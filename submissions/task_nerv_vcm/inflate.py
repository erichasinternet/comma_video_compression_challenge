#!/usr/bin/env python3
"""Inflate a task-NeRV prototype payload into official raw RGB frames."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import brotli
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frame_utils import camera_size
from submissions.task_nerv_vcm.model import HNeRVConfig, HNeRVRenderer


def load_payload(data_dir: Path) -> dict:
    plain = data_dir / "nerv_model.pt"
    compressed = data_dir / "nerv_model.pt.br"
    if plain.exists():
        return torch.load(plain, map_location="cpu")
    if compressed.exists():
        return torch.load(io.BytesIO(brotli.decompress(compressed.read_bytes())), map_location="cpu")
    archive = data_dir / "archive.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            return torch.load(io.BytesIO(brotli.decompress(zf.read("nerv_model.pt.br"))), map_location="cpu")
    raise FileNotFoundError(f"no nerv_model.pt(.br) payload in {data_dir}")


@torch.inference_mode()
def inflate_to_raw(data_dir: Path, _video_name: str, dst_path: Path, *, batch_size: int = 16) -> None:
    payload = load_payload(data_dir)
    cfg = HNeRVConfig(**payload["config"])
    model = HNeRVRenderer(cfg).eval()
    model.load_state_dict(payload["state_dict"])
    sample_ids = payload.get("sample_ids", list(range(cfg.n_frames // 2)))
    if sample_ids != list(range(len(sample_ids))):
        raise RuntimeError(
            "prototype inflate requires contiguous sample_ids starting at 0; "
            f"got first ids {sample_ids[:8]}"
        )
    target_h, target_w = camera_size[1], camera_size[0]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as f:
        for start in range(0, cfg.n_frames, batch_size):
            frame_ids = torch.arange(start, min(cfg.n_frames, start + batch_size))
            frames = model(frame_ids)
            frames = F.interpolate(frames, size=(target_h, target_w), mode="bicubic", align_corners=False)
            frames = frames.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            f.write(frames.numpy().tobytes())


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("Usage: inflate.py <archive_dir> <video_name> <dst.raw>")
    inflate_to_raw(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))


if __name__ == "__main__":
    main()

