#!/usr/bin/env python3
"""Inflate a task-token prototype archive into raw RGB frames."""

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
from submissions.task_token_vcm.train_capacity import DecoderConfig, TaskTokenDecoder


def load_payload(data_dir: Path) -> dict:
    plain = data_dir / "task_token.pt"
    compressed = data_dir / "task_token.pt.br"
    if plain.exists():
        return torch.load(plain, map_location="cpu")
    if compressed.exists():
        return torch.load(io.BytesIO(brotli.decompress(compressed.read_bytes())), map_location="cpu")
    archive = data_dir / "archive.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            return torch.load(io.BytesIO(brotli.decompress(zf.read("task_token.pt.br"))), map_location="cpu")
    raise FileNotFoundError(f"no task_token.pt(.br) payload in {data_dir}")


@torch.inference_mode()
def inflate_to_raw(data_dir: Path, _video_name: str, dst_path: Path, *, batch_size: int = 8) -> None:
    payload = load_payload(data_dir)
    cfg = DecoderConfig(**payload["decoder_config"])
    decoder = TaskTokenDecoder(cfg).eval()
    decoder.load_state_dict(payload["decoder_state_dict"])
    z_seg = payload["z_seg"].float()
    z_pose = payload["z_pose"].float()
    z_pair_payload = payload.get("z_pair")
    z_pair = z_pair_payload.float() if z_pair_payload is not None else None
    sample_ids = payload.get("sample_ids", list(range(z_seg.shape[0])))
    if sample_ids != list(range(len(sample_ids))):
        raise RuntimeError(
            "prototype inflate requires contiguous sample_ids starting at 0; "
            f"got first ids {sample_ids[:8]}"
        )
    target_h, target_w = camera_size[1], camera_size[0]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as f:
        for start in range(0, z_seg.shape[0], batch_size):
            end = min(z_seg.shape[0], start + batch_size)
            pair_batch = z_pair[start:end] if z_pair is not None else None
            frames = decoder(z_seg[start:end], z_pose[start:end], pair_batch).flatten(0, 1)
            frames = F.interpolate(frames, size=(target_h, target_w), mode="bicubic", align_corners=False)
            frames = frames.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            f.write(frames.numpy().tobytes())


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("Usage: inflate.py <archive_dir> <video_name> <dst.raw>")
    inflate_to_raw(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))


if __name__ == "__main__":
    main()
