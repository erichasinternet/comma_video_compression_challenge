#!/usr/bin/env python3
"""Inflate a commaVQ-token task-renderer prototype archive into raw RGB frames."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frame_utils import camera_size
from submissions.commavq_task.pack_tokens import unpack_uint10
from submissions.commavq_task.train_renderer import CommaVQTaskRenderer, RendererConfig


def read_archive_file(data_dir: Path, name: str) -> bytes:
    direct = data_dir / name
    if direct.exists():
        return direct.read_bytes()
    archive = data_dir / "archive.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            return zf.read(name)
    raise FileNotFoundError(f"missing {name} in {data_dir}")


def load_tokens(data_dir: Path) -> tuple[torch.Tensor, list[int]]:
    meta = json.loads(read_archive_file(data_dir, "tokens_meta.json").decode())
    compressed = read_archive_file(data_dir, "tokens.uint10.br")
    packed = brotli.decompress(compressed)
    tokens = unpack_uint10(packed, int(meta["token_count"])).reshape(meta["shape"]).astype(np.int64)
    return torch.from_numpy(tokens), list(meta.get("sample_ids", range(tokens.shape[0])))


def load_renderer(data_dir: Path) -> CommaVQTaskRenderer:
    payload = torch.load(io.BytesIO(brotli.decompress(read_archive_file(data_dir, "renderer.pt.br"))), map_location="cpu")
    cfg = RendererConfig(**payload["renderer_config"])
    renderer = CommaVQTaskRenderer(cfg).eval()
    renderer.load_state_dict(payload["renderer_state_dict"])
    return renderer


@torch.inference_mode()
def inflate_to_raw(data_dir: Path, _video_name: str, dst_path: Path, *, batch_size: int = 8) -> None:
    tokens, sample_ids = load_tokens(data_dir)
    if sample_ids != list(range(len(sample_ids))):
        raise RuntimeError(
            "prototype inflate requires contiguous sample_ids starting at 0; "
            f"got first ids {sample_ids[:8]}"
        )
    renderer = load_renderer(data_dir)
    target_h, target_w = camera_size[1], camera_size[0]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as f:
        for start in range(0, tokens.shape[0], batch_size):
            frames = renderer(tokens[start : start + batch_size]).flatten(0, 1)
            frames = F.interpolate(frames, size=(target_h, target_w), mode="bicubic", align_corners=False)
            frames = frames.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            f.write(frames.numpy().tobytes())


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("Usage: inflate.py <archive_dir> <video_name> <dst.raw>")
    inflate_to_raw(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))


if __name__ == "__main__":
    main()

