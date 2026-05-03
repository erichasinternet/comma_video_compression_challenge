#!/usr/bin/env python3
"""Inflate a TAVS archive by decoding its stored video into official raw frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb


VIDEO_NAMES = ("video.ivf", "video.mkv", "video.mp4", "synthetic.ivf", "synthetic.mkv", "synthetic.mp4")


def find_video(archive_dir: Path) -> Path:
    for name in VIDEO_NAMES:
        candidate = archive_dir / name
        if candidate.exists():
            return candidate
    videos = sorted(
        [
            path
            for suffix in ("*.ivf", "*.mkv", "*.mp4", "*.webm", "*.hevc")
            for path in archive_dir.glob(suffix)
        ]
    )
    if videos:
        return videos[0]
    raise FileNotFoundError(f"no TAVS video found in {archive_dir}")


def decode_video(video_path: Path) -> list[torch.Tensor]:
    fmt = "hevc" if video_path.suffix == ".hevc" else None
    container = av.open(str(video_path), format=fmt)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))
    container.close()
    if not frames:
        raise RuntimeError(f"decoded no frames from {video_path}")
    return frames


def resize_to_camera(frame: torch.Tensor) -> torch.Tensor:
    target_w, target_h = camera_size
    if tuple(frame.shape[:2]) == (target_h, target_w):
        return frame.contiguous()
    x = frame.permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8).contiguous()


def inflate(archive_dir: Path, out_dir: Path, file_list: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
    if not files:
        raise RuntimeError(f"empty file list: {file_list}")
    video_path = find_video(archive_dir)
    frames = decode_video(video_path)

    cursor = 0
    frames_per_file = len(frames) // len(files)
    if frames_per_file * len(files) != len(frames):
        raise RuntimeError(f"{len(frames)} decoded frames does not split over {len(files)} files")
    for file_name in files:
        raw_path = out_dir / f"{Path(file_name).stem}.raw"
        with raw_path.open("wb") as f:
            for frame in frames[cursor : cursor + frames_per_file]:
                f.write(resize_to_camera(frame).numpy().tobytes())
        cursor += frames_per_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, default=Path("public_test_video_names.txt"))
    args = parser.parse_args()
    inflate(args.archive_dir, args.out_dir, args.file_list)


if __name__ == "__main__":
    main()
