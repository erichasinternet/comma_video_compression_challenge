#!/usr/bin/env python3
"""Encode optimized TAVS source frames and package a submission archive."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from submissions.tavs_video.codec_roundtrip import codec_ext, encode_rawvideo, write_raw_rgb
from submissions.tavs_video.common import ROOT, metric_table, save_json


def load_frames(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if "best_frames" in data:
            return data["best_frames"]
        if "frames" in data:
            return data["frames"]
    if torch.is_tensor(data):
        return data
    raise ValueError(f"could not find frames tensor in {path}")


def package_video(video_path: Path, archive_zip: Path) -> int:
    archive_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(video_path, "video" + video_path.suffix)
    return archive_zip.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-pt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--codec", choices=["svtav1", "libaom-av1", "vp9", "x265"], default="svtav1")
    parser.add_argument("--crf", type=int, default=57)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.frames_pt)
    raw_path = args.out_dir / "source.rgb"
    _, width, height = write_raw_rgb(frames, raw_path)
    video_path = args.out_dir / f"video_{args.codec}_crf{args.crf}{codec_ext(args.codec)}"
    video_bytes = encode_rawvideo(
        raw_path=raw_path,
        out_path=video_path,
        width=width,
        height=height,
        codec=args.codec,
        crf=args.crf,
        fps=args.fps,
    )
    archive_path = args.archive or (args.out_dir / "archive.zip")
    archive_bytes = package_video(video_path, archive_path)
    metrics = {
        "frames_pt": str(args.frames_pt),
        "video_path": str(video_path),
        "video_bytes": int(video_bytes),
        "archive_zip": str(archive_path),
        "archive_bytes": int(archive_bytes),
        "codec": args.codec,
        "crf": int(args.crf),
        "frame_width": int(width),
        "frame_height": int(height),
    }
    if args.metrics_json:
        previous = json.loads(args.metrics_json.read_text()) if args.metrics_json.exists() else {}
        if "segnet_dist" in previous and "posenet_dist" in previous:
            previous.update(metric_table(previous["segnet_dist"], previous["posenet_dist"], archive_bytes))
        previous["pack"] = metrics
        save_json(args.metrics_json, previous)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
