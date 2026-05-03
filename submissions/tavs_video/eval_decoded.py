#!/usr/bin/env python3
"""Evaluate decoded TAVS frames against original evaluator targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from submissions.tavs_video.codec_roundtrip import decode_video_to_chw
from submissions.tavs_video.common import (
    build_distortion,
    collect_targets,
    evaluate_frames,
    load_original_pairs_by_indices,
    metric_table,
    parse_indices,
    save_json,
    FeatureTap,
    ROOT,
)


def load_video_pairs(video_path: Path) -> torch.Tensor:
    frames = decode_video_to_chw(video_path, target_hw=None)
    if frames.shape[0] % 2:
        raise RuntimeError(f"video has odd frame count: {frames.shape[0]}")
    return frames.reshape(-1, 2, 3, frames.shape[-2], frames.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--archive-bytes", type=int, default=0)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--subset", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    sample_ids = parse_indices(args.indices, offset=args.offset, subset=args.subset)
    frames = load_video_pairs(args.video)
    if frames.shape[0] != len(sample_ids):
        raise RuntimeError(f"video has {frames.shape[0]} pairs, but {len(sample_ids)} sample ids were provided")

    original = load_original_pairs_by_indices(
        data_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
        batch_size=max(args.batch_size, 8),
    )
    distortion = build_distortion(device)
    empty_seg_tap = FeatureTap(distortion.segnet, [])
    empty_pose_tap = FeatureTap(distortion.posenet, [])
    targets = collect_targets(
        distortion=distortion,
        original_cpu=original,
        device=device,
        batch_size=args.batch_size,
        seg_tap=empty_seg_tap,
        pose_tap=empty_pose_tap,
    )
    metrics = evaluate_frames(frames=frames, targets=targets, distortion=distortion, batch_size=args.batch_size)
    if args.archive_bytes:
        metrics.update(metric_table(metrics["segnet_dist"], metrics["posenet_dist"], args.archive_bytes))
    metrics["sample_ids"] = sample_ids
    metrics["video"] = str(args.video)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.out:
        save_json(args.out, metrics)


if __name__ == "__main__":
    main()
