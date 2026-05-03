#!/usr/bin/env python
"""Perfect hard-sample replacement oracle for the q55/qpack base layer.

This script does not build a submission. It measures whether selectively replacing
the hardest base-layer samples with original camera-domain frame pairs could create
enough score budget to justify an encoded enhancement sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from frame_utils import AVVideoDataset, TensorVideoDataset
from modules import DistortionNet, posenet_sd_path, segnet_sd_path
from q55_common import ORIGINAL_BYTES, quality_term, score_from_bytes, sha256_file, write_json


def parse_topk(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def load_video_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def per_sample_distortions(args) -> list[dict]:
    device = torch.device(args.device)
    names = load_video_names(args.video_names_file)

    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    ds_gt = AVVideoDataset(names, data_dir=args.uncompressed_dir, batch_size=args.batch_size, device=device)
    ds_gt.prepare_data()
    ds_base = TensorVideoDataset(names, data_dir=args.base_inflated_dir, batch_size=args.batch_size, device=device)
    ds_base.prepare_data()

    records: list[dict] = []
    global_index = 0
    with torch.inference_mode():
        for (path_gt, _, batch_gt), (path_base, _, batch_base) in zip(ds_gt, ds_base):
            batch_gt = batch_gt.to(device)
            batch_base = batch_base.to(device)
            if batch_gt.shape != batch_base.shape:
                raise RuntimeError(f"batch shape mismatch: {batch_gt.shape} vs {batch_base.shape}")
            pose, seg = net.compute_distortion(batch_gt, batch_base)
            pose = pose.detach().cpu()
            seg = seg.detach().cpu()
            for j in range(batch_gt.shape[0]):
                pose_j = float(pose[j].item())
                seg_j = float(seg[j].item())
                records.append(
                    {
                        "index": global_index,
                        "video": Path(str(path_gt)).name,
                        "batch_local_index": j,
                        "posenet_dist": pose_j,
                        "segnet_dist": seg_j,
                        "quality_contribution": quality_term(seg_j, pose_j),
                    }
                )
                global_index += 1
    if not records:
        raise RuntimeError("no samples evaluated")
    return records


def summarize_base(records: list[dict], archive_bytes: int) -> dict:
    n = len(records)
    seg = sum(r["segnet_dist"] for r in records) / n
    pose = sum(r["posenet_dist"] for r in records) / n
    quality = quality_term(seg, pose)
    return {
        "samples": n,
        "archive_bytes": archive_bytes,
        "segnet_dist": seg,
        "posenet_dist": pose,
        "quality": quality,
        "score": score_from_bytes(seg, pose, archive_bytes),
        "rate_term": 25.0 * archive_bytes / ORIGINAL_BYTES,
    }


def topk_curve(records: list[dict], base_archive_bytes: int, topk_values: list[int], target: float) -> list[dict]:
    n = len(records)
    ranked = sorted(records, key=lambda r: r["quality_contribution"], reverse=True)
    total_seg = sum(r["segnet_dist"] for r in records)
    total_pose = sum(r["posenet_dist"] for r in records)
    selected_seg = 0.0
    selected_pose = 0.0
    curve = []
    current_rank = 0

    wanted = sorted(set([0, *[min(k, n) for k in topk_values]]))
    for k in wanted:
        while current_rank < k:
            selected_seg += ranked[current_rank]["segnet_dist"]
            selected_pose += ranked[current_rank]["posenet_dist"]
            current_rank += 1
        seg_k = (total_seg - selected_seg) / n
        pose_k = (total_pose - selected_pose) / n
        quality_k = quality_term(seg_k, pose_k)
        max_total_archive = (target - quality_k) * ORIGINAL_BYTES / 25.0
        sidecar_allowed = max_total_archive - base_archive_bytes
        curve.append(
            {
                "top_k": k,
                "segnet_dist": seg_k,
                "posenet_dist": pose_k,
                "quality": quality_k,
                "score_without_sidecar_bytes": score_from_bytes(seg_k, pose_k, base_archive_bytes),
                "quality_gain": quality_term(total_seg / n, total_pose / n) - quality_k,
                "max_total_archive_bytes_for_target": math.floor(max_total_archive),
                "sidecar_allowed_bytes_for_target": math.floor(sidecar_allowed),
                "sidecar_allowed_positive": sidecar_allowed > 0,
                "selected_indices": [r["index"] for r in ranked[:k]],
            }
        )
    return curve


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=REPO_ROOT / "submissions/q55_fp16_pose_int10/archive.zip")
    parser.add_argument(
        "--base-inflated-dir",
        type=Path,
        default=REPO_ROOT / "submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int10_cpu/submission/inflated",
    )
    parser.add_argument("--uncompressed-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names-file", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "submissions/quantizr/experiments/q55_hard_sample_oracle")
    parser.add_argument("--top-k", default="4,8,16,32,48,64,96,128")
    parser.add_argument("--target-score", type=float, default=0.300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reuse-per-sample-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if not args.base_archive.exists():
        raise FileNotFoundError(args.base_archive)
    if not args.base_inflated_dir.exists():
        raise FileNotFoundError(
            f"{args.base_inflated_dir} does not exist. Inflate the base archive first or pass --base-inflated-dir."
        )

    archive_bytes = args.base_archive.stat().st_size
    records = read_jsonl(args.reuse_per_sample_jsonl) if args.reuse_per_sample_jsonl else per_sample_distortions(args)
    ranked = sorted(records, key=lambda r: r["quality_contribution"], reverse=True)
    topk_values = parse_topk(args.top_k)
    curve = topk_curve(records, archive_bytes, topk_values, args.target_score)
    base = summarize_base(records, archive_bytes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.out_dir / "per_sample_quality.jsonl"
    write_jsonl(per_sample_path, ranked)
    topk_path = args.out_dir / "topk_curve.json"
    record = {
        "base_archive": str(args.base_archive),
        "base_archive_sha256": sha256_file(args.base_archive),
        "base_inflated_dir": str(args.base_inflated_dir),
        "target_score": args.target_score,
        "sort_key": "per_sample_quality_contribution",
        "base": base,
        "topk_curve": curve,
        "outputs": {
            "per_sample_quality_jsonl": str(per_sample_path),
            "topk_curve_json": str(topk_path),
        },
    }
    write_json(topk_path, record)

    best_budget = max(curve, key=lambda r: r["sidecar_allowed_bytes_for_target"])
    print(
        json.dumps(
            {
                "base_quality": base["quality"],
                "base_score": base["score"],
                "best_sidecar_budget": best_budget,
                "topk_curve_json": str(topk_path),
                "per_sample_quality_jsonl": str(per_sample_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
