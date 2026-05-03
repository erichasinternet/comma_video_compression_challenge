#!/usr/bin/env python
"""Evaluate exact qpose tile injection into the PR #62 lowmask renderer.

This is a no-training oracle for the first post-boundary idea:
lowmask base + small exact semantic tile side-channel. It uses the PR #62
generator unchanged and measures whether exact tiles buy enough evaluator
quality per byte to justify a new family.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json
from submissions.search_vcm_v2.families.boundary_residual_codec import apply_records, boundary_map, dilate_bool, make_tile_records
from submissions.search_vcm_v2.families.lowmask_boundary_residual import (
    AUDIT_COMPRESSORS,
    _best_prefix_under_budget,
    _load_exact_classes,
    _load_lowmask,
    _seg_sensitivity,
    _tile_density,
    _top_tile_order,
)
from submissions.search_vcm_v2.families.lowmask_data import (
    FP4_ARCHIVE,
    archive_audit,
    decode_fp4_pose_stream,
    load_fp4_generator,
    split_fp4_archive,
)
from submissions.search_vcm_v2.families.qpose14_data import load_original_subset, select_torch_device
from submissions.search_vcm_v2.subsets import get_subset


OUT_DIR = EXPERIMENTS_DIR / "lowmask_exact_tile_oracle"


def _prefix_n(candidate_name: str) -> int:
    match = re.search(r"_top(\d+)", candidate_name)
    if not match:
        return 0
    return int(match.group(1))


def _ranked_tiles(*, exact: torch.Tensor, low: torch.Tensor, radius: int, tile_size: int) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    sensitivity = _seg_sensitivity()
    sample_weights = torch.tensor(sensitivity["sample_weights"], dtype=torch.float32).view(-1, 1, 1)
    exact_boundary = boundary_map(exact)
    low_boundary = boundary_map(low)
    class_diff = exact != low
    band = dilate_bool(exact_boundary | low_boundary, radius)
    band_error = class_diff & band
    boundary_xor = dilate_bool(exact_boundary ^ low_boundary, radius)
    error_density = _tile_density(band_error, tile_size)
    boundary_density = _tile_density(boundary_xor, tile_size)
    score_map = (4.0 * error_density * sample_weights + 2.0 * boundary_density + error_density).contiguous()
    stats = {
        "radius": radius,
        "tile_size": tile_size,
        "total_band_error_pixels": int(band_error.sum().item()),
        "total_boundary_xor_pixels": int(boundary_xor.sum().item()),
    }
    return _top_tile_order(score_map), stats


def _evaluate_masks(
    *,
    masks: torch.Tensor,
    poses: torch.Tensor,
    sample_ids: list[int],
    subset_name: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch_device = select_torch_device(device)
    generator = load_fp4_generator(split_fp4_archive(FP4_ARCHIVE)[1], torch_device)
    distortion = DistortionNet().eval().to(torch_device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, torch_device)
    original = load_original_subset(subset_name, sample_ids, device="cpu").float()

    per_sample = []
    with torch.inference_mode():
        for start in range(0, len(sample_ids), batch_size):
            ids = sample_ids[start : start + batch_size]
            m = masks[ids].to(torch_device).long()
            p = poses[ids].to(torch_device).float()
            f1, f2 = generator(m, p)
            pred = torch.stack([f1, f2], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            gt = original[start : start + len(ids)].to(torch_device)
            pose_dist, seg_dist = distortion.compute_distortion(gt, pred)
            for sample_id, pose_v, seg_v in zip(ids, pose_dist.detach().cpu().tolist(), seg_dist.detach().cpu().tolist(), strict=True):
                seg = float(seg_v)
                pose = float(pose_v)
                seg_term = 100.0 * seg
                pose_term = float(torch.sqrt(torch.tensor(10.0 * pose)).item())
                per_sample.append(
                    {
                        "sample_id": int(sample_id),
                        "segnet_dist": seg,
                        "posenet_dist": pose,
                        "seg_term": seg_term,
                        "pose_term": pose_term,
                        "quality": quality(seg, pose),
                    }
                )
    seg_mean = sum(row["segnet_dist"] for row in per_sample) / len(per_sample)
    pose_mean = sum(row["posenet_dist"] for row in per_sample) / len(per_sample)
    return {
        "subset": subset_name,
        "sample_count": len(per_sample),
        "segnet_dist": seg_mean,
        "posenet_dist": pose_mean,
        "seg_term": 100.0 * seg_mean,
        "pose_term": float(torch.sqrt(torch.tensor(10.0 * pose_mean)).item()),
        "quality": quality(seg_mean, pose_mean),
        "max_sample_quality": max(row["quality"] for row in per_sample),
        "sample60_pose_term": next((row["pose_term"] for row in per_sample if row["sample_id"] == 60), None),
        "per_sample": per_sample,
    }


def run_oracle(
    *,
    subset_name: str,
    budgets: list[int],
    radius: int,
    tile_size: int,
    device: str,
    batch_size: int,
    out: Path,
) -> dict[str, Any]:
    exact = _load_exact_classes()
    low, _ = _load_lowmask()
    mask_br, model_br, pose_br = split_fp4_archive(FP4_ARCHIVE)
    poses = decode_fp4_pose_stream(pose_br)
    archive = archive_audit(FP4_ARCHIVE)
    sample_ids = get_subset(subset_name)
    ordered, rank_stats = _ranked_tiles(exact=exact, low=low, radius=radius, tile_size=tile_size)

    rows = []
    for name, masks, residual_bytes, records_count, source_error_pixels in [
        ("fp4_lowmask_base", low, 0, 0, 0),
        ("fp4_exact_qpose_mask_upper", exact, None, None, int((exact != low).sum().item())),
    ]:
        metrics = _evaluate_masks(masks=masks, poses=poses, sample_ids=sample_ids, subset_name=subset_name, device=device, batch_size=batch_size)
        archive_bytes = int(archive["archive_bytes"]) if residual_bytes is not None else int(archive["archive_bytes"]) + 179_205
        rows.append(
            {
                "candidate": name,
                "residual_bytes": residual_bytes,
                "archive_bytes": archive_bytes,
                "score": metrics["quality"] + rate_term(archive_bytes),
                "records_count": records_count,
                "source_error_pixels": source_error_pixels,
                **metrics,
            }
        )

    for budget in budgets:
        candidate = _best_prefix_under_budget(exact=exact, low=low, ordered_tiles=ordered, tile_size=tile_size, budget_bytes=budget)
        n = _prefix_n(candidate["candidate"])
        records = make_tile_records(exact, low, ordered[:n], tile_size=tile_size)
        repaired = apply_records(low, records, tile_size=tile_size)
        metrics = _evaluate_masks(masks=repaired, poses=poses, sample_ids=sample_ids, subset_name=subset_name, device=device, batch_size=batch_size)
        archive_bytes = int(archive["archive_bytes"]) + int(candidate["residual_bytes"])
        rows.append(
            {
                "candidate": candidate["candidate"],
                "budget_bytes": budget,
                "residual_bytes": int(candidate["residual_bytes"]),
                "archive_bytes": archive_bytes,
                "score": metrics["quality"] + rate_term(archive_bytes),
                "records_count": int(candidate["selected_tiles"]),
                "source_error_pixels": int(candidate["source_error_pixels"]),
                "coverage_fraction": float(candidate.get("coverage_fraction", 0.0)),
                "compressor_breakdown": candidate["compressor_breakdown"],
                **metrics,
            }
        )

    base = next(row for row in rows if row["candidate"] == "fp4_lowmask_base")
    for row in rows:
        row["quality_delta_vs_fp4"] = row["quality"] - base["quality"]
        row["score_delta_vs_fp4"] = row["score"] - base["score"]

    best_by_score = min(rows, key=lambda row: row["score"])
    best_budgeted = min([row for row in rows if row["candidate"].startswith("R5a_")], key=lambda row: row["score"], default=None)
    summary = {
        "subset": subset_name,
        "device": str(select_torch_device(device)),
        "fp4_archive_bytes": int(archive["archive_bytes"]),
        "rank_stats": rank_stats,
        "budgets": budgets,
        "rows": rows,
        "best_by_score": best_by_score,
        "best_budgeted": best_budgeted,
        "decision": "continue" if best_budgeted and best_budgeted["quality_delta_vs_fp4"] <= -0.02 else "close_exact_tile_router",
    }
    write_json(out / f"{subset_name}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="hard8", choices=["hard8", "strat64"])
    parser.add_argument("--budgets", default="4096,8192,12288,16384,20480")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    budgets = [int(item.strip()) for item in args.budgets.split(",") if item.strip()]
    run_oracle(
        subset_name=args.subset,
        budgets=budgets,
        radius=args.radius,
        tile_size=args.tile_size,
        device=args.device,
        batch_size=args.batch_size,
        out=args.out,
    )


if __name__ == "__main__":
    main()
