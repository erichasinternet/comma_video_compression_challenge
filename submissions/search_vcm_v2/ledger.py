#!/usr/bin/env python
"""Build qpose14 baseline and append Search VCM v2 run ledgers."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.evaluator import (
    EXPERIMENTS_DIR,
    QPOSE14_LEDGER,
    QPOSE14_REFERENCE,
    QPOSE14_SUMMARY,
    append_jsonl,
    load_jsonl,
    qpose14_reference_summary,
    quality,
    score,
    write_json,
)


SOURCE_Q55_V1_LEDGER = REPO_ROOT / "submissions/search_vcm/experiments/search_vcm/base_q55_fp16_pose_int10_per_sample.jsonl"
QPOSE14_ARCHIVE = REPO_ROOT / "submissions/qpose14/archive.zip"


def _distill_q55_field(row: dict[str, Any], key: str) -> float:
    if key in row:
        return float(row[key])
    if key == "segnet_dist":
        return float(row["seg_term"]) / 100.0
    if key == "posenet_dist":
        return (float(row["pose_term"]) ** 2) / 10.0
    raise KeyError(key)


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks: dict[int, dict[str, int]] = {}
    rank_specs = (
        ("rank_by_quality", "qpose14_quality"),
        ("rank_by_pose", "qpose14_pose_term"),
        ("rank_by_seg", "qpose14_seg_term"),
    )
    for rank_key, metric_key in rank_specs:
        for rank, row in enumerate(sorted(rows, key=lambda item: float(item[metric_key]), reverse=True), start=1):
            ranks.setdefault(int(row["sample_id"]), {})[rank_key] = rank
    out = []
    for row in sorted(rows, key=lambda item: int(item["sample_id"])):
        out.append({**row, **ranks[int(row["sample_id"])]})
    return out


def build_proxy_qpose14_rows(source_ledger: Path) -> list[dict[str, Any]]:
    """Create a qpose14-shaped per-sample ledger from the q55 v1 ledger.

    The PR #63 archive is not present locally. This proxy preserves q55 per-sample
    ordering while rescaling full-set means to the qpose14 README metrics. It is
    suitable for governance plumbing and deterministic subsets, not for claiming
    local qpose14 evaluation equivalence.
    """

    if not source_ledger.exists():
        raise FileNotFoundError(f"missing source proxy ledger: {source_ledger}")
    source_rows = load_jsonl(source_ledger)
    if len(source_rows) != 600:
        raise RuntimeError(f"expected 600 source rows, got {len(source_rows)} from {source_ledger}")

    mean_seg = sum(_distill_q55_field(row, "segnet_dist") for row in source_rows) / len(source_rows)
    mean_pose = sum(_distill_q55_field(row, "posenet_dist") for row in source_rows) / len(source_rows)
    seg_scale = QPOSE14_REFERENCE["segnet_dist"] / mean_seg if mean_seg > 0 else 1.0
    pose_scale = QPOSE14_REFERENCE["posenet_dist"] / mean_pose if mean_pose > 0 else 1.0

    rows: list[dict[str, Any]] = []
    for row in source_rows:
        seg = max(0.0, _distill_q55_field(row, "segnet_dist") * seg_scale)
        pose = max(0.0, _distill_q55_field(row, "posenet_dist") * pose_scale)
        rows.append(
            {
                "sample_id": int(row["sample_id"]),
                "qpose14_segnet_dist": seg,
                "qpose14_posenet_dist": pose,
                "qpose14_seg_term": 100.0 * seg,
                "qpose14_pose_term": math.sqrt(10.0 * pose),
                "qpose14_quality": quality(seg, pose),
                "proxy_source_sample_quality": float(row.get("quality_i", quality(_distill_q55_field(row, "segnet_dist"), _distill_q55_field(row, "posenet_dist")))),
            }
        )
    return _rank_rows(rows)


def qpose_subset_summary(rows: list[dict[str, Any]], subset: list[int], *, archive_bytes: int) -> dict[str, Any]:
    by_id = {int(row["sample_id"]): row for row in rows}
    selected = [by_id[idx] for idx in subset if idx in by_id]
    if not selected:
        raise RuntimeError("subset has no qpose rows")
    seg = sum(float(row["qpose14_segnet_dist"]) for row in selected) / len(selected)
    pose = sum(float(row["qpose14_posenet_dist"]) for row in selected) / len(selected)
    return {
        "sample_count": len(selected),
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
        "quality": quality(seg, pose),
        "score": score(seg, pose, archive_bytes),
        "archive_bytes": archive_bytes,
    }


def build_qpose14_baseline(args: argparse.Namespace) -> dict[str, Any]:
    from submissions.search_vcm_v2.subsets import get_subset

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_proxy_qpose14_rows(args.source_ledger)

    QPOSE14_LEDGER.write_text("")
    for row in rows:
        append_jsonl(QPOSE14_LEDGER, row)

    reference = qpose14_reference_summary()
    archive_bytes = int(reference["archive_bytes"])
    subset_summaries = {
        "smoke": qpose_subset_summary(rows, get_subset("smoke"), archive_bytes=archive_bytes),
        "hard3": qpose_subset_summary(rows, get_subset("hard3"), archive_bytes=archive_bytes),
        "hard8": qpose_subset_summary(rows, get_subset("hard8"), archive_bytes=archive_bytes),
        "strat64": qpose_subset_summary(rows, get_subset("strat64"), archive_bytes=archive_bytes),
        "full600": qpose_subset_summary(rows, get_subset("full600"), archive_bytes=archive_bytes),
    }
    summary = {
        **reference,
        "ledger_source": "q55_fp16_pose_int10_proxy_rescaled_to_qpose14_reference",
        "source_ledger": str(args.source_ledger),
        "per_sample_ledger": str(QPOSE14_LEDGER),
        "row_count": len(rows),
        "artifact_status": "real_archive_present" if QPOSE14_ARCHIVE.exists() else "missing_real_qpose14_archive_using_proxy_ledger",
        "artifact_path": str(QPOSE14_ARCHIVE) if QPOSE14_ARCHIVE.exists() else None,
        "subset_summaries": subset_summaries,
        "hard3_quality": subset_summaries["hard3"]["quality"],
        "hard8_quality": subset_summaries["hard8"]["quality"],
        "strat64_quality": subset_summaries["strat64"]["quality"],
        "full600_quality": subset_summaries["full600"]["quality"],
    }
    write_json(QPOSE14_SUMMARY, summary)
    return summary


def cmd_build_qpose14(args: argparse.Namespace) -> None:
    summary = build_qpose14_baseline(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def evaluate_submission_per_sample(args: argparse.Namespace) -> list[dict[str, Any]]:
    from frame_utils import AVVideoDataset, TensorVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    with open(args.video_names_file, "r") as f:
        video_names = [line.strip() for line in f if line.strip()]

    distortion_net = DistortionNet().eval().to(device=device)
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    ds_gt = AVVideoDataset(
        video_names,
        data_dir=args.uncompressed_dir,
        batch_size=args.batch_size,
        device=device,
        num_threads=args.num_threads,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    ds_gt.prepare_data()
    ds_comp = TensorVideoDataset(
        video_names,
        data_dir=args.submission_dir / "inflated",
        batch_size=args.batch_size,
        device=device,
        num_threads=args.num_threads,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    ds_comp.prepare_data()

    rows = []
    sample_id = 0
    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_comp) in tqdm(
            zip(torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0), torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)),
            desc="qpose14 per-sample eval",
        ):
            batch_gt = batch_gt.to(device)
            batch_comp = batch_comp.to(device)
            posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
            for pose_v, seg_v in zip(posenet_dist.detach().cpu().tolist(), segnet_dist.detach().cpu().tolist(), strict=True):
                seg = float(seg_v)
                pose = float(pose_v)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "qpose14_segnet_dist": seg,
                        "qpose14_posenet_dist": pose,
                        "qpose14_seg_term": 100.0 * seg,
                        "qpose14_pose_term": math.sqrt(max(0.0, 10.0 * pose)),
                        "qpose14_quality": quality(seg, pose),
                    }
                )
                sample_id += 1
    return _rank_rows(rows)


def cmd_build_qpose14_real(args: argparse.Namespace) -> None:
    from submissions.search_vcm_v2.subsets import get_subset

    rows = evaluate_submission_per_sample(args)
    if len(rows) != 600:
        raise RuntimeError(f"expected 600 evaluated rows, got {len(rows)}")
    QPOSE14_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    QPOSE14_LEDGER.write_text("")
    for row in rows:
        append_jsonl(QPOSE14_LEDGER, row)

    archive_bytes = (args.submission_dir / "archive.zip").stat().st_size
    full = qpose_subset_summary(rows, list(range(600)), archive_bytes=archive_bytes)
    subset_summaries = {
        "smoke": qpose_subset_summary(rows, get_subset("smoke"), archive_bytes=archive_bytes),
        "hard3": qpose_subset_summary(rows, get_subset("hard3"), archive_bytes=archive_bytes),
        "hard8": qpose_subset_summary(rows, get_subset("hard8"), archive_bytes=archive_bytes),
        "strat64": qpose_subset_summary(rows, get_subset("strat64"), archive_bytes=archive_bytes),
        "full600": full,
    }
    summary = {
        "archive_bytes": archive_bytes,
        "segnet_dist": full["segnet_dist"],
        "posenet_dist": full["posenet_dist"],
        "seg_term": full["seg_term"],
        "pose_term": full["pose_term"],
        "quality": full["quality"],
        "score": full["score"],
        "rate_term": full["score"] - full["quality"],
        "source": f"local_{args.device or 'auto'}_per_sample_evaluator",
        "reference_pr63": qpose14_reference_summary(),
        "ledger_source": f"local_{args.device or 'auto'}_per_sample_evaluator",
        "per_sample_ledger": str(QPOSE14_LEDGER),
        "row_count": len(rows),
        "artifact_status": "real_archive_present",
        "artifact_path": str(args.submission_dir / "archive.zip"),
        "subset_summaries": subset_summaries,
        "hard3_quality": subset_summaries["hard3"]["quality"],
        "hard8_quality": subset_summaries["hard8"]["quality"],
        "strat64_quality": subset_summaries["strat64"]["quality"],
        "full600_quality": subset_summaries["full600"]["quality"],
    }
    write_json(QPOSE14_SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_append_run(args: argparse.Namespace) -> None:
    row = json.loads(args.row_json)
    append_jsonl(args.ledger, row)
    print(json.dumps(row, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    qpose = sub.add_parser("build-qpose14")
    qpose.add_argument("--source-ledger", type=Path, default=SOURCE_Q55_V1_LEDGER)
    qpose.add_argument("--out-dir", type=Path, default=EXPERIMENTS_DIR)
    qpose.set_defaults(func=cmd_build_qpose14)

    qpose_real = sub.add_parser("build-qpose14-real")
    qpose_real.add_argument("--submission-dir", type=Path, default=REPO_ROOT / "submissions/qpose14")
    qpose_real.add_argument("--uncompressed-dir", type=Path, default=REPO_ROOT / "videos")
    qpose_real.add_argument("--video-names-file", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    qpose_real.add_argument("--device", default="mps")
    qpose_real.add_argument("--batch-size", type=int, default=8)
    qpose_real.add_argument("--num-threads", type=int, default=2)
    qpose_real.add_argument("--prefetch-queue-depth", type=int, default=2)
    qpose_real.add_argument("--seed", type=int, default=1234)
    qpose_real.set_defaults(func=cmd_build_qpose14_real)

    append = sub.add_parser("append-run")
    append.add_argument("--ledger", type=Path, default=EXPERIMENTS_DIR / "runs.jsonl")
    append.add_argument("--row-json", required=True)
    append.set_defaults(func=cmd_append_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
