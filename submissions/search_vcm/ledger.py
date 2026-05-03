#!/usr/bin/env python
"""Build and append Search VCM ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm.evaluator import (
    EXPERIMENTS_DIR,
    Q55_METRICS,
    append_jsonl,
    load_jsonl,
    normalize_q55_metrics,
    quality,
    score,
    write_json,
)
from submissions.search_vcm.subsets import get_subset

SOURCE_Q55_LEDGER = REPO_ROOT / "submissions/quantizr/experiments/pareto_control_lab/base_metrics/base_per_sample.jsonl"


def rank_rows(rows: list[dict]) -> list[dict]:
    by_quality = sorted(rows, key=lambda r: float(r["quality_i"]), reverse=True)
    by_pose = sorted(rows, key=lambda r: float(r["pose_term"]), reverse=True)
    by_seg = sorted(rows, key=lambda r: float(r["seg_term"]), reverse=True)
    ranks = {}
    for key, ordered in (("rank_by_quality", by_quality), ("rank_by_pose", by_pose), ("rank_by_seg", by_seg)):
        for rank, row in enumerate(ordered, start=1):
            ranks.setdefault(int(row["sample_id"]), {})[key] = rank
    out = []
    for row in by_quality:
        sample_id = int(row["sample_id"])
        out.append({**row, **ranks[sample_id]})
    return out


def subset_metrics(rows: list[dict], subset: list[int], *, archive_bytes: int) -> dict:
    by_id = {int(row["sample_id"]): row for row in rows}
    selected = [by_id[idx] for idx in subset if idx in by_id]
    if not selected:
        raise RuntimeError("subset has no rows in ledger")
    seg = sum(float(row["segnet_dist"]) for row in selected) / len(selected)
    pose = sum(float(row["posenet_dist"]) for row in selected) / len(selected)
    return {
        "sample_count": len(selected),
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": (10.0 * pose) ** 0.5,
        "quality": quality(seg, pose),
        "score": score(seg, pose, archive_bytes),
        "archive_bytes": archive_bytes,
    }


def build_base(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source_ledger
    if not source.exists():
        raise FileNotFoundError(f"missing source per-sample ledger: {source}")
    rows = rank_rows(load_jsonl(source))
    out_ledger = out_dir / f"base_{args.base}_per_sample.jsonl"
    out_ledger.write_text("")
    for row in rows:
        append_jsonl(out_ledger, row)

    base = normalize_q55_metrics(args.base, Q55_METRICS[args.base])
    subset_summary = {}
    for subset_name in ("hard3", "hard8", "strat64", "full600"):
        subset_summary[subset_name] = subset_metrics(rows, get_subset(subset_name, out_ledger), archive_bytes=base["archive_bytes"])
    summary = {
        "base": args.base,
        "calibration_full600": base,
        "proxy_subsets": subset_summary,
        "source_ledger": str(source),
        "per_sample_ledger": str(out_ledger),
        "row_count": len(rows),
    }
    write_json(out_dir / "base_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def append_run(args: argparse.Namespace) -> None:
    row = json.loads(args.row_json)
    append_jsonl(args.ledger, row)
    print(json.dumps(row, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    base = sub.add_parser("build-base")
    base.add_argument("--base", choices=sorted(Q55_METRICS), default="q55_fp16_pose_int10")
    base.add_argument("--source-ledger", type=Path, default=SOURCE_Q55_LEDGER)
    base.add_argument("--out-dir", type=Path, default=EXPERIMENTS_DIR)
    base.set_defaults(func=build_base)

    append = sub.add_parser("append-run")
    append.add_argument("--ledger", type=Path, default=EXPERIMENTS_DIR / "runs.jsonl")
    append.add_argument("--row-json", required=True)
    append.set_defaults(func=append_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
