#!/usr/bin/env python
"""Payload accounting helpers for Pareto control candidates.

This intentionally does not mutate or build a contest archive. It takes the
validated rows emitted by pareto_control_lab.py and computes the byte/score
budget before any control is promoted to a real packer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from q55_common import ORIGINAL_BYTES, write_json


DEFAULT_IN = Path(__file__).resolve().parent / "experiments/pareto_control_lab/channel_audit/metrics.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "experiments/pareto_control_lab/pack_plan.json"


def required_quality_for_score(archive_bytes: int, target: float = 0.300) -> float:
    return target - 25.0 * int(archive_bytes) / ORIGINAL_BYTES


def score_from_quality(quality: float, archive_bytes: int) -> float:
    return float(quality) + 25.0 * int(archive_bytes) / ORIGINAL_BYTES


def build_pack_plan(metrics: dict, *, target_score: float) -> dict:
    base_archive = int(metrics.get("base_archive_bytes") or 0)
    if not base_archive:
        base_archive_path = metrics.get("base_archive")
        if base_archive_path:
            base_archive = Path(base_archive_path).stat().st_size

    baseline = metrics["baseline"]
    rows = []
    for control in metrics.get("controls", []):
        best = control.get("best")
        if not best:
            rows.append(
                {
                    "control": control["control"],
                    "promote": False,
                    "reason": "no accepted score-improving step",
                }
            )
            continue
        added = int(best["added_bytes_est"])
        total = base_archive + added
        quality_after = float(best["metrics"]["quality"])
        score_after = score_from_quality(quality_after, total)
        rows.append(
            {
                "control": control["control"],
                "promote": score_after < target_score,
                "added_bytes_est": added,
                "total_archive_bytes_est": total,
                "quality_after": quality_after,
                "score_after_est": score_after,
                "required_quality_for_target": required_quality_for_score(total, target_score),
                "quality_gap_to_target": quality_after - required_quality_for_score(total, target_score),
                "packer": control.get("packer"),
            }
        )
    return {
        "target_score": target_score,
        "base_archive_bytes": base_archive,
        "baseline_quality": baseline["quality"],
        "baseline_required_quality_for_target": required_quality_for_score(base_archive, target_score),
        "baseline_quality_gap_to_target": baseline["quality"] - required_quality_for_score(base_archive, target_score),
        "controls": rows,
        "decision": "promote_validated_control" if any(row.get("promote") for row in rows) else "do_not_pack",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-score", type=float, default=0.300)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    plan = build_pack_plan(metrics, target_score=args.target_score)
    write_json(args.out, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
