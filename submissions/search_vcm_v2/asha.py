#!/usr/bin/env python
"""ASHA-style governor for Search VCM v2."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.candidate_api import Budget, DecisionRow, stable_hash
from submissions.search_vcm_v2.evaluator import (
    EXPERIMENTS_DIR,
    QPOSE14_LEDGER,
    RUN_LEDGER,
    add_baseline_deltas,
    append_jsonl,
    baseline_for_subset,
    load_jsonl,
    qpose14_reference_summary,
    write_json,
)
from submissions.search_vcm_v2.negative_cache import NegativeCache
from submissions.search_vcm_v2.subsets import get_subset


FAMILY_MODULES = {
    "qpose14_baseline": "submissions.search_vcm_v2.families.qpose14_baseline",
    "qpose14_official_packaging": "submissions.search_vcm_v2.families.qpose14_official_packaging",
    "factorized_exactmask_pose_tokens": "submissions.search_vcm_v2.families.factorized_exactmask_pose_tokens",
    "lowmask_qpose_distill": "submissions.search_vcm_v2.families.lowmask_qpose_distill",
    "lowmask_boundary_residual": "submissions.search_vcm_v2.families.lowmask_boundary_residual",
}

ROUND_SUBSETS = {
    "gate0_audit": "full600",
    "gate0_byte_audit": "full600",
    "gate0b_temporal_subsampling": "full600",
    "smoke": "smoke",
    "hard8_capacity": "hard8_capacity",
    "hard8_free_boundary_capacity": "hard8_capacity",
    "hard8_budgeted_boundary": "hard8_capacity",
    "hard8_compressed_renderer": "hard8_compressed",
    "packability": "packability",
    "hard8_compressed": "hard8_compressed",
    "strat64": "strat64",
    "full600": "full600",
    "official_dry_run": "full600",
}


def load_family_candidates(family: str, round_name: str):
    if family not in FAMILY_MODULES:
        raise ValueError(f"unknown family: {family}")
    module = importlib.import_module(FAMILY_MODULES[family])
    return module.candidates(round_name)


def previous_promotions(ledger_path: Path = RUN_LEDGER) -> list[dict[str, Any]]:
    rows = load_jsonl(ledger_path)
    return [row for row in rows if str(row.get("decision", "")).startswith("promote")]


def blocked_row(*, run_id: str, family: str, candidate_name: str, config: dict[str, Any], reason: str, round_name: str, subset: str) -> DecisionRow:
    return DecisionRow(
        run_id=run_id,
        candidate_name=candidate_name,
        family=family,
        role="blocked",
        kind="blocked_negative_cache",
        packable=False,
        config_hash=stable_hash(config),
        novelty_reason=config.get("novelty_reason", ""),
        subset=subset,
        round=round_name,
        archive_bytes=None,
        added_bytes=0,
        quality=None,
        segnet_dist=None,
        posenet_dist=None,
        score=None,
        seg_delta=None,
        pose_delta=None,
        byte_delta=None,
        score_delta_vs_base=None,
        dominates_base=False,
        term_tradeoff="blocked_negative_cache",
        decision="blocked_negative_cache",
        failure_reason=reason,
        extra={"config": config},
    )


def normalize_decision(row: DecisionRow, *, round_name: str) -> DecisionRow:
    if row.role in {"baseline", "fallback_candidate"}:
        return row
    if row.failure_reason:
        if round_name == "hard8_capacity" and row.family == "factorized_exactmask_pose_tokens":
            if row.decision != "blocked_missing_qpose14_artifact":
                row.decision = "close_factorized_family"
        if round_name == "hard8_capacity" and row.family == "lowmask_qpose_distill":
            row.decision = "close_lowmask_qpose"
        if row.family == "lowmask_boundary_residual":
            if round_name == "gate0_byte_audit":
                row.decision = "close_lowmask_boundary" if row.decision == "fail_gate0_byte_audit" else row.decision
            if round_name == "gate0b_temporal_subsampling":
                row.decision = "close_temporal_subsampling" if row.decision == "fail_temporal_subsampling" else row.decision
        return row
    if round_name == "strat64":
        if row.score is not None and row.score <= 0.315:
            row.decision = "promote_full600"
            row.promotion_reason = "absolute_score_gate"
        elif row.archive_bytes is not None and row.archive_bytes <= 250_000 and row.quality is not None:
            row.decision = "relative_or_rate_candidate"
        else:
            row.decision = "reject_strat64_gate"
    elif round_name == "full600":
        if row.archive_bytes is not None and row.quality is not None:
            if row.archive_bytes <= 260_000 and row.quality <= 0.1269:
                row.decision = "promote_official_dry_run"
                row.promotion_reason = "full600_candidate_gate"
            elif row.archive_bytes <= 250_000 and row.quality <= 0.1335:
                row.decision = "promote_official_dry_run"
                row.promotion_reason = "full600_strong_candidate_gate"
            else:
                row.decision = "reject_full600_gate"
    elif round_name == "gate0_audit" and row.family == "lowmask_qpose_distill":
        if row.decision == "gate0_audit_complete":
            row.promotion_reason = "gate0_audit_ready"
    elif round_name == "gate0_byte_audit" and row.family == "lowmask_boundary_residual":
        if row.decision in {"promote_free_boundary_capacity", "diagnostic_boundary_candidate"}:
            row.promotion_reason = "boundary_residual_byte_audit"
    elif round_name == "gate0b_temporal_subsampling" and row.family == "lowmask_boundary_residual":
        if row.decision.startswith("promote"):
            row.promotion_reason = "temporal_subsampling_byte_audit"
    return row


def run(args: argparse.Namespace) -> None:
    run_id = args.run_id or f"{int(time.time())}_{args.round}"
    subset_name = ROUND_SUBSETS[args.round]
    subset_indices = get_subset(subset_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out_dir / "runs.jsonl"

    if args.promoted_only and not previous_promotions(ledger_path):
        report = {"run_id": run_id, "round": args.round, "decision": "no_promoted_candidates", "rows": []}
        write_json(args.out_dir / "reports" / f"{run_id}.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    baseline = baseline_for_subset(subset_name, subset_indices)
    baseline_rows_all = load_jsonl(QPOSE14_LEDGER)
    baseline_by_id = {int(row["sample_id"]): row for row in baseline_rows_all}
    baseline_samples = [baseline_by_id[idx] for idx in subset_indices if idx in baseline_by_id]
    ctx = {
        "run_id": run_id,
        "baseline": baseline,
        "baseline_samples": baseline_samples,
        "subset": subset_name,
        "subset_indices": subset_indices,
        "device": args.device,
        "max_steps": args.max_steps,
    }
    budget = Budget(round=args.round, subset=subset_name, max_steps=args.max_steps, max_minutes=args.max_minutes)
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    candidate_filter = {item.strip() for item in args.candidates.split(",") if item.strip()}
    cache = NegativeCache(args.negative_cache)
    rows: list[dict[str, Any]] = []

    for family in families:
        for candidate in load_family_candidates(family, args.round):
            if candidate_filter and candidate.name not in candidate_filter:
                continue
            novelty = getattr(candidate, "novelty_reason", candidate.config.get("novelty_reason", ""))
            cache_decision = cache.check(family=family, novelty_reason=novelty, allow_negative_cache=args.allow_negative_cache)
            if not cache_decision.allowed:
                row = blocked_row(
                    run_id=run_id,
                    family=family,
                    candidate_name=candidate.name,
                    config={**candidate.config, "novelty_reason": novelty},
                    reason=cache_decision.reason,
                    round_name=args.round,
                    subset=subset_name,
                )
            else:
                start = time.time()
                candidate.prepare(ctx)
                candidate.train_round(budget, ctx)
                row = candidate.decision_row(budget, ctx)
                row.extra["wall_time_sec"] = time.time() - start
                if not row.row_id:
                    row.row_id = f"{run_id}:{row.candidate_name}:{subset_name}"
                row = normalize_decision(row, round_name=args.round)
            row_baseline = baseline
            row_baseline_samples = baseline_samples
            if family == "qpose14_baseline":
                row_baseline = baseline_for_subset(subset_name, subset_indices)
                refreshed_rows = load_jsonl(QPOSE14_LEDGER)
                refreshed_by_id = {int(item["sample_id"]): item for item in refreshed_rows}
                row_baseline_samples = [refreshed_by_id[idx] for idx in subset_indices if idx in refreshed_by_id]
            row_dict = add_baseline_deltas(row.to_dict(), row_baseline, row_baseline_samples)
            append_jsonl(ledger_path, row_dict)
            rows.append(row_dict)

    report = {"run_id": run_id, "round": args.round, "subset": subset_name, "families": families, "rows": rows}
    write_json(args.out_dir / "reports" / f"{run_id}.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--families", default="qpose14_baseline")
    run_p.add_argument("--round", choices=sorted(ROUND_SUBSETS), default="smoke")
    run_p.add_argument("--promoted-only", action="store_true")
    run_p.add_argument("--allow-negative-cache", action="store_true")
    run_p.add_argument("--negative-cache", type=Path, default=Path(__file__).resolve().parent / "negative_cache.yaml")
    run_p.add_argument("--out-dir", type=Path, default=EXPERIMENTS_DIR)
    run_p.add_argument("--run-id", default="")
    run_p.add_argument("--max-steps", type=int, default=0)
    run_p.add_argument("--max-minutes", type=float, default=0.0)
    run_p.add_argument("--device", default="auto")
    run_p.add_argument("--candidates", default="", help="Optional comma-separated candidate name filter.")
    run_p.set_defaults(func=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
