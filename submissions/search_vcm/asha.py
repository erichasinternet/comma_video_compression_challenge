#!/usr/bin/env python
"""ASHA-style v1 runner for Search VCM."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm.candidate_api import Budget, DecisionRow, stable_hash
from submissions.search_vcm.evaluator import EXPERIMENTS_DIR, append_jsonl, base_metrics, classify_against_base, load_jsonl, write_json
from submissions.search_vcm.negative_cache import NegativeCache
from submissions.search_vcm.subsets import get_subset

FAMILY_MODULES = {
    "q55_fallback_packaging": "submissions.search_vcm.families.q55_fallback_packaging",
    "posenet_preprocess_oracle": "submissions.search_vcm.families.posenet_preprocess_oracle",
    "q55_pareto_controls": "submissions.search_vcm.families.q55_pareto_controls",
    "teacher_distilled_inflation": "submissions.search_vcm.families.teacher_distilled_inflation",
}

ROUND_SUBSETS = {
    "smoke": "smoke",
    "hard8": "hard8",
    "strat64": "strat64",
    "full600": "full600",
    "official": "full600",
}


def load_family_candidates(family: str, round_name: str):
    if family not in FAMILY_MODULES:
        raise ValueError(f"unknown family: {family}")
    module = importlib.import_module(FAMILY_MODULES[family])
    return module.candidates(round_name)


def blocked_row(*, run_id: str, family: str, candidate_name: str, config: dict, reason: str, round_name: str, subset: str) -> DecisionRow:
    return DecisionRow(
        run_id=run_id,
        candidate_name=candidate_name,
        family=family,
        role="blocked",
        kind="blocked",
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
        seg_delta=None,
        pose_delta=None,
        byte_delta=None,
        score=None,
        score_delta_vs_base=None,
        dominates_base=False,
        term_tradeoff="blocked_negative_cache",
        decision="blocked_negative_cache",
        failure_reason=reason,
        row_id="",
        parent_row_id="",
        oracle_parent=False,
        promotion_reason="",
        extra={"config": config},
    )


def apply_promotion(row: DecisionRow, *, round_name: str) -> DecisionRow:
    if row.kind == "oracle_only":
        if not row.failure_reason:
            row.decision = "diagnostic_only"
        row.term_tradeoff = "oracle_nonpackable"
        row.promotion_reason = "capacity_oracle"
        return row
    if row.role == "fallback_candidate":
        row.decision = "fallback_recorded"
        row.promotion_reason = "fallback_candidate"
        return row
    if row.decision == "blocked_negative_cache":
        return row
    if row.failure_reason:
        return row

    if round_name == "hard8":
        if row.family == "q55_pareto_controls":
            if row.score_delta_vs_base is not None and row.score_delta_vs_base <= -0.006:
                row.decision = "promote_strat64"
                row.promotion_reason = "net_score_improvement"
            elif row.quality is not None and row.quality <= 0.300:
                row.decision = "continue_diagnostic"
            else:
                row.decision = "reject_hard8_gate"
        elif row.score_delta_vs_base is not None and row.score_delta_vs_base <= -0.006:
            row.decision = "promote_strat64"
            row.promotion_reason = "net_score_improvement"
        elif row.quality is not None and row.quality <= 0.160:
            row.decision = "promote_strat64"
            row.promotion_reason = "capacity_oracle_pass"
        elif row.quality is not None and row.quality <= 0.300:
            row.decision = "continue_diagnostic"
        else:
            row.decision = "reject_hard8_gate"
    elif round_name == "strat64":
        if row.score is not None and row.score <= 0.315:
            row.decision = "promote_full600"
            row.promotion_reason = "net_score_improvement"
        else:
            row.decision = "reject_strat64_gate"
    return row


def previous_promotions(ledger_path: Path) -> list[dict]:
    rows = load_jsonl(ledger_path)
    return [row for row in rows if str(row.get("decision", "")).startswith("promote")]


def run(args: argparse.Namespace) -> None:
    run_id = args.run_id or f"{int(time.time())}_{args.round}"
    subset_name = ROUND_SUBSETS[args.round]
    subset_indices = get_subset(subset_name)
    ledger_path = args.out_dir / "runs.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.promoted_only and not previous_promotions(ledger_path):
        report = {"run_id": run_id, "round": args.round, "decision": "no_promoted_candidates", "rows": []}
        write_json(args.out_dir / "reports" / f"{run_id}.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    cache = NegativeCache(args.negative_cache)
    ctx = {
        "run_id": run_id,
        "base": base_metrics(),
        "subset": subset_name,
        "subset_indices": subset_indices,
        "device": args.device,
        "teacher_eval_every": args.teacher_eval_every,
        "teacher_batch_size": args.teacher_batch_size,
        "teacher_lr": args.teacher_lr,
        "teacher_pose_features": args.teacher_pose_features,
        "teacher_seg_features": args.teacher_seg_features,
    }
    budget = Budget(round=args.round, subset=subset_name, max_steps=args.max_steps, max_minutes=args.max_minutes)

    out_rows = []
    candidate_filter = {item.strip() for item in args.candidates.split(",") if item.strip()}
    for family in families:
        for candidate in load_family_candidates(family, args.round):
            if candidate_filter and candidate.name not in candidate_filter:
                continue
            config_id = candidate.config.get("config_id") or candidate.config.get("variant") or candidate.name
            novelty = getattr(candidate, "novelty_reason", candidate.config.get("novelty_reason", ""))
            cache_decision = cache.check(
                family=family,
                config_id=config_id,
                novelty_reason=novelty,
                allow_negative_cache=args.allow_negative_cache,
            )
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
                if not row.row_id:
                    row.row_id = f"{run_id}:{row.candidate_name}:{subset_name}"
                row.extra["wall_time_sec"] = time.time() - start
                row = apply_promotion(row, round_name=args.round)
                classify_against_base(row.extra.setdefault("_row_shadow", row.to_dict()), ctx["base"])
                row.extra.pop("_row_shadow", None)
            append_jsonl(ledger_path, row.to_dict())
            out_rows.append(row.to_dict())

    report = {
        "run_id": run_id,
        "round": args.round,
        "subset": subset_name,
        "families": families,
        "rows": out_rows,
    }
    write_json(args.out_dir / "reports" / f"{run_id}.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--families", default="q55_fallback_packaging,posenet_preprocess_oracle,q55_pareto_controls")
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
    run_p.add_argument("--teacher-eval-every", type=int, default=0)
    run_p.add_argument("--teacher-batch-size", type=int, default=0)
    run_p.add_argument("--teacher-lr", type=float, default=0.0)
    run_p.add_argument("--teacher-pose-features", default="summarizer")
    run_p.add_argument("--teacher-seg-features", default="")
    run_p.set_defaults(func=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
