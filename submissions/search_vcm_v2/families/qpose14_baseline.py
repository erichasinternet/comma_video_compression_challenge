#!/usr/bin/env python
"""qpose14 baseline integration family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm_v2.evaluator import (
    QPOSE14_LEDGER,
    QPOSE14_SUMMARY,
    add_baseline_deltas,
    load_json,
    qpose14_reference_summary,
    qpose14_summary,
)
from submissions.search_vcm_v2.ledger import SOURCE_Q55_V1_LEDGER, build_qpose14_baseline


class QPose14BaselineCandidate(Candidate):
    name = "qpose14_baseline"
    family = "qpose14_baseline"
    role = "baseline"
    kind = "baseline_reference"
    packable = True

    def __init__(self) -> None:
        self.config = {"config_id": self.name, "source": "qpose14_pr63_reference_plus_proxy_ledger"}
        self.summary: dict[str, Any] | None = None

    def prepare(self, ctx: dict[str, Any]) -> None:
        if not QPOSE14_SUMMARY.exists() or not QPOSE14_LEDGER.exists():
            args = argparse.Namespace(source_ledger=SOURCE_Q55_V1_LEDGER, out_dir=QPOSE14_SUMMARY.parent)
            self.summary = build_qpose14_baseline(args)
        else:
            self.summary = load_json(QPOSE14_SUMMARY)

    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        return None

    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        summary = self.summary or qpose14_summary()
        return PackageInfo(
            archive_bytes=int(summary["archive_bytes"]),
            payload_breakdown={"archive_zip": int(summary["archive_bytes"])},
            projected=str(summary.get("artifact_status", "")).startswith("missing"),
        )

    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        summary = self.summary or qpose14_summary()
        subset_summary = summary.get("subset_summaries", {}).get(subset)
        if subset_summary:
            return Metrics(
                segnet_dist=float(subset_summary["segnet_dist"]),
                posenet_dist=float(subset_summary["posenet_dist"]),
                quality=float(subset_summary["quality"]),
                seg_term=float(subset_summary["seg_term"]),
                pose_term=float(subset_summary["pose_term"]),
                rate_term=float(subset_summary.get("rate_term", 0.0)),
                score=float(subset_summary["score"]),
                sample_count=int(subset_summary["sample_count"]),
            )
        reference = qpose14_reference_summary()
        return Metrics(
            segnet_dist=float(reference["segnet_dist"]),
            posenet_dist=float(reference["posenet_dist"]),
            quality=float(reference["quality"]),
            seg_term=float(reference["seg_term"]),
            pose_term=float(reference["pose_term"]),
            rate_term=float(reference["rate_term"]),
            score=float(reference["score"]),
            sample_count=0,
        )

    def decision_row(self, budget: Budget, ctx: dict[str, Any]) -> DecisionRow:
        summary = self.summary or qpose14_summary()
        metrics = self.evaluate(budget.subset, ctx)
        package = self.package(ctx)
        row = DecisionRow(
            run_id=ctx["run_id"],
            candidate_name=self.name,
            family=self.family,
            role=self.role,
            kind=self.kind,
            packable=self.packable,
            config_hash=stable_hash(self.config),
            novelty_reason="",
            subset=budget.subset,
            round=budget.round,
            archive_bytes=package.archive_bytes,
            added_bytes=0,
            quality=metrics.quality,
            segnet_dist=metrics.segnet_dist,
            posenet_dist=metrics.posenet_dist,
            score=metrics.score,
            seg_delta=0.0,
            pose_delta=0.0,
            byte_delta=0,
            score_delta_vs_base=0.0,
            dominates_base=True,
            term_tradeoff="baseline",
            decision="baseline_recorded",
            failure_reason="",
            promotion_reason="baseline_recorded",
            extra={
                "summary_path": str(QPOSE14_SUMMARY),
                "per_sample_ledger": str(QPOSE14_LEDGER),
                "artifact_status": summary.get("artifact_status"),
                "ledger_source": summary.get("ledger_source"),
            },
        )
        add_baseline_deltas(row.extra.setdefault("_row_shadow", row.to_dict()), summary)
        row.extra.pop("_row_shadow", None)
        return row


def candidates(round_name: str) -> list[Candidate]:
    return [QPose14BaselineCandidate()]
