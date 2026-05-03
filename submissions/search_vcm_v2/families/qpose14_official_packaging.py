#!/usr/bin/env python
"""qpose14 packaging reference/fallback rows."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm_v2.evaluator import qpose14_reference_summary, qpose14_summary


class QPose14OfficialPackagingCandidate(Candidate):
    name = "qpose14_official_packaging"
    family = "qpose14_official_packaging"
    role = "fallback_candidate"
    kind = "packaging_reference"
    packable = True

    def __init__(self) -> None:
        self.config = {"config_id": self.name, "source": "qpose14_pr63_reference"}

    def prepare(self, ctx: dict) -> None:
        return None

    def train_round(self, budget: Budget, ctx: dict) -> None:
        return None

    def package(self, ctx: dict) -> PackageInfo:
        summary = qpose14_summary()
        return PackageInfo(
            archive_bytes=int(summary.get("archive_bytes", qpose14_reference_summary()["archive_bytes"])),
            payload_breakdown={"archive_zip": int(summary.get("archive_bytes", qpose14_reference_summary()["archive_bytes"]))},
            projected=True,
        )

    def evaluate(self, subset: str, ctx: dict) -> Metrics:
        summary = qpose14_summary()
        subset_summary = summary.get("subset_summaries", {}).get(subset)
        if subset_summary:
            return Metrics(
                segnet_dist=float(subset_summary["segnet_dist"]),
                posenet_dist=float(subset_summary["posenet_dist"]),
                quality=float(subset_summary["quality"]),
                seg_term=float(subset_summary["seg_term"]),
                pose_term=float(subset_summary["pose_term"]),
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

    def decision_row(self, budget: Budget, ctx: dict) -> DecisionRow:
        package = self.package(ctx)
        metrics = self.evaluate(budget.subset, ctx)
        baseline = ctx.get("baseline", qpose14_reference_summary())
        score_delta = None
        if metrics.score is not None and baseline.get("score") is not None:
            score_delta = float(metrics.score) - float(baseline["score"])
        return DecisionRow(
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
            score_delta_vs_base=score_delta,
            dominates_base=score_delta is not None and score_delta <= 0.0,
            term_tradeoff="fallback_packaging_reference",
            decision="fallback_recorded",
            failure_reason="",
            promotion_reason="fallback_candidate",
            extra={"note": "Reference row only; official dry run requires the real qpose14 archive artifact."},
        )


def candidates(round_name: str) -> list[Candidate]:
    return [QPose14OfficialPackagingCandidate()]
