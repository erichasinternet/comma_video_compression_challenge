#!/usr/bin/env python
"""Constrained q55 Pareto control candidate declarations."""

from __future__ import annotations

from typing import Any

from submissions.search_vcm.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm.evaluator import classify_against_base, score


class Q55ParetoControlCandidate(Candidate):
    family = "q55_pareto_controls"
    role = "exploratory_candidate"
    kind = "packable_candidate"
    packable = True

    def __init__(self, name: str, config: dict[str, Any], novelty_reason: str, added_bytes: int):
        self.name = name
        self.config = config
        self.novelty_reason = novelty_reason
        self.added_bytes = added_bytes

    def prepare(self, ctx: dict[str, Any]) -> None:
        return None

    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        return None

    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        return PackageInfo(
            archive_bytes=ctx["base"]["archive_bytes"] + self.added_bytes,
            added_bytes=self.added_bytes,
            projected=True,
        )

    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        base = ctx["base"]
        archive_bytes = base["archive_bytes"] + self.added_bytes
        return Metrics(
            segnet_dist=base["segnet_dist"],
            posenet_dist=base["posenet_dist"],
            quality=base["quality"],
            seg_term=base["seg_term"],
            pose_term=base["pose_term"],
            rate_term=25.0 * archive_bytes / 37_545_489,
            score=score(base["segnet_dist"], base["posenet_dist"], archive_bytes),
            sample_count=len(ctx.get("subset_indices", [])),
        )

    def decision_row(self, budget: Budget, ctx: dict[str, Any]) -> DecisionRow:
        base = ctx["base"]
        pkg = self.package(ctx)
        metrics = self.evaluate(budget.subset, ctx)
        row = {
            "run_id": ctx["run_id"],
            "candidate_name": self.name,
            "family": self.family,
            "role": self.role,
            "kind": self.kind,
            "packable": self.packable,
            "config_hash": stable_hash(self.config),
            "novelty_reason": self.novelty_reason,
            "subset": budget.subset,
            "round": budget.round,
            "archive_bytes": pkg.archive_bytes,
            "added_bytes": pkg.added_bytes,
            "quality": metrics.quality,
            "segnet_dist": metrics.segnet_dist,
            "posenet_dist": metrics.posenet_dist,
            "seg_delta": 0.0,
            "pose_delta": 0.0,
            "byte_delta": pkg.added_bytes,
            "score": metrics.score,
            "score_delta_vs_base": None if metrics.score is None else metrics.score - base["score"],
            "dominates_base": False,
            "term_tradeoff": "not_promotable",
            "decision": "registered_not_promoted",
            "failure_reason": "candidate requires optimizer execution before promotion",
            "row_id": "",
            "parent_row_id": "",
            "oracle_parent": False,
            "promotion_reason": "",
            "extra": {
                "config": self.config,
                "hard8_gate": "net_score_improvement>=0.006 with projected bytes included",
                "tail_gate": "reject_top5_quality_delta>0.050",
            },
        }
        classify_against_base(row, base)
        return DecisionRow(**row)


def candidates(round_name: str) -> list[Candidate]:
    if round_name not in {"smoke", "hard8"}:
        return []
    return [
        Q55ParetoControlCandidate(
            "c1_global_lut_v1",
            {"control": "c1", "variant": "global_lut", "payload_limit": 1024},
            "new_global_lut_payload_limited_not_existing_class_lut_run",
            added_bytes=256,
        ),
        Q55ParetoControlCandidate(
            "c1_vertical_band_lut_v1",
            {"control": "c1", "variant": "vertical_band_lut", "bands": 8, "payload_limit": 1024},
            "vertical_band_lut_not_previously_audited",
            added_bytes=768,
        ),
        Q55ParetoControlCandidate(
            "c3_shared_basis_pairres_v1",
            {"control": "c3", "variant": "shared_basis", "basis_count": 8, "payload_limit": 10_000},
            "shared_basis_global_payload_not_per_sample_grid",
            added_bytes=10_000,
        ),
    ]
