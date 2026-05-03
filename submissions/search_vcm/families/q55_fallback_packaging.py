#!/usr/bin/env python
"""Fallback q55 packaging candidates."""

from __future__ import annotations

import time
from typing import Any

from submissions.search_vcm.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm.evaluator import classify_against_base, normalize_q55_metrics, Q55_METRICS


class Q55FallbackCandidate(Candidate):
    family = "q55_fallback_packaging"
    role = "fallback_candidate"
    kind = "packable_candidate"
    packable = True

    def __init__(self, variant: str):
        self.name = variant
        self.config = {"variant": variant}
        self._metrics = normalize_q55_metrics(variant, Q55_METRICS[variant])

    def prepare(self, ctx: dict[str, Any]) -> None:
        return None

    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        return None

    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        return PackageInfo(
            archive_bytes=self._metrics["archive_bytes"],
            added_bytes=self._metrics["archive_bytes"] - ctx["base"]["archive_bytes"],
            archive_path=self._metrics.get("archive_path"),
            archive_sha=self._metrics.get("archive_sha"),
            payload_breakdown={p["filename"]: int(p["file_size"]) for p in self._metrics.get("payloads", [])},
        )

    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        return Metrics(
            segnet_dist=self._metrics["segnet_dist"],
            posenet_dist=self._metrics["posenet_dist"],
            quality=self._metrics["quality"],
            seg_term=self._metrics["seg_term"],
            pose_term=self._metrics["pose_term"],
            rate_term=self._metrics["rate_term"],
            score=self._metrics["score"],
            sample_count=600,
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
            "novelty_reason": "",
            "subset": budget.subset,
            "round": budget.round,
            "archive_bytes": pkg.archive_bytes,
            "added_bytes": pkg.added_bytes,
            "quality": metrics.quality,
            "segnet_dist": metrics.segnet_dist,
            "posenet_dist": metrics.posenet_dist,
            "seg_delta": metrics.seg_term - base["seg_term"],
            "pose_delta": metrics.pose_term - base["pose_term"],
            "byte_delta": pkg.added_bytes,
            "score": metrics.score,
            "score_delta_vs_base": None if metrics.score is None else metrics.score - base["score"],
            "dominates_base": False,
            "term_tradeoff": "incomplete",
            "decision": "fallback_recorded",
            "failure_reason": "",
            "row_id": "",
            "parent_row_id": "",
            "oracle_parent": False,
            "promotion_reason": "fallback_candidate",
            "extra": {
                "source_metrics": self._metrics["source_metrics"],
                "payload_breakdown": pkg.payload_breakdown,
                "wall_time_sec": 0.0,
            },
        }
        classify_against_base(row, base)
        return DecisionRow(**row)


def candidates(round_name: str) -> list[Candidate]:
    return [
        Q55FallbackCandidate("q55_fp16_only"),
        Q55FallbackCandidate("q55_fp16_pose_int12"),
        Q55FallbackCandidate("q55_fp16_pose_int10"),
    ]
