#!/usr/bin/env python
"""Common candidate interface and row schemas for Search VCM."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RoundName = Literal["smoke", "hard8", "strat64", "full600", "official"]


def stable_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class Budget:
    round: RoundName
    subset: str
    max_steps: int = 0
    max_minutes: float = 0.0


@dataclass(frozen=True)
class PackageInfo:
    archive_bytes: int | None
    added_bytes: int = 0
    payload_breakdown: dict[str, int] = field(default_factory=dict)
    archive_path: str | None = None
    archive_sha: str | None = None
    projected: bool = False


@dataclass(frozen=True)
class Metrics:
    segnet_dist: float
    posenet_dist: float
    quality: float
    seg_term: float
    pose_term: float
    rate_term: float = 0.0
    score: float | None = None
    sample_count: int = 0
    per_sample: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DecisionRow:
    run_id: str
    candidate_name: str
    family: str
    role: str
    kind: str
    packable: bool
    config_hash: str
    novelty_reason: str
    subset: str
    round: str
    archive_bytes: int | None
    added_bytes: int
    quality: float | None
    segnet_dist: float | None
    posenet_dist: float | None
    seg_delta: float | None
    pose_delta: float | None
    byte_delta: int | None
    score: float | None
    score_delta_vs_base: float | None
    dominates_base: bool
    term_tradeoff: str
    decision: str
    failure_reason: str
    row_id: str = ""
    parent_row_id: str = ""
    oracle_parent: bool = False
    promotion_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Candidate(ABC):
    name: str
    family: str
    role: str = "exploratory_candidate"
    kind: str = "packable_candidate"
    packable: bool = True
    config: dict[str, Any]

    @abstractmethod
    def prepare(self, ctx: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        raise NotImplementedError

    @abstractmethod
    def decision_row(self, budget: Budget, ctx: dict[str, Any]) -> DecisionRow:
        raise NotImplementedError
