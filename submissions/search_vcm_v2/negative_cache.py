#!/usr/bin/env python
"""Negative-cache enforcement for Search VCM v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_CACHE = Path(__file__).resolve().parent / "negative_cache.yaml"


@dataclass(frozen=True)
class NegativeCacheDecision:
    allowed: bool
    reason: str


class NegativeCache:
    def __init__(self, path: Path = DEFAULT_CACHE):
        self.path = path
        self.data = yaml.safe_load(path.read_text()) if path.exists() else {}
        self.allowed = set(self.data.get("allowed_v2_families", []))
        self.dead = self.data.get("dead_families", {}) or {}

    def check(self, *, family: str, novelty_reason: str = "", allow_negative_cache: bool = False) -> NegativeCacheDecision:
        if family in self.dead:
            if allow_negative_cache and novelty_reason.strip():
                return NegativeCacheDecision(True, "override_dead_family_with_novelty")
            reason = self.dead[family].get("reason", "") if isinstance(self.dead[family], dict) else str(self.dead[family])
            return NegativeCacheDecision(False, f"dead_family:{family}:{reason}")
        if self.allowed and family not in self.allowed:
            return NegativeCacheDecision(False, f"family_not_allowed_v2:{family}")
        return NegativeCacheDecision(True, "allowed")

