#!/usr/bin/env python
"""Config-level negative-cache enforcement."""

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
        self.dead_families = set(self.data.get("dead_families", []))
        self.dead_configs = self.data.get("dead_configs", {}) or {}
        self.dead_reasons = self.data.get("dead_reasons", {}) or {}

    def check(
        self,
        *,
        family: str,
        config_id: str,
        novelty_reason: str = "",
        allow_negative_cache: bool = False,
    ) -> NegativeCacheDecision:
        if allow_negative_cache:
            return NegativeCacheDecision(True, "override_allow_negative_cache")
        if family in self.dead_families:
            reason = self.dead_reasons.get(family, "")
            suffix = f":{reason}" if reason else ""
            return NegativeCacheDecision(False, f"dead_family:{family}{suffix}")
        dead_configs = set(self.dead_configs.get(family, []))
        if config_id in dead_configs and not novelty_reason.strip():
            return NegativeCacheDecision(False, f"dead_config_requires_novelty_reason:{family}/{config_id}")
        return NegativeCacheDecision(True, "allowed")
