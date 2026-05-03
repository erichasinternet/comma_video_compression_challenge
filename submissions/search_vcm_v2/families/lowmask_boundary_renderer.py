#!/usr/bin/env python
"""Placeholder renderer API for the lowmask boundary family.

Phase A-C intentionally stop before GPU renderer training. This module exists so
future Gate 1 code has a stable import target without making the CPU audit
pretend to be a trainable candidate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryRendererConfig:
    hidden: int = 128
    blocks: int = 6
    lowmask_embedding_dim: int = 32
    boundary_embedding_dim: int = 32
    z_pose_dim: int = 128


def capacity_config() -> BoundaryRendererConfig:
    return BoundaryRendererConfig()


__all__ = ["BoundaryRendererConfig", "capacity_config"]
