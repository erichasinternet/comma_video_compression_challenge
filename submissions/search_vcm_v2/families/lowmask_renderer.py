#!/usr/bin/env python
"""Low-rate mask renderer variants for Search VCM v2."""

from __future__ import annotations

from submissions.search_vcm_v2.families.factorized_renderer import FactorizedExactMaskRenderer


LOWMASK_RENDERER_CONFIGS = {
    "capacity": {"mask_emb_dim": 24, "hidden": 128, "blocks": 6, "z_pose_dim": 128, "pose_map_hw": (16, 24)},
    "L48": {"mask_emb_dim": 12, "hidden": 48, "blocks": 4, "z_pose_dim": 24, "pose_map_hw": (8, 12)},
    "L40": {"mask_emb_dim": 12, "hidden": 40, "blocks": 4, "z_pose_dim": 24, "pose_map_hw": (8, 12)},
    "L32": {"mask_emb_dim": 8, "hidden": 32, "blocks": 3, "z_pose_dim": 16, "pose_map_hw": (8, 12)},
}


def build_lowmask_renderer_config(name: str) -> dict:
    if name not in LOWMASK_RENDERER_CONFIGS:
        raise ValueError(f"unknown lowmask renderer config: {name}")
    return dict(LOWMASK_RENDERER_CONFIGS[name])


def build_lowmask_renderer(name: str = "capacity") -> FactorizedExactMaskRenderer:
    return FactorizedExactMaskRenderer(**build_lowmask_renderer_config(name))
