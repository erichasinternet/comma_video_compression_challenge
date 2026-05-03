#!/usr/bin/env python
"""Factorized exact-mask renderer for Search VCM v2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord_grid(batch: int, height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def mask_boundary_map(mask: torch.Tensor) -> torch.Tensor:
    """Return a [B,1,H,W] binary-ish boundary map from [B,H,W] class masks."""

    if mask.ndim != 3:
        raise ValueError(f"expected [B,H,W] mask, got {tuple(mask.shape)}")
    b, h, w = mask.shape
    boundary = torch.zeros(b, 1, h, w, device=mask.device, dtype=torch.float32)
    boundary[..., 1:, :] = torch.maximum(boundary[..., 1:, :], (mask[:, 1:, :] != mask[:, :-1, :]).float().unsqueeze(1))
    boundary[..., :-1, :] = torch.maximum(boundary[..., :-1, :], (mask[:, 1:, :] != mask[:, :-1, :]).float().unsqueeze(1))
    boundary[..., :, 1:] = torch.maximum(boundary[..., :, 1:], (mask[:, :, 1:] != mask[:, :, :-1]).float().unsqueeze(1))
    boundary[..., :, :-1] = torch.maximum(boundary[..., :, :-1], (mask[:, :, 1:] != mask[:, :, :-1]).float().unsqueeze(1))
    return boundary


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, *, cond_dim: int | None = None):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(max(1, min(8, channels // 4)), channels)
        self.act = nn.SiLU(inplace=True)
        self.film = nn.Linear(cond_dim, channels * 2) if cond_dim else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm(self.pw(self.dw(x)))
        if self.film is not None:
            if cond is None:
                raise ValueError("conditional block requires cond")
            gamma, beta = self.film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            y = y * (1.0 + gamma) + beta
        return self.act(x + y)


class FactorizedExactMaskRenderer(nn.Module):
    """Two-head renderer from exact mask plus pose/z tokens.

    The capacity configuration intentionally supports oversized hidden/z values.
    Compressed variants use the same class with smaller hidden, blocks, and z dim.
    """

    def __init__(
        self,
        *,
        mask_emb_dim: int = 16,
        hidden: int = 128,
        blocks: int = 6,
        z_pose_dim: int = 256,
        pose_dim: int = 6,
        pose_map_hw: tuple[int, int] | None = (16, 24),
    ):
        super().__init__()
        self.mask_emb_dim = mask_emb_dim
        self.hidden = hidden
        self.blocks = blocks
        self.z_pose_dim = z_pose_dim
        self.pose_dim = pose_dim
        self.pose_map_hw = pose_map_hw

        extra_pose_map_ch = 8 if pose_map_hw else 0
        self.embedding = nn.Embedding(5, mask_emb_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(mask_emb_dim + 3 + extra_pose_map_ch, hidden, 3, padding=1),
            nn.GroupNorm(max(1, min(8, hidden // 4)), hidden),
            nn.SiLU(inplace=True),
        )
        cond_dim = hidden
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim + z_pose_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, cond_dim),
        )
        self.pose_map = (
            nn.Sequential(nn.Linear(z_pose_dim, 8 * pose_map_hw[0] * pose_map_hw[1]), nn.SiLU(inplace=True))
            if pose_map_hw
            else None
        )
        self.shared = nn.ModuleList([DepthwiseSeparableBlock(hidden, cond_dim=cond_dim) for _ in range(blocks)])
        self.frame2_blocks = nn.ModuleList([DepthwiseSeparableBlock(hidden) for _ in range(2)])
        self.frame1_blocks = nn.ModuleList([DepthwiseSeparableBlock(hidden, cond_dim=cond_dim) for _ in range(2)])
        self.frame2_head = nn.Conv2d(hidden, 3, 1)
        self.frame1_head = nn.Conv2d(hidden, 3, 1)

    def config(self) -> dict:
        return {
            "mask_emb_dim": self.mask_emb_dim,
            "hidden": self.hidden,
            "blocks": self.blocks,
            "z_pose_dim": self.z_pose_dim,
            "pose_dim": self.pose_dim,
            "pose_map_hw": self.pose_map_hw,
        }

    def forward(self, mask: torch.Tensor, pose6: torch.Tensor, z_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.ndim != 3:
            raise ValueError(f"expected mask [B,H,W], got {tuple(mask.shape)}")
        b, h, w = mask.shape
        emb = self.embedding(mask.long()).permute(0, 3, 1, 2)
        coords = make_coord_grid(b, h, w, mask.device, emb.dtype)
        boundary = mask_boundary_map(mask).to(dtype=emb.dtype)
        inputs = [emb, coords, boundary]
        cond = self.pose_mlp(torch.cat([pose6.float(), z_pose.float()], dim=1))
        if self.pose_map is not None:
            ph, pw = self.pose_map_hw or (1, 1)
            zmap = self.pose_map(z_pose.float()).reshape(b, 8, ph, pw)
            zmap = F.interpolate(zmap, size=(h, w), mode="bilinear", align_corners=False)
            inputs.append(zmap)
        x = self.stem(torch.cat(inputs, dim=1))
        for block in self.shared:
            x = block(x, cond)
        x2 = x
        for block in self.frame2_blocks:
            x2 = block(x2)
        x1 = x
        for block in self.frame1_blocks:
            x1 = block(x1, cond)
        frame2 = torch.sigmoid(self.frame2_head(x2)) * 255.0
        frame1 = torch.sigmoid(self.frame1_head(x1)) * 255.0
        return frame1, frame2


def build_renderer_config(name: str) -> dict:
    configs = {
        "capacity": {"mask_emb_dim": 16, "hidden": 128, "blocks": 6, "z_pose_dim": 256, "pose_map_hw": (16, 24)},
        "F16": {"mask_emb_dim": 8, "hidden": 16, "blocks": 3, "z_pose_dim": 16, "pose_map_hw": (8, 12)},
        "F24": {"mask_emb_dim": 8, "hidden": 24, "blocks": 3, "z_pose_dim": 24, "pose_map_hw": (8, 12)},
        "F32": {"mask_emb_dim": 12, "hidden": 32, "blocks": 4, "z_pose_dim": 32, "pose_map_hw": (8, 12)},
    }
    if name not in configs:
        raise ValueError(f"unknown renderer config: {name}")
    return configs[name]


def build_renderer(name: str = "capacity") -> FactorizedExactMaskRenderer:
    return FactorizedExactMaskRenderer(**build_renderer_config(name))

