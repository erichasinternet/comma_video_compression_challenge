#!/usr/bin/env python3
"""Small HNeRV-style frame-index renderer for task-aware neural video."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_H, MODEL_W = 384, 512


@dataclass
class HNeRVConfig:
    n_frames: int
    embed_dim: int = 64
    hidden: int = 128
    base_h: int = 12
    base_w: int = 16
    num_blocks: int = 5
    mlp_hidden: int = 256


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 16 if channels >= 16 else 1
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        residual = x
        x = self.dw(x)
        x = self.pw(F.silu(x))
        x = self.norm(x)
        return F.silu(x + residual)


class HNeRVRenderer(nn.Module):
    """Map frame ids to RGB frames.

    This is intentionally simple for capacity tests: every frame has a learned
    embedding, and a shared upsampling renderer maps embeddings to images.
    """

    def __init__(self, cfg: HNeRVConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.n_frames, cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.mlp_hidden),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden, cfg.hidden * cfg.base_h * cfg.base_w),
        )
        self.blocks = nn.ModuleList([UpsampleBlock(cfg.hidden) for _ in range(cfg.num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1, groups=cfg.hidden),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden, 3, 1),
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, frame_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        z = self.embedding(frame_ids)
        x = self.mlp(z).view(frame_ids.shape[0], cfg.hidden, cfg.base_h, cfg.base_w)
        x = F.silu(x)
        for block in self.blocks:
            x = block(x)
        if x.shape[-2:] != (MODEL_H, MODEL_W):
            x = F.interpolate(x, size=(MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x)) * 255.0

    def render_pairs(self, rows: torch.Tensor) -> torch.Tensor:
        frame_ids = torch.stack([rows * 2, rows * 2 + 1], dim=1).flatten()
        frames = self(frame_ids)
        return frames.view(rows.shape[0], 2, 3, MODEL_H, MODEL_W)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

