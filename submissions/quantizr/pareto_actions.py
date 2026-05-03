#!/usr/bin/env python
"""Packable low-byte action surfaces for the Pareto control lab."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionSpec:
    name: str
    target: str
    description: str


class ParetoAction(torch.nn.Module):
    spec = ActionSpec("base", "none", "No-op action")

    def estimate_bytes(self, selected_samples: int | None = None) -> int:
        raise NotImplementedError

    def pack_dict(self) -> dict:
        return {"name": self.spec.name, "estimated_bytes": self.estimate_bytes()}

    def apply(self, frames: torch.Tensor, masks: torch.Tensor | None, sample_ids: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def regularizer(self) -> torch.Tensor:
        params = list(self.parameters())
        if not params:
            return torch.zeros([])
        return torch.stack([p.pow(2).mean() for p in params]).mean()


class C1ClassColorLUT(ParetoAction):
    spec = ActionSpec(
        "c1_class_color_lut",
        "segnet",
        "Frame2 class-conditioned RGB offset using exact q55 mask classes.",
    )

    def __init__(self, max_delta: float = 24.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.raw = torch.nn.Parameter(torch.zeros(5, 3))

    def estimate_bytes(self, selected_samples: int | None = None) -> int:
        # int8 delta table plus tiny metadata.
        return 5 * 3 + 64

    def apply(self, frames: torch.Tensor, masks: torch.Tensor | None, sample_ids: torch.Tensor | None = None) -> torch.Tensor:
        if masks is None:
            raise ValueError("C1 requires exact class masks")
        out = frames.clone()
        delta = self.max_delta * torch.tanh(self.raw).to(frames.device, frames.dtype)
        lut = delta[masks.long()].permute(0, 3, 1, 2)
        out[:, 1] = (out[:, 1] + lut).clamp(0.0, 255.0)
        return out

    def pack_dict(self) -> dict:
        q = (self.max_delta * torch.tanh(self.raw.detach())).round().clamp(-128, 127).to(torch.int8)
        return {
            **super().pack_dict(),
            "max_delta": self.max_delta,
            "delta_int8": q.cpu().tolist(),
        }


class _GridResidualBase(ParetoAction):
    frame_count = 1

    def __init__(self, sample_count: int, grid_hw: tuple[int, int] = (12, 16), max_delta: float = 18.0):
        super().__init__()
        self.sample_count = int(sample_count)
        self.grid_hw = tuple(int(x) for x in grid_hw)
        self.max_delta = float(max_delta)
        self.raw = torch.nn.Parameter(torch.zeros(self.sample_count, self.frame_count, 3, *self.grid_hw))

    def estimate_bytes(self, selected_samples: int | None = None) -> int:
        n = self.sample_count if selected_samples is None else int(selected_samples)
        # int8 grid coefficients plus a small per-action header.
        return n * self.frame_count * 3 * self.grid_hw[0] * self.grid_hw[1] + 128

    def _residual(self, frames: torch.Tensor, sample_ids: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.raw
        if sample_ids is not None:
            raw = raw.index_select(0, sample_ids.to(raw.device).long())
        residual = self.max_delta * torch.tanh(raw).to(frames.device, frames.dtype)
        sample_count = residual.shape[0]
        return F.interpolate(
            residual.flatten(0, 1),
            size=frames.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (sample_count, self.frame_count))

    def pack_dict(self) -> dict:
        q = (self.max_delta * torch.tanh(self.raw.detach())).round().clamp(-128, 127).to(torch.int8)
        return {
            **super().pack_dict(),
            "sample_count": self.sample_count,
            "grid_hw": list(self.grid_hw),
            "max_delta": self.max_delta,
            "residual_int8_shape": list(q.shape),
        }


class C2Frame2LowRankResidual(_GridResidualBase):
    spec = ActionSpec(
        "c2_frame2_lowrank_residual",
        "segnet",
        "Per-sample low-frequency frame2 residual grid with PoseNet hard cap.",
    )
    frame_count = 1

    def apply(self, frames: torch.Tensor, masks: torch.Tensor | None, sample_ids: torch.Tensor | None = None) -> torch.Tensor:
        out = frames.clone()
        out[:, 1] = (out[:, 1] + self._residual(frames, sample_ids)[:, 0]).clamp(0.0, 255.0)
        return out


class C3PairedLowFreqResidual(_GridResidualBase):
    spec = ActionSpec(
        "c3_paired_lowfreq_residual",
        "segnet+posenet",
        "Per-sample paired low-frequency residual grids for coherent frame1/frame2 moves.",
    )
    frame_count = 2

    def apply(self, frames: torch.Tensor, masks: torch.Tensor | None, sample_ids: torch.Tensor | None = None) -> torch.Tensor:
        return (frames + self._residual(frames, sample_ids)).clamp(0.0, 255.0)


class C4PoseCtrlCompensator(ParetoAction):
    spec = ActionSpec(
        "c4_pose_ctrl_compensator",
        "posenet",
        "Scaffold for pose-table compensation; only valid jointly with C2/C3 packaging.",
    )

    def estimate_bytes(self, selected_samples: int | None = None) -> int:
        n = 0 if selected_samples is None else int(selected_samples)
        return n * 6 * 2 + 128

    def apply(self, frames: torch.Tensor, masks: torch.Tensor | None, sample_ids: torch.Tensor | None = None) -> torch.Tensor:
        return frames


class C5ActionRouter:
    """Simple score-aware row selector for validated action rows."""

    def __init__(self, base_archive_bytes: int, original_bytes: int = 37_545_489):
        self.base_archive_bytes = int(base_archive_bytes)
        self.original_bytes = int(original_bytes)

    def score_delta(self, quality_delta: float, added_bytes: int) -> float:
        return float(quality_delta) + 25.0 * int(added_bytes) / self.original_bytes


def build_action(name: str, *, sample_count: int, grid_hw: tuple[int, int], max_delta: float) -> ParetoAction:
    if name == "c1":
        return C1ClassColorLUT(max_delta=max_delta)
    if name == "c2":
        return C2Frame2LowRankResidual(sample_count=sample_count, grid_hw=grid_hw, max_delta=max_delta)
    if name == "c3":
        return C3PairedLowFreqResidual(sample_count=sample_count, grid_hw=grid_hw, max_delta=max_delta)
    if name == "c4":
        return C4PoseCtrlCompensator()
    raise ValueError(f"unknown action {name}")
