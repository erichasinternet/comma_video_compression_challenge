#!/usr/bin/env python3
"""Shared helpers for the commaVQ-token task renderer prototype."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frame_utils import AVVideoDataset
from modules import DistortionNet, posenet_sd_path, segnet_sd_path


ORIGINAL_BYTES = 37_545_489
MODEL_HW = (384, 512)
DEFAULT_HARD8 = [59, 60, 62, 56, 57, 58, 61, 63]


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(max(0.0, 10.0 * posenet_dist))


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + 25.0 * archive_bytes / ORIGINAL_BYTES


def required_quality_for_score(archive_bytes: int, target: float = 0.300) -> float:
    return target - 25.0 * archive_bytes / ORIGINAL_BYTES


def parse_indices(text: str | None, *, offset: int, subset: int, preset: str) -> list[int]:
    if text:
        return [int(x) for x in text.replace(" ", "").split(",") if x]
    if preset == "hard8":
        return list(DEFAULT_HARD8)
    return list(range(offset, offset + subset))


def load_original_pairs_by_indices(
    *,
    data_dir: Path,
    video_names_file: Path,
    sample_indices: list[int],
    batch_size: int,
) -> torch.Tensor:
    requested = list(sample_indices)
    order = {idx: pos for pos, idx in enumerate(requested)}
    sorted_indices = sorted(requested)
    max_index = sorted_indices[-1]
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    ds = AVVideoDataset(names, data_dir=data_dir, batch_size=batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    out: list[torch.Tensor | None] = [None] * len(requested)
    seen = 0
    ptr = 0
    for _, _, batch in ds:
        batch_count = batch.shape[0]
        batch_start, batch_end = seen, seen + batch_count
        seen = batch_end
        while ptr < len(sorted_indices) and sorted_indices[ptr] < batch_end:
            sample_id = sorted_indices[ptr]
            if sample_id >= batch_start:
                out[order[sample_id]] = batch[sample_id - batch_start].contiguous()
            ptr += 1
        if seen > max_index:
            break
    if any(item is None for item in out):
        missing = [requested[i] for i, item in enumerate(out) if item is None]
        raise RuntimeError(f"failed to load samples: {missing}")
    return torch.stack([item for item in out if item is not None], dim=0)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def hard_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    target_logits = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    target_mask = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).bool()
    other_logits = logits.masked_fill(target_mask, -1.0e4).amax(dim=1)
    return F.relu(margin - (target_logits - other_logits)).mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    unit = x / 255.0
    return (unit[..., 1:, :] - unit[..., :-1, :]).abs().mean() + (
        unit[..., :, 1:] - unit[..., :, :-1]
    ).abs().mean()


class FeatureTap:
    def __init__(self, model: torch.nn.Module, names: list[str]):
        self.names = [name for name in names if name]
        modules = dict(model.named_modules())
        missing = [name for name in self.names if name not in modules]
        if missing:
            raise KeyError(f"missing feature modules: {missing}")
        self.features: dict[str, torch.Tensor] = {}
        self.handles = [modules[name].register_forward_hook(self._hook(name)) for name in self.names]

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.features[name] = output
        return hook

    def clear(self) -> None:
        self.features = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def feature_loss(got: dict[str, torch.Tensor], target: dict[str, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
    if not got or not target:
        return torch.zeros([], device=rows.device)
    losses = []
    for name, value in got.items():
        if name not in target:
            continue
        tgt = target[name].to(device=value.device, dtype=value.dtype).index_select(0, rows)
        denom = tgt.detach().pow(2).mean().clamp_min(1.0e-4)
        losses.append((value - tgt).pow(2).mean() / denom)
    if not losses:
        return torch.zeros([], device=rows.device)
    return torch.stack(losses).mean()


def build_distortion(device: torch.device) -> DistortionNet:
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in distortion.parameters():
        p.requires_grad_(False)
    return distortion


@torch.inference_mode()
def collect_targets(
    *,
    distortion: DistortionNet,
    original_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seg_tap: FeatureTap,
    pose_tap: FeatureTap,
) -> dict:
    seg_logits = []
    pose_targets = []
    seg_features: dict[str, list[torch.Tensor]] = {name: [] for name in seg_tap.names}
    pose_features: dict[str, list[torch.Tensor]] = {name: [] for name in pose_tap.names}
    for start in range(0, original_cpu.shape[0], batch_size):
        batch = original_cpu[start : start + batch_size].to(device).permute(0, 1, 4, 2, 3).float()
        seg_tap.clear()
        pose_tap.clear()
        seg_out = distortion.segnet(distortion.segnet.preprocess_input(batch))
        pose_out = distortion.posenet(distortion.posenet.preprocess_input(batch))["pose"][..., :6]
        seg_logits.append(seg_out.detach().cpu())
        pose_targets.append(pose_out.detach().cpu())
        for name, feat in seg_tap.features.items():
            seg_features[name].append(feat.detach().cpu())
        for name, feat in pose_tap.features.items():
            pose_features[name].append(feat.detach().cpu())
    logits = torch.cat(seg_logits, dim=0)
    return {
        "seg_logits": logits,
        "seg_argmax": logits.argmax(dim=1),
        "seg_prob": logits.softmax(dim=1),
        "pose": torch.cat(pose_targets, dim=0),
        "seg_features": {name: torch.cat(items, dim=0) for name, items in seg_features.items() if items},
        "pose_features": {name: torch.cat(items, dim=0) for name, items in pose_features.items() if items},
    }


@torch.inference_mode()
def evaluate_frames(
    *,
    frames: torch.Tensor,
    targets: dict,
    distortion: DistortionNet,
    batch_size: int,
) -> dict:
    total_seg = 0.0
    total_pose = 0.0
    total = 0
    device = next(distortion.parameters()).device
    for start in range(0, frames.shape[0], batch_size):
        end = min(frames.shape[0], start + batch_size)
        batch = round_ste(frames[start:end].to(device)).clamp(0, 255)
        rows = torch.arange(start, end, device=device)
        seg_logits = distortion.segnet(distortion.segnet.preprocess_input(batch))
        pose = distortion.posenet(distortion.posenet.preprocess_input(batch))["pose"][..., :6]
        target_seg = targets["seg_logits"].to(device).index_select(0, rows)
        target_pose = targets["pose"].to(device).index_select(0, rows)
        seg_dist = distortion.segnet.compute_distortion(target_seg, seg_logits)
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_seg += float(seg_dist.sum().item())
        total_pose += float(pose_dist.sum().item())
        total += end - start
    seg = total_seg / total
    pose = total_pose / total
    return {
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
        "quality": quality(seg, pose),
    }

