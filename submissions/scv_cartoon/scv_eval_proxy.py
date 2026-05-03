#!/usr/bin/env python
"""Shared metric and evaluator helpers for the SCV cartoon codec prototype."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from frame_utils import AVVideoDataset, camera_size, segnet_model_input_size
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path


ORIGINAL_BYTES = 37_545_489
CAMERA_W, CAMERA_H = camera_size
SEG_W, SEG_H = segnet_model_input_size


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(max(0.0, 10.0 * posenet_dist))


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + 25.0 * archive_bytes / ORIGINAL_BYTES


def required_quality_for_score(archive_bytes: int, target: float = 0.300) -> float:
    return target - 25.0 * archive_bytes / ORIGINAL_BYTES


def metric_table(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> dict:
    q = quality(segnet_dist, posenet_dist)
    s = score(segnet_dist, posenet_dist, archive_bytes)
    req = required_quality_for_score(archive_bytes)
    return {
        "archive_bytes": int(archive_bytes),
        "rate_term": 25.0 * archive_bytes / ORIGINAL_BYTES,
        "segnet_dist": float(segnet_dist),
        "posenet_dist": float(posenet_dist),
        "quality": float(q),
        "score": float(s),
        "required_quality_for_0.300": float(req),
        "gap_to_0.300": float(s - 0.300),
    }


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_evaluators(device: torch.device) -> tuple[SegNet, PoseNet]:
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for model in (segnet, posenet):
        for param in model.parameters():
            param.requires_grad = False
    return segnet, posenet


def load_rgb_subset(
    *,
    video_names: Path,
    video_dir: Path,
    subset: int,
    offset: int = 0,
    batch_size: int = 8,
) -> torch.Tensor:
    files = [line.strip() for line in video_names.read_text().splitlines() if line.strip()]
    ds = AVVideoDataset(files, data_dir=video_dir, batch_size=batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    selected = []
    seen = 0
    for _, _, batch in ds:
        for sample in batch:
            if seen >= offset:
                selected.append(sample.contiguous())
                if len(selected) >= subset:
                    return torch.stack(selected)
            seen += 1
    raise RuntimeError(f"loaded {len(selected)} samples, expected {subset}")


@torch.inference_mode()
def build_targets(
    gt_pairs_u8: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    batch_size: int = 8,
    include_logits: bool = False,
) -> dict:
    seg_targets, pose_targets, seg_logits_out = [], [], []
    for start in range(0, gt_pairs_u8.shape[0], batch_size):
        gt = gt_pairs_u8[start : start + batch_size].to(device).float()
        gt = einops.rearrange(gt, "b t h w c -> b t c h w")
        logits = segnet(segnet.preprocess_input(gt)).float()
        pose = posenet(posenet.preprocess_input(gt))["pose"][..., :6].float()
        seg_targets.append(logits.argmax(dim=1).cpu())
        pose_targets.append(pose.cpu())
        if include_logits:
            seg_logits_out.append(logits.half().cpu())
    out = {
        "seg_targets": torch.cat(seg_targets, dim=0).contiguous(),
        "pose_targets": torch.cat(pose_targets, dim=0).contiguous(),
    }
    if include_logits:
        out["seg_logits"] = torch.cat(seg_logits_out, dim=0).contiguous()
    return out


def frames_uint8_to_eval(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 5:
        raise ValueError(f"expected (N,2,H,W,3), got {tuple(frames.shape)}")
    return einops.rearrange(frames.float(), "b t h w c -> b t c h w").contiguous()


@torch.inference_mode()
def evaluate_pairs(
    comp_pairs_u8: torch.Tensor,
    seg_targets: torch.Tensor,
    pose_targets: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    *,
    batch_size: int = 8,
    archive_bytes: int = 0,
) -> dict:
    total_seg, total_pose, total = 0.0, 0.0, 0
    for start in range(0, comp_pairs_u8.shape[0], batch_size):
        comp = comp_pairs_u8[start : start + batch_size].to(device)
        x = frames_uint8_to_eval(comp)
        target_seg = seg_targets[start : start + comp.shape[0]].to(device)
        target_pose = pose_targets[start : start + comp.shape[0]].to(device)
        seg_logits = segnet(segnet.preprocess_input(x))
        pose = posenet(posenet.preprocess_input(x))["pose"][..., :6]
        seg_dist = (seg_logits.argmax(dim=1) != target_seg).float().mean(dim=(1, 2))
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_seg += float(seg_dist.sum().item())
        total_pose += float(pose_dist.sum().item())
        total += int(comp.shape[0])
    seg = total_seg / max(1, total)
    pose = total_pose / max(1, total)
    record = {
        "samples": total,
        "segnet_dist": float(seg),
        "posenet_dist": float(pose),
        "quality": quality(seg, pose),
    }
    if archive_bytes:
        record.update(metric_table(seg, pose, archive_bytes))
    return record


def read_raw_pairs(raw_path: Path, *, max_pairs: int | None = None) -> torch.Tensor:
    frame_bytes = CAMERA_H * CAMERA_W * 3
    data = np.memmap(raw_path, dtype=np.uint8, mode="r")
    if data.size % frame_bytes != 0:
        raise RuntimeError(f"{raw_path} size is not divisible by frame bytes")
    frames = data.reshape((-1, CAMERA_H, CAMERA_W, 3))
    pair_count = frames.shape[0] // 2
    if max_pairs is not None:
        pair_count = min(pair_count, max_pairs)
    arr = np.asarray(frames[: pair_count * 2]).reshape((pair_count, 2, CAMERA_H, CAMERA_W, 3))
    return torch.from_numpy(arr.copy())


def downsample_frame2_to_seg(frame2_u8: torch.Tensor) -> torch.Tensor:
    x = einops.rearrange(frame2_u8.float(), "b h w c -> b c h w")
    return F.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
