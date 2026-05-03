#!/usr/bin/env python
"""In-memory low-bit quantization oracle for PR #67 qpose14_qzs3.

This does not package a candidate. It answers whether the current QZS3 model has
enough quantization slack to justify a real QZS4/QAT packer.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import brotli
import einops
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBMISSION_DIR = REPO_ROOT / "submissions/qpose14_qzs3_filmq9g_slsb1_r55"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

import inflate as qzs3  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json
from submissions.search_vcm_v2.families.qpose14_data import load_original_subset, select_torch_device
from submissions.search_vcm_v2.subsets import get_subset


ARCHIVE = SUBMISSION_DIR / "archive.zip"
MASK_BYTES = 219_472
MODEL_BYTES = 56_093
ARCHIVE_BYTES = 276_564
OUT_DIR = EXPERIMENTS_DIR / "qzs3_lowbit_quant_oracle"


def _split_payload(archive: Path = ARCHIVE) -> tuple[bytes, bytes, bytes]:
    payload = zipfile.ZipFile(archive).read("p")
    return payload[:MASK_BYTES], payload[MASK_BYTES : MASK_BYTES + MODEL_BYTES], payload[MASK_BYTES + MODEL_BYTES :]


def _decode_masks(mask_br: bytes) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br))
        path = Path(tmp.name)
    try:
        return qzs3.load_encoded_mask_video(str(path))
    finally:
        path.unlink(missing_ok=True)


def _decode_pose(pose_br: bytes) -> torch.Tensor:
    raw = brotli.decompress(pose_br)
    if raw.startswith(b"QP1"):
        first = np.frombuffer(raw[3:5], dtype=np.uint16, count=1)[0]
        vals = [int(first)]
        cursor = 5
        while cursor < len(raw):
            shift = 0
            acc = 0
            while True:
                byte = raw[cursor]
                cursor += 1
                acc |= (byte & 0x7F) << shift
                if byte < 0x80:
                    break
                shift += 7
            vals.append(vals[-1] + ((acc >> 1) ^ -(acc & 1)))
        q_pose = np.zeros((len(vals), 6), dtype=np.uint16)
        q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
    else:
        q_pose = np.frombuffer(raw, dtype=np.uint16).reshape(-1, 6)
    pose = np.empty(q_pose.shape, dtype=np.float32)
    pose[:, 0] = q_pose[:, 0].astype(np.float32) / 512.0 + 20.0
    pose[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return torch.from_numpy(pose).float()


def load_generator(device: torch.device) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    mask_br, model_br, pose_br = _split_payload()
    model = qzs3.JointFrameGenerator().to(device)
    model.load_state_dict(qzs3.get_decoded_state_dict(brotli.decompress(model_br), device), strict=True)
    model.eval()
    return model, _decode_masks(mask_br), _decode_pose(pose_br)


def quantize_tensor_uniform(tensor: torch.Tensor, *, bits: int, block_size: int = 32) -> torch.Tensor:
    if bits >= 16:
        return tensor.clone()
    flat = tensor.detach().float().reshape(-1)
    n = int(flat.numel())
    blocks = (n + block_size - 1) // block_size
    padded = F.pad(flat, (0, blocks * block_size - n)).reshape(blocks, block_size)
    qmax = float((1 << (bits - 1)) - 1)
    if qmax <= 0:
        raise ValueError("bits must be >=2")
    scale = padded.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(padded / scale).clamp(-qmax, qmax)
    out = (q * scale).reshape(-1)[:n].reshape_as(tensor)
    return out.to(tensor.device, dtype=tensor.dtype)


def quantize_model(model: torch.nn.Module, *, conv_bits: int, qv_bits: int | None, dense_bits: int | None, scope: str) -> torch.nn.Module:
    out = copy.deepcopy(model).eval()
    with torch.no_grad():
        for name, module in out.named_modules():
            if isinstance(module, (qzs3.QConv2d, qzs3.QEmbedding)) and getattr(module, "quantize_weight", False):
                if scope in ("all", "conv", "frame1") and (scope != "frame1" or name.startswith("frame1_head") or name.startswith("pose_mlp")):
                    module.weight.copy_(quantize_tensor_uniform(module.weight, bits=conv_bits))
            if qv_bits is not None and hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                pass
        if qv_bits is not None:
            qv_names = set(qzs3.get_qv_specs())
            state = out.state_dict()
            for key in qv_names:
                if key in state:
                    state[key].copy_(quantize_tensor_uniform(state[key], bits=qv_bits))
        if dense_bits is not None:
            state = out.state_dict()
            skip = {f"{name}.weight" for name, module in out.named_modules() if isinstance(module, (qzs3.QConv2d, qzs3.QEmbedding))}
            skip |= set(qzs3.get_qv_specs())
            for key, value in state.items():
                if key in skip or not value.is_floating_point() or value.numel() < 16:
                    continue
                value.copy_(quantize_tensor_uniform(value, bits=dense_bits))
    return out


def estimate_model_raw_bytes(*, conv_bits: int, qv_bits: int | None, dense_bits: int | None, scope: str) -> dict[str, int]:
    template = qzs3.JointFrameGenerator()
    block_size = 32
    qv_specs = qzs3.get_qv_specs()
    covered = set()
    sizes = {"packed": 0, "scales": 0, "bias": 0, "dense_fp": 0, "fp_weight": 0, "dense_other": 0, "qv": 0}
    for name, module in template.named_modules():
        if not isinstance(module, (qzs3.QConv2d, qzs3.QEmbedding)):
            continue
        covered.add(f"{name}.weight")
        if getattr(module, "quantize_weight", False):
            numel = int(module.weight.numel())
            bits = conv_bits if (scope in ("all", "conv") or (scope == "frame1" and name.startswith("frame1_head"))) else 4
            sizes["packed"] += (numel * bits + 7) // 8
            sizes["scales"] += ((numel + block_size - 1) // block_size) * 2
        else:
            sizes["fp_weight"] += int(module.weight.numel()) * 2
        if isinstance(module, qzs3.QConv2d) and module.bias is not None:
            covered.add(f"{name}.bias")
            sizes["bias"] += int(module.bias.numel()) * 2
    for key, tensor in template.state_dict().items():
        if key in covered:
            continue
        count = int(tensor.numel())
        shape = tuple(tensor.shape)
        if key in qv_specs:
            old_bits, per_row = qv_specs[key]
            bits = qv_bits if qv_bits is not None else old_bits
            rows = shape[0] if per_row and len(shape) >= 2 else 1
            sizes["qv"] += rows * 4 + (count * bits + 7) // 8
        elif tensor.is_floating_point():
            if dense_bits is None:
                sizes["dense_fp"] += count * 2
            else:
                sizes["dense_fp"] += 4 + (count * dense_bits + 7) // 8
        else:
            sizes["dense_other"] += count * tensor.element_size()
    sizes["total_raw_plus_header"] = sum(sizes.values()) + 6
    return sizes


def evaluate_model(model: torch.nn.Module, masks: torch.Tensor, poses: torch.Tensor, *, subset_name: str, device: torch.device) -> dict[str, Any]:
    sample_ids = get_subset(subset_name)
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    original = load_original_subset(subset_name, sample_ids, device="cpu").float().to(device)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(sample_ids), 2):
            ids = sample_ids[start : start + 2]
            mask = masks[ids].to(device).long()
            pose = poses[ids].to(device).float()
            f1, f2 = model(mask, pose)
            f1 = F.interpolate(f1, size=(874, 1164), mode="bilinear", align_corners=False)
            f2 = F.interpolate(f2, size=(874, 1164), mode="bilinear", align_corners=False)
            pred = torch.stack([f1, f2], dim=1).clamp(0, 255).round()
            pred = einops.rearrange(pred, "b t c h w -> b t h w c")
            pose_dist, seg_dist = distortion.compute_distortion(original[start : start + len(ids)], pred)
            for sid, pose_v, seg_v in zip(ids, pose_dist.cpu().tolist(), seg_dist.cpu().tolist(), strict=True):
                seg = float(seg_v)
                pose_val = float(pose_v)
                rows.append(
                    {
                        "sample_id": int(sid),
                        "segnet_dist": seg,
                        "posenet_dist": pose_val,
                        "seg_term": 100.0 * seg,
                        "pose_term": float(torch.sqrt(torch.tensor(10.0 * pose_val)).item()),
                        "quality": quality(seg, pose_val),
                    }
                )
    seg_mean = sum(row["segnet_dist"] for row in rows) / len(rows)
    pose_mean = sum(row["posenet_dist"] for row in rows) / len(rows)
    return {
        "segnet_dist": seg_mean,
        "posenet_dist": pose_mean,
        "quality": quality(seg_mean, pose_mean),
        "score": quality(seg_mean, pose_mean) + rate_term(ARCHIVE_BYTES),
        "max_sample_quality": max(row["quality"] for row in rows),
        "per_sample": rows,
    }


def run_oracle(*, subset_name: str, device_name: str, out: Path) -> dict[str, Any]:
    device = select_torch_device(device_name)
    base_model, masks, poses = load_generator(device)
    variants = [
        {"name": "base", "conv_bits": 4, "qv_bits": None, "dense_bits": None, "scope": "all"},
        {"name": "qv8", "conv_bits": 4, "qv_bits": 8, "dense_bits": None, "scope": "all"},
        {"name": "qv6", "conv_bits": 4, "qv_bits": 6, "dense_bits": None, "scope": "all"},
        {"name": "conv3_all", "conv_bits": 3, "qv_bits": None, "dense_bits": None, "scope": "all"},
        {"name": "conv3_qv8", "conv_bits": 3, "qv_bits": 8, "dense_bits": None, "scope": "all"},
        {"name": "conv2_all", "conv_bits": 2, "qv_bits": None, "dense_bits": None, "scope": "all"},
        {"name": "conv3_frame1", "conv_bits": 3, "qv_bits": None, "dense_bits": None, "scope": "frame1"},
        {"name": "conv2_frame1", "conv_bits": 2, "qv_bits": None, "dense_bits": None, "scope": "frame1"},
        {"name": "conv3_dense8", "conv_bits": 3, "qv_bits": 8, "dense_bits": 8, "scope": "all"},
    ]
    rows = []
    base_metrics = None
    for cfg in variants:
        print(f"evaluating {cfg['name']}", flush=True)
        model = base_model if cfg["name"] == "base" else quantize_model(base_model, conv_bits=cfg["conv_bits"], qv_bits=cfg["qv_bits"], dense_bits=cfg["dense_bits"], scope=cfg["scope"]).to(device).eval()
        metrics = evaluate_model(model, masks, poses, subset_name=subset_name, device=device)
        sizes = estimate_model_raw_bytes(conv_bits=cfg["conv_bits"], qv_bits=cfg["qv_bits"], dense_bits=cfg["dense_bits"], scope=cfg["scope"])
        row = {**cfg, **metrics, "estimated_model_raw_bytes": sizes["total_raw_plus_header"], "estimated_raw_savings": 59_288 - sizes["total_raw_plus_header"], "segment_sizes": sizes}
        if base_metrics is None:
            base_metrics = metrics
        row["quality_delta_vs_base"] = metrics["quality"] - base_metrics["quality"]
        rows.append(row)
        print(json.dumps({k: row[k] for k in ["name", "quality", "quality_delta_vs_base", "estimated_raw_savings"]}, sort_keys=True), flush=True)
    summary = {
        "subset": subset_name,
        "device": str(device),
        "rows": rows,
        "best_tradeoff": min(rows[1:], key=lambda r: r["quality_delta_vs_base"] + max(0, 22_000 - r["estimated_raw_savings"]) / 100_000),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{subset_name}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="hard8")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    run_oracle(subset_name=args.subset, device_name=args.device, out=args.out)


if __name__ == "__main__":
    main()
