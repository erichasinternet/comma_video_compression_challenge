#!/usr/bin/env python
"""Pose-conditioned frame1 DCT residual oracle for qzs3.

This tests a packable actuator: predict low-frequency frame1 DCT coefficients
from existing pose/time features instead of storing per-sample coefficients.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import brotli
import einops
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import write_json  # noqa: E402
from submissions.search_vcm_v2.tools.qzs3_frame1_dct_oracle import (  # noqa: E402
    DEFAULT_SUBMISSION,
    eval_actual,
    load_subset_pairs,
    make_dct_basis,
    parse_sample_ids,
    posenet_preprocess_diff,
    select_device,
)


RANGE_MASK_BYTES = 159011
SPLIT_MODEL_REORDERED_BYTES = 55725
POSE_BYTES = 899
ROUTER_ACTION_BYTES = 225


def load_payload(submission_dir: Path) -> bytes:
    p = submission_dir / "archive" / "p"
    if p.exists():
        return p.read_bytes()
    with zipfile.ZipFile(submission_dir / "archive.zip") as zf:
        return zf.read("p")


def load_pose(submission_dir: Path) -> np.ndarray:
    payload = load_payload(submission_dir)
    pose_start = RANGE_MASK_BYTES + SPLIT_MODEL_REORDERED_BYTES
    pose_raw = brotli.decompress(payload[pose_start : pose_start + POSE_BYTES])
    if pose_raw.startswith(b"QP1"):
        first = np.frombuffer(pose_raw[3:5], dtype=np.uint16, count=1)[0]
        vals = [int(first)]
        cursor = 5
        while cursor < len(pose_raw):
            shift = 0
            acc = 0
            while True:
                byte = pose_raw[cursor]
                cursor += 1
                acc |= (byte & 0x7F) << shift
                if byte < 0x80:
                    break
                shift += 7
            delta = (acc >> 1) ^ -(acc & 1)
            vals.append(vals[-1] + delta)
        q_pose = np.zeros((len(vals), 6), dtype=np.uint16)
        q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
    else:
        q_pose = np.frombuffer(pose_raw, dtype=np.uint16).reshape(-1, 6)
    pose_np = np.empty(q_pose.shape, dtype=np.float32)
    pose_np[:, 0] = q_pose[:, 0].astype(np.float32) / 512.0 + 20.0
    pose_np[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return pose_np


def make_features(
    pose_np: np.ndarray,
    sample_ids: list[int],
    *,
    include_time: bool,
    comp: torch.Tensor | None = None,
    image_grid: int = 0,
) -> torch.Tensor:
    pose = torch.from_numpy(pose_np[sample_ids]).float()
    pose = (pose - pose.mean(dim=0, keepdim=True)) / pose.std(dim=0, keepdim=True).clamp_min(1e-6)
    pieces = [pose]
    if not include_time:
        pass
    else:
        t = torch.tensor(sample_ids, dtype=torch.float32).view(-1, 1) / 599.0
        t2 = 2.0 * t - 1.0
        time_feats = [t2, t2 * t2]
        for f in (1.0, 2.0, 4.0):
            time_feats.append(torch.sin(2.0 * math.pi * f * t))
            time_feats.append(torch.cos(2.0 * math.pi * f * t))
        pieces.extend(time_feats)

    if comp is not None and image_grid > 0:
        with torch.no_grad():
            # Low-frequency frame-pair statistics are available during inflate and
            # give the predictor scene/motion context without per-sample payload.
            x = einops.rearrange(comp.detach().float().cpu(), "b t h w c -> b (t c) h w") / 255.0
            pooled = torch.nn.functional.adaptive_avg_pool2d(x, (image_grid, image_grid)).flatten(1)
            global_mean = x.mean(dim=(-2, -1))
            global_std = x.std(dim=(-2, -1))
            diff = (x[:, :3] - x[:, 3:6])
            diff_mean = diff.mean(dim=(-2, -1))
            diff_std = diff.std(dim=(-2, -1))
            img = torch.cat([pooled, global_mean, global_std, diff_mean, diff_std], dim=1)
            img = (img - img.mean(dim=0, keepdim=True)) / img.std(dim=0, keepdim=True).clamp_min(1e-6)
            pieces.append(img)
    return torch.cat(pieces, dim=1)


class LinearDCT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 0):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def apply_alpha(comp: torch.Tensor, alpha: torch.Tensor, basis: torch.Tensor, max_delta: float) -> torch.Tensor:
    alpha = max_delta * torch.tanh(alpha)
    delta = torch.einsum("bk,kchw->bchw", alpha, basis)
    frame1 = (comp[:, 0] + einops.rearrange(delta, "b c h w -> b h w c")).clamp(0.0, 255.0)
    return torch.stack([frame1, comp[:, 1]], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="hard8")
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--basis-k", type=int, default=96)
    parser.add_argument("--max-delta", type=float, default=64.0)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--image-grid", type=int, default=0, help="include generated-frame low-frequency features on an NxN grid")
    parser.add_argument("--train-batch-size", type=int, default=0, help="0 means full-batch optimization")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = select_device(args.device)
    sample_ids = parse_sample_ids(args.sample_ids, args.sample_ids_file)
    ids, gt, comp = load_subset_pairs(args.submission_dir, args.subset, device, sample_ids)
    pose_np = load_pose(args.submission_dir)
    feats = make_features(pose_np, ids, include_time=args.include_time, comp=comp, image_grid=args.image_grid).to(device)
    model = LinearDCT(feats.shape[1], args.basis_k, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in distortion.parameters():
        p.requires_grad_(False)
    basis = make_dct_basis(args.basis_k, 874, 1164, device)
    with torch.no_grad():
        target_pose = distortion.posenet(posenet_preprocess_diff(gt))
    base_metrics = eval_actual(distortion, gt, comp, batch_size=args.eval_batch_size)
    best = {"step": 0, "metrics": base_metrics, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    history = [{"step": 0, **{k: v for k, v in base_metrics.items() if k != "rows"}}]

    all_idx = torch.arange(len(ids), device=device)
    for step in range(1, args.steps + 1):
        if args.train_batch_size and args.train_batch_size < len(ids):
            batch_idx = torch.randint(0, len(ids), (args.train_batch_size,), device=device)
        else:
            batch_idx = all_idx
        pred = apply_alpha(comp[batch_idx], model(feats[batch_idx]), basis, args.max_delta)
        pose_out = distortion.posenet(posenet_preprocess_diff(pred))
        pose_loss = (pose_out["pose"][..., :6] - target_pose["pose"][batch_idx, ..., :6]).pow(2).mean()
        reg = sum(param.pow(2).mean() for param in model.parameters())
        loss = pose_loss + args.l2 * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            pred_full = apply_alpha(comp, model(feats), basis, args.max_delta)
            metrics = eval_actual(distortion, gt, pred_full.detach(), batch_size=args.eval_batch_size)
            history.append({"step": step, "loss": float(loss.detach().cpu()), **{k: v for k, v in metrics.items() if k != "rows"}})
            if metrics["quality"] < best["metrics"]["quality"]:
                best = {"step": step, "metrics": metrics, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}

    param_count = sum(p.numel() for p in model.parameters())
    estimated_int8_bytes = param_count
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "subset": args.subset,
        "sample_ids": ids,
        "device": str(device),
        "basis_k": args.basis_k,
        "max_delta": args.max_delta,
        "hidden": args.hidden,
        "include_time": args.include_time,
        "image_grid": args.image_grid,
        "feature_dim": feats.shape[1],
        "param_count": param_count,
        "estimated_int8_bytes": estimated_int8_bytes,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_clip": args.grad_clip,
        "base": base_metrics,
        "best_step": int(best["step"]),
        "best": best["metrics"],
        "quality_delta_vs_base": best["metrics"]["quality"] - base_metrics["quality"],
        "history": history,
    }
    write_json(args.out / f"{args.subset}_k{args.basis_k}_h{args.hidden}_summary.json", summary)
    torch.save({"state": best["state"], "sample_ids": ids, "basis_k": args.basis_k, "max_delta": args.max_delta, "hidden": args.hidden, "include_time": args.include_time}, args.out / f"{args.subset}_k{args.basis_k}_h{args.hidden}_best.pt")
    print(json.dumps({k: summary[k] for k in ("subset", "basis_k", "hidden", "include_time", "param_count", "base", "best_step", "best", "quality_delta_vs_base")}, indent=2))


if __name__ == "__main__":
    main()
