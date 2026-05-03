#!/usr/bin/env python
"""Pixel-space evaluator oracle for the Quantizr #55 fixed-payload path.

This does not build a submission. It answers whether the decoded frames could,
in principle, reach a sub-0.300 score at the current byte budget if the decoder
were able to synthesize evaluator-optimal pixels from the exact #55 payload.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import brotli
import einops
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import inflate as q55_inflate
from frame_utils import AVVideoDataset, camera_size
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path
from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    ORIGINAL_BYTES,
    POSE_PAYLOAD,
    append_jsonl,
    score_from_bytes,
    sha256_file,
    unzip_archive,
    write_json,
)


LOW_SIZE = (384, 512)
CAMERA_SIZE_HW = (camera_size[1], camera_size[0])


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def logit_from_image(x: torch.Tensor) -> torch.Tensor:
    unit = (x / 255.0).clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(unit)


def image_from_param(param: torch.Tensor, start_low: torch.Tensor, args) -> torch.Tensor:
    if args.param_mode == "logit":
        return torch.sigmoid(param) * 255.0
    if args.param_mode == "delta":
        return (start_low + args.max_delta * torch.tanh(param)).clamp(0.0, 255.0)
    raise ValueError(f"unknown param mode: {args.param_mode}")


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    unit = x / 255.0
    dy = (unit[..., 1:, :] - unit[..., :-1, :]).abs().mean()
    dx = (unit[..., :, 1:] - unit[..., :, :-1]).abs().mean()
    return dx + dy


def saturation_guard(x: torch.Tensor) -> torch.Tensor:
    unit = x / 255.0
    return F.relu(0.02 - unit).mean() + F.relu(unit - 0.98).mean()


def render_for_eval(x_low: torch.Tensor, camera_sim: bool) -> torch.Tensor:
    if camera_sim:
        x = F.interpolate(x_low.flatten(0, 1), size=CAMERA_SIZE_HW, mode="bilinear", align_corners=False)
        x = round_ste(x).unflatten(0, (x_low.shape[0], 2))
    else:
        x = round_ste(x_low)
    return x.clamp(0.0, 255.0)


def seg_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    return seg_margin_map(logits, target, margin).mean()


def seg_margin_map(logits: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    target_logits = logits.gather(1, target[:, None]).squeeze(1)
    non_target = logits.masked_fill(
        F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool(),
        -1.0e6,
    ).amax(dim=1)
    return F.relu(margin - (target_logits - non_target))


def reduce_seg_loss(loss_map: torch.Tensor, hard_mask: torch.Tensor, args) -> torch.Tensor:
    if args.hard_pixels_only:
        denom = hard_mask.sum().clamp_min(1.0)
        return (loss_map * hard_mask).sum() / denom
    weights = 1.0 + args.hard_pixel_boost * hard_mask
    return (loss_map * weights).mean()


def load_rgb_subset(
    video_names: Path,
    video_dir: Path,
    *,
    offset: int,
    max_samples: int,
    decode_batch_size: int,
) -> torch.Tensor:
    files = [line.strip() for line in video_names.read_text().splitlines() if line.strip()]
    ds = AVVideoDataset(files, data_dir=video_dir, batch_size=decode_batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    selected = []
    seen = 0
    for _, _, batch in ds:
        for sample in batch:
            if seen >= offset:
                selected.append(sample.contiguous())
                if len(selected) >= max_samples:
                    return torch.stack(selected)
            seen += 1
    if len(selected) != max_samples:
        raise RuntimeError(f"loaded {len(selected)} samples, expected {max_samples}")
    return torch.stack(selected)


def load_start_frames_low(
    archive_zip: Path,
    *,
    indices: torch.Tensor,
    device: torch.device,
    gen_batch_size: int,
) -> torch.Tensor:
    with tempfile.TemporaryDirectory(prefix="q55_pixel_oracle_") as td:
        archive_dir = Path(td) / "archive"
        unzip_archive(archive_zip, archive_dir)

        generator = q55_inflate.JointFrameGenerator(**q55_inflate.load_arch_config(archive_dir)).to(device)
        model_qpack = archive_dir / MODEL_QPACK_PAYLOAD
        if model_qpack.exists():
            state = q55_inflate.get_qpack_state_dict(brotli.decompress(model_qpack.read_bytes()), device)
        else:
            state = q55_inflate.get_decoded_state_dict(brotli.decompress((archive_dir / MODEL_PAYLOAD).read_bytes()), device)
        generator.load_state_dict(state, strict=True)
        generator.shared_trunk.mask_adapter = q55_inflate.load_mask_adapter(archive_dir, device)
        generator.eval()

        mask_all = q55_inflate.load_mask_payload(archive_dir, archive_dir / MASK_PAYLOAD)
        pose_all = q55_inflate.load_pose_payload(archive_dir, archive_dir / POSE_PAYLOAD)

        lows = []
        with torch.inference_mode():
            for start in range(0, len(indices), gen_batch_size):
                idx = indices[start : start + gen_batch_size]
                mask = mask_all.index_select(0, idx.cpu()).to(device).long()
                pose = pose_all.index_select(0, idx.cpu()).to(device).float()
                p1, p2 = generator(mask, pose)
                lows.append(torch.stack([p1, p2], dim=1).cpu())
        return torch.cat(lows, dim=0).contiguous()


def load_evaluators(device: torch.device) -> tuple[SegNet, PoseNet]:
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for model in (segnet, posenet):
        for p in model.parameters():
            p.requires_grad = False
    return segnet, posenet


@torch.inference_mode()
def build_targets(
    gt_pairs_u8: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seg_targets, pose_targets = [], []
    for start in range(0, gt_pairs_u8.shape[0], batch_size):
        gt = gt_pairs_u8[start : start + batch_size].to(device).float()
        gt = einops.rearrange(gt, "b t h w c -> b t c h w")
        seg_logits = segnet(segnet.preprocess_input(gt))
        pose = posenet(posenet.preprocess_input(gt))["pose"][..., :6]
        seg_targets.append(seg_logits.argmax(dim=1).cpu())
        pose_targets.append(pose.cpu())
    return torch.cat(seg_targets, dim=0).contiguous(), torch.cat(pose_targets, dim=0).contiguous()


@torch.inference_mode()
def compute_metrics(
    frames_low: torch.Tensor,
    seg_targets: torch.Tensor,
    pose_targets: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    *,
    batch_size: int,
    camera_sim: bool,
    archive_bytes: int,
    sample_offset: int = 0,
    include_per_sample: bool = False,
    top_k: int = 16,
) -> dict:
    total_pose, total_seg, total = 0.0, 0.0, 0
    seg_values: list[float] = []
    pose_values: list[float] = []
    for start in range(0, frames_low.shape[0], batch_size):
        x_low = frames_low[start : start + batch_size].to(device).float()
        target_seg = seg_targets[start : start + batch_size].to(device)
        target_pose = pose_targets[start : start + batch_size].to(device)
        x_eval = render_for_eval(x_low, camera_sim)
        seg_logits = segnet(segnet.preprocess_input(x_eval))
        pose = posenet(posenet.preprocess_input(x_eval))["pose"][..., :6]
        seg_dist = (seg_logits.argmax(dim=1) != target_seg).float().mean(dim=(1, 2))
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_seg += seg_dist.sum().item()
        total_pose += pose_dist.sum().item()
        total += x_low.shape[0]
        if include_per_sample:
            seg_values.extend(float(x) for x in seg_dist.detach().cpu().tolist())
            pose_values.extend(float(x) for x in pose_dist.detach().cpu().tolist())
    seg = total_seg / max(1, total)
    pose = total_pose / max(1, total)
    quality = 100.0 * seg + math.sqrt(max(0.0, 10.0 * pose))
    result = {
        "samples": total,
        "segnet_dist": seg,
        "posenet_dist": pose,
        "quality_term": quality,
        "archive_bytes": archive_bytes,
        "rate_term": 25.0 * archive_bytes / ORIGINAL_BYTES,
        "projected_score_at_archive_bytes": score_from_bytes(seg, pose, archive_bytes),
        "quality_required_for_score_lt_0_300": 0.300 - 25.0 * archive_bytes / ORIGINAL_BYTES,
    }
    if include_per_sample:
        result.update(summarize_per_sample(seg_values, pose_values, sample_offset=sample_offset, top_k=top_k))
    return result


def _percentiles(values: np.ndarray, qs: tuple[int, ...] = (0, 25, 50, 75, 90, 95, 99, 100)) -> dict[str, float]:
    if values.size == 0:
        return {f"p{q}": 0.0 for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def summarize_per_sample(
    seg_values: list[float],
    pose_values: list[float],
    *,
    sample_offset: int,
    top_k: int,
) -> dict:
    seg = np.asarray(seg_values, dtype=np.float64)
    pose = np.asarray(pose_values, dtype=np.float64)
    quality_like = 100.0 * seg + np.sqrt(np.maximum(0.0, 10.0 * pose))
    count = int(seg.shape[0])

    def top(metric: np.ndarray) -> list[dict]:
        if count == 0:
            return []
        order = np.argsort(metric)[::-1][: max(0, min(top_k, count))]
        return [
            {
                "rank": int(rank + 1),
                "relative_index": int(i),
                "sample_index": int(sample_offset + i),
                "segnet_dist": float(seg[i]),
                "posenet_dist": float(pose[i]),
                "quality_like": float(quality_like[i]),
            }
            for rank, i in enumerate(order)
        ]

    return {
        "per_sample_summary": {
            "count": count,
            "segnet_dist_percentiles": _percentiles(seg),
            "posenet_dist_percentiles": _percentiles(pose),
            "quality_like_percentiles": _percentiles(quality_like),
            "top_by_quality_like": top(quality_like),
            "top_by_segnet_dist": top(seg),
            "top_by_posenet_dist": top(pose),
        },
        "per_sample": [
            {
                "relative_index": int(i),
                "sample_index": int(sample_offset + i),
                "segnet_dist": float(seg[i]),
                "posenet_dist": float(pose[i]),
                "quality_like": float(quality_like[i]),
            }
            for i in range(count)
        ],
    }


def optimize_chunk(
    chunk_id: int,
    start_low: torch.Tensor,
    target_seg: torch.Tensor,
    target_pose: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    args,
) -> tuple[torch.Tensor, dict]:
    start_low_device = start_low.to(device).float()
    if args.param_mode == "logit":
        init_param = logit_from_image(start_low_device)
    else:
        init_param = torch.zeros_like(start_low_device)
    param = init_param.detach().requires_grad_(True)
    target_seg = target_seg.to(device)
    target_pose = target_pose.to(device)
    opt = torch.optim.AdamW([param], lr=args.lr)
    with torch.no_grad():
        initial_eval = render_for_eval(start_low_device, args.camera_sim)
        initial_seg = segnet(segnet.preprocess_input(initial_eval))
        initial_pose = posenet(posenet.preprocess_input(initial_eval))["pose"][..., :6]
        hard_mask = (initial_seg.argmax(dim=1) != target_seg).float()
        initial_pose_per = (initial_pose - target_pose).pow(2).mean(dim=1)
        initial_seg_dist = hard_mask.mean().item()
        initial_pose_dist = initial_pose_per.mean().item()
        initial_quality = 100.0 * initial_seg_dist + math.sqrt(max(0.0, 10.0 * initial_pose_dist))
        best_quality = initial_quality
        best_low = start_low_device.detach().cpu().contiguous()
        best_step = 0
    pbar = tqdm(range(1, args.steps + 1), desc=f"Oracle chunk {chunk_id}", leave=False)
    last = {}
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        x_low = image_from_param(param, start_low_device, args)
        x_eval = render_for_eval(x_low, args.camera_sim)
        seg_logits = segnet(segnet.preprocess_input(x_eval))
        pose = posenet(posenet.preprocess_input(x_eval))["pose"][..., :6]

        ce = reduce_seg_loss(F.cross_entropy(seg_logits, target_seg, reduction="none"), hard_mask, args)
        margin = reduce_seg_loss(seg_margin_map(seg_logits, target_seg, args.margin), hard_mask, args)
        pose_per = (pose - target_pose).pow(2).mean(dim=1)
        pose_term = torch.sqrt(10.0 * pose_per + 1e-8).mean()
        pose_mse = pose_per.mean()
        tv = tv_loss(x_low)
        sat = saturation_guard(x_low)
        loss = (
            args.seg_ce_weight * ce
            + args.seg_margin_weight * margin
            + args.pose_term_weight * pose_term
            + args.pose_mse_weight * pose_mse
            + args.tv_weight * tv
            + args.saturation_weight * sat
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param], args.grad_clip)
        opt.step()

        with torch.no_grad():
            seg_dist = (seg_logits.argmax(dim=1) != target_seg).float().mean().item()
            pose_dist = pose_per.mean().item()
            quality = 100.0 * seg_dist + math.sqrt(max(0.0, 10.0 * pose_dist))
            if quality < best_quality - args.early_stop_epsilon:
                best_quality = quality
                best_step = step
                best_low = x_low.detach().cpu().contiguous()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                last = {
                    "step": step,
                    "best_step": best_step,
                    "best_quality_term": best_quality,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "margin": float(margin.item()),
                    "pose_term": float(pose_term.item()),
                    "pose_mse": float(pose_mse.item()),
                    "segnet_dist": float(seg_dist),
                    "quality_term": float(quality),
                }
                pbar.set_postfix(Q=f"{quality:.4f}", S=f"{seg_dist:.5f}", P=f"{pose_dist:.6f}")
            if (
                args.early_stop_patience > 0
                and step >= args.early_stop_min_step
                and step - best_step >= args.early_stop_patience
            ):
                last = {
                    "step": step,
                    "best_step": best_step,
                    "best_quality_term": best_quality,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "margin": float(margin.item()),
                    "pose_term": float(pose_term.item()),
                    "pose_mse": float(pose_mse.item()),
                    "segnet_dist": float(seg_dist),
                    "quality_term": float(quality),
                    "early_stopped": True,
                }
                break
    return best_low, {
        "chunk": chunk_id,
        "initial_quality_term": initial_quality,
        "best_quality_term": best_quality,
        "best_step": best_step,
        "hard_pixel_fraction": initial_seg_dist,
        **last,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="q55_pixel_oracle")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--opt-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--param-mode", choices=["delta", "logit"], default="delta")
    parser.add_argument("--max-delta", type=float, default=16.0)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--hard-pixels-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-pixel-boost", type=float, default=256.0)
    parser.add_argument("--seg-ce-weight", type=float, default=1.0)
    parser.add_argument("--seg-margin-weight", type=float, default=0.25)
    parser.add_argument("--pose-term-weight", type=float, default=8.0)
    parser.add_argument("--pose-mse-weight", type=float, default=3.0)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--saturation-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--early-stop-min-step", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-epsilon", type=float, default=1e-8)
    parser.add_argument("--camera-sim", action="store_true")
    parser.add_argument("--save-optimized-uint8", action="store_true")
    parser.add_argument("--save-per-sample-metrics", action="store_true")
    parser.add_argument("--tail-top-k", type=int, default=16)
    args = parser.parse_args()

    if not args.base_archive.exists():
        raise FileNotFoundError(args.base_archive)
    device = torch.device(args.device)
    run_dir = args.out_dir / args.label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    indices = torch.arange(args.offset, args.offset + args.max_samples, dtype=torch.long)
    segnet, posenet = load_evaluators(device)

    print("Loading source RGB subset...", flush=True)
    gt_pairs = load_rgb_subset(
        args.video_names,
        args.video_dir,
        offset=args.offset,
        max_samples=args.max_samples,
        decode_batch_size=args.decode_batch_size,
    )

    print("Building evaluator targets...", flush=True)
    seg_targets, pose_targets = build_targets(gt_pairs, segnet, posenet, device, args.target_batch_size)

    print("Generating #55 starting frames from archive payload...", flush=True)
    start_low = load_start_frames_low(
        args.base_archive,
        indices=indices,
        device=device,
        gen_batch_size=args.gen_batch_size,
    )

    archive_bytes = args.base_archive.stat().st_size
    initial_metrics = compute_metrics(
        start_low,
        seg_targets,
        pose_targets,
        segnet,
        posenet,
        device,
        batch_size=args.eval_batch_size,
        camera_sim=args.camera_sim,
        archive_bytes=archive_bytes,
        sample_offset=args.offset,
        include_per_sample=args.save_per_sample_metrics,
        top_k=args.tail_top_k,
    )
    print("Initial metrics:", json.dumps(initial_metrics, indent=2), flush=True)

    optimized_chunks, chunk_logs = [], []
    for chunk_id, start in enumerate(range(0, args.max_samples, args.opt_batch_size)):
        end = min(args.max_samples, start + args.opt_batch_size)
        final_low, chunk_log = optimize_chunk(
            chunk_id,
            start_low[start:end],
            seg_targets[start:end],
            pose_targets[start:end],
            segnet,
            posenet,
            device,
            args,
        )
        optimized_chunks.append(final_low)
        chunk_logs.append(chunk_log)

    final_low = torch.cat(optimized_chunks, dim=0).contiguous()
    final_metrics = compute_metrics(
        final_low,
        seg_targets,
        pose_targets,
        segnet,
        posenet,
        device,
        batch_size=args.eval_batch_size,
        camera_sim=args.camera_sim,
        archive_bytes=archive_bytes,
        sample_offset=args.offset,
        include_per_sample=args.save_per_sample_metrics,
        top_k=args.tail_top_k,
    )
    print("Final metrics:", json.dumps(final_metrics, indent=2), flush=True)

    record = {
        "label": args.label,
        "base_archive": str(args.base_archive),
        "base_archive_sha256": sha256_file(args.base_archive),
        "archive_bytes": archive_bytes,
        "device": str(device),
        "offset": args.offset,
        "max_samples": args.max_samples,
        "steps": args.steps,
        "opt_batch_size": args.opt_batch_size,
        "param_mode": args.param_mode,
        "max_delta": args.max_delta,
        "hard_pixels_only": args.hard_pixels_only,
        "hard_pixel_boost": args.hard_pixel_boost,
        "camera_sim": args.camera_sim,
        "initial": initial_metrics,
        "final": final_metrics,
        "chunks": chunk_logs,
    }
    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "pixel_oracle_results.jsonl", record)

    if args.save_per_sample_metrics:
        write_json(
            run_dir / "per_sample_metrics.json",
            {
                "label": args.label,
                "offset": args.offset,
                "max_samples": args.max_samples,
                "initial": initial_metrics.get("per_sample", []),
                "final": final_metrics.get("per_sample", []),
                "initial_summary": initial_metrics.get("per_sample_summary", {}),
                "final_summary": final_metrics.get("per_sample_summary", {}),
            },
        )

    if args.save_optimized_uint8:
        low_uint8 = final_low.clamp(0, 255).round().to(torch.uint8).numpy()
        np.save(run_dir / "optimized_low_uint8.npy", low_uint8)
        start_uint8 = start_low.clamp(0, 255).round().to(torch.uint8).numpy()
        np.save(run_dir / "start_low_uint8.npy", start_uint8)

    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
