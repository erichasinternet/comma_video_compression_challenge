#!/usr/bin/env python
"""Frame1 PoseNet-control oracles for the Quantizr #55 exact-mask path.

This does not build a submission. It tests whether the PoseNet-heavy tail is
controllable through frame1 while keeping the generated frame2 fixed. SegNet
uses only frame2, so any score movement here isolates PoseNet control.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from frame_utils import AVVideoDataset
from q55_common import DEFAULT_VIDEO_NAMES, ORIGINAL_BYTES, append_jsonl, score_from_bytes, sha256_file, write_json
from q55_pixel_oracle import (
    build_targets,
    load_evaluators,
    load_start_frames_low,
    render_for_eval,
    round_ste,
)


def parse_indices(args: argparse.Namespace) -> list[int]:
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        indices = list(range(args.offset, args.offset + args.max_samples))
    if not indices:
        raise ValueError("no sample indices selected")
    if min(indices) < 0:
        raise ValueError("sample indices must be non-negative")
    return indices


def load_rgb_indices(
    video_names: Path,
    video_dir: Path,
    *,
    indices: list[int],
    decode_batch_size: int,
) -> torch.Tensor:
    """Load arbitrary public-test sample indices in the requested order."""
    files = [line.strip() for line in video_names.read_text().splitlines() if line.strip()]
    ds = AVVideoDataset(files, data_dir=video_dir, batch_size=decode_batch_size, device=torch.device("cpu"))
    ds.prepare_data()

    wanted = {int(sample_idx): pos for pos, sample_idx in enumerate(indices)}
    selected: list[torch.Tensor | None] = [None] * len(indices)
    seen = 0
    remaining = set(wanted)
    for _, _, batch in ds:
        for sample in batch:
            if seen in remaining:
                selected[wanted[seen]] = sample.contiguous()
                remaining.remove(seen)
                if not remaining:
                    return torch.stack([x for x in selected if x is not None])
            seen += 1
    raise RuntimeError(f"missing sample indices after reading dataset: {sorted(remaining)}")


def _percentiles(values: np.ndarray, qs: tuple[int, ...] = (0, 25, 50, 75, 90, 95, 99, 100)) -> dict[str, float]:
    if values.size == 0:
        return {f"p{q}": 0.0 for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def summarize_per_sample(seg_values: list[float], pose_values: list[float], sample_indices: list[int], top_k: int) -> dict:
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
                "sample_index": int(sample_indices[i]),
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
                "sample_index": int(sample_indices[i]),
                "segnet_dist": float(seg[i]),
                "posenet_dist": float(pose[i]),
                "quality_like": float(quality_like[i]),
            }
            for i in range(count)
        ],
    }


@torch.inference_mode()
def compute_metrics_eval(
    frames_eval: torch.Tensor,
    seg_targets: torch.Tensor,
    pose_targets: torch.Tensor,
    segnet,
    posenet,
    device: torch.device,
    *,
    batch_size: int,
    archive_bytes: int,
    sample_indices: list[int],
    include_per_sample: bool,
    top_k: int,
) -> dict:
    total_pose, total_seg, total = 0.0, 0.0, 0
    seg_values: list[float] = []
    pose_values: list[float] = []
    for start in range(0, frames_eval.shape[0], batch_size):
        x = frames_eval[start : start + batch_size].to(device).float()
        target_seg = seg_targets[start : start + batch_size].to(device)
        target_pose = pose_targets[start : start + batch_size].to(device)
        seg_logits = segnet(segnet.preprocess_input(x))
        pose = posenet(posenet.preprocess_input(x))["pose"][..., :6]
        seg_dist = (seg_logits.argmax(dim=1) != target_seg).float().mean(dim=(1, 2))
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_seg += seg_dist.sum().item()
        total_pose += pose_dist.sum().item()
        total += x.shape[0]
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
        result.update(summarize_per_sample(seg_values, pose_values, sample_indices, top_k))
    return result


def to_eval_pairs(gt_pairs_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    return einops.rearrange(gt_pairs_u8, "b t h w c -> b t c h w").to(device).float()


def replace_frame1(base_eval: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
    out = base_eval.clone()
    out[:, 0] = round_ste(frame1).clamp(0.0, 255.0)
    return out


def lowres_frame1(frame1: torch.Tensor, size_hw: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "y":
        coeff = torch.tensor([0.299, 0.587, 0.114], device=frame1.device, dtype=frame1.dtype).view(1, 3, 1, 1)
        low_input = (frame1 * coeff).sum(dim=1, keepdim=True)
        small = F.interpolate(low_input, size=size_hw, mode="area")
        small = small.round().clamp(0.0, 255.0)
        up = F.interpolate(small, size=frame1.shape[-2:], mode="bilinear", align_corners=False)
        return up.repeat(1, 3, 1, 1)
    if mode == "rgb":
        small = F.interpolate(frame1, size=size_hw, mode="area")
        small = small.round().clamp(0.0, 255.0)
        return F.interpolate(small, size=frame1.shape[-2:], mode="bilinear", align_corners=False)
    raise ValueError(f"unknown lowres mode: {mode}")


def parse_size(text: str) -> tuple[int, int]:
    w, h = [int(x.strip()) for x in text.lower().split("x", 1)]
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid size: {text}")
    return h, w


def quality_from_pose_and_seg(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(max(0.0, 10.0 * posenet_dist))


def pose_only_quality(
    frames_eval: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[float, float]:
    total_pose, total = 0.0, 0
    for start in range(0, frames_eval.shape[0], batch_size):
        x = frames_eval[start : start + batch_size].to(device).float()
        target_pose = pose_targets[start : start + batch_size].to(device)
        pose = posenet(posenet.preprocess_input(x))["pose"][..., :6]
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_pose += pose_dist.sum().item()
        total += x.shape[0]
    pose = total_pose / max(1, total)
    return pose, math.sqrt(max(0.0, 10.0 * pose))


def optimize_affine_frame1(
    base_eval: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    chunks: list[torch.Tensor] = []
    logs = []
    identity_template = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    for chunk_id, start in enumerate(range(0, base_eval.shape[0], args.opt_batch_size)):
        end = min(base_eval.shape[0], start + args.opt_batch_size)
        base = base_eval[start:end].to(device).float()
        target_pose = pose_targets[start:end].to(device)
        batch = base.shape[0]
        identity = identity_template.unsqueeze(0).repeat(batch, 1, 1)
        theta = identity.clone().detach().requires_grad_(True)
        opt = torch.optim.AdamW([theta], lr=args.affine_lr)

        with torch.no_grad():
            base_pose, _ = pose_only_quality(base, target_pose, posenet, device, batch_size=batch)
            best_pose = base_pose
            best_frames = base.detach().cpu().contiguous()
            best_step = 0

        pbar = tqdm(range(1, args.affine_steps + 1), desc=f"Affine chunk {chunk_id}", leave=False)
        last = {}
        for step in pbar:
            opt.zero_grad(set_to_none=True)
            grid = F.affine_grid(theta, base[:, 0].shape, align_corners=False)
            warped_f1 = F.grid_sample(base[:, 0], grid, mode="bilinear", padding_mode="border", align_corners=False)
            candidate = base.clone()
            candidate[:, 0] = round_ste(warped_f1).clamp(0.0, 255.0)
            pose = posenet(posenet.preprocess_input(candidate))["pose"][..., :6]
            pose_per = (pose - target_pose).pow(2).mean(dim=1)
            pose_term = torch.sqrt(10.0 * pose_per + 1e-8).mean()
            pose_mse = pose_per.mean()
            reg = (theta - identity).pow(2).mean()
            loss = args.pose_term_weight * pose_term + args.pose_mse_weight * pose_mse + args.affine_reg_weight * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], args.grad_clip)
            opt.step()

            with torch.no_grad():
                pose_dist = pose_per.mean().item()
                if pose_dist < best_pose - args.early_stop_epsilon:
                    best_pose = pose_dist
                    best_frames = candidate.detach().cpu().contiguous()
                    best_step = step
                if step == 1 or step % args.log_every == 0 or step == args.affine_steps:
                    last = {
                        "step": step,
                        "best_step": best_step,
                        "pose_mse": float(pose_dist),
                        "pose_term": float(math.sqrt(max(0.0, 10.0 * pose_dist))),
                        "loss": float(loss.item()),
                        "reg": float(reg.item()),
                    }
                    pbar.set_postfix(P=f"{pose_dist:.6f}", T=f"{last['pose_term']:.4f}")
        chunks.append(best_frames)
        logs.append(
            {
                "chunk": chunk_id,
                "initial_posenet_dist": float(base_pose),
                "best_posenet_dist": float(best_pose),
                "best_step": int(best_step),
                **last,
            }
        )
    return torch.cat(chunks, dim=0).contiguous(), {"chunks": logs}


def patch_bounds(image_hw: tuple[int, int], patch_hw: tuple[int, int], position: str) -> tuple[int, int, int, int]:
    h, w = image_hw
    ph, pw = patch_hw
    if ph > h or pw > w:
        raise ValueError(f"patch {patch_hw} does not fit image {image_hw}")
    if position == "center":
        y0 = (h - ph) // 2
        x0 = (w - pw) // 2
    elif position == "bottom_center":
        y0 = h - ph - max(1, h // 12)
        x0 = (w - pw) // 2
    elif position == "top_center":
        y0 = max(0, h // 12)
        x0 = (w - pw) // 2
    elif position == "left_center":
        y0 = (h - ph) // 2
        x0 = max(0, w // 12)
    elif position == "right_center":
        y0 = (h - ph) // 2
        x0 = w - pw - max(1, w // 12)
    else:
        raise ValueError(f"unknown patch position: {position}")
    return y0, x0, y0 + ph, x0 + pw


def logit_from_unit(x: torch.Tensor) -> torch.Tensor:
    unit = x.clamp(0.0, 255.0) / 255.0
    return torch.logit(unit.clamp(1e-4, 1.0 - 1e-4))


def tv_patch(x: torch.Tensor) -> torch.Tensor:
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dx + dy


def optimize_patch_frame1(
    base_eval: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    patch_hw = parse_size(args.patch_size)
    grid_hw = parse_size(args.patch_grid)
    position_logs = []
    best_global_frames: torch.Tensor | None = None
    best_global_pose = float("inf")
    best_global_label = ""

    for position in args.patch_positions.split(","):
        position = position.strip()
        if not position:
            continue
        y0, x0, y1, x1 = patch_bounds(base_eval.shape[-2:], patch_hw, position)
        chunks: list[torch.Tensor] = []
        chunks_logs = []
        for chunk_id, start in enumerate(range(0, base_eval.shape[0], args.opt_batch_size)):
            end = min(base_eval.shape[0], start + args.opt_batch_size)
            base = base_eval[start:end].to(device).float()
            target_pose = pose_targets[start:end].to(device)
            current_patch = base[:, 0, :, y0:y1, x0:x1]
            init_small = F.interpolate(current_patch, size=grid_hw, mode="area")
            param = logit_from_unit(init_small).detach().requires_grad_(True)
            opt = torch.optim.AdamW([param], lr=args.patch_lr)

            with torch.no_grad():
                base_pose, _ = pose_only_quality(base, target_pose, posenet, device, batch_size=base.shape[0])
                best_pose = base_pose
                best_frames = base.detach().cpu().contiguous()
                best_step = 0

            pbar = tqdm(range(1, args.patch_steps + 1), desc=f"Patch {position} chunk {chunk_id}", leave=False)
            last = {}
            for step in pbar:
                opt.zero_grad(set_to_none=True)
                patch_small = torch.sigmoid(param)
                patch = F.interpolate(patch_small, size=patch_hw, mode="bilinear", align_corners=False) * 255.0
                candidate = base.clone()
                candidate[:, 0, :, y0:y1, x0:x1] = round_ste(patch).clamp(0.0, 255.0)
                pose = posenet(posenet.preprocess_input(candidate))["pose"][..., :6]
                pose_per = (pose - target_pose).pow(2).mean(dim=1)
                pose_term = torch.sqrt(10.0 * pose_per + 1e-8).mean()
                pose_mse = pose_per.mean()
                tv = tv_patch(patch_small)
                loss = args.pose_term_weight * pose_term + args.pose_mse_weight * pose_mse + args.patch_tv_weight * tv
                loss.backward()
                torch.nn.utils.clip_grad_norm_([param], args.grad_clip)
                opt.step()

                with torch.no_grad():
                    pose_dist = pose_per.mean().item()
                    if pose_dist < best_pose - args.early_stop_epsilon:
                        best_pose = pose_dist
                        best_frames = candidate.detach().cpu().contiguous()
                        best_step = step
                    if step == 1 or step % args.log_every == 0 or step == args.patch_steps:
                        last = {
                            "step": step,
                            "best_step": best_step,
                            "pose_mse": float(pose_dist),
                            "pose_term": float(math.sqrt(max(0.0, 10.0 * pose_dist))),
                            "loss": float(loss.item()),
                            "tv": float(tv.item()),
                        }
                        pbar.set_postfix(P=f"{pose_dist:.6f}", T=f"{last['pose_term']:.4f}")
            chunks.append(best_frames)
            chunks_logs.append(
                {
                    "chunk": chunk_id,
                    "initial_posenet_dist": float(base_pose),
                    "best_posenet_dist": float(best_pose),
                    "best_step": int(best_step),
                    **last,
                }
            )

        candidate_frames = torch.cat(chunks, dim=0).contiguous()
        pose_dist = sum(x["best_posenet_dist"] for x in chunks_logs) / max(1, len(chunks_logs))
        position_logs.append(
            {
                "position": position,
                "patch_size": args.patch_size,
                "patch_grid": args.patch_grid,
                "bounds_yxyx": [int(y0), int(x0), int(y1), int(x1)],
                "mean_best_posenet_dist_unweighted": float(pose_dist),
                "chunks": chunks_logs,
            }
        )
        if pose_dist < best_global_pose:
            best_global_pose = pose_dist
            best_global_frames = candidate_frames
            best_global_label = position

    if best_global_frames is None:
        raise RuntimeError("no patch positions were evaluated")
    return best_global_frames, {"best_position": best_global_label, "positions": position_logs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="q55_pose_control_oracle")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--indices", default="")
    parser.add_argument("--offset", type=int, default=56)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--controls",
        default="baseline,orig_f1,lowres_y,lowres_rgb,affine",
        help="comma-separated controls: baseline,orig_f1,lowres_y,lowres_rgb,affine,patch",
    )
    parser.add_argument("--lowres-sizes", default="16x12,24x18,32x24,48x36")
    parser.add_argument("--opt-batch-size", type=int, default=1)
    parser.add_argument("--pose-term-weight", type=float, default=80.0)
    parser.add_argument("--pose-mse-weight", type=float, default=20.0)
    parser.add_argument("--affine-steps", type=int, default=250)
    parser.add_argument("--affine-lr", type=float, default=0.002)
    parser.add_argument("--affine-reg-weight", type=float, default=0.01)
    parser.add_argument("--patch-steps", type=int, default=300)
    parser.add_argument("--patch-lr", type=float, default=0.02)
    parser.add_argument("--patch-size", default="64x32")
    parser.add_argument("--patch-grid", default="12x8")
    parser.add_argument("--patch-positions", default="bottom_center,center,top_center")
    parser.add_argument("--patch-tv-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-epsilon", type=float, default=1e-9)
    parser.add_argument("--save-per-sample-metrics", action="store_true")
    parser.add_argument("--tail-top-k", type=int, default=16)
    args = parser.parse_args()

    if not args.base_archive.exists():
        raise FileNotFoundError(args.base_archive)
    device = torch.device(args.device)
    controls = {x.strip() for x in args.controls.split(",") if x.strip()}
    indices = parse_indices(args)

    run_dir = args.out_dir / args.label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    segnet, posenet = load_evaluators(device)

    print(f"Loading source RGB samples: {indices}", flush=True)
    gt_pairs = load_rgb_indices(
        args.video_names,
        args.video_dir,
        indices=indices,
        decode_batch_size=args.decode_batch_size,
    )

    print("Building evaluator targets...", flush=True)
    seg_targets, pose_targets = build_targets(gt_pairs, segnet, posenet, device, args.target_batch_size)

    print("Generating #55/qpack starting frames from archive payload...", flush=True)
    start_low = load_start_frames_low(
        args.base_archive,
        indices=torch.tensor(indices, dtype=torch.long),
        device=device,
        gen_batch_size=args.gen_batch_size,
    )
    with torch.no_grad():
        base_eval = render_for_eval(start_low.to(device).float(), camera_sim=True).cpu().contiguous()
    gt_eval = to_eval_pairs(gt_pairs, device).cpu().contiguous()
    archive_bytes = args.base_archive.stat().st_size

    records: dict[str, dict] = {}

    def eval_and_record(name: str, frames: torch.Tensor, extra: dict | None = None) -> dict:
        metrics = compute_metrics_eval(
            frames,
            seg_targets,
            pose_targets,
            segnet,
            posenet,
            device,
            batch_size=args.eval_batch_size,
            archive_bytes=archive_bytes,
            sample_indices=indices,
            include_per_sample=args.save_per_sample_metrics,
            top_k=args.tail_top_k,
        )
        record = {"control": name, "metrics": metrics}
        if extra:
            record.update(extra)
        records[name] = record
        print(f"{name}: {json.dumps(metrics, indent=2)}", flush=True)
        return record

    eval_and_record("baseline", base_eval)

    if "orig_f1" in controls:
        eval_and_record("orig_f1", replace_frame1(base_eval, gt_eval[:, 0]))

    lowres_sizes = [x.strip() for x in args.lowres_sizes.split(",") if x.strip()]
    for control, mode in (("lowres_y", "y"), ("lowres_rgb", "rgb")):
        if control not in controls:
            continue
        for size_text in lowres_sizes:
            size_hw = parse_size(size_text)
            f1 = lowres_frame1(gt_eval[:, 0].to(device), size_hw, mode).cpu().contiguous()
            eval_and_record(f"{control}_{size_text}", replace_frame1(base_eval, f1), {"lowres_size": size_text})

    if "affine" in controls:
        frames, log = optimize_affine_frame1(base_eval, pose_targets, posenet, device, args)
        eval_and_record("affine", frames, {"optimization": log})

    if "patch" in controls:
        frames, log = optimize_patch_frame1(base_eval, pose_targets, posenet, device, args)
        eval_and_record("patch", frames, {"optimization": log})

    baseline = records["baseline"]["metrics"]
    sorted_records = sorted(records.values(), key=lambda x: x["metrics"]["quality_term"])
    best = sorted_records[0]
    pose_baseline = float(baseline["posenet_dist"])
    for record in records.values():
        pose = float(record["metrics"]["posenet_dist"])
        record["pose_dist_delta_vs_baseline"] = pose - pose_baseline
        record["pose_dist_drop_fraction_vs_baseline"] = (
            0.0 if pose_baseline <= 0 else (pose_baseline - pose) / pose_baseline
        )

    output = {
        "label": args.label,
        "base_archive": str(args.base_archive),
        "base_archive_sha256": sha256_file(args.base_archive),
        "archive_bytes": archive_bytes,
        "device": str(device),
        "sample_indices": indices,
        "controls": sorted(controls),
        "best_control": best["control"],
        "best_metrics": best["metrics"],
        "records": records,
    }
    write_json(run_dir / "metrics.json", output)
    append_jsonl(args.out_dir / "pose_control_oracle_results.jsonl", output)
    print(f"Wrote {run_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
