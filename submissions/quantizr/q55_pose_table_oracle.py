#!/usr/bin/env python
"""Pose-side-channel inversion oracle for the Quantizr #55 exact-mask path.

This does not create a contest submission by default. It tests whether the
stored 6D pose-conditioning table is a better control surface for the frozen
#55 generator than output-space frame edits. The evaluator target pose remains
the original-video PoseNet output; only the pose vector fed to the inflater is
optimized.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import inflate as q55_inflate
from pack_pose import build_pose_qpack
from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    ORIGINAL_BYTES,
    POSE_PAYLOAD,
    POSE_QPACK_PAYLOAD,
    append_jsonl,
    make_archive_zip,
    score_from_bytes,
    sha256_file,
    unzip_archive,
    write_json,
)
from q55_pixel_oracle import build_targets, load_evaluators, render_for_eval
from q55_pose_control_oracle import compute_metrics_eval, load_rgb_indices


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


def load_generator_payloads(
    archive_zip: Path,
    device: torch.device,
) -> tuple[q55_inflate.JointFrameGenerator, torch.Tensor, torch.Tensor, Path]:
    """Load the frozen inflater, decoded exact masks, and stored pose table."""
    tmp_path = Path(tempfile.mkdtemp(prefix="q55_pose_table_"))
    archive_dir = tmp_path / "archive"
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
    for p in generator.parameters():
        p.requires_grad = False

    mask_all = q55_inflate.load_mask_payload(archive_dir, archive_dir / MASK_PAYLOAD).contiguous()
    pose_all = q55_inflate.load_pose_payload(archive_dir, archive_dir / POSE_PAYLOAD).contiguous()

    return generator, mask_all, pose_all, tmp_path


def cleanup_loaded_temp(tmp_path: Path) -> None:
    shutil.rmtree(tmp_path, ignore_errors=True)


def pose_sigma(pose_all: torch.Tensor) -> torch.Tensor:
    sigma = pose_all.float().std(dim=0, unbiased=False)
    return sigma.clamp_min(1.0e-4)


def fake_quant_pose(
    pose: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    bits: int,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return pose
    levels = float((1 << bits) - 1)
    scale = (hi - lo).clamp_min(1.0e-12)
    q = ((pose - lo[None, :]) / scale[None, :] * levels).clamp(0.0, levels)
    q_round = q + (q.round() - q).detach()
    return q_round / levels * scale[None, :] + lo[None, :]


def pose_from_param(
    param: torch.Tensor,
    p0: torch.Tensor,
    sigma: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.param_mode == "bounded":
        return p0 + args.scale * sigma[None, :] * torch.tanh(param)
    if args.param_mode == "unbounded":
        return p0 + sigma[None, :] * param
    raise ValueError(f"unknown param mode: {args.param_mode}")


def init_param_from_pose(
    pose: torch.Tensor,
    p0: torch.Tensor,
    sigma: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.param_mode == "bounded":
        denom = (args.scale * sigma[None, :]).clamp_min(1.0e-12)
        x = ((pose - p0) / denom).clamp(-0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    if args.param_mode == "unbounded":
        return (pose - p0) / sigma[None, :].clamp_min(1.0e-12)
    raise ValueError(f"unknown param mode: {args.param_mode}")


def generate_eval_frames(
    generator: q55_inflate.JointFrameGenerator,
    mask: torch.Tensor,
    pose: torch.Tensor,
    *,
    camera_sim: bool = True,
) -> torch.Tensor:
    p1, p2 = generator(mask.long(), pose.float())
    low = torch.stack([p1, p2], dim=1)
    return render_for_eval(low, camera_sim=camera_sim)


def pose_loss_for_frames(
    frames_eval: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose_pred = posenet(posenet.preprocess_input(frames_eval))["pose"][..., :6]
    pose_per = (pose_pred - pose_targets).pow(2).mean(dim=1)
    pose_mse = pose_per.mean()
    pose_term = torch.sqrt(10.0 * pose_per + 1.0e-10).mean()
    return pose_mse, pose_term, pose_per


@torch.inference_mode()
def compute_baseline_frames(
    generator: q55_inflate.JointFrameGenerator,
    mask_all: torch.Tensor,
    pose_all: torch.Tensor,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    frames = []
    index_tensor = torch.tensor(indices, dtype=torch.long)
    for start in range(0, len(indices), batch_size):
        idx = index_tensor[start : start + batch_size]
        mask = mask_all.index_select(0, idx.cpu()).to(device)
        pose = pose_all.index_select(0, idx.cpu()).to(device).float()
        frames.append(generate_eval_frames(generator, mask, pose).cpu())
    return torch.cat(frames, dim=0).contiguous()


@torch.inference_mode()
def evaluate_pose_candidates(
    generator: q55_inflate.JointFrameGenerator,
    mask_one: torch.Tensor,
    candidates: torch.Tensor,
    target_pose: torch.Tensor,
    posenet,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    losses = []
    target = target_pose.to(device).float()
    for start in range(0, candidates.shape[0], args.candidate_batch_size):
        cand = candidates[start : start + args.candidate_batch_size].to(device).float()
        mask = mask_one.to(device).expand(cand.shape[0], -1, -1)
        frames = generate_eval_frames(generator, mask, cand)
        pose_pred = posenet(posenet.preprocess_input(frames))["pose"][..., :6]
        per = (pose_pred - target.expand(cand.shape[0], -1)).pow(2).mean(dim=1)
        losses.append(per.detach().cpu())
    return torch.cat(losses, dim=0)


def optimize_cem(
    generator: q55_inflate.JointFrameGenerator,
    mask_all: torch.Tensor,
    p0: torch.Tensor,
    pose_targets: torch.Tensor,
    sample_indices: list[int],
    sigma: torch.Tensor,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    posenet,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict]]:
    optimized = []
    logs = []
    elite_count = max(1, int(round(args.cem_candidates * args.cem_elite_frac)))
    sigma_cpu = sigma.detach().cpu()
    q_lo_cpu = q_lo.detach().cpu()
    q_hi_cpu = q_hi.detach().cpu()

    for local_idx, sample_idx in enumerate(sample_indices):
        mean = p0[local_idx].detach().cpu().float()
        std = (args.scale * sigma_cpu).clamp_min(1.0e-5)
        best_pose = mean.clone()
        best_loss = float("inf")
        iter_logs = []
        mask_one = mask_all[int(sample_idx)].contiguous()
        target_pose = pose_targets[local_idx].detach().cpu().float()

        for iteration in range(args.cem_iterations):
            noise = torch.randn(args.cem_candidates, 6) * std[None, :]
            candidates = mean[None, :] + noise
            if args.param_mode == "bounded":
                lo = p0[local_idx].detach().cpu() - args.scale * sigma_cpu
                hi = p0[local_idx].detach().cpu() + args.scale * sigma_cpu
                candidates = candidates.clamp(lo[None, :], hi[None, :])
            candidates = fake_quant_pose(
                candidates,
                q_lo_cpu,
                q_hi_cpu,
                bits=args.quant_bits,
                enabled=args.fake_quant,
            )
            losses = evaluate_pose_candidates(
                generator,
                mask_one,
                candidates,
                target_pose,
                posenet,
                device,
                args,
            )
            order = torch.argsort(losses)
            elites = candidates.index_select(0, order[:elite_count])
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp_min(args.cem_min_std * sigma_cpu)
            if float(losses[order[0]].item()) < best_loss:
                best_loss = float(losses[order[0]].item())
                best_pose = candidates[int(order[0])].detach().cpu()
            iter_logs.append(
                {
                    "iteration": iteration + 1,
                    "best_pose_mse": float(best_loss),
                    "elite_mean_pose_mse": float(losses[order[:elite_count]].mean().item()),
                    "std_mean_sigma_units": float((std / sigma_cpu).mean().item()),
                }
            )

        optimized.append(best_pose)
        logs.append(
            {
                "sample_index": int(sample_idx),
                "best_pose_mse": best_loss,
                "iterations": iter_logs,
                "delta_sigma_units": [
                    float(x) for x in ((best_pose - p0[local_idx].detach().cpu()) / sigma_cpu).tolist()
                ],
            }
        )
    return torch.stack(optimized, dim=0).contiguous(), logs


def optimize_adam(
    generator: q55_inflate.JointFrameGenerator,
    mask_all: torch.Tensor,
    init_pose: torch.Tensor,
    p0: torch.Tensor,
    pose_targets: torch.Tensor,
    sample_indices: list[int],
    sigma: torch.Tensor,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    posenet,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict]]:
    optimized_chunks = []
    logs = []
    index_tensor = torch.tensor(sample_indices, dtype=torch.long)
    sigma_device = sigma.to(device).float()
    q_lo_device = q_lo.to(device).float()
    q_hi_device = q_hi.to(device).float()

    for chunk_id, start in enumerate(range(0, len(sample_indices), args.opt_batch_size)):
        end = min(len(sample_indices), start + args.opt_batch_size)
        idx = index_tensor[start:end]
        chunk_indices = sample_indices[start:end]
        mask = mask_all.index_select(0, idx.cpu()).to(device)
        p0_chunk = p0[start:end].to(device).float()
        init_chunk = init_pose[start:end].to(device).float()
        target_pose = pose_targets[start:end].to(device).float()
        param = init_param_from_pose(init_chunk, p0_chunk, sigma_device, args).detach().requires_grad_(True)
        opt = torch.optim.AdamW([param], lr=args.lr)

        with torch.no_grad():
            base_pose = fake_quant_pose(
                p0_chunk,
                q_lo_device,
                q_hi_device,
                bits=args.quant_bits,
                enabled=args.fake_quant,
            )
            base_frames = generate_eval_frames(generator, mask, base_pose)
            base_mse, base_term, _ = pose_loss_for_frames(base_frames, target_pose, posenet)
            best_loss = float((args.pose_term_weight * base_term + args.pose_mse_weight * base_mse).item())
            best_pose = base_pose.detach().cpu().contiguous()
            best_step = 0

        pbar = tqdm(range(1, args.steps + 1), desc=f"Pose table chunk {chunk_id}", leave=False)
        last = {}
        for step in pbar:
            opt.zero_grad(set_to_none=True)
            pose_cont = pose_from_param(param, p0_chunk, sigma_device, args)
            pose_ctrl = fake_quant_pose(
                pose_cont,
                q_lo_device,
                q_hi_device,
                bits=args.quant_bits,
                enabled=args.fake_quant,
            )
            frames = generate_eval_frames(generator, mask, pose_ctrl)
            pose_mse, pose_term, pose_per = pose_loss_for_frames(frames, target_pose, posenet)
            reg = ((pose_cont - p0_chunk) / sigma_device[None, :]).pow(2).mean()
            loss = args.pose_term_weight * pose_term + args.pose_mse_weight * pose_mse + args.l2_weight * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param], args.grad_clip)
            opt.step()

            with torch.no_grad():
                loss_value = float(loss.item())
                if loss_value < best_loss - args.early_stop_epsilon:
                    best_loss = loss_value
                    best_step = step
                    best_pose = pose_ctrl.detach().cpu().contiguous()
                if step == 1 or step % args.log_every == 0 or step == args.steps:
                    last = {
                        "step": int(step),
                        "loss": loss_value,
                        "pose_mse": float(pose_mse.item()),
                        "pose_term": float(pose_term.item()),
                        "reg": float(reg.item()),
                        "best_step": int(best_step),
                        "best_loss": float(best_loss),
                    }
                    pbar.set_postfix(P=f"{float(pose_mse.item()):.6f}", T=f"{float(pose_term.item()):.4f}")
                if (
                    args.early_stop_patience > 0
                    and step >= args.early_stop_min_step
                    and step - best_step >= args.early_stop_patience
                ):
                    last["early_stopped"] = True
                    break

        optimized_chunks.append(best_pose)
        logs.append(
            {
                "chunk": int(chunk_id),
                "sample_indices": [int(x) for x in chunk_indices],
                "baseline_pose_mse": float(base_mse.item()),
                "baseline_pose_term": float(base_term.item()),
                "best_step": int(best_step),
                "best_loss": float(best_loss),
                **last,
            }
        )

    return torch.cat(optimized_chunks, dim=0).contiguous(), logs


@torch.inference_mode()
def frames_for_pose_table(
    generator: q55_inflate.JointFrameGenerator,
    mask_all: torch.Tensor,
    pose_ctrl: torch.Tensor,
    sample_indices: list[int],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    frames = []
    index_tensor = torch.tensor(sample_indices, dtype=torch.long)
    for start in range(0, len(sample_indices), batch_size):
        end = min(len(sample_indices), start + batch_size)
        idx = index_tensor[start:end]
        mask = mask_all.index_select(0, idx.cpu()).to(device)
        pose = pose_ctrl[start:end].to(device).float()
        frames.append(generate_eval_frames(generator, mask, pose).cpu())
    return torch.cat(frames, dim=0).contiguous()


def package_pose_archive(
    base_archive: Path,
    out_dir: Path,
    pose_all: torch.Tensor,
    pose_ctrl: torch.Tensor,
    sample_indices: list[int],
    variant: str,
) -> Path:
    archive_dir = out_dir / "archive"
    unzip_archive(base_archive, archive_dir)
    full_pose = pose_all.detach().cpu().numpy().astype(np.float32, copy=True)
    for local_idx, sample_idx in enumerate(sample_indices):
        full_pose[int(sample_idx)] = pose_ctrl[local_idx].detach().cpu().numpy().astype(np.float32, copy=False)
    qpack = build_pose_qpack(full_pose, variant)
    (archive_dir / POSE_QPACK_PAYLOAD).write_bytes(brotli.compress(qpack, quality=11, lgwin=24))
    legacy_pose = archive_dir / POSE_PAYLOAD
    if legacy_pose.exists():
        legacy_pose.unlink()
    archive_zip = out_dir / "archive.zip"
    make_archive_zip(archive_dir, archive_zip)
    return archive_zip


def add_delta_report(record: dict, baseline: dict, final: dict) -> None:
    base_pose = float(baseline["posenet_dist"])
    final_pose = float(final["posenet_dist"])
    base_quality = float(baseline["quality_term"])
    final_quality = float(final["quality_term"])
    record["pose_dist_drop_fraction_vs_baseline"] = (base_pose - final_pose) / max(base_pose, 1.0e-12)
    record["quality_delta_vs_baseline"] = final_quality - base_quality
    record["score_delta_vs_baseline"] = (
        float(final["projected_score_at_archive_bytes"]) - float(baseline["projected_score_at_archive_bytes"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="q55_pose_table_oracle")
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
    parser.add_argument("--opt-batch-size", type=int, default=3)
    parser.add_argument("--mode", choices=["adam", "cem", "cem_adam"], default="cem_adam")
    parser.add_argument("--param-mode", choices=["bounded", "unbounded"], default="bounded")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--pose-term-weight", type=float, default=80.0)
    parser.add_argument("--pose-mse-weight", type=float, default=20.0)
    parser.add_argument("--l2-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-min-step", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-epsilon", type=float, default=1e-10)
    parser.add_argument("--fake-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quant-bits", type=int, default=10)
    parser.add_argument("--pose-pack-variant", choices=["fp16", "int16_per_dim", "int12_per_dim", "int10_per_dim"], default="int10_per_dim")
    parser.add_argument("--cem-candidates", type=int, default=1024)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--cem-elite-frac", type=float, default=0.05)
    parser.add_argument("--cem-min-std", type=float, default=0.03)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--package", action="store_true")
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

    indices = parse_indices(args)
    print(f"Selected sample indices: {indices}", flush=True)

    segnet, posenet = load_evaluators(device)

    print("Loading source RGB samples...", flush=True)
    gt_pairs = load_rgb_indices(
        args.video_names,
        args.video_dir,
        indices=indices,
        decode_batch_size=args.decode_batch_size,
    )
    print("Building evaluator targets...", flush=True)
    seg_targets, pose_targets = build_targets(gt_pairs, segnet, posenet, device, args.target_batch_size)

    print("Loading frozen #55/qpack generator, masks, and pose table...", flush=True)
    tmp_path: Path | None = None
    try:
        generator, mask_all, pose_all, tmp_path = load_generator_payloads(args.base_archive, device)
        p0 = pose_all.index_select(0, torch.tensor(indices, dtype=torch.long)).float()
        sigma = pose_sigma(pose_all)
        q_lo = pose_all.float().min(dim=0).values
        q_hi = pose_all.float().max(dim=0).values

        archive_bytes = args.base_archive.stat().st_size
        baseline_frames = compute_baseline_frames(
            generator,
            mask_all,
            pose_all,
            indices,
            device,
            args.gen_batch_size,
        )
        baseline_metrics = compute_metrics_eval(
            baseline_frames,
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
        print("Baseline metrics:", json.dumps(baseline_metrics, indent=2), flush=True)

        logs: dict[str, list[dict]] = {}
        init_pose = p0.clone()
        if args.mode in {"cem", "cem_adam"}:
            print("Running CEM pose-input search...", flush=True)
            init_pose, cem_logs = optimize_cem(
                generator,
                mask_all,
                p0,
                pose_targets,
                indices,
                sigma,
                q_lo,
                q_hi,
                posenet,
                device,
                args,
            )
            logs["cem"] = cem_logs

        if args.mode in {"adam", "cem_adam"}:
            print("Running Adam pose-input refinement...", flush=True)
            final_pose, adam_logs = optimize_adam(
                generator,
                mask_all,
                init_pose,
                p0,
                pose_targets,
                indices,
                sigma,
                q_lo,
                q_hi,
                posenet,
                device,
                args,
            )
            logs["adam"] = adam_logs
        else:
            final_pose = init_pose

        final_frames = frames_for_pose_table(
            generator,
            mask_all,
            final_pose,
            indices,
            device,
            args.gen_batch_size,
        )
        final_metrics = compute_metrics_eval(
            final_frames,
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
        print("Final metrics:", json.dumps(final_metrics, indent=2), flush=True)

        np.save(run_dir / "pose_ctrl.npy", final_pose.detach().cpu().numpy().astype(np.float32))
        full_pose = pose_all.detach().cpu().numpy().astype(np.float32, copy=True)
        for local_idx, sample_idx in enumerate(indices):
            full_pose[int(sample_idx)] = final_pose[local_idx].detach().cpu().numpy().astype(np.float32, copy=False)
        np.save(run_dir / "pose_ctrl_full.npy", full_pose)

        archive_zip: Path | None = None
        packaged_archive_bytes = archive_bytes
        if args.package:
            archive_zip = package_pose_archive(
                args.base_archive,
                run_dir / "submission",
                pose_all,
                final_pose,
                indices,
                args.pose_pack_variant,
            )
            packaged_archive_bytes = archive_zip.stat().st_size

        record = {
            "label": args.label,
            "base_archive": str(args.base_archive),
            "base_archive_sha256": sha256_file(args.base_archive),
            "archive_bytes": archive_bytes,
            "packaged_archive_bytes": packaged_archive_bytes,
            "archive_zip": str(archive_zip) if archive_zip else "",
            "device": str(device),
            "indices": [int(x) for x in indices],
            "mode": args.mode,
            "param_mode": args.param_mode,
            "scale": args.scale,
            "steps": args.steps,
            "lr": args.lr,
            "fake_quant": args.fake_quant,
            "quant_bits": args.quant_bits,
            "pose_pack_variant": args.pose_pack_variant,
            "pose_sigma": [float(x) for x in sigma.tolist()],
            "pose_min": [float(x) for x in q_lo.tolist()],
            "pose_max": [float(x) for x in q_hi.tolist()],
            "baseline": baseline_metrics,
            "final": final_metrics,
            "logs": logs,
            "projected_full_score_if_same_quality": final_metrics["quality_term"] + 25.0 * packaged_archive_bytes / ORIGINAL_BYTES,
            "quality_required_for_score_lt_0_300_at_packaged_bytes": 0.300 - 25.0 * packaged_archive_bytes / ORIGINAL_BYTES,
        }
        add_delta_report(record, baseline_metrics, final_metrics)
        write_json(run_dir / "metrics.json", record)
        append_jsonl(args.out_dir / "pose_table_oracle_results.jsonl", record)
        print(f"Wrote {run_dir / 'metrics.json'}", flush=True)
    finally:
        if tmp_path is not None:
            cleanup_loaded_temp(tmp_path)


if __name__ == "__main__":
    main()
