#!/usr/bin/env python
"""Exact-mask student-model frontier for the Quantizr #55 path.

This tests the remaining high-leverage axis: keep the exact #55 mask/pose
payloads and replace the decoder with a smaller task-distilled student.
The output archive is inflate-compatible through arch.json.br + model.qpack.br.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
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

import compress as q55
import inflate as q55_inflate
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path
from pack_model import build_qpack
from q55_common import (
    ARCH_PAYLOAD,
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
    summarize_archive,
    unzip_archive,
    write_json,
)
from q55_pixel_oracle import build_targets, load_evaluators, render_for_eval
from q55_pose_control_oracle import compute_metrics_eval, load_rgb_indices


BUDGET_CONFIGS: dict[str, dict] = {
    "S8": {"cond_dim": 16, "depth_mult": 1, "shared_c1": 16, "shared_c2": 24, "frame_hidden": 16, "padding_mode": "zeros"},
    "S16": {"cond_dim": 24, "depth_mult": 1, "shared_c1": 24, "shared_c2": 32, "frame_hidden": 24, "padding_mode": "zeros"},
    "S28": {"cond_dim": 32, "depth_mult": 1, "shared_c1": 32, "shared_c2": 40, "frame_hidden": 32, "padding_mode": "zeros"},
    "S40": {"cond_dim": 40, "depth_mult": 1, "shared_c1": 40, "shared_c2": 48, "frame_hidden": 40, "padding_mode": "zeros"},
}


@dataclass
class LoadedArchive:
    archive_dir: Path
    tmp_path: Path
    teacher: q55.JointFrameGenerator
    mask_all: torch.Tensor
    pose_all: torch.Tensor
    arch_config: dict


def parse_indices(args: argparse.Namespace) -> list[int]:
    if args.indices:
        out = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        out = list(range(args.offset, args.offset + args.max_samples))
    if not out:
        raise ValueError("no indices selected")
    return out


def load_teacher_archive(base_archive: Path, device: torch.device) -> LoadedArchive:
    tmp_path = Path(tempfile.mkdtemp(prefix="q55_student_"))
    archive_dir = tmp_path / "archive"
    unzip_archive(base_archive, archive_dir)
    arch_config = q55_inflate.load_arch_config(archive_dir)

    teacher = q55.JointFrameGenerator(**arch_config).to(device)
    model_qpack = archive_dir / MODEL_QPACK_PAYLOAD
    if model_qpack.exists():
        state = q55_inflate.get_qpack_state_dict(brotli.decompress(model_qpack.read_bytes()), device)
    else:
        state = q55_inflate.get_decoded_state_dict(brotli.decompress((archive_dir / MODEL_PAYLOAD).read_bytes()), device)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    mask_all = q55_inflate.load_mask_payload(archive_dir, archive_dir / MASK_PAYLOAD).contiguous()
    pose_all = q55_inflate.load_pose_payload(archive_dir, archive_dir / POSE_PAYLOAD).contiguous()
    return LoadedArchive(archive_dir, tmp_path, teacher, mask_all, pose_all, arch_config)


def cleanup_loaded(loaded: LoadedArchive) -> None:
    shutil.rmtree(loaded.tmp_path, ignore_errors=True)


@torch.inference_mode()
def generate_low(
    model: q55.JointFrameGenerator,
    mask: torch.Tensor,
    pose: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    frames = []
    for start in range(0, mask.shape[0], batch_size):
        m = mask[start : start + batch_size].to(device).long()
        p = pose[start : start + batch_size].to(device).float()
        p1, p2 = model(m, p)
        frames.append(torch.stack([p1, p2], dim=1).detach().cpu())
    return torch.cat(frames, dim=0).contiguous()


@torch.inference_mode()
def build_target_logits(
    gt_pairs_u8: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seg_logits, seg_targets, pose_targets = [], [], []
    for start in range(0, gt_pairs_u8.shape[0], batch_size):
        gt = gt_pairs_u8[start : start + batch_size].to(device).float()
        gt = einops.rearrange(gt, "b t h w c -> b t c h w")
        logits = segnet(segnet.preprocess_input(gt)).float()
        pose = posenet(posenet.preprocess_input(gt))["pose"][..., :6].float()
        seg_logits.append(logits.cpu())
        seg_targets.append(logits.argmax(dim=1).cpu())
        pose_targets.append(pose.cpu())
    return torch.cat(seg_logits), torch.cat(seg_targets), torch.cat(pose_targets)


def kl_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature * temperature) / (
        teacher_logits.shape[-1] * teacher_logits.shape[-2]
    )


def sample_batch(
    sample_count: int,
    batch_size: int,
    hard_local: torch.Tensor,
    hard_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    hard_n = int(round(batch_size * hard_fraction)) if hard_local.numel() else 0
    hard_n = min(hard_n, batch_size)
    random_n = batch_size - hard_n
    parts = []
    if hard_n:
        hidx = torch.randint(0, hard_local.numel(), (hard_n,), device="cpu")
        parts.append(hard_local.index_select(0, hidx))
    if random_n:
        parts.append(torch.randint(0, sample_count, (random_n,), device="cpu"))
    idx = torch.cat(parts, dim=0)
    if idx.numel() > 1:
        idx = idx[torch.randperm(idx.numel())]
    return idx.to(device)


def student_loss(
    student: q55.JointFrameGenerator,
    batch_idx: torch.Tensor,
    mask_cpu: torch.Tensor,
    pose_cpu: torch.Tensor,
    teacher_low_cpu: torch.Tensor,
    seg_logits_cpu: torch.Tensor,
    seg_targets_cpu: torch.Tensor,
    pose_targets_cpu: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    args: argparse.Namespace,
    *,
    stage: str,
) -> tuple[torch.Tensor, dict]:
    mask = mask_cpu.index_select(0, batch_idx.cpu()).to(device).long()
    pose = pose_cpu.index_select(0, batch_idx.cpu()).to(device).float()
    teacher_low = teacher_low_cpu.index_select(0, batch_idx.cpu()).to(device).float()
    target_logits = seg_logits_cpu.index_select(0, batch_idx.cpu()).to(device).float()
    target_seg = seg_targets_cpu.index_select(0, batch_idx.cpu()).to(device).long()
    target_pose = pose_targets_cpu.index_select(0, batch_idx.cpu()).to(device).float()

    pred1, pred2 = student(mask, pose)
    low = torch.stack([pred1, pred2], dim=1)
    frame_loss = F.huber_loss(low / 255.0, teacher_low / 255.0, delta=0.05)

    if stage == "teacher":
        return args.frame_weight * frame_loss, {"frame": float(frame_loss.detach().item())}

    eval_frames = render_for_eval(low, camera_sim=True)
    seg_logits = segnet(segnet.preprocess_input(eval_frames))
    pose_pred = posenet(posenet.preprocess_input(eval_frames))["pose"][..., :6]
    ce = F.cross_entropy(seg_logits, target_seg)
    kl = kl_logits(seg_logits, target_logits, temperature=args.kl_temperature)
    pose_mse = F.mse_loss(pose_pred, target_pose)

    loss = (
        args.frame_anchor_weight * frame_loss
        + args.seg_ce_weight * ce
        + args.seg_kl_weight * kl
        + args.pose_mse_weight * pose_mse
    )
    return loss, {
        "frame": float(frame_loss.detach().item()),
        "ce": float(ce.detach().item()),
        "kl": float(kl.detach().item()),
        "pose_mse": float(pose_mse.detach().item()),
    }


def train_stage(
    student: q55.JointFrameGenerator,
    stage: str,
    steps: int,
    lr: float,
    mask_cpu: torch.Tensor,
    pose_cpu: torch.Tensor,
    teacher_low_cpu: torch.Tensor,
    seg_logits_cpu: torch.Tensor,
    seg_targets_cpu: torch.Tensor,
    pose_targets_cpu: torch.Tensor,
    hard_local: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    if steps <= 0:
        return []
    student.train()
    student.set_qat(stage == "qat")
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    logs = []
    pbar = tqdm(range(1, steps + 1), desc=f"{stage} train", leave=False)
    for step in pbar:
        idx = sample_batch(mask_cpu.shape[0], args.batch_size, hard_local, args.hard_fraction, device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = student_loss(
            student,
            idx,
            mask_cpu,
            pose_cpu,
            teacher_low_cpu,
            seg_logits_cpu,
            seg_targets_cpu,
            pose_targets_cpu,
            segnet,
            posenet,
            device,
            args,
            stage="teacher" if stage == "teacher" else "task",
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == steps:
            row = {"stage": stage, "step": step, "loss": float(loss.detach().item()), **parts}
            logs.append(row)
            pbar.set_postfix(L=f"{row['loss']:.4f}", P=f"{parts.get('pose_mse', 0.0):.5f}")
    student.set_qat(False)
    return logs


@torch.inference_mode()
def compute_student_metrics(
    student: q55.JointFrameGenerator,
    mask_cpu: torch.Tensor,
    pose_cpu: torch.Tensor,
    seg_targets: torch.Tensor,
    pose_targets: torch.Tensor,
    segnet: SegNet,
    posenet: PoseNet,
    device: torch.device,
    args: argparse.Namespace,
    archive_bytes: int,
    sample_indices: list[int],
) -> tuple[dict, torch.Tensor]:
    student.eval()
    low = generate_low(student, mask_cpu, pose_cpu, device, args.gen_batch_size)
    eval_frames = render_for_eval(low.to(device).float(), camera_sim=True).cpu()
    metrics = compute_metrics_eval(
        eval_frames,
        seg_targets,
        pose_targets,
        segnet,
        posenet,
        device,
        batch_size=args.eval_batch_size,
        archive_bytes=archive_bytes,
        sample_indices=sample_indices,
        include_per_sample=args.save_per_sample_metrics,
        top_k=args.tail_top_k,
    )
    return metrics, low


def write_arch_payload(archive_dir: Path, arch_config: dict) -> None:
    payload = {"format": "quantizr_arch_v1", "config": arch_config}
    (archive_dir / ARCH_PAYLOAD).write_bytes(
        brotli.compress(json.dumps(payload, sort_keys=True).encode("utf-8"), quality=11, lgwin=24)
    )


def export_student_archive(
    student: q55.JointFrameGenerator,
    loaded: LoadedArchive,
    run_dir: Path,
    arch_config: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Path, dict]:
    archive_dir = run_dir / "archive"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)

    shutil.copy2(loaded.archive_dir / MASK_PAYLOAD, archive_dir / MASK_PAYLOAD)
    pose_payload = POSE_QPACK_PAYLOAD if (loaded.archive_dir / POSE_QPACK_PAYLOAD).exists() else POSE_PAYLOAD
    shutil.copy2(loaded.archive_dir / pose_payload, archive_dir / pose_payload)
    write_arch_payload(archive_dir, arch_config)

    fp4_path = run_dir / "student_fp4.pt"
    q55.export_fp4_state_dict(student.cpu(), fp4_path)
    student.to(device)
    data = torch.load(fp4_path, map_location="cpu")
    qpack = build_qpack(data, quantize_fp16=args.qpack_quantize)
    model_qpack = archive_dir / MODEL_QPACK_PAYLOAD
    model_qpack.write_bytes(brotli.compress(qpack, quality=11, lgwin=24))

    archive_zip = run_dir / "archive.zip"
    make_archive_zip(archive_dir, archive_zip, [MASK_PAYLOAD, pose_payload, ARCH_PAYLOAD, MODEL_QPACK_PAYLOAD])
    report = {
        "model_qpack_bytes": model_qpack.stat().st_size,
        "arch_payload_bytes": (archive_dir / ARCH_PAYLOAD).stat().st_size,
        "pose_payload": pose_payload,
        "pose_payload_bytes": (archive_dir / pose_payload).stat().st_size,
        "mask_payload_bytes": (archive_dir / MASK_PAYLOAD).stat().st_size,
        "archive_summary": summarize_archive(archive_zip),
    }
    return archive_zip, report


def arch_for_budget(name: str) -> dict:
    if name not in BUDGET_CONFIGS:
        raise ValueError(f"unknown budget {name}; choose one of {sorted(BUDGET_CONFIGS)}")
    return dict(BUDGET_CONFIGS[name])


def init_student_from_teacher(student: q55.JointFrameGenerator, teacher: q55.JointFrameGenerator) -> dict:
    """Copy same-name overlapping tensor slices from #55 into a smaller student.

    This is structured pruning initialization, not a strict checkpoint load: channel
    prefixes are reused when dimensions shrink, while unmatched new capacity keeps
    its normal PyTorch initialization.
    """
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    exact, partial, skipped = [], [], []
    with torch.no_grad():
        for name, dst in student_state.items():
            src = teacher_state.get(name)
            if src is None:
                skipped.append(name)
                continue
            src = src.to(device=dst.device, dtype=dst.dtype if torch.is_floating_point(dst) else src.dtype)
            if tuple(src.shape) == tuple(dst.shape):
                dst.copy_(src)
                exact.append(name)
                continue
            if src.ndim != dst.ndim or src.ndim == 0:
                skipped.append(name)
                continue
            slices = tuple(slice(0, min(int(d), int(s))) for d, s in zip(dst.shape, src.shape))
            if any(s.stop == 0 for s in slices):
                skipped.append(name)
                continue
            dst[slices].copy_(src[slices])
            partial.append(
                {
                    "name": name,
                    "student_shape": [int(x) for x in dst.shape],
                    "teacher_shape": [int(x) for x in src.shape],
                    "copied_shape": [int(s.stop) for s in slices],
                }
            )
    student.load_state_dict(student_state, strict=True)
    return {
        "exact_tensors": len(exact),
        "partial_tensors": len(partial),
        "skipped_tensors": len(skipped),
        "partial": partial[:64],
        "skipped": skipped[:64],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--budget", choices=sorted(BUDGET_CONFIGS), required=True)
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--indices", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--teacher-steps", type=int, default=300)
    parser.add_argument("--task-steps", type=int, default=500)
    parser.add_argument("--qat-steps", type=int, default=150)
    parser.add_argument("--teacher-lr", type=float, default=8e-4)
    parser.add_argument("--task-lr", type=float, default=2e-4)
    parser.add_argument("--qat-lr", type=float, default=8e-5)
    parser.add_argument("--frame-weight", type=float, default=8.0)
    parser.add_argument("--frame-anchor-weight", type=float, default=0.25)
    parser.add_argument("--seg-ce-weight", type=float, default=1.0)
    parser.add_argument("--seg-kl-weight", type=float, default=0.5)
    parser.add_argument("--pose-mse-weight", type=float, default=80.0)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    parser.add_argument("--hard-indices", default="59,60,62")
    parser.add_argument("--hard-fraction", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--qpack-quantize", choices=["none", "int8", "int8_heads_fp16"], default="none")
    parser.add_argument("--no-teacher-init", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-per-sample-metrics", action="store_true")
    parser.add_argument("--tail-top-k", type=int, default=16)
    args = parser.parse_args()

    if not args.base_archive.exists():
        raise FileNotFoundError(args.base_archive)
    device = torch.device(args.device)
    indices = parse_indices(args)
    label = args.label or f"q55_student_{args.budget}_o{indices[0]}_{len(indices)}s"
    run_dir = args.out_dir / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    arch_config = arch_for_budget(args.budget)
    loaded: LoadedArchive | None = None
    try:
        print("Loading evaluator models...", flush=True)
        segnet, posenet = load_evaluators(device)

        print(f"Loading RGB samples: {indices[:8]}{'...' if len(indices) > 8 else ''}", flush=True)
        gt_pairs = load_rgb_indices(args.video_names, args.video_dir, indices=indices, decode_batch_size=args.decode_batch_size)

        print("Building evaluator targets...", flush=True)
        seg_logits, seg_targets, pose_targets = build_target_logits(
            gt_pairs, segnet, posenet, device, args.target_batch_size
        )

        print("Loading #55 teacher archive and exact mask/pose payloads...", flush=True)
        loaded = load_teacher_archive(args.base_archive, device)
        index_tensor = torch.tensor(indices, dtype=torch.long)
        mask_subset = loaded.mask_all.index_select(0, index_tensor.cpu()).contiguous()
        pose_subset = loaded.pose_all.index_select(0, index_tensor.cpu()).contiguous()

        print("Generating #55 teacher frames...", flush=True)
        teacher_low = generate_low(loaded.teacher, mask_subset, pose_subset, device, args.gen_batch_size).half()
        teacher_eval = render_for_eval(teacher_low.to(device).float(), camera_sim=True).cpu()
        teacher_metrics = compute_metrics_eval(
            teacher_eval,
            seg_targets,
            pose_targets,
            segnet,
            posenet,
            device,
            batch_size=args.eval_batch_size,
            archive_bytes=args.base_archive.stat().st_size,
            sample_indices=indices,
            include_per_sample=args.save_per_sample_metrics,
            top_k=args.tail_top_k,
        )
        print("Teacher metrics:", json.dumps(teacher_metrics, indent=2), flush=True)

        student = q55.JointFrameGenerator(**arch_config).to(device)
        init_report = None
        if not args.no_teacher_init:
            print("Initializing student from overlapping #55 teacher weights...", flush=True)
            init_report = init_student_from_teacher(student, loaded.teacher)
            print("Init report:", json.dumps(init_report, indent=2), flush=True)
        hard_globals = {int(x.strip()) for x in args.hard_indices.split(",") if x.strip()}
        hard_local = torch.tensor([i for i, global_i in enumerate(indices) if global_i in hard_globals], dtype=torch.long)

        logs = []
        logs.extend(
            train_stage(
                student,
                "teacher",
                args.teacher_steps,
                args.teacher_lr,
                mask_subset,
                pose_subset,
                teacher_low,
                seg_logits,
                seg_targets,
                pose_targets,
                hard_local,
                segnet,
                posenet,
                device,
                args,
            )
        )
        teacher_stage_metrics, _ = compute_student_metrics(
            student,
            mask_subset,
            pose_subset,
            seg_targets,
            pose_targets,
            segnet,
            posenet,
            device,
            args,
            args.base_archive.stat().st_size,
            indices,
        )
        print("After teacher stage:", json.dumps(teacher_stage_metrics, indent=2), flush=True)

        logs.extend(
            train_stage(
                student,
                "task",
                args.task_steps,
                args.task_lr,
                mask_subset,
                pose_subset,
                teacher_low,
                seg_logits,
                seg_targets,
                pose_targets,
                hard_local,
                segnet,
                posenet,
                device,
                args,
            )
        )
        task_stage_metrics, _ = compute_student_metrics(
            student,
            mask_subset,
            pose_subset,
            seg_targets,
            pose_targets,
            segnet,
            posenet,
            device,
            args,
            args.base_archive.stat().st_size,
            indices,
        )
        print("After task stage:", json.dumps(task_stage_metrics, indent=2), flush=True)

        logs.extend(
            train_stage(
                student,
                "qat",
                args.qat_steps,
                args.qat_lr,
                mask_subset,
                pose_subset,
                teacher_low,
                seg_logits,
                seg_targets,
                pose_targets,
                hard_local,
                segnet,
                posenet,
                device,
                args,
            )
        )

        archive_zip, package_report = export_student_archive(student, loaded, run_dir, arch_config, device, args)
        final_metrics, _ = compute_student_metrics(
            student,
            mask_subset,
            pose_subset,
            seg_targets,
            pose_targets,
            segnet,
            posenet,
            device,
            args,
            archive_zip.stat().st_size,
            indices,
        )
        print("Final metrics:", json.dumps(final_metrics, indent=2), flush=True)

        required_quality = 0.300 - 25.0 * archive_zip.stat().st_size / ORIGINAL_BYTES
        record = {
            "label": label,
            "budget": args.budget,
            "arch_config": arch_config,
            "base_archive": str(args.base_archive),
            "base_archive_sha256": sha256_file(args.base_archive),
            "archive_zip": str(archive_zip),
            "archive_sha256": sha256_file(archive_zip),
            "archive_bytes": archive_zip.stat().st_size,
            "quality_required_for_score_lt_0_300": required_quality,
            "device": str(device),
            "indices": [int(x) for x in indices],
            "teacher": teacher_metrics,
            "after_teacher_stage": teacher_stage_metrics,
            "after_task_stage": task_stage_metrics,
            "final": final_metrics,
            "package": package_report,
            "teacher_init": init_report,
            "steps": {
                "teacher": args.teacher_steps,
                "task": args.task_steps,
                "qat": args.qat_steps,
            },
            "logs_tail": logs[-32:],
        }
        write_json(run_dir / "metrics.json", record)
        append_jsonl(args.out_dir / "student_frontier_results.jsonl", record)
        print(f"Wrote {run_dir / 'metrics.json'}", flush=True)
    finally:
        if loaded is not None:
            cleanup_loaded(loaded)


if __name__ == "__main__":
    main()
