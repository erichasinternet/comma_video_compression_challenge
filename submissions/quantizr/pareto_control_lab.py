#!/usr/bin/env python
"""Constrained Pareto audit for low-byte Quantizr controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE, ROOT / "submissions" / "tavs_video"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pareto_actions import build_action
from q55_common import DEFAULT_VIDEO_NAMES, MASK_PAYLOAD, ORIGINAL_BYTES, sha256_file, unzip_archive, write_json
from submissions.tavs_video.common import (
    build_distortion,
    collect_targets,
    ensure_q55_inflated,
    load_original_pairs_by_indices,
    load_raw_pairs_by_indices,
    quality,
    score,
    to_model_chw,
)
from submissions.commavq_task.common import FeatureTap, hard_margin_loss, round_ste, tv_loss


DEFAULT_BASE = ROOT / "submissions/q55_fp16_pose_int10/archive.zip"
DEFAULT_OUT = ROOT / "submissions/quantizr/experiments/pareto_control_lab"
MODEL_HW = (384, 512)


def load_quantizr_inflate_module():
    spec = importlib.util.spec_from_file_location("quantizr_modified_inflate", HERE / "inflate.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load Quantizr inflate module from {HERE / 'inflate.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q55_inflate = load_quantizr_inflate_module()


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_indices(text: str | None, *, subset: int, offset: int) -> list[int]:
    if text:
        return [int(x) for x in text.replace(" ", "").split(",") if x.strip()]
    return list(range(offset, offset + subset))


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_q55_masks(base_archive: Path, indices: list[int], device: torch.device) -> torch.Tensor:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        archive_dir = Path(td) / "archive"
        unzip_archive(base_archive, archive_dir)
        masks = q55_inflate.load_mask_payload(archive_dir, archive_dir / MASK_PAYLOAD)
    idx = torch.as_tensor(indices, dtype=torch.long)
    return masks.index_select(0, idx).to(device).long()


def load_context(args: argparse.Namespace, indices: list[int], device: torch.device) -> dict:
    inflated = ensure_q55_inflated(
        q55_submission_dir=ROOT / "submissions/q55_fp16_pose_int10",
        cache_dir=args.cache_dir,
        file_list=args.video_names,
    )
    base_raw = load_raw_pairs_by_indices(
        raw_dir=inflated,
        video_names_file=args.video_names,
        sample_indices=indices,
    )
    original = load_original_pairs_by_indices(
        data_dir=args.video_dir,
        video_names_file=args.video_names,
        sample_indices=indices,
        batch_size=args.load_batch_size,
    )
    base_frames = to_model_chw(base_raw).to(device)
    original_cpu = original.cpu()

    distortion = build_distortion(device)
    seg_tap = FeatureTap(distortion.segnet, parse_csv(args.seg_feature_taps))
    pose_tap = FeatureTap(distortion.posenet, parse_csv(args.pose_feature_taps))
    try:
        targets = collect_targets(
            distortion=distortion,
            original_cpu=original_cpu,
            device=device,
            batch_size=args.eval_batch_size,
            seg_tap=seg_tap,
            pose_tap=pose_tap,
        )
    finally:
        seg_tap.close()
        pose_tap.close()
    masks = load_q55_masks(args.base_archive, indices, device)
    return {
        "indices": indices,
        "base_frames": base_frames,
        "original_cpu": original_cpu,
        "targets": targets,
        "distortion": distortion,
        "masks": masks,
        "inflated_dir": inflated,
    }


@torch.inference_mode()
def evaluate_with_per_sample(
    *,
    frames: torch.Tensor,
    targets: dict,
    distortion,
    batch_size: int,
    sample_indices: list[int],
) -> dict:
    total_seg = 0.0
    total_pose = 0.0
    total = 0
    per_sample = []
    device = next(distortion.parameters()).device
    for start in range(0, frames.shape[0], batch_size):
        end = min(frames.shape[0], start + batch_size)
        rows = torch.arange(start, end, device=device)
        batch = round_ste(frames[start:end].to(device)).clamp(0, 255)
        seg_logits = distortion.segnet(distortion.segnet.preprocess_input(batch))
        pose = distortion.posenet(distortion.posenet.preprocess_input(batch))["pose"][..., :6]
        target_seg_logits = targets["seg_logits"].to(device).index_select(0, rows)
        target_pose = targets["pose"].to(device).index_select(0, rows)
        seg_dist = distortion.segnet.compute_distortion(target_seg_logits, seg_logits)
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_seg += float(seg_dist.sum().item())
        total_pose += float(pose_dist.sum().item())
        total += end - start
        for local, seg_i, pose_i in zip(range(start, end), seg_dist.detach().cpu(), pose_dist.detach().cpu(), strict=True):
            seg_f = float(seg_i.item())
            pose_f = float(pose_i.item())
            per_sample.append(
                {
                    "sample_id": int(sample_indices[local]),
                    "local_index": int(local),
                    "segnet_dist": seg_f,
                    "posenet_dist": pose_f,
                    "seg_term": 100.0 * seg_f,
                    "pose_term": math.sqrt(max(0.0, 10.0 * pose_f)),
                    "quality_i": 100.0 * seg_f + math.sqrt(max(0.0, 10.0 * pose_f)),
                }
            )
    seg = total_seg / max(1, total)
    pose = total_pose / max(1, total)
    return {
        "samples": int(total),
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
        "quality": quality(seg, pose),
        "per_sample": per_sample,
    }


def loss_terms(
    frames: torch.Tensor,
    targets: dict,
    distortion,
    target_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = frames.device
    b = frames.shape[0]
    rows = torch.arange(0, b, device=device) if target_rows is None else target_rows.to(device).long()
    x = round_ste(frames).clamp(0, 255)
    seg_logits = distortion.segnet(distortion.segnet.preprocess_input(x))
    pose = distortion.posenet(distortion.posenet.preprocess_input(x))["pose"][..., :6]
    target_logits = targets["seg_logits"].to(device).index_select(0, rows)
    target_argmax = targets["seg_argmax"].to(device).index_select(0, rows)
    target_pose = targets["pose"].to(device).index_select(0, rows)
    ce = F.cross_entropy(seg_logits, target_argmax)
    kl = F.kl_div(
        F.log_softmax(seg_logits, dim=1),
        F.softmax(target_logits, dim=1),
        reduction="batchmean",
    ) / (seg_logits.shape[-1] * seg_logits.shape[-2])
    margin = hard_margin_loss(seg_logits, target_argmax, margin=2.0)
    seg_loss = ce + kl + 0.5 * margin
    pose_loss = (pose - target_pose).pow(2).mean()
    reg = tv_loss(x[:, 1]) + 0.5 * tv_loss(x[:, 0])
    return seg_loss, pose_loss, reg


def flatten_grads(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grads, params, strict=True):
        if grad is None:
            chunks.append(torch.zeros_like(param).reshape(-1))
        else:
            chunks.append(grad.reshape(-1))
    if not chunks:
        return torch.zeros([])
    return torch.cat(chunks)


def grads_or_zeros(loss: torch.Tensor, params: list[torch.nn.Parameter], *, retain_graph: bool) -> torch.Tensor:
    if not params:
        return torch.zeros([], device=loss.device)
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(param).reshape(-1) for param in params])
    return flatten_grads(torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True), params)


def assign_flat_grad(params: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    cursor = 0
    for param in params:
        count = param.numel()
        param.grad = flat[cursor : cursor + count].reshape_as(param).clone()
        cursor += count


def pcgrad_step(action, base_frames, masks, targets, distortion, args, local_rows: torch.Tensor | None = None) -> dict:
    params = [p for p in action.parameters() if p.requires_grad]
    if local_rows is None:
        batch_frames = base_frames
        batch_masks = masks
        target_rows = None
    else:
        rows = local_rows.to(base_frames.device).long()
        batch_frames = base_frames.index_select(0, rows)
        batch_masks = None if masks is None else masks.index_select(0, rows)
        target_rows = rows
    frames = action.apply(batch_frames, batch_masks, local_rows)
    seg_loss, pose_loss, reg_loss = loss_terms(frames, targets, distortion, target_rows=target_rows)
    if not params:
        return {
            "seg_loss": float(seg_loss.item()),
            "pose_loss": float(pose_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "grad_cosine": None,
            "conflict": False,
        }

    g_seg = grads_or_zeros(seg_loss, params, retain_graph=True)
    g_pose = grads_or_zeros(pose_loss, params, retain_graph=True)
    g_reg = grads_or_zeros(reg_loss, params, retain_graph=False)
    dot = torch.dot(g_seg, g_pose)
    denom = g_seg.norm().clamp_min(1e-12) * g_pose.norm().clamp_min(1e-12)
    cosine = float((dot / denom).detach().cpu().item()) if g_seg.numel() else 0.0
    conflict = bool(dot.detach().item() < 0)
    if conflict:
        g_seg = g_seg - dot / g_pose.pow(2).sum().clamp_min(1e-12) * g_pose
    combined = g_seg + args.pose_loss_weight * g_pose + args.reg_weight * g_reg
    assign_flat_grad(params, combined)
    return {
        "seg_loss": float(seg_loss.item()),
        "pose_loss": float(pose_loss.item()),
        "reg_loss": float(reg_loss.item()),
        "grad_cosine": cosine,
        "conflict": conflict,
    }


def cmd_base_metrics(args: argparse.Namespace) -> None:
    if args.stream_chunk_size and args.subset > args.stream_chunk_size and not args.indices:
        cmd_base_metrics_stream(args)
        return
    device = torch.device(args.device)
    indices = parse_indices(args.indices, subset=args.subset, offset=args.offset)
    ctx = load_context(args, indices, device)
    metrics = evaluate_with_per_sample(
        frames=ctx["base_frames"],
        targets=ctx["targets"],
        distortion=ctx["distortion"],
        batch_size=args.eval_batch_size,
        sample_indices=indices,
    )
    archive_bytes = args.base_archive.stat().st_size
    metrics.update(
        {
            "archive_bytes": archive_bytes,
            "rate_term": 25.0 * archive_bytes / ORIGINAL_BYTES,
            "score": score(metrics["segnet_dist"], metrics["posenet_dist"], archive_bytes),
            "base_archive": str(args.base_archive),
            "base_archive_sha256": sha256_file(args.base_archive),
            "inflated_dir": str(ctx["inflated_dir"]),
        }
    )
    ranked = sorted(metrics["per_sample"], key=lambda r: r["quality_i"], reverse=True)
    out_dir = args.out_dir / "base_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_dir / "base_per_sample.jsonl", ranked)
    write_json(out_dir / "metrics.json", {k: v for k, v in metrics.items() if k != "per_sample"} | {"top_samples": ranked[:32]})
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}, indent=2, sort_keys=True))


def cmd_base_metrics_stream(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    indices = parse_indices(args.indices, subset=args.subset, offset=args.offset)
    inflated = ensure_q55_inflated(
        q55_submission_dir=ROOT / "submissions/q55_fp16_pose_int10",
        cache_dir=args.cache_dir,
        file_list=args.video_names,
    )
    distortion = build_distortion(device)
    archive_bytes = args.base_archive.stat().st_size
    out_dir = args.out_dir / "base_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "base_per_sample.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    total_seg = 0.0
    total_pose = 0.0
    total = 0
    all_rows = []
    for start in tqdm(range(0, len(indices), args.stream_chunk_size), desc="base metrics chunks"):
        chunk = indices[start : start + args.stream_chunk_size]
        base_raw = load_raw_pairs_by_indices(
            raw_dir=inflated,
            video_names_file=args.video_names,
            sample_indices=chunk,
        )
        original = load_original_pairs_by_indices(
            data_dir=args.video_dir,
            video_names_file=args.video_names,
            sample_indices=chunk,
            batch_size=args.load_batch_size,
        )
        seg_tap = FeatureTap(distortion.segnet, parse_csv(args.seg_feature_taps))
        pose_tap = FeatureTap(distortion.posenet, parse_csv(args.pose_feature_taps))
        try:
            targets = collect_targets(
                distortion=distortion,
                original_cpu=original.cpu(),
                device=device,
                batch_size=args.eval_batch_size,
                seg_tap=seg_tap,
                pose_tap=pose_tap,
            )
        finally:
            seg_tap.close()
            pose_tap.close()
        frames = to_model_chw(base_raw).to(device)
        metrics = evaluate_with_per_sample(
            frames=frames,
            targets=targets,
            distortion=distortion,
            batch_size=args.eval_batch_size,
            sample_indices=chunk,
        )
        total_seg += metrics["segnet_dist"] * metrics["samples"]
        total_pose += metrics["posenet_dist"] * metrics["samples"]
        total += metrics["samples"]
        all_rows.extend(metrics["per_sample"])

    seg = total_seg / max(1, total)
    pose = total_pose / max(1, total)
    ranked = sorted(all_rows, key=lambda r: r["quality_i"], reverse=True)
    save_jsonl(ledger_path, ranked)
    metrics = {
        "samples": int(total),
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
        "quality": quality(seg, pose),
        "archive_bytes": archive_bytes,
        "rate_term": 25.0 * archive_bytes / ORIGINAL_BYTES,
        "score": score(seg, pose, archive_bytes),
        "base_archive": str(args.base_archive),
        "base_archive_sha256": sha256_file(args.base_archive),
        "inflated_dir": str(inflated),
        "stream_chunk_size": args.stream_chunk_size,
        "top_samples": ranked[:32],
    }
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "top_samples"}, indent=2, sort_keys=True))


def cmd_channel_audit(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    indices = parse_indices(args.indices, subset=args.subset, offset=args.offset)
    ctx = load_context(args, indices, device)
    baseline = evaluate_with_per_sample(
        frames=ctx["base_frames"],
        targets=ctx["targets"],
        distortion=ctx["distortion"],
        batch_size=args.eval_batch_size,
        sample_indices=indices,
    )
    pose_cap = baseline["pose_term"] + args.pose_cap
    action_rows = []
    out_dir = args.out_dir / getattr(args, "audit_subdir", "channel_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    for control in parse_csv(args.controls):
        action = build_action(control, sample_count=len(indices), grid_hw=(args.grid_h, args.grid_w), max_delta=args.max_delta).to(device)
        opt = torch.optim.AdamW(action.parameters(), lr=args.lr) if list(action.parameters()) else None
        best = None
        accepted = 0
        rejected = 0
        conflict_count = 0
        cosine_values = []
        history = []
        for step in tqdm(range(1, args.steps + 1), desc=f"audit {control}"):
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                if args.train_batch_size and args.train_batch_size < len(indices):
                    local_rows = torch.randperm(len(indices), device=device)[: args.train_batch_size]
                else:
                    local_rows = None
                grad_info = pcgrad_step(
                    action,
                    ctx["base_frames"],
                    ctx["masks"],
                    ctx["targets"],
                    ctx["distortion"],
                    args,
                    local_rows=local_rows,
                )
                opt.step()
                if grad_info["grad_cosine"] is not None:
                    cosine_values.append(float(grad_info["grad_cosine"]))
                if grad_info["conflict"]:
                    conflict_count += 1
            else:
                grad_info = {}
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                candidate_frames = action.apply(ctx["base_frames"], ctx["masks"])
                metrics = evaluate_with_per_sample(
                    frames=candidate_frames,
                    targets=ctx["targets"],
                    distortion=ctx["distortion"],
                    batch_size=args.eval_batch_size,
                    sample_indices=indices,
                )
                added_bytes = action.estimate_bytes(len(indices))
                score_before = score(baseline["segnet_dist"], baseline["posenet_dist"], args.base_archive.stat().st_size)
                score_after = score(metrics["segnet_dist"], metrics["posenet_dist"], args.base_archive.stat().st_size + added_bytes)
                row = {
                    "control": control,
                    "step": step,
                    "added_bytes_est": added_bytes,
                    "baseline": {k: baseline[k] for k in ("segnet_dist", "posenet_dist", "seg_term", "pose_term", "quality")},
                    "metrics": {k: metrics[k] for k in ("segnet_dist", "posenet_dist", "seg_term", "pose_term", "quality")},
                    "quality_delta": metrics["quality"] - baseline["quality"],
                    "score_delta": score_after - score_before,
                    "score_before": score_before,
                    "score_after_est": score_after,
                    "pose_cap": pose_cap,
                    "pose_cap_ok": metrics["pose_term"] <= pose_cap,
                    **grad_info,
                }
                ok = row["score_delta"] < -args.min_score_improvement and row["pose_cap_ok"]
                if ok:
                    accepted += 1
                    if best is None or row["score_delta"] < best["score_delta"]:
                        best = row
                else:
                    rejected += 1
                history.append(row)
                append_jsonl(out_dir / f"{control}_history.jsonl", row)
        summary = {
            "control": control,
            "target": action.spec.target,
            "description": action.spec.description,
            "estimated_bytes": action.estimate_bytes(len(indices)),
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "gradient_conflict_rate": conflict_count / max(1, args.steps),
            "gradient_cosine_mean": float(np.mean(cosine_values)) if cosine_values else None,
            "gradient_cosine_p10": float(np.percentile(cosine_values, 10)) if cosine_values else None,
            "gradient_cosine_p90": float(np.percentile(cosine_values, 90)) if cosine_values else None,
            "best": best,
            "packer": action.pack_dict(),
        }
        action_rows.append(summary)
        write_json(out_dir / f"{control}_summary.json", summary)

    report = {
        "base_archive": str(args.base_archive),
        "base_archive_sha256": sha256_file(args.base_archive),
        "indices": indices,
        "baseline": baseline,
        "controls": action_rows,
        "continue_channels": [
            row["control"]
            for row in action_rows
            if row["best"] is not None and row["best"]["quality_delta"] <= -0.005
        ],
        "decision": "continue_if_any_channel_positive" if any(row["best"] for row in action_rows) else "stop_no_positive_channel",
    }
    write_json(out_dir / "metrics.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_bundle_audit(args: argparse.Namespace) -> None:
    if not args.indices:
        ledger = args.out_dir / "base_metrics/base_per_sample.jsonl"
        if ledger.exists():
            records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
            args.indices = ",".join(str(row["sample_id"]) for row in records[: args.top_k])
            args.subset = args.top_k
    args.audit_subdir = "bundle_audit"
    cmd_channel_audit(args)


def cmd_full_compose(args: argparse.Namespace) -> None:
    metrics_path = args.metrics
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing audit metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    positive = []
    for control in metrics.get("controls", []):
        best = control.get("best")
        if not best:
            continue
        if best["score_delta"] < 0.0 and best["pose_cap_ok"]:
            positive.append(
                {
                    "control": control["control"],
                    "score_delta": best["score_delta"],
                    "quality_delta": best["quality_delta"],
                    "added_bytes_est": best["added_bytes_est"],
                }
            )
    decision = "continue_full600_action_routing" if positive else "stop_no_validated_controls"
    report = {
        "source_metrics": str(metrics_path),
        "positive_controls": positive,
        "decision": decision,
        "note": "This composer only promotes rows that already passed actual-score and pose-cap gates.",
    }
    write_json(args.out_dir / "full_compose/metrics.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-archive", type=Path, default=DEFAULT_BASE)
        p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
        p.add_argument("--cache-dir", type=Path, default=ROOT / "submissions/quantizr/experiments/cache/q55_fp16_pose_int10")
        p.add_argument("--video-dir", type=Path, default=ROOT / "videos")
        p.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
        p.add_argument("--device", default="cpu")
        p.add_argument("--indices", default=None)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--subset", type=int, default=64)
        p.add_argument("--load-batch-size", type=int, default=16)
        p.add_argument("--eval-batch-size", type=int, default=8)
        p.add_argument("--seg-feature-taps", default="")
        p.add_argument("--pose-feature-taps", default="")
        p.add_argument("--stream-chunk-size", type=int, default=0)

    base = sub.add_parser("base_metrics", help="write per-sample q55 base ledger")
    add_common(base)
    base.set_defaults(func=cmd_base_metrics)

    audit = sub.add_parser("channel_audit", help="audit low-byte controls on a subset")
    add_common(audit)
    audit.add_argument("--controls", default="c1,c2,c3")
    audit.add_argument("--steps", type=int, default=250)
    audit.add_argument("--eval-every", type=int, default=50)
    audit.add_argument("--train-batch-size", type=int, default=8)
    audit.add_argument("--lr", type=float, default=0.03)
    audit.add_argument("--grid-h", type=int, default=12)
    audit.add_argument("--grid-w", type=int, default=16)
    audit.add_argument("--max-delta", type=float, default=18.0)
    audit.add_argument("--pose-cap", type=float, default=0.005)
    audit.add_argument("--pose-loss-weight", type=float, default=2.0)
    audit.add_argument("--reg-weight", type=float, default=0.01)
    audit.add_argument("--min-score-improvement", type=float, default=0.003)
    audit.set_defaults(func=cmd_channel_audit, audit_subdir="channel_audit")

    bundle = sub.add_parser("bundle_audit", help="audit C2/C3-style bundles on top hard samples")
    add_common(bundle)
    bundle.add_argument("--controls", default="c2,c3")
    bundle.add_argument("--top-k", type=int, default=32)
    bundle.add_argument("--steps", type=int, default=250)
    bundle.add_argument("--eval-every", type=int, default=50)
    bundle.add_argument("--train-batch-size", type=int, default=8)
    bundle.add_argument("--lr", type=float, default=0.03)
    bundle.add_argument("--grid-h", type=int, default=12)
    bundle.add_argument("--grid-w", type=int, default=16)
    bundle.add_argument("--max-delta", type=float, default=18.0)
    bundle.add_argument("--pose-cap", type=float, default=0.005)
    bundle.add_argument("--pose-loss-weight", type=float, default=2.0)
    bundle.add_argument("--reg-weight", type=float, default=0.01)
    bundle.add_argument("--min-score-improvement", type=float, default=0.003)
    bundle.set_defaults(func=cmd_bundle_audit, audit_subdir="bundle_audit")

    compose = sub.add_parser("full_compose", help="promote validated audit rows to a conservative compose decision")
    compose.add_argument("--metrics", type=Path, default=DEFAULT_OUT / "channel_audit/metrics.json")
    compose.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    compose.set_defaults(func=cmd_full_compose)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
