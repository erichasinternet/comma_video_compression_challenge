#!/usr/bin/env python3
"""Capacity-first task-aware HNeRV/NeRV training.

This file deliberately separates:

1. RGB basin pretraining, which tries to enter the camera-domain PoseNet basin.
2. Evaluator fine-tuning, which optimizes SegNet/PoseNet against frozen targets.

It is an oracle/training harness, not a compressed final submission by itself.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from submissions.tavs_video.common import (
    FeatureTap,
    build_distortion,
    collect_targets,
    evaluate_frames,
    feature_loss,
    hard_margin_loss,
    load_original_pairs_by_indices,
    metric_table,
    save_json,
    to_model_chw,
    tv_loss,
)
from submissions.task_nerv_vcm.model import HNeRVConfig, HNeRVRenderer, count_parameters


DEFAULT_HARD8 = [59, 60, 62, 56, 57, 58, 61, 63]


def parse_indices(text: str | None, *, preset: str, offset: int, subset: int) -> list[int]:
    if text:
        return [int(x) for x in text.replace(" ", "").split(",") if x]
    if preset == "hard8":
        return list(DEFAULT_HARD8)
    return list(range(offset, offset + subset))


def choose_device(text: str) -> torch.device:
    if text != "auto":
        return torch.device(text)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_unit = pred / 255.0
    target_unit = target / 255.0
    losses = [
        (pred_unit[..., 1:, :] - pred_unit[..., :-1, :] - (target_unit[..., 1:, :] - target_unit[..., :-1, :])).abs().mean(),
        (pred_unit[..., :, 1:] - pred_unit[..., :, :-1] - (target_unit[..., :, 1:] - target_unit[..., :, :-1])).abs().mean(),
    ]
    return torch.stack(losses).mean()


def low_frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_low = F.interpolate(pred.flatten(0, 1), size=(96, 128), mode="area")
    target_low = F.interpolate(target.flatten(0, 1), size=(96, 128), mode="area")
    return F.smooth_l1_loss(pred_low / 255.0, target_low / 255.0)


@torch.inference_mode()
def render_all(model: HNeRVRenderer, n_samples: int, *, batch_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
    frames = []
    for start in range(0, n_samples, batch_size):
        rows = torch.arange(start, min(n_samples, start + batch_size), device=device)
        frames.append(model.render_pairs(rows).detach().cpu())
    return torch.cat(frames, dim=0)


def eval_model(
    *,
    model: HNeRVRenderer,
    n_samples: int,
    targets: dict,
    distortion,
    batch_size: int,
) -> dict:
    frames = render_all(model, n_samples, batch_size=batch_size)
    return evaluate_frames(frames=frames, targets=targets, distortion=distortion, batch_size=batch_size)


def write_history(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def checkpoint(
    path: Path,
    *,
    model: HNeRVRenderer,
    cfg: HNeRVConfig,
    sample_ids: list[int],
    stage: str,
    step: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "task_nerv_vcm",
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "sample_ids": sample_ids,
            "stage": stage,
            "step": int(step),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=["hard8", "sequential"], default="hard8")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--subset", type=int, default=8)
    parser.add_argument("--rgb-steps", type=int, default=2000)
    parser.add_argument("--task-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--base-h", type=int, default=12)
    parser.add_argument("--base-w", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--lr-embed", type=float, default=1.0e-3)
    parser.add_argument("--rgb-weight", type=float, default=1.0)
    parser.add_argument("--lowfreq-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--seg-ce-weight", type=float, default=1.0)
    parser.add_argument("--seg-kl-weight", type=float, default=1.0)
    parser.add_argument("--seg-margin-weight", type=float, default=2.0)
    parser.add_argument("--pose-weight", type=float, default=3.0)
    parser.add_argument("--pose-feature-weight", type=float, default=1.0)
    parser.add_argument("--seg-feature-weight", type=float, default=0.0)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--pose-feature-names", default="summarizer")
    parser.add_argument("--seg-feature-names", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    device = choose_device(args.device)
    sample_ids = parse_indices(args.indices, preset=args.preset, offset=args.offset, subset=args.subset)
    original_cpu = load_original_pairs_by_indices(
        data_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
        batch_size=max(args.batch_size, 8),
    )
    original_model = to_model_chw(original_cpu).to(device)

    distortion = build_distortion(device)
    seg_feature_names = [name for name in args.seg_feature_names.split(",") if name and args.seg_feature_weight]
    pose_feature_names = [name for name in args.pose_feature_names.split(",") if name and args.pose_feature_weight]
    seg_tap = FeatureTap(distortion.segnet, seg_feature_names)
    pose_tap = FeatureTap(distortion.posenet, pose_feature_names)
    targets = collect_targets(
        distortion=distortion,
        original_cpu=original_cpu,
        device=device,
        batch_size=max(1, min(args.batch_size, 4)),
        seg_tap=seg_tap,
        pose_tap=pose_tap,
    )

    cfg = HNeRVConfig(
        n_frames=len(sample_ids) * 2,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        base_h=args.base_h,
        base_w=args.base_w,
        num_blocks=args.num_blocks,
        mlp_hidden=args.mlp_hidden,
    )
    model = HNeRVRenderer(cfg).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.embedding.weight], "lr": args.lr_embed},
            {"params": [p for n, p in model.named_parameters() if n != "embedding.weight"], "lr": args.lr},
        ],
        weight_decay=0.0,
    )
    n = len(sample_ids)
    best = {"quality": float("inf")}

    def eval_and_maybe_save(stage: str, step: int) -> dict:
        nonlocal best
        metrics = eval_model(model=model, n_samples=n, targets=targets, distortion=distortion, batch_size=args.batch_size)
        row = {"stage": stage, "step": int(step), **metrics}
        write_history(history_path, row)
        if metrics["quality"] < best["quality"]:
            best = dict(row)
            checkpoint(
                args.out_dir / "best_nerv.pt",
                model=model,
                cfg=cfg,
                sample_ids=sample_ids,
                stage=stage,
                step=step,
                metrics=metrics,
            )
        save_json(
            args.out_dir / "metrics.json",
            {
                "kind": "task_nerv_vcm_capacity",
                "sample_ids": sample_ids,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "config": asdict(cfg),
                "parameter_count": count_parameters(model),
                "best": best,
                "score_targets": {
                    "N240_required_quality": metric_table(0.0, 0.0, 240_000)["required_quality_for_0.300"],
                    "N220_required_quality": metric_table(0.0, 0.0, 220_000)["required_quality_for_0.300"],
                    "N200_required_quality": metric_table(0.0, 0.0, 200_000)["required_quality_for_0.300"],
                },
            },
        )
        return row

    eval_and_maybe_save("init", 0)

    if args.rgb_steps > 0:
        pbar = tqdm(range(1, args.rgb_steps + 1), desc="task-nerv rgb")
        for step in pbar:
            rows = torch.randint(0, n, (min(args.batch_size, n),), device=device)
            pred = model.render_pairs(rows)
            target = original_model.index_select(0, rows)
            loss_rgb = F.smooth_l1_loss(pred / 255.0, target / 255.0)
            loss_low = low_frequency_loss(pred, target)
            loss_edge = edge_loss(pred, target)
            loss = args.rgb_weight * loss_rgb + args.lowfreq_weight * loss_low + args.edge_weight * loss_edge
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 10 == 0:
                pbar.set_postfix(loss=f"{float(loss.item()):.4f}", rgb=f"{float(loss_rgb.item()):.4f}")
            if step % args.eval_every == 0 or step == args.rgb_steps:
                row = eval_and_maybe_save("rgb", step)
                row.update({"rgb_loss": float(loss_rgb.detach().item()), "lowfreq_loss": float(loss_low.detach().item())})
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                checkpoint(args.out_dir / f"checkpoint_rgb_{step:06d}.pt", model=model, cfg=cfg, sample_ids=sample_ids, stage="rgb", step=step, metrics=best)

    if args.task_steps > 0:
        pbar = tqdm(range(1, args.task_steps + 1), desc="task-nerv task")
        for step in pbar:
            rows = torch.randint(0, n, (min(args.batch_size, n),), device=device)
            pred = model.render_pairs(rows).clamp(0, 255)
            target_arg = targets["seg_argmax"].to(device).index_select(0, rows)
            target_prob = targets["seg_prob"].to(device).index_select(0, rows)
            target_pose = targets["pose"].to(device).index_select(0, rows)
            seg_tap.clear()
            pose_tap.clear()
            seg_logits = distortion.segnet(distortion.segnet.preprocess_input(pred))
            pose = distortion.posenet(distortion.posenet.preprocess_input(pred))["pose"][..., :6]
            ce = F.cross_entropy(seg_logits, target_arg)
            kl = F.kl_div(F.log_softmax(seg_logits, dim=1), target_prob, reduction="none").sum(dim=1).mean()
            margin = hard_margin_loss(seg_logits, target_arg)
            pose_mse = (pose - target_pose).pow(2).mean()
            pose_feat = feature_loss(pose_tap.features, targets["pose_features"], rows)
            seg_feat = feature_loss(seg_tap.features, targets["seg_features"], rows)
            tv = tv_loss(pred)
            loss = (
                args.seg_ce_weight * ce
                + args.seg_kl_weight * kl
                + args.seg_margin_weight * margin
                + args.pose_weight * pose_mse
                + args.pose_feature_weight * pose_feat
                + args.seg_feature_weight * seg_feat
                + args.tv_weight * tv
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 10 == 0:
                pbar.set_postfix(loss=f"{float(loss.item()):.4f}", ce=f"{float(ce.item()):.3f}", pose=f"{float(pose_mse.item()):.4f}")
            if step % args.eval_every == 0 or step == args.task_steps:
                row = eval_and_maybe_save("task", step)
                row.update(
                    {
                        "loss": float(loss.detach().item()),
                        "ce": float(ce.detach().item()),
                        "kl": float(kl.detach().item()),
                        "margin": float(margin.detach().item()),
                        "pose_mse_loss": float(pose_mse.detach().item()),
                    }
                )
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                checkpoint(args.out_dir / f"checkpoint_task_{step:06d}.pt", model=model, cfg=cfg, sample_ids=sample_ids, stage="task", step=step, metrics=best)

    checkpoint(args.out_dir / "final_nerv.pt", model=model, cfg=cfg, sample_ids=sample_ids, stage="final", step=args.task_steps, metrics=best)
    seg_tap.close()
    pose_tap.close()


if __name__ == "__main__":
    main()

