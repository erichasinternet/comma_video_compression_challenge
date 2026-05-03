#!/usr/bin/env python3
"""Optimize a codec-friendly source video directly against SegNet/PoseNet."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from submissions.tavs_video.codec_roundtrip import check_ffmpeg_codec, roundtrip_frames
from submissions.tavs_video.common import (
    FeatureTap,
    MODEL_H,
    MODEL_W,
    ROOT,
    build_distortion,
    collect_targets,
    ensure_q55_inflated,
    evaluate_frames,
    feature_loss,
    hard_margin_loss,
    load_original_pairs_by_indices,
    load_raw_pairs_by_indices,
    metric_table,
    parse_indices,
    save_json,
    to_model_chw,
    tv_loss,
)
from submissions.commavq_task.common import round_ste


def rgb_to_yuv(rgb255: torch.Tensor) -> torch.Tensor:
    rgb = (rgb255 / 255.0).clamp(0.0, 1.0)
    r, g, b = rgb.unbind(dim=2)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) / 1.772
    v = (r - y) / 1.402
    return torch.stack([y, u, v], dim=2)


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    y, u, v = yuv.unbind(dim=2)
    r = y + 1.402 * v
    b = y + 1.772 * u
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.stack([r, g, b], dim=2).clamp(0.0, 1.0) * 255.0


def yuv420_proxy(frames: torch.Tensor) -> torch.Tensor:
    yuv = rgb_to_yuv(frames)
    y = yuv[:, :, 0:1]
    uv = yuv[:, :, 1:3].flatten(0, 1)
    uv_low = F.avg_pool2d(uv, kernel_size=2, stride=2)
    uv_up = F.interpolate(uv_low, size=(frames.shape[-2], frames.shape[-1]), mode="bilinear", align_corners=False)
    yuv420 = torch.cat([y.flatten(0, 1), uv_up], dim=1).reshape_as(yuv)
    rgb = yuv_to_rgb(yuv420)
    return round_ste(rgb)


def codec_aug(frames: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if not enabled:
        return round_ste(frames)
    out = yuv420_proxy(frames)
    if random.random() < 0.5:
        flat = out.flatten(0, 1)
        low_h = random.choice([288, 320, 352, 384])
        low_w = int(low_h * MODEL_W / MODEL_H)
        flat = F.interpolate(flat, size=(low_h, low_w), mode="bilinear", align_corners=False)
        flat = F.interpolate(flat, size=(MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
        out = flat.reshape_as(out)
    if random.random() < 0.35:
        flat = out.flatten(0, 1)
        flat = F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        out = flat.reshape_as(out)
    noise = torch.randn_like(out) * random.uniform(0.0, 1.5)
    return (out + noise).clamp(0, 255)


def highfreq_penalty(frames: torch.Tensor) -> torch.Tensor:
    flat = frames.flatten(0, 1)
    low = F.interpolate(flat, size=(96, 128), mode="area")
    low = F.interpolate(low, size=(MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
    return ((flat - low) / 255.0).pow(2).mean()


@dataclass
class SourceConfig:
    grid_h: int = 72
    grid_w: int = 96
    residual_scale: float = 0.25
    color_scale: float = 0.10


class LowFreqYUVSource(nn.Module):
    def __init__(self, base_frames: torch.Tensor, cfg: SourceConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("base_yuv", rgb_to_yuv(base_frames))
        n = base_frames.shape[0]
        self.residual = nn.Parameter(torch.zeros(n, 2, 3, cfg.grid_h, cfg.grid_w))
        self.gain = nn.Parameter(torch.zeros(n, 2, 3))
        self.bias = nn.Parameter(torch.zeros(n, 2, 3))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        base = self.base_yuv.index_select(0, rows)
        residual = torch.tanh(self.residual.index_select(0, rows)) * self.cfg.residual_scale
        residual = F.interpolate(
            residual.flatten(0, 1),
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape_as(base)
        gain = 1.0 + self.cfg.color_scale * torch.tanh(self.gain.index_select(0, rows))[:, :, :, None, None]
        bias = self.cfg.color_scale * torch.tanh(self.bias.index_select(0, rows))[:, :, :, None, None]
        yuv = base * gain + bias + residual
        return yuv_to_rgb(yuv)

    def residual_smoothness(self, rows: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.residual.index_select(0, rows))
        temporal = (residual[:, 1] - residual[:, 0]).abs().mean()
        spatial = (residual[..., 1:, :] - residual[..., :-1, :]).abs().mean()
        spatial = spatial + (residual[..., :, 1:] - residual[..., :, :-1]).abs().mean()
        return temporal + spatial


def load_base_frames(args: argparse.Namespace, sample_ids: list[int], original: torch.Tensor) -> torch.Tensor:
    original_model = to_model_chw(original)
    if args.init == "original":
        return original_model
    q55_inflated = ensure_q55_inflated(
        q55_submission_dir=args.q55_submission_dir,
        cache_dir=args.cache_dir / "q55_fp16_pose_int10",
        file_list=args.video_names_file,
    )
    q55_raw = load_raw_pairs_by_indices(
        raw_dir=q55_inflated,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
    )
    q55_model = to_model_chw(q55_raw)
    if args.init == "q55":
        return q55_model
    if args.init == "blend":
        low = F.interpolate(original_model.flatten(0, 1), size=(96, 128), mode="area")
        low = F.interpolate(low, size=(MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
        low = low.reshape_as(original_model)
        return (0.7 * q55_model + 0.3 * low).clamp(0, 255)
    raise ValueError(f"unknown init: {args.init}")


@torch.inference_mode()
def render_all(source: LowFreqYUVSource, *, batch_size: int) -> torch.Tensor:
    device = source.base_yuv.device
    frames = []
    for start in range(0, source.base_yuv.shape[0], batch_size):
        rows = torch.arange(start, min(source.base_yuv.shape[0], start + batch_size), device=device)
        frames.append(source(rows).detach().cpu())
    return torch.cat(frames, dim=0)


def evaluate_codec_candidates(
    *,
    frames: torch.Tensor,
    targets: dict,
    distortion,
    batch_size: int,
    out_dir: Path,
    step: int,
    sample_count: int,
    codec: str,
    crfs: list[int],
) -> list[dict]:
    rows = []
    if not check_ffmpeg_codec(codec):
        return [{"step": step, "codec": codec, "skipped": True, "reason": "ffmpeg encoder unavailable"}]
    for crf in crfs:
        result = roundtrip_frames(
            frames=frames,
            codec=codec,
            crf=crf,
            work_dir=out_dir / "codec_roundtrips",
            label=f"step{step:06d}",
        )
        metrics = evaluate_frames(
            frames=result["decoded"],
            targets=targets,
            distortion=distortion,
            batch_size=batch_size,
        )
        projected = int(round(result["encoded_bytes"] * 600 / sample_count))
        row = {
            "step": int(step),
            "codec": codec,
            "crf": int(crf),
            "encoded_64_bytes": int(result["encoded_bytes"]),
            "projected_full_video_bytes": projected,
            "video_path": result["video_path"],
            **metrics,
            **metric_table(metrics["segnet_dist"], metrics["posenet_dist"], projected),
        }
        rows.append(row)
    return rows


def choose_device(text: str) -> torch.device:
    if text != "auto":
        return torch.device(text)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def jsonable_config(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--init", choices=["q55", "original", "blend"], default="q55")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--subset", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--codec-eval-every", type=int, default=250)
    parser.add_argument("--codec", choices=["svtav1", "libaom-av1", "vp9", "x265"], default="svtav1")
    parser.add_argument("--codec-crfs", default="49,53,57,61")
    parser.add_argument("--grid-h", type=int, default=72)
    parser.add_argument("--grid-w", type=int, default=96)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seg-ce-weight", type=float, default=1.0)
    parser.add_argument("--seg-kl-weight", type=float, default=1.0)
    parser.add_argument("--seg-margin-weight", type=float, default=2.0)
    parser.add_argument("--pose-weight", type=float, default=3.0)
    parser.add_argument("--pose-feature-weight", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--highfreq-weight", type=float, default=0.05)
    parser.add_argument("--codec-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pose-feature-names", default="summarizer")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    parser.add_argument("--q55-submission-dir", type=Path, default=ROOT / "submissions/q55_fp16_pose_int10")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "experiments/cache")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"
    codec_history_path = args.out_dir / "codec_history.jsonl"
    history_path.unlink(missing_ok=True)
    codec_history_path.unlink(missing_ok=True)

    device = choose_device(args.device)
    sample_ids = parse_indices(args.indices, offset=args.offset, subset=args.subset)
    crfs = [int(x) for x in args.codec_crfs.replace(" ", "").split(",") if x]

    original = load_original_pairs_by_indices(
        data_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
        batch_size=max(args.batch_size, 8),
    )
    base_frames = load_base_frames(args, sample_ids, original).to(device)

    distortion = build_distortion(device)
    pose_tap = FeatureTap(distortion.posenet, [x for x in args.pose_feature_names.split(",") if x and args.pose_feature_weight])
    seg_tap = FeatureTap(distortion.segnet, [])
    targets = collect_targets(
        distortion=distortion,
        original_cpu=original,
        device=device,
        batch_size=max(1, min(args.batch_size, 4)),
        seg_tap=seg_tap,
        pose_tap=pose_tap,
    )

    source = LowFreqYUVSource(
        base_frames,
        SourceConfig(grid_h=args.grid_h, grid_w=args.grid_w, residual_scale=args.residual_scale),
    ).to(device)
    opt = torch.optim.AdamW(source.parameters(), lr=args.lr, weight_decay=0.0)

    baseline = evaluate_frames(frames=base_frames.detach().cpu(), targets=targets, distortion=distortion, batch_size=args.batch_size)
    best_proxy = {"quality": float("inf")}
    best_codec: dict | None = None
    best_frames = base_frames.detach().cpu()

    def evaluate_and_save(step: int) -> None:
        nonlocal best_proxy, best_codec, best_frames
        frames = render_all(source, batch_size=args.batch_size)
        metrics = evaluate_frames(frames=frames, targets=targets, distortion=distortion, batch_size=args.batch_size)
        row = {"step": int(step), **metrics}
        with history_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if metrics["quality"] < best_proxy["quality"]:
            best_proxy = row
            best_frames = frames
            torch.save(
                {
                    "best_frames": best_frames,
                    "sample_ids": sample_ids,
                    "step": int(step),
                    "metrics": metrics,
                    "config": jsonable_config(args),
                },
                args.out_dir / "best_frames.pt",
            )
        if args.codec_eval_every > 0 and (step == 0 or step % args.codec_eval_every == 0):
            for codec_row in evaluate_codec_candidates(
                frames=frames,
                targets=targets,
                distortion=distortion,
                batch_size=args.batch_size,
                out_dir=args.out_dir,
                step=step,
                sample_count=len(sample_ids),
                codec=args.codec,
                crfs=crfs,
            ):
                with codec_history_path.open("a") as f:
                    f.write(json.dumps({k: v for k, v in codec_row.items() if k != "decoded"}, sort_keys=True) + "\n")
                if not codec_row.get("skipped") and (
                    best_codec is None or codec_row["score"] < best_codec["score"]
                ):
                    best_codec = codec_row
        metrics_out = {
            "sample_ids": sample_ids,
            "config": jsonable_config(args),
            "source_config": asdict(source.cfg),
            "baseline": baseline,
            "best_proxy": best_proxy,
            "best_codec": best_codec,
        }
        save_json(args.out_dir / "metrics.json", metrics_out)

    evaluate_and_save(0)
    n = len(sample_ids)
    pbar = tqdm(range(1, args.steps + 1), desc="TAVS optimize")
    for step in pbar:
        rows = torch.randint(0, n, (min(args.batch_size, n),), device=device)
        frames = source(rows)
        frames_loss = codec_aug(frames, enabled=args.codec_aug)

        target_arg = targets["seg_argmax"].to(device).index_select(0, rows)
        target_prob = targets["seg_prob"].to(device).index_select(0, rows)
        target_pose = targets["pose"].to(device).index_select(0, rows)

        pose_tap.clear()
        seg_logits = distortion.segnet(distortion.segnet.preprocess_input(frames_loss))
        pose = distortion.posenet(distortion.posenet.preprocess_input(frames_loss))["pose"][..., :6]

        ce = F.cross_entropy(seg_logits, target_arg)
        kl = F.kl_div(F.log_softmax(seg_logits, dim=1), target_prob, reduction="none").sum(dim=1).mean()
        margin = hard_margin_loss(seg_logits, target_arg)
        pose_mse = (pose - target_pose).pow(2).mean()
        pfeat = feature_loss(pose_tap.features, targets["pose_features"], rows)
        tv = tv_loss(frames_loss)
        smooth = source.residual_smoothness(rows)
        high = highfreq_penalty(frames_loss)
        loss = (
            args.seg_ce_weight * ce
            + args.seg_kl_weight * kl
            + args.seg_margin_weight * margin
            + args.pose_weight * pose_mse
            + args.pose_feature_weight * pfeat
            + args.tv_weight * tv
            + args.smoothness_weight * smooth
            + args.highfreq_weight * high
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(source.parameters(), 5.0)
        opt.step()

        if step % 10 == 0:
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", ce=f"{float(ce.item()):.3f}", pose=f"{float(pose_mse.item()):.4f}")
        if step % args.eval_every == 0 or step == args.steps:
            evaluate_and_save(step)

    evaluate_and_save(args.steps)
    torch.save({"best_frames": best_frames, "sample_ids": sample_ids, "metrics": best_proxy}, args.out_dir / "best_frames.pt")


if __name__ == "__main__":
    main()
