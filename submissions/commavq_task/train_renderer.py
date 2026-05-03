#!/usr/bin/env python3
"""Train a small evaluator-facing renderer from pretrained commaVQ tokens."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from submissions.commavq_task.common import (
    MODEL_HW,
    FeatureTap,
    build_distortion,
    collect_targets,
    evaluate_frames,
    feature_loss,
    hard_margin_loss,
    load_original_pairs_by_indices,
    round_ste,
    tv_loss,
)
from frame_utils import camera_size


COMMAVQ_CROP_SIZE = (512, 256)
COMMAVQ_SCALE = 567 / 455
COMMAVQ_CY = 47.6


def make_coords(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1).expand(batch, 1, h, w)
    x = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w).expand(batch, 1, h, w)
    return torch.cat([x, y], dim=1)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8 if channels >= 8 else 1, channels)

    def forward(self, x: torch.Tensor, film: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.pw(F.silu(x))
        x = self.norm(x)
        if film is not None:
            gamma, beta = film
            x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.silu(x + residual)


@dataclass
class RendererConfig:
    vocab_size: int = 1024
    emb_dim: int = 24
    hidden: int = 64
    num_blocks: int = 5
    token_h: int = 8
    token_w: int = 16
    separate_heads: bool = False


class CommaVQTaskRenderer(nn.Module):
    def __init__(self, cfg: RendererConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.in_proj = nn.Conv2d(cfg.emb_dim + 3, cfg.hidden, 1)
        self.blocks = nn.ModuleList([DepthwiseSeparableBlock(cfg.hidden) for _ in range(cfg.num_blocks)])
        self.global_mlp = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.num_blocks * cfg.hidden * 2),
        )
        if cfg.separate_heads:
            self.frame1_head = nn.Sequential(
                DepthwiseSeparableBlock(cfg.hidden),
                nn.Conv2d(cfg.hidden, 3, 1),
            )
            self.frame2_head = nn.Sequential(
                DepthwiseSeparableBlock(cfg.hidden),
                nn.Conv2d(cfg.hidden, 3, 1),
            )
        else:
            self.head = nn.Sequential(
                DepthwiseSeparableBlock(cfg.hidden),
                nn.Conv2d(cfg.hidden, 3, 1),
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, 2, 128]
        b = tokens.shape[0]
        emb = self.embedding(tokens.long()).view(b, 2, self.cfg.token_h, self.cfg.token_w, self.cfg.emb_dim)
        emb = emb.permute(0, 1, 4, 2, 3).contiguous()
        global_in = emb.mean(dim=(-1, -2)).flatten(1)
        film = self.global_mlp(global_in).view(b, self.cfg.num_blocks, 2, self.cfg.hidden)

        x = emb.flatten(0, 1)
        coords = make_coords(b * 2, self.cfg.token_h, self.cfg.token_w, x.device)
        frame_id = torch.tensor([-1.0, 1.0], device=x.device).view(1, 2, 1, 1, 1).expand(b, 2, 1, self.cfg.token_h, self.cfg.token_w)
        frame_id = frame_id.flatten(0, 1)
        x = self.in_proj(torch.cat([x, coords, frame_id], dim=1))

        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            gamma = 0.1 * torch.tanh(film[:, i, 0]).repeat_interleave(2, dim=0)
            beta = 0.1 * torch.tanh(film[:, i, 1]).repeat_interleave(2, dim=0)
            x = block(x, (gamma, beta))
        if x.shape[-2:] != MODEL_HW:
            x = F.interpolate(x, size=MODEL_HW, mode="bilinear", align_corners=False)
        if self.cfg.separate_heads:
            x = x.reshape(b, 2, self.cfg.hidden, MODEL_HW[0], MODEL_HW[1])
            frame1 = torch.sigmoid(self.frame1_head(x[:, 0])) * 255.0
            frame2 = torch.sigmoid(self.frame2_head(x[:, 1])) * 255.0
            return torch.stack([frame1, frame2], dim=1)
        frames = torch.sigmoid(self.head(x)) * 255.0
        return frames.reshape(b, 2, 3, MODEL_HW[0], MODEL_HW[1])


@torch.inference_mode()
def evaluate_renderer(
    *,
    renderer: CommaVQTaskRenderer,
    tokens: torch.Tensor,
    targets: dict,
    distortion,
    batch_size: int,
) -> dict:
    frames = []
    for start in range(0, tokens.shape[0], batch_size):
        frames.append(renderer(tokens[start : start + batch_size]).detach().cpu())
    return evaluate_frames(frames=torch.cat(frames, dim=0), targets=targets, distortion=distortion, batch_size=batch_size)


def build_rgb_anchor(original_cpu: torch.Tensor, *, mode: str, fill: float, device: torch.device) -> torch.Tensor:
    original = original_cpu.to(device).permute(0, 1, 4, 2, 3).float()
    flat = original.flatten(0, 1)
    if mode == "original":
        out = F.interpolate(flat, size=MODEL_HW, mode="bicubic", align_corners=False).clamp(0, 255)
        return out.reshape(original.shape[0], 2, 3, MODEL_HW[0], MODEL_HW[1])
    if mode != "inverse_crop":
        raise ValueError(f"unknown rgb anchor mode: {mode}")
    target_h, target_w = camera_size[1], camera_size[0]
    crop_w = int(COMMAVQ_CROP_SIZE[0] * COMMAVQ_SCALE)
    crop_h = int(COMMAVQ_CROP_SIZE[1] * COMMAVQ_SCALE)
    x0 = target_h // 2 - crop_h // 2 - int(COMMAVQ_CY * COMMAVQ_SCALE) // 2
    y0 = target_w // 2 - crop_w // 2
    crop = flat[:, :, x0 : x0 + crop_h, y0 : y0 + crop_w]
    canvas = torch.full_like(flat, fill)
    canvas[:, :, x0 : x0 + crop_h, y0 : y0 + crop_w] = crop
    out = F.interpolate(canvas, size=MODEL_HW, mode="bicubic", align_corners=False).clamp(0, 255)
    return out.reshape(original.shape[0], 2, 3, MODEL_HW[0], MODEL_HW[1])


def gradient_loss(frames: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    frames = frames / 255.0
    target = target / 255.0
    fx = frames[..., :, 1:] - frames[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    fy = frames[..., 1:, :] - frames[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(fx, tx) + F.l1_loss(fy, ty)


def lowfreq_loss(frames: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    frames_low = F.interpolate(frames.flatten(0, 1), size=(96, 128), mode="area")
    target_low = F.interpolate(target.flatten(0, 1), size=(96, 128), mode="area")
    return F.smooth_l1_loss(frames_low / 255.0, target_low / 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rgb-anchor-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--emb-dim", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--separate-heads", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rgb-anchor-mode", choices=["original", "inverse_crop"], default="original")
    parser.add_argument("--rgb-anchor-fill", type=float, default=0.0)
    parser.add_argument("--rgb-huber-weight", type=float, default=1.0)
    parser.add_argument("--rgb-lowfreq-weight", type=float, default=0.5)
    parser.add_argument("--rgb-edge-weight", type=float, default=0.25)
    parser.add_argument("--seg-ce-weight", type=float, default=1.0)
    parser.add_argument("--seg-kl-weight", type=float, default=1.0)
    parser.add_argument("--seg-margin-weight", type=float, default=2.0)
    parser.add_argument("--seg-feature-weight", type=float, default=0.0)
    parser.add_argument("--pose-weight", type=float, default=3.0)
    parser.add_argument("--pose-feature-weight", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--seg-feature-names", default="encoder.model.blocks.4")
    parser.add_argument("--pose-feature-names", default="summarizer")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    tokens_np = np.load(args.tokens).astype(np.int64)
    if tokens_np.ndim != 3 or tokens_np.shape[1:] != (2, 128):
        raise ValueError(f"expected tokens shape [N,2,128], got {tokens_np.shape}")
    sample_ids = json.loads(args.sample_ids.read_text()) if args.sample_ids else list(range(tokens_np.shape[0]))
    tokens = torch.from_numpy(tokens_np).to(device)

    original = load_original_pairs_by_indices(
        data_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
        batch_size=max(args.batch_size, 8),
    )
    distortion = build_distortion(device)
    seg_names = [name for name in args.seg_feature_names.split(",") if name] if args.seg_feature_weight else []
    pose_names = [name for name in args.pose_feature_names.split(",") if name] if args.pose_feature_weight else []
    seg_tap = FeatureTap(distortion.segnet, seg_names)
    pose_tap = FeatureTap(distortion.posenet, pose_names)
    targets = collect_targets(
        distortion=distortion,
        original_cpu=original,
        device=device,
        batch_size=max(1, min(args.batch_size, 4)),
        seg_tap=seg_tap,
        pose_tap=pose_tap,
    )

    rgb_anchor = build_rgb_anchor(
        original,
        mode=args.rgb_anchor_mode,
        fill=args.rgb_anchor_fill,
        device=device,
    )

    cfg = RendererConfig(
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        num_blocks=args.num_blocks,
        separate_heads=args.separate_heads,
    )
    renderer = CommaVQTaskRenderer(cfg).to(device)
    optimizer = torch.optim.AdamW(renderer.parameters(), lr=args.lr, weight_decay=0.0)
    n = tokens.shape[0]

    best = evaluate_renderer(renderer=renderer, tokens=tokens, targets=targets, distortion=distortion, batch_size=args.batch_size)
    best.update({"step": 0, "stage": "init", "stage_step": 0})
    with history_path.open("a") as f:
        f.write(json.dumps(best, sort_keys=True) + "\n")
    print(json.dumps(best, sort_keys=True), flush=True)

    def save_best(current: dict) -> None:
        nonlocal best
        if current["quality"] < best["quality"]:
            best = dict(current)
            torch.save(
                {
                    "renderer_config": asdict(cfg),
                    "renderer_state_dict": renderer.state_dict(),
                    "sample_ids": sample_ids,
                    "best": best,
                },
                args.out_dir / "best_renderer.pt",
            )

    def eval_and_record(stage: str, stage_step: int, global_step: int, extra: dict | None = None) -> dict:
        current = evaluate_renderer(
            renderer=renderer,
            tokens=tokens,
            targets=targets,
            distortion=distortion,
            batch_size=args.batch_size,
        )
        current.update({"step": global_step, "stage": stage, "stage_step": stage_step})
        if extra:
            current.update(extra)
        with history_path.open("a") as f:
            f.write(json.dumps(current, sort_keys=True) + "\n")
        print(json.dumps(current, sort_keys=True), flush=True)
        save_best(current)
        return current

    global_step = 0

    for stage_step in tqdm(range(1, args.rgb_anchor_steps + 1), desc="commaVQ RGB anchor"):
        global_step += 1
        rows = torch.randint(0, n, (min(args.batch_size, n),), device=device)
        frames = renderer(tokens.index_select(0, rows))
        target_rgb = rgb_anchor.index_select(0, rows)
        huber = F.smooth_l1_loss(frames / 255.0, target_rgb / 255.0)
        low = lowfreq_loss(frames, target_rgb)
        edge = gradient_loss(frames, target_rgb)
        loss = (
            args.rgb_huber_weight * huber
            + args.rgb_lowfreq_weight * low
            + args.rgb_edge_weight * edge
            + args.tv_weight * tv_loss(frames)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(renderer.parameters(), 1.0)
        optimizer.step()
        if stage_step % args.eval_every == 0 or stage_step == args.rgb_anchor_steps:
            eval_and_record(
                "rgb_anchor",
                stage_step,
                global_step,
                {
                    "loss": float(loss.detach().item()),
                    "rgb_huber": float(huber.detach().item()),
                    "rgb_lowfreq": float(low.detach().item()),
                    "rgb_edge": float(edge.detach().item()),
                },
            )
        if global_step % args.checkpoint_every == 0:
            torch.save(
                {
                    "renderer_config": asdict(cfg),
                    "renderer_state_dict": renderer.state_dict(),
                    "sample_ids": sample_ids,
                    "step": global_step,
                    "stage": "rgb_anchor",
                },
                args.out_dir / f"checkpoint_step{global_step:06d}.pt",
            )

    for stage_step in tqdm(range(1, args.steps + 1), desc="commaVQ task renderer"):
        global_step += 1
        rows = torch.randint(0, n, (min(args.batch_size, n),), device=device)
        frames = renderer(tokens.index_select(0, rows))
        eval_frames = round_ste(frames).clamp(0, 255)
        seg_tap.clear()
        pose_tap.clear()
        seg_logits = distortion.segnet(distortion.segnet.preprocess_input(eval_frames))
        pose_out = distortion.posenet(distortion.posenet.preprocess_input(eval_frames))["pose"][..., :6]

        target_seg_argmax = targets["seg_argmax"].to(device).index_select(0, rows)
        target_seg_prob = targets["seg_prob"].to(device).index_select(0, rows)
        target_pose = targets["pose"].to(device).index_select(0, rows)
        ce = F.cross_entropy(seg_logits, target_seg_argmax)
        kl = F.kl_div(seg_logits.log_softmax(dim=1), target_seg_prob, reduction="batchmean") / (
            seg_logits.shape[-1] * seg_logits.shape[-2]
        )
        margin = hard_margin_loss(seg_logits, target_seg_argmax)
        pose = (pose_out - target_pose).pow(2).mean()
        seg_feat = feature_loss(seg_tap.features, targets["seg_features"], rows)
        pose_feat = feature_loss(pose_tap.features, targets["pose_features"], rows)
        smooth = tv_loss(frames)
        loss = (
            args.seg_ce_weight * ce
            + args.seg_kl_weight * kl
            + args.seg_margin_weight * margin
            + args.seg_feature_weight * seg_feat
            + args.pose_weight * pose
            + args.pose_feature_weight * pose_feat
            + args.tv_weight * smooth
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(renderer.parameters(), 1.0)
        optimizer.step()

        if stage_step % args.eval_every == 0 or stage_step == args.steps:
            eval_and_record(
                "task",
                stage_step,
                global_step,
                {
                    "loss": float(loss.detach().item()),
                    "ce": float(ce.detach().item()),
                    "kl": float(kl.detach().item()),
                    "margin": float(margin.detach().item()),
                    "pose_mse_loss": float(pose.detach().item()),
                    "seg_feature_loss": float(seg_feat.detach().item()),
                    "pose_feature_loss": float(pose_feat.detach().item()),
                    "tv": float(smooth.detach().item()),
                },
            )
        if global_step % args.checkpoint_every == 0:
            torch.save(
                {
                    "renderer_config": asdict(cfg),
                    "renderer_state_dict": renderer.state_dict(),
                    "sample_ids": sample_ids,
                    "step": global_step,
                    "stage": "task",
                },
                args.out_dir / f"checkpoint_step{global_step:06d}.pt",
            )

    result = {
        "kind": "commavq_task_renderer",
        "sample_ids": sample_ids,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "renderer_config": asdict(cfg),
        "best": best,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out_dir / 'metrics.json'}")
    seg_tap.close()
    pose_tap.close()


if __name__ == "__main__":
    main()
