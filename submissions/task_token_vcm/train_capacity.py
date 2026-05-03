#!/usr/bin/env python3
"""Continuous task-token capacity oracle.

This is deliberately not a final codec. It answers the first Karpathy-style
question for a task-token VCM representation:

  Can learned per-sample tokens plus a small decoder synthesize evaluator-facing
  frame pairs for a selected subset?

Compression, VQ, and entropy coding are intentionally downstream of this file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frame_utils import AVVideoDataset, camera_size
from modules import DistortionNet, posenet_sd_path, segnet_sd_path


ORIGINAL_BYTES = 37_545_489
MODEL_HW = (384, 512)
DEFAULT_HARD8 = [59, 60, 62, 56, 57, 58, 61, 63]


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(max(0.0, 10.0 * posenet_dist))


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + 25.0 * archive_bytes / ORIGINAL_BYTES


def parse_indices(text: str | None, *, offset: int, subset: int, preset: str) -> list[int]:
    if text:
        return [int(x) for x in text.replace(" ", "").split(",") if x]
    if preset == "hard8":
        return list(DEFAULT_HARD8)
    return list(range(offset, offset + subset))


def load_original_pairs_by_indices(
    *,
    data_dir: Path,
    video_names_file: Path,
    sample_indices: list[int],
    batch_size: int,
) -> torch.Tensor:
    requested = list(sample_indices)
    order = {idx: pos for pos, idx in enumerate(requested)}
    sorted_indices = sorted(requested)
    max_index = sorted_indices[-1]
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    ds = AVVideoDataset(names, data_dir=data_dir, batch_size=batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    out: list[torch.Tensor | None] = [None] * len(requested)
    seen = 0
    ptr = 0
    for _, _, batch in ds:
        batch_count = batch.shape[0]
        batch_start, batch_end = seen, seen + batch_count
        seen = batch_end
        while ptr < len(sorted_indices) and sorted_indices[ptr] < batch_end:
            sample_id = sorted_indices[ptr]
            if sample_id >= batch_start:
                out[order[sample_id]] = batch[sample_id - batch_start].contiguous()
            ptr += 1
        if seen > max_index:
            break
    if any(item is None for item in out):
        missing = [requested[i] for i, item in enumerate(out) if item is None]
        raise RuntimeError(f"failed to load samples: {missing}")
    return torch.stack([item for item in out if item is not None], dim=0)


def hard_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    target_logits = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    target_mask = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).bool()
    other_logits = logits.masked_fill(target_mask, -1.0e4).amax(dim=1)
    return F.relu(margin - (target_logits - other_logits)).mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    unit = x / 255.0
    return (unit[..., 1:, :] - unit[..., :-1, :]).abs().mean() + (
        unit[..., :, 1:] - unit[..., :, :-1]
    ).abs().mean()


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


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
class DecoderConfig:
    decoder_kind: str = "cnn"
    token_ch: int = 16
    pair_token_ch: int = 0
    pose_dim: int = 128
    hidden: int = 64
    grid_h: int = 24
    grid_w: int = 32
    num_blocks: int = 4


class TaskTokenDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Conv2d(cfg.token_ch + 2, cfg.hidden, 1)
        self.blocks = nn.ModuleList([DepthwiseSeparableBlock(cfg.hidden) for _ in range(cfg.num_blocks)])
        self.pose_mlp = nn.Sequential(
            nn.Linear(cfg.pose_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.num_blocks * cfg.hidden * 2),
        )
        head_extra = cfg.pair_token_ch + 2 if cfg.pair_token_ch else 0
        if head_extra:
            self.frame2_head = nn.Sequential(
                nn.Conv2d(cfg.hidden + head_extra, cfg.hidden, 1),
                DepthwiseSeparableBlock(cfg.hidden),
                nn.Conv2d(cfg.hidden, 3, 1),
            )
            self.frame1_head = nn.Sequential(
                nn.Conv2d(cfg.hidden + head_extra, cfg.hidden, 1),
                DepthwiseSeparableBlock(cfg.hidden),
                nn.Conv2d(cfg.hidden, 3, 1),
            )
        else:
            # Keep the original module layout so old capacity checkpoints still load.
            self.frame2_head = nn.Sequential(
                DepthwiseSeparableBlock(cfg.hidden),
                nn.Conv2d(cfg.hidden, 3, 1),
            )
            self.frame1_head = nn.Sequential(
                DepthwiseSeparableBlock(cfg.hidden),
                nn.Conv2d(cfg.hidden, 3, 1),
            )

    def _head_input(self, x: torch.Tensor, z_pair_frame: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.pair_token_ch == 0:
            return x
        if z_pair_frame is None:
            raise ValueError("pair_token_ch > 0 requires z_pair in decoder.forward")
        pair = F.interpolate(z_pair_frame, size=x.shape[-2:], mode="bilinear", align_corners=False)
        coords = make_coords(x.shape[0], x.shape[-2], x.shape[-1], x.device)
        return torch.cat([x, pair, coords], dim=1)

    def forward(
        self,
        z_seg: torch.Tensor,
        z_pose: torch.Tensor,
        z_pair: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.decoder_kind == "direct_rgb":
            if z_pair is None or z_pair.shape[2] < 3:
                raise ValueError("direct_rgb decoder requires z_pair with at least 3 channels")
            low = torch.sigmoid(z_pair[:, :, :3]) * 255.0
            flat = low.flatten(0, 1)
            flat = F.interpolate(flat, size=MODEL_HW, mode="bilinear", align_corners=False)
            return flat.reshape(low.shape[0], 2, 3, MODEL_HW[0], MODEL_HW[1])
        if self.cfg.decoder_kind != "cnn":
            raise ValueError(f"unknown decoder_kind: {self.cfg.decoder_kind}")
        b = z_seg.shape[0]
        coords = make_coords(b, z_seg.shape[-2], z_seg.shape[-1], z_seg.device)
        x = self.in_proj(torch.cat([z_seg, coords], dim=1))
        film_raw = self.pose_mlp(z_pose).view(b, self.cfg.num_blocks, 2, self.cfg.hidden)
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            gamma = 0.1 * torch.tanh(film_raw[:, i, 0])
            beta = 0.1 * torch.tanh(film_raw[:, i, 1])
            x = block(x, (gamma, beta))
        if x.shape[-2:] != MODEL_HW:
            x = F.interpolate(x, size=MODEL_HW, mode="bilinear", align_corners=False)
        z_pair_1 = z_pair[:, 0] if z_pair is not None else None
        z_pair_2 = z_pair[:, 1] if z_pair is not None else None
        frame2 = torch.sigmoid(self.frame2_head(self._head_input(x, z_pair_2))) * 255.0
        frame1 = torch.sigmoid(self.frame1_head(self._head_input(x, z_pair_1))) * 255.0
        return torch.stack([frame1, frame2], dim=1)


class FeatureTap:
    def __init__(self, model: nn.Module, names: list[str]):
        self.names = [name for name in names if name]
        modules = dict(model.named_modules())
        missing = [name for name in self.names if name not in modules]
        if missing:
            raise KeyError(f"missing feature modules: {missing}")
        self.features: dict[str, torch.Tensor] = {}
        self.handles = [
            modules[name].register_forward_hook(self._hook(name))
            for name in self.names
        ]

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.features[name] = output
        return hook

    def clear(self) -> None:
        self.features = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def feature_loss(got: dict[str, torch.Tensor], target: dict[str, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
    if not got or not target:
        return torch.zeros([], device=rows.device)
    losses = []
    for name, value in got.items():
        if name not in target:
            continue
        tgt = target[name].to(device=value.device, dtype=value.dtype).index_select(0, rows)
        denom = tgt.detach().pow(2).mean().clamp_min(1.0e-4)
        losses.append((value - tgt).pow(2).mean() / denom)
    if not losses:
        return torch.zeros([], device=rows.device)
    return torch.stack(losses).mean()


@torch.inference_mode()
def collect_targets(
    *,
    distortion: DistortionNet,
    original_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seg_tap: FeatureTap,
    pose_tap: FeatureTap,
) -> dict:
    seg_logits = []
    pose_targets = []
    seg_features: dict[str, list[torch.Tensor]] = {name: [] for name in seg_tap.names}
    pose_features: dict[str, list[torch.Tensor]] = {name: [] for name in pose_tap.names}
    for start in range(0, original_cpu.shape[0], batch_size):
        batch = original_cpu[start : start + batch_size].to(device).permute(0, 1, 4, 2, 3).float()
        seg_tap.clear()
        pose_tap.clear()
        seg_out = distortion.segnet(distortion.segnet.preprocess_input(batch))
        pose_out = distortion.posenet(distortion.posenet.preprocess_input(batch))["pose"][..., :6]
        seg_logits.append(seg_out.detach().cpu())
        pose_targets.append(pose_out.detach().cpu())
        for name, feat in seg_tap.features.items():
            seg_features[name].append(feat.detach().cpu())
        for name, feat in pose_tap.features.items():
            pose_features[name].append(feat.detach().cpu())
    return {
        "seg_logits": torch.cat(seg_logits, dim=0),
        "seg_argmax": torch.cat(seg_logits, dim=0).argmax(dim=1),
        "seg_prob": torch.cat(seg_logits, dim=0).softmax(dim=1),
        "pose": torch.cat(pose_targets, dim=0),
        "seg_features": {name: torch.cat(items, dim=0) for name, items in seg_features.items() if items},
        "pose_features": {name: torch.cat(items, dim=0) for name, items in pose_features.items() if items},
    }


@torch.inference_mode()
def evaluate(
    *,
    decoder: TaskTokenDecoder,
    z_seg: torch.Tensor,
    z_pose: torch.Tensor,
    z_pair: torch.Tensor | None,
    targets: dict,
    distortion: DistortionNet,
    batch_size: int,
    camera_sim: bool,
) -> dict:
    total_seg = 0.0
    total_pose = 0.0
    total = 0
    for start in range(0, z_seg.shape[0], batch_size):
        end = min(z_seg.shape[0], start + batch_size)
        rows = torch.arange(start, end, device=z_seg.device)
        pair_batch = z_pair[start:end] if z_pair is not None else None
        frames = decoder(z_seg[start:end], z_pose[start:end], pair_batch)
        eval_frames = round_ste(frames).clamp(0, 255)
        if camera_sim:
            flat = eval_frames.flatten(0, 1)
            flat = F.interpolate(flat, size=(camera_size[1], camera_size[0]), mode="bicubic", align_corners=False)
            eval_frames = flat.reshape(frames.shape[0], 2, 3, camera_size[1], camera_size[0]).clamp(0, 255)
        seg_logits = distortion.segnet(distortion.segnet.preprocess_input(eval_frames))
        pose = distortion.posenet(distortion.posenet.preprocess_input(eval_frames))["pose"][..., :6]
        target_seg = targets["seg_logits"].to(z_seg.device).index_select(0, rows)
        target_pose = targets["pose"].to(z_seg.device).index_select(0, rows)
        seg_dist = distortion.segnet.compute_distortion(target_seg, seg_logits)
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_seg += float(seg_dist.sum().item())
        total_pose += float(pose_dist.sum().item())
        total += end - start
    seg = total_seg / total
    pose = total_pose / total
    return {
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
        "quality": quality(seg, pose),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=["hard8", "sequential"], default="hard8")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--subset", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--decoder-kind", choices=["cnn", "direct_rgb"], default="cnn")
    parser.add_argument("--token-ch", type=int, default=16)
    parser.add_argument("--pair-token-ch", type=int, default=0)
    parser.add_argument("--pose-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--grid-h", type=int, default=24)
    parser.add_argument("--grid-w", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--lr-decoder", type=float, default=2e-4)
    parser.add_argument("--lr-token", type=float, default=1e-2)
    parser.add_argument("--seg-ce-weight", type=float, default=1.0)
    parser.add_argument("--seg-kl-weight", type=float, default=1.0)
    parser.add_argument("--seg-margin-weight", type=float, default=2.0)
    parser.add_argument("--pose-weight", type=float, default=3.0)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--token-l2-weight", type=float, default=1e-4)
    parser.add_argument("--seg-feature-weight", type=float, default=0.0)
    parser.add_argument("--pose-feature-weight", type=float, default=0.0)
    parser.add_argument("--seg-feature-names", default="encoder.model.blocks.4")
    parser.add_argument("--pose-feature-names", default="summarizer")
    parser.add_argument("--hard-boost", type=float, default=3.0)
    parser.add_argument("--init-direct-from-original", action="store_true")
    parser.add_argument("--camera-sim", action="store_true")
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

    sample_ids = parse_indices(args.indices, offset=args.offset, subset=args.subset, preset=args.preset)
    original_cpu = load_original_pairs_by_indices(
        data_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
        batch_size=max(8, args.batch_size),
    )

    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in distortion.parameters():
        p.requires_grad_(False)

    seg_feature_names = [name for name in args.seg_feature_names.split(",") if name] if args.seg_feature_weight else []
    pose_feature_names = [name for name in args.pose_feature_names.split(",") if name] if args.pose_feature_weight else []
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

    cfg = DecoderConfig(
        decoder_kind=args.decoder_kind,
        token_ch=args.token_ch,
        pair_token_ch=args.pair_token_ch,
        pose_dim=args.pose_dim,
        hidden=args.hidden,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        num_blocks=args.num_blocks,
    )
    decoder = TaskTokenDecoder(cfg).to(device)
    n = len(sample_ids)
    z_seg = nn.Parameter(torch.randn(n, cfg.token_ch, cfg.grid_h, cfg.grid_w, device=device) * 0.02)
    z_pose = nn.Parameter(torch.randn(n, cfg.pose_dim, device=device) * 0.02)
    z_pair = (
        nn.Parameter(torch.randn(n, 2, cfg.pair_token_ch, cfg.grid_h, cfg.grid_w, device=device) * 0.02)
        if cfg.pair_token_ch
        else None
    )
    if args.init_direct_from_original:
        if args.decoder_kind != "direct_rgb" or z_pair is None or cfg.pair_token_ch < 3:
            raise ValueError("--init-direct-from-original requires direct_rgb with pair_token_ch >= 3")
        with torch.no_grad():
            original = original_cpu.to(device).permute(0, 1, 4, 2, 3).float()
            low = F.interpolate(
                original.flatten(0, 1),
                size=(cfg.grid_h, cfg.grid_w),
                mode="area",
            ).reshape(n, 2, 3, cfg.grid_h, cfg.grid_w) / 255.0
            z_pair[:, :, :3].copy_(torch.logit(low.clamp(1.0e-4, 1.0 - 1.0e-4)))
    token_params: list[nn.Parameter] = [z_seg, z_pose]
    if z_pair is not None:
        token_params.append(z_pair)
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder.parameters(), "lr": args.lr_decoder},
            {"params": token_params, "lr": args.lr_token},
        ],
        weight_decay=0.0,
    )

    sample_weights = torch.ones(n, device=device)
    for i, sample_id in enumerate(sample_ids):
        if sample_id in DEFAULT_HARD8:
            sample_weights[i] = args.hard_boost
    sample_probs = sample_weights / sample_weights.sum()

    best = evaluate(
        decoder=decoder,
        z_seg=z_seg,
        z_pose=z_pose,
        z_pair=z_pair,
        targets=targets,
        distortion=distortion,
        batch_size=args.batch_size,
        camera_sim=args.camera_sim,
    )
    best.update({"step": 0})
    with history_path.open("a") as f:
        f.write(json.dumps(best, sort_keys=True) + "\n")
    print(json.dumps(best, sort_keys=True), flush=True)

    for step in tqdm(range(1, args.steps + 1), desc="task-token float capacity"):
        rows = torch.multinomial(sample_probs, num_samples=min(args.batch_size, n), replacement=True)
        rows = rows.to(device)
        pair_batch = z_pair.index_select(0, rows) if z_pair is not None else None
        frames = decoder(z_seg.index_select(0, rows), z_pose.index_select(0, rows), pair_batch)
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
        token_l2 = z_seg.index_select(0, rows).pow(2).mean() + z_pose.index_select(0, rows).pow(2).mean()
        if z_pair is not None:
            token_l2 = token_l2 + z_pair.index_select(0, rows).pow(2).mean()
        loss = (
            args.seg_ce_weight * ce
            + args.seg_kl_weight * kl
            + args.seg_margin_weight * margin
            + args.pose_weight * pose
            + args.seg_feature_weight * seg_feat
            + args.pose_feature_weight * pose_feat
            + args.tv_weight * smooth
            + args.token_l2_weight * token_l2
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + token_params, 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            current = evaluate(
                decoder=decoder,
                z_seg=z_seg,
                z_pose=z_pose,
                z_pair=z_pair,
                targets=targets,
                distortion=distortion,
                batch_size=args.batch_size,
                camera_sim=args.camera_sim,
            )
            current.update(
                {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "ce": float(ce.detach().item()),
                    "kl": float(kl.detach().item()),
                    "margin": float(margin.detach().item()),
                    "pose_mse_loss": float(pose.detach().item()),
                    "tv": float(smooth.detach().item()),
                    "token_l2": float(token_l2.detach().item()),
                }
            )
            with history_path.open("a") as f:
                f.write(json.dumps(current, sort_keys=True) + "\n")
            print(json.dumps(current, sort_keys=True), flush=True)
            if current["quality"] < best["quality"]:
                best = dict(current)
                torch.save(
                    {
                        "decoder_config": asdict(cfg),
                        "decoder_state_dict": decoder.state_dict(),
                        "z_seg": z_seg.detach().cpu(),
                        "z_pose": z_pose.detach().cpu(),
                        "z_pair": z_pair.detach().cpu() if z_pair is not None else None,
                        "sample_ids": sample_ids,
                        "best": best,
                    },
                    args.out_dir / "best_float_capacity.pt",
                )
        if step % args.checkpoint_every == 0:
            torch.save(
                {
                    "decoder_config": asdict(cfg),
                    "decoder_state_dict": decoder.state_dict(),
                    "z_seg": z_seg.detach().cpu(),
                    "z_pose": z_pose.detach().cpu(),
                    "z_pair": z_pair.detach().cpu() if z_pair is not None else None,
                    "sample_ids": sample_ids,
                    "step": step,
                },
                args.out_dir / f"checkpoint_step{step:06d}.pt",
            )

    result = {
        "kind": "float_capacity",
        "sample_ids": sample_ids,
        "args": vars(args) | {"out_dir": str(args.out_dir), "uncompressed_dir": str(args.uncompressed_dir), "video_names_file": str(args.video_names_file)},
        "decoder_config": asdict(cfg),
        "best": best,
        "score_table": {
            "allowed_quality_220kb": 0.300 - 25.0 * 220_000 / ORIGINAL_BYTES,
            "allowed_quality_200kb": 0.300 - 25.0 * 200_000 / ORIGINAL_BYTES,
            "allowed_quality_180kb": 0.300 - 25.0 * 180_000 / ORIGINAL_BYTES,
            "allowed_quality_160kb": 0.300 - 25.0 * 160_000 / ORIGINAL_BYTES,
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out_dir / 'metrics.json'}")
    seg_tap.close()
    pose_tap.close()


if __name__ == "__main__":
    main()
