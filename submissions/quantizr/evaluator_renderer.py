#!/usr/bin/env python
"""Evaluator-native latent renderer probe.

This is a short auto-decoder experiment, not a full submission path. It learns a
tiny renderer plus per-public-sample latent tables directly against the frozen
official SegNet/PoseNet targets on a proxy subset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_utils import AVVideoDataset, DaliVideoDataset, camera_size, segnet_model_input_size  # noqa: E402
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from compress import diff_rgb_to_yuv6, get_pose_tensor  # noqa: E402


def diff_round(x: torch.Tensor) -> torch.Tensor:
    clipped = x.clamp(0.0, 255.0)
    return clipped + (clipped.round() - clipped).detach()


def fake_quant_i4(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    scale = x.detach().abs().amax(dim=dims, keepdim=True).clamp_min(1e-4) / 7.0
    q = (x / scale).round().clamp(-8, 7) * scale
    return x + (q - x).detach()


def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx * 2.0 - 1.0
    yy = yy * 2.0 - 1.0
    rr = torch.sqrt((xx * xx + yy * yy).clamp_min(1e-8))
    horizon = yy
    return torch.stack([xx, yy, rr, horizon], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def boundary_weight(mask: torch.Tensor) -> torch.Tensor:
    # mask: B,H,W
    b = torch.zeros_like(mask, dtype=torch.float32)
    b[:, :, 1:] += (mask[:, :, 1:] != mask[:, :, :-1]).float()
    b[:, :, :-1] += (mask[:, :, 1:] != mask[:, :, :-1]).float()
    b[:, 1:, :] += (mask[:, 1:, :] != mask[:, :-1, :]).float()
    b[:, :-1, :] += (mask[:, 1:, :] != mask[:, :-1, :]).float()
    return 1.0 + 3.0 * (b > 0).float()


def seg_margin_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    target_scores = logits.gather(1, target[:, None]).squeeze(1)
    mask = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool()
    other_scores = logits.masked_fill(mask, -1e9).amax(dim=1)
    return (F.relu(margin - (target_scores - other_scores)) * weight).mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


class SepConvFiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(4, channels)
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.pw(self.dw(x)))
        gamma, beta = self.film(cond).chunk(2, dim=1)
        y = y * (1.0 + 0.1 * gamma[:, :, None, None]) + 0.1 * beta[:, :, None, None]
        return F.silu(x + y)


class EvaluatorRenderer(nn.Module):
    def __init__(
        self,
        z_channels: int,
        z_pose_dim: int,
        width: int = 32,
        blocks: int = 3,
        pose_hidden: int = 64,
    ):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(6 + z_pose_dim, pose_hidden),
            nn.SiLU(),
            nn.Linear(pose_hidden, pose_hidden),
            nn.SiLU(),
        )
        self.in_proj = nn.Conv2d(z_channels + 4, width, 1)
        self.blocks = nn.ModuleList([SepConvFiLMBlock(width, pose_hidden) for _ in range(blocks)])
        self.frame2_head = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU(), nn.Conv2d(width, 3, 1))
        self.frame1_head = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU(), nn.Conv2d(width, 3, 1))

    def forward(self, z_seg: torch.Tensor, z_pose: torch.Tensor, pose6: torch.Tensor, out_hw: tuple[int, int]):
        b = z_seg.shape[0]
        h, w = out_hw
        z_up = F.interpolate(z_seg, size=(h, w), mode="bilinear", align_corners=False)
        coords = make_coord_grid(b, h, w, z_up.device, z_up.dtype)
        cond = self.cond(torch.cat([pose6, z_pose], dim=1))
        feat = F.silu(self.in_proj(torch.cat([z_up, coords], dim=1)))
        for block in self.blocks:
            feat = block(feat, cond)
        frame2 = 127.5 + 127.5 * torch.tanh(self.frame2_head(feat))
        frame1 = 127.5 + 127.5 * torch.tanh(self.frame1_head(feat))
        return frame1, frame2


@dataclass
class ProbeConfig:
    preset: str
    samples: int
    z_channels: int
    z_h: int
    z_w: int
    z_pose_dim: int
    width: int
    blocks: int
    steps: int
    batch_size: int
    lr_model: float
    lr_latent: float
    pose_start_step: int
    eval_interval: int


def make_config(args: argparse.Namespace) -> ProbeConfig:
    if args.preset == "D0":
        return ProbeConfig(
            preset="D0",
            samples=args.samples,
            z_channels=8,
            z_h=8,
            z_w=12,
            z_pose_dim=64,
            width=args.width,
            blocks=args.blocks,
            steps=args.steps,
            batch_size=args.batch_size,
            lr_model=args.lr_model,
            lr_latent=args.lr_latent,
            pose_start_step=args.pose_start_step,
            eval_interval=args.eval_interval,
        )
    if args.preset == "D1":
        return ProbeConfig(
            preset="D1",
            samples=args.samples,
            z_channels=6,
            z_h=8,
            z_w=8,
            z_pose_dim=32,
            width=args.width,
            blocks=args.blocks,
            steps=args.steps,
            batch_size=args.batch_size,
            lr_model=args.lr_model,
            lr_latent=args.lr_latent,
            pose_start_step=args.pose_start_step,
            eval_interval=args.eval_interval,
        )
    raise ValueError(f"unknown preset: {args.preset}")


def load_proxy_targets(args: argparse.Namespace, device: torch.device, segnet: SegNet, posenet: PoseNet):
    batch_size = min(args.load_batch_size, args.samples)
    dataset_cls = DaliVideoDataset if device.type == "cuda" else AVVideoDataset
    files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]
    ds = dataset_cls(files, data_dir=args.video_dir, batch_size=batch_size, device=device, num_threads=2)
    ds.prepare_data()

    all_pose, all_logits, all_mask, all_weight = [], [], [], []
    loaded = 0
    h, w = segnet_model_input_size[1], segnet_model_input_size[0]
    for _, _, batch_rgb in ds:
        batch_rgb = batch_rgb[: args.samples - loaded]
        batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float().to(device)
        with torch.no_grad():
            real2 = F.interpolate(batch[:, 1], size=(h, w), mode="bilinear", align_corners=False)
            logits = segnet(real2).float()
            mask = logits.argmax(dim=1)
            pose = get_pose_tensor(posenet(posenet.preprocess_input(batch))).float()[..., :6]
            weight = boundary_weight(mask)
        all_logits.append(logits.detach().cpu().half())
        all_mask.append(mask.detach().cpu())
        all_pose.append(pose.detach().cpu())
        all_weight.append(weight.detach().cpu().half())
        loaded += batch.shape[0]
        if loaded >= args.samples:
            break

    if loaded < args.samples:
        raise RuntimeError(f"loaded only {loaded} samples, expected {args.samples}")

    targets = {
        "seg_logits": torch.cat(all_logits, dim=0),
        "seg_mask": torch.cat(all_mask, dim=0),
        "pose6": torch.cat(all_pose, dim=0),
        "hard_weight": torch.cat(all_weight, dim=0),
    }
    return targets


def pair_pose_input(pose6: torch.Tensor) -> torch.Tensor:
    # Pose side-channel starts with exact target pose; this is legal side info and
    # mirrors the Quantizr route. The renderer must still make PoseNet agree.
    return pose6


def proxy_eval(
    model: EvaluatorRenderer,
    z_seg_table: nn.Parameter,
    z_pose_table: nn.Parameter,
    targets: dict[str, torch.Tensor],
    device: torch.device,
    cfg: ProbeConfig,
    eval_batch_size: int,
) -> dict[str, float]:
    model.eval()
    h, w = segnet_model_input_size[1], segnet_model_input_size[0]
    segnet, posenet = proxy_eval.segnet, proxy_eval.posenet  # type: ignore[attr-defined]
    total_seg = 0.0
    total_pose = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, cfg.samples, eval_batch_size):
            idx = torch.arange(start, min(cfg.samples, start + eval_batch_size), device=device)
            target_mask = targets["seg_mask"][idx.cpu()].to(device)
            target_pose = targets["pose6"][idx.cpu()].to(device)
            z_seg = fake_quant_i4(z_seg_table[idx], dims=(1, 2, 3))
            z_pose = fake_quant_i4(z_pose_table[idx], dims=(1,))
            f1, f2 = model(z_seg, z_pose, pair_pose_input(target_pose), (h, w))
            f1 = diff_round(f1)
            f2 = diff_round(f2)
            logits = segnet(f2).float()
            seg_dist = (logits.argmax(dim=1) != target_mask).float().mean(dim=(1, 2))
            pair = torch.stack([f1, f2], dim=1)
            pred_pose = get_pose_tensor(posenet(posenet.preprocess_input(pair))).float()[..., :6]
            pose_dist = (pred_pose - target_pose).pow(2).mean(dim=1)
            total_seg += seg_dist.sum().item()
            total_pose += pose_dist.sum().item()
            seen += idx.numel()
    avg_seg = total_seg / seen
    avg_pose = total_pose / seen
    quality = 100.0 * avg_seg + math.sqrt(max(0.0, 10.0 * avg_pose))
    model.train()
    return {
        "segnet_dist": avg_seg,
        "posenet_dist": avg_pose,
        "quality": quality,
        "seg_term": 100.0 * avg_seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * avg_pose)),
    }


def parse_gate(text: str) -> list[tuple[int, float]]:
    if not text:
        return []
    out = []
    for part in text.split(","):
        step_s, quality_s = part.split(":", 1)
        out.append((int(step_s), float(quality_s)))
    return sorted(out)


def estimate_latent_raw_bytes(cfg: ProbeConfig) -> int:
    nibbles_per_sample = cfg.z_channels * cfg.z_h * cfg.z_w + cfg.z_pose_dim
    return int(math.ceil(cfg.samples * nibbles_per_sample / 2.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["D0", "D1"], default="D0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=ROOT_DIR / "videos")
    parser.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--load-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--pose-start-step", type=int, default=600)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--lr-model", type=float, default=8e-4)
    parser.add_argument("--lr-latent", type=float, default=3e-2)
    parser.add_argument("--gate", default="1000:2.0,3000:0.8,6000:0.35")
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = make_config(args)
    gates = parse_gate(args.gate)
    gate_idx = 0

    print(f"config={json.dumps(asdict(cfg), sort_keys=True)}", flush=True)
    print(f"latent_raw_bytes_estimate={estimate_latent_raw_bytes(cfg)}", flush=True)

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)

    proxy_eval.segnet = segnet  # type: ignore[attr-defined]
    proxy_eval.posenet = posenet  # type: ignore[attr-defined]

    print("precomputing official-model targets", flush=True)
    targets = load_proxy_targets(args, device, segnet, posenet)

    z_seg_table = nn.Parameter(torch.randn(cfg.samples, cfg.z_channels, cfg.z_h, cfg.z_w, device=device) * 0.05)
    z_pose_table = nn.Parameter(torch.randn(cfg.samples, cfg.z_pose_dim, device=device) * 0.05)
    model = EvaluatorRenderer(cfg.z_channels, cfg.z_pose_dim, width=cfg.width, blocks=cfg.blocks).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr_model, "weight_decay": 1e-5},
            {"params": [z_seg_table, z_pose_table], "lr": cfg.lr_latent, "weight_decay": 0.0},
        ]
    )

    h, w = segnet_model_input_size[1], segnet_model_input_size[0]
    results_path = args.out_dir / "probe_results.jsonl"
    start_time = time.time()
    model.train()

    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, cfg.samples, (cfg.batch_size,), device=device)
        idx_cpu = idx.detach().cpu()
        target_logits = targets["seg_logits"][idx_cpu].to(device).float()
        target_mask = targets["seg_mask"][idx_cpu].to(device)
        target_pose = targets["pose6"][idx_cpu].to(device)
        hard_weight = targets["hard_weight"][idx_cpu].to(device).float()

        z_seg = fake_quant_i4(z_seg_table[idx], dims=(1, 2, 3))
        z_pose = fake_quant_i4(z_pose_table[idx], dims=(1,))
        f1, f2 = model(z_seg, z_pose, pair_pose_input(target_pose), (h, w))
        f1_q, f2_q = diff_round(f1), diff_round(f2)

        pred_logits = segnet(f2_q).float()
        ce = (F.cross_entropy(pred_logits, target_mask, reduction="none") * hard_weight).mean()
        kl = F.kl_div(
            F.log_softmax(pred_logits / 2.0, dim=1),
            F.softmax(target_logits / 2.0, dim=1),
            reduction="batchmean",
        ) * (4.0 / (h * w))
        margin = seg_margin_loss(pred_logits, target_mask, hard_weight, margin=1.0)
        seg_loss = ce + 0.5 * kl + 0.5 * margin

        pose_loss = torch.tensor(0.0, device=device)
        if step >= cfg.pose_start_step:
            pair = torch.stack([f1_q, f2_q], dim=1)
            pred_pose = get_pose_tensor(posenet(posenet.preprocess_input(pair))).float()[..., :6]
            pose_loss = F.mse_loss(pred_pose, target_pose)

        latent_l1 = z_seg.abs().mean() + z_pose.abs().mean()
        smooth = tv_loss(f1 / 255.0) + tv_loss(f2 / 255.0)
        loss = 100.0 * seg_loss + (4.0 * pose_loss if step >= cfg.pose_start_step else 0.0) + 0.02 * smooth + 0.001 * latent_l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [z_seg_table, z_pose_table], 2.0)
        optimizer.step()

        if step % 20 == 0:
            print(
                f"step={step} loss={loss.item():.5f} seg_loss={seg_loss.item():.5f} "
                f"pose_mse={pose_loss.item():.6f} elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            metrics = proxy_eval(model, z_seg_table, z_pose_table, targets, device, cfg, args.eval_batch_size)
            row = {"step": step, **metrics, "elapsed_sec": time.time() - start_time}
            with results_path.open("a") as f:
                print(json.dumps(row, sort_keys=True), file=f, flush=True)
            print(
                "PROXY "
                f"step={step} quality={metrics['quality']:.5f} "
                f"seg_term={metrics['seg_term']:.5f} pose_term={metrics['pose_term']:.5f} "
                f"seg={metrics['segnet_dist']:.8f} pose={metrics['posenet_dist']:.8f}",
                flush=True,
            )

            while gate_idx < len(gates) and step >= gates[gate_idx][0]:
                gate_step, max_quality = gates[gate_idx]
                if metrics["quality"] > max_quality:
                    print(
                        f"GATE_FAIL step={step} gate_step={gate_step} "
                        f"quality={metrics['quality']:.5f} max={max_quality:.5f}",
                        flush=True,
                    )
                    ckpt = args.out_dir / f"{cfg.preset}_gate_fail_step{step}.pt"
                    torch.save(
                        {
                            "config": asdict(cfg),
                            "model": model.state_dict(),
                            "z_seg": z_seg_table.detach().cpu(),
                            "z_pose": z_pose_table.detach().cpu(),
                            "metrics": metrics,
                        },
                        ckpt,
                    )
                    sys.exit(2)
                gate_idx += 1

        if step % args.save_every == 0:
            torch.save(
                {
                    "config": asdict(cfg),
                    "model": model.state_dict(),
                    "z_seg": z_seg_table.detach().cpu(),
                    "z_pose": z_pose_table.detach().cpu(),
                },
                args.out_dir / f"{cfg.preset}_step{step}.pt",
            )

    final_metrics = proxy_eval(model, z_seg_table, z_pose_table, targets, device, cfg, args.eval_batch_size)
    torch.save(
        {
            "config": asdict(cfg),
            "model": model.state_dict(),
            "z_seg": z_seg_table.detach().cpu(),
            "z_pose": z_pose_table.detach().cpu(),
            "metrics": final_metrics,
        },
        args.out_dir / f"{cfg.preset}_final.pt",
    )
    print(f"FINAL {json.dumps(final_metrics, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
