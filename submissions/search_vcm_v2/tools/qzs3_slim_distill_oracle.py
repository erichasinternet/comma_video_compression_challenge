#!/usr/bin/env python
"""Slim qpose-compatible generator distillation oracle.

The exact mask appears to be the irreducible payload. Post-hoc lower-bit QZS3
quantization is precision-fragile. This oracle tests the remaining legitimate
reduction route: train a narrower generator that preserves the qpose data path
and exact mask manifold, instead of inventing a new renderer.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QZS3_DIR = REPO_ROOT / "submissions/qpose14_qzs3_filmq9g_slsb1_r55"
if str(QZS3_DIR) not in sys.path:
    sys.path.insert(0, str(QZS3_DIR))

import inflate as qzs3  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json  # noqa: E402
from submissions.search_vcm_v2.families.qpose14_data import load_original_subset, select_torch_device  # noqa: E402
from submissions.search_vcm_v2.subsets import get_subset  # noqa: E402
from submissions.search_vcm_v2.tools.qzs3_lowbit_quant_oracle import (  # noqa: E402
    ARCHIVE_BYTES,
    MASK_BYTES,
    evaluate_model,
    load_generator,
)
from submissions.search_vcm_v2.tools.qzs3_qat_oracle import (  # noqa: E402
    build_targets,
    frame_imitation_loss,
    qpose_fullres,
)


OUT_DIR = EXPERIMENTS_DIR / "qzs3_slim_distill_oracle"


@dataclass(frozen=True)
class SlimConfig:
    name: str
    emb: int
    c1: int
    c2: int
    hidden: int
    cond: int
    depth_mult: int = 1


CONFIGS = {
    "s48": SlimConfig("s48", emb=6, c1=48, c2=56, hidden=44, cond=40),
    "s44": SlimConfig("s44", emb=5, c1=44, c2=52, hidden=40, cond=36),
    "s40": SlimConfig("s40", emb=5, c1=40, c2=48, hidden=36, cond=32),
    "s36": SlimConfig("s36", emb=5, c1=36, c2=44, hidden=32, cond=28),
}


class SlimJointFrameGenerator(torch.nn.Module):
    def __init__(self, cfg: SlimConfig):
        super().__init__()
        self.cfg = cfg
        self.shared_trunk = qzs3.SharedMaskDecoder(
            num_classes=5,
            emb_dim=cfg.emb,
            c1=cfg.c1,
            c2=cfg.c2,
            depth_mult=cfg.depth_mult,
        )
        self.pose_mlp = torch.nn.Sequential(
            torch.nn.Linear(6, cfg.cond),
            torch.nn.SiLU(),
            torch.nn.Linear(cfg.cond, cfg.cond),
        )
        self.frame1_head = qzs3.FrameHead(
            in_ch=cfg.c1,
            cond_dim=cfg.cond,
            hidden=cfg.hidden,
            depth_mult=cfg.depth_mult,
        )
        self.frame2_head = qzs3.Frame2StaticHead(
            in_ch=cfg.c1,
            hidden=cfg.hidden,
            depth_mult=cfg.depth_mult,
        )

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor):
        b = mask2.shape[0]
        coords = qzs3.make_coord_grid(b, 384, 512, mask2.device, torch.float32)
        shared_feat = self.shared_trunk(mask2, coords)
        frame2 = self.frame2_head(shared_feat)
        cond = self.pose_mlp(pose6)
        frame1 = self.frame1_head(shared_feat, cond)
        return frame1, frame2


def estimate_slim_raw_bytes(model: torch.nn.Module) -> dict[str, int]:
    block_size = 32
    qv_specs = {
        "frame1_head.block1.film_proj.weight": (9, False),
        "pose_mlp.2.weight": (10, True),
    }
    covered: set[str] = set()
    sizes = {"packed": 0, "scales": 0, "bias": 0, "dense_fp": 0, "fp_weight": 0, "dense_other": 0, "qv": 0}
    for name, module in model.named_modules():
        if not isinstance(module, (qzs3.QConv2d, qzs3.QEmbedding)):
            continue
        covered.add(f"{name}.weight")
        if getattr(module, "quantize_weight", False):
            numel = int(module.weight.numel())
            sizes["packed"] += (numel * 4 + 7) // 8
            sizes["scales"] += ((numel + block_size - 1) // block_size) * 2
        else:
            sizes["fp_weight"] += int(module.weight.numel()) * 2
        if isinstance(module, qzs3.QConv2d) and module.bias is not None:
            covered.add(f"{name}.bias")
            sizes["bias"] += int(module.bias.numel()) * 2
    for key, tensor in model.state_dict().items():
        if key in covered:
            continue
        count = int(tensor.numel())
        if key in qv_specs:
            bits, per_row = qv_specs[key]
            rows = tensor.shape[0] if per_row and tensor.ndim >= 2 else 1
            sizes["qv"] += rows * 4 + (count * bits + 7) // 8
        elif tensor.is_floating_point():
            sizes["dense_fp"] += count * 2
        else:
            sizes["dense_other"] += count * tensor.element_size()
    sizes["total_raw_plus_header"] = sum(sizes.values()) + 6
    return sizes


def render_fullres(model: torch.nn.Module, mask: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    frame1, frame2 = model(mask.long(), pose.float())
    frame1 = F.interpolate(frame1, size=(874, 1164), mode="bilinear", align_corners=False)
    frame2 = F.interpolate(frame2, size=(874, 1164), mode="bilinear", align_corners=False)
    pred = torch.stack([frame1, frame2], dim=1).clamp(0, 255)
    return einops.rearrange(pred, "b t c h w -> b t h w c")


@torch.inference_mode()
def build_frame_targets(distortion: DistortionNet, frames: torch.Tensor) -> dict[str, Any]:
    posenet_in, segnet_in = distortion.preprocess_input(frames)
    pose_out, seg_logits = distortion.posenet(posenet_in), distortion.segnet(segnet_in)
    return {
        "pose": {k: v.detach() for k, v in pose_out.items()},
        "seg_logits": seg_logits.detach(),
        "seg_argmax": seg_logits.argmax(dim=1).detach(),
    }


def slim_task_loss(
    distortion: DistortionNet,
    pred: torch.Tensor,
    targets: dict[str, Any],
    qpose_pred: torch.Tensor | None,
    *,
    anchor_weight: float,
    pose_weight: float,
    seg_kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    posenet_in, segnet_in = distortion.preprocess_input(pred)
    pose_out = distortion.posenet(posenet_in)
    seg_logits = distortion.segnet(segnet_in)

    seg_ce = F.cross_entropy(seg_logits, targets["seg_argmax"])
    seg_kl = F.kl_div(
        F.log_softmax(seg_logits, dim=1),
        F.softmax(targets["seg_logits"], dim=1),
        reduction="batchmean",
    )
    pose_mse = (pose_out["pose"][..., :6] - targets["pose"]["pose"][..., :6]).pow(2).mean()
    loss = seg_ce + seg_kl_weight * seg_kl + pose_weight * pose_mse
    anchor = torch.zeros((), device=pred.device)
    if qpose_pred is not None and anchor_weight > 0:
        anchor = F.smooth_l1_loss(pred.contiguous() / 255.0, qpose_pred.contiguous() / 255.0)
        loss = loss + anchor_weight * anchor
    return loss, {
        "seg_ce": float(seg_ce.detach().cpu()),
        "seg_kl": float(seg_kl.detach().cpu()),
        "pose_mse": float(pose_mse.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
        "pose_weight": float(pose_weight),
        "seg_kl_weight": float(seg_kl_weight),
    }


def projected_archive_bytes(model_raw_bytes: int, pose_bytes: int = 899) -> int:
    # PR67 raw model 59,288 B compresses to 56,093 B. Use that ratio as a
    # conservative early estimate for single-member archive sizing.
    compressed_model_est = int(round(model_raw_bytes * (56_093 / 59_288)))
    return MASK_BYTES + compressed_model_est + pose_bytes + 100


def sliced_teacher_init(student: torch.nn.Module, teacher: torch.nn.Module) -> dict[str, int]:
    """Copy overlapping state_dict slices from teacher into a narrower student."""

    teacher_state = teacher.state_dict()
    copied = 0
    skipped = 0
    with torch.no_grad():
        for key, value in student.state_dict().items():
            source = teacher_state.get(key)
            if source is None or source.ndim != value.ndim:
                skipped += 1
                continue
            slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, source.shape, strict=True))
            value.zero_()
            value[slices].copy_(source[slices].to(device=value.device, dtype=value.dtype))
            copied += 1
    return {"copied": copied, "skipped": skipped}


def _topk_sorted(score: torch.Tensor, k: int) -> torch.Tensor:
    idx = torch.topk(score.float(), k=k).indices
    return torch.sort(idx).values


def _norm_score(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    axes = tuple(i for i in range(tensor.ndim) if i != dim)
    return tensor.detach().float().pow(2).sum(dim=axes).sqrt()


def _copy_sliced(dst: torch.Tensor, src: torch.Tensor, axes: dict[int, torch.Tensor]) -> None:
    value = src
    for dim in range(src.ndim):
        idx = axes.get(dim)
        if idx is not None:
            value = value.index_select(dim, idx.to(src.device))
        elif src.shape[dim] != dst.shape[dim]:
            value = value.narrow(dim, 0, dst.shape[dim])
    dst.copy_(value.to(device=dst.device, dtype=dst.dtype))


def indexed_teacher_init(student: SlimJointFrameGenerator, teacher: torch.nn.Module) -> dict[str, Any]:
    """Importance-pruned teacher copy for qpose-compatible slim models."""

    cfg = student.cfg
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    device = next(teacher.parameters()).device

    c1_score = _norm_score(t_state["shared_trunk.fuse_block.conv2.pw.weight"], dim=0)
    c1_idx = _topk_sorted(c1_score, cfg.c1).to(device)
    c2_score = _norm_score(t_state["shared_trunk.down_block.conv2.pw.weight"], dim=0)
    c2_idx = _topk_sorted(c2_score, cfg.c2).to(device)
    embcoord_idx = torch.tensor(list(range(cfg.emb)) + [6, 7], device=device)
    c1cat_idx = torch.cat([c1_idx, c1_idx + 56])
    film_idx = torch.cat([c1_idx, c1_idx + 56])
    cond_score = _norm_score(t_state["pose_mlp.2.weight"], dim=0)
    cond_idx = _topk_sorted(cond_score, cfg.cond).to(device)
    f1_hidden_idx = _topk_sorted(_norm_score(t_state["frame1_head.pre.pw.weight"], dim=0), cfg.hidden).to(device)
    f2_hidden_idx = _topk_sorted(_norm_score(t_state["frame2_head.pre.pw.weight"], dim=0), cfg.hidden).to(device)

    copied = 0
    skipped: list[str] = []

    def copy_key(key: str, axes: dict[int, torch.Tensor]) -> None:
        nonlocal copied
        if key not in t_state:
            skipped.append(key)
            return
        _copy_sliced(s_state[key], t_state[key], axes)
        copied += 1

    # Embedding and shared trunk transitions.
    copy_key("shared_trunk.embedding.weight", {1: torch.arange(cfg.emb, device=device)})
    for prefix in ["shared_trunk.stem_conv"]:
        copy_key(f"{prefix}.dw.weight", {0: embcoord_idx})
        copy_key(f"{prefix}.pw.weight", {0: c1_idx, 1: embcoord_idx})
        copy_key(f"{prefix}.pw.bias", {0: c1_idx})
        copy_key(f"{prefix}.norm.weight", {0: c1_idx})
        copy_key(f"{prefix}.norm.bias", {0: c1_idx})
    copy_key("shared_trunk.down_conv.dw.weight", {0: c1_idx})
    copy_key("shared_trunk.down_conv.pw.weight", {0: c2_idx, 1: c1_idx})
    copy_key("shared_trunk.down_conv.pw.bias", {0: c2_idx})
    copy_key("shared_trunk.down_conv.norm.weight", {0: c2_idx})
    copy_key("shared_trunk.down_conv.norm.bias", {0: c2_idx})
    copy_key("shared_trunk.up.1.dw.weight", {0: c2_idx})
    copy_key("shared_trunk.up.1.pw.weight", {0: c1_idx, 1: c2_idx})
    copy_key("shared_trunk.up.1.pw.bias", {0: c1_idx})
    copy_key("shared_trunk.up.1.norm.weight", {0: c1_idx})
    copy_key("shared_trunk.up.1.norm.bias", {0: c1_idx})
    copy_key("shared_trunk.fuse.dw.weight", {0: c1cat_idx})
    copy_key("shared_trunk.fuse.pw.weight", {0: c1_idx, 1: c1cat_idx})
    copy_key("shared_trunk.fuse.pw.bias", {0: c1_idx})
    copy_key("shared_trunk.fuse.norm.weight", {0: c1_idx})
    copy_key("shared_trunk.fuse.norm.bias", {0: c1_idx})

    def copy_cblock(prefix: str, idx: torch.Tensor) -> None:
        for sub in ["conv1", "conv2"]:
            copy_key(f"{prefix}.{sub}.dw.weight", {0: idx})
            copy_key(f"{prefix}.{sub}.pw.weight", {0: idx, 1: idx})
            if f"{prefix}.{sub}.pw.bias" in s_state:
                copy_key(f"{prefix}.{sub}.pw.bias", {0: idx})
            if f"{prefix}.{sub}.norm.weight" in s_state:
                copy_key(f"{prefix}.{sub}.norm.weight", {0: idx})
                copy_key(f"{prefix}.{sub}.norm.bias", {0: idx})
        if f"{prefix}.norm2.weight" in s_state:
            copy_key(f"{prefix}.norm2.weight", {0: idx})
            copy_key(f"{prefix}.norm2.bias", {0: idx})

    for prefix in ["shared_trunk.stem_block", "shared_trunk.fuse_block", "frame1_head.block2", "frame2_head.block1", "frame2_head.block2"]:
        copy_cblock(prefix, c1_idx)
    copy_cblock("shared_trunk.down_block", c2_idx)

    # Frame1 FiLM block: c1 channels plus condition vector.
    copy_cblock("frame1_head.block1", c1_idx)
    copy_key("frame1_head.block1.film_proj.weight", {0: film_idx, 1: cond_idx})
    copy_key("frame1_head.block1.film_proj.bias", {0: film_idx})

    # Heads.
    for head, hidx in [("frame1_head", f1_hidden_idx), ("frame2_head", f2_hidden_idx)]:
        copy_key(f"{head}.pre.dw.weight", {0: c1_idx})
        copy_key(f"{head}.pre.pw.weight", {0: hidx, 1: c1_idx})
        copy_key(f"{head}.pre.pw.bias", {0: hidx})
        copy_key(f"{head}.pre.norm.weight", {0: hidx})
        copy_key(f"{head}.pre.norm.bias", {0: hidx})
        copy_key(f"{head}.head.weight", {1: hidx})
        copy_key(f"{head}.head.bias", {})

    # Pose MLP.
    copy_key("pose_mlp.0.weight", {0: cond_idx})
    copy_key("pose_mlp.0.bias", {0: cond_idx})
    copy_key("pose_mlp.2.weight", {0: cond_idx, 1: cond_idx})
    copy_key("pose_mlp.2.bias", {0: cond_idx})

    return {
        "copied": copied,
        "skipped": skipped,
        "c1_idx": c1_idx.detach().cpu().tolist(),
        "c2_idx": c2_idx.detach().cpu().tolist(),
        "cond_idx": cond_idx.detach().cpu().tolist(),
    }


def train_slim(
    *,
    cfg: SlimConfig,
    subset_name: str,
    device_name: str,
    max_steps: int,
    eval_every: int,
    warmup_steps: int,
    batch_size: int,
    lr: float,
    init: str,
    target: str,
    pose_weight: float,
    seg_kl_weight: float,
    out: Path,
) -> dict[str, Any]:
    device = select_torch_device(device_name)
    sample_ids = get_subset(subset_name)
    teacher_model, masks, poses = load_generator(device)
    base_metrics = evaluate_model(teacher_model, masks, poses, subset_name=subset_name, device=device)
    qpose_teacher = qpose_fullres(teacher_model, masks, poses, sample_ids, device)

    model = SlimJointFrameGenerator(cfg).to(device)
    init_info = {"mode": init}
    if init == "sliced":
        init_info.update(sliced_teacher_init(model, teacher_model))
    elif init == "indexed":
        init_info.update(indexed_teacher_init(model, teacher_model))
    model.train()
    raw_sizes = estimate_slim_raw_bytes(model)
    projected_bytes = projected_archive_bytes(raw_sizes["total_raw_plus_header"])

    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for param in distortion.parameters():
        param.requires_grad_(False)
    original_targets = build_targets(sample_ids, device)
    teacher_targets = build_frame_targets(distortion, qpose_teacher)
    targets = teacher_targets if target == "teacher" else original_targets

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    index = torch.arange(len(sample_ids), device=device)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start_time = time.time()

    for step in range(1, max_steps + 1):
        if step <= warmup_steps:
            anchor_weight = 0.0
        elif step < max_steps * 0.75:
            anchor_weight = 0.5
        else:
            anchor_weight = 0.1
        perm = index[torch.randperm(len(sample_ids), device=device)[:batch_size]]
        ids = [sample_ids[int(i)] for i in perm.detach().cpu().tolist()]
        local = perm.detach().cpu().tolist()
        pred = render_fullres(model, masks[ids].to(device), poses[ids].to(device))
        if step <= warmup_steps:
            loss, parts = frame_imitation_loss(pred, qpose_teacher[local])
        else:
            batch_targets = {
                "seg_argmax": targets["seg_argmax"][local],
                "seg_logits": targets["seg_logits"][local],
                "pose": {k: v[local] for k, v in targets["pose"].items()},
            }
            loss, parts = slim_task_loss(
                distortion,
                pred,
                batch_targets,
                qpose_teacher[local],
                anchor_weight=anchor_weight,
                pose_weight=pose_weight,
                seg_kl_weight=seg_kl_weight,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            model.eval()
            metrics = evaluate_model(model, masks, poses, subset_name=subset_name, device=device)
            row = {
                "step": step,
                "config": cfg.__dict__,
                "loss": float(loss.detach().cpu()),
                **parts,
                **metrics,
                "quality_delta_vs_teacher": metrics["quality"] - base_metrics["quality"],
                "model_raw_bytes_est": raw_sizes["total_raw_plus_header"],
                "projected_archive_bytes": projected_bytes,
                "projected_score": metrics["quality"] + rate_term(projected_bytes),
                "elapsed_sec": time.time() - start_time,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "step": row["step"],
                        "config": cfg.name,
                        "quality": row["quality"],
                        "quality_delta_vs_teacher": row["quality_delta_vs_teacher"],
                        "projected_archive_bytes": row["projected_archive_bytes"],
                        "projected_score": row["projected_score"],
                        "loss": row["loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if best is None or row["projected_score"] < best["projected_score"]:
                best = row
            model.train()

    assert best is not None
    summary = {
        "subset": subset_name,
        "device": str(device),
        "base": base_metrics,
        "config": cfg.__dict__,
        "init": init_info,
        "target": target,
        "pose_weight": pose_weight,
        "seg_kl_weight": seg_kl_weight,
        "raw_sizes": raw_sizes,
        "projected_archive_bytes": projected_bytes,
        "target_note": "A legitimate 0.2x candidate needs projected_score < 0.300, preferably <=250-255KB archive.",
        "best": best,
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{subset_name}_{cfg.name}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="hard3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="s40")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init", choices=["random", "sliced", "indexed"], default="indexed")
    parser.add_argument("--target", choices=["teacher", "original"], default="teacher")
    parser.add_argument("--pose-weight", type=float, default=10.0)
    parser.add_argument("--seg-kl-weight", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    train_slim(
        cfg=CONFIGS[args.config],
        subset_name=args.subset,
        device_name=args.device,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        init=args.init,
        target=args.target,
        pose_weight=args.pose_weight,
        seg_kl_weight=args.seg_kl_weight,
        out=args.out,
    )


if __name__ == "__main__":
    main()
