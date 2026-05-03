#!/usr/bin/env python
"""Full PR #67 qpose14_qzs3 evaluator fine-tuning oracle.

After mask/model byte reduction probes failed, the remaining legitimate route is
same-size quality improvement: keep the exact mask, pose stream, and architecture
but fine-tune the generator directly against the frozen evaluator.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json  # noqa: E402
from submissions.search_vcm_v2.families.qpose14_data import load_original_subset, select_torch_device  # noqa: E402
from submissions.search_vcm_v2.subsets import get_subset  # noqa: E402
from submissions.search_vcm_v2.tools.qzs3_lowbit_quant_oracle import ARCHIVE_BYTES, load_generator  # noqa: E402


OUT_DIR = EXPERIMENTS_DIR / "qzs3_full_finetune_oracle"


def render_fullres(model: torch.nn.Module, mask: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    frame1, frame2 = model(mask.long(), pose.float())
    frame1 = F.interpolate(frame1, size=(874, 1164), mode="bilinear", align_corners=False)
    frame2 = F.interpolate(frame2, size=(874, 1164), mode="bilinear", align_corners=False)
    pred = torch.stack([frame1, frame2], dim=1).clamp(0, 255)
    return einops.rearrange(pred, "b t c h w -> b t h w c")


@torch.inference_mode()
def build_targets(sample_ids: list[int], device: torch.device) -> dict[str, Any]:
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    original = load_original_subset(
        "qzs3_ft_" + "_".join(map(str, sample_ids)),
        sample_ids,
        device="cpu",
        force=True,
    ).float().to(device)
    posenet_in, segnet_in = distortion.preprocess_input(original)
    pose_target = distortion.posenet(posenet_in)
    seg_logits = distortion.segnet(segnet_in)
    return {
        "original": original,
        "pose": {k: v.detach() for k, v in pose_target.items()},
        "seg_logits": seg_logits.detach(),
        "seg_argmax": seg_logits.argmax(dim=1).detach(),
    }


def evaluator_loss(
    distortion: DistortionNet,
    pred: torch.Tensor,
    targets: dict[str, Any],
    qpose_anchor: torch.Tensor,
    *,
    pose_weight: float,
    seg_kl_weight: float,
    anchor_weight: float,
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
    anchor = F.smooth_l1_loss(pred.contiguous() / 255.0, qpose_anchor.contiguous() / 255.0)
    loss = seg_ce + seg_kl_weight * seg_kl + pose_weight * pose_mse + anchor_weight * anchor
    return loss, {
        "seg_ce": float(seg_ce.detach().cpu()),
        "seg_kl": float(seg_kl.detach().cpu()),
        "pose_mse": float(pose_mse.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
    }


@torch.inference_mode()
def evaluate_with_original(
    model: torch.nn.Module,
    masks: torch.Tensor,
    poses: torch.Tensor,
    sample_ids: list[int],
    original: torch.Tensor,
    distortion: DistortionNet,
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for start in range(0, len(sample_ids), 2):
        ids = sample_ids[start : start + 2]
        pred = render_fullres(model, masks[ids].to(device), poses[ids].to(device))
        pose_dist, seg_dist = distortion.compute_distortion(original[start : start + len(ids)], pred)
        for sid, pose_v, seg_v in zip(ids, pose_dist.cpu().tolist(), seg_dist.cpu().tolist(), strict=True):
            seg = float(seg_v)
            pose_val = float(pose_v)
            rows.append(
                {
                    "sample_id": int(sid),
                    "segnet_dist": seg,
                    "posenet_dist": pose_val,
                    "seg_term": 100.0 * seg,
                    "pose_term": float(torch.sqrt(torch.tensor(10.0 * pose_val)).item()),
                    "quality": quality(seg, pose_val),
                }
            )
    seg_mean = sum(row["segnet_dist"] for row in rows) / len(rows)
    pose_mean = sum(row["posenet_dist"] for row in rows) / len(rows)
    return {
        "segnet_dist": seg_mean,
        "posenet_dist": pose_mean,
        "quality": quality(seg_mean, pose_mean),
        "score": quality(seg_mean, pose_mean) + rate_term(ARCHIVE_BYTES),
        "max_sample_quality": max(row["quality"] for row in rows),
        "per_sample": rows,
    }


@torch.inference_mode()
def qpose_fullres(model: torch.nn.Module, masks: torch.Tensor, poses: torch.Tensor, sample_ids: list[int], device: torch.device) -> torch.Tensor:
    chunks = []
    for sid in sample_ids:
        chunks.append(render_fullres(model, masks[sid : sid + 1].to(device), poses[sid : sid + 1].to(device)).detach())
    return torch.cat(chunks, dim=0)


def train_full_finetune(
    *,
    subset_name: str,
    device_name: str,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    lr: float,
    pose_weight: float,
    anchor_weight: float,
    train_scope: str,
    seed: int,
    out: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = select_torch_device(device_name)
    sample_ids = get_subset(subset_name)
    base_model, masks, poses = load_generator(device)
    anchor_frames = qpose_fullres(base_model, masks, poses, sample_ids, device)

    model = copy.deepcopy(base_model).to(device).train()
    for name, param in model.named_parameters():
        if train_scope == "all":
            trainable = True
        elif train_scope == "pose_mlp":
            trainable = name.startswith("pose_mlp.")
        elif train_scope == "frame1":
            trainable = name.startswith("pose_mlp.") or name.startswith("frame1_head.")
        else:
            raise ValueError(f"unknown train scope: {train_scope}")
        param.requires_grad_(trainable)

    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for param in distortion.parameters():
        param.requires_grad_(False)
    targets = build_targets(sample_ids, device)
    base_metrics = evaluate_with_original(base_model, masks, poses, sample_ids, targets["original"], distortion, device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"no trainable parameters for scope {train_scope}")
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-7)
    index = torch.arange(len(sample_ids), device=device)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start = time.time()

    out.mkdir(parents=True, exist_ok=True)
    best_checkpoint = out / f"{subset_name}_{train_scope}_best.pt"

    for step in range(1, max_steps + 1):
        perm = index[torch.randperm(len(sample_ids), device=device)[:batch_size]]
        ids = [sample_ids[int(i)] for i in perm.detach().cpu().tolist()]
        local = perm.detach().cpu().tolist()
        pred = render_fullres(model, masks[ids].to(device), poses[ids].to(device))
        batch_targets = {
            "seg_argmax": targets["seg_argmax"][local],
            "seg_logits": targets["seg_logits"][local],
            "pose": {k: v[local] for k, v in targets["pose"].items()},
        }
        loss, parts = evaluator_loss(
            distortion,
            pred,
            batch_targets,
            anchor_frames[local],
            pose_weight=pose_weight,
            seg_kl_weight=1e-5,
            anchor_weight=anchor_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            model.eval()
            metrics = evaluate_with_original(model, masks, poses, sample_ids, targets["original"], distortion, device)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                **parts,
                **metrics,
                "quality_delta_vs_base": metrics["quality"] - base_metrics["quality"],
                "score_at_pr67_bytes": metrics["quality"] + rate_term(ARCHIVE_BYTES),
                "score_delta_vs_base": metrics["quality"] - base_metrics["quality"],
                "elapsed_sec": time.time() - start,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ["step", "quality", "quality_delta_vs_base", "score_at_pr67_bytes", "loss"]}, sort_keys=True), flush=True)
            if best is None or row["quality"] < best["quality"]:
                best = row
                torch.save(
                    {
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "row": row,
                        "base": base_metrics,
                        "subset": subset_name,
                        "train_scope": train_scope,
                        "seed": seed,
                    },
                    best_checkpoint,
                )
            model.train()

    assert best is not None
    summary = {
        "subset": subset_name,
        "device": str(device),
        "archive_bytes": ARCHIVE_BYTES,
        "base": base_metrics,
        "best": best,
        "rows": rows,
        "pose_weight": pose_weight,
        "anchor_weight": anchor_weight,
        "train_scope": train_scope,
        "seed": seed,
        "trainable_params": sum(int(param.numel()) for param in trainable_params),
        "best_checkpoint": str(best_checkpoint),
        "target_note": "At PR67 bytes, visible <0.300 requires about 0.1158 quality.",
    }
    write_json(out / f"{subset_name}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="hard8")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--pose-weight", type=float, default=30.0)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--train-scope", choices=["all", "pose_mlp", "frame1"], default="all")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    train_full_finetune(
        subset_name=args.subset,
        device_name=args.device,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        lr=args.lr,
        pose_weight=args.pose_weight,
        anchor_weight=args.anchor_weight,
        train_scope=args.train_scope,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
