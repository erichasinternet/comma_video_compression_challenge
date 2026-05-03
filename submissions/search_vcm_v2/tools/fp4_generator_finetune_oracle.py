#!/usr/bin/env python
"""Fine-tune PR #62's low-byte FP4 generator against the frozen evaluator.

This is a capacity oracle for the most promising remaining byte regime:
keep PR #62's low-rate mask/pose/model architecture, start from its decoded
weights, and ask whether evaluator-backed training can move quality toward
qpose14 without adding payload.
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

import modules as challenge_modules  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json  # noqa: E402
from submissions.search_vcm_v2.families.lowmask_data import (  # noqa: E402
    FP4_ARCHIVE,
    decode_fp4_pose_stream,
    decode_lowmask_video,
    load_fp4_generator,
    split_fp4_archive,
)
from submissions.search_vcm_v2.families.qpose14_data import (  # noqa: E402
    load_original_subset,
    materialize_qpose14_subset,
    select_torch_device,
)
from submissions.search_vcm_v2.subsets import get_subset  # noqa: E402


OUT_DIR = EXPERIMENTS_DIR / "fp4_generator_finetune_oracle"
ARCHIVE_BYTES = 249_624


def _patch_allnorm_for_grad() -> None:
    """The evaluator's AllNorm uses view(), which can fail on grad tensors."""

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.bn(x.reshape(-1, 1)).reshape(x.shape)

    challenge_modules.AllNorm.forward = forward


def render_fullres(model: torch.nn.Module, mask: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    frame1, frame2 = model(mask.long(), pose.float())
    frame1 = F.interpolate(frame1, size=(874, 1164), mode="bilinear", align_corners=False)
    frame2 = F.interpolate(frame2, size=(874, 1164), mode="bilinear", align_corners=False)
    pred = torch.stack([frame1, frame2], dim=1).clamp(0, 255)
    return einops.rearrange(pred, "b t c h w -> b t h w c").contiguous()


def load_fp4_artifact(device: torch.device) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    mask_br, model_br, pose_br = split_fp4_archive(FP4_ARCHIVE)
    masks, _ = decode_lowmask_video(mask_br)
    poses = decode_fp4_pose_stream(pose_br)
    model = load_fp4_generator(model_br, device)
    return model, masks.contiguous(), poses.contiguous()


@torch.inference_mode()
def build_targets(sample_ids: list[int], device: torch.device) -> dict[str, Any]:
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    original = load_original_subset(
        "fp4_ft_" + "_".join(map(str, sample_ids)),
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


@torch.inference_mode()
def internal_teacher_fullres(kind: str, base_model: torch.nn.Module, masks: torch.Tensor, poses: torch.Tensor, sample_ids: list[int], device: torch.device) -> torch.Tensor:
    if kind == "fp4":
        return torch.cat([render_fullres(base_model, masks[sid : sid + 1].to(device), poses[sid : sid + 1].to(device)) for sid in sample_ids], dim=0)
    if kind == "qpose":
        qpose = materialize_qpose14_subset("fp4_ft_qpose_" + "_".join(map(str, sample_ids)), sample_ids, device=str(device))
        f1 = F.interpolate(qpose["qpose_frame1"].to(device).float(), size=(874, 1164), mode="bilinear", align_corners=False)
        f2 = F.interpolate(qpose["qpose_frame2"].to(device).float(), size=(874, 1164), mode="bilinear", align_corners=False)
        return einops.rearrange(torch.stack([f1, f2], dim=1).clamp(0, 255), "b t c h w -> b t h w c").contiguous()
    raise ValueError(f"unknown anchor kind: {kind}")


def evaluator_loss(
    distortion: DistortionNet,
    pred: torch.Tensor,
    targets: dict[str, Any],
    anchor_frames: torch.Tensor,
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
    ) / (targets["seg_logits"].shape[-1] * targets["seg_logits"].shape[-2])
    target_logit = seg_logits.gather(1, targets["seg_argmax"].unsqueeze(1)).squeeze(1)
    masked = seg_logits.masked_fill(F.one_hot(targets["seg_argmax"], 5).permute(0, 3, 1, 2).bool(), -1e4)
    margin = F.relu(masked.max(dim=1).values - target_logit + 0.25).mean()
    pose_mse = (pose_out["pose"][..., :6] - targets["pose"]["pose"][..., :6]).pow(2).mean()
    anchor = F.smooth_l1_loss(pred / 255.0, anchor_frames / 255.0)
    loss = seg_ce + seg_kl_weight * seg_kl + 0.5 * margin + pose_weight * pose_mse + anchor_weight * anchor
    return loss, {
        "seg_ce": float(seg_ce.detach().cpu()),
        "seg_kl": float(seg_kl.detach().cpu()),
        "margin": float(margin.detach().cpu()),
        "pose_mse": float(pose_mse.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    masks: torch.Tensor,
    poses: torch.Tensor,
    sample_ids: list[int],
    original: torch.Tensor,
    distortion: DistortionNet,
    device: torch.device,
    *,
    batch_size: int = 2,
) -> dict[str, Any]:
    rows = []
    for start in range(0, len(sample_ids), batch_size):
        ids = sample_ids[start : start + batch_size]
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
        "sample60": next((row for row in rows if row["sample_id"] == 60), None),
        "per_sample": rows,
    }


def set_train_scope(model: torch.nn.Module, scope: str) -> int:
    for name, param in model.named_parameters():
        if scope == "all":
            trainable = True
        elif scope == "heads":
            trainable = name.startswith("frame1_head.") or name.startswith("frame2_head.") or name.startswith("pose_mlp.")
        elif scope == "frame2":
            trainable = name.startswith("frame2_head.")
        elif scope == "frame1":
            trainable = name.startswith("frame1_head.") or name.startswith("pose_mlp.")
        elif scope == "pose_mlp":
            trainable = name.startswith("pose_mlp.")
        else:
            raise ValueError(f"unknown train scope: {scope}")
        param.requires_grad_(trainable)
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)


def train(
    *,
    subset: str,
    device_name: str,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    lr: float,
    pose_weight: float,
    anchor_weight: float,
    anchor_kind: str,
    train_scope: str,
    seed: int,
    out: Path,
) -> dict[str, Any]:
    _patch_allnorm_for_grad()
    if max_steps <= 50:
        torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)
    device = select_torch_device(device_name)
    sample_ids = get_subset(subset)
    base_model, masks, poses = load_fp4_artifact(device)
    anchor_frames = internal_teacher_fullres(anchor_kind, base_model, masks, poses, sample_ids, device)
    model = copy.deepcopy(base_model).to(device).train()
    trainable_params_count = set_train_scope(model, train_scope)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"no trainable parameters for scope {train_scope}")

    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for param in distortion.parameters():
        param.requires_grad_(False)
    targets = build_targets(sample_ids, device)
    base_metrics = evaluate(base_model, masks, poses, sample_ids, targets["original"], distortion, device)
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-7)

    out.mkdir(parents=True, exist_ok=True)
    best_checkpoint = out / f"{subset}_{train_scope}_{anchor_kind}_best.pt"
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start_time = time.time()
    local_index = torch.arange(len(sample_ids), device=device)

    for step in range(1, max_steps + 1):
        perm = local_index[torch.randperm(len(sample_ids), device=device)[:batch_size]]
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
            seg_kl_weight=0.25,
            anchor_weight=anchor_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            model.eval()
            metrics = evaluate(model, masks, poses, sample_ids, targets["original"], distortion, device)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                **parts,
                **metrics,
                "quality_delta_vs_base": metrics["quality"] - base_metrics["quality"],
                "score_at_fp4_bytes": metrics["quality"] + rate_term(ARCHIVE_BYTES),
                "elapsed_sec": time.time() - start_time,
            }
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ["step", "quality", "quality_delta_vs_base", "score_at_fp4_bytes", "loss"]},
                    sort_keys=True,
                ),
                flush=True,
            )
            if best is None or row["quality"] < best["quality"]:
                best = row
                torch.save(
                    {
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "row": row,
                        "base": base_metrics,
                        "subset": subset,
                        "train_scope": train_scope,
                        "anchor_kind": anchor_kind,
                        "seed": seed,
                    },
                    best_checkpoint,
                )
            model.train()

    assert best is not None
    summary = {
        "subset": subset,
        "device": str(device),
        "archive_bytes": ARCHIVE_BYTES,
        "base": base_metrics,
        "best": best,
        "rows": rows,
        "lr": lr,
        "pose_weight": pose_weight,
        "anchor_weight": anchor_weight,
        "anchor_kind": anchor_kind,
        "train_scope": train_scope,
        "seed": seed,
        "trainable_params": trainable_params_count,
        "best_checkpoint": str(best_checkpoint),
        "target_note": "At 249624 bytes, visible <0.300 requires quality <= about 0.1338.",
    }
    write_json(out / f"{subset}_{train_scope}_{anchor_kind}_summary.json", summary)
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
    parser.add_argument("--pose-weight", type=float, default=50.0)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--anchor-kind", choices=["fp4", "qpose"], default="fp4")
    parser.add_argument("--train-scope", choices=["all", "heads", "frame2", "frame1", "pose_mlp"], default="heads")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    train(
        subset=args.subset,
        device_name=args.device,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        lr=args.lr,
        pose_weight=args.pose_weight,
        anchor_weight=args.anchor_weight,
        anchor_kind=args.anchor_kind,
        train_scope=args.train_scope,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
