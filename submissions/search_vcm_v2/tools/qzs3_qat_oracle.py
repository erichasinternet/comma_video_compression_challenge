#!/usr/bin/env python
"""QAT feasibility oracle for PR #67 qpose14_qzs3.

This is intentionally not a packer. It trains the existing qpose14_qzs3
generator with straight-through fake quantization, then evaluates an actually
quantized copy on hard subsets. The question is whether the current working
architecture can survive the lower precision needed to save ~20KB.
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
    evaluate_model,
    estimate_model_raw_bytes,
    load_generator,
    quantize_model,
)


OUT_DIR = EXPERIMENTS_DIR / "qzs3_qat_oracle"


@dataclass(frozen=True)
class QuantPolicy:
    name: str
    conv_bits: int
    qv_bits: int | None = None
    dense_bits: int | None = None
    scope: str = "all"


POLICIES = {
    "q3_all_qv8_dense8": QuantPolicy("q3_all_qv8_dense8", conv_bits=3, qv_bits=8, dense_bits=8),
    "q2_all_qv8_dense8": QuantPolicy("q2_all_qv8_dense8", conv_bits=2, qv_bits=8, dense_bits=8),
    "q2_conv_qv10_dense16": QuantPolicy("q2_conv_qv10_dense16", conv_bits=2, qv_bits=None, dense_bits=None),
}


def fake_quant_uniform_ste(tensor: torch.Tensor, *, bits: int, block_size: int = 32) -> torch.Tensor:
    if bits >= 16:
        return tensor
    flat = tensor.float().reshape(-1)
    n = int(flat.numel())
    blocks = (n + block_size - 1) // block_size
    padded = F.pad(flat, (0, blocks * block_size - n)).reshape(blocks, block_size)
    qmax = float((1 << (bits - 1)) - 1)
    scale = padded.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(padded / scale).clamp(-qmax, qmax)
    deq = (q * scale).reshape(-1)[:n].reshape_as(tensor).to(dtype=tensor.dtype)
    return tensor + (deq - tensor).detach()


def annotate_names(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        module._qat_name = name  # type: ignore[attr-defined]


def should_quantize_conv(name: str, policy: QuantPolicy) -> bool:
    if policy.scope == "all":
        return True
    if policy.scope == "frame1":
        return name.startswith("frame1_head")
    if policy.scope == "conv":
        return True
    return False


def install_fake_quant_forwards(model: torch.nn.Module, policy: QuantPolicy) -> None:
    """Patch instance forwards so training sees fake-quantized weights."""

    for module in model.modules():
        name = getattr(module, "_qat_name", "")
        if isinstance(module, qzs3.QConv2d):
            original_forward = module.forward

            def qconv_forward(x, module=module, name=name, original_forward=original_forward):
                weight = module.weight
                if module.quantize_weight and should_quantize_conv(name, policy):
                    weight = fake_quant_uniform_ste(weight, bits=policy.conv_bits, block_size=module.block_size)
                return F.conv2d(x, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)

            module.forward = qconv_forward  # type: ignore[method-assign]
            module._qat_original_forward = original_forward  # type: ignore[attr-defined]
        elif isinstance(module, qzs3.QEmbedding):
            original_forward = module.forward

            def qemb_forward(x, module=module, name=name, original_forward=original_forward):
                weight = module.weight
                if module.quantize_weight and should_quantize_conv(name, policy):
                    weight = fake_quant_uniform_ste(weight, bits=policy.conv_bits, block_size=module.block_size)
                return F.embedding(
                    x,
                    weight,
                    module.padding_idx,
                    module.max_norm,
                    module.norm_type,
                    module.scale_grad_by_freq,
                    module.sparse,
                )

            module.forward = qemb_forward  # type: ignore[method-assign]
            module._qat_original_forward = original_forward  # type: ignore[attr-defined]
        elif isinstance(module, torch.nn.Linear):
            if name == "frame1_head.block1.film_proj" and policy.qv_bits is not None:
                bits = policy.qv_bits
            elif name == "pose_mlp.2" and policy.qv_bits is not None:
                bits = policy.qv_bits
            elif policy.dense_bits is not None and module.weight.numel() >= 16:
                bits = policy.dense_bits
            else:
                continue
            original_forward = module.forward

            def qlinear_forward(x, module=module, bits=bits, original_forward=original_forward):
                weight = fake_quant_uniform_ste(module.weight, bits=bits)
                bias = module.bias
                if bias is not None and bits <= 8 and bias.numel() >= 16:
                    bias = fake_quant_uniform_ste(bias, bits=bits)
                return F.linear(x, weight, bias)

            module.forward = qlinear_forward  # type: ignore[method-assign]
            module._qat_original_forward = original_forward  # type: ignore[attr-defined]


def render_fullres(model: torch.nn.Module, mask: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    frame1, frame2 = model(mask.long(), pose.float())
    frame1 = F.interpolate(frame1, size=(874, 1164), mode="bilinear", align_corners=False)
    frame2 = F.interpolate(frame2, size=(874, 1164), mode="bilinear", align_corners=False)
    pred = torch.stack([frame1, frame2], dim=1).clamp(0, 255)
    return einops.rearrange(pred, "b t c h w -> b t h w c")


@torch.inference_mode()
def build_targets(sample_ids: list[int], device: torch.device) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    original = load_original_subset("qat_" + "_".join(map(str, sample_ids)), sample_ids, device="cpu").float().to(device)
    posenet_in, segnet_in = distortion.preprocess_input(original)
    pose_target, seg_target = distortion.posenet(posenet_in), distortion.segnet(segnet_in)
    return {
        "original": original,
        "pose": {k: v.detach() for k, v in pose_target.items()},
        "seg_logits": seg_target.detach(),
        "seg_argmax": seg_target.argmax(dim=1).detach(),
    }


def task_loss(
    distortion: DistortionNet,
    pred: torch.Tensor,
    targets: dict[str, Any],
    qpose_pred: torch.Tensor | None,
    *,
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
    # Pixelwise KL is huge at image resolution; keep it as a weak stabilizer.
    loss = seg_ce + 1e-5 * seg_kl + 10.0 * pose_mse
    anchor = torch.zeros((), device=pred.device)
    if qpose_pred is not None and anchor_weight > 0:
        anchor = F.smooth_l1_loss(pred.contiguous() / 255.0, qpose_pred.contiguous() / 255.0)
        loss = loss + anchor_weight * anchor
    return loss, {
        "seg_ce": float(seg_ce.detach().cpu()),
        "seg_kl": float(seg_kl.detach().cpu()),
        "pose_mse": float(pose_mse.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
    }


def frame_imitation_loss(pred: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    pred_n = pred.contiguous() / 255.0
    teacher_n = teacher.contiguous() / 255.0
    huber = F.smooth_l1_loss(pred_n, teacher_n)
    pred_low = F.avg_pool2d(einops.rearrange(pred_n, "b t h w c -> (b t) c h w"), kernel_size=8, stride=8)
    teacher_low = F.avg_pool2d(einops.rearrange(teacher_n, "b t h w c -> (b t) c h w"), kernel_size=8, stride=8)
    low = F.smooth_l1_loss(pred_low.contiguous(), teacher_low.contiguous())
    loss = 10.0 * huber + 2.0 * low
    return loss, {"frame_huber": float(huber.detach().cpu()), "frame_low": float(low.detach().cpu())}


def qpose_fullres(base_model: torch.nn.Module, masks: torch.Tensor, poses: torch.Tensor, sample_ids: list[int], device: torch.device) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for sid in sample_ids:
            chunks.append(render_fullres(base_model, masks[sid : sid + 1].to(device), poses[sid : sid + 1].to(device)).detach())
    return torch.cat(chunks, dim=0)


def train_qat(
    *,
    policy: QuantPolicy,
    subset_name: str,
    device_name: str,
    max_steps: int,
    eval_every: int,
    warmup_steps: int,
    batch_size: int,
    lr: float,
    out: Path,
) -> dict[str, Any]:
    device = select_torch_device(device_name)
    sample_ids = get_subset(subset_name)
    base_model, masks, poses = load_generator(device)
    base_metrics = evaluate_model(base_model, masks, poses, subset_name=subset_name, device=device)
    qpose_teacher = qpose_fullres(base_model, masks, poses, sample_ids, device)

    model = copy.deepcopy(base_model).to(device).train()
    annotate_names(model)
    install_fake_quant_forwards(model, policy)
    for param in model.parameters():
        param.requires_grad_(True)

    targets = build_targets(sample_ids, device)
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for param in distortion.parameters():
        param.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    index = torch.arange(len(sample_ids), device=device)
    log_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start_time = time.time()

    for step in range(1, max_steps + 1):
        if step <= warmup_steps:
            anchor_weight = 0.0
        elif step < max_steps * 0.5:
            anchor_weight = 2.0
        elif step < max_steps * 0.8:
            anchor_weight = 0.25
        else:
            anchor_weight = 0.05
        perm = index[torch.randperm(len(sample_ids), device=device)[:batch_size]]
        ids = [sample_ids[int(i)] for i in perm.detach().cpu().tolist()]
        local = perm.detach().cpu().tolist()
        mask = masks[ids].to(device)
        pose = poses[ids].to(device)
        pred = render_fullres(model, mask, pose)
        batch_targets = {
            "seg_argmax": targets["seg_argmax"][local],
            "seg_logits": targets["seg_logits"][local],
            "pose": {k: v[local] for k, v in targets["pose"].items()},
        }
        if step <= warmup_steps:
            loss, parts = frame_imitation_loss(pred, qpose_teacher[local])
        else:
            loss, parts = task_loss(
                distortion,
                pred,
                batch_targets,
                qpose_teacher[local],
                anchor_weight=anchor_weight,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            quantized = quantize_model(
                model,
                conv_bits=policy.conv_bits,
                qv_bits=policy.qv_bits,
                dense_bits=policy.dense_bits,
                scope=policy.scope,
            ).to(device).eval()
            metrics = evaluate_model(quantized, masks, poses, subset_name=subset_name, device=device)
            sizes = estimate_model_raw_bytes(
                conv_bits=policy.conv_bits,
                qv_bits=policy.qv_bits,
                dense_bits=policy.dense_bits,
                scope=policy.scope,
            )
            row = {
                "step": step,
                "policy": policy.name,
                "loss": float(loss.detach().cpu()),
                **parts,
                **metrics,
                "quality_delta_vs_base": metrics["quality"] - base_metrics["quality"],
                "estimated_model_raw_bytes": sizes["total_raw_plus_header"],
                "estimated_raw_savings": 59_288 - sizes["total_raw_plus_header"],
                "elapsed_sec": time.time() - start_time,
            }
            log_rows.append(row)
            print(json.dumps({k: row[k] for k in ["step", "policy", "quality", "quality_delta_vs_base", "estimated_raw_savings", "loss"]}, sort_keys=True), flush=True)
            if best is None or row["quality"] < best["quality"]:
                best = row
            quantized.train(False)
            model.train(True)

    assert best is not None
    summary = {
        "subset": subset_name,
        "device": str(device),
        "policy": policy.__dict__,
        "base": base_metrics,
        "best": best,
        "rows": log_rows,
        "score_target_note": "Need ~22KB real savings at PR67 quality, or comparable quality improvement.",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{subset_name}_{policy.name}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="hard8")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="q2_all_qv8_dense8")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    summary = train_qat(
        policy=POLICIES[args.policy],
        subset_name=args.subset,
        device_name=args.device,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
