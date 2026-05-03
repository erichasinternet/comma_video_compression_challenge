#!/usr/bin/env python
"""Deep-compression diagnostics for the exact-mask Quantizr #55 model.

The first mode is a layer/block sensitivity audit. It keeps the exact mask and
pose payloads fixed, applies one compression perturbation to one block at a
time, and evaluates evaluator-quality deltas in memory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import brotli
import torch
import torch.nn as nn
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import inflate as q55_inflate
from pack_model import build_qpack
from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    ORIGINAL_BYTES,
    POSE_PAYLOAD,
    POSE_QPACK_PAYLOAD,
    append_jsonl,
    make_archive_zip,
    score_from_bytes,
    sha256_file,
    summarize_archive,
    unzip_archive,
    write_json,
)
from q55_pixel_oracle import build_targets, load_evaluators, render_for_eval
from q55_pose_control_oracle import compute_metrics_eval, load_rgb_indices


OUT_H = 874
OUT_W = 1164
DEFAULT_GROUPS = (
    "shared_trunk",
    "frame1_head",
    "frame2_head",
    "pose_mlp",
    "pointwise",
    "depthwise",
    "film_proj",
    "rgb_heads",
)


@dataclass(frozen=True)
class ParamRef:
    module_name: str
    param_name: str
    tensor: torch.Tensor

    @property
    def full_name(self) -> str:
        return f"{self.module_name}.{self.param_name}" if self.module_name else self.param_name

    @property
    def numel(self) -> int:
        return int(self.tensor.numel())


@dataclass(frozen=True)
class TransformResult:
    estimated_bytes: int
    extra: dict


def parse_indices(args: argparse.Namespace) -> list[int]:
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        indices = list(range(args.offset, args.offset + args.max_samples))
    if not indices:
        raise ValueError("no sample indices selected")
    return indices


def unpack_archive(base_archive: Path) -> tuple[Path, Path]:
    tmp_path = Path(tempfile.mkdtemp(prefix="q55_deep_compress_"))
    archive_dir = tmp_path / "archive"
    unzip_archive(base_archive, archive_dir)
    return tmp_path, archive_dir


def load_generator(archive_dir: Path, device: torch.device) -> q55_inflate.JointFrameGenerator:
    generator = q55_inflate.JointFrameGenerator(**q55_inflate.load_arch_config(archive_dir)).to(device)
    model_qpack = archive_dir / MODEL_QPACK_PAYLOAD
    if model_qpack.exists():
        state = q55_inflate.get_qpack_state_dict(brotli.decompress(model_qpack.read_bytes()), device)
    else:
        state = q55_inflate.get_decoded_state_dict(brotli.decompress((archive_dir / MODEL_PAYLOAD).read_bytes()), device)
    generator.load_state_dict(state, strict=True)
    generator.shared_trunk.mask_adapter = q55_inflate.load_mask_adapter(archive_dir, device)
    generator.eval()
    return generator


def load_mask_pose_subset(archive_dir: Path, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.tensor(indices, dtype=torch.long)
    mask_all = q55_inflate.load_mask_payload(archive_dir, archive_dir / MASK_PAYLOAD).contiguous()
    pose_all = q55_inflate.load_pose_payload(archive_dir, archive_dir / POSE_PAYLOAD).contiguous()
    return mask_all.index_select(0, idx), pose_all.index_select(0, idx)


@torch.inference_mode()
def generate_eval_frames(
    generator: q55_inflate.JointFrameGenerator,
    masks: torch.Tensor,
    poses: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    lows = []
    for start in range(0, masks.shape[0], batch_size):
        m = masks[start : start + batch_size].to(device).long()
        p = poses[start : start + batch_size].to(device).float()
        fake1, fake2 = generator(m, p)
        lows.append(torch.stack([fake1, fake2], dim=1).detach().cpu())
    low = torch.cat(lows, dim=0).contiguous()
    return render_for_eval(low.to(device).float(), camera_sim=True).cpu()


@torch.inference_mode()
def evaluate_generator(
    generator: q55_inflate.JointFrameGenerator,
    masks: torch.Tensor,
    poses: torch.Tensor,
    seg_targets: torch.Tensor,
    pose_targets: torch.Tensor,
    segnet,
    posenet,
    device: torch.device,
    args: argparse.Namespace,
    archive_bytes: int,
    sample_indices: list[int],
) -> dict:
    frames = generate_eval_frames(generator, masks, poses, device, args.gen_batch_size)
    return compute_metrics_eval(
        frames,
        seg_targets,
        pose_targets,
        segnet,
        posenet,
        device,
        batch_size=args.eval_batch_size,
        archive_bytes=archive_bytes,
        sample_indices=sample_indices,
        include_per_sample=args.save_per_sample_metrics,
        top_k=args.tail_top_k,
    )


def is_compressible_module(module: nn.Module, *, include_norm: bool) -> bool:
    if not hasattr(module, "weight") or getattr(module, "weight") is None:
        return False
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        return True
    if include_norm and isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
        return True
    return False


def collect_params(model: nn.Module, *, include_norm: bool, include_bias: bool) -> list[ParamRef]:
    refs: list[ParamRef] = []
    for name, module in model.named_modules():
        if not is_compressible_module(module, include_norm=include_norm):
            continue
        refs.append(ParamRef(name, "weight", module.weight))
        if include_bias and getattr(module, "bias", None) is not None:
            refs.append(ParamRef(name, "bias", module.bias))
    return refs


def group_predicate(group: str) -> Callable[[ParamRef], bool]:
    if group == "all":
        return lambda ref: True
    if group == "shared_trunk":
        return lambda ref: ref.module_name.startswith("shared_trunk.")
    if group == "frame1_head":
        return lambda ref: ref.module_name.startswith("frame1_head.")
    if group == "frame2_head":
        return lambda ref: ref.module_name.startswith("frame2_head.")
    if group == "pose_mlp":
        return lambda ref: ref.module_name.startswith("pose_mlp.")
    if group == "pointwise":
        return lambda ref: ref.module_name.endswith(".pw")
    if group == "depthwise":
        return lambda ref: ref.module_name.endswith(".dw")
    if group == "film_proj":
        return lambda ref: ref.module_name.endswith(".film_proj")
    if group == "rgb_heads":
        return lambda ref: ref.module_name in {"frame1_head.head", "frame2_head.head"}
    if group == "embedding":
        return lambda ref: ref.module_name == "shared_trunk.embedding"
    raise ValueError(f"unknown group: {group}")


def select_param_sets(refs: list[ParamRef], args: argparse.Namespace) -> list[tuple[str, list[ParamRef]]]:
    if args.scope == "groups":
        groups = [x.strip() for x in args.groups.split(",") if x.strip()] if args.groups else list(DEFAULT_GROUPS)
        out = []
        for group in groups:
            pred = group_predicate(group)
            selected = [ref for ref in refs if pred(ref)]
            if selected:
                out.append((group, selected))
        return out

    sorted_refs = sorted(refs, key=lambda ref: ref.numel, reverse=True)
    if args.top_layers > 0:
        sorted_refs = sorted_refs[: args.top_layers]
    return [(ref.full_name, [ref]) for ref in sorted_refs]


def quantize_uniform_(tensor: torch.Tensor, bits: int) -> TransformResult:
    data = tensor.data
    max_abs = data.abs().max()
    if float(max_abs) <= 1e-12:
        data.zero_()
        return TransformResult(estimated_bytes=4, extra={"scale": 0.0})
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / qmax
    tensor.data.copy_(torch.clamp(torch.round(data / scale), -qmax, qmax) * scale)
    estimated = math.ceil(tensor.numel() * bits / 8.0) + 4
    return TransformResult(estimated_bytes=estimated, extra={"bits": bits, "scale": float(scale.detach().cpu())})


def prune_magnitude_(tensor: torch.Tensor, sparsity: float) -> TransformResult:
    data = tensor.data
    n = data.numel()
    keep = max(1, int(round(n * (1.0 - sparsity))))
    flat_abs = data.detach().abs().reshape(-1)
    if keep >= n:
        return TransformResult(estimated_bytes=n * 2, extra={"sparsity": 0.0, "nonzero": n})
    threshold = torch.topk(flat_abs, keep, largest=True).values.min()
    mask = data.abs() >= threshold
    tensor.data.mul_(mask)
    nonzero = int(mask.sum().item())
    # Crude lower-bound estimate: one bit mask plus fp16 nonzero values.
    estimated = math.ceil(n / 8.0) + nonzero * 2
    return TransformResult(
        estimated_bytes=estimated,
        extra={"sparsity": float(1.0 - nonzero / max(1, n)), "nonzero": nonzero},
    )


def codebook_quantize_(tensor: torch.Tensor, k: int, iters: int = 10) -> TransformResult:
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == 0:
        return TransformResult(estimated_bytes=0, extra={"k": k})
    unique = torch.unique(flat)
    k_eff = min(k, int(unique.numel()))
    if k_eff <= 1:
        tensor.data.fill_(float(flat.mean().item()))
        return TransformResult(estimated_bytes=2, extra={"k": k_eff})
    qs = torch.linspace(0.0, 1.0, k_eff, device=flat.device)
    centroids = torch.quantile(flat, qs).contiguous()
    for _ in range(iters):
        dist = (flat[:, None] - centroids[None, :]).abs()
        idx = dist.argmin(dim=1)
        new = centroids.clone()
        for j in range(k_eff):
            mask = idx == j
            if bool(mask.any()):
                new[j] = flat[mask].mean()
        if torch.allclose(new, centroids, rtol=0.0, atol=1e-6):
            centroids = new
            break
        centroids = new
    idx = (flat[:, None] - centroids[None, :]).abs().argmin(dim=1)
    tensor.data.copy_(centroids[idx].reshape_as(tensor).to(dtype=tensor.dtype))
    bits = max(1, math.ceil(math.log2(k_eff)))
    estimated = math.ceil(tensor.numel() * bits / 8.0) + k_eff * 2
    return TransformResult(estimated_bytes=estimated, extra={"k": k_eff, "index_bits": bits})


def apply_transform(refs: list[ParamRef], variant: str) -> tuple[dict[str, torch.Tensor], dict]:
    originals = {ref.full_name: ref.tensor.detach().clone() for ref in refs}
    estimated_bytes = 0
    per_param = []
    for ref in refs:
        if variant.startswith("int"):
            bits = int(variant[3:])
            result = quantize_uniform_(ref.tensor, bits)
        elif variant.startswith("prune"):
            sparsity = float(variant[5:]) / 100.0
            result = prune_magnitude_(ref.tensor, sparsity)
        elif variant.startswith("codebook"):
            k = int(variant[8:])
            result = codebook_quantize_(ref.tensor, k)
        elif variant == "ternary":
            result = codebook_quantize_(ref.tensor, 3)
        else:
            raise ValueError(f"unknown variant: {variant}")
        estimated_bytes += result.estimated_bytes
        per_param.append(
            {
                "name": ref.full_name,
                "numel": ref.numel,
                "estimated_bytes": result.estimated_bytes,
                **result.extra,
            }
        )
    return originals, {"estimated_group_bytes": estimated_bytes, "params": per_param}


def restore_transform(refs: list[ParamRef], originals: dict[str, torch.Tensor]) -> None:
    for ref in refs:
        ref.tensor.data.copy_(originals[ref.full_name].to(device=ref.tensor.device, dtype=ref.tensor.dtype))


def quality_delta(candidate: dict, baseline: dict) -> dict:
    return {
        "delta_quality": candidate["quality_term"] - baseline["quality_term"],
        "delta_posenet": candidate["posenet_dist"] - baseline["posenet_dist"],
        "delta_segnet": candidate["segnet_dist"] - baseline["segnet_dist"],
        "delta_score_at_same_archive": candidate["projected_score_at_archive_bytes"]
        - baseline["projected_score_at_archive_bytes"],
    }


def parse_variants(text: str) -> list[str]:
    variants = [x.strip() for x in text.split(",") if x.strip()]
    if not variants:
        raise ValueError("no variants selected")
    for variant in variants:
        if variant.startswith("int"):
            bits = int(variant[3:])
            if bits < 2 or bits > 16:
                raise ValueError(f"bad int variant: {variant}")
        elif variant.startswith("prune"):
            pct = int(variant[5:])
            if pct <= 0 or pct >= 100:
                raise ValueError(f"bad prune variant: {variant}")
        elif variant.startswith("codebook"):
            k = int(variant[8:])
            if k < 2:
                raise ValueError(f"bad codebook variant: {variant}")
        elif variant != "ternary":
            raise ValueError(f"unknown variant: {variant}")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--indices", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--scope", choices=["groups", "layers"], default="groups")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--top-layers", type=int, default=24)
    parser.add_argument("--variants", default="int8,int6,int4,codebook16,prune50")
    parser.add_argument("--include-norm", action="store_true")
    parser.add_argument("--include-bias", action="store_true")
    parser.add_argument("--save-per-sample-metrics", action="store_true")
    parser.add_argument("--tail-top-k", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_archive = args.base_archive.resolve()
    if not base_archive.exists():
        raise FileNotFoundError(base_archive)
    device = torch.device(args.device)
    variants = parse_variants(args.variants)
    indices = parse_indices(args)
    label = args.label or f"q55_deep_compress_audit_{args.scope}_{len(indices)}s"
    run_dir = args.out_dir / label
    if run_dir.exists() and not args.dry_run:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        print("Loading evaluator models...", flush=True)
        segnet, posenet = load_evaluators(device)

        print(f"Loading RGB targets for {len(indices)} samples...", flush=True)
        gt_pairs = load_rgb_indices(
            args.video_names,
            args.video_dir,
            indices=indices,
            decode_batch_size=args.decode_batch_size,
        )
        seg_targets, pose_targets = build_targets(gt_pairs, segnet, posenet, device, args.target_batch_size)

        print("Loading #55 exact-mask archive...", flush=True)
        tmp_path, archive_dir = unpack_archive(base_archive)
        generator = load_generator(archive_dir, device)
        masks, poses = load_mask_pose_subset(archive_dir, indices)
        refs = collect_params(generator, include_norm=args.include_norm, include_bias=args.include_bias)
        param_sets = select_param_sets(refs, args)
        if args.dry_run:
            print(json.dumps(
                [
                    {"name": name, "params": len(params), "numel": sum(p.numel for p in params)}
                    for name, params in param_sets
                ],
                indent=2,
            ))
            return

        print("Evaluating baseline...", flush=True)
        baseline = evaluate_generator(
            generator,
            masks,
            poses,
            seg_targets,
            pose_targets,
            segnet,
            posenet,
            device,
            args,
            base_archive.stat().st_size,
            indices,
        )
        print("Baseline:", json.dumps(baseline, indent=2), flush=True)

        rows = []
        for set_name, selected_refs in tqdm(param_sets, desc="audit groups"):
            for variant in variants:
                originals, transform_report = apply_transform(selected_refs, variant)
                try:
                    metrics = evaluate_generator(
                        generator,
                        masks,
                        poses,
                        seg_targets,
                        pose_targets,
                        segnet,
                        posenet,
                        device,
                        args,
                        base_archive.stat().st_size,
                        indices,
                    )
                finally:
                    restore_transform(selected_refs, originals)
                row = {
                    "set": set_name,
                    "variant": variant,
                    "param_count": len(selected_refs),
                    "weight_numel": int(sum(ref.numel for ref in selected_refs)),
                    "metrics": metrics,
                    "delta": quality_delta(metrics, baseline),
                    "transform": transform_report,
                }
                rows.append(row)
                append_jsonl(run_dir / "audit_rows.jsonl", row)
                print(
                    f"{set_name:24s} {variant:10s} "
                    f"q={metrics['quality_term']:.6f} "
                    f"dq={row['delta']['delta_quality']:+.6f} "
                    f"pose={metrics['posenet_dist']:.6g} seg={metrics['segnet_dist']:.6g}",
                    flush=True,
                )

        sorted_rows = sorted(rows, key=lambda r: (r["delta"]["delta_quality"], r["weight_numel"]))
        record = {
            "label": label,
            "base_archive": str(base_archive),
            "base_archive_sha256": sha256_file(base_archive),
            "base_archive_summary": summarize_archive(base_archive),
            "device": str(device),
            "indices": indices,
            "sample_count": len(indices),
            "scope": args.scope,
            "variants": variants,
            "baseline": baseline,
            "rows": rows,
            "best_by_quality_delta": sorted_rows[:20],
            "worst_by_quality_delta": sorted(rows, key=lambda r: r["delta"]["delta_quality"], reverse=True)[:20],
            "decision_hints": {
                "continue_if_some_int4_or_codebook16_delta_quality_lt_0_010": any(
                    r["variant"] in {"int4", "codebook16"} and r["delta"]["delta_quality"] < 0.010
                    for r in rows
                ),
                "continue_if_some_prune50_delta_quality_lt_0_010": any(
                    r["variant"] == "prune50" and r["delta"]["delta_quality"] < 0.010 for r in rows
                ),
                "stop_if_all_large_groups_int8_or_prune50_damage_pose": all(
                    r["delta"]["delta_quality"] > 0.010
                    for r in rows
                    if r["weight_numel"] >= 2_000 and r["variant"] in {"int8", "prune50"}
                ),
            },
        }
        write_json(run_dir / "metrics.json", record)
        append_jsonl(args.out_dir / "deep_compress_audit_results.jsonl", record)
    finally:
        if tmp_path is not None:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    main()
