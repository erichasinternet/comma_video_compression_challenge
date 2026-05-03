#!/usr/bin/env python
"""Zero-payload postprocess sweep for the current qzs3 range-mask candidate.

This intentionally tests only deterministic transforms with a tiny fixed
description. If one wins, it can be implemented in inflate.py without hidden
side channels and then validated through the official evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import einops
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import TensorVideoDataset  # noqa: E402
from frame_utils import AVVideoDataset  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json  # noqa: E402
from submissions.search_vcm_v2.subsets import get_subset  # noqa: E402


OUT_DIR = EXPERIMENTS_DIR / "qzs3_postprocess_sweep"
DEFAULT_SUBMISSION = REPO_ROOT / "submissions/qzs3_range_mask_candidate"
ORIGINAL_BYTES = 37_545_489


@dataclass(frozen=True)
class Variant:
    name: str
    frame_scope: str
    channel: int | None
    bias: float = 0.0
    scale: float = 1.0
    gamma: float = 1.0
    op: str = "color"
    dx: int = 0
    dy: int = 0
    amount: float = 0.0

    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        out = batch.float()
        if self.op == "temporal_blend":
            out = out.clone()
            if self.frame_scope == "f1_from_f2":
                out[:, 0] = out[:, 0] * (1.0 - self.amount) + out[:, 1] * self.amount
            elif self.frame_scope == "f2_from_f1":
                out[:, 1] = out[:, 1] * (1.0 - self.amount) + out[:, 0] * self.amount
            else:
                raise ValueError(f"unknown temporal blend scope: {self.frame_scope}")
            return out.clamp(0.0, 255.0).round()

        if self.frame_scope == "f1":
            target = out[:, 0:1]
        elif self.frame_scope == "f2":
            target = out[:, 1:2]
        elif self.frame_scope == "both":
            target = out
        else:
            raise ValueError(f"unknown frame scope: {self.frame_scope}")

        if self.op in ("shift", "blur", "sharpen", "sat"):
            target = target.clone()
            flat = einops.rearrange(target, "b t h w c -> (b t) c h w")
            if self.op == "shift":
                top = max(self.dy, 0)
                bottom = max(-self.dy, 0)
                left = max(self.dx, 0)
                right = max(-self.dx, 0)
                padded = F.pad(flat, (left, right, top, bottom), mode="replicate")
                h, w = flat.shape[-2:]
                y0 = bottom
                x0 = right
                flat = padded[..., y0 : y0 + h, x0 : x0 + w]
            elif self.op in ("blur", "sharpen"):
                blur = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1)
                if self.op == "blur":
                    flat = flat * (1.0 - self.amount) + blur * self.amount
                else:
                    flat = flat + (flat - blur) * self.amount
            elif self.op == "sat":
                r, g, b = flat[:, 0:1], flat[:, 1:2], flat[:, 2:3]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                flat = gray + (flat - gray) * self.amount
            target = einops.rearrange(flat, "(b t) c h w -> b t h w c", b=target.shape[0], t=target.shape[1])
            if self.frame_scope == "f1":
                out = out.clone()
                out[:, 0:1] = target
            elif self.frame_scope == "f2":
                out = out.clone()
                out[:, 1:2] = target
            else:
                out = target
            return out.clamp(0.0, 255.0).round()

        if self.channel is None:
            view = target
        else:
            view = target[..., self.channel : self.channel + 1]

        x = view
        if self.scale != 1.0:
            x = (x - 128.0) * self.scale + 128.0
        if self.gamma != 1.0:
            x = torch.pow((x / 255.0).clamp(0.0, 1.0), self.gamma) * 255.0
        if self.bias:
            x = x + self.bias
        x = x.clamp(0.0, 255.0).round()

        if self.channel is None:
            target = x
        else:
            target = target.clone()
            target[..., self.channel : self.channel + 1] = x

        if self.frame_scope == "f2":
            out = out.clone()
            out[:, 1:2] = target
        else:
            out = target
        return out


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _subset_ids(name: str) -> set[int] | None:
    if name == "full600":
        return None
    return set(int(x) for x in get_subset(name))


def make_variants(mode: str) -> list[Variant]:
    variants = [Variant("identity", "both", None)]
    if mode in ("small", "wide", "advanced"):
        bias_values = [-4, -3, -2, -1, 1, 2, 3, 4] if mode == "wide" else [-2, -1, 1, 2]
        scale_values = [0.97, 0.98, 0.99, 1.01, 1.02, 1.03] if mode == "wide" else [0.99, 1.01]
        gamma_values = [0.96, 0.98, 1.02, 1.04] if mode == "wide" else [0.98, 1.02]
    else:
        raise ValueError(f"unknown mode: {mode}")

    for scope in ("f2", "both"):
        for b in bias_values:
            variants.append(Variant(f"{scope}_bias_all_{b:+g}", scope, None, bias=float(b)))
        for s in scale_values:
            variants.append(Variant(f"{scope}_scale_all_{s:g}", scope, None, scale=float(s)))
        for g in gamma_values:
            variants.append(Variant(f"{scope}_gamma_all_{g:g}", scope, None, gamma=float(g)))

    for scope in ("f2", "both"):
        for channel, cname in enumerate(("r", "g", "b")):
            for b in bias_values:
                variants.append(Variant(f"{scope}_bias_{cname}_{b:+g}", scope, channel, bias=float(b)))

    if mode == "advanced":
        for scope in ("f1", "f2", "both"):
            for dy, dx in ((-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                variants.append(Variant(f"{scope}_shift_y{dy:+d}_x{dx:+d}", scope, None, op="shift", dy=dy, dx=dx))
            for amount in (0.25, 0.50, 0.75):
                variants.append(Variant(f"{scope}_blur_{amount:g}", scope, None, op="blur", amount=amount))
            for amount in (0.25, 0.50, 0.75, 1.00):
                variants.append(Variant(f"{scope}_sharpen_{amount:g}", scope, None, op="sharpen", amount=amount))
            for amount in (0.80, 0.90, 1.10, 1.20):
                variants.append(Variant(f"{scope}_sat_{amount:g}", scope, None, op="sat", amount=amount))

        for amount in (0.05, 0.10, 0.15, 0.20):
            variants.append(Variant(f"f1_from_f2_{amount:g}", "f1_from_f2", None, op="temporal_blend", amount=amount))
            variants.append(Variant(f"f2_from_f1_{amount:g}", "f2_from_f1", None, op="temporal_blend", amount=amount))

    return variants


def filter_variants(variants: list[Variant], names: str | None) -> list[Variant]:
    if not names:
        return variants
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {variant.name: variant for variant in variants}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"unknown variant(s): {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def _iter_selected(
    *,
    submission_dir: Path,
    uncompressed_dir: Path,
    batch_size: int,
    device: torch.device,
    subset_ids: set[int] | None,
) -> Iterable[tuple[list[int], torch.Tensor, torch.Tensor]]:
    video_names = ["0.mkv"]
    ds_gt = AVVideoDataset(video_names, data_dir=uncompressed_dir, batch_size=batch_size, device=device, num_threads=2, seed=1234, prefetch_queue_depth=2)
    ds_comp = TensorVideoDataset(video_names, data_dir=submission_dir / "inflated", batch_size=batch_size, device=device, num_threads=2, seed=1234, prefetch_queue_depth=2)
    ds_gt.prepare_data()
    ds_comp.prepare_data()
    sample_id = 0
    for (_, _, gt), (_, _, comp) in zip(
        torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0),
        torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0),
    ):
        count = gt.shape[0]
        ids = list(range(sample_id, sample_id + count))
        sample_id += count
        if subset_ids is None:
            yield ids, gt.to(device), comp.to(device)
            continue
        keep = [i for i, sid in enumerate(ids) if sid in subset_ids]
        if keep:
            idx = torch.tensor(keep, dtype=torch.long)
            yield [ids[i] for i in keep], gt[idx].to(device), comp[idx].to(device)


def evaluate_variants(
    *,
    variants: list[Variant],
    subset: str,
    submission_dir: Path,
    uncompressed_dir: Path,
    batch_size: int,
    device_name: str | None,
    track_per_sample: bool,
    variant_chunk_size: int,
) -> dict:
    device = _device(device_name)
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    subset_ids = _subset_ids(subset)

    sums = {
        v.name: {
            "pose_sum": 0.0,
            "seg_sum": 0.0,
            "count": 0,
            "max_sample_quality": 0.0,
            "max_sample_id": None,
            "per_sample": [],
        }
        for v in variants
    }

    with torch.inference_mode():
        for ids, gt, comp in tqdm(
            _iter_selected(
                submission_dir=submission_dir,
                uncompressed_dir=uncompressed_dir,
                batch_size=batch_size,
                device=device,
                subset_ids=subset_ids,
            ),
            desc=f"postprocess {subset}",
        ):
            gt = gt.float()
            chunk_size = max(1, int(variant_chunk_size))
            for start in range(0, len(variants), chunk_size):
                chunk = variants[start : start + chunk_size]
                pred = torch.cat([variant.apply(comp) for variant in chunk], dim=0)
                gt_rep = gt.repeat((len(chunk), 1, 1, 1, 1))
                pose_dist, seg_dist = distortion.compute_distortion(gt_rep, pred)
                pose_chunks = pose_dist.detach().cpu().split(len(ids))
                seg_chunks = seg_dist.detach().cpu().split(len(ids))
                for variant, pose_tensor, seg_tensor in zip(chunk, pose_chunks, seg_chunks, strict=True):
                    pose_list = pose_tensor.tolist()
                    seg_list = seg_tensor.tolist()
                    rec = sums[variant.name]
                    rec["pose_sum"] += float(sum(pose_list))
                    rec["seg_sum"] += float(sum(seg_list))
                    rec["count"] += len(ids)
                    for sid, pose_v, seg_v in zip(ids, pose_list, seg_list, strict=True):
                        q = quality(float(seg_v), float(pose_v))
                        if q > rec["max_sample_quality"]:
                            rec["max_sample_quality"] = q
                            rec["max_sample_id"] = int(sid)
                        if track_per_sample:
                            rec["per_sample"].append(
                                {
                                    "sample_id": int(sid),
                                    "segnet_dist": float(seg_v),
                                    "posenet_dist": float(pose_v),
                                    "quality": q,
                                }
                            )

    archive_bytes = (submission_dir / "archive.zip").stat().st_size
    rate = rate_term(archive_bytes)
    rows = []
    for variant in variants:
        rec = sums[variant.name]
        if rec["count"] == 0:
            raise RuntimeError(f"no samples evaluated for {subset}")
        pose = rec["pose_sum"] / rec["count"]
        seg = rec["seg_sum"] / rec["count"]
        q = quality(seg, pose)
        rows.append(
            {
                "name": variant.name,
                "subset": subset,
                "sample_count": rec["count"],
                "segnet_dist": seg,
                "posenet_dist": pose,
                "seg_term": 100.0 * seg,
                "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
                "quality": q,
                "archive_bytes": archive_bytes,
                "rate_term": rate,
                "score": q + rate,
                "max_sample_quality": rec["max_sample_quality"],
                "max_sample_id": rec["max_sample_id"],
            }
        )
        if track_per_sample:
            rows[-1]["per_sample"] = sorted(rec["per_sample"], key=lambda r: r["sample_id"])
    base = next(row for row in rows if row["name"] == "identity")
    for row in rows:
        row["quality_delta_vs_identity"] = row["quality"] - base["quality"]
        row["score_delta_vs_identity"] = row["score"] - base["score"]
        row["seg_delta_vs_identity"] = row["seg_term"] - base["seg_term"]
        row["pose_delta_vs_identity"] = row["pose_term"] - base["pose_term"]
    rows.sort(key=lambda r: (r["score"], r["quality"]))
    summary = {
        "subset": subset,
        "device": str(device),
        "archive_bytes": archive_bytes,
        "target_score": 0.290,
        "identity": base,
        "best": rows[0],
        "rows": rows,
    }
    if track_per_sample:
        base_by_id = {int(row["sample_id"]): row for row in base["per_sample"]}
        best_by_id: dict[int, dict] = {}
        for row in rows:
            for sample in row["per_sample"]:
                sample_id = int(sample["sample_id"])
                candidate = {
                    "sample_id": sample_id,
                    "variant": row["name"],
                    "quality": float(sample["quality"]),
                    "segnet_dist": float(sample["segnet_dist"]),
                    "posenet_dist": float(sample["posenet_dist"]),
                    "quality_delta_vs_identity": float(sample["quality"]) - float(base_by_id[sample_id]["quality"]),
                }
                if sample_id not in best_by_id or candidate["quality"] < best_by_id[sample_id]["quality"]:
                    best_by_id[sample_id] = candidate
        oracle_samples = [best_by_id[k] for k in sorted(best_by_id)]
        oracle_seg = sum(s["segnet_dist"] for s in oracle_samples) / len(oracle_samples)
        oracle_pose = sum(s["posenet_dist"] for s in oracle_samples) / len(oracle_samples)
        oracle_q = quality(oracle_seg, oracle_pose)
        summary["oracle_best_per_sample"] = {
            "sample_count": len(oracle_samples),
            "segnet_dist": oracle_seg,
            "posenet_dist": oracle_pose,
            "quality": oracle_q,
            "score_at_same_bytes": oracle_q + rate,
            "quality_delta_vs_identity": oracle_q - base["quality"],
            "estimated_action_bytes_raw": math.ceil(len(oracle_samples) * math.ceil(math.log2(max(1, len(rows)))) / 8),
            "samples": oracle_samples,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="hard8")
    parser.add_argument("--mode", default="small", choices=["small", "wide", "advanced"])
    parser.add_argument("--only", default=None, help="comma-separated variant names to evaluate")
    parser.add_argument("--track-per-sample", action="store_true")
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--uncompressed-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--variant-chunk-size", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = evaluate_variants(
        variants=filter_variants(make_variants(args.mode), args.only),
        subset=args.subset,
        submission_dir=args.submission_dir,
        uncompressed_dir=args.uncompressed_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        track_per_sample=args.track_per_sample,
        variant_chunk_size=args.variant_chunk_size,
    )
    out = args.out or OUT_DIR / f"{args.subset}_{args.mode}_summary.json"
    write_json(out, summary)
    print(json.dumps({k: summary[k] for k in ("subset", "device", "identity", "best")}, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
