#!/usr/bin/env python
"""Pose side-channel oracle for PR #67 qpose14_qzs3.

PR #67 stores an aggressively compressed QP1 pose stream that only carries the
first pose input channel. This tool tests whether restoring cheap extra pose
channels improves evaluator quality enough to pay for the added bytes.
"""

from __future__ import annotations

import argparse
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

from frame_utils import AVVideoDataset  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json  # noqa: E402
from submissions.search_vcm_v2.families.qpose14_data import decode_pose_stream, select_torch_device, split_archive_payload  # noqa: E402
from submissions.search_vcm_v2.subsets import get_subset  # noqa: E402
from submissions.search_vcm_v2.tools.qzs3_lowbit_quant_oracle import ARCHIVE_BYTES, load_generator  # noqa: E402


OUT_DIR = EXPERIMENTS_DIR / "qzs3_pose_channel_oracle"


def render_fullres(model: torch.nn.Module, mask: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    frame1, frame2 = model(mask.long(), pose.float())
    frame1 = F.interpolate(frame1, size=(874, 1164), mode="bilinear", align_corners=False)
    frame2 = F.interpolate(frame2, size=(874, 1164), mode="bilinear", align_corners=False)
    pred = torch.stack([frame1, frame2], dim=1).clamp(0, 255).round()
    return einops.rearrange(pred, "b t c h w -> b t h w c")


def build_pose_variants(base_pose: torch.Tensor, qpose_pose: torch.Tensor) -> dict[str, torch.Tensor]:
    variants = {
        "pr67_qp1": base_pose,
        "qpose14_full_pose": qpose_pose,
        "pr67_col0_qpose_other5": qpose_pose.clone(),
        "qpose_col0_pr67_other0": base_pose.clone(),
    }
    variants["pr67_col0_qpose_other5"][:, 0] = base_pose[:, 0]
    variants["qpose_col0_pr67_other0"][:, 0] = qpose_pose[:, 0]
    return variants


def evaluate_streaming(
    *,
    subset_name: str,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    device = select_torch_device(device_name)
    model, masks, base_pose = load_generator(device)
    _, _, qpose_pose_br = split_archive_payload(REPO_ROOT / "submissions/qpose14/archive.zip")
    qpose_pose = decode_pose_stream(qpose_pose_br)
    variants = build_pose_variants(base_pose, qpose_pose)

    sample_ids = get_subset(subset_name)
    wanted = set(int(x) for x in sample_ids)
    max_needed = max(wanted)

    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for param in list(model.parameters()) + list(distortion.parameters()):
        param.requires_grad_(False)

    accum: dict[str, dict[str, Any]] = {
        name: {"rows": [], "seg_sum": 0.0, "pose_sum": 0.0, "count": 0}
        for name in variants
    }

    dataset = AVVideoDataset(["0.mkv"], data_dir=REPO_ROOT / "videos", batch_size=batch_size, device=torch.device("cpu"), num_threads=2, seed=1234, prefetch_queue_depth=2)
    dataset.prepare_data()
    cursor = 0
    start_time = time.time()
    with torch.inference_mode():
        for _, _, batch in dataset:
            ids_all = list(range(cursor, cursor + batch.shape[0]))
            keep_local = [i for i, sid in enumerate(ids_all) if sid in wanted]
            if keep_local:
                ids = [ids_all[i] for i in keep_local]
                original = batch[keep_local].float().to(device)
                mask = masks[ids].to(device)
                for name, pose_all in variants.items():
                    pose = pose_all[ids].to(device)
                    pred = render_fullres(model, mask, pose)
                    pose_dist, seg_dist = distortion.compute_distortion(original, pred)
                    for sid, pose_v, seg_v in zip(ids, pose_dist.cpu().tolist(), seg_dist.cpu().tolist(), strict=True):
                        seg = float(seg_v)
                        pose_val = float(pose_v)
                        row = {
                            "sample_id": int(sid),
                            "segnet_dist": seg,
                            "posenet_dist": pose_val,
                            "seg_term": 100.0 * seg,
                            "pose_term": float(torch.sqrt(torch.tensor(10.0 * pose_val)).item()),
                            "quality": quality(seg, pose_val),
                        }
                        accum[name]["rows"].append(row)
                        accum[name]["seg_sum"] += seg
                        accum[name]["pose_sum"] += pose_val
                        accum[name]["count"] += 1
            cursor += batch.shape[0]
            if cursor > max_needed and all(v["count"] == len(sample_ids) for v in accum.values()):
                break

    rows = []
    for name, data in accum.items():
        count = int(data["count"])
        seg_mean = data["seg_sum"] / count
        pose_mean = data["pose_sum"] / count
        q = quality(seg_mean, pose_mean)
        rows.append(
            {
                "name": name,
                "segnet_dist": seg_mean,
                "posenet_dist": pose_mean,
                "seg_term": 100.0 * seg_mean,
                "pose_term": float(torch.sqrt(torch.tensor(10.0 * pose_mean)).item()),
                "quality": q,
                "score_at_pr67_bytes": q + rate_term(ARCHIVE_BYTES),
                "max_sample_quality": max(row["quality"] for row in data["rows"]),
                "per_sample": sorted(data["rows"], key=lambda r: r["sample_id"]),
            }
        )
    base = next(row for row in rows if row["name"] == "pr67_qp1")
    for row in rows:
        row["quality_delta_vs_pr67_qp1"] = row["quality"] - base["quality"]
        row["score_delta_vs_pr67_qp1_at_same_bytes"] = row["score_at_pr67_bytes"] - base["score_at_pr67_bytes"]

    return {
        "subset": subset_name,
        "device": str(device),
        "batch_size": batch_size,
        "elapsed_sec": time.time() - start_time,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="hard8")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    summary = evaluate_streaming(subset_name=args.subset, device_name=args.device, batch_size=args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"{args.subset}_summary.json", summary)
    for row in summary["rows"]:
        print(
            json.dumps(
                {
                    "name": row["name"],
                    "quality": row["quality"],
                    "quality_delta_vs_pr67_qp1": row["quality_delta_vs_pr67_qp1"],
                    "seg_term": row["seg_term"],
                    "pose_term": row["pose_term"],
                    "score_at_pr67_bytes": row["score_at_pr67_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
