#!/usr/bin/env python
"""PR #62 fp4_mask_gen artifact helpers for Search VCM v2."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, REPO_ROOT, quality, score, sha256, write_json
from submissions.search_vcm_v2.families.qpose14_data import (
    load_original_subset,
    materialize_qpose14_subset,
    select_torch_device,
)
from submissions.search_vcm_v2.ledger import _rank_rows, qpose_subset_summary
from submissions.search_vcm_v2.subsets import get_subset


FP4_ARCHIVE_URL = "https://github.com/user-attachments/files/27186314/archive.zip"
FP4_SUBMISSION_DIR = REPO_ROOT / "submissions/fp4_mask_gen"
FP4_ARCHIVE = FP4_SUBMISSION_DIR / "archive.zip"
LOWMASK_DIR = EXPERIMENTS_DIR / "lowmask_qpose"
FP4_LEDGER = LOWMASK_DIR / "fp4_mask_gen_per_sample.jsonl"
FP4_SUMMARY = LOWMASK_DIR / "fp4_mask_gen_summary.json"
LOWMASK_CACHE = LOWMASK_DIR / "lowmask_cache.pt"
QPOSE_TEACHER_CACHE = LOWMASK_DIR / "qpose14_teacher_cache.pt"

FP4_REFERENCE = {
    "archive_bytes": 249_624,
    "segnet_dist": 0.00112878,
    "posenet_dist": 0.00063958,
}


def fp4_reference_summary() -> dict[str, Any]:
    archive = FP4_REFERENCE["archive_bytes"]
    seg = FP4_REFERENCE["segnet_dist"]
    pose = FP4_REFERENCE["posenet_dist"]
    q = quality(seg, pose)
    return {
        **FP4_REFERENCE,
        "seg_term": 100.0 * seg,
        "pose_term": float(np.sqrt(10.0 * pose)),
        "quality": q,
        "rate_term": 25.0 * archive / 37_545_489,
        "score": score(seg, pose, archive),
        "source": "fp4_mask_gen_pr62_reference",
    }


def ensure_fp4_archive(path: Path = FP4_ARCHIVE) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FP4_ARCHIVE_URL, path)
    return path


def split_fp4_archive(archive_path: Path = FP4_ARCHIVE) -> tuple[bytes, bytes, bytes]:
    ensure_fp4_archive(archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        return zf.read("mask.obu.br"), zf.read("model.pt.br"), zf.read("pose.bin.br")


def archive_audit(archive_path: Path = FP4_ARCHIVE) -> dict[str, Any]:
    ensure_fp4_archive(archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        members = [{"name": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size} for info in zf.infolist()]
    payload_breakdown = {item["name"]: int(item["bytes"]) for item in members}
    return {
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "members": members,
        "payload_breakdown": payload_breakdown,
    }


def decode_lowmask_video(mask_br_data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode PR #62 mask stream to class masks and grayscale source frames."""

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br_data))
        tmp_path = Path(tmp.name)
    try:
        container = av.open(str(tmp_path))
        cls_frames = []
        gray_frames = []
        for frame in container.decode(video=0):
            gray = frame.to_ndarray(format="gray")
            gray_frames.append(gray.astype(np.uint8))
            cls_frames.append(np.clip(np.round(gray / 63.0), 0, 4).astype(np.uint8))
        container.close()
    finally:
        tmp_path.unlink(missing_ok=True)
    return torch.from_numpy(np.stack(cls_frames)).long().contiguous(), torch.from_numpy(np.stack(gray_frames)).contiguous()


def decode_fp4_pose_stream(pose_br_data: bytes) -> torch.Tensor:
    from submissions.fp4_mask_gen.inflate import decode_pose_bin

    return decode_pose_bin(brotli.decompress(pose_br_data)).float().contiguous()


def load_fp4_generator(model_br_data: bytes, device: torch.device) -> torch.nn.Module:
    from submissions.fp4_mask_gen.inflate import JointFrameGenerator, get_decoded_state_dict

    weights_data = brotli.decompress(model_br_data)
    generator = JointFrameGenerator().to(device)
    generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
    generator.eval()
    return generator


def render_fp4_internal(
    masks: torch.Tensor,
    poses: torch.Tensor,
    *,
    device: str = "auto",
    batch_size: int = 4,
    archive_path: Path = FP4_ARCHIVE,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, model_br, _ = split_fp4_archive(archive_path)
    torch_device = select_torch_device(device)
    generator = load_fp4_generator(model_br, torch_device)
    frame1_chunks = []
    frame2_chunks = []
    with torch.inference_mode():
        for start in range(0, masks.shape[0], batch_size):
            m = masks[start : start + batch_size].to(torch_device).long()
            p = poses[start : start + batch_size].to(torch_device).float()
            f1, f2 = generator(m, p)
            frame1_chunks.append(f1.detach().cpu())
            frame2_chunks.append(f2.detach().cpu())
    return torch.cat(frame1_chunks, dim=0).contiguous(), torch.cat(frame2_chunks, dim=0).contiguous()


def fp4_pred_to_bt_hw3(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    return torch.stack([frame1, frame2], dim=1).permute(0, 1, 3, 4, 2).contiguous()


def evaluate_fp4_per_sample(
    *,
    device: str = "auto",
    batch_size: int = 4,
    force: bool = False,
    archive_path: Path = FP4_ARCHIVE,
) -> list[dict[str, Any]]:
    if FP4_LEDGER.exists() and not force:
        return [json.loads(line) for line in FP4_LEDGER.read_text().splitlines() if line.strip()]

    from frame_utils import AVVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    ensure_fp4_archive(archive_path)
    mask_br, model_br, pose_br = split_fp4_archive(archive_path)
    masks_all, _ = decode_lowmask_video(mask_br)
    poses_all = decode_fp4_pose_stream(pose_br)
    torch_device = select_torch_device(device)
    generator = load_fp4_generator(model_br, torch_device)
    distortion = DistortionNet().eval().to(device=torch_device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, torch_device)

    video_names = ["0.mkv"]
    loader_device = torch_device if torch_device.type != "cuda" else torch.device("cpu")
    ds_gt = AVVideoDataset(video_names, data_dir=REPO_ROOT / "videos", batch_size=batch_size, device=loader_device, num_threads=2, seed=1234, prefetch_queue_depth=2)
    ds_gt.prepare_data()

    rows: list[dict[str, Any]] = []
    sample_id = 0
    with torch.inference_mode():
        for _, _, batch_gt in tqdm(ds_gt, desc="fp4_mask_gen per-sample eval"):
            count = batch_gt.shape[0]
            m = masks_all[sample_id : sample_id + count].to(torch_device).long()
            p = poses_all[sample_id : sample_id + count].to(torch_device).float()
            f1, f2 = generator(m, p)
            pred_bt = fp4_pred_to_bt_hw3(f1, f2).to(torch_device)
            gt = batch_gt.to(torch_device).float()
            posenet_dist, segnet_dist = distortion.compute_distortion(gt, pred_bt)
            for pose_v, seg_v in zip(posenet_dist.detach().cpu().tolist(), segnet_dist.detach().cpu().tolist(), strict=True):
                seg = float(seg_v)
                pose = float(pose_v)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "fp4_segnet_dist": seg,
                        "fp4_posenet_dist": pose,
                        "fp4_seg_term": 100.0 * seg,
                        "fp4_pose_term": float(np.sqrt(max(0.0, 10.0 * pose))),
                        "fp4_quality": quality(seg, pose),
                    }
                )
                sample_id += 1

    rows = _rank_rows(
        [
            {
                "sample_id": row["sample_id"],
                "qpose14_segnet_dist": row["fp4_segnet_dist"],
                "qpose14_posenet_dist": row["fp4_posenet_dist"],
                "qpose14_seg_term": row["fp4_seg_term"],
                "qpose14_pose_term": row["fp4_pose_term"],
                "qpose14_quality": row["fp4_quality"],
                **row,
            }
            for row in rows
        ]
    )
    for row in rows:
        row["rank_by_quality_fp4"] = row.pop("rank_by_quality")
        row["rank_by_pose_fp4"] = row.pop("rank_by_pose")
        row["rank_by_seg_fp4"] = row.pop("rank_by_seg")
        row.pop("qpose14_segnet_dist", None)
        row.pop("qpose14_posenet_dist", None)
        row.pop("qpose14_seg_term", None)
        row.pop("qpose14_pose_term", None)
        row.pop("qpose14_quality", None)

    FP4_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    FP4_LEDGER.write_text("")
    for row in rows:
        with FP4_LEDGER.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def materialize_lowmask_subset(
    subset_name: str,
    sample_ids: list[int],
    *,
    device: str = "auto",
    force: bool = False,
    archive_path: Path = FP4_ARCHIVE,
) -> dict[str, Any]:
    if LOWMASK_CACHE.exists() and not force:
        data = torch.load(LOWMASK_CACHE, map_location="cpu")
        if [int(x) for x in data.get("sample_ids", [])] == [int(x) for x in sample_ids]:
            return data

    mask_br, _, pose_br = split_fp4_archive(archive_path)
    masks_all, gray_all = decode_lowmask_video(mask_br)
    poses_all = decode_fp4_pose_stream(pose_br)
    masks = masks_all[sample_ids].contiguous()
    grays = gray_all[sample_ids].contiguous()
    poses = poses_all[sample_ids].contiguous()
    fp4_frame1, fp4_frame2 = render_fp4_internal(masks, poses, device=device, archive_path=archive_path)
    qpose = materialize_qpose14_subset(subset_name, sample_ids, device=device)
    qpose_teacher = {
        "sample_ids": [int(x) for x in sample_ids],
        "qpose_mask": qpose["mask"],
        "qpose_pose6": qpose["pose6"],
        "qpose_frame1": qpose["qpose_frame1"],
        "qpose_frame2": qpose["qpose_frame2"],
        "source_archive": qpose.get("source_archive"),
    }
    LOWMASK_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(qpose_teacher, QPOSE_TEACHER_CACHE)
    data = {
        "sample_ids": [int(x) for x in sample_ids],
        "lowmask_class": masks,
        "lowmask_gray": grays,
        "fp4_pose6": poses,
        "fp4_frame1": fp4_frame1,
        "fp4_frame2": fp4_frame2,
        "qpose_pose6": qpose["pose6"],
        "qpose_frame1": qpose["qpose_frame1"],
        "qpose_frame2": qpose["qpose_frame2"],
        "source_archive": str(archive_path),
        "frame_space": "internal_384x512_rgb_0_255",
    }
    torch.save(data, LOWMASK_CACHE)
    return data


def write_fp4_summary(rows: list[dict[str, Any]] | None = None, *, archive_path: Path = FP4_ARCHIVE) -> dict[str, Any]:
    audit = archive_audit(archive_path)
    rows = rows or ([json.loads(line) for line in FP4_LEDGER.read_text().splitlines() if line.strip()] if FP4_LEDGER.exists() else [])
    if rows:
        qpose_shaped = [
            {
                "sample_id": row["sample_id"],
                "qpose14_segnet_dist": row["fp4_segnet_dist"],
                "qpose14_posenet_dist": row["fp4_posenet_dist"],
                "qpose14_seg_term": row["fp4_seg_term"],
                "qpose14_pose_term": row["fp4_pose_term"],
                "qpose14_quality": row["fp4_quality"],
            }
            for row in rows
        ]
        full = qpose_subset_summary(qpose_shaped, list(range(600)), archive_bytes=audit["archive_bytes"])
        subset_summaries = {
            "hard3": qpose_subset_summary(qpose_shaped, get_subset("hard3"), archive_bytes=audit["archive_bytes"]),
            "hard8": qpose_subset_summary(qpose_shaped, get_subset("hard8"), archive_bytes=audit["archive_bytes"]),
            "strat64": qpose_subset_summary(qpose_shaped, get_subset("strat64"), archive_bytes=audit["archive_bytes"]),
            "full600": full,
        }
        source = "local_per_sample_evaluator"
    else:
        ref = fp4_reference_summary()
        full = ref
        subset_summaries = {}
        source = "fp4_mask_gen_pr62_reference_no_local_ledger"
    summary = {
        **audit,
        "segnet_dist": full["segnet_dist"],
        "posenet_dist": full["posenet_dist"],
        "seg_term": full["seg_term"],
        "pose_term": full["pose_term"],
        "quality": full["quality"],
        "score": full["score"],
        "source": source,
        "reference_pr62": fp4_reference_summary(),
        "per_sample_ledger": str(FP4_LEDGER) if rows else None,
        "row_count": len(rows),
        "subset_summaries": subset_summaries,
        "hard3_quality": subset_summaries.get("hard3", {}).get("quality"),
        "hard8_quality": subset_summaries.get("hard8", {}).get("quality"),
        "strat64_quality": subset_summaries.get("strat64", {}).get("quality"),
        "full600_quality": subset_summaries.get("full600", {}).get("quality", full["quality"]),
    }
    write_json(FP4_SUMMARY, summary)
    return summary
