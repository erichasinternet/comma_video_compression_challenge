#!/usr/bin/env python3
"""Shared helpers for task-aware video source optimization."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frame_utils import AVVideoDataset, camera_size
from submissions.commavq_task.common import (
    FeatureTap,
    build_distortion,
    collect_targets,
    evaluate_frames,
    feature_loss,
    hard_margin_loss,
    load_original_pairs_by_indices,
    round_ste,
    tv_loss,
)


ORIGINAL_BYTES = 37_545_489
MODEL_H, MODEL_W = 384, 512
PAIRS_PER_FILE = 600


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(max(0.0, 10.0 * posenet_dist))


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + 25.0 * archive_bytes / ORIGINAL_BYTES


def required_quality_for_score(archive_bytes: int, target: float = 0.300) -> float:
    return target - 25.0 * archive_bytes / ORIGINAL_BYTES


def metric_table(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> dict:
    q = quality(segnet_dist, posenet_dist)
    rate_term = 25.0 * archive_bytes / ORIGINAL_BYTES
    return {
        "archive_bytes": int(archive_bytes),
        "rate_term": rate_term,
        "segnet_dist": float(segnet_dist),
        "posenet_dist": float(posenet_dist),
        "seg_term": 100.0 * float(segnet_dist),
        "pose_term": math.sqrt(max(0.0, 10.0 * float(posenet_dist))),
        "quality": q,
        "score": q + rate_term,
        "required_quality_for_0.300": required_quality_for_score(archive_bytes),
        "gap_to_0.300": q + rate_term - 0.300,
    }


def parse_indices(indices: str | None, *, offset: int, subset: int) -> list[int]:
    if indices:
        return [int(x) for x in indices.replace(" ", "").split(",") if x]
    return list(range(offset, offset + subset))


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def run_checked(cmd: list[str | Path], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def ensure_q55_inflated(
    *,
    q55_submission_dir: Path = ROOT / "submissions/q55_fp16_pose_int10",
    cache_dir: Path = HERE / "experiments/cache/q55_fp16_pose_int10",
    file_list: Path = ROOT / "public_test_video_names.txt",
) -> Path:
    """Inflate the safe q55 package once and return the inflated raw directory."""

    inflated_dir = cache_dir / "inflated"
    names = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
    if names and all((inflated_dir / f"{Path(name).stem}.raw").exists() for name in names):
        return inflated_dir

    archive_zip = q55_submission_dir / "archive.zip"
    if not archive_zip.exists():
        raise FileNotFoundError(f"missing q55 archive: {archive_zip}")

    archive_dir = cache_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    inflated_dir.mkdir(parents=True, exist_ok=True)
    marker = archive_dir / ".unzipped"
    if not marker.exists():
        for child in archive_dir.iterdir():
            if child.is_file():
                child.unlink()
        with zipfile.ZipFile(archive_zip, "r") as zf:
            zf.extractall(archive_dir)
        marker.write_text("ok\n")

    run_checked([sys.executable, q55_submission_dir / "inflate.py", archive_dir, inflated_dir, file_list])
    return inflated_dir


def load_raw_pairs_by_indices(
    *,
    raw_dir: Path,
    video_names_file: Path,
    sample_indices: list[int],
) -> torch.Tensor:
    """Load selected two-frame samples from official raw layout as [N,2,H,W,3]."""

    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"empty video names file: {video_names_file}")
    out: list[torch.Tensor] = []
    W, H = camera_size
    frame_bytes = H * W * 3
    memmaps: dict[int, np.memmap] = {}
    try:
        for sample_id in sample_indices:
            file_idx = sample_id // PAIRS_PER_FILE
            pair_idx = sample_id % PAIRS_PER_FILE
            if file_idx >= len(names):
                raise IndexError(f"sample {sample_id} maps past file list length {len(names)}")
            if file_idx not in memmaps:
                raw_path = raw_dir / f"{Path(names[file_idx]).stem}.raw"
                if not raw_path.exists():
                    raise FileNotFoundError(f"missing raw file: {raw_path}")
                n_frames = os.path.getsize(raw_path) // frame_bytes
                memmaps[file_idx] = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(n_frames, H, W, 3))
            frames = memmaps[file_idx][pair_idx * 2 : pair_idx * 2 + 2].copy()
            if frames.shape[0] != 2:
                raise RuntimeError(f"sample {sample_id} has incomplete raw pair")
            out.append(torch.from_numpy(frames))
    finally:
        memmaps.clear()
    return torch.stack(out, dim=0)


def to_model_chw(frames_bthwc: torch.Tensor) -> torch.Tensor:
    """Convert raw/video frames [B,2,H,W,3] uint8/float to [B,2,3,384,512]."""

    x = frames_bthwc.permute(0, 1, 4, 2, 3).float()
    flat = x.flatten(0, 1)
    flat = F.interpolate(flat, size=(MODEL_H, MODEL_W), mode="bicubic", align_corners=False)
    return flat.clamp(0, 255).reshape(x.shape[0], 2, 3, MODEL_H, MODEL_W)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
