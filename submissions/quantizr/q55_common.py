#!/usr/bin/env python
"""Shared helpers for Quantizr #55 restart experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
QUANTIZR_DIR = Path(__file__).resolve().parent
EVALUATE_SH = REPO_ROOT / "evaluate.sh"
DEFAULT_VIDEO_NAMES = REPO_ROOT / "public_test_video_names.txt"
ORIGINAL_BYTES = 37_545_489
Q55_REF = "7366288"

MODEL_PAYLOAD = "model.pt.br"
MODEL_QPACK_PAYLOAD = "model.qpack.br"
MASK_PAYLOAD = "mask.obu.br"
POSE_PAYLOAD = "pose.npy.br"
POSE_QPACK_PAYLOAD = "pose.qpack.br"
ARCH_PAYLOAD = "arch.json.br"
REQUIRED_LEGACY_PAYLOADS = (MODEL_PAYLOAD, MASK_PAYLOAD, POSE_PAYLOAD)


@dataclass(frozen=True)
class Metrics:
    posenet_dist: float
    segnet_dist: float
    archive_bytes: int
    original_bytes: int
    rate: float
    quality_term: float
    rate_term: float
    score: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def archive_payloads(archive_zip: Path) -> list[dict]:
    with zipfile.ZipFile(archive_zip) as z:
        return [
            {
                "filename": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "crc": f"{info.CRC:08x}",
            }
            for info in sorted(z.infolist(), key=lambda x: x.filename)
            if not info.is_dir()
        ]


def summarize_archive(archive_zip: Path) -> dict:
    return {
        "archive_zip": str(archive_zip),
        "archive_bytes": archive_zip.stat().st_size,
        "archive_sha256": sha256_file(archive_zip),
        "payloads": archive_payloads(archive_zip),
    }


def unzip_archive(archive_zip: Path, archive_dir: Path) -> None:
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive_zip) as z:
        z.extractall(archive_dir)


def make_archive_zip(archive_dir: Path, archive_zip: Path, payload_names: Iterable[str] | None = None) -> None:
    archive_zip.parent.mkdir(parents=True, exist_ok=True)
    if payload_names is None:
        payload_names = sorted(p.name for p in archive_dir.iterdir() if p.is_file() and p.suffix == ".br")
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as z:
        for name in payload_names:
            path = archive_dir / name
            if path.exists():
                z.write(path, arcname=name)


def materialize_submission(
    archive_zip: Path,
    submission_dir: Path,
    inflate_mode: str,
    upstream_ref: str = Q55_REF,
) -> None:
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True)
    shutil.copy2(archive_zip, submission_dir / "archive.zip")
    shutil.copy2(QUANTIZR_DIR / "inflate.sh", submission_dir / "inflate.sh")

    if inflate_mode == "upstream":
        pinned_inflate = QUANTIZR_DIR / f"inflate_upstream_{upstream_ref}.py"
        if pinned_inflate.exists():
            shutil.copy2(pinned_inflate, submission_dir / "inflate.py")
        else:
            inflate_src = subprocess.check_output(
                ["git", "show", f"{upstream_ref}:submissions/quantizr/inflate.py"],
                cwd=REPO_ROOT,
            )
            (submission_dir / "inflate.py").write_bytes(inflate_src)
    elif inflate_mode == "modified":
        shutil.copy2(QUANTIZR_DIR / "inflate.py", submission_dir / "inflate.py")
    else:
        raise ValueError(f"unknown inflate mode: {inflate_mode}")


def parse_report(report_path: Path) -> Metrics:
    text = report_path.read_text()
    pose = _extract_float(text, r"Average PoseNet Distortion:\s*([0-9.]+)")
    seg = _extract_float(text, r"Average SegNet Distortion:\s*([0-9.]+)")
    archive_bytes = _extract_int(text, r"Submission file size:\s*([0-9,]+) bytes")
    original_bytes = _extract_int(text, r"Original uncompressed size:\s*([0-9,]+) bytes")
    rate = archive_bytes / original_bytes
    quality = quality_term(seg, pose)
    score = quality + 25.0 * rate
    return Metrics(
        posenet_dist=pose,
        segnet_dist=seg,
        archive_bytes=archive_bytes,
        original_bytes=original_bytes,
        rate=rate,
        quality_term=quality,
        rate_term=25.0 * rate,
        score=score,
    )


def _extract_float(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"missing metric pattern: {pattern}")
    return float(m.group(1).replace(",", ""))


def _extract_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"missing metric pattern: {pattern}")
    return int(m.group(1).replace(",", ""))


def quality_term(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(max(0.0, 10.0 * posenet_dist))


def score_from_bytes(segnet_dist: float, posenet_dist: float, archive_bytes: int, original_bytes: int = ORIGINAL_BYTES) -> float:
    return quality_term(segnet_dist, posenet_dist) + 25.0 * archive_bytes / original_bytes


def run_evaluate_submission(
    submission_dir: Path,
    device: str,
    video_names_file: Path = DEFAULT_VIDEO_NAMES,
    env: dict | None = None,
) -> Path:
    cmd = [
        "bash",
        str(EVALUATE_SH),
        "--submission-dir",
        str(submission_dir),
        "--video-names-file",
        str(video_names_file),
        "--device",
        device,
    ]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)
    return submission_dir / "report.txt"


def ensure_legacy_payloads(archive_dir: Path) -> None:
    missing = [name for name in REQUIRED_LEGACY_PAYLOADS if not (archive_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"archive missing required payload(s): {missing}")


def metric_record(
    *,
    label: str,
    archive_zip: Path,
    device: str,
    report_path: Path,
    extra: dict | None = None,
) -> dict:
    metrics = parse_report(report_path)
    record = {
        "label": label,
        "device": device,
        "report_path": str(report_path),
        "archive_zip": str(archive_zip),
        "archive_sha256": sha256_file(archive_zip),
        "archive_bytes": metrics.archive_bytes,
        "original_bytes": metrics.original_bytes,
        "rate": metrics.rate,
        "rate_term": metrics.rate_term,
        "segnet_dist": metrics.segnet_dist,
        "posenet_dist": metrics.posenet_dist,
        "quality_term": metrics.quality_term,
        "score": metrics.score,
        "payloads": archive_payloads(archive_zip),
    }
    if extra:
        record.update(extra)
    return record
