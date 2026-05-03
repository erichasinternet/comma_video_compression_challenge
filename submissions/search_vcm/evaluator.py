#!/usr/bin/env python
"""Score math, q55 metric ingestion, dominance, and tail-regression helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ORIGINAL_BYTES = 37_545_489
REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SEARCH_DIR / "experiments" / "search_vcm"

Q55_METRICS = {
    "q55_fp16_pose_int10": REPO_ROOT / "submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int10_cpu/metrics.json",
    "q55_fp16_pose_int12": REPO_ROOT / "submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int12_cpu/metrics.json",
    "q55_fp16_only": REPO_ROOT / "submissions/quantizr/experiments/q55_restart/q1_fp16_cpu/metrics.json",
}

Q55_EMBEDDED_METRICS = {
    # Fallback calibration for remote Search VCM oracle jobs where local experiment
    # artifacts are intentionally not copied into the container.
    "q55_fp16_pose_int10": {
        "archive_bytes": 288_268,
        "segnet_dist": 0.0007222,
        "posenet_dist": 0.00065137,
        "archive_sha256": "9f1f005a48514a7215e72b7ddbae36c87a4aeef94f361c02bbf9e15c60d0de03",
        "payloads": [
            {"filename": "mask.obu.br", "compress_size": 219_472, "file_size": 219_472},
            {"filename": "model.qpack.br", "compress_size": 63_680, "file_size": 63_680},
            {"filename": "pose.qpack.br", "compress_size": 4_790, "file_size": 4_790},
        ],
    },
    "q55_fp16_pose_int12": {
        "archive_bytes": 289_127,
        "segnet_dist": 0.00072222,
        "posenet_dist": 0.00064976,
    },
    "q55_fp16_only": {
        "archive_bytes": 296_659,
        "segnet_dist": 0.00072222,
        "posenet_dist": 0.00064992,
    },
}


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * float(segnet_dist) + math.sqrt(max(0.0, 10.0 * float(posenet_dist)))


def rate_term(archive_bytes: int) -> float:
    return 25.0 * int(archive_bytes) / ORIGINAL_BYTES


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + rate_term(archive_bytes)


def required_quality_for_score(archive_bytes: int, target: float = 0.300) -> float:
    return target - rate_term(archive_bytes)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_q55_metrics(name: str, metrics_path: Path | None = None) -> dict[str, Any]:
    path = metrics_path or Q55_METRICS[name]
    raw = load_json(path) if path.exists() else Q55_EMBEDDED_METRICS[name]
    seg = float(raw["segnet_dist"])
    pose = float(raw["posenet_dist"])
    archive_bytes = int(raw["archive_bytes"])
    return {
        "name": name,
        "archive_bytes": archive_bytes,
        "archive_sha": raw.get("archive_sha256"),
        "archive_path": raw.get("archive_zip"),
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
        "quality": quality(seg, pose),
        "rate_term": rate_term(archive_bytes),
        "score": score(seg, pose, archive_bytes),
        "payloads": raw.get("payloads", []),
        "source_metrics": str(path) if path.exists() else "embedded_q55_metrics",
    }


def base_metrics() -> dict[str, Any]:
    return normalize_q55_metrics("q55_fp16_pose_int10")


def term_tradeoff(row: dict[str, Any]) -> str:
    if row.get("kind") == "oracle_only":
        return "oracle_nonpackable"
    score_delta = row.get("score_delta_vs_base")
    if score_delta is None:
        return "incomplete"
    if row.get("dominates_base"):
        return "dominant"
    if score_delta < 0:
        return "rate_quality_tradeoff_with_positive_net_score"
    seg_delta = row.get("seg_delta")
    pose_delta = row.get("pose_delta")
    byte_delta = row.get("byte_delta")
    if seg_delta is not None and seg_delta < 0 and ((pose_delta or 0) > 0 or (byte_delta or 0) > 0):
        return "segnet_improved_but_not_net_positive"
    return "not_promotable"


def classify_against_base(row: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    base = base or base_metrics()
    archive_bytes = row.get("archive_bytes")
    quality_value = row.get("quality")
    score_value = row.get("score")
    dominates = False
    if archive_bytes is not None and quality_value is not None and score_value is not None:
        dominates = (
            score_value <= base["score"]
            and int(archive_bytes) <= int(base["archive_bytes"])
            and float(quality_value) <= float(base["quality"])
        )
    row["dominates_base"] = dominates
    row["term_tradeoff"] = term_tradeoff(row)
    return row


def top_tail_regression(
    base_samples: list[dict[str, Any]],
    candidate_samples: list[dict[str, Any]],
    *,
    top_k: int = 5,
    threshold: float = 0.050,
) -> dict[str, Any]:
    base_by_id = {int(r["sample_id"]): r for r in base_samples}
    candidate_by_id = {int(r["sample_id"]): r for r in candidate_samples}
    top_ids = [
        int(r["sample_id"])
        for r in sorted(base_samples, key=lambda r: float(r["quality_i"]), reverse=True)[:top_k]
    ]
    deltas = []
    for sample_id in top_ids:
        if sample_id not in candidate_by_id:
            continue
        deltas.append(
            {
                "sample_id": sample_id,
                "quality_delta": float(candidate_by_id[sample_id]["quality_i"]) - float(base_by_id[sample_id]["quality_i"]),
            }
        )
    worst = max((row["quality_delta"] for row in deltas), default=0.0)
    return {
        "top_k": top_k,
        "threshold": threshold,
        "worst_delta": worst,
        "reject": worst > threshold,
        "deltas": deltas,
    }
