#!/usr/bin/env python
"""Score math and qpose14 baseline helpers for Search VCM v2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ORIGINAL_BYTES = 37_545_489
REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SEARCH_DIR / "experiments"
RUN_LEDGER = EXPERIMENTS_DIR / "runs.jsonl"
QPOSE14_LEDGER = EXPERIMENTS_DIR / "qpose14_per_sample.jsonl"
QPOSE14_SUMMARY = EXPERIMENTS_DIR / "qpose14_summary.json"

QPOSE14_REFERENCE = {
    "archive_bytes": 287_573,
    "segnet_dist": 0.00061261,
    "posenet_dist": 0.00052154,
}


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * float(segnet_dist) + math.sqrt(max(0.0, 10.0 * float(posenet_dist)))


def rate_term(archive_bytes: int) -> float:
    return 25.0 * int(archive_bytes) / ORIGINAL_BYTES


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + rate_term(archive_bytes)


def required_quality_for_score(archive_bytes: int, target: float = 0.300) -> float:
    return target - rate_term(archive_bytes)


def qpose14_reference_summary() -> dict[str, Any]:
    seg = QPOSE14_REFERENCE["segnet_dist"]
    pose = QPOSE14_REFERENCE["posenet_dist"]
    archive_bytes = QPOSE14_REFERENCE["archive_bytes"]
    return {
        **QPOSE14_REFERENCE,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(10.0 * pose),
        "quality": quality(seg, pose),
        "rate_term": rate_term(archive_bytes),
        "score": score(seg, pose, archive_bytes),
        "source": "qpose14_pr63_readme_reference",
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qpose14_summary() -> dict[str, Any]:
    if QPOSE14_SUMMARY.exists():
        return load_json(QPOSE14_SUMMARY)
    return qpose14_reference_summary()


def subset_metrics(rows: list[dict[str, Any]], subset: list[int], *, archive_bytes: int | None = None) -> dict[str, Any]:
    by_id = {int(row["sample_id"]): row for row in rows}
    selected = [by_id[idx] for idx in subset if idx in by_id]
    if not selected:
        raise RuntimeError("subset has no qpose14 ledger rows")
    seg = sum(float(row["qpose14_segnet_dist"]) for row in selected) / len(selected)
    pose = sum(float(row["qpose14_posenet_dist"]) for row in selected) / len(selected)
    archive = int(archive_bytes if archive_bytes is not None else qpose14_summary()["archive_bytes"])
    return {
        "sample_count": len(selected),
        "segnet_dist": seg,
        "posenet_dist": pose,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose)),
        "quality": quality(seg, pose),
        "archive_bytes": archive,
        "rate_term": rate_term(archive),
        "score": score(seg, pose, archive),
    }


def baseline_for_subset(subset_name: str, subset_indices: list[int]) -> dict[str, Any]:
    rows = load_jsonl(QPOSE14_LEDGER)
    if rows:
        return subset_metrics(rows, subset_indices)
    return qpose14_reference_summary()


def add_baseline_deltas(row: dict[str, Any], baseline: dict[str, Any], baseline_samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    row["baseline_name"] = "qpose14"
    row["baseline_subset_quality"] = baseline.get("quality")
    if row.get("quality") is not None and baseline.get("quality") is not None:
        row["quality_delta_vs_baseline"] = float(row["quality"]) - float(baseline["quality"])
    if row.get("segnet_dist") is not None and baseline.get("segnet_dist") is not None:
        row["seg_delta_vs_baseline"] = 100.0 * (float(row["segnet_dist"]) - float(baseline["segnet_dist"]))
    if row.get("posenet_dist") is not None and baseline.get("posenet_dist") is not None:
        row["pose_delta_vs_baseline"] = math.sqrt(max(0.0, 10.0 * float(row["posenet_dist"]))) - math.sqrt(
            max(0.0, 10.0 * float(baseline["posenet_dist"]))
        )
    if row.get("archive_bytes") is not None and baseline.get("archive_bytes") is not None:
        row["rate_delta_vs_baseline"] = rate_term(int(row["archive_bytes"])) - rate_term(int(baseline["archive_bytes"]))
    if row.get("score") is not None and baseline.get("score") is not None:
        row["estimated_score_delta_vs_baseline"] = float(row["score"]) - float(baseline["score"])
    samples = row.get("extra", {}).get("per_sample", [])
    if samples:
        row["max_sample_quality"] = max(float(s.get("quality_i", s.get("quality", 0.0))) for s in samples)
        sample60 = next((s for s in samples if int(s.get("sample_id", -1)) == 60), None)
        if sample60:
            row["sample60_pose_term"] = float(sample60.get("pose_term", 0.0))
    if baseline_samples and samples:
        base_by_id = {int(s["sample_id"]): s for s in baseline_samples}
        deltas = []
        for sample in samples:
            sample_id = int(sample.get("sample_id", -1))
            if sample_id in base_by_id:
                deltas.append(float(sample.get("quality_i", sample.get("quality", 0.0))) - float(base_by_id[sample_id]["qpose14_quality"]))
                if sample_id == 60:
                    row["sample60_pose_delta_vs_baseline"] = float(sample.get("pose_term", 0.0)) - float(
                        base_by_id[sample_id]["qpose14_pose_term"]
                    )
        if deltas:
            row["max_sample_delta_vs_baseline"] = max(deltas)
    return row

