#!/usr/bin/env python
"""Deterministic class-mask cleaner diagnostics for Quantizr #55.

This is a recoverability probe, not a byte-accurate submission path. It decodes
a lossy predictor mask, applies zero-byte deterministic cleanup rules, compares
the result to the exact #55 decoded class tensor, and can write a diagnostic
archive using mask_clean.bin.br so the cleaned tensor can be inflated/evaluated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import zipfile
from pathlib import Path

import brotli
import numpy as np

from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    POSE_PAYLOAD,
    POSE_QPACK_PAYLOAD,
    append_jsonl,
    make_archive_zip,
    materialize_submission,
    metric_record,
    run_evaluate_submission,
    summarize_archive,
    unzip_archive,
    write_json,
)
from q55_mask_alloc import decode_archive_masks


MASK_CLEAN_PAYLOAD = "mask_clean.bin.br"


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for raw in values.astype(np.uint16, copy=False).reshape(-1):
        value = int(raw)
        if value & ~mask:
            raise ValueError(f"value {value} exceeds {bits} bits")
        acc |= value << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def write_clean_payload(path: Path, masks: np.ndarray) -> dict:
    masks = np.ascontiguousarray(masks.astype(np.uint8, copy=False))
    header = {
        "format": "quantizr_mask_clean_v1",
        "shape": list(masks.shape),
        "bits": 3,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = pack_bits(masks, 3)
    payload = b"QMC1" + struct.pack("<I", len(header_bytes)) + header_bytes + body
    path.write_bytes(brotli.compress(payload, quality=11, lgwin=24))
    return {
        "shape": list(masks.shape),
        "packed_bytes_raw": len(payload),
        "payload_bytes_br": path.stat().st_size,
    }


def choose_payload(archive_dir: Path, primary: str, fallback: str) -> str:
    if (archive_dir / primary).exists():
        return primary
    if (archive_dir / fallback).exists():
        return fallback
    raise FileNotFoundError(f"missing payload: {primary} or {fallback}")


def decode_mask_from_archive(archive_zip: Path) -> np.ndarray:
    with zipfile.ZipFile(archive_zip) as z:
        names = set(z.namelist())
        if MASK_PAYLOAD not in names:
            raise FileNotFoundError(f"{archive_zip} has no {MASK_PAYLOAD}")
        with z.open(MASK_PAYLOAD) as src:
            tmp = Path("/tmp") / f"q55_mask_cleaner_{archive_zip.stem}_{id(archive_zip)}.obu.br"
            tmp.write_bytes(src.read())
    try:
        return decode_archive_masks(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def spatial_singleton(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    center = mask[:, 1:-1, 1:-1]
    up = mask[:, :-2, 1:-1]
    down = mask[:, 2:, 1:-1]
    left = mask[:, 1:-1, :-2]
    right = mask[:, 1:-1, 2:]
    replace = (up == down) & (up == left) & (up == right) & (center != up)
    inner = out[:, 1:-1, 1:-1]
    inner[replace] = up[replace]
    return out


def window_majority(mask: np.ndarray, *, k: int, min_majority: int, max_center_count: int, chunk: int = 12) -> np.ndarray:
    if k % 2 != 1:
        raise ValueError("window size must be odd")
    pad = k // 2
    out = mask.copy()
    n, h, w = mask.shape
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        block = mask[start:stop]
        padded = np.pad(block, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
        counts = np.zeros((5, stop - start, h, w), dtype=np.uint8)
        for dy in range(k):
            for dx in range(k):
                vals = padded[:, dy : dy + h, dx : dx + w]
                for cls in range(5):
                    counts[cls] += vals == cls
        majority = counts.argmax(axis=0).astype(np.uint8)
        majority_count = counts.max(axis=0)
        center_count = np.take_along_axis(counts, block[None, ...], axis=0)[0]
        replace = (majority != block) & (majority_count >= min_majority) & (center_count <= max_center_count)
        out[start:stop][replace] = majority[replace]
    return out


def temporal_singleton(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if mask.shape[0] < 3:
        return out
    prev_ = mask[:-2]
    cur = mask[1:-1]
    next_ = mask[2:]
    replace = (prev_ == next_) & (cur != prev_)
    middle = out[1:-1]
    middle[replace] = prev_[replace]
    return out


def spatiotemporal_majority(mask: np.ndarray) -> np.ndarray:
    # Conservative final pass: require temporal agreement and weak local support.
    out = mask.copy()
    if mask.shape[0] < 3:
        return out
    spatial = window_majority(mask, k=3, min_majority=5, max_center_count=2)
    prev_ = mask[:-2]
    cur = mask[1:-1]
    next_ = mask[2:]
    temporal_class = prev_
    replace = (prev_ == next_) & (cur != prev_) & (spatial[1:-1] == temporal_class)
    middle = out[1:-1]
    middle[replace] = temporal_class[replace]
    return out


CLEANER_FUNCS = {
    "none": lambda x: x.copy(),
    "d0": spatial_singleton,
    "d1": lambda x: window_majority(spatial_singleton(x), k=3, min_majority=6, max_center_count=2),
    "d1_5": lambda x: window_majority(
        window_majority(spatial_singleton(x), k=3, min_majority=6, max_center_count=2),
        k=5,
        min_majority=18,
        max_center_count=4,
    ),
    "d2": lambda x: temporal_singleton(window_majority(spatial_singleton(x), k=3, min_majority=6, max_center_count=2)),
    "d3": lambda x: spatiotemporal_majority(
        temporal_singleton(window_majority(spatial_singleton(x), k=3, min_majority=6, max_center_count=2))
    ),
}


def compare_masks(exact: np.ndarray, predicted: np.ndarray, cleaned: np.ndarray) -> dict:
    before = predicted != exact
    after = cleaned != exact
    changed_by_cleaner = cleaned != predicted
    repaired = before & ~after
    introduced = ~before & after
    worsened = before & after & (cleaned != predicted)
    total = exact.size
    return {
        "total_pixels": int(total),
        "errors_before": int(before.sum()),
        "errors_before_fraction": float(before.sum() / total),
        "errors_after": int(after.sum()),
        "errors_after_fraction": float(after.sum() / total),
        "errors_repaired": int(repaired.sum()),
        "new_errors_introduced": int(introduced.sum()),
        "wrong_errors_changed": int(worsened.sum()),
        "cleaner_changed_pixels": int(changed_by_cleaner.sum()),
        "cleaner_changed_fraction": float(changed_by_cleaner.sum() / total),
        "repaired_fraction_of_errors": float(repaired.sum() / max(1, before.sum())),
        "false_edit_fraction_of_correct": float(introduced.sum() / max(1, (~before).sum())),
    }


def build_clean_archive(base_archive: Path, archive_dir: Path, archive_zip: Path, cleaned: np.ndarray) -> dict:
    unzip_archive(base_archive, archive_dir)
    model_payload = choose_payload(archive_dir, MODEL_QPACK_PAYLOAD, MODEL_PAYLOAD)
    pose_payload = choose_payload(archive_dir, POSE_QPACK_PAYLOAD, POSE_PAYLOAD)
    for path in archive_dir.glob("mask*.br"):
        path.unlink()
    payload_report = write_clean_payload(archive_dir / MASK_CLEAN_PAYLOAD, cleaned)
    make_archive_zip(archive_dir, archive_zip, [model_payload, pose_payload, MASK_CLEAN_PAYLOAD])
    return {
        "model_payload": model_payload,
        "pose_payload": pose_payload,
        "clean_payload": payload_report,
    }


def run_one(args, cleaner_name: str, exact: np.ndarray, predicted: np.ndarray) -> dict:
    label = args.label or f"qmask_clean_{cleaner_name}"
    if len(args.cleaners) > 1:
        label = f"{label}_{cleaner_name}"
    run_dir = args.out_dir / label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    submission_dir = run_dir / "submission"
    run_dir.mkdir(parents=True, exist_ok=True)

    cleaned = CLEANER_FUNCS[cleaner_name](predicted)
    stats = compare_masks(exact, predicted, cleaned)
    archive_report = build_clean_archive(args.base_archive.resolve(), archive_dir, archive_zip, cleaned)
    record = {
        "label": label,
        "cleaner": cleaner_name,
        "base_archive": str(args.base_archive.resolve()),
        "predictor_mask_archive": str(args.predictor_mask_archive.resolve()),
        "base_archive_summary": summarize_archive(args.base_archive.resolve()),
        "predictor_archive_summary": summarize_archive(args.predictor_mask_archive.resolve()),
        "archive_zip": str(archive_zip),
        "archive_summary": summarize_archive(archive_zip),
        "archive_bytes": archive_zip.stat().st_size,
        "diagnostic_archive_rate_is_not_candidate_rate": True,
        **archive_report,
        **stats,
    }

    materialize_submission(archive_zip=archive_zip, submission_dir=submission_dir, inflate_mode="modified")
    if args.evaluate:
        env = {"FORCE_AV_DATASET": "1"} if args.force_av_dataset else None
        report_path = run_evaluate_submission(submission_dir, args.device, args.video_names, env=env)
        record = metric_record(
            label=label,
            archive_zip=submission_dir / "archive.zip",
            device=args.device,
            report_path=report_path,
            extra=record,
        )

    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "mask_cleaner_results.jsonl", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--predictor-mask-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cleaner", choices=[*CLEANER_FUNCS.keys(), "all"], default="all")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cpu")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--label", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--force-av-dataset", action="store_true")
    args = parser.parse_args()

    args.base_archive = args.base_archive.resolve()
    args.predictor_mask_archive = args.predictor_mask_archive.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.base_archive.exists():
        raise FileNotFoundError(args.base_archive)
    if not args.predictor_mask_archive.exists():
        raise FileNotFoundError(args.predictor_mask_archive)

    exact = decode_mask_from_archive(args.base_archive)
    predicted = decode_mask_from_archive(args.predictor_mask_archive)
    if exact.shape != predicted.shape:
        raise RuntimeError(f"mask shape mismatch: {exact.shape} vs {predicted.shape}")

    cleaners = [name for name in CLEANER_FUNCS if name != "none"] if args.cleaner == "all" else [args.cleaner]
    args.cleaners = cleaners
    records = [run_one(args, cleaner, exact, predicted) for cleaner in cleaners]
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
