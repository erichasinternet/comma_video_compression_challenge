#!/usr/bin/env python
"""Entropy oracle and transformed-stream baselines for the exact #55 mask.

This script does not build a submission codec. It answers the first source
coding question: whether the exact rounded class tensor has enough structure to
beat the AV1 mask payload by a meaningful margin.
"""

from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
import shutil
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np

from q55_common import ORIGINAL_BYTES, append_jsonl, archive_payloads, sha256_file, write_json
from q55_mask_alloc import decode_archive_masks


MASK_PAYLOAD = "mask.obu.br"
DEFAULT_LIMITS = (270_000, 265_000, 260_000, 250_947, 240_000, 230_000, 221_000)


@dataclass(frozen=True)
class StreamPart:
    name: str
    payload: bytes


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


def empirical_entropy_bits(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64, copy=False)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    nz = counts[counts > 0]
    return float(np.sum(nz * np.log2(total / nz)))


def grouped_entropy_bits(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64, copy=False)
    if counts.ndim != 2:
        raise ValueError("grouped entropy expects a 2D array")
    bits = 0.0
    for row in counts:
        bits += empirical_entropy_bits(row)
    return float(bits)


def mask_payload_size(archive_zip: Path) -> int:
    with zipfile.ZipFile(archive_zip) as z:
        return z.getinfo(MASK_PAYLOAD).file_size


def extract_mask_payload(archive_zip: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(archive_zip) as z:
        z.extract(MASK_PAYLOAD, out_dir)
    return out_dir / MASK_PAYLOAD


def load_exact_masks(archive_zip: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        payload = extract_mask_payload(archive_zip, Path(td))
        return decode_archive_masks(payload)


def make_context_arrays(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sentinel = np.uint8(5)
    prev = np.empty_like(masks)
    prev[0].fill(sentinel)
    prev[1:] = masks[:-1]

    left = np.empty_like(masks)
    left[:, :, 0].fill(sentinel)
    left[:, :, 1:] = masks[:, :, :-1]

    up = np.empty_like(masks)
    up[:, 0, :].fill(sentinel)
    up[:, 1:, :] = masks[:, :-1, :]
    return prev, left, up


def p6_events(masks: np.ndarray, prev: np.ndarray, left: np.ndarray, up: np.ndarray) -> np.ndarray:
    event = np.full(masks.shape, 3, dtype=np.uint8)
    same_prev = masks == prev
    same_left = (~same_prev) & (masks == left)
    same_up = (~same_prev) & (~same_left) & (masks == up)
    event[same_prev] = 0
    event[same_left] = 1
    event[same_up] = 2
    return event


def p6_entropy_report(masks: np.ndarray, prev: np.ndarray, left: np.ndarray, up: np.ndarray, event: np.ndarray) -> dict:
    class_counts = np.bincount(masks.reshape(-1), minlength=5)
    event_counts = np.bincount(event.reshape(-1), minlength=4)
    residual = masks[event == 3]
    residual_counts = np.bincount(residual.reshape(-1), minlength=5)

    context = (prev.astype(np.uint16) * 36 + left.astype(np.uint16) * 6 + up.astype(np.uint16)).reshape(-1)
    event_flat = event.reshape(-1).astype(np.uint16)
    event_context_counts = np.bincount(context * 4 + event_flat, minlength=216 * 4).reshape(216, 4)

    if residual.size:
        residual_context = context[event_flat == 3]
        residual_context_counts = np.bincount(
            residual_context * 5 + residual.astype(np.uint16),
            minlength=216 * 5,
        ).reshape(216, 5)
    else:
        residual_context_counts = np.zeros((216, 5), dtype=np.int64)

    p0_bits = empirical_entropy_bits(class_counts)
    p6_uncontext_bits = empirical_entropy_bits(event_counts) + empirical_entropy_bits(residual_counts)
    p6_context_bits = grouped_entropy_bits(event_context_counts) + grouped_entropy_bits(residual_context_counts)

    same_prev = int(event_counts[0])
    same_left = int(event_counts[1])
    same_up = int(event_counts[2])
    residual_count = int(event_counts[3])
    total = int(event.size)
    return {
        "symbol_count": total,
        "class_counts": [int(x) for x in class_counts],
        "class_fractions": [float(x / total) for x in class_counts],
        "class_entropy_bits": p0_bits,
        "class_entropy_bytes": int(math.ceil(p0_bits / 8.0)),
        "p6_event_counts": {
            "same_prev": same_prev,
            "same_left": same_left,
            "same_up": same_up,
            "residual": residual_count,
        },
        "p6_event_fractions": {
            "same_prev": same_prev / total,
            "same_left": same_left / total,
            "same_up": same_up / total,
            "residual": residual_count / total,
        },
        "p6_residual_class_counts": [int(x) for x in residual_counts],
        "p6_static_uncontext_entropy_bits": p6_uncontext_bits,
        "p6_static_uncontext_entropy_bytes": int(math.ceil(p6_uncontext_bits / 8.0)),
        "p6_static_context_entropy_bits": p6_context_bits,
        "p6_static_context_entropy_bytes": int(math.ceil(p6_context_bits / 8.0)),
        "p6_contexts_nonempty": int((event_context_counts.sum(axis=1) > 0).sum()),
        "p6_residual_contexts_nonempty": int((residual_context_counts.sum(axis=1) > 0).sum()),
    }


def build_streams(masks: np.ndarray, event: np.ndarray) -> dict[str, list[StreamPart]]:
    residual = masks[event == 3]
    event_flat = event.reshape(-1)
    same_prev = event_flat == 0
    not_same_prev_values = masks.reshape(-1)[~same_prev]

    lo = (event_flat & 1).astype(np.uint8)
    hi = ((event_flat >> 1) & 1).astype(np.uint8)

    return {
        "s0_raw_u8": [StreamPart("classes.u8", masks.reshape(-1).tobytes())],
        "s0_raw_3bit": [StreamPart("classes.3b", pack_bits(masks, 3))],
        "s1_prev_same_residual": [
            StreamPart("same_prev.1b", pack_bits(same_prev.astype(np.uint8), 1)),
            StreamPart("not_same_prev_classes.3b", pack_bits(not_same_prev_values, 3)),
        ],
        "s2_p6_event_residual": [
            StreamPart("event.2b", pack_bits(event, 2)),
            StreamPart("residual_classes.3b", pack_bits(residual, 3)),
        ],
        "s3_p6_event_bitplanes_residual": [
            StreamPart("event_lo.1b", pack_bits(lo, 1)),
            StreamPart("event_hi.1b", pack_bits(hi, 1)),
            StreamPart("residual_classes.3b", pack_bits(residual, 3)),
        ],
    }


def compress_payload(payload: bytes, method: str) -> bytes:
    if method == "brotli11":
        return brotli.compress(payload, quality=11, lgwin=24)
    if method == "xz9e":
        return lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)
    if method == "bz2_9":
        return bz2.compress(payload, compresslevel=9)
    if method == "zlib9":
        return zlib.compress(payload, level=9)
    raise ValueError(method)


def available_external_methods() -> list[str]:
    methods = []
    if shutil.which("zstd"):
        methods.append("zstd22")
    if shutil.which("zpaq"):
        methods.append("zpaq")
    return methods


def compress_external(payload: bytes, method: str) -> int:
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "stream.bin"
        src.write_bytes(payload)
        if method == "zstd22":
            dst = root / "stream.bin.zst"
            subprocess.run(["zstd", "-22", "--ultra", "-q", "-f", str(src), "-o", str(dst)], check=True)
            return dst.stat().st_size
        if method == "zpaq":
            dst = root / "stream.zpaq"
            subprocess.run(["zpaq", "add", str(dst), str(src), "-m5", "-quiet"], check=True)
            return dst.stat().st_size
    raise ValueError(method)


def compressed_stream_report(
    streams: dict[str, list[StreamPart]],
    *,
    out_dir: Path | None,
    keep_streams: bool,
) -> list[dict]:
    methods = ["brotli11", "xz9e", "bz2_9", "zlib9"]
    external = available_external_methods()
    report = []
    if out_dir and keep_streams:
        out_dir.mkdir(parents=True, exist_ok=True)

    for transform, parts in streams.items():
        raw_total = sum(len(part.payload) for part in parts)
        record = {
            "transform": transform,
            "raw_bytes": raw_total,
            "parts": [{"name": p.name, "raw_bytes": len(p.payload)} for p in parts],
            "methods": {},
        }
        for method in methods:
            sizes = [len(compress_payload(part.payload, method)) for part in parts]
            record["methods"][method] = {
                "part_bytes": sizes,
                "total_bytes": int(sum(sizes)),
            }
        for method in external:
            sizes = [compress_external(part.payload, method) for part in parts]
            record["methods"][method] = {
                "part_bytes": sizes,
                "total_bytes": int(sum(sizes)),
            }
        best_method, best_info = min(record["methods"].items(), key=lambda kv: kv[1]["total_bytes"])
        record["best_method"] = best_method
        record["best_total_bytes"] = int(best_info["total_bytes"])

        if out_dir and keep_streams:
            transform_dir = out_dir / transform
            transform_dir.mkdir(parents=True, exist_ok=True)
            for part in parts:
                (transform_dir / part.name).write_bytes(part.payload)
        report.append(record)
    return report


def target_table(*, archive_bytes: int, mask_bytes: int, quality_terms: dict[str, float]) -> dict:
    non_mask_bytes = archive_bytes - mask_bytes
    rows = []
    for limit in DEFAULT_LIMITS:
        rows.append(
            {
                "archive_limit": limit,
                "mask_target_bytes": limit - non_mask_bytes,
                "rate_term": 25.0 * limit / ORIGINAL_BYTES,
                "required_quality_for_lt_0_300": 0.300 - 25.0 * limit / ORIGINAL_BYTES,
            }
        )
    quality_rows = {}
    for name, quality in quality_terms.items():
        archive_limit = math.floor((0.300 - quality) * ORIGINAL_BYTES / 25.0)
        quality_rows[name] = {
            "quality_term": quality,
            "archive_limit_for_lt_0_300": archive_limit,
            "mask_target_bytes": archive_limit - non_mask_bytes,
        }
    return {
        "non_mask_bytes": non_mask_bytes,
        "archive_limit_rows": rows,
        "quality_rows": quality_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="q55_mask_entropy_oracle")
    parser.add_argument("--keep-streams", action="store_true")
    parser.add_argument("--local-quality", type=float, default=0.15292750)
    parser.add_argument("--pr-quality", type=float, default=0.13290)
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.exists():
        raise FileNotFoundError(archive)
    run_dir = args.out_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)

    masks = load_exact_masks(archive)
    mask_bytes = mask_payload_size(archive)
    prev, left, up = make_context_arrays(masks)
    event = p6_events(masks, prev, left, up)
    entropy = p6_entropy_report(masks, prev, left, up, event)

    streams = build_streams(masks, event)
    stream_report = compressed_stream_report(
        streams,
        out_dir=run_dir / "streams",
        keep_streams=args.keep_streams,
    )
    best_stream = min(stream_report, key=lambda rec: rec["best_total_bytes"])

    record = {
        "label": args.label,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "payloads": archive_payloads(archive),
        "mask_payload_bytes": mask_bytes,
        "mask_shape": list(masks.shape),
        "mask_symbols": int(masks.size),
        "current_mask_bits_per_pixel": float(mask_bytes * 8 / masks.size),
        "entropy": entropy,
        "stream_transforms": stream_report,
        "best_transformed_stream": {
            "transform": best_stream["transform"],
            "method": best_stream["best_method"],
            "total_bytes": best_stream["best_total_bytes"],
            "savings_vs_current_mask_bytes": int(mask_bytes - best_stream["best_total_bytes"]),
        },
        "target_table": target_table(
            archive_bytes=archive.stat().st_size,
            mask_bytes=mask_bytes,
            quality_terms={
                "local_q1_fp16_pose_int10": args.local_quality,
                "pr_like_q55": args.pr_quality,
            },
        ),
        "decision_hints": {
            "weak_pass_mask_bytes": 196_000,
            "strong_pass_mask_bytes": 182_000,
            "dream_pass_mask_bytes": 152_000,
            "build_real_codec_if_best_actual_le_200k": best_stream["best_total_bytes"] <= 200_000,
            "build_real_codec_if_context_entropy_le_180k": entropy["p6_static_context_entropy_bytes"] <= 180_000,
            "stop_if_best_actual_gt_240k_and_entropy_gt_220k": (
                best_stream["best_total_bytes"] > 240_000
                and entropy["p6_static_context_entropy_bytes"] > 220_000
            ),
        },
    }
    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "mask_entropy_oracle_results.jsonl", record)

    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
