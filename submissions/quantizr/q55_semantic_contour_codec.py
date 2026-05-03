#!/usr/bin/env python
"""Lossless semantic-map contour diagnostics for the exact Quantizr #55 mask.

Phase 1 is a scanline contour codec: each row is represented as class runs,
then rows are predicted from the same row in the previous frame and/or the
previous row in the current frame. This is a source-coding diagnostic, not yet
a contest inflater integration.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import brotli
import numpy as np

from q55_common import MASK_PAYLOAD, ORIGINAL_BYTES, REPO_ROOT, sha256_file, write_json
from q55_mask_alloc import decode_archive_masks


DEFAULT_ARCHIVE = REPO_ROOT / "submissions/q55_fp16_pose_int10/archive.zip"
DEFAULT_OUT = REPO_ROOT / "submissions/quantizr/experiments/q55_contour_scanline_v0"
WIDTH = 512
HEIGHT = 384
FRAME_COUNT = 600


@dataclass(frozen=True)
class RowRuns:
    classes: tuple[int, ...]
    bounds: tuple[int, ...]


@dataclass(frozen=True)
class EncodedScanline:
    strategy: str
    meta: dict
    parts: dict[str, bytes]
    mode_counts: dict[str, int]
    stats: dict


class UVarintReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.pos = 0

    def read(self) -> int:
        shift = 0
        value = 0
        while True:
            if self.pos >= len(self.payload):
                raise EOFError("uvarint stream ended early")
            b = self.payload[self.pos]
            self.pos += 1
            value |= (b & 0x7F) << shift
            if not (b & 0x80):
                return value
            shift += 7
            if shift > 63:
                raise ValueError("uvarint is too long")

    def consumed(self) -> bool:
        return self.pos == len(self.payload)


class BitReader:
    def __init__(self, payload: bytes, bits: int):
        self.payload = payload
        self.bits = bits
        self.mask = (1 << bits) - 1
        self.byte_pos = 0
        self.acc = 0
        self.acc_bits = 0
        self.values_read = 0

    def read(self) -> int:
        while self.acc_bits < self.bits:
            if self.byte_pos >= len(self.payload):
                raise EOFError("bit stream ended early")
            self.acc |= int(self.payload[self.byte_pos]) << self.acc_bits
            self.byte_pos += 1
            self.acc_bits += 8
        value = self.acc & self.mask
        self.acc >>= self.bits
        self.acc_bits -= self.bits
        self.values_read += 1
        return value


def pack_bits(values: Iterable[int] | np.ndarray, bits: int) -> bytes:
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for raw in np.asarray(list(values), dtype=np.uint16).reshape(-1):
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


def encode_uvarint(value: int, out: bytearray) -> None:
    if value < 0:
        raise ValueError(value)
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def encode_uvarints(values: Iterable[int]) -> bytes:
    out = bytearray()
    for value in values:
        encode_uvarint(int(value), out)
    return bytes(out)


def zigzag(value: int) -> int:
    return (value << 1) if value >= 0 else ((-value << 1) - 1)


def unzigzag(value: int) -> int:
    return (value >> 1) if (value & 1) == 0 else -((value >> 1) + 1)


def row_runs(row: np.ndarray) -> RowRuns:
    changes = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], changes))
    classes = tuple(int(x) for x in row[starts])
    bounds = tuple(int(x) for x in changes)
    return RowRuns(classes=classes, bounds=bounds)


def fill_row(out: np.ndarray, runs: RowRuns) -> None:
    start = 0
    for cls, end in zip(runs.classes, (*runs.bounds, WIDTH), strict=True):
        if end <= start or end > WIDTH:
            raise ValueError(f"invalid run boundary {start}->{end}")
        out[start:end] = cls
        start = end


def boundary_delta_bytes(current: RowRuns, pred: RowRuns) -> int:
    return len(encode_uvarints(zigzag(a - b) for a, b in zip(current.bounds, pred.bounds, strict=True)))


def raw_row_estimate_bytes(runs: RowRuns) -> int:
    lengths = []
    start = 0
    for bound in runs.bounds:
        lengths.append(bound - start)
        start = bound
    return 1 + len(runs.classes) + len(encode_uvarints(lengths))


def choose_mode(
    current: RowRuns,
    prev_frame: RowRuns | None,
    up_row: RowRuns | None,
    strategy: str,
) -> tuple[int, RowRuns | None]:
    if strategy == "raw":
        return 4, None

    if prev_frame is not None and current == prev_frame:
        return 0, prev_frame
    if strategy == "prev_up" and up_row is not None and current == up_row:
        return 1, up_row

    candidates: list[tuple[int, int, RowRuns]] = []
    if prev_frame is not None and current.classes == prev_frame.classes and len(current.bounds) == len(prev_frame.bounds):
        candidates.append((boundary_delta_bytes(current, prev_frame), 2, prev_frame))
    if (
        strategy == "prev_up"
        and up_row is not None
        and current.classes == up_row.classes
        and len(current.bounds) == len(up_row.bounds)
    ):
        candidates.append((boundary_delta_bytes(current, up_row), 3, up_row))

    raw_cost = raw_row_estimate_bytes(current)
    if candidates:
        best_cost, best_mode, best_pred = min(candidates, key=lambda x: (x[0], x[1]))
        if best_cost < raw_cost:
            return best_mode, best_pred
    return 4, None


def encode_scanline(masks: np.ndarray, strategy: str) -> EncodedScanline:
    if strategy not in {"raw", "prev", "prev_up"}:
        raise ValueError(strategy)

    frames, height, width = masks.shape
    if (height, width) != (HEIGHT, WIDTH):
        raise ValueError(f"expected {HEIGHT}x{WIDTH}, got {height}x{width}")

    mode_values: list[int] = []
    raw_run_counts: list[int] = []
    raw_classes: list[int] = []
    raw_lengths: list[int] = []
    delta_values: list[int] = []
    run_counts: list[int] = []
    changed_vs_prev_frame = 0
    unchanged_vs_prev_frame = 0
    previous_frame_runs: list[RowRuns] | None = None
    all_frame_runs: list[list[RowRuns]] = []

    for t in range(frames):
        frame_runs: list[RowRuns] = []
        for y in range(height):
            current = row_runs(masks[t, y])
            run_counts.append(len(current.classes))
            prev_frame = previous_frame_runs[y] if previous_frame_runs is not None else None
            up_row = frame_runs[y - 1] if y > 0 else None

            mode, pred = choose_mode(current, prev_frame, up_row, strategy)
            mode_values.append(mode)
            if mode == 4:
                raw_run_counts.append(len(current.classes) - 1)
                raw_classes.extend(current.classes)
                start = 0
                for bound in current.bounds:
                    raw_lengths.append(bound - start)
                    start = bound
            elif mode in (2, 3):
                if pred is None:
                    raise AssertionError("delta mode missing predictor")
                delta_values.extend(zigzag(a - b) for a, b in zip(current.bounds, pred.bounds, strict=True))

            if prev_frame is not None:
                if current == prev_frame:
                    unchanged_vs_prev_frame += 1
                else:
                    changed_vs_prev_frame += 1
            frame_runs.append(current)
        all_frame_runs.append(frame_runs)
        previous_frame_runs = frame_runs

    parts = {
        "mode.3b": pack_bits(mode_values, 3),
        "raw_run_count_minus1.varint": encode_uvarints(raw_run_counts),
        "raw_classes.3b": pack_bits(raw_classes, 3),
        "raw_run_lengths.varint": encode_uvarints(raw_lengths),
        "boundary_deltas.zzvarint": encode_uvarints(delta_values),
    }
    mode_names = {
        0: "same_prev_frame_row",
        1: "same_previous_row",
        2: "delta_prev_frame_row",
        3: "delta_previous_row",
        4: "raw_row",
    }
    mode_counts = {mode_names[i]: int(mode_values.count(i)) for i in range(5)}
    run_counts_arr = np.asarray(run_counts, dtype=np.int32)
    stats = {
        "rows": int(frames * height),
        "average_runs_per_row": float(run_counts_arr.mean()),
        "median_runs_per_row": float(np.median(run_counts_arr)),
        "max_runs_per_row": int(run_counts_arr.max()),
        "total_boundaries": int(run_counts_arr.sum() - run_counts_arr.size),
        "unchanged_rows_vs_prev_frame": int(unchanged_vs_prev_frame),
        "changed_rows_vs_prev_frame": int(changed_vs_prev_frame),
        "unchanged_row_fraction_vs_prev_frame": float(
            unchanged_vs_prev_frame / max(1, unchanged_vs_prev_frame + changed_vs_prev_frame)
        ),
        "raw_rows": int(mode_counts["raw_row"]),
        "delta_rows": int(mode_counts["delta_prev_frame_row"] + mode_counts["delta_previous_row"]),
    }
    meta = {
        "format": "q55_scanline_contour_v0",
        "strategy": strategy,
        "shape": [int(frames), int(height), int(width)],
        "class_count": 5,
        "mode_count": len(mode_values),
        "raw_class_count": len(raw_classes),
    }
    return EncodedScanline(strategy=strategy, meta=meta, parts=parts, mode_counts=mode_counts, stats=stats)


def decode_scanline(encoded: EncodedScanline) -> np.ndarray:
    frames, height, width = encoded.meta["shape"]
    if (height, width) != (HEIGHT, WIDTH):
        raise ValueError(f"expected {HEIGHT}x{WIDTH}, got {height}x{width}")
    out = np.empty((frames, height, width), dtype=np.uint8)
    mode_reader = BitReader(encoded.parts["mode.3b"], 3)
    raw_count_reader = UVarintReader(encoded.parts["raw_run_count_minus1.varint"])
    raw_class_reader = BitReader(encoded.parts["raw_classes.3b"], 3)
    raw_length_reader = UVarintReader(encoded.parts["raw_run_lengths.varint"])
    delta_reader = UVarintReader(encoded.parts["boundary_deltas.zzvarint"])

    previous_frame_runs: list[RowRuns] | None = None
    for t in range(frames):
        frame_runs: list[RowRuns] = []
        for y in range(height):
            mode = mode_reader.read()
            if mode == 0:
                if previous_frame_runs is None:
                    raise ValueError("same previous-frame row used on frame 0")
                runs = previous_frame_runs[y]
            elif mode == 1:
                if y == 0:
                    raise ValueError("same previous-row used on row 0")
                runs = frame_runs[y - 1]
            elif mode in (2, 3):
                pred = previous_frame_runs[y] if mode == 2 and previous_frame_runs is not None else None
                if mode == 3 and y > 0:
                    pred = frame_runs[y - 1]
                if pred is None:
                    raise ValueError(f"delta mode {mode} missing predictor at frame {t} row {y}")
                bounds = tuple(
                    int(bound + unzigzag(delta_reader.read()))
                    for bound in pred.bounds
                )
                runs = RowRuns(classes=pred.classes, bounds=bounds)
            elif mode == 4:
                run_count = raw_count_reader.read() + 1
                classes = tuple(raw_class_reader.read() for _ in range(run_count))
                start = 0
                bounds_list = []
                for _ in range(run_count - 1):
                    start += raw_length_reader.read()
                    bounds_list.append(start)
                runs = RowRuns(classes=classes, bounds=tuple(bounds_list))
            else:
                raise ValueError(f"unknown row mode {mode}")

            fill_row(out[t, y], runs)
            frame_runs.append(runs)
        previous_frame_runs = frame_runs

    if not raw_count_reader.consumed():
        raise ValueError("raw run-count stream has trailing bytes")
    if not raw_length_reader.consumed():
        raise ValueError("raw run-length stream has trailing bytes")
    if not delta_reader.consumed():
        raise ValueError("delta stream has trailing bytes")
    return out


def extract_mask_payload(archive_zip: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(archive_zip) as z:
        z.extract(MASK_PAYLOAD, out_dir)
    return out_dir / MASK_PAYLOAD


def mask_payload_size(archive_zip: Path) -> int:
    with zipfile.ZipFile(archive_zip) as z:
        return z.getinfo(MASK_PAYLOAD).file_size


def load_masks(args: argparse.Namespace) -> np.ndarray:
    if args.exact_mask_cache and args.exact_mask_cache.exists():
        path = args.exact_mask_cache
        if path.suffix == ".npy":
            masks = np.load(path)
        elif path.suffix == ".npz":
            data = np.load(path)
            masks = data["masks"] if "masks" in data else data[data.files[0]]
        elif path.suffix in {".pt", ".pth"}:
            import torch

            obj = torch.load(path, map_location="cpu")
            if hasattr(obj, "detach"):
                masks = obj.detach().cpu().numpy()
            elif isinstance(obj, dict):
                for key in ("masks", "mask", "classes", "class_masks"):
                    if key in obj:
                        value = obj[key]
                        masks = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
                        break
                else:
                    raise KeyError(f"no mask tensor key found in {path}")
            else:
                masks = np.asarray(obj)
        else:
            raise ValueError(f"unsupported mask cache suffix: {path.suffix}")
    else:
        with tempfile.TemporaryDirectory() as td:
            payload = extract_mask_payload(args.archive, Path(td))
            masks = decode_archive_masks(payload)
        if args.write_cache:
            args.write_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.write_cache, masks.astype(np.uint8, copy=False))

    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[1:] != (HEIGHT, WIDTH):
        raise ValueError(f"expected masks shape (*,{HEIGHT},{WIDTH}), got {masks.shape}")
    if masks.shape[0] != FRAME_COUNT:
        print(f"warning: expected {FRAME_COUNT} masks, got {masks.shape[0]}", flush=True)
    if int(masks.min()) < 0 or int(masks.max()) > 4:
        raise ValueError(f"mask class range must be 0..4, got {int(masks.min())}..{int(masks.max())}")
    return masks


def compress_payload(payload: bytes, method: str) -> bytes:
    if method == "brotli":
        return brotli.compress(payload, quality=11, lgwin=24)
    if method == "xz":
        return lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)
    if method == "zstd":
        if not shutil.which("zstd"):
            raise RuntimeError("zstd command not found")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "part.bin"
            dst = root / "part.bin.zst"
            src.write_bytes(payload)
            subprocess.run(["zstd", "-22", "--ultra", "-q", "-f", str(src), "-o", str(dst)], check=True)
            return dst.read_bytes()
    raise ValueError(method)


def available_methods(requested: str) -> list[str]:
    methods = []
    for item in [x.strip() for x in requested.split(",") if x.strip()]:
        if item == "zstd" and not shutil.which("zstd"):
            print("warning: zstd requested but not found; skipping", flush=True)
            continue
        if item not in {"brotli", "xz", "zstd"}:
            raise ValueError(f"unknown compressor: {item}")
        methods.append(item)
    if not methods:
        raise ValueError("no compressors available")
    return methods


def compressed_report(encoded: EncodedScanline, methods: list[str], out_dir: Path, keep_streams: bool) -> dict:
    meta_payload = json.dumps(encoded.meta, sort_keys=True, separators=(",", ":")).encode()
    parts = {**encoded.parts, "meta.json": meta_payload}
    part_records = []
    totals = {method: 0 for method in methods}
    best_per_stream_total = 0

    streams_dir = out_dir / encoded.strategy / "streams"
    if keep_streams:
        streams_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in parts.items():
        sizes = {}
        best_size = None
        best_method = None
        for method in methods:
            compressed = compress_payload(payload, method)
            sizes[method] = len(compressed)
            totals[method] += len(compressed)
            if best_size is None or len(compressed) < best_size:
                best_size = len(compressed)
                best_method = method
            if keep_streams:
                suffix = {"brotli": ".br", "xz": ".xz", "zstd": ".zst"}[method]
                (streams_dir / f"{name}{suffix}").write_bytes(compressed)
        if best_size is None or best_method is None:
            raise AssertionError("compressor loop produced no size")
        best_per_stream_total += best_size
        part_records.append(
            {
                "name": name,
                "raw_bytes": len(payload),
                "compressed_bytes": sizes,
                "best_method": best_method,
                "best_bytes": best_size,
            }
        )

    best_whole_method = min(totals.items(), key=lambda x: x[1])
    return {
        "strategy": encoded.strategy,
        "raw_total_bytes": sum(len(x) for x in parts.values()),
        "compressed_totals": totals,
        "best_whole_method": best_whole_method[0],
        "best_whole_method_bytes": best_whole_method[1],
        "best_per_stream_bytes": best_per_stream_total,
        "parts": part_records,
        "mode_counts": encoded.mode_counts,
        "stats": encoded.stats,
        "meta": encoded.meta,
    }


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(10.0 * posenet_dist)


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + 25.0 * archive_bytes / ORIGINAL_BYTES


def required_quality_for_score(archive_bytes: int, target: float = 0.300) -> float:
    return target - 25.0 * archive_bytes / ORIGINAL_BYTES


def phase_label(mask_bytes: int) -> str:
    if mask_bytes <= 152_000:
        return "dream_local_cpu_0p2x"
    if mask_bytes <= 182_000:
        return "strong_pr_like_0p2x"
    if mask_bytes <= 196_000:
        return "weak_first_place_path"
    if mask_bytes <= 210_000:
        return "continue_to_contour_codec"
    if mask_bytes <= 240_000:
        return "near_miss_only_continue_if_boundary_stats_promising"
    return "hard_fail"


def cmd_scanline(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods = available_methods(args.compressors)
    masks = load_masks(args)
    av1_mask_bytes = mask_payload_size(args.archive)
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    records = []
    for strategy in strategies:
        print(f"encoding strategy={strategy}", flush=True)
        encoded = encode_scanline(masks, strategy)
        if args.verify:
            decoded = decode_scanline(encoded)
            if not np.array_equal(decoded, masks):
                diff = int(np.count_nonzero(decoded != masks))
                raise RuntimeError(f"decode verification failed for {strategy}: {diff} pixels differ")
        report = compressed_report(encoded, methods, args.out_dir, args.keep_streams)
        report["verified_exact"] = bool(args.verify)
        report["best_per_stream_label"] = phase_label(int(report["best_per_stream_bytes"]))
        report["best_whole_method_label"] = phase_label(int(report["best_whole_method_bytes"]))
        records.append(report)
        write_json(args.out_dir / f"{strategy}_metrics.json", report)
        print(
            f"{strategy}: best_per_stream={report['best_per_stream_bytes']:,} "
            f"best_whole={report['best_whole_method']}:{report['best_whole_method_bytes']:,}",
            flush=True,
        )

    best = min(records, key=lambda x: x["best_per_stream_bytes"])
    projected_nonmask_bytes = args.nonmask_bytes
    projected_archive_bytes = projected_nonmask_bytes + int(best["best_per_stream_bytes"])
    summary = {
        "archive": str(args.archive),
        "archive_sha256": sha256_file(args.archive),
        "av1_mask_payload_bytes": av1_mask_bytes,
        "mask_shape": [int(x) for x in masks.shape],
        "compressors": methods,
        "strategies": records,
        "best_strategy": best["strategy"],
        "best_mask_bytes": int(best["best_per_stream_bytes"]),
        "best_mask_label": phase_label(int(best["best_per_stream_bytes"])),
        "best_vs_av1_delta_bytes": int(best["best_per_stream_bytes"]) - int(av1_mask_bytes),
        "best_vs_av1_delta_fraction": float((best["best_per_stream_bytes"] - av1_mask_bytes) / av1_mask_bytes),
        "projected_nonmask_bytes": projected_nonmask_bytes,
        "projected_archive_bytes": projected_archive_bytes,
        "projected_rate_term": 25.0 * projected_archive_bytes / ORIGINAL_BYTES,
        "required_quality_for_0p300": required_quality_for_score(projected_archive_bytes),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.out_dir / "metrics.json", summary)
    (args.out_dir / "scanline_results.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    scanline = sub.add_parser("scanline", help="run phase-1 scanline contour diagnostic")
    scanline.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    scanline.add_argument("--exact-mask-cache", type=Path, default=None)
    scanline.add_argument("--write-cache", type=Path, default=None)
    scanline.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    scanline.add_argument("--compressors", default="brotli,zstd,xz")
    scanline.add_argument("--strategies", default="raw,prev,prev_up")
    scanline.add_argument("--nonmask-bytes", type=int, default=68_796)
    scanline.add_argument("--verify", action="store_true")
    scanline.add_argument("--keep-streams", action="store_true")
    scanline.set_defaults(func=cmd_scanline)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
