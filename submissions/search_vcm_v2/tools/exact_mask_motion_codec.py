#!/usr/bin/env python
"""Motion-compensated exact semantic-mask codec oracle.

This is a source-coding diagnostic for qpose/Quantizr masks. It does not change
the decoded class tensor. The question is whether exact inter-frame motion plus
sparse repairs can beat the current AV1/Brotli exact mask payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, write_json
from submissions.search_vcm_v2.families.boundary_residual_codec import (
    compress_streams,
    decode_varints,
    encode_varints,
    pack_bits,
    unpack_bits,
)
from submissions.search_vcm_v2.families.qpose14_data import (
    MASK_BYTES,
    QPOSE14_ARCHIVE,
    decode_mask_stream,
    split_archive_payload,
)


QPOSE14_ARCHIVE_BYTES = 287_573
MODE_COPY = 1
MODE_SPARSE = 2
MODE_RAW = 3


def zigzag_encode(value: int) -> int:
    value = int(value)
    return (value << 1) if value >= 0 else ((-value << 1) - 1)


def zigzag_decode(value: int) -> int:
    value = int(value)
    return (value >> 1) if (value & 1) == 0 else -((value >> 1) + 1)


def make_shifts(search: int, step: int) -> list[tuple[int, int]]:
    """Return deterministic motion candidates, zero and local moves first."""

    shifts: list[tuple[int, int]] = [(0, 0)]
    seen = {(0, 0)}

    local = min(2, search)
    for dy in range(-local, local + 1):
        for dx in range(-local, local + 1):
            if (dy, dx) not in seen:
                shifts.append((dy, dx))
                seen.add((dy, dx))

    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            if (dy, dx) not in seen:
                shifts.append((dy, dx))
                seen.add((dy, dx))
    return shifts


def shift_frame(frame: np.ndarray, dy: int, dx: int, fill: int = 255) -> np.ndarray:
    """Shift previous frame into destination coordinates."""

    height, width = frame.shape
    out = np.full_like(frame, fill)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def pred_tile(prev: np.ndarray, y0: int, x0: int, block_size: int, dy: int, dx: int, fill: int = 255) -> np.ndarray:
    """Extract one shifted prediction tile without materializing the full shift."""

    out = np.full((block_size, block_size), fill, dtype=prev.dtype)
    src_y0 = y0 - dy
    src_y1 = src_y0 + block_size
    src_x0 = x0 - dx
    src_x1 = src_x0 + block_size
    in_y0 = max(src_y0, 0)
    in_y1 = min(src_y1, prev.shape[0])
    in_x0 = max(src_x0, 0)
    in_x1 = min(src_x1, prev.shape[1])
    if in_y1 <= in_y0 or in_x1 <= in_x0:
        return out
    dst_y0 = in_y0 - src_y0
    dst_x0 = in_x0 - src_x0
    out[dst_y0 : dst_y0 + (in_y1 - in_y0), dst_x0 : dst_x0 + (in_x1 - in_x0)] = prev[in_y0:in_y1, in_x0:in_x1]
    return out


def block_mismatch_counts(curr: np.ndarray, shifted_prev: np.ndarray, block_size: int) -> np.ndarray:
    diff = curr != shifted_prev
    by = curr.shape[0] // block_size
    bx = curr.shape[1] // block_size
    return diff.reshape(by, block_size, bx, block_size).sum(axis=(1, 3)).astype(np.uint16)


def compute_best_motion(
    classes: np.ndarray,
    *,
    block_size: int,
    shifts: list[tuple[int, int]],
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute best motion vector per block for every inter frame."""

    frames, height, width = classes.shape
    if height % block_size or width % block_size:
        raise ValueError(f"block_size={block_size} does not divide {(height, width)}")
    by = height // block_size
    bx = width // block_size
    best_counts = np.empty((frames - 1, by, bx), dtype=np.uint16)
    best_shift_idx = np.empty((frames - 1, by, bx), dtype=np.uint16)

    for t in range(1, frames):
        curr = classes[t]
        prev = classes[t - 1]
        counts0 = block_mismatch_counts(curr, shift_frame(prev, *shifts[0]), block_size)
        best = counts0
        best_idx = np.zeros((by, bx), dtype=np.uint16)
        for shift_idx, (dy, dx) in enumerate(shifts[1:], start=1):
            counts = block_mismatch_counts(curr, shift_frame(prev, dy, dx), block_size)
            update = counts < best
            if np.any(update):
                best = np.where(update, counts, best)
                best_idx = np.where(update, shift_idx, best_idx)
        best_counts[t - 1] = best
        best_shift_idx[t - 1] = best_idx
        if progress and (t == 1 or t % 50 == 0 or t == frames - 1):
            exact = int((best == 0).sum())
            print(f"motion frame {t:04d}/{frames - 1}: exact_blocks={exact}/{by * bx} avg_mismatch={float(best.mean()):.2f}", flush=True)
    return best_counts, best_shift_idx


@dataclass(frozen=True)
class MotionRecord:
    tile_id: int
    mode: int
    dy: int
    dx: int
    bitmap: bytes = b""
    classes: bytes = b""
    raw: bytes = b""
    changed_pixels: int = 0


def _record_tile_id(frame: int, tile_y: int, tile_x: int, tiles_y: int, tiles_x: int) -> int:
    return frame * tiles_y * tiles_x + tile_y * tiles_x + tile_x


def build_motion_records(
    classes: np.ndarray,
    *,
    block_size: int,
    shifts: list[tuple[int, int]],
    best_counts: np.ndarray,
    best_shift_idx: np.ndarray,
    sparse_threshold: int,
) -> list[MotionRecord]:
    frames, height, width = classes.shape
    tiles_y = height // block_size
    tiles_x = width // block_size
    records: list[MotionRecord] = []

    for frame in range(1, frames):
        curr = classes[frame]
        prev = classes[frame - 1]
        counts = best_counts[frame - 1]
        shift_idx = best_shift_idx[frame - 1]
        for ty in range(tiles_y):
            y0 = ty * block_size
            for tx in range(tiles_x):
                x0 = tx * block_size
                idx = int(shift_idx[ty, tx])
                dy, dx = shifts[idx]
                mismatch = int(counts[ty, tx])
                tile_id = _record_tile_id(frame, ty, tx, tiles_y, tiles_x)
                if mismatch == 0 and dy == 0 and dx == 0:
                    continue
                if mismatch == 0:
                    records.append(MotionRecord(tile_id=tile_id, mode=MODE_COPY, dy=dy, dx=dx))
                    continue
                curr_tile = curr[y0 : y0 + block_size, x0 : x0 + block_size]
                if mismatch <= sparse_threshold:
                    pred = pred_tile(prev, y0, x0, block_size, dy, dx)
                    diff = pred != curr_tile
                    records.append(
                        MotionRecord(
                            tile_id=tile_id,
                            mode=MODE_SPARSE,
                            dy=dy,
                            dx=dx,
                            bitmap=pack_bits(diff),
                            classes=curr_tile[diff].astype(np.uint8).tobytes(),
                            changed_pixels=int(diff.sum()),
                        )
                    )
                else:
                    records.append(
                        MotionRecord(
                            tile_id=tile_id,
                            mode=MODE_RAW,
                            dy=0,
                            dx=0,
                            raw=curr_tile.astype(np.uint8, copy=False).tobytes(),
                            changed_pixels=block_size * block_size,
                        )
                    )
    records.sort(key=lambda record: record.tile_id)
    return records


def pack_motion_streams(classes: np.ndarray, records: list[MotionRecord], *, block_size: int, search: int, step: int) -> dict[str, bytes]:
    return pack_motion_streams_with_sparse_codec(
        classes,
        records,
        block_size=block_size,
        search=search,
        step=step,
        sparse_codec="bitmap",
    )


def pack_motion_streams_with_sparse_codec(
    classes: np.ndarray,
    records: list[MotionRecord],
    *,
    block_size: int,
    search: int,
    step: int,
    sparse_codec: str,
) -> dict[str, bytes]:
    frames, height, width = classes.shape
    tiles_y = height // block_size
    tiles_x = width // block_size
    deltas = []
    prev_id = 0
    for i, record in enumerate(records):
        deltas.append(record.tile_id if i == 0 else record.tile_id - prev_id)
        prev_id = record.tile_id

    meta = {
        "shape": [int(x) for x in classes.shape],
        "block_size": int(block_size),
        "search": int(search),
        "step": int(step),
        "record_count": int(len(records)),
        "tiles_y": int(tiles_y),
        "tiles_x": int(tiles_x),
        "copy_records": int(sum(record.mode == MODE_COPY for record in records)),
        "sparse_records": int(sum(record.mode == MODE_SPARSE for record in records)),
        "raw_records": int(sum(record.mode == MODE_RAW for record in records)),
        "sparse_changed_pixels": int(sum(record.changed_pixels for record in records if record.mode == MODE_SPARSE)),
        "sparse_codec": sparse_codec,
        "sparse_offset_dtype": "uint8" if block_size * block_size <= 256 else "uint16le",
    }
    sparse_records = [record for record in records if record.mode == MODE_SPARSE]
    streams = {
        "meta.json": json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        "key0.bin": classes[0].astype(np.uint8, copy=False).tobytes(),
        "tile_deltas.bin": encode_varints(deltas),
        "modes.bin": bytes(record.mode for record in records),
        "mv_dy.bin": encode_varints(zigzag_encode(record.dy) for record in records),
        "mv_dx.bin": encode_varints(zigzag_encode(record.dx) for record in records),
        "sparse_classes.bin": b"".join(record.classes for record in sparse_records),
        "raw_tiles.bin": b"".join(record.raw for record in records if record.mode == MODE_RAW),
    }
    if sparse_codec == "bitmap":
        streams["bitmaps.bin"] = b"".join(record.bitmap for record in sparse_records)
    elif sparse_codec == "offsets":
        counts: list[int] = []
        offsets = bytearray()
        offset_dtype = np.uint8 if block_size * block_size <= 256 else np.dtype("<u2")
        for record in sparse_records:
            mask = unpack_bits(record.bitmap, block_size * block_size)
            where = np.flatnonzero(mask)
            counts.append(int(len(where)))
            offsets.extend(where.astype(offset_dtype).tobytes())
        streams["sparse_counts.bin"] = encode_varints(counts)
        streams["sparse_offsets.bin"] = bytes(offsets)
    else:
        raise ValueError(f"unknown sparse_codec={sparse_codec!r}")
    return streams


def decode_motion_streams(streams: dict[str, bytes]) -> np.ndarray:
    meta = json.loads(streams["meta.json"].decode("utf-8"))
    shape = tuple(int(x) for x in meta["shape"])
    block_size = int(meta["block_size"])
    _, height, width = shape
    tiles_y = height // block_size
    tiles_x = width // block_size
    classes = np.empty(shape, dtype=np.uint8)
    classes[0] = np.frombuffer(streams["key0.bin"], dtype=np.uint8).reshape(height, width)

    tile_deltas = decode_varints(streams["tile_deltas.bin"])
    modes = np.frombuffer(streams["modes.bin"], dtype=np.uint8)
    dy_values = [zigzag_decode(x) for x in decode_varints(streams["mv_dy.bin"])]
    dx_values = [zigzag_decode(x) for x in decode_varints(streams["mv_dx.bin"])]
    if not (len(tile_deltas) == len(modes) == len(dy_values) == len(dx_values)):
        raise ValueError("motion stream length mismatch")

    sparse_codec = meta.get("sparse_codec", "bitmap")
    bitmap_bytes = (block_size * block_size + 7) // 8
    bitmaps = streams.get("bitmaps.bin", b"")
    sparse_counts = decode_varints(streams.get("sparse_counts.bin", b"")) if sparse_codec == "offsets" else []
    sparse_offsets = streams.get("sparse_offsets.bin", b"")
    offset_dtype_name = meta.get("sparse_offset_dtype", "uint8")
    offset_dtype = np.uint8 if offset_dtype_name == "uint8" else np.dtype("<u2")
    offset_itemsize = int(np.dtype(offset_dtype).itemsize)
    sparse_classes = streams["sparse_classes.bin"]
    raw_tiles = streams["raw_tiles.bin"]
    bitmap_cursor = 0
    offsets_cursor = 0
    sparse_record_cursor = 0
    sparse_cursor = 0
    raw_cursor = 0
    record_cursor = 0
    tile_id = 0
    records_by_frame: dict[int, list[tuple[int, int, int, int, int, bytes | None, bytes | None]]] = {}
    for i, delta in enumerate(tile_deltas):
        tile_id = tile_id + delta if i else delta
        frame = tile_id // (tiles_y * tiles_x)
        rem = tile_id % (tiles_y * tiles_x)
        tile_y = rem // tiles_x
        tile_x = rem % tiles_x
        mode = int(modes[i])
        dy = int(dy_values[i])
        dx = int(dx_values[i])
        bitmap: bytes | None = None
        payload: bytes | None = None
        if mode == MODE_SPARSE:
            if sparse_codec == "bitmap":
                bitmap = bitmaps[bitmap_cursor : bitmap_cursor + bitmap_bytes]
                bitmap_cursor += bitmap_bytes
                changed = int(unpack_bits(bitmap, block_size * block_size).sum())
            elif sparse_codec == "offsets":
                changed = int(sparse_counts[sparse_record_cursor])
                offset_bytes = changed * offset_itemsize
                offsets = np.frombuffer(sparse_offsets[offsets_cursor : offsets_cursor + offset_bytes], dtype=offset_dtype)
                offsets_cursor += offset_bytes
                bits = np.zeros(block_size * block_size, dtype=bool)
                bits[offsets.astype(np.int64)] = True
                bitmap = pack_bits(bits)
                sparse_record_cursor += 1
            else:
                raise ValueError(f"unknown sparse_codec={sparse_codec!r}")
            payload = sparse_classes[sparse_cursor : sparse_cursor + changed]
            sparse_cursor += changed
        elif mode == MODE_RAW:
            payload = raw_tiles[raw_cursor : raw_cursor + block_size * block_size]
            raw_cursor += block_size * block_size
        elif mode != MODE_COPY:
            raise ValueError(f"unknown mode {mode}")
        records_by_frame.setdefault(frame, []).append((tile_y, tile_x, mode, dy, dx, bitmap, payload))
        record_cursor += 1

    if record_cursor != len(tile_deltas):
        raise ValueError("record cursor mismatch")

    for frame in range(1, shape[0]):
        prev = classes[frame - 1]
        out = prev.copy()
        for tile_y, tile_x, mode, dy, dx, bitmap, payload in records_by_frame.get(frame, []):
            y0 = tile_y * block_size
            x0 = tile_x * block_size
            if mode == MODE_COPY:
                out[y0 : y0 + block_size, x0 : x0 + block_size] = pred_tile(prev, y0, x0, block_size, dy, dx)
            elif mode == MODE_SPARSE:
                tile = pred_tile(prev, y0, x0, block_size, dy, dx)
                mask = unpack_bits(bitmap or b"", block_size * block_size).reshape(block_size, block_size)
                values = np.frombuffer(payload or b"", dtype=np.uint8)
                tile[mask] = values
                out[y0 : y0 + block_size, x0 : x0 + block_size] = tile
            else:
                out[y0 : y0 + block_size, x0 : x0 + block_size] = np.frombuffer(payload or b"", dtype=np.uint8).reshape(block_size, block_size)
        classes[frame] = out
    return classes


def _decision(mask_bytes: int) -> str:
    if mask_bytes <= 182 * 1024:
        return "pass_sub_0p300_candidate"
    if mask_bytes <= 196 * 1024:
        return "near_first_place_or_pr_like_candidate"
    if mask_bytes <= 205 * 1024:
        return "near_fail"
    return "fail"


def run_variant(
    classes: np.ndarray,
    *,
    block_size: int,
    search: int,
    step: int,
    sparse_thresholds: Iterable[int],
    sparse_codecs: Iterable[str],
    compressors: tuple[str, ...],
    verify: bool,
) -> list[dict[str, Any]]:
    shifts = make_shifts(search, step)
    print(f"variant motion block={block_size} search={search} step={step} shifts={len(shifts)}", flush=True)
    best_counts, best_shift_idx = compute_best_motion(classes, block_size=block_size, shifts=shifts)
    rows = []
    for sparse_threshold in sparse_thresholds:
        records = build_motion_records(
            classes,
            block_size=block_size,
            shifts=shifts,
            best_counts=best_counts,
            best_shift_idx=best_shift_idx,
            sparse_threshold=int(sparse_threshold),
        )
        for sparse_codec in sparse_codecs:
            print(f"packing block={block_size} search={search} threshold={sparse_threshold} sparse={sparse_codec}", flush=True)
            streams = pack_motion_streams_with_sparse_codec(
                classes,
                records,
                block_size=block_size,
                search=search,
                step=step,
                sparse_codec=sparse_codec,
            )
            if verify:
                decoded = decode_motion_streams(streams)
                if not np.array_equal(decoded, classes):
                    raise AssertionError(f"decode mismatch for b{block_size}_s{search}_thr{sparse_threshold}_{sparse_codec}")
            compressed = compress_streams(streams, compressors)
            mask_payload = int(compressed["total_best_bytes"])
            row = {
                "candidate": f"motion_b{block_size}_s{search}_step{step}_thr{sparse_threshold}_{sparse_codec}",
                "block_size": int(block_size),
                "search": int(search),
                "step": int(step),
                "shift_count": len(shifts),
                "sparse_threshold": int(sparse_threshold),
                "sparse_codec": sparse_codec,
                "records": len(records),
                "copy_records": int(sum(record.mode == MODE_COPY for record in records)),
                "sparse_records": int(sum(record.mode == MODE_SPARSE for record in records)),
                "raw_records": int(sum(record.mode == MODE_RAW for record in records)),
                "sparse_changed_pixels": int(sum(record.changed_pixels for record in records if record.mode == MODE_SPARSE)),
                "mask_payload_bytes": mask_payload,
                "current_qpose_mask_bytes": MASK_BYTES,
                "mask_delta_vs_qpose": mask_payload - MASK_BYTES,
                "projected_qpose_archive_bytes": QPOSE14_ARCHIVE_BYTES - MASK_BYTES + mask_payload,
                "verified_exact": bool(verify),
                "decision": _decision(mask_payload),
                "compressor_breakdown": compressed,
            }
            print(
                json.dumps(
                    {
                        "candidate": row["candidate"],
                        "mask_payload_bytes": row["mask_payload_bytes"],
                        "mask_delta_vs_qpose": row["mask_delta_vs_qpose"],
                        "projected_qpose_archive_bytes": row["projected_qpose_archive_bytes"],
                        "decision": row["decision"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            rows.append(row)
    return rows


def load_qpose_classes(qpose_archive: Path) -> np.ndarray:
    mask_br, _, _ = split_archive_payload(qpose_archive)
    return decode_mask_stream(mask_br).to(torch.uint8).numpy()


def run_sweep(
    *,
    qpose_archive: Path,
    out: Path,
    block_sizes: list[int],
    searches: list[int],
    step: int,
    sparse_thresholds: list[int],
    sparse_codecs: list[str],
    compressors: tuple[str, ...],
    verify: bool,
) -> dict[str, Any]:
    classes = load_qpose_classes(qpose_archive)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for block_size in block_sizes:
        for search in searches:
            rows.extend(
                run_variant(
                    classes,
                    block_size=block_size,
                    search=search,
                    step=step,
                    sparse_thresholds=sparse_thresholds,
                    sparse_codecs=sparse_codecs,
                    compressors=compressors,
                    verify=verify,
                )
            )
            partial = {
                "qpose_archive": str(qpose_archive),
                "shape": list(classes.shape),
                "rows": rows,
                "best": min(rows, key=lambda item: int(item["mask_payload_bytes"])) if rows else None,
            }
            write_json(out / "partial_summary.json", partial)
    best = min(rows, key=lambda item: int(item["mask_payload_bytes"])) if rows else None
    summary = {
        "qpose_archive": str(qpose_archive),
        "shape": list(classes.shape),
        "current_qpose_mask_bytes": MASK_BYTES,
        "current_qpose_archive_bytes": QPOSE14_ARCHIVE_BYTES,
        "best": best,
        "decision": _decision(int(best["mask_payload_bytes"])) if best else "fail",
        "rows": rows,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qpose-archive", type=Path, default=QPOSE14_ARCHIVE)
    parser.add_argument("--out", type=Path, default=EXPERIMENTS_DIR / "exact_mask_motion_codec")
    parser.add_argument("--block-sizes", default="16")
    parser.add_argument("--searches", default="8")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--sparse-thresholds", default="16,32,64,128")
    parser.add_argument("--sparse-codecs", default="bitmap,offsets")
    parser.add_argument("--compressors", default="brotli,zstd")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    compressors = tuple(item.strip() for item in args.compressors.split(",") if item.strip())
    run_sweep(
        qpose_archive=args.qpose_archive,
        out=args.out,
        block_sizes=_parse_ints(args.block_sizes),
        searches=_parse_ints(args.searches),
        step=int(args.step),
        sparse_thresholds=_parse_ints(args.sparse_thresholds),
        sparse_codecs=[item.strip() for item in args.sparse_codecs.split(",") if item.strip()],
        compressors=compressors,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
