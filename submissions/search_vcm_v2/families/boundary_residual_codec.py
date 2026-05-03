#!/usr/bin/env python
"""Sparse boundary residual codec diagnostics for lowmask/qpose audits."""

from __future__ import annotations

import json
import lzma
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import brotli
import numpy as np
import torch
import torch.nn.functional as F


def boundary_map(classes: torch.Tensor) -> torch.Tensor:
    """Return [T,H,W] bool map where any 4-neighbor has a different class."""

    if classes.ndim != 3:
        raise ValueError(f"expected [T,H,W], got {tuple(classes.shape)}")
    out = torch.zeros_like(classes, dtype=torch.bool)
    diff_y = classes[:, 1:, :] != classes[:, :-1, :]
    diff_x = classes[:, :, 1:] != classes[:, :, :-1]
    out[:, 1:, :] |= diff_y
    out[:, :-1, :] |= diff_y
    out[:, :, 1:] |= diff_x
    out[:, :, :-1] |= diff_x
    return out


def dilate_bool(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[:, 0].bool()


def boundary_band(classes: torch.Tensor, radius: int) -> torch.Tensor:
    return dilate_bool(boundary_map(classes), radius)


def pack_bits(mask: np.ndarray) -> bytes:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    return np.packbits(flat, bitorder="little").tobytes()


def unpack_bits(payload: bytes, count: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:count].astype(bool)


def encode_varints(values: Iterable[int]) -> bytes:
    out = bytearray()
    for value in values:
        v = int(value)
        if v < 0:
            raise ValueError("varint cannot encode negative values")
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def decode_varints(payload: bytes) -> list[int]:
    values = []
    shift = 0
    value = 0
    for byte in payload:
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            values.append(value)
            value = 0
            shift = 0
        else:
            shift += 7
    if shift:
        raise ValueError("truncated varint stream")
    return values


@dataclass(frozen=True)
class TileResidualRecord:
    frame: int
    tile_y: int
    tile_x: int
    bitmap: bytes
    classes: bytes
    changed_pixels: int


def tile_grid(shape: tuple[int, int, int], tile_size: int) -> tuple[int, int]:
    _, height, width = shape
    if height % tile_size or width % tile_size:
        raise ValueError(f"tile_size={tile_size} does not divide {(height, width)}")
    return height // tile_size, width // tile_size


def make_tile_records(
    exact: torch.Tensor,
    low: torch.Tensor,
    selected: list[tuple[int, int, int]],
    *,
    tile_size: int,
) -> list[TileResidualRecord]:
    records = []
    exact_np = exact.cpu().numpy().astype(np.uint8, copy=False)
    low_np = low.cpu().numpy().astype(np.uint8, copy=False)
    for frame, tile_y, tile_x in selected:
        y0 = tile_y * tile_size
        x0 = tile_x * tile_size
        exact_tile = exact_np[frame, y0 : y0 + tile_size, x0 : x0 + tile_size]
        low_tile = low_np[frame, y0 : y0 + tile_size, x0 : x0 + tile_size]
        diff = exact_tile != low_tile
        changed = int(diff.sum())
        if changed == 0:
            continue
        records.append(
            TileResidualRecord(
                frame=int(frame),
                tile_y=int(tile_y),
                tile_x=int(tile_x),
                bitmap=pack_bits(diff),
                classes=exact_tile[diff].astype(np.uint8).tobytes(),
                changed_pixels=changed,
            )
        )
    records.sort(key=lambda rec: (rec.frame, rec.tile_y, rec.tile_x))
    return records


def pack_records(records: list[TileResidualRecord], *, shape: tuple[int, int, int], tile_size: int) -> dict[str, bytes]:
    tiles_y, tiles_x = tile_grid(shape, tile_size)
    ids = [rec.frame * tiles_y * tiles_x + rec.tile_y * tiles_x + rec.tile_x for rec in records]
    prev = 0
    deltas = []
    for idx, tile_id in enumerate(ids):
        deltas.append(tile_id if idx == 0 else tile_id - prev)
        prev = tile_id
    meta = {
        "shape": list(shape),
        "tile_size": tile_size,
        "record_count": len(records),
        "changed_pixels": sum(rec.changed_pixels for rec in records),
    }
    return {
        "meta.json": json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        "tile_deltas.bin": encode_varints(deltas),
        "modes.bin": bytes([2]) * len(records),
        "bitmaps.bin": b"".join(rec.bitmap for rec in records),
        "sparse_classes.bin": b"".join(rec.classes for rec in records),
    }


def unpack_records(streams: dict[str, bytes]) -> list[TileResidualRecord]:
    meta = json.loads(streams["meta.json"].decode("utf-8"))
    shape = tuple(meta["shape"])
    tile_size = int(meta["tile_size"])
    tiles_y, tiles_x = tile_grid(shape, tile_size)
    deltas = decode_varints(streams["tile_deltas.bin"])
    bitmap_bytes = (tile_size * tile_size + 7) // 8
    bitmaps = streams["bitmaps.bin"]
    classes = streams["sparse_classes.bin"]
    records = []
    tile_id = 0
    class_cursor = 0
    for i, delta in enumerate(deltas):
        tile_id = tile_id + delta if i else delta
        frame = tile_id // (tiles_y * tiles_x)
        rem = tile_id % (tiles_y * tiles_x)
        tile_y = rem // tiles_x
        tile_x = rem % tiles_x
        bitmap = bitmaps[i * bitmap_bytes : (i + 1) * bitmap_bytes]
        changed = int(unpack_bits(bitmap, tile_size * tile_size).sum())
        cls = classes[class_cursor : class_cursor + changed]
        class_cursor += changed
        records.append(TileResidualRecord(frame, tile_y, tile_x, bitmap, cls, changed))
    return records


def apply_records(base: torch.Tensor, records: list[TileResidualRecord], *, tile_size: int) -> torch.Tensor:
    out = base.clone()
    for rec in records:
        y0 = rec.tile_y * tile_size
        x0 = rec.tile_x * tile_size
        mask = torch.from_numpy(unpack_bits(rec.bitmap, tile_size * tile_size).reshape(tile_size, tile_size)).to(out.device)
        classes = torch.from_numpy(np.frombuffer(rec.classes, dtype=np.uint8).copy()).to(out.device)
        tile = out[rec.frame, y0 : y0 + tile_size, x0 : x0 + tile_size]
        tile[mask] = classes.to(tile.dtype)
    return out


def _external_compress(name: str, data: bytes) -> bytes | None:
    exe = shutil.which(name)
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "stream.bin"
        src.write_bytes(data)
        if name == "zstd":
            dst = Path(td) / "stream.bin.zst"
            cmd = [exe, "--ultra", "-22", "-q", "-f", str(src), "-o", str(dst)]
        elif name == "zpaq":
            dst = Path(td) / "stream.zpaq"
            cmd = [exe, "a", str(dst), str(src), "-method", "5"]
        else:
            return None
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return dst.read_bytes()
        except Exception:
            return None


def compress_bytes(data: bytes, compressors: tuple[str, ...] = ("brotli", "xz", "zstd")) -> dict:
    candidates: dict[str, int] = {"raw": len(data), "zlib9": len(zlib.compress(data, 9))}
    if "brotli" in compressors:
        candidates["brotli"] = len(brotli.compress(data, quality=11, lgwin=24))
    if "xz" in compressors:
        candidates["xz"] = len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))
    for external in ("zstd", "zpaq"):
        if external in compressors:
            payload = _external_compress(external, data)
            if payload is not None:
                candidates[external] = len(payload)
    best_name = min(candidates, key=candidates.get)
    return {"raw_bytes": len(data), "best": best_name, "best_bytes": candidates[best_name], "candidates": candidates}


def compress_streams(streams: dict[str, bytes], compressors: tuple[str, ...] = ("brotli", "xz", "zstd")) -> dict:
    breakdown = {name: compress_bytes(data, compressors) for name, data in streams.items()}
    return {
        "streams": breakdown,
        "total_raw_bytes": sum(len(data) for data in streams.values()),
        "total_best_bytes": sum(item["best_bytes"] for item in breakdown.values()),
    }


def candidate_from_records(
    *,
    name: str,
    records: list[TileResidualRecord],
    shape: tuple[int, int, int],
    tile_size: int,
    compressors: tuple[str, ...] = ("brotli", "xz", "zstd"),
) -> dict:
    streams = pack_records(records, shape=shape, tile_size=tile_size)
    compressed = compress_streams(streams, compressors)
    return {
        "candidate": name,
        "tile_size": tile_size,
        "selected_tiles": len(records),
        "source_error_pixels": sum(rec.changed_pixels for rec in records),
        "residual_bytes": int(compressed["total_best_bytes"]),
        "compressor_breakdown": compressed,
    }
