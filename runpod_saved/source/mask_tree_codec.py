#!/usr/bin/env python
"""Boundary-preserving mask codec for the Quantizr inflater.

This codec is optimized for the fixed 600-frame mask side-channel used by the
challenge submission, not for general image compression. It emits three archive
payloads consumed directly by inflate.py:

  mask_tree_meta.json.br
  mask_tree_tokens.bin.br
  mask_tree_codebook.bin.br
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np
import torch


MASK_TREE_META_PAYLOAD_NAME = "mask_tree_meta.json.br"
MASK_TREE_TOKENS_PAYLOAD_NAME = "mask_tree_tokens.bin.br"
MASK_TREE_CODEBOOK_PAYLOAD_NAME = "mask_tree_codebook.bin.br"

MODE_PREV = 1
MODE_UNIFORM = 2
MODE_LEFT = 3
MODE_ABOVE = 4
MODE_TWO_CLASS = 5
MODE_DICT8 = 7
MODE_DICT16 = 8
MODE_RAW_RLE = 9
MODE_SPLIT = 10


class ByteWriter:
    def __init__(self):
        self.buf = bytearray()

    def u8(self, value: int):
        self.buf.append(int(value) & 0xFF)

    def u16(self, value: int):
        self.buf.extend(struct.pack("<H", int(value)))

    def u32(self, value: int):
        self.buf.extend(struct.pack("<I", int(value)))

    def bytes(self, value: bytes):
        self.buf.extend(value)

    def finish(self) -> bytes:
        return bytes(self.buf)


class ByteReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def u8(self) -> int:
        value = self.data[self.pos]
        self.pos += 1
        return int(value)

    def u16(self) -> int:
        value = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return int(value)

    def u32(self) -> int:
        value = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return int(value)

    def bytes(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out


@dataclass
class Node:
    mode: int
    recon: np.ndarray
    err: float
    est_bytes: int
    cost: float
    payload: object = None
    children: list["Node"] | None = None


def brotli_write(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(brotli.compress(payload, quality=11, lgwin=24))


def brotli_read(path: Path) -> bytes:
    return brotli.decompress(path.read_bytes())


def load_masks(path: Path) -> np.ndarray:
    if path.is_dir():
        candidates = [
            path / "reconstructed_masks.pt",
            *sorted(path.glob("mask_frames*.pt")),
            *sorted(path.glob("*.pt")),
            *sorted(path.glob("*.npy")),
            *sorted(path.glob("*.npz")),
        ]
        candidates = [p for p in candidates if p.exists()]
        if not candidates:
            raise FileNotFoundError(f"No mask tensor found under {path}")
        return load_masks(candidates[0])
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            key = "masks" if "masks" in data else data.files[0]
            arr = data[key]
    else:
        arr = torch.load(path, map_location="cpu").numpy()
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected mask tensor [frames,h,w], got {arr.shape} from {path}")
    if arr.min() < 0 or arr.max() > 4:
        raise ValueError(f"Expected class IDs 0..4 in {path}, got min={arr.min()} max={arr.max()}")
    return np.ascontiguousarray(arr)


def boundary_weight(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=np.float32)
    boundary[1:, :] = np.maximum(boundary[1:, :], mask[1:, :] != mask[:-1, :])
    boundary[:-1, :] = np.maximum(boundary[:-1, :], mask[1:, :] != mask[:-1, :])
    boundary[:, 1:] = np.maximum(boundary[:, 1:], mask[:, 1:] != mask[:, :-1])
    boundary[:, :-1] = np.maximum(boundary[:, :-1], mask[:, 1:] != mask[:, :-1])
    return 1.0 + (4.0 * boundary)


def tile_matrix(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape
    return (
        frame.reshape(h // size, size, w // size, size)
        .transpose(0, 2, 1, 3)
        .reshape(h // size, w // size, size * size)
    )


def untile_matrix(tiles: np.ndarray, height: int, width: int, size: int) -> np.ndarray:
    return (
        tiles.reshape(height // size, width // size, size, size)
        .transpose(0, 2, 1, 3)
        .reshape(height, width)
    )


def rle_byte_estimates(tiles: np.ndarray, size: int) -> np.ndarray:
    rows = tiles.reshape(-1, size, size)
    runs = 1 + (rows[:, :, 1:] != rows[:, :, :-1]).sum(axis=2)
    return (1 + (2 * size) + (3 * runs.sum(axis=1))).astype(np.int32)


def encode_rle_tile_bytes(tile: np.ndarray) -> bytes:
    writer = ByteWriter()
    h, w = tile.shape
    for y in range(h):
        row = tile[y]
        runs = []
        x = 0
        while x < w:
            cls = int(row[x])
            end = x + 1
            while end < w and int(row[end]) == cls:
                end += 1
            runs.append((end - x, cls))
            x = end
        writer.u16(len(runs))
        for length, cls in runs:
            writer.u16(length)
            writer.u8(cls)
    return writer.finish()


def write_rle_tile(writer: ByteWriter, tile: np.ndarray):
    writer.bytes(encode_rle_tile_bytes(tile))


def decode_rle_tile(reader: ByteReader, h: int, w: int) -> np.ndarray:
    out = np.empty((h, w), dtype=np.uint8)
    for y in range(h):
        x = 0
        run_count = reader.u16()
        for _ in range(run_count):
            length = reader.u16()
            cls = reader.u8()
            out[y, x : x + length] = cls
            x += length
        if x != w:
            raise ValueError(f"RLE row width mismatch: {x} != {w}")
    return out


def encode_row_rle(masks: np.ndarray) -> tuple[bytes, np.ndarray]:
    writer = ByteWriter()
    writer.bytes(b"MTK1")
    for frame in masks:
        write_rle_tile(writer, frame)
    return writer.finish(), masks.copy()


def decode_row_rle(reader: ByteReader, frames: int, h: int, w: int) -> np.ndarray:
    out = np.empty((frames, h, w), dtype=np.uint8)
    for i in range(frames):
        out[i] = decode_rle_tile(reader, h, w)
    return out


def encode_temporal_rle(masks: np.ndarray) -> tuple[bytes, np.ndarray]:
    frames, h, w = masks.shape
    writer = ByteWriter()
    writer.bytes(b"MTK1")
    write_rle_tile(writer, masks[0])
    for i in range(1, frames):
        cur = masks[i]
        prev = masks[i - 1]
        for y in range(h):
            diff = cur[y] != prev[y]
            runs = []
            x = 0
            while x < w:
                while x < w and not diff[x]:
                    x += 1
                if x >= w:
                    break
                start = x
                while x < w and diff[x]:
                    x += 1
                segment = cur[y, start:x]
                if np.all(segment == segment[0]):
                    runs.append((start, x - start, 0, int(segment[0]), b""))
                else:
                    runs.append((start, x - start, 1, 0, segment.tobytes()))
            writer.u16(len(runs))
            for start, length, kind, cls, raw in runs:
                writer.u16(start)
                writer.u16(length)
                writer.u8(kind)
                if kind == 0:
                    writer.u8(cls)
                else:
                    writer.bytes(raw)
    return writer.finish(), masks.copy()


def decode_temporal_rle(reader: ByteReader, frames: int, h: int, w: int) -> np.ndarray:
    out = np.empty((frames, h, w), dtype=np.uint8)
    out[0] = decode_rle_tile(reader, h, w)
    for i in range(1, frames):
        cur = out[i - 1].copy()
        for y in range(h):
            changed_count = reader.u16()
            for _ in range(changed_count):
                start_x = reader.u16()
                length = reader.u16()
                kind = reader.u8()
                if kind == 0:
                    cur[y, start_x : start_x + length] = reader.u8()
                elif kind == 1:
                    cur[y, start_x : start_x + length] = np.frombuffer(reader.bytes(length), dtype=np.uint8)
                else:
                    raise ValueError(f"Unknown temporal run kind: {kind}")
        out[i] = cur
    return out


def pack_bitmap(mask: np.ndarray) -> bytes:
    return np.packbits(mask.reshape(-1).astype(np.uint8), bitorder="little").tobytes()


def unpack_bitmap(bits: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(bits, dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:count].astype(bool)


def weighted_error(tile: np.ndarray, recon: np.ndarray, weight: np.ndarray) -> float:
    return float((weight * (tile != recon)).sum())


def weighted_majority(tile: np.ndarray, weight: np.ndarray, classes: int = 5) -> int:
    scores = np.zeros(classes, dtype=np.float64)
    for cls in range(classes):
        scores[cls] = float(weight[tile == cls].sum())
    return int(scores.argmax())


def top_two_classes(tile: np.ndarray, weight: np.ndarray, classes: int = 5) -> tuple[int, int]:
    scores = np.zeros(classes, dtype=np.float64)
    for cls in range(classes):
        scores[cls] = float(weight[tile == cls].sum())
    order = np.argsort(scores)[::-1]
    return int(order[0]), int(order[1])


def best_dict8(tile: np.ndarray, weight: np.ndarray, codebook: np.ndarray) -> tuple[int, np.ndarray, float]:
    if codebook.shape[0] == 0:
        raise ValueError("empty codebook")
    errs = ((codebook[:, : tile.shape[0], : tile.shape[1]] != tile) * weight[None, :, :]).sum(axis=(1, 2))
    idx = int(np.argmin(errs))
    recon = codebook[idx, : tile.shape[0], : tile.shape[1]].copy()
    return idx, recon, float(errs[idx])


def train_codebook(masks: np.ndarray, k: int, max_patches: int = 250_000) -> np.ndarray:
    if k <= 0:
        return np.zeros((0, 8, 8), dtype=np.uint8)
    frames, h, w = masks.shape
    tiles_y = h // 8
    tiles_x = w // 8
    total = frames * tiles_y * tiles_x
    rng = np.random.default_rng(123)
    sample_count = min(max_patches, total)
    sample_ids = rng.choice(total, size=sample_count, replace=False) if sample_count < total else np.arange(total)
    counts: Counter[bytes] = Counter()
    for sid in sample_ids:
        frame_idx = sid // (tiles_y * tiles_x)
        rem = sid % (tiles_y * tiles_x)
        ty = rem // tiles_x
        tx = rem % tiles_x
        patch = masks[frame_idx, ty * 8 : ty * 8 + 8, tx * 8 : tx * 8 + 8]
        if np.unique(patch).size > 1:
            counts[patch.tobytes()] += 1
    if not counts:
        return np.zeros((0, 8, 8), dtype=np.uint8)
    patches = [np.frombuffer(raw, dtype=np.uint8).reshape(8, 8).copy() for raw, _ in counts.most_common(k)]
    return np.stack(patches).astype(np.uint8)


def make_candidate(mode: int, recon: np.ndarray, err: float, est_bytes: int, lam: float, payload=None) -> Node:
    return Node(mode=mode, recon=recon.astype(np.uint8, copy=False), err=err, est_bytes=est_bytes, cost=err + (lam * est_bytes), payload=payload)


def encode_tree_node(
    tile: np.ndarray,
    weight: np.ndarray,
    prev_frame: np.ndarray | None,
    context: np.ndarray,
    codebook: np.ndarray,
    y: int,
    x: int,
    lam: float,
    min_tile: int,
) -> Node:
    h, w = tile.shape
    candidates: list[Node] = []

    if prev_frame is not None:
        recon = prev_frame[y : y + h, x : x + w].copy()
        candidates.append(make_candidate(MODE_PREV, recon, weighted_error(tile, recon, weight), 1, lam))

    if x >= w:
        recon = context[y : y + h, x - w : x].copy()
        candidates.append(make_candidate(MODE_LEFT, recon, weighted_error(tile, recon, weight), 1, lam))
    if y >= h:
        recon = context[y - h : y, x : x + w].copy()
        candidates.append(make_candidate(MODE_ABOVE, recon, weighted_error(tile, recon, weight), 1, lam))

    cls = weighted_majority(tile, weight)
    recon = np.full((h, w), cls, dtype=np.uint8)
    candidates.append(make_candidate(MODE_UNIFORM, recon, weighted_error(tile, recon, weight), 2, lam, payload=cls))

    if h in (8, 16) and w in (8, 16):
        cls_a, cls_b = top_two_classes(tile, weight)
        use_b = tile == cls_b
        recon = np.where(use_b, cls_b, cls_a).astype(np.uint8)
        est_bytes = 3 + math.ceil(h * w / 8)
        candidates.append(make_candidate(MODE_TWO_CLASS, recon, weighted_error(tile, recon, weight), est_bytes, lam, payload=(cls_a, cls_b, pack_bitmap(use_b))))

    if h == 8 and w == 8 and codebook.shape[0] > 0:
        idx, recon, err = best_dict8(tile, weight, codebook)
        candidates.append(make_candidate(MODE_DICT8, recon, err, 3, lam, payload=idx))

    if h == 16 and w == 16 and codebook.shape[0] > 0:
        recon = np.empty((16, 16), dtype=np.uint8)
        indices = []
        err = 0.0
        for dy in (0, 8):
            for dx in (0, 8):
                idx, sub_recon, sub_err = best_dict8(tile[dy : dy + 8, dx : dx + 8], weight[dy : dy + 8, dx : dx + 8], codebook)
                recon[dy : dy + 8, dx : dx + 8] = sub_recon
                indices.append(idx)
                err += sub_err
        candidates.append(make_candidate(MODE_DICT16, recon, err, 9, lam, payload=indices))

    raw_payload = encode_rle_tile_bytes(tile)
    candidates.append(make_candidate(MODE_RAW_RLE, tile.copy(), 0.0, 1 + len(raw_payload), lam, payload=raw_payload))

    if h > min_tile or w > min_tile:
        split_context = context.copy()
        children = []
        recon = np.empty((h, w), dtype=np.uint8)
        err = 0.0
        est_bytes = 1
        h1 = h // 2
        w1 = w // 2
        for cy, ch, oy in ((y, h1, 0), (y + h1, h - h1, h1)):
            for cx, cw, ox in ((x, w1, 0), (x + w1, w - w1, w1)):
                child = encode_tree_node(
                    tile[oy : oy + ch, ox : ox + cw],
                    weight[oy : oy + ch, ox : ox + cw],
                    prev_frame,
                    split_context,
                    codebook,
                    cy,
                    cx,
                    lam,
                    min_tile,
                )
                split_context[cy : cy + ch, cx : cx + cw] = child.recon
                recon[oy : oy + ch, ox : ox + cw] = child.recon
                children.append(child)
                err += child.err
                est_bytes += child.est_bytes
        candidates.append(Node(MODE_SPLIT, recon, err, est_bytes, err + (lam * est_bytes), children=children))

    return min(candidates, key=lambda node: node.cost)


def write_tree_node(writer: ByteWriter, node: Node):
    writer.u8(node.mode)
    if node.mode == MODE_UNIFORM:
        writer.u8(int(node.payload))
    elif node.mode == MODE_TWO_CLASS:
        cls_a, cls_b, bitmap = node.payload
        writer.u8(cls_a)
        writer.u8(cls_b)
        writer.bytes(bitmap)
    elif node.mode == MODE_DICT8:
        writer.u16(int(node.payload))
    elif node.mode == MODE_DICT16:
        for idx in node.payload:
            writer.u16(int(idx))
    elif node.mode == MODE_RAW_RLE:
        writer.bytes(node.payload)
    elif node.mode == MODE_SPLIT:
        for child in node.children or []:
            write_tree_node(writer, child)


def encode_tree(masks: np.ndarray, codebook: np.ndarray, lam: float, root_tile: int, min_tile: int) -> tuple[bytes, np.ndarray]:
    frames, h, w = masks.shape
    writer = ByteWriter()
    writer.bytes(b"MTK1")
    recon_all = np.empty_like(masks)
    for i in range(frames):
        frame = np.zeros((h, w), dtype=np.uint8)
        prev = recon_all[i - 1] if i > 0 else None
        weight = boundary_weight(masks[i])
        for y in range(0, h, root_tile):
            for x in range(0, w, root_tile):
                th = min(root_tile, h - y)
                tw = min(root_tile, w - x)
                node = encode_tree_node(
                    masks[i, y : y + th, x : x + tw],
                    weight[y : y + th, x : x + tw],
                    prev,
                    frame,
                    codebook,
                    y,
                    x,
                    lam,
                    min_tile,
                )
                write_tree_node(writer, node)
                frame[y : y + th, x : x + tw] = node.recon
        recon_all[i] = frame
    return writer.finish(), recon_all


def direct_block_choices(
    frame: np.ndarray,
    weight: np.ndarray,
    prev_recon: np.ndarray | None,
    codebook: np.ndarray,
    lam: float,
    size: int,
) -> dict[str, np.ndarray]:
    h, w = frame.shape
    tiles = tile_matrix(frame, size)
    weights = tile_matrix(weight, size).astype(np.float32, copy=False)
    shape = tiles.shape[:2]
    flat_tiles = tiles.reshape(-1, size * size)
    flat_weights = weights.reshape(-1, size * size)
    total_weight = flat_weights.sum(axis=1)

    hist = np.empty((flat_tiles.shape[0], 5), dtype=np.float32)
    for cls in range(5):
        hist[:, cls] = (flat_weights * (flat_tiles == cls)).sum(axis=1)

    best_cls = hist.argmax(axis=1).astype(np.uint8)
    best_err = total_weight - hist[np.arange(hist.shape[0]), best_cls]
    best_mode = np.full(flat_tiles.shape[0], MODE_UNIFORM, dtype=np.uint8)
    best_a = best_cls.copy()
    best_b = np.zeros(flat_tiles.shape[0], dtype=np.uint8)
    best_idx = np.zeros(flat_tiles.shape[0], dtype=np.uint16)
    best_est_bytes = np.full(flat_tiles.shape[0], 2, dtype=np.int32)
    best_cost = best_err + (lam * best_est_bytes)

    if prev_recon is not None:
        prev_tiles = tile_matrix(prev_recon, size).reshape(-1, size * size)
        prev_err = (flat_weights * (flat_tiles != prev_tiles)).sum(axis=1)
        prev_bytes = np.ones_like(best_est_bytes)
        prev_cost = prev_err + (lam * prev_bytes)
        take = prev_cost < best_cost
        best_mode[take] = MODE_PREV
        best_err[take] = prev_err[take]
        best_est_bytes[take] = prev_bytes[take]
        best_cost[take] = prev_cost[take]

    order = np.argsort(hist, axis=1)
    class_a = order[:, -1].astype(np.uint8)
    class_b = order[:, -2].astype(np.uint8)
    two_err = total_weight - hist[np.arange(hist.shape[0]), class_a] - hist[np.arange(hist.shape[0]), class_b]
    two_bytes = np.full(flat_tiles.shape[0], 3 + math.ceil((size * size) / 8), dtype=np.int32)
    two_cost = two_err + (lam * two_bytes)
    take = two_cost < best_cost
    best_mode[take] = MODE_TWO_CLASS
    best_a[take] = class_a[take]
    best_b[take] = class_b[take]
    best_err[take] = two_err[take]
    best_est_bytes[take] = two_bytes[take]
    best_cost[take] = two_cost[take]

    if size == 8 and codebook.shape[0] > 0:
        dict_err = np.empty(flat_tiles.shape[0], dtype=np.float32)
        dict_idx = np.empty(flat_tiles.shape[0], dtype=np.uint16)
        code_flat = codebook.reshape(codebook.shape[0], 64)
        chunk = 8192
        for start in range(0, flat_tiles.shape[0], chunk):
            end = min(start + chunk, flat_tiles.shape[0])
            matches = flat_tiles[start:end, None, :] == code_flat[None, :, :]
            score = (matches * flat_weights[start:end, None, :]).sum(axis=2)
            err = total_weight[start:end, None] - score
            idx = err.argmin(axis=1)
            dict_idx[start:end] = idx.astype(np.uint16)
            dict_err[start:end] = err[np.arange(end - start), idx]
        dict_bytes = np.full(flat_tiles.shape[0], 3, dtype=np.int32)
        dict_cost = dict_err + (lam * dict_bytes)
        take = dict_cost < best_cost
        best_mode[take] = MODE_DICT8
        best_idx[take] = dict_idx[take]
        best_err[take] = dict_err[take]
        best_est_bytes[take] = dict_bytes[take]
        best_cost[take] = dict_cost[take]

    raw_bytes = rle_byte_estimates(tiles, size)
    raw_err = np.zeros_like(best_err)
    raw_cost = lam * raw_bytes
    take = raw_cost < best_cost
    best_mode[take] = MODE_RAW_RLE
    best_err[take] = raw_err[take]
    best_est_bytes[take] = raw_bytes[take]
    best_cost[take] = raw_cost[take]

    recon_tiles = np.empty_like(flat_tiles, dtype=np.uint8)
    uniform = best_mode == MODE_UNIFORM
    if uniform.any():
        recon_tiles[uniform] = best_a[uniform, None]
    prev = best_mode == MODE_PREV
    if prev.any() and prev_recon is not None:
        recon_tiles[prev] = tile_matrix(prev_recon, size).reshape(-1, size * size)[prev]
    two = best_mode == MODE_TWO_CLASS
    if two.any():
        src = flat_tiles[two]
        b = best_b[two]
        a = best_a[two]
        recon_tiles[two] = np.where(src == b[:, None], b[:, None], a[:, None]).astype(np.uint8)
    dict_take = best_mode == MODE_DICT8
    if dict_take.any():
        recon_tiles[dict_take] = codebook.reshape(codebook.shape[0], 64)[best_idx[dict_take]]
    raw = best_mode == MODE_RAW_RLE
    if raw.any():
        recon_tiles[raw] = flat_tiles[raw]

    return {
        "mode": best_mode.reshape(shape),
        "a": best_a.reshape(shape),
        "b": best_b.reshape(shape),
        "idx": best_idx.reshape(shape),
        "err": best_err.reshape(shape),
        "bytes": best_est_bytes.reshape(shape),
        "cost": best_cost.reshape(shape),
        "recon": untile_matrix(recon_tiles.reshape(*shape, size * size), h, w, size),
    }


def split_sums(child: np.ndarray) -> np.ndarray:
    h, w = child.shape
    return child.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))


def merge_split_recon(direct_recon: np.ndarray, child_recon: np.ndarray, split_mask: np.ndarray, size: int) -> np.ndarray:
    out = direct_recon.copy()
    for ty, tx in np.argwhere(split_mask):
        y = int(ty) * size
        x = int(tx) * size
        out[y : y + size, x : x + size] = child_recon[y : y + size, x : x + size]
    return out


def apply_root_context_modes(frame: np.ndarray, weight: np.ndarray, selected: dict[str, np.ndarray], lam: float, size: int) -> dict[str, np.ndarray]:
    """Greedily add root-level LEFT/ABOVE modes using decoded context."""
    h, w = frame.shape
    out = np.empty_like(frame)
    tiles = tile_matrix(frame, size).reshape(h // size, w // size, size, size)
    weights = tile_matrix(weight, size).reshape(h // size, w // size, size, size)
    mode = selected["mode"].copy()
    err = selected["err"].copy()
    est_bytes = selected["bytes"].copy()
    cost = selected["cost"].copy()
    recon = selected["recon"].copy()

    for ty in range(h // size):
        for tx in range(w // size):
            y = ty * size
            x = tx * size
            tile = tiles[ty, tx]
            wt = weights[ty, tx]
            best_mode = int(mode[ty, tx])
            best_err = float(err[ty, tx])
            best_bytes = int(est_bytes[ty, tx])
            best_cost = float(cost[ty, tx])
            best_recon = recon[y : y + size, x : x + size].copy()

            if tx > 0:
                candidate = out[y : y + size, x - size : x]
                cand_err = float((wt * (tile != candidate)).sum())
                cand_cost = cand_err + lam
                if cand_cost < best_cost:
                    best_mode = MODE_LEFT
                    best_err = cand_err
                    best_bytes = 1
                    best_cost = cand_cost
                    best_recon = candidate.copy()

            if ty > 0:
                candidate = out[y - size : y, x : x + size]
                cand_err = float((wt * (tile != candidate)).sum())
                cand_cost = cand_err + lam
                if cand_cost < best_cost:
                    best_mode = MODE_ABOVE
                    best_err = cand_err
                    best_bytes = 1
                    best_cost = cand_cost
                    best_recon = candidate.copy()

            mode[ty, tx] = best_mode
            err[ty, tx] = best_err
            est_bytes[ty, tx] = best_bytes
            cost[ty, tx] = best_cost
            out[y : y + size, x : x + size] = best_recon

    updated = dict(selected)
    updated.update({"mode": mode, "err": err, "bytes": est_bytes, "cost": cost, "recon": out})
    return updated


def write_vector_tree_node(
    writer: ByteWriter,
    choices: dict[int, dict[str, np.ndarray]],
    masks: np.ndarray,
    frame_idx: int,
    y: int,
    x: int,
    size: int,
):
    ty = y // size
    tx = x // size
    mode = int(choices[size]["mode"][frame_idx, ty, tx])
    writer.u8(mode)
    if mode == MODE_SPLIT:
        half = size // 2
        for cy in (y, y + half):
            for cx in (x, x + half):
                write_vector_tree_node(writer, choices, masks, frame_idx, cy, cx, half)
    elif mode == MODE_UNIFORM:
        writer.u8(int(choices[size]["a"][frame_idx, ty, tx]))
    elif mode == MODE_TWO_CLASS:
        cls_a = int(choices[size]["a"][frame_idx, ty, tx])
        cls_b = int(choices[size]["b"][frame_idx, ty, tx])
        tile = masks[frame_idx, y : y + size, x : x + size]
        writer.u8(cls_a)
        writer.u8(cls_b)
        writer.bytes(pack_bitmap(tile == cls_b))
    elif mode == MODE_DICT8:
        writer.u16(int(choices[size]["idx"][frame_idx, ty, tx]))
    elif mode == MODE_RAW_RLE:
        write_rle_tile(writer, masks[frame_idx, y : y + size, x : x + size])
    elif mode in (MODE_PREV, MODE_LEFT, MODE_ABOVE):
        pass
    else:
        raise ValueError(f"Unsupported vector tree mode: {mode}")


def encode_tree_vectorized(masks: np.ndarray, codebook: np.ndarray, lam: float, root_tile: int, min_tile: int) -> tuple[bytes, np.ndarray]:
    if min_tile != 8:
        raise ValueError("Vectorized tree currently requires --min-tile 8")
    if root_tile not in (8, 16, 32, 64):
        raise ValueError("Vectorized tree supports root tiles 8, 16, 32, or 64")
    frames, h, w = masks.shape
    sizes = [8, 16, 32, 64]
    sizes = [s for s in sizes if s <= root_tile]
    choices: dict[int, dict[str, np.ndarray]] = {}
    for size in sizes:
        choices[size] = {
            "mode": np.empty((frames, h // size, w // size), dtype=np.uint8),
            "a": np.zeros((frames, h // size, w // size), dtype=np.uint8),
            "b": np.zeros((frames, h // size, w // size), dtype=np.uint8),
            "idx": np.zeros((frames, h // size, w // size), dtype=np.uint16),
        }

    recon_all = np.empty_like(masks)
    for frame_idx in range(frames):
        frame = masks[frame_idx]
        weight = boundary_weight(frame)
        prev_recon = recon_all[frame_idx - 1] if frame_idx > 0 else None

        direct8 = direct_block_choices(frame, weight, prev_recon, codebook, lam, 8)
        choices[8]["mode"][frame_idx] = direct8["mode"]
        choices[8]["a"][frame_idx] = direct8["a"]
        choices[8]["b"][frame_idx] = direct8["b"]
        choices[8]["idx"][frame_idx] = direct8["idx"]
        cost_by_size = {8: direct8["cost"]}
        err_by_size = {8: direct8["err"]}
        bytes_by_size = {8: direct8["bytes"]}
        recon_by_size = {8: direct8["recon"]}

        for size in sizes[1:]:
            child_size = size // 2
            direct = direct_block_choices(frame, weight, prev_recon, np.zeros((0, 8, 8), dtype=np.uint8), lam, size)
            split_cost = split_sums(cost_by_size[child_size]) + lam
            split_err = split_sums(err_by_size[child_size])
            split_bytes = split_sums(bytes_by_size[child_size]) + 1
            split = split_cost < direct["cost"]
            mode = direct["mode"].copy()
            mode[split] = MODE_SPLIT
            choices[size]["mode"][frame_idx] = mode
            choices[size]["a"][frame_idx] = direct["a"]
            choices[size]["b"][frame_idx] = direct["b"]
            choices[size]["idx"][frame_idx] = direct["idx"]
            cost = np.where(split, split_cost, direct["cost"])
            err = np.where(split, split_err, direct["err"])
            est_bytes = np.where(split, split_bytes, direct["bytes"])
            recon = merge_split_recon(direct["recon"], recon_by_size[child_size], split, size)
            cost_by_size[size] = cost
            err_by_size[size] = err
            bytes_by_size[size] = est_bytes
            recon_by_size[size] = recon

        root_selected = {
            "mode": choices[root_tile]["mode"][frame_idx],
            "a": choices[root_tile]["a"][frame_idx],
            "b": choices[root_tile]["b"][frame_idx],
            "idx": choices[root_tile]["idx"][frame_idx],
            "err": err_by_size[root_tile],
            "bytes": bytes_by_size[root_tile],
            "cost": cost_by_size[root_tile],
            "recon": recon_by_size[root_tile],
        }
        root_selected = apply_root_context_modes(frame, weight, root_selected, lam, root_tile)
        choices[root_tile]["mode"][frame_idx] = root_selected["mode"]
        choices[root_tile]["a"][frame_idx] = root_selected["a"]
        choices[root_tile]["b"][frame_idx] = root_selected["b"]
        choices[root_tile]["idx"][frame_idx] = root_selected["idx"]
        recon_by_size[root_tile] = root_selected["recon"]
        recon_all[frame_idx] = recon_by_size[root_tile]

    writer = ByteWriter()
    writer.bytes(b"MTK1")
    for frame_idx in range(frames):
        for y in range(0, h, root_tile):
            for x in range(0, w, root_tile):
                write_vector_tree_node(writer, choices, masks, frame_idx, y, x, root_tile)
    return writer.finish(), recon_all


def decode_tree_node(reader: ByteReader, out: np.ndarray, prev: np.ndarray | None, codebook: np.ndarray, y: int, x: int, h: int, w: int):
    mode = reader.u8()
    if mode == MODE_PREV:
        if prev is None:
            raise ValueError("PREV mode used on first frame")
        out[y : y + h, x : x + w] = prev[y : y + h, x : x + w]
    elif mode == MODE_UNIFORM:
        out[y : y + h, x : x + w] = reader.u8()
    elif mode == MODE_LEFT:
        out[y : y + h, x : x + w] = out[y : y + h, x - w : x]
    elif mode == MODE_ABOVE:
        out[y : y + h, x : x + w] = out[y - h : y, x : x + w]
    elif mode == MODE_TWO_CLASS:
        cls_a = reader.u8()
        cls_b = reader.u8()
        bits = unpack_bitmap(reader.bytes((h * w + 7) // 8), h * w).reshape(h, w)
        out[y : y + h, x : x + w] = np.where(bits, cls_b, cls_a).astype(np.uint8)
    elif mode == MODE_DICT8:
        idx = reader.u16()
        out[y : y + h, x : x + w] = codebook[idx, :h, :w]
    elif mode == MODE_DICT16:
        for dy in (0, 8):
            for dx in (0, 8):
                idx = reader.u16()
                out[y + dy : y + dy + 8, x + dx : x + dx + 8] = codebook[idx]
    elif mode == MODE_RAW_RLE:
        out[y : y + h, x : x + w] = decode_rle_tile(reader, h, w)
    elif mode == MODE_SPLIT:
        h1 = h // 2
        w1 = w // 2
        for cy, ch in ((y, h1), (y + h1, h - h1)):
            for cx, cw in ((x, w1), (x + w1, w - w1)):
                decode_tree_node(reader, out, prev, codebook, cy, cx, ch, cw)
    else:
        raise ValueError(f"Unknown tree mode: {mode}")


def decode_tree(reader: ByteReader, frames: int, h: int, w: int, root_tile: int, codebook: np.ndarray) -> np.ndarray:
    out = np.empty((frames, h, w), dtype=np.uint8)
    for i in range(frames):
        frame = np.zeros((h, w), dtype=np.uint8)
        prev = out[i - 1] if i > 0 else None
        for y in range(0, h, root_tile):
            for x in range(0, w, root_tile):
                decode_tree_node(reader, frame, prev, codebook, y, x, min(root_tile, h - y), min(root_tile, w - x))
        out[i] = frame
    return out


def append_residual_stream(tokens: bytes, source: np.ndarray, reconstructed: np.ndarray, budget: int) -> tuple[bytes, np.ndarray, int]:
    if budget <= 0:
        return tokens, reconstructed, 0

    candidates = []
    frames, h, w = source.shape
    for frame_idx in range(frames):
        weight = boundary_weight(source[frame_idx])
        diff = source[frame_idx] != reconstructed[frame_idx]
        for y in range(h):
            x = 0
            while x < w:
                while x < w and not diff[y, x]:
                    x += 1
                if x >= w:
                    break
                start = x
                while x < w and diff[y, x]:
                    x += 1
                raw = source[frame_idx, y, start:x].tobytes()
                est_bytes = 8 + len(raw)
                gain = float(weight[y, start:x].sum())
                candidates.append((gain / est_bytes, gain, est_bytes, frame_idx, y, start, raw))

    candidates.sort(reverse=True)
    selected = []
    used = 8
    corrected = reconstructed.copy()
    for _, _, est_bytes, frame_idx, y, start, raw in candidates:
        if used + est_bytes > budget:
            continue
        selected.append((frame_idx, y, start, raw))
        length = len(raw)
        corrected[frame_idx, y, start : start + length] = np.frombuffer(raw, dtype=np.uint8)
        used += est_bytes

    writer = ByteWriter()
    writer.bytes(tokens)
    writer.bytes(b"RS1\0")
    writer.u32(len(selected))
    for frame_idx, y, start, raw in selected:
        writer.u16(frame_idx)
        writer.u16(y)
        writer.u16(start)
        writer.u16(len(raw))
        writer.bytes(raw)
    return writer.finish(), corrected, len(selected)


def apply_residual_stream(reader: ByteReader, masks: np.ndarray) -> np.ndarray:
    if reader.pos >= len(reader.data):
        return masks
    if reader.bytes(4) != b"RS1\0":
        raise ValueError("Invalid residual stream magic")
    out = masks.copy()
    count = reader.u32()
    for _ in range(count):
        frame_idx = reader.u16()
        y = reader.u16()
        start = reader.u16()
        length = reader.u16()
        out[frame_idx, y, start : start + length] = np.frombuffer(reader.bytes(length), dtype=np.uint8)
    return out


def decode_payload(codec_dir: Path) -> np.ndarray:
    meta = json.loads(brotli_read(codec_dir / MASK_TREE_META_PAYLOAD_NAME).decode("utf-8"))
    token_payload = brotli_read(codec_dir / MASK_TREE_TOKENS_PAYLOAD_NAME)
    codebook_payload = brotli_read(codec_dir / MASK_TREE_CODEBOOK_PAYLOAD_NAME)
    codebook = np.frombuffer(codebook_payload, dtype=np.uint8).reshape(-1, 8, 8) if codebook_payload else np.zeros((0, 8, 8), dtype=np.uint8)
    reader = ByteReader(token_payload)
    if reader.bytes(4) != b"MTK1":
        raise ValueError("Invalid token stream magic")
    frames = int(meta["frames"])
    h = int(meta["height"])
    w = int(meta["width"])
    if meta["variant"] == "row_rle":
        masks = decode_row_rle(reader, frames, h, w)
    elif meta["variant"] == "temporal_rle":
        masks = decode_temporal_rle(reader, frames, h, w)
    elif meta["variant"] == "tree":
        masks = decode_tree(reader, frames, h, w, int(meta.get("root_tile", 64)), codebook)
        masks = apply_residual_stream(reader, masks)
    else:
        raise ValueError(f"Unknown variant: {meta['variant']}")
    crc = f"{zlib.crc32(masks.tobytes()) & 0xFFFFFFFF:08x}"
    if meta.get("recon_crc32") and meta["recon_crc32"] != crc:
        raise ValueError(f"Decoded mask CRC mismatch: {crc} != {meta['recon_crc32']}")
    return masks


def write_codec(out_dir: Path, masks: np.ndarray, reconstructed: np.ndarray, tokens: bytes, codebook: np.ndarray, args):
    frames, h, w = masks.shape
    stored_variant = "tree" if args.variant == "tree_recursive" else args.variant
    meta = {
        "version": 1,
        "variant": stored_variant,
        "frames": int(frames),
        "height": int(h),
        "width": int(w),
        "classes": 5,
        "patch": 8,
        "root_tile": int(args.root_tile),
        "min_tile": int(args.min_tile),
        "lambda": float(args.lam),
        "residual_budget": int(args.residual_budget),
        "residual_runs": int(getattr(args, "_residual_runs", 0)),
        "codebook_size": int(codebook.shape[0]),
        "source_crc32": f"{zlib.crc32(masks.tobytes()) & 0xFFFFFFFF:08x}",
        "recon_crc32": f"{zlib.crc32(reconstructed.tobytes()) & 0xFFFFFFFF:08x}",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    brotli_write(out_dir / MASK_TREE_META_PAYLOAD_NAME, json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    brotli_write(out_dir / MASK_TREE_TOKENS_PAYLOAD_NAME, tokens)
    brotli_write(out_dir / MASK_TREE_CODEBOOK_PAYLOAD_NAME, codebook.astype(np.uint8).tobytes())
    torch.save(torch.from_numpy(reconstructed).contiguous(), out_dir / "reconstructed_masks.pt")
    (out_dir / "mask_tree_report.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "payload_sizes": {
                    MASK_TREE_META_PAYLOAD_NAME: (out_dir / MASK_TREE_META_PAYLOAD_NAME).stat().st_size,
                    MASK_TREE_TOKENS_PAYLOAD_NAME: (out_dir / MASK_TREE_TOKENS_PAYLOAD_NAME).stat().st_size,
                    MASK_TREE_CODEBOOK_PAYLOAD_NAME: (out_dir / MASK_TREE_CODEBOOK_PAYLOAD_NAME).stat().st_size,
                },
                "mask_error_rate": float((masks != reconstructed).mean()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_fit(args):
    masks = load_masks(args.mask_cache)
    args._residual_runs = 0
    if args.variant == "row_rle":
        codebook = np.zeros((0, 8, 8), dtype=np.uint8)
        tokens, reconstructed = encode_row_rle(masks)
    elif args.variant == "temporal_rle":
        codebook = np.zeros((0, 8, 8), dtype=np.uint8)
        tokens, reconstructed = encode_temporal_rle(masks)
    elif args.variant == "tree":
        codebook = train_codebook(masks, args.codebook_size, max_patches=args.max_codebook_patches)
        tokens, reconstructed = encode_tree_vectorized(masks, codebook, args.lam, args.root_tile, args.min_tile)
        tokens, reconstructed, args._residual_runs = append_residual_stream(tokens, masks, reconstructed, args.residual_budget)
    elif args.variant == "tree_recursive":
        codebook = train_codebook(masks, args.codebook_size, max_patches=args.max_codebook_patches)
        tokens, reconstructed = encode_tree(masks, codebook, args.lam, args.root_tile, args.min_tile)
        tokens, reconstructed, args._residual_runs = append_residual_stream(tokens, masks, reconstructed, args.residual_budget)
    else:
        raise ValueError(args.variant)
    write_codec(args.out, masks, reconstructed, tokens, codebook, args)
    report = json.loads((args.out / "mask_tree_report.json").read_text())
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_decode(args):
    masks = decode_payload(args.codec_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".npy":
        np.save(args.out, masks)
    else:
        torch.save(torch.from_numpy(masks).contiguous(), args.out)
    print(json.dumps({"out": str(args.out), "shape": list(masks.shape), "crc32": f"{zlib.crc32(masks.tobytes()) & 0xFFFFFFFF:08x}"}, sort_keys=True))


def cmd_inspect(args):
    meta = json.loads(brotli_read(args.codec_dir / MASK_TREE_META_PAYLOAD_NAME).decode("utf-8"))
    payload_sizes = {
        name: (args.codec_dir / name).stat().st_size
        for name in (MASK_TREE_META_PAYLOAD_NAME, MASK_TREE_TOKENS_PAYLOAD_NAME, MASK_TREE_CODEBOOK_PAYLOAD_NAME)
        if (args.codec_dir / name).exists()
    }
    print(json.dumps({"meta": meta, "payload_sizes": payload_sizes, "total_payload": sum(payload_sizes.values())}, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--mask-cache", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--variant", choices=["row_rle", "temporal_rle", "tree", "tree_recursive"], default="temporal_rle")
    fit.add_argument("--codebook-size", type=int, default=128)
    fit.add_argument("--max-codebook-patches", type=int, default=250_000)
    fit.add_argument("--lambda", dest="lam", type=float, default=0.2)
    fit.add_argument("--root-tile", type=int, default=64)
    fit.add_argument("--min-tile", type=int, default=8)
    fit.add_argument("--residual-budget", type=int, default=0, help="Accepted for matrix bookkeeping; v1 tree uses lambda RD selection.")
    fit.set_defaults(func=cmd_fit)

    dec = sub.add_parser("decode")
    dec.add_argument("--codec-dir", type=Path, required=True)
    dec.add_argument("--out", type=Path, required=True)
    dec.set_defaults(func=cmd_decode)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--codec-dir", type=Path, required=True)
    inspect.set_defaults(func=cmd_inspect)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
