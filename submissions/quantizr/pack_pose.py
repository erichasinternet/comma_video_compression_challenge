#!/usr/bin/env python
"""Pack Quantizr pose.npy.br into a compact inflate-compatible pose.qpack.br."""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import brotli
import numpy as np


POSE_PAYLOAD_NAME = "pose.npy.br"
POSE_QPACK_PAYLOAD_NAME = "pose.qpack.br"


def load_pose_payload(path: Path) -> np.ndarray:
    return np.load(io.BytesIO(brotli.decompress(path.read_bytes()))).astype(np.float32, copy=False)


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    flat = values.astype(np.uint16, copy=False).reshape(-1)
    out = bytearray((flat.size * bits + 7) // 8)
    acc = 0
    acc_bits = 0
    j = 0
    mask = (1 << bits) - 1
    for value in flat:
        acc |= (int(value) & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out[j] = acc & 0xFF
            j += 1
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out[j] = acc & 0xFF
    return bytes(out)


def build_pose_qpack(pose: np.ndarray, variant: str) -> bytes:
    pose = np.ascontiguousarray(pose.astype(np.float32, copy=False))
    header = {
        "format": "quantizr_pose_qpack_v1",
        "shape": [int(x) for x in pose.shape],
        "variant": variant,
    }
    if variant == "fp16":
        body = pose.astype("<f2").tobytes(order="C")
        header["kind"] = "fp16"
    elif variant == "int16_per_dim":
        lo = pose.min(axis=0).astype(np.float32)
        hi = pose.max(axis=0).astype(np.float32)
        scale = np.maximum(hi - lo, np.float32(1e-12))
        q = np.clip(np.round((pose - lo[None, :]) / scale[None, :] * 65535.0), 0, 65535).astype("<u2")
        body = q.tobytes(order="C")
        header.update({"kind": "uint16_per_dim", "bits": 16, "min": lo.tolist(), "max": hi.tolist()})
    elif variant in {"int12_per_dim", "int10_per_dim"}:
        bits = 12 if variant.startswith("int12") else 10
        lo = pose.min(axis=0).astype(np.float32)
        hi = pose.max(axis=0).astype(np.float32)
        levels = float((1 << bits) - 1)
        scale = np.maximum(hi - lo, np.float32(1e-12))
        q = np.clip(np.round((pose - lo[None, :]) / scale[None, :] * levels), 0, levels).astype(np.uint16)
        body = pack_bits(q, bits)
        header.update({"kind": "packed_uint_per_dim", "bits": bits, "min": lo.tolist(), "max": hi.tolist()})
    else:
        raise ValueError(f"unknown pose pack variant: {variant}")

    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"PQP1" + struct.pack("<I", len(header_bytes)) + header_bytes + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--pose-br", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--variant", choices=["fp16", "int16_per_dim", "int12_per_dim", "int10_per_dim"], required=True)
    args = parser.parse_args()

    pose_br = args.pose_br
    if pose_br is None:
        if args.archive_dir is None:
            raise ValueError("Pass --archive-dir or --pose-br")
        pose_br = args.archive_dir / POSE_PAYLOAD_NAME
    out = args.out
    if out is None:
        if args.archive_dir is None:
            raise ValueError("--out is required without --archive-dir")
        out = args.archive_dir / POSE_QPACK_PAYLOAD_NAME

    pose = load_pose_payload(pose_br)
    qpack = build_pose_qpack(pose, args.variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(brotli.compress(qpack, quality=11, lgwin=24))

    report = {
        "out": str(out),
        "variant": args.variant,
        "shape": [int(x) for x in pose.shape],
        "pose_npy_br_bytes": pose_br.stat().st_size,
        "pose_qpack_br_bytes": out.stat().st_size,
        "saved_bytes": pose_br.stat().st_size - out.stat().st_size,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
