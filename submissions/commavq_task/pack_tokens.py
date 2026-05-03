#!/usr/bin/env python3
"""Pack commaVQ uint10 tokens into a prototype archive."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import brotli
import numpy as np


def pack_uint10(values: np.ndarray) -> bytes:
    vals = values.astype(np.uint16).reshape(-1)
    if vals.size and int(vals.max()) >= 1024:
        raise ValueError(f"uint10 token out of range: max={int(vals.max())}")
    out = bytearray()
    acc = 0
    bits = 0
    for value in vals:
        acc |= int(value) << bits
        bits += 10
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_uint10(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.uint16)
    acc = 0
    bits = 0
    idx = 0
    for byte in data:
        acc |= int(byte) << bits
        bits += 8
        while bits >= 10 and idx < count:
            out[idx] = acc & 0x3FF
            acc >>= 10
            bits -= 10
            idx += 1
    if idx != count:
        raise ValueError(f"decoded {idx} tokens, expected {count}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", type=int, default=11)
    args = parser.parse_args()

    tokens = np.load(args.tokens).astype(np.uint16)
    packed = pack_uint10(tokens)
    if not np.array_equal(unpack_uint10(packed, tokens.size).reshape(tokens.shape), tokens):
        raise RuntimeError("uint10 pack/unpack roundtrip failed")
    compressed = brotli.compress(packed, quality=args.quality)
    sample_ids = json.loads(args.sample_ids.read_text()) if args.sample_ids else list(range(tokens.shape[0]))
    meta = {
        "shape": list(tokens.shape),
        "dtype": "uint10",
        "sample_ids": sample_ids,
        "token_count": int(tokens.size),
        "vocab_size": 1024,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tokens.uint10.br", compressed)
        zf.writestr("tokens_meta.json", json.dumps(meta, sort_keys=True).encode())
    metrics = {
        "tokens": str(args.tokens),
        "archive": str(args.out),
        "shape": list(tokens.shape),
        "raw_uint16_bytes": int(tokens.nbytes),
        "raw_uint10_bytes": len(packed),
        "brotli_bytes": len(compressed),
        "archive_bytes": args.out.stat().st_size,
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

