#!/usr/bin/env python3
"""Pack a selfcomp `segmap_inference.pt` checkpoint into dcpack format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import brotli
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from unpack_segmap import _deep_equal, pack_payload, unpack_payload, write_dcpack


def _count_tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_count_tensor_bytes(v) for v in value.values())
    return 0


def pack_checkpoint(input_path: Path, out_path: Path, *, raw_out: Path | None = None) -> dict:
    payload = torch.load(input_path, map_location="cpu")
    raw = pack_payload(payload)
    compressed = brotli.compress(raw, quality=11, lgwin=24)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(compressed)
    if raw_out is not None:
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_bytes(raw)

    roundtrip = unpack_payload(brotli.decompress(compressed))
    _deep_equal(payload, roundtrip)

    return {
        "input": str(input_path),
        "output": str(out_path),
        "raw_dcpack_bytes": len(raw),
        "brotli_dcpack_bytes": len(compressed),
        "torch_checkpoint_bytes": input_path.stat().st_size,
        "raw_tensor_bytes": _count_tensor_bytes(payload),
        "compression_ratio_vs_pt": len(compressed) / input_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input segmap_inference.pt")
    parser.add_argument("--out", type=Path, required=True, help="Output segmap.dcpack.br")
    parser.add_argument("--raw-out", type=Path, help="Optional uncompressed segmap.dcpack")
    parser.add_argument("--metrics", type=Path, help="Optional JSON metrics path")
    args = parser.parse_args()

    metrics = pack_checkpoint(args.input, args.out, raw_out=args.raw_out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
