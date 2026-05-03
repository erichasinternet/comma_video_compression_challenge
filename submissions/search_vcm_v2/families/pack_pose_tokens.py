#!/usr/bin/env python
"""Pose-token quantization helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import brotli
import numpy as np
import torch


def quantize_pose_tokens(tokens: torch.Tensor, *, bits: int = 8) -> tuple[torch.Tensor, dict]:
    if bits != 8:
        raise ValueError("v2 currently supports int8 token packing only")
    x = tokens.detach().float().cpu()
    min_v = x.amin(dim=0)
    max_v = x.amax(dim=0)
    scale = (max_v - min_v).clamp_min(1e-8) / 255.0
    q = torch.round((x - min_v) / scale).clamp(0, 255).to(torch.uint8)
    meta = {"bits": bits, "shape": list(x.shape), "min": min_v.tolist(), "scale": scale.tolist()}
    return q, meta


def dequantize_pose_tokens(q: torch.Tensor, meta: dict) -> torch.Tensor:
    min_v = torch.tensor(meta["min"], dtype=torch.float32)
    scale = torch.tensor(meta["scale"], dtype=torch.float32)
    return q.float() * scale + min_v


def pack_pose_tokens(tokens: torch.Tensor) -> bytes:
    q, meta = quantize_pose_tokens(tokens)
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = len(meta_bytes).to_bytes(4, "little") + meta_bytes + q.numpy().tobytes()
    return brotli.compress(payload, quality=11)


def unpack_pose_tokens(payload: bytes) -> torch.Tensor:
    raw = brotli.decompress(payload)
    meta_len = int.from_bytes(raw[:4], "little")
    meta = json.loads(raw[4 : 4 + meta_len].decode("utf-8"))
    q = np.frombuffer(raw[4 + meta_len :], dtype=np.uint8).reshape(meta["shape"])
    return dequantize_pose_tokens(torch.from_numpy(q.copy()), meta)


def estimate_pose_token_bytes(num_samples: int, z_dim: int) -> dict:
    dummy = torch.zeros(num_samples, z_dim)
    packed = pack_pose_tokens(dummy)
    raw_bytes = num_samples * z_dim
    return {"num_samples": num_samples, "z_dim": z_dim, "raw_int8_bytes": raw_bytes, "packed_zero_bytes": len(packed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--z-dim", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = estimate_pose_token_bytes(args.num_samples, args.z_dim)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

