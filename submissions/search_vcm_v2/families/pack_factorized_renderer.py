#!/usr/bin/env python
"""Renderer byte-estimation and simple int8 state packing."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import brotli
import torch

from submissions.search_vcm_v2.families.factorized_renderer import build_renderer


def int8_state_payload(model: torch.nn.Module) -> bytes:
    records = {}
    for name, tensor in model.state_dict().items():
        x = tensor.detach().cpu()
        if torch.is_floating_point(x):
            max_abs = x.abs().max().clamp_min(1e-8)
            q = torch.round(x / max_abs * 127.0).clamp(-127, 127).to(torch.int8)
            records[name] = {"kind": "int8_symmetric", "shape": list(x.shape), "scale": float(max_abs / 127.0), "data": q.numpy().tobytes()}
        else:
            records[name] = {"kind": "raw", "shape": list(x.shape), "dtype": str(x.numpy().dtype), "data": x.numpy().tobytes()}
    buffer = io.BytesIO()
    torch.save(records, buffer)
    return brotli.compress(buffer.getvalue(), quality=11)


def estimate_renderer_bytes(config_name: str) -> dict:
    model = build_renderer(config_name)
    params = sum(p.numel() for p in model.parameters())
    fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    packed = int8_state_payload(model)
    return {
        "config_name": config_name,
        "params": params,
        "fp32_param_bytes": fp32_bytes,
        "int8_brotli_bytes_random_init": len(packed),
        "config": model.config(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["capacity", "F16", "F24", "F32"], default="F24")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = estimate_renderer_bytes(args.config)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
