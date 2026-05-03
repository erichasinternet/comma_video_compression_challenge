#!/usr/bin/env python
"""Pack Quantizr FP4 model payloads into a compact pickle-free qpack format."""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import brotli
import numpy as np
import torch


MODEL_PAYLOAD_NAME = "model.pt.br"
MODEL_QPACK_PAYLOAD_NAME = "model.qpack.br"
HEAD_SENSITIVE_PREFIXES = (
    "frame1_head.head",
    "frame2_head.head",
    "pose_mlp",
)


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().newbyteorder("<")
    return np.ascontiguousarray(arr)


def add_array(arrays: list[tuple[str, np.ndarray]], name: str, tensor: torch.Tensor | np.ndarray) -> str:
    if isinstance(tensor, torch.Tensor):
        arr = tensor_to_array(tensor)
    else:
        arr = np.ascontiguousarray(tensor)
    arrays.append((name, arr))
    return name


def quantize_symmetric_int8(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = arr.astype(np.float32, copy=False)
    max_abs = np.max(np.abs(arr))
    scale = np.float32(max_abs / 127.0) if max_abs > 1e-12 else np.float32(1.0)
    q = np.clip(np.round(arr / float(scale)), -127, 127).astype(np.int8)
    return q, np.asarray(scale, dtype=np.float32)


def is_sensitive_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in HEAD_SENSITIVE_PREFIXES)


def should_int8(name: str, quantize_fp16: str) -> bool:
    if quantize_fp16 == "none":
        return False
    if quantize_fp16 == "int8":
        return True
    if quantize_fp16 == "int8_heads_fp16":
        return not is_sensitive_name(name)
    raise ValueError(f"unknown quantize mode: {quantize_fp16}")


def load_model_payload(args) -> dict:
    if args.fp4 is not None:
        return torch.load(args.fp4, map_location="cpu")
    model_br = args.model_br
    if model_br is None:
        if args.archive_dir is None:
            raise ValueError("Pass --archive-dir, --model-br, or --fp4")
        model_br = args.archive_dir / MODEL_PAYLOAD_NAME
    with open(model_br, "rb") as f:
        return torch.load(io.BytesIO(brotli.decompress(f.read())), map_location="cpu")


def build_qpack(data: dict, quantize_fp16: str = "none") -> bytes:
    source = data.get("quantized", data.get("tensors", {}))
    arrays: list[tuple[str, np.ndarray]] = []
    header = {
        "format": "quantizr_qpack_v1",
        "__meta__": data.get("__meta__", {}),
        "quantized": {},
        "dense_fp16": {},
        "arrays": [],
    }

    for module_name, rec in source.items():
        out_rec = {
            "type": rec.get("type", "tensor"),
            "weight_kind": rec["weight_kind"],
            "weight_shape": [int(x) for x in rec["weight_shape"]],
        }
        for key in ("stride", "padding", "dilation", "groups"):
            if key in rec:
                out_rec[key] = rec[key]
        safe_name = module_name.replace(".", "__")
        if rec["weight_kind"] == "fp4_packed":
            out_rec["packed_weight"] = add_array(arrays, f"q/{safe_name}/packed_weight", rec["packed_weight"])
            out_rec["scales_fp16"] = add_array(arrays, f"q/{safe_name}/scales_fp16", rec["scales_fp16"])
        else:
            weight_arr = tensor_to_array(rec["weight_fp16"])
            if should_int8(module_name, quantize_fp16) and np.issubdtype(weight_arr.dtype, np.floating):
                q_weight, scale = quantize_symmetric_int8(weight_arr)
                out_rec["weight_kind"] = "int8_symmetric"
                out_rec["weight_int8"] = add_array(arrays, f"q/{safe_name}/weight_int8", q_weight)
                out_rec["weight_scale_fp16"] = add_array(arrays, f"q/{safe_name}/weight_scale_fp16", scale)
            else:
                out_rec["weight_fp16"] = add_array(arrays, f"q/{safe_name}/weight_fp16", weight_arr)
        if rec.get("bias_fp16") is not None:
            out_rec["bias_fp16"] = add_array(arrays, f"q/{safe_name}/bias_fp16", rec["bias_fp16"])
        else:
            out_rec["bias_fp16"] = None
        header["quantized"][module_name] = out_rec

    for name, tensor in data.get("dense_fp16", {}).items():
        arr = tensor_to_array(tensor)
        if should_int8(name, quantize_fp16) and np.issubdtype(arr.dtype, np.floating):
            q_arr, scale = quantize_symmetric_int8(arr)
            header["dense_fp16"][name] = {
                "kind": "int8_symmetric",
                "values": add_array(arrays, f"d/{name}/values", q_arr),
                "scale": add_array(arrays, f"d/{name}/scale", scale),
            }
        else:
            header["dense_fp16"][name] = {"kind": "raw", "values": add_array(arrays, f"d/{name}", arr)}

    offset = 0
    blob_parts = []
    for name, arr in arrays:
        raw = arr.tobytes(order="C")
        header["arrays"].append(
            {
                "name": name,
                "dtype": arr.dtype.str,
                "shape": [int(x) for x in arr.shape],
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        blob_parts.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"QPK1" + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(blob_parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--model-br", type=Path)
    parser.add_argument("--fp4", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--copy-into-archive", action="store_true")
    parser.add_argument("--quantize-fp16", choices=["none", "int8", "int8_heads_fp16"], default="none")
    args = parser.parse_args()

    data = load_model_payload(args)
    qpack = build_qpack(data, quantize_fp16=args.quantize_fp16)
    out = args.out
    if out is None:
        if args.archive_dir is None:
            raise ValueError("--out is required without --archive-dir")
        out = args.archive_dir / MODEL_QPACK_PAYLOAD_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(brotli.compress(qpack, quality=11, lgwin=24))

    old_size = None
    if args.model_br is not None and args.model_br.exists():
        old_size = args.model_br.stat().st_size
    elif args.archive_dir is not None and (args.archive_dir / MODEL_PAYLOAD_NAME).exists():
        old_size = (args.archive_dir / MODEL_PAYLOAD_NAME).stat().st_size
    report = {
        "out": str(out),
        "quantize_fp16": args.quantize_fp16,
        "qpack_bytes": out.stat().st_size,
        "model_pt_br_bytes": old_size,
    }
    if old_size:
        report["saved_bytes"] = old_size - out.stat().st_size
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
