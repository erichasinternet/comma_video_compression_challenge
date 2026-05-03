#!/usr/bin/env python3
"""Binary pack/unpack helpers for selfcomp segmap inference payloads.

The original PR #56 payload is a PyTorch checkpoint containing a small
structured dict.  This module stores the same structure in a deterministic
binary tree format so the inflater can reconstruct the exact payload without
PyTorch pickle/zip container overhead.
"""

from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path
from typing import Any

import brotli
import torch


MAGIC = b"SCVDC1\0"

TAG_NONE = b"n"
TAG_BOOL = b"b"
TAG_INT = b"i"
TAG_FLOAT = b"f"
TAG_STR = b"s"
TAG_DICT = b"d"
TAG_LIST = b"l"
TAG_TENSOR = b"t"

DTYPE_TO_CODE = {
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.bool: 8,
}
CODE_TO_DTYPE = {value: key for key, value in DTYPE_TO_CODE.items()}


def _write_varint(out: io.BytesIO, value: int) -> None:
    if value < 0:
        raise ValueError(f"varint cannot encode negative value: {value}")
    while value >= 0x80:
        out.write(bytes([(value & 0x7F) | 0x80]))
        value >>= 7
    out.write(bytes([value]))


def _read_varint(inp: memoryview, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if pos >= len(inp):
            raise ValueError("truncated varint")
        byte = int(inp[pos])
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint is too large")


def _write_bytes(out: io.BytesIO, data: bytes) -> None:
    _write_varint(out, len(data))
    out.write(data)


def _read_bytes(inp: memoryview, pos: int) -> tuple[bytes, int]:
    length, pos = _read_varint(inp, pos)
    end = pos + length
    if end > len(inp):
        raise ValueError("truncated byte payload")
    return bytes(inp[pos:end]), end


def _write_string(out: io.BytesIO, value: str) -> None:
    _write_bytes(out, value.encode("utf-8"))


def _read_string(inp: memoryview, pos: int) -> tuple[str, int]:
    data, pos = _read_bytes(inp, pos)
    return data.decode("utf-8"), pos


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype not in DTYPE_TO_CODE:
        raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.numpy().tobytes(order="C")


def _encode_value(out: io.BytesIO, value: Any) -> None:
    if value is None:
        out.write(TAG_NONE)
        return
    if isinstance(value, bool):
        out.write(TAG_BOOL)
        out.write(b"\x01" if value else b"\x00")
        return
    if isinstance(value, int):
        out.write(TAG_INT)
        out.write(struct.pack("<q", value))
        return
    if isinstance(value, float):
        out.write(TAG_FLOAT)
        out.write(struct.pack("<d", value))
        return
    if isinstance(value, str):
        out.write(TAG_STR)
        _write_string(out, value)
        return
    if isinstance(value, dict):
        out.write(TAG_DICT)
        keys = sorted(value.keys())
        _write_varint(out, len(keys))
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(f"only string dict keys are supported, got {type(key)}")
            _write_string(out, key)
            _encode_value(out, value[key])
        return
    if isinstance(value, (list, tuple)):
        out.write(TAG_LIST)
        _write_varint(out, len(value))
        for item in value:
            _encode_value(out, item)
        return
    if torch.is_tensor(value):
        out.write(TAG_TENSOR)
        dtype_code = DTYPE_TO_CODE[value.dtype]
        out.write(bytes([dtype_code]))
        _write_varint(out, value.ndim)
        for dim in value.shape:
            _write_varint(out, int(dim))
        _write_bytes(out, _tensor_to_bytes(value))
        return
    raise TypeError(f"unsupported payload value: {type(value)}")


def _decode_value(inp: memoryview, pos: int) -> tuple[Any, int]:
    if pos >= len(inp):
        raise ValueError("truncated payload")
    tag = bytes(inp[pos : pos + 1])
    pos += 1
    if tag == TAG_NONE:
        return None, pos
    if tag == TAG_BOOL:
        if pos >= len(inp):
            raise ValueError("truncated bool")
        return bool(inp[pos]), pos + 1
    if tag == TAG_INT:
        end = pos + 8
        if end > len(inp):
            raise ValueError("truncated int")
        return struct.unpack("<q", inp[pos:end])[0], end
    if tag == TAG_FLOAT:
        end = pos + 8
        if end > len(inp):
            raise ValueError("truncated float")
        return struct.unpack("<d", inp[pos:end])[0], end
    if tag == TAG_STR:
        return _read_string(inp, pos)
    if tag == TAG_DICT:
        length, pos = _read_varint(inp, pos)
        out = {}
        for _ in range(length):
            key, pos = _read_string(inp, pos)
            out[key], pos = _decode_value(inp, pos)
        return out, pos
    if tag == TAG_LIST:
        length, pos = _read_varint(inp, pos)
        out = []
        for _ in range(length):
            item, pos = _decode_value(inp, pos)
            out.append(item)
        return out, pos
    if tag == TAG_TENSOR:
        if pos >= len(inp):
            raise ValueError("truncated tensor dtype")
        dtype_code = int(inp[pos])
        pos += 1
        dtype = CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(f"unsupported tensor dtype code: {dtype_code}")
        ndim, pos = _read_varint(inp, pos)
        shape = []
        numel = 1
        for _ in range(ndim):
            dim, pos = _read_varint(inp, pos)
            shape.append(dim)
            numel *= dim
        raw, pos = _read_bytes(inp, pos)
        element_size = torch.empty((), dtype=dtype).element_size()
        expected = numel * element_size
        if len(raw) != expected:
            raise ValueError(f"tensor byte length mismatch: got {len(raw)}, expected {expected}")
        storage = torch.frombuffer(bytearray(raw), dtype=dtype)
        return storage.reshape(tuple(shape)).clone(), pos
    raise ValueError(f"unsupported value tag: {tag!r}")


def pack_payload(payload: dict[str, Any]) -> bytes:
    out = io.BytesIO()
    out.write(MAGIC)
    _encode_value(out, payload)
    return out.getvalue()


def unpack_payload(data: bytes) -> dict[str, Any]:
    if data.startswith(MAGIC):
        payload, pos = _decode_value(memoryview(data), len(MAGIC))
        if pos != len(data):
            raise ValueError(f"trailing bytes after dcpack payload: {len(data) - pos}")
        if not isinstance(payload, dict):
            raise TypeError("dcpack root payload is not a dict")
        return payload
    raise ValueError("not a selfcomp dcpack payload")


def read_dcpack(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if path.name.endswith(".br"):
        data = brotli.decompress(data)
    return unpack_payload(data)


def write_dcpack(payload: dict[str, Any], path: Path, *, compress: bool = True) -> None:
    data = pack_payload(payload)
    if compress:
        data = brotli.compress(data, quality=11, lgwin=24)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _deep_equal(a: Any, b: Any, path: str = "$") -> None:
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise AssertionError(f"{path}: tensor/type mismatch")
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f"{path}: tensor metadata mismatch")
        if not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError(f"{path}: tensor values differ")
        return
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            raise AssertionError(f"{path}: dict/type mismatch")
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"{path}: dict keys differ")
        for key in sorted(a.keys()):
            _deep_equal(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            raise AssertionError(f"{path}: list/type mismatch")
        if len(a) != len(b):
            raise AssertionError(f"{path}: list length mismatch")
        for idx, (left, right) in enumerate(zip(a, b)):
            _deep_equal(left, right, f"{path}[{idx}]")
        return
    if a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or unpack a selfcomp dcpack payload.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-pt", type=Path, help="Optional path to write a PyTorch checkpoint.")
    parser.add_argument("--compare-pt", type=Path, help="Optional original checkpoint for exact structure comparison.")
    args = parser.parse_args()

    payload = read_dcpack(args.input)
    print(f"Loaded {args.input} with keys: {', '.join(sorted(payload.keys()))}")
    if args.compare_pt:
        original = torch.load(args.compare_pt, map_location="cpu")
        _deep_equal(original, payload)
        print("Exact payload comparison passed.")
    if args.out_pt:
        args.out_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.out_pt)
        print(f"Wrote {args.out_pt}")


if __name__ == "__main__":
    main()
