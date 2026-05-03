#!/usr/bin/env python
"""Compare decoded semantic mask classes between two archive.zip files."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import brotli
import numpy as np

import av

from q55_mask_alloc import decode_archive_masks


def extract_mask_payload(archive_zip: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(archive_zip) as z:
        z.extract("mask.obu.br", out_dir)
    return out_dir / "mask.obu.br"


def decode_mask_grayscale(mask_br: Path) -> np.ndarray:
    obu = mask_br.with_suffix("")
    obu.write_bytes(brotli.decompress(mask_br.read_bytes()))
    try:
        container = av.open(str(obu))
        frames = [frame.to_ndarray(format="gray").copy() for frame in container.decode(video=0)]
        container.close()
    finally:
        obu.unlink(missing_ok=True)
    if not frames:
        raise RuntimeError(f"no frames decoded from {mask_br}")
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a_payload = extract_mask_payload(args.a, root / "a")
        b_payload = extract_mask_payload(args.b, root / "b")
        a_gray = decode_mask_grayscale(a_payload)
        b_gray = decode_mask_grayscale(b_payload)
        a_mask = np.clip(np.round(a_gray / 63.0).astype(np.uint8), 0, 4)
        b_mask = np.clip(np.round(b_gray / 63.0).astype(np.uint8), 0, 4)

    if a_mask.shape != b_mask.shape:
        raise RuntimeError(f"shape mismatch: {a_mask.shape} vs {b_mask.shape}")

    diff = a_mask != b_mask
    per_frame = diff.reshape(diff.shape[0], -1).mean(axis=1)
    transition = np.zeros((5, 5), dtype=np.int64)
    for src in range(5):
        for dst in range(5):
            transition[src, dst] = int(((a_mask == src) & (b_mask == dst)).sum())

    print(f"frames: {a_mask.shape[0]}")
    print(f"shape: {a_mask.shape[1]}x{a_mask.shape[2]}")
    raw_abs = np.abs(a_gray.astype(np.int16) - b_gray.astype(np.int16))
    print(f"raw_changed_fraction: {float((a_gray != b_gray).mean()):.8f}")
    print(f"raw_abs_mean: {float(raw_abs.mean()):.8f}")
    print(f"raw_abs_p50: {float(np.percentile(raw_abs, 50)):.8f}")
    print(f"raw_abs_p90: {float(np.percentile(raw_abs, 90)):.8f}")
    print(f"raw_abs_p99: {float(np.percentile(raw_abs, 99)):.8f}")
    print(f"raw_abs_max: {int(raw_abs.max())}")
    print(f"changed_pixels: {int(diff.sum())}")
    print(f"changed_fraction: {float(diff.mean()):.8f}")
    print(f"per_frame_mean: {float(per_frame.mean()):.8f}")
    print(f"per_frame_p50: {float(np.percentile(per_frame, 50)):.8f}")
    print(f"per_frame_p90: {float(np.percentile(per_frame, 90)):.8f}")
    print(f"per_frame_p99: {float(np.percentile(per_frame, 99)):.8f}")
    print("top_frames:")
    for idx in np.argsort(-per_frame)[: args.top_k]:
        print(f"  {int(idx):4d}: {float(per_frame[idx]):.8f}")
    print("transition_counts rows=a cols=b:")
    for src in range(5):
        print("  " + " ".join(str(int(x)) for x in transition[src]))


if __name__ == "__main__":
    main()
