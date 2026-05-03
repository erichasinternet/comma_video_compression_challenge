#!/usr/bin/env python
"""One-time exact-mask lossless sanity sweep.

This is deliberately limited to standard codecs/simple streams. Prior geometry
codecs already failed; this tool closes the remaining standard-codec doubt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, write_json
from submissions.search_vcm_v2.families.boundary_residual_codec import compress_bytes, compress_streams, pack_bits
from submissions.search_vcm_v2.families.qpose14_data import QPOSE14_ARCHIVE, decode_mask_stream, split_archive_payload


def _decision(mask_bytes: int) -> str:
    if mask_bytes <= 182 * 1024:
        return "pass"
    if mask_bytes <= 196 * 1024:
        return "near"
    if mask_bytes <= 205 * 1024:
        return "near_fail"
    return "fail"


def _same_prev_stream(classes: np.ndarray) -> dict[str, bytes]:
    same = np.zeros(classes.shape[0], dtype=np.uint8)
    changed_payload = bytearray()
    prev = np.zeros_like(classes[0])
    for i, frame in enumerate(classes):
        if i > 0 and np.array_equal(frame, prev):
            same[i] = 1
        else:
            changed_payload.extend(frame.tobytes())
            prev = frame
    return {"same_flags.bin": pack_bits(same.astype(bool)), "changed_frames.bin": bytes(changed_payload)}


def _bitplane_streams(classes: np.ndarray) -> dict[str, bytes]:
    streams = {}
    for bit in range(3):
        streams[f"bitplane{bit}.bin"] = pack_bits(((classes >> bit) & 1).astype(bool))
    return streams


def _ffmpeg_lossless(classes: np.ndarray, *, codec: str, qpose_archive: Path) -> dict[str, Any] | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    height, width = classes.shape[1:]
    gray = classes.astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        raw = td_path / "classes.raw"
        raw.write_bytes(gray.tobytes())
        out = td_path / "out.mkv"
        if codec == "ffv1":
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                f"{width}x{height}",
                "-i",
                str(raw),
                "-c:v",
                "ffv1",
                "-level",
                "3",
                str(out),
            ]
        elif codec == "x265_lossless":
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                f"{width}x{height}",
                "-i",
                str(raw),
                "-c:v",
                "libx265",
                "-x265-params",
                "lossless=1",
                "-pix_fmt",
                "gray",
                str(out),
            ]
        elif codec == "libaom_lossless":
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                f"{width}x{height}",
                "-i",
                str(raw),
                "-c:v",
                "libaom-av1",
                "-crf",
                "0",
                "-b:v",
                "0",
                "-cpu-used",
                "6",
                str(out),
            ]
        else:
            return None
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        except Exception as exc:
            return {"candidate": codec, "status": f"encode_failed:{exc}", "bytes": None, "verified_exact": False}

        decoded = td_path / "decoded.raw"
        dec_cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(out),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            str(decoded),
        ]
        try:
            subprocess.run(dec_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
            arr = np.frombuffer(decoded.read_bytes(), dtype=np.uint8).reshape(classes.shape)
            verified = bool(np.array_equal(arr, classes))
        except Exception:
            verified = False
        return {"candidate": codec, "status": "ok", "bytes": out.stat().st_size if out.exists() else None, "verified_exact": verified}


def run_sweep(*, qpose_archive: Path, out: Path, compressors: list[str]) -> dict[str, Any]:
    mask_br, _, _ = split_archive_payload(qpose_archive)
    classes = decode_mask_stream(mask_br).to(torch.uint8).numpy()
    palette = (classes.astype(np.uint16) * 63).astype(np.uint8)
    rows: list[dict[str, Any]] = []

    raw_class = compress_bytes(classes.tobytes(), tuple(compressors))
    rows.append({"candidate": "uint8_class_raw_stream", "bytes": raw_class["best_bytes"], "verified_exact": True, "compressor_breakdown": raw_class})
    raw_palette = compress_bytes(palette.tobytes(), tuple(compressors))
    rows.append({"candidate": "palette_grayscale_raw_stream", "bytes": raw_palette["best_bytes"], "verified_exact": True, "compressor_breakdown": raw_palette})
    bitplanes = compress_streams(_bitplane_streams(classes), tuple(compressors))
    rows.append({"candidate": "uint8_class_bitplanes", "bytes": bitplanes["total_best_bytes"], "verified_exact": True, "compressor_breakdown": bitplanes})
    same_prev = compress_streams(_same_prev_stream(classes), tuple(compressors))
    rows.append({"candidate": "same_as_prev_events", "bytes": same_prev["total_best_bytes"], "verified_exact": True, "compressor_breakdown": same_prev})

    for codec in ("ffv1", "x265_lossless", "libaom_lossless"):
        if codec not in compressors:
            continue
        item = _ffmpeg_lossless(classes, codec=codec, qpose_archive=qpose_archive)
        if item is not None:
            rows.append(item)

    verified = [row for row in rows if row.get("verified_exact") and row.get("bytes") is not None]
    best = min(verified, key=lambda row: int(row["bytes"])) if verified else None
    summary = {
        "qpose_archive": str(qpose_archive),
        "shape": list(classes.shape),
        "best": best,
        "decision": _decision(int(best["bytes"])) if best else "fail",
        "rows": rows,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qpose-archive", type=Path, default=QPOSE14_ARCHIVE)
    parser.add_argument("--out", type=Path, default=EXPERIMENTS_DIR / "exact_mask_lossless_sweep")
    parser.add_argument("--compressors", default="ffv1,x265_lossless,libaom_lossless,xz,zstd,brotli")
    args = parser.parse_args()
    compressors = [item.strip() for item in args.compressors.split(",") if item.strip()]
    run_sweep(qpose_archive=args.qpose_archive, out=args.out, compressors=compressors)


if __name__ == "__main__":
    main()
