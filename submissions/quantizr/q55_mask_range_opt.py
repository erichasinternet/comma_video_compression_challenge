#!/usr/bin/env python
"""Class-preserving grayscale source optimization for the Quantizr #55 mask.

The Quantizr inflater rounds decoded grayscale values with round(img / 63) and
feeds only the resulting class IDs to the generator. This diagnostic searches
for grayscale mask videos that stay inside each class's legal rounding interval
but are easier for AV1/VP9 to encode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np

from q55_common import MASK_PAYLOAD, ORIGINAL_BYTES, REPO_ROOT, sha256_file, write_json
from q55_mask_alloc import decode_archive_masks


DEFAULT_ARCHIVE = REPO_ROOT / "submissions/q55_fp16_pose_int10/archive.zip"
DEFAULT_OUT = REPO_ROOT / "submissions/quantizr/experiments/q55_range_mask"
WIDTH = 512
HEIGHT = 384
ROUNDING_INTERVALS = np.asarray(
    [
        [0, 31],
        [32, 94],
        [95, 157],
        [158, 220],
        [221, 255],
    ],
    dtype=np.int16,
)


@dataclass(frozen=True)
class CodecSpec:
    name: str
    encoder: str
    container: str
    suffix: str
    max_crf: int = 63


CODECS = {
    "libaom": CodecSpec("libaom", "libaom-av1", "obu", ".obu"),
    "svtav1": CodecSpec("svtav1", "libsvtav1", "ivf", ".ivf"),
    "vp9": CodecSpec("vp9", "libvpx-vp9", "ivf", ".ivf"),
}


def get_ffmpeg_path() -> str:
    local_ffmpeg = Path(__file__).resolve().parent / "ffmpeg"
    if local_ffmpeg.is_file() and os.access(local_ffmpeg, os.X_OK):
        return str(local_ffmpeg.resolve())
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return exe
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise FileNotFoundError("ffmpeg not found")


def available_encoders(ffmpeg_path: str) -> set[str]:
    proc = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-encoders"],
        check=True,
        text=True,
        capture_output=True,
    )
    return {spec.name for spec in CODECS.values() if spec.encoder in proc.stdout}


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def legal_intervals(margin: int) -> np.ndarray:
    intervals = ROUNDING_INTERVALS.copy()
    for cls in range(5):
        if cls > 0:
            intervals[cls, 0] += margin
        if cls < 4:
            intervals[cls, 1] -= margin
    if np.any(intervals[:, 0] > intervals[:, 1]):
        raise ValueError(f"margin {margin} makes an empty class interval: {intervals.tolist()}")
    return intervals


def round_classes(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.round(gray.astype(np.float32) / 63.0), 0, 4).astype(np.uint8)


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for raw in values.astype(np.uint16, copy=False).reshape(-1):
        value = int(raw)
        if value & ~mask:
            raise ValueError(f"value {value} exceeds {bits} bits")
        acc |= value << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def encode_varints(values: np.ndarray) -> bytes:
    out = bytearray()
    for raw in values.astype(np.int64, copy=False).reshape(-1):
        value = int(raw)
        if value < 0:
            raise ValueError("varints must be non-negative")
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
    return bytes(out)


def residual_estimate(exact: np.ndarray, decoded: np.ndarray) -> dict:
    flat_exact = exact.reshape(-1)
    flat_decoded = decoded.reshape(-1)
    changed = np.flatnonzero(flat_exact != flat_decoded).astype(np.int64)
    if changed.size:
        deltas = np.empty_like(changed)
        deltas[0] = changed[0]
        deltas[1:] = changed[1:] - changed[:-1]
        values = flat_exact[changed].astype(np.uint8)
    else:
        deltas = changed
        values = np.empty((0,), dtype=np.uint8)

    pos_bytes = encode_varints(deltas)
    value_bytes = pack_bits(values, 3)
    raw = (
        b"QMR1"
        + int(changed.size).to_bytes(4, "little")
        + len(pos_bytes).to_bytes(4, "little")
        + len(value_bytes).to_bytes(4, "little")
        + pos_bytes
        + value_bytes
    )
    br = brotli.compress(raw, quality=11, lgwin=24) if changed.size else b""
    manifest = {
        "format": "q55_mask_range_residual_estimate_v0",
        "num_frames": int(exact.shape[0]),
        "height": int(exact.shape[1]),
        "width": int(exact.shape[2]),
        "changed_pixels": int(changed.size),
    }
    manifest_br = (
        brotli.compress(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(), quality=11, lgwin=24)
        if changed.size
        else b""
    )
    return {
        "changed_pixels": int(changed.size),
        "changed_fraction": float(changed.size / flat_exact.size),
        "position_bytes_raw": len(pos_bytes),
        "value_bytes_raw": len(value_bytes),
        "residual_payload_raw_bytes": len(raw) if changed.size else 0,
        "residual_payload_br_bytes": len(br),
        "residual_manifest_br_bytes_est": len(manifest_br),
        "residual_total_br_bytes_est": len(br) + len(manifest_br),
    }


def extract_mask_payload(archive_zip: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(archive_zip) as z:
        z.extract(MASK_PAYLOAD, out_dir)
    return out_dir / MASK_PAYLOAD


def mask_payload_size(archive_zip: Path) -> int:
    with zipfile.ZipFile(archive_zip) as z:
        return z.getinfo(MASK_PAYLOAD).file_size


def load_classes(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        arr = data["classes"] if "classes" in data else data[data.files[0]]
    else:
        raise ValueError(f"unsupported class cache suffix: {path.suffix}")
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[1:] != (HEIGHT, WIDTH):
        raise ValueError(f"expected classes shape (*,{HEIGHT},{WIDTH}), got {arr.shape}")
    if int(arr.min()) < 0 or int(arr.max()) > 4:
        raise ValueError(f"class tensor must be in 0..4, got {int(arr.min())}..{int(arr.max())}")
    return arr


def load_classes_from_archive(archive: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        payload = extract_mask_payload(archive, Path(td))
        return decode_archive_masks(payload)


def class_adjacency_counts(classes: np.ndarray) -> np.ndarray:
    counts = np.zeros((5, 5), dtype=np.float64)

    def add_pairs(a: np.ndarray, b: np.ndarray) -> None:
        mask = a != b
        if not np.any(mask):
            return
        aa = a[mask].astype(np.int64)
        bb = b[mask].astype(np.int64)
        pair_counts = np.bincount(aa * 5 + bb, minlength=25).reshape(5, 5)
        counts[:] += pair_counts

    add_pairs(classes[:, :, :-1], classes[:, :, 1:])
    add_pairs(classes[:, :-1, :], classes[:, 1:, :])
    add_pairs(classes[:-1], classes[1:])
    counts = counts + counts.T
    np.fill_diagonal(counts, 0)
    return counts


def optimize_palette(classes: np.ndarray, margin: int) -> list[int]:
    intervals = legal_intervals(margin)
    weights = class_adjacency_counts(classes)
    p = intervals.mean(axis=1).astype(np.float64)
    for _ in range(1000):
        old = p.copy()
        for i in range(5):
            denom = weights[i].sum()
            if denom > 0:
                p[i] = float(np.dot(weights[i], p) / denom)
            p[i] = np.clip(p[i], intervals[i, 0], intervals[i, 1])
        if float(np.max(np.abs(p - old))) < 1e-6:
            break
    return [int(round(x)) for x in p]


def palette_for_kind(classes: np.ndarray, kind: str, margin: int) -> list[int]:
    intervals = legal_intervals(margin)
    if kind == "standard":
        p = np.asarray([0, 63, 126, 189, 252], dtype=np.int16)
        return [int(np.clip(p[i], intervals[i, 0], intervals[i, 1])) for i in range(5)]
    if kind == "center":
        return [int(round(x)) for x in intervals.mean(axis=1)]
    if kind == "low":
        return [int(x) for x in intervals[:, 0]]
    if kind == "high":
        return [int(x) for x in intervals[:, 1]]
    if kind == "adjopt":
        return optimize_palette(classes, margin)
    raise ValueError(f"unknown palette kind: {kind}")


def source_from_palette(classes: np.ndarray, palette: list[int]) -> np.ndarray:
    lut = np.asarray(palette, dtype=np.uint8)
    return lut[classes]


def diffuse_source(classes: np.ndarray, palette: list[int], margin: int, iters: int, alpha: float = 0.65) -> np.ndarray:
    intervals = legal_intervals(margin)
    lo = intervals[:, 0].astype(np.float32)[classes]
    hi = intervals[:, 1].astype(np.float32)[classes]
    x = source_from_palette(classes, palette).astype(np.float32)
    for _ in range(iters):
        acc = 2.0 * x
        weight = 2.0
        acc[:, :, 1:] += x[:, :, :-1]
        acc[:, :, :-1] += x[:, :, 1:]
        acc[:, 1:, :] += x[:, :-1, :]
        acc[:, :-1, :] += x[:, 1:, :]
        weight += 4.0
        acc[1:] += x[:-1]
        acc[:-1] += x[1:]
        weight += 2.0
        avg = acc / weight
        x = np.clip((1.0 - alpha) * x + alpha * avg, lo, hi)
    return np.rint(x).astype(np.uint8)


def write_raw_gray(source: np.ndarray, path: Path) -> None:
    if source.dtype != np.uint8:
        raise ValueError("source must be uint8")
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(source).tobytes())


def encode_video(source: np.ndarray, codec: str, crf: int, out_dir: Path, label: str) -> tuple[Path, Path]:
    spec = CODECS[codec]
    if crf > spec.max_crf:
        raise ValueError(f"{codec} CRF {crf} exceeds max {spec.max_crf}")
    ffmpeg_path = get_ffmpeg_path()
    raw_path = out_dir / f"{label}.gray"
    video_path = out_dir / f"{label}{spec.suffix}"
    br_path = out_dir / f"{label}{spec.suffix}.br"
    write_raw_gray(source, raw_path)

    cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        "10",
        "-i",
        str(raw_path),
    ]
    if codec == "libaom":
        cmd += [
            "-c:v",
            spec.encoder,
            "-crf",
            str(crf),
            "-b:v",
            "0",
            "-cpu-used",
            "0",
            "-row-mt",
            "1",
            "-g",
            "1200",
            "-keyint_min",
            "1200",
            "-lag-in-frames",
            "48",
            "-arnr-strength",
            "0",
            "-aq-mode",
            "0",
            "-aom-params",
            "enable-cdef=0:enable-intrabc=1:enable-obmc=0",
            "-f",
            spec.container,
            str(video_path),
        ]
    elif codec == "svtav1":
        cmd += [
            "-vf",
            "format=yuv420p",
            "-c:v",
            spec.encoder,
            "-crf",
            str(crf),
            "-preset",
            "4",
            "-g",
            "1200",
            "-svtav1-params",
            "enable-overlays=0",
            "-f",
            spec.container,
            str(video_path),
        ]
    elif codec == "vp9":
        cmd += [
            "-vf",
            "format=yuv420p",
            "-c:v",
            spec.encoder,
            "-crf",
            str(crf),
            "-b:v",
            "0",
            "-deadline",
            "good",
            "-cpu-used",
            "0",
            "-row-mt",
            "1",
            "-g",
            "1200",
            "-f",
            spec.container,
            str(video_path),
        ]
    else:
        raise ValueError(codec)
    try:
        subprocess.run(cmd, check=True)
        br_path.write_bytes(brotli.compress(video_path.read_bytes(), quality=11, lgwin=24))
    finally:
        raw_path.unlink(missing_ok=True)
    return video_path, br_path


def decode_video_classes(video_path: Path) -> np.ndarray:
    import av

    frames = []
    container = av.open(str(video_path))
    try:
        for frame in container.decode(video=0):
            gray = frame.to_ndarray(format="gray")
            frames.append(round_classes(gray))
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"no decoded frames from {video_path}")
    return np.stack(frames).astype(np.uint8, copy=False)


def evaluate_source(
    *,
    classes: np.ndarray,
    source: np.ndarray,
    out_dir: Path,
    label: str,
    codec: str,
    crf: int,
    keep_video: bool,
) -> dict:
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    video_path, br_path = encode_video(source, codec, crf, candidate_dir, label)
    decoded = decode_video_classes(video_path)
    if decoded.shape != classes.shape:
        raise RuntimeError(f"decoded shape {decoded.shape}, expected {classes.shape}")
    residual = residual_estimate(classes, decoded)
    base_bytes = br_path.stat().st_size
    exact_total = base_bytes + residual["residual_total_br_bytes_est"]
    record = {
        "label": label,
        "codec": codec,
        "crf": crf,
        "base_video_br_bytes": base_bytes,
        "residual": residual,
        "exact_or_repaired_mask_bytes_est": exact_total,
        "mask_label": mask_label(exact_total),
        "decoded_exact_without_residual": residual["changed_pixels"] == 0,
        "video_path": str(video_path),
        "video_br_path": str(br_path),
    }
    if not keep_video:
        video_path.unlink(missing_ok=True)
        br_path.unlink(missing_ok=True)
        record["video_path"] = None
        record["video_br_path"] = None
    return record


def mask_label(mask_bytes: int) -> str:
    if mask_bytes <= 152_000:
        return "dream_local_cpu_0p2x"
    if mask_bytes <= 182_000:
        return "strong_pr_like_0p2x"
    if mask_bytes <= 196_000:
        return "near_first_place_or_pr_path"
    if mask_bytes <= 205_000:
        return "near_miss_continue_if_promising"
    return "hard_fail"


def resolve_codecs(requested: str) -> list[str]:
    ffmpeg_path = get_ffmpeg_path()
    available = available_encoders(ffmpeg_path)
    if requested == "auto":
        order = ["libaom", "svtav1"]
    elif requested == "auto-all":
        order = ["libaom", "svtav1", "vp9"]
    else:
        order = parse_csv(requested)
    codecs = []
    for codec in order:
        if codec not in CODECS:
            raise ValueError(f"unknown codec {codec}; choices are {sorted(CODECS)}")
        if codec not in available:
            print(f"warning: {CODECS[codec].encoder} unavailable; skipping {codec}", flush=True)
            continue
        codecs.append(codec)
    if not codecs:
        raise RuntimeError(f"none of requested codecs are available; available={sorted(available)}")
    return codecs


def cmd_extract(args: argparse.Namespace) -> None:
    classes = load_classes_from_archive(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, classes)
    record = {
        "archive": str(args.archive),
        "archive_sha256": sha256_file(args.archive),
        "mask_payload_bytes": mask_payload_size(args.archive),
        "classes": str(args.out),
        "shape": [int(x) for x in classes.shape],
        "class_counts": [int(x) for x in np.bincount(classes.reshape(-1), minlength=5)],
    }
    write_json(args.out.with_suffix(".json"), record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


def load_or_extract(args: argparse.Namespace) -> np.ndarray:
    if args.classes:
        return load_classes(args.classes)
    return load_classes_from_archive(args.archive)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def add_projection(record: dict, nonmask_bytes: int) -> None:
    total_archive = nonmask_bytes + int(record["exact_or_repaired_mask_bytes_est"])
    record["projected_nonmask_bytes"] = nonmask_bytes
    record["projected_archive_bytes"] = total_archive
    record["projected_rate_term"] = 25.0 * total_archive / ORIGINAL_BYTES
    record["required_quality_for_0p300"] = 0.300 - record["projected_rate_term"]


def summarize_records(records: list[dict], out_dir: Path, archive: Path, classes: np.ndarray, nonmask_bytes: int) -> dict:
    best = min(records, key=lambda r: int(r["exact_or_repaired_mask_bytes_est"]))
    summary = {
        "archive": str(archive),
        "archive_sha256": sha256_file(archive) if archive.exists() else None,
        "current_mask_payload_bytes": mask_payload_size(archive) if archive.exists() else None,
        "class_shape": [int(x) for x in classes.shape],
        "candidate_count": len(records),
        "best": best,
        "nonmask_bytes": nonmask_bytes,
        "elapsed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary


def cmd_palette_sweep(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    classes = load_or_extract(args)
    codecs = resolve_codecs(args.codecs)
    crfs = [x for x in parse_csv_ints(args.crfs) if x <= 63]
    margins = parse_csv_ints(args.margins)
    kinds = parse_csv(args.palette_kinds)
    records = []
    for margin in margins:
        for kind in kinds:
            palette = palette_for_kind(classes, kind, margin)
            source = source_from_palette(classes, palette)
            assert np.array_equal(round_classes(source), classes)
            for codec in codecs:
                for crf in crfs:
                    label = f"palette_m{margin}_{kind}_{codec}_crf{crf}"
                    print(f"encoding {label} palette={palette}", flush=True)
                    record = evaluate_source(
                        classes=classes,
                        source=source,
                        out_dir=args.out_dir,
                        label=label,
                        codec=codec,
                        crf=crf,
                        keep_video=args.keep_video,
                    )
                    record.update({"mode": "palette", "margin": margin, "palette_kind": kind, "palette": palette})
                    add_projection(record, args.nonmask_bytes)
                    records.append(record)
                    append_jsonl(args.out_dir / "palette_results.jsonl", record)
                    print(
                        f"{label}: base={record['base_video_br_bytes']:,} "
                        f"errors={record['residual']['changed_pixels']:,} "
                        f"exact_est={record['exact_or_repaired_mask_bytes_est']:,}",
                        flush=True,
                    )
    summary = summarize_records(records, args.out_dir, args.archive, classes, args.nonmask_bytes)
    summary["elapsed_seconds"] = time.time() - started
    write_json(args.out_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def cmd_diffuse_sweep(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    classes = load_or_extract(args)
    codecs = resolve_codecs(args.codecs)
    crfs = [x for x in parse_csv_ints(args.crfs) if x <= 63]
    margins = parse_csv_ints(args.margins)
    iters_list = parse_csv_ints(args.iters)
    records = []
    for margin in margins:
        palette = palette_for_kind(classes, args.palette_kind, margin)
        for iters in iters_list:
            print(f"diffusing margin={margin} iters={iters} palette={palette}", flush=True)
            source = diffuse_source(classes, palette, margin, iters, alpha=args.alpha)
            if not np.array_equal(round_classes(source), classes):
                diff = int(np.count_nonzero(round_classes(source) != classes))
                raise RuntimeError(f"diffusion source violated class intervals: {diff} pixels")
            for codec in codecs:
                for crf in crfs:
                    label = f"diffuse_m{margin}_i{iters}_{args.palette_kind}_{codec}_crf{crf}"
                    print(f"encoding {label}", flush=True)
                    record = evaluate_source(
                        classes=classes,
                        source=source,
                        out_dir=args.out_dir,
                        label=label,
                        codec=codec,
                        crf=crf,
                        keep_video=args.keep_video,
                    )
                    record.update(
                        {
                            "mode": "diffuse",
                            "margin": margin,
                            "iters": iters,
                            "alpha": args.alpha,
                            "palette_kind": args.palette_kind,
                            "palette": palette,
                        }
                    )
                    add_projection(record, args.nonmask_bytes)
                    records.append(record)
                    append_jsonl(args.out_dir / "diffuse_results.jsonl", record)
                    print(
                        f"{label}: base={record['base_video_br_bytes']:,} "
                        f"errors={record['residual']['changed_pixels']:,} "
                        f"exact_est={record['exact_or_repaired_mask_bytes_est']:,}",
                        flush=True,
                    )
    summary = summarize_records(records, args.out_dir, args.archive, classes, args.nonmask_bytes)
    summary["elapsed_seconds"] = time.time() - started
    write_json(args.out_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser("extract", help="extract exact rounded class tensor from archive")
    extract.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    extract.add_argument("--out", type=Path, default=DEFAULT_OUT / "classes.npy")
    extract.set_defaults(func=cmd_extract)

    pal = sub.add_parser("palette_sweep", help="search legal global class palettes")
    pal.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    pal.add_argument("--classes", type=Path, default=None)
    pal.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "palette")
    pal.add_argument("--margins", default="0,2,4,8,12")
    pal.add_argument("--crfs", default="40,44,48,52,56,60,63")
    pal.add_argument("--codecs", default="auto")
    pal.add_argument("--palette-kinds", default="standard,center,adjopt")
    pal.add_argument("--nonmask-bytes", type=int, default=68_796)
    pal.add_argument("--keep-video", action="store_true")
    pal.set_defaults(func=cmd_palette_sweep)

    diff = sub.add_parser("diffuse_sweep", help="projected smoothing within class intervals")
    diff.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    diff.add_argument("--classes", type=Path, default=None)
    diff.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "diffuse")
    diff.add_argument("--margins", default="0,2,4,8,12")
    diff.add_argument("--iters", default="10,30,100")
    diff.add_argument("--crfs", default="44,48,52,56,60,63")
    diff.add_argument("--codecs", default="auto")
    diff.add_argument("--palette-kind", default="adjopt")
    diff.add_argument("--alpha", type=float, default=0.65)
    diff.add_argument("--nonmask-bytes", type=int, default=68_796)
    diff.add_argument("--keep-video", action="store_true")
    diff.set_defaults(func=cmd_diffuse_sweep)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
