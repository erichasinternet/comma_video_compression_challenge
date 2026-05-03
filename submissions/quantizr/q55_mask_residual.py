#!/usr/bin/env python
"""Build an exact-class AV1 mask predictor plus sparse residual repair.

The archive decodes a lossy AV1 mask stream, rounds it back to semantic
classes, applies a sparse class residual, and feeds the exact #55 class tensor
to the generator. If the residual verifies, quality should match the base
archive; the diagnostic is whether base+residual is smaller than mask.obu.br.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import brotli
import numpy as np

from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    POSE_PAYLOAD,
    POSE_QPACK_PAYLOAD,
    append_jsonl,
    make_archive_zip,
    materialize_submission,
    metric_record,
    run_evaluate_submission,
    summarize_archive,
    unzip_archive,
    write_json,
)
from q55_mask_alloc import (
    PALETTES,
    decode_archive_masks,
    mask_boundary_scores,
    mask_histograms,
    order_group,
    parse_palette,
)


MASK_RESIDUAL_MANIFEST = "mask_residual.json.br"
MASK_RESIDUAL_PAYLOAD = "mask_residual.bin.br"
MASK_BASE_PAYLOAD = "mask_base.obu.br"
RESIDUAL_FORMAT = "quantizr_mask_residual_v1"


def encode_varints(values: np.ndarray) -> bytes:
    out = bytearray()
    for raw in values.astype(np.int64, copy=False):
        value = int(raw)
        if value < 0:
            raise ValueError("varint values must be non-negative")
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
    return bytes(out)


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


def unpack_bits(payload: bytes, count: int, bits: int) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(count, dtype=np.uint16)
    acc = 0
    acc_bits = 0
    j = 0
    mask = (1 << bits) - 1
    for byte in data:
        acc |= int(byte) << acc_bits
        acc_bits += 8
        while acc_bits >= bits and j < count:
            out[j] = acc & mask
            acc >>= bits
            acc_bits -= bits
            j += 1
    if j != count:
        raise RuntimeError(f"decoded {j} packed values, expected {count}")
    return out


def decode_varints(payload: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int64)
    value = 0
    shift = 0
    j = 0
    for byte in payload:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        if j >= count:
            raise RuntimeError("too many varints in residual stream")
        out[j] = value
        j += 1
        value = 0
        shift = 0
    if shift:
        raise RuntimeError("varint stream ended mid-value")
    if j != count:
        raise RuntimeError(f"decoded {j} varints, expected {count}")
    return out


def decode_obu_br(mask_br: Path, palette: list[int]) -> np.ndarray:
    import av

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
        tmp_obu.write(brotli.decompress(mask_br.read_bytes()))
        tmp_obu_path = tmp_obu.name
    frames = []
    palette_arr = np.asarray(palette, dtype=np.float32)
    try:
        container = av.open(tmp_obu_path)
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="gray")
            dist = np.abs(img.astype(np.float32)[..., None] - palette_arr[None, None, :])
            frames.append(dist.argmin(axis=-1).astype(np.uint8))
        container.close()
    finally:
        os.remove(tmp_obu_path)
    if not frames:
        raise RuntimeError(f"no frames decoded from {mask_br}")
    return np.stack(frames).astype(np.uint8, copy=False)


def encode_base_mask_stream(
    *,
    masks: np.ndarray,
    encoded_order: list[int],
    palette: list[int],
    crf: int,
    archive_dir: Path,
    cpu_used: int,
    ffmpeg_path: str,
) -> Path:
    raw_path = archive_dir / "mask_base.yuv"
    obu_path = archive_dir / "mask_base.obu"
    br_path = archive_dir / MASK_BASE_PAYLOAD
    palette_arr = np.asarray(palette, dtype=np.uint8)
    with open(raw_path, "wb") as f:
        for idx in encoded_order:
            f.write(palette_arr[masks[idx]].tobytes())

    ffmpeg_cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        "512x384",
        "-r",
        "10",
        "-i",
        str(raw_path),
        "-c:v",
        "libaom-av1",
        "-crf",
        str(crf),
        "-cpu-used",
        str(cpu_used),
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
        "obu",
        str(obu_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    br_path.write_bytes(brotli.compress(obu_path.read_bytes(), quality=11, lgwin=24))
    raw_path.unlink(missing_ok=True)
    obu_path.unlink(missing_ok=True)
    return br_path


def restored_order(decoded_ordered: np.ndarray, encoded_order: list[int]) -> np.ndarray:
    restored = np.empty_like(decoded_ordered)
    for local_idx, original_idx in enumerate(encoded_order):
        restored[int(original_idx)] = decoded_ordered[local_idx]
    return restored


def build_residual(exact: np.ndarray, predicted: np.ndarray) -> tuple[bytes, dict, np.ndarray, np.ndarray]:
    if exact.shape != predicted.shape:
        raise RuntimeError(f"shape mismatch: exact {exact.shape}, predicted {predicted.shape}")
    flat_exact = exact.reshape(-1)
    flat_pred = predicted.reshape(-1)
    changed = np.flatnonzero(flat_exact != flat_pred).astype(np.int64, copy=False)
    values = flat_exact[changed].astype(np.uint8, copy=False)
    if changed.size:
        deltas = np.empty_like(changed)
        deltas[0] = changed[0]
        deltas[1:] = changed[1:] - changed[:-1]
    else:
        deltas = changed
    pos_bytes = encode_varints(deltas)
    value_bytes = pack_bits(values, 3)
    payload = b"QMR1" + len(changed).to_bytes(4, "little") + len(pos_bytes).to_bytes(4, "little")
    payload += len(value_bytes).to_bytes(4, "little") + pos_bytes + value_bytes
    stats = {
        "changed_pixels": int(changed.size),
        "changed_fraction": float(changed.size / flat_exact.size),
        "position_bytes_raw": len(pos_bytes),
        "value_bytes_raw": len(value_bytes),
        "payload_bytes_raw": len(payload),
    }
    return payload, stats, changed, values


def verify_residual(predicted: np.ndarray, changed: np.ndarray, values: np.ndarray, exact: np.ndarray) -> None:
    repaired = predicted.copy().reshape(-1)
    repaired[changed] = values
    repaired = repaired.reshape(exact.shape)
    if not np.array_equal(repaired, exact):
        diff = int(np.count_nonzero(repaired != exact))
        raise RuntimeError(f"residual repair failed, {diff} class pixels still differ")


def choose_payload(archive_dir: Path, primary: str, fallback: str) -> str:
    if (archive_dir / primary).exists():
        return primary
    if (archive_dir / fallback).exists():
        return fallback
    raise FileNotFoundError(f"missing payload: {primary} or {fallback}")


def get_ffmpeg_path() -> str:
    local_ffmpeg = Path(__file__).resolve().parent / "ffmpeg"
    if local_ffmpeg.is_file() and os.access(local_ffmpeg, os.X_OK):
        return str(local_ffmpeg.resolve())
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    raise FileNotFoundError("ffmpeg not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--crf", type=int, required=True)
    parser.add_argument("--predictor-mask-archive", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cpu")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--palette", default="legacy")
    parser.add_argument("--order", choices=["original", "boundary", "hist"], default="original")
    parser.add_argument("--cpu-used", type=int, default=0)
    parser.add_argument("--label", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--force-av-dataset", action="store_true")
    args = parser.parse_args()

    base_archive = args.base_archive.resolve()
    if not base_archive.exists():
        raise FileNotFoundError(base_archive)

    palette = parse_palette(args.palette)
    label = args.label or f"qmask_resid_crf{args.crf}_{args.order}_{args.palette}"
    run_dir = args.out_dir / label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    submission_dir = run_dir / "submission"
    run_dir.mkdir(parents=True, exist_ok=True)

    unzip_archive(base_archive, archive_dir)
    model_payload = choose_payload(archive_dir, MODEL_QPACK_PAYLOAD, MODEL_PAYLOAD)
    pose_payload = choose_payload(archive_dir, POSE_QPACK_PAYLOAD, POSE_PAYLOAD)

    exact_masks = decode_archive_masks(archive_dir / MASK_PAYLOAD)
    scores = mask_boundary_scores(exact_masks)
    hists = mask_histograms(exact_masks)
    all_indices = np.arange(exact_masks.shape[0])
    encoded_order = order_group(all_indices, scores, hists, args.order)

    for path in archive_dir.glob("mask*.obu*"):
        path.unlink()
    for name in (MASK_RESIDUAL_MANIFEST, MASK_RESIDUAL_PAYLOAD, "mask_mix.json.br"):
        (archive_dir / name).unlink(missing_ok=True)

    if args.predictor_mask_archive:
        predictor_archive = args.predictor_mask_archive.resolve()
        if not predictor_archive.exists():
            raise FileNotFoundError(predictor_archive)
        with zipfile.ZipFile(predictor_archive) as z:
            if MASK_PAYLOAD in z.namelist():
                with z.open(MASK_PAYLOAD) as src, open(archive_dir / MASK_BASE_PAYLOAD, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            elif MASK_BASE_PAYLOAD in z.namelist():
                with z.open(MASK_BASE_PAYLOAD) as src, open(archive_dir / MASK_BASE_PAYLOAD, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                raise FileNotFoundError(f"{predictor_archive} has no mask predictor payload")
        base_payload_path = archive_dir / MASK_BASE_PAYLOAD
    else:
        base_payload_path = encode_base_mask_stream(
            masks=exact_masks,
            encoded_order=encoded_order,
            palette=palette,
            crf=args.crf,
            archive_dir=archive_dir,
            cpu_used=args.cpu_used,
            ffmpeg_path=get_ffmpeg_path(),
        )
    decoded_ordered = decode_obu_br(base_payload_path, palette=palette)
    predicted_masks = restored_order(decoded_ordered, encoded_order)

    residual_raw, residual_stats, changed, values = build_residual(exact_masks, predicted_masks)
    verify_residual(predicted_masks, changed, values, exact_masks)
    residual_br = archive_dir / MASK_RESIDUAL_PAYLOAD
    residual_br.write_bytes(brotli.compress(residual_raw, quality=11, lgwin=24))

    manifest = {
        "format": RESIDUAL_FORMAT,
        "num_frames": int(exact_masks.shape[0]),
        "height": int(exact_masks.shape[1]),
        "width": int(exact_masks.shape[2]),
        "base_payload": MASK_BASE_PAYLOAD,
        "residual_payload": MASK_RESIDUAL_PAYLOAD,
        "palette": palette,
        "palette_name": args.palette if args.palette in PALETTES else None,
        "crf": args.crf,
        "cpu_used": args.cpu_used,
        "predictor_mask_archive": str(args.predictor_mask_archive.resolve()) if args.predictor_mask_archive else None,
        "order": args.order,
        "encoded_order": encoded_order,
        "residual": residual_stats | {"payload_bytes_br": residual_br.stat().st_size},
        "base_payload_bytes_br": base_payload_path.stat().st_size,
    }
    manifest_br = archive_dir / MASK_RESIDUAL_MANIFEST
    manifest_br.write_bytes(
        brotli.compress(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"), quality=11, lgwin=24)
    )

    payload_names = [model_payload, pose_payload, MASK_BASE_PAYLOAD, MASK_RESIDUAL_PAYLOAD, MASK_RESIDUAL_MANIFEST]
    make_archive_zip(archive_dir, archive_zip, payload_names)
    archive_summary = summarize_archive(archive_zip)
    mask_payload_bytes = (
        base_payload_path.stat().st_size + residual_br.stat().st_size + manifest_br.stat().st_size
    )
    codec_record = {
        "label": label,
        "base_archive": str(base_archive),
        "predictor_mask_archive": str(args.predictor_mask_archive.resolve()) if args.predictor_mask_archive else None,
        "base_archive_summary": summarize_archive(base_archive),
        "archive_zip": str(archive_zip),
        "archive_summary": archive_summary,
        "archive_bytes": archive_summary["archive_bytes"],
        "archive_sha256": archive_summary["archive_sha256"],
        "crf": args.crf,
        "palette": palette,
        "palette_name": args.palette if args.palette in PALETTES else None,
        "order": args.order,
        "model_payload": model_payload,
        "pose_payload": pose_payload,
        "mask_payload_bytes": mask_payload_bytes,
        "mask_payload_savings_vs_q55": 219_472 - mask_payload_bytes,
        "exact_class_recovered": True,
        "residual": manifest["residual"],
        "base_payload_bytes_br": base_payload_path.stat().st_size,
        "manifest_bytes_br": manifest_br.stat().st_size,
    }

    if args.evaluate:
        materialize_submission(archive_zip=archive_zip, submission_dir=submission_dir, inflate_mode="modified")
        env = {"FORCE_AV_DATASET": "1"} if args.force_av_dataset else None
        report_path = run_evaluate_submission(submission_dir, args.device, args.video_names, env=env)
        eval_record = metric_record(
            label=label,
            archive_zip=submission_dir / "archive.zip",
            device=args.device,
            report_path=report_path,
            extra=codec_record,
        )
        write_json(run_dir / "metrics.json", eval_record)
        append_jsonl(args.out_dir / "mask_residual_results.jsonl", eval_record)
    else:
        if submission_dir.exists():
            shutil.rmtree(submission_dir)
        submission_dir.mkdir(parents=True)
        shutil.copy2(archive_zip, submission_dir / "archive.zip")
        materialize_submission(archive_zip=archive_zip, submission_dir=submission_dir, inflate_mode="modified")
        write_json(run_dir / "metrics.json", codec_record)
        append_jsonl(args.out_dir / "mask_residual_results.jsonl", codec_record)

    print(json.dumps(codec_record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
