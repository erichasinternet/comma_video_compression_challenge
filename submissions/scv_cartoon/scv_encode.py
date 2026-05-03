#!/usr/bin/env python
"""Encode/decode helpers for SCV frame sequences."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from scv_eval_proxy import CAMERA_H, CAMERA_W, write_json


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg not found")
    return exe


def available_encoder(name: str) -> bool:
    proc = subprocess.run([ffmpeg(), "-hide_banner", "-encoders"], check=True, capture_output=True, text=True)
    return name in proc.stdout


def codec_plan(codec: str, crf: int, out_path: Path) -> list[str]:
    if codec == "svtav1":
        return ["-c:v", "libsvtav1", "-crf", str(crf), "-preset", "8", "-pix_fmt", "yuv420p", str(out_path)]
    if codec == "libaom-av1":
        return ["-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0", "-cpu-used", "6", "-row-mt", "1", "-pix_fmt", "yuv420p", str(out_path)]
    if codec == "vp9":
        return ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-deadline", "good", "-cpu-used", "4", "-pix_fmt", "yuv420p", str(out_path)]
    if codec == "x265":
        return ["-c:v", "libx265", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", str(out_path)]
    raise ValueError(f"unknown codec: {codec}")


def encoder_name(codec: str) -> str:
    return {
        "svtav1": "libsvtav1",
        "libaom-av1": "libaom-av1",
        "vp9": "libvpx-vp9",
        "x265": "libx265",
    }[codec]


def default_codecs() -> list[str]:
    out = []
    for codec in ("svtav1", "libaom-av1", "vp9", "x265"):
        if available_encoder(encoder_name(codec)):
            out.append(codec)
    return out


def crf_grid(codec: str, quick: bool) -> list[int]:
    if quick:
        return {"svtav1": [45, 55], "libaom-av1": [45, 55], "vp9": [45, 55], "x265": [36, 44]}[codec]
    return {
        "svtav1": [35, 40, 45, 50, 55, 60],
        "libaom-av1": [35, 40, 45, 50, 55, 60],
        "vp9": [35, 40, 45, 50, 55, 60],
        "x265": [28, 32, 36, 40, 44],
    }[codec]


def encode_video(frames_dir: Path, out_video: Path, *, codec: str, crf: int, fps: int = 20) -> dict:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    if out_video.exists():
        out_video.unlink()
    cmd = [
        ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        *codec_plan(codec, crf, out_video),
    ]
    subprocess.run(cmd, check=True)
    return {"video": str(out_video), "video_bytes": out_video.stat().st_size, "codec": codec, "crf": crf, "fps": fps}


def decode_video_to_raw(video_path: Path, raw_path: Path) -> dict:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raw_path.unlink()
    cmd = [
        ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={CAMERA_W}:{CAMERA_H}:flags=bicubic",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        str(raw_path),
    ]
    subprocess.run(cmd, check=True)
    return {"raw": str(raw_path), "raw_bytes": raw_path.stat().st_size}


def make_archive(video_path: Path, archive_zip: Path, *, subset: int, codec: str, crf: int, fps: int = 20) -> dict:
    archive_zip.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "scv_cartoon_v1",
        "video_payload": video_path.name,
        "subset": int(subset),
        "codec": codec,
        "crf": int(crf),
        "fps": int(fps),
        "width": CAMERA_W,
        "height": CAMERA_H,
    }
    meta_path = archive_zip.parent / "meta.json"
    meta_path.write_text(json.dumps(meta, sort_keys=True) + "\n")
    if archive_zip.exists():
        archive_zip.unlink()
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as z:
        z.write(video_path, arcname=video_path.name)
        z.write(meta_path, arcname="meta.json")
    return {"archive_zip": str(archive_zip), "archive_bytes": archive_zip.stat().st_size, "meta": meta}


def encode_one(frames_dir: Path, out_dir: Path, *, codec: str, crf: int, subset: int, fps: int = 20) -> dict:
    suffix = "mkv"
    video_path = out_dir / f"synthetic_{codec}_crf{crf}.{suffix}"
    enc = encode_video(frames_dir, video_path, codec=codec, crf=crf, fps=fps)
    archive_zip = out_dir / f"archive_{codec}_crf{crf}.zip"
    archive = make_archive(video_path, archive_zip, subset=subset, codec=codec, crf=crf, fps=fps)
    raw = decode_video_to_raw(video_path, out_dir / f"decoded_{codec}_crf{crf}.raw")
    record = {**enc, **archive, **raw}
    write_json(out_dir / f"encode_{codec}_crf{crf}.json", record)
    return record

