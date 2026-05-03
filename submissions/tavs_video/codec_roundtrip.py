#!/usr/bin/env python3
"""FFmpeg encode/decode helpers for TAVS source-video candidates."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frame_utils import yuv420_to_rgb

from submissions.tavs_video.common import MODEL_H, MODEL_W


def flatten_frames(frames: torch.Tensor) -> torch.Tensor:
    """Return uint8 NHWC frames from [B,2,3,H,W] or [N,3,H,W]."""

    if frames.ndim == 5:
        frames = frames.flatten(0, 1)
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected [B,2,3,H,W] or [N,3,H,W], got {tuple(frames.shape)}")
    return frames.detach().clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def write_raw_rgb(frames: torch.Tensor, path: Path) -> tuple[int, int, int]:
    nhwc = flatten_frames(frames).cpu()
    n, h, w, _ = nhwc.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(nhwc.numpy().tobytes())
    return n, w, h


def codec_ext(codec: str) -> str:
    return ".mkv" if codec in {"svtav1", "libaom-av1", "vp9", "x265"} else ".mkv"


def codec_args(codec: str, crf: int) -> list[str]:
    if codec == "svtav1":
        return ["-c:v", "libsvtav1", "-crf", str(crf), "-preset", "10", "-pix_fmt", "yuv420p"]
    if codec == "libaom-av1":
        return ["-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0", "-cpu-used", "8", "-row-mt", "1", "-pix_fmt", "yuv420p"]
    if codec == "vp9":
        return ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-deadline", "good", "-cpu-used", "6", "-pix_fmt", "yuv420p"]
    if codec == "x265":
        return ["-c:v", "libx265", "-crf", str(crf), "-preset", "fast", "-x265-params", "log-level=error", "-pix_fmt", "yuv420p"]
    raise ValueError(f"unknown codec: {codec}")


def check_ffmpeg_codec(codec: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], check=True, text=True, capture_output=True).stdout
    required = {
        "svtav1": "libsvtav1",
        "libaom-av1": "libaom-av1",
        "vp9": "libvpx-vp9",
        "x265": "libx265",
    }[codec]
    return required in encoders


def encode_rawvideo(
    *,
    raw_path: Path,
    out_path: Path,
    width: int,
    height: int,
    codec: str,
    crf: int,
    fps: int = 20,
    quiet: bool = True,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error" if quiet else "info",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(raw_path),
        *codec_args(codec, crf),
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path.stat().st_size


def decode_video_to_chw(video_path: Path, *, target_hw: tuple[int, int] | None = (MODEL_H, MODEL_W)) -> torch.Tensor:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame).permute(2, 0, 1).float()
        if target_hw is not None and tuple(rgb.shape[-2:]) != tuple(target_hw):
            rgb = F.interpolate(rgb.unsqueeze(0), size=target_hw, mode="bicubic", align_corners=False).squeeze(0)
        frames.append(rgb.clamp(0, 255))
    container.close()
    if not frames:
        raise RuntimeError(f"decoded no frames from {video_path}")
    return torch.stack(frames, dim=0)


def roundtrip_frames(
    *,
    frames: torch.Tensor,
    codec: str,
    crf: int,
    work_dir: Path,
    label: str = "roundtrip",
    fps: int = 20,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / f"{label}.rgb"
    n, width, height = write_raw_rgb(frames, raw_path)
    video_path = work_dir / f"{label}_{codec}_crf{crf}{codec_ext(codec)}"
    encoded_bytes = encode_rawvideo(
        raw_path=raw_path,
        out_path=video_path,
        width=width,
        height=height,
        codec=codec,
        crf=crf,
        fps=fps,
    )
    decoded = decode_video_to_chw(video_path, target_hw=(height, width))
    if decoded.shape[0] != n:
        raise RuntimeError(f"decoded {decoded.shape[0]} frames, expected {n}")
    decoded = decoded.reshape(-1, 2, 3, height, width)
    return {
        "codec": codec,
        "crf": int(crf),
        "encoded_bytes": int(encoded_bytes),
        "video_path": str(video_path),
        "decoded": decoded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-pt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--codec", default="svtav1")
    parser.add_argument("--crf", type=int, default=57)
    args = parser.parse_args()

    frames = torch.load(args.frames_pt, map_location="cpu")
    if isinstance(frames, dict):
        frames = frames["frames"]
    result = roundtrip_frames(frames=frames, codec=args.codec, crf=args.crf, work_dir=args.out_dir)
    print({k: v for k, v in result.items() if k != "decoded"})


if __name__ == "__main__":
    main()
