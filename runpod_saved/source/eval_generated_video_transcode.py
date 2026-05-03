#!/usr/bin/env python
"""Transcode an already-generated submission output and evaluate it.

This is a diagnostic for the "store Candidate A output directly" path. It does
not create a contest-ready inflater; it packages the compressed generated video
only so the official archive-byte term is measured honestly, then evaluates the
decoded raw frames with evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
THIS_DIR = Path(__file__).resolve().parent

WIDTH = 1164
HEIGHT = 874
FPS = 20
CHANNELS = 3
ORIGINAL_BYTES = 37_545_489


@dataclass(frozen=True)
class TranscodeSpec:
    name: str
    codec: str
    crf: int | None
    preset: str
    bitrate: str | None = None

    @property
    def suffix(self) -> str:
        if self.codec in {"svtav1", "aomav1", "vp9"}:
            return ".mkv"
        if self.codec == "x265":
            return ".mkv"
        raise ValueError(f"unknown codec: {self.codec}")


def default_specs(suite: str) -> list[TranscodeSpec]:
    if suite == "target":
        return [
            TranscodeSpec("x265_b24k", "x265", None, "veryfast", "24k"),
            TranscodeSpec("x265_b32k", "x265", None, "veryfast", "32k"),
            TranscodeSpec("svtav1_b24k", "svtav1", None, "8", "24k"),
            TranscodeSpec("svtav1_b32k", "svtav1", None, "8", "32k"),
            TranscodeSpec("svtav1_b48k", "svtav1", None, "8", "48k"),
            TranscodeSpec("x265_crf51", "x265", 51, "veryfast"),
            TranscodeSpec("svtav1_crf63", "svtav1", 63, "8"),
            TranscodeSpec("vp9_b32k", "vp9", None, "4", "32k"),
        ]
    if suite == "quick":
        return [
            TranscodeSpec("svtav1_crf52", "svtav1", 52, "8"),
            TranscodeSpec("svtav1_crf56", "svtav1", 56, "8"),
            TranscodeSpec("svtav1_crf60", "svtav1", 60, "8"),
            TranscodeSpec("x265_crf40", "x265", 40, "veryfast"),
            TranscodeSpec("x265_crf44", "x265", 44, "veryfast"),
            TranscodeSpec("x265_crf48", "x265", 48, "veryfast"),
        ]
    if suite == "full":
        specs = [TranscodeSpec(f"svtav1_crf{crf}", "svtav1", crf, "8") for crf in (44, 48, 52, 56, 60, 64)]
        specs += [TranscodeSpec(f"x265_crf{crf}", "x265", crf, "veryfast") for crf in (36, 40, 44, 48)]
        specs += [TranscodeSpec(f"vp9_crf{crf}", "vp9", crf, "4") for crf in (42, 48, 54, 60)]
        return specs
    raise ValueError(f"unknown suite: {suite}")


def parse_spec(text: str) -> TranscodeSpec:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("candidate must look like NAME,codec=svtav1,crf=56[,preset=8]")
    name = parts[0]
    opts: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"candidate option lacks '=': {part}")
        key, value = part.split("=", 1)
        opts[key.strip()] = value.strip()
    try:
        codec = opts["codec"]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"missing option: {exc.args[0]}") from exc
    crf = int(opts["crf"]) if "crf" in opts else None
    bitrate = opts.get("bitrate")
    if crf is None and bitrate is None:
        raise argparse.ArgumentTypeError("candidate needs crf=VALUE or bitrate=VALUE")
    preset = opts.get("preset")
    if preset is None:
        preset = "8" if codec in {"svtav1", "aomav1"} else "veryfast" if codec == "x265" else "4"
    return TranscodeSpec(name, codec, crf, preset, bitrate)


def run(cmd: list[str], *, cwd: Path | None = None):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ffmpeg_encoders() -> set[str]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    encoders = set()
    for line in proc.stdout.splitlines():
        m = re.match(r"\s*[A-Z.]{6}\s+(\S+)", line)
        if m:
            encoders.add(m.group(1))
    return encoders


def encoder_name(codec: str) -> str:
    return {
        "svtav1": "libsvtav1",
        "aomav1": "libaom-av1",
        "x265": "libx265",
        "vp9": "libvpx-vp9",
    }[codec]


def encode_cmd(spec: TranscodeSpec, in_raw: Path, out_video: Path, *, fps: int, gop: int) -> list[str]:
    base = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        str(in_raw),
        "-an",
        "-pix_fmt",
        "yuv420p",
    ]
    rate_args = ["-crf", str(spec.crf)] if spec.crf is not None else ["-b:v", str(spec.bitrate)]
    if spec.codec == "svtav1":
        return base + [
            "-c:v",
            "libsvtav1",
            "-preset",
            spec.preset,
            "-g",
            str(gop),
        ] + rate_args + [str(out_video)]
    if spec.codec == "aomav1":
        aom_rate_args = ["-crf", str(spec.crf), "-b:v", "0"] if spec.crf is not None else ["-b:v", str(spec.bitrate)]
        return base + [
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            spec.preset,
            "-g",
            str(gop),
        ] + aom_rate_args + [str(out_video)]
    if spec.codec == "x265":
        x265_rate_args = ["-crf", str(spec.crf)] if spec.crf is not None else ["-b:v", str(spec.bitrate)]
        return base + [
            "-c:v",
            "libx265",
            "-preset",
            spec.preset,
            "-x265-params",
            f"log-level=error:keyint={gop}:min-keyint={gop}:scenecut=0",
        ] + x265_rate_args + [str(out_video)]
    if spec.codec == "vp9":
        vp9_rate_args = ["-crf", str(spec.crf), "-b:v", "0"] if spec.crf is not None else ["-b:v", str(spec.bitrate)]
        return base + [
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "good",
            "-cpu-used",
            spec.preset,
            "-row-mt",
            "1",
            "-g",
            str(gop),
        ] + vp9_rate_args + [str(out_video)]
    raise ValueError(f"unknown codec: {spec.codec}")


def decode_cmd(in_video: Path, out_raw: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        str(out_raw),
    ]


def unzip_archive(archive_zip: Path, out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        zf.extractall(out_dir)
    return out_dir


def inflate_base(args: argparse.Namespace, base_archive_dir: Path, base_inflated_dir: Path):
    if base_inflated_dir.exists() and not args.force_inflate:
        raws = list(base_inflated_dir.glob("*.raw"))
        if raws:
            print(f"Using existing inflated base raw files in {base_inflated_dir}", flush=True)
            return
    if base_inflated_dir.exists():
        shutil.rmtree(base_inflated_dir)
    base_inflated_dir.mkdir(parents=True, exist_ok=True)
    python_bin = args.python_bin or str(ROOT_DIR / ".venv" / "bin" / "python")
    if not Path(python_bin).exists():
        python_bin = sys.executable
    run(
        [
            python_bin,
            str(THIS_DIR / "inflate.py"),
            str(base_archive_dir),
            str(base_inflated_dir),
            str(args.video_names),
        ],
        cwd=ROOT_DIR,
    )


def raw_paths(video_names: Path, inflated_dir: Path) -> list[Path]:
    out = []
    for line in video_names.read_text().splitlines():
        name = line.strip()
        if not name:
            continue
        out.append(inflated_dir / f"{Path(name).stem}.raw")
    return out


def frame_count(raw_path: Path) -> int:
    frame_bytes = WIDTH * HEIGHT * CHANNELS
    size = raw_path.stat().st_size
    if size % frame_bytes:
        raise RuntimeError(f"raw file size is not frame-aligned: {raw_path} ({size} bytes)")
    return size // frame_bytes


def package_encoded(encoded_paths: list[Path], archive_zip: Path) -> int:
    if archive_zip.exists():
        archive_zip.unlink()
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in encoded_paths:
            zf.write(path, arcname=f"video_payload/{path.name}")
    return archive_zip.stat().st_size


def parse_report(report_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    line_map = {
        "Average PoseNet Distortion": "posenet_dist",
        "Average SegNet Distortion": "segnet_dist",
        "Compression Rate": "rate",
        "Submission file size": "archive_bytes",
    }
    for line in report_path.read_text().splitlines():
        for prefix, key in line_map.items():
            if prefix in line:
                value = line.split(":", 1)[1].strip().replace(",", "")
                metrics[key] = float(value.split()[0])
    missing = {"segnet_dist", "posenet_dist", "rate"} - metrics.keys()
    if missing:
        raise RuntimeError(f"failed to parse {report_path}; missing {sorted(missing)}")
    metrics["score"] = 100.0 * metrics["segnet_dist"] + math.sqrt(10.0 * metrics["posenet_dist"]) + 25.0 * metrics["rate"]
    metrics["quality_term"] = 100.0 * metrics["segnet_dist"] + math.sqrt(10.0 * metrics["posenet_dist"])
    return metrics


def evaluate_candidate(args: argparse.Namespace, candidate_dir: Path) -> dict[str, float]:
    python_bin = args.python_bin or str(ROOT_DIR / ".venv" / "bin" / "python")
    if not Path(python_bin).exists():
        python_bin = sys.executable
    report = candidate_dir / "report.txt"
    cmd = [
        python_bin,
        str(ROOT_DIR / "evaluate.py"),
        "--submission-dir",
        str(candidate_dir),
        "--uncompressed-dir",
        str(ROOT_DIR / "videos"),
        "--video-names-file",
        str(args.video_names),
        "--report",
        str(report),
        "--device",
        args.eval_device,
        "--batch-size",
        str(args.eval_batch_size),
    ]
    run(cmd, cwd=ROOT_DIR)
    return parse_report(report)


def write_summary(path: Path, row: dict[str, object]):
    with path.open("a") as f:
        print(json.dumps(row, sort_keys=True), file=f, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-archive-dir", type=Path)
    base.add_argument("--base-archive-zip", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--base-inflated-dir", type=Path)
    parser.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    parser.add_argument("--suite", choices=["target", "quick", "full"], default="target")
    parser.add_argument("--candidate", action="append", type=parse_spec)
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--gop", type=int, default=600)
    parser.add_argument("--python-bin", type=str, default=None)
    parser.add_argument("--force-inflate", action="store_true")
    parser.add_argument("--force-encode", action="store_true")
    parser.add_argument("--skip-unavailable", action="store_true", default=True)
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--keep-decoded", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.base_archive_zip is not None:
        base_archive_dir = unzip_archive(args.base_archive_zip, args.work_dir / "_base_archive")
    else:
        base_archive_dir = args.base_archive_dir
    if base_archive_dir is None or not base_archive_dir.exists():
        raise FileNotFoundError(f"base archive dir not found: {base_archive_dir}")

    base_inflated_dir = args.base_inflated_dir or (args.work_dir / "_base_inflated")
    inflate_base(args, base_archive_dir, base_inflated_dir)

    base_raws = raw_paths(args.video_names, base_inflated_dir)
    for raw in base_raws:
        if not raw.exists():
            raise FileNotFoundError(f"missing inflated raw: {raw}")
        print(f"base_raw {raw} frames={frame_count(raw)} bytes={raw.stat().st_size}", flush=True)

    specs = args.candidate or default_specs(args.suite)
    if args.max_candidates is not None:
        specs = specs[: args.max_candidates]

    available = ffmpeg_encoders()
    summary_path = args.work_dir / "generated_video_transcode_results.jsonl"
    print(f"Writing results to {summary_path}", flush=True)

    for spec in specs:
        enc = encoder_name(spec.codec)
        if enc not in available:
            msg = f"Skipping {spec.name}: ffmpeg encoder {enc} is unavailable"
            if args.skip_unavailable:
                print(msg, flush=True)
                continue
            raise RuntimeError(msg)

        candidate_dir = args.work_dir / spec.name
        encoded_dir = candidate_dir / "encoded"
        inflated_dir = candidate_dir / "inflated"
        encoded_dir.mkdir(parents=True, exist_ok=True)
        inflated_dir.mkdir(parents=True, exist_ok=True)

        encoded_paths: list[Path] = []
        for raw in base_raws:
            out_video = encoded_dir / f"{raw.stem}_{spec.name}{spec.suffix}"
            if args.force_encode or not out_video.exists():
                run(encode_cmd(spec, raw, out_video, fps=args.fps, gop=args.gop), cwd=ROOT_DIR)
            encoded_paths.append(out_video)

            out_raw = inflated_dir / raw.name
            if args.force_encode or not out_raw.exists():
                run(decode_cmd(out_video, out_raw), cwd=ROOT_DIR)
            expected_size = raw.stat().st_size
            actual_size = out_raw.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(f"decoded raw size mismatch for {out_raw}: {actual_size} != {expected_size}")

        archive_bytes = package_encoded(encoded_paths, candidate_dir / "archive.zip")
        row: dict[str, object] = {
            "name": spec.name,
            "codec": spec.codec,
            "crf": spec.crf,
            "bitrate": spec.bitrate,
            "preset": spec.preset,
            "archive_zip": str(candidate_dir / "archive.zip"),
            "archive_bytes": archive_bytes,
            "encoded_bytes": sum(p.stat().st_size for p in encoded_paths),
            "projected_rate_term": 25.0 * archive_bytes / ORIGINAL_BYTES,
        }

        if not args.encode_only:
            metrics = evaluate_candidate(args, candidate_dir)
            row.update(metrics)
            print(
                f"[{spec.name}] score={metrics['score']:.5f} quality={metrics['quality_term']:.5f} "
                f"seg={metrics['segnet_dist']:.8f} pose={metrics['posenet_dist']:.8f} bytes={archive_bytes}",
                flush=True,
            )
        else:
            print(f"[{spec.name}] bytes={archive_bytes}", flush=True)

        write_summary(summary_path, row)
        if not args.keep_decoded and inflated_dir.exists():
            shutil.rmtree(inflated_dir)

    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
