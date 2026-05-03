#!/usr/bin/env python
"""Pack and optionally evaluate a hard-sample original-pair sidecar.

The base layer is an existing q55/qpack inflated output. The sidecar stores both
original frames for selected hard samples, encoded chronologically as a small
video stream. Evaluation replaces only those selected pairs with decoded sidecar
frames and computes the official distortion terms locally.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from frame_utils import AVVideoDataset, TensorVideoDataset, camera_size
from modules import DistortionNet, posenet_sd_path, segnet_sd_path
from q55_common import ORIGINAL_BYTES, quality_term, score_from_bytes, sha256_file, write_json


CAMERA_W, CAMERA_H = camera_size


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg not found")
    return exe


def parse_resolution(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x", 1)
    return int(w), int(h)


def load_ranked_indices(path: Path, top_k: int) -> list[int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [int(row["index"]) for row in rows[:top_k]]
    return sorted(selected)


def load_video_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def save_sidecar_frames(args, selected_indices: list[int], frames_dir: Path) -> None:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    selected = set(selected_indices)
    names = load_video_names(args.video_names_file)
    ds = AVVideoDataset(names, data_dir=args.uncompressed_dir, batch_size=args.batch_size, device=torch.device("cpu"))
    ds.prepare_data()

    out_w, out_h = parse_resolution(args.resolution)
    global_index = 0
    sidecar_frame_index = 0
    for _, _, batch in ds:
        for pair in batch:
            if global_index in selected:
                for t in range(2):
                    arr = pair[t].numpy()
                    img = Image.fromarray(arr)
                    if (out_w, out_h) != (CAMERA_W, CAMERA_H):
                        img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
                    img.save(frames_dir / f"{sidecar_frame_index:06d}.png", optimize=True)
                    sidecar_frame_index += 1
            global_index += 1
    expected = 2 * len(selected_indices)
    if sidecar_frame_index != expected:
        raise RuntimeError(f"wrote {sidecar_frame_index} sidecar frames, expected {expected}")


def codec_args(codec: str, crf: int, out_path: Path) -> list[str]:
    if codec == "svtav1":
        return ["-c:v", "libsvtav1", "-crf", str(crf), "-preset", "8", "-pix_fmt", "yuv420p", str(out_path)]
    if codec == "vp9":
        return ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-deadline", "good", "-cpu-used", "4", "-pix_fmt", "yuv420p", str(out_path)]
    if codec == "x265":
        return ["-c:v", "libx265", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", str(out_path)]
    raise ValueError(f"unsupported codec: {codec}")


def encode_sidecar(frames_dir: Path, video_path: Path, *, codec: str, crf: int, fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists():
        video_path.unlink()
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
        *codec_args(codec, crf, video_path),
    ]
    subprocess.run(cmd, check=True)


def make_sidecar_zip(video_path: Path, zip_path: Path, meta: dict) -> None:
    meta_path = zip_path.parent / "hard_sidecar_meta.json"
    meta_path.write_text(json.dumps(meta, sort_keys=True) + "\n")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        z.write(video_path, arcname=video_path.name)
        z.write(meta_path, arcname=meta_path.name)


def decode_sidecar_to_raw(video_path: Path, raw_path: Path) -> None:
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
        f"scale={CAMERA_W}:{CAMERA_H}:flags=lanczos",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        str(raw_path),
    ]
    subprocess.run(cmd, check=True)


def evaluate_mixed(args, selected_indices: list[int], decoded_raw: Path, total_archive_bytes: int) -> dict:
    device = torch.device(args.device)
    names = load_video_names(args.video_names_file)
    index_to_sidecar = {idx: pos for pos, idx in enumerate(selected_indices)}
    sidecar = np.memmap(decoded_raw, dtype=np.uint8, mode="r").reshape((len(selected_indices), 2, CAMERA_H, CAMERA_W, 3))

    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    ds_gt = AVVideoDataset(names, data_dir=args.uncompressed_dir, batch_size=args.batch_size, device=device)
    ds_gt.prepare_data()
    ds_base = TensorVideoDataset(names, data_dir=args.base_inflated_dir, batch_size=args.batch_size, device=device)
    ds_base.prepare_data()

    global_index = 0
    total_pose = 0.0
    total_seg = 0.0
    total = 0
    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_base) in zip(ds_gt, ds_base):
            comp = batch_base.clone()
            for j in range(comp.shape[0]):
                pos = index_to_sidecar.get(global_index)
                if pos is not None:
                    comp[j] = torch.from_numpy(np.asarray(sidecar[pos]).copy())
                global_index += 1
            batch_gt = batch_gt.to(device)
            comp = comp.to(device)
            pose, seg = net.compute_distortion(batch_gt, comp)
            total_pose += float(pose.sum().item())
            total_seg += float(seg.sum().item())
            total += int(comp.shape[0])
    seg_mean = total_seg / total
    pose_mean = total_pose / total
    quality = quality_term(seg_mean, pose_mean)
    return {
        "samples": total,
        "segnet_dist": seg_mean,
        "posenet_dist": pose_mean,
        "quality": quality,
        "total_archive_bytes": total_archive_bytes,
        "rate_term": 25.0 * total_archive_bytes / ORIGINAL_BYTES,
        "score": score_from_bytes(seg_mean, pose_mean, total_archive_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=REPO_ROOT / "submissions/q55_fp16_pose_int10/archive.zip")
    parser.add_argument(
        "--base-inflated-dir",
        type=Path,
        default=REPO_ROOT / "submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int10_cpu/submission/inflated",
    )
    parser.add_argument(
        "--per-sample-jsonl",
        type=Path,
        default=REPO_ROOT / "submissions/quantizr/experiments/q55_hard_sample_oracle/per_sample_quality.jsonl",
    )
    parser.add_argument("--uncompressed-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names-file", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--resolution", default="256x192")
    parser.add_argument("--codec", choices=["svtav1", "vp9", "x265"], default="svtav1")
    parser.add_argument("--crf", type=int, default=63)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    selected_indices = load_ranked_indices(args.per_sample_jsonl, args.top_k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    video_path = args.out_dir / f"hard_sidecar_{args.codec}_crf{args.crf}_{args.resolution}.mkv"
    zip_path = args.out_dir / "hard_sidecar.zip"
    raw_path = args.out_dir / "decoded_sidecar.raw"

    save_sidecar_frames(args, selected_indices, frames_dir)
    encode_sidecar(frames_dir, video_path, codec=args.codec, crf=args.crf, fps=args.fps)
    meta = {
        "format": "q55_hard_sample_sidecar_v1",
        "indices": selected_indices,
        "top_k": args.top_k,
        "resolution": args.resolution,
        "codec": args.codec,
        "crf": args.crf,
        "fps": args.fps,
        "video_payload": video_path.name,
    }
    make_sidecar_zip(video_path, zip_path, meta)

    total_archive_bytes = args.base_archive.stat().st_size + zip_path.stat().st_size
    record = {
        "base_archive": str(args.base_archive),
        "base_archive_sha256": sha256_file(args.base_archive),
        "base_archive_bytes": args.base_archive.stat().st_size,
        "sidecar_zip": str(zip_path),
        "sidecar_zip_bytes": zip_path.stat().st_size,
        "sidecar_video_bytes": video_path.stat().st_size,
        "total_archive_bytes_estimate": total_archive_bytes,
        "selected_indices": selected_indices,
        "top_k": args.top_k,
        "resolution": args.resolution,
        "codec": args.codec,
        "crf": args.crf,
    }

    if not args.skip_eval:
        decode_sidecar_to_raw(video_path, raw_path)
        record["decoded_metrics"] = evaluate_mixed(args, selected_indices, raw_path, total_archive_bytes)
        if not args.keep_raw:
            raw_path.unlink(missing_ok=True)

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    write_json(args.out_dir / "metrics.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
