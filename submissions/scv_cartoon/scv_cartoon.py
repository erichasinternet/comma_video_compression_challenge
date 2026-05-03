#!/usr/bin/env python
"""CLI for the Semantic Cartoon Video Codec prototype."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from scv_encode import crf_grid, default_codecs, encode_one, ffmpeg
from scv_eval_proxy import (
    CAMERA_H,
    CAMERA_W,
    evaluate_pairs,
    load_evaluators,
    load_rgb_subset,
    metric_table,
    pick_device,
    read_raw_pairs,
    write_json,
    build_targets,
)
from scv_render import render_variant


def cmd_render(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    record = render_variant(args)
    print(json.dumps(record, indent=2, sort_keys=True))


def evaluate_decoded(args, raw_path: Path, archive_bytes: int) -> dict:
    device = pick_device(args.device)
    segnet, posenet = load_evaluators(device)
    gt = load_rgb_subset(
        video_names=args.video_names,
        video_dir=args.video_dir,
        subset=args.subset,
        offset=args.offset,
        batch_size=args.decode_batch_size,
    )
    targets = build_targets(gt, segnet, posenet, device, batch_size=args.eval_batch_size)
    pairs = read_raw_pairs(raw_path, max_pairs=args.subset)
    return evaluate_pairs(
        pairs,
        targets["seg_targets"],
        targets["pose_targets"],
        segnet,
        posenet,
        device,
        batch_size=args.eval_batch_size,
        archive_bytes=archive_bytes,
    )


def cmd_encode(args) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_dir = args.out.parent
    codecs = [args.codec] if args.codec else default_codecs()
    if not codecs:
        raise RuntimeError("no supported ffmpeg encoders found")
    eval_ctx = None
    if args.evaluate:
        device = pick_device(args.device)
        segnet, posenet = load_evaluators(device)
        gt = load_rgb_subset(
            video_names=args.video_names,
            video_dir=args.video_dir,
            subset=args.subset,
            offset=args.offset,
            batch_size=args.decode_batch_size,
        )
        targets = build_targets(gt, segnet, posenet, device, batch_size=args.eval_batch_size)
        eval_ctx = (device, segnet, posenet, targets)
    records = []
    for codec in codecs:
        for crf in ([args.crf] if args.crf is not None else crf_grid(codec, quick=args.quick)):
            rec = encode_one(args.frames, out_dir, codec=codec, crf=crf, subset=args.subset, fps=args.fps)
            rec["extrapolated_600_bytes"] = int(round(rec["archive_bytes"] * 600 / args.subset))
            rec["byte_gates"] = {
                "weak_extrapolated_le_240k": rec["extrapolated_600_bytes"] <= 240_000,
                "strong_extrapolated_le_220k": rec["extrapolated_600_bytes"] <= 220_000,
                "excellent_extrapolated_le_200k": rec["extrapolated_600_bytes"] <= 200_000,
            }
            if args.evaluate:
                assert eval_ctx is not None
                device, segnet, posenet, targets = eval_ctx
                pairs = read_raw_pairs(Path(rec["raw"]), max_pairs=args.subset)
                metrics = evaluate_pairs(
                    pairs,
                    targets["seg_targets"],
                    targets["pose_targets"],
                    segnet,
                    posenet,
                    device,
                    batch_size=args.eval_batch_size,
                    archive_bytes=rec["extrapolated_600_bytes"],
                )
                rec["decoded_metrics"] = metrics
                rec["score_table_extrapolated"] = metric_table(
                    metrics["segnet_dist"],
                    metrics["posenet_dist"],
                    rec["extrapolated_600_bytes"],
                )
                if not args.keep_decoded_raw:
                    Path(rec["raw"]).unlink(missing_ok=True)
                    rec["raw"] = None
            records.append(rec)
    if args.evaluate:
        records = sorted(records, key=lambda r: (r["score_table_extrapolated"]["score"], r["archive_bytes"]))
    else:
        records = sorted(records, key=lambda r: r["archive_bytes"])
    write_json(out_dir / "encode_grid_metrics.json", {"records": records})

    best = records[0]
    # Keep a stable archive.zip for evaluate.sh/inflate.sh smoke tests.
    shutil.copy2(best["archive_zip"], args.out)
    print(json.dumps({"best": best, "records": records}, indent=2, sort_keys=True))


def cmd_inflate(args) -> None:
    archive_dir = args.archive_dir
    out_dir = args.out_dir
    file_list = args.file_list
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = archive_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        video = archive_dir / meta["video_payload"]
    else:
        videos = sorted([p for p in archive_dir.iterdir() if p.suffix.lower() in {".mkv", ".mp4", ".ivf", ".webm"}])
        if not videos:
            raise FileNotFoundError(f"no synthetic video payload found in {archive_dir}")
        video = videos[0]
    files = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
    if not files:
        raise ValueError("empty file list")
    # Prototype archives contain one concatenated synthetic stream for the first public file.
    base = Path(files[0]).stem
    raw_path = out_dir / f"{base}.raw"
    cmd = [
        ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"scale={CAMERA_W}:{CAMERA_H}:flags=bicubic",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        str(raw_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"decoded {video.name} -> {raw_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    render = sub.add_parser("render")
    render.add_argument("--subset", type=int, default=64)
    render.add_argument("--offset", type=int, default=0)
    render.add_argument("--variant", choices=["scv0", "scv1", "scvt"], default="scv0")
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--cache-dir", type=Path, default=HERE / "experiments" / "cache")
    render.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    render.add_argument("--video-names", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    render.add_argument("--device", default=None)
    render.add_argument("--decode-batch-size", type=int, default=8)
    render.add_argument("--eval-batch-size", type=int, default=8)
    render.add_argument("--cache-logits", action="store_true")
    render.add_argument("--rebuild-cache", action="store_true")
    render.add_argument("--texture-size", type=int, nargs=2, default=[48, 64], metavar=("H", "W"))
    render.add_argument("--texture-strength", type=float, default=0.35)
    render.add_argument("--scvt-blur-size", type=int, nargs=2, default=[146, 194], metavar=("H", "W"))
    render.add_argument("--scvt-color-step", type=int, default=12)
    render.add_argument("--scvt-mask-smooth", type=float, default=0.28)
    render.add_argument("--scvt-edge-keep", type=float, default=0.65)
    render.add_argument("--archive-bytes-for-projection", type=int, default=220_000)
    render.add_argument("--keep-raw", action="store_true")
    render.set_defaults(func=cmd_render)

    encode = sub.add_parser("encode")
    encode.add_argument("--frames", type=Path, required=True)
    encode.add_argument("--out", type=Path, required=True)
    encode.add_argument("--codec", choices=["svtav1", "libaom-av1", "vp9", "x265"], default=None)
    encode.add_argument("--crf", type=int, default=None)
    encode.add_argument("--subset", type=int, default=64)
    encode.add_argument("--offset", type=int, default=0)
    encode.add_argument("--fps", type=int, default=20)
    encode.add_argument("--quick", action="store_true")
    encode.add_argument("--evaluate", action="store_true")
    encode.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    encode.add_argument("--video-names", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    encode.add_argument("--device", default=None)
    encode.add_argument("--decode-batch-size", type=int, default=8)
    encode.add_argument("--eval-batch-size", type=int, default=8)
    encode.add_argument("--keep-decoded-raw", action="store_true")
    encode.set_defaults(func=cmd_encode)

    inflate = sub.add_parser("inflate")
    inflate.add_argument("--archive-dir", type=Path, required=True)
    inflate.add_argument("--out-dir", type=Path, required=True)
    inflate.add_argument("--file-list", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    inflate.set_defaults(func=cmd_inflate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
