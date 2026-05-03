#!/usr/bin/env python
"""Regenerate #55 SegNet masks at a requested CRF and run zero-step eval."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    POSE_PAYLOAD,
    REPO_ROOT,
    append_jsonl,
    ensure_legacy_payloads,
    make_archive_zip,
    materialize_submission,
    metric_record,
    run_evaluate_submission,
    summarize_archive,
    unzip_archive,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--crf", type=int, choices=[50, 52, 54, 56, 58, 60], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inflate-mode", choices=["upstream", "modified"], default="modified")
    parser.add_argument("--label", default=None)
    parser.add_argument("--force-av-dataset", action="store_true")
    args = parser.parse_args()

    base_archive = args.base_archive.resolve()
    if not base_archive.exists():
        raise FileNotFoundError(base_archive)

    import torch
    import compress as q55
    from frame_utils import AVVideoDataset

    label = args.label or f"q55_crf{args.crf}_{args.device}"
    run_dir = args.out_dir / label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    submission_dir = run_dir / "submission"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "crf_swap.log")],
    )

    unzip_archive(base_archive, archive_dir)
    ensure_legacy_payloads(archive_dir)
    for path in archive_dir.glob("mask*.obu*"):
        path.unlink()

    device = torch.device(args.device)
    files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]

    logging.info("Loading SegNet and source videos for CRF mask swap...")
    segnet = q55.SegNet().eval().to(device)
    segnet.load_state_dict(q55.load_file(q55.segnet_sd_path, device=str(device)))
    for p in segnet.parameters():
        p.requires_grad = False

    if args.force_av_dataset or device.type != "cuda":
        logging.info("Preloading raw video RGB pairs via AVVideoDataset...")
        os.environ["FORCE_AV_DATASET"] = "1"
        ds = AVVideoDataset(files, data_dir=args.video_dir, batch_size=args.batch_size, device=torch.device("cpu"))
        ds.prepare_data()
        batches = [batch.cpu().contiguous() for _, _, batch in ds]
        if not batches:
            raise RuntimeError("No video data was loaded by AVVideoDataset.")
        rgb_pairs_all = torch.cat(batches, dim=0).contiguous()
    else:
        rgb_pairs_all = q55.preload_video_pair_cache_dali(files, args.video_dir, args.batch_size, device)
    q55.extract_and_compress_masks(rgb_pairs_all, segnet, device, args.crf, archive_dir, batch_size=args.batch_size)

    if not (archive_dir / MASK_PAYLOAD).exists():
        raise FileNotFoundError(archive_dir / MASK_PAYLOAD)
    make_archive_zip(archive_dir, archive_zip, [MODEL_PAYLOAD, MASK_PAYLOAD, POSE_PAYLOAD])

    materialize_submission(
        archive_zip=archive_zip,
        submission_dir=submission_dir,
        inflate_mode=args.inflate_mode,
    )
    env = {"FORCE_AV_DATASET": "1"} if args.force_av_dataset else None
    report_path = run_evaluate_submission(submission_dir, args.device, args.video_names, env=env)

    record = metric_record(
        label=label,
        archive_zip=submission_dir / "archive.zip",
        device=args.device,
        report_path=report_path,
        extra={
            "crf": args.crf,
            "inflate_mode": args.inflate_mode,
            "force_av_dataset": args.force_av_dataset,
            "base_archive": str(base_archive),
            "base_archive_summary": summarize_archive(base_archive),
        },
    )
    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "crf_swap_results.jsonl", record)
    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
