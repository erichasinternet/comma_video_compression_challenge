#!/usr/bin/env python
"""Re-encode the decoded #55 mask payload at a single CRF and evaluate it."""

from __future__ import annotations

import argparse
import logging
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
from q55_mask_alloc import decode_archive_masks, encode_mask_group, parse_palette


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--crf", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--palette", default="legacy")
    parser.add_argument("--inflate-mode", choices=["upstream", "modified"], default="modified")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    base_archive = args.base_archive.resolve()
    if not base_archive.exists():
        raise FileNotFoundError(base_archive)

    import compress as q55

    palette = parse_palette(args.palette)
    label = args.label or f"qrecode_crf{args.crf}_{args.device}_{args.palette}"
    run_dir = args.out_dir / label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    submission_dir = run_dir / "submission"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "mask_recode.log")],
    )

    unzip_archive(base_archive, archive_dir)
    ensure_legacy_payloads(archive_dir)
    logging.info("Decoding original #55 mask payload...")
    masks = decode_archive_masks(archive_dir / MASK_PAYLOAD)

    for path in archive_dir.glob("mask*.obu*"):
        path.unlink()

    logging.info(f"Re-encoding decoded #55 masks at CRF {args.crf}...")
    payload = encode_mask_group(
        masks=masks,
        frame_indices=list(range(masks.shape[0])),
        palette=palette,
        crf=args.crf,
        name="mask",
        run_dir=archive_dir,
        q55=q55,
    )
    if payload != MASK_PAYLOAD:
        raise RuntimeError(f"expected {MASK_PAYLOAD}, got {payload}")

    make_archive_zip(archive_dir, archive_zip, [MODEL_PAYLOAD, MASK_PAYLOAD, POSE_PAYLOAD])
    materialize_submission(archive_zip=archive_zip, submission_dir=submission_dir, inflate_mode=args.inflate_mode)
    report_path = run_evaluate_submission(submission_dir, args.device, args.video_names)

    record = metric_record(
        label=label,
        archive_zip=submission_dir / "archive.zip",
        device=args.device,
        report_path=report_path,
        extra={
            "base_archive": str(base_archive),
            "base_archive_summary": summarize_archive(base_archive),
            "crf": args.crf,
            "palette": palette,
            "mask_source": "archive",
        },
    )
    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "mask_recode_results.jsonl", record)
    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
