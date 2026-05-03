#!/usr/bin/env python
"""Evaluate the exact Quantizr #55 artifact under a pinned evaluator path."""

from __future__ import annotations

import argparse
from pathlib import Path

from q55_common import (
    DEFAULT_VIDEO_NAMES,
    Q55_REF,
    append_jsonl,
    materialize_submission,
    metric_record,
    run_evaluate_submission,
    summarize_archive,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--inflate-mode", choices=["upstream", "modified"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--upstream-ref", default=Q55_REF)
    parser.add_argument("--label", default=None)
    parser.add_argument("--force-av-dataset", action="store_true")
    args = parser.parse_args()

    archive_zip = args.archive_zip.resolve()
    if not archive_zip.exists():
        raise FileNotFoundError(archive_zip)

    label = args.label or f"q55_{args.inflate_mode}_{args.device}"
    run_dir = args.out_dir / label
    submission_dir = run_dir / "submission"
    materialize_submission(
        archive_zip=archive_zip,
        submission_dir=submission_dir,
        inflate_mode=args.inflate_mode,
        upstream_ref=args.upstream_ref,
    )
    env = {"FORCE_AV_DATASET": "1"} if args.force_av_dataset else None
    report_path = run_evaluate_submission(submission_dir, args.device, args.video_names, env=env)

    record = metric_record(
        label=label,
        archive_zip=submission_dir / "archive.zip",
        device=args.device,
        report_path=report_path,
        extra={
            "inflate_mode": args.inflate_mode,
            "force_av_dataset": args.force_av_dataset,
            "upstream_ref": args.upstream_ref,
            "source_archive": str(archive_zip),
            "source_archive_summary": summarize_archive(archive_zip),
        },
    )
    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "calibration_results.jsonl", record)
    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
