#!/usr/bin/env python
"""Pack #55 pose payload into pose.qpack.br and evaluate net official score."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from q55_common import (
    DEFAULT_VIDEO_NAMES,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--variant", choices=["fp16", "int16_per_dim", "int12_per_dim", "int10_per_dim"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--label", default=None)
    parser.add_argument("--force-av-dataset", action="store_true")
    args = parser.parse_args()

    base_archive = args.base_archive.resolve()
    if not base_archive.exists():
        raise FileNotFoundError(base_archive)

    label = args.label or f"q55_pose_{args.variant}_{args.device}"
    run_dir = args.out_dir / label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    submission_dir = run_dir / "submission"
    unzip_archive(base_archive, archive_dir)

    pack_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "pack_pose.py"),
        "--archive-dir",
        str(archive_dir),
        "--variant",
        args.variant,
    ]
    proc = subprocess.run(pack_cmd, check=True, capture_output=True, text=True)
    pack_report = json.loads(proc.stdout)

    pose_path = archive_dir / POSE_PAYLOAD
    if pose_path.exists():
        pose_path.unlink()
    if not (archive_dir / POSE_QPACK_PAYLOAD).exists():
        raise FileNotFoundError(archive_dir / POSE_QPACK_PAYLOAD)
    make_archive_zip(archive_dir, archive_zip)

    materialize_submission(archive_zip=archive_zip, submission_dir=submission_dir, inflate_mode="modified")
    env = {"FORCE_AV_DATASET": "1"} if args.force_av_dataset else None
    report_path = run_evaluate_submission(submission_dir, args.device, args.video_names, env=env)

    record = metric_record(
        label=label,
        archive_zip=submission_dir / "archive.zip",
        device=args.device,
        report_path=report_path,
        extra={
            "variant": args.variant,
            "force_av_dataset": args.force_av_dataset,
            "pack_report": pack_report,
            "base_archive": str(base_archive),
            "base_archive_summary": summarize_archive(base_archive),
        },
    )
    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "pose_qpack_results.jsonl", record)
    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
