#!/usr/bin/env python3
"""Pack a task-NeRV checkpoint into a Brotli-compressed prototype payload."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import brotli
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submissions.tavs_video.common import metric_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quality", type=float, default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = io.BytesIO()
    torch.save(payload, raw)
    raw_bytes = raw.getvalue()
    br = brotli.compress(raw_bytes, quality=11)
    payload_path = args.out_dir / "nerv_model.pt.br"
    payload_path.write_bytes(br)
    archive_path = args.out_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(payload_path, "nerv_model.pt.br")
    archive_bytes = archive_path.stat().st_size
    metrics = {
        "checkpoint": str(args.checkpoint),
        "payload_bytes_raw": len(raw_bytes),
        "payload_bytes_br": len(br),
        "archive_bytes": archive_bytes,
        "metric_table_if_quality_zero": metric_table(0.0, 0.0, archive_bytes),
    }
    if args.quality is not None:
        metrics["quality"] = args.quality
        metrics["rate_term"] = metric_table(0.0, 0.0, archive_bytes)["rate_term"]
        metrics["projected_score"] = args.quality + metrics["rate_term"]
    (args.out_dir / "archive.metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

