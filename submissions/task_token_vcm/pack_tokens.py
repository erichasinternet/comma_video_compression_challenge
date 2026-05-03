#!/usr/bin/env python3
"""Pack a task-token capacity checkpoint into a prototype archive.

This is intentionally simple: it stores the float-token checkpoint as a Brotli
payload so inflate.py can render it. Real VQ/ANS token packing belongs after
the float/VQ capacity gates pass.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import brotli
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", type=int, default=11)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    buf = io.BytesIO()
    torch.save(payload, buf)
    compressed = brotli.compress(buf.getvalue(), quality=args.quality)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("task_token.pt.br", compressed)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "archive": str(args.out),
        "raw_payload_bytes": len(buf.getvalue()),
        "brotli_payload_bytes": len(compressed),
        "archive_bytes": args.out.stat().st_size,
        "note": "float-token prototype archive; not a final compressed VQ bitstream",
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
