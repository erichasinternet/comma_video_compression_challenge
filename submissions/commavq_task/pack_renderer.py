#!/usr/bin/env python3
"""Pack a commaVQ task-renderer checkpoint into Brotli payload form."""

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
    parser.add_argument("--tokens-archive", type=Path, default=None)
    parser.add_argument("--quality", type=int, default=11)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    buf = io.BytesIO()
    torch.save(payload, buf)
    compressed = brotli.compress(buf.getvalue(), quality=args.quality)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as zf:
        if args.tokens_archive:
            with zipfile.ZipFile(args.tokens_archive) as src:
                for name in src.namelist():
                    zf.writestr(name, src.read(name))
        zf.writestr("renderer.pt.br", compressed)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "tokens_archive": str(args.tokens_archive) if args.tokens_archive else None,
        "archive": str(args.out),
        "raw_renderer_bytes": len(buf.getvalue()),
        "brotli_renderer_bytes": len(compressed),
        "archive_bytes": args.out.stat().st_size,
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

