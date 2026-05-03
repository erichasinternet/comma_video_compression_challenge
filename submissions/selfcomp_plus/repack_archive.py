#!/usr/bin/env python3
"""Repack a selfcomp PR #56 archive using `segmap.dcpack.br`."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

from pack_segmap import pack_checkpoint


def repack_archive(base_archive: Path, out_archive: Path, work_dir: Path | None = None) -> dict:
    owned_tmp = None
    if work_dir is None:
        owned_tmp = tempfile.TemporaryDirectory(prefix="selfcomp_repack_")
        root = Path(owned_tmp.name)
    else:
        root = work_dir
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

    try:
        archive_dir = root / "archive"
        payload_dir = root / "payload"
        archive_dir.mkdir(parents=True)
        payload_dir.mkdir(parents=True)
        with zipfile.ZipFile(base_archive) as zf:
            zf.extractall(archive_dir)
        with tarfile.open(archive_dir / "payload.tar.xz", mode="r:xz") as tf:
            tf.extractall(payload_dir)

        pack_metrics = pack_checkpoint(
            payload_dir / "segmap_inference.pt",
            payload_dir / "segmap.dcpack.br",
            raw_out=payload_dir / "segmap.dcpack",
        )
        (payload_dir / "segmap_inference.pt").unlink()
        (payload_dir / "segmap.dcpack").unlink()

        payload_tar = root / "payload.tar.xz"
        with tarfile.open(payload_tar, mode="w:xz") as tf:
            tf.add(payload_dir / "0.mkv", arcname="0.mkv")
            tf.add(payload_dir / "segmap.dcpack.br", arcname="segmap.dcpack.br")

        out_archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(payload_tar, arcname="payload.tar.xz")

        metrics = {
            "base_archive": str(base_archive),
            "out_archive": str(out_archive),
            "base_archive_bytes": base_archive.stat().st_size,
            "out_archive_bytes": out_archive.stat().st_size,
            "bytes_saved": base_archive.stat().st_size - out_archive.stat().st_size,
            "payload_tar_bytes": payload_tar.stat().st_size,
            "model_pack": pack_metrics,
        }
        return metrics
    finally:
        if owned_tmp is not None:
            owned_tmp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()

    metrics = repack_archive(args.base_archive, args.out, work_dir=args.work_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
