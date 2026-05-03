#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="${1:-archive}"
OUT_DIR="${2:-inflated}"

mkdir -p "$OUT_DIR"
python submissions/commavq_task/inflate.py "$ARCHIVE_DIR" 0.mkv "$OUT_DIR/0.raw"

