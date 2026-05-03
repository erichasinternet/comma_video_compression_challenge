#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

ARCHIVE_DIR="${1:-archive}"
OUT_DIR="${2:-inflated}"
FILE_LIST="${3:-$ROOT/public_test_video_names.txt}"

mkdir -p "$OUT_DIR"

"${PYTHON:-python3}" "$HERE/inflate.py" \
  --archive-dir "$ARCHIVE_DIR" \
  --out-dir "$OUT_DIR" \
  --file-list "$FILE_LIST"

