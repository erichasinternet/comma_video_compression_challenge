#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT/.venv/bin/python}"
DEVICE="${SCV_DEVICE:-cpu}"
SUBSET="${SCV_SUBSET:-64}"

cd "$ROOT"

"$PYTHON_BIN" submissions/scv_cartoon/scv_cartoon.py render \
  --subset "$SUBSET" \
  --variant scv0 \
  --device "$DEVICE" \
  --out submissions/scv_cartoon/experiments/scv0_64

"$PYTHON_BIN" submissions/scv_cartoon/scv_cartoon.py encode \
  --frames submissions/scv_cartoon/experiments/scv0_64/frames \
  --out submissions/scv_cartoon/experiments/scv0_64/archive.zip \
  --subset "$SUBSET" \
  --quick \
  --evaluate \
  --device "$DEVICE"

"$PYTHON_BIN" submissions/scv_cartoon/scv_cartoon.py render \
  --subset "$SUBSET" \
  --variant scv1 \
  --device "$DEVICE" \
  --out submissions/scv_cartoon/experiments/scv1_64

"$PYTHON_BIN" submissions/scv_cartoon/scv_cartoon.py encode \
  --frames submissions/scv_cartoon/experiments/scv1_64/frames \
  --out submissions/scv_cartoon/experiments/scv1_64/archive.zip \
  --subset "$SUBSET" \
  --quick \
  --evaluate \
  --device "$DEVICE"

"$PYTHON_BIN" submissions/scv_cartoon/scv_cartoon.py render \
  --subset "$SUBSET" \
  --variant scvt \
  --device "$DEVICE" \
  --out submissions/scv_cartoon/experiments/scvt_64

"$PYTHON_BIN" submissions/scv_cartoon/scv_cartoon.py encode \
  --frames submissions/scv_cartoon/experiments/scvt_64/frames \
  --out submissions/scv_cartoon/experiments/scvt_64/archive.zip \
  --subset "$SUBSET" \
  --quick \
  --evaluate \
  --device "$DEVICE"
