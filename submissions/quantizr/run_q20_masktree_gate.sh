#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MASKTREE_DIR="${1:-}"
if [[ -z "$MASKTREE_DIR" ]]; then
  echo "Usage: $0 <masktree_dir> [archive_root]" >&2
  exit 1
fi

ARCHIVE_ROOT="${2:-submissions/quantizr/experiments/q20_masktree_metric_finetune_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-submissions/quantizr/experiments/gpt_pose_gate}"
BASE_ARCHIVE="${BASE_ARCHIVE:-$SOURCE_ROOT/crf50_c156_c264_h52_cond48_z0_repro/archive}"
BASE_FP4="${BASE_FP4:-$BASE_ARCHIVE/run6_pose_rescue_best_fp4.pt}"
DEVICE="${DEVICE:-cuda:0}"
OFFICIAL_EVAL_DEVICE="${OFFICIAL_EVAL_DEVICE:-cpu}"
SELECTION_METRIC="${SELECTION_METRIC:-proxy}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_TAIL="${EVAL_TAIL:-4}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos}"
VIDEO_NAMES="${VIDEO_NAMES:-$ROOT_DIR/public_test_video_names.txt}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-$SOURCE_ROOT/_shared_cache}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
QPACK_QUANTIZE_FP16="${QPACK_QUANTIZE_FP16:-int8}"

SEG_RESCUE_EPOCHS="${SEG_RESCUE_EPOCHS:-60}"
SEG_RESCUE_LR="${SEG_RESCUE_LR:-5e-6}"
SEG_RESCUE_ERROR_BOOST="${SEG_RESCUE_ERROR_BOOST:-25}"
SEG_RESCUE_POSE_WEIGHT="${SEG_RESCUE_POSE_WEIGHT:-0.20}"
POSE_RESCUE_EPOCHS="${POSE_RESCUE_EPOCHS:-60}"
POSE_RESCUE_LR="${POSE_RESCUE_LR:-5e-6}"
POSE_RESCUE_POSE_SCALE="${POSE_RESCUE_POSE_SCALE:-50.0}"
POSE_RESCUE_SEG2_SCALE="${POSE_RESCUE_SEG2_SCALE:-1.0}"

MIN_ARCHIVE_BYTES="${MIN_ARCHIVE_BYTES:-1}"
MAX_ORACLE_ARCHIVE_BYTES="${MAX_ORACLE_ARCHIVE_BYTES:-205000}"
MAX_ORACLE_SCORE="${MAX_ORACLE_SCORE:-0.300}"

mkdir -p "$ARCHIVE_ROOT"

if [[ ! -f "$BASE_FP4" ]]; then
  BASE_FP4="$BASE_ARCHIVE/candidate_model_fp4.pt"
fi
if [[ ! -f "$BASE_FP4" ]]; then
  echo "Missing Candidate A FP4 checkpoint: $BASE_FP4" >&2
  exit 1
fi

QPACK_SUFFIX="${QPACK_QUANTIZE_FP16}"
QPACK_PATH="$BASE_ARCHIVE/model_${QPACK_SUFFIX}.qpack.br"
if [[ ! -f "$QPACK_PATH" ]]; then
  "$PYTHON_BIN" submissions/quantizr/pack_model.py --archive-dir "$BASE_ARCHIVE" --out "$QPACK_PATH" --quantize-fp16 "$QPACK_QUANTIZE_FP16"
fi

ORACLE_ROOT="$ARCHIVE_ROOT/oracle"
"$PYTHON_BIN" submissions/quantizr/eval_mask_codec_oracle.py \
  --base-archive-dir "$BASE_ARCHIVE" \
  --out-root "$ORACLE_ROOT" \
  --video-names "$VIDEO_NAMES" \
  --eval-device "$OFFICIAL_EVAL_DEVICE" \
  --run-official \
  --enforce-m0 \
  --candidate M0:copy \
  --candidate "Q20:masktree_qpack,masktree=$MASKTREE_DIR,qpack=$QPACK_PATH"

"$PYTHON_BIN" - "$ORACLE_ROOT/mask_codec_oracle_results.jsonl" "$MAX_ORACLE_ARCHIVE_BYTES" "$MAX_ORACLE_SCORE" "$MIN_ARCHIVE_BYTES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_bytes = int(sys.argv[2])
max_score = float(sys.argv[3])
min_bytes = int(sys.argv[4])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
q20 = [row for row in rows if row["name"] == "Q20"][-1]
score = float(q20["score"])
archive_bytes = int(q20["archive_bytes"])
if archive_bytes < min_bytes:
    raise SystemExit(f"oracle archive bytes suspiciously low: {archive_bytes}")
if archive_bytes > max_bytes or score >= max_score:
    raise SystemExit(f"oracle gate failed: score={score:.5f} bytes={archive_bytes}")
print(f"oracle gate passed: score={score:.5f} bytes={archive_bytes}")
PY

NAME="q20_masktree_metric_finetune_v1"
ARCHIVE_DIR="$ARCHIVE_ROOT/$NAME/archive"
ARCHIVE_ZIP="$ARCHIVE_ROOT/$NAME/archive.zip"
mkdir -p "$ARCHIVE_DIR"

exec "$PYTHON_BIN" submissions/quantizr/compress.py \
  --video-dir "$VIDEO_DIR" \
  --video-names "$VIDEO_NAMES" \
  --device "$DEVICE" \
  --official-eval-device "$OFFICIAL_EVAL_DEVICE" \
  --shared-cache-root "$SHARED_CACHE_ROOT" \
  --decode-backend av \
  --selection-metric "$SELECTION_METRIC" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-tail "$EVAL_TAIL" \
  --crf 50 \
  --mask-source masktree \
  --mask-tree-dir "$MASKTREE_DIR" \
  --mask-payload-kind masktree \
  --c1 56 \
  --c2 64 \
  --hidden 52 \
  --cond-dim 48 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --init-fp4 "$BASE_FP4" \
  --archive-dir "$ARCHIVE_DIR" \
  --output-archive-zip "$ARCHIVE_ZIP" \
  --pipeline-preset seg_rescue \
  --include-pose-rescue \
  --seg-rescue-epochs "$SEG_RESCUE_EPOCHS" \
  --seg-rescue-lr "$SEG_RESCUE_LR" \
  --seg-rescue-error-boost "$SEG_RESCUE_ERROR_BOOST" \
  --seg-rescue-pose-weight "$SEG_RESCUE_POSE_WEIGHT" \
  --pose-rescue-epochs "$POSE_RESCUE_EPOCHS" \
  --pose-rescue-lr "$POSE_RESCUE_LR" \
  --pose-rescue-pose-scale "$POSE_RESCUE_POSE_SCALE" \
  --pose-rescue-seg2-scale "$POSE_RESCUE_SEG2_SCALE" \
  --z-dim 0
