#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CANDIDATE="${1:-}"
if [[ -z "$CANDIDATE" ]]; then
  echo "Usage: $0 <A|B|D> [archive_root]"
  exit 1
fi

ARCHIVE_ROOT="${2:-submissions/quantizr/experiments/gpt_pose_gate}"
DEVICE="${DEVICE:-cuda:0}"
OFFICIAL_EVAL_DEVICE="${OFFICIAL_EVAL_DEVICE:-cpu}"
SELECTION_METRIC="${SELECTION_METRIC:-proxy}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
EVAL_TAIL="${EVAL_TAIL:-10}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos}"
VIDEO_NAMES="${VIDEO_NAMES:-$ROOT_DIR/public_test_video_names.txt}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-$ARCHIVE_ROOT/_shared_cache}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
POSE_RESCUE_EPOCHS="${POSE_RESCUE_EPOCHS:-120}"
POSE_RESCUE_LR="${POSE_RESCUE_LR:-5e-6}"
POSE_RESCUE_POSE_SCALE="${POSE_RESCUE_POSE_SCALE:-50.0}"
POSE_RESCUE_SEG2_SCALE="${POSE_RESCUE_SEG2_SCALE:-1.0}"

mkdir -p "$ARCHIVE_ROOT"

COMMON_ARGS=(
  --video-dir "$VIDEO_DIR"
  --video-names "$VIDEO_NAMES"
  --device "$DEVICE"
  --official-eval-device "$OFFICIAL_EVAL_DEVICE"
  --shared-cache-root "$SHARED_CACHE_ROOT"
  --decode-backend av
  --pipeline-preset full
  --selection-metric "$SELECTION_METRIC"
  --eval-interval "$EVAL_INTERVAL"
  --eval-tail "$EVAL_TAIL"
  --include-pose-rescue
  --pose-rescue-epochs "$POSE_RESCUE_EPOCHS"
  --pose-rescue-lr "$POSE_RESCUE_LR"
  --pose-rescue-qat-start-epoch 0
  --pose-rescue-pose-scale "$POSE_RESCUE_POSE_SCALE"
  --pose-rescue-seg2-scale "$POSE_RESCUE_SEG2_SCALE"
)

case "$CANDIDATE" in
  A|a)
    NAME="crf50_c156_c264_h52_cond48_z0_repro"
    EXTRA_ARGS=(
      --crf 50
      --c1 56
      --c2 64
      --hidden 52
      --cond-dim 48
      --z-dim 0
      --batch-size 1
      --grad-accum-steps 4
    )
    ;;
  B|b)
    NAME="crf50_c156_c264_h52_cond48_z32_pose_rescue"
    EXTRA_ARGS=(
      --crf 50
      --c1 56
      --c2 64
      --hidden 52
      --cond-dim 48
      --z-dim 32
      --frame1-only-latent
      --batch-size 1
      --grad-accum-steps 4
    )
    ;;
  D|d)
    NAME="crf50_c152_c264_h52_cond48_z32_pose_rescue"
    EXTRA_ARGS=(
      --crf 50
      --c1 52
      --c2 64
      --hidden 52
      --cond-dim 48
      --z-dim 32
      --frame1-only-latent
      --batch-size 1
      --grad-accum-steps 4
    )
    ;;
  *)
    echo "Unknown candidate: $CANDIDATE"
    exit 1
    ;;
esac

ARCHIVE_DIR="$ARCHIVE_ROOT/$NAME/archive"
ARCHIVE_ZIP="$ARCHIVE_ROOT/$NAME/archive.zip"
mkdir -p "$ARCHIVE_DIR"

exec "$PYTHON_BIN" submissions/quantizr/compress.py \
  "${COMMON_ARGS[@]}" \
  --archive-dir "$ARCHIVE_DIR" \
  --output-archive-zip "$ARCHIVE_ZIP" \
  "${EXTRA_ARGS[@]}"
