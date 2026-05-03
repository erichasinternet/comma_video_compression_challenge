#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CANDIDATE="${1:-NM1}"
ARCHIVE_ROOT="${2:-submissions/quantizr/experiments/gpt_no_mask}"
SOURCE_ROOT="${SOURCE_ROOT:-submissions/quantizr/experiments/gpt_pose_gate}"
DEVICE="${DEVICE:-cuda:0}"
OFFICIAL_EVAL_DEVICE="${OFFICIAL_EVAL_DEVICE:-cpu}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos}"
VIDEO_NAMES="${VIDEO_NAMES:-$ROOT_DIR/public_test_video_names.txt}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-$SOURCE_ROOT/_shared_cache}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

mkdir -p "$ARCHIVE_ROOT"

CANDIDATE_A_ARCHIVE="$SOURCE_ROOT/crf50_c156_c264_h52_cond48_z0_repro/archive"
CANDIDATE_A_FP4="$CANDIDATE_A_ARCHIVE/run6_pose_rescue_best_fp4.pt"
if [[ ! -f "$CANDIDATE_A_FP4" ]]; then
  CANDIDATE_A_FP4="$CANDIDATE_A_ARCHIVE/candidate_model_fp4.pt"
fi
if [[ ! -f "$CANDIDATE_A_FP4" ]]; then
  echo "Missing init checkpoint: $CANDIDATE_A_FP4" >&2
  exit 1
fi

COMMON_ARGS=(
  --video-dir "$VIDEO_DIR"
  --video-names "$VIDEO_NAMES"
  --device "$DEVICE"
  --official-eval-device "$OFFICIAL_EVAL_DEVICE"
  --shared-cache-root "$SHARED_CACHE_ROOT"
  --decode-backend av
  --selection-metric proxy
  --eval-interval 4
  --eval-tail 4
  --crf 50
  --c1 56
  --c2 64
  --hidden 52
  --cond-dim 48
  --batch-size 1
  --grad-accum-steps 4
  --init-fp4 "$CANDIDATE_A_FP4"
  --pipeline-preset split_latent
  --split-latent-epochs 64
  --split-latent-lr 5e-5
  --split-latent-pose-weight 0.35
  --latent-quant-bits 4
  --mask-input-mode zero
  --omit-mask-payload
  --frame1-only-latent
)

case "$CANDIDATE" in
  NM1|nm1)
    NAME="crf50_nomask_zpose8i4_zseg2x8x12i4"
    EXTRA_ARGS=(--z-dim 8 --z-seg-channels 2 --z-seg-h 8 --z-seg-w 12)
    ;;
  NM2|nm2)
    NAME="crf50_nomask_zpose8i4_zseg3x8x12i4"
    EXTRA_ARGS=(--z-dim 8 --z-seg-channels 3 --z-seg-h 8 --z-seg-w 12)
    ;;
  NM3|nm3)
    NAME="crf50_latentmask_zpose8i4_zseg5x8x12i4"
    EXTRA_ARGS=(
      --mask-input-mode decoded
      --mask-from-latent
      --init-zseg-from-mask
      --z-dim 8
      --z-seg-channels 5
      --z-seg-h 8
      --z-seg-w 12
    )
    ;;
  *)
    echo "Unknown candidate: $CANDIDATE" >&2
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
