#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CANDIDATE="${1:-}"
if [[ -z "$CANDIDATE" ]]; then
  echo "Usage: $0 <S0|S1|S2> [archive_root]"
  exit 1
fi

ARCHIVE_ROOT="${2:-submissions/quantizr/experiments/gpt_seg_split}"
SOURCE_ROOT="${SOURCE_ROOT:-submissions/quantizr/experiments/gpt_pose_gate}"
DEVICE="${DEVICE:-cuda:0}"
OFFICIAL_EVAL_DEVICE="${OFFICIAL_EVAL_DEVICE:-cpu}"
SELECTION_METRIC="${SELECTION_METRIC:-proxy}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
EVAL_TAIL="${EVAL_TAIL:-5}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos}"
VIDEO_NAMES="${VIDEO_NAMES:-$ROOT_DIR/public_test_video_names.txt}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-$SOURCE_ROOT/_shared_cache}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SEG_RESCUE_EPOCHS="${SEG_RESCUE_EPOCHS:-60}"
SEG_RESCUE_LR="${SEG_RESCUE_LR:-5e-6}"
SEG_RESCUE_ERROR_BOOST="${SEG_RESCUE_ERROR_BOOST:-9.0}"
SEG_RESCUE_POSE_WEIGHT="${SEG_RESCUE_POSE_WEIGHT:-0.25}"
SPLIT_LATENT_EPOCHS="${SPLIT_LATENT_EPOCHS:-80}"
SPLIT_LATENT_LR="${SPLIT_LATENT_LR:-1e-5}"
SPLIT_LATENT_ERROR_BOOST="${SPLIT_LATENT_ERROR_BOOST:-9.0}"
SPLIT_LATENT_POSE_WEIGHT="${SPLIT_LATENT_POSE_WEIGHT:-0.5}"
CRF="${CRF:-50}"
MASK_ENCODE_SIZE="${MASK_ENCODE_SIZE:-512x384}"

mkdir -p "$ARCHIVE_ROOT"

CANDIDATE_A_ARCHIVE="$SOURCE_ROOT/crf50_c156_c264_h52_cond48_z0_repro/archive"
CANDIDATE_A_FP4="$CANDIDATE_A_ARCHIVE/run6_pose_rescue_best_fp4.pt"
if [[ ! -f "$CANDIDATE_A_FP4" ]]; then
  CANDIDATE_A_FP4="$CANDIDATE_A_ARCHIVE/candidate_model_fp4.pt"
fi

S0_ARCHIVE="$ARCHIVE_ROOT/crf50_c156_c264_h52_cond48_z0_seg_rescue/archive"
S0_FP4="$S0_ARCHIVE/run7_seg_rescue_best_fp4.pt"

COMMON_ARGS=(
  --video-dir "$VIDEO_DIR"
  --video-names "$VIDEO_NAMES"
  --device "$DEVICE"
  --official-eval-device "$OFFICIAL_EVAL_DEVICE"
  --shared-cache-root "$SHARED_CACHE_ROOT"
  --decode-backend av
  --selection-metric "$SELECTION_METRIC"
  --eval-interval "$EVAL_INTERVAL"
  --eval-tail "$EVAL_TAIL"
  --crf "$CRF"
  --mask-encode-size "$MASK_ENCODE_SIZE"
  --c1 56
  --c2 64
  --hidden 52
  --cond-dim 48
  --batch-size 1
  --grad-accum-steps 4
)

case "$CANDIDATE" in
  S0|s0)
    NAME="crf50_c156_c264_h52_cond48_z0_seg_rescue"
    INIT_FP4="$CANDIDATE_A_FP4"
    EXTRA_ARGS=(
      --pipeline-preset seg_rescue
      --seg-rescue-epochs "$SEG_RESCUE_EPOCHS"
      --seg-rescue-lr "$SEG_RESCUE_LR"
      --seg-rescue-error-boost "$SEG_RESCUE_ERROR_BOOST"
      --seg-rescue-pose-weight "$SEG_RESCUE_POSE_WEIGHT"
      --z-dim 0
    )
    ;;
  S1|s1)
    NAME="crf50_c156_c264_h52_cond48_zpose8i4_zseg2x4x6i4_split"
    INIT_FP4="${INIT_FP4:-$S0_FP4}"
    if [[ ! -f "$INIT_FP4" ]]; then
      INIT_FP4="$CANDIDATE_A_FP4"
    fi
    EXTRA_ARGS=(
      --pipeline-preset split_latent
      --split-latent-epochs "$SPLIT_LATENT_EPOCHS"
      --split-latent-lr "$SPLIT_LATENT_LR"
      --split-latent-error-boost "$SPLIT_LATENT_ERROR_BOOST"
      --split-latent-pose-weight "$SPLIT_LATENT_POSE_WEIGHT"
      --z-dim 8
      --z-seg-channels 2
      --z-seg-h 4
      --z-seg-w 6
      --latent-quant-bits 4
      --frame1-only-latent
    )
    ;;
  S2|s2)
    NAME="crf50_c156_c264_h52_cond48_zpose8i4_zseg1x6x8i4_split"
    INIT_FP4="${INIT_FP4:-$S0_FP4}"
    if [[ ! -f "$INIT_FP4" ]]; then
      INIT_FP4="$CANDIDATE_A_FP4"
    fi
    EXTRA_ARGS=(
      --pipeline-preset split_latent
      --split-latent-epochs "$SPLIT_LATENT_EPOCHS"
      --split-latent-lr "$SPLIT_LATENT_LR"
      --split-latent-error-boost "$SPLIT_LATENT_ERROR_BOOST"
      --split-latent-pose-weight "$SPLIT_LATENT_POSE_WEIGHT"
      --z-dim 8
      --z-seg-channels 1
      --z-seg-h 6
      --z-seg-w 8
      --latent-quant-bits 4
      --frame1-only-latent
    )
    ;;
  *)
    echo "Unknown candidate: $CANDIDATE"
    exit 1
    ;;
esac

if [[ ! -f "$INIT_FP4" ]]; then
  echo "Missing init checkpoint: $INIT_FP4" >&2
  exit 1
fi

ARCHIVE_DIR="$ARCHIVE_ROOT/$NAME/archive"
ARCHIVE_ZIP="$ARCHIVE_ROOT/$NAME/archive.zip"
mkdir -p "$ARCHIVE_DIR"

exec "$PYTHON_BIN" submissions/quantizr/compress.py \
  "${COMMON_ARGS[@]}" \
  --init-fp4 "$INIT_FP4" \
  --archive-dir "$ARCHIVE_DIR" \
  --output-archive-zip "$ARCHIVE_ZIP" \
  "${EXTRA_ARGS[@]}"
