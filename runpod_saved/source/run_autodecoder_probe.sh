#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"

RUN_NAME="${RUN_NAME:-autodecoder_D0_64_001}"
PRESET="${PRESET:-D0}"
OUT_DIR="${OUT_DIR:-${HERE}/experiments/${RUN_NAME}}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-64}"
STEPS="${STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
POSE_START_STEP="${POSE_START_STEP:-600}"
WIDTH="${WIDTH:-32}"
BLOCKS="${BLOCKS:-3}"
LR_MODEL="${LR_MODEL:-8e-4}"
LR_LATENT="${LR_LATENT:-3e-2}"
GATE="${GATE:-1000:2.0,3000:0.8,6000:0.35}"

mkdir -p "${OUT_DIR}"

exec "${PYTHON_BIN}" "${HERE}/evaluator_renderer.py" \
  --preset "${PRESET}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --samples "${SAMPLES}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --pose-start-step "${POSE_START_STEP}" \
  --width "${WIDTH}" \
  --blocks "${BLOCKS}" \
  --lr-model "${LR_MODEL}" \
  --lr-latent "${LR_LATENT}" \
  --gate "${GATE}"
