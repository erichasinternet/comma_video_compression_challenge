#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

ARCHIVE_ZIP="${1:?usage: run_q55_restart_gate.sh /path/to/q55/archive.zip [out_dir] [device]}"
OUT_DIR="${2:-${ROOT}/submissions/quantizr/experiments/q55_restart}"
DEVICE="${3:-cuda}"

mkdir -p "${OUT_DIR}"

if [[ "${RUN_Q0:-1}" == "1" ]]; then
  python "${HERE}/q55_calibrate.py" \
    --archive-zip "${ARCHIVE_ZIP}" \
    --inflate-mode upstream \
    --device "${DEVICE}" \
    --out-dir "${OUT_DIR}" \
    --label "q0_upstream_${DEVICE}"

  python "${HERE}/q55_calibrate.py" \
    --archive-zip "${ARCHIVE_ZIP}" \
    --inflate-mode modified \
    --device "${DEVICE}" \
    --out-dir "${OUT_DIR}" \
    --label "q0_modified_${DEVICE}"

  python "${HERE}/q55_gate.py" q0 \
    --upstream "${OUT_DIR}/q0_upstream_${DEVICE}/metrics.json" \
    --modified "${OUT_DIR}/q0_modified_${DEVICE}/metrics.json"
fi

if [[ "${RUN_Q1:-1}" == "1" ]]; then
  for variant in fp16 mixed_int8 mixed_int8_heads_fp16; do
    python "${HERE}/q55_qpack_eval.py" \
      --base-archive "${ARCHIVE_ZIP}" \
      --variant "${variant}" \
      --device "${DEVICE}" \
      --out-dir "${OUT_DIR}" \
      --label "q1_${variant}_${DEVICE}"
  done
fi

if [[ "${RUN_QCRF:-0}" == "1" ]]; then
  for crf in 50 52 54 56 58 60; do
    python "${HERE}/q55_crf_swap.py" \
      --base-archive "${ARCHIVE_ZIP}" \
      --crf "${crf}" \
      --device "${DEVICE}" \
      --out-dir "${OUT_DIR}" \
      --label "qcrf${crf}_${DEVICE}"
  done
  python "${HERE}/q55_gate.py" crf \
    --base "${OUT_DIR}/q0_modified_${DEVICE}/metrics.json" \
    --crf50 "${OUT_DIR}/qcrf50_${DEVICE}/metrics.json" \
    --crf52 "${OUT_DIR}/qcrf52_${DEVICE}/metrics.json" \
    --crf54 "${OUT_DIR}/qcrf54_${DEVICE}/metrics.json"
fi

if [[ "${RUN_QMASK:-0}" == "1" ]]; then
  for spec in "50:0.20,54:0.35,58:0.45" "52:0.25,56:0.35,60:0.40"; do
    safe_spec="${spec//[,:.]/_}"
    python "${HERE}/q55_mask_alloc.py" \
      --base-archive "${ARCHIVE_ZIP}" \
      --device "${DEVICE}" \
      --eval-device "${DEVICE}" \
      --decode-backend "${QMASK_DECODE_BACKEND:-av}" \
      --mask-source "${QMASK_MASK_SOURCE:-archive}" \
      --group-spec "${spec}" \
      --order "${QMASK_ORDER:-hist}" \
      --palette "${QMASK_PALETTE:-legacy}" \
      --out-dir "${OUT_DIR}" \
      --label "qmask_${safe_spec}_${DEVICE}"
  done
fi

echo "Gate outputs written under ${OUT_DIR}"
