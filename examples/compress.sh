#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/.." && pwd)"

CRF=""
SCALE=""
IN_DIR=""
JOBS=""
VIDEO_NAMES_FILE=""
ENCODER=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --crf)
      CRF="$2"; shift 2 ;;
    --scale)
      SCALE="$2"; shift 2 ;;
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --encoder)
      ENCODER="$2"; shift 2 ;;
    --out-dir|--out_dir)
      OUT_DIR="${2%/}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 --crf <crf> --scale <scale> --in-dir <in_dir> --out-dir <out_dir> --jobs <jobs> --video-names-file <video-names-file> [--encoder cpu|gpu]" >&2
      exit 2 ;;
  esac
done

# Auto-detect encoder if not specified
if [[ -z "$ENCODER" ]]; then
  if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q hevc_nvenc; then
    ENCODER="gpu"
  else
    ENCODER="cpu"
  fi
  echo "Auto-detected encoder: $ENCODER"
fi


TMPDIR="$(mktemp -d)"
OUTDIR="${OUT_DIR:-$PD/submission}"

export CRF IN_DIR TMPDIR SCALE VIDEO_NAMES_FILE JOBS ENCODER

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  OUT="${TMPDIR}/${rel}"
  mkdir -p "$(dirname "$OUT")"

  echo "→ ${IN}  CRF=${CRF}  scale=${SCALE}  →  ${OUT}"

  if [[ "$SCALE" =~ ^1(\.0+)?$ ]]; then
    SCALE_VF=""
  else
    SCALE_VF="-vf scale=trunc(iw*${SCALE}/2)*2:trunc(ih*${SCALE}/2)*2:flags=lanczos"
  fi

  if [[ "$ENCODER" == "gpu" ]]; then
    ffmpeg -nostdin -y -hide_banner -loglevel warning \
      -r 20 -fflags +genpts -i "$IN" \
      ${SCALE_VF} \
      -c:v hevc_nvenc -preset p1 -cq "$CRF" \
      -g 1 -bf 0 \
      -r 20 -f hevc "$OUT"
  else
    ffmpeg -nostdin -y -hide_banner -loglevel warning \
      -r 20 -fflags +genpts -i "$IN" \
      ${SCALE_VF} \
      -c:v libx265 -preset ultrafast -crf "$CRF" \
      -g 1 -bf 0 -x265-params "keyint=1:min-keyint=1:scenecut=0:frame-threads=1:log-level=warning" \
      -r 20 -f hevc "$OUT"
  fi
' _ {}

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
(
  cd "$TMPDIR"
  cp -r . "$OUTDIR"
)
rm -rf "$TMPDIR"
echo "All done. Saved $OUTDIR"
