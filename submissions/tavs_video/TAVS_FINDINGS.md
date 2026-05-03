# TAVS Findings

## Status

Task-Aware Video Source Optimization has been scaffolded as a new submission family under `submissions/tavs_video`.

The current implementation optimizes a low-frequency YUV residual source at evaluator resolution (`512x384`), evaluates SegNet/PoseNet directly, and can periodically run real FFmpeg codec round trips. The final archive format is a normal compressed video decoded by `inflate.sh` into official `*.raw` frames.

## 64-Sample Gate Results

### SVT-AV1 Run

```text
experiment: tavs_64_q55_yuvgrid96_s5000
init:       q55_fp16_pose_int10 decoded frames
steps:      5000
codec:      svtav1
```

This run is not a valid TAVS decision signal because the Modal FFmpeg build did not expose the SVT-AV1 encoder.

```text
best_proxy:
  step:    0
  quality: 0.173374

best_codec:
  null

codec_history:
  every checkpoint skipped with "ffmpeg encoder unavailable"
```

The proxy optimization also did not improve over step 0, so the source optimizer showed no useful uncompressed movement.

### VP9 Real-Codec Run

```text
experiment: tavs_64_q55_vp9_yuvgrid96_s5000
init:       q55_fp16_pose_int10 decoded frames
steps:      5000
codec:      vp9
CRFs:       35,40,45,50,55,60
```

Baseline proxy quality for this 64-sample subset:

```text
quality: 0.1734248054
SegNet:  0.0006532669
PoseNet: 0.0011685202
```

Best uncompressed proxy result:

```text
step:    0
quality: 0.1733744792
```

Best real-codec score:

```text
step:                 0
codec:                vp9
crf:                  55
encoded_64_bytes:     208,613
projected_full_bytes: 1,955,747
quality:              3.2250226345
score:                4.5272742845
```

Aggregate codec-history checks:

```text
real codec rows:       72
min projected bytes:   891,431
min quality:           0.7239406409
rows <=260 KB:         0
```

Best-quality row:

```text
step:                 0
codec:                vp9
crf:                  35
projected_full_bytes: 11,059,716
quality:              0.7239406409
score:                8.0881515585
```

Smallest-byte row:

```text
step:                 0
codec:                vp9
crf:                  60
projected_full_bytes: 891,431
quality:              6.0447481786
score:                6.6383154911
```

## Decision

TAVS fails the 64-sample decision table by a wide margin.

Kill condition:

```text
after 2k steps:
  no real-codec candidate has decoded quality <=0.30 at projected <=260 KB
```

Observed:

```text
no real-codec candidate reached projected <=260 KB at all
best quality was 0.724 at ~11.1 MB projected
smallest byte candidate was ~891 KB projected with quality 6.04
training degraded PoseNet heavily after step 0
```

This is not a small miss, so the planned blend-init follow-up is not justified. TAVS should be stopped as a visible `0.2x` path.

## Implementation Notes

The Modal runner now accepts an explicit codec:

```bash
./.venv/bin/python -m modal run submissions/tavs_video/modal_tavs.py \
  --stage optimize64 \
  --init q55 \
  --codec vp9 \
  --steps 5000 \
  --batch-size 4 \
  --eval-every 100 \
  --codec-eval-every 250 \
  --codec-crfs 35,40,45,50,55,60
```

The original SVT-AV1 command remains useful only on environments where FFmpeg exposes `libsvtav1`.

## First Gate Command

Run the 64-sample oracle before any full-set work:

```bash
./.venv/bin/python -m modal run submissions/tavs_video/modal_tavs.py \
  --stage optimize64 \
  --init q55 \
  --codec svtav1 \
  --steps 5000 \
  --batch-size 4 \
  --eval-every 100 \
  --codec-eval-every 250 \
  --codec-crfs 49,53,57,61
```

## Decision Rules

Continue only if real codec round-trip metrics satisfy one of:

- Projected bytes `<=240 KB` and decoded quality `<=0.180`.
- Projected bytes `<=220 KB` and decoded quality `<=0.153`.
- Projected bytes `<=200 KB` and decoded quality `<=0.167`.

Kill if no real-codec candidate reaches quality `<=0.30` at projected `<=260 KB` after about 2k steps, or `<=0.20` at projected `<=240 KB` after about 5k steps.

## Local Smoke

Implemented smoke run:

```text
experiment: submissions/tavs_video/experiments/smoke_original_1
init:       original
subset:     1
steps:      1
codec:      SVT-AV1 CRF 61 pack/inflate smoke
```

The optimizer/evaluator path produced `metrics.json`, `best_frames.pt`, and a tiny `archive.zip`. The archive inflated to the official raw layout (`inflated/0.raw`).

This smoke is not a quality signal. It only verifies TAVS plumbing. The meaningful 64-sample GPU oracle is the VP9 run summarized above, and it fails the stop gate.
