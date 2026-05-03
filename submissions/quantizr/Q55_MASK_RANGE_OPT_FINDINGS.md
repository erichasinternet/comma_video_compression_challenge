# Q55 Mask Range Optimization Findings

## Purpose

This branch tested class-preserving AV1 source optimization for the exact
Quantizr #55 mask tensor. The inflater consumes only:

```python
round(decoded_gray / 63.0).clip(0, 4)
```

so the grayscale source can vary inside each class's legal rounding interval
without changing evaluator-facing class IDs.

Targets:

```text
<=196 KB exact/repaired mask: continue
<=182 KB exact/repaired mask: strong PR-like visible-0.2x path
>205 KB exact/repaired mask: stop
```

Local note:

```text
The local FFmpeg exposes libsvtav1 but not libaom-av1, so the completed runs
used SVT-AV1. The script supports libaom automatically when available.
```

## Implementation

Script:

```text
submissions/quantizr/q55_mask_range_opt.py
```

Implemented modes:

```text
extract:
  decode current mask.obu.br to exact rounded class tensor

palette_sweep:
  generate legal global class palettes, encode/decode, compare class tensor,
  estimate sparse residual bytes

diffuse_sweep:
  projected spatial/temporal smoothing inside per-class legal intervals,
  encode/decode, compare class tensor, estimate sparse residual bytes
```

Class intervals:

```text
class 0:   0..31
class 1:  32..94
class 2:  95..157
class 3: 158..220
class 4: 221..255
```

## Extract

Run:

```bash
./.venv/bin/python submissions/quantizr/q55_mask_range_opt.py extract \
  --archive submissions/q55_fp16_pose_int10/archive.zip \
  --out submissions/quantizr/experiments/q55_range_mask/classes.npy
```

Result:

```text
mask payload: 219,472 B
shape:        600 x 384 x 512
class counts:
  0: 27,408,427
  1:    690,063
  2: 58,413,695
  3:  1,459,867
  4: 29,992,748
```

## Phase 1: Global Palette Sweep

Run:

```bash
./.venv/bin/python submissions/quantizr/q55_mask_range_opt.py palette_sweep \
  --classes submissions/quantizr/experiments/q55_range_mask/classes.npy \
  --out-dir submissions/quantizr/experiments/q55_range_mask/palette_grid_svt \
  --margins 0,4,8,12 \
  --palette-kinds standard,center,adjopt \
  --crfs 40,48,56,60,63 \
  --codecs svtav1
```

Best result:

```text
label:              palette_m0_standard_svtav1_crf48
palette:            [0, 63, 126, 189, 252]
base video:         232,565 B
changed pixels:     166,385
residual estimate:  195,080 B
exact/repaired:     427,645 B
```

Interpretation:

```text
Global range palettes reduce the base stream in some cases, but class flips are
far too numerous for sparse residual repair. The best exact/repaired result is
~208 KB worse than the current AV1 mask.
```

## Phase 2: Projected Diffusion Sweep

Run:

```bash
./.venv/bin/python submissions/quantizr/q55_mask_range_opt.py diffuse_sweep \
  --classes submissions/quantizr/experiments/q55_range_mask/classes.npy \
  --out-dir submissions/quantizr/experiments/q55_range_mask/diffuse_svt_small \
  --margins 4,8,12 \
  --iters 10,30 \
  --crfs 40,48,56,60 \
  --codecs svtav1 \
  --palette-kind adjopt
```

Best result:

```text
label:              diffuse_m8_i10_adjopt_svtav1_crf60
palette:            [23, 40, 103, 166, 229]
base video:          60,212 B
changed pixels:   1,020,226
residual estimate: 472,018 B
exact/repaired:    532,230 B
```

Interpretation:

```text
Projected smoothing makes the base video very small, but it massively increases
decoded class errors. The residual dominates, so exact recovery is much worse
than the current mask.
```

## Decision

```text
Stop q55_mask_range_opt as a main 0.2x path.
Do not integrate mask_range video or residual support into inflate.py.
Do not run package_best.
```

Reason:

```text
Both required gates failed by a large margin:

global palette best exact/repaired: 427,645 B
diffusion best exact/repaired:      532,230 B
stop threshold:                     >205,000 B
current exact AV1 mask:             219,472 B
```

The missed grayscale degree of freedom is real, but under local SVT-AV1 it
trades mask bytes for far too many class flips. Exact residual repair does not
become viable.
