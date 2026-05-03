# Q55 Semantic Contour Codec Findings

## Purpose

This branch tests whether the exact rounded Quantizr #55 class mask tensor can
be encoded below the current `mask.obu.br` payload without changing any
evaluator-facing signal.

Target mask payloads:

```text
<=210 KB: continue to a real shared-boundary contour codec
<=196 KB: likely first-place path
<=182 KB: PR-like visible-0.2x path
<=152 KB: local-CPU visible-0.2x path
```

## Phase 1: Scanline Contour Diagnostic

Implementation:

```text
submissions/quantizr/q55_semantic_contour_codec.py
```

The diagnostic encodes each semantic-map row as class runs and tests three
lossless strategies:

```text
raw:     raw row-runs
prev:    exact/delta prediction from the same row in the previous frame
prev_up: exact/delta prediction from previous frame row or current previous row
```

Each run verifies exact reconstruction before byte results are accepted.

Run:

```bash
./.venv/bin/python submissions/quantizr/q55_semantic_contour_codec.py scanline \
  --archive submissions/q55_fp16_pose_int10/archive.zip \
  --out-dir submissions/quantizr/experiments/q55_contour_scanline_v0 \
  --compressors brotli,zstd,xz \
  --verify
```

## Decision

Scanline result:

```text
current AV1 mask.obu.br: 219,472 B

raw row-runs:
  best_per_stream: 339,294 B

previous-frame row prediction:
  best_per_stream: 331,889 B

previous-frame + previous-row prediction:
  best_per_stream: 270,010 B
  delta vs AV1:     +50,538 B
```

Best strategy details:

```text
strategy: prev_up
verified exact: true
rows: 230,400
average runs/row: 2.4731
median runs/row: 1
total boundaries: 339,405
unchanged rows vs previous frame: 157,002
unchanged row fraction vs previous frame: 0.68257

mode counts:
  same previous-frame row: 157,002
  same previous row:         7,152
  delta previous-frame row: 11,688
  delta previous row:       32,858
  raw row:                  21,700

best compressed streams:
  mode.3b:                    16,675 B
  raw_run_count_minus1:        9,808 B
  raw_classes.3b:             23,238 B
  raw_run_lengths.varint:    126,040 B
  boundary_deltas.zzvarint:   94,144 B
  meta:                          105 B
```

Projected archive with the safe non-mask payload:

```text
non-mask bytes:             68,796 B
best scanline mask bytes:  270,010 B
projected archive:         338,806 B
rate term:                 0.22560
required quality <0.300:   0.07440
```

Decision:

```text
Hard fail by the Phase 1 stop rule.
Do not implement the shared-boundary contour codec from this diagnostic.
Do not integrate a contour-mask path into inflate.py.
```

Reason:

```text
The scanline geometry transform is substantially worse than AV1 on the exact
mask tensor. The dominant streams are raw run lengths and boundary deltas,
which means this row-run representation is not close enough for chain coding to
be a disciplined next step under the stated gates.
```
