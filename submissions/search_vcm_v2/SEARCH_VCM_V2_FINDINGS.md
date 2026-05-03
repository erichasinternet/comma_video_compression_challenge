# Search VCM v2 Findings

## Context

Search VCM v2 tests one hypothesis: keep the exact Quantizr/qpose semantic
mask, replace the bulky renderer/pose machinery with a tiny factorized renderer
and compact per-sample pose tokens.

This branch intentionally does not reopen mask recompression, residual repair,
SCV/TAVS, commaVQ, task-token VCM, task-NeRV, selfcomp++, teacher inversion,
PoseNet-preprocess projection, Quantizr model compression, or hard-sample
sidecars.

## qpose14 Baseline Math

Reference PR #63 values:

```text
archive_bytes = 287,573
segnet_dist   = 0.00061261
posenet_dist  = 0.00052154
quality       = 100*segnet + sqrt(10*posenet) ~= 0.13348
score         ~= 0.32496
```

The qpose14 PR #63 GitHub attachment has been downloaded to
`submissions/qpose14/archive.zip` and is exactly `287,573` bytes. It contains a
single zip member named `p`, matching the PR's single-member archive packing.

The per-sample qpose14 ledger is now built from a real local MPS evaluator run
over the downloaded qpose14 archive. Local CPU reproduced the same aggregate
metrics within reported precision, but both CPU and MPS are worse than the PR
CUDA reference. Search VCM v2 early gates therefore use the local qpose14 ledger
as the comparator.

## Ledger Summary

Run:

```bash
python submissions/search_vcm_v2/asha.py run \
  --families qpose14_baseline \
  --round smoke
```

Expected outputs:

```text
submissions/search_vcm_v2/experiments/qpose14_per_sample.jsonl
submissions/search_vcm_v2/experiments/qpose14_summary.json
```

Current local MPS subset summary:

```text
hard3_quality   = 0.21530885
hard8_quality   = 0.18193570
strat64_quality = 0.19723204
full600_quality = 0.15362587
```

These values come from a real local MPS per-sample evaluation of the downloaded
qpose14 archive. The same inflated output evaluated on local CPU matched MPS
within reported precision:

```text
local MPS: PoseNet=0.00066267 SegNet=0.00072222 score~=0.35
local CPU: PoseNet=0.00066266 SegNet=0.00072220 score~=0.35
PR CUDA:   PoseNet=0.00052154 SegNet=0.00061261 score~=0.325
```

Interpretation: the downloaded artifact is correct, but local CPU/MPS
inflation/evaluation does not reproduce CUDA PR numbers exactly. CUDA remains
the official-path reference.

## Gate 1: hard8 Capacity

Status: qpose14 masks, pose tokens, and hard8 internal 384x512 qpose teacher
frames are materialized at:

```text
submissions/search_vcm_v2/experiments/qpose14_cache/hard8_capacity.pt
```

A one-step Stage-A teacher-imitation smoke validated the exact archive split,
mask decode, pose decode, qpose frame rendering, renderer forward path, and
optimizer plumbing.

The real evaluator-backed Gate 1 run completed on Modal CUDA:

```bash
python submissions/search_vcm_v2/asha.py run \
  --families factorized_exactmask_pose_tokens \
  --round hard8_capacity \
  --device cuda \
  --max-steps 5000 \
  --run-id v2_gate1_hard8_cuda_s5000 \
  --candidates factorized_capacity_joint
```

Pass condition remains:

```text
hard8_quality <= qpose14_hard8_quality + 0.010
max_sample_quality <= qpose14_max_hard8_sample_quality + 0.025
sample60_pose_term <= qpose14_sample60_pose_term + 0.015
```

Result:

```text
decision                  = close_factorized_family
gate_pass                 = false
strong_pass               = false
qpose14_hard8_quality     = 0.18193570
best_hard8_quality        = 0.87274584
quality_delta_vs_qpose14  = +0.69081014
best_seg_term             = 0.72879791
best_pose_term            = 0.14394793
best_max_sample_quality   = 0.96583945
max_sample_delta_vs_qpose = +0.79419113
sample60_pose_term        = 0.15253811
sample60_pose_delta       = -0.00143668
best_step                 = 4300
best_stage                = polish
```

The capacity renderer did reduce PoseNet for sample 60 relative to qpose14, but
SegNet quality remained far outside the qpose-relative gate. Since the
uncompressed overpowered capacity model could not match qpose14 hard8 behavior,
compressed variants are not justified.

## Packability

Implemented packability estimates for:

```text
F16
F24
F32
```

The estimate includes:

```text
mask stream bytes
int8+Brotli renderer state estimate
int8+Brotli pose-token estimate
single-member archive overhead estimate
```

Compressed rounds must not run unless packability projects `<=260 KB`.

## Decision

Current decision: close `factorized_exactmask_pose_tokens`.

Do not run packability, hard8 compressed, strat64, or full600 for this family.
The hard8 capacity gate failed by a wide margin against the local qpose14
baseline.
