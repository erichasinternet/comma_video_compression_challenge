# Search VCM v1 Findings

## Infrastructure

`submissions/search_vcm` is now the experiment governor for v1 search:

```text
fixed subsets
negative cache
append-only JSONL run ledger
fallback rows
oracle-only rows
q55 Pareto registration rows
promotion gates
```

The current ledger is:

```text
submissions/search_vcm/experiments/search_vcm/runs.jsonl
```

The base per-sample q55 ledger is:

```text
submissions/search_vcm/experiments/search_vcm/base_q55_fp16_pose_int10_per_sample.jsonl
```

## PoseNet Preprocess Oracle

B1 direct PoseNet-preprocessed tensor optimization is implemented as a non-packable oracle.

B1 original-preprocess sanity:

```text
run_id: b1_sanity_interp_hard8_v2
subset: [59, 60, 62, 56, 57, 58, 61, 63]
sanity pose term: 0.00000000
gate: pass
```

This confirms the target, preprocessing, and sample indexing are consistent.

B1 q55-to-original interpolation:

```text
alphas: 0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0
first hard8 gate alpha: 1.0
sample60 first gate alpha: 1.0
conclusion: only_original_reaches_gate
```

Intermediate interpolation points are worse than the q55 starting point. Under the decision table, this does not justify B2 projection or a stronger per-sample B1 run.

Hard8 run:

```text
run_id: b1_hard8_200step
subset: [59, 60, 62, 56, 57, 58, 61, 63]
steps: 200
device: mps
```

Result:

```text
baseline pose term:  0.11827796
optimized pose term: 0.08974943
average pose drop:   41.35%
sample 60 drop:      36.53%
gate:                fail
```

Required gate:

```text
hard8 PoseNet term <=0.050
```

B2 projection did not run because there was no passing hard8 B1 artifact:

```text
decision: b2_blocked_or_failed
reason: no passing B1 artifact found
```

## Decision

Do not run B2 or any packable PoseNet-projection work from this B1 result. The abstract PoseNet input space is movable, but the current B1 optimizer did not clear the reachability gate.

`posenet_preprocess_oracle` is now marked closed in `negative_cache.yaml`:

```text
PoseNet target is reachable and preprocessing is correct, but hard8 interpolation reaches the gate only at alpha=1.0, meaning the q55-to-original path requires essentially full original PoseNet-space information. Not a low-byte local control surface.
```

## Fallback Dry Scores

Root `evaluate.sh` was run on MPS for the three q55 fallback packages. This exercises:

```text
archive.zip -> unzip -> inflate.sh -> inflated raw video -> evaluate.py
```

Results:

```text
q55_fp16_pose_int10
  bytes:   288,268
  PoseNet: 0.00065135
  SegNet:  0.00072222
  score:   0.34487409

q55_fp16_pose_int12
  bytes:   289,127
  PoseNet: 0.00064976
  SegNet:  0.00072222
  score:   0.34534749

q55_fp16_only
  bytes:   296,659
  PoseNet: 0.00064992
  SegNet:  0.00072222
  score:   0.35037267
```

Decision:

```text
q55_fp16_pose_int10 is the best local fallback among these three.
None of the local MPS dry scores beats the public 0.33 target.
CUDA/T4 official-runner calibration was not available locally.
```

## Evaluator Validation

Trusted local evaluator work is now documented in:

```text
submissions/search_vcm/EVALUATOR_VALIDATION.md
```

Harness:

```text
submissions/search_vcm/evaluator_validation.py
```

Key local evaluator issue found:

```text
evaluate.sh and submission inflate.sh scripts call `python`.
The local shell did not expose `python` globally, so fresh evaluation failed
until the harness prepended `.venv/bin` to PATH.
```

Fresh root-evaluator reproductions:

```text
q55_fp16_pose_int10
  score: 0.3448740862
  status: pass

q55_fp16_pose_int12
  score: 0.3453474935
  status: pass

selfcomp_pr56_eval
  score: 0.3642577293
  status: pass
```

The trusted local score path is now:

```text
archive.zip -> evaluate.sh -> inflate.sh -> inflated raw -> evaluate.py -> parsed full-precision JSON
```

Proxy/subset ledgers remain useful for ranking but are not trusted submission scores.

## Teacher-Distilled Inflation

`teacher_distilled_inflation` was admitted as an opt-in Search VCM family, not part of the default v1 family list.

Novelty test:

```text
1. Not a mask perturbation.
2. Not local q55 output control as a packable candidate.
3. Does not depend on PoseNet moving locally from q55 toward original at inflate time.
4. Targets the 245-260 KB region through a future tiny inflater.
5. Has a hard8 teacher oracle before any student/full run.
```

Implemented live stage:

```text
Gate 1 variants:
  teacher_gate1_t1_direct
    q55-init direct RGB/YUV residual
  teacher_gate1_t2_lowmid
    q55-init low+mid-frequency residual, larger delta
  teacher_gate1_t3_continuation
    q55 blended with low-frequency original, then optimized

kind: oracle_only
packable: false
input: q55 frames as initialization, original evaluator targets
output: non-packable teacher frames for possible student distillation
hard8 gate: teacher quality <=0.080
strat64 gate: teacher quality <=0.110
```

Local MPS smoke:

```text
run_id: teacher_gate1_smoke_v2
subset: smoke [0, 1]
steps: 1
baseline quality: 0.15401387
best quality:     0.15053108
decision: plumbing ok
```

Local MPS hard8 pilot:

```text
run_id: teacher_gate1_hard8_mps25
subset: [59, 60, 62, 56, 57, 58, 61, 63]
steps: 25
q55-init baseline quality: 0.29048857
best teacher quality:      0.29048857
best step:                 0
original-frame upper:      0.04880480
gate:                      fail
```

Decision:

```text
This pilot verifies the family plumbing and shows original-frame targets can clear Gate 1, but the current q55-init teacher optimizer did not move in 25 MPS steps. This is not a final Gate 1 close; the real admission gate remains a bounded L4/A10 hard8 run.
```

Real L4/A10 Gate 1 sweep:

```text
run_id: teacher_gate1_hard8_gpu_s1500_b2_e50
subset: [59, 60, 62, 56, 57, 58, 61, 63]
steps: 1500
batch: 2
eval_every: 50
remote run: https://modal.com/apps/erichasinternet/main/ap-ZQJKDVQFUggS0NENW14LGK
local summary: submissions/search_vcm/experiments/search_vcm/modal_returned/teacher_gate1_hard8_gpu_s1500_b2_e50/modal_result.json
```

Results:

```text
T1 direct
  baseline quality: 0.29011440
  best quality:     0.29011440
  best step:        0
  step250:          6.17466041
  step750:          10.51410322
  step1500:         16.91557583
  gate:             fail

T2 low+mid residual
  baseline quality: 0.29011440
  best quality:     0.29011440
  best step:        0
  step250:          37.47927774
  step750:          38.07714210
  step1500:         37.74588604
  gate:             fail

T3 continuation
  baseline quality: 0.29011440
  best quality:     0.84520578
  best step:        100
  step250:          1.58354000
  step750:          3.88789277
  step1500:         7.13992050
  gate:             fail

Original upper:
  quality:          0.04670672
```

Decision:

```text
Close teacher_distilled_inflation.

Original frames pass the target, but offline teacher inversion from q55/regularized continuations cannot find a distillable hard8 target under the required quality gate.
T1/T2 did not improve over q55 at any selected checkpoint; T3 reduced SegNet but left PoseNet far outside the basin.
No student distillation work is authorized.
```
