# Next 0.2x Audits

This records the sequential audits run after closing `lowmask_boundary_residual`.

## 1. Hybrid Lowmask / Exact-Tile Router

Tool:

```text
submissions/search_vcm_v2/tools/lowmask_exact_tile_oracle.py
```

Stable CPU hard8 result:

```text
PR #62 lowmask base:
  quality: 0.33622929
  archive: 249,624 B

Best budgeted exact-tile side channel:
  candidate: R5a_tile8_budget4k_top492
  residual: 4,093 B
  archive: 253,717 B
  quality: 0.33336265
  score delta vs FP4 base: -0.00014

Full qpose exact mask through PR #62 renderer:
  archive proxy: 428,829 B
  quality: 0.65060685
```

Decision:

```text
close_exact_tile_router
```

The best budgeted exact-tile helper barely offsets its bytes, and larger helpers get worse. The full exact-mask upper bound through the PR #62 renderer is worse than the lowmask base, so exact-tile routing is not a viable 0.2x path.

## 2. Mask-Stream Context Distillation

Tool:

```text
submissions/search_vcm_v2/tools/mask_context_distill_audit.py
```

Best deterministic context model:

```text
context: 3x3 lowmask class neighborhood
selected contexts: 1,523
table bytes: 2,211 B
residual bytes: 165,260 B
total model+residual: 167,471 B
total mask payload: 342,807 B
```

Gate:

```text
pass target: <=205 KiB total mask payload
strong target: <=182 KiB total mask payload
```

Decision:

```text
close_mask_context_distill
```

The lowmask stream is already 175,336 B, leaving only about 34 KiB for model plus residual at the weak gate. The best context model still needs 167,471 B beyond the lowmask stream.

## 3. qpose14 Model Byte Audit

Artifacts:

```text
submissions/search_vcm_v2/experiments/qpose_model_audit/
```

Payload sizes:

```text
current model.pt.br:               66,841 B
qpack, no extra quantization:      63,680 B
qpack, dense fp16 -> int8:         56,288 B
qpack, heads fp16:                 58,551 B
```

Hard8 relative eval:

```text
qpack none:
  quality delta vs base: 0.0000
  projected score delta: -0.0021

qpack int8:
  quality delta vs base: +0.0656
  projected score delta: +0.0585

qpack int8_heads_fp16:
  quality delta vs base: +0.0473
  projected score delta: +0.0418
```

Decision:

```text
safe qpack is useful but too small; int8 variants are not quality-safe.
```

Even using the PR #63 quality reference, a 3.1 KB model saving only moves qpose14 from roughly `0.325` to roughly `0.323`. This does not approach visible `<0.300`.

## Overall Decision

The remaining lowmask/exact-mask composition routes did not produce a 0.2x candidate:

```text
exact tiles: too little quality gain per byte
mask context distillation: residual remains far too large
qpose14 model compression: safe savings too small; larger savings damage quality
```

The practical contest fallback remains the best safe qpose/q55 packaging candidate unless a genuinely new source representation is admitted under the negative-cache novelty rules.

## 4. PR #67 QZS3 Low-Bit / QAT Audit

Artifacts:

```text
submissions/qpose14_qzs3_filmq9g_slsb1_r55/archive.zip
submissions/search_vcm_v2/tools/qzs3_lowbit_quant_oracle.py
submissions/search_vcm_v2/tools/qzs3_qat_oracle.py
submissions/search_vcm_v2/experiments/qzs3_lowbit_quant_oracle/hard8_summary.json
submissions/search_vcm_v2/experiments/qzs3_qat_oracle_hard3_q2_100/hard3_q2_all_qv8_dense8_summary.json
submissions/search_vcm_v2/experiments/qzs3_qat_oracle_hard3_q3_framewarm_80/hard3_q3_all_qv8_dense8_summary.json
submissions/search_vcm_v2/experiments/qzs3_qat_oracle_hard3_q2conv_100/hard3_q2_conv_qv10_dense16_summary.json
```

PR #67 payload:

```text
archive: 276,564 B
mask:    219,472 B
model:    56,093 B
pose:        899 B
```

At PR #67 quality, visible `<0.300` needs roughly:

```text
archive <=254 KB
required savings ~=22 KB
```

Post-hoc low-bit hard8 result:

```text
base hard8 quality:     0.19053
qv8 savings:            1,328 B raw, quality 1.86403
conv3_all savings:      9,568 B raw, quality 10.81347
conv2_all savings:     19,056 B raw, quality 53.77268
conv3_frame1 savings:   2,327 B raw, quality 4.37413
```

QAT hard3 probes:

```text
q2_all_qv8_dense8:
  savings: 21,764 B raw
  best hard3 quality: 12.43479 without frame warmup
  best hard3 quality:  9.76215 with frame warmup

q3_all_qv8_dense8:
  savings: 12,276 B raw
  best hard3 quality:  2.64722

q2_conv_qv10_dense16:
  savings: 19,056 B raw
  best hard3 quality: 12.00289
```

Decision:

```text
close_qzs3_lowbit_model_reduction
```

The model bytes that need to be removed are exactly the precision-sensitive
ones. Even QAT on hard3 remains orders of magnitude away from the baseline
quality, so a full hard8/full600 GPU run is not justified.

## 5. Slim Qpose-Compatible Structural Distillation

Tool:

```text
submissions/search_vcm_v2/tools/qzs3_slim_distill_oracle.py
```

Hypothesis:

```text
Keep the exact PR #67/qpose mask and pose stream.
Replace only the qpose generator with a width-reduced qpose-compatible model.
Train by qpose frame imitation and evaluator distillation.
```

This differs from the failed factorized renderer because it preserves the qpose
module topology and mask/pose data path. It differs from low-bit QAT because it
removes structure instead of forcing precision-sensitive weights to 2-bit/3-bit.

Byte estimates:

```text
s40:
  raw model estimate: 33,724 B
  projected archive: 252,378 B

s44:
  raw model estimate: 39,674 B
  projected archive: 258,007 B

s48:
  raw model estimate: 46,106 B
  projected archive: 264,092 B
```

Local short hard3 probes:

```text
s40 random, 120 steps:
  projected archive: 252,378 B
  best hard3 quality: 23.25570

s40 sliced teacher init, 80 steps:
  projected archive: 252,378 B
  best hard3 quality: 16.30347

s44 random, 160 steps:
  projected archive: 258,007 B
  best hard3 quality: 8.75086

s48 random, 120 steps:
  projected archive: 264,092 B
  best hard3 quality: 13.99383
```

Decision:

```text
keep_structural_distillation_as_the_only_live_legitimate_reduction_route
```

The short local runs are not candidate-quality. However, this is still the best
new legitimate reduction route because it is the only remaining idea with the
right byte math that does not perturb the exact mask or exploit uncounted code
payload. A real GPU gate should train `s44` and `s40` much longer before closing
the family.

Follow-up hard3 probes:

```text
s44 indexed teacher target, 700 steps:
  projected archive: 258,007 B
  best hard3 quality: 2.88507
  blocker: sample 60 PoseNet term stayed 4.219; qpose hard3 is ~0.2 quality.

s44 indexed teacher target, pose_weight=80, batch=3, 500 steps:
  projected archive: 258,007 B
  best hard3 quality: 7.12675
  blocker: all-sample pose-heavy optimization worsened the basin instead of preserving it.

s44 frame-only teacher imitation, 1000 steps:
  best hard3 quality: 6.2104
  blocker: low RGB Huber error did not imply evaluator compatibility.
```

Decision update:

```text
close_qzs3_slim_structural_distillation
```

The qpose-compatible topology is still highly width-sensitive. The model can
imitate broad RGB structure but cannot preserve PoseNet's temporal basin after
removing the ~20 KB needed for visible `0.2x`.

## 6. PR #67 Pose-Side / Same-Byte Quality Audits

Hypothesis:

```text
If bytes cannot be removed safely, maybe PR #67 can hit <0.300 by improving
quality at roughly the same archive size.
```

At PR #67's archive size (`276,564 B`), visible `<0.300` requires roughly:

```text
quality <= 0.1158
```

Pose channel audit:

```text
PR67 payload pose stream: 899 B QP1 col0-only
qpose14 pose stream: also effectively col0-only for useful channels
CPU hard8 qpose extra-channel/swap test: no net quality improvement
```

Full-model evaluator fine-tune:

```text
all weights, hard8, 300 steps:
  destructive; quality jumped above 0.8-1.3 after task updates.

pose_mlp only, hard8:
  proxy MPS runs produced apparent one-step wins but were not stable.
  independent CPU verification of the best checkpoint:
    base hard8 quality:      0.18018
    candidate hard8 quality: 0.18031
    delta:                  +0.00013
```

Decision:

```text
close_qzs3_pose_channel_expansion
close_qzs3_pose_mlp_finetune
```

Same-byte qpose quality tuning did not produce a verified gain. Larger
evaluator-gradient moves cause sample-level PoseNet blowups; constrained
pose-MLP moves are too small to matter.

## 7. PR #62 FP4 Generator Fine-Tune Audit

Tool:

```text
submissions/search_vcm_v2/tools/fp4_generator_finetune_oracle.py
```

Hypothesis:

```text
Keep PR #62's 249.6 KB byte regime and trained lowmask generator.
Fine-tune the actual generator weights toward evaluator/qpose targets.
```

This differs from the failed `lowmask_qpose_distill` branch because it starts
from the public PR #62 generator that already works somewhat, rather than
training a new factorized renderer from scratch.

CPU hard3 probes:

```text
heads, fp4 anchor, lr=1e-5, 120 steps:
  base quality: 0.23288
  best quality: 0.23197 at step 20
  final quality: 0.24962
  decision: tiny early win, then PoseNet drift.

frame2-only, qpose anchor, lr=5e-6, 60 steps:
  base quality: 0.23288
  best quality: 0.23059 at step 20
  final quality: 0.23682
  decision: best verified win only 0.00229, then drift.

all weights, qpose anchor, lr=2e-6, 60 steps:
  base quality: 0.23288
  best quality: 0.23198 at step 1
  final quality: 0.39732
  decision: destructive once weights move materially.
```

Target math:

```text
At 249,624 B, visible <0.300 requires quality around <=0.1338.
The observed best hard3 improvement was ~0.0023, while the needed improvement is ~0.099.
```

Decision:

```text
close_pr62_fp4_generator_finetune
```

The real PR #62 generator has a small local descent direction, but it is not a
candidate-quality control surface. Meaningful updates quickly leave the PoseNet
basin, and safe updates are two orders of magnitude too small for `0.2x`.

## 8. PR #67 Exact-Mask Range Coding

Tool:

```text
submissions/search_vcm_v2/tools/exact_mask_range_codec.cpp
```

Candidate:

```text
submissions/qzs3_range_mask_candidate/archive.zip
```

Hypothesis:

```text
Keep PR #67's exact evaluator-facing mask, model, and pose stream.
Replace only the mask payload representation with a real adaptive arithmetic
codec for the exact class tensor.
```

This is not a mask perturbation and not a payload-in-code loophole. The archive
stores a compressed exact class tensor; `inflate.py` decodes it with an
algorithmic range decoder and then runs the unchanged PR #67 renderer path.

Mask codec sweep:

```text
PR #67 AV1/Brotli mask:          219,472 B
adaptive5 range + Brotli:        209,920 B
adaptive6pr range + Brotli:      204,527 B
adaptive7prpd range + Brotli:    197,027 B
adaptive8prpdup2 range+Brotli:   189,654 B
adaptive9up2left2 range+Brotli:  188,214 B
```

The winning context is:

```text
event predictor: prev-frame same pixel, current left, current up
adaptive context:
  prev, left, up, up-left, up-right, prev-right, prev-down, current up2, current left2
```

Candidate byte split:

```text
mask:     188,214 B
model:     56,093 B
pose:         899 B
zip ovh:      100 B
archive:  245,306 B
sha256:   115aee70f14c0e97661388bc3b2e7ca10c237c1a3277566576ef312add50ec61
```

Validation:

```text
range decode exactness:
  decoded class tensor == qpose exact class tensor

inflated output equivalence:
  qzs3_range_mask_candidate/inflated/0.raw
  ==
  qpose14_qzs3_filmq9g_slsb1_r55/inflated/0.raw
```

Public PR #67 projected score:

```text
reported SegNet:  0.00061000
reported PoseNet: 0.00048597
quality:          0.13071155
archive:          245,306 B
rate term:        0.16333920
projected score:  0.29405074
```

Local notes:

```text
uv-run evaluate.sh path succeeds locally on MPS:
  archive: 245,306 B
  PoseNet: 0.00063294
  SegNet:  0.00072216
  score:   0.32 (local evaluator drift, not output mismatch)

Local MPS/CPU evaluator remains pessimistic versus public CUDA, as seen with
q55/qpose/qzs3 earlier. Modal CUDA evaluation reached the evaluator but DALI
failed with an NVML internal driver error before scoring. This is an evaluator
infrastructure issue, not a candidate decode or quality mismatch.
```

Decision:

```text
package_qzs3_range_mask_candidate
```

This is the first legitimate candidate in the audit sequence with enough byte
reduction to project below visible `0.300` while preserving PR #67 output
bit-for-bit.
