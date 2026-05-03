# Lowmask Qpose Findings

## Context

This family tests the PR #62 low-rate mask regime as the rate prior and qpose14
as the quality teacher.

Hypothesis:

```text
PR #62 lowmask stream + trained renderer + qpose14 distillation
can approach qpose14 quality at roughly 250 KB.
```

This is distinct from qrecode recovery: the renderer is trained on the lowmask
distribution from the start instead of asking a qpose/Quantizr renderer to
tolerate a perturbed exact mask.

## Gate 0

Required outputs:

```text
experiments/lowmask_qpose/fp4_mask_gen_summary.json
experiments/lowmask_qpose/fp4_mask_gen_per_sample.jsonl
experiments/lowmask_qpose/qpose14_teacher_cache.pt
experiments/lowmask_qpose/lowmask_cache.pt
```

The PR #62 archive is expected at:

```text
submissions/fp4_mask_gen/archive.zip
```

Gate 0 completed locally with MPS evaluator.

Archive audit:

```text
archive_bytes = 249,624
sha256        = 9002026e12fdbae0c84b394c180a4e7c8a5078974cfd03b490a02132c735a4b1
model.pt.br   = 66,607 B
mask.obu.br   = 175,336 B
pose.bin.br   = 7,209 B
```

Local MPS fp4 ledger:

```text
full600_quality = 0.64183221
hard3_quality   = 0.36539403
hard8_quality   = 0.45243086
strat64_quality = 1.04365132
```

The local MPS result is much worse than PR #62's CUDA reference, mainly from
PoseNet. This mirrors the already-observed local-vs-CUDA drift, so Gate 1 uses
qpose-relative hard8 metrics under the same local baseline.

## Gate 1

Gate 1 asks whether an overpowered lowmask renderer can match qpose14 hard8
behavior.

Pass:

```text
hard8_quality <= qpose14_hard8_quality + 0.020
max_sample_quality <= qpose14_max_hard8 + 0.040
sample60_pose_term <= qpose14_sample60_pose + 0.020
```

If Gate 1 fails, close `lowmask_qpose_distill` and do not run compressed
variants.

Gate 1 completed on Modal CUDA:

```bash
python submissions/search_vcm_v2/asha.py run \
  --families lowmask_qpose_distill \
  --round hard8_capacity \
  --device cuda \
  --max-steps 5000 \
  --run-id v2_lowmask-qpose-distill_hard8-capacity_cuda_s5000 \
  --candidates lowmask_capacity_qpose14
```

Result:

```text
decision                  = close_lowmask_qpose
gate_pass                 = false
strong_pass               = false
qpose14_hard8_quality     = 0.18193570
best_hard8_quality        = 0.97094486
quality_delta_vs_qpose14  = +0.78900916
best_seg_term             = 0.86873379
best_pose_term            = 0.10221107
best_max_sample_quality   = 1.08943835
max_sample_delta_vs_qpose = +0.91059566
sample60_pose_term        = 0.11640603
sample60_pose_delta       = -0.03756875
best_step                 = 4500
best_stage                = polish
```

The capacity model moved PoseNet into a qpose-compatible regime, including a
better sample-60 pose term than qpose14. It failed because SegNet stayed far
outside the qpose-relative gate. Since the overpowered capacity model cannot
match qpose14 hard8 quality from the PR #62 lowmask stream, compressed
variants are not justified.

## Current Decision

Close `lowmask_qpose_distill`.

Do not run packability, hard8 compressed, strat64, full600, or official dry
scoring for this family.
