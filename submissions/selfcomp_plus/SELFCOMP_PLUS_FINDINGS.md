# Selfcomp++ Findings

## Phase 1: Model Repack

Implemented a custom `dcpack` model payload:

```text
submissions/selfcomp_plus/pack_segmap.py
submissions/selfcomp_plus/unpack_segmap.py
submissions/selfcomp_plus/repack_archive.py
submissions/selfcomp_plus/inflate.py
submissions/selfcomp_plus/inflate.sh
```

The `selfcomp_plus` inflater prefers:

```text
segmap.dcpack.br
segmap.dcpack
segmap_inference.pt
```

in that order. Rendering is delegated to the original PR #56 selfcomp implementation to avoid behavior drift.

## Byte Results

Baseline #56 archive:

```text
archive.zip:          279,036 B
payload.tar.xz:       278,820 B
0.mkv:                206,573 B
segmap_inference.pt:  1,239,122 B raw
model-only xz:         ~74,676 B
```

Repacked archive:

```text
archive.zip:          273,579 B
payload.tar.xz:       273,368 B
segmap.dcpack.br:      68,315 B
bytes saved:            5,457 B
```

Repack metrics:

```text
raw dcpack:            1,197,572 B
raw tensor bytes:      1,186,211 B
brotli dcpack:            68,315 B
```

Model equivalence check:

```text
max_abs_diff on random input: 0.0
torch.equal output:           True
```

## Score Geometry

Using local reproduced selfcomp quality:

```text
PoseNet: 0.00039916
SegNet:  0.00115278
quality: 0.17845711
```

At the repacked byte count:

```text
archive bytes:              273,579
rate term:                  0.18216
required quality for 0.300: 0.11784
quality gap:                ~0.06062
```

With PoseNet unchanged, the repacked archive still needs roughly:

```text
SegNet dist <= 0.000546
```

## Decision

The generic exact repack is a real improvement but misses the planned model-payload gate:

```text
weak gate:   <=60 KB model payload
actual:       68.3 KB model payload
```

This does not kill selfcomp++, but it changes the priority:

```text
1. Do not spend more time on generic pickle/container repacking.
2. Proceed to the 64-sample SegNet capacity oracle.
3. Return to model entropy only if SegNet repair passes and the final byte target is close.
```

Reason: the remaining gap is still dominated by SegNet quality, not the 5-10 KB available from safer container work. A stronger specialized entropy format may save a few more KB, but it is not the decisive gate.

## Phase 2 Harness Status

Added the 64-sample capacity oracle harness:

```text
submissions/selfcomp_plus/selfcomp_train_seg_oracle.py
```

Supported modes:

```text
model-only
latent-only
latent+model
```

The script trains against official evaluator components:

```text
SegNet CE to original argmax
SegNet KL to original logits
hard-margin SegNet loss
optional PoseNet guard
TV smoothness on generated output
```

Smoke tests:

```text
zero-step, 2 samples, CPU:
  posenet_dist: 0.00023134
  segnet_dist:  0.00101217
  quality:      0.14931430

one train step, 2 samples, CPU, pose guard off:
  posenet_dist: 0.00021235
  segnet_dist:  0.00100962
  quality:      0.14704393
```

The backward path works, but CPU is too slow:

```text
2 samples, 1 model-only step: ~60 seconds
```

Decision:

```text
Run the real 64-sample capacity oracle on GPU/Modal.
Do not run 1k-2k step experiments locally.
```

## Modal U0 Capacity Oracle

Added Modal runner:

```text
submissions/selfcomp_plus/modal_selfcomp_plus.py
```

GPU smoke:

```text
stage:       smoke
mode:        latent+model
subset:      2
steps:       1
result path: submissions/selfcomp_plus/experiments/modal_returned/gpu_smoke_latent_model_2x1

step 0:
  seg_term:  0.10121664
  pose_term: 0.04709023
  quality:   0.14830686

step 1:
  seg_term:  0.09969076
  pose_term: 0.05428002
  quality:   0.15397078
```

The smoke validated CUDA forward/backward/eval. It also showed the need for conservative learning rates.

U0 `latent+model` capacity oracle:

```text
stage:       u0
mode:        latent+model
subset:      64
batch:       1
max steps:   2000
eval every:  100
result path: submissions/selfcomp_plus/experiments/modal_returned/u0_latent_model_64_s2000_b1
```

Operational fixes before the final U0 run:

```text
eval_now was patched to use inference mode; otherwise baseline eval retained the full autograd graph and OOMed.
the training loop was patched so `steps` means optimizer updates, not full passes over the 64-sample subset.
batch size was reduced to 1 because selfcomp renderer training OOMed on L4 at batch 2/4.
```

Final U0 history:

```text
step 0:
  seg_term:  0.10508697
  pose_term: 0.04270943
  quality:   0.14779640

step 100:
  seg_term:  0.10213058
  pose_term: 0.44727519
  quality:   0.54940577

step 200:
  seg_term:  0.10278225
  pose_term: 0.79661210
  quality:   0.89939435

step 300:
  seg_term:  0.09653569
  pose_term: 1.07237800
  quality:   1.16891369
```

Gate result:

```text
early gate failed at step 300:
  SegNet term improvement = 8.1%, required >=15%
  PoseNet term also worsened catastrophically
```

Interpretation:

```text
The upper-bound latent+model selfcomp SegNet repair did not pass the first capacity gate.
The run did reduce SegNet slightly by step 300, but far too slowly and only by destroying PoseNet.
Per the plan, this stops selfcomp++ as a main 0.2x route unless a new, materially different objective/control idea is introduced.
```

## Pose-Locked SegNet Oracle

Added the frame2-only, pose-locked oracle:

```text
submissions/selfcomp_plus/selfcomp_pose_locked_seg_oracle.py
```

and a Modal entrypoint:

```text
python -m modal run submissions/selfcomp_plus/modal_selfcomp_plus.py \
  --stage pose-lock --variant ...
```

The oracle freezes selfcomp frame1, modifies only frame2, and accepts a checkpoint only if:

```text
pose_term <= baseline_pose_term + pose_eps
seg_term improves over the best accepted state
```

This tests whether a SegNet-improving direction exists in the PoseNet nullspace, instead of repeating the failed global latent/model training.

Results on the same 64-sample subset:

```text
P0 global color, eps=0.005:
  baseline: seg=0.10547638 pose=0.05605318 quality=0.16152957
  best:     seg=0.10536512 pose=0.06073340 quality=0.16609852
  seg improvement: 0.105%
  result: failed 10% gate

P1 8-band vertical color, eps=0.005:
  baseline: seg=0.10547638 pose=0.05605318 quality=0.16152957
  best:     seg=0.10444324 pose=0.06080259 quality=0.16524582
  seg improvement: 0.980%
  result: failed 10% gate

P4 full frame2 residual upper bound, eps=0.005:
  baseline: seg=0.10550817 pose=0.05605293 quality=0.16156110
  best:     seg=0.09628932 pose=0.06091944 quality=0.15720876
  seg improvement: 8.738%
  result: failed 10% gate and far from seg <=0.075 target

P4 full frame2 residual upper bound, eps=0.010:
  baseline: seg=0.10550817 pose=0.05605293 quality=0.16156110
  best:     seg=0.09261767 pose=0.06594552 quality=0.15856319
  seg improvement: 12.218%
  result: relaxed diagnostic only; still far from seg <=0.075 target and spends nearly all relaxed pose budget
```

Important rejected P4 states:

```text
step 25, eps=0.005/0.010:
  seg_term ≈0.0778
  pose_term ≈0.1156
```

Interpretation:

```text
Full pixel residuals can move SegNet toward the target, but the useful moves also move PoseNet out of basin.
The accepted PoseNet-safe nullspace is too small: even the full-frame residual upper bound cannot reach the required SegNet term.
P0/P1 do not justify P2/P3; they fail the cheap color/statistics gate.
```

Decision:

```text
Stop selfcomp++ as a visible-0.2x route.
The remaining practical path is first-place fallback packaging and official-path dry scoring, not more selfcomp architecture or bitrate hardening.
```
