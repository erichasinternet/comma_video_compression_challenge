# Quantizr #55 Restart Progress Report For GPT Pro

Prepared: 2026-04-26

Workspace: `/Users/eric/comma_video_compression_challenge`

Important constraint: do not create a contest PR or submit anything yet. The current task is candidate generation, local/Modal evaluation, and decision-making toward a visible `0.2x` result.

## Executive Status

The restart is still correctly anchored on the exact public Quantizr #55 artifact. Since the last plan, the major empirical update is negative but useful: blind mask recompression and regenerated/mixed-CRF masks are much more dangerous than expected. Even CRF50 recoding of the decoded #55 mask, with only `0.0201%` hard class changes, caused PoseNet to degrade severely. The safe byte wins so far are model fp16 qpack and pose int10 qpack, producing a local CPU score around `0.3449` at `288,268` bytes.

The current best local package is:

```text
q1_fp16_pose_int10_cpu
archive: 288,268 bytes
sha256: 9f1f005a48514a7215e72b7ddbae36c87a4aeef94f361c02bbf9e15c60d0de03
PoseNet: 0.00065137
SegNet:  0.00072220
quality: 0.15292750
score:   0.34487333 local CPU
```

Modal exact-mask polish from logs slightly improved this to approximately:

```text
archive: 287,918 bytes
sha256: a2d7745e3d2280b3225bba2258f6f4864abb639938ba3bd9db4f1a815cec338f
PoseNet: 0.00064770
SegNet:  0.00072204
quality: 0.15268381
score:   0.34439659 local CPU/AV path
```

That improvement is only about `0.00048`, so exact-mask polish is not a high-ROI scaling path.

Post-verdict addendum 1: an exact-class AV1+residual diagnostic has now been implemented and tested. It restores the exact rounded class tensor, but the first real CRF50 predictor is not byte-competitive:

```text
qmask_resid_from_qrecode50_int10
predictor: qrecode50_archive_cpu mask.obu.br
base mask predictor: 215,818 bytes
residual:            38,455 bytes
manifest:             1,153 bytes
mask total:          255,426 bytes
delta vs original mask.obu.br: -35,954 bytes
archive total:       324,466 bytes
exact class recovered: true
```

This means exact-class residual repair works mechanically, but the naive sparse residual format plus CRF50 predictor is too large. Row-run and class-bitplane checks were also worse because the changed pixels are mostly isolated.

Post-verdict addendum 2: deterministic mask cleaners and learned class-denoising were also tested on `qrecode50`. Both failed the sanity gate. The best learned gray-feature threshold changed only `3` pixels, repaired `2` of `23,747` errors, introduced `1` new error, and left `23,746` errors. Lower thresholds repaired more true defects but introduced huge numbers of false edits. This means the current mask defects are not locally identifiable well enough from class/gray context to justify CRF52/54 denoiser training.

Post-verdict addendum 3: a tiny no-op-initialized evaluator-level mask embedding adapter was implemented and trained on `qrecode50`. It also failed. The step-0 proxy quality was `0.31514`; after 1000 Modal steps it was `0.31784`, worse than start. The adapter package was small (`2,191` bytes), but it did not move the PoseNet failure.

Post-verdict addendum 4: the stronger full-generator qrecode50 warm-start sanity was launched on Modal and stopped early after it failed the gate. After two mask-adaptation epochs plus one SegNet hard-margin epoch, proxy eval was still `Seg(x100)=0.0829`, `Pose=sqrt(10*pose)=0.2322`, quality `0.3151`. This is essentially unchanged from raw qrecode50 quality (`~0.3148`) and far above the `<=0.220` early gate. Decision: stop compressed-mask recovery for now and pivot to exact-mask architecture/padding fallback.

Post-verdict addendum 5: exact-mask architecture payload support was added so small Quantizr variants can be evaluated with `arch.json.br`. The first padding-mode probe, `reflect`, failed catastrophically at zero-step even though it used the exact #55 mask stream:

```text
q55_exact_mask_polish_fp16_pose_int10_per_dim_reflect_h52_c48_1a_4s_2p_cpu_av zero-step
archive: 288,486 bytes
PoseNet: 0.15922065
SegNet:  0.00090440
score:   1.54
```

This rejects padding-mode changes as a likely path. A zero-step score gate has now been added to the warm-start runner so future architecture probes abort before training if the pretrained package is already invalid.

Post-verdict addendum 6: the plan has pivoted from mask reduction to a fixed-mask pixel-space evaluator oracle. A new diagnostic, `q55_pixel_oracle.py`, directly optimizes decoded frames while keeping the current safe package bytes fixed (`288,268` bytes: exact #55 mask, fp16 model qpack, int10 pose qpack). Early unconstrained/logit optimization destroyed PoseNet, so the oracle was tightened to:

```text
bounded delta around #55 decoded frames
max per-pixel delta: 4
camera-resize simulation enabled
hard-pixel-only SegNet loss on initial SegNet mismatches
best actual evaluator-quality frame tracked every optimizer step
pose-term guard weight: 20
```

The first meaningful 64-sample Modal probes are positive but not sufficient:

```text
q55_pixel_oracle_delta4_64s_100steps_b2_camera_av
archive bytes used for projection: 288,268
initial 64-sample quality:        0.12335298
final/best 64-sample quality:     0.11760255
projected score:                  0.31529881 -> 0.30954838
PoseNet:                          0.00033037 -> 0.00031694
SegNet:                           0.00065875 -> 0.00061305

q55_pixel_oracle_delta4_64s_500steps_b2_camera_av
archive bytes used for projection: 288,268
initial 64-sample quality:        0.12335298
final/best 64-sample quality:     0.11746203
projected score:                  0.31529881 -> 0.30940786
PoseNet:                          0.00033037 -> 0.00031420
SegNet:                           0.00065875 -> 0.00061409
```

Interpretation: fixed-mask adversarial frame optimization can improve evaluator quality without changing the payload, but the current bounded-delta objective stalls around `0.1175`, above the `~0.10805` quality needed for `<0.300` at `288,268` bytes. Longer optimization did not help: median best step was `6`, and the best step never exceeded `33`. Decision: do not build residual-head distillation yet. Add early-stop support and run a small objective sweep with lower learning rates, alternate hard-pixel weighting, and stronger PoseNet guards.

Post-verdict addendum 7: the first objective sweep is complete. Full-frame weighted SegNet loss worked better than hard-pixel-only loss, but the best 64-sample oracle is still not under the sub-`0.300` gate:

```text
q55_pixel_oracle_sweep_full_d4_lr002_p30_hb64_64s
archive bytes used for projection: 288,268
initial 64-sample quality:        0.12335298
final/best 64-sample quality:     0.11526622
projected score:                  0.31529881 -> 0.30721205
PoseNet:                          0.00033037 -> 0.00030947
SegNet:                           0.00065875 -> 0.00059636
```

The oracle improvement is real (`-0.00809` quality on the 64-sample subset), but it still misses the fixed-byte requirement:

```text
required quality at 288,268 bytes: <=0.108054
best oracle quality so far:         0.115266
remaining gap:                      ~0.00721 quality
```

Decision: do not train/distill adversarial residual heads yet. The next step is tail diagnosis, not another generic sweep. The sweep logs show the best steps usually happen early and that the hard tail appears concentrated near the last chunks of the first 64-sample subset. `q55_pixel_oracle.py` now has per-sample metric output, and `modal_q55_restart.py` now has a tail-focused stage:

```text
stage: q55-pixel-oracle-tail-av
subset: offset 56, 8 samples
goal: determine whether the remaining oracle gap is dominated by a few hard PoseNet/SegNet samples
```

If tail-focused optimization gets the hard subset near or below the same quality as the rest of the batch, residual-head distillation becomes plausible again. If the hard samples remain above the gate even under direct pixel optimization, then fixed-mask adversarial residuals are not a credible `0.2x` path without a stronger oracle objective.

Post-verdict addendum 8: the offset-56 tail run is complete and confirms the hard-tail hypothesis, but it is negative for the current mixed SegNet/PoseNet oracle:

```text
offset 56, 8 samples
initial quality: 0.182435

best mixed-objective variant:
  q55_pixel_oracle_tail_o56_full_d4_lr0015_p40_hb64_8s
  final quality: 0.176692
  projected score: 0.368638
  PoseNet: 0.00139312 -> 0.00143039
  SegNet:  0.00064405 -> 0.00057093
```

The tail is dominated by PoseNet, not SegNet. The worst samples after optimization remain:

```text
sample 60: quality_like 0.22879, PoseNet 0.00266276
sample 59: quality_like 0.21230, PoseNet 0.00238147
sample 62: quality_like 0.18751, PoseNet 0.00169093
```

Interpretation: the current oracle mostly improves SegNet while failing to repair the PoseNet-heavy tail. A final narrowed oracle was added before abandoning the fixed-mask adversarial-residual path:

```text
stage: q55-pixel-oracle-tail-pose-av
objective: PoseNet-only / PoseNet-dominant optimization on offset 56-63
gate: PoseNet must materially drop on samples 59, 60, and 62
```

If the pose-only oracle cannot move those samples, then residual-head distillation should be stopped; the direct output-space oracle itself cannot solve the hard tail.

Post-verdict addendum 9: the PoseNet-only tail oracle also failed. It barely moved the aggregate tail PoseNet term and did not materially improve the hard samples:

```text
q55_pixel_oracle_tail_pose_o56_d8_lr0005_p200_m50_8s
quality: 0.180971 -> 0.180426
projected score: 0.372917 -> 0.372372
PoseNet: 0.00136471 -> 0.00135496
SegNet:  0.00064150 -> 0.00064023

q55_pixel_oracle_tail_pose_o56_d16_lr0003_p300_m100_8s
quality: 0.180971 -> 0.180608
projected score: 0.372917 -> 0.372554
PoseNet: 0.00136471 -> 0.00135477
SegNet:  0.00064150 -> 0.00064214
```

The hard samples remain:

```text
sample 60: PoseNet ~0.00239-0.00240, quality_like ~0.229
sample 59: PoseNet ~0.00204, quality_like ~0.209
sample 62: PoseNet ~0.00166, quality_like ~0.196
```

Decision: stop the fixed-mask adversarial residual-head path for now. The direct output-space oracle cannot repair the PoseNet tail under bounded perturbations up to `16` pixels, so distilling that oracle into a tiny decoder head is not credible.

## Evaluation Anchors

Official score formula:

```text
score = 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate
rate = archive.zip bytes / 37,545,489
```

Known #55 artifact:

```text
archive bytes: 299,970
archive sha256: af61d60863243b0a7093b72266c925d9a179bb990e8afd60efbff9fd373e652a
payloads:
  mask.obu.br: 219,472 bytes
  model.pt.br: 66,841 bytes
  pose.npy.br: 13,185 bytes
```

#55 PR/GitHub-reported CUDA result:

```text
PoseNet: 0.00051328
SegNet:  0.00061261
quality: ~0.13290
score:   ~0.33
```

Local devcontainer CPU result for exact #55:

```text
PoseNet: 0.00064988
SegNet:  0.00072220
quality: 0.15283514
score:   0.35257285
```

RunPod RTX 4090 CUDA/DALI result for exact #55:

```text
PoseNet: 0.00104610
SegNet:  0.00077994
quality: 0.18027303
score:   ~0.38001074
```

Interpretation: evaluator/device variance is real. Local CPU is useful as a relative filter, not final leaderboard truth. RunPod 4090 CUDA/DALI did not match the PR score and should not be used as final truth. Every row must remain runner-tagged.

## Result Table Since Last Plan

All rows below are local CPU unless explicitly noted.

| Label | Bytes | Score | Quality | Required quality for `<0.300` | PoseNet | SegNet | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `q0_modified_cpu_current` | 299,970 | 0.352573 | 0.152835 | 0.100262 | 0.00064988 | 0.00072220 | Baseline |
| `q1_fp16_cpu` | 296,659 | 0.350368 | 0.152835 | 0.102467 | 0.00064988 | 0.00072220 | Safe small win |
| `q1_fp16_pose_int16_cpu` | 290,891 | 0.346533 | 0.152841 | 0.106308 | 0.00064997 | 0.00072220 | Safe |
| `q1_fp16_pose_int12_cpu` | 289,127 | 0.345345 | 0.152827 | 0.107482 | 0.00064975 | 0.00072220 | Safe |
| `q1_fp16_pose_int10_cpu` | 288,268 | 0.344873 | 0.152927 | 0.108054 | 0.00065137 | 0.00072220 | Current saved best |
| `q1_mixed_int8_heads_fp16_cpu` | 291,530 | 0.376046 | 0.181928 | 0.105882 | 0.00120339 | 0.00072229 | Reject without recovery |
| `modal_qpack_recover_int10_cpu_av` | 283,346 | 0.357919 | 0.169251 | 0.111332 | 0.00094146 | 0.00072222 | Reject |
| `qrecode50_archive_cpu` | 296,160 | 0.512009 | 0.314808 | 0.102799 | 0.00537290 | 0.00083013 | Reject |
| `qcrf50_cpu` | 299,360 | 0.635546 | 0.436214 | 0.100668 | 0.01368456 | 0.00066288 | Reject regenerated masks |
| `qmask_50_0_20_54_0_35_58_0_45_cpu` | 279,187 | 0.907712 | 0.721813 | 0.114101 | 0.03605160 | 0.00121383 | Reject |
| `qmask_52_0_25_56_0_35_60_0_40_cpu` | 261,112 | 1.007594 | 0.833730 | 0.126136 | 0.04711421 | 0.00147332 | Reject |
| `modal_exact_mask_polish_int10` | 287,918 | 0.344397 | 0.152684 | 0.108287 | 0.00064770 | 0.00072204 | Minor win only |
| `modal_reflect_padding_zero_step` | 288,486 | 1.54 | n/a | n/a | 0.15922065 | 0.00090440 | Reject padding-mode change |
| `q55_pixel_oracle_delta4_64s_100steps_b2_camera_av` | 288,268 | 0.309548 projected | 0.117603 | 0.108054 | 0.00031694 | 0.00061305 | Positive oracle, continue to 500-step |
| `q55_pixel_oracle_delta4_64s_500steps_b2_camera_av` | 288,268 | 0.309408 projected | 0.117462 | 0.108054 | 0.00031420 | 0.00061409 | Saturated; tune oracle objective before distillation |
| `q55_pixel_oracle_sweep_hard_d4_lr002_p30_64s` | 288,268 | 0.309536 projected | 0.117591 | 0.108054 | 0.00031671 | 0.00061313 | Similar to 500-step hard-only; not enough |
| `q55_pixel_oracle_sweep_full_d4_lr002_p30_hb64_64s` | 288,268 | 0.307212 projected | 0.115266 | 0.108054 | 0.00030947 | 0.00059636 | Best oracle so far; still above gate |
| `q55_pixel_oracle_sweep_hard_d8_lr001_p40_64s` | 288,268 | 0.309285 projected | 0.117339 | 0.108054 | 0.00031442 | 0.00061266 | Slight hard-only improvement; not enough |
| `q55_pixel_oracle_tail_o56_full_d4_lr0015_p40_hb64_8s` | 288,268 | 0.368638 projected | 0.176692 | 0.108054 | 0.00143039 | 0.00057093 | Tail remains PoseNet-dominated |
| `q55_pixel_oracle_tail_o56_full_d8_lr001_p60_hb64_8s` | 288,268 | 0.369169 projected | 0.177223 | 0.108054 | 0.00143854 | 0.00057284 | No improvement over d4 |
| `q55_pixel_oracle_tail_o56_hard_d8_lr001_p60_8s` | 288,268 | 0.371078 projected | 0.179132 | 0.108054 | 0.00147078 | 0.00057856 | Worse than full-frame tail |
| `q55_pixel_oracle_tail_pose_o56_d8_lr0005_p200_m50_8s` | 288,268 | 0.372372 projected | 0.180426 | 0.108054 | 0.00135496 | 0.00064023 | Pose-only does not move hard tail |
| `q55_pixel_oracle_tail_pose_o56_d16_lr0003_p300_m100_8s` | 288,268 | 0.372554 projected | 0.180608 | 0.108054 | 0.00135477 | 0.00064214 | Pose-only does not move hard tail |

Mask-repair diagnostics without full score rows:

| Label | Mask / archive bytes | Class errors before | Class errors after | Key result | Decision |
|---|---:|---:|---:|---|---|
| `qmask_resid_from_qrecode50_int10` | mask total `255,426`, archive `324,466` | 23,747 | 0 | Exact class recovery works, but costs `+35,954` mask bytes vs original | Reject as main path |
| `qmask_clean_qrecode50_int10_d0` | diagnostic only | 23,747 | 24,734 | Repaired 195, introduced 1,182 | Reject |
| `qmask_clean_qrecode50_int10_d1` | diagnostic only | 23,747 | 33,487 | Repaired 880, introduced 10,620 | Reject |
| `qmask_clean_qrecode50_int10_d1_5` | diagnostic only | 23,747 | 44,683 | Repaired 1,131, introduced 22,067 | Reject |
| `qmask_clean_qrecode50_int10_d2` | diagnostic only | 23,747 | 622,636 | Temporal singleton repair explodes false edits | Reject |
| `qmask_clean_qrecode50_int10_d3` | diagnostic only | 23,747 | 618,471 | Spatiotemporal repair explodes false edits | Reject |
| `qmask_denoise_qrecode50_sanity_1000s` | diagnostic archive `485,666` | 23,747 | 1,760,312 | Raw argmax repairs many true errors but edits 1.78M pixels | Reject |
| `qmask_denoise_qrecode50_sanity_thr_1000s` | diagnostic archive `485,666` | 23,747 | 23,747 | Best threshold is do-nothing; threshold 10 repairs 136 but introduces 325 | Reject |
| `qmask_denoise_qrecode50_sanity_gray_thr_1000s` | diagnostic archive `485,906` | 23,747 | 23,746 | Best threshold repairs 2 and introduces 1; threshold 6 repairs 1,074 but introduces 4,418 | Reject |
| `qmask_adapter_qrecode50_sanity_1000s` | archive `286,917`, adapter `2,191` | n/a | n/a | Proxy quality `0.31514 -> 0.31784`; no recovery | Reject tiny adapter |
| `qrecode50_full_recover_fp16_pose_int10...` | interrupted sanity run | n/a | n/a | Proxy quality remained `~0.3151` after early full-generator warm-start | Reject compressed-mask recovery |

## What Changed Since The Last Plan

### 1. QMASK mixed-CRF was implemented and falsified in zero-step form

The mixed-mask format was implemented with:

```text
mask_mix.json.br
mask_gXX_crfYY.obu.br
```

`inflate.py` can decode grouped mask streams and restore the original frame order.

Two mixed-CRF tests were run:

```text
qmask_50_0_20_54_0_35_58_0_45_cpu
archive: 279,187 bytes
score:   0.907712
PoseNet: 0.03605160
SegNet:  0.00121383

qmask_52_0_25_56_0_35_60_0_40_cpu
archive: 261,112 bytes
score:   1.007594
PoseNet: 0.04711421
SegNet:  0.00147332
```

Decision: current QMASK based on regenerated source masks is dead without a trainable adapter/recovery mechanism. Do not spend more time on blind mixed-CRF allocation from regenerated masks.

### 2. Regenerated CRF50 masks do not reproduce #55

CRF50 source-mask regeneration result:

```text
qcrf50_cpu
archive: 299,360 bytes
score:   0.635546
PoseNet: 0.01368456
SegNet:  0.00066288
```

This fails the original hard gate. It means the source-mask generation path does not match the exact #55 mask distribution closely enough. Any CRF52/54/56 result from this path would be misleading.

Decision: do not continue regenerated-mask CRF training until the mask extraction/reproduction issue is solved or a deliberate adapter is added.

### 3. Re-encoding the decoded #55 mask also failed

Archive-mask recode at CRF50:

```text
qrecode50_archive_cpu
archive: 296,160 bytes
score:   0.512009
PoseNet: 0.00537290
SegNet:  0.00083013
```

Mask comparison against exact #55:

```text
frames: 600
shape: 384x512
raw_changed_fraction: 0.00887508
raw_abs_mean: 0.05015352
hard class changed_fraction: 0.00020131
hard class changed pixels: 23,747
```

Important implication: the Quantizr generator is extremely sensitive to small decoded-mask perturbations. The model consumes class indices after grayscale decode/rounding, but only `0.0201%` hard class changes were enough to make PoseNet explode. Blind lossy mask recoding is not safe.

Decision: stop treating "CRF50 reproduces closely" as likely. It has been falsified on the local CPU path.

### 4. qpack and pose quantization produced safe byte wins

Safe packages:

```text
fp16 model qpack:
  archive: 296,659 bytes
  score:   0.350368
  quality unchanged vs #55 CPU

fp16 model qpack + int12 pose:
  archive: 289,127 bytes
  score:   0.345345
  quality essentially unchanged

fp16 model qpack + int10 pose:
  archive: 288,268 bytes
  score:   0.344873
  slight PoseNet change, acceptable
```

Rejected:

```text
mixed_int8_heads_fp16:
  archive: 291,530 bytes
  score:   0.376046
  PoseNet worsened to 0.00120339

Modal mixed-int8 recovery:
  archive: 283,346 bytes
  score:   0.357919
  PoseNet still too high at 0.00094146
```

Decision: keep fp16 model qpack plus int10 pose as the clean low-risk package. Do not use mixed-int8 unless a later pack-aware training stage can recover PoseNet.

### 5. Modal exact-mask polish was tested and is low ROI

Modal run used the exact #55 mask stream and the int10 pose/fp16 qpack package. It was intended to see whether metric fine-tuning can improve quality without changing the mask.

Modal issues:

```text
first worker was preempted around 60% of stage 1
Modal restarted automatically
archive-return capture was patched after this run, so this archive was not saved locally
```

Final result from Modal logs:

```text
archive: 287,918 bytes
payloads:
  mask.obu.br:    219,472 bytes
  model.qpack.br: 63,330 bytes
  pose.qpack.br:  4,790 bytes
PoseNet: 0.00064770
SegNet:  0.00072204
quality: 0.15268381
score:   0.34439659
```

Decision: exact-mask polish works technically but barely moves the score. It is not the path to `0.2x`.

### 6. Lossless class-mask compression is worse than the existing AV1 mask stream

Decoded #55 class masks:

```text
shape: 600 x 384 x 512
raw class bytes: 117,964,800
existing mask.obu.br: 219,472 bytes
```

Quick lossless probes:

```text
raw classes zstd22:      439,241 bytes
raw classes zlib9:       581,384 bytes
raw classes brotli q6:   503,996 bytes
3-bit classes zstd22:    482,809 bytes
3-bit classes brotli q6: 522,080 bytes
mod5 horizontal zstd22:  471,334 bytes
```

Decision: do not replace the AV1 mask stream with direct lossless class storage. The original `mask.obu.br` is already very efficient.

## Current Gap To 0.2x

For `q1_fp16_pose_int10_cpu`:

```text
archive bytes: 288,268
rate term:     0.19194583
quality:       0.15292750
score:         0.34487333
required quality for score <0.300 at this byte size: 0.10805417
quality gap:   0.04487333
```

If this package somehow achieved PR-like #55 quality `~0.13290`, the score would still be:

```text
0.13290 + 0.19194583 = 0.32484583
```

So qpack plus pose quantization alone is insufficient even under the favorable PR-quality assumption.

At current local CPU quality `~0.15293`, the archive would need to be about:

```text
<=220,876 bytes
```

At PR-like #55 quality `~0.13290`, the archive needs to be about:

```text
<=250,954 bytes
```

Implication: a true `0.2x` candidate needs one of:

```text
1. Reduce bytes by another ~37 KB while preserving PR-like #55 quality.
2. Reduce bytes by another ~67 KB while preserving local CPU #55 quality.
3. Improve quality dramatically, roughly from 0.1529 to 0.1081 at 288 KB.
4. Combine a smaller, trained mask representation with quality recovery.
```

The latest tests make option 4 harder than expected because even mild mask changes are damaging.

## Current Artifact Inventory

Clean local packages created for non-submission evaluation:

```text
submissions/q55_fp16_pose_int10/archive.zip
submissions/q55_fp16_pose_int10/inflate.py
submissions/q55_fp16_pose_int10/inflate.sh

submissions/q55_fp16_pose_int12/archive.zip
submissions/q55_fp16_pose_int12/inflate.py
submissions/q55_fp16_pose_int12/inflate.sh
```

Experiment source and results:

```text
submissions/quantizr/experiments/q55_restart/source/q55_archive.zip
submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int10_cpu/submission/archive.zip
submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int12_cpu/submission/archive.zip
submissions/quantizr/experiments/q55_restart/*/metrics.json
```

Modal support:

```text
submissions/quantizr/modal_q55_restart.py
```

Exact-class residual support:

```text
submissions/quantizr/q55_mask_residual.py
submissions/quantizr/inflate.py supports mask_residual.json.br + mask_residual.bin.br
```

Mask-cleaner / class-denoiser support:

```text
submissions/quantizr/q55_mask_cleaner.py
submissions/quantizr/q55_mask_denoise.py
submissions/quantizr/q55_mask_adapter_train.py
submissions/quantizr/inflate.py supports diagnostic mask_clean.bin.br
submissions/quantizr/inflate.py supports optional mask_adapter.pt.br
submissions/quantizr/modal_q55_restart.py includes qmask-denoise-qrecode50 stage
```

Modal script now includes archive-return support for future runs:

```text
load_metrics_with_archive(label)
save_returned_archive(result)
local output: submissions/quantizr/experiments/q55_restart/modal_returned/<label>/
```

Validation already run:

```text
python3 -m py_compile submissions/quantizr/modal_q55_restart.py
```

## Decision Log

1. Keep Quantizr #55 as base. Candidate A and replacement representations are not competitive.

2. Keep local CPU as a relative filter. Do not assume CPU, Modal CUDA, RunPod CUDA, and GitHub/T4 are numerically interchangeable.

3. Keep fp16 model qpack plus int10 pose qpack as the current safe package.

4. Reject regenerated source-mask CRF paths until reproduction is fixed or a trainable adapter is introduced.

5. Reject blind archive-mask CRF recoding. CRF50 recoding barely saves bytes and severely damages PoseNet.

6. Do not run more uniform CRF54/56 or mixed-CRF from the current mask generation code. The CRF50 reproduction gate failed.

7. Do not spend more time on lossless class-mask storage. It is at least about `2x` larger than `mask.obu.br`.

8. Exact-mask polish is not enough. It produced a real but tiny improvement.

9. Exact-class AV1+residual repair is implemented and verifies exact class recovery, but CRF50 residual repair is currently not byte-competitive. Do not run a large residual sweep until either the predictor gets much better or the residual coding becomes substantially more efficient.

10. Deterministic class cleaning is rejected for qrecode50. Every tested cleaner either worsened the class tensor or exploded false edits.

11. Supervised learned class-denoising is rejected for qrecode50. The thresholded model cannot separate true CRF defects from normal mask boundaries/context; useful repair thresholds cause too many false edits.

12. Do not launch CRF52/54/56 learned class-denoiser training from this state. The required sanity gate was qrecode50 recovery, and it failed.

13. The tiny evaluator-level embedding adapter also failed qrecode50 recovery. It is not enough capacity or not the right intervention.

14. Pixel-space evaluator inversion is the active oracle. It improved the 64-sample projected score from `0.31530` to `0.30721` without changing payload bytes.

15. Residual-head distillation is not justified yet. The best direct pixel oracle quality is `0.115266`, still above the `0.108054` fixed-byte gate.

16. The next experiment should diagnose the hard tail with per-sample metrics and a targeted offset-56 oracle run, not launch another mask-codec or architecture sweep.

17. The offset-56 tail is confirmed and remains PoseNet-dominated. Mixed-objective pixel optimization improved SegNet but did not reduce tail PoseNet; best tail quality only moved `0.182435 -> 0.176692`.

18. Run one final PoseNet-only tail oracle. If samples 59, 60, and 62 do not move, stop fixed-mask adversarial-residual distillation.

19. PoseNet-only tail optimization failed. Bounded direct pixel-space optimization cannot repair samples 59/60/62, so do not build adversarial residual heads from this oracle.

## Recommended Next Work

### Immediate infrastructure still worth doing

Add persistent checkpointing to Modal before any serious training:

```text
Modal Volume for checkpoints and decoded caches
save checkpoint every N steps or every few minutes
resume automatically after preemption
return final archive.zip and metrics.json to local machine
```

Archive-return capture is already patched. Checkpoint/resume is still the main infrastructure gap.

### Candidate path A: fixed-mask pixel oracle tail diagnosis

This path has now been tested and should be stopped unless GPT Pro proposes a materially different oracle. The first direct-pixel oracle showed that evaluator-targeted frame optimization can reduce quality at fixed bytes, but the generic sweep did not clear the gate, the offset-56 tail stayed high, and PoseNet-only tail optimization did not move the hard samples.

Results:

```text
best broad 64-sample oracle:
  quality 0.123353 -> 0.115266
  projected score 0.315299 -> 0.307212

best offset-56 mixed-objective tail:
  quality 0.182435 -> 0.176692

best offset-56 PoseNet-only tail:
  quality 0.180971 -> 0.180426
```

Additional frame1-only PoseNet-control oracles were run on the hard samples `59`, `60`, and `62`, holding generated frame2 fixed because SegNet ignores frame1:

```text
baseline on samples 59/60/62:
  quality:       0.216640
  posenet_dist:  0.00215967
  projected:     0.408586

original frame1 replacement:
  quality:       3.806598
  posenet_dist:  1.396454
  result:        catastrophic mismatch with generated frame2

best low-res original-frame1 sidecar:
  lowres RGB 48x36
  quality:       4.702563
  posenet_dist:  2.146359
  result:        catastrophic mismatch

affine warp of generated frame1:
  quality:       0.204331
  posenet_dist:  0.00181304
  pose drop:     16.0%
  sample 60:     0.00248145 -> 0.00224059, about 9.7% drop

64x64 learned frame1 patch:
  quality:       0.205241
  posenet_dist:  0.00183762
  pose drop:     14.9%
  sample 60:     0.00248145 -> 0.00221296, about 10.8% drop
```

Artifacts:

```text
submissions/quantizr/q55_pose_control_oracle.py
submissions/quantizr/experiments/q55_restart/modal_returned/q55_pose_control_indices_59_60_62_orig_lowres_affine_av/metrics.json
submissions/quantizr/experiments/q55_restart/modal_returned/q55_pose_control_indices_59_60_62_patch64_av/metrics.json
```

Decision:

```text
do not build residual-head distillation from this oracle
do not add z_adv latents until a direct pixel oracle can show a reachable target
do not build original-frame1 sidecars, affine warps, or patch renderers
```

The path is not logically impossible, but the current output-space and frame1-control oracles cannot supply a target worth distilling. The hard tail is not fixed by adding original-frame1 information; that actually breaks PoseNet because frame1 and generated frame2 become mismatched. Small frame1 controls only produce `~15-16%` PoseNet reduction on the hard three samples, far short of the `>=40%` gate.

### Candidate path B: full qrecode50 warm-start sanity

This was run and failed. Full generator warm-start on the qrecode50 archive-mask stream did not recover the PoseNet failure:

```text
base: qrecode50 archive mask
model: full #55 generator warm-start
packaging: fp16 model qpack + int10 pose qpack
eval: local/Modal CPU relative path
result: proxy quality stayed around 0.3151 after the early gate
```

Decision: stop CRF52/54/56 compressed-mask recovery work for now. If qrecode50 cannot recover under full-generator warm-start, stronger mask compression is unlikely to recover under this architecture.

### Candidate path B2: pose-table inversion oracle

This tested the newly proposed byte-neutral control variable: optimize the stored 6D pose side-channel as the generator input while keeping the evaluator pose target fixed.

Implementation:

```text
submissions/quantizr/q55_pose_table_oracle.py
submissions/quantizr/modal_q55_restart.py stage q55-pose-table-oracle-tail-av
```

Setup:

```text
base archive: fp16 model qpack + int10 pose qpack
mask: exact #55 mask
generator: frozen #55/qpack generator
indices: 59,60,62
method: CEM + Adam
fake quant: int10 enabled
bounded scales: 0.5σ, 1.0σ, 2.0σ
```

Baseline on hard three:

```text
quality:       0.216640
posenet_dist:  0.00215967
projected:     0.408586
```

Results:

```text
scale 0.5σ:
  quality:       0.204437
  posenet_dist:  0.00181590
  pose drop:     15.92%

scale 1.0σ:
  quality:       0.204392
  posenet_dist:  0.00181467
  pose drop:     15.97%

scale 2.0σ:
  quality:       0.204341
  posenet_dist:  0.00181332
  pose drop:     16.04%

unbounded, initialized at 2.0σ CEM:
  quality:       0.204135
  posenet_dist:  0.00180776
  pose drop:     16.29%
```

Best per-sample result from the unbounded run:

```text
sample 59: 0.00238028 -> 0.00176142
sample 60: 0.00248145 -> 0.00223787
sample 62: 0.00161728 -> 0.00142398
```

Artifacts:

```text
submissions/quantizr/experiments/q55_restart/modal_returned/q55_pose_table_indices_59_60_62_cem_adam_s0p5_av/metrics.json
submissions/quantizr/experiments/q55_restart/modal_returned/q55_pose_table_indices_59_60_62_cem_adam_s1p0_av/metrics.json
submissions/quantizr/experiments/q55_restart/modal_returned/q55_pose_table_indices_59_60_62_cem_adam_s2p0_av/metrics.json
submissions/quantizr/experiments/q55_restart/modal_returned/q55_pose_table_indices_59_60_62_cem_adam_unbounded_s2p0_av/metrics.json
```

Decision:

```text
do not run offset 56-63
do not run 64-sample pose-table inversion
do not package optimized pose table
```

Reason: the hard-three continuation gate required at least `25%` average PoseNet drop. The best result was `16.29%`, and bounded/unbounded searches saturated at essentially the same level. This falsifies pose-table inversion as the missing high-leverage control variable under the current #55 generator.

### Candidate path C: exact-mask architecture sweep as fallback/bridge

Since mask compression routes are now repeatedly failing, a small exact-mask architecture/quality sweep is the safest fallback. The first padding-mode change (`reflect`) failed at zero-step with score `1.54`, so padding-mode changes should be treated as rejected unless there is a specific boundary-only implementation that preserves pretrained behavior. The fallback probably cannot solve `0.2x` alone, but it may improve first place and test Quantizr's conv-dim comment:

```text
same exact #55 mask stream
fp16 model qpack + int10 pose package
partial weight loading from #55 where possible
try at most 2-3 conv/hidden/cond variants
train with metric losses, not RGB reconstruction
abort if zero-step score is catastrophic
evaluate local CPU after zero-step and after each stage
```

Run at most 2-3 configs. Do not let this become an open-ended quality-only project unless the first config improves quality by at least `0.01`, not `0.0005`. All remaining architecture probes should use `--zero-score-max` so Modal exits after zero-step if partial weight loading or architecture drift destroys the pretrained generator.

### Candidate path D: predictor search only if it reduces hard class errors

The exact residual route is only alive if a predictor reduces hard class errors by at least `5x-10x`. Do not optimize the residual coder further until the predictor improves:

```text
current qrecode50 errors: 23,747
weak predictor target:    <=6,000 errors
strong predictor target:  <=2,000 errors
```

Only palette/AV1-setting changes that hit those class-error targets are worth pairing with exact residual repair.

### Candidate path E: mixed-int8 only as pack-aware late stage

Mixed-int8 saved bytes but hurt PoseNet too much:

```text
archive: 291,530 bytes raw mixed-int8
archive: 283,346 bytes after Modal mixed-int8 recovery + int10 pose
best recovered score: 0.357919
```

Do not use it now. Revisit only if a stronger pack-aware training loop exists.

## Questions For GPT Pro

1. The fixed-mask pixel oracle improved the easy/mid subset but failed on the PoseNet-heavy tail. Is there any reason to try an unbounded or patch-style output-space oracle after bounded `delta=16` PoseNet-only optimization failed, or should the fixed-mask adversarial-residual path be considered falsified?

2. Samples 59/60/62 remain high-PoseNet under direct pixel optimization and frame1-control oracles. Original-frame1 replacement and low-res original-frame1 sidecars are catastrophic; affine and patch controls only reduce PoseNet by ~15-16%. Is this more likely an evaluator target/preprocessing issue, a local optimum from bounded/low-dimensional controls, or evidence that #55's generated pair is already near the best reachable PoseNet basin for those samples?

3. If direct pixel optimization cannot repair the hard PoseNet tail, is there any plausible value in per-sample `z_adv16`/`z_adv32`, or would that just distill a non-existent oracle?

4. With mask reduction, qrecode recovery, exact residual repair, exact-mask polish, padding changes, fixed-mask pixel inversion, frame1 controls, and pose-table inversion all failing or insufficient, what is the next non-falsified Quantizr-family route to a visible `0.2x`? Is there one, or should the target be reset to first-place improvement?

5. Would a GitHub/T4 dry-run of the current safe package change any strategic decision, or is the gap now large enough that runner variance is not the blocking uncertainty?

## Current Recommendation

Continue from Quantizr #55, but change the emphasis:

```text
Stop blind mask recompression.
Keep fp16 model qpack + int10 pose as the safe byte baseline.
Keep the exact #55 mask.
Stop fixed-mask adversarial residual-head distillation for now.
The direct pixel oracle could not repair the PoseNet-heavy tail.
Frame1 original sidecars, affine warps, and localized patches did not repair it either.
Pose-table inversion also did not repair it; the best hard-three drop was only 16.29%.
Do not launch more Modal training until a new oracle-level hypothesis exists.
Use the report to choose the next non-falsified route with GPT Pro.
Do not create a contest PR or submit yet.
```

The fixed-payload quality path is now also blocked by the PoseNet tail. The best broad oracle reached `0.115266`, but the hard tail remains around `0.180-0.204` quality under pixel, frame1, patch, affine, and pose-table controls. The next step should be a strategic reset, not another small patch.
