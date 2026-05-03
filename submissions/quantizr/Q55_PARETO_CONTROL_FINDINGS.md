# Q55 Pareto Control Lab Findings

## Scope

Implemented a constrained Pareto control audit for the safe Quantizr/qpack base:

```text
base archive: submissions/q55_fp16_pose_int10/archive.zip
archive bytes: 288,268
```

The lab tests low-byte controls against actual evaluator terms and includes an estimated byte penalty in the accept/reject gate.

## Implemented

```text
submissions/quantizr/pareto_control_lab.py
submissions/quantizr/pareto_actions.py
submissions/quantizr/pareto_pack.py
submissions/quantizr/pareto_report.py
```

Implemented controls:

```text
C1 frame2 class-conditioned color/LUT correction
C2 frame2 low-frequency residual grid
C3 paired frame1/frame2 low-frequency residual grid
C4 pose compensator scaffold
C5 conservative action-router/pack accounting
```

The optimizer computes separate SegNet/PoseNet/regularizer gradients, applies PCGrad-style projection when gradients conflict, and then accepts candidates only if actual measured score improves under the pose cap and byte penalty.

## 64-sample audit

Command output:

```text
submissions/quantizr/experiments/pareto_control_lab_64/channel_audit/metrics.json
submissions/quantizr/experiments/pareto_control_lab_64/PARETO_CONTROL_REPORT.md
submissions/quantizr/experiments/pareto_control_lab_64/pack_plan.json
submissions/quantizr/experiments/pareto_control_lab_64/full_compose/metrics.json
```

Subset:

```text
samples 0..63
includes hard tail samples 58, 59, 60, 61, 62, 63
```

Baseline on this 64-sample audit:

```text
quality:      0.17367025
SegNet term:  0.06534259
PoseNet term: 0.10832766
rate term:    0.19194583
```

Control results:

| control | estimated bytes | accepted steps | decision |
| --- | ---: | ---: | --- |
| C1 class color/LUT | 79 | 0 | reject |
| C2 frame2 grid | 18,560 | 0 | reject |
| C3 paired grid | 36,992 | 0 | reject |

Pack plan:

```text
decision: do_not_pack
```

Full compose:

```text
decision: stop_no_validated_controls
positive_controls: []
```

## Full 600 base ledger

The all-at-once full-ledger run was killed by memory pressure, so `base_metrics` now supports streaming target collection:

```text
--stream-chunk-size 64
```

Streaming output:

```text
submissions/quantizr/experiments/pareto_control_lab/base_metrics/base_per_sample.jsonl
submissions/quantizr/experiments/pareto_control_lab/base_metrics/metrics.json
```

Full-set proxy metrics from the Pareto lab:

```text
samples:      600
quality:      0.19726397
SegNet term:  0.07209439
PoseNet term: 0.12516958
score:        0.38920979
```

This ledger is a model-size proxy used by the control lab, not a replacement for the official raw-frame local CPU score. It is useful for hard-tail ranking; the known official-style local score for `q55_fp16_pose_int10` remains the calibration reference.

Top hard-tail samples by this ledger begin with:

```text
514, 270, 271, 258, 365, 62, 58, 65, 430, 464
```

## Decision

The Pareto composer hit the first stop rule:

```text
No individual channel has positive net score on 64 samples.
```

Do not run bundle routing or full-600 composition from these C1/C2/C3 controls. At the tested byte scales, C2/C3 are already too expensive for 64 samples, and none produced an actual-score-improving accepted point under the pose cap.

## Notes

PoseNet preprocessing uses a `torch.no_grad()` RGB-to-YUV conversion in the official modules, so PoseNet gradients are not available through the standard evaluator path. The lab handles this by treating missing PoseNet gradients as zero-gradient and enforcing PoseNet with hard measured accept/reject instead of relying on soft pose gradients.
