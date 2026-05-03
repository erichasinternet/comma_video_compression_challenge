# Task-NeRV VCM Findings

## Purpose

This branch tests a task-aware HNeRV/NeRV representation:

```text
frame id -> learned frame embedding -> compact upsampling renderer -> RGB frames
```

The goal is to determine whether a neural video representation can enter the PoseNet/SegNet evaluator manifold without a fragile mask, conventional video bitstream, or task-token decoder.

## Current Implementation

Files:

```text
submissions/task_nerv_vcm/model.py
submissions/task_nerv_vcm/train_nerv.py
submissions/task_nerv_vcm/pack_nerv.py
submissions/task_nerv_vcm/inflate.py
submissions/task_nerv_vcm/inflate.sh
submissions/task_nerv_vcm/modal_task_nerv.py
```

Training stages:

```text
Stage 1: RGB basin pretraining against original camera-domain frames.
Stage 2: frozen SegNet/PoseNet task fine-tuning.
```

The first implementation is capacity-first. The packed payload is only a Brotli-compressed PyTorch checkpoint; compression-aware pruning/codebooks are intentionally downstream of the hard8 and 64-sample gates.

## Decision Table

Hard8 continue:

```text
quality <=0.120
strong continue <=0.090
```

64-sample capacity continue:

```text
weak pass   <=0.180
real pass   <=0.150
strong pass <=0.130
```

Compressed 64-sample continue:

```text
projected archive <=240 KB and quality <=0.140
projected archive <=220 KB and quality <=0.153
projected archive <=200 KB and quality <=0.167
```

## First Run

Launch hard8 capacity:

```bash
./.venv/bin/python -m modal run submissions/task_nerv_vcm/modal_task_nerv.py \
  --stage hard8 \
  --rgb-steps 2000 \
  --task-steps 5000 \
  --batch-size 2 \
  --eval-every 100 \
  --hidden 128 \
  --embed-dim 64
```

Stop this branch if hard8 cannot get below `0.120` quality. Only run 64-sample capacity if hard8 passes.

## Hard8 Result

Run:

```bash
./.venv/bin/python -m modal run submissions/task_nerv_vcm/modal_task_nerv.py \
  --stage hard8 \
  --rgb-steps 2000 \
  --task-steps 5000 \
  --batch-size 2 \
  --eval-every 100 \
  --hidden 128 \
  --embed-dim 64
```

The run was stopped after the task-stage `2k` gate failed. Relevant observed checkpoints:

```text
initial:
  quality  ~92.7243
  seg term ~51.0445
  pose term ~41.6798

RGB stage:
  step 1200 quality ~11.0752, seg term ~4.9383, pose term ~6.1369
  RGB pretraining improved the frames, but did not enter the required PoseNet basin.

task stage:
  step 700  quality ~33.8093, seg term ~0.3381, pose term ~33.4712
  step 900  quality ~36.2549, seg term ~0.3064, pose term ~35.9485
  step 1800 quality ~33.1107, seg term ~0.2314, pose term ~32.8793
  step 2100 quality ~32.8653, seg term ~0.2109, pose term ~32.6545
```

Interpretation:

```text
SegNet is learnable from the NeRV output.
PoseNet remains catastrophically outside the temporal manifold.
The hard8 quality gate was <=0.120; the observed task-stage quality at the 2k checkpoint was ~32.9.
```

Decision:

```text
Do not run subset64 capacity.
Do not run compression-aware NeRV packing.
Stop task_nerv_vcm as a visible-0.2x route.
```
