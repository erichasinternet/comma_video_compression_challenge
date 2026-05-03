# Quantizr Contest Handoff For GPT Pro

## Objective

We forked Quantizr PR #55 and tried the simplest high-leverage extension for the comma video compression challenge:

- keep the metric-distillation inflater approach
- add a tiny per-pair latent code
- select candidates by real official score
- sweep CRF and a small architecture neighborhood

This document captures the exact code changes, run history, measured results, failure modes, and the decision we need next.

## Repo And Branch

- Repo: `/Users/eric/comma_video_compression_challenge`
- Active branch: `codex/quantizr-latent-sweep`
- RunPod workspace path: `/workspace/comma_video_compression_challenge`
- Snapshot time: `2026-04-22 19:12:20 MDT`

## Code Changes Already Implemented

These are already on the branch and pushed:

### 1. Quantizr latent fork

Main files:

- `/Users/eric/comma_video_compression_challenge/submissions/quantizr/compress.py`
- `/Users/eric/comma_video_compression_challenge/submissions/quantizr/inflate.py`
- `/Users/eric/comma_video_compression_challenge/submissions/quantizr/sweep_official.py`

Implemented changes:

- added per-pair latent code support (`z_dim`)
- conditioned both heads on pose plus latent
- stored latent side-channel as `z.npz.br`
- exposed sweep knobs for `c1`, `c2`, `hidden`, `cond_dim`, `z_dim`, `crf`
- packaging uses only actual payload artifacts

### 2. Official-score sweep path

Sweep driver uses:

- `pipeline-preset fast`
- `selection-metric proxy`
- real official score only at candidate end

### 3. Shared caches

To avoid repeated preprocessing across candidates:

- shared RGB cache
- shared pose cache
- shared per-CRF mask cache

### 4. AV decode instead of DALI for sweep cache

This was necessary because the earlier DALI path caused large non-PyTorch GPU allocations and OOMs.

### 5. Eval recovery / CPU fallback

Implemented:

- official eval defaults to CPU in the sweep path
- GPU eval falls back to CPU if it fails
- if a candidate fails after training but before final export/eval completes, the run can recover from the saved proxy-best checkpoint rather than retraining from scratch

## Run Infrastructure

GPU box used:

- RunPod
- RTX A5000

Important environment facts:

- training was done on CUDA
- official sweep-time evaluation was forced to CPU after earlier GPU eval OOMs
- full sweep was later intentionally stopped to avoid further spend while reassessing

## Search Plan We Actually Ran

Initial shortlist:

- `crf in {50,54,58}`
- `c1 in {48,56}`
- `c2 = c1 + 8`
- `hidden = 48`
- `cond_dim = 40`
- `z_dim in {0,8}`
- `batch_size = 4`

Then we narrowed after an OOM to only the viable width:

- `c1 = 48`
- `c2 = 56`
- `crf in {50,54,58}`
- `z_dim in {0,8}`

## Finished Results

Saved locally in:

- `/Users/eric/comma_video_compression_challenge/runpod_snapshots/2026-04-22_shortlist_a5000_001/results.jsonl`

Completed candidates:

1. `crf50_c148_c256_h48_cond40_z0`
   - score: `0.8233766585423492`
   - segnet_dist: `0.00102431`
   - posenet_dist: `0.02823848`
   - rate: `0.00758187`
   - archive_bytes: `284665`

2. `crf50_c148_c256_h48_cond40_z8`
   - score: `0.8122018277350104`
   - segnet_dist: `0.00097656`
   - posenet_dist: `0.02697271`
   - rate: `0.00780773`
   - archive_bytes: `293145`

3. `crf54_c148_c256_h48_cond40_z0`
   - score: `0.7960072417880449`
   - segnet_dist: `0.00166985`
   - posenet_dist: `0.02141283`
   - rate: `0.00665129`
   - archive_bytes: `249726`

### Observed deltas

Versus `crf50 z0` baseline:

- `crf50 z8`
  - score delta: `-0.011175`
  - bytes delta: `+8480`
  - seg delta: `-0.00004775`
  - pose delta: `-0.00126577`
  - rate delta: `+0.00022586`

- `crf54 z0`
  - score delta: `-0.027369`
  - bytes delta: `-34939`
  - seg delta: `+0.00064554`
  - pose delta: `-0.00682565`
  - rate delta: `-0.00093058`

### Current interpretation

- `z=8` helps a little at `crf50`
- moving from `crf50` to `crf54` helped more than latent did
- the branch is improving, but the absolute score is still nowhere near competitive
- target zone remains roughly around `0.33`

## Unfinished / Interrupted Candidates

### 1. `crf50_c156_c264_h48_cond40_z0`

Status:

- failed
- no `final_metrics.json`

Failure:

- training CUDA OOM during `run1_anchor`
- width `c1=56, c2=64` is not viable on this A5000 at `batch_size=4`

Important excerpt:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 60.00 MiB.
GPU 0 has a total capacity of 23.56 GiB of which 3.25 MiB is free.
Including non-PyTorch memory, this process has 23.55 GiB memory in use.
```

### 2. `crf54_c148_c256_h48_cond40_z8`

Status:

- training finished
- official CPU eval was in progress
- manually terminated when we paused the pod run
- no final score saved

Last known eval progress:

```text
1it [00:23, 23.97s/it]
2it [00:45, 22.24s/it]
3it [01:07, 22.44s/it]
4it [01:30, 22.62s/it]
5it [01:53, 22.77s/it]
6it [02:15, 22.46s/it]
7it [02:38, 22.77s/it]
8it [03:01, 22.66s/it]
9it [03:23, 22.66s/it]
10it [03:45, 22.31s/it]
11it [04:05, 21.48s/it]
12it [04:24, 20.90s/it]Terminated
```

## Important Operational History

These failures already happened and were patched around:

1. DALI-based preload consumed too much GPU memory
   - fixed by switching sweep cache preload to PyAV on CPU

2. official GPU eval OOMed
   - fixed by making sweep-time eval use CPU
   - also added GPU-to-CPU fallback

3. post-training failures wasted work
   - fixed by allowing recovery from saved proxy-best checkpoints

4. repeated preprocessing made shortlist sweeps too slow
   - fixed by adding shared caches

These fixes improved stability, but not competitiveness.

## Current Strategic Read

The direct interpretation of the current evidence is:

- this exact neighborhood is not currently on a winning trajectory
- the improvements are real but small
- the best finished score, `0.7960`, is still massively worse than leaderboard-leading territory

What the evidence says:

- latent helps some
- stronger mask compression helps more than latent in the tested range
- the width increase to `56/64` is operationally expensive and currently unstable at batch size 4
- the "simple latent plus CRF sweep" idea has not come close enough to justify brute-forcing more of the same without a better hypothesis

## Local Snapshot Files

Saved in this repo:

- `/Users/eric/comma_video_compression_challenge/runpod_snapshots/2026-04-22_shortlist_a5000_001/results.jsonl`
- `/Users/eric/comma_video_compression_challenge/runpod_snapshots/2026-04-22_shortlist_a5000_001/log_excerpts.txt`

## Questions For GPT Pro

Please reason from the actual measured results and the code modifications already in this branch. I want a concrete next-step plan, not generic ideas.

Questions:

1. Given these results, is the current Quantizr-latent direction fundamentally too weak to win, or is there still a narrow, plausible continuation path?

2. If this direction is still viable, what is the single highest-leverage next experiment?
   - Be specific about architecture, conditioning, side-channel, training recipe, or evaluator exploitation.

3. If this direction is not viable, what should replace it?
   - Prefer options that reuse as much of the current branch as possible.

4. How should we think about the bad-but-improving pattern here?
   - Is this just a local neighborhood that needs a sharper sweep?
   - Or is the score scale telling us we are fundamentally missing the real exploit?

5. If you had only one more serious GPU run, what exact candidate set would you launch next?
   - Give explicit hyperparameters and stop rules.

6. Should we finish `crf54_c148_c256_h48_cond40_z8` and possibly `crf58` on this branch, or is that likely wasted compute?

7. Are there stronger evaluator-targeting modifications that remain "simple" relative to rewriting the entire method?
   - Examples: learned mask encoding, different latent placement, frame-2-only specialization, direct SegNet-targeted residual side-channel, harder pose quantization, etc.

## My Current Recommendation Before GPT Pro Weighs In

If forced to choose right now, I would *not* spend another long run just extending this same narrow sweep. I would first ask whether the current method is missing the actual exploit entirely, because the absolute scores are too far from the target for this to look like normal hyperparameter cleanup.
