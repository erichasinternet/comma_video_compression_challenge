# Task-Token VCM Findings

## Intent

This is a first-principles Video Coding for Machines path:

```text
per-sample task tokens
+ tiny two-frame decoder
+ frozen SegNet/PoseNet losses
-> evaluator-facing frames
```

It intentionally avoids Quantizr masks, selfcomp latent video, conventional video sidecars, and hand-designed SCV cartoons.

## Implemented

```text
train_capacity.py   continuous-token capacity oracle
train_vq.py         gated VQ placeholder
pack_tokens.py      prototype float-token archive packer
inflate.py          prototype renderer/inflater
inflate.sh          challenge-compatible inflate wrapper
modal_task_token.py Modal GPU runner
```

## Gates

Run order:

```text
1. CPU/GPU smoke
2. hard8 float capacity
3. 64-sample float capacity
4. VQ only if float capacity passes
```

Capacity gates:

```text
hard8:
  2k steps: quality <=0.120
  5k steps: quality <=0.090 and sample-60 PoseNet drops strongly

64-sample:
  weak pass:   quality <=0.120
  strong pass: quality <=0.100
```

Budget gates after VQ:

```text
B220: archive <=220 KB and quality <=0.153
B200: archive <=200 KB and quality <=0.167
B180: archive <=180 KB and quality <=0.180
```

## Current Status

Smoke tests passed locally and on Modal. The hard8 capacity gate has not passed.

## Hard8 Results

All rows use samples:

```text
59, 60, 62, 56, 57, 58, 61, 63
```

| run | best step | quality | SegNet term | PoseNet term | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `hard8_float_s500_b4` | 200 | 35.0144 | 2.0100 | 33.0044 | fails; SegNet learns, PoseNet remains huge |
| `hard8_pair_float_s500_b4` | 50 | 30.2522 | 3.9384 | 26.3138 | fails; explicit per-frame tokens improve PoseNet but not nearly enough |
| `hard8_direct_rgb_s500_b4` | 500 | 36.5072 | 0.6748 | 35.8324 | fails; direct low-res RGB tokens learn SegNet, not PoseNet |
| `hard8_direct_originit_s0_b4` | 0 | 7.7355 | 3.4781 | 4.2574 | fails; even original-initialized 48x64 frame tokens are far from the gate |

The hard8 gate was:

```text
2k steps: quality <=0.120
5k steps: quality <=0.090 and sample-60 PoseNet drops strongly
```

The current task-token family is multiple orders of magnitude off that gate. Do not run VQ, budget ladders, or 64-sample scaling from these checkpoints.

## Interpretation

The learned CNN token decoder can rapidly generate SegNet-friendly stimuli, but the generated frame pairs stay far outside PoseNet's temporal basin. Adding explicit per-frame spatial tokens helps PoseNet somewhat, which confirms the first decoder was under-controlled, but the improvement is still not close to useful.

The direct low-resolution RGB-token upper bound is more damaging: random low-res tokens still do not optimize PoseNet, and original-initialized 48x64 frame tokens only reach quality `7.7355`. That means a byte-realistic low-res frame-token stream is not a credible `0.2x` route.

## Decision

Stop the current task-token VCM path before VQ/rate work unless a materially different PoseNet control representation is introduced. The existing capacity oracles did not prove the required evaluator manifold is reachable.
