# Selfcomp #56 Baseline Audit

## Source

- Branch imported from upstream PR #56 into `submissions/selfcomp`.
- Archive attachment downloaded from GitHub user attachments:
  `https://github.com/user-attachments/files/27106535/archive.zip`
- Local archive path:
  `submissions/selfcomp/experiments/source/selfcomp_pr56_archive.zip`
- Archive SHA256:
  `241da6aa0a82fb01bf2a287d7d6561040342733689d2d9ef1f1b2db939eee0e4`

## PR-Reported Metrics

```text
archive bytes: 279,036
PoseNet:       0.00055221
SegNet:        0.00122167
score:         0.38
```

Score decomposition:

```text
rate term: 25 * 279036 / 37545489 = 0.18580
pose term: sqrt(10 * 0.00055221)  = 0.07431
seg term:  100 * 0.00122167      = 0.12217
total:                                0.38228
```

## Payload Breakdown

`archive.zip` contains one file:

```text
payload.tar.xz: 278,820 bytes inside zip
archive.zip:    279,036 bytes total
```

`payload.tar.xz` expands to:

```text
0.mkv:               206,573 bytes
segmap_inference.pt: 1,239,122 bytes
```

Independent xz baselines:

```text
0.mkv.xz:                  204,092 bytes
video_only.tar.xz:         204,200 bytes
segmap_inference.pt.xz:     73,988 bytes
model_only.tar.xz:          74,676 bytes
combined payload.tar.xz:   278,820 bytes
```

Interpretation: selfcomp is primarily video-rate limited already. The model is large raw but highly compressible; under xz it contributes roughly 74 KB.

## Model Configuration

Checkpoint format:

```text
format:              segmap_inference_integer_aux8_v1
hidden:              64
block_hidden:        128
num_blocks:          8
num_fourier_bands:   6
max_frame_index:     1200
latent_input_scale:  0.1
weight layout:       HWOI
aux tensor codec:    linear_minmax_uint8_v1
```

Renderer summary:

```text
grayscale AV1 side video -> Gaussian soft 5-class LUT
+ shared latent canvas with per-frame affine embedding
+ 8 residual conv blocks
-> RGB frame pair output
```

The side video is:

```text
codec:      AV1
container:  Matroska
pix_fmt:    gray
size:       512x384
fps:        20
duration:   30 seconds
```

## Local Reproduction

CPU evaluation was run in:

```text
submissions/selfcomp_pr56_eval
```

Result:

```text
PoseNet:       0.00039916
SegNet:        0.00115278
archive bytes: 279,036
quality:       0.17845711
rate term:     0.18579862
score:         0.36425573
```

Sub-0.300 requirement at this byte size:

```text
required quality: 0.11420138
quality gap:      0.06425573
```

The local CPU path is better than the PR-reported CUDA score, mostly because PoseNet is lower locally. The strategic conclusion is unchanged: PoseNet is already strong, and SegNet is the dominant repair target.

Practical note: local CPU inflation took the bulk of runtime because it rendered 1200 full-resolution raw frames through the conv renderer. The next harness should support 64-sample/chunked evaluation without writing a full 3.4 GB raw output for every early candidate.

## Current Read

Selfcomp has the right failure shape for a follow-up:

```text
good:  lower archive size than Quantizr #55
good:  PoseNet already close to Quantizr
bad:   SegNet is much worse than Quantizr
```

The next useful experiment is fixed-byte SegNet repair at the existing payload size before any rate reduction. If SegNet cannot move down sharply without PoseNet damage, selfcomp++ should stop early.
