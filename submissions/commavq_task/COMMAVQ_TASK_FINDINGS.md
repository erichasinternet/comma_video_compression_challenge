# commaVQ Task Renderer Findings

## Intent

Use pretrained driving-video commaVQ tokens as the compact camera-domain representation, then train a small renderer directly for the frozen SegNet/PoseNet evaluator.

This is deliberately separate from:

```text
Quantizr masks
selfcomp latent video
random task-token VCM
SCV cartoons
original-frame sidecars
```

## Implemented

```text
encode_tokens.py          offline commaVQ encoder path
eval_commavq_decoder.py   real commaVQ decoder oracle
train_renderer.py         token -> evaluator-frame renderer
pack_tokens.py            uint10 token packing + Brotli archive
pack_renderer.py          renderer checkpoint Brotli packer
inflate.py / inflate.sh   prototype renderer inflate path
modal_commavq_task.py     Modal GPU runner
```

## Gates

```text
real commaVQ decoder:
  weak pass:   quality <=0.180
  strong pass: quality <=0.150

hard8 renderer:
  after RGB anchoring: quality <=2.0
  after 2k task steps: quality <=0.500
  after 5k task steps: quality <=0.250
  strong:              quality <=0.150

64-sample renderer:
  weak pass:   quality <=0.180
  real pass:   quality <=0.150
  strong pass: quality <=0.130
```

## Current Status

Scaffold is implemented. Hard8 commaVQ tokenization succeeded. The stock commaVQ decoder failed, and the follow-up overpowered hard8 task-renderer capacity oracle also failed because PoseNet stayed far off-manifold.

## Hard8 Results

Samples:

```text
59, 60, 62, 56, 57, 58, 61, 63
```

Tokenization:

```text
tokens shape:     8 x 2 x 128
raw 10-bit bytes: 2,560
token range:      1..1023
```

Real commaVQ decoder oracle:

```text
placement:        inverse_crop
fill:             0
SegNet term:      14.8053
PoseNet term:      1.8638
quality:          16.6691
```

The initial stretch-based oracle was worse:

```text
SegNet term:      31.6994
PoseNet term:      4.8396
quality:          36.5389
```

The corrected inverse-crop placement is the appropriate geometry for commaVQ's crop, but it is still far above the `quality <=0.300` hard-stop threshold.

Overpowered hard8 task renderer:

```text
run:              renderer_hard8_big_rgb2000_task2000_b2
architecture:     1024-token embedding x 64, hidden 128, 6 upsampling blocks, separate heads
RGB anchor:       2,000 steps to original frames
task optimize:    2,000 steps with SegNet CE/KL/margin + Seg/Pose features + Pose MSE
batch size:       2
```

RGB-anchor result:

```text
step:             2,000
SegNet term:      50.7949
PoseNet term:     37.6645
quality:          88.4594
```

Best task-stage result:

```text
step:             4,000
task step:        2,000
SegNet term:       0.6173
PoseNet term:     35.4902
quality:          36.1075
SegNet dist:       0.00617282
PoseNet dist:    125.95527840
```

The task renderer learned a strong SegNet-facing frame2 stimulus from commaVQ tokens, but it did not recover PoseNet-compatible frame-pair dynamics. The failure is not ambiguous: the required 2k-task gate was `quality <=0.500`, and the observed quality was `36.1075`.

## Decision

Stop the commaVQ-token task-renderer path before VQ/rate work or 64-sample scaling.

The corrected interpretation is:

```text
Stock commaVQ decoder:
  not evaluator-compatible.

Overpowered token-to-task renderer:
  can drive SegNet much lower,
  but cannot enter the hard8 PoseNet manifold after RGB anchoring + 2k task steps.
```

So the branch is closed for the current `0.2x` objective unless a materially different PoseNet-specific representation is introduced. More token packing, renderer shrinking, or 64-sample scaling would be premature.
