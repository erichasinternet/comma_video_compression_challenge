# Semantic Cartoon Video Codec Prototype

This directory is an isolated SCV prototype for the comma video compression challenge. It deliberately avoids the prior Quantizr restart code paths: no mask recompression, no pose-table inversion, no qrecode recovery, and no contest submission plumbing beyond a local `inflate.sh`.

## What It Tests

SCV renders a low-entropy synthetic driving video from evaluator-visible semantic information, then encodes that video directly with commodity video codecs.

- `scv0`: flat five-class semantic palette, frame1 copied from frame2.
- `scv1`: semantic palette estimated from original frames, low-frequency texture, and boundary treatment, frame1 copied from frame2.
- `scvt`: camera-like stylized original frames using low-pass blur, semantic mask-guided smoothing, edge-preserving blend, and RGB quantization.
- `scv2`: intentionally not implemented until SCV-1 clears the gate.

## Commands

Run the 64-sample prototype:

```bash
bash submissions/scv_cartoon/compress.sh
```

Render one variant:

```bash
python submissions/scv_cartoon/scv_cartoon.py render \
  --subset 64 \
  --variant scvt \
  --device cpu \
  --out submissions/scv_cartoon/experiments/scvt_64
```

Encode and evaluate a codec grid:

```bash
python submissions/scv_cartoon/scv_cartoon.py encode \
  --frames submissions/scv_cartoon/experiments/scvt_64/frames \
  --out submissions/scv_cartoon/experiments/scvt_64/archive.zip \
  --subset 64 \
  --quick \
  --evaluate \
  --device cpu
```

Inflate an archive:

```bash
unzip -o submissions/scv_cartoon/experiments/scv1_64/archive.zip -d /tmp/scv_archive
bash submissions/scv_cartoon/inflate.sh /tmp/scv_archive /tmp/scv_inflated public_test_video_names.txt
```

## Gate

The prototype should stop before SCV-2 unless SCV-1 reaches:

- 64-sample SegNet term `<= 0.08`
- extrapolated 600-sample archive `<= 240 KB`

Every metrics file includes the official score decomposition:

```text
quality = 100*segnet_dist + sqrt(10*posenet_dist)
score = quality + 25*(archive_bytes/37_545_489)
```
