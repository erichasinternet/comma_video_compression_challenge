# SCV Findings

## Summary

The 64-sample Semantic Cartoon Video Codec prototype is implemented through SCV-T. SCV-0/1 prove that mask-colored cartoons are not recognized well enough by SegNet/PoseNet. SCV-T confirms that camera-like stylized original frames are much better semantically, but the byte/quality tradeoff is still far from a candidate.

Decision:

```text
Do not scale to 600.
Do not implement SCV-2 yet.
Stop this SCV branch unless a materially stronger camera-like renderer is introduced.
```

## Artifacts

```text
submissions/scv_cartoon/experiments/scv0_64/metrics.json
submissions/scv_cartoon/experiments/scv0_64/encode_grid_metrics.json
submissions/scv_cartoon/experiments/scv0_64/preview_grid.jpg
submissions/scv_cartoon/experiments/scv1_64/metrics.json
submissions/scv_cartoon/experiments/scv1_64/encode_grid_metrics.json
submissions/scv_cartoon/experiments/scv1_64/preview_grid.jpg
submissions/scv_cartoon/experiments/scv1_64/archive.zip
submissions/scv_cartoon/experiments/scvt_64/metrics.json
submissions/scv_cartoon/experiments/scvt_64/encode_grid_metrics.json
submissions/scv_cartoon/experiments/scvt_64/preview_grid.jpg
submissions/scv_cartoon/experiments/scvt_64_hi/encode_grid_metrics.json
submissions/scv_cartoon/experiments/scvt_64_hi63/encode_grid_metrics.json
```

`inflate.sh` was smoke-tested against `scv1_64/archive.zip` and produced the expected raw output file.

## SCV-0

SCV-0 renders a flat semantic palette and copies frame2 to frame1.

Uncompressed 64-sample metrics:

```text
segnet_dist:  0.13638758
posenet_dist: 186.98481560
quality:      56.88049928
```

Best decoded codec result by projected score:

```text
codec:                    x265
crf:                      44
extrapolated_600_bytes:   494,512
decoded_quality:          28.74291868
projected_score:          29.07219393
```

Result: failed both quality and byte gates.

## SCV-1

SCV-1 uses a class palette estimated from original frames, low-frequency frame2 texture, and boundary darkening. It still copies frame2 to frame1.

Uncompressed 64-sample metrics:

```text
segnet_dist:  0.00600370
segnet_term:  0.60036977
posenet_dist: 187.58209038
quality:      43.91111790
```

SCV-1 gate:

```text
required segnet_term <= 0.08
actual segnet_term   = 0.60037
gate result          = fail
```

Best decoded codec result by projected score:

```text
codec:                    x265
crf:                      36
extrapolated_600_bytes:   321,759
decoded_quality:          27.91818623
projected_score:          28.13243234
```

Best byte result:

```text
codec:                    svtav1
crf:                      55
extrapolated_600_bytes:   185,334
decoded_quality:          43.19377645
projected_score:          43.31718275
```

Result: byte-only gate can pass, but evaluator quality is orders of magnitude off target. The PoseNet term dominates because the initial cartoon variants do not encode coherent frame-to-frame motion.

## SCV-T

SCV-T stylizes the original camera frames rather than rendering mask-colored cartoons. It applies low-pass resize blur, semantic mask-guided smoothing, edge-preserving detail blend, and RGB quantization to both frame1 and frame2.

Uncompressed 64-sample metrics:

```text
segnet_dist:  0.00816035
segnet_term:  0.81603527
posenet_dist: 0.36784112
quality:      2.73395373
```

Best decoded codec result by projected score from the quick grid:

```text
codec:                    vp9
crf:                      45
extrapolated_600_bytes:   1,453,228
decoded_quality:          4.49144969
projected_score:          5.45909457
```

Best weak-byte result:

```text
codec:                    svtav1
crf:                      60
extrapolated_600_bytes:   228,056
decoded_quality:          5.69535766
projected_score:          5.84721079
```

Lowest-byte probe:

```text
codec:                    svtav1
crf:                      63
extrapolated_600_bytes:   134,494
decoded_quality:          9.41602953
projected_score:          9.50558356
```

Result: SCV-T is the right direction relative to SCV-0/1 because it remains camera-like and preserves more semantic signal, but it does not create a credible `0.2x` path. The quality gap is still more than an order of magnitude too large at competitive archive sizes.

## Recommendation

Do not proceed to SCV-2 under the current renderer. The original SCV-2 prerequisite was:

```text
SCV-1 SegNet term <= 0.08
SCV-1 extrapolated archive <= 240 KB
```

SCV-1 only satisfies the byte side for the most aggressive SVT-AV1 setting and misses the SegNet gate by about `7.5x`; SCV-T improves visual plausibility and PoseNet dramatically, but still has uncompressed quality `2.73` and codec-roundtrip quality `5.70` near the weak byte gate.

If this branch is revisited, the next experiment should not be pose-flow. It would need a stronger camera-like renderer that gets uncompressed SCV-T quality below roughly `0.3` before codec tuning or motion parameterization is worth funding.
