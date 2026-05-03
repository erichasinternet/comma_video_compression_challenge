# Q55 Hard-Sample Sidecar Findings

## Summary

The selective hard-sample enhancement idea was tested with an exact perfect-replacement oracle and two actual encoded original-pair sidecar probes.

Decision:

```text
Do not build a full hard-sample original-pair sidecar submission.
Do not run a broad sidecar codec sweep.
```

The perfect oracle only creates positive sidecar budget after replacing a very large tail. Once the sidecar is compressed enough to fit that budget, the replacement frames are too damaged and make PoseNet far worse than the Quantizr base.

## Implemented

```text
submissions/quantizr/q55_hard_sample_oracle.py
submissions/quantizr/q55_hard_sidecar_pack.py
```

Outputs:

```text
submissions/quantizr/experiments/q55_hard_sample_oracle/per_sample_quality.jsonl
submissions/quantizr/experiments/q55_hard_sample_oracle/topk_curve.json
submissions/quantizr/experiments/q55_hard_sample_oracle_extended/topk_curve.json
submissions/quantizr/experiments/q55_hard_sidecar_k192_256x192_svt63_eval/metrics.json
submissions/quantizr/experiments/q55_hard_sidecar_k256_256x192_svt60_eval/metrics.json
```

## Base

Base package:

```text
archive: submissions/q55_fp16_pose_int10/archive.zip
bytes:   288,268
quality: 0.15292687
score:   0.34487270
```

This matches the known local CPU safe qpack result.

## Perfect Replacement Oracle

Oracle assumption:

```text
Replace both frames of the selected top-K hard samples with original frames.
Selected samples get zero PoseNet and SegNet distortion.
No sidecar bytes counted yet.
```

Requested K curve:

| K | quality | score before sidecar | allowed sidecar bytes |
| -: | ------: | -------------------: | --------------------: |
| 4 | 0.151098 | 0.343044 | -64,645 |
| 8 | 0.149543 | 0.341489 | -62,310 |
| 16 | 0.146640 | 0.338586 | -57,950 |
| 32 | 0.141248 | 0.333194 | -49,852 |
| 48 | 0.136080 | 0.328026 | -42,090 |
| 64 | 0.131136 | 0.323082 | -34,665 |
| 96 | 0.121963 | 0.313909 | -20,890 |
| 128 | 0.113130 | 0.305076 | -7,623 |

Result for requested K values:

```text
No K up to 128 can hit 0.300 even with perfect replacement and zero sidecar bytes.
```

Extended K curve:

| K | quality | score before sidecar | allowed sidecar bytes |
| -: | ------: | -------------------: | --------------------: |
| 160 | 0.104752 | 0.296698 | 4,959 |
| 192 | 0.096561 | 0.288507 | 17,260 |
| 256 | 0.080593 | 0.272539 | 41,241 |
| 320 | 0.065332 | 0.257278 | 64,160 |
| 384 | 0.050386 | 0.242332 | 86,607 |
| 448 | 0.035980 | 0.227926 | 108,242 |
| 512 | 0.021903 | 0.213848 | 129,384 |
| 600 | 0.000000 | 0.191946 | 162,277 |

Interpretation:

```text
The branch only becomes byte-positive after replacing a very large number of samples.
That means the sidecar must encode hundreds of original frame pairs in only tens of KB.
```

## Actual Sidecar Probes

### K=192, 256x192, SVT-AV1 CRF63

Perfect budget:

```text
allowed sidecar bytes: 17,260
```

Actual sidecar:

```text
sidecar zip bytes: 17,609
total archive estimate: 305,877
```

Mixed decoded evaluation:

```text
segnet_dist:  0.01653858
posenet_dist: 46.30163089
quality:      23.17167152
score:        23.37534246
```

Result:

```text
Fails. It is close on bytes but catastrophically worse than the base.
```

### K=256, 256x192, SVT-AV1 CRF60

Perfect budget:

```text
allowed sidecar bytes: 41,241
```

Actual sidecar:

```text
sidecar zip bytes: 32,032
total archive estimate: 320,300
```

Mixed decoded evaluation:

```text
segnet_dist:  0.01188816
posenet_dist: 41.02712536
quality:      21.44396941
score:        21.65724403
```

Result:

```text
Fails. It fits the byte budget but the compressed low-resolution original pairs destroy PoseNet.
```

## Conclusion

The hard-sample sidecar upper bound is informative but not actionable for full-frame original-pair rescue.

```text
For small K, perfect replacement cannot hit 0.300.
For large K, the allowed sidecar budget is too small for camera-domain original pairs.
Compressed low-resolution replacements are worse than the q55 base output.
```

If this family is revisited, it should not start with full-frame original-pair sidecars. The only plausible remaining sidecar variants would be residual or ROI overlays that preserve the q55 base frame and avoid replacing the whole generated pair. Those should require a new oracle before implementation.
