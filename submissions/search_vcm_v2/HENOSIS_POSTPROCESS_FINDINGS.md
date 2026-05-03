# Henosis PR #65 Postprocess Oracle Findings

## Context

PR #65 (`henosis_qz_n3z_r25_clean`) is the current public frontier we audited:

```text
archive: 284,425 B
PoseNet: 0.00035283
SegNet:  0.00070896
score:   ~0.31968
```

Payload audit:

```text
mask:      219,472 B
model:      57,074 B
pose:        1,487 B
controls:   ~6.3 KB
```

At `284,425 B`, a `<0.300` score needs quality around `0.1106`, so PR #65
needs roughly `0.020` more quality improvement if bytes stay fixed.

## Audits

All decisive rows below use CPU verification because MPS produced unstable
PoseNet metrics on this branch.

| Control family | CPU hard8 base quality | CPU hard8 oracle quality | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| Frame0 extra motion / fractional shifts | `0.162075` | `0.162014` | `-0.000062` | close |
| Frame0/frame1 RGB bias shortlist | `0.162075` | `0.161557` | `-0.000519` | close |
| Frame2 class-conditioned bias shortlist | `0.162075` | `0.161825` | `-0.000251` | close |

The larger MPS sweeps showed apparent `-0.018` to `-0.041` wins, but those did
not reproduce on CPU. Treat those as MPS evaluator artifacts, not candidate
signals.

## Decision

Close incremental PR #65 postprocess controls as a `<0.300` route. The verified
CPU improvement is orders of magnitude below the needed `~0.020` quality gain.

PR #65 remains the best practical fallback target, but the remaining 0.2x gap is
not available through another tiny LUT/shift/bias control layer.
