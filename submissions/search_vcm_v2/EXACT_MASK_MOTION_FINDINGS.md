# Exact Mask Motion Codec Findings

## Context

This was a final exact-mask source-coding audit for qpose14. The current qpose14
mask payload is `219,472 B`; a useful exact-mask path needs roughly:

- `<=182 KiB` for a visible PR-like `<0.300` path.
- `<=196 KiB` for a near candidate / first-place packaging path.
- `<=205 KiB` to justify further exact-mask codec work.

The codec is lossless over the rounded qpose14 class tensor. It predicts each
mask frame from the previous exact mask using block motion, then stores exact
sparse or raw repairs. All reported rows verified exact decode.

## Results

| Variant | Best exact mask payload | Delta vs qpose mask | Projected qpose archive | Decision |
| --- | ---: | ---: | ---: | --- |
| `16x16`, search `8`, bitmap sparse | `358,055 B` | `+138,583 B` | `426,156 B` | fail |
| `16x16`, search `8`, offset sparse | `340,696 B` | `+121,224 B` | `408,797 B` | fail |
| `8x8`, search `8`, mixed sparse | `349,580 B` | `+130,108 B` | `417,681 B` | fail |
| `32x32`, search `16`, offset sparse | `348,660 B` | `+129,188 B` | `416,761 B` | fail |

Best verified result:

```text
motion_b16_s8_step2_thr4_offsets
mask_payload_bytes: 340,696
current_qpose_mask_bytes: 219,472
projected_qpose_archive_bytes: 408,797
```

## Decision

Close exact-mask motion coding. Even after block motion, sparse repair offset
coding, multiple block sizes, and exact decode verification, the best result is
`~121 KB` larger than the current AV1/Brotli mask. This does not approach the
`<=205 KiB` continue gate.

The practical conclusion is that qpose14's existing AV1 mask stream is already
using temporal/spatial structure better than our exact semantic-map source
coding attempts. The exact-mask-byte route should remain in the negative cache
unless a genuinely different entropy model is introduced.
