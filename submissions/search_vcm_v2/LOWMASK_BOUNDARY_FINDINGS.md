# Lowmask Boundary Residual Findings

## Context

This audit tests the narrow hypothesis: PR #62's low-byte mask regime plus a tiny boundary helper may recover qpose14-like SegNet quality without giving up the byte advantage.

## Baselines

- qpose14 local archive bytes: `287573`
- qpose14 local full600 quality: `0.15362587198943095`
- qpose14 local hard8 quality: `0.18193569744702692`
- PR #62 local archive bytes: `249624`
- PR #62 local full600 quality: `0.6418322142901499`
- PR #62 local hard8 quality: `0.45243085937790645`

## Phase A: Exact-Mask Lossless Sweep

- Best exact-mask payload: `439409` bytes via `uint8_class_bitplanes`
- Decision: `fail`
- Compressor scope: bounded fast audit using `zstd` plus `ffv1`. Initial `brotli`/`xz` full-stream attempts were stopped because they blocked before producing a result.

## Phase B: Gate 0 Boundary Residual Byte Audit

- Best full R3a repair: `179205` bytes, coverage `1.0000073894537715`.
- Best budgeted R5a by coverage: `20471` bytes, coverage `0.15797057541251322`.
- Selected Gate 0 candidate: `R5a_tile8_budget20k_top3466`, bytes `20471`, coverage `0.15797057541251322`.
- Projected archive with selected residual: `270095` bytes.
- Decision: `diagnostic_only`.
- Compressor scope: bounded CPU audit using `zstd` accounting for residual streams.

## Phase C: Gate 0b Temporal Subsampling

- Best temporal candidate: `temporal_k2_linear_sparse_residual`.
- Total mask payload: `543958` bytes.
- Projected archive: `612571` bytes.
- Decision: `fail`.

## Decision

Stop after Phase A-C as requested. Do not launch GPU Gate 1 until this byte audit is explicitly reviewed.
