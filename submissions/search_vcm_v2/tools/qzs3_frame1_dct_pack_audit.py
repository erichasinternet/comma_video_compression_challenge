#!/usr/bin/env python
"""Quantization and byte audit for qzs3 frame1 DCT actuator checkpoints."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
from pathlib import Path

import brotli
import torch
try:
    import zstandard as zstd
except ModuleNotFoundError:  # pragma: no cover - optional local dependency.
    zstd = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.evaluator import quality, rate_term, write_json  # noqa: E402
from submissions.search_vcm_v2.tools.qzs3_frame1_dct_oracle import (  # noqa: E402
    DEFAULT_SUBMISSION,
    eval_actual,
    load_subset_pairs,
    make_dct_basis,
    select_device,
)
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402


def pack_signed(values: torch.Tensor, bits: int) -> bytes:
    """Pack signed quantized values into a bitstream after offsetting to unsigned."""
    if bits == 8:
        return (values.to(torch.int16).clamp(-128, 127) + 128).to(torch.uint8).cpu().numpy().tobytes()
    limit = (1 << (bits - 1)) - 1
    unsigned = (values.to(torch.int16).clamp(-limit, limit) + limit).flatten().tolist()
    out = bytearray()
    acc = 0
    nbits = 0
    mask = (1 << bits) - 1
    for v in unsigned:
        acc |= (int(v) & mask) << nbits
        nbits += bits
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def compress_sizes(payload: bytes) -> dict[str, int]:
    sizes = {
        "raw": len(payload),
        "brotli_q11": len(brotli.compress(payload, quality=11)),
        "xz_9": len(lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)),
    }
    if zstd is not None:
        cctx = zstd.ZstdCompressor(level=22)
        sizes["zstd_22"] = len(cctx.compress(payload))
    return sizes


def quantize_dense(alpha: torch.Tensor, max_delta: float, bits: int) -> tuple[torch.Tensor, bytes]:
    limit = (1 << (bits - 1)) - 1
    q = torch.round(alpha / max_delta * limit).clamp(-limit, limit).to(torch.int16)
    deq = q.float() / limit * max_delta
    return deq, pack_signed(q, bits)


def quantize_topm(alpha: torch.Tensor, max_delta: float, bits: int, topm: int) -> tuple[torch.Tensor, bytes]:
    limit = (1 << (bits - 1)) - 1
    deq = torch.zeros_like(alpha)
    idx_rows = []
    val_rows = []
    for row in alpha:
        idx = torch.topk(row.abs(), k=min(topm, row.numel())).indices.sort().values
        q = torch.round(row[idx] / max_delta * limit).clamp(-limit, limit).to(torch.int16)
        deq[len(idx_rows), idx] = q.float() / limit * max_delta
        idx_rows.append(idx.to(torch.uint8).cpu())
        val_rows.append(q)
    # Fixed topm per sample; sample ids live in checkpoint metadata, so stream is
    # just coefficient indices and quantized values.
    idx_payload = torch.cat(idx_rows).numpy().tobytes()
    val_payload = pack_signed(torch.cat(val_rows), bits)
    return deq, idx_payload + val_payload


def apply_alpha(comp: torch.Tensor, alpha: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    import einops

    delta = torch.einsum("bk,kchw->bchw", alpha.to(comp.device), basis)
    frame1 = (comp[:, 0] + einops.rearrange(delta, "b c h w -> b h w c")).clamp(0.0, 255.0)
    return torch.stack([frame1, comp[:, 1]], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bits", default="8,6,4")
    parser.add_argument("--topm", default="8,16,24,32,48")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--archive-bytes", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sample_ids = [int(x) for x in ckpt["sample_ids"]]
    basis_k = int(ckpt["basis_k"])
    max_delta = float(ckpt["max_delta"])
    alpha = ckpt["alpha"].float()
    device = select_device(args.device)
    ids, gt, comp = load_subset_pairs(args.submission_dir, "custom", device, sample_ids)
    if ids != sample_ids:
        raise ValueError(f"loaded ids differ from checkpoint ids: {ids} != {sample_ids}")

    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in distortion.parameters():
        p.requires_grad_(False)

    basis = make_dct_basis(basis_k, 874, 1164, device)
    base_metrics = eval_actual(distortion, gt, comp, batch_size=args.eval_batch_size)
    fp_metrics = eval_actual(distortion, gt, apply_alpha(comp, alpha.to(device), basis), batch_size=args.eval_batch_size)
    archive_bytes = args.archive_bytes or int((args.submission_dir / "archive.zip").stat().st_size)

    rows = []
    for bits in [int(x) for x in args.bits.split(",") if x.strip()]:
        deq, payload = quantize_dense(alpha, max_delta, bits)
        metrics = eval_actual(distortion, gt, apply_alpha(comp, deq.to(device), basis), batch_size=args.eval_batch_size)
        sizes = compress_sizes(payload)
        rows.append(
            {
                "mode": "dense",
                "bits": bits,
                "topm": None,
                "payload_bytes": sizes,
                "quality": metrics["quality"],
                "quality_delta_vs_base": metrics["quality"] - base_metrics["quality"],
                "quality_delta_vs_float": metrics["quality"] - fp_metrics["quality"],
                "projected_score_brotli": metrics["quality"] + rate_term(archive_bytes + sizes["brotli_q11"]),
                "metrics": {k: v for k, v in metrics.items() if k != "rows"},
            }
        )
        for topm in [int(x) for x in args.topm.split(",") if x.strip()]:
            if topm >= basis_k:
                continue
            deq, payload = quantize_topm(alpha, max_delta, bits, topm)
            metrics = eval_actual(distortion, gt, apply_alpha(comp, deq.to(device), basis), batch_size=args.eval_batch_size)
            sizes = compress_sizes(payload)
            rows.append(
                {
                    "mode": "topm",
                    "bits": bits,
                    "topm": topm,
                    "payload_bytes": sizes,
                    "quality": metrics["quality"],
                    "quality_delta_vs_base": metrics["quality"] - base_metrics["quality"],
                    "quality_delta_vs_float": metrics["quality"] - fp_metrics["quality"],
                    "projected_score_brotli": metrics["quality"] + rate_term(archive_bytes + sizes["brotli_q11"]),
                    "metrics": {k: v for k, v in metrics.items() if k != "rows"},
                }
            )

    payload = {
        "checkpoint": str(args.checkpoint),
        "sample_ids": sample_ids,
        "basis_k": basis_k,
        "max_delta": max_delta,
        "archive_bytes": archive_bytes,
        "base": {k: v for k, v in base_metrics.items() if k != "rows"},
        "float": {k: v for k, v in fp_metrics.items() if k != "rows"},
        "float_quality_delta_vs_base": fp_metrics["quality"] - base_metrics["quality"],
        "rows": sorted(rows, key=lambda row: row["projected_score_brotli"]),
    }
    write_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in ("basis_k", "base", "float", "float_quality_delta_vs_base")}, indent=2))
    print("best rows:")
    for row in payload["rows"][:10]:
        print(json.dumps({k: row[k] for k in ("mode", "bits", "topm", "payload_bytes", "quality", "quality_delta_vs_base", "quality_delta_vs_float", "projected_score_brotli")}, sort_keys=True))


if __name__ == "__main__":
    main()
