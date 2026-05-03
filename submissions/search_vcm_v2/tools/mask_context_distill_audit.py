#!/usr/bin/env python
"""CPU source-coding audit: predict exact qpose masks from PR #62 lowmask.

This tests the "mask-stream distillation" idea with deterministic context
tables before spending GPU time on a neural predictor. The measured payload is:

    PR62 lowmask stream + context table + sparse residual back to exact mask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, write_json
from submissions.search_vcm_v2.families.boundary_residual_codec import (
    candidate_from_records,
    compress_streams,
    encode_varints,
    make_tile_records,
)
from submissions.search_vcm_v2.families.lowmask_boundary_residual import _load_exact_classes, _load_lowmask
from submissions.search_vcm_v2.families.lowmask_data import archive_audit, FP4_ARCHIVE


OUT_DIR = EXPERIMENTS_DIR / "mask_context_distill"
LOWMASK_BYTES = 175_336


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
    y0 = 1 + dy
    x0 = 1 + dx
    return out[:, y0 : y0 + arr.shape[1], x0 : x0 + arr.shape[2]]


def _context_keys(low: np.ndarray, kind: str) -> tuple[np.ndarray, int, np.ndarray]:
    center = low
    if kind == "center":
        return center.astype(np.int32), 5, center
    if kind == "plus5":
        parts = [
            center,
            _shift(low, -1, 0),
            _shift(low, 1, 0),
            _shift(low, 0, -1),
            _shift(low, 0, 1),
        ]
    elif kind == "x3":
        parts = [
            _shift(low, -1, -1),
            _shift(low, -1, 0),
            _shift(low, -1, 1),
            _shift(low, 0, -1),
            center,
            _shift(low, 0, 1),
            _shift(low, 1, -1),
            _shift(low, 1, 0),
            _shift(low, 1, 1),
        ]
    else:
        raise ValueError(f"unknown context kind: {kind}")
    key = np.zeros_like(center, dtype=np.int32)
    mul = 1
    for part in parts:
        key += part.astype(np.int32) * mul
        mul *= 5
    return key, mul, center


def _build_counts(low: np.ndarray, exact: np.ndarray, kind: str, *, chunk: int = 25) -> tuple[np.ndarray, np.ndarray]:
    _, key_count, _ = _context_keys(low[:1], kind)
    counts = np.zeros((key_count, 5), dtype=np.int64)
    low_center_for_key = np.zeros(key_count, dtype=np.uint8)
    for start in range(0, low.shape[0], chunk):
        keys, _, center = _context_keys(low[start : start + chunk], kind)
        exact_chunk = exact[start : start + chunk]
        flat_keys = keys.reshape(-1)
        flat_exact = exact_chunk.reshape(-1)
        flat_center = center.reshape(-1)
        for cls in range(5):
            counts[:, cls] += np.bincount(flat_keys[flat_exact == cls], minlength=key_count)
        # The center class is deterministic for a context key in these encodings.
        first = np.unique(flat_keys, return_index=True)
        low_center_for_key[first[0]] = flat_center[first[1]]
    return counts, low_center_for_key


def _select_table(counts: np.ndarray, center_for_key: np.ndarray) -> dict[str, np.ndarray]:
    totals = counts.sum(axis=1)
    majority = counts.argmax(axis=1).astype(np.uint8)
    center_counts = counts[np.arange(counts.shape[0]), center_for_key]
    majority_counts = counts[np.arange(counts.shape[0]), majority]
    improvement = majority_counts.astype(np.int64) - center_counts.astype(np.int64)
    selected = (totals > 0) & (majority != center_for_key) & (improvement > 0)
    return {
        "keys": np.nonzero(selected)[0].astype(np.int64),
        "classes": majority[selected].astype(np.uint8),
        "improvement": improvement[selected].astype(np.int64),
        "totals": totals[selected].astype(np.int64),
    }


def _pack_table(keys: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    if keys.size == 0:
        streams = {"key_deltas.bin": b"", "classes.bin": b""}
    else:
        order = np.argsort(keys)
        sorted_keys = keys[order]
        sorted_classes = classes[order]
        deltas = np.empty_like(sorted_keys)
        deltas[0] = sorted_keys[0]
        deltas[1:] = sorted_keys[1:] - sorted_keys[:-1]
        streams = {
            "key_deltas.bin": encode_varints(deltas.tolist()),
            "classes.bin": sorted_classes.tobytes(),
        }
    compressed = compress_streams(streams, ("zstd",))
    return {"streams": compressed, "bytes": int(compressed["total_best_bytes"])}


def _apply_table(low: np.ndarray, kind: str, keys: np.ndarray, classes: np.ndarray, *, chunk: int = 25) -> np.ndarray:
    pred = low.copy()
    if keys.size == 0:
        return pred
    table = np.full(_context_keys(low[:1], kind)[1], 255, dtype=np.uint8)
    table[keys] = classes
    for start in range(0, low.shape[0], chunk):
        context, _, _ = _context_keys(low[start : start + chunk], kind)
        mapped = table[context]
        mask = mapped != 255
        pred[start : start + chunk][mask] = mapped[mask]
    return pred


def _residual_payload(exact: torch.Tensor, pred: torch.Tensor, *, kind: str) -> dict[str, Any]:
    diff = exact != pred
    selected8 = [(int(t), int(y), int(x)) for t, y, x in (torch.nn.functional.avg_pool2d(diff.float().unsqueeze(1), 8, 8)[:, 0] > 0).nonzero(as_tuple=False).tolist()]
    selected16 = [(int(t), int(y), int(x)) for t, y, x in (torch.nn.functional.avg_pool2d(diff.float().unsqueeze(1), 16, 16)[:, 0] > 0).nonzero(as_tuple=False).tolist()]
    rec8 = make_tile_records(exact, pred, selected8, tile_size=8)
    rec16 = make_tile_records(exact, pred, selected16, tile_size=16)
    cand8 = candidate_from_records(name=f"{kind}_residual_tile8", records=rec8, shape=tuple(exact.shape), tile_size=8, compressors=("zstd",))
    cand16 = candidate_from_records(name=f"{kind}_residual_tile16", records=rec16, shape=tuple(exact.shape), tile_size=16, compressors=("zstd",))
    best = cand8 if int(cand8["residual_bytes"]) <= int(cand16["residual_bytes"]) else cand16
    best["remaining_error_pixels"] = int(diff.sum().item())
    return best


def run_audit(*, contexts: list[str], out: Path) -> dict[str, Any]:
    exact_t = _load_exact_classes().to(torch.uint8)
    low_t, _ = _load_lowmask()
    low_t = low_t.to(torch.uint8)
    exact = exact_t.numpy().astype(np.uint8, copy=False)
    low = low_t.numpy().astype(np.uint8, copy=False)
    rows = []
    fp4 = archive_audit(FP4_ARCHIVE)
    baseline_residual = _residual_payload(exact_t, low_t, kind="identity_lowmask")
    rows.append(
        {
            "candidate": "identity_lowmask_plus_sparse_residual",
            "context": "identity",
            "table_bytes": 0,
            "residual_bytes": int(baseline_residual["residual_bytes"]),
            "total_model_residual_bytes": int(baseline_residual["residual_bytes"]),
            "total_mask_payload_bytes": LOWMASK_BYTES + int(baseline_residual["residual_bytes"]),
            "remaining_error_pixels": int(baseline_residual["remaining_error_pixels"]),
            "residual": baseline_residual,
        }
    )
    for kind in contexts:
        counts, center_for_key = _build_counts(low, exact, kind)
        selected = _select_table(counts, center_for_key)
        table = _pack_table(selected["keys"], selected["classes"])
        pred_np = _apply_table(low, kind, selected["keys"], selected["classes"])
        pred_t = torch.from_numpy(pred_np).to(torch.uint8)
        residual = _residual_payload(exact_t, pred_t, kind=kind)
        total_model_residual = int(table["bytes"]) + int(residual["residual_bytes"])
        rows.append(
            {
                "candidate": f"context_{kind}_table_plus_sparse_residual",
                "context": kind,
                "observed_contexts": int((counts.sum(axis=1) > 0).sum()),
                "selected_contexts": int(selected["keys"].size),
                "table_bytes": int(table["bytes"]),
                "residual_bytes": int(residual["residual_bytes"]),
                "total_model_residual_bytes": total_model_residual,
                "total_mask_payload_bytes": LOWMASK_BYTES + total_model_residual,
                "predicted_error_pixels": int((pred_t != exact_t).sum().item()),
                "raw_improvement_pixels": int(selected["improvement"].sum()) if selected["improvement"].size else 0,
                "table": table,
                "residual": residual,
            }
        )
    best = min(rows, key=lambda row: int(row["total_mask_payload_bytes"]))
    summary = {
        "fp4_archive_bytes": int(fp4["archive_bytes"]),
        "lowmask_bytes": LOWMASK_BYTES,
        "thresholds": {"pass": 205 * 1024, "strong": 182 * 1024},
        "rows": rows,
        "best": best,
        "decision": "continue" if int(best["total_mask_payload_bytes"]) <= 205 * 1024 else "close_mask_context_distill",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="center,plus5,x3")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    contexts = [item.strip() for item in args.contexts.split(",") if item.strip()]
    run_audit(contexts=contexts, out=args.out)


if __name__ == "__main__":
    main()
