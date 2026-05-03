#!/usr/bin/env python
"""CPU audits for the lowmask + boundary residual family.

Phases A-C are intentionally byte/accounting diagnostics only. They do not
launch renderer training or claim a packable submission candidate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm_v2.evaluator import (
    EXPERIMENTS_DIR,
    QPOSE14_LEDGER,
    QPOSE14_SUMMARY,
    load_json,
    load_jsonl,
    qpose14_summary,
    rate_term,
    score,
    write_json,
)
from submissions.search_vcm_v2.families.boundary_residual_codec import (
    boundary_map,
    candidate_from_records,
    compress_bytes,
    compress_streams,
    dilate_bool,
    make_tile_records,
    pack_bits,
)
from submissions.search_vcm_v2.families.lowmask_data import (
    FP4_ARCHIVE,
    FP4_LEDGER,
    FP4_SUMMARY,
    archive_audit,
    decode_lowmask_video,
    split_fp4_archive,
    write_fp4_summary,
)
from submissions.search_vcm_v2.families.qpose14_data import (
    MASK_BYTES as QPOSE_MASK_BYTES,
    QPOSE14_ARCHIVE,
    decode_mask_stream,
    materialize_qpose14_subset,
    split_archive_payload,
)
from submissions.search_vcm_v2.subsets import HARD8


BOUNDARY_DIR = EXPERIMENTS_DIR / "lowmask_boundary"
CACHE_DIR = BOUNDARY_DIR / "cache"
BYTE_AUDIT_JSON = BOUNDARY_DIR / "gate0_byte_audit.json"
TEMPORAL_JSON = BOUNDARY_DIR / "gate0b_temporal_subsampling.json"
FINDINGS = Path(__file__).resolve().parents[1] / "LOWMASK_BOUNDARY_FINDINGS.md"
SINGLE_MEMBER_OVERHEAD_ESTIMATE = 512
AUDIT_COMPRESSORS = ("zstd",)


def _load_exact_classes() -> torch.Tensor:
    cache_path = CACHE_DIR / "qpose_exact_classes.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu").to(torch.uint8)
    mask_br, _, _ = split_archive_payload(QPOSE14_ARCHIVE)
    exact = decode_mask_stream(mask_br).to(torch.uint8).contiguous()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(exact, cache_path)
    return exact


def _load_lowmask() -> tuple[torch.Tensor, torch.Tensor]:
    cls_path = CACHE_DIR / "pr62_lowmask_classes.pt"
    gray_path = CACHE_DIR / "pr62_lowmask_probs.pt"
    if cls_path.exists() and gray_path.exists():
        return torch.load(cls_path, map_location="cpu").to(torch.uint8), torch.load(gray_path, map_location="cpu")
    mask_br, _, _ = split_fp4_archive(FP4_ARCHIVE)
    low, gray = decode_lowmask_video(mask_br)
    low = low.to(torch.uint8).contiguous()
    gray = gray.to(torch.uint8).contiguous()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(low, cls_path)
    torch.save(gray, gray_path)
    return low, gray


def _materialize_required_caches() -> dict[str, str]:
    exact = _load_exact_classes()
    low, gray = _load_lowmask()
    qpose_teacher_path = CACHE_DIR / "qpose_teacher_frames.pt"
    if not qpose_teacher_path.exists():
        hard8 = materialize_qpose14_subset("hard8_capacity", list(HARD8), device="cpu")
        torch.save(
            {
                "sample_ids": hard8["sample_ids"],
                "qpose_frame1": hard8["qpose_frame1"],
                "qpose_frame2": hard8["qpose_frame2"],
                "qpose_pose6": hard8["pose6"],
                "note": "hard8-only teacher cache for lowmask boundary Gate 0.",
            },
            qpose_teacher_path,
        )
    sensitivity_path = CACHE_DIR / "seg_sensitivity.pt"
    if not sensitivity_path.exists():
        torch.save(_seg_sensitivity(), sensitivity_path)
    return {
        "qpose_exact_classes": str(CACHE_DIR / "qpose_exact_classes.pt"),
        "pr62_lowmask_classes": str(CACHE_DIR / "pr62_lowmask_classes.pt"),
        "pr62_lowmask_probs": str(CACHE_DIR / "pr62_lowmask_probs.pt"),
        "qpose_teacher_frames": str(qpose_teacher_path),
        "seg_sensitivity": str(sensitivity_path),
        "exact_shape": list(exact.shape),
        "low_shape": list(low.shape),
        "gray_shape": list(gray.shape),
    }


def _seg_sensitivity() -> dict[str, Any]:
    qpose_rows = {int(row["sample_id"]): row for row in load_jsonl(QPOSE14_LEDGER)}
    fp4_rows = {int(row["sample_id"]): row for row in load_jsonl(FP4_LEDGER)}
    weights = []
    gaps = []
    for sample_id in range(600):
        qrow = qpose_rows.get(sample_id, {})
        frow = fp4_rows.get(sample_id, {})
        q_seg = float(qrow.get("qpose14_seg_term", 0.0))
        f_seg = float(frow.get("fp4_seg_term", q_seg))
        gap = max(0.0, f_seg - q_seg)
        gaps.append(gap)
    max_gap = max(gaps) if gaps else 1.0
    for gap in gaps:
        weights.append(1.0 + (gap / max_gap if max_gap > 0 else 0.0))
    return {"seg_gap_terms": gaps, "sample_weights": weights}


def _tile_density(mask: torch.Tensor, tile_size: int) -> torch.Tensor:
    x = mask.float().unsqueeze(1)
    pooled = F.avg_pool2d(x, kernel_size=tile_size, stride=tile_size)
    return pooled[:, 0].contiguous()


def _selected_tiles_from_mask(tile_mask: torch.Tensor) -> list[tuple[int, int, int]]:
    coords = tile_mask.nonzero(as_tuple=False)
    return [(int(t), int(y), int(x)) for t, y, x in coords.tolist()]


def _top_tile_order(score: torch.Tensor, *, limit: int = 200_000) -> list[tuple[int, int, int]]:
    flat = score.reshape(-1)
    positive = int((flat > 0).sum().item())
    if positive == 0:
        return []
    k = min(limit, positive)
    values, indices = torch.topk(flat, k=k, largest=True, sorted=True)
    tiles_y, tiles_x = score.shape[1:]
    out = []
    for value, idx in zip(values.tolist(), indices.tolist(), strict=True):
        if value <= 0:
            break
        t = idx // (tiles_y * tiles_x)
        rem = idx % (tiles_y * tiles_x)
        y = rem // tiles_x
        x = rem % tiles_x
        out.append((int(t), int(y), int(x)))
    return out


def _best_prefix_under_budget(
    *,
    exact: torch.Tensor,
    low: torch.Tensor,
    ordered_tiles: list[tuple[int, int, int]],
    tile_size: int,
    budget_bytes: int,
) -> dict[str, Any]:
    if not ordered_tiles:
        return candidate_from_records(
            name=f"R5a_tile{tile_size}_budget{budget_bytes // 1024}k_empty",
            records=[],
            shape=tuple(exact.shape),
            tile_size=tile_size,
            compressors=AUDIT_COMPRESSORS,
        )

    def evaluate(n: int) -> dict[str, Any]:
        records = make_tile_records(exact, low, ordered_tiles[:n], tile_size=tile_size)
        return candidate_from_records(
            name=f"R5a_tile{tile_size}_budget{budget_bytes // 1024}k_top{n}",
            records=records,
            shape=tuple(exact.shape),
            tile_size=tile_size,
            compressors=AUDIT_COMPRESSORS,
        )

    lo = 0
    hi = 1
    best = evaluate(0)
    while hi <= len(ordered_tiles):
        candidate = evaluate(hi)
        if int(candidate["residual_bytes"]) <= budget_bytes:
            best = candidate
            lo = hi
            hi *= 2
        else:
            break
    hi = min(hi, len(ordered_tiles))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = evaluate(mid)
        if int(candidate["residual_bytes"]) <= budget_bytes:
            best = candidate
            lo = mid
        else:
            hi = mid - 1
    return best


def _boundary_xor_candidate(boundary_xor: torch.Tensor, *, radius: int) -> dict[str, Any]:
    payload = pack_bits(boundary_xor.cpu().numpy())
    compressed = compress_bytes(payload, AUDIT_COMPRESSORS)
    return {
        "candidate": f"R2_boundary_xor_r{radius}",
        "residual_kind": "R2",
        "radius": radius,
        "source_error_pixels": int(boundary_xor.sum().item()),
        "residual_bytes": int(compressed["best_bytes"]),
        "compressor_breakdown": compressed,
        "decision": _byte_decision(int(compressed["best_bytes"])),
        "diagnostic_only": True,
    }


def _byte_decision(residual_bytes: int) -> str:
    if residual_bytes <= 8 * 1024:
        return "strong_pass"
    if residual_bytes <= 12 * 1024:
        return "pass"
    if residual_bytes <= 15 * 1024:
        return "weak_pass"
    if residual_bytes <= 20 * 1024:
        return "diagnostic_only"
    return "fail"


def run_gate0_byte_audit() -> dict[str, Any]:
    caches = _materialize_required_caches()
    exact = _load_exact_classes()
    low, _ = _load_lowmask()
    if tuple(exact.shape) != tuple(low.shape):
        raise RuntimeError(f"exact/low shape mismatch: {tuple(exact.shape)} vs {tuple(low.shape)}")

    sensitivity = torch.load(CACHE_DIR / "seg_sensitivity.pt", map_location="cpu")
    sample_weights = torch.tensor(sensitivity["sample_weights"], dtype=torch.float32).view(-1, 1, 1)
    exact_boundary = boundary_map(exact)
    low_boundary = boundary_map(low)
    class_diff = exact != low
    result_rows: list[dict[str, Any]] = []

    for radius in (1, 2, 3):
        band = dilate_bool(exact_boundary | low_boundary, radius)
        band_error = class_diff & band
        boundary_xor = dilate_bool(exact_boundary ^ low_boundary, radius)
        total_band_error = int(band_error.sum().item())
        total_boundary_xor = int(boundary_xor.sum().item())
        r2 = _boundary_xor_candidate(boundary_xor, radius=radius)
        r2.update({"total_band_error_pixels": total_band_error, "total_boundary_xor_pixels": total_boundary_xor})
        result_rows.append(r2)

        for tile_size in (8, 16):
            error_density = _tile_density(band_error, tile_size)
            boundary_density = _tile_density(boundary_xor, tile_size)
            score_map = (4.0 * error_density * sample_weights + 2.0 * boundary_density + error_density).contiguous()
            error_tiles = error_density > 0
            selected = _selected_tiles_from_mask(error_tiles)
            records = make_tile_records(exact, low, selected, tile_size=tile_size)
            full = candidate_from_records(
                name=f"R3a_tile{tile_size}_r{radius}_full_boundary_band",
                records=records,
                shape=tuple(exact.shape),
                tile_size=tile_size,
                compressors=AUDIT_COMPRESSORS,
            )
            full.update(
                {
                    "residual_kind": "R3a",
                    "radius": radius,
                    "total_band_error_pixels": total_band_error,
                    "total_boundary_xor_pixels": total_boundary_xor,
                    "coverage_fraction": 1.0 if total_band_error == 0 else full["source_error_pixels"] / total_band_error,
                    "decision": _byte_decision(int(full["residual_bytes"])),
                }
            )
            result_rows.append(full)

            ordered = _top_tile_order(score_map)
            for budget in (4 * 1024, 8 * 1024, 12 * 1024, 16 * 1024, 20 * 1024):
                item = _best_prefix_under_budget(exact=exact, low=low, ordered_tiles=ordered, tile_size=tile_size, budget_bytes=budget)
                item.update(
                    {
                        "residual_kind": "R5a",
                        "radius": radius,
                        "budget_bytes": budget,
                        "total_band_error_pixels": total_band_error,
                        "total_boundary_xor_pixels": total_boundary_xor,
                        "coverage_fraction": 0.0 if total_band_error == 0 else item["source_error_pixels"] / total_band_error,
                        "decision": _byte_decision(int(item["residual_bytes"])),
                    }
                )
                result_rows.append(item)

    full_repair = [row for row in result_rows if row.get("residual_kind") == "R3a"]
    budgeted = [row for row in result_rows if row.get("residual_kind") == "R5a"]
    diagnostics = [row for row in result_rows if row.get("residual_kind") == "R2"]
    best_full = min(full_repair, key=lambda row: int(row["residual_bytes"])) if full_repair else None
    best_budgeted = max(
        budgeted,
        key=lambda row: (float(row.get("coverage_fraction", 0.0)), -int(row["residual_bytes"])),
    ) if budgeted else None
    best_budgeted_by_bytes = min(budgeted, key=lambda row: int(row["residual_bytes"])) if budgeted else None
    best_diagnostic = min(diagnostics, key=lambda row: int(row["residual_bytes"])) if diagnostics else None
    selected_for_gate = best_budgeted or best_full or best_diagnostic
    selected_bytes = int(selected_for_gate["residual_bytes"]) if selected_for_gate else math.inf
    decision = _byte_decision(selected_bytes)

    fp4_audit = archive_audit(FP4_ARCHIVE)
    summary = {
        "round": "gate0_byte_audit",
        "inputs": {
            "qpose_archive": str(QPOSE14_ARCHIVE),
            "fp4_archive": str(FP4_ARCHIVE),
            "caches": caches,
        },
        "fp4_archive_bytes": int(fp4_audit["archive_bytes"]),
        "fp4_payload_breakdown": fp4_audit["payload_breakdown"],
        "best_full_repair": best_full,
        "best_budgeted_by_coverage": best_budgeted,
        "best_budgeted_by_bytes": best_budgeted_by_bytes,
        "best_boundary_xor": best_diagnostic,
        "selected_gate_candidate": selected_for_gate,
        "selected_projected_archive_bytes": None if selected_for_gate is None else int(fp4_audit["archive_bytes"]) + int(selected_for_gate["residual_bytes"]),
        "decision": decision,
        "rows": result_rows,
    }
    write_json(BYTE_AUDIT_JSON, summary)
    return summary


def _nearest_prediction(exact: torch.Tensor, k: int) -> torch.Tensor:
    idx = torch.arange(exact.shape[0])
    key = (idx // k) * k
    return exact[key.clamp(max=exact.shape[0] - 1)].contiguous()


def _linear_prediction(exact: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.empty_like(exact)
    t_count = exact.shape[0]
    for start in range(0, t_count, k):
        end = min(start + k, t_count - 1)
        start_frame = exact[start].float()
        end_frame = exact[end].float()
        denom = max(1, end - start)
        for t in range(start, min(start + k, t_count)):
            alpha = (t - start) / denom
            out[t] = torch.round((1.0 - alpha) * start_frame + alpha * end_frame).clamp(0, 4).to(torch.uint8)
    return out


def _temporal_payload(exact: torch.Tensor, pred: torch.Tensor, k: int, variant: str) -> dict[str, Any]:
    key_indices = list(range(0, exact.shape[0], k))
    key_payload = exact[key_indices].contiguous().numpy().astype("uint8", copy=False).tobytes()
    diff = pred != exact
    residual_streams = {
        "changed_bitmap.bin": pack_bits(diff.numpy()),
        "changed_classes.bin": exact[diff].contiguous().numpy().astype("uint8", copy=False).tobytes(),
    }
    key_compressed = compress_bytes(key_payload, AUDIT_COMPRESSORS)
    residual_compressed = compress_streams(residual_streams, AUDIT_COMPRESSORS)
    total = int(key_compressed["best_bytes"]) + int(residual_compressed["total_best_bytes"])
    return {
        "candidate": f"temporal_k{k}_{variant}_sparse_residual",
        "residual_kind": "temporal_subsample",
        "k": k,
        "variant": variant,
        "key_frames": len(key_indices),
        "changed_pixels": int(diff.sum().item()),
        "key_mask_bytes": int(key_compressed["best_bytes"]),
        "residual_bytes": int(residual_compressed["total_best_bytes"]),
        "total_mask_payload_bytes": total,
        "key_compressor_breakdown": key_compressed,
        "residual_compressor_breakdown": residual_compressed,
        "decision": _temporal_decision(total),
    }


def _temporal_decision(payload_bytes: int) -> str:
    if payload_bytes <= 182 * 1024:
        return "strong_pass"
    if payload_bytes <= 196 * 1024:
        return "pass"
    if payload_bytes <= 205 * 1024:
        return "near"
    return "fail"


def run_gate0b_temporal_subsampling() -> dict[str, Any]:
    exact = _load_exact_classes()
    rows: list[dict[str, Any]] = []
    for k in (2, 3, 4, 5):
        rows.append(_temporal_payload(exact, _nearest_prediction(exact, k), k, "nearest"))
        rows.append(_temporal_payload(exact, _linear_prediction(exact, k), k, "linear"))
    best = min(rows, key=lambda row: int(row["total_mask_payload_bytes"]))
    qpose = qpose14_summary()
    archive_without_mask = int(qpose["archive_bytes"]) - QPOSE_MASK_BYTES
    summary = {
        "round": "gate0b_temporal_subsampling",
        "qpose_archive_bytes": int(qpose["archive_bytes"]),
        "qpose_mask_bytes": QPOSE_MASK_BYTES,
        "archive_without_mask": archive_without_mask,
        "best": best,
        "best_projected_archive_bytes": archive_without_mask + int(best["total_mask_payload_bytes"]) + SINGLE_MEMBER_OVERHEAD_ESTIMATE,
        "decision": best["decision"],
        "rows": rows,
    }
    write_json(TEMPORAL_JSON, summary)
    return summary


def update_findings() -> None:
    exact_summary_path = EXPERIMENTS_DIR / "exact_mask_lossless_sweep" / "summary.json"
    exact_summary = load_json(exact_summary_path) if exact_summary_path.exists() else None
    gate0 = load_json(BYTE_AUDIT_JSON) if BYTE_AUDIT_JSON.exists() else None
    temporal = load_json(TEMPORAL_JSON) if TEMPORAL_JSON.exists() else None
    qpose = qpose14_summary()
    fp4_summary = load_json(FP4_SUMMARY) if FP4_SUMMARY.exists() else {}
    lines = [
        "# Lowmask Boundary Residual Findings",
        "",
        "## Context",
        "",
        "This audit tests the narrow hypothesis: PR #62's low-byte mask regime plus a tiny boundary helper may recover qpose14-like SegNet quality without giving up the byte advantage.",
        "",
        "## Baselines",
        "",
        f"- qpose14 local archive bytes: `{qpose.get('archive_bytes')}`",
        f"- qpose14 local full600 quality: `{qpose.get('quality')}`",
        f"- qpose14 local hard8 quality: `{qpose.get('hard8_quality')}`",
        f"- PR #62 local archive bytes: `{fp4_summary.get('archive_bytes')}`",
        f"- PR #62 local full600 quality: `{fp4_summary.get('quality')}`",
        f"- PR #62 local hard8 quality: `{fp4_summary.get('hard8_quality')}`",
        "",
        "## Phase A: Exact-Mask Lossless Sweep",
        "",
    ]
    if exact_summary:
        lines.extend(
            [
                f"- Best exact-mask payload: `{exact_summary.get('best', {}).get('bytes')}` bytes via `{exact_summary.get('best', {}).get('candidate')}`",
                f"- Decision: `{exact_summary.get('decision')}`",
            ]
        )
    else:
        lines.append("- Not run yet.")
    lines.extend(["", "## Phase B: Gate 0 Boundary Residual Byte Audit", ""])
    if gate0:
        best_full = gate0.get("best_full_repair") or {}
        best_budget = gate0.get("best_budgeted_by_coverage") or {}
        selected = gate0.get("selected_gate_candidate") or {}
        lines.extend(
            [
                f"- Best full R3a repair: `{best_full.get('residual_bytes')}` bytes, coverage `{best_full.get('coverage_fraction')}`.",
                f"- Best budgeted R5a by coverage: `{best_budget.get('residual_bytes')}` bytes, coverage `{best_budget.get('coverage_fraction')}`.",
                f"- Selected Gate 0 candidate: `{selected.get('candidate')}`, bytes `{selected.get('residual_bytes')}`, coverage `{selected.get('coverage_fraction')}`.",
                f"- Projected archive with selected residual: `{gate0.get('selected_projected_archive_bytes')}` bytes.",
                f"- Decision: `{gate0.get('decision')}`.",
            ]
        )
    else:
        lines.append("- Not run yet.")
    lines.extend(["", "## Phase C: Gate 0b Temporal Subsampling", ""])
    if temporal:
        best = temporal.get("best") or {}
        lines.extend(
            [
                f"- Best temporal candidate: `{best.get('candidate')}`.",
                f"- Total mask payload: `{best.get('total_mask_payload_bytes')}` bytes.",
                f"- Projected archive: `{temporal.get('best_projected_archive_bytes')}` bytes.",
                f"- Decision: `{temporal.get('decision')}`.",
            ]
        )
    else:
        lines.append("- Not run yet.")
    lines.extend(["", "## Decision", ""])
    if gate0:
        decision = gate0.get("decision")
        if decision == "fail":
            lines.append("Close `lowmask_boundary_residual` before GPU: Gate 0 best residual exceeded 20 KB.")
        else:
            lines.append("Stop after Phase A-C as requested. Do not launch GPU Gate 1 until this byte audit is explicitly reviewed.")
    else:
        lines.append("Pending Gate 0 results.")
    FINDINGS.write_text("\n".join(lines) + "\n")


class LowmaskBoundaryResidualCandidate(Candidate):
    family = "lowmask_boundary_residual"
    role = "byte_audit"
    kind = "cpu_residual_audit"
    packable = False

    def __init__(self, *, variant: str) -> None:
        self.variant = variant
        self.name = f"lowmask_boundary_{variant}"
        self.config = {"config_id": self.name, "variant": variant}
        self._package = PackageInfo(archive_bytes=None, added_bytes=0, projected=True)
        self._metrics = Metrics(None, None, None, sample_count=600)
        self._decision = "not_run"
        self._failure_reason = ""
        self._extra: dict[str, Any] = {}

    def prepare(self, ctx: dict[str, Any]) -> None:
        if not QPOSE14_SUMMARY.exists() or not QPOSE14_LEDGER.exists():
            raise RuntimeError("qpose14 baseline ledger is required before lowmask boundary audits")
        if not FP4_ARCHIVE.exists():
            raise FileNotFoundError(f"missing PR #62 archive: {FP4_ARCHIVE}")
        if not FP4_SUMMARY.exists() or not FP4_LEDGER.exists():
            rows = []
            if FP4_LEDGER.exists():
                rows = load_jsonl(FP4_LEDGER)
            write_fp4_summary(rows)
        self._extra["qpose14_summary"] = str(QPOSE14_SUMMARY)
        self._extra["fp4_summary"] = str(FP4_SUMMARY)

    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        if budget.round == "smoke":
            self._decision = "smoke_pass"
            return
        if budget.round == "gate0_byte_audit":
            summary = run_gate0_byte_audit()
            selected = summary.get("selected_gate_candidate") or {}
            archive = summary.get("selected_projected_archive_bytes")
            residual_bytes = int(selected.get("residual_bytes", 0) or 0)
            self._package = PackageInfo(
                archive_bytes=int(archive) if archive is not None else None,
                added_bytes=residual_bytes,
                payload_breakdown={
                    "fp4_archive": int(summary.get("fp4_archive_bytes", 0)),
                    "boundary_residual": residual_bytes,
                },
                projected=True,
            )
            self._extra["gate0_byte_audit"] = summary
            decision = summary.get("decision", "fail")
            if decision in {"strong_pass", "pass", "weak_pass"}:
                self._decision = "promote_free_boundary_capacity"
            elif decision == "diagnostic_only":
                self._decision = "diagnostic_boundary_candidate"
            else:
                self._decision = "fail_gate0_byte_audit"
                self._failure_reason = "Gate 0 best residual exceeded 20 KB."
            update_findings()
            return
        if budget.round == "gate0b_temporal_subsampling":
            summary = run_gate0b_temporal_subsampling()
            best = summary.get("best") or {}
            archive = int(summary.get("best_projected_archive_bytes", 0))
            self._package = PackageInfo(
                archive_bytes=archive,
                added_bytes=max(0, int(best.get("total_mask_payload_bytes", 0)) - QPOSE_MASK_BYTES),
                payload_breakdown={
                    "archive_without_mask": int(summary.get("archive_without_mask", 0)),
                    "temporal_mask_payload": int(best.get("total_mask_payload_bytes", 0)),
                    "overhead": SINGLE_MEMBER_OVERHEAD_ESTIMATE,
                },
                projected=True,
            )
            qpose = qpose14_summary()
            self._metrics = Metrics(
                segnet_dist=float(qpose["segnet_dist"]),
                posenet_dist=float(qpose["posenet_dist"]),
                quality=float(qpose["quality"]),
                seg_term=float(qpose["seg_term"]),
                pose_term=float(qpose["pose_term"]),
                score=score(float(qpose["segnet_dist"]), float(qpose["posenet_dist"]), archive),
                sample_count=600,
            )
            self._extra["gate0b_temporal_subsampling"] = summary
            decision = summary.get("decision", "fail")
            if decision in {"strong_pass", "pass"}:
                self._decision = "promote_temporal_mask_subsampling"
            elif decision == "near":
                self._decision = "near_temporal_mask_subsampling"
            else:
                self._decision = "fail_temporal_subsampling"
                self._failure_reason = "Temporal subsampling payload exceeded 205 KB."
            update_findings()
            return
        self._decision = "blocked_phase_not_implemented"
        self._failure_reason = "GPU renderer phases are intentionally not implemented in Phase A-C."

    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        return self._package

    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        return self._metrics

    def decision_row(self, budget: Budget, ctx: dict[str, Any]) -> DecisionRow:
        package = self.package(ctx)
        metrics = self.evaluate(budget.subset, ctx)
        projected_score = metrics.score
        if projected_score is None and metrics.quality is not None and package.archive_bytes is not None:
            projected_score = float(metrics.quality) + rate_term(int(package.archive_bytes))
        baseline = ctx.get("baseline", qpose14_summary())
        byte_delta = int(package.archive_bytes) - int(baseline["archive_bytes"]) if package.archive_bytes is not None else None
        score_delta = float(projected_score) - float(baseline["score"]) if projected_score is not None and baseline.get("score") is not None else None
        return DecisionRow(
            run_id=ctx["run_id"],
            candidate_name=self.name,
            family=self.family,
            role=self.role,
            kind=self.kind,
            packable=self.packable,
            config_hash=stable_hash(self.config),
            novelty_reason="",
            subset=budget.subset,
            round=budget.round,
            archive_bytes=package.archive_bytes,
            added_bytes=package.added_bytes,
            quality=metrics.quality,
            segnet_dist=metrics.segnet_dist,
            posenet_dist=metrics.posenet_dist,
            score=projected_score,
            seg_delta=None,
            pose_delta=None,
            byte_delta=byte_delta,
            score_delta_vs_base=score_delta,
            dominates_base=score_delta is not None and score_delta <= 0 and (byte_delta is None or byte_delta <= 0),
            term_tradeoff="cpu_boundary_byte_audit",
            decision=self._decision,
            failure_reason=self._failure_reason,
            extra=self._extra,
        )


def candidates(round_name: str) -> list[Candidate]:
    if round_name == "gate0_byte_audit":
        return [LowmaskBoundaryResidualCandidate(variant="gate0_byte_audit")]
    if round_name == "gate0b_temporal_subsampling":
        return [LowmaskBoundaryResidualCandidate(variant="gate0b_temporal_subsampling")]
    return [LowmaskBoundaryResidualCandidate(variant="smoke")]
