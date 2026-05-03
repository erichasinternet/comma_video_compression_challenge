#!/usr/bin/env python
"""PoseNet-preprocess-space B1/B2 oracle family.

B1 is a non-submission reachability oracle: optimize the tensor consumed by
PoseNet directly. B2 is a projection oracle: try to realize a passing B1 tensor
with legal RGB frames. Neither B1 nor B2 emits a packable archive candidate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import PoseNet, posenet_sd_path
from submissions.search_vcm.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm.evaluator import EXPERIMENTS_DIR, classify_against_base, quality, write_json
from submissions.search_vcm.subsets import HARD3, HARD8
from submissions.tavs_video.common import (
    build_distortion,
    collect_targets,
    ensure_q55_inflated,
    load_original_pairs_by_indices,
    load_raw_pairs_by_indices,
)
from submissions.commavq_task.common import FeatureTap, round_ste


DEFAULT_CACHE = REPO_ROOT / "submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int10_cpu/submission"
VIDEO_NAMES = REPO_ROOT / "public_test_video_names.txt"
VIDEO_DIR = REPO_ROOT / "videos"
Q55_DIR = REPO_ROOT / "submissions/q55_fp16_pose_int10"
ORACLE_DIR = EXPERIMENTS_DIR / "posenet_preprocess_oracle"
B1_INTERPOLATION_ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
SANITY_POSE_TERM_GATE = 0.005


def resolve_device(ctx: dict[str, Any]) -> torch.device:
    requested = ctx.get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_posenet(device: torch.device) -> PoseNet:
    model = PoseNet().eval().to(device)
    model.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def to_chw(frames_bthwc: torch.Tensor, device: torch.device) -> torch.Tensor:
    return frames_bthwc.permute(0, 1, 4, 2, 3).float().to(device)


def diff_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    h, w = rgb_chw.shape[-2], rgb_chw.shape[-1]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :, : 2 * h2, : 2 * w2]
    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]
    y = (r * 0.299 + g * 0.587 + b * 0.114).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    u_sub = (u[..., 0::2, 0::2] + u[..., 1::2, 0::2] + u[..., 0::2, 1::2] + u[..., 1::2, 1::2]) * 0.25
    v_sub = (v[..., 0::2, 0::2] + v[..., 1::2, 0::2] + v[..., 0::2, 1::2] + v[..., 1::2, 1::2]) * 0.25
    return torch.stack(
        [
            y[..., 0::2, 0::2],
            y[..., 1::2, 0::2],
            y[..., 0::2, 1::2],
            y[..., 1::2, 1::2],
            u_sub,
            v_sub,
        ],
        dim=-3,
    )


def diff_posenet_preprocess(frames: torch.Tensor) -> torch.Tensor:
    b, t, *_ = frames.shape
    flat = frames.reshape(b * t, 3, frames.shape[-2], frames.shape[-1])
    flat = F.interpolate(flat, size=(384, 512), mode="bilinear", align_corners=False)
    yuv = diff_rgb_to_yuv6(flat)
    return yuv.reshape(b, t * 6, yuv.shape[-2], yuv.shape[-1])


def pose_term_from_dist(dist: float) -> float:
    return math.sqrt(max(0.0, 10.0 * float(dist)))


def load_pairs(indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inflated = ensure_q55_inflated(q55_submission_dir=Q55_DIR, cache_dir=DEFAULT_CACHE, file_list=VIDEO_NAMES)
    base = load_raw_pairs_by_indices(raw_dir=inflated, video_names_file=VIDEO_NAMES, sample_indices=indices)
    original = load_original_pairs_by_indices(
        data_dir=VIDEO_DIR,
        video_names_file=VIDEO_NAMES,
        sample_indices=indices,
        batch_size=min(16, max(1, len(indices))),
    )
    return to_chw(base, device), to_chw(original, device)


@torch.no_grad()
def pose_outputs(posenet: PoseNet, pre: torch.Tensor) -> torch.Tensor:
    return posenet(pre)["pose"][..., :6].detach()


def per_sample_pose_dist(got: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (got - target).pow(2).mean(dim=1)


def latest_passing_b1(*, require_hard8: bool = True) -> dict[str, Any] | None:
    candidates = sorted(ORACLE_DIR.glob("*_b1_result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        data = json.loads(path.read_text())
        if not data.get("b1_gate_pass"):
            continue
        if require_hard8 and set(data.get("indices", [])) != set(HARD8):
            continue
        if data.get("b1_gate_pass"):
            data["result_json"] = str(path)
            return data
    return None


class PoseNetPreprocessOracle(Candidate):
    family = "posenet_preprocess_oracle"
    role = "oracle"
    kind = "oracle_only"
    packable = False

    def __init__(self, stage: str):
        self.name = f"posenet_{stage}"
        self.config = {"stage": stage, "packable": False}
        self.stage = stage
        self.result: dict[str, Any] = {}

    def prepare(self, ctx: dict[str, Any]) -> None:
        ORACLE_DIR.mkdir(parents=True, exist_ok=True)

    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        if self.stage == "b1_orig_sanity":
            self.result = self._run_orig_sanity(budget, ctx)
        elif self.stage == "b1_interpolation_curve":
            self.result = self._run_interpolation_curve(budget, ctx)
        elif self.stage == "b1_direct_preprocess":
            self.result = self._run_b1(budget, ctx)
        elif self.stage == "b2_projection_gate":
            self.result = self._run_b2(budget, ctx)
        else:
            raise ValueError(f"unknown oracle stage: {self.stage}")

    def _preprocess_state(
        self,
        indices: list[int],
        device: torch.device,
    ) -> tuple[PoseNet, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        posenet = load_posenet(device)
        base_frames, original_frames = load_pairs(indices, device)
        with torch.no_grad():
            base_pre = posenet.preprocess_input(base_frames).detach()
            target_pre = posenet.preprocess_input(original_frames).detach()
            target_pose = pose_outputs(posenet, target_pre)
            base_pose = pose_outputs(posenet, base_pre)
            base_dist = per_sample_pose_dist(base_pose, target_pose)
        return posenet, base_pre, target_pre, target_pose, base_dist

    def _sample_rows_from_dist(
        self,
        *,
        indices: list[int],
        before_dist: torch.Tensor,
        after_dist: torch.Tensor,
        after_key: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for local, sample_id in enumerate(indices):
            before = float(before_dist[local].item())
            after = float(after_dist[local].item())
            drop = 0.0 if before <= 0 else (before - after) / before
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "baseline_posenet_dist": before,
                    after_key: after,
                    "drop_fraction": drop,
                    "baseline_pose_term": pose_term_from_dist(before),
                    "optimized_pose_term": pose_term_from_dist(after),
                }
            )
        return rows

    def _run_orig_sanity(self, budget: Budget, ctx: dict[str, Any]) -> dict[str, Any]:
        device = resolve_device(ctx)
        indices = list(ctx.get("subset_indices") or HARD8)
        posenet, _base_pre, target_pre, target_pose, base_dist = self._preprocess_state(indices, device)
        with torch.no_grad():
            sanity_pose = pose_outputs(posenet, target_pre)
            sanity_dist = per_sample_pose_dist(sanity_pose, target_pose).detach().cpu()
        base_dist_cpu = base_dist.detach().cpu()
        sanity_pose_term = pose_term_from_dist(float(sanity_dist.mean().item()))
        sample_rows = self._sample_rows_from_dist(
            indices=indices,
            before_dist=base_dist_cpu,
            after_dist=sanity_dist,
            after_key="sanity_posenet_dist",
        )
        gate_pass = sanity_pose_term <= SANITY_POSE_TERM_GATE
        result_json = ORACLE_DIR / f"{ctx['run_id']}_{budget.subset}_b1_orig_sanity_result.json"
        result = {
            "stage": "b1_orig_sanity",
            "row_id": f"{ctx['run_id']}:posenet_b1_orig_sanity:{budget.subset}",
            "indices": indices,
            "baseline_pose_term": pose_term_from_dist(float(base_dist_cpu.mean().item())),
            "sanity_pose_term": sanity_pose_term,
            "pose_delta": sanity_pose_term - pose_term_from_dist(float(base_dist_cpu.mean().item())),
            "sanity_gate_pass": bool(gate_pass),
            "b1_gate_pass": bool(gate_pass),
            "failure_reason": "" if gate_pass else "B1 original-preprocess sanity failed",
            "sample_rows": sample_rows,
        }
        write_json(result_json, result)
        result["result_json"] = str(result_json)
        return result

    def _run_interpolation_curve(self, budget: Budget, ctx: dict[str, Any]) -> dict[str, Any]:
        device = resolve_device(ctx)
        indices = list(ctx.get("subset_indices") or HARD8)
        posenet, base_pre, target_pre, target_pose, base_dist = self._preprocess_state(indices, device)
        base_dist_cpu = base_dist.detach().cpu()
        rows = []
        first_gate_alpha: float | None = None
        sample60_first_gate_alpha: float | None = None
        with torch.no_grad():
            for alpha in B1_INTERPOLATION_ALPHAS:
                candidate = (1.0 - alpha) * base_pre + alpha * target_pre
                pose = pose_outputs(posenet, candidate)
                dist = per_sample_pose_dist(pose, target_pose).detach().cpu()
                pose_term = pose_term_from_dist(float(dist.mean().item()))
                sample60_term = None
                sample_rows = self._sample_rows_from_dist(
                    indices=indices,
                    before_dist=base_dist_cpu,
                    after_dist=dist,
                    after_key="interpolated_posenet_dist",
                )
                for row in sample_rows:
                    if row["sample_id"] == 60:
                        sample60_term = row["optimized_pose_term"]
                        break
                if first_gate_alpha is None and pose_term <= 0.050:
                    first_gate_alpha = alpha
                if sample60_first_gate_alpha is None and sample60_term is not None and sample60_term <= 0.050:
                    sample60_first_gate_alpha = alpha
                rows.append(
                    {
                        "alpha": alpha,
                        "pose_term": pose_term,
                        "sample60_pose_term": sample60_term,
                        "sample_rows": sample_rows,
                    }
                )
        if first_gate_alpha is None:
            conclusion = "gate_not_reached"
        elif first_gate_alpha <= 0.10:
            conclusion = "small_residual_projection_plausible"
        elif first_gate_alpha <= 0.35:
            conclusion = "nontrivial_projection_possible"
        elif first_gate_alpha <= 0.50:
            conclusion = "large_projection_required"
        elif first_gate_alpha < 1.0:
            conclusion = "too_far_for_low_byte_projection"
        else:
            conclusion = "only_original_reaches_gate"
        practical_gate = first_gate_alpha is not None and first_gate_alpha <= 0.50
        if first_gate_alpha is None:
            failure_reason = "B1 interpolation gate not reached"
        elif not practical_gate:
            failure_reason = f"B1 interpolation practical gate failed: {conclusion}"
        else:
            failure_reason = ""
        best_row = min(rows, key=lambda row: row["pose_term"])
        result_json = ORACLE_DIR / f"{ctx['run_id']}_{budget.subset}_b1_interpolation_result.json"
        result = {
            "stage": "b1_interpolation_curve",
            "row_id": f"{ctx['run_id']}:posenet_b1_interpolation_curve:{budget.subset}",
            "indices": indices,
            "alphas": B1_INTERPOLATION_ALPHAS,
            "baseline_pose_term": pose_term_from_dist(float(base_dist_cpu.mean().item())),
            "best_pose_term": float(best_row["pose_term"]),
            "first_gate_alpha": first_gate_alpha,
            "sample60_first_gate_alpha": sample60_first_gate_alpha,
            "interpolation_conclusion": conclusion,
            "b1_gate_pass": first_gate_alpha is not None,
            "practical_projection_gate_pass": bool(practical_gate),
            "failure_reason": failure_reason,
            "curve": rows,
        }
        write_json(result_json, result)
        result["result_json"] = str(result_json)
        return result

    def _run_b1(self, budget: Budget, ctx: dict[str, Any]) -> dict[str, Any]:
        device = resolve_device(ctx)
        indices = HARD3 if budget.subset == "smoke" else list(ctx.get("subset_indices") or HARD8)
        steps = budget.max_steps or (2 if budget.subset == "smoke" else 200)
        lr = float(ctx.get("oracle_lr", 0.005))
        posenet, base_pre, _target_pre, target_pose, base_dist = self._preprocess_state(indices, device)

        opt_pre = torch.nn.Parameter(base_pre.clone())
        opt = torch.optim.AdamW([opt_pre], lr=lr)
        best = {
            "pose_term": pose_term_from_dist(float(base_dist.mean().item())),
            "dist": base_dist.detach().cpu(),
            "tensor": base_pre.detach().cpu(),
            "step": 0,
        }
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            candidate = opt_pre.clamp(0.0, 255.0)
            pose = posenet(candidate)["pose"][..., :6]
            pose_loss = (pose - target_pose).pow(2).mean()
            reg = ((candidate - base_pre) / 64.0).pow(2).mean()
            loss = pose_loss + 0.001 * reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                dist = per_sample_pose_dist(posenet(opt_pre.clamp(0.0, 255.0))["pose"][..., :6], target_pose)
                term = pose_term_from_dist(float(dist.mean().item()))
                if term < best["pose_term"]:
                    best = {"pose_term": term, "dist": dist.detach().cpu(), "tensor": opt_pre.detach().cpu(), "step": step}

        base_dist_cpu = base_dist.detach().cpu()
        sample_rows = self._sample_rows_from_dist(
            indices=indices,
            before_dist=base_dist_cpu,
            after_dist=best["dist"],
            after_key="optimized_posenet_dist",
        )
        drops = [float(row["drop_fraction"]) for row in sample_rows]
        sample60_drop = next((row["drop_fraction"] for row in sample_rows if row["sample_id"] == 60), None)
        avg_drop = float(sum(drops) / max(1, len(drops)))
        pose_term = float(best["pose_term"])
        hard3_gate = sample60_drop is not None and sample60_drop >= 0.80 and avg_drop >= 0.70
        hard8_gate = pose_term <= 0.050
        gate_pass = hard3_gate if set(indices) == set(HARD3) else hard8_gate
        artifact = ORACLE_DIR / f"{ctx['run_id']}_{budget.subset}_b1_pre.pt"
        result_json = ORACLE_DIR / f"{ctx['run_id']}_{budget.subset}_b1_result.json"
        torch.save({"pre_tensor": best["tensor"], "indices": indices, "target_pose": target_pose.detach().cpu()}, artifact)
        result = {
            "stage": "b1",
            "row_id": f"{ctx['run_id']}:posenet_b1_direct_preprocess:{budget.subset}",
            "indices": indices,
            "steps": steps,
            "best_step": int(best["step"]),
            "artifact": str(artifact),
            "baseline_pose_term": pose_term_from_dist(float(base_dist_cpu.mean().item())),
            "optimized_pose_term": pose_term,
            "pose_delta": pose_term - pose_term_from_dist(float(base_dist_cpu.mean().item())),
            "average_pose_drop_fraction": avg_drop,
            "sample60_pose_drop_fraction": sample60_drop,
            "hard3_gate_pass": bool(hard3_gate),
            "hard8_gate_pass": bool(hard8_gate),
            "b1_gate_pass": bool(gate_pass),
            "failure_reason": "" if gate_pass else "B1 gate failed",
            "sample_rows": sample_rows,
        }
        write_json(result_json, result)
        result["result_json"] = str(result_json)
        return result

    def _run_b2(self, budget: Budget, ctx: dict[str, Any]) -> dict[str, Any]:
        parent = latest_passing_b1(require_hard8=True)
        if not parent:
            return {
                "stage": "b2",
                "b2_gate_pass": False,
                "failure_reason": "no passing B1 artifact found",
            }
        device = resolve_device(ctx)
        indices = [int(x) for x in parent["indices"]]
        steps = budget.max_steps or 200
        lr = float(ctx.get("projection_lr", 0.02))
        data = torch.load(parent["artifact"], map_location="cpu")
        target_pre = data["pre_tensor"].to(device)
        base_frames, original_frames = load_pairs(indices, device)
        base_frames_384 = F.interpolate(base_frames.flatten(0, 1), size=(384, 512), mode="bicubic", align_corners=False).reshape(
            base_frames.shape[0], 2, 3, 384, 512
        )
        distortion = build_distortion(device)
        seg_tap = FeatureTap(distortion.segnet, [])
        pose_tap = FeatureTap(distortion.posenet, [])
        try:
            targets = collect_targets(
                distortion=distortion,
                original_cpu=original_frames.detach().cpu().permute(0, 1, 3, 4, 2).contiguous(),
                device=device,
                batch_size=min(8, len(indices)),
                seg_tap=seg_tap,
                pose_tap=pose_tap,
            )
        finally:
            seg_tap.close()
            pose_tap.close()

        rgb = torch.nn.Parameter(base_frames_384.clone())
        opt = torch.optim.AdamW([rgb], lr=lr)
        best: dict[str, Any] | None = None
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            candidate = rgb.clamp(0.0, 255.0)
            pre = diff_posenet_preprocess(candidate)
            pre_loss = (pre - target_pre).pow(2).mean()
            pose = distortion.posenet(pre)["pose"][..., :6]
            pose_loss = (pose - targets["pose"].to(device)).pow(2).mean()
            tv = (candidate[..., 1:, :] - candidate[..., :-1, :]).abs().mean() / 255.0
            loss = pre_loss / 1000.0 + pose_loss + 0.001 * tv
            loss.backward()
            opt.step()
            if step == steps or step % max(1, steps // 5) == 0:
                metrics = self._evaluate_rgb(candidate.detach(), targets, distortion)
                if best is None or metrics["quality"] < best["quality"]:
                    best = {**metrics, "step": step, "rgb": candidate.detach().cpu()}

        if best is None:
            best = self._evaluate_rgb(rgb.detach().clamp(0.0, 255.0), targets, distortion) | {"step": 0, "rgb": rgb.detach().cpu()}

        baseline_pose = float(parent["baseline_pose_term"])
        b1_pose = float(parent["optimized_pose_term"])
        b2_pose = float(best["pose_term"])
        denom = max(1e-9, baseline_pose - b1_pose)
        preserved = (baseline_pose - b2_pose) / denom
        pass_gate = preserved >= 0.60 and best["quality"] <= 0.120
        strong_gate = preserved >= 0.60 and best["quality"] <= 0.095
        result_json = ORACLE_DIR / f"{ctx['run_id']}_{budget.subset}_b2_result.json"
        result = {
            "stage": "b2",
            "indices": indices,
            "steps": steps,
            "best_step": int(best["step"]),
            "parent_row_id": parent.get("row_id", ""),
            "parent_result_json": parent.get("result_json", ""),
            "parent_artifact": parent.get("artifact", ""),
            "quality": float(best["quality"]),
            "segnet_dist": float(best["segnet_dist"]),
            "posenet_dist": float(best["posenet_dist"]),
            "seg_term": float(best["seg_term"]),
            "pose_term": b2_pose,
            "projection_preserved_fraction": float(preserved),
            "b2_gate_pass": bool(pass_gate),
            "b2_strong_gate_pass": bool(strong_gate),
            "failure_reason": "" if pass_gate else "B2 projection failed quality/preservation gate",
        }
        write_json(result_json, result)
        result["result_json"] = str(result_json)
        return result

    @torch.no_grad()
    def _evaluate_rgb(self, frames: torch.Tensor, targets: dict[str, Any], distortion) -> dict[str, float]:
        x = round_ste(frames).clamp(0, 255)
        seg_logits = distortion.segnet(distortion.segnet.preprocess_input(x))
        pose = distortion.posenet(diff_posenet_preprocess(x))["pose"][..., :6]
        target_seg = targets["seg_logits"].to(x.device)
        target_pose = targets["pose"].to(x.device)
        seg_dist = distortion.segnet.compute_distortion(target_seg, seg_logits).mean()
        pose_dist = (pose - target_pose).pow(2).mean()
        seg_f = float(seg_dist.item())
        pose_f = float(pose_dist.item())
        return {
            "segnet_dist": seg_f,
            "posenet_dist": pose_f,
            "seg_term": 100.0 * seg_f,
            "pose_term": pose_term_from_dist(pose_f),
            "quality": quality(seg_f, pose_f),
        }

    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        return PackageInfo(archive_bytes=None, added_bytes=0, projected=True)

    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        base = ctx["base"]
        if self.result and self.stage == "b1_direct_preprocess":
            pose_dist = (float(self.result["optimized_pose_term"]) ** 2) / 10.0
            return Metrics(
                segnet_dist=base["segnet_dist"],
                posenet_dist=pose_dist,
                quality=base["seg_term"] + float(self.result["optimized_pose_term"]),
                seg_term=base["seg_term"],
                pose_term=float(self.result["optimized_pose_term"]),
                rate_term=0.0,
                score=None,
                sample_count=len(self.result.get("indices", [])),
            )
        if self.result and self.stage == "b1_orig_sanity":
            pose_dist = (float(self.result["sanity_pose_term"]) ** 2) / 10.0
            return Metrics(
                segnet_dist=base["segnet_dist"],
                posenet_dist=pose_dist,
                quality=base["seg_term"] + float(self.result["sanity_pose_term"]),
                seg_term=base["seg_term"],
                pose_term=float(self.result["sanity_pose_term"]),
                rate_term=0.0,
                score=None,
                sample_count=len(self.result.get("indices", [])),
            )
        if self.result and self.stage == "b1_interpolation_curve":
            pose_dist = (float(self.result["best_pose_term"]) ** 2) / 10.0
            return Metrics(
                segnet_dist=base["segnet_dist"],
                posenet_dist=pose_dist,
                quality=base["seg_term"] + float(self.result["best_pose_term"]),
                seg_term=base["seg_term"],
                pose_term=float(self.result["best_pose_term"]),
                rate_term=0.0,
                score=None,
                sample_count=len(self.result.get("indices", [])),
            )
        if self.result and self.stage == "b2_projection_gate" and "quality" in self.result:
            return Metrics(
                segnet_dist=float(self.result["segnet_dist"]),
                posenet_dist=float(self.result["posenet_dist"]),
                quality=float(self.result["quality"]),
                seg_term=float(self.result["seg_term"]),
                pose_term=float(self.result["pose_term"]),
                rate_term=0.0,
                score=None,
                sample_count=len(self.result.get("indices", [])),
            )
        return Metrics(
            segnet_dist=base["segnet_dist"],
            posenet_dist=base["posenet_dist"],
            quality=base["quality"],
            seg_term=base["seg_term"],
            pose_term=base["pose_term"],
            rate_term=0.0,
            score=None,
            sample_count=len(ctx.get("subset_indices", [])),
        )

    def decision_row(self, budget: Budget, ctx: dict[str, Any]) -> DecisionRow:
        metrics = self.evaluate(budget.subset, ctx)
        failure = self.result.get("failure_reason", "") if self.result else ""
        if self.stage == "b1_direct_preprocess":
            decision = "diagnostic_only" if not failure else "b1_gate_failed"
            extra_note = "B1 direct preprocessed tensor optimization is oracle-only and cannot promote a candidate."
            parent_row_id = ""
        elif self.stage == "b1_orig_sanity":
            decision = "diagnostic_only" if not failure else "b1_sanity_failed"
            extra_note = "B1 original-preprocessed tensor sanity is oracle-only and verifies target/preprocess/indexing."
            parent_row_id = ""
        elif self.stage == "b1_interpolation_curve":
            decision = "diagnostic_only" if not failure else "b1_interpolation_gate_failed"
            extra_note = "B1 interpolation curve is oracle-only and estimates distance from q55 to original PoseNet basin."
            parent_row_id = ""
        else:
            decision = "diagnostic_only" if not failure else "b2_blocked_or_failed"
            extra_note = "B2 projection is non-packable; a future B3 packable parameterization is required before candidate promotion."
            parent_row_id = self.result.get("parent_row_id", "") if self.result else ""
        row = {
            "run_id": ctx["run_id"],
            "candidate_name": self.name,
            "family": self.family,
            "role": self.role,
            "kind": self.kind,
            "packable": self.packable,
            "config_hash": stable_hash(self.config),
            "novelty_reason": "",
            "subset": budget.subset,
            "round": budget.round,
            "archive_bytes": None,
            "added_bytes": 0,
            "quality": metrics.quality,
            "segnet_dist": metrics.segnet_dist,
            "posenet_dist": metrics.posenet_dist,
            "seg_delta": metrics.seg_term - ctx["base"]["seg_term"],
            "pose_delta": metrics.pose_term - ctx["base"]["pose_term"],
            "byte_delta": None,
            "score": None,
            "score_delta_vs_base": None,
            "dominates_base": False,
            "term_tradeoff": "oracle_nonpackable",
            "decision": decision,
            "failure_reason": failure,
            "row_id": self.result.get("row_id", ""),
            "parent_row_id": parent_row_id,
            "oracle_parent": True,
            "promotion_reason": "capacity_oracle",
            "extra": {
                "packable": False,
                "stage": self.stage,
                "b1_hard3_gate": "sample60_drop>=0.80 and hard3_avg_drop>=0.70",
                "b1_hard8_gate": "pose_term<=0.050",
                "b2_required_before_candidate": "projection_preserves>=0.60_of_B1_pose_improvement",
                "note": extra_note,
                **(self.result or {}),
            },
        }
        classify_against_base(row, ctx["base"])
        row["term_tradeoff"] = "oracle_nonpackable"
        return DecisionRow(**row)


def candidates(round_name: str) -> list[Candidate]:
    if round_name == "smoke":
        return [PoseNetPreprocessOracle("b1_direct_preprocess")]
    if round_name == "hard8":
        return [
            PoseNetPreprocessOracle("b1_orig_sanity"),
            PoseNetPreprocessOracle("b1_interpolation_curve"),
            PoseNetPreprocessOracle("b1_direct_preprocess"),
            PoseNetPreprocessOracle("b2_projection_gate"),
        ]
    if round_name == "strat64":
        return [PoseNetPreprocessOracle("b2_projection_gate")]
    return []
