#!/usr/bin/env python
"""Teacher-distilled inflation Gate 1 family.

Gate 1 is non-packable. It tests whether a slow offline optimizer can create
teacher frames from q55 initialization that beat the hard8/strat64 quality
thresholds before any student or packing work is allowed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.commavq_task.common import FeatureTap, feature_loss, hard_margin_loss, round_ste, tv_loss
from submissions.search_vcm.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm.evaluator import EXPERIMENTS_DIR, classify_against_base, quality, write_json
from submissions.search_vcm.subsets import HARD8
from submissions.tavs_video.common import (
    build_distortion,
    collect_targets,
    ensure_q55_inflated,
    evaluate_frames,
    load_original_pairs_by_indices,
    load_raw_pairs_by_indices,
    to_model_chw,
)


VIDEO_NAMES = REPO_ROOT / "public_test_video_names.txt"
VIDEO_DIR = REPO_ROOT / "videos"
Q55_DIR = REPO_ROOT / "submissions/q55_fp16_pose_int10"
CACHE_DIR = EXPERIMENTS_DIR / "cache"
TEACHER_DIR = EXPERIMENTS_DIR / "teacher_distilled_inflation"
STUDENT_TARGET_BYTES_WEAK = 270_000
HARD8_QUALITY_GATE = 0.080
STRAT64_QUALITY_GATE = 0.110
STUDENT_HARD8_GATE = 0.120
MAX_HARD8_SAMPLE_QUALITY = 0.150
SAMPLE60_MIN_IMPROVEMENT = 0.030


VARIANTS: dict[str, dict[str, Any]] = {
    "t1_direct": {
        "name": "teacher_gate1_t1_direct",
        "label": "T1 q55-init direct RGB/YUV residual",
        "parameterization": "direct_pixels",
        "lr": 1.0,
        "anchor_weight": 0.001,
        "tv_weight": 0.02,
        "high_weight": 0.02,
    },
    "t2_lowmid": {
        "name": "teacher_gate1_t2_lowmid",
        "label": "T2 q55-init low+mid-frequency residual",
        "parameterization": "low_mid_residual",
        "lr": 0.035,
        "low_grid": (72, 96),
        "mid_grid": (144, 192),
        "low_scale": 96.0,
        "mid_scale": 64.0,
        "anchor_weight": 0.0005,
        "tv_weight": 0.03,
        "high_weight": 0.03,
    },
    "t3_continuation": {
        "name": "teacher_gate1_t3_continuation",
        "label": "T3 q55 blended with low-frequency original continuation",
        "parameterization": "direct_pixels",
        "lr": 0.75,
        "blend_original_lowfreq": 0.35,
        "anchor_weight": 0.0005,
        "tv_weight": 0.02,
        "high_weight": 0.02,
    },
}


def resolve_device(ctx: dict[str, Any]) -> torch.device:
    requested = ctx.get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_base_and_original(indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inflated = ensure_q55_inflated(
        q55_submission_dir=Q55_DIR,
        cache_dir=CACHE_DIR / "q55_fp16_pose_int10",
        file_list=VIDEO_NAMES,
    )
    q55_raw = load_raw_pairs_by_indices(raw_dir=inflated, video_names_file=VIDEO_NAMES, sample_indices=indices)
    original = load_original_pairs_by_indices(
        data_dir=VIDEO_DIR,
        video_names_file=VIDEO_NAMES,
        sample_indices=indices,
        batch_size=min(16, max(1, len(indices))),
    )
    return to_model_chw(q55_raw).to(device), to_model_chw(original).to(device)


def highfreq_penalty(frames: torch.Tensor) -> torch.Tensor:
    flat = frames.flatten(0, 1)
    low = F.interpolate(flat, size=(96, 128), mode="area")
    low = F.interpolate(low, size=flat.shape[-2:], mode="bilinear", align_corners=False)
    return ((flat - low) / 255.0).pow(2).mean()


def lowfreq_like(frames: torch.Tensor, *, size: tuple[int, int] = (96, 128)) -> torch.Tensor:
    flat = frames.flatten(0, 1)
    low = F.interpolate(flat, size=size, mode="area")
    low = F.interpolate(low, size=flat.shape[-2:], mode="bilinear", align_corners=False)
    return low.reshape_as(frames)


def frame_distance_metrics(frames: torch.Tensor, base: torch.Tensor, original: torch.Tensor) -> dict[str, float]:
    x = frames.detach().float().cpu()
    b = base.detach().float().cpu()
    o = original.detach().float().cpu()
    q55_delta = x - b
    original_delta = x - o
    return {
        "distance_to_q55_l1": float(q55_delta.abs().mean().item()),
        "distance_to_q55_rmse": float(q55_delta.pow(2).mean().sqrt().item()),
        "distance_to_original_l1": float(original_delta.abs().mean().item()),
        "distance_to_original_rmse": float(original_delta.pow(2).mean().sqrt().item()),
    }


@torch.inference_mode()
def evaluate_frames_detailed(
    *,
    frames: torch.Tensor,
    targets: dict[str, Any],
    distortion: torch.nn.Module,
    batch_size: int,
    sample_ids: list[int],
) -> dict[str, Any]:
    """Evaluate aggregate and per-sample SegNet/PoseNet terms against original targets."""

    total_seg = 0.0
    total_pose = 0.0
    total = 0
    per_sample: list[dict[str, Any]] = []
    device = next(distortion.parameters()).device
    for start in range(0, frames.shape[0], batch_size):
        end = min(frames.shape[0], start + batch_size)
        batch = round_ste(frames[start:end].to(device)).clamp(0, 255)
        rows = torch.arange(start, end, device=device)
        seg_logits = distortion.segnet(distortion.segnet.preprocess_input(batch))
        pose = distortion.posenet(distortion.posenet.preprocess_input(batch))["pose"][..., :6]
        target_seg = targets["seg_logits"].to(device).index_select(0, rows)
        target_pose = targets["pose"].to(device).index_select(0, rows)
        seg_dist = distortion.segnet.compute_distortion(target_seg, seg_logits)
        pose_dist = (pose - target_pose).pow(2).mean(dim=1)
        total_seg += float(seg_dist.sum().item())
        total_pose += float(pose_dist.sum().item())
        total += end - start
        for offset in range(end - start):
            seg_i = float(seg_dist[offset].item())
            pose_i = float(pose_dist[offset].item())
            per_sample.append(
                {
                    "sample_id": int(sample_ids[start + offset]),
                    "segnet_dist": seg_i,
                    "posenet_dist": pose_i,
                    "seg_term": 100.0 * seg_i,
                    "pose_term": math.sqrt(max(0.0, 10.0 * pose_i)),
                    "quality_i": 100.0 * seg_i + math.sqrt(max(0.0, 10.0 * pose_i)),
                }
            )
    seg = total_seg / max(1, total)
    pose_value = total_pose / max(1, total)
    return {
        "segnet_dist": seg,
        "posenet_dist": pose_value,
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(max(0.0, 10.0 * pose_value)),
        "quality": quality(seg, pose_value),
        "per_sample": per_sample,
    }


def jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in config.items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


class TeacherParameterization:
    def __init__(self, *, variant: dict[str, Any], base_frames: torch.Tensor, original_frames: torch.Tensor) -> None:
        self.variant = variant
        self.base_frames = base_frames
        self.original_frames = original_frames
        self.kind = str(variant["parameterization"])
        if self.kind == "direct_pixels":
            init = base_frames.detach().clone()
            blend = float(variant.get("blend_original_lowfreq", 0.0))
            if blend > 0:
                init = ((1.0 - blend) * init + blend * lowfreq_like(original_frames)).clamp(0.0, 255.0)
            self.direct = torch.nn.Parameter(init)
            self._params = [self.direct]
        elif self.kind == "low_mid_residual":
            n = base_frames.shape[0]
            channels = base_frames.shape[1] * base_frames.shape[2]
            low_h, low_w = variant["low_grid"]
            mid_h, mid_w = variant["mid_grid"]
            self.low = torch.nn.Parameter(torch.zeros(n, channels, low_h, low_w, device=base_frames.device))
            self.mid = torch.nn.Parameter(torch.zeros(n, channels, mid_h, mid_w, device=base_frames.device))
            self._params = [self.low, self.mid]
        else:
            raise ValueError(f"unknown teacher parameterization: {self.kind}")

    def parameters(self) -> list[torch.nn.Parameter]:
        return self._params

    def frames(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        if self.kind == "direct_pixels":
            x = self.direct if rows is None else self.direct.index_select(0, rows)
            return x.clamp(0.0, 255.0)
        base = self.base_frames if rows is None else self.base_frames.index_select(0, rows)
        n, t, c, h, w = base.shape
        low = self.low if rows is None else self.low.index_select(0, rows)
        mid = self.mid if rows is None else self.mid.index_select(0, rows)
        low = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).reshape(n, t, c, h, w)
        mid = F.interpolate(mid, size=(h, w), mode="bilinear", align_corners=False).reshape(n, t, c, h, w)
        residual = float(self.variant["low_scale"]) * low.tanh() + float(self.variant["mid_scale"]) * mid.tanh()
        return (base + residual).clamp(0.0, 255.0)

    @torch.no_grad()
    def clamp_(self) -> None:
        if self.kind == "direct_pixels":
            self.direct.clamp_(0.0, 255.0)
        else:
            # Keep tanh residual controls in a numerically useful range.
            self.low.clamp_(-3.0, 3.0)
            self.mid.clamp_(-3.0, 3.0)


class TeacherGate1Oracle(Candidate):
    family = "teacher_distilled_inflation"
    role = "oracle"
    kind = "oracle_only"
    packable = False
    novelty_reason = (
        "offline_solver_distills_teacher_frames_for_future_tiny_inflater;"
        "not_mask_perturbation_not_local_q55_control_not_pose_projection"
    )

    def __init__(self, variant_id: str = "t1_direct") -> None:
        self.variant_id = variant_id
        self.variant = VARIANTS[variant_id]
        self.name = str(self.variant["name"])
        self.config = {
            "config_id": self.name,
            "stage": "gate1_teacher_inversion",
            "variant_id": variant_id,
            "variant_label": self.variant["label"],
            "parameterization": self.variant["parameterization"],
            "packable": False,
            "novelty_reason": self.novelty_reason,
            "optimizer": "adamw_per_sample_teacher_frames",
            "student_target_archive_bytes": STUDENT_TARGET_BYTES_WEAK,
            "hard8_teacher_quality_gate": HARD8_QUALITY_GATE,
            "student_hard8_gate_if_teacher_passes": STUDENT_HARD8_GATE,
        }
        for key in (
            "lr",
            "low_grid",
            "mid_grid",
            "low_scale",
            "mid_scale",
            "blend_original_lowfreq",
            "anchor_weight",
            "tv_weight",
            "high_weight",
        ):
            if key in self.variant:
                self.config[key] = self.variant[key]
        self.result: dict[str, Any] = {}

    def prepare(self, ctx: dict[str, Any]) -> None:
        TEACHER_DIR.mkdir(parents=True, exist_ok=True)

    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        self.result = self._run_gate1(budget, ctx)

    def _run_gate1(self, budget: Budget, ctx: dict[str, Any]) -> dict[str, Any]:
        device = resolve_device(ctx)
        indices = list(ctx.get("subset_indices") or HARD8)
        steps = budget.max_steps or (3 if budget.subset == "smoke" else 300)
        default_eval_every = 25 if steps <= 300 else 100
        requested_eval_every = int(ctx.get("teacher_eval_every") or 0)
        eval_every = max(1, requested_eval_every or default_eval_every)
        requested_batch_size = int(ctx.get("teacher_batch_size") or 0)
        batch_size = min(requested_batch_size or 4, len(indices))
        requested_lr = float(ctx.get("teacher_lr") or 0.0)
        lr = requested_lr or float(self.variant["lr"])
        pose_feature_names = [x for x in str(ctx.get("teacher_pose_features", "summarizer")).split(",") if x]
        seg_feature_names = [x for x in str(ctx.get("teacher_seg_features", "")).split(",") if x]

        base_frames, original_frames = load_base_and_original(indices, device)
        distortion = build_distortion(device)
        pose_tap = FeatureTap(distortion.posenet, pose_feature_names)
        seg_tap = FeatureTap(distortion.segnet, seg_feature_names)
        try:
            targets = collect_targets(
                distortion=distortion,
                original_cpu=original_frames.detach().cpu().permute(0, 1, 3, 4, 2).contiguous(),
                device=device,
                batch_size=max(1, min(batch_size, 4)),
                seg_tap=seg_tap,
                pose_tap=pose_tap,
            )
        finally:
            seg_tap.close()
            pose_tap.close()

        baseline = evaluate_frames_detailed(
            frames=base_frames.detach().cpu(),
            targets=targets,
            distortion=distortion,
            batch_size=batch_size,
            sample_ids=indices,
        )
        original_upper = evaluate_frames_detailed(
            frames=original_frames.detach().cpu(),
            targets=targets,
            distortion=distortion,
            batch_size=batch_size,
            sample_ids=indices,
        )

        teacher = TeacherParameterization(variant=self.variant, base_frames=base_frames, original_frames=original_frames)
        opt = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=0.0)
        init_frames = teacher.frames().detach()
        init_metrics = evaluate_frames_detailed(
            frames=init_frames.cpu(),
            targets=targets,
            distortion=distortion,
            batch_size=batch_size,
            sample_ids=indices,
        )
        init_metrics.update(frame_distance_metrics(init_frames, base_frames, original_frames))
        best = {**init_metrics, "step": 0}
        best_frames = init_frames.detach().cpu()
        history: list[dict[str, Any]] = [{"step": 0, "phase": "init", **init_metrics}]

        n = len(indices)
        pose_tap_train = FeatureTap(distortion.posenet, pose_feature_names)
        seg_tap_train = FeatureTap(distortion.segnet, seg_feature_names)
        try:
            for step in range(1, steps + 1):
                if batch_size >= n:
                    rows = torch.arange(n, device=device)
                else:
                    rows = torch.randint(0, n, (batch_size,), device=device)
                frames = round_ste(teacher.frames(rows)).clamp(0.0, 255.0)
                target_arg = targets["seg_argmax"].to(device).index_select(0, rows)
                target_prob = targets["seg_prob"].to(device).index_select(0, rows)
                target_pose = targets["pose"].to(device).index_select(0, rows)

                seg_tap_train.clear()
                pose_tap_train.clear()
                seg_logits = distortion.segnet(distortion.segnet.preprocess_input(frames))
                pose = distortion.posenet(distortion.posenet.preprocess_input(frames))["pose"][..., :6]
                ce = F.cross_entropy(seg_logits, target_arg)
                kl = F.kl_div(F.log_softmax(seg_logits, dim=1), target_prob, reduction="none").sum(dim=1).mean()
                margin = hard_margin_loss(seg_logits, target_arg, margin=2.0)
                pose_mse = (pose - target_pose).pow(2).mean()
                pfeat = feature_loss(pose_tap_train.features, targets["pose_features"], rows)
                sfeat = feature_loss(seg_tap_train.features, targets["seg_features"], rows)

                tv = tv_loss(frames)
                high = highfreq_penalty(frames)
                anchor = F.huber_loss(frames / 255.0, base_frames.index_select(0, rows) / 255.0, delta=0.05)
                original_distance = F.huber_loss(
                    frames / 255.0,
                    original_frames.index_select(0, rows) / 255.0,
                    delta=0.05,
                )
                loss = (
                    1.0 * ce
                    + 1.0 * kl
                    + 2.0 * margin
                    + 6.0 * pose_mse
                    + 1.0 * pfeat
                    + 0.5 * sfeat
                    + float(self.variant.get("tv_weight", 0.02)) * tv
                    + float(self.variant.get("high_weight", 0.02)) * high
                    + float(self.variant.get("anchor_weight", 0.001)) * anchor
                    + 0.0005 * original_distance
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
                opt.step()
                teacher.clamp_()

                if step == 1 or step % eval_every == 0 or step == steps:
                    full_frames = teacher.frames().detach()
                    metrics = evaluate_frames_detailed(
                        frames=full_frames.cpu(),
                        targets=targets,
                        distortion=distortion,
                        batch_size=batch_size,
                        sample_ids=indices,
                    )
                    metrics.update(frame_distance_metrics(full_frames, base_frames, original_frames))
                    row = {
                        "step": int(step),
                        "loss": float(loss.detach().item()),
                        "ce": float(ce.detach().item()),
                        "kl": float(kl.detach().item()),
                        "margin": float(margin.detach().item()),
                        "pose_mse": float(pose_mse.detach().item()),
                        "pose_feature_loss": float(pfeat.detach().item()),
                        "seg_feature_loss": float(sfeat.detach().item()),
                        "tv": float(tv.detach().item()),
                        "highfreq": float(high.detach().item()),
                        "anchor_to_q55": float(anchor.detach().item()),
                        "distance_to_original_loss": float(original_distance.detach().item()),
                        **metrics,
                    }
                    history.append(row)
                    if metrics["quality"] < best["quality"]:
                        best = {**metrics, "step": int(step)}
                        best_frames = full_frames.detach().cpu().contiguous()
        finally:
            pose_tap_train.close()
            seg_tap_train.close()

        if budget.subset == "hard8":
            gate = HARD8_QUALITY_GATE
        elif budget.subset == "strat64":
            gate = STRAT64_QUALITY_GATE
        else:
            gate = 0.300
        gate_checks = self._gate_checks(history=history, baseline=baseline, best=best, subset=budget.subset, sample_ids=indices)
        gate_pass = bool(gate_checks["final_gate_pass"])
        artifact = TEACHER_DIR / f"{ctx['run_id']}_{self.name}_{budget.subset}_teacher_frames.pt"
        result_json = TEACHER_DIR / f"{ctx['run_id']}_{self.name}_{budget.subset}_teacher_result.json"
        torch.save(
            {
                "teacher_frames": best_frames,
                "indices": indices,
                "best_metrics": best,
                "baseline": baseline,
                "original_upper": original_upper,
                "config": jsonable_config(self.config),
                "gate_checks": gate_checks,
            },
            artifact,
        )
        result = {
            "stage": "gate1_teacher_inversion",
            "variant_id": self.variant_id,
            "variant_label": self.variant["label"],
            "row_id": f"{ctx['run_id']}:{self.name}:{budget.subset}",
            "indices": indices,
            "steps": int(steps),
            "best_step": int(best["step"]),
            "quality_gate": gate,
            "gate_pass": bool(gate_pass),
            "baseline": baseline,
            "original_upper": original_upper,
            "best": best,
            "gate_checks": gate_checks,
            "history": history,
            "artifact": str(artifact),
            "failure_reason": "" if gate_pass else gate_checks["failure_reason"],
        }
        write_json(result_json, result)
        result["result_json"] = str(result_json)
        return result

    def _gate_checks(
        self,
        *,
        history: list[dict[str, Any]],
        baseline: dict[str, Any],
        best: dict[str, Any],
        subset: str,
        sample_ids: list[int],
    ) -> dict[str, Any]:
        def first_at_or_after(step: int) -> dict[str, Any] | None:
            candidates = [row for row in history if int(row["step"]) >= step]
            return min(candidates, key=lambda row: int(row["step"])) if candidates else None

        baseline_quality = float(baseline["quality"])
        baseline_by_id = {int(row["sample_id"]): row for row in baseline.get("per_sample", [])}
        best_by_id = {int(row["sample_id"]): row for row in best.get("per_sample", [])}
        sample60_baseline = float(baseline_by_id.get(60, {}).get("quality_i", 0.0))
        sample60_best = float(best_by_id.get(60, {}).get("quality_i", sample60_baseline))
        sample60_improvement = sample60_baseline - sample60_best if 60 in sample_ids else None
        max_sample_quality = max((float(row["quality_i"]) for row in best.get("per_sample", [])), default=float(best["quality"]))

        step250 = first_at_or_after(250)
        step750 = first_at_or_after(750)
        step1500 = first_at_or_after(1500)
        checks = {
            "step250_quality_improvement": None if step250 is None else baseline_quality - float(step250["quality"]),
            "step250_pass": None if step250 is None else (baseline_quality - float(step250["quality"]) >= 0.030),
            "step750_quality": None if step750 is None else float(step750["quality"]),
            "step750_pass": None if step750 is None else (float(step750["quality"]) <= 0.180),
            "step1500_quality": None if step1500 is None else float(step1500["quality"]),
            "step1500_pass": None if step1500 is None else (float(step1500["quality"]) <= 0.110),
            "sample60_quality_baseline": sample60_baseline if 60 in sample_ids else None,
            "sample60_quality_best": sample60_best if 60 in sample_ids else None,
            "sample60_quality_improvement": sample60_improvement,
            "sample60_materially_improved": True
            if 60 not in sample_ids
            else bool(sample60_improvement is not None and sample60_improvement >= SAMPLE60_MIN_IMPROVEMENT),
            "max_sample_quality": max_sample_quality,
            "max_sample_quality_pass": bool(max_sample_quality <= MAX_HARD8_SAMPLE_QUALITY),
            "quality_gate_pass": bool(float(best["quality"]) <= (HARD8_QUALITY_GATE if subset == "hard8" else STRAT64_QUALITY_GATE if subset == "strat64" else 0.300)),
        }
        if subset == "hard8":
            final_gate_pass = (
                checks["quality_gate_pass"]
                and checks["sample60_materially_improved"]
                and checks["max_sample_quality_pass"]
            )
        elif subset == "strat64":
            final_gate_pass = checks["quality_gate_pass"]
        else:
            final_gate_pass = checks["quality_gate_pass"]
        failures = []
        if not checks["quality_gate_pass"]:
            failures.append("teacher quality gate failed")
        if subset == "hard8" and not checks["sample60_materially_improved"]:
            failures.append("sample 60 did not materially improve")
        if subset == "hard8" and not checks["max_sample_quality_pass"]:
            failures.append("hard8 worst-sample quality remained above tail cap")
        return {
            **checks,
            "final_gate_pass": bool(final_gate_pass),
            "failure_reason": "; ".join(failures) or "",
        }

    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        return PackageInfo(archive_bytes=None, added_bytes=0, projected=True)

    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        base = ctx["base"]
        best = self.result.get("best") if self.result else None
        if best:
            return Metrics(
                segnet_dist=float(best["segnet_dist"]),
                posenet_dist=float(best["posenet_dist"]),
                quality=float(best["quality"]),
                seg_term=float(best["seg_term"]),
                pose_term=float(best["pose_term"]),
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
        decision = "diagnostic_only" if not failure else "gate1_failed"
        row = {
            "run_id": ctx["run_id"],
            "candidate_name": self.name,
            "family": self.family,
            "role": self.role,
            "kind": self.kind,
            "packable": self.packable,
            "config_hash": stable_hash(self.config),
            "novelty_reason": self.novelty_reason,
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
            "parent_row_id": "",
            "oracle_parent": True,
            "promotion_reason": "capacity_oracle",
            "extra": {
                "packable": False,
                "stage": "gate1_teacher_inversion",
                "hard8_gate": f"teacher quality <= {HARD8_QUALITY_GATE:.3f}",
                "strat64_gate": f"teacher quality <= {STRAT64_QUALITY_GATE:.3f}",
                "next_gate": "student hard8 quality <=0.120 only if Gate 1 passes",
                "variant_id": self.variant_id,
                "variant_label": self.variant["label"],
                "note": "Gate 1 teacher frames are non-packable distillation targets, not a submission candidate.",
                **(self.result or {}),
            },
        }
        classify_against_base(row, ctx["base"])
        row["term_tradeoff"] = "oracle_nonpackable"
        return DecisionRow(**row)


def candidates(round_name: str) -> list[Candidate]:
    if round_name == "smoke":
        return [TeacherGate1Oracle("t1_direct")]
    if round_name in {"hard8", "strat64"}:
        return [TeacherGate1Oracle("t1_direct"), TeacherGate1Oracle("t2_lowmid"), TeacherGate1Oracle("t3_continuation")]
    return []
