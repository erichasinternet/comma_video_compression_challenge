#!/usr/bin/env python
"""Search family: PR #62 low-rate mask + qpose14 distillation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.candidate_api import Budget, Candidate, DecisionRow, Metrics, PackageInfo, stable_hash
from submissions.search_vcm_v2.evaluator import QPOSE14_LEDGER, QPOSE14_SUMMARY, load_jsonl, qpose14_reference_summary, rate_term
from submissions.search_vcm_v2.families.factorized_exactmask_pose_tokens import pred_frames_to_bt_hw3, rgb_to_yuv6_diff
from submissions.search_vcm_v2.families.lowmask_data import (
    FP4_ARCHIVE,
    FP4_LEDGER,
    FP4_SUMMARY,
    LOWMASK_CACHE,
    archive_audit,
    evaluate_fp4_per_sample,
    materialize_lowmask_subset,
    write_fp4_summary,
)
from submissions.search_vcm_v2.families.lowmask_renderer import build_lowmask_renderer
from submissions.search_vcm_v2.families.pack_lowmask_renderer import estimate_lowmask_renderer_bytes
from submissions.search_vcm_v2.families.pack_pose_tokens import estimate_pose_token_bytes
from submissions.search_vcm_v2.families.qpose14_data import load_original_subset, select_torch_device


SINGLE_MEMBER_OVERHEAD_ESTIMATE = 512


class LowmaskQposeDistillCandidate(Candidate):
    family = "lowmask_qpose_distill"
    role = "exploratory_candidate"
    kind = "capacity_or_compressed_candidate"
    packable = True

    def __init__(self, *, variant: str, pose_source: str = "qpose14") -> None:
        self.variant = variant
        self.pose_source = pose_source
        self.name = f"lowmask_{variant.lower()}_{pose_source}"
        self.config = {"config_id": self.name, "variant": variant, "pose_source": pose_source}
        self._package: PackageInfo | None = None
        self._metrics: Metrics | None = None
        self._decision = "not_run"
        self._failure_reason = ""
        self._extra: dict[str, Any] = {}

    def prepare(self, ctx: dict[str, Any]) -> None:
        if not QPOSE14_SUMMARY.exists() or not QPOSE14_LEDGER.exists():
            raise RuntimeError("qpose14 baseline ledger is required before lowmask runs; run qpose14_baseline smoke first")
        self._extra["qpose14_summary"] = str(QPOSE14_SUMMARY)
        self._extra["qpose14_ledger"] = str(QPOSE14_LEDGER)
        self._extra["fp4_archive_present"] = FP4_ARCHIVE.exists()

    def train_round(self, budget: Budget, ctx: dict[str, Any]) -> None:
        if budget.round == "smoke":
            model = build_lowmask_renderer("L32")
            mask = torch.randint(0, 5, (2, 32, 48), dtype=torch.long)
            pose6 = torch.zeros(2, 6)
            z_pose = torch.zeros(2, model.z_pose_dim)
            with torch.no_grad():
                frame1, frame2 = model(mask, pose6, z_pose)
            self._extra["dummy_forward"] = {
                "frame1_shape": list(frame1.shape),
                "frame2_shape": list(frame2.shape),
                "config": model.config(),
            }
            self._decision = "smoke_pass"
            return
        if budget.round == "gate0_audit":
            self._run_gate0_audit(ctx)
            return
        if budget.round == "packability":
            self._decision = "packability_estimated"
            return
        if budget.round == "hard8_capacity":
            if not FP4_SUMMARY.exists() or not FP4_LEDGER.exists() or not LOWMASK_CACHE.exists():
                self._decision = "blocked_missing_gate0_audit"
                self._failure_reason = "Gate 0 audit and lowmask/qpose caches are required before lowmask hard8 capacity training."
                return
            if budget.max_steps <= 0:
                self._decision = "blocked_no_training_budget"
                self._failure_reason = "hard8_capacity requires --max-steps > 0."
                return
            self._run_hard8_capacity(budget, ctx)
            return
        if budget.round in {"hard8_compressed", "strat64", "full600"}:
            self._decision = "blocked_no_capacity_checkpoint"
            self._failure_reason = "Compressed/final rounds require a promoted lowmask hard8_capacity checkpoint."
            return

    def _run_gate0_audit(self, ctx: dict[str, Any]) -> None:
        device = ctx.get("device", "auto")
        audit = archive_audit(FP4_ARCHIVE)
        subset_ids = [59, 60, 62, 56, 57, 58, 61, 63]
        materialize_lowmask_subset("hard8_capacity", subset_ids, device=device)
        batch_size = 2 if device == "cpu" else 4
        rows = evaluate_fp4_per_sample(device=device, batch_size=batch_size)
        summary = write_fp4_summary(rows)
        self._metrics = Metrics(
            segnet_dist=float(summary["segnet_dist"]),
            posenet_dist=float(summary["posenet_dist"]),
            quality=float(summary["quality"]),
            seg_term=float(summary["seg_term"]),
            pose_term=float(summary["pose_term"]),
            score=float(summary["score"]),
            sample_count=int(summary.get("row_count", 0)),
        )
        self._package = PackageInfo(
            archive_bytes=int(summary["archive_bytes"]),
            added_bytes=0,
            payload_breakdown={k: int(v) for k, v in audit["payload_breakdown"].items()},
            archive_path=str(FP4_ARCHIVE),
            archive_sha=str(audit["archive_sha256"]),
            projected=False,
        )
        self._extra["gate0_audit"] = {
            "archive": audit,
            "summary": summary,
            "fp4_ledger": str(FP4_LEDGER),
            "fp4_summary": str(FP4_SUMMARY),
            "lowmask_cache": str(LOWMASK_CACHE),
        }
        self._decision = "gate0_audit_complete"

    def _run_hard8_capacity(self, budget: Budget, ctx: dict[str, Any]) -> None:
        subset_ids = [int(x) for x in ctx.get("subset_indices", [])]
        data = materialize_lowmask_subset(budget.subset, subset_ids, device=ctx.get("device", "auto"))
        device = select_torch_device(ctx.get("device", "auto"))
        model = build_lowmask_renderer("capacity").to(device)
        z_pose = torch.nn.Parameter(torch.zeros(len(subset_ids), model.z_pose_dim, device=device))
        optimizer = torch.optim.AdamW(list(model.parameters()) + [z_pose], lr=2e-4, weight_decay=1e-5)

        masks = data["lowmask_class"].to(device).long()
        pose_key = "qpose_pose6" if self.pose_source == "qpose14" else "fp4_pose6"
        pose6 = data[pose_key].to(device).float()
        target1 = data["qpose_frame1"].to(device).float()
        target2 = data["qpose_frame2"].to(device).float()
        original = load_original_subset(budget.subset, subset_ids, device="cpu").to(device).float()
        baseline_samples = ctx.get("baseline_samples") or []
        baseline_by_id = {int(row["sample_id"]): row for row in baseline_samples}

        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        distortion = DistortionNet().eval().to(device)
        distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
        for param in distortion.parameters():
            param.requires_grad_(False)

        with torch.no_grad():
            target_pose_out, target_seg_logits = distortion(original)
            target_pose_vec = target_pose_out["pose"][..., :6].detach()
            target_seg_logits = target_seg_logits.detach()
            target_seg_argmax = target_seg_logits.argmax(dim=1).detach()
            target_pose_summary = self._posenet_summary(distortion, original).detach()

        stage_a = min(1000, max(1, budget.max_steps // 5))
        stage_c = min(1000, max(0, budget.max_steps // 5))
        stage_b = max(0, budget.max_steps - stage_a - stage_c)
        if budget.max_steps <= 20:
            stage_a, stage_b, stage_c = budget.max_steps, 0, 0
        schedule = [("teacher", stage_a), ("task", stage_b), ("polish", stage_c)]
        eval_every = max(1, min(100, budget.max_steps // 10 or 1))
        checkpoint_dir = QPOSE14_SUMMARY.parent / "lowmask_checkpoints" / f"{ctx['run_id']}_{self.name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_quality: dict[str, Any] | None = None
        best_gate: dict[str, Any] | None = None
        best_sample60: dict[str, Any] | None = None
        history = []

        def save_checkpoint(name: str, step: int) -> str:
            path = checkpoint_dir / f"{name}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "z_pose": z_pose.detach().cpu(),
                    "config": model.config(),
                    "sample_ids": subset_ids,
                    "pose_source": self.pose_source,
                },
                path,
            )
            return str(path)

        def eval_checkpoint(stage_name: str, step: int) -> dict[str, Any]:
            with torch.no_grad():
                pred1, pred2 = model(masks, pose6, z_pose)
                pred_bt = pred_frames_to_bt_hw3(pred1, pred2)
                pred_pose_out, pred_seg_logits = distortion(pred_bt)
                pose_dist = (pred_pose_out["pose"][..., :6] - target_pose_vec).pow(2).mean(dim=1)
                seg_dist = (pred_seg_logits.argmax(dim=1) != target_seg_argmax).float().mean(dim=(1, 2))
                per_sample = []
                for i, sample_id in enumerate(subset_ids):
                    seg_term = 100.0 * float(seg_dist[i].item())
                    pose_term = float(torch.sqrt(torch.clamp(10.0 * pose_dist[i], min=0.0)).item())
                    q = seg_term + pose_term
                    base = baseline_by_id.get(sample_id, {})
                    per_sample.append(
                        {
                            "sample_id": sample_id,
                            "segnet_dist": float(seg_dist[i].item()),
                            "posenet_dist": float(pose_dist[i].item()),
                            "seg_term": seg_term,
                            "pose_term": pose_term,
                            "quality": q,
                            "quality_delta_vs_qpose14": q - float(base.get("qpose14_quality", 0.0)),
                            "pose_delta_vs_qpose14": pose_term - float(base.get("qpose14_pose_term", 0.0)),
                        }
                    )
                seg_mean = float(seg_dist.mean().item())
                pose_mean = float(pose_dist.mean().item())
                seg_term = 100.0 * seg_mean
                pose_term = float(torch.sqrt(torch.tensor(10.0 * pose_mean)).item())
                q_mean = seg_term + pose_term
                baseline_quality = float(ctx["baseline"]["quality"])
                max_sample = max(per_sample, key=lambda row: row["quality"])
                sample60 = next((row for row in per_sample if row["sample_id"] == 60), None)
                qpose_l1 = 0.5 * ((pred1 - target1).abs().mean() + (pred2 - target2).abs().mean())
                qpose_rmse = torch.sqrt(0.5 * ((pred1 - target1).pow(2).mean() + (pred2 - target2).pow(2).mean()))
                return {
                    "stage": stage_name,
                    "step": step,
                    "quality": q_mean,
                    "segnet_dist": seg_mean,
                    "posenet_dist": pose_mean,
                    "seg_term": seg_term,
                    "pose_term": pose_term,
                    "quality_delta_vs_qpose14": q_mean - baseline_quality,
                    "max_sample_quality": float(max_sample["quality"]),
                    "max_sample_delta_vs_qpose14": float(max(row["quality_delta_vs_qpose14"] for row in per_sample)),
                    "sample60_pose_term": float(sample60["pose_term"]) if sample60 else None,
                    "sample60_pose_delta_vs_qpose14": float(sample60["pose_delta_vs_qpose14"]) if sample60 else None,
                    "distance_to_qpose_l1": float(qpose_l1.item()),
                    "distance_to_qpose_rmse": float(qpose_rmse.item()),
                    "per_sample": per_sample,
                }

        global_step = 0
        tail_indices = torch.tensor([i for i, sample_id in enumerate(subset_ids) if sample_id in {59, 60, 62}], device=device, dtype=torch.long)
        for stage_name, stage_steps in schedule:
            for _ in range(stage_steps):
                global_step += 1
                if stage_name == "polish" and tail_indices.numel() > 0 and torch.rand((), device=device) < 0.7:
                    idx = tail_indices[torch.randint(0, tail_indices.numel(), (1,), device=device)]
                else:
                    idx = torch.randint(0, len(subset_ids), (1,), device=device)
                pred1, pred2 = model(masks[idx], pose6[idx], z_pose[idx])
                loss = self._stage_loss(
                    stage_name,
                    distortion,
                    pred1,
                    pred2,
                    target1[idx],
                    target2[idx],
                    target_pose_vec[idx],
                    target_seg_logits[idx],
                    target_seg_argmax[idx],
                    target_pose_summary[idx],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if global_step == 1 or global_step % eval_every == 0 or global_step == budget.max_steps:
                    item = eval_checkpoint(stage_name, global_step)
                    history.append(item)
                    if best_quality is None or item["quality"] < best_quality["quality"]:
                        item["checkpoint"] = save_checkpoint("best_quality", global_step)
                        best_quality = item
                    gate_score = (
                        max(0.0, item["quality"] - (float(ctx["baseline"]["quality"]) + 0.020))
                        + max(0.0, item["max_sample_delta_vs_qpose14"] - 0.040)
                        + max(0.0, (item["sample60_pose_delta_vs_qpose14"] or 0.0) - 0.020)
                    )
                    if best_gate is None or gate_score < best_gate["gate_score"]:
                        best_gate = {**item, "gate_score": gate_score, "checkpoint": save_checkpoint("best_gate", global_step)}
                    if item["sample60_pose_term"] is not None and (best_sample60 is None or item["sample60_pose_term"] < best_sample60["sample60_pose_term"]):
                        best_sample60 = {**item, "checkpoint": save_checkpoint("best_sample60_pose", global_step)}

        selected = best_gate or best_quality
        gate_pass = bool(
            selected
            and selected["quality"] <= float(ctx["baseline"]["quality"]) + 0.020
            and selected["max_sample_delta_vs_qpose14"] <= 0.040
            and (selected["sample60_pose_delta_vs_qpose14"] is None or selected["sample60_pose_delta_vs_qpose14"] <= 0.020)
        )
        strong_pass = bool(selected and selected["quality"] <= float(ctx["baseline"]["quality"]))
        if selected:
            self._metrics = Metrics(
                segnet_dist=selected["segnet_dist"],
                posenet_dist=selected["posenet_dist"],
                quality=selected["quality"],
                seg_term=selected["seg_term"],
                pose_term=selected["pose_term"],
                score=None,
                sample_count=len(subset_ids),
                per_sample=selected["per_sample"],
            )
        self._extra["lowmask_capacity"] = {
            "lowmask_cache": str(LOWMASK_CACHE),
            "sample_ids": subset_ids,
            "device": str(device),
            "max_steps": budget.max_steps,
            "schedule": [{"stage": name, "steps": steps} for name, steps in schedule],
            "best_quality": best_quality,
            "best_gate": best_gate,
            "best_sample60_pose": best_sample60,
            "history": history,
            "gate_pass": gate_pass,
            "strong_pass": strong_pass,
            "pose_source": self.pose_source,
        }
        if gate_pass:
            self._decision = "promote_packability"
        elif budget.max_steps < 100:
            self._decision = "diagnostic_gate1_smoke_complete"
        else:
            self._decision = "close_lowmask_qpose"

    def _posenet_summary(self, distortion, batch_bt_hw3: torch.Tensor) -> torch.Tensor:
        b, t, *_ = batch_bt_hw3.shape
        x = batch_bt_hw3.permute(0, 1, 4, 2, 3).contiguous().float()
        x = x.reshape(b * t, 3, x.shape[-2], x.shape[-1])
        x = F.interpolate(x, size=(384, 512), mode="bilinear", align_corners=False)
        posenet_in = rgb_to_yuv6_diff(x).reshape(b, t * 6, 192, 256)
        vision = distortion.posenet.vision((posenet_in - distortion.posenet._mean) / distortion.posenet._std)
        return distortion.posenet.summarizer(vision)

    def _stage_loss(
        self,
        stage_name: str,
        distortion,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor,
        target_pose_vec: torch.Tensor,
        target_seg_logits: torch.Tensor,
        target_seg_argmax: torch.Tensor,
        target_pose_summary: torch.Tensor,
    ) -> torch.Tensor:
        teacher = F.huber_loss(pred1, target1, delta=8.0) + F.huber_loss(pred2, target2, delta=8.0)
        low_teacher = F.huber_loss(F.avg_pool2d(pred1, 8), F.avg_pool2d(target1, 8), delta=4.0) + F.huber_loss(
            F.avg_pool2d(pred2, 8), F.avg_pool2d(target2, 8), delta=4.0
        )
        if stage_name == "teacher":
            return teacher + 0.2 * low_teacher

        pred_bt = pred_frames_to_bt_hw3(pred1, pred2)
        pred_chw = pred_bt.permute(0, 1, 4, 2, 3).contiguous().float()
        b, t, c, h, w = pred_chw.shape
        flat = pred_chw.reshape(b * t, c, h, w)
        flat_small = F.interpolate(flat, size=(384, 512), mode="bilinear", align_corners=False)
        posenet_in = rgb_to_yuv6_diff(flat_small).reshape(b, t * 6, 192, 256)
        pred_vision = distortion.posenet.vision((posenet_in - distortion.posenet._mean) / distortion.posenet._std)
        pred_summary = distortion.posenet.summarizer(pred_vision)
        pred_pose_out = distortion.posenet.hydra(pred_summary)["pose"][..., :6]

        frame2 = pred_chw[:, -1]
        seg_in = F.interpolate(frame2, size=(384, 512), mode="bilinear", align_corners=False)
        pred_seg_logits = distortion.segnet(seg_in)
        seg_ce = F.cross_entropy(pred_seg_logits, target_seg_argmax)
        seg_kl = F.kl_div(F.log_softmax(pred_seg_logits, dim=1), F.softmax(target_seg_logits, dim=1), reduction="batchmean") / (
            target_seg_logits.shape[-1] * target_seg_logits.shape[-2]
        )
        target_logit = pred_seg_logits.gather(1, target_seg_argmax.unsqueeze(1)).squeeze(1)
        masked = pred_seg_logits.masked_fill(F.one_hot(target_seg_argmax, 5).permute(0, 3, 1, 2).bool(), -1e4)
        margin = F.relu(masked.max(dim=1).values - target_logit + 0.25).mean()
        pose_mse = F.mse_loss(pred_pose_out, target_pose_vec)
        pose_feat = F.mse_loss(pred_summary, target_pose_summary)
        task = seg_ce + 0.25 * seg_kl + 0.5 * margin + 75.0 * pose_mse + 0.2 * pose_feat
        anchor = 0.01 * teacher + 0.02 * low_teacher
        return task + anchor

    def package(self, ctx: dict[str, Any]) -> PackageInfo:
        if self._package is not None:
            return self._package
        if self.variant in {"L48", "L40", "L32"}:
            audit = archive_audit(FP4_ARCHIVE)
            mask_bytes = int(audit["payload_breakdown"].get("mask.obu.br", 175_336))
            pose_bytes = int(audit["payload_breakdown"].get("pose.bin.br", 7_209))
            renderer = estimate_lowmask_renderer_bytes(self.variant)
            z_dim = int(renderer["config"]["z_pose_dim"])
            z_est = estimate_pose_token_bytes(600, z_dim)
            renderer_bytes = int(renderer["int8_brotli_bytes_random_init"])
            z_bytes = int(z_est["packed_zero_bytes"])
            archive_bytes = mask_bytes + pose_bytes + renderer_bytes + z_bytes + SINGLE_MEMBER_OVERHEAD_ESTIMATE
            self._extra["byte_estimate"] = {
                "mask_stream_bytes": mask_bytes,
                "pose_stream_bytes": pose_bytes,
                "renderer": renderer,
                "pose_tokens": z_est,
                "single_member_overhead_estimate": SINGLE_MEMBER_OVERHEAD_ESTIMATE,
                "projected_archive_bytes": archive_bytes,
            }
            self._package = PackageInfo(
                archive_bytes=archive_bytes,
                added_bytes=max(0, archive_bytes - int(qpose14_reference_summary()["archive_bytes"])),
                payload_breakdown={
                    "mask_stream": mask_bytes,
                    "pose_stream": pose_bytes,
                    "renderer_qpack_estimate": renderer_bytes,
                    "z_pose_estimate": z_bytes,
                    "overhead": SINGLE_MEMBER_OVERHEAD_ESTIMATE,
                },
                projected=True,
            )
            return self._package
        self._package = PackageInfo(archive_bytes=None, added_bytes=0, projected=True)
        return self._package

    def evaluate(self, subset: str, ctx: dict[str, Any]) -> Metrics:
        if self._metrics is not None:
            return self._metrics
        baseline = ctx.get("baseline", qpose14_reference_summary())
        if self._decision in {"smoke_pass", "packability_estimated"}:
            self._metrics = Metrics(
                segnet_dist=float(baseline["segnet_dist"]),
                posenet_dist=float(baseline["posenet_dist"]),
                quality=float(baseline["quality"]),
                seg_term=float(baseline["seg_term"]),
                pose_term=float(baseline["pose_term"]),
                score=None,
                sample_count=len(ctx.get("subset_indices", [])),
            )
        else:
            self._metrics = Metrics(None, None, None, sample_count=len(ctx.get("subset_indices", [])))
        return self._metrics

    def _gate_result(self, budget: Budget, package: PackageInfo, metrics: Metrics, baseline: dict[str, Any]) -> tuple[str, str, str]:
        if self._failure_reason:
            return self._decision, self._failure_reason, ""
        if budget.round == "smoke":
            return "smoke_pass", "", "smoke"
        if budget.round == "gate0_audit":
            return "gate0_audit_complete", "", "gate0_audit_ready"
        if budget.round == "packability":
            archive = package.archive_bytes or 10**9
            if self.variant in {"L48", "L40"} and archive <= 255_000:
                return "promote_hard8_compressed", "", "projected_archive_bytes_le_255k"
            if self.variant == "L32" and archive <= 250_000:
                return "promote_hard8_compressed", "", "projected_archive_bytes_le_250k"
            return "reject_packability_gate", f"projected archive {archive} exceeds tier budget", ""
        return self._decision, self._failure_reason, ""

    def decision_row(self, budget: Budget, ctx: dict[str, Any]) -> DecisionRow:
        package = self.package(ctx)
        metrics = self.evaluate(budget.subset, ctx)
        baseline = ctx.get("baseline", qpose14_reference_summary())
        decision, failure_reason, promotion_reason = self._gate_result(budget, package, metrics, baseline)
        projected_score = metrics.score
        if projected_score is None and metrics.quality is not None and package.archive_bytes is not None:
            projected_score = float(metrics.quality) + rate_term(int(package.archive_bytes))
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
            novelty_reason=self.config.get("novelty_reason", ""),
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
            term_tradeoff="lowmask_qpose_distill",
            decision=decision,
            failure_reason=failure_reason,
            promotion_reason=promotion_reason,
            extra=self._extra,
        )


def candidates(round_name: str) -> list[Candidate]:
    if round_name == "gate0_audit":
        return [LowmaskQposeDistillCandidate(variant="gate0", pose_source="qpose14")]
    if round_name == "hard8_capacity":
        return [LowmaskQposeDistillCandidate(variant="capacity", pose_source="qpose14")]
    if round_name in {"packability", "hard8_compressed", "strat64", "full600"}:
        return [
            LowmaskQposeDistillCandidate(variant="L48", pose_source="qpose14"),
            LowmaskQposeDistillCandidate(variant="L40", pose_source="qpose14"),
            LowmaskQposeDistillCandidate(variant="L32", pose_source="qpose14"),
        ]
    return [LowmaskQposeDistillCandidate(variant="L32", pose_source="qpose14")]
