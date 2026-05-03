#!/usr/bin/env python3
"""Pose-locked SegNet oracle for selfcomp++.

This script tests whether a SegNet-improving frame2 direction exists while
keeping selfcomp's already-good PoseNet behavior fixed. It freezes rendered
frame1, modifies only frame2 through a small parameterization, and accepts an
update only when actual measured SegNet improves and PoseNet stays inside a
hard term budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frame_utils import AVVideoDataset
from modules import DistortionNet, posenet_sd_path, segnet_sd_path
from submissions.selfcomp import inflate as base
from submissions.selfcomp_plus.inflate import load_segmap


ORIGINAL_BYTES = 37_545_489


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(10.0 * posenet_dist)


def extract_archive(archive_zip: Path, out_dir: Path) -> Path:
    archive_dir = out_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        zf.extractall(archive_dir)
    payload = archive_dir / "payload.tar.xz"
    if payload.exists():
        with tarfile.open(payload, mode="r:xz") as tf:
            tf.extractall(archive_dir)
    return archive_dir


def load_original_pairs(
    *,
    data_dir: Path,
    video_names_file: Path,
    offset: int,
    subset: int,
    batch_size: int,
) -> torch.Tensor:
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    ds = AVVideoDataset(names, data_dir=data_dir, batch_size=batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    batches = []
    seen = 0
    stop = offset + subset
    for _, _, batch in ds:
        batch_count = batch.shape[0]
        batch_start, batch_end = seen, seen + batch_count
        seen = batch_end
        if batch_end <= offset:
            continue
        if batch_start >= stop:
            break
        left = max(offset - batch_start, 0)
        right = min(stop - batch_start, batch_count)
        batches.append(batch[left:right])
        if sum(item.shape[0] for item in batches) >= subset:
            break
    out = torch.cat(batches, dim=0)
    if out.shape[0] != subset:
        raise RuntimeError(f"expected {subset} samples, loaded {out.shape[0]}")
    return out


def load_probability_maps(data_dir: Path, *, offset: int, subset: int, device: torch.device) -> torch.Tensor:
    lut = base.create_gaussian_softmax_lut().to(device)
    video_path = data_dir / "0.mkv"
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    maps = []
    stop = offset + subset
    for idx, frame in enumerate(container.decode(stream)):
        if idx < offset:
            continue
        if idx >= stop:
            break
        gray = torch.from_numpy(frame.to_ndarray(format="gray").copy()).to(device=device, dtype=torch.float32)
        if tuple(gray.shape) != (base.SEGMAP_INPUT_SIZE[1], base.SEGMAP_INPUT_SIZE[0]):
            gray = (
                F.interpolate(
                    gray.unsqueeze(0).unsqueeze(0),
                    size=(base.SEGMAP_INPUT_SIZE[1], base.SEGMAP_INPUT_SIZE[0]),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
        gray = gray.round().clamp(0, 255).long()
        maps.append(F.embedding(gray, lut).permute(2, 0, 1).contiguous())
    container.close()
    if len(maps) != subset:
        raise RuntimeError(f"expected {subset} latent frames, decoded {len(maps)}")
    return torch.stack(maps, dim=0)


def render_selfcomp(model: torch.nn.Module, prob_maps: torch.Tensor, offset: int, batch_size: int) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, prob_maps.shape[0], batch_size):
            end = min(prob_maps.shape[0], start + batch_size)
            sample_indices = torch.arange(offset + start, offset + end, device=prob_maps.device)
            frame_indices = torch.stack([2 * sample_indices, 2 * sample_indices + 1], dim=1).reshape(-1).long()
            rendered = model(prob_maps[start:end].repeat_interleave(2, dim=0), frame_indices)
            chunks.append(rendered.reshape(end - start, 2, *rendered.shape[1:]))
    return torch.cat(chunks, dim=0)


def hard_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    target_logits = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    mask = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).bool()
    other_logits = logits.masked_fill(mask, -1e4).max(dim=1).values
    return F.relu(margin - (target_logits - other_logits)).mean()


def tensor_tv(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


class Frame2Correction(torch.nn.Module):
    def __init__(self, variant: str, subset: int, h: int, w: int):
        super().__init__()
        self.variant = variant
        self.subset = subset
        self.h = h
        self.w = w
        if variant == "p0":
            self.scale_raw = torch.nn.Parameter(torch.zeros(1, 3, 1, 1))
            self.bias = torch.nn.Parameter(torch.zeros(1, 3, 1, 1))
        elif variant == "p1":
            self.scale_raw = torch.nn.Parameter(torch.zeros(1, 3, 8, 1))
            self.bias = torch.nn.Parameter(torch.zeros(1, 3, 8, 1))
        elif variant == "p2":
            self.class_scale_raw = torch.nn.Parameter(torch.zeros(5, 3))
            self.class_bias = torch.nn.Parameter(torch.zeros(5, 3))
        elif variant == "p3":
            self.residual_grid = torch.nn.Parameter(torch.zeros(subset, 3, 12, 16))
        elif variant == "p4":
            self.residual = torch.nn.Parameter(torch.zeros(subset, 3, h, w))
        else:
            raise ValueError(f"unknown variant: {variant}")

    def forward(
        self,
        base_frame2: torch.Tensor,
        prob_maps: torch.Tensor,
        *,
        indices: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        if self.variant == "p0":
            scale = 1.0 + 0.15 * torch.tanh(self.scale_raw)
            bias = 20.0 * torch.tanh(self.bias / 20.0)
            return (base_frame2 * scale + bias).clamp(0, 255)
        if self.variant == "p1":
            scale = 1.0 + 0.15 * torch.tanh(
                F.interpolate(self.scale_raw, size=(self.h, self.w), mode="bilinear", align_corners=False)
            )
            bias = 20.0 * torch.tanh(
                F.interpolate(self.bias / 20.0, size=(self.h, self.w), mode="bilinear", align_corners=False)
            )
            return (base_frame2 * scale + bias).clamp(0, 255)
        if self.variant == "p2":
            scale_delta = torch.einsum("bkhw,kc->bchw", prob_maps, self.class_scale_raw)
            bias = torch.einsum("bkhw,kc->bchw", prob_maps, self.class_bias)
            scale = 1.0 + 0.15 * torch.tanh(scale_delta)
            bias = 20.0 * torch.tanh(bias / 20.0)
            return (base_frame2 * scale + bias).clamp(0, 255)
        if self.variant == "p3":
            grid = self.residual_grid if indices is None else self.residual_grid[indices]
            residual = F.interpolate(grid, size=(self.h, self.w), mode="bicubic", align_corners=False)
            return (base_frame2 + 24.0 * torch.tanh(residual / 24.0)).clamp(0, 255)
        raw_residual = self.residual if indices is None else self.residual[indices]
        # P4 is an upper bound, not a packable representation; allow a wider
        # pixel-space move than the low-entropy P3 grid while still avoiding
        # unbounded adversarial blowups.
        residual = 64.0 * torch.tanh(raw_residual / 64.0)
        return (base_frame2 + residual).clamp(0, 255)


def clone_state(module: torch.nn.Module) -> dict:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def compute_terms(
    *,
    distortion: DistortionNet,
    target_seg_logits: torch.Tensor,
    target_pose: torch.Tensor,
    base_frame1: torch.Tensor,
    frame2: torch.Tensor,
    metric_batch_size: int,
) -> dict:
    seg_weighted = 0.0
    pose_weighted = 0.0
    total = frame2.shape[0]
    with torch.inference_mode():
        for start in range(0, total, metric_batch_size):
            end = min(total, start + metric_batch_size)
            pair = torch.stack([base_frame1[start:end], frame2[start:end]], dim=1)
            seg_out = distortion.segnet(pair[:, -1])
            seg_dist = distortion.segnet.compute_distortion(target_seg_logits[start:end], seg_out).mean().item()
            pose_out = distortion.posenet(distortion.posenet.preprocess_input(pair))
            pose_dist = (pose_out["pose"][..., :6] - target_pose[start:end]).pow(2).mean(dim=1).mean().item()
            weight = end - start
            seg_weighted += seg_dist * weight
            pose_weighted += pose_dist * weight
    seg_dist = seg_weighted / total
    pose_dist = pose_weighted / total
    return {
        "segnet_dist": float(seg_dist),
        "posenet_dist": float(pose_dist),
        "seg_term": 100.0 * float(seg_dist),
        "pose_term": math.sqrt(10.0 * float(pose_dist)),
        "quality": quality(float(seg_dist), float(pose_dist)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["p0", "p1", "p2", "p3", "p4"], required=True)
    parser.add_argument("--subset", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pose-eps", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render-batch-size", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--metric-batch-size", type=int, default=4)
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="selfcomp_pose_lock_") as tmp:
        archive_dir = extract_archive(args.archive, Path(tmp))
        model = load_segmap(archive_dir / "segmap_inference.pt", device)
        for param in model.parameters():
            param.requires_grad_(False)

        original_camera = load_original_pairs(
            data_dir=args.uncompressed_dir,
            video_names_file=args.video_names_file,
            offset=args.offset,
            subset=args.subset,
            batch_size=8,
        ).to(device)
        original_chw = original_camera.permute(0, 1, 4, 2, 3).float()

        distortion = DistortionNet().eval().to(device)
        distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
        for param in distortion.parameters():
            param.requires_grad_(False)

        prob_maps = load_probability_maps(archive_dir, offset=args.offset, subset=args.subset, device=device)
        rendered = render_selfcomp(model, prob_maps, args.offset, args.render_batch_size)
        base_frame1 = rendered[:, 0].detach().clone()
        base_frame2 = rendered[:, 1].detach().clone()
        _, _, h, w = base_frame2.shape

        with torch.inference_mode():
            target_seg_logits = distortion.segnet(distortion.segnet.preprocess_input(original_chw))
            target_seg_prob = target_seg_logits.softmax(dim=1)
            target_seg_argmax = target_seg_logits.argmax(dim=1)
            target_pose = distortion.posenet(distortion.posenet.preprocess_input(original_chw))["pose"][..., :6]
        target_seg_logits = target_seg_logits.detach().clone()
        target_seg_prob = target_seg_prob.detach().clone()
        target_seg_argmax = target_seg_argmax.detach().clone()
        target_pose = target_pose.detach().clone()

        baseline = compute_terms(
            distortion=distortion,
            target_seg_logits=target_seg_logits,
            target_pose=target_pose,
            base_frame1=base_frame1,
            frame2=base_frame2,
            metric_batch_size=args.metric_batch_size,
        )
        baseline["step"] = 0
        print(json.dumps(baseline, sort_keys=True), flush=True)
        with history_path.open("a") as f:
            f.write(json.dumps({"accepted": True, **baseline}, sort_keys=True) + "\n")

        correction = Frame2Correction(args.variant, args.subset, h, w).to(device)
        best_state = clone_state(correction)
        best_frame2 = base_frame2
        best = dict(baseline)
        optimizer = torch.optim.AdamW(correction.parameters(), lr=args.lr, weight_decay=0.0)

        for step in tqdm(range(1, args.steps + 1), desc=f"pose-lock {args.variant}"):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, args.subset, args.train_batch_size):
                end = min(args.subset, start + args.train_batch_size)
                idx = torch.arange(start, end, device=device)
                candidate = correction(base_frame2[start:end], prob_maps[start:end], indices=idx)
                seg_logits = distortion.segnet(candidate)
                ce = F.cross_entropy(seg_logits, target_seg_argmax[start:end])
                kl = F.kl_div(
                    seg_logits.log_softmax(dim=1),
                    target_seg_prob[start:end],
                    reduction="batchmean",
                ) / (seg_logits.shape[-1] * seg_logits.shape[-2])
                margin = hard_margin_loss(seg_logits, target_seg_argmax[start:end])
                smooth = tensor_tv((candidate - base_frame2[start:end]) / 255.0)
                loss = ce + kl + 2.0 * margin + 0.02 * smooth
                (loss * ((end - start) / args.subset)).backward()
            torch.nn.utils.clip_grad_norm_(correction.parameters(), 1.0)
            optimizer.step()

            if step % args.eval_every != 0 and step != args.steps:
                continue

            candidate_chunks = []
            with torch.inference_mode():
                for start in range(0, args.subset, args.metric_batch_size):
                    end = min(args.subset, start + args.metric_batch_size)
                    idx = torch.arange(start, end, device=device)
                    candidate_chunks.append(correction(base_frame2[start:end], prob_maps[start:end], indices=idx))
            candidate = torch.cat(candidate_chunks, dim=0)
            current = compute_terms(
                distortion=distortion,
                target_seg_logits=target_seg_logits,
                target_pose=target_pose,
                base_frame1=base_frame1,
                frame2=candidate,
                metric_batch_size=args.metric_batch_size,
            )
            current["step"] = step
            accepted = current["pose_term"] <= baseline["pose_term"] + args.pose_eps and current["seg_term"] < best["seg_term"]
            current["accepted"] = accepted
            with history_path.open("a") as f:
                f.write(json.dumps(current, sort_keys=True) + "\n")
            print(json.dumps(current, sort_keys=True), flush=True)
            if accepted:
                best = dict(current)
                best_state = clone_state(correction)
                best_frame2 = candidate.detach()
            else:
                correction.load_state_dict(best_state)
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5

        result = {
            "archive": str(args.archive),
            "variant": args.variant,
            "subset": args.subset,
            "offset": args.offset,
            "steps": args.steps,
            "pose_eps": args.pose_eps,
            "baseline": baseline,
            "best": best,
            "seg_improvement": (baseline["seg_term"] - best["seg_term"]) / max(baseline["seg_term"], 1e-12),
            "pose_worsening": best["pose_term"] - baseline["pose_term"],
        }
        metrics_path = args.out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        torch.save({"variant": args.variant, "state_dict": best_state, "best": best}, args.out_dir / "best_correction.pt")
        print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
