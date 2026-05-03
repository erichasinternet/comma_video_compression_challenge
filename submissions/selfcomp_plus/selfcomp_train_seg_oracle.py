#!/usr/bin/env python3
"""64-sample SegNet capacity oracle for selfcomp++.

This is intentionally an oracle, not a final compressor. It answers whether the
existing selfcomp representation can move SegNet toward Quantizr-level quality
when the decoded soft semantic video and/or renderer weights are allowed to
train before any AV1 bitrate hardening.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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

from frame_utils import AVVideoDataset, camera_size, seq_len
from modules import DistortionNet, posenet_sd_path, segnet_sd_path
from submissions.selfcomp import inflate as base
from submissions.selfcomp_plus.inflate import load_segmap


ORIGINAL_BYTES = 37_545_489


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(10.0 * posenet_dist)


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + 25.0 * archive_bytes / ORIGINAL_BYTES


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
    if not batches:
        raise RuntimeError("no original samples loaded")
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
        prob = F.embedding(gray, lut).permute(2, 0, 1).contiguous()
        maps.append(prob)
    container.close()
    if len(maps) != subset:
        raise RuntimeError(f"expected {subset} latent frames, decoded {len(maps)}")
    return torch.stack(maps, dim=0)


def render_model(
    model: torch.nn.Module,
    prob_maps: torch.Tensor,
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    batch = prob_maps.shape[0]
    frame_indices = torch.stack([2 * sample_indices, 2 * sample_indices + 1], dim=1).reshape(-1).long()
    rendered = model(prob_maps.repeat_interleave(2, dim=0), frame_indices)
    return rendered.reshape(batch, 2, *rendered.shape[1:])  # B, 2, 3, 384, 512


def to_camera_size(x: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = x.shape
    flat = x.reshape(b * t, c, h, w)
    up = F.interpolate(flat, size=(camera_size[1], camera_size[0]), mode="bicubic", align_corners=False)
    return up.reshape(b, t, c, camera_size[1], camera_size[0]).clamp(0, 255)


def hard_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    target_logits = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    masked = logits.masked_fill(F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).bool(), -1e4)
    other_logits = masked.max(dim=1).values
    return F.relu(margin - (target_logits - other_logits)).mean()


def tensor_tv(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def build_optimizer(
    *,
    model: torch.nn.Module,
    latent_logits: torch.nn.Parameter,
    train_model: bool,
    train_latent: bool,
    lr_model: float,
    lr_affine: float,
    lr_shared: float,
    lr_latent: float,
) -> torch.optim.Optimizer:
    groups = []
    if train_model:
        shared = [model.shared_latent_base]
        affine = list(model.frame_affine_embedding.parameters())
        special_ids = {id(p) for p in shared + affine}
        rest = [p for p in model.parameters() if id(p) not in special_ids]
        groups.extend(
            [
                {"name": "renderer_conv", "params": rest, "lr": lr_model},
                {"name": "shared_latent_base", "params": shared, "lr": lr_shared},
                {"name": "frame_affine_embedding", "params": affine, "lr": lr_affine},
            ]
        )
    if train_latent:
        groups.append({"name": "decoded_latent_logits", "params": [latent_logits], "lr": lr_latent})
    return torch.optim.AdamW(groups, weight_decay=0.0)


@torch.inference_mode()
def compute_metrics(
    *,
    distortion: DistortionNet,
    original_camera_btchw: torch.Tensor,
    generated_model_btchw: torch.Tensor,
) -> dict:
    generated_camera = to_camera_size(generated_model_btchw).permute(0, 1, 3, 4, 2).round().to(torch.uint8)
    posenet_dist, segnet_dist = distortion.compute_distortion(original_camera_btchw, generated_camera)
    pose = float(posenet_dist.mean().item())
    seg = float(segnet_dist.mean().item())
    return {
        "posenet_dist": pose,
        "segnet_dist": seg,
        "quality": quality(seg, pose),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["model-only", "latent-only", "latent+model"], required=True)
    parser.add_argument("--subset", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr-model", type=float, default=3e-5)
    parser.add_argument("--lr-affine", type=float, default=3e-4)
    parser.add_argument("--lr-shared", type=float, default=1e-4)
    parser.add_argument("--lr-latent", type=float, default=1e-2)
    parser.add_argument("--pose-weight", type=float, default=0.5)
    parser.add_argument("--pose-margin", type=float, default=0.005)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--latent-tv-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--apply-gates", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="selfcomp_oracle_") as tmp:
        archive_dir = extract_archive(args.archive, Path(tmp))
        model = load_segmap(archive_dir / "segmap_inference.pt", device)

        original_camera = load_original_pairs(
            data_dir=args.uncompressed_dir,
            video_names_file=args.video_names_file,
            offset=args.offset,
            subset=args.subset,
            batch_size=max(args.batch_size, 1),
        ).to(device)
        original_chw = original_camera.permute(0, 1, 4, 2, 3).float()

        distortion = DistortionNet().eval().to(device)
        distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
        for param in distortion.parameters():
            param.requires_grad_(False)

        with torch.inference_mode():
            target_seg_logits = distortion.segnet(distortion.segnet.preprocess_input(original_chw))
            target_seg_prob = target_seg_logits.softmax(dim=1)
            target_seg_argmax = target_seg_logits.argmax(dim=1)
            target_pose_out = distortion.posenet(distortion.posenet.preprocess_input(original_chw))
            target_pose = target_pose_out["pose"][..., :6]

        prob_maps = load_probability_maps(archive_dir, offset=args.offset, subset=args.subset, device=device)
        latent_logits = torch.nn.Parameter((prob_maps + 1e-6).log())

        train_model = args.mode in {"model-only", "latent+model"}
        train_latent = args.mode in {"latent-only", "latent+model"}
        for param in model.parameters():
            param.requires_grad_(train_model)
        latent_logits.requires_grad_(train_latent)

        if not train_model and not train_latent:
            raise RuntimeError("no trainable parameters selected")
        optimizer = build_optimizer(
            model=model,
            latent_logits=latent_logits,
            train_model=train_model,
            train_latent=train_latent,
            lr_model=args.lr_model,
            lr_affine=args.lr_affine,
            lr_shared=args.lr_shared,
            lr_latent=args.lr_latent,
        )

        sample_indices_all = torch.arange(args.offset, args.offset + args.subset, device=device)
        history = []
        stopped_reason = None
        history_jsonl = args.out_dir / "history.jsonl"
        history_jsonl.unlink(missing_ok=True)
        checkpoint_path = args.out_dir / "checkpoint.pt"

        with torch.inference_mode():
            baseline_pose_dists = []
            for start in range(0, args.subset, args.batch_size):
                end = min(args.subset, start + args.batch_size)
                generated = render_model(model, prob_maps[start:end], sample_indices_all[start:end])
                gen_camera = to_camera_size(generated)
                pose_out = distortion.posenet(distortion.posenet.preprocess_input(gen_camera))
                pose_dist = (pose_out["pose"][..., :6] - target_pose[start:end]).pow(2).mean(dim=1)
                baseline_pose_dists.append(pose_dist)
            baseline_pose_dists = torch.cat(baseline_pose_dists, dim=0).detach()

        def eval_now(step: int) -> dict:
            model.eval()
            with torch.inference_mode():
                chunks = []
                for start in range(0, args.subset, args.batch_size):
                    end = min(args.subset, start + args.batch_size)
                    probs = latent_logits[start:end].softmax(dim=1) if train_latent else prob_maps[start:end]
                    chunks.append(render_model(model, probs, sample_indices_all[start:end]))
                generated = torch.cat(chunks, dim=0)
                metrics = compute_metrics(
                    distortion=distortion,
                    original_camera_btchw=original_camera,
                    generated_model_btchw=generated,
                )
            metrics["step"] = step
            metrics["seg_term"] = 100.0 * metrics["segnet_dist"]
            metrics["pose_term"] = math.sqrt(10.0 * metrics["posenet_dist"])
            history.append(metrics)
            with history_jsonl.open("a") as f:
                f.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)
            return metrics

        baseline = eval_now(0)
        model.train()

        for step in tqdm(range(1, args.steps + 1), desc=f"{args.mode} oracle"):
            idx = torch.randint(0, args.subset, (args.batch_size,), device=device)
            probs = latent_logits[idx].softmax(dim=1) if train_latent else prob_maps[idx]
            generated = render_model(model, probs, sample_indices_all[idx])
            gen_seg_logits = distortion.segnet(generated[:, -1])
            ce = F.cross_entropy(gen_seg_logits, target_seg_argmax[idx])
            kl = F.kl_div(
                gen_seg_logits.log_softmax(dim=1),
                target_seg_prob[idx],
                reduction="batchmean",
            ) / (gen_seg_logits.shape[-1] * gen_seg_logits.shape[-2])
            margin = hard_margin_loss(gen_seg_logits, target_seg_argmax[idx])
            pose_loss = torch.zeros((), device=device)
            if args.pose_weight > 0:
                gen_camera = to_camera_size(generated)
                pose_out = distortion.posenet(distortion.posenet.preprocess_input(gen_camera))
                pose_dist = (pose_out["pose"][..., :6] - target_pose[idx]).pow(2).mean(dim=1)
                pose_term = torch.sqrt(10.0 * pose_dist.clamp_min(1e-12))
                baseline_pose_term = torch.sqrt(10.0 * baseline_pose_dists[idx].clamp_min(1e-12))
                pose_loss = F.relu(pose_term - baseline_pose_term - args.pose_margin).mean()
            latent_smooth = torch.zeros((), device=device)
            if train_latent and args.latent_tv_weight > 0:
                latent_smooth = tensor_tv(latent_logits[idx].softmax(dim=1))
            loss = (
                ce
                + kl
                + 2.0 * margin
                + args.pose_weight * pose_loss
                + args.tv_weight * tensor_tv(generated / 255.0)
                + args.latent_tv_weight * latent_smooth
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                    args.grad_clip,
                )
            optimizer.step()

            if step % args.eval_every == 0 or step == args.steps:
                current = eval_now(step)
                if args.apply_gates:
                    seg_improvement = (baseline["seg_term"] - current["seg_term"]) / max(baseline["seg_term"], 1e-12)
                    pose_worsening = current["pose_term"] - baseline["pose_term"]
                    if step >= 250 and seg_improvement < 0.15:
                        stopped_reason = (
                            f"early gate failed at step {step}: "
                            f"SegNet term improvement {seg_improvement:.3f} < 0.150"
                        )
                    elif step >= 250 and pose_worsening > 0.010:
                        stopped_reason = (
                            f"early gate failed at step {step}: "
                            f"PoseNet term worsening {pose_worsening:.5f} > 0.010"
                        )
                    elif step >= 750 and seg_improvement < 0.30:
                        stopped_reason = (
                            f"mid gate failed at step {step}: "
                            f"SegNet term improvement {seg_improvement:.3f} < 0.300"
                        )
                    elif step >= 750 and current["quality"] > baseline["quality"] - 0.035:
                        stopped_reason = (
                            f"mid gate failed at step {step}: "
                            f"quality improvement {baseline['quality'] - current['quality']:.5f} < 0.035"
                        )
                    elif step >= 750 and pose_worsening > 0.015:
                        stopped_reason = (
                            f"mid gate failed at step {step}: "
                            f"PoseNet term worsening {pose_worsening:.5f} > 0.015"
                        )
                    if stopped_reason is not None:
                        print(stopped_reason, flush=True)
                        break
                model.train()
            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0 or step == args.steps):
                torch.save(
                    {
                        "step": step,
                        "mode": args.mode,
                        "model_state_dict": model.state_dict(),
                        "latent_logits": latent_logits.detach().cpu(),
                        "history": history,
                    },
                    checkpoint_path,
                )

        final = history[-1]
        result = {
            "archive": str(args.archive),
            "mode": args.mode,
            "subset": args.subset,
            "offset": args.offset,
            "steps": args.steps,
            "baseline": baseline,
            "final": final,
            "history": history,
            "stopped_reason": stopped_reason,
        }
        metrics_path = args.out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
