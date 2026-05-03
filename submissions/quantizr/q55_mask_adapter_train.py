#!/usr/bin/env python
"""Train a tiny evaluator-level mask adapter for a lossy #55 mask stream."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import brotli
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    ORIGINAL_BYTES,
    POSE_PAYLOAD,
    POSE_QPACK_PAYLOAD,
    REPO_ROOT,
    append_jsonl,
    make_archive_zip,
    materialize_submission,
    metric_record,
    run_evaluate_submission,
    score_from_bytes,
    summarize_archive,
    unzip_archive,
    write_json,
)
from q55_mask_alloc import decode_archive_masks
from inflate import (
    MASK_ADAPTER_PAYLOAD_NAME,
    MaskEmbeddingAdapter,
    get_qpack_state_dict,
    load_pose_payload,
)


def choose_payload(archive_dir: Path, primary: str, fallback: str) -> str:
    if (archive_dir / primary).exists():
        return primary
    if (archive_dir / fallback).exists():
        return fallback
    raise FileNotFoundError(f"missing payload: {primary} or {fallback}")


def load_generator_from_archive(archive_dir: Path, device: torch.device, q55):
    generator = q55.JointFrameGenerator().to(device)
    model_qpack = archive_dir / MODEL_QPACK_PAYLOAD
    model_br = archive_dir / MODEL_PAYLOAD
    if model_qpack.exists():
        state = get_qpack_state_dict(brotli.decompress(model_qpack.read_bytes()), device)
        generator.load_state_dict(state, strict=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(brotli.decompress(model_br.read_bytes()))
            tmp_path = Path(tmp.name)
        try:
            q55.load_fp4_state_dict(generator, tmp_path, device)
        finally:
            tmp_path.unlink(missing_ok=True)
    generator.float().eval()
    return generator


def load_rgb_pairs(files: list[str], video_dir: Path, batch_size: int, q55):
    from frame_utils import AVVideoDataset

    os.environ["FORCE_AV_DATASET"] = "1"
    ds = AVVideoDataset(files, data_dir=video_dir, batch_size=batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    batches = [batch.cpu().contiguous() for _, _, batch in ds]
    if not batches:
        raise RuntimeError("No video data was loaded by AVVideoDataset.")
    return torch.cat(batches, dim=0).contiguous()


class AdapterGenerator(nn.Module):
    def __init__(self, base, adapter: MaskEmbeddingAdapter):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def shared_features(self, mask2: torch.Tensor) -> torch.Tensor:
        trunk = self.base.shared_trunk
        coords = self._coords(mask2.shape[0], mask2.device)
        e2 = trunk.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        e2_up = self.adapter(e2_up, mask2.long(), coords)
        s = trunk.stem_block(trunk.stem_conv(torch.cat([e2_up, coords], dim=1)))
        z = trunk.up(trunk.down_block(trunk.down_conv(s)))
        return trunk.fuse_block(trunk.fuse(torch.cat([z, s], dim=1)))

    def _coords(self, batch: int, device: torch.device) -> torch.Tensor:
        import compress as q55

        return q55.make_coord_grid(batch, 384, 512, device, torch.float32)

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor):
        feat = self.shared_features(mask2)
        pred_frame2 = self.base.frame2_head(feat)
        pred_frame1 = self.base.frame1_head(feat, self.base.pose_mlp(pose6))
        return pred_frame1, pred_frame2, feat


def base_features_and_frames(generator, mask2: torch.Tensor, pose6: torch.Tensor):
    coords = generator.shared_trunk.embedding.weight.new_empty(0)
    import compress as q55

    coords = q55.make_coord_grid(mask2.shape[0], 384, 512, mask2.device, torch.float32)
    feat = generator.shared_trunk(mask2, coords)
    frame2 = generator.frame2_head(feat)
    frame1 = generator.frame1_head(feat, generator.pose_mlp(pose6))
    return frame1, frame2, feat


def distill_loss(student_out, teacher_out, args):
    s1, s2, sf = student_out
    t1, t2, tf = teacher_out
    frame_loss = F.smooth_l1_loss(s1 / 255.0, t1 / 255.0) + F.smooth_l1_loss(s2 / 255.0, t2 / 255.0)
    feat_loss = F.mse_loss(sf, tf)
    return args.frame_weight * frame_loss + args.feature_weight * feat_loss, frame_loss.detach(), feat_loss.detach()


def evaluator_loss(student_frames, real_batch, segnet, posenet, args, q55):
    if args.seg_weight <= 0 and args.pose_weight <= 0:
        z = student_frames[0].new_tensor(0.0)
        return z, z, z

    pred_frame1, pred_frame2 = student_frames
    batch = einops.rearrange(real_batch, "b t h w c -> b t c h w").float()
    real1 = F.interpolate(batch[:, 0], size=(384, 512), mode="bilinear", align_corners=False)
    real2 = F.interpolate(batch[:, 1], size=(384, 512), mode="bilinear", align_corners=False)

    fake1_up = F.interpolate(pred_frame1, size=(874, 1164), mode="bilinear", align_corners=False)
    fake2_up = F.interpolate(pred_frame2, size=(874, 1164), mode="bilinear", align_corners=False)
    fake1_down = F.interpolate(q55.diff_round(fake1_up.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)
    fake2_down = F.interpolate(q55.diff_round(fake2_up.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)

    seg_loss = pred_frame1.new_tensor(0.0)
    pose_loss = pred_frame1.new_tensor(0.0)

    if args.seg_weight > 0:
        with torch.no_grad():
            gt_logits2 = segnet(real2).float()
            gt_mask2 = gt_logits2.argmax(dim=1)
        fake_logits2 = segnet(fake2_down).float()
        seg_ce = F.cross_entropy(fake_logits2, gt_mask2)
        seg_kl = q55.kl_on_logits(fake_logits2, gt_logits2, 2.0) / (384 * 512)
        seg_loss = seg_ce + 0.5 * seg_kl

    if args.pose_weight > 0:
        with torch.no_grad():
            gt_pose = q55.get_pose_tensor(posenet(posenet.preprocess_input(batch))).float()[..., :6]
        fake_pose = q55.get_pose_tensor(posenet(q55.pack_pair_yuv6(fake1_down, fake2_down).float())).float()[..., :6]
        pose_loss = F.mse_loss(fake_pose, gt_pose)

    return args.seg_weight * seg_loss + args.pose_weight * pose_loss, seg_loss.detach(), pose_loss.detach()


@torch.inference_mode()
def proxy_eval(student: AdapterGenerator, rgb_pairs, masks, poses, distortion_net, device, batch_size: int) -> dict:
    import compress as q55

    student.eval()
    total_pose = 0.0
    total_seg = 0.0
    samples = 0
    for start in range(0, masks.shape[0], batch_size):
        end = min(start + batch_size, masks.shape[0])
        mask = masks[start:end].to(device).long()
        pose = poses[start:end].to(device).float()
        real = rgb_pairs[start:end].to(device)
        p1, p2, _ = student(mask, pose)
        comp = torch.stack(
            [
                F.interpolate(p1, size=(874, 1164), mode="bilinear", align_corners=False),
                F.interpolate(p2, size=(874, 1164), mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        comp = einops.rearrange(comp, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
        p_dist, s_dist = distortion_net.compute_distortion(real, comp)
        total_pose += p_dist.sum().item()
        total_seg += s_dist.sum().item()
        samples += end - start
    pose = total_pose / max(1, samples)
    seg = total_seg / max(1, samples)
    return {
        "samples": samples,
        "posenet_dist": pose,
        "segnet_dist": seg,
        "quality_term": 100.0 * seg + (10.0 * pose) ** 0.5,
    }


def save_adapter_payload(adapter: MaskEmbeddingAdapter, path: Path) -> dict:
    payload = {
        "format": "quantizr_mask_adapter_v1",
        "hidden": adapter.hidden,
        "emb_dim": adapter.emb_dim,
        "num_classes": adapter.num_classes,
        "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
    }
    raw = io.BytesIO()
    torch.save(payload, raw)
    path.write_bytes(brotli.compress(raw.getvalue(), quality=11, lgwin=24))
    return {"adapter_payload_bytes_br": path.stat().st_size}


def package_archive(args, run_dir: Path, adapter: MaskEmbeddingAdapter) -> tuple[Path, dict]:
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    base_dir = run_dir / "base_archive"
    predictor_dir = run_dir / "predictor_archive"
    unzip_archive(args.base_package_archive, base_dir)
    unzip_archive(args.predictor_mask_archive, predictor_dir)
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)

    model_payload = choose_payload(base_dir, MODEL_QPACK_PAYLOAD, MODEL_PAYLOAD)
    pose_payload = choose_payload(base_dir, POSE_QPACK_PAYLOAD, POSE_PAYLOAD)
    shutil.copy2(base_dir / model_payload, archive_dir / model_payload)
    shutil.copy2(base_dir / pose_payload, archive_dir / pose_payload)
    shutil.copy2(predictor_dir / MASK_PAYLOAD, archive_dir / MASK_PAYLOAD)
    adapter_report = save_adapter_payload(adapter, archive_dir / MASK_ADAPTER_PAYLOAD_NAME)
    make_archive_zip(archive_dir, archive_zip, [MASK_PAYLOAD, model_payload, pose_payload, MASK_ADAPTER_PAYLOAD_NAME])
    return archive_zip, {
        "model_payload": model_payload,
        "pose_payload": pose_payload,
        "mask_payload": MASK_PAYLOAD,
        **adapter_report,
        "archive_summary": summarize_archive(archive_zip),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-archive", type=Path, required=True)
    parser.add_argument("--predictor-mask-archive", type=Path, required=True)
    parser.add_argument("--base-package-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="qmask_adapter_qrecode50_sanity")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--frame-weight", type=float, default=8.0)
    parser.add_argument("--feature-weight", type=float, default=0.02)
    parser.add_argument("--seg-weight", type=float, default=0.25)
    parser.add_argument("--pose-weight", type=float, default=2.0)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--eval-device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()

    run_dir = args.out_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    import compress as q55
    from safetensors.torch import load_file
    from modules import DistortionNet, PoseNet, SegNet, posenet_sd_path, segnet_sd_path

    exact_dir = run_dir / "exact_source"
    predictor_dir = run_dir / "predictor_source"
    base_dir = run_dir / "base_source"
    unzip_archive(args.exact_archive, exact_dir)
    unzip_archive(args.predictor_mask_archive, predictor_dir)
    unzip_archive(args.base_package_archive, base_dir)

    files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]
    print("Loading source videos via AVVideoDataset...", flush=True)
    rgb_pairs = load_rgb_pairs(files, args.video_dir, args.eval_batch_size, q55)
    exact_masks = torch.from_numpy(decode_archive_masks(exact_dir / MASK_PAYLOAD)).long().contiguous()
    predictor_masks = torch.from_numpy(decode_archive_masks(predictor_dir / MASK_PAYLOAD)).long().contiguous()
    poses = load_pose_payload(base_dir, base_dir / POSE_PAYLOAD).float().contiguous()

    teacher = load_generator_from_archive(base_dir, device, q55)
    for p in teacher.parameters():
        p.requires_grad = False
    base = load_generator_from_archive(base_dir, device, q55)
    for p in base.parameters():
        p.requires_grad = False
    adapter = MaskEmbeddingAdapter(hidden=args.hidden).to(device)
    student = AdapterGenerator(base, adapter).to(device)

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    distortion_net = DistortionNet().eval().to(device)
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for model in (segnet, posenet, distortion_net):
        for p in model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, betas=(0.9, 0.99))
    history = []

    def run_proxy(step: int) -> dict:
        rec = proxy_eval(student, rgb_pairs, predictor_masks, poses, distortion_net, device, args.eval_batch_size)
        archive_bytes_est = summarize_archive(args.base_package_archive)["archive_bytes"]
        rec.update(
            {
                "step": step,
                "score_at_base_bytes": score_from_bytes(
                    rec["segnet_dist"], rec["posenet_dist"], archive_bytes_est, ORIGINAL_BYTES
                ),
            }
        )
        print(json.dumps(rec, sort_keys=True), flush=True)
        history.append(rec)
        student.train()
        return rec

    run_proxy(0)
    n = predictor_masks.shape[0]
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,))
        exact_mask = exact_masks[idx].to(device)
        pred_mask = predictor_masks[idx].to(device)
        pose = poses[idx].to(device)
        real_batch = rgb_pairs[idx].to(device)

        with torch.no_grad():
            teacher_out = base_features_and_frames(teacher, exact_mask, pose)
        student_out = student(pred_mask, pose)
        loss_distill, frame_loss, feat_loss = distill_loss(student_out, teacher_out, args)
        loss_eval, seg_loss, pose_loss = evaluator_loss(student_out[:2], real_batch, segnet, posenet, args, q55)
        loss = loss_distill + loss_eval

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "frame_loss": float(frame_loss.cpu()),
                        "feature_loss": float(feat_loss.cpu()),
                        "seg_loss": float(seg_loss.cpu()),
                        "pose_loss": float(pose_loss.cpu()),
                        "alpha": float(adapter.alpha.detach().cpu()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            run_proxy(step)

    archive_zip, package_report = package_archive(args, run_dir, adapter)
    record = {
        "label": args.label,
        "exact_archive": str(args.exact_archive),
        "predictor_mask_archive": str(args.predictor_mask_archive),
        "base_package_archive": str(args.base_package_archive),
        "steps": args.steps,
        "hidden": args.hidden,
        "lr": args.lr,
        "frame_weight": args.frame_weight,
        "feature_weight": args.feature_weight,
        "seg_weight": args.seg_weight,
        "pose_weight": args.pose_weight,
        "history": history,
        "final_proxy": history[-1] if history else None,
        "archive_zip": str(archive_zip),
        **package_report,
    }

    if args.final_eval:
        submission_dir = run_dir / "submission"
        materialize_submission(archive_zip, submission_dir, inflate_mode="modified")
        report_path = run_evaluate_submission(submission_dir, args.eval_device, args.video_names, env={"FORCE_AV_DATASET": "1"})
        record = metric_record(
            label=args.label,
            archive_zip=submission_dir / "archive.zip",
            device=args.eval_device,
            report_path=report_path,
            extra=record,
        )

    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "mask_adapter_results.jsonl", record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
