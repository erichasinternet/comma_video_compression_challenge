#!/usr/bin/env python
"""Warm-start metric fine-tune from the exact Quantizr #55 payload."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from q55_common import (
    ARCH_PAYLOAD,
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    POSE_PAYLOAD,
    POSE_QPACK_PAYLOAD,
    REPO_ROOT,
    append_jsonl,
    make_archive_zip,
    materialize_submission,
    metric_record,
    run_evaluate_submission,
    summarize_archive,
    unzip_archive,
    write_json,
)


DEFAULT_ARCH_CONFIG = {
    "cond_dim": 48,
    "depth_mult": 1,
    "shared_c1": 56,
    "shared_c2": 64,
    "frame_hidden": 52,
    "padding_mode": "zeros",
}


VARIANT_TO_PACK_MODE = {
    "fp16": "none",
    "mixed_int8": "int8",
    "mixed_int8_heads_fp16": "int8_heads_fp16",
}


def load_pose_payload(path: Path) -> torch.Tensor:
    import brotli
    import numpy as np
    import torch

    with open(path, "rb") as f:
        pose_bytes = brotli.decompress(f.read())
    return torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float().contiguous()


def load_model_from_payload(generator, model_br: Path, device, q55) -> dict:
    import brotli
    import torch

    data = torch.load(io.BytesIO(brotli.decompress(model_br.read_bytes())), map_location=device)
    state_dict, source = {}, data.get("tensors", data.get("quantized", {}))
    for name, rec in source.items():
        if rec["weight_kind"] == "fp4_packed":
            nibbles = q55.unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w = q55.FP4Codebook.dequantize_from_nibbles(
                nibbles, rec["scales_fp16"].to(device), rec["weight_shape"]
            )
        else:
            w = rec["weight_fp16"].to(device).float()
        state_dict[f"{name}.weight"] = w
        if rec.get("bias_fp16") is not None:
            state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for k, v in data.get("dense_fp16", {}).items():
        state_dict[k] = v.to(device).float()

    target = generator.state_dict()
    compatible = {
        k: v
        for k, v in state_dict.items()
        if k in target and tuple(target[k].shape) == tuple(v.shape)
    }
    skipped = sorted(set(state_dict) - set(compatible))
    generator.load_state_dict(compatible, strict=False)
    logging.info(
        "Loaded %d compatible #55 tensors into current architecture; skipped %d shape-mismatched tensors.",
        len(compatible),
        len(skipped),
    )
    if skipped:
        logging.info("Skipped tensors: %s", ", ".join(skipped[:24]) + (" ..." if len(skipped) > 24 else ""))
    generator.float()
    return {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}


def build_pipeline(args, q55):
    runs = []
    if args.adapt_epochs > 0:
        runs.append(
            q55.PipelineRun(
                name="q55_stage1_mask_adapt",
                stage=q55.Stage.ANCHOR,
                epochs=args.adapt_epochs,
                lr=args.adapt_lr,
                qat_start_epoch=0,
                frame1_fade_epochs=0,
                error_boost=args.error_boost,
            )
        )
    if args.seg_epochs > 0:
        runs.append(
            q55.PipelineRun(
                name="q55_stage2_seg_hardmargin",
                stage=q55.Stage.JOINT,
                epochs=args.seg_epochs,
                lr=args.seg_lr,
                qat_start_epoch=0,
                frame1_fade_epochs=max(1, min(args.seg_epochs, args.frame1_fade_epochs)),
                pose_weight=args.pose_weight,
                error_boost=args.error_boost,
            )
        )
    if args.pose_epochs > 0:
        runs.append(
            q55.PipelineRun(
                name="q55_stage3_pose_restore",
                stage=q55.Stage.FINETUNE,
                epochs=args.pose_epochs,
                lr=args.pose_lr,
                qat_start_epoch=0,
                frame1_fade_epochs=0,
                pose_weight=args.pose_weight,
                error_boost=args.error_boost,
            )
        )
    return runs


def maybe_pack_qpack(archive_dir: Path, variant: str | None) -> None:
    if variant is None:
        return
    import brotli
    from pack_model import build_qpack, load_model_payload

    class Args:
        pass

    pack_args = Args()
    pack_args.archive_dir = archive_dir
    pack_args.model_br = None
    pack_args.fp4 = None

    data = load_model_payload(pack_args)
    qpack = build_qpack(data, quantize_fp16=VARIANT_TO_PACK_MODE[variant])
    out = archive_dir / MODEL_QPACK_PAYLOAD
    out.write_bytes(brotli.compress(qpack, quality=11, lgwin=24))
    (archive_dir / MODEL_PAYLOAD).unlink(missing_ok=True)


def maybe_pack_pose_qpack(archive_dir: Path, variant: str | None) -> dict | None:
    if variant is None:
        return None
    import brotli
    from pack_pose import build_pose_qpack, load_pose_payload as load_pose_array

    pose_path = archive_dir / POSE_PAYLOAD
    pose = load_pose_array(pose_path)
    qpack = build_pose_qpack(pose, variant)
    out = archive_dir / POSE_QPACK_PAYLOAD
    out.write_bytes(brotli.compress(qpack, quality=11, lgwin=24))
    report = {
        "variant": variant,
        "pose_npy_br_bytes": pose_path.stat().st_size,
        "pose_qpack_br_bytes": out.stat().st_size,
        "saved_bytes": pose_path.stat().st_size - out.stat().st_size,
    }
    pose_path.unlink(missing_ok=True)
    return report


def arch_config_from_args(args) -> dict:
    return {
        "cond_dim": args.cond_dim,
        "depth_mult": args.depth_mult,
        "shared_c1": args.shared_c1,
        "shared_c2": args.shared_c2,
        "frame_hidden": args.frame_hidden,
        "padding_mode": args.padding_mode,
    }


def arch_config_is_default(config: dict) -> bool:
    return all(config.get(k) == v for k, v in DEFAULT_ARCH_CONFIG.items())


def maybe_write_arch_payload(archive_dir: Path, config: dict) -> None:
    arch_path = archive_dir / ARCH_PAYLOAD
    if arch_config_is_default(config):
        arch_path.unlink(missing_ok=True)
        return
    import brotli

    payload = {"format": "quantizr_arch_v1", "config": config}
    arch_path.write_bytes(brotli.compress(json.dumps(payload, sort_keys=True).encode("utf-8"), quality=11, lgwin=24))


def archive_payloads_for(args) -> list[str]:
    model_payload = MODEL_QPACK_PAYLOAD if args.qpack_variant else MODEL_PAYLOAD
    pose_payload = POSE_QPACK_PAYLOAD if args.pose_pack_variant else POSE_PAYLOAD
    payloads = [MASK_PAYLOAD, pose_payload, model_payload]
    if not arch_config_is_default(arch_config_from_args(args)):
        payloads.append(ARCH_PAYLOAD)
    return payloads


def preload_rgb_pairs(files, video_dir: Path, batch_size: int, device, decode_backend: str, q55):
    import torch

    if decode_backend == "dali":
        return q55.preload_video_pair_cache_dali(files, video_dir, batch_size, device)
    if decode_backend != "av":
        raise ValueError(f"unknown decode backend: {decode_backend}")

    from frame_utils import AVVideoDataset

    logging.info("Preloading raw video RGB pairs via AVVideoDataset on CPU...")
    os.environ["FORCE_AV_DATASET"] = "1"
    ds = AVVideoDataset(files, data_dir=video_dir, batch_size=batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    batches = [batch.cpu().contiguous() for _, _, batch in ds]
    if not batches:
        raise RuntimeError("No video data was loaded by AVVideoDataset.")
    return torch.cat(batches, dim=0).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--crf", type=int, choices=[50, 52, 54, 56], default=None)
    parser.add_argument("--mask-source", choices=["archive", "crf"], default="crf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-backend", choices=["dali", "av"], default="dali")
    parser.add_argument("--eval-device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--zero-eval", action="store_true")
    parser.add_argument(
        "--zero-score-max",
        type=float,
        default=None,
        help="Abort before training if the zero-step official score is above this value.",
    )
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--qpack-variant", choices=sorted(VARIANT_TO_PACK_MODE), default=None)
    parser.add_argument(
        "--pose-pack-variant",
        choices=["fp16", "int16_per_dim", "int12_per_dim", "int10_per_dim"],
        default=None,
    )
    parser.add_argument("--adapt-epochs", type=int, default=12)
    parser.add_argument("--seg-epochs", type=int, default=16)
    parser.add_argument("--pose-epochs", type=int, default=12)
    parser.add_argument("--adapt-lr", type=float, default=1e-5)
    parser.add_argument("--seg-lr", type=float, default=5e-6)
    parser.add_argument("--pose-lr", type=float, default=3e-6)
    parser.add_argument("--error-boost", type=float, default=19.0)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--frame1-fade-epochs", type=int, default=12)
    parser.add_argument("--cond-dim", type=int, default=DEFAULT_ARCH_CONFIG["cond_dim"])
    parser.add_argument("--depth-mult", type=int, default=DEFAULT_ARCH_CONFIG["depth_mult"])
    parser.add_argument("--shared-c1", type=int, default=DEFAULT_ARCH_CONFIG["shared_c1"])
    parser.add_argument("--shared-c2", type=int, default=DEFAULT_ARCH_CONFIG["shared_c2"])
    parser.add_argument("--frame-hidden", type=int, default=DEFAULT_ARCH_CONFIG["frame_hidden"])
    parser.add_argument(
        "--padding-mode",
        choices=["zeros", "reflect", "replicate", "circular"],
        default=DEFAULT_ARCH_CONFIG["padding_mode"],
    )
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if args.mask_source == "crf" and args.crf is None:
        raise ValueError("--crf is required when --mask-source=crf")

    base_archive = args.base_archive.resolve()
    if not base_archive.exists():
        raise FileNotFoundError(base_archive)

    import torch
    import compress as q55

    label = args.label or (
        f"q55_crf{args.crf}_warmstart" if args.mask_source == "crf" else "q55_archive_mask_warmstart"
    )
    run_dir = args.out_dir / label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    run_dir.mkdir(parents=True, exist_ok=True)
    arch_config = arch_config_from_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "warmstart.log")],
    )

    unzip_archive(base_archive, archive_dir)
    maybe_write_arch_payload(archive_dir, arch_config)
    if args.mask_source == "crf":
        for path in archive_dir.glob("mask*.obu*"):
            path.unlink()

    device = torch.device(args.device)
    files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]

    logging.info("Loading evaluator models and source videos...")
    segnet = q55.SegNet().eval().to(device)
    segnet.load_state_dict(q55.load_file(q55.segnet_sd_path, device=str(device)))
    posenet = q55.PoseNet().eval().to(device)
    posenet.load_state_dict(q55.load_file(q55.posenet_sd_path, device=str(device)))
    dist_net = q55.DistortionNet().eval().to(device)
    dist_net.load_state_dicts(q55.posenet_sd_path, q55.segnet_sd_path, device)
    for model in (segnet, posenet, dist_net):
        for p in model.parameters():
            p.requires_grad = False

    rgb_pairs_all = preload_rgb_pairs(files, args.video_dir, args.batch_size, device, args.decode_backend, q55)
    if args.mask_source == "archive":
        from q55_mask_alloc import decode_archive_masks

        logging.info("Decoding unchanged #55 archive masks for training cache...")
        mask_frames_all = torch.from_numpy(decode_archive_masks(archive_dir / MASK_PAYLOAD)).long().contiguous()
    else:
        mask_frames_all = q55.extract_and_compress_masks(
            rgb_pairs_all,
            segnet,
            device,
            args.crf,
            archive_dir,
            batch_size=args.batch_size,
        )
    pose6_all = load_pose_payload(archive_dir / POSE_PAYLOAD)

    if args.zero_eval:
        zero_archive_dir = run_dir / "zero_step_archive"
        if zero_archive_dir.exists():
            shutil.rmtree(zero_archive_dir)
        shutil.copytree(archive_dir, zero_archive_dir)
        maybe_write_arch_payload(zero_archive_dir, arch_config)
        maybe_pack_qpack(zero_archive_dir, args.qpack_variant)
        zero_pose_pack_report = maybe_pack_pose_qpack(zero_archive_dir, args.pose_pack_variant)
        make_archive_zip(zero_archive_dir, archive_zip, archive_payloads_for(args))
        zero_submission = run_dir / "zero_step_submission"
        materialize_submission(archive_zip, zero_submission, inflate_mode="modified")
        zero_report = run_evaluate_submission(zero_submission, args.eval_device, args.video_names)
        zero_record = metric_record(
            label=f"{label}_zero_step",
            archive_zip=zero_submission / "archive.zip",
            device=args.eval_device,
            report_path=zero_report,
            extra={
                "crf": args.crf,
                "mask_source": args.mask_source,
                "stage": "zero_step",
                "qpack_variant": args.qpack_variant,
                "pose_pack_variant": args.pose_pack_variant,
                "pose_pack_report": zero_pose_pack_report,
                "base_archive": str(base_archive),
                "arch_config": arch_config,
            },
        )
        write_json(run_dir / "zero_step_metrics.json", zero_record)
        append_jsonl(args.out_dir / "warmstart_results.jsonl", zero_record)
        if args.zero_score_max is not None and zero_record["score"] > args.zero_score_max:
            abort_record = dict(zero_record)
            abort_record.update(
                {
                    "label": label,
                    "stage": "zero_step_abort",
                    "aborted": True,
                    "abort_reason": (
                        f"zero-step score {zero_record['score']:.6f} exceeded "
                        f"--zero-score-max {args.zero_score_max:.6f}"
                    ),
                    "zero_step_label": zero_record["label"],
                    "zero_score_max": args.zero_score_max,
                }
            )
            logging.warning(abort_record["abort_reason"])
            write_json(run_dir / "metrics.json", abort_record)
            append_jsonl(args.out_dir / "warmstart_results.jsonl", abort_record)
            return

    loader = q55.CachedPairLoader(rgb_pairs_all, mask_frames_all, pose6_all, args.batch_size, device)
    generator = q55.JointFrameGenerator(**arch_config).to(device)
    current_state_dict = load_model_from_payload(generator, archive_dir / MODEL_PAYLOAD, device, q55)

    for run in build_pipeline(args, q55):
        logging.info("\n" + "=" * 50)
        logging.info(f"STARTING WARMSTART RUN: {run.name}")
        logging.info("=" * 50)
        current_state_dict = q55.train_run(
            run,
            generator,
            loader,
            device,
            archive_dir,
            (segnet, posenet, dist_net),
            current_state_dict,
        )

    maybe_pack_qpack(archive_dir, args.qpack_variant)
    pose_pack_report = maybe_pack_pose_qpack(archive_dir, args.pose_pack_variant)
    payloads = archive_payloads_for(args)
    make_archive_zip(archive_dir, archive_zip, payloads)

    final_record = {
        "label": f"{label}_packaged",
        "stage": "packaged",
        "crf": args.crf,
        "mask_source": args.mask_source,
        "qpack_variant": args.qpack_variant,
        "pose_pack_variant": args.pose_pack_variant,
        "pose_pack_report": pose_pack_report,
        "archive_zip": str(archive_zip),
        "archive_summary": summarize_archive(archive_zip),
        "base_archive": str(base_archive),
        "base_archive_summary": summarize_archive(base_archive),
        "arch_config": arch_config,
    }
    if args.final_eval:
        final_submission = run_dir / "final_submission"
        materialize_submission(archive_zip, final_submission, inflate_mode="modified")
        final_report = run_evaluate_submission(final_submission, args.eval_device, args.video_names)
        final_record = metric_record(
            label=f"{label}_final",
            archive_zip=final_submission / "archive.zip",
            device=args.eval_device,
            report_path=final_report,
            extra={
                "crf": args.crf,
                "stage": "final",
                "mask_source": args.mask_source,
                "qpack_variant": args.qpack_variant,
                "pose_pack_variant": args.pose_pack_variant,
                "pose_pack_report": pose_pack_report,
                "base_archive": str(base_archive),
                "base_archive_summary": summarize_archive(base_archive),
                "arch_config": arch_config,
            },
        )

    write_json(run_dir / "metrics.json", final_record)
    append_jsonl(args.out_dir / "warmstart_results.jsonl", final_record)
    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
