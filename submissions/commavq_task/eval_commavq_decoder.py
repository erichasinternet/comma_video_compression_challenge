#!/usr/bin/env python3
"""Evaluate real commaVQ decoder output as an information-content oracle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from submissions.commavq_task.common import (
    MODEL_HW,
    FeatureTap,
    build_distortion,
    collect_targets,
    evaluate_frames,
    load_original_pairs_by_indices,
)
from frame_utils import camera_size


COMMAVQ_CROP_SIZE = (512, 256)
COMMAVQ_SCALE = 567 / 455
COMMAVQ_CY = 47.6


def import_commavq(commavq_root: Path):
    if not commavq_root.exists():
        raise FileNotFoundError(
            f"commaVQ root not found: {commavq_root}. "
            "Clone https://github.com/commaai/commavq and pass --commavq-root."
        )
    if str(commavq_root) not in sys.path:
        sys.path.insert(0, str(commavq_root))
    from utils.vqvae import CompressorConfig, Decoder

    return CompressorConfig, Decoder


def place_commavq_crop(decoded: torch.Tensor, *, fill: float) -> torch.Tensor:
    """Invert commaVQ's camera crop approximately, then resize to evaluator size."""
    target_h, target_w = camera_size[1], camera_size[0]
    crop_w = int(COMMAVQ_CROP_SIZE[0] * COMMAVQ_SCALE)
    crop_h = int(COMMAVQ_CROP_SIZE[1] * COMMAVQ_SCALE)
    x0 = target_h // 2 - crop_h // 2 - int(COMMAVQ_CY * COMMAVQ_SCALE) // 2
    y0 = target_w // 2 - crop_w // 2
    crop = F.interpolate(decoded, size=(crop_h, crop_w), mode="bicubic", align_corners=False).clamp(0, 255)
    canvas = torch.full(
        (decoded.shape[0], 3, target_h, target_w),
        fill,
        device=decoded.device,
        dtype=decoded.dtype,
    )
    canvas[:, :, x0 : x0 + crop_h, y0 : y0 + crop_w] = crop
    return F.interpolate(canvas, size=MODEL_HW, mode="bicubic", align_corners=False).clamp(0, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commavq-root", type=Path, default=Path(os.environ.get("COMMAVQ_ROOT", "/tmp/commavq")))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--placement", choices=["inverse_crop", "stretch"], default="inverse_crop")
    parser.add_argument("--fill", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokens_np = np.load(args.tokens).astype(np.int64)
    if tokens_np.ndim != 3 or tokens_np.shape[1:] != (2, 128):
        raise ValueError(f"expected tokens shape [N,2,128], got {tokens_np.shape}")
    sample_ids = json.loads(args.sample_ids.read_text()) if args.sample_ids else list(range(tokens_np.shape[0]))

    original = load_original_pairs_by_indices(
        data_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
        batch_size=max(args.batch_size, 8),
    )
    distortion = build_distortion(device)
    seg_tap = FeatureTap(distortion.segnet, [])
    pose_tap = FeatureTap(distortion.posenet, [])
    targets = collect_targets(
        distortion=distortion,
        original_cpu=original,
        device=device,
        batch_size=args.batch_size,
        seg_tap=seg_tap,
        pose_tap=pose_tap,
    )

    CompressorConfig, Decoder = import_commavq(args.commavq_root)
    config = CompressorConfig()
    with torch.device("meta"):
        decoder = Decoder(config)
    decoder.load_state_dict_from_url(
        "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin",
        assign=True,
    )
    decoder = decoder.eval().to(device)

    flat_tokens = torch.from_numpy(tokens_np.reshape(-1, 128)).to(device)
    decoded = []
    with torch.inference_mode():
        for start in tqdm(range(0, flat_tokens.shape[0], args.batch_size), desc="commaVQ decode"):
            out = decoder(flat_tokens[start : start + args.batch_size]).clamp(0, 255)
            if args.placement == "stretch":
                out = F.interpolate(out, size=MODEL_HW, mode="bicubic", align_corners=False).clamp(0, 255)
            else:
                out = place_commavq_crop(out, fill=args.fill)
            decoded.append(out.cpu())
    frames = torch.cat(decoded, dim=0).reshape(tokens_np.shape[0], 2, 3, MODEL_HW[0], MODEL_HW[1])
    metrics = evaluate_frames(frames=frames, targets=targets, distortion=distortion, batch_size=args.batch_size)
    metrics.update(
        {
            "kind": "commavq_decoder_oracle",
            "sample_ids": sample_ids,
            "tokens_shape": list(tokens_np.shape),
            "raw_10bit_bytes": int((tokens_np.size * 10 + 7) // 8),
            "commavq_root": str(args.commavq_root),
            "placement": args.placement,
            "fill": args.fill,
        }
    )
    torch.save({"frames": frames.to(torch.uint8), "sample_ids": sample_ids}, args.out_dir / "decoded_frames.pt")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    seg_tap.close()
    pose_tap.close()


if __name__ == "__main__":
    main()
