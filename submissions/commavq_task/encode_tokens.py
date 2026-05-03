#!/usr/bin/env python3
"""Encode challenge frame pairs into commaVQ tokens.

This is an offline compression-side tool. The commaVQ encoder weights are not
used by inflate.py and should not be included in a submitted archive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from submissions.commavq_task.common import (
    DEFAULT_HARD8,
    load_original_pairs_by_indices,
    parse_indices,
)


def import_commavq(commavq_root: Path):
    if not commavq_root.exists():
        raise FileNotFoundError(
            f"commaVQ root not found: {commavq_root}. "
            "Clone https://github.com/commaai/commavq and pass --commavq-root."
        )
    if str(commavq_root) not in sys.path:
        sys.path.insert(0, str(commavq_root))
    from utils.video import transform_img
    from utils.vqvae import CompressorConfig, Encoder

    return transform_img, CompressorConfig, Encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commavq-root", type=Path, default=Path(os.environ.get("COMMAVQ_ROOT", "/tmp/commavq")))
    parser.add_argument("--preset", choices=["hard8", "sequential"], default="hard8")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--subset", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
    parser.add_argument("--uncompressed-dir", type=Path, default=ROOT / "videos")
    args = parser.parse_args()

    transform_img, CompressorConfig, Encoder = import_commavq(args.commavq_root)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = parse_indices(args.indices, offset=args.offset, subset=args.subset, preset=args.preset)
    original = load_original_pairs_by_indices(
        data_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        sample_indices=sample_ids,
        batch_size=max(args.batch_size, 8),
    )

    flat_frames = original.reshape(-1, *original.shape[2:]).numpy()
    transformed = np.stack([transform_img(frame) for frame in flat_frames], axis=0)
    frames = torch.from_numpy(transformed).permute(0, 3, 1, 2).float()

    config = CompressorConfig()
    with torch.device("meta"):
        encoder = Encoder(config)
    encoder.load_state_dict_from_url(
        "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin",
        assign=True,
    )
    encoder = encoder.eval().to(device)

    encoded = []
    with torch.inference_mode():
        for start in tqdm(range(0, frames.shape[0], args.batch_size), desc="commaVQ encode"):
            batch = frames[start : start + args.batch_size].to(device)
            encoded.append(encoder(batch).detach().cpu())
    tokens = torch.cat(encoded, dim=0).numpy().astype(np.uint16).reshape(len(sample_ids), 2, -1)
    if tokens.shape[-1] != 128:
        raise RuntimeError(f"expected 128 tokens/frame, got shape {tokens.shape}")
    if int(tokens.max()) >= config.vocab_size:
        raise RuntimeError(f"token out of range: max={int(tokens.max())}, vocab={config.vocab_size}")

    np.save(args.out_dir / "tokens.npy", tokens)
    (args.out_dir / "sample_ids.json").write_text(json.dumps(sample_ids) + "\n")
    metrics = {
        "kind": "commavq_tokens",
        "sample_ids": sample_ids,
        "tokens_shape": list(tokens.shape),
        "token_min": int(tokens.min()),
        "token_max": int(tokens.max()),
        "raw_10bit_bytes": int((tokens.size * 10 + 7) // 8),
        "raw_uint16_bytes": int(tokens.nbytes),
        "commavq_root": str(args.commavq_root),
        "hard8_reference": DEFAULT_HARD8,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

