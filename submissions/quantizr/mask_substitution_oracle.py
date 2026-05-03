#!/usr/bin/env python
"""Build and optionally evaluate archive variants with substituted mask payloads.

This is a pre-GPU diagnostic for the next strategic question:
can a smaller or cleaner mask representation preserve Candidate A quality when
plugged into the existing Quantizr inflater?
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT_DIR))

import compress as q  # noqa: E402
from modules import SegNet, segnet_sd_path  # noqa: E402


def parse_candidate_spec(spec: str) -> tuple[str, str, dict[str, str]]:
    """Parse NAME:mode,k=v,k=v."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError("candidate must look like NAME:copy or NAME:av1,crf=48,size=512x384")
    name, rest = spec.split(":", 1)
    if not name:
        raise argparse.ArgumentTypeError("candidate name cannot be empty")
    parts = [part.strip() for part in rest.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("candidate mode cannot be empty")
    mode = parts[0]
    opts: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"candidate option lacks '=': {part}")
        key, value = part.split("=", 1)
        opts[key.strip()] = value.strip()
    return name, mode, opts


def unzip_archive(archive_zip: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        zf.extractall(out_dir)
    return out_dir


def copy_base_payloads(base_archive_dir: Path, out_archive_dir: Path):
    out_archive_dir.mkdir(parents=True, exist_ok=True)
    for name in (q.MODEL_PAYLOAD_NAME, q.MASK_PAYLOAD_NAME, q.POSE_PAYLOAD_NAME, q.LATENT_PAYLOAD_NAME):
        src = base_archive_dir / name
        if src.exists():
            shutil.copy2(src, out_archive_dir / name)
    for required in (q.MODEL_PAYLOAD_NAME, q.POSE_PAYLOAD_NAME):
        if not (out_archive_dir / required).exists():
            raise FileNotFoundError(f"missing required base payload: {base_archive_dir / required}")


def payload_sizes(archive_dir: Path) -> dict[str, int]:
    sizes = {}
    for path in sorted(archive_dir.glob("*")):
        if path.is_file():
            sizes[path.name] = path.stat().st_size
    return sizes


def load_segnet(device: torch.device) -> SegNet:
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    for p in segnet.parameters():
        p.requires_grad = False
    return segnet


def build_av1_mask_payload(
    *,
    candidate_name: str,
    out_archive_dir: Path,
    rgb_pairs: torch.Tensor,
    segnet: SegNet,
    device: torch.device,
    crf: int,
    size: tuple[int, int],
    batch_size: int,
    shared_cache_root: Path | None,
):
    cache_dir = shared_cache_root / candidate_name if shared_cache_root is not None else out_archive_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    q.extract_and_compress_masks(
        rgb_pairs,
        segnet,
        device,
        crf,
        out_archive_dir,
        batch_size=batch_size,
        cache_dir=cache_dir,
        mask_encode_size=size,
    )


def package_variant(out_archive_dir: Path, archive_zip: Path) -> int:
    return q.package_submission_archive(
        out_archive_dir,
        archive_zip,
        include_latents=(out_archive_dir / q.LATENT_PAYLOAD_NAME).exists(),
        include_mask=(out_archive_dir / q.MASK_PAYLOAD_NAME).exists(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-archive-dir", type=Path)
    base.add_argument("--base-archive-zip", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=ROOT_DIR / "videos")
    parser.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    parser.add_argument("--shared-cache-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--decode-backend", choices=["av", "dali"], default="av")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="NAME:copy or NAME:av1,crf=48,size=512x384. Can be repeated.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.shared_cache_root is not None:
        args.shared_cache_root.mkdir(parents=True, exist_ok=True)

    if args.base_archive_zip is not None:
        base_archive_dir = unzip_archive(args.base_archive_zip, args.out_root / "_base_archive")
    else:
        base_archive_dir = args.base_archive_dir
    if base_archive_dir is None or not base_archive_dir.exists():
        raise FileNotFoundError(f"base archive dir not found: {base_archive_dir}")

    candidate_specs = args.candidate or [
        "M0:copy",
        "M2_crf48:av1,crf=48,size=512x384",
        "M_clean_crf0:av1,crf=0,size=512x384",
    ]
    specs = [parse_candidate_spec(spec) for spec in candidate_specs]
    needs_masks = any(mode == "av1" for _, mode, _ in specs)

    rgb_pairs = None
    segnet = None
    device = torch.device(args.device)
    if needs_masks:
        files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]
        rgb_pairs = q.get_rgb_pairs(
            files,
            args.video_dir,
            args.batch_size,
            device,
            args.shared_cache_root,
            args.decode_backend,
        )
        segnet = load_segnet(device)

    results_path = args.out_root / "mask_substitution_results.jsonl"
    with results_path.open("a") as results_file:
        for name, mode, opts in specs:
            variant_root = args.out_root / name
            out_archive_dir = variant_root / "archive"
            archive_zip = variant_root / "archive.zip"
            copy_base_payloads(base_archive_dir, out_archive_dir)

            if mode == "copy":
                logging.info("[%s] using base mask payload", name)
            elif mode == "av1":
                if rgb_pairs is None or segnet is None:
                    raise RuntimeError("internal error: av1 candidate requested without loaded masks")
                crf = int(opts.get("crf", "50"))
                size = q.parse_size_arg(opts.get("size", "512x384"))
                logging.info("[%s] generating AV1 mask payload crf=%d size=%dx%d", name, crf, size[0], size[1])
                build_av1_mask_payload(
                    candidate_name=name,
                    out_archive_dir=out_archive_dir,
                    rgb_pairs=rgb_pairs,
                    segnet=segnet,
                    device=device,
                    crf=crf,
                    size=size,
                    batch_size=args.batch_size,
                    shared_cache_root=args.shared_cache_root,
                )
            else:
                raise ValueError(f"unknown candidate mode: {mode}")

            archive_bytes = package_variant(out_archive_dir, archive_zip)
            row: dict[str, object] = {
                "name": name,
                "mode": mode,
                "options": opts,
                "archive_zip": str(archive_zip),
                "archive_bytes": archive_bytes,
                "payload_sizes": payload_sizes(out_archive_dir),
            }

            if args.run_official:
                metrics = q.run_official_evaluation(
                    submission_dir=THIS_DIR,
                    archive_zip_path=archive_zip,
                    video_names_file=args.video_names,
                    eval_device=args.eval_device,
                )
                row.update(metrics)
                logging.info(
                    "[%s] score=%.5f seg=%.8f pose=%.8f rate=%.8f bytes=%d",
                    name,
                    metrics["score"],
                    metrics["segnet_dist"],
                    metrics["posenet_dist"],
                    metrics["rate"],
                    int(metrics["archive_bytes"]),
                )
            else:
                logging.info("[%s] archive=%s bytes=%d payloads=%s", name, archive_zip, archive_bytes, row["payload_sizes"])

            print(json.dumps(row, sort_keys=True), file=results_file, flush=True)

    logging.info("Wrote %s", results_path)


if __name__ == "__main__":
    main()
