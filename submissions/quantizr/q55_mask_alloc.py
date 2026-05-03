#!/usr/bin/env python
"""Build mixed-CRF mask payloads for Quantizr #55 zero-step evaluation.

This is a rate-allocation diagnostic, not a new representation. It keeps the
full-resolution hard-mask interface, but encodes hard/easy samples into
separate AV1 mask streams with different CRFs and a tiny manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import brotli
import numpy as np

from q55_common import (
    DEFAULT_VIDEO_NAMES,
    MASK_PAYLOAD,
    MODEL_PAYLOAD,
    POSE_PAYLOAD,
    REPO_ROOT,
    append_jsonl,
    ensure_legacy_payloads,
    make_archive_zip,
    materialize_submission,
    metric_record,
    run_evaluate_submission,
    summarize_archive,
    unzip_archive,
    write_json,
)


MASK_MIX_MANIFEST = "mask_mix.json.br"
MIX_FORMAT = "quantizr_mask_mix_v1"
PALETTES = {
    "legacy": [0, 63, 126, 189, 252],
    "wide": [0, 32, 64, 128, 255],
    "mid": [0, 48, 96, 160, 224],
    "unit": [0, 1, 2, 3, 4],
}


def parse_palette(value: str) -> list[int]:
    if value in PALETTES:
        return PALETTES[value]
    parts = [int(x) for x in value.split(",") if x.strip()]
    if len(parts) != 5 or min(parts) < 0 or max(parts) > 255:
        raise ValueError("--palette must be a named palette or five uint8 values")
    return parts


def parse_group_spec(value: str) -> list[tuple[int, float]]:
    groups = []
    for part in value.split(","):
        if not part.strip():
            continue
        crf_s, frac_s = part.split(":", 1)
        crf = int(crf_s)
        frac = float(frac_s)
        if crf < 0 or frac <= 0:
            raise ValueError(f"invalid group spec item: {part}")
        groups.append((crf, frac))
    if not groups:
        raise ValueError("--group-spec produced no groups")
    total = sum(frac for _, frac in groups)
    return [(crf, frac / total) for crf, frac in groups]


def load_rgb_pairs(files, video_dir: Path, batch_size: int, device, decode_backend: str, q55):
    import torch

    if decode_backend == "dali":
        return q55.preload_video_pair_cache_dali(files, video_dir, batch_size, device)
    if decode_backend != "av":
        raise ValueError(f"unknown decode backend: {decode_backend}")

    from frame_utils import AVVideoDataset

    logging.info("Preloading source video RGB pairs via AVVideoDataset on CPU...")
    os.environ["FORCE_AV_DATASET"] = "1"
    ds = AVVideoDataset(files, data_dir=video_dir, batch_size=batch_size, device=torch.device("cpu"))
    ds.prepare_data()
    batches = [batch.cpu().contiguous() for _, _, batch in ds]
    if not batches:
        raise RuntimeError("No video data was loaded by AVVideoDataset.")
    return torch.cat(batches, dim=0).contiguous()


def extract_source_masks(rgb_pairs_all, segnet, device, batch_size: int) -> np.ndarray:
    import einops
    import torch
    from tqdm import tqdm

    masks = []
    with torch.inference_mode():
        for start in tqdm(range(0, rgb_pairs_all.shape[0], batch_size), desc="Extracting source masks"):
            batch = rgb_pairs_all[start : start + batch_size].to(device).float()
            batch = einops.rearrange(batch, "b t h w c -> b t c h w")
            odd_frames = batch[:, 1]
            resized = torch.nn.functional.interpolate(odd_frames, size=(384, 512), mode="bilinear")
            out = segnet(resized)
            masks.append(out.argmax(dim=1).to(torch.uint8).cpu().numpy())
    return np.concatenate(masks, axis=0).astype(np.uint8, copy=False)


def decode_archive_masks(mask_br: Path) -> np.ndarray:
    import av

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
        tmp_obu.write(brotli.decompress(mask_br.read_bytes()))
        tmp_obu_path = tmp_obu.name
    try:
        container = av.open(tmp_obu_path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="gray")
            cls_img = np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4)
            frames.append(cls_img)
        container.close()
    finally:
        os.remove(tmp_obu_path)
    if not frames:
        raise RuntimeError(f"no mask frames decoded from {mask_br}")
    return np.stack(frames).astype(np.uint8, copy=False)


def mask_boundary_scores(masks: np.ndarray) -> np.ndarray:
    horiz = masks[:, :, 1:] != masks[:, :, :-1]
    vert = masks[:, 1:, :] != masks[:, :-1, :]
    return horiz.mean(axis=(1, 2)) + vert.mean(axis=(1, 2))


def mask_histograms(masks: np.ndarray) -> np.ndarray:
    out = np.zeros((masks.shape[0], 5), dtype=np.float32)
    flat = masks.reshape(masks.shape[0], -1)
    for i in range(masks.shape[0]):
        out[i] = np.bincount(flat[i], minlength=5)[:5]
    out /= float(flat.shape[1])
    return out


def assign_groups(scores: np.ndarray, group_spec: list[tuple[int, float]]) -> np.ndarray:
    n = scores.shape[0]
    hardest_first = np.argsort(-scores, kind="mergesort")
    counts = [int(round(frac * n)) for _, frac in group_spec]
    counts[-1] += n - sum(counts)
    if counts[-1] < 0:
        raise ValueError(f"group fractions over-assigned frames: {counts}")

    group_for_frame = np.empty(n, dtype=np.int32)
    cursor = 0
    for group_id, count in enumerate(counts):
        frame_ids = hardest_first[cursor : cursor + count]
        group_for_frame[frame_ids] = group_id
        cursor += count
    return group_for_frame


def order_group(indices: np.ndarray, scores: np.ndarray, hists: np.ndarray, strategy: str) -> list[int]:
    if strategy == "original":
        return [int(x) for x in np.sort(indices)]
    if strategy == "boundary":
        return [int(x) for x in indices[np.argsort(-scores[indices], kind="mergesort")]]
    if strategy == "hist":
        keys = [indices]
        for c in reversed(range(hists.shape[1])):
            keys.append(hists[indices, c])
        order = np.lexsort(tuple(keys))
        return [int(x) for x in indices[order]]
    raise ValueError(f"unknown order strategy: {strategy}")


def encode_mask_group(
    *,
    masks: np.ndarray,
    frame_indices: list[int],
    palette: list[int],
    crf: int,
    name: str,
    run_dir: Path,
    q55,
) -> str:
    raw_path = run_dir / f"{name}.yuv"
    obu_path = run_dir / f"{name}.obu"
    br_path = run_dir / f"{name}.obu.br"
    palette_arr = np.asarray(palette, dtype=np.uint8)
    with open(raw_path, "wb") as f:
        for idx in frame_indices:
            f.write(palette_arr[masks[idx]].tobytes())

    ffmpeg_cmd = [
        q55.get_ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        "512x384",
        "-r",
        "10",
        "-i",
        str(raw_path),
        "-c:v",
        "libaom-av1",
        "-crf",
        str(crf),
        "-cpu-used",
        "0",
        "-row-mt",
        "1",
        "-g",
        "1200",
        "-keyint_min",
        "1200",
        "-lag-in-frames",
        "48",
        "-arnr-strength",
        "0",
        "-aq-mode",
        "0",
        "-aom-params",
        "enable-cdef=0:enable-intrabc=1:enable-obmc=0",
        "-f",
        "obu",
        str(obu_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    br_path.write_bytes(brotli.compress(obu_path.read_bytes(), quality=11, lgwin=24))
    raw_path.unlink(missing_ok=True)
    obu_path.unlink(missing_ok=True)
    return br_path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument("--eval-device", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--video-names", type=Path, default=DEFAULT_VIDEO_NAMES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-backend", choices=["dali", "av"], default="dali")
    parser.add_argument("--mask-source", choices=["archive", "source"], default="archive")
    parser.add_argument("--group-spec", default="50:0.20,54:0.35,58:0.45")
    parser.add_argument("--order", choices=["original", "boundary", "hist"], default="hist")
    parser.add_argument("--palette", default="legacy")
    parser.add_argument("--inflate-mode", choices=["upstream", "modified"], default="modified")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    base_archive = args.base_archive.resolve()
    if not base_archive.exists():
        raise FileNotFoundError(base_archive)

    import torch
    import compress as q55

    group_spec = parse_group_spec(args.group_spec)
    palette = parse_palette(args.palette)
    label = args.label or (
        "qmask_"
        + "_".join(f"crf{crf}_{frac:.2f}" for crf, frac in group_spec).replace(".", "p")
        + f"_{args.order}_{args.palette}_{args.device}"
    )
    run_dir = args.out_dir / label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    submission_dir = run_dir / "submission"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "mask_alloc.log")],
    )

    unzip_archive(base_archive, archive_dir)
    ensure_legacy_payloads(archive_dir)
    device = torch.device(args.device)
    decode_backend = args.decode_backend
    if args.mask_source == "archive":
        logging.info("Decoding masks from the base #55 archive payload...")
        masks = decode_archive_masks(archive_dir / MASK_PAYLOAD)
        decode_backend = "archive"
    else:
        files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]
        logging.info("Loading SegNet and source videos for QMASK...")
        segnet = q55.SegNet().eval().to(device)
        segnet.load_state_dict(q55.load_file(q55.segnet_sd_path, device=str(device)))
        for p in segnet.parameters():
            p.requires_grad = False

        if device.type != "cuda":
            decode_backend = "av"
        rgb_pairs_all = load_rgb_pairs(files, args.video_dir, args.batch_size, device, decode_backend, q55)
        masks = extract_source_masks(rgb_pairs_all, segnet, device, args.batch_size)

    for path in archive_dir.glob("mask*.obu*"):
        path.unlink()
    scores = mask_boundary_scores(masks)
    hists = mask_histograms(masks)
    group_for_frame = assign_groups(scores, group_spec)

    manifest_groups = []
    entries: list[list[int]] = [[-1, -1] for _ in range(masks.shape[0])]
    payload_names = [MODEL_PAYLOAD, POSE_PAYLOAD, MASK_MIX_MANIFEST]

    for group_id, (crf, frac) in enumerate(group_spec):
        indices = np.nonzero(group_for_frame == group_id)[0]
        if indices.size == 0:
            continue
        ordered = order_group(indices, scores, hists, args.order)
        name = f"mask_g{len(manifest_groups):02d}_crf{crf}"
        payload = encode_mask_group(
            masks=masks,
            frame_indices=ordered,
            palette=palette,
            crf=crf,
            name=name,
            run_dir=archive_dir,
            q55=q55,
        )
        local_group_id = len(manifest_groups)
        for local_idx, frame_idx in enumerate(ordered):
            entries[frame_idx] = [local_group_id, local_idx]
        manifest_groups.append(
            {
                "name": name,
                "payload": payload,
                "crf": crf,
                "requested_fraction": frac,
                "count": len(ordered),
                "payload_bytes": (archive_dir / payload).stat().st_size,
            }
        )
        payload_names.append(payload)

    if any(group_id < 0 for group_id, _ in entries):
        raise RuntimeError("not all mask frames were assigned to mixed payload groups")

    manifest = {
        "format": MIX_FORMAT,
        "num_frames": int(masks.shape[0]),
        "height": int(masks.shape[1]),
        "width": int(masks.shape[2]),
        "palette": palette,
        "group_spec": [{"crf": crf, "fraction": frac} for crf, frac in group_spec],
        "assignment": "boundary_descending",
        "order": args.order,
        "groups": manifest_groups,
        "entries": entries,
    }
    (archive_dir / MASK_MIX_MANIFEST).write_bytes(
        brotli.compress(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"), quality=11, lgwin=24)
    )

    make_archive_zip(archive_dir, archive_zip, payload_names)
    materialize_submission(archive_zip=archive_zip, submission_dir=submission_dir, inflate_mode=args.inflate_mode)
    eval_device = args.eval_device or args.device
    env = {"FORCE_AV_DATASET": "1"} if eval_device == "cuda" and decode_backend == "av" else None
    report_path = run_evaluate_submission(submission_dir, eval_device, args.video_names, env=env)

    record = metric_record(
        label=label,
        archive_zip=submission_dir / "archive.zip",
        device=eval_device,
        report_path=report_path,
        extra={
            "base_archive": str(base_archive),
            "base_archive_summary": summarize_archive(base_archive),
            "decode_backend": decode_backend,
            "mask_source": args.mask_source,
            "group_spec": manifest["group_spec"],
            "order": args.order,
            "palette": palette,
            "mask_mix_manifest_bytes": (archive_dir / MASK_MIX_MANIFEST).stat().st_size,
            "mask_mix_groups": manifest_groups,
        },
    )
    write_json(run_dir / "metrics.json", record)
    append_jsonl(args.out_dir / "mask_alloc_results.jsonl", record)
    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
