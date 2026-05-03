#!/usr/bin/env python
"""Render semantic/cartoon frame sequences for the SCV prototype."""

from __future__ import annotations

import shutil
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from scv_eval_proxy import (
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_W,
    build_targets,
    downsample_frame2_to_seg,
    evaluate_pairs,
    load_evaluators,
    load_rgb_subset,
    metric_table,
    pick_device,
    write_json,
)


CLASS_COLORS = torch.tensor(
    [
        [135, 180, 215],
        [45, 45, 48],
        [218, 218, 190],
        [155, 85, 65],
        [82, 122, 88],
    ],
    dtype=torch.float32,
)

MASK_PREVIEW_COLORS = torch.tensor(
    [
        [70, 130, 180],
        [40, 40, 40],
        [230, 220, 120],
        [190, 80, 80],
        [90, 150, 90],
    ],
    dtype=torch.uint8,
)


def cache_path(cache_dir: Path, subset: int, offset: int) -> Path:
    return cache_dir / f"targets_o{offset}_{subset}.pt"


def load_or_build_cache(args, device: torch.device, segnet, posenet) -> dict:
    path = cache_path(args.cache_dir, args.subset, args.offset)
    if path.exists() and not args.rebuild_cache:
        record = torch.load(path, map_location="cpu")
        if "seg_targets_f1" not in record:
            record["seg_targets_f1"] = build_frame_seg_targets(
                record["gt_pairs"][:, 0],
                segnet,
                device,
                batch_size=args.eval_batch_size,
            )
            torch.save(record, path)
        return record

    gt_pairs = load_rgb_subset(
        video_names=args.video_names,
        video_dir=args.video_dir,
        subset=args.subset,
        offset=args.offset,
        batch_size=args.decode_batch_size,
    )
    targets = build_targets(
        gt_pairs,
        segnet,
        posenet,
        device,
        batch_size=args.eval_batch_size,
        include_logits=args.cache_logits,
    )
    record = {
        "gt_pairs": gt_pairs,
        "seg_targets": targets["seg_targets"],
        "seg_targets_f1": build_frame_seg_targets(
            gt_pairs[:, 0],
            segnet,
            device,
            batch_size=args.eval_batch_size,
        ),
        "pose_targets": targets["pose_targets"],
        "offset": args.offset,
        "subset": args.subset,
    }
    if "seg_logits" in targets:
        record["seg_logits"] = targets["seg_logits"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)
    return record


@torch.inference_mode()
def build_frame_seg_targets(frames_u8: torch.Tensor, segnet, device: torch.device, batch_size: int = 8) -> torch.Tensor:
    preds = []
    for start in range(0, frames_u8.shape[0], batch_size):
        frame = frames_u8[start : start + batch_size]
        pair = torch.stack([frame, frame], dim=1).float()
        x = einops.rearrange(pair, "b t h w c -> b t c h w").to(device)
        preds.append(segnet(segnet.preprocess_input(x)).argmax(dim=1).cpu())
    return torch.cat(preds, dim=0).contiguous()


def palette_from_original(gt_pairs: torch.Tensor, seg_targets: torch.Tensor) -> torch.Tensor:
    frame2_low = downsample_frame2_to_seg(gt_pairs[:, 1])
    colors = []
    fallback = CLASS_COLORS.clone()
    for cls in range(5):
        mask = seg_targets == cls
        if bool(mask.any()):
            pixels = frame2_low.permute(0, 2, 3, 1)[mask]
            colors.append(pixels.float().mean(dim=0))
        else:
            colors.append(fallback[cls])
    return torch.stack(colors).clamp(0, 255)


def mask_edges(mask: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(mask, dtype=torch.float32)
    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], (mask[:, :, 1:] != mask[:, :, :-1]).float())
    edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], (mask[:, :, 1:] != mask[:, :, :-1]).float())
    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], (mask[:, 1:, :] != mask[:, :-1, :]).float())
    edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], (mask[:, 1:, :] != mask[:, :-1, :]).float())
    return edge


def render_palette(mask: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    low = palette[mask.long()].permute(0, 3, 1, 2)
    up = F.interpolate(low, size=(CAMERA_H, CAMERA_W), mode="nearest")
    return up


def render_scv0(cache: dict) -> tuple[torch.Tensor, dict]:
    seg_targets = cache["seg_targets"]
    palette = CLASS_COLORS.clone()
    frame2 = render_palette(seg_targets, palette)
    pairs = torch.stack([frame2, frame2], dim=1)
    pairs = einops.rearrange(pairs, "b t c h w -> b t h w c").round().clamp(0, 255).to(torch.uint8)
    return pairs, {"palette": palette.tolist(), "texture": "none", "frame1": "copy_frame2"}


def render_scv1(cache: dict, *, texture_size: tuple[int, int] = (48, 64), texture_strength: float = 0.35) -> tuple[torch.Tensor, dict]:
    gt_pairs = cache["gt_pairs"]
    seg_targets = cache["seg_targets"]
    palette = palette_from_original(gt_pairs, seg_targets)
    palette_frame = render_palette(seg_targets, palette)

    orig = einops.rearrange(gt_pairs[:, 1].float(), "b h w c -> b c h w")
    low = F.interpolate(orig, size=texture_size, mode="area")
    low = F.interpolate(low, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)

    edge_low = mask_edges(seg_targets)
    edge = F.interpolate(edge_low[:, None], size=(CAMERA_H, CAMERA_W), mode="nearest")
    cartoon = (1.0 - texture_strength) * palette_frame + texture_strength * low
    cartoon = cartoon * (1.0 - 0.35 * edge) + 245.0 * (0.08 * edge)
    cartoon = cartoon.clamp(0, 255)
    pairs = torch.stack([cartoon, cartoon], dim=1)
    pairs = einops.rearrange(pairs, "b t c h w -> b t h w c").round().clamp(0, 255).to(torch.uint8)
    return pairs, {
        "palette": palette.tolist(),
        "texture": "low_frequency_original_frame2",
        "texture_size_hw": list(texture_size),
        "texture_strength": texture_strength,
        "boundary_treatment": "darken_edges",
        "frame1": "copy_frame2",
    }


def per_sample_palettes(frames_u8: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    low = downsample_frame2_to_seg(frames_u8)
    pixels = low.permute(0, 2, 3, 1)
    palettes = []
    for b in range(frames_u8.shape[0]):
        fallback = pixels[b].reshape(-1, 3).float().mean(dim=0)
        colors = []
        for cls in range(5):
            cls_mask = masks[b] == cls
            if bool(cls_mask.any()):
                colors.append(pixels[b][cls_mask].float().mean(dim=0))
            else:
                colors.append(fallback)
        palettes.append(torch.stack(colors))
    return torch.stack(palettes).clamp(0, 255)


def render_palette_batch(masks: torch.Tensor, palettes: torch.Tensor) -> torch.Tensor:
    frames = []
    for b in range(masks.shape[0]):
        frames.append(palettes[b][masks[b].long()].permute(2, 0, 1))
    low = torch.stack(frames)
    return F.interpolate(low, size=(CAMERA_H, CAMERA_W), mode="nearest")


def quantize_rgb(x: torch.Tensor, step: int) -> torch.Tensor:
    return (torch.round(x / float(step)) * float(step)).clamp(0, 255)


def stylize_camera_like(
    frames_u8: torch.Tensor,
    masks: torch.Tensor,
    *,
    blur_size: tuple[int, int],
    color_step: int,
    mask_smooth: float,
    edge_keep: float,
) -> torch.Tensor:
    x = einops.rearrange(frames_u8.float(), "b h w c -> b c h w")
    low = F.interpolate(x, size=blur_size, mode="area")
    low = F.interpolate(low, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)

    palettes = per_sample_palettes(frames_u8, masks)
    class_smooth = render_palette_batch(masks, palettes)
    edge_low = F.max_pool2d(mask_edges(masks)[:, None], kernel_size=3, stride=1, padding=1)
    edge = F.interpolate(edge_low, size=(CAMERA_H, CAMERA_W), mode="nearest")

    # Bilateral-like behavior: smooth aggressively inside semantic regions, but keep
    # more low-pass camera detail near semantic boundaries.
    smooth = (1.0 - mask_smooth) * low + mask_smooth * class_smooth
    smooth = smooth * (1.0 - edge_keep * edge) + low * (edge_keep * edge)
    return quantize_rgb(smooth, color_step)


def render_scvt(
    cache: dict,
    *,
    blur_size: tuple[int, int] = (146, 194),
    color_step: int = 12,
    mask_smooth: float = 0.28,
    edge_keep: float = 0.65,
) -> tuple[torch.Tensor, dict]:
    gt_pairs = cache["gt_pairs"]
    f1 = stylize_camera_like(
        gt_pairs[:, 0],
        cache["seg_targets_f1"],
        blur_size=blur_size,
        color_step=color_step,
        mask_smooth=mask_smooth,
        edge_keep=edge_keep,
    )
    f2 = stylize_camera_like(
        gt_pairs[:, 1],
        cache["seg_targets"],
        blur_size=blur_size,
        color_step=color_step,
        mask_smooth=mask_smooth,
        edge_keep=edge_keep,
    )
    pairs = torch.stack([f1, f2], dim=1)
    pairs = einops.rearrange(pairs, "b t c h w -> b t h w c").round().clamp(0, 255).to(torch.uint8)
    return pairs, {
        "style": "camera_like_texture",
        "blur": "area_downsample_bilinear_upsample",
        "blur_size_hw": list(blur_size),
        "color_quant_step": int(color_step),
        "mask_guided_smoothing": float(mask_smooth),
        "edge_detail_keep": float(edge_keep),
        "frame1": "stylized_original_frame1",
        "frame2": "stylized_original_frame2",
    }


def save_frames(pairs: torch.Tensor, frames_dir: Path, *, keep_raw: bool = False) -> Path | None:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    stale_raw = frames_dir.parent / "frames.raw"
    if stale_raw.exists() and not keep_raw:
        stale_raw.unlink()
    flat = pairs.reshape((-1, CAMERA_H, CAMERA_W, 3)).contiguous().numpy()
    raw_path = None
    if keep_raw:
        raw_path = frames_dir.parent / "frames.raw"
        raw_path.write_bytes(flat.tobytes())
    for idx, frame in enumerate(flat):
        Image.fromarray(frame).save(frames_dir / f"{idx:06d}.png", optimize=True)
    return raw_path


def colorize_mask(mask: torch.Tensor, *, out_size: tuple[int, int] = (180, 240)) -> Image.Image:
    rgb = MASK_PREVIEW_COLORS[mask.long()].numpy()
    img = Image.fromarray(rgb)
    return img.resize((out_size[1], out_size[0]), Image.Resampling.NEAREST)


def resize_img(arr: np.ndarray, *, out_size: tuple[int, int] = (180, 240)) -> Image.Image:
    return Image.fromarray(arr).resize((out_size[1], out_size[0]), Image.Resampling.BILINEAR)


@torch.inference_mode()
def seg_predictions(pairs: torch.Tensor, segnet, device: torch.device, batch_size: int) -> torch.Tensor:
    preds = []
    for start in range(0, pairs.shape[0], batch_size):
        x = einops.rearrange(pairs[start : start + batch_size].float(), "b t h w c -> b t c h w").to(device)
        pred = segnet(segnet.preprocess_input(x)).argmax(dim=1).cpu()
        preds.append(pred)
    return torch.cat(preds, dim=0)


def save_previews(cache: dict, pairs: torch.Tensor, seg_pred: torch.Tensor, out_dir: Path, *, rows: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = cache["gt_pairs"]
    target = cache["seg_targets"]
    n = min(rows, pairs.shape[0])
    thumb_h, thumb_w = 180, 240
    cols = 5
    sheet = Image.new("RGB", (cols * thumb_w, n * thumb_h), (20, 20, 20))
    labels = ["original f2", "target mask", "scv f2", "scv seg", "scv pair"]
    draw = ImageDraw.Draw(sheet)
    for i in range(n):
        imgs = [
            resize_img(gt[i, 1].numpy(), out_size=(thumb_h, thumb_w)),
            colorize_mask(target[i], out_size=(thumb_h, thumb_w)),
            resize_img(pairs[i, 1].numpy(), out_size=(thumb_h, thumb_w)),
            colorize_mask(seg_pred[i], out_size=(thumb_h, thumb_w)),
            resize_img(np.concatenate([pairs[i, 0].numpy(), pairs[i, 1].numpy()], axis=1), out_size=(thumb_h, thumb_w)),
        ]
        for j, img in enumerate(imgs):
            sheet.paste(img, (j * thumb_w, i * thumb_h))
            if i == 0:
                draw.text((j * thumb_w + 4, 4), labels[j], fill=(255, 255, 255))
    sheet.save(out_dir / "preview_grid.jpg", quality=92)
    sheet.save(out_dir / "preview_original_vs_scv.jpg", quality=92)

    mask_sheet = Image.new("RGB", (2 * thumb_w, n * thumb_h), (20, 20, 20))
    for i in range(n):
        mask_sheet.paste(colorize_mask(target[i], out_size=(thumb_h, thumb_w)), (0, i * thumb_h))
        mask_sheet.paste(colorize_mask(seg_pred[i], out_size=(thumb_h, thumb_w)), (thumb_w, i * thumb_h))
    mask_sheet.save(out_dir / "preview_masks.jpg", quality=92)
    mask_sheet.save(out_dir / "preview_segnet_outputs.jpg", quality=92)


def render_variant(args) -> dict:
    device = pick_device(args.device)
    segnet, posenet = load_evaluators(device)
    cache = load_or_build_cache(args, device, segnet, posenet)
    if args.variant == "scv0":
        pairs, render_info = render_scv0(cache)
    elif args.variant == "scv1":
        pairs, render_info = render_scv1(
            cache,
            texture_size=tuple(args.texture_size),
            texture_strength=args.texture_strength,
        )
    elif args.variant == "scvt":
        pairs, render_info = render_scvt(
            cache,
            blur_size=tuple(args.scvt_blur_size),
            color_step=args.scvt_color_step,
            mask_smooth=args.scvt_mask_smooth,
            edge_keep=args.scvt_edge_keep,
        )
    else:
        raise ValueError(f"unsupported render variant: {args.variant}")

    raw_path = save_frames(pairs, args.out / "frames", keep_raw=args.keep_raw)
    metrics = evaluate_pairs(
        pairs,
        cache["seg_targets"],
        cache["pose_targets"],
        segnet,
        posenet,
        device,
        batch_size=args.eval_batch_size,
        archive_bytes=args.archive_bytes_for_projection,
    )
    seg_pred = seg_predictions(pairs, segnet, device, args.eval_batch_size)
    save_previews(cache, pairs, seg_pred, args.out)
    record = {
        "variant": args.variant,
        "subset": args.subset,
        "offset": args.offset,
        "frames_dir": str(args.out / "frames"),
        "raw_frames": str(raw_path) if raw_path is not None else None,
        "render": render_info,
        "uncompressed_metrics": metrics,
        "score_table": metric_table(
            metrics["segnet_dist"],
            metrics["posenet_dist"],
            args.archive_bytes_for_projection,
        ),
        "gates": {
            "scv1_segnet_term_le_0.08": 100.0 * metrics["segnet_dist"] <= 0.08,
            "scv1_segnet_term_le_0.10": 100.0 * metrics["segnet_dist"] <= 0.10,
        },
    }
    write_json(args.out / "metrics.json", record)
    return record
