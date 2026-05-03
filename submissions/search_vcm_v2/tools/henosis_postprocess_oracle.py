#!/usr/bin/env python
"""Cheap postprocess-control oracle for PR #65 henosis/qpose variant.

This starts from the public PR #65 decoded frames, then tests one additional
hardcoded per-sample postprocess stage. The payload would be a tiny choice
stream if a family wins, so this is the remaining plausible fixed-byte lever.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import brotli
import einops
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import DistortionNet, posenet_sd_path, segnet_sd_path
from submissions.henosis_qz_n3z_r25_clean import inflate as henosis
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json
from submissions.search_vcm_v2.families.qpose14_data import load_original_subset, select_torch_device
from submissions.search_vcm_v2.subsets import get_subset


HENOSIS_ARCHIVE = REPO_ROOT / "submissions/henosis_qz_n3z_r25_clean/archive.zip"
OUT_DIR = EXPERIMENTS_DIR / "henosis_postprocess_oracle"
HENOSIS_ARCHIVE_BYTES = 284_425


@dataclass(frozen=True)
class ExtraControls:
    shift: torch.Tensor | None = None
    frac: torch.Tensor | None = None
    frac2: torch.Tensor | None = None
    frac3: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    region: torch.Tensor | None = None
    randmulti: list[tuple[torch.Tensor, int, int, int]] | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    apply: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class EvalContext:
    distortion: DistortionNet
    original: torch.Tensor
    sample_ids: list[int]
    subset_name: str
    device: torch.device


def _load_bundle(archive: Path) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(td_path)
        return henosis.load_compact_archive_bundle(td_path)


def _decode_masks(bundle: dict[str, bytes]) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(bundle["mask"]))
        tmp_path = Path(tmp.name)
    try:
        return henosis.load_encoded_mask_video(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_generator(bundle: dict[str, bytes], device: torch.device) -> torch.nn.Module:
    generator = henosis.JointFrameGenerator().to(device)
    weights = brotli.decompress(bundle["model"])
    generator.load_state_dict(henosis.get_decoded_state_dict(weights, device), strict=True)
    generator.eval()
    return generator


def _sparse_default_delta(raw: bytes, *, magic_full: bytes, magic_delta: bytes, center: int) -> np.ndarray:
    magic = raw[:3]
    if magic == magic_full:
        return np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
    if magic == magic_delta:
        d = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
        return np.where(d == 0, center, d - 1).astype(np.int64)
    raise ValueError(f"bad sparse/default payload magic {magic!r}")


def _load_extra_controls(bundle: dict[str, bytes], device: torch.device) -> ExtraControls:
    shift = frac = frac2 = frac3 = bias = region = None
    randmulti = None

    if bundle.get("shift"):
        raw = brotli.decompress(bundle["shift"])
        shift = torch.from_numpy(_sparse_default_delta(raw, magic_full=b"SH4", magic_delta=b"SD4", center=40)).to(device)
    if bundle.get("frac"):
        raw = brotli.decompress(bundle["frac"])
        if raw[:3] == b"FH1":
            arr = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
        elif raw[:3] == b"FV1":
            cnt = int.from_bytes(raw[3:5], "little")
            pos = 5
            arr = np.full(600, 4, dtype=np.int64)
            idx = -1
            inds = []
            for _ in range(cnt):
                acc = 0
                sh = 0
                while True:
                    by = raw[pos]
                    pos += 1
                    acc |= (by & 127) << sh
                    if by & 128:
                        sh += 7
                    else:
                        break
                idx += acc + 1
                inds.append(idx)
            vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos).astype(np.int64)
            for ii, vv in zip(inds, vals, strict=True):
                arr[ii] = vv - 1
        else:
            raise ValueError("bad frac payload")
        frac = torch.from_numpy(arr).to(device)
    if bundle.get("frac2"):
        raw = brotli.decompress(bundle["frac2"])
        if raw[:3] != b"FH2":
            raise ValueError("bad frac2 payload")
        frac2 = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)).to(device)
    if bundle.get("frac3"):
        raw = brotli.decompress(bundle["frac3"])
        frac3 = torch.from_numpy(_sparse_default_delta(raw, magic_full=b"FH3", magic_delta=b"FD3", center=4)).to(device)
    if bundle.get("bias"):
        raw = brotli.decompress(bundle["bias"])
        magic = raw[:3]
        center = 13
        if magic in (b"BH1", b"BD1"):
            bias = torch.from_numpy(_sparse_default_delta(raw, magic_full=b"BH1", magic_delta=b"BD1", center=center)).to(device)
        elif magic == b"BV1":
            cnt = int.from_bytes(raw[3:5], "little")
            pos = 5
            arr = np.full(600, center, dtype=np.int64)
            idx = -1
            inds = []
            for _ in range(cnt):
                acc = 0
                sh = 0
                while True:
                    by = raw[pos]
                    pos += 1
                    acc |= (by & 127) << sh
                    if by & 128:
                        sh += 7
                    else:
                        break
                idx += acc + 1
                inds.append(idx)
            vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos).astype(np.int64)
            for ii, vv in zip(inds, vals, strict=True):
                arr[ii] = vv - 1
            bias = torch.from_numpy(arr).to(device)
        else:
            raise ValueError("bad bias payload")
    if bundle.get("region"):
        raw = brotli.decompress(bundle["region"])
        magic = raw[:3]
        if magic in (b"RH1", b"RD1"):
            region = torch.from_numpy(_sparse_default_delta(raw, magic_full=b"RH1", magic_delta=b"RD1", center=0)).to(device)
        elif magic == b"RV1":
            cnt = int.from_bytes(raw[3:5], "little")
            pos = 5
            arr = np.zeros(600, dtype=np.int64)
            idx = -1
            inds = []
            for _ in range(cnt):
                acc = 0
                sh = 0
                while True:
                    by = raw[pos]
                    pos += 1
                    acc |= (by & 127) << sh
                    if by & 128:
                        sh += 7
                    else:
                        break
                idx += acc + 1
                inds.append(idx)
            vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos).astype(np.int64)
            for ii, vv in zip(inds, vals, strict=True):
                arr[ii] = vv - 1
            region = torch.from_numpy(arr).to(device)
        else:
            raise ValueError("bad region payload")
    if bundle.get("randmulti"):
        raw = brotli.decompress(bundle["randmulti"])
        randmulti = []
        if raw[:3] == b"NM2":
            pos = 4
            group_count = int(raw[3])
            for _ in range(group_count):
                lh = int(raw[pos])
                lw = int(raw[pos + 1])
                amp = int(raw[pos + 2])
                scount = int(raw[pos + 3])
                pos += 4
                arr = np.frombuffer(raw, dtype=np.uint8, count=scount * 600, offset=pos).reshape(scount, 600).astype(np.int64)
                pos += scount * 600
                randmulti.append((torch.from_numpy(arr).to(device), lh, lw, amp))
        else:
            specs = [
                (24, 32, 1, 12),
                (12, 16, 1, 1),
                (6, 8, 1, 1),
                (3, 4, 1, 1),
                (2, 2, 1, 1),
                (8, 8, 1, 1),
                (4, 4, 1, 1),
                (4, 8, 1, 1),
                (2, 4, 1, 1),
                (2, 8, 1, 1),
                (1, 2, 1, 1),
                (1, 4, 1, 1),
                (2, 1, 1, 1),
                (4, 1, 1, 1),
                (8, 1, 1, 1),
                (1, 8, 1, 1),
            ]
            pos = 0
            for lh, lw, amp, scount in specs:
                rows = np.zeros((scount, 600), dtype=np.uint8)
                for si in range(scount):
                    cnt = int(raw[pos])
                    pos += 1
                    if cnt == 255:
                        cnt = int.from_bytes(raw[pos : pos + 2], "little")
                        pos += 2
                    idx = -1
                    inds = []
                    for _ in range(cnt):
                        acc = 0
                        sh = 0
                        while True:
                            by = raw[pos]
                            pos += 1
                            acc |= (by & 127) << sh
                            if by & 128:
                                sh += 7
                            else:
                                break
                        idx += acc + 1
                        inds.append(idx)
                    vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos)
                    pos += cnt
                    if cnt:
                        rows[si, np.array(inds, dtype=np.int64)] = vals
                randmulti.append((torch.from_numpy(rows.astype(np.int64)).to(device), lh, lw, amp))
            if pos != len(raw):
                raise ValueError("bad headerless randmulti payload")
    return ExtraControls(shift=shift, frac=frac, frac2=frac2, frac3=frac3, bias=bias, region=region, randmulti=randmulti)


def _apply_int_shift(img_hwc: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    img = img_hwc.permute(2, 0, 1).unsqueeze(0)
    left, right = max(dx, 0), max(-dx, 0)
    top, bottom = max(dy, 0), max(-dy, 0)
    imgp = F.pad(img, (left, right, top, bottom), mode="replicate")
    y0, x0 = bottom, right
    return imgp[0, :, y0 : y0 + img_hwc.shape[0], x0 : x0 + img_hwc.shape[1]].permute(1, 2, 0)


def _frac_shift(img_hwc: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    height, width = img_hwc.shape[:2]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=img_hwc.device, dtype=torch.float32),
        torch.arange(width, device=img_hwc.device, dtype=torch.float32),
        indexing="ij",
    )
    gx = ((xx - dx) + 0.5) * 2.0 / width - 1.0
    gy = ((yy - dy) + 0.5) * 2.0 / height - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    img = img_hwc.permute(2, 0, 1).unsqueeze(0).float()
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].permute(1, 2, 0)


def _apply_existing_post(batch_hwc: torch.Tensor, sample_ids: list[int], postprocess, extras: ExtraControls) -> torch.Tensor:
    batch_hwc = batch_hwc.clone()
    device = batch_hwc.device
    if postprocess is not None:
        idx_tensor = torch.tensor(sample_ids, dtype=torch.long, device=device)
        for gains, biases, choices in postprocess:
            idx = choices[idx_tensor].clamp(0, gains.shape[0] - 1)
            batch_hwc = (batch_hwc * gains[idx] + biases[idx]).clamp(0, 255).round()
    for bi, sample_id in enumerate(sample_ids):
        if extras.shift is not None:
            ch = int(extras.shift[sample_id].item())
            if ch != 40:
                batch_hwc[bi, 0] = _apply_int_shift(batch_hwc[bi, 0], ch // 9 - 4, ch % 9 - 4)
        for arr, scale in ((extras.frac, 0.5), (extras.frac2, 0.25), (extras.frac3, 0.125)):
            if arr is not None:
                ch = int(arr[sample_id].item())
                if ch != 4:
                    batch_hwc[bi, 0] = _frac_shift(batch_hwc[bi, 0], (ch // 3 - 1) * scale, (ch % 3 - 1) * scale).clamp(0, 255).round()
        if extras.bias is not None:
            ch = int(extras.bias[sample_id].item())
            if ch != 13:
                br = ch // 9 - 1
                bg = (ch // 3) % 3 - 1
                bb = ch % 3 - 1
                batch_hwc[bi, 0] = (batch_hwc[bi, 0] + torch.tensor([br, bg, bb], device=device)).clamp(0, 255).round()
        if extras.region is not None:
            ch = int(extras.region[sample_id].item())
            if ch != 0:
                height, width = batch_hwc.shape[2:4]
                yy = torch.arange(height, device=device).view(height, 1).expand(height, width)
                xx = torch.arange(width, device=device).view(1, width).expand(height, width)
                j = ch - 1
                val = float([-2, -1, 1, 2][j % 4])
                j //= 4
                ci = j % 4
                j //= 4
                if j == 0:
                    mask = yy < height // 2
                elif j == 1:
                    mask = yy >= height // 2
                elif j == 2:
                    mask = xx < width // 2
                elif j == 3:
                    mask = xx >= width // 2
                elif j == 4:
                    mask = (yy >= height // 3) & (yy < 2 * height // 3)
                else:
                    mask = (xx >= width // 3) & (xx < 2 * width // 3)
                if ci == 0:
                    batch_hwc[bi, 0][mask, :] = (batch_hwc[bi, 0][mask, :] + val).clamp(0, 255).round()
                else:
                    batch_hwc[bi, 0][mask, ci - 1] = (batch_hwc[bi, 0][mask, ci - 1] + val).clamp(0, 255).round()
    if extras.randmulti is not None:
        height, width = batch_hwc.shape[2:4]
        cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
        for bi, sample_id in enumerate(sample_ids):
            for arr_choices, lh, lw, amp in extras.randmulti:
                for st in range(arr_choices.shape[0]):
                    ch = int(arr_choices[st, sample_id].item())
                    if ch == 0:
                        continue
                    key = (lh, lw, amp, ch)
                    if key not in cache:
                        rng = np.random.default_rng(1000 + ch)
                        arr = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(3, lh, lw)).astype(np.float32) * float(amp)
                        cache[key] = torch.from_numpy(arr).to(device)
                    pat = F.interpolate(cache[key].unsqueeze(0), size=(height, width), mode="nearest")[0].permute(1, 2, 0)
                    batch_hwc[bi, 0] = (batch_hwc[bi, 0] + pat).clamp(0, 255).round()
    return batch_hwc


def render_henosis_subset(sample_ids: list[int], *, device: torch.device, archive: Path) -> torch.Tensor:
    bundle = _load_bundle(archive)
    generator = _load_generator(bundle, device)
    masks = _decode_masks(bundle)
    poses = henosis.load_pose_frames_from_payload(bundle["pose"])
    post = henosis.load_postprocess(Path("."), device, bundle["post"])
    extras = _load_extra_controls(bundle, device)
    out = []
    with torch.inference_mode():
        for start in range(0, len(sample_ids), 2):
            ids = sample_ids[start : start + 2]
            mask = masks[ids].to(device).long()
            pose = poses[ids].to(device).float()
            f1, f2 = generator(mask, pose)
            f1 = F.interpolate(f1, size=(874, 1164), mode="bilinear", align_corners=False)
            f2 = F.interpolate(f2, size=(874, 1164), mode="bilinear", align_corners=False)
            batch = torch.stack([f1, f2], dim=1).clamp(0, 255).round()
            batch_hwc = einops.rearrange(batch, "b t c h w -> b t h w c")
            batch_hwc = _apply_existing_post(batch_hwc, ids, post, extras)
            out.append(batch_hwc.detach().cpu())
    return torch.cat(out, dim=0).contiguous()


def make_eval_context(sample_ids: list[int], *, subset_name: str, device: torch.device) -> EvalContext:
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    original = load_original_subset(subset_name, sample_ids, device="cpu").float().to(device)
    return EvalContext(distortion=distortion, original=original, sample_ids=sample_ids, subset_name=subset_name, device=device)


def evaluate_frames(frames: torch.Tensor, ctx: EvalContext) -> dict:
    sample_ids = ctx.sample_ids
    distortion = ctx.distortion
    original = ctx.original
    device = ctx.device
    pred = frames.to(device).float()
    per_sample = []
    with torch.inference_mode():
        for i in range(0, len(sample_ids), 2):
            pose_dist, seg_dist = distortion.compute_distortion(original[i : i + 2], pred[i : i + 2])
            for sid, pose_v, seg_v in zip(sample_ids[i : i + 2], pose_dist.cpu().tolist(), seg_dist.cpu().tolist(), strict=True):
                seg = float(seg_v)
                pose = float(pose_v)
                per_sample.append(
                    {
                        "sample_id": int(sid),
                        "segnet_dist": seg,
                        "posenet_dist": pose,
                        "seg_term": 100.0 * seg,
                        "pose_term": float(torch.sqrt(torch.tensor(10.0 * pose)).item()),
                        "quality": quality(seg, pose),
                    }
                )
    seg_mean = sum(row["segnet_dist"] for row in per_sample) / len(per_sample)
    pose_mean = sum(row["posenet_dist"] for row in per_sample) / len(per_sample)
    return {
        "segnet_dist": seg_mean,
        "posenet_dist": pose_mean,
        "seg_term": 100.0 * seg_mean,
        "pose_term": float(torch.sqrt(torch.tensor(10.0 * pose_mean)).item()),
        "quality": quality(seg_mean, pose_mean),
        "score": quality(seg_mean, pose_mean) + rate_term(HENOSIS_ARCHIVE_BYTES),
        "max_sample_quality": max(row["quality"] for row in per_sample),
        "per_sample": per_sample,
    }


def make_variants(device: torch.device, families: list[str], mask_up: torch.Tensor | None = None) -> list[Variant]:
    variants = [Variant("none", lambda x: x.clone())]
    if "f0_int_shift" in families:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                variants.append(Variant(f"f0_int_shift_{dy}_{dx}", lambda x, dy=dy, dx=dx: _apply_one_frame0_shift(x, dy, dx)))
    if "f0_frac16" in families:
        for dy in [-0.0625, 0.0, 0.0625]:
            for dx in [-0.0625, 0.0, 0.0625]:
                if dy == 0.0 and dx == 0.0:
                    continue
                variants.append(Variant(f"f0_frac16_{dy}_{dx}", lambda x, dy=dy, dx=dx: _apply_one_frame0_frac(x, dy, dx)))
    if "f0_bias" in families:
        for r in [-2, -1, 0, 1, 2]:
            for g in [-2, -1, 0, 1, 2]:
                for b in [-2, -1, 0, 1, 2]:
                    if (r, g, b) != (0, 0, 0):
                        bias = torch.tensor([r, g, b], dtype=torch.float32, device=device)
                        variants.append(Variant(f"f0_bias_{r}_{g}_{b}", lambda x, bias=bias: _apply_frame_bias(x, 0, bias)))
    if "f1_bias" in families:
        for r in [-2, -1, 0, 1, 2]:
            for g in [-2, -1, 0, 1, 2]:
                for b in [-2, -1, 0, 1, 2]:
                    if (r, g, b) != (0, 0, 0):
                        bias = torch.tensor([r, g, b], dtype=torch.float32, device=device)
                        variants.append(Variant(f"f1_bias_{r}_{g}_{b}", lambda x, bias=bias: _apply_frame_bias(x, 1, bias)))
    if "f1_class_bias" in families:
        if mask_up is None:
            raise ValueError("f1_class_bias requires exact masks")
        for cls in range(5):
            for val in [-3, -2, -1, 1, 2, 3]:
                for channel in ["all", "r", "g", "b"]:
                    variants.append(
                        Variant(
                            f"f1_class_bias_c{cls}_{channel}_{val}",
                            lambda x, cls=cls, channel=channel, val=val: _apply_frame2_class_bias(x, mask_up, cls, channel, val),
                        )
                    )
    return variants


def _apply_one_frame0_shift(frames: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = frames.clone()
    for i in range(out.shape[0]):
        out[i, 0] = _apply_int_shift(out[i, 0], dy, dx)
    return out


def _apply_one_frame0_frac(frames: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    out = frames.clone()
    for i in range(out.shape[0]):
        out[i, 0] = _frac_shift(out[i, 0], dy, dx).clamp(0, 255).round()
    return out


def _apply_frame_bias(frames: torch.Tensor, frame_idx: int, bias: torch.Tensor) -> torch.Tensor:
    out = frames.clone()
    out[:, frame_idx] = (out[:, frame_idx] + bias.view(1, 1, 1, 3)).clamp(0, 255).round()
    return out


def _apply_frame2_class_bias(frames: torch.Tensor, mask_up: torch.Tensor, cls: int, channel: str, val: int) -> torch.Tensor:
    out = frames.clone()
    masks = mask_up.to(out.device) == int(cls)
    delta = torch.zeros(3, dtype=out.dtype, device=out.device)
    if channel == "all":
        delta[:] = float(val)
    else:
        delta[{"r": 0, "g": 1, "b": 2}[channel]] = float(val)
    changed = (out[:, 1] + delta.view(1, 1, 1, 3)).clamp(0, 255)
    out[:, 1] = torch.where(masks.unsqueeze(-1), changed, out[:, 1]).round()
    return out


def _load_upsampled_masks(sample_ids: list[int], *, archive: Path, device: torch.device) -> torch.Tensor:
    bundle = _load_bundle(archive)
    masks = _decode_masks(bundle)[sample_ids].long().to(device)
    up = F.interpolate(masks[:, None].float(), size=(874, 1164), mode="nearest")[:, 0].to(torch.uint8)
    return up


def run_oracle(*, subset_name: str, device_name: str, families: list[str], archive: Path, out: Path, only_variants: set[str] | None = None) -> dict:
    sample_ids = get_subset(subset_name)
    device = select_torch_device(device_name)
    base_frames = render_henosis_subset(sample_ids, device=device, archive=archive)
    eval_ctx = make_eval_context(sample_ids, subset_name=subset_name, device=device)
    base = evaluate_frames(base_frames, eval_ctx)
    best_frames = base_frames.clone()
    best_quality_by_sample = {row["sample_id"]: row["quality"] for row in base["per_sample"]}
    choices = {row["sample_id"]: "none" for row in base["per_sample"]}
    mask_up = _load_upsampled_masks(sample_ids, archive=archive, device=device) if "f1_class_bias" in families else None
    variants = make_variants(device, families, mask_up=mask_up)
    if only_variants:
        variants = [variant for variant in variants if variant.name in only_variants or variant.name == "none"]
        missing = sorted(only_variants - {variant.name for variant in variants})
        if missing:
            raise ValueError(f"unknown --only-variants entries: {missing}")
    rows = []
    for variant in variants:
        cand_frames = variant.apply(base_frames.to(device)).detach().cpu()
        metrics = evaluate_frames(cand_frames, eval_ctx)
        rows.append({"variant": variant.name, **{k: v for k, v in metrics.items() if k != "per_sample"}})
        for i, sample_row in enumerate(metrics["per_sample"]):
            sid = sample_row["sample_id"]
            if sample_row["quality"] < best_quality_by_sample[sid]:
                best_quality_by_sample[sid] = sample_row["quality"]
                choices[sid] = variant.name
                best_frames[i] = cand_frames[i]
    chosen = evaluate_frames(best_frames, eval_ctx)
    rows.sort(key=lambda row: row["quality"])
    summary = {
        "archive": str(archive),
        "archive_bytes": HENOSIS_ARCHIVE_BYTES,
        "subset": subset_name,
        "device": str(device),
        "families": families,
        "base": base,
        "best_single_variants": rows[:20],
        "oracle_chosen": chosen,
        "choices": choices,
        "quality_delta_oracle_vs_base": chosen["quality"] - base["quality"],
        "score_delta_oracle_vs_base": chosen["score"] - base["score"],
        "decision": "continue" if chosen["quality"] <= base["quality"] - 0.01 else "close_postprocess_increment",
    }
    write_json(out / f"{subset_name}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="hard8", choices=["hard8", "strat64"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--families", default="f0_int_shift,f0_frac16,f0_bias,f1_bias")
    parser.add_argument("--only-variants", default="")
    parser.add_argument("--archive", type=Path, default=HENOSIS_ARCHIVE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    only = {item.strip() for item in args.only_variants.split(",") if item.strip()} or None
    run_oracle(subset_name=args.subset, device_name=args.device, families=families, archive=args.archive, out=args.out, only_variants=only)


if __name__ == "__main__":
    main()
