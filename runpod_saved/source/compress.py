#!/usr/bin/env python
import os
import sys
import math
import mmap
import json
import argparse
import av
import subprocess
import shutil
import zipfile
import numpy as np
import logging
import warnings
import brotli
import io
import tempfile
import gc
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from tqdm import tqdm
from safetensors.torch import load_file

# -----------------------------
# Path Resolution & Imports
# -----------------------------
# compress.py is in ./submissions/quantizr/, we need to access ./modules.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frame_utils import AVVideoDataset, DaliVideoDataset, segnet_model_input_size
from modules import SegNet, PoseNet, DistortionNet, segnet_sd_path, posenet_sd_path

# -----------------------------
# Constants & Config
# -----------------------------
SEQ_LEN = 2
SEGNET_MODEL_INPUT_SIZE = (512, 384)
MODEL_PAYLOAD_NAME = "model.pt.br"
MODEL_QPACK_PAYLOAD_NAME = "model.qpack.br"
MASK_PAYLOAD_NAME = "mask.obu.br"
MASK_TREE_META_PAYLOAD_NAME = "mask_tree_meta.json.br"
MASK_TREE_TOKENS_PAYLOAD_NAME = "mask_tree_tokens.bin.br"
MASK_TREE_CODEBOOK_PAYLOAD_NAME = "mask_tree_codebook.bin.br"
MASK_TREE_PAYLOAD_NAMES = (
    MASK_TREE_META_PAYLOAD_NAME,
    MASK_TREE_TOKENS_PAYLOAD_NAME,
    MASK_TREE_CODEBOOK_PAYLOAD_NAME,
)
POSE_PAYLOAD_NAME = "pose.npy.br"
LATENT_PAYLOAD_NAME = "z.npz.br"
RGB_CACHE_NAME = "rgb_pairs.pt"
POSE_CACHE_NAME = "pose6.pt"


def parse_size_arg(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if "x" not in normalized:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, for example 128x96")
    w_raw, h_raw = normalized.split("x", 1)
    try:
        width = int(w_raw)
        height = int(h_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected integer WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Mask encode dimensions must be positive")
    return width, height


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 5
    pose_dim: int = 6
    z_dim: int = 8
    z_seg_channels: int = 0
    z_seg_h: int = 0
    z_seg_w: int = 0
    latent_quant_bits: int = 8
    cond_dim: int = 48
    c1: int = 56
    c2: int = 64
    hidden: int = 52
    depth_mult: int = 1
    frame2_uses_latent: bool = True
    mask_from_latent: bool = False

    @property
    def z_seg_dim(self) -> int:
        return self.z_seg_channels * self.z_seg_h * self.z_seg_w

    @property
    def total_latent_dim(self) -> int:
        return self.z_dim + self.z_seg_dim

    def to_meta(self) -> dict[str, int]:
        return {
            "num_classes": self.num_classes,
            "pose_dim": self.pose_dim,
            "z_dim": self.z_dim,
            "z_seg_channels": self.z_seg_channels,
            "z_seg_h": self.z_seg_h,
            "z_seg_w": self.z_seg_w,
            "latent_quant_bits": self.latent_quant_bits,
            "cond_dim": self.cond_dim,
            "c1": self.c1,
            "c2": self.c2,
            "hidden": self.hidden,
            "depth_mult": self.depth_mult,
            "frame2_uses_latent": int(self.frame2_uses_latent),
            "mask_from_latent": int(self.mask_from_latent),
        }


@dataclass
class PipelineState:
    model_state: dict[str, torch.Tensor] | None = None
    latent_state: dict[str, torch.Tensor] | None = None

class Stage(Enum):
    ANCHOR = "anchor"     
    FINETUNE = "finetune" 
    JOINT = "joint"
    POSE_RESCUE = "pose_rescue"
    SEG_RESCUE = "seg_rescue"
    SPLIT_LATENT = "split_latent"

@dataclass
class PipelineRun:
    name: str
    stage: Stage
    epochs: int
    lr: float
    qat_start_epoch: int
    frame1_fade_epochs: int = 0
    error_boost: float = 4.0
    ce_weight: float = 1.0
    pose_weight: float = 1.0
    warmup_epochs: int = 2
    ema_decay: float = 0.99
    grad_clip: float = 1.0
    pose_rescue_pose_scale: float = 50.0
    pose_rescue_seg2_scale: float = 1.0
    tv_weight: float = 0.0

# -----------------------------
# System Helpers
# -----------------------------
def get_ffmpeg_path():
    """Find local ffmpeg binary first, fallback to system ffmpeg."""
    local_ffmpeg = ROOT_DIR / "ffmpeg"
    if local_ffmpeg.is_file() and os.access(local_ffmpeg, os.X_OK):
        return str(local_ffmpeg.resolve())
    
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
        
    raise FileNotFoundError("FFmpeg binary not found locally or in system PATH.")

def diff_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def boundary_weight_from_mask(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    boundary = torch.zeros_like(mask, dtype=torch.float32)
    boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], (mask[:, 1:, :] != mask[:, :-1, :]).float())
    boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], (mask[:, 1:, :] != mask[:, :-1, :]).float())
    boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], (mask[:, :, 1:] != mask[:, :, :-1]).float())
    boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], (mask[:, :, 1:] != mask[:, :, :-1]).float())
    if radius > 0:
        k = (radius * 2) + 1
        boundary = F.max_pool2d(boundary.unsqueeze(1), kernel_size=k, stride=1, padding=radius).squeeze(1)
    return 1.0 + (2.0 * boundary)


def seg_margin_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, margin: float = 0.75) -> torch.Tensor:
    target_logits = logits.gather(1, target[:, None]).squeeze(1)
    target_mask = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool()
    other_logits = logits.masked_fill(target_mask, -1e9).amax(dim=1)
    return (F.relu(margin - (target_logits - other_logits)) * weight).mean()

# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# -----------------------------
# Evaluator Helpers
# -----------------------------
def diff_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., : 2 * h2, : 2 * w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    y00, y10, y01, y11 = y[:, 0::2, 0::2], y[:, 1::2, 0::2], y[:, 0::2, 1::2], y[:, 1::2, 1::2]
    u_sub = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
    v_sub = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=1)

def pack_pair_yuv6(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    return torch.cat([diff_rgb_to_yuv6(frame1), diff_rgb_to_yuv6(frame2)], dim=1)

def get_pose_tensor(posenet_out):
    if isinstance(posenet_out, dict): return posenet_out["pose"]
    if hasattr(posenet_out, "pose"): return posenet_out.pose
    return posenet_out["pose"]

def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)

def kl_on_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)

def assert_finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"non-finite tensor in {name}: shape={tuple(x.shape)}")


def quantize_latents(latents: torch.Tensor, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    lat_np = latents.detach().cpu().float().numpy()
    scales = np.max(np.abs(lat_np), axis=0, keepdims=False).astype(np.float32)
    max_q = 127 if bits == 8 else 7
    scales = np.where(scales > 1e-8, scales / float(max_q), np.ones_like(scales, dtype=np.float32))
    quantized = np.clip(np.round(lat_np / scales), -max_q, max_q).astype(np.int8)
    return quantized, scales.astype(np.float16)


def pack_int4_np(values: np.ndarray) -> np.ndarray:
    flat = (values.astype(np.int16).reshape(-1) + 8).astype(np.uint8)
    if flat.size % 2 == 1:
        flat = np.pad(flat, (0, 1), constant_values=8)
    return ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)


def unpack_int4_np(packed: np.ndarray, count: int) -> np.ndarray:
    flat = packed.reshape(-1).astype(np.uint8)
    out = np.empty(flat.size * 2, dtype=np.uint8)
    out[0::2] = (flat >> 4) & 0x0F
    out[1::2] = flat & 0x0F
    return out[:count].astype(np.int16) - 8


def save_quantized_latents(latents: torch.Tensor, out_path: Path, bits: int = 8):
    if bits not in (4, 8):
        raise ValueError(f"Unsupported latent quantization bits: {bits}")
    quantized, scales = quantize_latents(latents, bits=bits)
    buffer = io.BytesIO()
    if bits == 4:
        np.savez(
            buffer,
            bits=np.array(bits, dtype=np.int16),
            shape=np.array(quantized.shape, dtype=np.int32),
            values_packed=pack_int4_np(quantized),
            scales=scales,
        )
    else:
        np.savez(buffer, bits=np.array(bits, dtype=np.int16), values=quantized, scales=scales)
    buffer.seek(0)
    with open(out_path, "wb") as f_out:
        f_out.write(brotli.compress(buffer.read(), quality=11, lgwin=24))


def load_quantized_latents(path: Path) -> torch.Tensor:
    with open(path, "rb") as f_in:
        payload = brotli.decompress(f_in.read())
    with np.load(io.BytesIO(payload)) as data:
        bits = int(data["bits"]) if "bits" in data else 8
        if bits == 4:
            shape = tuple(int(x) for x in data["shape"])
            values = unpack_int4_np(data["values_packed"], int(np.prod(shape))).reshape(shape).astype(np.float32)
        else:
            values = data["values"].astype(np.float32)
        scales = data["scales"].astype(np.float32)
    return torch.from_numpy(values * scales[None, :]).float()


def package_submission_archive(
    archive_dir: Path,
    archive_zip_path: Path,
    include_latents: bool,
    include_mask: bool = True,
    mask_payload_kind: str | None = None,
) -> int:
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    model_payload_name = MODEL_QPACK_PAYLOAD_NAME if (archive_dir / MODEL_QPACK_PAYLOAD_NAME).exists() else MODEL_PAYLOAD_NAME
    artifact_names = [model_payload_name]

    if mask_payload_kind is None:
        mask_payload_kind = "av1" if include_mask else "none"
    if mask_payload_kind == "auto":
        if all((archive_dir / name).exists() for name in MASK_TREE_PAYLOAD_NAMES):
            mask_payload_kind = "masktree"
        elif include_mask and (archive_dir / MASK_PAYLOAD_NAME).exists():
            mask_payload_kind = "av1"
        else:
            mask_payload_kind = "none"

    if mask_payload_kind == "masktree":
        artifact_names.extend(MASK_TREE_PAYLOAD_NAMES)
    elif mask_payload_kind == "av1":
        if include_mask:
            artifact_names.append(MASK_PAYLOAD_NAME)
    elif mask_payload_kind == "none":
        pass
    else:
        raise ValueError(f"Unknown mask payload kind: {mask_payload_kind}")

    artifact_names.append(POSE_PAYLOAD_NAME)
    if include_latents and (archive_dir / LATENT_PAYLOAD_NAME).exists():
        artifact_names.append(LATENT_PAYLOAD_NAME)

    if archive_zip_path.exists():
        archive_zip_path.unlink()

    with zipfile.ZipFile(archive_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for artifact_name in artifact_names:
            artifact_path = archive_dir / artifact_name
            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing artifact for archive packaging: {artifact_path}")
            zf.write(artifact_path, arcname=artifact_name)

    return archive_zip_path.stat().st_size


def parse_official_report(report_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    line_map = {
        "Average PoseNet Distortion": "posenet_dist",
        "Average SegNet Distortion": "segnet_dist",
        "Compression Rate": "rate",
        "Submission file size": "archive_bytes",
    }
    for line in report_path.read_text().splitlines():
        for prefix, key in line_map.items():
            if prefix in line:
                value = line.split(":", 1)[1].strip().replace(",", "")
                metrics[key] = float(value.split()[0])
    if {"segnet_dist", "posenet_dist", "rate"} - metrics.keys():
        raise RuntimeError(f"Failed to parse official report: {report_path}")
    metrics["score"] = 100.0 * metrics["segnet_dist"] + math.sqrt(10.0 * metrics["posenet_dist"]) + 25.0 * metrics["rate"]
    return metrics


def load_cached_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def copy_if_different(src: Path, dst: Path):
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)


def run_official_evaluation(submission_dir: Path, archive_zip_path: Path, video_names_file: Path, eval_device: str | None = None) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="quantizr_eval_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        shutil.copy2(archive_zip_path, tmp_dir / "archive.zip")
        shutil.copy2(submission_dir / "inflate.py", tmp_dir / "inflate.py")
        shutil.copy2(submission_dir / "inflate.sh", tmp_dir / "inflate.sh")
        python_bin = ROOT_DIR / ".venv" / "bin" / "python"

        cmd = [
            "bash",
            str(ROOT_DIR / "evaluate.sh"),
            "--submission-dir",
            str(tmp_dir),
            "--video-names-file",
            str(video_names_file),
        ]
        if eval_device:
            cmd.extend(["--device", eval_device])
        env = os.environ.copy()
        if python_bin.exists():
            env["PYTHON_BIN"] = str(python_bin)
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError:
            if eval_device == "cuda":
                logging.warning("Official eval on CUDA failed; retrying on CPU.")
                retry_cmd = [
                    "bash",
                    str(ROOT_DIR / "evaluate.sh"),
                    "--submission-dir",
                    str(tmp_dir),
                    "--video-names-file",
                    str(video_names_file),
                    "--device",
                    "cpu",
                ]
                subprocess.run(retry_cmd, check=True, env=env)
            else:
                raise
        return parse_official_report(tmp_dir / "report.txt")

# -----------------------------
# Data Extractors & Preloaders
# -----------------------------
def hevc_frame_count(path: str) -> int:
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        frames, i = 0, 0
        find = b.find
        while True:
            j = find(b"\x00\x00\x01", i)
            if j < 0: return frames
            p = j + 3
            if ((b[p] >> 1) & 0x3F) <= 31: frames += 1
            i = p

def container_frame_count(path: str) -> int:
    container = av.open(path)
    stream = container.streams.video[0]
    n = stream.frames
    # If the container header lacks the frame count, demux and count manually
    if n == 0: 
        n = sum(1 for packet in container.demux(stream) if packet.size > 0)
    container.close()
    return n

def preload_video_pair_cache_dali(file_names, data_dir, batch_size, device, num_threads=4, prefetch_queue_depth=4):
    logging.info("Preloading raw video RGB pairs into memory via DALI...")
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    warnings.filterwarnings("ignore", category=Warning, module=r"nvidia\.dali\.plugin\.base_iterator")

    @pipeline_def
    def pipe():
        return fn.experimental.inputs.video(name="inbuf", sequence_length=SEQ_LEN, device="mixed", no_copy=True, blocking=False, last_sequence_policy="pad")

    all_batches = []
    for fnm in file_names:
        path = str(data_dir / fnm)
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        
        # Use the robust frame counter
        frames_count = hevc_frame_count(path) if path.endswith(".hevc") else container_frame_count(path)
        
        it_size = math.ceil((frames_count // SEQ_LEN) / batch_size)
        
        p = pipe(batch_size=batch_size, num_threads=num_threads, device_id=device.index or 0, prefetch_queue_depth=prefetch_queue_depth)
        p.build()
        p.feed_input("inbuf", [mv])
        it = DALIGenericIterator([p], output_map=["video"], auto_reset=False, last_batch_policy=LastBatchPolicy.PARTIAL)
        try:
            for _ in range(it_size): 
                all_batches.append(next(it)[0]["video"].cpu().contiguous())
        finally:
            torch.cuda.synchronize()
            it.reset(); del it, p; mv.release(); mm.close(); f.close()
            
    if not all_batches:
        raise RuntimeError("No video data was loaded. Please check if your video directory and file list are correct.")
        
    out = torch.cat(all_batches, dim=0).contiguous()
    del all_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def preload_video_pair_cache_av(file_names, data_dir, batch_size):
    logging.info("Preloading raw video RGB pairs into memory via PyAV on CPU...")
    dataset = AVVideoDataset(
        file_names=file_names,
        data_dir=data_dir,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    dataset.prepare_data()
    all_batches = []
    for _, _, vid in dataset:
        all_batches.append(vid.contiguous())
    if not all_batches:
        raise RuntimeError("No video data was loaded with AV preload.")
    return torch.cat(all_batches, dim=0).contiguous()


def get_rgb_pairs(file_names, data_dir, batch_size, device, cache_dir: Path | None, decode_backend: str):
    cache_path = cache_dir / RGB_CACHE_NAME if cache_dir else None
    if cache_path is not None and cache_path.exists():
        logging.info(f"Loading cached RGB pairs from {cache_path}...")
        return load_cached_tensor(cache_path).contiguous()

    if decode_backend == "av":
        rgb_pairs_all = preload_video_pair_cache_av(file_names, data_dir, batch_size)
    else:
        rgb_pairs_all = preload_video_pair_cache_dali(file_names, data_dir, batch_size, device)
    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(rgb_pairs_all.cpu(), cache_path)
    return rgb_pairs_all.contiguous()

def extract_and_compress_masks(
    rgb_pairs_all,
    segnet,
    device,
    crf,
    archive_dir,
    batch_size=8,
    cache_dir: Path | None = None,
    mask_encode_size: tuple[int, int] = SEGNET_MODEL_INPUT_SIZE,
):
    expected_frames = rgb_pairs_all.shape[0]
    cache_dir = cache_dir or archive_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_w, target_h = mask_encode_size
    full_w, full_h = SEGNET_MODEL_INPUT_SIZE
    cache_tag = f"crf{crf}" if (target_w, target_h) == (full_w, full_h) else f"crf{crf}_{target_w}x{target_h}"
    
    # Versioned filenames for caching
    raw_path = cache_dir / f"raw_masks_{cache_tag}.yuv"
    obu_path = cache_dir / f"mask_{cache_tag}.obu"
    obu_br_path = cache_dir / f"mask_{cache_tag}.obu.br"
    mask_tensor_path = cache_dir / f"mask_frames_{cache_tag}.pt"
    
    # Stable filename for inflate.py compatibility
    stable_obu_br_path = archive_dir / MASK_PAYLOAD_NAME

    # --- 1. Cache Check & Validation ---
    if obu_br_path.exists() and mask_tensor_path.exists():
        logging.info(f"Loading cached mask artifacts for CRF {crf} at {target_w}x{target_h} from {cache_dir}...")
        try:
            frames = load_cached_tensor(mask_tensor_path).contiguous()
            if frames.shape[0] == expected_frames and tuple(frames.shape[-2:]) == (target_h, target_w):
                copy_if_different(obu_br_path, stable_obu_br_path)
                return frames
            logging.warning(f"Cached mask tensor incomplete ({frames.shape[0]}/{expected_frames} frames). Regenerating...")
        except Exception as e:
            logging.warning(f"Failed to load cached mask ({e}). Regenerating...")

    # --- 2. Generation & Extraction ---
    logging.info(f"Generating odd-frame raw masks from cached RGB pairs at {target_w}x{target_h}...")
    with open(raw_path, "wb") as f_out:
        with torch.inference_mode():
            for start in tqdm(range(0, expected_frames, batch_size), desc="Extracting Masks"):
                batch = rgb_pairs_all[start:start+batch_size].to(device).float()
                batch = einops.rearrange(batch, 'b t h w c -> b t c h w')
                odd_frames = batch[:, 1] 
                
                resized = torch.nn.functional.interpolate(
                    odd_frames, 
                    size=(SEGNET_MODEL_INPUT_SIZE[1], SEGNET_MODEL_INPUT_SIZE[0]), 
                    mode='bilinear'
                )
                
                out = segnet(resized)
                mask = out.argmax(dim=1).to(torch.uint8)
                if (target_w, target_h) != (full_w, full_h):
                    mask = F.interpolate(
                        mask[:, None].float(),
                        size=(target_h, target_w),
                        mode="nearest",
                    ).squeeze(1).to(torch.uint8)
                mask_scaled = mask * 63 
                f_out.write(mask_scaled.contiguous().cpu().numpy().tobytes())

    # --- 3. Compression ---
    logging.info(f"Compressing {target_w}x{target_h} masks to OBU using FFmpeg (CRF {crf})...")
    ffmpeg_cmd = [
        get_ffmpeg_path(), "-y", "-hide_banner",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{target_w}x{target_h}", 
        "-r", "10", 
        "-i", str(raw_path),
        "-c:v", "libaom-av1",
        "-crf", str(crf),
        "-cpu-used", "0",
        "-row-mt", "1",
        "-g", "1200",
        "-keyint_min", "1200",
        "-lag-in-frames", "48",
        "-arnr-strength", "0",
        "-aq-mode", "0",
        "-aom-params", "enable-cdef=0:enable-intrabc=1:enable-obmc=0",
        "-f", "obu",
        str(obu_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    logging.info("Applying Brotli compression to OBU...")
    with open(obu_path, "rb") as f_in, open(obu_br_path, "wb") as f_out:
        f_out.write(brotli.compress(f_in.read(), quality=11, lgwin=24))

    # Provide the stable filename for inflate.py
    copy_if_different(obu_br_path, stable_obu_br_path)

    # --- 4. Validation & RAM Decoding ---
    logging.info("Decoding OBU artifacts to RAM for training cache...")
    container = av.open(str(obu_path))
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        cls_img = np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4)
        frames.append(cls_img)
    container.close()
    
    # Completeness Check
    if len(frames) != expected_frames:
        raise RuntimeError(f"FFmpeg encoding failed! Generated {len(frames)} frames, expected {expected_frames}.")
    
    # Cleanup temporary files (keep the .obu.br files)
    obu_path.unlink()
    raw_path.unlink()
    
    mask_tensor = torch.from_numpy(np.stack(frames)).contiguous()
    torch.save(mask_tensor, mask_tensor_path)
    return mask_tensor

def extract_and_compress_poses(rgb_pairs_all, posenet, device, archive_dir, batch_size=8, cache_dir: Path | None = None):
    cache_dir = cache_dir or archive_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    br_path = cache_dir / POSE_PAYLOAD_NAME
    pose_tensor_path = cache_dir / POSE_CACHE_NAME

    if br_path.exists() and pose_tensor_path.exists():
        logging.info(f"Loading cached poses from {cache_dir}...")
        pose_tensor = load_cached_tensor(pose_tensor_path).float().contiguous()
        copy_if_different(br_path, archive_dir / POSE_PAYLOAD_NAME)
        return pose_tensor

    all_pose6 = []

    logging.info("Generating FP32 Poses from cached RGB pairs...")
    with torch.inference_mode():
        for start in tqdm(range(0, rgb_pairs_all.shape[0], batch_size), desc="Extracting Poses"):
            batch = rgb_pairs_all[start:start+batch_size].to(device).float()
            batch = einops.rearrange(batch, "b t h w c -> b t c h w")
            
            posenet_in = posenet.preprocess_input(batch)
            out = posenet(posenet_in)
            pose6 = out["pose"][..., :6].to(torch.float32)
            all_pose6.append(pose6.cpu().numpy())

    pose_arr = np.concatenate(all_pose6, axis=0).astype(np.float16)
    
    buffer = io.BytesIO()
    np.save(buffer, pose_arr)
    buffer.seek(0)
    
    logging.info("Applying Brotli compression to Poses...")
    with open(br_path, "wb") as f_out:
        f_out.write(brotli.compress(buffer.read(), quality=11, lgwin=24))

    pose_tensor = torch.from_numpy(pose_arr).float().contiguous()
    torch.save(pose_tensor.cpu(), pose_tensor_path)
    copy_if_different(br_path, archive_dir / POSE_PAYLOAD_NAME)
    return pose_tensor


def load_mask_tensor(path: Path, expected_frames: int | None = None) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Mask tensor not found: {path}")
    if path.suffix == ".npy":
        mask_tensor = torch.from_numpy(np.load(path)).contiguous()
    elif path.suffix == ".npz":
        with np.load(path) as data:
            key = "masks" if "masks" in data else data.files[0]
            mask_tensor = torch.from_numpy(data[key]).contiguous()
    else:
        mask_tensor = torch.load(path, map_location="cpu").contiguous()
    if mask_tensor.ndim != 3:
        raise ValueError(f"Expected mask tensor with shape [frames,h,w], got {tuple(mask_tensor.shape)} from {path}")
    if expected_frames is not None and mask_tensor.shape[0] != expected_frames:
        raise ValueError(f"Mask tensor frame count mismatch: got {mask_tensor.shape[0]}, expected {expected_frames}")
    return mask_tensor.to(torch.uint8)


def load_mask_tree_artifacts(mask_tree_dir: Path, archive_dir: Path, expected_frames: int | None = None) -> torch.Tensor:
    for name in MASK_TREE_PAYLOAD_NAMES:
        src = mask_tree_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing mask-tree payload: {src}")
        copy_if_different(src, archive_dir / name)

    tensor_path = mask_tree_dir / "reconstructed_masks.pt"
    if tensor_path.exists():
        return load_mask_tensor(tensor_path, expected_frames=expected_frames)

    raise FileNotFoundError(
        f"Missing {tensor_path}. Run mask_tree_codec.py decode or fit before using --mask-source masktree."
    )

class CachedPairLoader:
    def __init__(self, rgb_pairs_cpu, mask2_cpu, pose6_cpu, batch_size, device, seed=123, shuffle=True):
        self.rgb_pairs = rgb_pairs_cpu.contiguous()
        self.mask2 = mask2_cpu.contiguous()
        self.pose6 = pose6_cpu.contiguous()
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = rgb_pairs_cpu.shape[0]

    def set_epoch(self, epoch: int): self.epoch = int(epoch)
    def __len__(self): return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.num_samples, generator=g) if self.shuffle else torch.arange(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield (
                idx,
                self.rgb_pairs.index_select(0, idx).to(self.device, non_blocking=True),
                self.mask2.index_select(0, idx).to(self.device, non_blocking=True),
                self.pose6.index_select(0, idx).to(self.device, non_blocking=True),
            )

# -----------------------------
# FP4 Logic
# -----------------------------
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    @staticmethod
    def quantize_blockwise(x: torch.Tensor, block_size: int = 32):
        orig_shape = x.shape
        flat = x.reshape(-1)
        pad = (block_size - (flat.numel() % block_size)) % block_size
        if pad: flat = F.pad(flat, (0, pad))
        blocks = flat.view(-1, block_size)
        max_abs = blocks.abs().amax(dim=1, keepdim=True)
        scales = torch.where(max_abs > 0, max_abs / 6.0, torch.ones_like(max_abs))
        norm = blocks / scales
        signs = (norm < 0).to(torch.int16)
        levels = FP4Codebook.pos_levels.to(x.device, x.dtype).view(1, 1, -1)
        mag_idx = (norm.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.int16)
        q = torch.where(signs.bool(), -levels[0, 0, mag_idx.long()], levels[0, 0, mag_idx.long()])
        return (q * scales).view(-1)[:x.numel()].view(orig_shape), ((signs << 3) | mag_idx).to(torch.uint8), scales.squeeze(1)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        nibbles = nibbles.view(-1, nibbles.numel() // scales.numel())
        signs, mag_idx = (nibbles >> 3).to(torch.int64), (nibbles & 0x7).to(torch.int64)
        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].to(torch.float32)).view(-1)[:flat_n].reshape(orig_shape)

def fake_quant_fp4_ste(x, block_size=32):
    dq, _, _ = FP4Codebook.quantize_blockwise(x, block_size=block_size)
    return x + (dq - x).detach()

def pack_nibbles(nib):
    flat = nib.reshape(-1)
    if flat.numel() % 2 == 1: flat = F.pad(flat, (0, 1))
    return ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)

def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2], out[1::2] = (flat >> 4) & 0x0F, flat & 0x0F
    return out[:count]

def load_fp4_state_dict(model, fp4_path, device):
    data = torch.load(fp4_path, map_location=device)
    state_dict, source = {}, data.get("tensors", data.get("quantized", {}))
    for name, rec in source.items():
        if rec["weight_kind"] == "fp4_packed":
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w = FP4Codebook.dequantize_from_nibbles(nibbles, rec["scales_fp16"].to(device), rec["weight_shape"])
        else: w = rec["weight_fp16"].to(device).float()
        state_dict[f"{name}.weight"] = w
        if rec.get("bias_fp16") is not None: state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for k, v in data.get("dense_fp16", {}).items(): state_dict[k] = v.to(device).float()
    model.load_state_dict(state_dict, strict=False)
    return data.get("__meta__", {})

def export_fp4_state_dict(model, out_path, block_size=32, meta: dict[str, Any] | None = None):
    export = {
        "__format__": "fp4_standalone",
        "__block_size__": block_size,
        "__codebook__": FP4Codebook.pos_levels.clone(),
        "__meta__": meta or {},
        "quantized": {},
        "dense_fp16": {},
    }
    covered_keys = set()
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QEmbedding)):
            rec = {"type": "conv2d" if isinstance(m, QConv2d) else "embedding"}
            w = m.weight.detach().float().cpu()
            rec["weight_shape"] = list(w.shape)
            covered_keys.add(f"{name}.weight")
            if isinstance(m, QConv2d):
                rec["stride"], rec["padding"], rec["dilation"], rec["groups"] = list(m.stride) if isinstance(m.stride, tuple) else [m.stride]*2, list(m.padding) if isinstance(m.padding, tuple) else [m.padding]*2, list(m.dilation) if isinstance(m.dilation, tuple) else [m.dilation]*2, int(m.groups)
                rec["bias_fp16"] = m.bias.detach().half().cpu() if m.bias is not None else None
                if m.bias is not None: covered_keys.add(f"{name}.bias")
            if getattr(m, 'quantize_weight', False):
                _, nib, scales = FP4Codebook.quantize_blockwise(w, block_size=block_size)
                rec.update({"weight_kind": "fp4_packed", "weight_numel": int(w.numel()), "packed_weight": pack_nibbles(nib.cpu()), "scales_fp16": scales.half().cpu()})
            else: rec.update({"weight_kind": "fp16", "weight_fp16": w.half().cpu()})
            export["quantized"][name] = rec
    for k, v in model.state_dict().items():
        if k not in covered_keys: export["dense_fp16"][k] = v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()
    torch.save(export, out_path, _use_new_zipfile_serialization=False)

# -----------------------------
# Quantizable Modules
# -----------------------------
class QMixin:
    def set_qat(self, enabled: bool, act_enabled: bool = False):
        self.qat_enabled = enabled
        self.qat_act_enabled = act_enabled

class QConv2d(nn.Conv2d, QMixin):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.quantize_weight = quantize_weight
        self.qat_enabled = False

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QEmbedding(nn.Embedding, QMixin):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.quantize_weight = quantize_weight
        self.qat_enabled = False

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.embedding(x, w, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

# -----------------------------
# Architecture
# -----------------------------
class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
    def forward(self, x): return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x, cond_emb):
        x_base = self.norm2(self.conv2(self.conv1(x)))
        gamma, beta = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(x + (x_base * (1.0 + gamma) + beta))

class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=56, c2=64, depth_mult=1):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), SepConvGNAct(c2, c1, depth_mult=depth_mult))
        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2, coords, mask_logits=None):
        if mask_logits is None:
            e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        else:
            probs = F.softmax(mask_logits.float(), dim=1)
            emb = self.embedding.weight.float()
            e2 = torch.einsum("bchw,ce->behw", probs, emb)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        s = self.stem_block(self.stem_conv(torch.cat([e2_up, coords], dim=1)))
        z = self.up(self.down_block(self.down_conv(s)))
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))

class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=48, hidden=52, depth_mult=1):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)
    def forward(self, feat, cond_emb): return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat, cond_emb))))) * 255.0

class SegGridAdapter(nn.Module):
    def __init__(self, grid_ch, feat_ch):
        super().__init__()
        self.proj = nn.Sequential(
            QConv2d(grid_ch, feat_ch, 1, quantize_weight=False),
            nn.SiLU(inplace=True),
            QConv2d(feat_ch, feat_ch, 1, quantize_weight=False),
        )
        with torch.no_grad():
            self.proj[-1].weight.zero_()
            if self.proj[-1].bias is not None:
                self.proj[-1].bias.zero_()

    def forward(self, z_grid, size):
        z_up = F.interpolate(z_grid, size=size, mode="bilinear", align_corners=False)
        return self.proj(z_up)

class LatentMaskAdapter(nn.Module):
    def __init__(self, grid_ch, num_classes):
        super().__init__()
        self.proj = QConv2d(grid_ch, num_classes, 1, quantize_weight=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            if self.proj.bias is not None:
                self.proj.bias.zero_()
            if grid_ch == num_classes:
                for i in range(num_classes):
                    self.proj.weight[i, i, 0, 0] = 4.0

    def forward(self, z_grid, size):
        z_up = F.interpolate(z_grid, size=size, mode="bilinear", align_corners=False)
        return self.proj(z_up)

class JointFrameGenerator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.shared_trunk = SharedMaskDecoder(
            num_classes=config.num_classes,
            emb_dim=6,
            c1=config.c1,
            c2=config.c2,
            depth_mult=config.depth_mult,
        )
        self.pose_mlp = nn.Sequential(
            nn.Linear(config.pose_dim, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.latent_mlp = None
        if config.z_dim > 0:
            self.latent_mlp = nn.Sequential(
                nn.Linear(config.z_dim, config.cond_dim),
                nn.SiLU(),
                nn.Linear(config.cond_dim, config.cond_dim),
            )
            with torch.no_grad():
                self.latent_mlp[-1].weight.zero_()
                self.latent_mlp[-1].bias.zero_()
        self.seg_grid_adapter = None
        if config.z_seg_dim > 0:
            self.seg_grid_adapter = SegGridAdapter(config.z_seg_channels, config.c1)
        self.latent_mask_adapter = None
        if config.mask_from_latent:
            if config.z_seg_dim <= 0:
                raise ValueError("mask_from_latent requires a spatial z_seg grid.")
            self.latent_mask_adapter = LatentMaskAdapter(config.z_seg_channels, config.num_classes)
        self.frame1_head = FrameHead(
            in_ch=config.c1,
            cond_dim=config.cond_dim,
            hidden=config.hidden,
            depth_mult=config.depth_mult,
        )
        self.frame2_head = FrameHead(
            in_ch=config.c1,
            cond_dim=config.cond_dim,
            hidden=config.hidden,
            depth_mult=config.depth_mult,
        )

    def split_latent(self, z):
        if z is None:
            return None, None
        z_pose = z[:, : self.config.z_dim] if self.config.z_dim > 0 else None
        z_seg = None
        if self.config.z_seg_dim > 0:
            start = self.config.z_dim
            end = start + self.config.z_seg_dim
            z_seg = z[:, start:end].view(
                z.shape[0],
                self.config.z_seg_channels,
                self.config.z_seg_h,
                self.config.z_seg_w,
            )
        return z_pose, z_seg

    def set_qat(self, enabled: bool):
        for m in self.modules():
            if isinstance(m, (QConv2d, QEmbedding)): m.set_qat(enabled=enabled)

    def make_cond_embedding(self, pose6, z=None, include_latent=True):
        cond_emb = self.pose_mlp(pose6)
        if self.latent_mlp is not None and include_latent:
            if z is None:
                raise ValueError("Latent conditioning is enabled but no latent tensor was provided.")
            cond_emb = cond_emb + self.latent_mlp(z)
        return cond_emb

    def forward(self, mask2, pose6, z=None):
        coords = make_coord_grid(mask2.shape[0], 384, 512, mask2.device, torch.float32)
        z_pose, z_seg = self.split_latent(z)
        mask_logits = None
        if self.latent_mask_adapter is not None:
            if z_seg is None:
                raise ValueError("mask_from_latent requires z_seg latents.")
            mask_logits = self.latent_mask_adapter(z_seg, (384, 512))
        shared_feat = self.shared_trunk(mask2, coords, mask_logits=mask_logits)
        use_latent = self.latent_mlp is not None and z_pose is not None
        cond_emb1 = self.make_cond_embedding(pose6, z_pose, include_latent=use_latent)
        cond_emb2 = self.make_cond_embedding(
            pose6,
            z_pose,
            include_latent=use_latent and self.config.frame2_uses_latent,
        )
        frame2_feat = shared_feat
        if self.seg_grid_adapter is not None and z_seg is not None:
            frame2_feat = frame2_feat + self.seg_grid_adapter(z_seg, shared_feat.shape[-2:])
        return self.frame1_head(shared_feat, cond_emb1), self.frame2_head(frame2_feat, cond_emb2)

# -----------------------------
# Freeze Control & Training Engine
# -----------------------------
def apply_freeze_state(model: JointFrameGenerator, stage: Stage):
    for p in model.parameters(): p.requires_grad = True
    
    if stage == Stage.ANCHOR:
        logging.info("STAGE: ANCHOR -> Freezing Frame 1 and Pose.")
        for p in model.frame1_head.parameters(): p.requires_grad = False
        for p in model.pose_mlp.parameters(): p.requires_grad = False
    elif stage == Stage.FINETUNE:
        logging.info("STAGE: FINETUNE -> Freezing Shared Trunk and Frame 2.")
        for p in model.shared_trunk.parameters(): p.requires_grad = False
        for p in model.frame2_head.parameters(): p.requires_grad = False
    elif stage == Stage.POSE_RESCUE:
        logging.info("STAGE: POSE_RESCUE -> Freezing Shared Trunk and Frame 2; optimizing Pose + Frame 1.")
        for p in model.shared_trunk.parameters(): p.requires_grad = False
        for p in model.frame2_head.parameters(): p.requires_grad = False
    elif stage == Stage.SEG_RESCUE:
        logging.info("STAGE: SEG_RESCUE -> Optimizing Frame 2 + late shared decoder for SegNet.")
        for p in model.parameters(): p.requires_grad = False
        for p in model.frame2_head.parameters(): p.requires_grad = True
        for p in model.shared_trunk.fuse.parameters(): p.requires_grad = True
        for p in model.shared_trunk.fuse_block.parameters(): p.requires_grad = True
        if model.seg_grid_adapter is not None:
            for p in model.seg_grid_adapter.parameters(): p.requires_grad = True
    elif stage == Stage.SPLIT_LATENT:
        logging.info("STAGE: SPLIT_LATENT -> Optimizing split latent adapters, frame heads, and pose conditioning.")
        for p in model.parameters(): p.requires_grad = False
        for module in (model.frame1_head, model.frame2_head, model.pose_mlp):
            for p in module.parameters(): p.requires_grad = True
        if model.latent_mlp is not None:
            for p in model.latent_mlp.parameters(): p.requires_grad = True
        if model.seg_grid_adapter is not None:
            for p in model.seg_grid_adapter.parameters(): p.requires_grad = True
        if model.latent_mask_adapter is not None:
            for p in model.latent_mask_adapter.parameters(): p.requires_grad = True
    elif stage == Stage.JOINT:
        logging.info("STAGE: JOINT -> All parameters unfrozen.")

    if stage in [Stage.FINETUNE, Stage.POSE_RESCUE, Stage.SPLIT_LATENT]:
        model.shared_trunk.eval()
    if stage in [Stage.FINETUNE, Stage.POSE_RESCUE]:
        model.frame2_head.eval()

def build_optimizer(generator: JointFrameGenerator, pair_latents: nn.Embedding | None, lr: float):
    param_groups = [{"params": [p for p in generator.parameters() if p.requires_grad]}]
    if pair_latents is not None:
        param_groups.append({"params": list(pair_latents.parameters()), "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.99))


def get_trainable_params(generator: JointFrameGenerator, pair_latents: nn.Embedding | None):
    params = [p for p in generator.parameters() if p.requires_grad]
    if pair_latents is not None:
        params.extend(list(pair_latents.parameters()))
    return params


def score_proxy_metrics(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * segnet_dist + math.sqrt(max(0.0, 10.0 * posenet_dist))


def export_candidate_artifacts(
    generator: JointFrameGenerator,
    pair_latents: nn.Embedding | None,
    device: torch.device,
    archive_dir: Path,
    output_archive_zip: Path,
    model_config: ModelConfig,
    include_mask_payload: bool = True,
    mask_payload_kind: str | None = None,
) -> Path:
    fp4_path = archive_dir / "candidate_model_fp4.pt"
    export_fp4_state_dict(generator.cpu(), fp4_path, meta=model_config.to_meta())
    generator.to(device)

    stale_qpack = archive_dir / MODEL_QPACK_PAYLOAD_NAME
    if stale_qpack.exists():
        stale_qpack.unlink()

    with open(fp4_path, "rb") as f_in, open(archive_dir / MODEL_PAYLOAD_NAME, "wb") as f_out:
        f_out.write(brotli.compress(f_in.read(), quality=11, lgwin=24))

    latent_path = archive_dir / LATENT_PAYLOAD_NAME
    if pair_latents is not None and model_config.total_latent_dim > 0:
        save_quantized_latents(pair_latents.weight.detach(), latent_path, bits=model_config.latent_quant_bits)
    elif latent_path.exists():
        latent_path.unlink()

    package_submission_archive(
        archive_dir,
        output_archive_zip,
        include_latents=model_config.total_latent_dim > 0,
        include_mask=include_mask_payload,
        mask_payload_kind=mask_payload_kind,
    )
    return fp4_path


def load_best_artifacts(
    generator: JointFrameGenerator,
    pair_latents: nn.Embedding | None,
    archive_dir: Path,
    run_name: str,
    device: torch.device,
) -> PipelineState:
    best_model_path = archive_dir / f"{run_name}_best_fp4.pt"
    load_fp4_state_dict(generator, best_model_path, device)
    generator.float()

    latent_state = None
    if pair_latents is not None:
        latent_path = archive_dir / f"{run_name}_best_latents.pt"
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent checkpoint: {latent_path}")
        latent_state = torch.load(latent_path, map_location="cpu")
        pair_latents.load_state_dict(latent_state)

    return PipelineState(
        model_state={k: v.detach().cpu().clone() for k, v in generator.state_dict().items()},
        latent_state={k: v.detach().cpu().clone() for k, v in pair_latents.state_dict().items()} if pair_latents is not None else None,
    )


def train_run(
    run: PipelineRun,
    generator: JointFrameGenerator,
    pair_latents: nn.Embedding | None,
    loader: CachedPairLoader,
    device,
    archive_dir: Path,
    output_archive_zip: Path,
    submission_dir: Path,
    video_names_file: Path,
    official_eval_device: str | None,
    model_config: ModelConfig,
    aux_models,
    selection_metric: str,
    eval_interval: int,
    eval_tail: int,
    grad_accum_steps: int,
    include_mask_payload: bool,
    mask_payload_kind: str,
    skip_final_official: bool,
    state_to_load: PipelineState | None = None,
):
    segnet, posenet, distortion_net = aux_models
    apply_freeze_state(generator, run.stage)

    optimizer = build_optimizer(generator, pair_latents, run.lr)
    start_epoch, best_metric = 0, float("inf")
    best_state_path = archive_dir / f"{run.name}_best_state.pt"
    best_proxy_metrics_path = archive_dir / f"{run.name}_best_proxy_metrics.json"
    best_model_path = archive_dir / f"{run.name}_best_fp4.pt"
    
    latest_path = archive_dir / f"{run.name}_latest.pt"
    if latest_path.exists():
        logging.info(f"Resuming {run.name} from {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        generator.load_state_dict(checkpoint["model_state"])
        if pair_latents is not None and checkpoint.get("latent_state") is not None:
            pair_latents.load_state_dict(checkpoint["latent_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
    elif selection_metric == "proxy" and best_state_path.exists() and not best_model_path.exists():
        logging.info(f"Recovering {run.name} from best proxy state at {best_state_path}")
        best_state = torch.load(best_state_path, map_location=device)
        generator.load_state_dict(best_state["model_state"])
        if pair_latents is not None and best_state.get("latent_state") is not None:
            pair_latents.load_state_dict(best_state["latent_state"])
        if best_proxy_metrics_path.exists():
            best_metric = json.loads(best_proxy_metrics_path.read_text())["proxy_score"]
        start_epoch = run.epochs
    elif state_to_load is not None:
        logging.info("Loading previous stage state dict into Generator...")
        if state_to_load.model_state is not None:
            generator.load_state_dict(state_to_load.model_state)
        if pair_latents is not None and state_to_load.latent_state is not None:
            pair_latents.load_state_dict(state_to_load.latent_state)

    ema = EMA(generator, decay=run.ema_decay) if run.ema_decay > 0 else None
    if ema and latest_path.exists() and checkpoint.get("ema_state"):
        ema.shadow = {k: v.to(device) for k, v in checkpoint["ema_state"].items()}

    qat_warmup = min(run.warmup_epochs, max(1, (run.epochs - run.qat_start_epoch) // 2)) if run.qat_start_epoch == 0 else run.warmup_epochs
    warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=qat_warmup)
    main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, run.epochs - qat_warmup))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sch, main_sch], milestones=[qat_warmup])

    for _ in range(start_epoch): scheduler.step()

    for epoch in range(start_epoch, run.epochs):
        generator.train()
        if run.stage in [Stage.FINETUNE, Stage.POSE_RESCUE, Stage.SPLIT_LATENT]:
            generator.shared_trunk.eval()
        if run.stage in [Stage.FINETUNE, Stage.POSE_RESCUE]:
            generator.frame2_head.eval()

        loader.set_epoch(epoch)
        qat_on = epoch >= run.qat_start_epoch
        generator.set_qat(qat_on)

        if epoch == run.qat_start_epoch and run.qat_start_epoch > 0:
            logging.info(f"--- QAT Phase Initiated. Resetting Optimizer ---")
            optimizer = build_optimizer(generator, pair_latents, run.lr)
            warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=qat_warmup)
            main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, (run.epochs - epoch) - qat_warmup))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sch, main_sch], milestones=[qat_warmup])

        kl_ce_alpha = min(1.0, epoch / max(1, run.qat_start_epoch // 2)) if run.qat_start_epoch > 0 else 1.0
        seg2_kl_w = 0.9 - (0.9 * kl_ce_alpha)
        seg2_ce_w = 0.1 + (0.9 * kl_ce_alpha)
        frame1_sem_w = 0.0
        if run.stage != Stage.POSE_RESCUE and run.frame1_fade_epochs > 0:
            frame1_sem_w = max(0.0, 1.0 - (epoch / run.frame1_fade_epochs))

        total_loss_sum, total_seg2_ce, total_seg1_ce, total_pose_dist, batches = 0.0, 0.0, 0.0, 0.0, 0

        accum_steps = max(1, grad_accum_steps)
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Run: {run.name} | Epoch {epoch+1}/{run.epochs}", leave=False)
        for batch_idx, (idx_cpu, batch_rgb, in_mask2, in_pose6) in enumerate(pbar):
            batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float().to(device)
            in_mask2, in_pose6 = in_mask2.to(device).long(), in_pose6.to(device).float()
            z_batch = pair_latents(idx_cpu.to(device)).float() if pair_latents is not None else None

            need_pose = run.stage in [Stage.FINETUNE, Stage.JOINT, Stage.POSE_RESCUE, Stage.SEG_RESCUE, Stage.SPLIT_LATENT]
            need_seg2 = run.stage in [Stage.ANCHOR, Stage.JOINT, Stage.POSE_RESCUE, Stage.SEG_RESCUE, Stage.SPLIT_LATENT]
            need_seg1 = frame1_sem_w > 0
            need_gt_logits2 = run.stage in [Stage.ANCHOR, Stage.JOINT, Stage.SEG_RESCUE, Stage.SPLIT_LATENT]
            need_gt_logits1 = run.stage == Stage.JOINT and frame1_sem_w > 0

            with torch.no_grad():
                gt_logits1 = gt_logits2 = gt_pose = None
                gt_mask1 = gt_mask2 = None
                if need_seg1:
                    real1 = F.interpolate(batch[:, 0], size=(384, 512), mode="bilinear", align_corners=False)
                    if need_gt_logits1:
                        gt_logits1 = segnet(real1).float()
                        gt_mask1 = gt_logits1.argmax(dim=1)
                    else:
                        gt_mask1 = segnet(real1).float().argmax(dim=1)
                if need_seg2:
                    real2 = F.interpolate(batch[:, 1], size=(384, 512), mode="bilinear", align_corners=False)
                    if need_gt_logits2:
                        gt_logits2 = segnet(real2).float()
                        gt_mask2 = gt_logits2.argmax(dim=1)
                    else:
                        gt_mask2 = segnet(real2).float().argmax(dim=1)
                if need_pose:
                    gt_pose = get_pose_tensor(posenet(posenet.preprocess_input(batch))).float()[..., :6]

            pred_frame1, pred_frame2 = generator(in_mask2, in_pose6, z_batch)

            fake1_up = F.interpolate(pred_frame1, size=(874, 1164), mode="bilinear", align_corners=False)
            fake2_up = F.interpolate(pred_frame2, size=(874, 1164), mode="bilinear", align_corners=False)
            fake1_down = F.interpolate(diff_round(fake1_up.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)
            fake2_down = F.interpolate(diff_round(fake2_up.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)

            loss, loss_pose, loss_seg2, loss_seg1 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            loss_seg2_ce, loss_seg1_ce = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            if need_pose:
                fake_pose = get_pose_tensor(posenet(pack_pair_yuv6(fake1_down, fake2_down).float())).float()[..., :6]
                loss_pose = F.mse_loss(fake_pose, gt_pose)

            if run.stage in [Stage.ANCHOR, Stage.JOINT]:
                fake_logits2 = segnet(fake2_down).float()
                ce_unreduced2 = F.cross_entropy(fake_logits2, gt_mask2, reduction='none')
                with torch.no_grad():
                    boost2 = 1.0 + ((fake_logits2.argmax(dim=1) != gt_mask2).float() * run.error_boost)
                loss_seg2_ce = (ce_unreduced2 * boost2).mean()
                loss_seg2_kl = kl_on_logits(fake_logits2, gt_logits2, 2.0) / (384 * 512)
                loss_seg2 = 100.0 * ((seg2_kl_w * loss_seg2_kl) + (seg2_ce_w * 0.5 * run.ce_weight * loss_seg2_ce))
            elif run.stage == Stage.POSE_RESCUE:
                fake_logits2 = segnet(fake2_down).float()
                loss_seg2_ce = F.cross_entropy(fake_logits2, gt_mask2)
                loss_seg2 = run.pose_rescue_seg2_scale * loss_seg2_ce
            elif run.stage in [Stage.SEG_RESCUE, Stage.SPLIT_LATENT]:
                fake_logits2 = segnet(fake2_down).float()
                ce_unreduced2 = F.cross_entropy(fake_logits2, gt_mask2, reduction="none")
                with torch.no_grad():
                    mismatch2 = (fake_logits2.argmax(dim=1) != gt_mask2).float()
                    hard_weight2 = boundary_weight_from_mask(gt_mask2) + (mismatch2 * run.error_boost)
                loss_seg2_ce = (ce_unreduced2 * hard_weight2).mean()
                loss_seg2_kl = kl_on_logits(fake_logits2, gt_logits2, 2.0) / (384 * 512)
                loss_seg2_margin = seg_margin_loss(fake_logits2, gt_mask2, hard_weight2)
                loss_seg2 = 100.0 * (
                    (0.4 * loss_seg2_kl)
                    + (0.4 * run.ce_weight * loss_seg2_ce)
                    + (0.2 * loss_seg2_margin)
                )

            if frame1_sem_w > 0:
                fake_logits1 = segnet(fake1_down).float()
                ce_unreduced1 = F.cross_entropy(fake_logits1, gt_mask1, reduction='none')
                with torch.no_grad():
                    boost1 = 1.0 + ((fake_logits1.argmax(dim=1) != gt_mask1).float() * run.error_boost)
                loss_seg1_ce = (ce_unreduced1 * boost1).mean()
                
                if run.stage == Stage.JOINT:
                    loss_seg1_kl = kl_on_logits(fake_logits1, gt_logits1, 2.0) / (384 * 512)
                    loss_seg1 = 100.0 * frame1_sem_w * ((seg2_kl_w * loss_seg1_kl) + (seg2_ce_w * 0.5 * run.ce_weight * loss_seg1_ce))
                else: 
                    loss_seg1 = 100.0 * frame1_sem_w * (run.ce_weight * loss_seg1_ce)

            if run.stage == Stage.ANCHOR: loss = loss_seg2
            elif run.stage == Stage.FINETUNE: loss = loss_seg1 + (run.pose_weight * loss_pose * 10.0)
            elif run.stage == Stage.JOINT: loss = loss_seg2 + loss_seg1 + (30.0 * run.pose_weight * loss_pose)
            elif run.stage == Stage.POSE_RESCUE:
                loss = (run.pose_rescue_pose_scale * loss_pose) + loss_seg2
                if run.tv_weight > 0:
                    loss = loss + (run.tv_weight * total_variation_loss(pred_frame1 / 255.0))
            elif run.stage == Stage.SEG_RESCUE:
                loss = loss_seg2 + (10.0 * run.pose_weight * loss_pose)
            elif run.stage == Stage.SPLIT_LATENT:
                loss = loss_seg2 + (20.0 * run.pose_weight * loss_pose)

            assert_finite("loss", loss)
            (loss / accum_steps).backward()

            should_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(generator, pair_latents), max_norm=run.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema and epoch >= run.warmup_epochs:
                    ema.update(generator)
            
            total_loss_sum += loss.item(); total_seg2_ce += loss_seg2_ce.item()
            total_seg1_ce += loss_seg1_ce.item(); total_pose_dist += loss_pose.item(); batches += 1
            pbar.set_postfix({"L": f"{loss.item():.2f}", "S2": f"{loss_seg2_ce.item():.2f}", "P": f"{loss_pose.item():.4f}"})

        scheduler.step()
        avg_loss, avg_s2, avg_s1, avg_p = total_loss_sum/max(1, batches), total_seg2_ce/max(1, batches), total_seg1_ce/max(1, batches), total_pose_dist/max(1, batches)

        logging.info(f"\nEpoch {epoch+1}/{run.epochs} {'[QAT ACTIVE]' if qat_on else ''}")
        logging.info(f"  Avg Loss:      {avg_loss:.4f}")
        if run.stage in [Stage.ANCHOR, Stage.JOINT, Stage.POSE_RESCUE, Stage.SEG_RESCUE, Stage.SPLIT_LATENT]: logging.info(f"  Avg Seg2 CE:   {avg_s2:.4f}")
        if frame1_sem_w > 0: logging.info(f"  Avg Seg1 CE:   {avg_s1:.4f}")
        if run.stage in [Stage.FINETUNE, Stage.JOINT, Stage.POSE_RESCUE, Stage.SEG_RESCUE, Stage.SPLIT_LATENT]: logging.info(f"  Avg Pose MSE:  {avg_p:.6f}")

        is_eval_epoch = qat_on and (
            ((epoch - run.qat_start_epoch) % max(1, eval_interval) == 0)
            or (run.epochs - epoch <= max(1, eval_tail))
        )

        if is_eval_epoch:
            if ema: ema.apply_shadow(generator)
            generator.eval()
            total_seg, total_pose, samples = 0.0, 0.0, 0
            
            with torch.inference_mode():
                eval_pbar = tqdm(loader, desc=f"Eval: {run.name} Ep {epoch+1}", leave=False)
                for idx_cpu, batch_rgb, in_mask2, in_pose6 in eval_pbar:
                    batch_gt = batch_rgb.to(device)
                    z_batch = pair_latents(idx_cpu.to(device)).float() if pair_latents is not None else None
                    p1, p2 = generator(in_mask2.to(device).long(), in_pose6.to(device).float(), z_batch)
                    
                    b_comp = torch.stack([F.interpolate(p1, size=(874, 1164), mode="bilinear", align_corners=False), 
                                          F.interpolate(p2, size=(874, 1164), mode="bilinear", align_corners=False)], dim=1)
                    b_comp = einops.rearrange(b_comp, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
                    
                    p_dist, s_dist = distortion_net.compute_distortion(batch_gt, b_comp)
                    total_seg += s_dist.sum().item()
                    total_pose += p_dist.sum().item()
                    samples += batch_gt.shape[0]
            
            avg_seg = total_seg / max(1, samples)
            avg_pose = total_pose / max(1, samples)
            proxy_score = score_proxy_metrics(avg_seg, avg_pose)

            logging.info(
                "  [Proxy] Score: %.5f | Seg(x100): %.5f | Pose(√x10): %.5f"
                % (
                    proxy_score,
                    100.0 * avg_seg,
                    math.sqrt(max(0.0, 10.0 * avg_pose)),
                )
            )

            if selection_metric == "proxy":
                if proxy_score < best_metric:
                    best_metric = proxy_score
                    torch.save(
                        {
                            "model_state": generator.state_dict(),
                            "latent_state": pair_latents.state_dict() if pair_latents is not None else None,
                        },
                        best_state_path,
                    )
                    with open(best_proxy_metrics_path, "w") as f_out:
                        json.dump(
                            {
                                "proxy_score": proxy_score,
                                "segnet_dist": avg_seg,
                                "posenet_dist": avg_pose,
                            },
                            f_out,
                            indent=2,
                            sort_keys=True,
                        )
                    logging.info(f"  *** New Best Proxy Score: {best_metric:.5f} ***")
            else:
                candidate_fp4_path = export_candidate_artifacts(
                    generator=generator,
                    pair_latents=pair_latents,
                    device=device,
                    archive_dir=archive_dir,
                    output_archive_zip=output_archive_zip,
                    model_config=model_config,
                    include_mask_payload=include_mask_payload,
                    mask_payload_kind=mask_payload_kind,
                )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                official_metrics = run_official_evaluation(
                    submission_dir=submission_dir,
                    archive_zip_path=output_archive_zip,
                    video_names_file=video_names_file,
                    eval_device=official_eval_device,
                )
                logging.info(
                    "  [Official] Score: %.5f | Seg(x100): %.5f | Pose(√x10): %.5f | Rate(x25): %.5f"
                    % (
                        official_metrics["score"],
                        100.0 * official_metrics["segnet_dist"],
                        math.sqrt(10.0 * official_metrics["posenet_dist"]),
                        25.0 * official_metrics["rate"],
                    )
                )

                if official_metrics["score"] < best_metric:
                    best_metric = official_metrics["score"]
                    shutil.copyfile(candidate_fp4_path, archive_dir / f"{run.name}_best_fp4.pt")
                    shutil.copyfile(archive_dir / MODEL_PAYLOAD_NAME, archive_dir / f"{run.name}_best_model.pt.br")
                    shutil.copyfile(output_archive_zip, archive_dir / f"{run.name}_best_archive.zip")
                    if pair_latents is not None:
                        torch.save(pair_latents.state_dict(), archive_dir / f"{run.name}_best_latents.pt")
                        shutil.copyfile(archive_dir / LATENT_PAYLOAD_NAME, archive_dir / f"{run.name}_best_{LATENT_PAYLOAD_NAME}")
                    with open(archive_dir / f"{run.name}_best_metrics.json", "w") as f_out:
                        json.dump(official_metrics, f_out, indent=2, sort_keys=True)
                    logging.info(f"  *** New Best Official Score: {best_metric:.5f} ***")

            if ema: ema.restore(generator)

        torch.save({
            "epoch": epoch, "best_metric": best_metric,
            "model_state": generator.state_dict(),
            "latent_state": pair_latents.state_dict() if pair_latents is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "ema_state": {k: v.cpu() for k, v in ema.shadow.items()} if ema else None
        }, latest_path)

    if latest_path.exists(): latest_path.unlink()
    if selection_metric == "proxy" and best_state_path.exists():
        best_state = torch.load(best_state_path, map_location=device)
        generator.load_state_dict(best_state["model_state"])
        if pair_latents is not None and best_state.get("latent_state") is not None:
            pair_latents.load_state_dict(best_state["latent_state"])

    if not best_model_path.exists():
        export_candidate_artifacts(
            generator,
            pair_latents,
            device,
            archive_dir,
            output_archive_zip,
            model_config,
            include_mask_payload=include_mask_payload,
            mask_payload_kind=mask_payload_kind,
        )
        shutil.copyfile(archive_dir / "candidate_model_fp4.pt", best_model_path)
        shutil.copyfile(archive_dir / MODEL_PAYLOAD_NAME, archive_dir / f"{run.name}_best_model.pt.br")
        shutil.copyfile(output_archive_zip, archive_dir / f"{run.name}_best_archive.zip")
        if pair_latents is not None:
            torch.save(pair_latents.state_dict(), archive_dir / f"{run.name}_best_latents.pt")
        if skip_final_official:
            official_metrics = {
                "skipped_official": True,
                "archive_bytes": float(output_archive_zip.stat().st_size),
                "best_proxy": json.loads(best_proxy_metrics_path.read_text()) if best_proxy_metrics_path.exists() else None,
            }
        else:
            official_metrics = run_official_evaluation(
                submission_dir=submission_dir,
                archive_zip_path=output_archive_zip,
                video_names_file=video_names_file,
                eval_device=official_eval_device,
            )
        with open(archive_dir / f"{run.name}_best_metrics.json", "w") as f_out:
            json.dump(official_metrics, f_out, indent=2, sort_keys=True)

    return load_best_artifacts(generator, pair_latents, archive_dir, run.name, device)

# -----------------------------
# Main Setup
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", type=Path, default=ROOT_DIR / "videos")
    p.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    p.add_argument("--crf", type=int, default=50, help="CRF value for AV1 OBU mask compression")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--archive-dir", type=Path, default=Path(__file__).parent / "archive")
    p.add_argument("--output-archive-zip", type=Path, default=Path(__file__).parent / "archive.zip")
    p.add_argument("--official-eval-device", type=str, default=None, help="Device passed to evaluate.sh; defaults to the training device type.")
    p.add_argument("--mask-encode-size", type=parse_size_arg, default=SEGNET_MODEL_INPUT_SIZE, help="Encode mask side-channel as WIDTHxHEIGHT before AV1 compression.")
    p.add_argument("--mask-source", choices=["av1", "masktree", "tensor"], default="av1", help="Training mask source: generated AV1 masks, mask-tree reconstructed masks, or a tensor file.")
    p.add_argument("--mask-tree-dir", type=Path, default=None, help="Directory containing mask-tree payloads and reconstructed_masks.pt.")
    p.add_argument("--mask-tensor-path", type=Path, default=None, help="Mask tensor path for --mask-source tensor.")
    p.add_argument("--mask-payload-kind", choices=["auto", "av1", "masktree", "none"], default="auto", help="Mask payload to include in archive.zip.")
    p.add_argument("--z-dim", type=int, default=8)
    p.add_argument("--z-seg-channels", type=int, default=0)
    p.add_argument("--z-seg-h", type=int, default=0)
    p.add_argument("--z-seg-w", type=int, default=0)
    p.add_argument("--latent-quant-bits", type=int, choices=[4, 8], default=8)
    p.add_argument("--cond-dim", type=int, default=48)
    p.add_argument("--c1", type=int, default=56)
    p.add_argument("--c2", type=int, default=64)
    p.add_argument("--hidden", type=int, default=52)
    p.add_argument("--depth-mult", type=int, default=1)
    p.add_argument("--frame1-only-latent", action="store_true", help="Apply latent conditioning to frame 1 only; frame 2 uses pose conditioning only.")
    p.add_argument("--mask-input-mode", choices=["decoded", "zero"], default="decoded", help="Use decoded SegNet mask side-channel or a synthetic all-zero mask input.")
    p.add_argument("--omit-mask-payload", action="store_true", help="Do not include mask.obu.br in archive.zip. Inflate will use all-zero masks.")
    p.add_argument("--mask-from-latent", action="store_true", help="Synthesize the shared-trunk mask embedding from z_seg instead of archive mask frames.")
    p.add_argument("--init-zseg-from-mask", action="store_true", help="Initialize z_seg grid from downsampled decoded mask one-hot probabilities.")
    p.add_argument("--pipeline-preset", choices=["full", "fast", "smoke", "seg_rescue", "split_latent"], default="full")
    p.add_argument("--include-pose-rescue", action="store_true", help="Append a frame1-focused PoseNet rescue stage after the base pipeline.")
    p.add_argument("--init-fp4", type=Path, default=None, help="Optional FP4 checkpoint to initialize from before running the selected pipeline.")
    p.add_argument("--init-latents", type=Path, default=None, help="Optional latent embedding state_dict to initialize from.")
    p.add_argument("--pose-rescue-epochs", type=int, default=120)
    p.add_argument("--pose-rescue-lr", type=float, default=5e-6)
    p.add_argument("--pose-rescue-qat-start-epoch", type=int, default=0)
    p.add_argument("--pose-rescue-frame1-fade-epochs", type=int, default=0)
    p.add_argument("--pose-rescue-pose-scale", type=float, default=50.0)
    p.add_argument("--pose-rescue-seg2-scale", type=float, default=1.0)
    p.add_argument("--pose-rescue-tv-weight", type=float, default=0.0)
    p.add_argument("--seg-rescue-epochs", type=int, default=60)
    p.add_argument("--seg-rescue-lr", type=float, default=5e-6)
    p.add_argument("--seg-rescue-qat-start-epoch", type=int, default=0)
    p.add_argument("--seg-rescue-error-boost", type=float, default=9.0)
    p.add_argument("--seg-rescue-pose-weight", type=float, default=0.25)
    p.add_argument("--split-latent-epochs", type=int, default=80)
    p.add_argument("--split-latent-lr", type=float, default=1e-5)
    p.add_argument("--split-latent-qat-start-epoch", type=int, default=0)
    p.add_argument("--split-latent-error-boost", type=float, default=9.0)
    p.add_argument("--split-latent-pose-weight", type=float, default=0.5)
    p.add_argument("--selection-metric", choices=["official", "proxy"], default="official")
    p.add_argument("--skip-final-official", action="store_true", help="For bounded probes, export artifacts but skip the final full official evaluation.")
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-tail", type=int, default=10)
    p.add_argument("--shared-cache-root", type=Path, default=None)
    p.add_argument("--decode-backend", choices=["dali", "av"], default="dali")
    return p.parse_args()


def build_pipeline(args) -> list[PipelineRun]:
    preset = args.pipeline_preset
    if preset == "fast":
        pipeline = [
            PipelineRun(name="run1_anchor", stage=Stage.ANCHOR, epochs=60, lr=5e-4, qat_start_epoch=20, frame1_fade_epochs=20, error_boost=9.0),
            PipelineRun(name="run2_anchor_boost", stage=Stage.ANCHOR, epochs=12, lr=2e-5, qat_start_epoch=0, frame1_fade_epochs=0, error_boost=49.0),
            PipelineRun(name="run3_finetune", stage=Stage.FINETUNE, epochs=48, lr=5e-5, qat_start_epoch=16, frame1_fade_epochs=20, pose_weight=1.0),
            PipelineRun(name="run4_finish", stage=Stage.JOINT, epochs=24, lr=1e-5, qat_start_epoch=0, frame1_fade_epochs=12, pose_weight=1.0),
        ]
    elif preset == "smoke":
        pipeline = [
            PipelineRun(name="run1_anchor", stage=Stage.ANCHOR, epochs=12, lr=5e-4, qat_start_epoch=4, frame1_fade_epochs=4, error_boost=9.0),
            PipelineRun(name="run2_finetune", stage=Stage.FINETUNE, epochs=12, lr=5e-5, qat_start_epoch=4, frame1_fade_epochs=4, pose_weight=1.0),
            PipelineRun(name="run3_finish", stage=Stage.JOINT, epochs=8, lr=1e-5, qat_start_epoch=0, frame1_fade_epochs=4, pose_weight=1.0),
        ]
    elif preset == "seg_rescue":
        pipeline = [
            PipelineRun(
                name="run7_seg_rescue",
                stage=Stage.SEG_RESCUE,
                epochs=args.seg_rescue_epochs,
                lr=args.seg_rescue_lr,
                qat_start_epoch=args.seg_rescue_qat_start_epoch,
                frame1_fade_epochs=0,
                error_boost=args.seg_rescue_error_boost,
                pose_weight=args.seg_rescue_pose_weight,
            ),
        ]
    elif preset == "split_latent":
        pipeline = [
            PipelineRun(
                name="run8_split_latent",
                stage=Stage.SPLIT_LATENT,
                epochs=args.split_latent_epochs,
                lr=args.split_latent_lr,
                qat_start_epoch=args.split_latent_qat_start_epoch,
                frame1_fade_epochs=0,
                error_boost=args.split_latent_error_boost,
                pose_weight=args.split_latent_pose_weight,
            ),
        ]
    else:
        pipeline = [
            PipelineRun(name="run1_anchor", stage=Stage.ANCHOR, epochs=400, lr=5e-4, qat_start_epoch=200, frame1_fade_epochs=50, error_boost=9.0),
            PipelineRun(name="run2_anchor_boost", stage=Stage.ANCHOR, epochs=80, lr=1e-5, qat_start_epoch=0, frame1_fade_epochs=0, error_boost=49.0),
            PipelineRun(name="run3_finetune", stage=Stage.FINETUNE, epochs=320, lr=5e-5, qat_start_epoch=120, frame1_fade_epochs=60, pose_weight=1.0),
            PipelineRun(name="run4_finish", stage=Stage.JOINT, epochs=160, lr=1e-5, qat_start_epoch=0, frame1_fade_epochs=40, pose_weight=1.0),
            PipelineRun(name="run5_micro", stage=Stage.FINETUNE, epochs=120, lr=5e-6, qat_start_epoch=0, frame1_fade_epochs=0, pose_weight=1.0),
        ]

    if args.include_pose_rescue:
        pipeline.append(
            PipelineRun(
                name="run6_pose_rescue",
                stage=Stage.POSE_RESCUE,
                epochs=args.pose_rescue_epochs,
                lr=args.pose_rescue_lr,
                qat_start_epoch=args.pose_rescue_qat_start_epoch,
                frame1_fade_epochs=args.pose_rescue_frame1_fade_epochs,
                pose_rescue_pose_scale=args.pose_rescue_pose_scale,
                pose_rescue_seg2_scale=args.pose_rescue_seg2_scale,
                tv_weight=args.pose_rescue_tv_weight,
                pose_weight=1.0,
            )
        )

    return pipeline

def main():
    args = parse_args()
    device = torch.device(args.device)
    official_eval_device = args.official_eval_device or device.type
    model_config = ModelConfig(
        z_dim=args.z_dim,
        z_seg_channels=args.z_seg_channels,
        z_seg_h=args.z_seg_h,
        z_seg_w=args.z_seg_w,
        latent_quant_bits=args.latent_quant_bits,
        cond_dim=args.cond_dim,
        c1=args.c1,
        c2=args.c2,
        hidden=args.hidden,
        depth_mult=args.depth_mult,
        frame2_uses_latent=not args.frame1_only_latent,
        mask_from_latent=args.mask_from_latent,
    )
    
    archive_dir = args.archive_dir
    archive_dir.mkdir(exist_ok=True, parents=True)
    shared_cache_root = args.shared_cache_root
    if shared_cache_root is not None:
        shared_cache_root.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(archive_dir / "pipeline.log")])
    logging.info("Initializing Full End-to-End Pipeline Execution...")
    logging.info(f"Model config: {model_config}")
    logging.info(f"Pipeline preset: {args.pipeline_preset} | Selection metric: {args.selection_metric}")
    logging.info(
        "Runtime knobs: batch_size=%d grad_accum_steps=%d frame1_only_latent=%s include_pose_rescue=%s total_latent_dim=%d mask_input_mode=%s mask_source=%s mask_payload_kind=%s omit_mask_payload=%s mask_encode_size=%dx%d"
        % (
            args.batch_size,
            args.grad_accum_steps,
            args.frame1_only_latent,
            args.include_pose_rescue,
            model_config.total_latent_dim,
            args.mask_input_mode,
            args.mask_source,
            args.mask_payload_kind,
            args.omit_mask_payload,
            args.mask_encode_size[0],
            args.mask_encode_size[1],
        )
    )

    # Load Auxiliary Networks once to use in Extraction and Validation
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for model in (segnet, posenet, dist_net):
        for p in model.parameters(): p.requires_grad = False

    # --- 1. PRELOAD DATA TO RAM ---
    files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]
    rgb_pairs_all = get_rgb_pairs(files, args.video_dir, args.batch_size, device, shared_cache_root, args.decode_backend)

    # --- 2. EXTRACT MASKS & POSES ---
    for stale_name in (MASK_PAYLOAD_NAME, *MASK_TREE_PAYLOAD_NAMES):
        stale_path = archive_dir / stale_name
        if stale_path.exists() and args.mask_payload_kind != "auto":
            stale_path.unlink()

    if args.mask_source == "masktree":
        if args.mask_tree_dir is None:
            raise ValueError("--mask-source masktree requires --mask-tree-dir")
        logging.info(f"Loading mask-tree reconstructed masks from {args.mask_tree_dir}")
        mask_frames_all = load_mask_tree_artifacts(
            args.mask_tree_dir,
            archive_dir,
            expected_frames=rgb_pairs_all.shape[0],
        )
        if args.mask_payload_kind == "auto":
            args.mask_payload_kind = "masktree"
    elif args.mask_source == "tensor":
        if args.mask_tensor_path is None:
            raise ValueError("--mask-source tensor requires --mask-tensor-path")
        logging.info(f"Loading mask tensor from {args.mask_tensor_path}")
        mask_frames_all = load_mask_tensor(args.mask_tensor_path, expected_frames=rgb_pairs_all.shape[0])
        if args.mask_payload_kind == "auto":
            args.mask_payload_kind = "none"
    elif args.mask_input_mode == "zero" and args.omit_mask_payload:
        logging.info("Using all-zero mask inputs and omitting mask payload from archive.")
        mask_frames_all = torch.zeros(
            (rgb_pairs_all.shape[0], SEGNET_MODEL_INPUT_SIZE[1], SEGNET_MODEL_INPUT_SIZE[0]),
            dtype=torch.uint8,
        )
        if args.mask_payload_kind == "auto":
            args.mask_payload_kind = "none"
    else:
        mask_frames_all = extract_and_compress_masks(
            rgb_pairs_all,
            segnet,
            device,
            args.crf,
            archive_dir,
            batch_size=args.batch_size,
            cache_dir=shared_cache_root,
            mask_encode_size=args.mask_encode_size,
        )
        if args.mask_input_mode == "zero":
            logging.info("Using all-zero mask inputs for training while keeping mask payload in archive.")
            mask_frames_all = torch.zeros_like(mask_frames_all)
        if args.mask_payload_kind == "auto":
            args.mask_payload_kind = "av1"
    pose6_all = extract_and_compress_poses(
        rgb_pairs_all,
        posenet,
        device,
        archive_dir,
        batch_size=args.batch_size,
        cache_dir=shared_cache_root,
    )

    # Initialize Train Loader
    loader = CachedPairLoader(rgb_pairs_all, mask_frames_all, pose6_all, args.batch_size, device)
    generator = JointFrameGenerator(model_config).to(device)
    if args.init_fp4 is not None:
        logging.info(f"Initializing generator from FP4 checkpoint: {args.init_fp4}")
        load_fp4_state_dict(generator, args.init_fp4, device)
        generator.float()
    pair_latents = nn.Embedding(rgb_pairs_all.shape[0], model_config.total_latent_dim).to(device) if model_config.total_latent_dim > 0 else None
    if pair_latents is not None:
        with torch.no_grad():
            pair_latents.weight.zero_()
            if args.init_zseg_from_mask:
                if model_config.z_seg_channels != model_config.num_classes:
                    raise ValueError("--init-zseg-from-mask requires z_seg_channels == num_classes.")
                if model_config.z_seg_h <= 0 or model_config.z_seg_w <= 0:
                    raise ValueError("--init-zseg-from-mask requires a spatial z_seg grid.")
                logging.info("Initializing z_seg grid from downsampled decoded masks.")
                one_hot = F.one_hot(mask_frames_all.long(), num_classes=model_config.num_classes).permute(0, 3, 1, 2).float()
                grid = F.interpolate(
                    one_hot,
                    size=(model_config.z_seg_h, model_config.z_seg_w),
                    mode="area",
                )
                z_init = (grid * 2.0 - 1.0).reshape(mask_frames_all.shape[0], -1).to(device)
                pair_latents.weight[:, model_config.z_dim : model_config.z_dim + model_config.z_seg_dim].copy_(z_init)
        if args.init_latents is not None:
            logging.info(f"Initializing latents from checkpoint: {args.init_latents}")
            pair_latents.load_state_dict(torch.load(args.init_latents, map_location=device))

    # --- 3. PIPELINE EXECUTION ---
    PIPELINE = build_pipeline(args)

    current_state = None
    for run in PIPELINE:
        best_path = archive_dir / f"{run.name}_best_fp4.pt"
        best_latent_path = archive_dir / f"{run.name}_best_latents.pt"
        latest_path = archive_dir / f"{run.name}_latest.pt"
        
        # A run is ONLY fully complete if it has a best checkpoint AND the active latest.pt file was cleaned up.
        if best_path.exists() and not latest_path.exists() and (pair_latents is None or best_latent_path.exists()):
            logging.info(f"\n[SKIP] Run '{run.name}' is already completed. Loading state to pass to next stage...")
            current_state = load_best_artifacts(generator, pair_latents, archive_dir, run.name, device)
            continue
        
        if latest_path.exists():
            logging.info(f"\n[RESUME] Found interrupted state for '{run.name}'. Resuming...")
        
        logging.info(f"\n" + "="*50)
        logging.info(f"STARTING PIPELINE RUN: {run.name}")
        logging.info("="*50)
        
        current_state = train_run(
            run=run,
            generator=generator,
            pair_latents=pair_latents,
            loader=loader,
            device=device,
            archive_dir=archive_dir,
            output_archive_zip=args.output_archive_zip,
            submission_dir=Path(__file__).parent,
            video_names_file=args.video_names,
            official_eval_device=official_eval_device,
            model_config=model_config,
            aux_models=(segnet, posenet, dist_net),
            selection_metric=args.selection_metric,
            eval_interval=args.eval_interval,
            eval_tail=args.eval_tail,
            grad_accum_steps=args.grad_accum_steps,
            include_mask_payload=not args.omit_mask_payload,
            mask_payload_kind=args.mask_payload_kind,
            skip_final_official=args.skip_final_official,
            state_to_load=current_state,
        )

    export_candidate_artifacts(
        generator,
        pair_latents,
        device,
        archive_dir,
        args.output_archive_zip,
        model_config,
        include_mask_payload=not args.omit_mask_payload,
        mask_payload_kind=args.mask_payload_kind,
    )
    if args.skip_final_official:
        final_metrics = {
            "skipped_official": True,
            "archive_bytes": float(args.output_archive_zip.stat().st_size),
        }
    else:
        final_metrics = run_official_evaluation(
            submission_dir=Path(__file__).parent,
            archive_zip_path=args.output_archive_zip,
            video_names_file=args.video_names,
            eval_device=official_eval_device,
        )
    with open(archive_dir / "final_metrics.json", "w") as f_out:
        json.dump(final_metrics, f_out, indent=2, sort_keys=True)

    logging.info("\nEnd-to-End Execution Fully Completed.")
    logging.info(f"Final official score: {final_metrics['score']:.5f}")
    logging.info(f"Final payload saved to: {args.output_archive_zip}")

if __name__ == "__main__":
    main()
