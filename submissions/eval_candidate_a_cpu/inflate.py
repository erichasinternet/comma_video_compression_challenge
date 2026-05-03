#!/usr/bin/env python
import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


MODEL_PAYLOAD_NAME = "model.pt.br"
MODEL_QPACK_PAYLOAD_NAME = "model.qpack.br"
MASK_PAYLOAD_NAME = "mask.obu.br"
MASK_TREE_META_PAYLOAD_NAME = "mask_tree_meta.json.br"
MASK_TREE_TOKENS_PAYLOAD_NAME = "mask_tree_tokens.bin.br"
MASK_TREE_CODEBOOK_PAYLOAD_NAME = "mask_tree_codebook.bin.br"
POSE_PAYLOAD_NAME = "pose.npy.br"
LATENT_PAYLOAD_NAME = "z.npz.br"

MODE_PREV = 1
MODE_UNIFORM = 2
MODE_LEFT = 3
MODE_ABOVE = 4
MODE_TWO_CLASS = 5
MODE_DICT8 = 7
MODE_DICT16 = 8
MODE_RAW_RLE = 9
MODE_SPLIT = 10


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 5
    pose_dim: int = 6
    z_dim: int = 0
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

    @classmethod
    def from_meta(cls, meta: dict | None):
        meta = meta or {}
        return cls(
            num_classes=int(meta.get("num_classes", 5)),
            pose_dim=int(meta.get("pose_dim", 6)),
            z_dim=int(meta.get("z_dim", 0)),
            z_seg_channels=int(meta.get("z_seg_channels", 0)),
            z_seg_h=int(meta.get("z_seg_h", 0)),
            z_seg_w=int(meta.get("z_seg_w", 0)),
            latent_quant_bits=int(meta.get("latent_quant_bits", 8)),
            cond_dim=int(meta.get("cond_dim", 48)),
            c1=int(meta.get("c1", 56)),
            c2=int(meta.get("c2", 64)),
            hidden=int(meta.get("hidden", 52)),
            depth_mult=int(meta.get("depth_mult", 1)),
            frame2_uses_latent=bool(meta.get("frame2_uses_latent", 1)),
            mask_from_latent=bool(meta.get("mask_from_latent", 0)),
        )


# -----------------------------
# FP4 Dequantization Tools
# -----------------------------
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def dequantize_from_nibbles(nibbles: torch.Tensor, scales: torch.Tensor, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        block_size = nibbles.numel() // scales.numel()

        nibbles = nibbles.view(-1, block_size)
        signs = (nibbles >> 3).to(torch.int64)
        mag_idx = (nibbles & 0x7).to(torch.int64)

        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = levels[mag_idx]
        q = torch.where(signs.bool(), -q, q)
        dq = q * scales[:, None].to(torch.float32)
        return dq.view(-1)[:flat_n].reshape(orig_shape)

def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    flat = packed.reshape(-1)
    hi = (flat >> 4) & 0x0F
    lo = flat & 0x0F
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]

def get_decoded_state_dict(payload_data, device: torch.device):
    data = torch.load(io.BytesIO(payload_data), map_location=device)
    state_dict = {}

    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            padded_count = rec["packed_weight"].numel() * 2
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), padded_count)
            w = FP4Codebook.dequantize_from_nibbles(
                nibbles, rec["scales_fp16"].to(device), rec["weight_shape"]
            )
        else:
            w = rec["weight_fp16"].to(device).float()

        state_dict[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()

    for name, tensor in data["dense_fp16"].items():
        state_dict[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)

    return state_dict, ModelConfig.from_meta(data.get("__meta__"))


def _read_qpack_arrays(payload: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    if not payload.startswith(b"QPK1"):
        raise ValueError("Invalid qpack payload magic")
    header_len = struct.unpack_from("<I", payload, 4)[0]
    header_start = 8
    header_end = header_start + header_len
    header = json.loads(payload[header_start:header_end].decode("utf-8"))
    blob = memoryview(payload)[header_end:]
    arrays = {}
    for rec in header["arrays"]:
        dtype = np.dtype(rec["dtype"])
        arr_bytes = blob[rec["offset"] : rec["offset"] + rec["nbytes"]]
        arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(rec["shape"])
        arrays[rec["name"]] = arr
    return header, arrays


def get_qpack_state_dict(payload_data: bytes, device: torch.device):
    header, arrays = _read_qpack_arrays(payload_data)
    state_dict = {}

    for name, rec in header["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            packed = torch.from_numpy(arrays[rec["packed_weight"]].copy()).to(device)
            scales = torch.from_numpy(arrays[rec["scales_fp16"]].copy()).to(device)
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            w = FP4Codebook.dequantize_from_nibbles(nibbles, scales, rec["weight_shape"])
        elif rec["weight_kind"] == "int8_symmetric":
            values = torch.from_numpy(arrays[rec["weight_int8"]].copy()).to(device).float()
            scale = torch.from_numpy(arrays[rec["weight_scale_fp16"]].copy()).to(device).float()
            w = values * scale
        else:
            w = torch.from_numpy(arrays[rec["weight_fp16"]].copy()).to(device).float()
        state_dict[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            state_dict[f"{name}.bias"] = torch.from_numpy(arrays[rec["bias_fp16"]].copy()).to(device).float()

    for name, dense_rec in header["dense_fp16"].items():
        if isinstance(dense_rec, str):
            arr = arrays[dense_rec]
            tensor = torch.from_numpy(arr.copy()).to(device)
        elif dense_rec["kind"] == "int8_symmetric":
            values = torch.from_numpy(arrays[dense_rec["values"]].copy()).to(device).float()
            scale = torch.from_numpy(arrays[dense_rec["scale"]].copy()).to(device).float()
            tensor = values * scale
        else:
            arr = arrays[dense_rec["values"]]
            tensor = torch.from_numpy(arr.copy()).to(device)
        state_dict[name] = tensor.float() if torch.is_floating_point(tensor) else tensor

    return state_dict, ModelConfig.from_meta(header.get("__meta__"))


def load_quantized_latents(path: Path) -> torch.Tensor:
    with open(path, "rb") as f_in:
        payload = brotli.decompress(f_in.read())
    with np.load(io.BytesIO(payload)) as data:
        bits = int(data["bits"]) if "bits" in data else 8
        if bits == 4:
            packed = data["values_packed"].reshape(-1).astype(np.uint8)
            shape = tuple(int(x) for x in data["shape"])
            out = np.empty(packed.size * 2, dtype=np.uint8)
            out[0::2] = (packed >> 4) & 0x0F
            out[1::2] = packed & 0x0F
            values = (out[: int(np.prod(shape))].astype(np.int16) - 8).reshape(shape).astype(np.float32)
        else:
            values = data["values"].astype(np.float32)
        scales = data["scales"].astype(np.float32)
    return torch.from_numpy(values * scales[None, :]).float()

# -----------------------------
# Architecture (Inference Only)
# -----------------------------

class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class SepConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, depth_mult: int = 4, quantize_weight: bool = True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult

        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, depth_mult: int = 4, quantize_weight: bool = True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult

        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)

    def forward(self, x):
        return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch: int, depth_mult: int = 4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int, depth_mult: int = 4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)

        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond_emb):
        residual = x
        x = self.norm2(self.conv2(self.conv1(x)))

        film = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        x = x * (1.0 + gamma) + beta

        return self.act(residual + x)

class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=56, c2=64, depth_mult=1):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, depth_mult=depth_mult),
        )
        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, coords: torch.Tensor, mask_logits=None):
        if mask_logits is None:
            e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        else:
            probs = F.softmax(mask_logits.float(), dim=1)
            emb = self.embedding.weight.float()
            e2 = torch.einsum("bchw,ce->behw", probs, emb)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        f = self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))
        return f

class FrameHead(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int = 48, hidden: int = 52, depth_mult: int = 1):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat, cond_emb)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0

class SegGridAdapter(nn.Module):
    def __init__(self, grid_ch: int, feat_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            QConv2d(grid_ch, feat_ch, 1, quantize_weight=False),
            nn.SiLU(inplace=True),
            QConv2d(feat_ch, feat_ch, 1, quantize_weight=False),
        )

    def forward(self, z_grid: torch.Tensor, size) -> torch.Tensor:
        z_up = F.interpolate(z_grid, size=size, mode="bilinear", align_corners=False)
        return self.proj(z_up)

class LatentMaskAdapter(nn.Module):
    def __init__(self, grid_ch: int, num_classes: int):
        super().__init__()
        self.proj = QConv2d(grid_ch, num_classes, 1, quantize_weight=False)

    def forward(self, z_grid: torch.Tensor, size) -> torch.Tensor:
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
        self.seg_grid_adapter = None
        if config.z_seg_dim > 0:
            self.seg_grid_adapter = SegGridAdapter(config.z_seg_channels, config.c1)
        self.latent_mask_adapter = None
        if config.mask_from_latent:
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

    def split_latent(self, z: torch.Tensor | None):
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

    def make_cond_embedding(self, pose6: torch.Tensor, z: torch.Tensor | None = None, include_latent: bool = True):
        cond_emb = self.pose_mlp(pose6)
        if self.latent_mlp is not None and include_latent:
            if z is None:
                raise ValueError("Latent conditioning is enabled but no latent tensor was provided.")
            cond_emb = cond_emb + self.latent_mlp(z)
        return cond_emb

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor, z: torch.Tensor | None = None):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)

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
        pred_frame1 = self.frame1_head(shared_feat, cond_emb1)
        frame2_feat = shared_feat
        if self.seg_grid_adapter is not None and z_seg is not None:
            frame2_feat = frame2_feat + self.seg_grid_adapter(z_seg, shared_feat.shape[-2:])
        pred_frame2 = self.frame2_head(frame2_feat, cond_emb2)

        return pred_frame1, pred_frame2

def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


# -----------------------------
# Inference Helpers & Main
# -----------------------------
def load_encoded_mask_video(path: str) -> torch.Tensor:
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        cls_img = np.round(img / 63.0).astype(np.uint8)
        cls_img = np.clip(cls_img, 0, 4)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()


class _ByteReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def u8(self) -> int:
        value = self.data[self.pos]
        self.pos += 1
        return int(value)

    def u16(self) -> int:
        value = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return int(value)

    def u32(self) -> int:
        value = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return int(value)

    def bytes(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out


def _unpack_bitmap(bits: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(bits, dtype=np.uint8)
    unpacked = np.unpackbits(raw, bitorder="little")
    return unpacked[:count].astype(bool)


def _decode_rle_tile(reader: _ByteReader, h: int, w: int) -> np.ndarray:
    out = np.empty((h, w), dtype=np.uint8)
    for y in range(h):
        x = 0
        run_count = reader.u16()
        for _ in range(run_count):
            length = reader.u16()
            cls = reader.u8()
            out[y, x : x + length] = cls
            x += length
        if x != w:
            raise ValueError(f"RLE row width mismatch: got {x}, expected {w}")
    return out


def _decode_row_rle_frames(reader: _ByteReader, frames: int, h: int, w: int) -> np.ndarray:
    out = np.empty((frames, h, w), dtype=np.uint8)
    for i in range(frames):
        out[i] = _decode_rle_tile(reader, h, w)
    return out


def _decode_temporal_rle_frames(reader: _ByteReader, frames: int, h: int, w: int) -> np.ndarray:
    out = np.empty((frames, h, w), dtype=np.uint8)
    out[0] = _decode_rle_tile(reader, h, w)
    for i in range(1, frames):
        cur = out[i - 1].copy()
        for y in range(h):
            changed_count = reader.u16()
            for _ in range(changed_count):
                start_x = reader.u16()
                length = reader.u16()
                kind = reader.u8()
                if kind == 0:
                    cur[y, start_x : start_x + length] = reader.u8()
                elif kind == 1:
                    cur[y, start_x : start_x + length] = np.frombuffer(reader.bytes(length), dtype=np.uint8)
                else:
                    raise ValueError(f"Unknown temporal RLE run kind: {kind}")
        out[i] = cur
    return out


def _decode_tree_node(reader: _ByteReader, out: np.ndarray, prev: np.ndarray | None, codebook: np.ndarray, y: int, x: int, h: int, w: int):
    mode = reader.u8()
    if mode == MODE_PREV:
        if prev is None:
            raise ValueError("PREV mode used on first frame")
        out[y : y + h, x : x + w] = prev[y : y + h, x : x + w]
    elif mode == MODE_UNIFORM:
        out[y : y + h, x : x + w] = reader.u8()
    elif mode == MODE_LEFT:
        if x < w:
            raise ValueError("LEFT mode lacks a decoded neighbor")
        out[y : y + h, x : x + w] = out[y : y + h, x - w : x]
    elif mode == MODE_ABOVE:
        if y < h:
            raise ValueError("ABOVE mode lacks a decoded neighbor")
        out[y : y + h, x : x + w] = out[y - h : y, x : x + w]
    elif mode == MODE_TWO_CLASS:
        cls_a = reader.u8()
        cls_b = reader.u8()
        bits = _unpack_bitmap(reader.bytes((h * w + 7) // 8), h * w).reshape(h, w)
        out[y : y + h, x : x + w] = np.where(bits, cls_b, cls_a).astype(np.uint8)
    elif mode == MODE_DICT8:
        idx = reader.u16()
        out[y : y + h, x : x + w] = codebook[idx, :h, :w]
    elif mode == MODE_DICT16:
        if h != 16 or w != 16:
            raise ValueError("DICT16 is defined only for 16x16 tiles")
        for dy in (0, 8):
            for dx in (0, 8):
                idx = reader.u16()
                out[y + dy : y + dy + 8, x + dx : x + dx + 8] = codebook[idx]
    elif mode == MODE_RAW_RLE:
        out[y : y + h, x : x + w] = _decode_rle_tile(reader, h, w)
    elif mode == MODE_SPLIT:
        h1 = h // 2
        w1 = w // 2
        for cy, ch in ((y, h1), (y + h1, h - h1)):
            for cx, cw in ((x, w1), (x + w1, w - w1)):
                _decode_tree_node(reader, out, prev, codebook, cy, cx, ch, cw)
    else:
        raise ValueError(f"Unknown mask-tree mode: {mode}")


def _decode_tree_frames(reader: _ByteReader, frames: int, h: int, w: int, root_tile: int, codebook: np.ndarray) -> np.ndarray:
    out = np.empty((frames, h, w), dtype=np.uint8)
    for i in range(frames):
        frame = np.zeros((h, w), dtype=np.uint8)
        prev = out[i - 1] if i > 0 else None
        for y in range(0, h, root_tile):
            for x in range(0, w, root_tile):
                _decode_tree_node(reader, frame, prev, codebook, y, x, min(root_tile, h - y), min(root_tile, w - x))
        out[i] = frame
    return out


def _apply_mask_tree_residuals(reader: _ByteReader, masks: np.ndarray) -> np.ndarray:
    if reader.pos >= len(reader.data):
        return masks
    if reader.bytes(4) != b"RS1\0":
        raise ValueError("Invalid mask-tree residual stream magic")
    out = masks.copy()
    count = reader.u32()
    for _ in range(count):
        frame_idx = reader.u16()
        y = reader.u16()
        start = reader.u16()
        length = reader.u16()
        out[frame_idx, y, start : start + length] = np.frombuffer(reader.bytes(length), dtype=np.uint8)
    return out


def load_mask_tree_payload(data_dir: Path) -> torch.Tensor:
    with open(data_dir / MASK_TREE_META_PAYLOAD_NAME, "rb") as f:
        meta = json.loads(brotli.decompress(f.read()).decode("utf-8"))
    with open(data_dir / MASK_TREE_TOKENS_PAYLOAD_NAME, "rb") as f:
        token_payload = brotli.decompress(f.read())
    codebook = np.zeros((0, 8, 8), dtype=np.uint8)
    codebook_path = data_dir / MASK_TREE_CODEBOOK_PAYLOAD_NAME
    if codebook_path.exists():
        with open(codebook_path, "rb") as f:
            raw_codebook = brotli.decompress(f.read())
        if raw_codebook:
            codebook = np.frombuffer(raw_codebook, dtype=np.uint8).reshape(-1, 8, 8)

    reader = _ByteReader(token_payload)
    magic = reader.bytes(4)
    if magic != b"MTK1":
        raise ValueError("Invalid mask-tree token magic")

    frames = int(meta["frames"])
    h = int(meta["height"])
    w = int(meta["width"])
    variant = meta["variant"]
    if variant == "row_rle":
        masks = _decode_row_rle_frames(reader, frames, h, w)
    elif variant == "temporal_rle":
        masks = _decode_temporal_rle_frames(reader, frames, h, w)
    elif variant == "tree":
        masks = _decode_tree_frames(reader, frames, h, w, int(meta.get("root_tile", 64)), codebook)
        masks = _apply_mask_tree_residuals(reader, masks)
    else:
        raise ValueError(f"Unknown mask-tree variant: {variant}")

    return torch.from_numpy(masks).contiguous()

def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]

    model_qpack_br = data_dir / MODEL_QPACK_PAYLOAD_NAME
    model_br = data_dir / MODEL_PAYLOAD_NAME
    mask_br = data_dir / MASK_PAYLOAD_NAME
    pose_br = data_dir / POSE_PAYLOAD_NAME
    latent_br = data_dir / LATENT_PAYLOAD_NAME

    # 1. Load Weights
    if model_qpack_br.exists():
        with open(model_qpack_br, "rb") as f:
            weights_data = brotli.decompress(f.read())
        decoded_state, model_config = get_qpack_state_dict(weights_data, device)
    else:
        with open(model_br, "rb") as f:
            weights_data = brotli.decompress(f.read())
        decoded_state, model_config = get_decoded_state_dict(weights_data, device)
    generator = JointFrameGenerator(model_config).to(device)
    generator.load_state_dict(decoded_state, strict=True)
    generator.eval()

    # 2. Load Pose Vectors
    with open(pose_br, "rb") as f:
        pose_bytes = brotli.decompress(f.read())
    pose_frames_all = torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()

    # 3. Load masks. Prefer mask-tree payloads, then legacy OBU, then synthesize zeros.
    if (data_dir / MASK_TREE_META_PAYLOAD_NAME).exists() and (data_dir / MASK_TREE_TOKENS_PAYLOAD_NAME).exists():
        mask_frames_all = load_mask_tree_payload(data_dir)
    elif mask_br.exists():
        with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
            with open(mask_br, "rb") as f:
                tmp_obu.write(brotli.decompress(f.read()))
            tmp_obu_path = tmp_obu.name

        mask_frames_all = load_encoded_mask_video(tmp_obu_path)
        os.remove(tmp_obu_path)
    else:
        mask_frames_all = torch.zeros((pose_frames_all.shape[0], 384, 512), dtype=torch.uint8)

    latent_frames_all = None
    if model_config.total_latent_dim > 0:
        if not latent_br.exists():
            raise FileNotFoundError(f"Expected latent payload not found: {latent_br}")
        latent_frames_all = load_quantized_latents(latent_br)

    out_h, out_w = 874, 1164
    cursor = 0
    batch_size = 4 
    pairs_per_file = mask_frames_all.shape[0] // max(1, len(files))

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"
            
            # Retrieve exactly the pairs mapping to this file
            file_masks = mask_frames_all[cursor : cursor + pairs_per_file]
            file_poses = pose_frames_all[cursor : cursor + pairs_per_file]
            file_latents = latent_frames_all[cursor : cursor + pairs_per_file] if latent_frames_all is not None else None
            cursor += pairs_per_file
            
            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}")
                
                for i in pbar:
                    in_mask2 = file_masks[i : i + batch_size].to(device).long()
                    in_pose6 = file_poses[i : i + batch_size].to(device).float()
                    in_z = file_latents[i : i + batch_size].to(device).float() if file_latents is not None else None

                    fake1, fake2 = generator(in_mask2, in_pose6, in_z)

                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)

                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")

                    output_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(output_bytes.cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
