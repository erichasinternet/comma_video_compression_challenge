#!/usr/bin/env python
import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

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
MASK_MIX_MANIFEST_NAME = "mask_mix.json.br"
POSE_PAYLOAD_NAME = "pose.npy.br"
POSE_QPACK_PAYLOAD_NAME = "pose.qpack.br"


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

    return state_dict

def _read_qpack_arrays(payload: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    if not payload.startswith(b"QPK1"):
        raise ValueError("Invalid qpack payload magic")
    header_len = struct.unpack_from("<I", payload, 4)[0]
    header_start = 8
    header_end = header_start + header_len
    header = json.loads(payload[header_start:header_end].decode("utf-8"))
    blob = memoryview(payload)[header_end:]
    arrays: dict[str, np.ndarray] = {}
    for rec in header["arrays"]:
        dtype = np.dtype(rec["dtype"])
        arr_bytes = blob[rec["offset"]: rec["offset"] + rec["nbytes"]]
        arrays[rec["name"]] = np.frombuffer(arr_bytes, dtype=dtype).reshape(rec["shape"])
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
        if dense_rec["kind"] == "int8_symmetric":
            values = torch.from_numpy(arrays[dense_rec["values"]].copy()).to(device).float()
            scale = torch.from_numpy(arrays[dense_rec["scale"]].copy()).to(device).float()
            tensor = values * scale
        else:
            tensor = torch.from_numpy(arrays[dense_rec["values"]].copy()).to(device)
        state_dict[name] = tensor.float() if torch.is_floating_point(tensor) else tensor

    return state_dict

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
    def __init__(self, num_classes=5, emb_dim=6, c1=40, c2=44, depth_mult=4):
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

    def forward(self, mask2: torch.Tensor, coords: torch.Tensor):
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        f = self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))
        return f

class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 36, depth_mult: int = 4):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0

class FrameHead(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int = 32, hidden: int = 36, depth_mult: int = 4):
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

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(
            num_classes=num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult)

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        self.frame1_head = FrameHead(
            in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)

        self.frame2_head = Frame2StaticHead(
            in_ch=56, hidden=52, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)

        shared_feat = self.shared_trunk(mask2, coords)
        pred_frame2 = self.frame2_head(shared_feat)

        cond_emb = self.pose_mlp(pose6)
        pred_frame1 = self.frame1_head(shared_feat, cond_emb)

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
def load_encoded_mask_video(path: str, palette: list[int] | None = None) -> torch.Tensor:
    container = av.open(path)
    frames = []
    palette_arr = None
    if palette is not None:
        palette_arr = np.asarray(palette, dtype=np.float32)
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        if palette_arr is None:
            cls_img = np.round(img / 63.0).astype(np.uint8)
            cls_img = np.clip(cls_img, 0, 4)
        else:
            # Decode arbitrary palette values back to semantic classes.
            dist = np.abs(img.astype(np.float32)[..., None] - palette_arr[None, None, :])
            cls_img = dist.argmin(axis=-1).astype(np.uint8)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()

def load_mask_payload(data_dir: Path, legacy_mask_br: Path) -> torch.Tensor:
    mix_manifest_br = data_dir / MASK_MIX_MANIFEST_NAME
    if not mix_manifest_br.exists():
        with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
            with open(legacy_mask_br, "rb") as f:
                tmp_obu.write(brotli.decompress(f.read()))
            tmp_obu_path = tmp_obu.name
        try:
            return load_encoded_mask_video(tmp_obu_path)
        finally:
            os.remove(tmp_obu_path)

    manifest = json.loads(brotli.decompress(mix_manifest_br.read_bytes()).decode("utf-8"))
    if manifest.get("format") != "quantizr_mask_mix_v1":
        raise ValueError(f"unknown mask mix format: {manifest.get('format')}")
    palette = manifest.get("palette")
    group_masks = []
    for group in manifest["groups"]:
        payload_path = data_dir / group["payload"]
        with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
            tmp_obu.write(brotli.decompress(payload_path.read_bytes()))
            tmp_obu_path = tmp_obu.name
        try:
            decoded = load_encoded_mask_video(tmp_obu_path, palette=palette)
        finally:
            os.remove(tmp_obu_path)
        if decoded.shape[0] != int(group["count"]):
            raise RuntimeError(
                f"mask group {group['name']} decoded {decoded.shape[0]} frames, expected {group['count']}"
            )
        group_masks.append(decoded)

    entries = manifest["entries"]
    frames = []
    for group_id, local_idx in entries:
        frames.append(group_masks[int(group_id)][int(local_idx)])
    mask_frames = torch.stack(frames).contiguous()
    expected = int(manifest["num_frames"])
    if mask_frames.shape[0] != expected:
        raise RuntimeError(f"mixed mask payload reconstructed {mask_frames.shape[0]} frames, expected {expected}")
    return mask_frames

def _unpack_bits(payload: bytes, count: int, bits: int) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(count, dtype=np.uint16)
    acc = 0
    acc_bits = 0
    j = 0
    mask = (1 << bits) - 1
    for byte in data:
        acc |= int(byte) << acc_bits
        acc_bits += 8
        while acc_bits >= bits and j < count:
            out[j] = acc & mask
            acc >>= bits
            acc_bits -= bits
            j += 1
    if j != count:
        raise RuntimeError(f"bit-packed pose payload decoded {j} values, expected {count}")
    return out

def load_pose_payload(data_dir: Path, legacy_pose_br: Path) -> torch.Tensor:
    pose_qpack = data_dir / POSE_QPACK_PAYLOAD_NAME
    if not pose_qpack.exists():
        with open(legacy_pose_br, "rb") as f:
            pose_bytes = brotli.decompress(f.read())
        return torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()

    payload = brotli.decompress(pose_qpack.read_bytes())
    if not payload.startswith(b"PQP1"):
        raise ValueError("Invalid pose qpack payload magic")
    header_len = struct.unpack_from("<I", payload, 4)[0]
    header_start = 8
    header_end = header_start + header_len
    header = json.loads(payload[header_start:header_end].decode("utf-8"))
    body = payload[header_end:]
    shape = tuple(int(x) for x in header["shape"])
    kind = header["kind"]
    if kind == "fp16":
        arr = np.frombuffer(body, dtype=np.dtype("<f2")).reshape(shape).astype(np.float32)
    elif kind in {"uint16_per_dim", "packed_uint_per_dim"}:
        lo = np.asarray(header["min"], dtype=np.float32)
        hi = np.asarray(header["max"], dtype=np.float32)
        levels = float((1 << int(header["bits"])) - 1)
        count = int(np.prod(shape))
        if kind == "uint16_per_dim":
            q = np.frombuffer(body, dtype=np.dtype("<u2")).reshape(shape).astype(np.float32)
        else:
            q = _unpack_bits(body, count, int(header["bits"])).reshape(shape).astype(np.float32)
        arr = q / levels * (hi - lo)[None, :] + lo[None, :]
    else:
        raise ValueError(f"unknown pose qpack kind: {kind}")
    return torch.from_numpy(np.ascontiguousarray(arr)).float()

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
 
    generator = JointFrameGenerator().to(device)

    # 1. Load Weights
    if model_qpack_br.exists():
        with open(model_qpack_br, "rb") as f:
            weights_data = brotli.decompress(f.read())
        decoded_state = get_qpack_state_dict(weights_data, device)
    else:
        with open(model_br, "rb") as f:
            weights_data = brotli.decompress(f.read())
        decoded_state = get_decoded_state_dict(weights_data, device)
    
    generator.load_state_dict(decoded_state, strict=True)
    generator.eval()

    # 2. Load Mask Video (.obu) or mixed-mask payload.
    mask_frames_all = load_mask_payload(data_dir, mask_br)

    # 3. Load Pose Vectors
    pose_frames_all = load_pose_payload(data_dir, pose_br)

    out_h, out_w = 874, 1164
    cursor = 0
    batch_size = 4 
    
    # 1 mask per generated pair, assume 600 pairs per standard 1200 frame chunk.
    pairs_per_file = 600

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"
            
            # Retrieve exactly the pairs mapping to this file
            file_masks = mask_frames_all[cursor : cursor + pairs_per_file]
            file_poses = pose_frames_all[cursor : cursor + pairs_per_file]
            cursor += pairs_per_file
            
            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Decoding {file_name}")
                
                for i in pbar:
                    in_mask2 = file_masks[i : i + batch_size].to(device).long()
                    in_pose6 = file_poses[i : i + batch_size].to(device).float()

                    fake1, fake2 = generator(in_mask2, in_pose6)

                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)

                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")

                    output_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(output_bytes.cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
