#!/usr/bin/env python
"""Local qpose14 artifact decoding helpers for Search VCM v2."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import av
import brotli
import numpy as np
import torch

from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, REPO_ROOT


QPOSE14_ARCHIVE = REPO_ROOT / "submissions/qpose14/archive.zip"
QPOSE14_CACHE_DIR = EXPERIMENTS_DIR / "qpose14_cache"
MASK_BYTES = 219_472
MODEL_BYTES = 66_841


def select_torch_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return select_torch_device("auto")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        return select_torch_device("auto")
    return torch.device(requested)


def split_archive_payload(archive_path: Path = QPOSE14_ARCHIVE) -> tuple[bytes, bytes, bytes]:
    if not archive_path.exists():
        raise FileNotFoundError(f"missing qpose14 archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if "p" in names:
            payload = zf.read("p")
            return payload[:MASK_BYTES], payload[MASK_BYTES : MASK_BYTES + MODEL_BYTES], payload[MASK_BYTES + MODEL_BYTES :]
        mask = zf.read("a") if "a" in names else zf.read("mask.obu.br")
        model = zf.read("b") if "b" in names else zf.read("model.pt.br")
        pose = zf.read("c") if "c" in names else zf.read("poseq.bin.br")
        return mask, model, pose


def decode_mask_stream(mask_br_data: bytes) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br_data))
        tmp_path = Path(tmp.name)
    try:
        container = av.open(str(tmp_path))
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="gray")
            cls_img = np.round(img / 63.0).astype(np.uint8)
            cls_img = np.clip(cls_img, 0, 4)
            frames.append(cls_img)
        container.close()
    finally:
        tmp_path.unlink(missing_ok=True)
    return torch.from_numpy(np.stack(frames)).contiguous().long()


def decode_pose_stream(pose_q_br_data: bytes) -> torch.Tensor:
    q = np.frombuffer(brotli.decompress(pose_q_br_data), dtype=np.uint16).reshape(-1, 6)
    pose_np = np.empty(q.shape, dtype=np.float32)
    pose_np[:, 0] = q[:, 0].astype(np.float32) / 512.0 + 20.0
    pose_np[:, 1:] = q[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return torch.from_numpy(pose_np).float()


def load_qpose14_generator(model_br_data: bytes, device: torch.device) -> torch.nn.Module:
    from submissions.qpose14.inflate import JointFrameGenerator, get_decoded_state_dict

    generator = JointFrameGenerator().to(device)
    weights_data = brotli.decompress(model_br_data)
    generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
    generator.eval()
    return generator


def qpose14_cache_path(subset_name: str) -> Path:
    safe = subset_name.replace("/", "_")
    return QPOSE14_CACHE_DIR / f"{safe}.pt"


def original_subset_cache_path(subset_name: str) -> Path:
    safe = subset_name.replace("/", "_")
    return QPOSE14_CACHE_DIR / f"{safe}_original.pt"


def load_original_subset(
    subset_name: str,
    sample_ids: list[int],
    *,
    device: str = "cpu",
    force: bool = False,
) -> torch.Tensor:
    """Load original frame pairs for sample ids as uint8 [B,2,H,W,3]."""

    cache_path = original_subset_cache_path(subset_name)
    if cache_path.exists() and not force:
        data = torch.load(cache_path, map_location="cpu")
        if [int(x) for x in data.get("sample_ids", [])] == [int(x) for x in sample_ids]:
            return data["frames"]

    from frame_utils import AVVideoDataset

    video_names = ["0.mkv"]
    torch_device = select_torch_device(device)
    if torch_device.type == "cuda":
        torch_device = torch.device("cpu")
    dataset = AVVideoDataset(video_names, data_dir=REPO_ROOT / "videos", batch_size=16, device=torch_device, num_threads=2, seed=1234, prefetch_queue_depth=2)
    dataset.prepare_data()
    wanted = set(int(x) for x in sample_ids)
    found: dict[int, torch.Tensor] = {}
    sample_cursor = 0
    for _, _, batch in dataset:
        for local_idx in range(batch.shape[0]):
            sample_id = sample_cursor + local_idx
            if sample_id in wanted:
                found[sample_id] = batch[local_idx].cpu()
        sample_cursor += batch.shape[0]
        if len(found) == len(wanted):
            break
    missing = [idx for idx in sample_ids if idx not in found]
    if missing:
        raise RuntimeError(f"missing original samples: {missing}")
    frames = torch.stack([found[int(idx)] for idx in sample_ids], dim=0).contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_ids": [int(x) for x in sample_ids], "frames": frames}, cache_path)
    return frames


def materialize_qpose14_subset(
    subset_name: str,
    sample_ids: list[int],
    *,
    device: str = "auto",
    archive_path: Path = QPOSE14_ARCHIVE,
    force: bool = False,
) -> dict:
    """Return exact qpose14 masks, poses, and internal 384x512 teacher frames."""

    cache_path = qpose14_cache_path(subset_name)
    if cache_path.exists() and not force:
        data = torch.load(cache_path, map_location="cpu")
        if [int(x) for x in data.get("sample_ids", [])] == [int(x) for x in sample_ids]:
            return data

    mask_br_data, model_br_data, pose_q_br_data = split_archive_payload(archive_path)
    masks_all = decode_mask_stream(mask_br_data)
    poses_all = decode_pose_stream(pose_q_br_data)
    if masks_all.shape[0] < max(sample_ids) + 1:
        raise RuntimeError(f"qpose mask stream has only {masks_all.shape[0]} masks; need sample {max(sample_ids)}")

    masks = masks_all[sample_ids].contiguous()
    poses = poses_all[sample_ids].contiguous()
    torch_device = select_torch_device(device)
    generator = load_qpose14_generator(model_br_data, torch_device)

    frame1_chunks = []
    frame2_chunks = []
    with torch.inference_mode():
        for i in range(0, len(sample_ids), 1):
            mask = masks[i : i + 1].to(torch_device)
            pose = poses[i : i + 1].to(torch_device)
            frame1, frame2 = generator(mask, pose)
            frame1_chunks.append(frame1.detach().cpu())
            frame2_chunks.append(frame2.detach().cpu())

    data = {
        "sample_ids": [int(x) for x in sample_ids],
        "mask": masks.cpu(),
        "pose6": poses.cpu(),
        "qpose_frame1": torch.cat(frame1_chunks, dim=0).cpu(),
        "qpose_frame2": torch.cat(frame2_chunks, dim=0).cpu(),
        "source_archive": str(archive_path),
        "frame_space": "qpose14_internal_384x512_rgb_0_255",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data
