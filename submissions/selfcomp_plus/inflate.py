#!/usr/bin/env python3
"""selfcomp++ inflater.

This delegates rendering to the PR #56 selfcomp inflater but replaces the
checkpoint loader so archives may contain `segmap.dcpack.br` or `segmap.dcpack`
instead of `segmap_inference.pt`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from submissions.selfcomp import inflate as base
from unpack_segmap import read_dcpack


def _load_payload(checkpoint_path: Path) -> dict:
    data_dir = checkpoint_path.parent
    for name in ("segmap.dcpack.br", "segmap.dcpack"):
        candidate = data_dir / name
        if candidate.exists():
            return read_dcpack(candidate)
    return torch.load(checkpoint_path, map_location="cpu")


def load_segmap(checkpoint_path: Path, device: torch.device) -> base.SegMap:
    payload = _load_payload(checkpoint_path)
    if payload.get("learned_fullres_residual", False):
        raise ValueError("learned_fullres_residual export is not supported by this submission")
    if payload.get("lowfreq_frame_channel", False):
        raise ValueError("lowfreq_frame_channel export is not supported by this submission")
    state = payload["inference_state_dict"]
    model = base.SegMap(
        hidden=int(payload["hidden"]),
        block_hidden=int(payload.get("block_hidden") or payload["hidden"]),
        num_blocks=int(payload["num_blocks"]),
        max_frame_index=int(payload["max_frame_index"]),
        affine_max_zoom_delta=float(payload.get("affine_max_zoom_delta", 0.12)),
        affine_max_aspect_delta=float(payload.get("affine_max_aspect_delta", 0.03)),
        affine_max_shear=float(payload.get("affine_max_shear", 0.03)),
        affine_max_translation=float(payload.get("affine_max_translation", 0.08)),
        latent_input_scale=float(payload.get("latent_input_scale", 1.0)),
    ).to(device)
    with torch.no_grad():
        model.shared_latent_base.copy_(base.decode_tensor_payload(state["shared_latent_base"]).to(device))
        model.frame_affine_embedding.weight.copy_(
            base.decode_tensor_payload(state["frame_affine_embedding.weight"]).to(device)
        )
        model.layer_in.weight.copy_(base.reconstruct_weight(payload, state, "layer_in").to(device))
        model.layer_in.bias.copy_(base.decode_tensor_payload(state["layer_in.bias"]).to(device))
        model.layer_out.weight.copy_(base.reconstruct_weight(payload, state, "layer_out").to(device))
        model.layer_out.bias.copy_(base.decode_tensor_payload(state["layer_out.bias"]).to(device))
        for block_idx, block in enumerate(model.blocks):
            prefix = f"blocks.{block_idx}.conv1"
            block.conv1.weight.copy_(base.reconstruct_weight(payload, state, prefix).to(device))
            block.conv1.bias.copy_(base.decode_tensor_payload(state[f"{prefix}.bias"]).to(device))
            prefix = f"blocks.{block_idx}.conv2"
            block.conv2.weight.copy_(base.reconstruct_weight(payload, state, prefix).to(device))
            block.conv2.bias.copy_(base.decode_tensor_payload(state[f"{prefix}.bias"]).to(device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


base.load_segmap = load_segmap


if __name__ == "__main__":
    base.inflate_to_raw(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
