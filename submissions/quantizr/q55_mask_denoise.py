#!/usr/bin/env python
"""Train a tiny no-op-initialized mask denoiser against exact #55 masks."""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from pathlib import Path

import brotli
import numpy as np

from q55_common import (
    MODEL_PAYLOAD,
    MODEL_QPACK_PAYLOAD,
    POSE_PAYLOAD,
    POSE_QPACK_PAYLOAD,
    append_jsonl,
    make_archive_zip,
    materialize_submission,
    summarize_archive,
    unzip_archive,
    write_json,
)
from q55_mask_alloc import decode_archive_masks
from q55_mask_cleaner import MASK_CLEAN_PAYLOAD, compare_masks, write_clean_payload


class MaskDenoiserConfig:
    num_classes = 5
    coord_channels = 2
    boundary_channels = 1
    temporal_class_frames = 3


def choose_payload(archive_dir: Path, primary: str, fallback: str) -> str:
    if (archive_dir / primary).exists():
        return primary
    if (archive_dir / fallback).exists():
        return fallback
    raise FileNotFoundError(f"missing payload: {primary} or {fallback}")


def load_masks_from_archive(archive_zip: Path) -> np.ndarray:
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(archive_zip) as z:
            z.extract("mask.obu.br", root)
        return decode_archive_masks(root / "mask.obu.br")


def load_mask_gray_and_class_from_archive(archive_zip: Path) -> tuple[np.ndarray, np.ndarray]:
    import av
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(archive_zip) as z:
            z.extract("mask.obu.br", root)
        mask_br = root / "mask.obu.br"
        with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
            tmp_obu.write(brotli.decompress(mask_br.read_bytes()))
            tmp_obu_path = tmp_obu.name
        frames = []
        try:
            container = av.open(tmp_obu_path)
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="gray").astype(np.uint8, copy=False))
            container.close()
        finally:
            Path(tmp_obu_path).unlink(missing_ok=True)
    if not frames:
        raise RuntimeError(f"no mask frames decoded from {archive_zip}")
    gray = np.stack(frames).astype(np.uint8, copy=False)
    classes = np.clip(np.round(gray / 63.0).astype(np.uint8), 0, 4)
    return gray, classes


def build_clean_archive(base_archive: Path, archive_dir: Path, archive_zip: Path, cleaned: np.ndarray) -> dict:
    unzip_archive(base_archive, archive_dir)
    model_payload = choose_payload(archive_dir, MODEL_QPACK_PAYLOAD, MODEL_PAYLOAD)
    pose_payload = choose_payload(archive_dir, POSE_QPACK_PAYLOAD, POSE_PAYLOAD)
    for path in archive_dir.glob("mask*.br"):
        path.unlink()
    payload_report = write_clean_payload(archive_dir / MASK_CLEAN_PAYLOAD, cleaned)
    make_archive_zip(archive_dir, archive_zip, [model_payload, pose_payload, MASK_CLEAN_PAYLOAD])
    return {
        "model_payload": model_payload,
        "pose_payload": pose_payload,
        "clean_payload": payload_report,
    }


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def make_coord_grid(torch, batch: int, height: int, width: int, device) -> object:
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def boundary_map(torch, cur) -> object:
    b = torch.zeros_like(cur, dtype=torch.bool)
    b[:, :, 1:] |= cur[:, :, 1:] != cur[:, :, :-1]
    b[:, :, :-1] |= cur[:, :, 1:] != cur[:, :, :-1]
    b[:, 1:, :] |= cur[:, 1:, :] != cur[:, :-1, :]
    b[:, :-1, :] |= cur[:, 1:, :] != cur[:, :-1, :]
    return b.float().unsqueeze(1)


def make_features(torch, F, masks, gray, indices, device) -> tuple[object, object]:
    idx = torch.as_tensor(indices, device=device, dtype=torch.long)
    n = masks.shape[0]
    prev_idx = torch.clamp(idx - 1, min=0)
    next_idx = torch.clamp(idx + 1, max=n - 1)
    prev_m = masks[prev_idx]
    cur_m = masks[idx]
    next_m = masks[next_idx]
    onehots = [
        F.one_hot(x.long(), num_classes=5).permute(0, 3, 1, 2).float()
        for x in (prev_m, cur_m, next_m)
    ]
    coords = make_coord_grid(torch, cur_m.shape[0], cur_m.shape[1], cur_m.shape[2], device)
    gray_features = []
    if gray is not None:
        prev_g = gray[prev_idx].float() / 255.0
        cur_g = gray[idx].float() / 255.0
        next_g = gray[next_idx].float() / 255.0
        center = (cur_m.float() * 63.0).clamp(0, 252)
        center_dist = ((gray[idx].float() - center).abs() / 31.5).clamp(0, 4)
        gray_features = [x.unsqueeze(1) for x in (prev_g, cur_g, next_g, center_dist)]
    return torch.cat([*onehots, boundary_map(torch, cur_m), coords, *gray_features], dim=1), cur_m


def make_model(hidden: int, base_logit: float, use_gray: bool):
    torch, nn, _ = import_torch()

    class MaskDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            in_ch = 5 * 3 + 1 + 2 + (4 if use_gray else 0)
            self.base_logit = float(base_logit)
            self.in_proj = nn.Conv2d(in_ch, hidden, 1)
            self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
            self.out_proj = nn.Conv2d(hidden, 5, 1)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        def forward(self, features, cur_mask):
            base = torch.nn.functional.one_hot(cur_mask.long(), num_classes=5).permute(0, 3, 1, 2).float()
            x = torch.nn.functional.silu(self.in_proj(features))
            x = torch.nn.functional.silu(self.dw(x))
            return base * self.base_logit + self.out_proj(x)

    return MaskDenoiser()


def eval_model(
    model,
    torch,
    F,
    pred_masks,
    pred_gray,
    exact_masks,
    device,
    batch_size: int,
    thresholds: list[float],
) -> tuple[np.ndarray, dict]:
    model.eval()
    cleaned_by_threshold = {float(t): [] for t in thresholds}
    with torch.inference_mode():
        for start in range(0, pred_masks.shape[0], batch_size):
            idxs = list(range(start, min(pred_masks.shape[0], start + batch_size)))
            features, cur = make_features(torch, F, pred_masks, pred_gray, idxs, device)
            logits = model(features, cur)
            proposed = logits.argmax(dim=1)
            current_logits = logits.gather(1, cur.long().unsqueeze(1)).squeeze(1)
            proposed_logits = logits.gather(1, proposed.long().unsqueeze(1)).squeeze(1)
            margin = proposed_logits - current_logits
            for threshold in thresholds:
                apply = (proposed != cur) & (margin >= float(threshold))
                cleaned = cur.clone()
                cleaned[apply] = proposed[apply]
                cleaned_by_threshold[float(threshold)].append(cleaned.to(torch.uint8).cpu().numpy())

    exact_np = exact_masks.cpu().numpy()
    pred_np = pred_masks.cpu().numpy()
    threshold_stats = []
    cleaned_outputs = {}
    for threshold in thresholds:
        key = float(threshold)
        cleaned_np = np.concatenate(cleaned_by_threshold[key], axis=0).astype(np.uint8, copy=False)
        stats = compare_masks(exact_np, pred_np, cleaned_np)
        stats["threshold"] = key
        threshold_stats.append(stats)
        cleaned_outputs[key] = cleaned_np

    best = min(
        threshold_stats,
        key=lambda rec: (rec["errors_after"], rec["new_errors_introduced"], -rec["errors_repaired"]),
    )
    best_cleaned = cleaned_outputs[float(best["threshold"])]
    return best_cleaned, {"best_threshold": best["threshold"], "threshold_stats": threshold_stats, **best}


def save_adapter(path: Path, model, config: dict) -> int:
    torch, _, _ = import_torch()
    buffer = io.BytesIO()
    torch.save({"config": config, "state_dict": model.state_dict()}, buffer)
    path.write_bytes(brotli.compress(buffer.getvalue(), quality=11, lgwin=24))
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-archive", type=Path, required=True)
    parser.add_argument("--predictor-mask-archive", type=Path, required=True)
    parser.add_argument("--base-package-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="qmask_denoise_qrecode50_sanity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--changed-weight", type=float, default=4096.0)
    parser.add_argument("--base-logit", type=float, default=8.0)
    parser.add_argument("--no-gray-features", action="store_true")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--thresholds", default="0,0.5,1,2,3,4,5,6,8,10,12,16,24")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    for path in (args.exact_archive, args.predictor_mask_archive, args.base_package_archive):
        if not path.exists():
            raise FileNotFoundError(path)

    torch, nn, F = import_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    run_dir = args.out_dir / args.label
    archive_dir = run_dir / "archive"
    archive_zip = run_dir / "archive.zip"
    run_dir.mkdir(parents=True, exist_ok=True)

    exact_np = load_masks_from_archive(args.exact_archive)
    pred_gray_np, pred_np = load_mask_gray_and_class_from_archive(args.predictor_mask_archive)
    if exact_np.shape != pred_np.shape:
        raise RuntimeError(f"mask shape mismatch: {exact_np.shape} vs {pred_np.shape}")

    exact = torch.from_numpy(exact_np).to(device=device, dtype=torch.long)
    pred = torch.from_numpy(pred_np).to(device=device, dtype=torch.long)
    pred_gray = None if args.no_gray_features else torch.from_numpy(pred_gray_np).to(device=device, dtype=torch.uint8)
    changed = pred != exact

    use_gray = pred_gray is not None
    model = make_model(args.hidden, args.base_logit, use_gray).to(device)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps))
    latest_path = run_dir / "latest.pt"
    start_step = 0
    history = []
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"]) + 1
        history = ckpt.get("history", [])

    n = exact.shape[0]
    for step in range(start_step, args.steps):
        model.train()
        idxs = torch.randint(0, n, (args.batch_size,), device=device).tolist()
        features, cur = make_features(torch, F, pred, pred_gray, idxs, device)
        target = exact[torch.as_tensor(idxs, device=device)]
        logits = model(features, cur)
        loss_map = F.cross_entropy(logits, target, reduction="none")
        weights = 1.0 + changed[torch.as_tensor(idxs, device=device)].float() * args.changed_weight
        loss = (loss_map * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            cleaned, stats = eval_model(model, torch, F, pred, pred_gray, exact, device, args.eval_batch_size, thresholds)
            stats = {"step": step + 1, "loss": float(loss.item()), **stats}
            history.append(stats)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                },
                latest_path,
            )
            print(json.dumps(stats, sort_keys=True), flush=True)

    cleaned, final_stats = eval_model(model, torch, F, pred, pred_gray, exact, device, args.eval_batch_size, thresholds)
    adapter_bytes = save_adapter(
        run_dir / "mask_denoiser.pt.br",
        model.cpu(),
        {
            "hidden": args.hidden,
            "base_logit": args.base_logit,
            "use_gray": use_gray,
            "input": "onehot_prev_cur_next_boundary_xy_gray" if use_gray else "onehot_prev_cur_next_boundary_xy",
        },
    )
    archive_report = build_clean_archive(args.base_package_archive.resolve(), archive_dir, archive_zip, cleaned)
    materialize_submission(archive_zip=archive_zip, submission_dir=run_dir / "submission", inflate_mode="modified")

    final_record = {
        "label": args.label,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "lr": args.lr,
        "changed_weight": args.changed_weight,
        "base_logit": args.base_logit,
        "use_gray": use_gray,
        "thresholds": thresholds,
        "exact_archive": str(args.exact_archive.resolve()),
        "predictor_mask_archive": str(args.predictor_mask_archive.resolve()),
        "base_package_archive": str(args.base_package_archive.resolve()),
        "exact_archive_summary": summarize_archive(args.exact_archive.resolve()),
        "predictor_archive_summary": summarize_archive(args.predictor_mask_archive.resolve()),
        "base_package_summary": summarize_archive(args.base_package_archive.resolve()),
        "adapter_payload_bytes_br": adapter_bytes,
        "archive_zip": str(archive_zip),
        "archive_summary": summarize_archive(archive_zip),
        "diagnostic_archive_rate_is_not_candidate_rate": True,
        "history": history,
        **archive_report,
        **final_stats,
    }
    write_json(run_dir / "metrics.json", final_record)
    append_jsonl(args.out_dir / "mask_denoise_results.jsonl", final_record)
    latest_path.unlink(missing_ok=True)
    print(json.dumps(final_record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
