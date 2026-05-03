#!/usr/bin/env python
"""Frame1 low-frequency DCT residual oracle for the qzs3 candidate.

This tests whether a small, packable frame1-only correction can reduce PoseNet
without touching frame2/SegNet. It optimizes per-sample DCT coefficients against
the frozen PoseNet target and reports actual evaluator quality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frame_utils import AVVideoDataset  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from submissions.search_vcm_v2.evaluator import EXPERIMENTS_DIR, quality, rate_term, write_json  # noqa: E402
from submissions.search_vcm_v2.subsets import get_subset  # noqa: E402


DEFAULT_SUBMISSION = REPO_ROOT / "submissions/qzs3_range_mask_candidate"
OUT_DIR = EXPERIMENTS_DIR / "qzs3_frame1_dct_oracle"


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dct_basis(k: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    freqs: list[tuple[int, int, int]] = []
    for fy in range(16):
        for fx in range(16):
            if fx == 0 and fy == 0:
                continue
            freqs.append((fx, fy, fx * fx + fy * fy))
    freqs.sort(key=lambda item: item[2])
    patterns = []
    for channel in range(3):
        for fx, fy, _ in freqs:
            pat = torch.cos(math.pi * fx * xx) * torch.cos(math.pi * fy * yy)
            chans = torch.zeros(3, h, w, device=device)
            chans[channel] = pat
            patterns.append(chans)
            if len(patterns) >= k:
                basis = torch.stack(patterns, dim=0)
                return basis / basis.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    raise ValueError(f"not enough basis patterns for k={k}")


def rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
    h, w = rgb_chw.shape[-2], rgb_chw.shape[-1]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :, : 2 * h2, : 2 * w2]
    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    u_sub = (u[..., 0::2, 0::2] + u[..., 1::2, 0::2] + u[..., 0::2, 1::2] + u[..., 1::2, 1::2]) * 0.25
    v_sub = (v[..., 0::2, 0::2] + v[..., 1::2, 0::2] + v[..., 0::2, 1::2] + v[..., 1::2, 1::2]) * 0.25
    return torch.stack([y[..., 0::2, 0::2], y[..., 1::2, 0::2], y[..., 0::2, 1::2], y[..., 1::2, 1::2], u_sub, v_sub], dim=-3)


def posenet_preprocess_diff(x_bthwc: torch.Tensor) -> torch.Tensor:
    b, t, *_ = x_bthwc.shape
    x = einops.rearrange(x_bthwc, "b t h w c -> (b t) c h w", b=b, t=t, c=3)
    x = F.interpolate(x, size=(384, 512), mode="bilinear")
    return einops.rearrange(rgb_to_yuv6_diff(x), "(b t) c h w -> b (t c) h w", b=b, t=t, c=6)


def parse_sample_ids(raw: str | None, path: Path | None) -> list[int] | None:
    if raw and path:
        raise ValueError("use only one of --sample-ids or --sample-ids-file")
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    if path:
        text = path.read_text().strip()
        if not text:
            return []
        if text.startswith("["):
            data = json.loads(text)
            return [int(x) for x in data]
        return [int(x.strip()) for x in text.replace("\n", ",").split(",") if x.strip()]
    return None


def load_subset_pairs(submission_dir: Path, subset: str, device: torch.device, sample_ids: list[int] | None = None) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    wanted_ids = sample_ids if sample_ids is not None else get_subset(subset)
    wanted = set(int(x) for x in wanted_ids)
    if not wanted:
        raise ValueError("empty sample id set")
    max_needed = max(wanted)
    raw_path = submission_dir / "inflated" / "0.raw"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing inflated raw: {raw_path}")
    comp_np = torch.from_file(str(raw_path), dtype=torch.uint8, size=1200 * 874 * 1164 * 3)
    comp_np = comp_np.reshape(1200, 874, 1164, 3)

    dataset = AVVideoDataset(["0.mkv"], data_dir=REPO_ROOT / "videos", batch_size=16, device=torch.device("cpu"), num_threads=2, seed=1234, prefetch_queue_depth=2)
    dataset.prepare_data()
    gt_rows = []
    ids_out = []
    cursor = 0
    for _, _, batch in dataset:
        ids_all = list(range(cursor, cursor + batch.shape[0]))
        keep_local = [i for i, sid in enumerate(ids_all) if sid in wanted]
        if keep_local:
            ids_out.extend(ids_all[i] for i in keep_local)
            gt_rows.append(batch[keep_local])
        cursor += batch.shape[0]
        if cursor > max_needed:
            break
    gt = torch.cat(gt_rows, dim=0).float().to(device)
    comp_pairs = []
    for sid in ids_out:
        comp_pairs.append(torch.stack([comp_np[2 * sid], comp_np[2 * sid + 1]], dim=0))
    comp = torch.stack(comp_pairs, dim=0).float().to(device)
    return ids_out, gt, comp


def eval_actual(distortion: DistortionNet, gt: torch.Tensor, pred: torch.Tensor, *, batch_size: int = 16) -> dict:
    with torch.inference_mode():
        pose_chunks = []
        seg_chunks = []
        for start in range(0, gt.shape[0], batch_size):
            pose_dist, seg_dist = distortion.compute_distortion(gt[start : start + batch_size], pred[start : start + batch_size])
            pose_chunks.append(pose_dist.detach().cpu())
            seg_chunks.append(seg_dist.detach().cpu())
    pose = torch.cat(pose_chunks)
    seg = torch.cat(seg_chunks)
    rows = []
    for pose_v, seg_v in zip(pose.tolist(), seg.tolist(), strict=True):
        rows.append({"posenet_dist": float(pose_v), "segnet_dist": float(seg_v), "quality": quality(float(seg_v), float(pose_v))})
    pose_mean = float(pose.mean().item())
    seg_mean = float(seg.mean().item())
    return {
        "posenet_dist": pose_mean,
        "segnet_dist": seg_mean,
        "pose_term": math.sqrt(10.0 * pose_mean),
        "seg_term": 100.0 * seg_mean,
        "quality": quality(seg_mean, pose_mean),
        "max_sample_quality": max(row["quality"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="hard8")
    parser.add_argument("--sample-ids", default=None, help="comma-separated sample ids; overrides --subset")
    parser.add_argument("--sample-ids-file", type=Path, default=None, help="JSON list or comma/newline-separated sample ids; overrides --subset")
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--basis-k", type=int, default=24)
    parser.add_argument("--max-delta", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=0, help="0 means full-batch optimization")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = select_device(args.device)
    sample_ids = parse_sample_ids(args.sample_ids, args.sample_ids_file)
    ids, gt, comp = load_subset_pairs(args.submission_dir, args.subset, device, sample_ids)
    distortion = DistortionNet().eval().to(device)
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in distortion.parameters():
        p.requires_grad_(False)

    basis = make_dct_basis(args.basis_k, 874, 1164, device)
    raw_alpha = torch.zeros(len(ids), args.basis_k, device=device, requires_grad=True)
    opt = torch.optim.AdamW([raw_alpha], lr=args.lr)

    with torch.no_grad():
        target_pose = distortion.posenet(posenet_preprocess_diff(gt))
    base_metrics = eval_actual(distortion, gt, comp, batch_size=args.eval_batch_size)
    best = {"step": 0, "metrics": base_metrics, "alpha": raw_alpha.detach().cpu()}
    history = [{"step": 0, **{k: v for k, v in base_metrics.items() if k != "rows"}}]

    for step in tqdm(range(1, args.steps + 1), desc="frame1 dct optimize"):
        if args.train_batch_size and args.train_batch_size < len(ids):
            batch_idx = torch.randint(0, len(ids), (args.train_batch_size,), device=device)
        else:
            batch_idx = torch.arange(len(ids), device=device)
        alpha = args.max_delta * torch.tanh(raw_alpha)
        delta = torch.einsum("bk,kchw->bchw", alpha[batch_idx], basis)
        frame1 = (comp[batch_idx, 0] + einops.rearrange(delta, "b c h w -> b h w c")).clamp(0.0, 255.0)
        pred = torch.stack([frame1, comp[batch_idx, 1]], dim=1)
        pose_out = distortion.posenet(posenet_preprocess_diff(pred))
        pose_loss = (pose_out["pose"][..., :6] - target_pose["pose"][batch_idx, ..., :6]).pow(2).mean()
        loss = pose_loss + args.l2 * alpha[batch_idx].pow(2).mean() + args.l1 * alpha[batch_idx].abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            full_delta = torch.einsum("bk,kchw->bchw", alpha, basis)
            full_frame1 = (comp[:, 0] + einops.rearrange(full_delta, "b c h w -> b h w c")).clamp(0.0, 255.0)
            full_pred = torch.stack([full_frame1, comp[:, 1]], dim=1)
            metrics = eval_actual(distortion, gt, full_pred.detach(), batch_size=args.eval_batch_size)
            rec = {"step": step, "loss": float(loss.detach().cpu().item()), **{k: v for k, v in metrics.items() if k != "rows"}}
            history.append(rec)
            if metrics["quality"] < best["metrics"]["quality"]:
                best = {"step": step, "metrics": metrics, "alpha": alpha.detach().cpu()}

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "subset": args.subset,
        "device": str(device),
        "sample_ids": ids,
        "basis_k": args.basis_k,
        "max_delta": args.max_delta,
        "steps": args.steps,
        "lr": args.lr,
        "l2": args.l2,
        "l1": args.l1,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "archive_bytes": int((args.submission_dir / "archive.zip").stat().st_size),
        "base": base_metrics,
        "best_step": int(best["step"]),
        "best": best["metrics"],
        "quality_delta_vs_base": float(best["metrics"]["quality"] - base_metrics["quality"]),
        "score_delta_vs_base": float(best["metrics"]["quality"] - base_metrics["quality"]),
        "history": history,
        "estimated_alpha_int8_bytes_raw": int(len(ids) * args.basis_k),
    }
    write_json(args.out / f"{args.subset}_k{args.basis_k}_summary.json", summary)
    torch.save({"sample_ids": ids, "alpha": best["alpha"], "basis_k": args.basis_k, "max_delta": args.max_delta}, args.out / f"{args.subset}_k{args.basis_k}_best.pt")
    print(json.dumps({k: summary[k] for k in ("subset", "device", "basis_k", "base", "best_step", "best", "quality_delta_vs_base", "estimated_alpha_int8_bytes_raw")}, indent=2))


if __name__ == "__main__":
    main()
