#!/usr/bin/env python
"""Evaluate mask-tree payload substitutions against the unchanged inflater/model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT_DIR))

import compress as q  # noqa: E402


BASELINE_SCORE = 0.4041945
BASELINE_SEG = 0.00099684
BASELINE_POSE = 0.00107137
ORIGINAL_BYTES = 37_545_489


def parse_candidate_spec(spec: str) -> tuple[str, str, dict[str, str]]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError("candidate must look like NAME:copy or NAME:masktree,path=DIR")
    name, rest = spec.split(":", 1)
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    if not name or not parts:
        raise argparse.ArgumentTypeError(f"invalid candidate spec: {spec}")
    mode = parts[0]
    opts = {}
    for part in parts[1:]:
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"candidate option lacks '=': {part}")
        key, value = part.split("=", 1)
        opts[key.strip()] = value.strip()
    return name, mode, opts


def unzip_archive(archive_zip: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        zf.extractall(out_dir)
    return out_dir


def copy_base_payloads(base_archive_dir: Path, out_archive_dir: Path):
    out_archive_dir.mkdir(parents=True, exist_ok=True)
    names = [
        q.MODEL_PAYLOAD_NAME,
        q.MODEL_QPACK_PAYLOAD_NAME,
        q.MASK_PAYLOAD_NAME,
        *q.MASK_TREE_PAYLOAD_NAMES,
        q.POSE_PAYLOAD_NAME,
        q.LATENT_PAYLOAD_NAME,
    ]
    for name in names:
        src = base_archive_dir / name
        if src.exists():
            shutil.copy2(src, out_archive_dir / name)
    if not ((out_archive_dir / q.MODEL_PAYLOAD_NAME).exists() or (out_archive_dir / q.MODEL_QPACK_PAYLOAD_NAME).exists()):
        raise FileNotFoundError(f"missing model payload in {base_archive_dir}")
    if not (out_archive_dir / q.POSE_PAYLOAD_NAME).exists():
        raise FileNotFoundError(f"missing pose payload in {base_archive_dir}")


def clear_mask_payloads(out_archive_dir: Path):
    for name in (q.MASK_PAYLOAD_NAME, *q.MASK_TREE_PAYLOAD_NAMES):
        path = out_archive_dir / name
        if path.exists():
            path.unlink()


def copy_masktree_payloads(masktree_dir: Path, out_archive_dir: Path):
    clear_mask_payloads(out_archive_dir)
    for name in q.MASK_TREE_PAYLOAD_NAMES:
        src = masktree_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing mask-tree payload: {src}")
        shutil.copy2(src, out_archive_dir / name)


def copy_qpack(qpack_path: Path, out_archive_dir: Path):
    if qpack_path.is_dir():
        qpack_path = qpack_path / q.MODEL_QPACK_PAYLOAD_NAME
    if not qpack_path.exists():
        raise FileNotFoundError(f"missing qpack payload: {qpack_path}")
    stale = out_archive_dir / q.MODEL_PAYLOAD_NAME
    if stale.exists():
        stale.unlink()
    shutil.copy2(qpack_path, out_archive_dir / q.MODEL_QPACK_PAYLOAD_NAME)


def payload_sizes(archive_dir: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in sorted(archive_dir.glob("*")) if p.is_file()}


def package_variant(out_archive_dir: Path, archive_zip: Path) -> int:
    return q.package_submission_archive(
        out_archive_dir,
        archive_zip,
        include_latents=(out_archive_dir / q.LATENT_PAYLOAD_NAME).exists(),
        include_mask=True,
        mask_payload_kind="auto",
    )


def quality_term(segnet_dist: float, posenet_dist: float) -> float:
    return (100.0 * segnet_dist) + ((10.0 * posenet_dist) ** 0.5)


def enforce_m0(
    row: dict[str, object],
    *,
    baseline_score: float,
    baseline_seg: float,
    baseline_pose: float,
    score_tol: float,
    dist_tol: float,
):
    score = float(row["score"])
    seg = float(row["segnet_dist"])
    pose = float(row["posenet_dist"])
    failures = []
    if abs(score - baseline_score) > score_tol:
        failures.append(f"score {score:.7f} differs from {baseline_score:.7f}")
    if abs(seg - baseline_seg) > dist_tol:
        failures.append(f"seg {seg:.8f} differs from {baseline_seg:.8f}")
    if abs(pose - baseline_pose) > dist_tol:
        failures.append(f"pose {pose:.8f} differs from {baseline_pose:.8f}")
    if failures:
        raise RuntimeError("M0 invariant failed: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-archive-dir", type=Path)
    base.add_argument("--base-archive-zip", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--enforce-m0", action="store_true")
    parser.add_argument("--baseline-score", type=float, default=BASELINE_SCORE)
    parser.add_argument("--baseline-segnet-dist", type=float, default=BASELINE_SEG)
    parser.add_argument("--baseline-posenet-dist", type=float, default=BASELINE_POSE)
    parser.add_argument("--baseline-score-tol", type=float, default=0.003)
    parser.add_argument("--baseline-dist-tol", type=float, default=0.00005)
    parser.add_argument("--projected-model-bytes", type=int, default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="NAME:copy, NAME:masktree,path=DIR, NAME:qpack,path=FILE_OR_DIR, or NAME:masktree_qpack,masktree=DIR,qpack=FILE_OR_DIR",
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.base_archive_zip is not None:
        base_archive_dir = unzip_archive(args.base_archive_zip, args.out_root / "_base_archive")
    else:
        base_archive_dir = args.base_archive_dir
    if base_archive_dir is None or not base_archive_dir.exists():
        raise FileNotFoundError(f"base archive dir not found: {base_archive_dir}")

    specs = [parse_candidate_spec(s) for s in (args.candidate or ["M0:copy"])]
    results_path = args.out_root / "mask_codec_oracle_results.jsonl"
    with results_path.open("a") as results_file:
        for name, mode, opts in specs:
            variant_root = args.out_root / name
            out_archive_dir = variant_root / "archive"
            archive_zip = variant_root / "archive.zip"
            if out_archive_dir.exists():
                shutil.rmtree(out_archive_dir)
            copy_base_payloads(base_archive_dir, out_archive_dir)

            if mode == "copy":
                pass
            elif mode == "masktree":
                copy_masktree_payloads(Path(opts["path"]), out_archive_dir)
            elif mode == "qpack":
                copy_qpack(Path(opts["path"]), out_archive_dir)
            elif mode == "masktree_qpack":
                copy_masktree_payloads(Path(opts["masktree"]), out_archive_dir)
                copy_qpack(Path(opts["qpack"]), out_archive_dir)
            else:
                raise ValueError(f"unknown candidate mode: {mode}")

            archive_bytes = package_variant(out_archive_dir, archive_zip)
            sizes = payload_sizes(out_archive_dir)
            row: dict[str, object] = {
                "name": name,
                "mode": mode,
                "options": opts,
                "archive_zip": str(archive_zip),
                "archive_bytes": archive_bytes,
                "payload_sizes": sizes,
                "projected_rate_term": 25.0 * archive_bytes / ORIGINAL_BYTES,
            }

            if args.projected_model_bytes is not None:
                current_model = sizes.get(q.MODEL_QPACK_PAYLOAD_NAME, sizes.get(q.MODEL_PAYLOAD_NAME, 0))
                projected_bytes = archive_bytes - current_model + args.projected_model_bytes
                row["projected_archive_bytes_with_model_target"] = projected_bytes
                row["projected_rate_term_with_model_target"] = 25.0 * projected_bytes / ORIGINAL_BYTES

            if args.run_official:
                metrics = q.run_official_evaluation(
                    submission_dir=THIS_DIR,
                    archive_zip_path=archive_zip,
                    video_names_file=args.video_names,
                    eval_device=args.eval_device,
                )
                row.update(metrics)
                row["quality_term"] = quality_term(metrics["segnet_dist"], metrics["posenet_dist"])
                if args.projected_model_bytes is not None:
                    row["projected_score_with_model_target"] = row["quality_term"] + row["projected_rate_term_with_model_target"]
                if args.enforce_m0 and name == "M0":
                    enforce_m0(
                        row,
                        baseline_score=args.baseline_score,
                        baseline_seg=args.baseline_segnet_dist,
                        baseline_pose=args.baseline_posenet_dist,
                        score_tol=args.baseline_score_tol,
                        dist_tol=args.baseline_dist_tol,
                    )
                print(
                    f"[{name}] score={metrics['score']:.5f} quality={row['quality_term']:.5f} "
                    f"seg={metrics['segnet_dist']:.8f} pose={metrics['posenet_dist']:.8f} bytes={archive_bytes}"
                )
            else:
                print(f"[{name}] bytes={archive_bytes} payloads={sizes}")

            print(json.dumps(row, sort_keys=True), file=results_file, flush=True)

    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
