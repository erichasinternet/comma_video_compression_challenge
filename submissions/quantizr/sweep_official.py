#!/usr/bin/env python3
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMPRESS_PY = HERE / "compress.py"


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def candidate_name(crf: int, c1: int, c2: int, hidden: int, cond_dim: int, z_dim: int) -> str:
    return f"crf{crf}_c1{c1}_c2{c2}_h{hidden}_cond{cond_dim}_z{z_dim}"


def main():
    parser = argparse.ArgumentParser(description="Run official-score sweeps for the Quantizr latent variant.")
    parser.add_argument("--archive-root", type=Path, default=HERE / "experiments")
    parser.add_argument("--crf-values", type=parse_csv_ints, default=parse_csv_ints("50,54,58"))
    parser.add_argument("--c1-values", type=parse_csv_ints, default=parse_csv_ints("44,48,52,56,60,64"))
    parser.add_argument("--c2-offset-values", type=parse_csv_ints, default=parse_csv_ints("4,8,12"))
    parser.add_argument("--hidden-values", type=parse_csv_ints, default=parse_csv_ints("40,44,48,52,56,60"))
    parser.add_argument("--cond-dim-values", type=parse_csv_ints, default=parse_csv_ints("32,40,48,64"))
    parser.add_argument("--z-dim-values", type=parse_csv_ints, default=parse_csv_ints("0,4,8,16"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--official-eval-device", type=str, default="cpu")
    parser.add_argument("--video-dir", type=Path, default=HERE.parent.parent / "videos")
    parser.add_argument("--video-names", type=Path, default=HERE.parent.parent / "public_test_video_names.txt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shared-cache-root", type=Path, default=None)
    parser.add_argument("--decode-backend", choices=["dali", "av"], default="av")
    parser.add_argument("--pipeline-preset", choices=["full", "fast", "smoke"], default="fast")
    parser.add_argument("--selection-metric", choices=["official", "proxy"], default="proxy")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--eval-tail", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of candidates to run.")
    parser.add_argument("--rerun", action="store_true", help="Re-run candidates even if final_metrics.json already exists.")
    args = parser.parse_args()

    args.archive_root.mkdir(parents=True, exist_ok=True)
    if args.shared_cache_root is None:
        args.shared_cache_root = args.archive_root / "_shared_cache"
    args.shared_cache_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.archive_root / "results.jsonl"

    candidates = []
    for crf, c1, c2_offset, hidden, cond_dim, z_dim in itertools.product(
        args.crf_values,
        args.c1_values,
        args.c2_offset_values,
        args.hidden_values,
        args.cond_dim_values,
        args.z_dim_values,
    ):
        c2 = c1 + c2_offset
        candidates.append((crf, c1, c2, hidden, cond_dim, z_dim))

    if args.limit > 0:
        candidates = candidates[: args.limit]

    all_results = []
    for crf, c1, c2, hidden, cond_dim, z_dim in candidates:
        name = candidate_name(crf, c1, c2, hidden, cond_dim, z_dim)
        exp_dir = args.archive_root / name
        archive_dir = exp_dir / "archive"
        archive_zip = exp_dir / "archive.zip"
        final_metrics_path = archive_dir / "final_metrics.json"
        exp_dir.mkdir(parents=True, exist_ok=True)

        if final_metrics_path.exists() and not args.rerun:
            metrics = json.loads(final_metrics_path.read_text())
            metrics["name"] = name
            all_results.append(metrics)
            print(f"[skip] {name}: {metrics['score']:.5f}")
            continue

        cmd = [
            sys.executable,
            str(COMPRESS_PY),
            "--video-dir",
            str(args.video_dir),
            "--video-names",
            str(args.video_names),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--shared-cache-root",
            str(args.shared_cache_root),
            "--decode-backend",
            args.decode_backend,
            "--pipeline-preset",
            args.pipeline_preset,
            "--selection-metric",
            args.selection_metric,
            "--eval-interval",
            str(args.eval_interval),
            "--eval-tail",
            str(args.eval_tail),
            "--archive-dir",
            str(archive_dir),
            "--output-archive-zip",
            str(archive_zip),
            "--crf",
            str(crf),
            "--c1",
            str(c1),
            "--c2",
            str(c2),
            "--hidden",
            str(hidden),
            "--cond-dim",
            str(cond_dim),
            "--z-dim",
            str(z_dim),
        ]
        if args.official_eval_device:
            cmd.extend(["--official-eval-device", args.official_eval_device])

        print(f"[run] {name}")
        subprocess.run(cmd, check=True)

        metrics = json.loads(final_metrics_path.read_text())
        metrics["name"] = name
        all_results.append(metrics)
        with open(summary_path, "a") as f_out:
            f_out.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(f"[done] {name}: {metrics['score']:.5f}")

    if not all_results:
        return

    all_results.sort(key=lambda item: item["score"])
    best = all_results[0]
    print(
        "best=%s score=%.5f seg=%.8f pose=%.8f rate=%.8f"
        % (
            best["name"],
            best["score"],
            best["segnet_dist"],
            best["posenet_dist"],
            best["rate"],
        )
    )


if __name__ == "__main__":
    main()
