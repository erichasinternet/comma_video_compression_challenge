#!/usr/bin/env python3
"""Modal entrypoint for Search VCM v2 Gate 1 runs."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-search-vcm-v2"
REMOTE_REPO = Path("/root/comma_search_vcm_v2")
REMOTE_VOLUME_ROOT = Path("/root/search_vcm_v2_volume")
REMOTE_OUT_DIR = REMOTE_VOLUME_ROOT / "experiments"


def infer_local_repo() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


LOCAL_REPO = infer_local_repo()


def ignore_repo_path(path: Path) -> bool:
    parts = set(path.parts)
    if ".git" in parts or any(part.startswith(".venv") for part in parts):
        return True
    if "__pycache__" in parts or path.name.endswith(".pyc"):
        return True
    if "runpod_saved" in parts or "inflated" in parts:
        return True
    if path.suffix == ".raw":
        return True
    if "experiments" in parts:
        keep_v2_seed = (
            "search_vcm_v2" in parts
            and (
                path.name in {"qpose14_per_sample.jsonl", "qpose14_summary.json"}
                or "qpose14_cache" in parts
                or "lowmask_qpose" in parts
            )
        )
        return not keep_v2_seed
    return False


base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git", "zip", "unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy",
        "einops",
        "timm",
        "safetensors",
        "segmentation-models-pytorch",
        "tqdm",
        "pillow",
        "av",
        "brotli",
        "pyyaml",
        "nvidia-dali-cuda120",
    )
    .add_local_dir(LOCAL_REPO, str(REMOTE_REPO), copy=True, ignore=ignore_repo_path)
)

volume = modal.Volume.from_name("comma-search-vcm-v2", create_if_missing=True)
app = modal.App(APP_NAME, image=base_image)


def run(cmd: list[str | Path], cwd: Path = REMOTE_REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def mount_experiments_dir() -> None:
    REMOTE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    repo_exp = REMOTE_REPO / "submissions/search_vcm_v2/experiments"

    # Seed qpose baseline/cache from the image into the persistent volume.
    seed_files = ["qpose14_per_sample.jsonl", "qpose14_summary.json"]
    for name in seed_files:
        src = repo_exp / name
        dst = REMOTE_OUT_DIR / name
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    src_cache = repo_exp / "qpose14_cache"
    dst_cache = REMOTE_OUT_DIR / "qpose14_cache"
    if src_cache.exists() and not dst_cache.exists():
        shutil.copytree(src_cache, dst_cache)
    src_lowmask = repo_exp / "lowmask_qpose"
    dst_lowmask = REMOTE_OUT_DIR / "lowmask_qpose"
    if src_lowmask.exists() and not dst_lowmask.exists():
        shutil.copytree(src_lowmask, dst_lowmask)

    repo_exp.parent.mkdir(parents=True, exist_ok=True)
    if repo_exp.is_symlink():
        repo_exp.unlink()
    elif repo_exp.exists():
        shutil.rmtree(repo_exp)
    repo_exp.symlink_to(REMOTE_OUT_DIR, target_is_directory=True)


def load_remote_outputs(run_id: str) -> dict:
    report_path = REMOTE_OUT_DIR / "reports" / f"{run_id}.json"
    summary_path = REMOTE_OUT_DIR / "qpose14_summary.json"
    lowmask_summary_path = REMOTE_OUT_DIR / "lowmask_qpose" / "fp4_mask_gen_summary.json"
    runs_path = REMOTE_OUT_DIR / "runs.jsonl"
    checkpoints = []
    for ckpt_root in (REMOTE_OUT_DIR / "factorized_checkpoints", REMOTE_OUT_DIR / "lowmask_checkpoints"):
        if ckpt_root.exists():
            checkpoints.extend(str(path) for path in ckpt_root.glob(f"{run_id}_*/best_*.pt"))
    checkpoints = sorted(checkpoints)
    return {
        "run_id": run_id,
        "remote_out_dir": str(REMOTE_OUT_DIR),
        "report": json.loads(report_path.read_text()) if report_path.exists() else None,
        "qpose14_summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
        "lowmask_summary": json.loads(lowmask_summary_path.read_text()) if lowmask_summary_path.exists() else None,
        "runs_jsonl_tail": runs_path.read_text().splitlines()[-20:] if runs_path.exists() else [],
        "checkpoints": checkpoints,
    }


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 8, volumes={str(REMOTE_VOLUME_ROOT): volume})
def search_run(family: str = "factorized_exactmask_pose_tokens", round_name: str = "hard8_capacity", steps: int = 5000, candidates: str = "") -> dict:
    mount_experiments_dir()
    safe_family = family.replace("_", "-")
    safe_round = round_name.replace("_", "-")
    run_id = f"v2_{safe_family}_{safe_round}_cuda_s{steps}"
    cmd = [
        "python",
        "submissions/search_vcm_v2/asha.py",
        "run",
        "--families",
        family,
        "--round",
        round_name,
        "--device",
        "cuda",
        "--max-steps",
        str(steps),
        "--run-id",
        run_id,
    ]
    if candidates:
        cmd.extend(["--candidates", candidates])
    run(cmd)
    volume.commit()
    return load_remote_outputs(run_id)


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 4, volumes={str(REMOTE_VOLUME_ROOT): volume})
def eval_submission(
    submission_dir: str = "submissions/qzs3_range_mask_candidate",
    batch_size: int = 16,
    num_threads: int = 2,
    prefetch_queue_depth: int = 4,
) -> dict:
    mount_experiments_dir()
    subdir = REMOTE_REPO / submission_dir
    run_id = "eval_" + submission_dir.replace("/", "_")
    archive_dir = subdir / "archive"
    inflated_dir = subdir / "inflated"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    if inflated_dir.exists():
        shutil.rmtree(inflated_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    run(["unzip", "-q", str(subdir / "archive.zip"), "-d", str(archive_dir)])
    run(["python", str(subdir / "inflate.py"), str(archive_dir), str(inflated_dir), "public_test_video_names.txt"])
    report_path = REMOTE_OUT_DIR / f"{run_id}_cuda_report.txt"
    attempts = [
        (batch_size, num_threads, prefetch_queue_depth),
        (8, 1, 1),
        (4, 1, 1),
        (2, 1, 1),
    ]
    seen_attempts = set()
    errors = []
    for bs, threads, prefetch in attempts:
        key = (bs, threads, prefetch)
        if key in seen_attempts:
            continue
        seen_attempts.add(key)
        try:
            run(
                [
                    "python",
                    "evaluate.py",
                    "--submission-dir",
                    str(subdir),
                    "--device",
                    "cuda",
                    "--batch-size",
                    str(bs),
                    "--num-threads",
                    str(threads),
                    "--prefetch-queue-depth",
                    str(prefetch),
                    "--report",
                    str(report_path),
                ]
            )
            break
        except subprocess.CalledProcessError as exc:
            errors.append({"batch_size": bs, "num_threads": threads, "prefetch_queue_depth": prefetch, "returncode": exc.returncode})
            if report_path.exists():
                report_path.unlink()
    else:
        raise RuntimeError(f"CUDA evaluation failed for all DALI retry configs: {errors}")
    volume.commit()
    return {
        "run_id": run_id,
        "submission_dir": submission_dir,
        "archive_bytes": (subdir / "archive.zip").stat().st_size,
        "errors": errors,
        "report": report_path.read_text() if report_path.exists() else "",
    }


def save_local_result(result: dict) -> Path:
    run_id = result["run_id"]
    local_out = LOCAL_REPO / "submissions/search_vcm_v2/experiments/modal_returned" / run_id
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "modal_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = result.get("report")
    if report is not None:
        (local_out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return local_out


@app.local_entrypoint()
def main(
    steps: int = 5000,
    candidates: str = "",
    family: str = "factorized_exactmask_pose_tokens",
    round_name: str = "hard8_capacity",
    eval_submission_dir: str = "",
    eval_batch_size: int = 16,
    eval_num_threads: int = 2,
    eval_prefetch_queue_depth: int = 4,
) -> None:
    if eval_submission_dir:
        result = eval_submission.remote(
            submission_dir=eval_submission_dir,
            batch_size=eval_batch_size,
            num_threads=eval_num_threads,
            prefetch_queue_depth=eval_prefetch_queue_depth,
        )
    else:
        result = search_run.remote(family=family, round_name=round_name, steps=steps, candidates=candidates)
    local_out = save_local_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved local result to {local_out}")
