#!/usr/bin/env python3
"""Modal entrypoint for teacher-distilled inflation Gate 1 sweeps."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-search-vcm-teacher"
REMOTE_REPO = Path("/root/comma_search_vcm")
REMOTE_VOLUME_ROOT = Path("/root/search_vcm_volume")
REMOTE_OUT_DIR = REMOTE_VOLUME_ROOT / "experiments" / "search_vcm"


def infer_local_repo() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


LOCAL_REPO = infer_local_repo()


def ignore_repo_path(path: Path) -> bool:
    parts = set(path.parts)
    return (
        ".git" in parts
        or any(part.startswith(".venv") for part in path.parts)
        or "__pycache__" in parts
        or "runpod_saved" in parts
        or "inflated" in parts
        or path.name.endswith(".pyc")
        or ("experiments" in parts and path.is_file())
    )


base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "zip", "unzip", "libgl1", "libglib2.0-0")
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
        "opencv-python-headless",
        "pyyaml",
    )
    .add_local_dir(LOCAL_REPO, str(REMOTE_REPO), copy=True, ignore=ignore_repo_path)
)

volume = modal.Volume.from_name("comma-search-vcm-teacher", create_if_missing=True)
app = modal.App(APP_NAME, image=base_image)


def run(cmd: list[str | Path], cwd: Path = REMOTE_REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def mount_experiments_dir() -> None:
    REMOTE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    experiments_dir = REMOTE_REPO / "submissions/search_vcm/experiments/search_vcm"
    experiments_dir.parent.mkdir(parents=True, exist_ok=True)
    if experiments_dir.is_symlink():
        experiments_dir.unlink()
    elif experiments_dir.exists():
        shutil.rmtree(experiments_dir)
    experiments_dir.symlink_to(REMOTE_OUT_DIR, target_is_directory=True)


def load_remote_outputs(run_id: str) -> dict:
    report_path = REMOTE_OUT_DIR / "reports" / f"{run_id}.json"
    result_dir = REMOTE_OUT_DIR / "teacher_distilled_inflation"
    result_files = sorted(result_dir.glob(f"{run_id}_*_teacher_result.json")) if result_dir.exists() else []
    artifact_files = sorted(result_dir.glob(f"{run_id}_*_teacher_frames.pt")) if result_dir.exists() else []
    return {
        "run_id": run_id,
        "remote_out_dir": str(REMOTE_OUT_DIR),
        "report": json.loads(report_path.read_text()) if report_path.exists() else None,
        "teacher_results": [json.loads(path.read_text()) for path in result_files],
        "teacher_frame_artifacts": [str(path) for path in artifact_files],
        "runs_jsonl_tail": (REMOTE_OUT_DIR / "runs.jsonl").read_text().splitlines()[-20:]
        if (REMOTE_OUT_DIR / "runs.jsonl").exists()
        else [],
    }


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 10, volumes={str(REMOTE_VOLUME_ROOT): volume})
def gate1_hard8_sweep(
    steps: int = 1500,
    batch_size: int = 2,
    eval_every: int = 50,
    candidates: str = "",
) -> dict:
    mount_experiments_dir()
    run_id = f"teacher_gate1_hard8_gpu_s{steps}_b{batch_size}_e{eval_every}"
    cmd = [
        "python",
        "submissions/search_vcm/asha.py",
        "run",
        "--families",
        "teacher_distilled_inflation",
        "--round",
        "hard8",
        "--run-id",
        run_id,
        "--max-steps",
        str(steps),
        "--device",
        "cuda",
        "--teacher-batch-size",
        str(batch_size),
        "--teacher-eval-every",
        str(eval_every),
    ]
    if candidates:
        cmd.extend(["--candidates", candidates])
    run(cmd)
    volume.commit()
    return load_remote_outputs(run_id)


def save_local_result(result: dict) -> Path:
    run_id = result["run_id"]
    local_out = LOCAL_REPO / "submissions/search_vcm/experiments/search_vcm/modal_returned" / run_id
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "modal_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return local_out


@app.local_entrypoint()
def main(
    stage: str = "gate1_hard8",
    steps: int = 1500,
    batch_size: int = 2,
    eval_every: int = 50,
    candidates: str = "",
) -> None:
    if stage != "gate1_hard8":
        raise ValueError(f"unknown stage: {stage}")
    result = gate1_hard8_sweep.remote(
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        candidates=candidates,
    )
    local_out = save_local_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved local result to {local_out}")
