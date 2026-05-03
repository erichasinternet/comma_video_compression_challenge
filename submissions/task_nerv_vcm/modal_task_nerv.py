#!/usr/bin/env python3
"""Modal entrypoints for task-aware HNeRV capacity oracles."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-task-nerv-vcm"
REMOTE_REPO = Path("/root/comma_task_nerv_vcm")
REMOTE_OUT_DIR = Path("/root/task_nerv_vcm_volume/experiments")


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
    )
    .add_local_dir(LOCAL_REPO, str(REMOTE_REPO), copy=True, ignore=ignore_repo_path)
)

volume = modal.Volume.from_name("comma-task-nerv-vcm", create_if_missing=True)
app = modal.App(APP_NAME, image=base_image)


def run(cmd: list[str | Path], cwd: Path = REMOTE_REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def load_result(label: str) -> dict:
    out_dir = REMOTE_OUT_DIR / label
    result = json.loads((out_dir / "metrics.json").read_text())
    result["label"] = label
    result["remote_out_dir"] = str(out_dir)
    history = out_dir / "history.jsonl"
    if history.exists():
        result["history_jsonl"] = history.read_text()
    return result


def save_local_result(result: dict) -> Path:
    label = result.get("label", "modal_result")
    local_out = LOCAL_REPO / "submissions/task_nerv_vcm/experiments/modal_returned" / label
    local_out.mkdir(parents=True, exist_ok=True)
    history = result.pop("history_jsonl", None)
    if history is not None:
        (local_out / "history.jsonl").write_text(history)
    (local_out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return local_out


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 8, volumes={"/root/task_nerv_vcm_volume": volume})
def hard8_capacity(
    rgb_steps: int = 2000,
    task_steps: int = 5000,
    batch_size: int = 2,
    eval_every: int = 100,
    hidden: int = 128,
    embed_dim: int = 64,
) -> dict:
    label = f"hard8_hnerv_h{hidden}_e{embed_dim}_rgb{rgb_steps}_task{task_steps}_b{batch_size}"
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/task_nerv_vcm/train_nerv.py",
            "--out-dir",
            out_dir,
            "--preset",
            "hard8",
            "--subset",
            "8",
            "--rgb-steps",
            str(rgb_steps),
            "--task-steps",
            str(task_steps),
            "--batch-size",
            str(batch_size),
            "--eval-every",
            str(eval_every),
            "--hidden",
            str(hidden),
            "--embed-dim",
            str(embed_dim),
            "--device",
            "cuda",
        ]
    )
    volume.commit()
    return load_result(label)


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 12, volumes={"/root/task_nerv_vcm_volume": volume})
def subset64_capacity(
    rgb_steps: int = 3000,
    task_steps: int = 5000,
    batch_size: int = 2,
    eval_every: int = 100,
    hidden: int = 128,
    embed_dim: int = 64,
) -> dict:
    label = f"subset64_hnerv_h{hidden}_e{embed_dim}_rgb{rgb_steps}_task{task_steps}_b{batch_size}"
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/task_nerv_vcm/train_nerv.py",
            "--out-dir",
            out_dir,
            "--preset",
            "sequential",
            "--subset",
            "64",
            "--rgb-steps",
            str(rgb_steps),
            "--task-steps",
            str(task_steps),
            "--batch-size",
            str(batch_size),
            "--eval-every",
            str(eval_every),
            "--hidden",
            str(hidden),
            "--embed-dim",
            str(embed_dim),
            "--device",
            "cuda",
        ]
    )
    volume.commit()
    return load_result(label)


@app.local_entrypoint()
def main(
    stage: str = "hard8",
    rgb_steps: int = 2000,
    task_steps: int = 5000,
    batch_size: int = 2,
    eval_every: int = 100,
    hidden: int = 128,
    embed_dim: int = 64,
) -> None:
    if stage == "hard8":
        result = hard8_capacity.remote(
            rgb_steps=rgb_steps,
            task_steps=task_steps,
            batch_size=batch_size,
            eval_every=eval_every,
            hidden=hidden,
            embed_dim=embed_dim,
        )
    elif stage == "subset64":
        result = subset64_capacity.remote(
            rgb_steps=rgb_steps,
            task_steps=task_steps,
            batch_size=batch_size,
            eval_every=eval_every,
            hidden=hidden,
            embed_dim=embed_dim,
        )
    else:
        raise ValueError(f"unknown stage: {stage}")
    local_out = save_local_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved local result to {local_out}")

