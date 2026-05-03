#!/usr/bin/env python3
"""Modal entrypoints for selfcomp++ capacity oracles."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-selfcomp-plus"
REMOTE_REPO = Path("/root/comma_selfcomp_plus")
REMOTE_ARCHIVE = REMOTE_REPO / "submissions/selfcomp_plus/experiments/repack_pr56/archive.zip"
REMOTE_OUT_DIR = Path("/root/selfcomp_plus_volume/experiments")


def infer_local_repo() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


LOCAL_REPO = infer_local_repo()


def ignore_repo_path(path: Path) -> bool:
    parts = set(path.parts)
    path_text = path.as_posix()
    name = path.name
    allowed_experiment_file = (
        path_text.endswith("submissions/selfcomp/experiments/source/selfcomp_pr56_archive.zip")
        or path_text.endswith("submissions/selfcomp_plus/experiments/repack_pr56/archive.zip")
        or path_text.endswith("submissions/selfcomp_plus/experiments/repack_pr56/segmap.dcpack.br")
        or path_text.endswith("submissions/selfcomp_plus/experiments/repack_pr56/pack_metrics.json")
        or path_text.endswith("submissions/selfcomp_plus/experiments/repack_pr56/repack_metrics.json")
    )
    return (
        ".git" in parts
        or any(part.startswith(".venv") for part in path.parts)
        or "__pycache__" in parts
        or "runpod_saved" in parts
        or "selfcomp_pr56_eval" in parts
        or "inflated" in parts
        or name.endswith(".pyc")
        or (
            "experiments" in parts
            and path.is_file()
            and not allowed_experiment_file
            and not path_text.startswith("submissions/selfcomp_plus/experiments/modal_returned/")
        )
    )


GPU_POOL = ["L4", "A10"]

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
        "charset-normalizer",
        "requests",
        "urllib3",
        "brotli",
    )
    .add_local_dir(LOCAL_REPO, str(REMOTE_REPO), copy=True, ignore=ignore_repo_path)
)

volume = modal.Volume.from_name("comma-selfcomp-plus", create_if_missing=True)
app = modal.App(APP_NAME, image=base_image)


def run(cmd: list[str | Path], cwd: Path = REMOTE_REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def load_result(label: str) -> dict:
    out_dir = REMOTE_OUT_DIR / label
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    result = json.loads(metrics_path.read_text())
    result["label"] = label
    result["remote_out_dir"] = str(out_dir)
    if history_path.exists():
        result["history_jsonl"] = history_path.read_text()
    return result


def run_oracle(
    *,
    label: str,
    mode: str,
    subset: int,
    steps: int,
    batch_size: int,
    eval_every: int,
    checkpoint_every: int,
    pose_weight: float,
    lr_latent: float,
    lr_model: float,
    lr_affine: float,
    lr_shared: float,
    grad_clip: float,
    apply_gates: bool,
) -> dict:
    if not REMOTE_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_ARCHIVE)
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/selfcomp_plus/selfcomp_train_seg_oracle.py",
            "--archive",
            REMOTE_ARCHIVE,
            "--out-dir",
            out_dir,
            "--mode",
            mode,
            "--subset",
            str(subset),
            "--steps",
            str(steps),
            "--eval-every",
            str(eval_every),
            "--checkpoint-every",
            str(checkpoint_every),
            "--batch-size",
            str(batch_size),
            "--device",
            "cuda",
            "--pose-weight",
            str(pose_weight),
            "--lr-latent",
            str(lr_latent),
            "--lr-model",
            str(lr_model),
            "--lr-affine",
            str(lr_affine),
            "--lr-shared",
            str(lr_shared),
            "--grad-clip",
            str(grad_clip),
        ]
        + (["--apply-gates"] if apply_gates else [])
    )
    volume.commit()
    return load_result(label)


def run_pose_lock(
    *,
    label: str,
    variant: str,
    subset: int,
    steps: int,
    lr: float,
    pose_eps: float,
    eval_every: int,
    train_batch_size: int,
    metric_batch_size: int,
) -> dict:
    if not REMOTE_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_ARCHIVE)
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/selfcomp_plus/selfcomp_pose_locked_seg_oracle.py",
            "--archive",
            REMOTE_ARCHIVE,
            "--out-dir",
            out_dir,
            "--variant",
            variant,
            "--subset",
            str(subset),
            "--steps",
            str(steps),
            "--lr",
            str(lr),
            "--pose-eps",
            str(pose_eps),
            "--eval-every",
            str(eval_every),
            "--train-batch-size",
            str(train_batch_size),
            "--metric-batch-size",
            str(metric_batch_size),
            "--render-batch-size",
            "2",
            "--device",
            "cuda",
        ]
    )
    volume.commit()
    return load_result(label)


@app.function(gpu=GPU_POOL, cpu=8, memory=32768, timeout=60 * 30, volumes={"/root/selfcomp_plus_volume": volume})
def gpu_smoke() -> dict:
    return run_oracle(
        label="gpu_smoke_latent_model_2x1",
        mode="latent+model",
        subset=2,
        steps=1,
        batch_size=1,
        eval_every=1,
        checkpoint_every=1,
        pose_weight=0.5,
        lr_latent=1e-3,
        lr_model=1e-5,
        lr_affine=1e-4,
        lr_shared=3e-5,
        grad_clip=0.5,
        apply_gates=False,
    )


@app.function(gpu=GPU_POOL, cpu=8, memory=49152, timeout=60 * 60 * 8, volumes={"/root/selfcomp_plus_volume": volume})
def u0_latent_model(
    steps: int = 2000,
    batch_size: int = 4,
    eval_every: int = 100,
    checkpoint_every: int = 250,
) -> dict:
    return run_oracle(
        label=f"u0_latent_model_64_s{steps}_b{batch_size}",
        mode="latent+model",
        subset=64,
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        pose_weight=0.5,
        lr_latent=1e-3,
        lr_model=1e-5,
        lr_affine=1e-4,
        lr_shared=3e-5,
        grad_clip=0.5,
        apply_gates=True,
    )


@app.function(gpu=GPU_POOL, cpu=8, memory=49152, timeout=60 * 60 * 4, volumes={"/root/selfcomp_plus_volume": volume})
def pose_lock(
    variant: str = "p0",
    steps: int = 500,
    lr: float = 1e-2,
    pose_eps: float = 0.005,
    eval_every: int = 25,
    train_batch_size: int = 4,
    metric_batch_size: int = 4,
) -> dict:
    eps_label = str(pose_eps).replace(".", "p")
    lr_label = str(lr).replace(".", "p")
    label = f"pose_lock_{variant}_64_s{steps}_lr{lr_label}_eps{eps_label}"
    return run_pose_lock(
        label=label,
        variant=variant,
        subset=64,
        steps=steps,
        lr=lr,
        pose_eps=pose_eps,
        eval_every=eval_every,
        train_batch_size=train_batch_size,
        metric_batch_size=metric_batch_size,
    )


def save_local_result(result: dict) -> Path:
    label = result.get("label", "modal_result")
    local_out = LOCAL_REPO / "submissions/selfcomp_plus/experiments/modal_returned" / label
    local_out.mkdir(parents=True, exist_ok=True)
    history = result.pop("history_jsonl", None)
    (local_out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if history is not None:
        (local_out / "history.jsonl").write_text(history)
    return local_out


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    steps: int = 2000,
    batch_size: int = 4,
    eval_every: int = 100,
    variant: str = "p0",
    lr: float = 1e-2,
    pose_eps: float = 0.005,
) -> None:
    if stage == "smoke":
        result = gpu_smoke.remote()
    elif stage == "u0":
        result = u0_latent_model.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    elif stage == "pose-lock":
        result = pose_lock.remote(
            variant=variant,
            steps=steps,
            lr=lr,
            pose_eps=pose_eps,
            eval_every=eval_every,
            train_batch_size=batch_size,
            metric_batch_size=batch_size,
        )
    else:
        raise ValueError(f"unknown stage: {stage}")
    local_out = save_local_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved local result to {local_out}")
