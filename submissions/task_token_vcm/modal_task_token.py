#!/usr/bin/env python3
"""Modal entrypoints for task-token VCM capacity oracles."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-task-token-vcm"
REMOTE_REPO = Path("/root/comma_task_token_vcm")
REMOTE_OUT_DIR = Path("/root/task_token_vcm_volume/experiments")


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
        or (path.name.endswith(".pyc"))
        or ("experiments" in parts and path.is_file())
    )


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
    )
    .add_local_dir(LOCAL_REPO, str(REMOTE_REPO), copy=True, ignore=ignore_repo_path)
)

volume = modal.Volume.from_name("comma-task-token-vcm", create_if_missing=True)
app = modal.App(APP_NAME, image=base_image)


def run(cmd: list[str | Path], cwd: Path = REMOTE_REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def load_result(label: str) -> dict:
    out_dir = REMOTE_OUT_DIR / label
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.jsonl"
    result = json.loads(metrics_path.read_text())
    result["label"] = label
    result["remote_out_dir"] = str(out_dir)
    if history_path.exists():
        result["history_jsonl"] = history_path.read_text()
    return result


def run_capacity(
    *,
    label: str,
    preset: str,
    subset: int,
    steps: int,
    batch_size: int,
    eval_every: int,
    decoder_kind: str,
    hidden: int,
    token_ch: int,
    pair_token_ch: int,
    pose_dim: int,
    grid_h: int,
    grid_w: int,
    num_blocks: int,
    lr_decoder: float,
    lr_token: float,
    pose_weight: float,
    seg_feature_weight: float,
    pose_feature_weight: float,
    init_direct_from_original: bool = False,
) -> dict:
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/task_token_vcm/train_capacity.py",
            "--out-dir",
            out_dir,
            "--preset",
            preset,
            "--subset",
            str(subset),
            "--steps",
            str(steps),
            "--batch-size",
            str(batch_size),
            "--eval-every",
            str(eval_every),
            "--decoder-kind",
            decoder_kind,
            "--hidden",
            str(hidden),
            "--token-ch",
            str(token_ch),
            "--pair-token-ch",
            str(pair_token_ch),
            "--pose-dim",
            str(pose_dim),
            "--grid-h",
            str(grid_h),
            "--grid-w",
            str(grid_w),
            "--num-blocks",
            str(num_blocks),
            "--lr-decoder",
            str(lr_decoder),
            "--lr-token",
            str(lr_token),
            "--pose-weight",
            str(pose_weight),
            "--seg-feature-weight",
            str(seg_feature_weight),
            "--pose-feature-weight",
            str(pose_feature_weight),
            "--device",
            "cuda",
        ]
        + (["--init-direct-from-original"] if init_direct_from_original else [])
    )
    volume.commit()
    return load_result(label)


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 30, volumes={"/root/task_token_vcm_volume": volume})
def capacity_smoke() -> dict:
    return run_capacity(
        label="capacity_smoke_2_s1",
        preset="sequential",
        subset=2,
        steps=1,
        batch_size=1,
        eval_every=1,
        decoder_kind="cnn",
        hidden=32,
        token_ch=8,
        pair_token_ch=0,
        pose_dim=32,
        grid_h=24,
        grid_w=32,
        num_blocks=4,
        lr_decoder=2e-4,
        lr_token=1e-2,
        pose_weight=3.0,
        seg_feature_weight=0.0,
        pose_feature_weight=0.0,
    )


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 8, volumes={"/root/task_token_vcm_volume": volume})
def hard8_float(steps: int = 2000, batch_size: int = 4, eval_every: int = 100) -> dict:
    return run_capacity(
        label=f"hard8_float_s{steps}_b{batch_size}",
        preset="hard8",
        subset=8,
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        decoder_kind="cnn",
        hidden=64,
        token_ch=16,
        pair_token_ch=0,
        pose_dim=128,
        grid_h=24,
        grid_w=32,
        num_blocks=4,
        lr_decoder=2e-4,
        lr_token=1e-2,
        pose_weight=3.0,
        seg_feature_weight=0.0,
        pose_feature_weight=0.0,
    )


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 8, volumes={"/root/task_token_vcm_volume": volume})
def hard8_pair_float(steps: int = 1000, batch_size: int = 4, eval_every: int = 50) -> dict:
    return run_capacity(
        label=f"hard8_pair_float_s{steps}_b{batch_size}",
        preset="hard8",
        subset=8,
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        decoder_kind="cnn",
        hidden=64,
        token_ch=16,
        pair_token_ch=16,
        pose_dim=128,
        grid_h=24,
        grid_w=32,
        num_blocks=4,
        lr_decoder=2e-4,
        lr_token=1e-2,
        pose_weight=30.0,
        seg_feature_weight=0.0,
        pose_feature_weight=0.25,
    )


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 8, volumes={"/root/task_token_vcm_volume": volume})
def hard8_direct_rgb(steps: int = 1000, batch_size: int = 4, eval_every: int = 50) -> dict:
    return run_capacity(
        label=f"hard8_direct_rgb_s{steps}_b{batch_size}",
        preset="hard8",
        subset=8,
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        decoder_kind="direct_rgb",
        hidden=64,
        token_ch=1,
        pair_token_ch=3,
        pose_dim=1,
        grid_h=48,
        grid_w=64,
        num_blocks=1,
        lr_decoder=0.0,
        lr_token=5e-2,
        pose_weight=30.0,
        seg_feature_weight=0.0,
        pose_feature_weight=0.25,
    )


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 2, volumes={"/root/task_token_vcm_volume": volume})
def hard8_direct_original_init(steps: int = 0, batch_size: int = 4, eval_every: int = 50) -> dict:
    return run_capacity(
        label=f"hard8_direct_originit_s{steps}_b{batch_size}",
        preset="hard8",
        subset=8,
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        decoder_kind="direct_rgb",
        hidden=64,
        token_ch=1,
        pair_token_ch=3,
        pose_dim=1,
        grid_h=48,
        grid_w=64,
        num_blocks=1,
        lr_decoder=0.0,
        lr_token=5e-2,
        pose_weight=30.0,
        seg_feature_weight=0.0,
        pose_feature_weight=0.25,
        init_direct_from_original=True,
    )


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 10, volumes={"/root/task_token_vcm_volume": volume})
def subset64_float(steps: int = 3000, batch_size: int = 4, eval_every: int = 100) -> dict:
    return run_capacity(
        label=f"subset64_float_s{steps}_b{batch_size}",
        preset="sequential",
        subset=64,
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        decoder_kind="cnn",
        hidden=64,
        token_ch=16,
        pair_token_ch=0,
        pose_dim=128,
        grid_h=24,
        grid_w=32,
        num_blocks=4,
        lr_decoder=2e-4,
        lr_token=1e-2,
        pose_weight=3.0,
        seg_feature_weight=0.0,
        pose_feature_weight=0.0,
    )


def save_local_result(result: dict) -> Path:
    label = result.get("label", "modal_result")
    local_out = LOCAL_REPO / "submissions/task_token_vcm/experiments/modal_returned" / label
    local_out.mkdir(parents=True, exist_ok=True)
    history = result.pop("history_jsonl", None)
    (local_out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if history is not None:
        (local_out / "history.jsonl").write_text(history)
    return local_out


@app.local_entrypoint()
def main(stage: str = "smoke", steps: int = 2000, batch_size: int = 4, eval_every: int = 100) -> None:
    if stage == "smoke":
        result = capacity_smoke.remote()
    elif stage == "hard8":
        result = hard8_float.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    elif stage == "hard8_pair":
        result = hard8_pair_float.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    elif stage == "hard8_direct":
        result = hard8_direct_rgb.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    elif stage == "hard8_direct_orig":
        result = hard8_direct_original_init.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    elif stage == "subset64":
        result = subset64_float.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    else:
        raise ValueError(f"unknown stage: {stage}")
    local_out = save_local_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved local result to {local_out}")
