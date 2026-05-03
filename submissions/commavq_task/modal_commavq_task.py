#!/usr/bin/env python3
"""Modal entrypoints for commaVQ-token task-renderer oracles."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-commavq-task"
REMOTE_REPO = Path("/root/comma_commavq_task")
REMOTE_OUT_DIR = Path("/root/commavq_task_volume/experiments")
REMOTE_COMMAVQ = Path("/opt/commavq")


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
        "opencv-python-headless",
    )
    .run_commands(f"git clone --depth 1 https://github.com/commaai/commavq.git {REMOTE_COMMAVQ}")
    .add_local_dir(LOCAL_REPO, str(REMOTE_REPO), copy=True, ignore=ignore_repo_path)
)

volume = modal.Volume.from_name("comma-commavq-task", create_if_missing=True)
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


def ensure_tokens(label: str, preset: str, subset: int, batch_size: int) -> Path:
    token_dir = REMOTE_OUT_DIR / label
    if not (token_dir / "tokens.npy").exists():
        token_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "python",
                "submissions/commavq_task/encode_tokens.py",
                "--out-dir",
                token_dir,
                "--commavq-root",
                REMOTE_COMMAVQ,
                "--preset",
                preset,
                "--subset",
                str(subset),
                "--batch-size",
                str(batch_size),
                "--device",
                "cuda",
            ]
        )
        volume.commit()
    return token_dir


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 4, volumes={"/root/commavq_task_volume": volume})
def encode_hard8() -> dict:
    token_dir = ensure_tokens("tokens_hard8", "hard8", 8, 8)
    return json.loads((token_dir / "metrics.json").read_text())


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 4, volumes={"/root/commavq_task_volume": volume})
def decoder_oracle_hard8() -> dict:
    token_dir = ensure_tokens("tokens_hard8", "hard8", 8, 8)
    out_dir = REMOTE_OUT_DIR / "decoder_oracle_hard8"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/commavq_task/eval_commavq_decoder.py",
            "--tokens",
            token_dir / "tokens.npy",
            "--sample-ids",
            token_dir / "sample_ids.json",
            "--out-dir",
            out_dir,
            "--commavq-root",
            REMOTE_COMMAVQ,
            "--batch-size",
            "4",
            "--device",
            "cuda",
        ]
    )
    volume.commit()
    return load_result("decoder_oracle_hard8")


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 8, volumes={"/root/commavq_task_volume": volume})
def train_hard8(steps: int = 2000, batch_size: int = 4, eval_every: int = 100) -> dict:
    token_dir = ensure_tokens("tokens_hard8", "hard8", 8, 8)
    label = f"renderer_hard8_s{steps}_b{batch_size}"
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/commavq_task/train_renderer.py",
            "--tokens",
            token_dir / "tokens.npy",
            "--sample-ids",
            token_dir / "sample_ids.json",
            "--out-dir",
            out_dir,
            "--steps",
            str(steps),
            "--batch-size",
            str(batch_size),
            "--eval-every",
            str(eval_every),
            "--device",
            "cuda",
        ]
    )
    volume.commit()
    return load_result(label)


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 10, volumes={"/root/commavq_task_volume": volume})
def train_hard8_big(rgb_steps: int = 2000, task_steps: int = 2000, batch_size: int = 2, eval_every: int = 100) -> dict:
    token_dir = ensure_tokens("tokens_hard8", "hard8", 8, 8)
    label = f"renderer_hard8_big_rgb{rgb_steps}_task{task_steps}_b{batch_size}"
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/commavq_task/train_renderer.py",
            "--tokens",
            token_dir / "tokens.npy",
            "--sample-ids",
            token_dir / "sample_ids.json",
            "--out-dir",
            out_dir,
            "--rgb-anchor-steps",
            str(rgb_steps),
            "--steps",
            str(task_steps),
            "--batch-size",
            str(batch_size),
            "--eval-every",
            str(eval_every),
            "--checkpoint-every",
            "500",
            "--emb-dim",
            "64",
            "--hidden",
            "128",
            "--num-blocks",
            "6",
            "--separate-heads",
            "--lr",
            "0.0002",
            "--rgb-anchor-mode",
            "original",
            "--pose-weight",
            "10.0",
            "--pose-feature-weight",
            "1.0",
            "--seg-feature-weight",
            "0.25",
            "--device",
            "cuda",
        ]
    )
    volume.commit()
    return load_result(label)


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 10, volumes={"/root/commavq_task_volume": volume})
def train_subset64(steps: int = 3000, batch_size: int = 4, eval_every: int = 100) -> dict:
    token_dir = ensure_tokens("tokens_subset64", "sequential", 64, 8)
    label = f"renderer_subset64_s{steps}_b{batch_size}"
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/commavq_task/train_renderer.py",
            "--tokens",
            token_dir / "tokens.npy",
            "--sample-ids",
            token_dir / "sample_ids.json",
            "--out-dir",
            out_dir,
            "--steps",
            str(steps),
            "--batch-size",
            str(batch_size),
            "--eval-every",
            str(eval_every),
            "--device",
            "cuda",
        ]
    )
    volume.commit()
    return load_result(label)


def save_local_result(result: dict) -> Path:
    label = result.get("label", "modal_result")
    local_out = LOCAL_REPO / "submissions/commavq_task/experiments/modal_returned" / label
    local_out.mkdir(parents=True, exist_ok=True)
    history = result.pop("history_jsonl", None)
    (local_out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if history is not None:
        (local_out / "history.jsonl").write_text(history)
    return local_out


@app.local_entrypoint()
def main(
    stage: str = "encode_hard8",
    steps: int = 2000,
    rgb_steps: int = 2000,
    task_steps: int = 2000,
    batch_size: int = 4,
    eval_every: int = 100,
) -> None:
    if stage == "encode_hard8":
        result = encode_hard8.remote()
        result["label"] = "tokens_hard8"
    elif stage == "decoder_hard8":
        result = decoder_oracle_hard8.remote()
    elif stage == "train_hard8":
        result = train_hard8.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    elif stage == "train_hard8_big":
        result = train_hard8_big.remote(rgb_steps=rgb_steps, task_steps=task_steps, batch_size=batch_size, eval_every=eval_every)
    elif stage == "train64":
        result = train_subset64.remote(steps=steps, batch_size=batch_size, eval_every=eval_every)
    else:
        raise ValueError(f"unknown stage: {stage}")
    local_out = save_local_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved local result to {local_out}")
