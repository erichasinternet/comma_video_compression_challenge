#!/usr/bin/env python3
"""Modal entrypoints for TAVS 64-sample codec-in-loop oracles."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-tavs-video"
REMOTE_REPO = Path("/root/comma_tavs_video")
REMOTE_OUT_DIR = Path("/root/tavs_video_volume/experiments")


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

volume = modal.Volume.from_name("comma-tavs-video", create_if_missing=True)
app = modal.App(APP_NAME, image=base_image)


def run(cmd: list[str | Path], cwd: Path = REMOTE_REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def load_result(label: str) -> dict:
    out_dir = REMOTE_OUT_DIR / label
    metrics_path = out_dir / "metrics.json"
    result = json.loads(metrics_path.read_text())
    result["label"] = label
    result["remote_out_dir"] = str(out_dir)
    for name in ("history.jsonl", "codec_history.jsonl"):
        path = out_dir / name
        if path.exists():
            result[name] = path.read_text()
    return result


@app.function(gpu=["L4", "A10"], cpu=8, memory=49152, timeout=60 * 60 * 10, volumes={"/root/tavs_video_volume": volume})
def optimize64(
    init: str = "q55",
    codec: str = "svtav1",
    steps: int = 5000,
    batch_size: int = 4,
    eval_every: int = 100,
    codec_eval_every: int = 250,
    codec_crfs: str = "49,53,57,61",
) -> dict:
    label = f"tavs_64_{init}_{codec}_yuvgrid96_s{steps}"
    out_dir = REMOTE_OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "submissions/tavs_video/optimize_source.py",
            "--out-dir",
            out_dir,
            "--init",
            init,
            "--subset",
            "64",
            "--grid-h",
            "72",
            "--grid-w",
            "96",
            "--steps",
            str(steps),
            "--batch-size",
            str(batch_size),
            "--eval-every",
            str(eval_every),
            "--codec-eval-every",
            str(codec_eval_every),
            "--codec",
            codec,
            "--codec-crfs",
            codec_crfs,
            "--device",
            "cuda",
            "--cache-dir",
            REMOTE_OUT_DIR / "cache",
        ]
    )
    volume.commit()
    return load_result(label)


def save_local_result(result: dict) -> Path:
    label = result.get("label", "modal_result")
    local_out = LOCAL_REPO / "submissions/tavs_video/experiments/modal_returned" / label
    local_out.mkdir(parents=True, exist_ok=True)
    for name in ("history.jsonl", "codec_history.jsonl"):
        text = result.pop(name, None)
        if text is not None:
            (local_out / name).write_text(text)
    (local_out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return local_out


@app.local_entrypoint()
def main(
    stage: str = "optimize64",
    init: str = "q55",
    codec: str = "svtav1",
    steps: int = 5000,
    batch_size: int = 4,
    eval_every: int = 100,
    codec_eval_every: int = 250,
    codec_crfs: str = "49,53,57,61",
) -> None:
    if stage != "optimize64":
        raise ValueError(f"unknown stage: {stage}")
    result = optimize64.remote(
        init=init,
        codec=codec,
        steps=steps,
        batch_size=batch_size,
        eval_every=eval_every,
        codec_eval_every=codec_eval_every,
        codec_crfs=codec_crfs,
    )
    local_out = save_local_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved local result to {local_out}")
