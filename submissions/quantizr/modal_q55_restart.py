#!/usr/bin/env python
"""Modal entrypoints for the Quantizr #55 restart gates."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import modal


APP_NAME = "comma-q55-restart"
REMOTE_REPO = Path("/root/comma_q55_restart")
REMOTE_Q55_ARCHIVE = REMOTE_REPO / "submissions/quantizr/experiments/q55_restart/source/q55_archive.zip"
REMOTE_Q55_INT10_ARCHIVE = (
    REMOTE_REPO / "submissions/quantizr/experiments/q55_restart/q1_fp16_pose_int10_cpu/submission/archive.zip"
)
REMOTE_QRECODE50_ARCHIVE = (
    REMOTE_REPO / "submissions/quantizr/experiments/q55_restart/qrecode50_archive_cpu/submission/archive.zip"
)
REMOTE_OUT_DIR = REMOTE_REPO / "submissions/quantizr/experiments/modal_q55_restart"


def infer_local_repo() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path("/root/comma_q55_restart")


LOCAL_REPO = infer_local_repo()


def ignore_repo_path(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name
    path_text = path.as_posix()
    allowed_experiment_file = (
        name == "q55_archive.zip"
        or path_text.endswith("q1_fp16_pose_int10_cpu/submission/archive.zip")
        or path_text.endswith("qrecode50_archive_cpu/submission/archive.zip")
    )
    return (
        ".git" in parts
        or any(part.startswith(".venv") for part in path.parts)
        or ".venv-modal" in parts
        or "runpod_saved" in parts
        or "__pycache__" in parts
        or "submissions/q55_manual/" in path_text
        or "submissions/eval_candidate_a_cpu/" in path_text
        or "submissions/pairwise_semantic_residual/" in path_text
        or "inflated/" in path_text
        or name.endswith(".pyc")
        or (
            {"submissions", "quantizr", "experiments"} <= parts
            and path.is_file()
            and not allowed_experiment_file
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
    .pip_install("nvidia-dali-cuda120==1.53.0", extra_index_url="https://pypi.nvidia.com")
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
        "imageio-ffmpeg",
    )
    .add_local_dir(LOCAL_REPO, str(REMOTE_REPO), copy=True, ignore=ignore_repo_path)
)

app = modal.App(APP_NAME, image=base_image)


def run(cmd: list[str], cwd: Path = REMOTE_REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def load_metrics(label: str) -> dict:
    path = REMOTE_OUT_DIR / label / "metrics.json"
    return json.loads(path.read_text())


def load_metrics_with_archive(label: str) -> dict:
    metrics = load_metrics(label)
    archive_zip = metrics.get("archive_zip")
    if archive_zip and Path(archive_zip).exists():
        metrics["archive_zip_b64"] = base64.b64encode(Path(archive_zip).read_bytes()).decode("ascii")
    return metrics


def save_returned_archive(result: dict) -> dict:
    archive_b64 = result.pop("archive_zip_b64", None)
    label = result.get("label", "modal_result")
    local_out = LOCAL_REPO / "submissions/quantizr/experiments/q55_restart/modal_returned" / label
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if archive_b64:
        (local_out / "archive.zip").write_bytes(base64.b64decode(archive_b64.encode("ascii")))
        result["local_archive_zip"] = str(local_out / "archive.zip")
    result["local_metrics_json"] = str(local_out / "metrics.json")
    return result


def enforce_crf50_reproduction_gate(metrics: dict) -> None:
    """Fail fast if regenerated CRF50 masks do not reproduce the baseline."""
    base_path = REMOTE_OUT_DIR / "q0_upstream_cuda_av" / "metrics.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        checks = (
            ("score", 0.006),
            ("segnet_dist", 0.00008),
            ("posenet_dist", 0.00008),
        )
        failures = []
        for key, tol in checks:
            delta = abs(float(metrics[key]) - float(base[key]))
            if delta > tol:
                failures.append(f"{key} delta {delta:.8f} > {tol}")
        if failures:
            print("CRF50 reproduction metrics:", json.dumps(metrics, indent=2, sort_keys=True), flush=True)
            raise RuntimeError("CRF50 reproduction gate failed: " + "; ".join(failures))
        return

    # Fallback when q0-av was not run in this ephemeral Modal app.
    if float(metrics["score"]) > 0.45 or float(metrics["posenet_dist"]) > 0.002:
        print("CRF50 reproduction metrics:", json.dumps(metrics, indent=2, sort_keys=True), flush=True)
        raise RuntimeError("CRF50 reproduction gate failed without baseline metrics")


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 3, cpu=8, memory=32768)
def q0_calibration() -> dict:
    """Run only the hard Q0 calibration gate on a CUDA GPU."""
    if not REMOTE_Q55_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_ARCHIVE)
    REMOTE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    run(
        [
            "python",
            "submissions/quantizr/q55_calibrate.py",
            "--archive-zip",
            REMOTE_Q55_ARCHIVE,
            "--inflate-mode",
            "upstream",
            "--device",
            "cuda",
            "--out-dir",
            REMOTE_OUT_DIR,
            "--label",
            "q0_upstream_cuda",
        ]
    )
    run(
        [
            "python",
            "submissions/quantizr/q55_calibrate.py",
            "--archive-zip",
            REMOTE_Q55_ARCHIVE,
            "--inflate-mode",
            "modified",
            "--device",
            "cuda",
            "--out-dir",
            REMOTE_OUT_DIR,
            "--label",
            "q0_modified_cuda",
        ]
    )
    run(
        [
            "python",
            "submissions/quantizr/q55_gate.py",
            "q0",
            "--upstream",
            REMOTE_OUT_DIR / "q0_upstream_cuda/metrics.json",
            "--modified",
            REMOTE_OUT_DIR / "q0_modified_cuda/metrics.json",
        ]
    )
    return {
        "upstream": load_metrics("q0_upstream_cuda"),
        "modified": load_metrics("q0_modified_cuda"),
    }


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 3, cpu=8, memory=32768)
def q0_calibration_cuda_av() -> dict:
    """Run Q0 with CUDA inference and AV video loading when Modal DALI/NVML fails."""
    if not REMOTE_Q55_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_ARCHIVE)
    REMOTE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in ("upstream", "modified"):
        label = f"q0_{mode}_cuda_av"
        run(
            [
                "python",
                "submissions/quantizr/q55_calibrate.py",
                "--archive-zip",
                REMOTE_Q55_ARCHIVE,
                "--inflate-mode",
                mode,
                "--device",
                "cuda",
                "--force-av-dataset",
                "--out-dir",
                REMOTE_OUT_DIR,
                "--label",
                label,
            ]
        )
    return {
        "upstream": load_metrics("q0_upstream_cuda_av"),
        "modified": load_metrics("q0_modified_cuda_av"),
    }


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 4, cpu=8, memory=32768)
def q1_qpack() -> dict:
    """Run qpack variants after Q0 has passed."""
    results = {}
    for variant in ("fp16", "mixed_int8", "mixed_int8_heads_fp16"):
        label = f"q1_{variant}_cuda"
        run(
            [
                "python",
                "submissions/quantizr/q55_qpack_eval.py",
                "--base-archive",
                REMOTE_Q55_ARCHIVE,
                "--variant",
                variant,
                "--device",
                "cuda",
                "--out-dir",
                REMOTE_OUT_DIR,
                "--label",
                label,
            ]
        )
        results[variant] = load_metrics(label)
    return results


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 4, cpu=8, memory=32768)
def q1_qpack_cuda_av() -> dict:
    """Run qpack variants with CUDA inference and AV video loading as a diagnostic."""
    results = {}
    for variant in ("fp16", "mixed_int8", "mixed_int8_heads_fp16"):
        label = f"q1_{variant}_cuda_av"
        run(
            [
                "python",
                "submissions/quantizr/q55_qpack_eval.py",
                "--base-archive",
                REMOTE_Q55_ARCHIVE,
                "--variant",
                variant,
                "--device",
                "cuda",
                "--force-av-dataset",
                "--out-dir",
                REMOTE_OUT_DIR,
                "--label",
                label,
            ]
        )
        results[variant] = load_metrics(label)
    return results


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 8, cpu=8, memory=32768)
def qcrf_cuda_av() -> dict:
    """Run CRF zero-step swaps with CUDA inference and AV video loading as a diagnostic."""
    results = {}
    for crf in (50, 52, 54, 56):
        label = f"qcrf{crf}_cuda_av"
        run(
            [
                "python",
                "submissions/quantizr/q55_crf_swap.py",
                "--base-archive",
                REMOTE_Q55_ARCHIVE,
                "--crf",
                str(crf),
                "--device",
                "cuda",
                "--force-av-dataset",
                "--out-dir",
                REMOTE_OUT_DIR,
                "--label",
                label,
            ]
        )
        metrics = load_metrics(label)
        results[str(crf)] = metrics
        if crf == 50:
            enforce_crf50_reproduction_gate(metrics)
    return results


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 8, cpu=8, memory=32768)
def qmask_av(eval_device: str = "cpu") -> dict:
    """Run mixed-CRF mask allocation with AV source decode."""
    results = {}
    specs = (
        "50:0.20,54:0.35,58:0.45",
        "52:0.25,56:0.35,60:0.40",
        "50:0.15,54:0.35,58:0.35,60:0.15",
    )
    for spec in specs:
        safe = spec.replace(":", "_").replace(",", "_").replace(".", "p")
        label = f"qmask_{safe}_{eval_device}_av"
        run(
            [
                "python",
                "submissions/quantizr/q55_mask_alloc.py",
                "--base-archive",
                REMOTE_Q55_ARCHIVE,
                "--device",
                "cuda",
                "--eval-device",
                eval_device,
                "--decode-backend",
                "av",
                "--group-spec",
                spec,
                "--order",
                "hist",
                "--palette",
                "legacy",
                "--out-dir",
                REMOTE_OUT_DIR,
                "--label",
                label,
            ]
        )
        results[spec] = load_metrics(label)
    return results


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 10, cpu=8, memory=32768)
def q3_warmstart_av(crf: int = 54, eval_device: str = "cpu") -> dict:
    """Run a bounded CRF warm-start probe with AV source decode."""
    label = f"q3_crf{crf}_warmstart_{eval_device}_av"
    run(
        [
            "python",
            "submissions/quantizr/q55_warmstart.py",
            "--base-archive",
            REMOTE_Q55_ARCHIVE,
            "--crf",
            str(crf),
            "--device",
            "cuda:0",
            "--eval-device",
            eval_device,
            "--decode-backend",
            "av",
            "--out-dir",
            REMOTE_OUT_DIR,
            "--label",
            label,
            "--zero-eval",
            "--final-eval",
        ]
    )
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 10, cpu=8, memory=32768)
def qpack_pose_recover_av(pose_variant: str = "int10_per_dim", eval_device: str = "cpu") -> dict:
    """Recover mixed-int8 qpack PoseNet loss while preserving the exact #55 mask stream."""
    label = f"qpack_recover_mixed_int8_heads_fp16_pose_{pose_variant}_{eval_device}_av"
    run(
        [
            "python",
            "submissions/quantizr/q55_warmstart.py",
            "--base-archive",
            REMOTE_Q55_ARCHIVE,
            "--mask-source",
            "archive",
            "--device",
            "cuda:0",
            "--eval-device",
            eval_device,
            "--decode-backend",
            "av",
            "--batch-size",
            "1",
            "--qpack-variant",
            "mixed_int8_heads_fp16",
            "--pose-pack-variant",
            pose_variant,
            "--adapt-epochs",
            "0",
            "--seg-epochs",
            "0",
            "--pose-epochs",
            "4",
            "--pose-lr",
            "5e-6",
            "--pose-weight",
            "1.0",
            "--out-dir",
            REMOTE_OUT_DIR,
            "--label",
            label,
            "--zero-eval",
            "--final-eval",
        ]
    )
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 8, cpu=8, memory=32768)
def q55_exact_mask_polish_av(
    pose_variant: str = "int10_per_dim",
    eval_device: str = "cpu",
    adapt_epochs: int = 1,
    seg_epochs: int = 4,
    pose_epochs: int = 2,
    padding_mode: str = "zeros",
    frame_hidden: int = 52,
    cond_dim: int = 48,
    zero_score_max: float | None = 0.60,
) -> dict:
    """Short metric polish on the exact #55 mask stream plus safe fp16/pose packing."""
    label = (
        "q55_exact_mask_polish_fp16_pose_"
        f"{pose_variant}_{padding_mode}_h{frame_hidden}_c{cond_dim}_"
        f"{adapt_epochs}a_{seg_epochs}s_{pose_epochs}p_{eval_device}_av"
    )
    cmd = [
        "python",
        "submissions/quantizr/q55_warmstart.py",
        "--base-archive",
        REMOTE_Q55_ARCHIVE,
        "--mask-source",
        "archive",
        "--device",
        "cuda:0",
        "--eval-device",
        eval_device,
        "--decode-backend",
        "av",
        "--batch-size",
        "1",
        "--qpack-variant",
        "fp16",
        "--pose-pack-variant",
        pose_variant,
        "--adapt-epochs",
        str(adapt_epochs),
        "--seg-epochs",
        str(seg_epochs),
        "--pose-epochs",
        str(pose_epochs),
        "--adapt-lr",
        "5e-6",
        "--seg-lr",
        "3e-6",
        "--pose-lr",
        "2e-6",
        "--pose-weight",
        "1.0",
        "--padding-mode",
        padding_mode,
        "--frame-hidden",
        str(frame_hidden),
        "--cond-dim",
        str(cond_dim),
        "--out-dir",
        REMOTE_OUT_DIR,
        "--label",
        label,
        "--zero-eval",
        "--final-eval",
    ]
    if zero_score_max is not None:
        cmd.extend(["--zero-score-max", str(zero_score_max)])
    run(cmd)
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 10, cpu=8, memory=32768)
def qrecode50_full_recover_av(
    pose_variant: str = "int10_per_dim",
    eval_device: str = "cpu",
    adapt_epochs: int = 2,
    seg_epochs: int = 8,
    pose_epochs: int = 4,
) -> dict:
    """Full-generator warm-start sanity probe on the qrecode50 archive-mask stream."""
    if not REMOTE_QRECODE50_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_QRECODE50_ARCHIVE)
    label = (
        "qrecode50_full_recover_fp16_pose_"
        f"{pose_variant}_{adapt_epochs}a_{seg_epochs}s_{pose_epochs}p_{eval_device}_av"
    )
    run(
        [
            "python",
            "submissions/quantizr/q55_warmstart.py",
            "--base-archive",
            REMOTE_QRECODE50_ARCHIVE,
            "--mask-source",
            "archive",
            "--device",
            "cuda:0",
            "--eval-device",
            eval_device,
            "--decode-backend",
            "av",
            "--batch-size",
            "1",
            "--qpack-variant",
            "fp16",
            "--pose-pack-variant",
            pose_variant,
            "--adapt-epochs",
            str(adapt_epochs),
            "--seg-epochs",
            str(seg_epochs),
            "--pose-epochs",
            str(pose_epochs),
            "--adapt-lr",
            "1e-5",
            "--seg-lr",
            "5e-6",
            "--pose-lr",
            "3e-6",
            "--pose-weight",
            "1.0",
            "--out-dir",
            REMOTE_OUT_DIR,
            "--label",
            label,
            "--final-eval",
        ]
    )
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 3, cpu=8, memory=32768)
def qmask_denoise_qrecode50_sanity(steps: int = 1000, eval_every: int = 250) -> dict:
    """Supervised tiny mask-denoiser sanity probe on qrecode50 archive masks."""
    if not REMOTE_Q55_INT10_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_INT10_ARCHIVE)
    if not REMOTE_QRECODE50_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_QRECODE50_ARCHIVE)
    label = f"qmask_denoise_qrecode50_sanity_gray_thr_{steps}s"
    run(
        [
            "python",
            "submissions/quantizr/q55_mask_denoise.py",
            "--exact-archive",
            REMOTE_Q55_INT10_ARCHIVE,
            "--predictor-mask-archive",
            REMOTE_QRECODE50_ARCHIVE,
            "--base-package-archive",
            REMOTE_Q55_INT10_ARCHIVE,
            "--device",
            "cuda:0",
            "--steps",
            str(steps),
            "--batch-size",
            "2",
            "--eval-batch-size",
            "4",
            "--hidden",
            "12",
            "--lr",
            "0.002",
            "--changed-weight",
            "512",
            "--eval-every",
            str(eval_every),
            "--out-dir",
            REMOTE_OUT_DIR,
            "--label",
            label,
        ]
    )
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 4, cpu=8, memory=32768)
def qmask_adapter_qrecode50_sanity(steps: int = 1000, eval_every: int = 250) -> dict:
    """Evaluator-level tiny mask-adapter sanity probe on qrecode50 archive masks."""
    if not REMOTE_Q55_INT10_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_INT10_ARCHIVE)
    if not REMOTE_QRECODE50_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_QRECODE50_ARCHIVE)
    label = f"qmask_adapter_qrecode50_sanity_{steps}s"
    run(
        [
            "python",
            "submissions/quantizr/q55_mask_adapter_train.py",
            "--exact-archive",
            REMOTE_Q55_INT10_ARCHIVE,
            "--predictor-mask-archive",
            REMOTE_QRECODE50_ARCHIVE,
            "--base-package-archive",
            REMOTE_Q55_INT10_ARCHIVE,
            "--device",
            "cuda:0",
            "--steps",
            str(steps),
            "--batch-size",
            "2",
            "--eval-batch-size",
            "4",
            "--hidden",
            "12",
            "--lr",
            "0.002",
            "--frame-weight",
            "8.0",
            "--feature-weight",
            "0.02",
            "--seg-weight",
            "0.25",
            "--pose-weight",
            "2.0",
            "--eval-every",
            str(eval_every),
            "--out-dir",
            REMOTE_OUT_DIR,
            "--label",
            label,
        ]
    )
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 8, cpu=8, memory=32768)
def q55_pixel_oracle_av(
    offset: int = 0,
    max_samples: int = 64,
    steps: int = 500,
    opt_batch_size: int = 2,
    max_delta: float = 4.0,
    lr: float = 0.005,
    pose_term_weight: float = 20.0,
    hard_pixels_only: bool = True,
    hard_pixel_boost: float = 256.0,
    seg_ce_weight: float = 1.0,
    seg_margin_weight: float = 0.25,
    pose_mse_weight: float = 3.0,
    tv_weight: float = 0.01,
    saturation_weight: float = 0.001,
    early_stop_patience: int = 0,
    tag: str = "",
) -> dict:
    """Direct pixel-space evaluator inversion oracle at the current safe byte budget."""
    if not REMOTE_Q55_INT10_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_INT10_ARCHIVE)
    delta_label = str(max_delta).replace(".", "p")
    lr_label = str(lr).replace(".", "p")
    pose_label = str(pose_term_weight).replace(".", "p")
    mode_label = "hard" if hard_pixels_only else "full"
    label = tag or (
        f"q55_pixel_oracle_{mode_label}_d{delta_label}_lr{lr_label}_p{pose_label}_"
        f"o{offset}_{max_samples}s_{steps}steps_b{opt_batch_size}_camera_av"
    )
    cmd = [
        "python",
        "submissions/quantizr/q55_pixel_oracle.py",
        "--base-archive",
        REMOTE_Q55_INT10_ARCHIVE,
        "--device",
        "cuda:0",
        "--offset",
        str(offset),
        "--max-samples",
        str(max_samples),
        "--steps",
        str(steps),
        "--opt-batch-size",
        str(opt_batch_size),
        "--decode-batch-size",
        "8",
        "--target-batch-size",
        "8",
        "--gen-batch-size",
        "8",
        "--eval-batch-size",
        "4",
        "--camera-sim",
        "--max-delta",
        str(max_delta),
        "--lr",
        str(lr),
        "--pose-term-weight",
        str(pose_term_weight),
        "--seg-ce-weight",
        str(seg_ce_weight),
        "--seg-margin-weight",
        str(seg_margin_weight),
        "--pose-mse-weight",
        str(pose_mse_weight),
        "--tv-weight",
        str(tv_weight),
        "--saturation-weight",
        str(saturation_weight),
        "--hard-pixel-boost",
        str(hard_pixel_boost),
        "--log-every",
        "25",
        "--save-per-sample-metrics",
        "--out-dir",
        REMOTE_OUT_DIR,
        "--label",
        label,
    ]
    if not hard_pixels_only:
        cmd.append("--no-hard-pixels-only")
    if early_stop_patience > 0:
        cmd.extend(["--early-stop-min-step", "20", "--early-stop-patience", str(early_stop_patience)])
    run(cmd)
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 3, cpu=8, memory=32768)
def q55_pose_control_oracle_av(
    indices: str = "59,60,62",
    offset: int = 56,
    max_samples: int = 8,
    controls: str = "baseline,orig_f1,lowres_y,lowres_rgb,affine",
    lowres_sizes: str = "16x12,24x18,32x24,48x36",
    affine_steps: int = 250,
    affine_lr: float = 0.002,
    patch_steps: int = 300,
    patch_size: str = "64x32",
    patch_grid: str = "12x8",
    tag: str = "",
) -> dict:
    """Frame1-only PoseNet control oracle for the hard-tail samples."""
    if not REMOTE_Q55_INT10_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_INT10_ARCHIVE)
    if tag:
        label = tag
    elif indices:
        safe_indices = indices.replace(",", "_")
        label = f"q55_pose_control_indices_{safe_indices}_av"
    else:
        label = f"q55_pose_control_o{offset}_{max_samples}s_av"
    cmd = [
        "python",
        "submissions/quantizr/q55_pose_control_oracle.py",
        "--base-archive",
        REMOTE_Q55_INT10_ARCHIVE,
        "--device",
        "cuda:0",
        "--controls",
        controls,
        "--lowres-sizes",
        lowres_sizes,
        "--decode-batch-size",
        "8",
        "--target-batch-size",
        "8",
        "--gen-batch-size",
        "8",
        "--eval-batch-size",
        "4",
        "--opt-batch-size",
        "1",
        "--affine-steps",
        str(affine_steps),
        "--affine-lr",
        str(affine_lr),
        "--patch-steps",
        str(patch_steps),
        "--patch-size",
        patch_size,
        "--patch-grid",
        patch_grid,
        "--save-per-sample-metrics",
        "--out-dir",
        REMOTE_OUT_DIR,
        "--label",
        label,
    ]
    if indices:
        cmd.extend(["--indices", indices])
    else:
        cmd.extend(["--offset", str(offset), "--max-samples", str(max_samples)])
    run(cmd)
    return load_metrics(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 3, cpu=8, memory=32768)
def q55_pose_table_oracle_av(
    indices: str = "59,60,62",
    offset: int = 56,
    max_samples: int = 8,
    mode: str = "cem_adam",
    param_mode: str = "bounded",
    scale: float = 1.0,
    steps: int = 300,
    lr: float = 0.05,
    cem_candidates: int = 512,
    cem_iterations: int = 2,
    opt_batch_size: int = 3,
    package: bool = False,
    tag: str = "",
) -> dict:
    """Optimize the stored pose-conditioning table for the frozen #55 generator."""
    if not REMOTE_Q55_INT10_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_INT10_ARCHIVE)
    scale_label = str(scale).replace(".", "p")
    if tag:
        label = tag
    elif indices:
        safe_indices = indices.replace(",", "_")
        label = f"q55_pose_table_{safe_indices}_{mode}_s{scale_label}_av"
    else:
        label = f"q55_pose_table_o{offset}_{max_samples}s_{mode}_s{scale_label}_av"
    cmd = [
        "python",
        "submissions/quantizr/q55_pose_table_oracle.py",
        "--base-archive",
        REMOTE_Q55_INT10_ARCHIVE,
        "--device",
        "cuda:0",
        "--mode",
        mode,
        "--param-mode",
        param_mode,
        "--scale",
        str(scale),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--cem-candidates",
        str(cem_candidates),
        "--cem-iterations",
        str(cem_iterations),
        "--candidate-batch-size",
        "32",
        "--opt-batch-size",
        str(opt_batch_size),
        "--decode-batch-size",
        "8",
        "--target-batch-size",
        "8",
        "--gen-batch-size",
        "8",
        "--eval-batch-size",
        "4",
        "--log-every",
        "25",
        "--save-per-sample-metrics",
        "--out-dir",
        REMOTE_OUT_DIR,
        "--label",
        label,
    ]
    if indices:
        cmd.extend(["--indices", indices])
    else:
        cmd.extend(["--offset", str(offset), "--max-samples", str(max_samples)])
    if package:
        cmd.append("--package")
    run(cmd)
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 60 * 5, cpu=8, memory=32768)
def q55_student_frontier_av(
    budget: str = "S28",
    offset: int = 0,
    max_samples: int = 64,
    indices: str = "",
    teacher_steps: int = 300,
    task_steps: int = 500,
    qat_steps: int = 150,
    batch_size: int = 2,
    teacher_lr: float = 8e-4,
    task_lr: float = 2e-4,
    qat_lr: float = 8e-5,
    frame_anchor_weight: float = 0.25,
    seg_ce_weight: float = 1.0,
    seg_kl_weight: float = 0.5,
    pose_mse_weight: float = 80.0,
    qpack_quantize: str = "none",
    teacher_init: bool = True,
    tag: str = "",
) -> dict:
    """Train/package a reduced exact-mask student decoder for the model-byte frontier."""
    if not REMOTE_Q55_INT10_ARCHIVE.exists():
        raise FileNotFoundError(REMOTE_Q55_INT10_ARCHIVE)
    if tag:
        label = tag
    elif indices:
        safe_indices = indices.replace(",", "_")
        label = f"q55_student_{budget}_idx_{safe_indices}_av"
    else:
        label = f"q55_student_{budget}_o{offset}_{max_samples}s_av"
    cmd = [
        "python",
        "submissions/quantizr/q55_student_frontier.py",
        "--base-archive",
        REMOTE_Q55_INT10_ARCHIVE,
        "--device",
        "cuda:0",
        "--budget",
        budget,
        "--teacher-steps",
        str(teacher_steps),
        "--task-steps",
        str(task_steps),
        "--qat-steps",
        str(qat_steps),
        "--teacher-lr",
        str(teacher_lr),
        "--task-lr",
        str(task_lr),
        "--qat-lr",
        str(qat_lr),
        "--batch-size",
        str(batch_size),
        "--frame-anchor-weight",
        str(frame_anchor_weight),
        "--seg-ce-weight",
        str(seg_ce_weight),
        "--seg-kl-weight",
        str(seg_kl_weight),
        "--pose-mse-weight",
        str(pose_mse_weight),
        "--decode-batch-size",
        "8",
        "--target-batch-size",
        "8",
        "--gen-batch-size",
        "8",
        "--eval-batch-size",
        "4",
        "--qpack-quantize",
        qpack_quantize,
        "--save-per-sample-metrics",
        "--out-dir",
        REMOTE_OUT_DIR,
        "--label",
        label,
    ]
    if indices:
        cmd.extend(["--indices", indices])
    else:
        cmd.extend(["--offset", str(offset), "--max-samples", str(max_samples)])
    if not teacher_init:
        cmd.append("--no-teacher-init")
    run(cmd)
    return load_metrics_with_archive(label)


@app.function(gpu=GPU_POOL, timeout=60 * 20, cpu=4, memory=16384)
def dali_video_smoke() -> dict:
    """Minimal proof that Modal can run the official DALI video input operator."""
    run(
        [
            "python",
            "-c",
            "\n".join(
                [
                    "import json, torch",
                    "from pathlib import Path",
                    "from frame_utils import DaliVideoDataset",
                    "files = ['0.mkv']",
                    "device = torch.device('cuda:0')",
                    "ds = DaliVideoDataset(files, data_dir=Path('videos'), batch_size=4, device=device, num_threads=2, prefetch_queue_depth=2)",
                    "ds.prepare_data()",
                    "path, idx, batch = next(iter(ds))",
                    "torch.cuda.synchronize()",
                    "print(json.dumps({'path': path, 'idx': idx, 'shape': list(batch.shape), 'dtype': str(batch.dtype), 'device': str(batch.device), 'cuda_name': torch.cuda.get_device_name(0)}, sort_keys=True))",
                ]
            ),
        ]
    )
    return {"status": "passed"}


@app.local_entrypoint()
def main(stage: str = "q0"):
    if stage == "q0":
        print(json.dumps(q0_calibration.remote(), indent=2, sort_keys=True))
    elif stage == "q0-av":
        print(json.dumps(q0_calibration_cuda_av.remote(), indent=2, sort_keys=True))
    elif stage == "q1":
        print(json.dumps(q1_qpack.remote(), indent=2, sort_keys=True))
    elif stage == "q1-av":
        print(json.dumps(q1_qpack_cuda_av.remote(), indent=2, sort_keys=True))
    elif stage == "qcrf-av":
        print(json.dumps(qcrf_cuda_av.remote(), indent=2, sort_keys=True))
    elif stage == "qmask-av":
        print(json.dumps(qmask_av.remote(eval_device="cpu"), indent=2, sort_keys=True))
    elif stage == "q3-crf54-av":
        print(json.dumps(save_returned_archive(q3_warmstart_av.remote(crf=54, eval_device="cpu")), indent=2, sort_keys=True))
    elif stage == "q3-crf56-av":
        print(json.dumps(save_returned_archive(q3_warmstart_av.remote(crf=56, eval_device="cpu")), indent=2, sort_keys=True))
    elif stage == "qpack-recover-int10-av":
        print(json.dumps(save_returned_archive(qpack_pose_recover_av.remote(pose_variant="int10_per_dim", eval_device="cpu")), indent=2, sort_keys=True))
    elif stage == "qpack-recover-int12-av":
        print(json.dumps(save_returned_archive(qpack_pose_recover_av.remote(pose_variant="int12_per_dim", eval_device="cpu")), indent=2, sort_keys=True))
    elif stage == "q55-polish-int10-fast-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_exact_mask_polish_av.remote(
                        pose_variant="int10_per_dim",
                        eval_device="cpu",
                        adapt_epochs=1,
                        seg_epochs=4,
                        pose_epochs=2,
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-polish-int12-fast-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_exact_mask_polish_av.remote(
                        pose_variant="int12_per_dim",
                        eval_device="cpu",
                        adapt_epochs=1,
                        seg_epochs=4,
                        pose_epochs=2,
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-polish-int10-reflect-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_exact_mask_polish_av.remote(
                        pose_variant="int10_per_dim",
                        eval_device="cpu",
                        adapt_epochs=1,
                        seg_epochs=4,
                        pose_epochs=2,
                        padding_mode="reflect",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-polish-int10-replicate-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_exact_mask_polish_av.remote(
                        pose_variant="int10_per_dim",
                        eval_device="cpu",
                        adapt_epochs=1,
                        seg_epochs=4,
                        pose_epochs=2,
                        padding_mode="replicate",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "qrecode50-full-recover-av":
        print(json.dumps(save_returned_archive(qrecode50_full_recover_av.remote()), indent=2, sort_keys=True))
    elif stage == "qmask-denoise-qrecode50":
        print(json.dumps(save_returned_archive(qmask_denoise_qrecode50_sanity.remote()), indent=2, sort_keys=True))
    elif stage == "qmask-adapter-qrecode50":
        print(json.dumps(save_returned_archive(qmask_adapter_qrecode50_sanity.remote()), indent=2, sort_keys=True))
    elif stage == "q55-pixel-oracle-av":
        print(json.dumps(save_returned_archive(q55_pixel_oracle_av.remote()), indent=2, sort_keys=True))
    elif stage == "q55-pixel-oracle-probe-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_pixel_oracle_av.remote(
                        max_samples=64,
                        steps=100,
                        opt_batch_size=2,
                        max_delta=4.0,
                        lr=0.005,
                        pose_term_weight=20.0,
                        tag="q55_pixel_oracle_delta4_64s_100steps_b2_camera_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-pixel-oracle-sweep-av":
        configs = [
            {
                "tag": "q55_pixel_oracle_sweep_hard_d4_lr002_p30_64s",
                "max_delta": 4.0,
                "lr": 0.002,
                "pose_term_weight": 30.0,
                "hard_pixels_only": True,
                "hard_pixel_boost": 256.0,
            },
            {
                "tag": "q55_pixel_oracle_sweep_full_d4_lr002_p30_hb64_64s",
                "max_delta": 4.0,
                "lr": 0.002,
                "pose_term_weight": 30.0,
                "hard_pixels_only": False,
                "hard_pixel_boost": 64.0,
            },
            {
                "tag": "q55_pixel_oracle_sweep_hard_d8_lr001_p40_64s",
                "max_delta": 8.0,
                "lr": 0.001,
                "pose_term_weight": 40.0,
                "hard_pixels_only": True,
                "hard_pixel_boost": 256.0,
            },
        ]
        results = []
        for cfg in configs:
            results.append(
                save_returned_archive(
                    q55_pixel_oracle_av.remote(
                        offset=0,
                        max_samples=64,
                        steps=100,
                        opt_batch_size=2,
                        early_stop_patience=35,
                        **cfg,
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-pixel-oracle-tail-av":
        configs = [
            {
                "tag": "q55_pixel_oracle_tail_o56_full_d4_lr0015_p40_hb64_8s",
                "max_delta": 4.0,
                "lr": 0.0015,
                "pose_term_weight": 40.0,
                "hard_pixels_only": False,
                "hard_pixel_boost": 64.0,
            },
            {
                "tag": "q55_pixel_oracle_tail_o56_full_d8_lr001_p60_hb64_8s",
                "max_delta": 8.0,
                "lr": 0.001,
                "pose_term_weight": 60.0,
                "hard_pixels_only": False,
                "hard_pixel_boost": 64.0,
            },
            {
                "tag": "q55_pixel_oracle_tail_o56_hard_d8_lr001_p60_8s",
                "max_delta": 8.0,
                "lr": 0.001,
                "pose_term_weight": 60.0,
                "hard_pixels_only": True,
                "hard_pixel_boost": 256.0,
            },
        ]
        results = []
        for cfg in configs:
            results.append(
                save_returned_archive(
                    q55_pixel_oracle_av.remote(
                        offset=56,
                        max_samples=8,
                        steps=300,
                        opt_batch_size=2,
                        early_stop_patience=80,
                        **cfg,
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-pixel-oracle-tail-pose-av":
        configs = [
            {
                "tag": "q55_pixel_oracle_tail_pose_o56_d8_lr0005_p200_m50_8s",
                "max_delta": 8.0,
                "lr": 0.0005,
                "pose_term_weight": 200.0,
                "pose_mse_weight": 50.0,
                "seg_ce_weight": 0.0,
                "seg_margin_weight": 0.0,
                "hard_pixels_only": False,
                "hard_pixel_boost": 0.0,
            },
            {
                "tag": "q55_pixel_oracle_tail_pose_o56_d16_lr0003_p300_m100_8s",
                "max_delta": 16.0,
                "lr": 0.0003,
                "pose_term_weight": 300.0,
                "pose_mse_weight": 100.0,
                "seg_ce_weight": 0.0,
                "seg_margin_weight": 0.0,
                "hard_pixels_only": False,
                "hard_pixel_boost": 0.0,
            },
        ]
        results = []
        for cfg in configs:
            results.append(
                save_returned_archive(
                    q55_pixel_oracle_av.remote(
                        offset=56,
                        max_samples=8,
                        steps=500,
                        opt_batch_size=2,
                        early_stop_patience=120,
                        **cfg,
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-pose-control-oracle-tail-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_pose_control_oracle_av.remote(
                        indices="59,60,62",
                        controls="baseline,orig_f1,lowres_y,lowres_rgb,affine",
                        tag="q55_pose_control_indices_59_60_62_orig_lowres_affine_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-pose-control-oracle-tail-patch-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_pose_control_oracle_av.remote(
                        indices="59,60,62",
                        controls="baseline,patch",
                        patch_steps=500,
                        patch_size="64x64",
                        patch_grid="12x12",
                        tag="q55_pose_control_indices_59_60_62_patch64_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-pose-control-oracle-o56-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_pose_control_oracle_av.remote(
                        indices="",
                        offset=56,
                        max_samples=8,
                        controls="baseline,orig_f1,lowres_y,lowres_rgb,affine",
                        tag="q55_pose_control_o56_8s_orig_lowres_affine_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-pose-table-oracle-tail-av":
        results = []
        for scale in (0.5, 1.0, 2.0):
            results.append(
                save_returned_archive(
                    q55_pose_table_oracle_av.remote(
                        indices="59,60,62",
                        mode="cem_adam",
                        param_mode="bounded",
                        scale=scale,
                        steps=250,
                        lr=0.05,
                        cem_candidates=512,
                        cem_iterations=2,
                        opt_batch_size=3,
                        tag=f"q55_pose_table_indices_59_60_62_cem_adam_s{str(scale).replace('.', 'p')}_av",
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-pose-table-oracle-o56-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_pose_table_oracle_av.remote(
                        indices="",
                        offset=56,
                        max_samples=8,
                        mode="cem_adam",
                        param_mode="bounded",
                        scale=1.0,
                        steps=300,
                        lr=0.05,
                        cem_candidates=512,
                        cem_iterations=2,
                        opt_batch_size=4,
                        tag="q55_pose_table_o56_8s_cem_adam_s1p0_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-pose-table-oracle-tail-unbounded-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_pose_table_oracle_av.remote(
                        indices="59,60,62",
                        mode="cem_adam",
                        param_mode="unbounded",
                        scale=2.0,
                        steps=300,
                        lr=0.03,
                        cem_candidates=768,
                        cem_iterations=2,
                        opt_batch_size=3,
                        tag="q55_pose_table_indices_59_60_62_cem_adam_unbounded_s2p0_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-student-frontier-smoke-av":
        configs = [
            {
                "budget": "S16",
                "tag": "q55_student_S16_64s_smoke_av",
            },
            {
                "budget": "S28",
                "tag": "q55_student_S28_64s_smoke_av",
            },
        ]
        results = []
        for cfg in configs:
            results.append(
                save_returned_archive(
                    q55_student_frontier_av.remote(
                        offset=0,
                        max_samples=64,
                        teacher_steps=80,
                        task_steps=120,
                        qat_steps=40,
                        batch_size=2,
                        **cfg,
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-student-capacity8-av":
        configs = [
            {
                "budget": "S28",
                "teacher_init": False,
                "tag": "q55_student_S28_8s_teacher2000_random_av",
            },
            {
                "budget": "S28",
                "teacher_init": True,
                "tag": "q55_student_S28_8s_teacher2000_prunedinit_av",
            },
            {
                "budget": "S40",
                "teacher_init": True,
                "tag": "q55_student_S40_8s_teacher2000_prunedinit_av",
            },
        ]
        results = []
        for cfg in configs:
            results.append(
                save_returned_archive(
                    q55_student_frontier_av.remote(
                        offset=0,
                        max_samples=8,
                        teacher_steps=2000,
                        task_steps=0,
                        qat_steps=0,
                        batch_size=2,
                        **cfg,
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-student-task8-av":
        configs = [
            {
                "tag": "q55_student_S40_8s_taskoverfit_pose300_av",
                "teacher_steps": 600,
                "task_steps": 3000,
                "task_lr": 3e-4,
                "frame_anchor_weight": 0.05,
                "pose_mse_weight": 300.0,
            },
            {
                "tag": "q55_student_S40_8s_taskoverfit_pose600_av",
                "teacher_steps": 600,
                "task_steps": 3000,
                "task_lr": 4e-4,
                "frame_anchor_weight": 0.02,
                "pose_mse_weight": 600.0,
            },
        ]
        results = []
        for cfg in configs:
            results.append(
                save_returned_archive(
                    q55_student_frontier_av.remote(
                        budget="S40",
                        offset=0,
                        max_samples=8,
                        qat_steps=0,
                        batch_size=2,
                        teacher_init=True,
                        **cfg,
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-student-task8-stable-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_student_frontier_av.remote(
                        budget="S40",
                        offset=0,
                        max_samples=8,
                        teacher_steps=2000,
                        task_steps=1000,
                        qat_steps=0,
                        batch_size=2,
                        teacher_init=True,
                        task_lr=5e-5,
                        frame_anchor_weight=0.25,
                        pose_mse_weight=80.0,
                        tag="q55_student_S40_8s_task1000_stable_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "q55-student-frontier-64-av":
        configs = [
            ("S8", 250, 400, 100),
            ("S16", 300, 500, 150),
            ("S28", 350, 600, 175),
            ("S40", 350, 650, 200),
        ]
        results = []
        for budget, teacher_steps, task_steps, qat_steps in configs:
            results.append(
                save_returned_archive(
                    q55_student_frontier_av.remote(
                        budget=budget,
                        offset=0,
                        max_samples=64,
                        teacher_steps=teacher_steps,
                        task_steps=task_steps,
                        qat_steps=qat_steps,
                        batch_size=2,
                        tag=f"q55_student_{budget}_64s_frontier_av",
                    )
                )
            )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif stage == "q55-pixel-oracle-smoke-av":
        print(
            json.dumps(
                save_returned_archive(
                    q55_pixel_oracle_av.remote(
                        offset=0,
                        max_samples=4,
                        steps=5,
                        opt_batch_size=2,
                        tag="q55_pixel_oracle_delta4_4s_5steps_b2_camera_av",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif stage == "dali-smoke":
        print(json.dumps(dali_video_smoke.remote(), indent=2, sort_keys=True))
    else:
        raise ValueError(
            "stage must be q0, q0-av, q1, q1-av, qcrf-av, qmask-av, "
            "q3-crf54-av, q3-crf56-av, qpack-recover-int10-av, "
            "qpack-recover-int12-av, q55-polish-int10-fast-av, "
            "q55-polish-int12-fast-av, q55-polish-int10-reflect-av, "
            "q55-polish-int10-replicate-av, qmask-denoise-qrecode50, "
            "qmask-adapter-qrecode50, q55-pixel-oracle-av, "
            "q55-pixel-oracle-probe-av, q55-pixel-oracle-sweep-av, "
            "q55-pixel-oracle-tail-av, q55-pixel-oracle-tail-pose-av, "
            "q55-pose-control-oracle-tail-av, q55-pose-control-oracle-tail-patch-av, "
            "q55-pose-control-oracle-o56-av, "
            "q55-student-frontier-smoke-av, q55-student-capacity8-av, "
            "q55-student-task8-av, q55-student-task8-stable-av, "
            "q55-student-frontier-64-av, "
            "q55-pixel-oracle-smoke-av, "
            "qrecode50-full-recover-av, or dali-smoke"
        )
