#!/usr/bin/env python
"""Validate and reproduce official-style local evaluator runs.

This is intentionally wrapped around the repository root ``evaluate.sh`` path.
Search/proxy evaluators are useful for ranking experiments, but trusted
submission scoring should go through:

  archive.zip -> inflate.sh -> inflated/*.raw -> evaluate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SEARCH_DIR / "experiments" / "evaluator_validation"
ORIGINAL_BYTES = 37_545_489

REPORT_PATTERNS = {
    "samples": re.compile(r"Evaluation results over\s+([0-9,]+)\s+samples"),
    "posenet_dist": re.compile(r"Average PoseNet Distortion:\s+([0-9.eE+-]+)"),
    "segnet_dist": re.compile(r"Average SegNet Distortion:\s+([0-9.eE+-]+)"),
    "archive_bytes": re.compile(r"Submission file size:\s+([0-9,]+)\s+bytes"),
    "original_bytes": re.compile(r"Original uncompressed size:\s+([0-9,]+)\s+bytes"),
    "rate": re.compile(r"Compression Rate:\s+([0-9.eE+-]+)"),
    "printed_score": re.compile(r"Final score:.*=\s+([0-9.]+)"),
}


REFERENCE_METRICS = {
    # These are local full-root-evaluator calibration values from this machine.
    # They are not proxy metrics and are intentionally rounded to report precision.
    "q55_fp16_pose_int10": {
        "archive_bytes": 288_268,
        "posenet_dist": 0.00065135,
        "segnet_dist": 0.00072222,
    },
    "q55_fp16_pose_int12": {
        "archive_bytes": 289_127,
        "posenet_dist": 0.00064976,
        "segnet_dist": 0.00072222,
    },
    "q55_fp16_only": {
        "archive_bytes": 296_659,
        "posenet_dist": 0.00064992,
        "segnet_dist": 0.00072222,
    },
    "selfcomp_pr56_eval": {
        "archive_bytes": 279_036,
        "posenet_dist": 0.00039916,
        "segnet_dist": 0.00115278,
    },
}

DEFAULT_SUITE = [
    "submissions/q55_fp16_pose_int10",
    "submissions/q55_fp16_pose_int12",
    "submissions/q55_manual",
    "submissions/selfcomp_pr56_eval",
]


def quality(segnet_dist: float, posenet_dist: float) -> float:
    return 100.0 * float(segnet_dist) + math.sqrt(max(0.0, 10.0 * float(posenet_dist)))


def score(segnet_dist: float, posenet_dist: float, archive_bytes: int, original_bytes: int) -> float:
    return quality(segnet_dist, posenet_dist) + 25.0 * int(archive_bytes) / int(original_bytes)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_int(text: str) -> int:
    return int(text.replace(",", ""))


def parse_report(path: Path) -> dict[str, Any]:
    text = path.read_text()
    out: dict[str, Any] = {"report_path": str(path)}
    for key, pattern in REPORT_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1)
        if key in {"samples", "archive_bytes", "original_bytes"}:
            out[key] = parse_int(value)
        else:
            out[key] = float(value)
    missing = [key for key in ("samples", "posenet_dist", "segnet_dist", "archive_bytes", "original_bytes") if key not in out]
    if missing:
        raise ValueError(f"missing fields in {path}: {missing}")
    out["quality"] = quality(out["segnet_dist"], out["posenet_dist"])
    out["score_full_precision"] = score(
        out["segnet_dist"],
        out["posenet_dist"],
        out["archive_bytes"],
        out["original_bytes"],
    )
    out["rate_full_precision"] = out["archive_bytes"] / out["original_bytes"]
    return out


def archive_summary(submission_dir: Path) -> dict[str, Any]:
    archive = submission_dir / "archive.zip"
    if not archive.exists():
        raise FileNotFoundError(f"missing archive.zip: {archive}")
    payloads = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in sorted(zf.infolist(), key=lambda item: item.filename):
            payloads.append(
                {
                    "filename": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "crc": f"{info.CRC:08x}",
                }
            )
    return {
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "payloads": payloads,
    }


def file_sha(path: Path) -> str | None:
    return sha256(path) if path.exists() else None


def environment_summary() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        torch_info = {"import_error": repr(exc)}
    else:
        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "mps_available": bool(torch.backends.mps.is_available()),
        }
        if torch.cuda.is_available():
            torch_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_info,
        "evaluate_py_sha256": file_sha(REPO_ROOT / "evaluate.py"),
        "evaluate_sh_sha256": file_sha(REPO_ROOT / "evaluate.sh"),
        "modules_py_sha256": file_sha(REPO_ROOT / "modules.py"),
        "posenet_sha256": file_sha(REPO_ROOT / "models/posenet.safetensors"),
        "segnet_sha256": file_sha(REPO_ROOT / "models/segnet.safetensors"),
        "video_names_sha256": file_sha(REPO_ROOT / "public_test_video_names.txt"),
    }


def validate_inflated_layout(submission_dir: Path, video_names_file: Path) -> dict[str, Any]:
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    inflated = submission_dir / "inflated"
    from frame_utils import camera_size, seq_len

    expected_frame_bytes = camera_size[1] * camera_size[0] * 3
    expected_raw_bytes_per_file = seq_len * expected_frame_bytes
    files = []
    missing = []
    bad_size = []
    for name in names:
        raw = inflated / f"{Path(name).stem}.raw"
        if not raw.exists():
            missing.append(str(raw))
            continue
        size = raw.stat().st_size
        row = {"path": str(raw), "bytes": size, "expected_bytes": expected_raw_bytes_per_file}
        files.append(row)
        if size != expected_raw_bytes_per_file:
            bad_size.append(row)
    return {
        "inflated_dir": str(inflated),
        "video_count": len(names),
        "files": files,
        "missing": missing,
        "bad_size": bad_size,
        "ok": not missing and not bad_size,
    }


def compare_reference(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    ref = REFERENCE_METRICS.get(name)
    if not ref:
        return {"available": False}
    deltas = {
        "archive_bytes": int(metrics["archive_bytes"]) - int(ref["archive_bytes"]),
        "posenet_dist": float(metrics["posenet_dist"]) - float(ref["posenet_dist"]),
        "segnet_dist": float(metrics["segnet_dist"]) - float(ref["segnet_dist"]),
    }
    tolerances = {
        "archive_bytes": 0,
        "posenet_dist": 5e-6,
        "segnet_dist": 5e-6,
    }
    passed = all(abs(deltas[key]) <= tolerances[key] for key in tolerances)
    return {
        "available": True,
        "reference": ref,
        "deltas": deltas,
        "tolerances": tolerances,
        "pass": passed,
    }


def submission_name(submission_dir: Path) -> str:
    return submission_dir.resolve().name


def run_evaluate_sh(
    *,
    submission_dir: Path,
    device: str,
    video_names_file: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    cmd = [
        "bash",
        str(REPO_ROOT / "evaluate.sh"),
        "--submission-dir",
        str(submission_dir),
        "--video-names-file",
        str(video_names_file),
        "--device",
        device,
    ]
    env = os.environ.copy()
    venv_bin = REPO_ROOT / ".venv" / "bin"
    if venv_bin.exists():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_time_sec": time.time() - start,
        "stdout": proc.stdout,
    }


def validate_one(
    *,
    submission_dir: Path,
    device: str,
    video_names_file: Path,
    out_dir: Path,
    run: bool,
    timeout_sec: int,
) -> dict[str, Any]:
    submission_dir = submission_dir.resolve()
    name = submission_name(submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "name": name,
        "submission_dir": str(submission_dir),
        "device": device,
        "run_requested": run,
        "environment": environment_summary(),
        "archive": archive_summary(submission_dir),
    }
    if run:
        eval_result = run_evaluate_sh(
            submission_dir=submission_dir,
            device=device,
            video_names_file=video_names_file,
            timeout_sec=timeout_sec,
        )
        result["evaluate_sh"] = eval_result
        if eval_result["returncode"] != 0:
            result["status"] = "evaluate_failed"
            result["failure_reason"] = f"evaluate.sh returned {eval_result['returncode']}"
            return result
    report = submission_dir / "report.txt"
    result["metrics"] = parse_report(report)
    result["layout"] = validate_inflated_layout(submission_dir, video_names_file)
    result["reference_compare"] = compare_reference(name, result["metrics"])
    result["status"] = "pass" if result["layout"]["ok"] and result["reference_compare"].get("pass", True) else "check"
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def command_parse_report(args: argparse.Namespace) -> None:
    data = parse_report(args.report)
    print(json.dumps(data, indent=2, sort_keys=True))


def command_run_one(args: argparse.Namespace) -> None:
    result = validate_one(
        submission_dir=args.submission_dir,
        device=args.device,
        video_names_file=args.video_names_file,
        out_dir=args.out_dir,
        run=not args.no_run,
        timeout_sec=args.timeout_sec,
    )
    out_path = args.out_dir / f"{result['name']}_{args.device}_validation.json"
    write_json(out_path, result)
    print(json.dumps({"out": str(out_path), "status": result["status"], "metrics": result.get("metrics")}, indent=2, sort_keys=True))


def command_run_suite(args: argparse.Namespace) -> None:
    rows = []
    for item in args.submissions:
        result = validate_one(
            submission_dir=(REPO_ROOT / item).resolve(),
            device=args.device,
            video_names_file=args.video_names_file,
            out_dir=args.out_dir,
            run=not args.no_run,
            timeout_sec=args.timeout_sec,
        )
        out_path = args.out_dir / f"{result['name']}_{args.device}_validation.json"
        write_json(out_path, result)
        metrics = result.get("metrics", {})
        rows.append(
            {
                "name": result["name"],
                "status": result["status"],
                "archive_bytes": metrics.get("archive_bytes"),
                "posenet_dist": metrics.get("posenet_dist"),
                "segnet_dist": metrics.get("segnet_dist"),
                "quality": metrics.get("quality"),
                "score": metrics.get("score_full_precision"),
                "reference_pass": result.get("reference_compare", {}).get("pass"),
                "out": str(out_path),
            }
        )
    summary = {
        "device": args.device,
        "run_requested": not args.no_run,
        "rows": rows,
        "environment": environment_summary(),
    }
    write_json(args.out_dir / f"suite_{args.device}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_summarize(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted(args.out_dir.glob("*_validation.json")):
        result = read_json(path)
        metrics = result.get("metrics", {})
        rows.append(
            {
                "name": result["name"],
                "device": result.get("device"),
                "status": result.get("status"),
                "archive_bytes": metrics.get("archive_bytes"),
                "posenet_dist": metrics.get("posenet_dist"),
                "segnet_dist": metrics.get("segnet_dist"),
                "quality": metrics.get("quality"),
                "score": metrics.get("score_full_precision"),
                "reference_pass": result.get("reference_compare", {}).get("pass"),
            }
        )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": rows}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    parse_p = sub.add_parser("parse-report")
    parse_p.add_argument("report", type=Path)
    parse_p.set_defaults(func=command_parse_report)

    one_p = sub.add_parser("run-one")
    one_p.add_argument("--submission-dir", type=Path, required=True)
    one_p.add_argument("--device", default="mps")
    one_p.add_argument("--video-names-file", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    one_p.add_argument("--timeout-sec", type=int, default=60 * 30)
    one_p.add_argument("--no-run", action="store_true", help="Parse existing report/inflated outputs without invoking evaluate.sh.")
    one_p.set_defaults(func=command_run_one)

    suite_p = sub.add_parser("run-suite")
    suite_p.add_argument("--submissions", nargs="+", default=DEFAULT_SUITE)
    suite_p.add_argument("--device", default="mps")
    suite_p.add_argument("--video-names-file", type=Path, default=REPO_ROOT / "public_test_video_names.txt")
    suite_p.add_argument("--timeout-sec", type=int, default=60 * 30)
    suite_p.add_argument("--no-run", action="store_true", help="Parse existing reports/inflated outputs without invoking evaluate.sh.")
    suite_p.set_defaults(func=command_run_suite)

    summarize_p = sub.add_parser("summarize")
    summarize_p.set_defaults(func=command_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
