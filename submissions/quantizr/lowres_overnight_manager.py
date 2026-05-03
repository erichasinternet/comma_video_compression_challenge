#!/usr/bin/env python
import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RUN_LOG_DIR = ROOT_DIR / "run_logs"

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
PROXY_RE = re.compile(
    r"\[Proxy\]\s+Score:\s+([0-9.]+)\s+\|\s+Seg\(x100\):\s+([0-9.]+)\s+\|\s+Pose\(.*?\):\s+([0-9.]+)"
)
FINAL_RE = re.compile(r"Final official score:\s+([0-9.]+)")


def parse_log(path: Path) -> dict[str, float | int | None]:
    status: dict[str, float | int | None] = {
        "epoch": None,
        "total_epochs": None,
        "proxy": None,
        "seg_x100": None,
        "pose_sqrt10": None,
        "final_score": None,
    }
    if not path.exists():
        return status
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return status

    for match in EPOCH_RE.finditer(text):
        status["epoch"] = int(match.group(1))
        status["total_epochs"] = int(match.group(2))
    for match in PROXY_RE.finditer(text):
        status["proxy"] = float(match.group(1))
        status["seg_x100"] = float(match.group(2))
        status["pose_sqrt10"] = float(match.group(3))
    for match in FINAL_RE.finditer(text):
        status["final_score"] = float(match.group(1))
    return status


def gate_reason(candidate: str, status: dict[str, float | int | None]) -> str | None:
    epoch = status["epoch"]
    proxy = status["proxy"]
    if epoch is None or proxy is None:
        return None

    epoch = int(epoch)
    proxy = float(proxy)
    if proxy > 2.0:
        return f"catastrophic proxy {proxy:.4f} at epoch {epoch}"
    if epoch >= 4 and proxy > 0.75:
        return f"proxy {proxy:.4f} still above 0.75 at epoch {epoch}"

    if candidate.lower() in {"lr128", "lr96"}:
        gates = [(8, 0.45), (16, 0.32), (32, 0.25)]
    else:
        gates = [(8, 0.55), (16, 0.40), (32, 0.31)]
    for min_epoch, max_proxy in gates:
        if epoch >= min_epoch and proxy > max_proxy:
            return f"proxy {proxy:.4f} above {max_proxy:.2f} at epoch {epoch}"
    return None


def terminate_process(process: subprocess.Popen, reason: str, log_file):
    print(f"[manager] stopping pid={process.pid}: {reason}", flush=True)
    print(f"[manager] stopping pid={process.pid}: {reason}", file=log_file, flush=True)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def maybe_stop_pod():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("[manager] RUNPOD_POD_ID is not set; skipping pod stop.", flush=True)
        return
    try:
        subprocess.run(["runpodctl", "stop", "pod", pod_id], check=False)
    except FileNotFoundError:
        print("[manager] runpodctl not found; skipping pod stop.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run low-res mask candidates overnight with hard proxy gates.")
    parser.add_argument("--candidates", nargs="+", default=["LR128", "LR96", "LR192"])
    parser.add_argument("--archive-root", type=Path, default=ROOT_DIR / "submissions/quantizr/experiments/gpt_lowres_mask")
    parser.add_argument("--runner", type=Path, default=ROOT_DIR / "submissions/quantizr/run_lowres_mask_probe.sh")
    parser.add_argument("--deadline-hours", type=float, default=12.0)
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--stop-pod-on-exit", action="store_true")
    args = parser.parse_args()

    RUN_LOG_DIR.mkdir(exist_ok=True)
    deadline = time.time() + args.deadline_hours * 3600
    manager_log = RUN_LOG_DIR / "lowres_overnight_manager.log"
    with manager_log.open("a", buffering=1) as mlog:
        print(f"[manager] started candidates={args.candidates} deadline_hours={args.deadline_hours}", flush=True)
        print(f"[manager] started candidates={args.candidates} deadline_hours={args.deadline_hours}", file=mlog, flush=True)

        for candidate in args.candidates:
            if time.time() >= deadline:
                print("[manager] deadline reached before next candidate.", flush=True)
                break

            log_path = RUN_LOG_DIR / f"gpt_lowres_{candidate.lower()}.log"
            pid_path = RUN_LOG_DIR / f"gpt_lowres_{candidate.lower()}.pid"
            cmd = [str(args.runner), candidate, str(args.archive_root)]
            print(f"[manager] launching {' '.join(cmd)}", flush=True)
            print(f"[manager] launching {' '.join(cmd)}", file=mlog, flush=True)
            with log_path.open("a", buffering=1) as lf:
                process = subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=lf, stderr=subprocess.STDOUT)
                pid_path.write_text(f"{process.pid}\n")
                final_status: dict[str, float | int | None] = {}

                while True:
                    time.sleep(args.poll_seconds)
                    final_status = parse_log(log_path)
                    print(f"[manager] {candidate} pid={process.pid} status={final_status}", flush=True)
                    print(f"[manager] {candidate} pid={process.pid} status={final_status}", file=mlog, flush=True)

                    reason = gate_reason(candidate, final_status)
                    if reason is not None:
                        terminate_process(process, reason, mlog)
                        break

                    if process.poll() is not None:
                        print(f"[manager] {candidate} exited rc={process.returncode}", flush=True)
                        print(f"[manager] {candidate} exited rc={process.returncode}", file=mlog, flush=True)
                        break

                    if time.time() >= deadline:
                        terminate_process(process, "hard deadline reached", mlog)
                        break

                final_status = parse_log(log_path)
                final_score = final_status.get("final_score")
                if isinstance(final_score, float) and final_score < 0.30:
                    print(f"[manager] keeping completed {candidate}: final_score={final_score:.5f}", flush=True)
                    print(f"[manager] keeping completed {candidate}: final_score={final_score:.5f}", file=mlog, flush=True)
                    break

        print("[manager] finished.", flush=True)
        print("[manager] finished.", file=mlog, flush=True)

    if args.stop_pod_on_exit:
        maybe_stop_pod()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)
