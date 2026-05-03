#!/usr/bin/env python3
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


RUN_RE = re.compile(r"Run:\s+(\w+)\s+\|\s+Epoch\s+(\d+)/(\d+)")
START_RE = re.compile(r"STARTING PIPELINE RUN:\s+(\w+)")


def parse_args():
    p = argparse.ArgumentParser(description="Watch a pose-gate candidate and stop it on failing proxy thresholds.")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--log-path", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--run3-max-posenet", type=float, default=0.006)
    p.add_argument("--run6-epoch-threshold", type=int, default=80)
    p.add_argument("--run6-max-posenet", type=float, default=0.002)
    p.add_argument("--run6-max-proxy-score", type=float, default=0.45)
    return p.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")


def read_metrics(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_progress(log_text: str):
    progress = {}
    started = set()
    for name in START_RE.findall(log_text):
        started.add(name)
    for name, epoch_s, total_s in RUN_RE.findall(log_text):
        epoch = int(epoch_s)
        total = int(total_s)
        current = progress.get(name, {"max_epoch": 0, "total": total})
        current["max_epoch"] = max(current["max_epoch"], epoch)
        current["total"] = total
        progress[name] = current
    return progress, started


def terminate(pid: int):
    subprocess.run(["pkill", "-P", str(pid)], check=False)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    time.sleep(5)
    if pid_alive(pid):
        subprocess.run(["pkill", "-9", "-P", str(pid)], check=False)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def main():
    args = parse_args()
    reason_path = args.archive_dir / "watcher_stop_reason.txt"
    run3_metrics_path = args.archive_dir / "run3_finetune_best_proxy_metrics.json"
    run6_metrics_path = args.archive_dir / "run6_pose_rescue_best_proxy_metrics.json"

    print(f"[watch] pid={args.pid} log={args.log_path} archive={args.archive_dir}", flush=True)

    while True:
        if not pid_alive(args.pid):
            print("[watch] target process exited", flush=True)
            return 0

        text = read_text(args.log_path)
        progress, started = parse_progress(text)

        run3_metrics = read_metrics(run3_metrics_path)
        run6_metrics = read_metrics(run6_metrics_path)

        run3_progress = progress.get("run3_finetune")
        run6_progress = progress.get("run6_pose_rescue")

        run4_started = "run4_finish" in started
        run3_finished = bool(run3_progress and run3_progress["max_epoch"] >= run3_progress["total"])
        if run3_metrics and (run4_started or run3_finished):
            if run3_metrics["posenet_dist"] > args.run3_max_posenet:
                reason = (
                    f"Stopped after run3_finetune: posenet_dist={run3_metrics['posenet_dist']:.6f} "
                    f"> {args.run3_max_posenet:.6f}"
                )
                reason_path.write_text(reason + "\n")
                print(f"[watch] {reason}", flush=True)
                terminate(args.pid)
                return 0

        if run6_metrics and run6_progress:
            if run6_progress["max_epoch"] >= args.run6_epoch_threshold and run6_metrics["posenet_dist"] > args.run6_max_posenet:
                reason = (
                    f"Stopped at run6_pose_rescue epoch {run6_progress['max_epoch']}: "
                    f"posenet_dist={run6_metrics['posenet_dist']:.6f} > {args.run6_max_posenet:.6f}"
                )
                reason_path.write_text(reason + "\n")
                print(f"[watch] {reason}", flush=True)
                terminate(args.pid)
                return 0

            if run6_progress["max_epoch"] >= run6_progress["total"] and run6_metrics["proxy_score"] > args.run6_max_proxy_score:
                reason = (
                    f"Stopped after run6_pose_rescue: proxy_score={run6_metrics['proxy_score']:.5f} "
                    f"> {args.run6_max_proxy_score:.5f}"
                )
                reason_path.write_text(reason + "\n")
                print(f"[watch] {reason}", flush=True)
                terminate(args.pid)
                return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
