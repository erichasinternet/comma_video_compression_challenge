#!/usr/bin/env python
"""Stop a training process when proxy scores miss configured recovery gates."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


PROXY_RE = re.compile(
    r"\[Proxy\]\s+Score:\s+([0-9.]+)\s+\|\s+Seg\(x100\):\s+([0-9.]+)\s+\|\s+Pose\(.*?\):\s+([0-9.]+)"
)


def parse_gate(spec: str) -> tuple[int, float]:
    try:
        index_s, threshold_s = spec.split(":", 1)
        index = int(index_s)
        threshold = float(threshold_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gate must look like PROXY_INDEX:MAX_SCORE, e.g. 1:1.80") from exc
    if index <= 0:
        raise argparse.ArgumentTypeError("proxy index must be >= 1")
    return index, threshold


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True, help="Parent process to terminate.")
    p.add_argument("--log-path", type=Path, required=True)
    p.add_argument("--reason-path", type=Path, default=None)
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument(
        "--gate",
        action="append",
        type=parse_gate,
        default=[(1, 1.80), (2, 1.20), (4, 0.75), (8, 0.45)],
        help="Gate as PROXY_INDEX:MAX_SCORE. Can be repeated.",
    )
    return p.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_bytes().replace(b"\x00", b"").replace(b"\r", b"\n").decode("utf-8", errors="replace")


def parse_proxies(text: str) -> list[tuple[float, float, float]]:
    return [(float(a), float(b), float(c)) for a, b, c in PROXY_RE.findall(text)]


def terminate_tree(pid: int):
    subprocess.run(["pkill", "-TERM", "-P", str(pid)], check=False)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    time.sleep(5)
    if pid_alive(pid):
        subprocess.run(["pkill", "-KILL", "-P", str(pid)], check=False)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def main():
    args = parse_args()
    reason_path = args.reason_path or args.log_path.with_suffix(args.log_path.suffix + ".stop_reason")
    gates = sorted(args.gate)
    last_count = 0
    print(f"[proxy-watch] pid={args.pid} log={args.log_path} gates={gates}", flush=True)

    while True:
        if not pid_alive(args.pid):
            print("[proxy-watch] target process exited", flush=True)
            return 0

        proxies = parse_proxies(read_log(args.log_path))
        if len(proxies) != last_count:
            last_count = len(proxies)
            score, seg, pose = proxies[-1]
            print(
                f"[proxy-watch] proxy#{last_count} score={score:.5f} seg_x100={seg:.5f} pose_sqrt10={pose:.5f}",
                flush=True,
            )

        for proxy_index, max_score in gates:
            if len(proxies) >= proxy_index:
                score = proxies[proxy_index - 1][0]
                if score > max_score:
                    reason = f"proxy#{proxy_index} score={score:.5f} > {max_score:.5f}"
                    reason_path.write_text(reason + "\n")
                    print(f"[proxy-watch] stopping: {reason}", flush=True)
                    terminate_tree(args.pid)
                    return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)
