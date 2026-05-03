#!/usr/bin/env python
"""Hard gates for Quantizr #55 restart metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fail(msg: str) -> None:
    raise SystemExit(f"GATE FAIL: {msg}")


def cmd_q0(args) -> None:
    upstream = load(args.upstream)
    modified = load(args.modified)
    score = upstream["score"]
    if not (args.min_score <= score <= args.max_score):
        fail(f"upstream score {score:.6f} outside [{args.min_score}, {args.max_score}]")
    for key, tol in (
        ("score", args.score_tol),
        ("segnet_dist", args.seg_tol),
        ("posenet_dist", args.pose_tol),
    ):
        delta = abs(modified[key] - upstream[key])
        if delta > tol:
            fail(f"modified {key} delta {delta:.8f} > {tol}")
    print("Q0 gate passed")


def cmd_crf(args) -> None:
    crf50 = load(args.crf50)
    crf52 = load(args.crf52)
    crf54 = load(args.crf54)
    base = load(args.base) if args.base else crf50
    for key, tol in (
        ("score", args.score_tol),
        ("segnet_dist", args.seg_tol),
        ("posenet_dist", args.pose_tol),
    ):
        delta = abs(crf50[key] - base[key])
        if delta > tol:
            fail(f"CRF50 {key} delta {delta:.8f} > {tol}")
    if crf52["quality_term"] > args.crf52_quality_max:
        fail(f"CRF52 quality {crf52['quality_term']:.6f} > {args.crf52_quality_max}")
    if crf54["quality_term"] > args.crf54_quality_max:
        fail(f"CRF54 quality {crf54['quality_term']:.6f} > {args.crf54_quality_max}")
    print("QCRF gate passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    q0 = sub.add_parser("q0")
    q0.add_argument("--upstream", type=Path, required=True)
    q0.add_argument("--modified", type=Path, required=True)
    q0.add_argument("--min-score", type=float, default=0.332)
    q0.add_argument("--max-score", type=float, default=0.334)
    q0.add_argument("--score-tol", type=float, default=0.003)
    q0.add_argument("--seg-tol", type=float, default=0.00005)
    q0.add_argument("--pose-tol", type=float, default=0.00005)
    q0.set_defaults(func=cmd_q0)

    crf = sub.add_parser("crf")
    crf.add_argument("--base", type=Path)
    crf.add_argument("--crf50", type=Path, required=True)
    crf.add_argument("--crf52", type=Path, required=True)
    crf.add_argument("--crf54", type=Path, required=True)
    crf.add_argument("--score-tol", type=float, default=0.006)
    crf.add_argument("--seg-tol", type=float, default=0.00008)
    crf.add_argument("--pose-tol", type=float, default=0.00008)
    crf.add_argument("--crf52-quality-max", type=float, default=0.160)
    crf.add_argument("--crf54-quality-max", type=float, default=0.180)
    crf.set_defaults(func=cmd_crf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
