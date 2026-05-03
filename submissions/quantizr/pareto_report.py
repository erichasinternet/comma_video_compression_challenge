#!/usr/bin/env python
"""Render Pareto control audit metrics into a concise Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from q55_common import ORIGINAL_BYTES


DEFAULT_METRICS = Path(__file__).resolve().parent / "experiments/pareto_control_lab/channel_audit/metrics.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "experiments/pareto_control_lab/PARETO_CONTROL_REPORT.md"


def rate_term(archive_bytes: int) -> float:
    return 25.0 * int(archive_bytes) / ORIGINAL_BYTES


def render(metrics: dict) -> str:
    archive_path = metrics.get("base_archive")
    archive_bytes = Path(archive_path).stat().st_size if archive_path else 0
    baseline = metrics["baseline"]
    lines = [
        "# Pareto Control Audit",
        "",
        "## Baseline",
        "",
        f"- Archive bytes: `{archive_bytes:,}`",
        f"- Quality: `{baseline['quality']:.8f}`",
        f"- SegNet term: `{baseline['seg_term']:.8f}`",
        f"- PoseNet term: `{baseline['pose_term']:.8f}`",
        f"- Rate term: `{rate_term(archive_bytes):.8f}`",
        "",
        "## Controls",
        "",
        "| control | target | bytes est | best quality delta | best score delta | pose ok | conflict rate | decision |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for control in metrics.get("controls", []):
        best = control.get("best")
        if best:
            q_delta = best["quality_delta"]
            s_delta = best["score_delta"]
            pose_ok = str(best["pose_cap_ok"])
            decision = "positive" if s_delta < 0 and best["pose_cap_ok"] else "reject"
        else:
            q_delta = None
            s_delta = None
            pose_ok = "-"
            decision = "reject"
        lines.append(
            "| {control} | {target} | {bytes_est} | {q_delta} | {s_delta} | {pose_ok} | {conflict:.3f} | {decision} |".format(
                control=control["control"],
                target=control.get("target", ""),
                bytes_est=control.get("estimated_bytes", 0),
                q_delta="-" if q_delta is None else f"{q_delta:.8f}",
                s_delta="-" if s_delta is None else f"{s_delta:.8f}",
                pose_ok=pose_ok,
                conflict=float(control.get("gradient_conflict_rate") or 0.0),
                decision=decision,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{metrics.get('decision', 'unknown')}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    report = render(metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
