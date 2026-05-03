#!/usr/bin/env python3
"""VQ task-token trainer placeholder.

This file exists to keep the experiment ladder explicit. Do not run VQ until
`train_capacity.py` passes the hard8 and 64-sample float gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "blocked",
        "reason": "VQ training is intentionally gated on float-token capacity passing first.",
        "capacity_checkpoint": str(args.capacity_checkpoint),
        "next_gate": "hard8 quality <=0.090 after capacity training, then 64-sample quality <=0.120",
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
