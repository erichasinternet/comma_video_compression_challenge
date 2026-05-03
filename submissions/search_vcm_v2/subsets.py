#!/usr/bin/env python
"""Fixed subsets for Search VCM v2, based on qpose14 ledger when available."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm_v2.evaluator import QPOSE14_LEDGER, load_jsonl, write_json


HARD3 = [59, 60, 62]
HARD8 = [59, 60, 62, 56, 57, 58, 61, 63]
FULL600 = list(range(600))


def validate_subset(indices: list[int]) -> None:
    if len(indices) != len(set(indices)):
        raise ValueError(f"subset has duplicates: {indices}")
    invalid = [idx for idx in indices if idx < 0 or idx >= 600]
    if invalid:
        raise ValueError(f"subset has invalid sample ids: {invalid}")


def strat64_from_qpose14(ledger_path: Path = QPOSE14_LEDGER, *, seed: int = 63) -> list[int]:
    rows = load_jsonl(ledger_path)
    if not rows:
        rng = random.Random(seed)
        rest = [idx for idx in FULL600 if idx not in HARD8]
        subset = HARD8 + rng.sample(rest, 64 - len(HARD8))
        validate_subset(subset)
        return subset
    ranked = sorted(rows, key=lambda row: float(row["qpose14_quality"]), reverse=True)
    top32 = [int(row["sample_id"]) for row in ranked[:32]]
    remaining = [idx for idx in FULL600 if idx not in set(top32)]
    rng = random.Random(seed)
    bins = [remaining[i::32] for i in range(32)]
    complement = [rng.choice(items) for items in bins if items]
    subset = top32 + complement[: 64 - len(top32)]
    validate_subset(subset)
    return subset


def get_subset(name: str, ledger_path: Path = QPOSE14_LEDGER) -> list[int]:
    if name == "smoke":
        return [0, 1]
    if name == "hard3":
        return list(HARD3)
    if name == "hard8" or name == "hard8_capacity" or name == "hard8_compressed":
        return list(HARD8)
    if name == "strat64":
        return strat64_from_qpose14(ledger_path)
    if name == "full600":
        return list(FULL600)
    if name == "packability":
        return list(HARD8)
    raise ValueError(f"unknown subset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["smoke", "hard3", "hard8", "hard8_capacity", "hard8_compressed", "packability", "strat64", "full600"])
    parser.add_argument("--ledger", type=Path, default=QPOSE14_LEDGER)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    subset = get_subset(args.name, args.ledger)
    payload = {"name": args.name, "indices": subset, "count": len(subset)}
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
