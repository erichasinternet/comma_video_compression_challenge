#!/usr/bin/env python
"""Fixed comparable subsets for Search VCM."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions.search_vcm.evaluator import EXPERIMENTS_DIR, load_jsonl, write_json

HARD3 = [59, 60, 62]
HARD8 = [59, 60, 62, 56, 57, 58, 61, 63]
FULL600 = list(range(600))
DEFAULT_LEDGER = EXPERIMENTS_DIR / "base_q55_fp16_pose_int10_per_sample.jsonl"


def validate_subset(indices: list[int]) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError(f"subset has duplicates: {indices}")
    invalid = [idx for idx in indices if idx < 0 or idx >= 600]
    if invalid:
        raise ValueError(f"subset has invalid sample ids: {invalid}")


def strat64_from_ledger(ledger_path: Path = DEFAULT_LEDGER, *, seed: int = 55) -> list[int]:
    rows = load_jsonl(ledger_path)
    if not rows:
        # Deterministic fallback if the base ledger has not been generated yet.
        rng = random.Random(seed)
        rest = [idx for idx in FULL600 if idx not in HARD8]
        return HARD8 + rng.sample(rest, 64 - len(HARD8))
    ranked = sorted(rows, key=lambda r: float(r["quality_i"]), reverse=True)
    top32 = [int(row["sample_id"]) for row in ranked[:32]]
    remaining = [idx for idx in FULL600 if idx not in set(top32)]
    rng = random.Random(seed)
    # Stratified over sample id order to avoid an all-tail or all-contiguous complement.
    bins = [remaining[i::32] for i in range(32)]
    complement = [rng.choice(bin_items) for bin_items in bins if bin_items]
    subset = top32 + complement[: 64 - len(top32)]
    validate_subset(subset)
    return subset


def get_subset(name: str, ledger_path: Path = DEFAULT_LEDGER) -> list[int]:
    if name == "hard3":
        return list(HARD3)
    if name == "hard8":
        return list(HARD8)
    if name == "strat64":
        return strat64_from_ledger(ledger_path)
    if name == "full600":
        return list(FULL600)
    if name == "smoke":
        return [0, 1]
    raise ValueError(f"unknown subset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["smoke", "hard3", "hard8", "strat64", "full600"])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    subset = get_subset(args.name, args.ledger)
    payload = {"name": args.name, "indices": subset, "count": len(subset)}
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
