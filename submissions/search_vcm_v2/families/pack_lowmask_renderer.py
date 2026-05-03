#!/usr/bin/env python
"""Byte estimates for lowmask renderer variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submissions.search_vcm_v2.families.lowmask_renderer import LOWMASK_RENDERER_CONFIGS, build_lowmask_renderer
from submissions.search_vcm_v2.families.pack_factorized_renderer import int8_state_payload


def estimate_lowmask_renderer_bytes(config_name: str) -> dict:
    model = build_lowmask_renderer(config_name)
    params = sum(p.numel() for p in model.parameters())
    fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    packed = int8_state_payload(model)
    return {
        "config_name": config_name,
        "params": params,
        "fp32_param_bytes": fp32_bytes,
        "int8_brotli_bytes_random_init": len(packed),
        "config": model.config(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(LOWMASK_RENDERER_CONFIGS), default="L48")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = estimate_lowmask_renderer_bytes(args.config)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
