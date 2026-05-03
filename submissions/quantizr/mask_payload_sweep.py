#!/usr/bin/env python
import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_ffmpeg_path() -> str:
    local_ffmpeg = ROOT_DIR / "ffmpeg"
    if local_ffmpeg.is_file() and local_ffmpeg.stat().st_mode & 0o111:
        return str(local_ffmpeg)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise FileNotFoundError("ffmpeg not found")


def resize_masks(mask: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    x = mask[:, None].float()
    y = F.interpolate(x, size=(height, width), mode="nearest").squeeze(1)
    return y.clamp(0, 4).byte().cpu().numpy()


def write_raw_mask_video(mask_np: np.ndarray, path: Path):
    scaled = (mask_np.astype(np.uint8) * 63).clip(0, 255)
    path.write_bytes(scaled.tobytes())


def encode_av1(raw_path: Path, out_path: Path, width: int, height: int, crf: int):
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        "10",
        "-i",
        str(raw_path),
        "-c:v",
        "libaom-av1",
        "-crf",
        str(crf),
        "-cpu-used",
        "0",
        "-row-mt",
        "1",
        "-g",
        "1200",
        "-keyint_min",
        "1200",
        "-lag-in-frames",
        "48",
        "-arnr-strength",
        "0",
        "-aq-mode",
        "0",
        "-aom-params",
        "enable-cdef=0:enable-intrabc=1:enable-obmc=0",
        "-f",
        "obu",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def brotli_size(path: Path) -> int:
    payload = path.read_bytes()
    return len(brotli.compress(payload, quality=11, lgwin=24))


def main():
    parser = argparse.ArgumentParser(description="Measure alternative mask side-channel payload sizes.")
    parser.add_argument("--mask-tensor", type=Path, required=True, help="Torch tensor of decoded class masks: N,H,W uint8/long.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["512x384", "384x288", "256x192", "192x144", "128x96"])
    parser.add_argument("--crfs", nargs="+", type=int, default=[42, 46, 50, 54, 58, 62])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask = torch.load(args.mask_tensor, map_location="cpu").contiguous()
    if mask.ndim != 3:
        raise ValueError(f"expected mask tensor shape N,H,W, got {tuple(mask.shape)}")

    rows = []
    with tempfile.TemporaryDirectory(prefix="mask_payload_sweep_") as tmp_name:
        tmp = Path(tmp_name)
        for size_str in args.sizes:
            width_s, height_s = size_str.lower().split("x", 1)
            width, height = int(width_s), int(height_s)
            resized = resize_masks(mask, (height, width))
            raw_path = tmp / f"mask_{width}x{height}.yuv"
            write_raw_mask_video(resized, raw_path)
            raw_brotli = brotli_size(raw_path)
            rows.append(
                {
                    "mode": "raw_brotli",
                    "width": width,
                    "height": height,
                    "crf": "",
                    "obu_bytes": "",
                    "brotli_bytes": raw_brotli,
                }
            )
            for crf in args.crfs:
                obu_path = tmp / f"mask_{width}x{height}_crf{crf}.obu"
                encode_av1(raw_path, obu_path, width, height, crf)
                rows.append(
                    {
                        "mode": "av1_obu_brotli",
                        "width": width,
                        "height": height,
                        "crf": crf,
                        "obu_bytes": obu_path.stat().st_size,
                        "brotli_bytes": brotli_size(obu_path),
                    }
                )

    csv_path = args.out_dir / "mask_payload_sweep.csv"
    json_path = args.out_dir / "mask_payload_sweep.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "width", "height", "crf", "obu_bytes", "brotli_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    for row in sorted(rows, key=lambda r: int(r["brotli_bytes"]))[:20]:
        print(row)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
