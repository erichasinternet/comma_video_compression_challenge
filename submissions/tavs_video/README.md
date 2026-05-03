# TAVS Video

Task-Aware Video Source Optimization stores only a conventional compressed video in the archive. During compression, the source video is optimized against the frozen challenge evaluator with codec-like augmentations and periodic real codec round trips.

Inflation is intentionally simple:

```bash
./submissions/tavs_video/inflate.sh archive inflated public_test_video_names.txt
```

The inflater searches the archive directory for `video.ivf`, `video.mkv`, or `video.mp4`, decodes it with PyAV/FFmpeg semantics, resizes frames to the official camera size, and writes `inflated/0.raw`.

The initial experiment is a 64-sample oracle:

```bash
./.venv/bin/python submissions/tavs_video/optimize_source.py \
  --out-dir submissions/tavs_video/experiments/tavs_64_q55init_yuvgrid96_svtav1 \
  --init q55 \
  --subset 64 \
  --steps 5000 \
  --codec-eval-every 250
```

For GPU execution, use `modal_tavs.py`.

