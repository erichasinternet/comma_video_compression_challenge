# Experiment Archive Note

This branch preserves the experiment source code, reports, and findings that led
to the final `qzs3_range_mask_candidate` submission.

Large generated artifacts are intentionally not committed to GitHub history:

- inflated raw videos under `*/inflated/`
- extracted archive directories under `*/archive/`
- large tensor/checkpoint caches under `*/experiments/`
- local preview renders under `submissions/qzs3_range_mask_candidate/previews/`
- large RunPod logs and downloaded experiment artifacts under `runpod_saved/`

Those files remain local workspace artifacts. The clean submission branch should
contain only evaluator-facing files.
