# Evaluator Validation

## Trusted Path

Use the repository root evaluator path for submission scoring:

```text
archive.zip
-> evaluate.sh unzip
-> submission inflate.sh
-> inflated/*.raw
-> evaluate.py
-> report.txt
```

Search/proxy evaluators are not trusted submission scores. They are useful only
for ranking experiments and selecting subsets.

## Harness

Validation harness:

```bash
python submissions/search_vcm/evaluator_validation.py run-one \
  --submission-dir submissions/q55_fp16_pose_int10 \
  --device mps

python submissions/search_vcm/evaluator_validation.py run-suite \
  --no-run \
  --device mps
```

The harness records:

```text
archive bytes and SHA256
payload file list
evaluate.py/evaluate.sh/modules.py hashes
model weight hashes
torch/platform/device metadata
inflated raw layout checks
parsed report metrics
full-precision score recomputation
reference deltas
```

Local output directory:

```text
submissions/search_vcm/experiments/evaluator_validation/
```

## Local Issue Found

The root evaluator and several submission `inflate.sh` scripts call `python`.
This shell does not expose `python` globally; it only has the project virtualenv
when invoked explicitly. The validation harness prepends:

```text
.venv/bin
```

to `PATH` before invoking `evaluate.sh`. Without this, fresh local evaluation
can fail before inflation with:

```text
python: command not found
```

## Validated MPS Runs

Fresh root `evaluate.sh` runs:

```text
q55_fp16_pose_int10
  archive: 288,268 B
  PoseNet: 0.00065135
  SegNet:  0.00072222
  quality: 0.1529282575
  score:   0.3448740862
  status:  pass

q55_fp16_pose_int12
  archive: 289,127 B
  PoseNet: 0.00064976
  SegNet:  0.00072222
  quality: 0.1528296919
  score:   0.3453474935
  status:  pass

selfcomp_pr56_eval
  archive: 279,036 B
  PoseNet: 0.00039916
  SegNet:  0.00115280
  quality: 0.1784591105
  score:   0.3642577293
  status:  pass
```

Parsed existing report:

```text
q55_manual
  archive: 299,970 B
  PoseNet: 0.00064988
  SegNet:  0.00072220
  quality: 0.1528351351
  score:   0.3525728452
  status:  pass
```

## Current Trust Statement

The local MPS root evaluator is trusted for relative local submission scoring
across q55 and selfcomp-style submissions. It reproduces existing reference
rows under the same archive bytes, raw layout, model hashes, and score formula.

Remaining gap:

```text
CUDA/T4 official-runner calibration is still unavailable locally.
```
