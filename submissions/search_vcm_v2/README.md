# Search VCM v2

Search VCM v2 tests one live hypothesis:

```text
exact semantic mask + tiny factorized renderer + per-sample pose tokens
```

v2 is separate from `submissions/search_vcm`. It uses qpose14 as the public
baseline/target and requires qpose-relative gates before any compressed or
full-run work is allowed.

## First Commands

```bash
python submissions/search_vcm_v2/asha.py run \
  --families qpose14_baseline \
  --round smoke

python -m unittest discover -s submissions/search_vcm_v2/tests
```

If no real `submissions/qpose14/archive.zip` exists, Gate 0 can materialize a
local qpose14-compatible proxy from `submissions/q55_manual`. This is useful
for plumbing and subset construction, but it is not an official qpose14 score.

