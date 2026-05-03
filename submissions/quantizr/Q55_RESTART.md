# Quantizr #55 Restart

This worktree is pinned to commit `7366288`, the merge context for Quantizr PR #55. Use it as a clean restart path; do not mix in Candidate A or mask-tree artifacts.

## Artifact

Download the official #55 attachment outside git:

```bash
mkdir -p submissions/quantizr/experiments/q55_restart/source
curl -L 'https://github.com/user-attachments/files/26863641/archive.zip' \
  -o submissions/quantizr/experiments/q55_restart/source/q55_archive.zip
```

## Gate Sequence

Run Q0 and Q1:

```bash
bash submissions/quantizr/run_q55_restart_gate.sh \
  submissions/quantizr/experiments/q55_restart/source/q55_archive.zip \
  submissions/quantizr/experiments/q55_restart \
  cuda
```

Run QCRF after Q0 is recorded. Scores must be runner-tagged; local CPU is a useful relative filter, while GitHub/T4 is leaderboard-like truth when available:

```bash
RUN_QCRF=1 bash submissions/quantizr/run_q55_restart_gate.sh \
  submissions/quantizr/experiments/q55_restart/source/q55_archive.zip \
  submissions/quantizr/experiments/q55_restart \
  cuda
```

Run QMASK mixed-CRF allocation before treating uniform CRF56 as the main path:

```bash
python submissions/quantizr/q55_mask_alloc.py \
  --base-archive submissions/quantizr/experiments/q55_restart/source/q55_archive.zip \
  --device cuda \
  --eval-device cpu \
  --decode-backend av \
  --group-spec 50:0.20,54:0.35,58:0.45 \
  --order hist \
  --palette legacy \
  --out-dir submissions/quantizr/experiments/q55_restart
```

Start warm-start training only after QCRF/QMASK gates identify a recoverable byte-quality point:

```bash
python submissions/quantizr/q55_warmstart.py \
  --base-archive submissions/quantizr/experiments/q55_restart/source/q55_archive.zip \
  --crf 54 \
  --device cuda:0 \
  --decode-backend av \
  --out-dir submissions/quantizr/experiments/q55_restart \
  --zero-eval \
  --final-eval
```

## Gates

Do not block all candidate generation on reproducing one local evaluator mode. Record Q0 for each runner and keep all results tagged by runner/device/archive SHA.

Hard stop QCRF only within a runner if regenerated CRF50 does not reproduce #55 closely on that same runner. Proceed only if CRF52/54/56 zero-step quality is recoverable relative to that runner's #55 baseline.

Run QMASK before uniform Q3/CRF56. Initial targets:

```text
mask <=185 KB for the easier PR/GitHub-quality sub-0.300 path
mask <=160 KB for the harder local CPU-quality sub-0.300 path
```

Sub-0.300 requires one of these regimes:

```text
<=251 KB at #55 quality (~0.13290 quality term)
<=260 KB at quality <=0.1269
<=270 KB at quality <=0.1202
<=221 KB at local CPU #55 quality (~0.15284 quality term)
```

Q2/CRF52 is a bridge and first-place candidate. Q3 should train the best uniform-CRF or QMASK mixed-CRF payload, with CRF54 favored before CRF56 unless zero-step proves CRF56 is mild.
