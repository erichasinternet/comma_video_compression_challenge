# RunPod Artifact Save

Saved before deleting the RunPod pod/volume.

## Contents

- `run_logs/`: complete remote `run_logs/` directory, including long training logs.
- `experiments/`: selected experiment artifacts from `submissions/quantizr/experiments/`.
- `source/`: copied remote Quantizr scripts used for the final probes.
- `manifests/remote_all_files.tsv`: full remote file inventory for `run_logs` and Quantizr experiments.
- `manifests/remote_status.txt`: remote git/status/GPU snapshot before shutdown.
- `manifests/local_file_list.txt`: saved local file list.
- `manifests/local_sha256.txt`: SHA-256 checksums for saved local files.

## Excluded

Large disposable intermediates were intentionally excluded:

- inflated raw RGB files
- `.mkv` / `.mp4` transcode intermediates
- `rgb_pairs.pt`
- `mask_frames*.pt`
- `reconstructed_masks.pt`

These can be regenerated from local videos and saved code if needed.

## Key Conclusions Preserved

- Current Candidate A truth is the re-evaluated `~0.404`, not stale `0.3653336`.
- Mask-tree byte targets were reached, but evaluator quality collapsed.
- Mask-tree adaptation failed its first useful proxy gate.
- Direct generated-video storage failed for both x265 and SVT-AV1 target-bitrate probes.
- Evaluator-native latent renderer D0 failed the first hard capacity gate at step 1000.

## Key Saved Artifacts

- Candidate A archive and stage checkpoints under `experiments/gpt_pose_gate/`.
- Qpack/mask-tree oracle outputs under `experiments/masktree_v1/`.
- Direct-video oracle outputs under `experiments/generated_transcode_oracle_*`.
- Auto-decoder probe outputs under `experiments/autodecoder_*`.
