# Task-NeRV VCM

Prototype task-aware neural video representation for the comma video compression challenge.

This is not a conventional video codec path. It stores a neural renderer and learned frame embeddings, then inflates by rendering frames from frame ids.

Current status is capacity-oracle only. Do not treat the packed PyTorch payload as final compression.

