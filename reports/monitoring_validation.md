# Training monitor validation — 2026-09-11

Local test: four A800 80GB GPUs, two local c10d agents × two GPUs, HSDP shard size2. This is not a physical two-node / 16-GPU test. Config: `configs/monitoring_480p.json`; native Wan1.3B, 832×480, 81 frames, batch1, accumulation2, two optimizer updates, checkpoint writing disabled. Training input is the existing 128-caption engineering cache, not the full450k run. Preview uses two held-out fixed prompts, seed42, NFE4/8; all four ranks participate in sampling, including padded ranks.

Both agents completed successfully. Previews at step0,1,2 each produced four MP4/GIF/JPG sets and COMPLETE manifests. Callback times: 38.46s, 36.76s, 36.84s. Training continued after callbacks; losses and gradients were finite. All GPU processes were released afterward.

`python scripts/verify_monitoring.py outputs/monitoring_480p` verified all12 MP4s as832×480,81frames,16FPS. TensorBoard train/loss at steps1/2 exactly matches metrics.jsonl loss_mean; all12 TensorBoard animated GIF entries were readable at208×120. See `monitoring_media_check.json` and per-agent logs. This is callback functionality validation, not a converged model quality benchmark. It does not establish bitwise training equivalence to runs without previews.

18 CPU unit tests passed, including equal collective forward counts with padded prompts, preview scheduling, RNG/mode/coefficients restoration after exceptions, media writing and scalar event reading. Launcher tests and shell syntax checks passed after adding runtime dependency checks. Ruff checks passed for monitoring and new utility/test files.

Cluster logs supplied by the user show missing diffusers in /usr/local/bin/python (Python3.12), after the platform skipped nonexistent root requirements.txt. No conda path is hardcoded in PDD launchers. Added root requirements.txt for cluster runtime packages while preserving image-provided CUDA PyTorch, and a once-per-node import check before torchrun. Local runtime check passed; installation and actual execution in the remote Python3.12 cluster image have not been tested here.
