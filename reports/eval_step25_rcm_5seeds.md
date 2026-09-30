# Step25 / RCM six prompts / five seeds

Completed on four local A800 80GB GPUs using the shared Wan Python environment.
Checkpoint: /mnt/data/butong/PDD/outputs/wan1p3b_480p_450k_fsdp16/step_000025 (COMPLETE, strict student state load).
Prompts: assets/rcm_fixed_prompts/prompts_000000000.pt, all6 original RCM prompts.
Seeds: 42,43,44,45,46. NFE4 and8. All60 samples produced finite latents and MP4 videos verified at832×480,81frames. The renderer requests16FPS. No teacher baseline was sampled in this run.

Mean sampling time: NFE4=7.341s; NFE8=14.690s. Excludes model loading/transfers, warmup, VAE decode and disk writing. These are latency measurements, not quality-matched speedups. Latent diversity statistics in summary.json are diagnostics, not perceptual quality scores.

Output: outputs/eval_step25_rcm_5seeds. index.html provides prompt/seed/method filters and video players with midpoint thumbnails. COMPLETE confirms gallery verification. All GPU processes exited and GPU memory was released.

Reproduction: bash scripts/eval_step25_rcm_5seeds.sh (set OUTPUT to a new directory to avoid mixing runs).
