# Official teacher source snapshot

Source repository: https://github.com/Wan-Video/Wan2.1
Pinned commit: 9737cba9c1c3c4d04b33fcad41c111989865d315
Files and SHA256 checksums: provenance.json. Source files are unmodified.

The comparison uses the official README's T2V-1.3B 832×480 example: shift8, guidance6, the CLI's UniPC solver and T2V50-step defaults. All81 frames,16FPS; no prompt extension. Official shared_config.sample_neg_prompt was extracted and encoded with the native checkpoint's UMT5 into assets/wan21_official_negative.

Official native WanModel weights remain FP32 with BF16 autocast, matching the source loading and sampling pattern. PDD inference uses its existing Diffusers backbone implementation and BF16 parameters. Both use the same six original RCM prompts and seeded noise, and the native Wan2.1 VAE implementation from ../Wan2.2. Native teacher has no skip-layer modification. These are different sampler/backend settings, so latency ratios are not quality-matched speedup claims.
