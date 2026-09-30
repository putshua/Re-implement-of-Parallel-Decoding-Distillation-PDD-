# Standard endpoint DMD experiment (2026-09-17)

Submitted to two confirmed idle 8-GPU nodes.

Implementation:

- `AnyFlow/pdd/joint_training.py` now supports `dmd_endpoint_mode=final`.
- Final mode differentiably rolls all four student phases and applies DMD only
  to the final clean endpoint; it does not use local phase PDD trajectory loss.
- `fake_num_heads=1` and `fake_updates_per_g=1` are used for the ordinary-DMD
  baseline.
- PDD/DMD joint defaults were changed to student LR `2e-5`, fake LR `1e-5`.
- `configs/wan1p3b_480p_450k_dmd_endpoint_from_pdd.json` initializes weights-only
  from the completed pure-PDD checkpoint
  `outputs/wan1p3b_480p_450k_rcm6_fsdp16_20260912/step_000100`.

The prepared 16-GPU experiment uses BS=1 per rank and GA=4, hence global batch
64. It uses FSDP shard16, original Wan1.3B teacher weights, 832x480/81f and the
450k weighted prompt cache. `RESUME=none` is intentional: the previous PDD
checkpoint is imported through `student_init` as weights only; optimizer state
and the old PDD/fake state are not resumed.

Prepared launcher:
`scripts/train_wan1p3b_480p_450k_dmd_endpoint_from_pdd.sh`

Keeper payload:
`reports/keeper_dmd_endpoint_20260917/task.json`

Submission:

- `10.32.10.51` (`dlc1brf5zeo88nfz-master-0`): task
  `t20260917-200018-1`
- `10.32.9.209` (`dlc1qh3h1mhywsxc-master-0`): task
  `t20260917-200018-1`
- The first submission failed because the IP endpoint could not establish the
  c10d store. It exited with `DistNetworkError` after 60 seconds.
- Retry submission (active): task `t20260918-105051-2` on both nodes, using
  rendezvous `dlc1brf5zeo88nfz-master-0:29617`.
- output: `outputs/wan1p3b_480p_450k_dmd_endpoint_from_pdd_step100_gbs64_fsdp16_20260917`
- submission record: `reports/keeper_dmd_endpoint_20260917/submission.jsonl`

Current fleet check:

- `10.32.4.42`: all 8 GPUs occupied by `h3-dmd-v3-r5-resume300-20260917`.
- `10.32.9.182`: all 8 GPUs occupied by `h3-dmd-v3-r5-resume300-20260917`.
- `10.32.10.43`: all 8 GPUs occupied by `h3-dmd-v3-r5-resume300-20260917`.
- `10.32.10.49`: all 8 GPUs occupied by `h3-dmd-v3-r5-resume300-20260917`.
- `10.32.10.51`: was idle; now running the submitted task.
- `10.32.9.209`: was idle; now running the submitted task.
- `10.32.1.20`: idle but only 4 GPUs.
- `10.32.4.70`, `10.32.6.74`, `10.32.6.82`, `10.32.8.44`: keeper status request timed out during the check; earlier fleet state showed them occupied, so they were not selected.

No pre-existing running task was stopped or modified. The retry is currently
`running` on both nodes and all 16 ranks have initialized the student
checkpoint; the first loss has not yet been emitted.
