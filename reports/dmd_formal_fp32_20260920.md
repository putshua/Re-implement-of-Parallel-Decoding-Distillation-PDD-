# Formal DMD continuation and optimizer verification

The short audit job was replaced with the formal 250-update run on
10.32.10.51 and 10.32.9.209. Both keeper task IDs are t20260920-125205-22.
Only the owned audit task was stopped; keeper daemons were not stopped.

Initialization: pure PDD step100 weights, fresh G/fake optimizer state;
not the collapsed DMD checkpoint and not a resume of the short audit.
Configuration: 16 GPUs, FSDP16, BS1, GA4, global batch64; normalized endpoint
DMD; G LR 2e-5; fake LR 1e-5; fake:G 5:1; DMD weight1, trajectory weight0.
Rolling checkpoint every5, permanent every50; 32 specified prompts every25.
Output: `outputs/wan13b_dmd_normalized_fp32_250step_20260920`.
Payload and API submission responses: `reports/keeper_dmd_formal_20260920*.json`.

## Precision evidence

Read both model.pt and fake.pt from the completed audit step000010 with
torch.load(mmap=True, weights_only=True). For each model:

- 825 parameter tensors: torch.float32.
- 825 Adam exp_avg tensors: torch.float32.
- 825 Adam exp_avg_sq tensors: torch.float32.
- 825 Adam step tensors: torch.float32.
- Saved optimizer LR: generator 2e-5, fake 1e-5.

FSDP casts forward parameters to BF16, reduces gradients in FP32 and keeps
original master shards in FP32. New runtime checks inspect live parameters,
gradients, exp_avg and exp_avg_sq immediately after first optimizer.step;
all ranks participate and any non-FP32 tensor causes a failure.

## Loss scaling evidence

`dmd_weight` is the DMD loss multiplier. There is no separate loss_scale
config key or dynamic FP16 GradScaler here. The generator backward receives
`(traj_weight*trajectory + dmd_weight*ramp*dmd)/grad_accum`.
The loss reduces over all latent elements using mean, with the explicit 0.5
surrogate factor. Test invoking actual JointTraining.generator_micro and
backward with GA4 confirms changing dmd_weight from 0.5 to2 multiplies the
gradient by4. Six standard endpoint tests pass. Gradient clipping and Adam
mean parameter updates are not necessarily proportional to loss scale.
