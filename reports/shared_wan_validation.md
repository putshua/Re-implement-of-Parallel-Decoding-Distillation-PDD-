# Shared Wan environment validation — 2026-09-11

Default interpreter is now hardcoded in scripts/shared_env.sh:
/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python

The requested /mnt/data/butong/miniconda directory does not exist on this mount; the actual directory is miniconda3. The selected environment reports Python3.10.20, Torch2.8.0+cu128, Diffusers0.39.0, Transformers5.13.1, Accelerate1.14.0. TensorBoard2.21.0 and its missing dependencies were added using this interpreter. Existing Torch/Diffusers/Transformers were retained. Runtime imports pass (shared_wan_runtime.log).

Training, data preflight, evaluation and local validation launchers source the same shared interpreter; no fallback to node-local Python. PYTHONNOUSERSITE=1 prevents per-node user site packages from shadowing the shared environment. Explicit PYTHON_BIN override remains supported. All nodes need the same absolute shared mount.

19 CPU tests passed under the shared interpreter (shared_wan_unit.log). Four-GPU HSDP shard2 contract passed: forward/backward, accumulation, clipping, full-state export and exact next-step checkpoint resume (shared_wan_fsdp_check.log). This is a tiny-model correctness test, not a resolution benchmark.

Real native Wan1.3B test uses configs/shared_wan_480p.json: 832×480,81frames,4 local GPUs,2 local c10d agents,shard2,batch1,accumulation2,one optimizer update,engineering128-caption training cache. Training loss_mean=0.18638514, grad_norm=2.65114427,31.64seconds. Preview uses one held-out prompt,seed42,NFE4,at step0 and1. Full450k training and physical cross-node16-GPU execution are not tested here. This newer software environment is not a bitwise reproduction of the previous Torch2.3.1/Diffusers0.34 environment.

Both local agents exited0. Post-run verification confirmed both MP4s at832×480,81frames,16FPS; TensorBoard loss exactly matches JSON metrics, both animated preview entries are readable. See shared_wan_media_check.json. No GPU processes remain after the test.
