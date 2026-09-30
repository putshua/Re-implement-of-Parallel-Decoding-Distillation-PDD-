# Parallel Decoding Distillation — Community Implementation

An **unofficial community implementation** of [Parallel Decoding Distillation for Fast Image and Video Generation](https://arxiv.org/abs/2607.26004), based on [NVlabs/AnyFlow](https://github.com/NVlabs/AnyFlow). Supports Wan2.1 T2V 1.3B and 14B, with optional DMD extensions. This is not the authors' official implementation or a claim to reproduce their reported results. The adapted implementation is in `AnyFlow/pdd/`; the [upstream license](AnyFlow/LICENSE) is retained.

## Environment

Linux, Python 3.10–3.12, CUDA-enabled PyTorch, and NVIDIA GPUs with BF16 support are required. The tested dependency versions are PyTorch 2.8.0, Diffusers 0.39.0, Transformers 5.13.1, and Accelerate 1.14.0. GPU memory depends on model size, batch size, and video length; start with batch size 1. Install PyTorch for your CUDA runtime first:

```bash
python -m venv .venv
source .venv/bin/activate
# Example CUDA build; choose the index appropriate for your system.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/check_runtime.py
```

Install `ffmpeg` and `ffprobe` for video galleries. Download a **native Wan2.1 T2V checkpoint**, including its transformer weights, UMT5 encoder/tokenizer, and VAE. Obtain a [Wan2.2 source checkout](https://github.com/Wan-Video/Wan2.2) with `wan/modules/t5.py` and `wan/modules/vae2_1.py` (used here for its Wan2.1-compatible encoder and VAE). Set your own paths in the configuration; no weights or datasets are included. Transformers run through Diffusers; prompt encoding and VAE decoding use the supplied Wan source.

Scripts use Python from the active environment. Set `PYTHON_BIN` to an executable path if needed. Multi-node runs require compatible environments, reachable rendezvous networking, identical model/data paths, and a shared output directory on every node.

## Training

### Prepare your prompt dataset

Training uses prompt embeddings and generated noise, without ground-truth videos. Supply a UTF-8 text file with one prompt per line, or JSONL rows such as `{"id":"example-1","caption":"A dog runs across a grassy field."}` (`id` is optional).

```bash
python scripts/prepare_embeddings.py \
  --prompts /path/to/train_prompts.txt \
  --checkpoint /path/to/Wan2.1-T2V-1.3B \
  --wan-root /path/to/Wan-source \
  --output /path/to/train_embeddings \
  --negative-prompt "low quality, blurry"
```

For JSONL, replace `--prompts` with `--manifest`. The command writes resumable `prompts_*.pt` shards and, when requested, `negative_embeddings.pt`. Each shard contains `prompts` and `t5_text_embeddings` shaped `[N,512,4096]`; negative embeddings have shape `[512,4096]`. Encode a separate small prompt collection for evaluation and callbacks.

Copy one of these configurations and replace its `/path/to/...` placeholders:

| Configuration | Objective |
|---|---|
| `configs/pdd_wan1p3b.example.json` | Pure PDD trajectory distillation |
| `configs/dmd_wan1p3b.example.json` | Final-endpoint DMD with the PDD student architecture |
| `configs/phased_dmd_pdd_wan1p3b.example.json` | Phased endpoint DMD plus trajectory MSE |

Set `embedding_dir` to your shard directory, **or** replace it with `prompt_embeddings` pointing to one prepared `.pt` file. Set `negative_embeddings`, `checkpoint`, and `wan_root` explicitly. `height`, `width`, and `frames` select generated video dimensions. Wan2.1 14B uses the same entries with a native T2V-14B checkpoint; start with batch 1 and a larger FSDP group, then measure memory before increasing accumulation.

### Single node

```bash
cp configs/pdd_wan1p3b.example.json configs/my_pdd.json
# Edit configs/my_pdd.json before running.
CONFIG=configs/my_pdd.json NPROC_PER_NODE=4 \
BATCH_SIZE=1 GRAD_ACCUM=8 FSDP_SHARD_SIZE=4 \
OUTPUT=outputs/my_pdd bash scripts/train_single_node.sh
```

For DMD or joint training, copy the corresponding configuration and pass it through `CONFIG` to the same script. The examples enable teacher FSDP and require at least two ranks; for one GPU, disable `teacher_fsdp` and use a sharding size of 1 if the model fits.

### Multiple nodes

Run the same command on **every participating node**:

```bash
CONFIG=/shared/project/configs/my_dmd.json \
NNODES=2 NPROC_PER_NODE=8 \
RDZV_ENDPOINT=MASTER_IP:29571 RDZV_ID=my-unique-dmd-run \
BATCH_SIZE=1 GRAD_ACCUM=2 FSDP_SHARD_SIZE=8 \
OUTPUT=/shared/project/outputs/my_dmd \
bash scripts/train_multi_node.sh
```

Replace `MASTER_IP` with an address reachable from all nodes. The scripts launch local workers; they do not submit jobs to other machines. Global batch is `NNODES × NPROC_PER_NODE × BATCH_SIZE × GRAD_ACCUM` (32 above). FSDP group size controls parameter sharding separately. Use `DRY_RUN=1` to inspect the command. CLI overrides include `--steps 1000` and `--set lr=2e-5`.

The DMD example initializes from the teacher, uses generator/fake learning rates `2e-5`/`1e-5`, warms up the fake score for 50 updates, then performs four fake updates per generator update. Its 1000-step budget counts every fake or generator update, including warmup. `rollout_nfes=[1,2,3,4]` cycles deterministic trajectory rollout lengths; it does not re-noise between rollout intervals. Pure DMD has `traj_weight=0`; the joint example uses phase-endpoint DMD and trajectory MSE. Set `student_init` to a completed PDD checkpoint to import student weights only; use `RESUME` to restore the full training state instead.

The examples save every 5 steps and retain every 50th checkpoint. Resume with the same distributed layout and batch settings:

```bash
CONFIG=configs/my_pdd.json NPROC_PER_NODE=4 FSDP_SHARD_SIZE=4 \
BATCH_SIZE=1 GRAD_ACCUM=8 OUTPUT=outputs/my_pdd RESUME=auto \
bash scripts/train_single_node.sh
```

Completed checkpoints contain a `COMPLETE` marker. Ordinary logs are in `OUTPUT/logs`; TensorBoard loss is in `OUTPUT/tensorboard`. For training previews, set `fixed_prompt_enabled=true`, `fixed_prompt_embeddings` to your evaluation file, `fixed_prompt_every=25`, `fixed_prompt_count` to your prompt count, `fixed_prompt_nfe=[4]`, and `fixed_prompt_decode=true`. The preview page is `OUTPUT/fixed_prompt/index.html`.

## Inference

Use the run's saved configuration and a completed checkpoint:

```bash
CONFIG=outputs/my_pdd/config.json \
PROMPTS=/path/to/eval_embeddings/prompts_000000000.pt \
WAN_ROOT=/path/to/Wan-source NPROC_PER_NODE=4 \
OUTPUT=outputs/evaluation bash scripts/eval_pdd.sh \
  --student outputs/my_pdd/step_001000 \
  --nfe 4 --limit 16 --seeds 42 43 44 45 46 --skip-teacher --decode
```

Set `--limit` no higher than the number of evaluation prompts. NFE must fit the checkpoint's trained interval range: the PDD example permits 4 or 8; the DMD example permits 1–4. Student inference uses one conditioned network call per interval, with CFG learned during distillation. Removing `--skip-teacher` also evaluates the training teacher using its configured CFG and an Euler trajectory. `--teacher-only --official-teacher-root /path/to/Wan2.1 --official-negative /path/to/official_negative_embeddings.pt` selects the separate Wan2.1 1.3B official-example comparison: native model, UniPC50, shift8, CFG6. Prepare the official negative text with the same UMT5 encoder. These teacher settings differ from the student's trajectory grid.

To generate a gallery after decoded evaluation:

```bash
python scripts/eval_gallery.py outputs/evaluation
python -m http.server 8080 --bind 127.0.0.1 --directory outputs/evaluation
```

Use `--resume-evaluation` with unchanged evaluation arguments to continue an interrupted inference run.
