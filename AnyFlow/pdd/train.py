import argparse
import copy
import hashlib
import json
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pdd.distributed import (
    FSDP,
    wrap_student,
    checkpoint_state,
    restore_state,
    move_optimizer_state,
    verify_optimizer_fp32,
)
from pdd.monitoring import TrainingMonitor, evaluation_state
from pdd.run_state import (
    resolve_resume,
    reconcile_metrics,
    checkpoint_due,
    prune_checkpoints,
)
from pdd.core import pd_loss, sigma_grid
from pdd.checkpoint_io import prefetch_checkpoint
from pdd.model import PDDWan, guided_velocity, load_wan
from pdd.data import PromptEmbeddings, WeightedPromptCache, ShardedPromptEmbeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_VALUE")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for override in args.set:
        key, value = override.split("=", 1)
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    for key in ("output", "batch_size", "steps", "grad_accum", "resume"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("distributed_strategy", "ddp")
    cfg.setdefault("fsdp_shard_size", 1)
    cfg["head_sampling"] = "per_example"
    cfg["code_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    joint_mode = cfg.get("training_objective") == "phased_dmd_pdd"
    count_optimizer_updates = joint_mode and bool(cfg.get("count_optimizer_updates", False))
    updates_per_cycle = int(cfg.get("fake_updates_per_g", 0)) + 1 if joint_mode else 1
    warmup_steps = int(cfg.get("fake_warmup_steps", 0))
    if warmup_steps < 0 or (warmup_steps and (not joint_mode or cfg["dmd_weight"] <= 0)):
        raise ValueError("fake_warmup_steps requires enabled DMD and must be nonnegative")
    step_offset = warmup_steps if count_optimizer_updates else 0
    if cfg["steps"] <= step_offset:
        raise ValueError("steps must leave at least one generator cycle after fake warmup")
    if count_optimizer_updates and (cfg["steps"] - step_offset) % updates_per_cycle:
        raise ValueError("steps minus fake warmup must be divisible by fake_updates_per_g + 1")
    cycle_steps = ((cfg["steps"] - step_offset) // updates_per_cycle
                   if count_optimizer_updates else cfg["steps"])
    rank, world, local = [
        int(os.environ.get(k, d))
        for k, d in [("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)]
    ]
    if cfg.get("optimizer_cpu_offload") == "auto":
        cfg["optimizer_cpu_offload"] = (
            cfg["distributed_strategy"] == "fsdp"
            and min(cfg["fsdp_shard_size"], world) < 8
        )
    if not isinstance(cfg.get("optimizer_cpu_offload", False), bool):
        raise ValueError("optimizer_cpu_offload must be true, false or auto")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training. Check your PyTorch installation and GPU access."
        )
    torch.cuda.set_device(local)
    if cfg.get("checkpoint_prefetch", False) and rank == 0:
        print("[rank 0] sequential checkpoint prefetch started", flush=True)
        prefetch_begin = time.perf_counter()
        prefetched = prefetch_checkpoint(
            cfg["checkpoint"], cfg.get("checkpoint_prefetch_workers", 6)
        )
        print(
            f"[rank 0] checkpoint prefetched: {prefetched / 2**30:.2f} GiB in "
            f"{time.perf_counter() - prefetch_begin:.2f}s",
            flush=True,
        )
    if world > 1:
        dist.init_process_group(
            "nccl",
            timeout=timedelta(seconds=cfg.get("distributed_timeout_seconds", 600)),
        )
    device = torch.device("cuda", local)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    N, Lmin, Lmax = cfg["num_heads"], cfg["block_min"], cfg["block_max"]
    if N % Lmin or not 1 <= Lmin <= Lmax <= N:
        raise ValueError(
            "Require N divisible by block_min and 1 <= block_min <= block_max <= N"
        )
    batch, accum = cfg["batch_size"], cfg["grad_accum"]
    if min(batch, accum, cfg["steps"]) < 1:
        raise ValueError("Batch, accumulation and steps must be positive")
    output = Path(cfg["output"])
    # Rank0 selects once; all nodes receive the same checkpoint path.
    restart = [None, None]
    if rank == 0:
        try:
            restart[0] = resolve_resume(
                output,
                cfg.get("resume"),
                world,
                extra_files=(
                    ["joint_state.json"]
                    + (["fake.pt"] if cfg["dmd_weight"] > 0 else [])
                )
                if joint_mode
                else [],
                restart_without_checkpoint=cfg.get(
                    "auto_restart_without_checkpoint", False
                ),
            )
        except Exception as exc:
            restart[1] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(restart, src=0)
    if restart[1]:
        raise RuntimeError(restart[1])
    cfg["resume"] = restart[0]
    if local == 0:
        print(
            json.dumps(
                {
                    "event": "restart_selection",
                    "rank": rank,
                    "output": str(output),
                    "resume": cfg["resume"],
                    "mode": "resume" if cfg["resume"] else "fresh",
                }
            ),
            flush=True,
        )
    if (output / "metrics.jsonl").exists() and not cfg.get("resume"):
        raise FileExistsError(
            f"{output} already contains a run; choose a new output or RESUME"
        )
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "config.json").write_text(json.dumps(cfg, indent=2))
        print(
            json.dumps(
                {
                    "event": "init",
                    "world": world,
                    "global_batch": batch * accum * world,
                    "torch": torch.__version__,
                    "pytorch_cuda_alloc_conf": os.environ.get(
                        "PYTORCH_CUDA_ALLOC_CONF", "default"
                    ),
                    "device": torch.cuda.get_device_name(),
                    "config": cfg,
                }
            ),
            flush=True,
        )
    sharded = (
        cfg["distributed_strategy"] == "fsdp" and min(cfg["fsdp_shard_size"], world) > 1
    )
    if cfg.get("teacher_fsdp", False) and not sharded:
        raise ValueError("teacher_fsdp requires FSDP with at least two ranks/shards")
    if cfg.get("low_memory_init", False):
        student = PDDWan(
            load_wan(
                cfg["checkpoint"], dtype=torch.float32, init_only=sharded and rank != 0
            ),
            N,
        )
        teacher = None
    else:
        teacher = load_wan(cfg["checkpoint"])
        student = PDDWan(copy.deepcopy(teacher), N)
    if cfg.get("student_init") and not cfg.get("resume") and (not sharded or rank == 0):
        initial = Path(cfg["student_init"])
        initial = initial / "model.pt" if initial.is_dir() else initial
        state = torch.load(initial, map_location="cpu", mmap=True, weights_only=True)
        for key in ("num_heads", "shift", "height", "width", "frames"):
            if state["config"][key] != cfg[key]:
                raise ValueError(f"Student initialization config mismatch: {key}")
        student.load_state_dict(state["student"], strict=True)
        del state
    print(f"[rank {rank}] student checkpoint initialized", flush=True)
    if cfg.get("gradient_checkpointing", True):
        student.backbone.enable_gradient_checkpointing()
    student = wrap_student(student, cfg, device)
    if teacher is None:
        teacher = load_wan(
            cfg["checkpoint"],
            dtype=torch.bfloat16,
            init_only=cfg.get("teacher_fsdp", False) and rank != 0,
        )
    teacher = teacher.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    teacher = (
        wrap_student(teacher, cfg, device, frozen=True)
        if cfg.get("teacher_fsdp", False)
        else teacher.to(device)
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg["lr"], weight_decay=0.0, foreach=False
    )
    neg = torch.load(cfg["negative_embeddings"], map_location="cpu", weights_only=True)
    if neg.ndim == 2:
        neg = neg.unsqueeze(0)
    negative = neg.to(device, dtype=torch.bfloat16)
    print(f"[rank {rank}] models and optimizer ready", flush=True)
    if cfg.get("embedding_dir"):
        dataset = ShardedPromptEmbeddings(cfg["embedding_dir"], seed, rank)
        sampler = None
    elif cfg.get("weight_index"):
        dataset = WeightedPromptCache(
            cfg["weight_index"], seed, rank, cfg.get("path_map")
        )
        sampler = None
    else:
        dataset = PromptEmbeddings(cfg["prompt_embeddings"])
        sampler = (
            DistributedSampler(dataset, world, rank, shuffle=True, seed=seed)
            if world > 1
            else None
        )
    loader = DataLoader(
        dataset,
        batch_size=batch,
        sampler=sampler,
        num_workers=cfg.get("workers", 0),
        pin_memory=True,
        drop_last=True,
    )
    if hasattr(dataset, "__len__") and len(dataset) < batch * world:
        raise ValueError("Insufficient prompts for one global microbatch")
    iterator, epoch, cursor = iter(loader), 0, 0

    def next_context():
        nonlocal iterator, epoch, cursor
        try:
            item = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            cursor = 0
            item = next(iterator)
        cursor += 1
        return item["context"].to(device, dtype=torch.bfloat16)

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    grid = sigma_grid(N, cfg["shift"], device)
    shape = (
        batch,
        16,
        (cfg["frames"] - 1) // 4 + 1,
        cfg["height"] // 8,
        cfg["width"] // 8,
    )
    if (cfg["frames"] - 1) % 4 or cfg["height"] % 16 or cfg["width"] % 16:
        raise ValueError("Wan requires 4k+1 frames and height/width divisible by 16")
    joint = None
    if joint_mode:
        from pdd.joint_training import JointTraining

        joint = JointTraining(
            cfg, student, teacher, negative, grid, device, rank, world
        )
    # One retained on-policy stream per accumulation microbatch; no retained graph.
    streams = [None] * accum
    start = 0
    resume_step = 0
    if cfg.get("resume"):
        checkpoint = Path(cfg["resume"])
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError("Cannot resume incomplete checkpoint")
        state = torch.load(
            checkpoint / "model.pt", map_location="cpu", weights_only=True, mmap=True
        )
        if state["config"].get("training_objective") != cfg.get("training_objective"):
            raise ValueError(
                "Cannot resume a different training objective; use student_init for weights only"
            )
        if joint_mode:
            for key in (
                "phase_edges",
                "dmd_weight",
                "traj_weight",
                "fake_lr",
                "fake_updates_per_g",
                "fake_warmup_steps",
                "dmd_ramp_steps",
                "dmd_time_shift",
                "dmd_min_gap",
                "dmd_min_time",
                "dmd_max_time",
                "dmd_normalization",
                "dmd_teacher_skip_layer",
                "fake_variance_cap",
                "dmd_endpoint_mode",
                "fake_num_heads",
                "rollout_nfes",
                "count_optimizer_updates",
                "grad_accum",
                "batch_size",
                "weight_index",
            ):
                if state["config"].get(key) != cfg.get(key):
                    raise ValueError(f"Joint resume config mismatch: {key}")
        for key in (
            "num_heads",
            "shift",
            "block_min",
            "block_max",
            "solver",
            "height",
            "width",
            "frames",
            "seed",
            "guidance",
            "skip_layer",
            "lr",
        ):
            if state["config"][key] != cfg[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        if state["config"].get("head_sampling") != cfg["head_sampling"]:
            raise ValueError("Resume requires matching per-example head sampling")
        for key, default in (("distributed_strategy", "ddp"), ("fsdp_shard_size", 1)):
            if state["config"].get(key, default) != cfg[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        restore_state(student, optimizer, state)
        if cfg.get("optimizer_cpu_offload", False):
            move_optimizer_state(optimizer, "cpu")
            torch.cuda.empty_cache()
        saved_step = state["step"]
        resume_step = saved_step
        if count_optimizer_updates:
            if saved_step >= step_offset and (saved_step - step_offset) % updates_per_cycle:
                raise ValueError("Checkpoint is not at a complete fake/G cycle")
            start = max(0, saved_step - step_offset) // updates_per_cycle
        else:
            start = saved_step
        if joint:
            joint.restore(checkpoint, saved_step)
        del state
        local_state = torch.load(
            checkpoint / f"rank_{rank}.pt", map_location="cpu", weights_only=False
        )
        if (
            local_state["world"] != world
            or local_state["batch"] != batch
            or len(local_state["streams"]) != accum
        ):
            raise ValueError("Exact resume requires same world, batch and accumulation")
        # Replay data cursor before restoring RNG used for policy trajectories.
        for _ in range(local_state["total_batches"]):
            next_context()
        streams = [
            None
            if s is None
            else (
                s[0].to("cpu" if cfg.get("offload_streams", False) else device),
                s[1].to("cpu" if cfg.get("offload_streams", False) else device),
                s[2],
            )
            for s in local_state["streams"]
        ]
        torch.set_rng_state(local_state["cpu_rng"])
        torch.cuda.set_rng_state(local_state["cuda_rng"], device)
        random.setstate(local_state["python_rng"])
    total_batches = (
        epoch * len(loader) + cursor if hasattr(dataset, "__len__") else cursor
    )
    if rank == 0 and cfg.get("resume"):
        backup = reconcile_metrics(
            output, resume_step
        )
        if count_optimizer_updates:
            reconcile_metrics(output, resume_step, "optimizer_updates.jsonl")
            if warmup_steps:
                reconcile_metrics(output, min(resume_step, warmup_steps), "fake_warmup.jsonl")
        print(
            json.dumps(
                {
                    "event": "resume_ready",
                    "step": resume_step,
                    "checkpoint": cfg["resume"],
                    "metrics_backup": backup,
                }
            ),
            flush=True,
        )
    log = open(output / "metrics.jsonl", "a", buffering=1) if rank == 0 else None
    updates_log = (
        open(output / "optimizer_updates.jsonl", "a", buffering=1)
        if rank == 0 and count_optimizer_updates else None
    )
    def save_checkpoint(display_step):
        keep_every = cfg.get("checkpoint_keep_every", 0)
        if cfg.get("save_checkpoints", True) and (
            checkpoint_due(
                display_step,
                cfg.get("save_every", 25),
                cfg["steps"],
                cfg.get("save_first_step", False),
            )
            or (keep_every > 0 and display_step % keep_every == 0)
        ):
            folder = output / f"step_{display_step:06d}"
            folder.mkdir(exist_ok=True)
            if rank == 0:
                (folder / "COMPLETE").unlink(missing_ok=True)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_start",
                            "step": display_step,
                            "output": str(folder),
                        }
                    ),
                    flush=True,
                )
            if world > 1:
                dist.barrier()
            if cfg.get("optimizer_cpu_offload", False):
                move_optimizer_state(optimizer, device)
            model_state, optim_state = checkpoint_state(student, optimizer)
            if cfg.get("optimizer_cpu_offload", False):
                move_optimizer_state(optimizer, "cpu")
                torch.cuda.empty_cache()
            if rank == 0:
                torch.save(
                    {
                        "student": model_state,
                        "optimizer": optim_state,
                        "step": display_step,
                        "config": cfg,
                    },
                    folder / "model.pt.tmp",
                )
                (folder / "model.pt.tmp").replace(folder / "model.pt")
            del model_state, optim_state
            if joint:
                joint.save(folder, display_step)
            torch.save(
                {
                    "world": world,
                    "batch": batch,
                    "streams": streams,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "python_rng": random.getstate(),
                    "total_batches": total_batches,
                },
                folder / f"rank_{rank}.pt",
            )
            if world > 1:
                dist.barrier()
            if rank == 0:
                (folder / "COMPLETE").touch()
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_complete",
                            "step": display_step,
                            "output": str(folder),
                        }
                    ),
                    flush=True,
                )
            if rank == 0 and keep_every > 0:
                removed = prune_checkpoints(
                    output,
                    display_step,
                    keep_every,
                    world,
                    (["joint_state.json"] + (["fake.pt"] if joint.enabled else []))
                    if joint
                    else [],
                )
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_retention",
                            "latest_step": display_step,
                            "keep_every": keep_every,
                            "removed": removed,
                        }
                    ),
                    flush=True,
                )
            if world > 1:
                dist.barrier()

    student.train()
    # Optional imports/preview initialization must not perturb the training RNG.
    with evaluation_state(student, device):
        monitor = TrainingMonitor(
            cfg, output, device,
            start=resume_step,
        )
    monitor.run_preview(
        student, resume_step,
        startup=True,
    )
    if joint and joint.fake_updates < warmup_steps:
        if rank == 0:
            print(json.dumps({"event": "fake_warmup_start", "completed": joint.fake_updates,
                              "target": warmup_steps, "student_init": cfg.get("student_init")}),
                  flush=True)
        for warmup in range(joint.fake_updates, warmup_steps):
            joint.begin_step(warmup, accum, shape, next_context, warmup=True)
            total_batches += accum
            for update_record in joint.fake_update_records:
                if rank == 0:
                    if updates_log:
                        updates_log.write(json.dumps(update_record) + "\n")
                    with open(output / "fake_warmup.jsonl", "a") as warmup_log:
                        warmup_log.write(json.dumps(update_record) + "\n")
                monitor.log_optimizer_update(update_record)
            if count_optimizer_updates:
                save_checkpoint(warmup + 1)
                monitor.run_preview(student, warmup + 1)
        if rank == 0:
            print(json.dumps({"event": "fake_warmup_complete", "fake_updates": joint.fake_updates}),
                  flush=True)
    for step in range(start, cycle_steps):
        display_step = step_offset + (step + 1) * updates_per_cycle if count_optimizer_updates else step + 1
        torch.cuda.synchronize()
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        # Retain DDP bucket views across steps, including no_sync accumulation.
        # Dropping them allocates an extra full FP32 gradient copy on the next microbatch.
        optimizer.zero_grad(set_to_none=False)
        losses, heads = [], []
        if joint:
            joint.begin_step(step, accum, shape, next_context)
            total_batches += accum
            if count_optimizer_updates:
                for update_record in joint.fake_update_records:
                    if rank == 0:
                        updates_log.write(json.dumps(update_record) + "\n")
                    monitor.log_optimizer_update(update_record)
        for micro in range(accum):
            if joint:
                loss, k = joint.generator_micro(micro, step)
                (loss / accum).backward()
                losses.append(loss.detach())
                heads.append(k.cpu().tolist())
            else:
                if streams[micro] is None:
                    context = next_context()
                    total_batches += 1
                    x, n = torch.randn(shape, device=device), 0
                else:
                    x, context, n = streams[micro]
                    x, context = x.to(device), context.to(device)
                k = torch.randint(
                    n,
                    min(n + Lmax, N),
                    (batch, 2 if cfg["solver"] == "euler" else 1),
                    device=device,
                )

                def velocity(state, sigma):
                    return guided_velocity(
                        teacher,
                        state,
                        sigma,
                        context,
                        negative,
                        cfg["guidance"],
                        cfg["skip_layer"],
                    )

                sync = (
                    student.no_sync()
                    if isinstance(student, DDP) and micro < accum - 1
                    else nullcontext()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    activation_context = (
                        torch.autograd.graph.save_on_cpu(pin_memory=True)
                        if cfg.get("activation_cpu_offload", False)
                        else nullcontext()
                    )
                    with activation_context:
                        loss, next_x = pd_loss(
                            student,
                            velocity,
                            x,
                            context,
                            grid,
                            n,
                            k,
                            Lmin,
                            cfg["solver"],
                        )
                    (loss / accum).backward()
                losses.append(loss.detach())
                heads.append(k.cpu().tolist())
                if cfg.get("offload_streams", False):
                    next_x, context = next_x.cpu(), context.cpu()
                streams[micro] = None if n + Lmin == N else (next_x, context, n + Lmin)
            micro_every = int(cfg.get("log_micro_every", 0))
            if (
                local == 0
                and micro_every > 0
                and (micro == 0 or (micro + 1) % micro_every == 0 or micro + 1 == accum)
            ):
                print(
                    f"[rank {rank}] step={display_step}/{cfg['steps']} "
                    f"micro={micro + 1}/{accum} loss_local={loss.detach().item():.8f} "
                    f"elapsed={time.perf_counter() - begin:.2f}s",
                    flush=True,
                )
        grad = (
            student.clip_grad_norm_(cfg["max_grad_norm"])
            if isinstance(student, FSDP)
            else torch.nn.utils.clip_grad_norm_(
                student.parameters(), cfg["max_grad_norm"]
            )
        )
        finite = torch.tensor(int(torch.isfinite(grad)), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"Non-finite gradients at step {display_step}")
        limit = cfg.get("abort_grad_norm_above")
        if limit is not None:
            excessive = torch.tensor(int(float(grad) > limit), device=device)
            if world > 1:
                dist.all_reduce(excessive, op=dist.ReduceOp.MAX)
            if excessive.item():
                raise FloatingPointError(
                    f"Generator gradient norm exceeded {limit} at step {display_step}"
                )
        if cfg.get("optimizer_cpu_offload", False):
            move_optimizer_state(optimizer, device)
        optimizer.step()
        if step == start:
            verify_optimizer_fp32(optimizer, device, "generator")
        if cfg.get("optimizer_cpu_offload", False):
            move_optimizer_state(optimizer, "cpu")
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        metrics = torch.tensor(
            [
                torch.stack(losses).mean().item(),
                elapsed,
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            ],
            device=device,
        )
        if world > 1:
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        loss_mean = torch.stack(losses).mean().clone()
        if world > 1:
            dist.all_reduce(loss_mean, op=dist.ReduceOp.SUM)
            loss_mean /= world
        record = {
            "step": display_step,
            "loss_mean": loss_mean.item(),
            "loss_rank_max": metrics[0].item(),
            "seconds": metrics[1].item(),
            "samples_per_second": batch * accum * world / metrics[1].item(),
            "peak_allocated_gib": metrics[2].item(),
            "peak_reserved_gib": metrics[3].item(),
            "grad_norm": float(grad),
            "global_batch": batch * accum * world,
            "head_indices_rank0": heads,
        }
        if joint:
            record.update(joint.metrics())
            if count_optimizer_updates:
                record["g_step"] = step + 1
                record["update_type"] = "generator"
                record["rollout_nfe"] = joint.g_rollout_nfe
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            if count_optimizer_updates:
                updates_log.write(json.dumps({
                    "step": display_step,
                    "update_type": "generator",
                    "rollout_nfe": joint.g_rollout_nfe,
                    "loss_mean": record["loss_mean"],
                    "grad_norm": record["grad_norm"],
                    "seconds": record["seconds"],
                    "global_batch": record["global_batch"],
                }) + "\n")
        if count_optimizer_updates:
            monitor.log_optimizer_update({
                "step": display_step,
                "update_type": "generator",
                "rollout_nfe": joint.g_rollout_nfe,
                "loss_mean": record["loss_mean"],
                "grad_norm": record["grad_norm"],
            })
        # One readable progress line per node, using globally reduced mean loss.
        if local == 0:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[rank {rank}] step={display_step}/{cfg['steps']} "
                f"loss={record['loss_mean']:.8f} "
                f"loss_rank_max={record['loss_rank_max']:.8f} "
                f"grad_norm={record['grad_norm']:.6f} "
                + (f"fake_grad_norm={record['joint/fake_grad_norm']:.6f} " if joint else "")
                + f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"step_seconds={record['seconds']:.2f}",
                flush=True,
            )
        if joint and local == 0:
            print(
                f"[joint] step={display_step} nfe={joint.g_rollout_nfe} "
                f"traj={record['joint/traj_loss']:.7f} "
                f"dmd={record['joint/dmd_loss']:.7f} fake={record['joint/fake_loss']:.7f} "
                f"dmd_weight={record['joint/dmd_weight']:.5f} fake_updates={joint.fake_updates}",
                flush=True,
            )
        monitor.log_step(record, optimizer.param_groups[0]["lr"])
        save_checkpoint(display_step)
        monitor.run_preview(student, display_step, final=display_step == cfg["steps"])
    monitor.close()
    if log:
        log.close()
    if updates_log:
        updates_log.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
