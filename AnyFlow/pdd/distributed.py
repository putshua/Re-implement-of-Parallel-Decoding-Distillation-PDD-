"""Block-wise FSDP/HSDP for the student and optional frozen teacher."""

from functools import partial
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
import json


def verify_optimizer_fp32(optimizer, device, label):
    """Check live master shards, gradients and Adam moments after a real update."""
    counts = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            tensors = {"parameter": p, "gradient": p.grad}
            tensors.update(
                {
                    k: optimizer.state.get(p, {}).get(k)
                    for k in ("exp_avg", "exp_avg_sq")
                }
            )
            for kind, value in tensors.items():
                if value is not None and value.numel():
                    key = f"{kind}:{value.dtype}"
                    counts[key] = counts.get(key, 0) + value.numel()
    ok = all(key.endswith(":torch.float32") for key in counts)
    valid = torch.tensor(int(ok), device=device)
    if dist.is_initialized():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise RuntimeError(f"{label}: optimizer is not FP32: {counts}")
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "event": "optimizer_precision",
                    "model": label,
                    "counts_rank0": counts,
                    "all_ranks_fp32": True,
                }
            ),
            flush=True,
        )


def wrap_student(student, cfg, device, *, frozen=False):
    world = dist.get_world_size() if dist.is_initialized() else 1
    requested = int(cfg.get("fsdp_shard_size", 1))
    size = min(requested, world)
    if size < 1 or world % size:
        raise ValueError("FSDP shard size must be positive and divide world size")
    if cfg.get("distributed_strategy", "ddp") not in ("ddp", "fsdp"):
        raise ValueError("distributed_strategy must be ddp or fsdp")
    if cfg.get("distributed_strategy", "ddp") == "fsdp" and size > 1:
        replicas = world // size
        mesh = init_device_mesh(
            "cuda", (replicas, size), mesh_dim_names=("replicate", "shard")
        )
        block_type = type(student.blocks[0] if frozen else student.backbone.blocks[0])
        student = FSDP(
            student,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={block_type}
            ),
            device_mesh=mesh if replicas > 1 else mesh["shard"],
            sharding_strategy=ShardingStrategy.HYBRID_SHARD
            if replicas > 1
            else ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
                # Preserve FP32 noise time and integration coefficients at the root.
                # The training loop already autocasts compute operations to BF16.
                cast_root_forward_inputs=False,
            ),
            device_id=device,
            param_init_fn=lambda module: module.to_empty(device=device, recurse=False),
            sync_module_states=True,
            use_orig_params=True,
            limit_all_gathers=True,
        )
    elif world > 1 and not frozen:
        student = student.to(device)
        student = DDP(
            student,
            device_ids=[device.index],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    else:
        student = student.to(device)
    return student.eval() if frozen else student


def checkpoint_state(student, optimizer):
    if isinstance(student, FSDP):
        # All ranks participate in both collectives; only global rank 0 owns the result.
        with FSDP.state_dict_type(
            student,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return student.state_dict(), FSDP.optim_state_dict(student, optimizer)
    raw = student.module if isinstance(student, DDP) else student
    return raw.state_dict(), optimizer.state_dict()


def restore_state(student, optimizer, state):
    if isinstance(student, FSDP):
        with FSDP.state_dict_type(
            student,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            student.load_state_dict(state["student"], strict=True)
            optimizer.load_state_dict(
                FSDP.optim_state_dict_to_load(student, optimizer, state["optimizer"])
            )
    else:
        raw = student.module if isinstance(student, DDP) else student
        raw.load_state_dict(state["student"], strict=True)
        optimizer.load_state_dict(state["optimizer"])


def move_optimizer_state(optimizer, device):
    """Stage Adam moments without changing their dtype or CPU step counters."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device=device)
