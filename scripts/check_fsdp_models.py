"""GPU regression: meta initialization, frozen-teacher CFG/skip, save and restore.
Run with torchrun --standalone --nproc_per_node=2; uses tiny synthetic tensors.
"""

import copy
import os
from pathlib import Path
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from diffusers import WanTransformer3DModel
from pdd.distributed import (
    wrap_student,
    checkpoint_state,
    restore_state,
    move_optimizer_state,
)
from pdd.model import PDDWan, guided_velocity
from pdd.core import sigma_grid, training_coefficients

rank = int(os.environ["RANK"])
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl")
torch.manual_seed(123)
kwargs = dict(
    num_attention_heads=2,
    attention_head_dim=8,
    in_channels=16,
    out_channels=16,
    freq_dim=8,
    ffn_dim=32,
    num_layers=2,
    text_dim=16,
)
base = WanTransformer3DModel(**kwargs)
reference = (
    copy.deepcopy(base).to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
)
if rank:
    with init_empty_weights():
        base = WanTransformer3DModel(**kwargs)
student = PDDWan(base, 4)
cfg = dict(
    distributed_strategy="fsdp",
    fsdp_shard_size=int(os.environ.get("FSDP_SHARD_SIZE", dist.get_world_size())),
)
student = wrap_student(student, cfg, device)
if rank:
    with init_empty_weights():
        teacher = WanTransformer3DModel(**kwargs)
    teacher = teacher.to(dtype=torch.bfloat16)
else:
    teacher = copy.deepcopy(reference).cpu()
teacher = wrap_student(teacher.eval().requires_grad_(False), cfg, device, frozen=True)
torch.manual_seed(456)
x = torch.randn(1, 16, 1, 2, 2, device=device)
c = torch.randn(1, 4, 16, device=device, dtype=torch.bfloat16)
neg = torch.randn_like(c)
grid = sigma_grid(4, device=device)
a = training_coefficients(grid, 0, 1, 1)
with torch.autocast("cuda", dtype=torch.bfloat16):
    for skip in [-1, 1]:
        actual = guided_velocity(teacher, x, grid[1], c, neg, 5, skip)
        expected = guided_velocity(reference, x, grid[1], c, neg, 5, skip)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
opt = torch.optim.AdamW(student.parameters(), lr=1e-4, foreach=False)
for _ in range(2):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        student(x, grid[0], c, a).square().mean().div(2).backward()
expected_grads = [
    None if p.grad is None else p.grad.detach().clone() for p in student.parameters()
]
opt.zero_grad(set_to_none=True)
for _ in range(2):
    with (
        torch.autograd.graph.save_on_cpu(pin_memory=True),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        loss = student(x, grid[0], c, a).square().mean().div(2)
    loss.backward()
for p, expected_grad in zip(student.parameters(), expected_grads):
    if expected_grad is None:
        assert p.grad is None
    else:
        torch.testing.assert_close(p.grad, expected_grad, rtol=0, atol=0)
assert torch.isfinite(student.clip_grad_norm_(1.0))
opt.step()
moments = {
    id(p): {k: v.clone() for k, v in state.items()} for p, state in opt.state.items()
}
move_optimizer_state(opt, "cpu")
assert all(
    v.device.type == "cpu" for state in opt.state.values() for v in state.values()
)
move_optimizer_state(opt, device)
for p, state in opt.state.items():
    for k, v in state.items():
        torch.testing.assert_close(v, moments[id(p)][k], rtol=0, atol=0)
state, optim = checkpoint_state(student, opt)
path = Path(os.environ.get("CHECK_FSDP_FILE", "/tmp/pdd_fsdp_models_test.pt"))
if rank == 0:
    torch.save(dict(student=state, optimizer=optim), path)
dist.barrier()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    before = student(x, grid[0], c, a).clone()
with torch.no_grad():
    for p in student.parameters():
        p.add_(0.1)
restore_state(student, opt, torch.load(path, map_location="cpu", weights_only=True))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    after = student(x, grid[0], c, a)
torch.testing.assert_close(before, after, rtol=0, atol=0)
assert not any(p.requires_grad for p in teacher.parameters())
if rank == 0:
    print(
        "PASS: meta sync, sharded teacher CFG with/without skip, accumulation, CPU activation offload gradient equality, Adam moment staging, checkpoint restore",
        flush=True,
    )
dist.destroy_process_group()
