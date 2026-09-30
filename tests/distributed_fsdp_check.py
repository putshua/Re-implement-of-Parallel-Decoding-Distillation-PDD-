"""Four-GPU communication/checkpoint contract on a tiny Wan; not a resolution benchmark."""

import argparse
import copy
import os
from pathlib import Path
import torch
import torch.distributed as dist
from diffusers import WanTransformer3DModel
from pdd.model import PDDWan
from pdd.core import pd_loss, sigma_grid
from pdd.distributed import FSDP, wrap_student, checkpoint_state, restore_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local)
    torch.manual_seed(11)
    base = PDDWan(
        WanTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            text_dim=12,
            freq_dim=16,
            ffn_dim=48,
            num_layers=2,
        ),
        8,
    )

    def check_root_precision(module, inputs):
        assert inputs[1].dtype == torch.float32
        assert inputs[3].dtype == torch.float32

    base.register_forward_pre_hook(check_root_precision)
    base.backbone.enable_gradient_checkpointing()
    cfg = {"distributed_strategy": "fsdp", "fsdp_shard_size": args.shard_size}
    student = wrap_student(copy.deepcopy(base).to(device), cfg, device)
    assert isinstance(student, FSDP)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, foreach=False)
    torch.manual_seed(123 + rank)
    x = torch.randn(2, 4, 1, 4, 4, device=device)
    context = torch.randn(2, 8, 12, device=device)
    grid = sigma_grid(8, device=device)
    indices = torch.tensor([[rank % 4], [(rank + 2) % 4]], device=device)

    def update(model, opt):
        model.train()
        opt.zero_grad(set_to_none=False)
        losses = []
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = pd_loss(
                    model,
                    lambda y, t: y * 0.3 + t[:, None, None, None, None],
                    x,
                    context,
                    grid,
                    0,
                    indices,
                    2,
                    "midpoint",
                )
                (loss / 2).backward()
                losses.append(loss.detach())
        norm = model.clip_grad_norm_(1.0)
        assert torch.isfinite(norm)
        opt.step()
        return torch.stack(losses), norm

    update(student, optimizer)
    weights, optim_state = checkpoint_state(student, optimizer)
    folder = Path(args.output)
    folder.mkdir(exist_ok=True, parents=True)
    if rank == 0:
        # Full model state has no FSDP-specific key names and loads into plain PDDWan.
        base.load_state_dict(weights, strict=True)
        torch.save({"student": weights, "optimizer": optim_state}, folder / "model.pt")
    dist.barrier()
    del weights, optim_state
    expected, expected_norm = update(student, optimizer)
    restored = wrap_student(copy.deepcopy(base).to(device), cfg, device)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3, foreach=False)
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    restore_state(restored, restored_opt, state)
    actual, actual_norm = update(restored, restored_opt)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_norm, expected_norm, atol=1e-5, rtol=1e-4)
    # Different-rank head selections must produce one synchronized global update.
    weights, _ = checkpoint_state(restored, restored_opt)
    if rank == 0:
        assert any((weights[k] - state["student"][k]).abs().max() > 0 for k in weights)
        print(
            f"PASS: FSDP shard={args.shard_size}, world={dist.get_world_size()}, accumulation, clipping, full-state export, exact next-step resume",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
