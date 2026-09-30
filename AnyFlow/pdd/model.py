import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import WanTransformer3DModel
from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
from safetensors.torch import load_file
from accelerate import init_empty_weights


def load_wan(path, *, dtype=None, init_only=False):
    """Load native Alibaba weights without downloading or silently dropping keys."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    if cfg.get("_class_name") == "WanModel":
        with init_empty_weights():
            model = WanTransformer3DModel(
                num_attention_heads=cfg["num_heads"],
                attention_head_dim=cfg["dim"] // cfg["num_heads"],
                in_channels=cfg["in_dim"],
                out_channels=cfg["out_dim"],
                freq_dim=cfg["freq_dim"],
                ffn_dim=cfg["ffn_dim"],
                num_layers=cfg["num_layers"],
                eps=cfg["eps"],
            )
        if dtype is not None:
            model.to(dtype=dtype)
        if init_only:
            return model
        index_path = path / "diffusion_pytorch_model.safetensors.index.json"
        weight_map = (
            json.loads(index_path.read_text())["weight_map"]
            if index_path.exists()
            else None
        )
        files = (
            sorted(set(weight_map.values()))
            if weight_map
            else ["diffusion_pytorch_model.safetensors"]
        )
        expected, loaded = set(model.state_dict()), set()
        for filename in files:
            shard = load_file(str(path / filename))
            if weight_map is not None:
                indexed = {k for k, v in weight_map.items() if v == filename}
                if set(shard) != indexed:
                    raise ValueError(
                        f"Checkpoint index does not match shard {filename}"
                    )
            state = convert_wan_transformer_to_diffusers(shard)
            unexpected, duplicate = set(state) - expected, set(state) & loaded
            if unexpected or duplicate:
                raise ValueError(
                    f"Invalid checkpoint keys: unexpected={unexpected}, duplicate={duplicate}"
                )
            if dtype is not None:
                state = {
                    k: v.to(dtype=dtype) if v.is_floating_point() else v
                    for k, v in state.items()
                }
            model.load_state_dict(state, strict=False, assign=True)
            loaded.update(state)
            del state, shard
        if expected != loaded:
            raise ValueError(f"Missing checkpoint keys: {expected - loaded}")
        return model
    if init_only:
        with init_empty_weights():
            model = WanTransformer3DModel.from_config(cfg)
        return model.to(dtype=dtype) if dtype is not None else model
    return WanTransformer3DModel.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype or torch.float32
    )


class ParallelProjection(nn.Module):
    """N absolute-time heads. Fuse BEFORE unpatchify, preserving [patch, channel]."""

    def __init__(self, linear, num_heads, patch_volume, channels):
        super().__init__()
        self.weight = nn.Parameter(
            linear.weight.detach().unsqueeze(0).repeat(num_heads, 1, 1)
        )
        self.bias = nn.Parameter(linear.bias.detach().unsqueeze(0).repeat(num_heads, 1))
        self.patch_volume, self.channels = patch_volume, channels
        self.coefficients = None

    def forward(self, x):
        a = self.coefficients.to(device=self.weight.device, dtype=self.weight.dtype)
        if a.ndim == 3:
            if a.shape[0] != x.shape[0]:
                raise ValueError("Coefficient batch does not match latent batch")
            w = torch.einsum("bqn,noi->bqoi", a, self.weight)
            b = torch.einsum("bqn,no->bqo", a, self.bias)
            w = w.reshape(x.shape[0], -1, self.patch_volume, self.channels, w.shape[-1])
            w = w.permute(0, 2, 1, 3, 4).flatten(1, 3)
            b = b.reshape(x.shape[0], -1, self.patch_volume, self.channels)
            b = b.permute(0, 2, 1, 3).flatten(1)
            return torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
        if a.ndim != 2:
            raise ValueError("Coefficients must be [Q,N] or [B,Q,N]")
        w = torch.einsum("qn,noi->qoi", a, self.weight)
        b = a @ self.bias
        # Each projected output must be [patch, selected-head, channel].
        w = (
            w.reshape(-1, self.patch_volume, self.channels, w.shape[-1])
            .permute(1, 0, 2, 3)
            .flatten(0, 2)
        )
        b = b.reshape(-1, self.patch_volume, self.channels).permute(1, 0, 2).flatten()
        return F.linear(x, w, b)


class PDDWan(nn.Module):
    def __init__(self, backbone, num_heads):
        super().__init__()
        self.backbone = backbone
        self.num_heads = num_heads
        self.channels = backbone.config.out_channels
        backbone.proj_out = ParallelProjection(
            backbone.proj_out,
            num_heads,
            math.prod(backbone.config.patch_size),
            self.channels,
        )

    def forward(self, x, sigma, context, coefficients):
        self.backbone.proj_out.coefficients = coefficients
        y = self.backbone(
            x,
            timestep=sigma.expand(x.shape[0]) * 1000,
            encoder_hidden_states=context,
            return_dict=False,
        )[0]
        return y.unflatten(1, (coefficients.shape[-2], self.channels)).float()


class SkipBlock(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


@torch.no_grad()
def guided_velocity(teacher, x, sigma, context, negative, cfg=5.0, skip_layer=10):
    t = sigma.expand(x.shape[0]) * 1000
    cond = teacher(x, timestep=t, encoder_hidden_states=context, return_dict=False)[
        0
    ].float()
    if cfg == 1:
        return cond
    block = None
    if skip_layer >= 0:
        block = teacher.blocks[skip_layer]
        teacher.blocks[skip_layer] = SkipBlock()
    try:
        uncond = teacher(
            x,
            timestep=t,
            encoder_hidden_states=negative.expand(x.shape[0], -1, -1),
            return_dict=False,
        )[0].float()
    finally:
        if block is not None:
            teacher.blocks[skip_layer] = block
    return uncond + cfg * (cond - uncond)
