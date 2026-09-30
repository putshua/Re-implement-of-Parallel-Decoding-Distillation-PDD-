"""Phase-balanced on-policy PDD + DMD training and fake-score state."""

import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from pdd.distributed import (
    FSDP,
    checkpoint_state,
    restore_state,
    wrap_student,
    move_optimizer_state,
    verify_optimizer_fp32,
)
from pdd.model import PDDWan, load_wan, guided_velocity
from pdd.core import rollout_edges
from pdd.phased_dmd import (
    conditional_renoise,
    fake_score_loss,
    endpoint_dmd_loss,
    joint_phase_forward,
)


def cyclic_rollout_nfe(update_index, nfes):
    if update_index < 0 or not nfes:
        raise ValueError("Invalid rollout cycle")
    return nfes[update_index % len(nfes)]


class JointTraining:
    def __init__(self, cfg, student, teacher, negative, grid, device, rank, world):
        self.cfg, self.student, self.teacher = cfg, student, teacher
        self.negative, self.grid, self.device = negative, grid, device
        self.rank, self.world = rank, world
        self.edges = cfg["phase_edges"]
        if self.edges != list(range(0, cfg["num_heads"] + 1, cfg["block_min"])):
            raise ValueError("phase_edges must tile block_min and num_heads")
        self.phases = len(self.edges) - 1
        self.rollout_nfes = list(cfg.get("rollout_nfes", [self.phases]))
        if not self.rollout_nfes or any(
            nfe < 1 or nfe > cfg["num_heads"] for nfe in self.rollout_nfes
        ):
            raise ValueError("rollout_nfes must contain valid positive step counts")
        for nfe in self.rollout_nfes:
            edges = rollout_edges(cfg["num_heads"], nfe)
            widths = [right - left for left, right in zip(edges[:-1], edges[1:])]
            if min(widths) < cfg["block_min"] or max(widths) > cfg["block_max"]:
                raise ValueError(f"Rollout NFE {nfe} is outside trained block range")
        self.count_optimizer_updates = bool(cfg.get("count_optimizer_updates", False))
        self.updates_per_cycle = int(cfg["fake_updates_per_g"]) + 1
        self.dmd_endpoint_mode = cfg.get("dmd_endpoint_mode", "phase")
        if self.dmd_endpoint_mode not in ("phase", "final"):
            raise ValueError("dmd_endpoint_mode must be phase or final")
        if self.dmd_endpoint_mode == "final" and cfg["traj_weight"] != 0:
            raise ValueError(
                "final endpoint mode does not implement trajectory loss; set traj_weight=0"
            )
        self.fake_phases = int(
            cfg.get(
                "fake_num_heads",
                self.phases if self.dmd_endpoint_mode == "phase" else 1,
            )
        )
        if self.fake_phases < 1:
            raise ValueError("fake_num_heads must be positive")
        if self.dmd_endpoint_mode == "phase" and self.fake_phases != self.phases:
            raise ValueError("Each phased endpoint needs its own fake head")
        self.enabled = cfg["dmd_weight"] > 0
        if cfg["dmd_weight"] < 0 or cfg["fake_updates_per_g"] < 1:
            raise ValueError("Invalid DMD weight/update ratio")
        self.fake_updates = 0
        self.fake = self.fake_optimizer = None
        if self.enabled:
            sharded = (
                cfg["distributed_strategy"] == "fsdp"
                and min(cfg["fsdp_shard_size"], world) > 1
            )
            fake = PDDWan(
                load_wan(
                    cfg["checkpoint"],
                    dtype=torch.float32,
                    init_only=sharded and rank != 0,
                ),
                self.fake_phases,
            )
            fake.backbone.enable_gradient_checkpointing()
            self.fake = wrap_student(fake, cfg, device)
            self.fake_optimizer = torch.optim.AdamW(
                self.fake.parameters(), lr=cfg["fake_lr"], foreach=False
            )
        self.rows = []

    def activation_context(self):
        return (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if self.cfg.get("activation_cpu_offload")
            else nullcontext()
        )

    def _rollout(self, noise, context, phase, nfe=None):
        if self.dmd_endpoint_mode == "final":
            return self._full_rollout(noise, context, nfe=nfe).detach()
        x = noise
        for left, right in zip(self.edges[: phase + 1], self.edges[1 : phase + 2]):
            a = self.grid.new_zeros(1, self.cfg["num_heads"])
            a[0, left:right] = self.grid.diff()[left:right]
            x = x.float() + self.student(x, self.grid[left], context, a)[:, 0]
        return x.detach()

    def _full_rollout(self, noise, context, nfe=None):
        """Differentiable complete student rollout for ordinary endpoint DMD."""
        x = noise
        edges = (
            self.edges
            if nfe is None
            else rollout_edges(self.cfg["num_heads"], nfe)
        )
        for left, right in zip(edges[:-1], edges[1:]):
            a = self.grid.new_zeros(1, self.cfg["num_heads"])
            a[0, left:right] = self.grid.diff()[left:right]
            x = x.float() + self.student(x, self.grid[left], context, a)[:, 0]
        return x

    def _score_sample(self, xs, phase):
        endpoint = (
            self.grid[-1]
            if self.dmd_endpoint_mode == "final"
            else self.grid[self.edges[phase + 1]]
        )
        s = endpoint.float()
        gap = self.cfg["dmd_min_gap"]
        low = torch.maximum(s + gap, s.new_tensor(self.cfg.get("dmd_min_time", gap)))
        high = s.new_tensor(self.cfg.get("dmd_max_time", 1 - gap))
        if not (0 <= s < low < high < 1):
            raise ValueError("DMD score-time interval is empty")
        shift = self.cfg["dmd_time_shift"]
        if shift <= 0:
            raise ValueError("dmd_time_shift must be positive")

        # Draw shifted uniform conditional on the valid interval, not clamping.
        def cdf(t):
            return t / (shift - (shift - 1) * t)

        u = cdf(low) + torch.rand(xs.shape[0], device=self.device) * (
            cdf(high) - cdf(low)
        )
        t = shift * u / (1 + (shift - 1) * u)
        noise = torch.randn_like(xs)
        xt, _, _ = conditional_renoise(xs.detach().float(), s, t, noise)
        return s, t, noise, xt

    def _fake_velocity(self, xt, t, context, phase):
        a = self.grid.new_zeros(1, self.fake_phases)
        a[0, min(phase, self.fake_phases - 1)] = 1
        # Wan expects one sigma for each example; PDDWan supports [B] times.
        return self.fake(xt, t, context, a)[:, 0]

    def begin_step(self, step, accum, shape, next_context):
        self.rows = []
        self.data = []
        self.fake_update_records = []
        self.g_rollout_nfe = cyclic_rollout_nfe(
            (step * self.updates_per_cycle + self.updates_per_cycle - 1)
            if self.count_optimizer_updates else step,
            self.rollout_nfes,
        )
        # Identical phase order across ranks; rank-local prompts/noise remain independent.
        order = list(range(self.phases))
        random.Random(self.cfg["seed"] + step).shuffle(order)
        input_rng = torch.Generator(device="cpu").manual_seed(
            self.cfg["seed"] + 1000003 * self.rank + 7919 * step
        )
        self.head_rng = torch.Generator(device=self.device).manual_seed(
            self.cfg["seed"] + 1000003 * self.rank + 7919 * step + 1
        )
        for micro in range(accum):
            self.data.append(
                (
                    torch.randn(shape, device="cpu", generator=input_rng),
                    next_context().cpu(),
                    (
                        self.phases - 1
                        if self.dmd_endpoint_mode == "final"
                        else order[micro % self.phases]
                    ),
                )
            )
        if not self.enabled:
            return
        self.fake.train()
        for update in range(self.cfg["fake_updates_per_g"]):
            update_started = time.perf_counter()
            nfe = cyclic_rollout_nfe(
                (step * self.updates_per_cycle + update)
                if self.count_optimizer_updates else step,
                self.rollout_nfes,
            )
            update_losses = []
            self.fake_optimizer.zero_grad(set_to_none=True)
            for micro, (noise, context, phase) in enumerate(self.data):
                noise, context = noise.to(self.device), context.to(self.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    xs = self._rollout(noise, context, phase, nfe=nfe)
                s, t, eps, xt = self._score_sample(xs, phase)
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16),
                    self.activation_context(),
                ):
                    v = self._fake_velocity(xt, t, context, phase)
                    if self.dmd_endpoint_mode == "final":
                        # Ordinary RF flow matching: do not suppress the low-noise
                        # score learning with the phased variance-cap weighting.
                        loss = (
                            (v.float() - (eps.float() - xs.detach().float()))
                            .square()
                            .mean()
                        )
                    else:
                        loss = fake_score_loss(
                            v.float(),
                            xs.float(),
                            s,
                            t,
                            eps,
                            self.cfg["fake_variance_cap"],
                        )
                (loss / accum).backward()
                self.rows.append((phase, "fake_loss", loss.detach().item()))
                update_losses.append(loss.detach().item())
                if self.rank == 0 and (
                    micro == 0 or (micro + 1) % self.cfg["log_micro_every"] == 0
                ):
                    print(
                        f"[fake] step={step * self.updates_per_cycle + update + 1} "
                        f"Gcycle={step + 1} update={update + 1} nfe={nfe} "
                        f"micro={micro + 1}/{accum} phase={phase} loss={loss.item():.7f}",
                        flush=True,
                    )
            grad = (
                self.fake.clip_grad_norm_(self.cfg["max_grad_norm"])
                if isinstance(self.fake, FSDP)
                else torch.nn.utils.clip_grad_norm_(
                    self.fake.parameters(), self.cfg["max_grad_norm"]
                )
            )
            finite = torch.tensor(int(torch.isfinite(grad)), device=self.device)
            if self.world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite fake gradient")
            excessive = torch.tensor(
                int(float(grad) > self.cfg.get("abort_grad_norm_above", float("inf"))),
                device=self.device,
            )
            if self.world > 1:
                dist.all_reduce(excessive, op=dist.ReduceOp.MAX)
            if excessive.item():
                raise FloatingPointError(
                    f"Fake gradient norm {float(grad):.6g} exceeds configured limit"
                )
            self.rows.append((phase, "fake_grad_norm", float(grad)))
            self.rows.append((phase, "fake_rollout_nfe", float(nfe)))
            summary = torch.tensor(
                [sum(update_losses), len(update_losses)],
                device=self.device,
                dtype=torch.float64,
            )
            if self.world > 1:
                dist.all_reduce(summary)
            self.fake_update_records.append({
                "step": step * self.updates_per_cycle + update + 1,
                "update_type": "fake",
                "rollout_nfe": nfe,
                "loss_mean": (summary[0] / summary[1]).item(),
                "grad_norm": float(grad),
                "seconds": time.perf_counter() - update_started,
                "global_batch": self.data[0][0].shape[0] * accum * self.world,
            })
            if self.rank == 0:
                print(
                    f"[fake] step={step * self.updates_per_cycle + update + 1} "
                    f"Gcycle={step + 1} update={update + 1} nfe={nfe} "
                    f"grad_norm={float(grad):.6f}",
                    flush=True,
                )
            if self.cfg.get("optimizer_cpu_offload", False):
                move_optimizer_state(self.fake_optimizer, self.device)
            self.fake_optimizer.step()
            if self.fake_updates == 0:
                verify_optimizer_fp32(self.fake_optimizer, self.device, "fake")
            self.fake_optimizer.zero_grad(set_to_none=True)
            if self.cfg.get("optimizer_cpu_offload", False):
                move_optimizer_state(self.fake_optimizer, "cpu")
                torch.cuda.empty_cache()
            self.fake_updates += 1
        self.fake.eval()

    def generator_micro(self, micro, step):
        noise, context, phase = self.data[micro]
        noise, context = noise.to(self.device), context.to(self.device)
        n, end = self.edges[phase : phase + 2]
        k = torch.randint(
            n, end, (noise.shape[0], 1), device=self.device, generator=self.head_rng
        )

        def velocity(x, t):
            return guided_velocity(
                self.teacher,
                x,
                t,
                context,
                self.negative,
                self.cfg["guidance"],
                self.cfg["skip_layer"],
            )

        with torch.autocast("cuda", dtype=torch.bfloat16), self.activation_context():
            if self.dmd_endpoint_mode == "final":
                trajectory = noise.new_zeros(())
                xs = self._full_rollout(noise, context, nfe=getattr(self, "g_rollout_nfe", None))
            else:
                trajectory, xs = joint_phase_forward(
                    self.student,
                    velocity,
                    noise,
                    context,
                    self.grid,
                    self.edges,
                    phase,
                    k,
                    self.cfg["solver"],
                )
        dmd = trajectory.new_zeros(())
        if self.enabled:
            s, t, eps, xt = self._score_sample(xs, phase)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                fake = self._fake_velocity(xt, t, context, phase)
                real = guided_velocity(
                    self.teacher,
                    xt,
                    t,
                    context,
                    self.negative,
                    self.cfg["guidance"],
                    self.cfg["dmd_teacher_skip_layer"],
                )
            dmd, direction = endpoint_dmd_loss(
                xs,
                real.float(),
                fake.float(),
                s,
                t,
                self.cfg["dmd_normalization"],
                noisy_sample=xt,
            )
            self.rows += [
                (phase, "direction_rms", direction.square().mean().sqrt().item()),
                (phase, "score_time", t.mean().item()),
                (
                    phase,
                    "endpoint_rms",
                    xs.detach().float().square().mean().sqrt().item(),
                ),
                (phase, "score_time_min", t.min().item()),
            ]
        ramp = min(1.0, (step + 1) / max(1, self.cfg["dmd_ramp_steps"]))
        weight = self.cfg["dmd_weight"] * ramp
        self.rows += [
            (phase, "traj_loss", trajectory.detach().item()),
            (phase, "dmd_loss", dmd.detach().item()),
        ]
        self.weight = weight
        return self.cfg["traj_weight"] * trajectory + weight * dmd, k

    def metrics(self):
        values = {}
        for name in (
            "traj_loss",
            "dmd_loss",
            "fake_loss",
            "fake_grad_norm",
            "fake_rollout_nfe",
            "direction_rms",
            "score_time",
            "endpoint_rms",
            "score_time_min",
        ):
            for phase in range(self.phases):
                rows = [v for p, key, v in self.rows if p == phase and key == name]
                pair = torch.tensor(
                    [sum(rows), len(rows)], device=self.device, dtype=torch.float64
                )
                if self.world > 1:
                    dist.all_reduce(pair)
                values[f"phase/{phase}/{name}"] = (
                    (pair[0] / pair[1]).item() if pair[1] else 0.0
                )
        for name in ("traj_loss", "dmd_loss", "fake_loss", "fake_grad_norm"):
            rows = [v for _, key, v in self.rows if key == name]
            pair = torch.tensor(
                [sum(rows), len(rows)], device=self.device, dtype=torch.float64
            )
            if self.world > 1:
                dist.all_reduce(pair)
            values[f"joint/{name}"] = (pair[0] / pair[1]).item() if pair[1] else 0.0
        values["joint/dmd_weight"] = self.weight
        values["joint/fake_updates"] = self.fake_updates
        values["joint/g_rollout_nfe"] = getattr(self, "g_rollout_nfe", self.phases)
        return values

    def save(self, folder, step):
        if self.enabled:
            if self.cfg.get("optimizer_cpu_offload", False):
                move_optimizer_state(self.fake_optimizer, self.device)
            model, optimizer = checkpoint_state(self.fake, self.fake_optimizer)
            if self.cfg.get("optimizer_cpu_offload", False):
                move_optimizer_state(self.fake_optimizer, "cpu")
            if self.rank == 0:
                torch.save(
                    {"student": model, "optimizer": optimizer, "step": step},
                    folder / "fake.pt.tmp",
                )
                (folder / "fake.pt.tmp").replace(folder / "fake.pt")
            del model, optimizer
        if self.rank == 0:
            (folder / "joint_state.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "step": step,
                        "fake_updates": self.fake_updates,
                        "dmd_enabled": self.enabled,
                    }
                )
            )

    def restore(self, folder, step):
        folder = Path(folder)
        info = json.loads((folder / "joint_state.json").read_text())
        if (
            info["version"] != 1
            or info["step"] != step
            or info["dmd_enabled"] != self.enabled
        ):
            raise ValueError("Joint checkpoint mismatch")
        if self.enabled:
            state = torch.load(
                folder / "fake.pt", map_location="cpu", mmap=True, weights_only=True
            )
            if state["step"] != step:
                raise ValueError("Fake checkpoint step mismatch")
            restore_state(self.fake, self.fake_optimizer, state)
            if self.cfg.get("optimizer_cpu_offload", False):
                move_optimizer_state(self.fake_optimizer, "cpu")
        self.fake_updates = info["fake_updates"]
