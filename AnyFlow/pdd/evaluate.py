import argparse
import copy
import importlib.util
import json
import os
import time
from pathlib import Path
import torch
from pdd.core import sample, sigma_grid, rollout_edges
from pdd.data import PromptEmbeddings
from pdd.model import PDDWan, guided_velocity, load_wan
from pdd.native import wan_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--student", help="Completed step directory; omit for initialized-head baseline"
    )
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--nfe", type=int, nargs="+", default=[4, 8])
    p.add_argument("--teacher-steps", type=int, default=50)
    p.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Generate student previews without a teacher comparison",
    )
    p.add_argument(
        "--official-teacher-root",
        help="Pinned official Wan2.1 source tree; enables UniPC50, shift8, CFG6, native full model",
    )
    p.add_argument(
        "--official-negative", help="Native UMT5 cache of official negative prompt"
    )
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--wan-root", default="../Wan2.2")
    p.add_argument("--decode", action="store_true")
    p.add_argument("--resume-evaluation", action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    for nfe in a.nfe:
        edges = rollout_edges(cfg["num_heads"], nfe)
        widths = [right - left for left, right in zip(edges[:-1], edges[1:])]
        if min(widths) < cfg["block_min"] or max(widths) > cfg["block_max"]:
            raise ValueError(f"NFE {nfe} is outside the trained block-size range")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    completed = set()
    if a.resume_evaluation:
        previous = json.loads((out / "evaluation_config.json").read_text())
        for key in (
            "student",
            "prompts",
            "nfe",
            "teacher_steps",
            "limit",
            "seeds",
            "decode",
            "skip_teacher",
            "official_teacher_root",
            "official_negative",
        ):
            if previous["arguments"].get(key) != vars(a).get(key):
                raise ValueError(f"Evaluation resume argument mismatch: {key}")
        for key in (
            "checkpoint",
            "num_heads",
            "shift",
            "block_min",
            "block_max",
            "height",
            "width",
            "frames",
            "guidance",
            "skip_layer",
            "solver",
        ):
            if previous["model_config"][key] != cfg[key]:
                raise ValueError(f"Evaluation resume model mismatch: {key}")
        for path in out.glob("metrics_rank*.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = (row["prompt_index"], row["seed"], row["mode"])
                stem = f"prompt{key[0]:04d}_seed{key[1]}_{key[2]}"
                extensions = [".pt", ".mp4"] if a.decode else [".pt"]
                if key in completed or any(
                    not (out / (stem + ext)).is_file()
                    or (out / (stem + ext)).stat().st_size == 0
                    for ext in extensions
                ):
                    raise ValueError(f"Inconsistent completed evaluation record: {key}")
                completed.add(key)
        print(
            json.dumps(
                {
                    "event": "evaluation_resume",
                    "rank": rank,
                    "completed": len(completed),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
            ),
            flush=True,
        )
        metric_file = out / f"metrics_rank{rank}_retry_{time.time_ns()}.jsonl"
    else:
        metric_file = out / f"metrics_rank{rank}.jsonl"
        if metric_file.exists():
            raise FileExistsError(
                "Choose a new evaluation output directory or --resume-evaluation"
            )
        if rank == 0:
            (out / "evaluation_config.json").write_text(
                json.dumps(
                    {
                        "model_config": cfg,
                        "arguments": vars(a),
                        "world_size": world,
                        "torch_version": torch.__version__,
                    },
                    indent=2,
                )
            )
    teacher = load_wan(cfg["checkpoint"])
    student = PDDWan(copy.deepcopy(teacher), cfg["num_heads"])
    if a.official_teacher_root:
        if a.skip_teacher or a.teacher_steps != 50 or not a.official_negative:
            raise ValueError(
                "Official teacher requires 50 steps and --official-negative"
            )
        del teacher
        native_class = wan_module(a.official_teacher_root, "model").WanModel
        teacher, info = native_class.from_pretrained(
            cfg["checkpoint"], output_loading_info=True
        )
        if any(
            info.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Native teacher checkpoint loading mismatch: {info}")
        source = Path(a.official_teacher_root) / "wan/utils/fm_solvers_unipc.py"
        spec = importlib.util.spec_from_file_location("pdd_official_unipc", source)
        solver_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solver_module)
        scheduler_class = solver_module.FlowUniPCMultistepScheduler
        print(
            json.dumps(
                {
                    "event": "official_teacher_loaded",
                    "rank": rank,
                    "steps": 50,
                    "solver": "unipc",
                    "shift": 8,
                    "guidance": 6,
                    "skip_layer": -1,
                }
            ),
            flush=True,
        )

    if a.student:
        folder = Path(a.student)
        if not (folder / "COMPLETE").is_file():
            raise ValueError("Incomplete student checkpoint")
        state = torch.load(
            folder / "model.pt", map_location="cpu", weights_only=True, mmap=True
        )
        for key in (
            "num_heads",
            "shift",
            "block_min",
            "block_max",
            "height",
            "width",
            "frames",
            "solver",
            "guidance",
            "skip_layer",
        ):
            if state["config"][key] != cfg[key]:
                raise ValueError(f"Checkpoint config mismatch: {key}")
        student.load_state_dict(state["student"], strict=True)
        del state
    # Keep both models on CPU until the sampling loop activates one at a time.
    student = student.to(dtype=torch.bfloat16).eval()
    teacher = (
        teacher.to(dtype=torch.float32 if a.official_teacher_root else torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )
    negative = torch.load(
        cfg["negative_embeddings"], map_location="cpu", weights_only=True
    ).to(device, dtype=torch.bfloat16)
    if negative.ndim == 2:
        negative = negative.unsqueeze(0)
    if a.official_teacher_root:
        negative = (
            PromptEmbeddings(a.official_negative)[0]["context"]
            .unsqueeze(0)
            .to(device, dtype=torch.bfloat16)
        )

    def teacher_velocity(x, sigma, context):
        if not a.official_teacher_root:
            return guided_velocity(
                teacher, x, sigma, context, negative, cfg["guidance"], cfg["skip_layer"]
            )
        t = sigma.reshape(1) * 1000
        seq_len = x.shape[2] * x.shape[3] * x.shape[4] // 4
        cond = teacher([x[0]], t=t, context=[context[0]], seq_len=seq_len)[0]
        uncond = teacher([x[0]], t=t, context=[negative[0]], seq_len=seq_len)[0]
        return (uncond + 6 * (cond - uncond)).unsqueeze(0)

    dataset = PromptEmbeddings(a.prompts)
    shape = (1, 16, (cfg["frames"] - 1) // 4 + 1, cfg["height"] // 8, cfg["width"] // 8)
    grid = sigma_grid(cfg["num_heads"], cfg["shift"], device)
    vae = None
    if a.decode:
        vae = wan_module(a.wan_root, "vae2_1").Wan2_1_VAE(
            vae_pth=str(Path(cfg["checkpoint"]) / "Wan2.1_VAE.pth"),
            dtype=torch.bfloat16,
            device=device,
        )
    with (
        open(metric_file, "w", buffering=1) as log,
        torch.no_grad(),
    ):
        for index in range(min(a.limit, len(dataset))):
            item = dataset[index]
            context = item["context"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            for seed_index, seed in enumerate(a.seeds):
                if (index * len(a.seeds) + seed_index) % world != rank:
                    continue
                modes = ([] if a.skip_teacher else ["teacher"]) + [
                    f"pdd_{n}" for n in a.nfe
                ]
                if all((index, seed, mode) in completed for mode in modes):
                    continue
                noise = torch.randn(
                    shape,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(seed),
                )
                reference = None
                for mode in modes:
                    if (index, seed, mode) in completed:
                        if mode == "teacher":
                            reference = torch.load(
                                out / f"prompt{index:04d}_seed{seed}_teacher.pt",
                                map_location=device,
                                weights_only=True,
                            )["latent"]
                        continue
                    # Keep only the active network on GPU; transfers and warmup are untimed.
                    if mode == "teacher":
                        student.to("cpu")
                        teacher.to(device)
                    else:
                        teacher.to("cpu")
                        student.to(device)
                    torch.cuda.empty_cache()
                    # Warmup outside timed sampling, including CUDA kernel initialization.
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if mode == "teacher":
                            teacher_velocity(noise, grid[0], context)
                        else:
                            coeff = grid.new_zeros(1, cfg["num_heads"])
                            coeff[0, 0] = 1
                            student(noise, grid[0], context, coeff)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    x = noise.clone()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if mode == "teacher":
                            if a.official_teacher_root:
                                scheduler = scheduler_class(
                                    num_train_timesteps=1000,
                                    shift=1,
                                    use_dynamic_shifting=False,
                                )
                                scheduler.set_timesteps(50, device=device, shift=8)
                                for solver_step, t in enumerate(scheduler.timesteps):
                                    v = teacher_velocity(x, t.float() / 1000, context)
                                    x = scheduler.step(v, t, x, return_dict=False)[0]
                                    if (solver_step + 1) % 10 == 0:
                                        print(
                                            json.dumps(
                                                {
                                                    "event": "teacher_progress",
                                                    "rank": rank,
                                                    "prompt_index": index,
                                                    "seed": seed,
                                                    "step": solver_step + 1,
                                                    "total": 50,
                                                }
                                            ),
                                            flush=True,
                                        )
                            else:
                                tg = sigma_grid(a.teacher_steps, cfg["shift"], device)
                                for j in range(a.teacher_steps):
                                    v = teacher_velocity(x, tg[j], context)
                                    x = x + (tg[j + 1] - tg[j]) * v
                        else:
                            x = sample(
                                student, x, context, grid, int(mode.split("_")[1])
                            )
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - begin
                    if not torch.isfinite(x).all():
                        raise FloatingPointError(f"Non-finite sample: {mode}")
                    if mode == "teacher":
                        reference = x.clone()
                    stem = f"prompt{index:04d}_seed{seed}_{mode}"
                    torch.save(
                        {"latent": x.cpu(), "prompt": item["prompt"], "seed": seed},
                        out / f"{stem}.pt",
                    )
                    record = {
                        "prompt_index": index,
                        "seed": seed,
                        "worker_rank": rank,
                        "worker_world_size": world,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "mode": mode,
                        "seconds": seconds,
                        "backbone_calls": a.teacher_steps
                        * (1 if cfg["guidance"] == 1 else 2)
                        if mode == "teacher"
                        else int(mode.split("_")[1]),
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "memory_scope": "active model, latent buffers, and VAE if decoding",
                        "latent_mse_to_teacher": (x - reference).square().mean().item()
                        if reference is not None
                        else None,
                        "latent_temporal_delta": (x[:, :, 1:] - x[:, :, :-1])
                        .abs()
                        .mean()
                        .item(),
                        "teacher_preset": "official_wan13b_unipc50_shift8_cfg6"
                        if a.official_teacher_root
                        else "training_teacher_euler",
                        "trained_checkpoint": a.student,
                        "height": cfg["height"],
                        "width": cfg["width"],
                        "frames": cfg["frames"],
                    }
                    if vae is not None:
                        import imageio.v3 as iio

                        video = vae.decode([x[0]])[0]
                        frames = (
                            ((video.permute(1, 2, 3, 0) + 1) * 127.5)
                            .clamp(0, 255)
                            .byte()
                            .cpu()
                            .numpy()
                        )
                        iio.imwrite(out / f"{stem}.mp4", frames, fps=16)
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
