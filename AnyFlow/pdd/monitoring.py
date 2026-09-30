"""Live training scalars and collective-safe fixed-prompt student previews."""

from contextlib import contextmanager
import gc
import hashlib
import html
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from pdd.core import sample, sigma_grid, rollout_edges
from pdd.data import PromptEmbeddings
from pdd.native import wan_module


@contextmanager
def evaluation_state(model, device):
    modes = [(module, module.training) for module in model.modules()]
    coefficients = [
        (m, m.coefficients) for m in model.modules() if hasattr(m, "coefficients")
    ]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            with torch.no_grad():
                yield
    finally:
        for module, mode in modes:
            module.training = mode
        for module, value in coefficients:
            module.coefficients = value
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def prompt_slots(count, world, rank):
    """Pad rounds so every FSDP rank executes exactly the same number of forwards."""
    for offset in range(0, count, world):
        index = offset + rank
        yield (index if index < count else 0), index < count


def preview_due(step, every, *, startup=False, at_start=True, final=False, at_end=True):
    return (
        at_start if startup else step > 0 and (step % every == 0 or (final and at_end))
    )


def collective_error(error, device):
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(
            "Preview/monitor error: "
            + "; ".join(f"rank {i}: {e}" for i, e in enumerate(errors) if e)
        )


def barrier():
    if dist.is_initialized():
        dist.barrier()


def write_preview_media(video, stem, fps=16, preview_width=208):
    """Full-resolution MP4 + small animated GIF and three-frame contact sheet."""
    import imageio.v3 as iio
    from PIL import Image

    frames = (
        ((video.permute(1, 2, 3, 0).float() + 1) * 127.5)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
    temporary = stem.with_suffix(".tmp.mp4")
    iio.imwrite(temporary, frames, fps=fps)
    temporary.replace(stem.with_suffix(".mp4"))
    height = max(1, round(frames.shape[1] * preview_width / frames.shape[2]))
    thumbnails = [Image.fromarray(f).resize((preview_width, height)) for f in frames]
    gif = stem.with_suffix(".tmp.gif")
    thumbnails[0].save(
        gif,
        save_all=True,
        append_images=thumbnails[1:],
        duration=round(1000 / fps),
        loop=0,
    )
    gif.replace(stem.with_suffix(".gif"))
    strip = Image.new("RGB", (preview_width * 3, height))
    for col, index in enumerate([0, len(thumbnails) // 2, len(thumbnails) - 1]):
        strip.paste(thumbnails[index], (col * preview_width, 0))
    strip.save(stem.with_suffix(".jpg"))
    return preview_width, height


class TrainingMonitor:
    def __init__(self, cfg, output, device, start=0):
        self.cfg, self.output, self.device = cfg, Path(output), device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.writer = None
        self.vae = None
        self.enabled = cfg.get("fixed_prompt_enabled", False)
        self.every = int(cfg.get("fixed_prompt_every", 25))
        self.log_every = int(cfg.get("tensorboard_every", 1))
        self.at_start = cfg.get("fixed_prompt_at_start", True)
        self.at_end = cfg.get("fixed_prompt_at_end", True)
        self.decode = cfg.get("fixed_prompt_decode", True)
        self.nfes = list(cfg.get("fixed_prompt_nfe", [4, 8]))
        self.seed = int(cfg.get("fixed_prompt_seed", 42))
        self.preview_width = int(cfg.get("fixed_prompt_preview_width", 208))
        error = None
        try:
            if min(self.every, self.log_every, self.preview_width) < 1:
                raise ValueError(
                    "Logging/preview intervals and preview width must be positive"
                )
            if self.enabled:
                if not self.nfes or len(set(self.nfes)) != len(self.nfes):
                    raise ValueError("fixed_prompt_nfe must be a nonempty unique list")
                for nfe in self.nfes:
                    try:
                        edges = rollout_edges(cfg["num_heads"], nfe)
                    except ValueError as exc:
                        raise ValueError(f"Invalid preview NFE {nfe}") from exc
                    widths = [b - a for a, b in zip(edges[:-1], edges[1:])]
                    if min(widths) < cfg["block_min"] or max(widths) > cfg["block_max"]:
                        raise ValueError(
                            f"Preview NFE {nfe} is outside the trained block range"
                        )
                self.dataset = PromptEmbeddings(cfg["fixed_prompt_embeddings"])
                self.count = int(cfg.get("fixed_prompt_count", 4))
                if not 1 <= self.count <= len(self.dataset):
                    raise ValueError(
                        "fixed_prompt_count must fit the fixed embedding dataset"
                    )
                self.spec = {
                    "prompts": self.dataset.prompts[: self.count],
                    "seed": self.seed,
                    "nfe": self.nfes,
                    "height": cfg["height"],
                    "width": cfg["width"],
                    "frames": cfg["frames"],
                    "num_heads": cfg["num_heads"],
                    "shift": cfg["shift"],
                    "decode": self.decode,
                    "preview_width": self.preview_width,
                    "source": str(Path(cfg["fixed_prompt_embeddings"]).resolve()),
                }
                digest = hashlib.sha256(
                    json.dumps(self.spec, sort_keys=True, ensure_ascii=False).encode()
                )
                for index in range(self.count):
                    digest.update(
                        self.dataset[index]["context"]
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
                self.signature = digest.hexdigest()
            if cfg.get("tensorboard_enabled", True) and self.rank == 0:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(
                    str(self.output / "tensorboard"),
                    purge_step=start + 1 if cfg.get("resume") else None,
                    flush_secs=10,
                )
                self.writer.add_text("run/config", json.dumps(cfg, indent=2), start)
                self.writer.flush()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        collective_error(error, device)

    def log_step(self, record, lr):
        if self.writer is None or record["step"] % self.log_every:
            return
        values = {
            "train/loss": record["loss_mean"],
            "train/loss_rank_max": record["loss_rank_max"],
            "train/grad_norm": record["grad_norm"],
            "train/lr": lr,
            "train/global_batch": record["global_batch"],
            "perf/step_seconds": record["seconds"],
            "perf/samples_per_second": record["samples_per_second"],
            "memory/peak_allocated_gib": record["peak_allocated_gib"],
            "memory/peak_reserved_gib": record["peak_reserved_gib"],
        }
        values.update(
            {k: v for k, v in record.items() if k.startswith(("phase/", "joint/"))}
        )
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, record["step"])
        self.writer.flush()

    def log_optimizer_update(self, record):
        if self.writer is None or record["step"] % self.log_every:
            return
        prefix = f"updates/{record['update_type']}"
        for key in ("loss_mean", "grad_norm", "rollout_nfe"):
            self.writer.add_scalar(f"{prefix}/{key}", record[key], record["step"])
        self.writer.flush()

    def run_preview(self, student, step, *, startup=False, final=False):
        if not self.enabled or not preview_due(
            step,
            self.every,
            startup=startup,
            at_start=self.at_start,
            final=final,
            at_end=self.at_end,
        ):
            return
        root = self.output / "fixed_prompt" / f"step_{step:06d}"
        # Read once on rank0 and broadcast so nodes cannot take divergent collective paths.
        status = [None, str(root)]
        if self.rank == 0:
            try:
                # Keep old previews when the fixed prompt set changes on resume.
                if (root / "COMPLETE").exists():
                    previous = json.loads((root / "manifest.json").read_text())
                    if previous["signature"] != self.signature:
                        root = root.with_name(root.name + "_" + self.signature[:12])
                status[1] = str(root)
                if (root / "COMPLETE").exists():
                    spec = json.loads((root / "manifest.json").read_text())
                    status[0] = (
                        "skip"
                        if spec["signature"] == self.signature
                        else "error: preview signature collision"
                    )
                else:
                    status[0] = "run"
            except Exception as exc:
                status[0] = f"error: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(status, src=0)
        if status[0] == "skip":
            return
        if status[0] != "run":
            raise RuntimeError(status[0])
        root = Path(status[1])
        started = time.perf_counter()
        if self.rank == 0:
            print(
                json.dumps(
                    {
                        "event": "fixed_prompt_start",
                        "step": step,
                        "count": self.count,
                        "nfe": self.nfes,
                    }
                ),
                flush=True,
            )
        root.mkdir(parents=True, exist_ok=True)
        pending, records = [], []
        with evaluation_state(student, self.device):
            cfg = self.cfg
            grid = sigma_grid(cfg["num_heads"], cfg["shift"], self.device)
            shape = (
                1,
                16,
                (cfg["frames"] - 1) // 4 + 1,
                cfg["height"] // 8,
                cfg["width"] // 8,
            )
            for index, keep in prompt_slots(self.count, self.world, self.rank):
                item = self.dataset[index]
                context = (
                    item["context"].unsqueeze(0).to(self.device, dtype=torch.bfloat16)
                )
                noise = torch.randn(
                    shape,
                    device=self.device,
                    generator=torch.Generator(device=self.device).manual_seed(
                        self.seed
                    ),
                )
                for nfe in self.nfes:
                    with torch.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                        cache_enabled=False,
                    ):
                        latent = sample(student, noise.clone(), context, grid, nfe)
                    if not torch.isfinite(latent).all():
                        raise FloatingPointError(
                            f"Non-finite preview at step {step}, prompt {index}"
                        )
                    if keep:
                        stem = f"prompt{index:04d}_seed{self.seed}_pdd_{nfe}"
                        pending.append((stem, latent[0].cpu()))
                        records.append(
                            {
                                "stem": stem,
                                "prompt_index": index,
                                "prompt": item["prompt"],
                                "seed": self.seed,
                                "nfe": nfe,
                                "step": step,
                                "rank": self.rank,
                                "weights": "current_student",
                            }
                        )
                    del latent
                del context, noise
            barrier()  # No VAE or disk work overlaps sharded model collectives.
            error = None
            try:
                if self.decode and pending:
                    if self.vae is None:
                        self.vae = wan_module(cfg["wan_root"], "vae2_1").Wan2_1_VAE(
                            vae_pth=str(Path(cfg["checkpoint"]) / "Wan2.1_VAE.pth"),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                    else:
                        self.vae.model.to(self.device)
                for (stem, latent), record in zip(pending, records):
                    path = root / stem
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save({"latent": latent, "metadata": record}, temporary)
                    temporary.replace(path.with_suffix(".pt"))
                    if self.decode:
                        video = self.vae.decode([latent.to(self.device)])[0]
                        write_preview_media(
                            video, path, preview_width=self.preview_width
                        )
                        del video
                (root / f"rank_{self.rank:05d}.json").write_text(
                    json.dumps(records, ensure_ascii=False, indent=2)
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if self.vae is not None:
                    self.vae.model.to("cpu")
                pending.clear()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            collective_error(error, self.device)
            error = None
            if self.rank == 0:
                try:
                    all_records = [
                        r
                        for rank in range(self.world)
                        for r in json.loads(
                            (root / f"rank_{rank:05d}.json").read_text()
                        )
                    ]
                    all_records.sort(key=lambda r: (r["prompt_index"], r["nfe"]))
                    self._publish(
                        root, step, all_records, time.perf_counter() - started
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            collective_error(error, self.device)
        barrier()
        if self.rank == 0:
            print(
                json.dumps(
                    {
                        "event": "fixed_prompt_complete",
                        "step": step,
                        "seconds": time.perf_counter() - started,
                        "output": str(root),
                    }
                ),
                flush=True,
            )

    def _publish(self, root, step, records, seconds):
        manifest = {
            "step": step,
            "signature": self.signature,
            "settings": self.spec,
            "seconds": seconds,
            "samples": records,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )
        cards = []
        for record in records:
            stem = record["stem"]
            prompt = html.escape(record["prompt"])
            if self.decode:
                cards.append(
                    f'<article><h3>Prompt {record["prompt_index"]} · {record["nfe"]} NFE · seed {self.seed}</h3><p>{prompt}</p><video controls loop preload="metadata" width="416" src="{stem}.mp4"></video></article>'
                )
                if self.writer is not None:
                    import imageio.v3 as iio
                    from PIL import Image
                    from tensorboard.compat.proto.summary_pb2 import Summary

                    tag = f"fixed_prompt/prompt{record['prompt_index']:04d}/pdd_{record['nfe']}"
                    self.writer.add_image(
                        tag + "/frames",
                        iio.imread(root / f"{stem}.jpg"),
                        step,
                        dataformats="HWC",
                    )
                    # TensorBoard's add_video also stores an animated GIF in Summary.Image.
                    with Image.open(root / f"{stem}.gif") as gif:
                        width, height = gif.size
                    summary = Summary(
                        value=[
                            Summary.Value(
                                tag=tag + "/video",
                                image=Summary.Image(
                                    height=height,
                                    width=width,
                                    colorspace=3,
                                    encoded_image_string=(
                                        root / f"{stem}.gif"
                                    ).read_bytes(),
                                ),
                            )
                        ]
                    )
                    self.writer.file_writer.add_summary(summary, step)
                    self.writer.add_text(tag + "/prompt", record["prompt"], step)
        page = f'<!doctype html><html><meta charset="utf-8"><title>PDD fixed prompts — step {step}</title><style>body{{font-family:sans-serif;margin:24px;background:#161616;color:#eee}}main{{display:flex;flex-wrap:wrap;gap:24px}}article{{width:416px}}p{{max-height:100px;overflow:auto}}</style><h1>PDD current student — step {step}</h1><p>{self.spec["width"]}×{self.spec["height"]}, {self.spec["frames"]} frames. Fixed seed {self.seed}.</p><main>{"".join(cards)}</main></html>'
        (root / "index.html").write_text(page)
        if self.writer is not None:
            self.writer.add_scalar("preview/seconds", seconds, step)
            self.writer.flush()
        (root / "COMPLETE").touch()
        latest = root.parent / "index.html.tmp"
        latest.write_text(
            f'<!doctype html><meta http-equiv="refresh" content="0;url={root.name}/index.html"><a href="{root.name}/index.html">Latest preview: step {step}</a>'
        )
        latest.replace(root.parent / "index.html")

    def close(self):
        if self.writer is not None:
            self.writer.close()
