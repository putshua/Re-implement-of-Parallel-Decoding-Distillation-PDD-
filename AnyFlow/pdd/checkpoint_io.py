"""Warm native checkpoint pages with bounded-memory sequential reads on shared disks."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path


def prefetch_checkpoint(path, workers=6):
    path = Path(path)
    index = path / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["diffusion_pytorch_model.safetensors"]
    if not files or workers < 1:
        raise ValueError("Need checkpoint files and positive prefetch workers")

    def read(name):
        size = 0
        with (path / name).open("rb") as stream:
            while chunk := stream.read(16 * 1024**2):
                size += len(chunk)
        if size == 0:
            raise ValueError(f"Empty checkpoint shard: {name}")
        return size

    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
        return sum(pool.map(read, files))
