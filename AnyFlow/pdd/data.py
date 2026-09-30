import io
import json
import random
import tarfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class PromptEmbeddings(Dataset):
    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.embeddings = blob["t5_text_embeddings"]
        self.prompts = blob["prompts"]
        if len(self.prompts) != len(self.embeddings):
            raise ValueError("Prompt/embedding count mismatch")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return {"context": self.embeddings[i], "prompt": self.prompts[i]}


class WeightedPromptCache(IterableDataset):
    """Bidirectional 450613-row weight index; prompt-only, original cache resolutions.

    Draw shards proportional to max(weight), then scan every member and accept each
    member with weight/max(weight). Accepted prompts follow the SFT weights.
    Each rank/worker owns an independent reproducible RNG stream.
    """

    def __init__(self, index, seed=42, rank=0, path_map=None):
        self.index = Path(index)
        self.meta = json.loads(self.index.read_text())
        if (
            self.meta["version"] != "taxonomy_wds_repeat_weight_v1"
            or not self.meta["coverage_complete"]
        ):
            raise ValueError("Incomplete or unsupported SFT weight index")
        self.seed, self.rank = seed, rank
        self.path_map = path_map or {}
        self.entries = []
        for root, filename in self.meta["arrays"].items():
            mapped = root
            for src, dst in self.path_map.items():
                if mapped.startswith(src):
                    mapped = dst + mapped[len(src) :]
                    break
            weights = np.load(self.index.parent / filename, mmap_mode="r")
            shards = sorted(Path(mapped).glob("shard_*.tar"))
            if not shards:
                raise FileNotFoundError(
                    f"No prompt cache shards at {mapped}; use path_map or prepare_embeddings.py"
                )
            # Cache shards contain consecutive groups of 256 source sample IDs.
            for shard in shards:
                sid = int(shard.stem.split("_")[-1])
                part = weights[sid * 256 : (sid + 1) * 256]
                if len(part) and part.max() > 0:
                    self.entries.append(
                        (
                            str(shard),
                            weights,
                            float(part.max()),
                            float(part.max()),
                        )
                    )
        if not self.entries:
            raise ValueError("No positive-weight shards")

    def __iter__(self):
        worker = get_worker_info()
        rng = random.Random(
            self.seed + 1000003 * self.rank + (worker.id if worker else 0)
        )
        masses = [e[3] for e in self.entries]
        while True:
            path, weights, maximum, _ = rng.choices(self.entries, weights=masses, k=1)[
                0
            ]
            with tarfile.open(path) as tar:
                for member in tar:
                    if not member.name.endswith(".embed.pt"):
                        continue
                    key = int(Path(member.name).name.split(".")[0])
                    if key >= len(weights):
                        raise ValueError(
                            f"Cache key {key} outside weight array: {path}"
                        )
                    w = float(weights[key])
                    if w > maximum + 1e-5:
                        raise ValueError(
                            "Cache shard layout differs from expected 256 slots per shard"
                        )
                    if rng.random() * maximum >= w:
                        continue
                    emb = torch.load(
                        io.BytesIO(tar.extractfile(member).read()),
                        map_location="cpu",
                        weights_only=True,
                    )
                    if emb.shape != (512, 4096) or not torch.isfinite(emb).all():
                        raise ValueError(f"Invalid embedding {path}:{key}")
                    yield {"context": emb, "prompt": f"{path}:{key}"}


class ShardedPromptEmbeddings(IterableDataset):
    def __init__(self, directory, seed=42, rank=0):
        self.paths = sorted(Path(directory).glob("prompts_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No prepared embedding shards in {directory}")
        self.seed, self.rank = seed, rank

    def __iter__(self):
        worker = get_worker_info()
        rng = random.Random(
            self.seed + 1000003 * self.rank + (worker.id if worker else 0)
        )
        while True:
            # Select shards uniformly and emit every row; long-run row mass is uniform.
            path = rng.choice(self.paths)
            blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            order = list(range(len(blob["prompts"])))
            rng.shuffle(order)
            for i in order:
                yield {
                    "context": blob["t5_text_embeddings"][i],
                    "prompt": blob["prompts"][i],
                }
