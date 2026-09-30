"""Encode exact 450k-manifest captions when original SFT cache mounts are absent.

Output one atomic .pt per shard, never concatenate the 450k token tensors in RAM.
Run with torchrun for rank-sharded preparation; existing finished shards are checked.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AnyFlow"))
import torch
from pdd.native import wan_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", default="../Wan2.2/checkpoints/Wan2.1-T2V-1.3B")
    p.add_argument("--wan-root", default="../Wan2.2")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=128)
    a = p.parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(a.manifest) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("caption"):
                raise ValueError("Manifest row missing caption")
            rows.append(row)
            if a.limit and len(rows) >= a.limit:
                break
    ckpt = Path(a.checkpoint)
    model = wan_module(a.wan_root, "t5").T5EncoderModel(
        512,
        device=device,
        checkpoint_path=str(ckpt / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(ckpt / "google/umt5-xxl"),
    )
    for start in range(rank * a.shard_size, len(rows), world * a.shard_size):
        subset = rows[start : start + a.shard_size]
        prompts = [r["caption"] for r in subset]
        digest = hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode()
        ).hexdigest()
        path = out / f"prompts_{start:09d}.pt"
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if old["caption_sha256"] != digest:
                raise ValueError(f"Stale embeddings: {path}")
            continue
        values = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(prompts), a.batch_size):
                for context in model(prompts[i : i + a.batch_size], device):
                    values.append(
                        torch.nn.functional.pad(
                            context, (0, 0, 0, 512 - len(context))
                        ).cpu()
                    )
        torch.save(
            {
                "prompts": prompts,
                "t5_text_embeddings": torch.stack(values),
                "manifest_ids": [r["id"] for r in subset],
                "caption_sha256": digest,
                "manifest": str(Path(a.manifest).resolve()),
                "row_start": start,
            },
            str(path) + ".tmp",
        )
        Path(str(path) + ".tmp").replace(path)
        print(
            json.dumps({"rank": rank, "path": str(path), "rows": len(prompts)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
