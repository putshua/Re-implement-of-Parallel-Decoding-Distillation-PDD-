import argparse
import collections
import itertools
import json
from pathlib import Path
import statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("directory")
a = p.parse_args()
root = Path(a.directory)
rows = [
    json.loads(line)
    for path in root.glob("metrics_rank*.jsonl")
    for line in path.read_text().splitlines()
]
if not rows:
    raise SystemExit("No evaluation results")
groups = collections.defaultdict(list)
for row in rows:
    groups[row["mode"]].append(row)
teachers = {
    (r["prompt_index"], r["seed"]): r["seconds"] for r in rows if r["mode"] == "teacher"
}
summary = {}
for mode, items in groups.items():
    pairs = []
    for prompt in sorted(set(x["prompt_index"] for x in items)):
        paths = [
            root / f"prompt{prompt:04d}_seed{x['seed']}_{mode}.pt"
            for x in items
            if x["prompt_index"] == prompt
        ]
        for x, y in itertools.combinations(paths, 2):
            vx = torch.load(x, map_location="cpu", weights_only=True)["latent"].float()
            vy = torch.load(y, map_location="cpu", weights_only=True)["latent"].float()
            pairs.append((vx - vy).square().mean().sqrt().item())
    summary[mode] = {
        "samples": len(items),
        "mean_seconds": statistics.mean(x["seconds"] for x in items),
        "median_seconds": statistics.median(x["seconds"] for x in items),
        "min_seconds": min(x["seconds"] for x in items),
        "max_seconds": max(x["seconds"] for x in items),
        "mean_speedup_vs_teacher": statistics.mean(
            teachers[(x["prompt_index"], x["seed"])] / x["seconds"] for x in items
        )
        if teachers
        else None,
        "mean_latent_mse_to_teacher": statistics.mean(
            x["latent_mse_to_teacher"] for x in items
        )
        if all(x["latent_mse_to_teacher"] is not None for x in items)
        else None,
        "mean_pairwise_latent_rms": statistics.mean(pairs) if pairs else None,
        "peak_allocated_gib": max(x["peak_allocated_gib"] for x in items),
    }
report = {
    "results": summary,
    "note": "Latent distances are diagnostics, not VBench or perceptual quality/diversity scores.",
}
(root / "summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
