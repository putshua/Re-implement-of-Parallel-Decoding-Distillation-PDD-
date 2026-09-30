"""Report all timing pairs plus a separate view excluding observed GPU0 contention."""

import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
root = a.directory
cfg = json.loads((root / "evaluation_config.json").read_text())
rows = [
    json.loads(line)
    for path in root.glob("metrics_rank*.jsonl")
    for line in path.read_text().splitlines()
]
seeds = cfg["arguments"]["seeds"]
lookup = {(r["prompt_index"], r["seed"], r["mode"]): r for r in rows}
pairs = []
for index, seed in sorted({(r["prompt_index"], r["seed"]) for r in rows}):
    t, s = lookup[index, seed, "teacher"], lookup[index, seed, "pdd_4"]
    # Initial attempt used physical GPUs0-3. Retry explicitly used GPUs1-3.
    excluded = any(
        r.get("cuda_visible_devices") != "1,2,3"
        and (index * len(seeds) + seeds.index(seed)) % cfg["world_size"] == 0
        for r in (t, s)
    )
    pairs.append(
        dict(
            prompt_index=index,
            seed=seed,
            teacher_seconds=t["seconds"],
            pdd4_seconds=s["seconds"],
            ratio=t["seconds"] / s["seconds"],
            initial_gpu0_pair=excluded,
        )
    )


def summarize(items):
    return dict(
        pairs=len(items),
        teacher_mean_seconds=statistics.mean(x["teacher_seconds"] for x in items),
        pdd4_mean_seconds=statistics.mean(x["pdd4_seconds"] for x in items),
        teacher_median_seconds=statistics.median(x["teacher_seconds"] for x in items),
        pdd4_median_seconds=statistics.median(x["pdd4_seconds"] for x in items),
        mean_paired_latency_ratio=statistics.mean(x["ratio"] for x in items),
    )


report = dict(
    all_pairs=summarize(pairs),
    excluding_initial_gpu0_pairs=summarize(
        [x for x in pairs if not x["initial_gpu0_pair"]]
    ),
    excluded_pairs=[
        {"prompt_index": x["prompt_index"], "seed": x["seed"]}
        for x in pairs
        if x["initial_gpu0_pair"]
    ],
    pairs=pairs,
    note="GPU0 had a competing process and the first attempt ended in VAE OOM. Separate timing summary excludes initial GPU0 pairs. Other GPUs were checked at discrete times, not continuously. Backend and sampler settings differ; these ratios are not quality-matched speedups.",
)
(root / "latency_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
