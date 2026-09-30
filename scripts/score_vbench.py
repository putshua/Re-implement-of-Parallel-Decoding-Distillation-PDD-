"""Score generated MP4s per method using VBench's custom-input interface.

This reports dimensions on these custom prompts, not the official aggregate.
Uses local metric weights; configure VBENCH_CACHE_DIR before launching.
"""

import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("directory")
p.add_argument(
    "--dimensions",
    nargs="+",
    default=[
        "temporal_flickering",
        "motion_smoothness",
        "subject_consistency",
        "background_consistency",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
    ],
)
p.add_argument("--methods", nargs="+", default=["teacher", "pdd_2", "pdd_4", "pdd_8"])
a = p.parse_args()
root = Path(a.directory).resolve()
from vbench import VBench  # noqa: E402

for method in a.methods:
    videos = sorted(root.glob(f"*_{method}.mp4"))
    if not videos:
        raise FileNotFoundError(f"No decoded {method} videos in {root}")
    stage = root / "vbench" / method
    stage.mkdir(parents=True, exist_ok=True)
    prompts = {}
    for path in videos:
        target = stage / path.name
        if not target.exists():
            target.symlink_to(path)
        prompts[path.name] = torch.load(
            path.with_suffix(".pt"), map_location="cpu", weights_only=True
        )["prompt"]
    info = stage / "empty_standard_info.json"
    info.write_text("[]")
    evaluator = VBench(torch.device("cuda"), str(info), str(stage / "results"))
    evaluator.evaluate(
        videos_path=str(stage),
        name=method,
        prompt_list=prompts,
        dimension_list=a.dimensions,
        mode="custom_input",
        local=True,
    )
