"""Validate a completed live-monitor smoke run's media and TensorBoard events."""

import argparse
import io
import json
from pathlib import Path
import subprocess

from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
a = p.parse_args()
rows = [
    json.loads(line) for line in (a.output / "metrics.jsonl").read_text().splitlines()
]
events = EventAccumulator(
    str(a.output / "tensorboard"), size_guidance={"images": 0, "scalars": 0}
).Reload()
losses = events.Scalars("train/loss")
assert [(r["step"], r["loss_mean"]) for r in rows] == [
    (e.step, e.value) for e in losses
]
folders = sorted((a.output / "fixed_prompt").glob("step_*"))
video_count = 0
for folder in folders:
    assert (folder / "COMPLETE").is_file(), folder
    manifest = json.loads((folder / "manifest.json").read_text())
    for sample in manifest["samples"]:
        stem = folder / sample["stem"]
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,r_frame_rate",
                    "-of",
                    "json",
                    str(stem.with_suffix(".mp4")),
                ]
            )
        )["streams"][0]
        assert (probe["width"], probe["height"], int(probe["nb_frames"])) == (
            832,
            480,
            81,
        ), probe
        assert probe["r_frame_rate"] == "16/1", probe
        tag = (
            f"fixed_prompt/prompt{sample['prompt_index']:04d}/pdd_{sample['nfe']}/video"
        )
        event = next(e for e in events.Images(tag) if e.step == manifest["step"])
        with Image.open(io.BytesIO(event.encoded_image_string)) as gif:
            assert gif.size == (208, 120)
            assert gif.n_frames > 1
        video_count += 1
print(
    json.dumps(
        {
            "steps": [r["step"] for r in rows],
            "preview_steps": [int(f.name.split("_")[1]) for f in folders],
            "mp4_verified": video_count,
            "tensorboard_loss_matches_json": True,
            "animated_tensorboard_images_verified": True,
        },
        indent=2,
    )
)
