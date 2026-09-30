"""Verify decoded evaluation videos and create a filterable local HTML gallery."""

import argparse
import html
import json
from pathlib import Path
import subprocess
import torch

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
root = a.directory
config = json.loads((root / "evaluation_config.json").read_text())
args = config["arguments"]
rows = [
    json.loads(line)
    for path in sorted(root.glob("metrics_rank*.jsonl"))
    for line in path.read_text().splitlines()
]
expected = (
    args["limit"]
    * len(args["seeds"])
    * (len(args["nfe"]) + (not args.get("skip_teacher", False)))
)
assert len(rows) == expected, (len(rows), expected)
assert len({(r["prompt_index"], r["seed"], r["mode"]) for r in rows}) == expected
prompts = torch.load(args["prompts"], map_location="cpu", weights_only=True, mmap=True)[
    "prompts"
]
cards = []
for row in sorted(rows, key=lambda r: (r["prompt_index"], r["seed"], r["mode"])):
    stem = f"prompt{row['prompt_index']:04d}_seed{row['seed']}_{row['mode']}"
    video = root / (stem + ".mp4")
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "json",
                str(video),
            ]
        )
    )["streams"][0]
    assert (probe["width"], probe["height"], int(probe["nb_frames"])) == (
        row["width"],
        row["height"],
        row["frames"],
    ), video
    poster = root / (stem + ".jpg")
    if not poster.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-ss",
                "2.5",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=416:-1",
                "-threads",
                "1",
                str(poster),
            ],
            check=True,
        )
    cards.append(
        f'''<article data-prompt="{row["prompt_index"]}" data-seed="{row["seed"]}" data-mode="{row["mode"]}"><h3>Prompt {row["prompt_index"] + 1} · seed {row["seed"]} · {row["mode"]}</h3><video controls loop preload="none" poster="{poster.name}" src="{video.name}"></video><details><summary>Prompt</summary>{html.escape(prompts[row["prompt_index"]])}</details><p>{row["seconds"]:.2f}s sampling</p></article>'''
    )
selectors = []
for key in ["prompt", "seed", "mode"]:
    values = sorted(
        {str(r["prompt_index"] if key == "prompt" else r[key]) for r in rows}
    )
    selectors.append(
        f'<label>{key} <select id="{key}"><option value="">All</option>'
        + "".join(
            f'<option value="{v}">{int(v) + 1 if key == "prompt" else v}</option>'
            for v in values
        )
        + "</select></label>"
    )
page = """<!doctype html><meta charset="utf-8"><title>PDD evaluation</title><style>body{background:#181818;color:#eee;font:16px sans-serif;margin:24px}main{display:flex;flex-wrap:wrap;gap:20px}article{width:416px}video{width:100%}label{margin-right:20px}select{font-size:16px}details{max-height:180px;overflow:auto}nav{margin:24px 0}</style>"""
page += (
    f"<h1>PDD — {html.escape(Path(args['student']).name)}</h1><p>{expected} videos · 832×480 · 81 frames · seeds {html.escape(str(args['seeds']))}</p><nav>"
    + "".join(selectors)
    + "</nav><main>"
    + "".join(cards)
    + "</main>"
)
page += """<script>document.querySelectorAll('select').forEach(s=>s.onchange=()=>{document.querySelectorAll('article').forEach(a=>{a.hidden=['prompt','seed','mode'].some(k=>document.getElementById(k).value && a.dataset[k]!==document.getElementById(k).value);if(a.hidden)a.querySelector('video').pause()})})</script>"""
(root / "index.html").write_text(page)
(root / "COMPLETE").write_text(json.dumps({"verified_videos": len(rows)}))
print(root / "index.html")
