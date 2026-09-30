"""Build paired teacher/student players and seed contact sheets after eval_gallery."""

import argparse
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from gallery_loss import build as build_loss, loss_panel

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
root = a.directory
cfg = json.loads((root / "evaluation_config.json").read_text())
summary = json.loads((root / "summary.json").read_text())["results"]
rows = [
    json.loads(line)
    for path in root.glob("metrics_rank*.jsonl")
    for line in path.read_text().splitlines()
]
lookup = {(r["prompt_index"], r["seed"], r["mode"]): r for r in rows}
cache = torch.load(
    cfg["arguments"]["prompts"], map_location="cpu", mmap=True, weights_only=True
)
prompts = cache["prompts"]
indices = sorted({r["prompt_index"] for r in rows})
seeds = sorted({r["seed"] for r in rows})
sections = []
for index in indices:
    sheet = Image.new("RGB", (832, 272 * len(seeds)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for row, seed in enumerate(seeds):
        cards = []
        for col, mode in enumerate(("teacher", "pdd_4")):
            metric = lookup[index, seed, mode]
            stem = f"prompt{index:04d}_seed{seed}_{mode}"
            title = (
                "Teacher / UniPC 50 / CFG6 / shift8"
                if mode == "teacher"
                else "PDD / 4 steps / shift6"
            )
            cards.append(
                f'<div><h3>{title}</h3><video controls loop preload="none" poster="{stem}.jpg" src="{stem}.mp4"></video><p>Sampling {metric["seconds"]:.2f}s</p></div>'
            )
            with Image.open(root / f"{stem}.jpg") as frame:
                sheet.paste(
                    frame.convert("RGB").resize((416, 240)), (col * 416, row * 272 + 32)
                )
            draw.text(
                (col * 416 + 8, row * 272 + 8),
                f"Prompt {index + 1} / seed {seed} / {mode}",
                fill="white",
            )
        sections.append(
            f'<section data-prompt="{index}" data-seed="{seed}"><h2>Prompt {index + 1} · seed {seed}</h2><details><summary>Prompt text</summary>{html.escape(prompts[index])}</details><button onclick="playPair(this)">Play both from start</button><div class="pair">'
            + "".join(cards)
            + "</div></section>"
        )
    sheet.save(root / f"comparison_prompt{index:04d}_5seeds.jpg", quality=92)
teacher, student = summary["teacher"], summary["pdd_4"]
page = """<!doctype html><meta charset="utf-8"><title>Official Wan teacher vs PDD4</title><style>body{font:16px sans-serif;background:#181818;color:#eee;margin:24px}section{border-top:1px solid #555;padding:16px 0;max-width:1200px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%}select,button{font-size:16px;margin:12px 12px 12px 0}details{max-width:1000px}a{color:#8cd3ff}</style>"""
page += f"<h1>Step 100 PDD4 vs official Wan1.3B teacher</h1><p>6 RCM prompts × 5 seeds (42–46) · 832×480 · 81 frames · 16 FPS.</p><p>Mean sampling: teacher {teacher['mean_seconds']:.2f}s; PDD4 {student['mean_seconds']:.2f}s. Excludes loading, warmup, VAE decode and file writing. Native official teacher and Diffusers PDD backends differ; this is measured latency, not a quality-matched speedup.</p><p>Same original prompts and initial noise per pair. Teacher uses UniPC50, shift8, CFG6, official negative prompt, all blocks. PDD uses its trained shift6 schedule. No prompt extension.</p>"
page += '<p>Timing caveat: the first attempt encountered GPU0 contention and a VAE OOM. Completed videos were retained; remaining pairs ran on GPUs1–3. See <a href="latency_comparison.json">timing statistics with initial GPU0 pairs excluded</a>.</p>'
page += loss_panel(build_loss(root))
page += (
    '<label>Prompt <select id="prompt"><option value="">All</option>'
    + "".join(f'<option value="{i}">{i + 1}</option>' for i in indices)
    + '</select></label><label>Seed <select id="seed"><option value="">All</option>'
    + "".join(f"<option>{s}</option>" for s in seeds)
    + "</select></label>"
)
page += (
    "".join(sections)
    + """<script>function playPair(b){b.closest('section').querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})}document.querySelectorAll('select').forEach(s=>s.onchange=()=>document.querySelectorAll('section').forEach(e=>{e.hidden=['prompt','seed'].some(k=>document.getElementById(k).value&&e.dataset[k]!==document.getElementById(k).value);if(e.hidden)e.querySelectorAll('video').forEach(v=>v.pause())}))</script>"""
)
(root / "comparison.html").write_text(page)
print(root / "comparison.html")
