"""Publish completed pairs while a long teacher evaluation is still running."""

import html
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = {}
for path in root.glob("metrics_rank*.jsonl"):
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        records[r["prompt_index"], r["seed"], r["mode"]] = r
pairs = sorted(
    {
        (i, s)
        for i, s, m in records
        if (i, s, "teacher") in records and (i, s, "pdd_4") in records
    }
)
page = '<!doctype html><meta charset="utf-8"><title>PDD comparison — running</title><style>body{font:16px sans-serif;background:#181818;color:#eee;margin:24px}.pair{display:flex;gap:16px}video{width:416px}section{margin-bottom:30px}button{padding:8px}</style>'
page += f"<h1>Step100 PDD4 / official teacher — {len(pairs)}/30 pairs completed</h1><p>Evaluation in progress. Refresh for newly published pairs.</p>"
for index, seed in pairs:
    page += f'<section><h2>Prompt {index + 1} / seed {seed}</h2><button onclick="this.parentElement.querySelectorAll(\'video\').forEach(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}})">Play both</button><div class="pair">'
    for mode in ["teacher", "pdd_4"]:
        stem = f"prompt{index:04d}_seed{seed}_{mode}"
        title = (
            "Teacher UniPC50 / shift8 / CFG6"
            if mode == "teacher"
            else "PDD 4 steps / shift6"
        )
        page += f'<div><h3>{html.escape(title)}</h3><video controls loop preload="none" src="{stem}.mp4"></video></div>'
    page += "</div></section>"
temp = root / "partial_comparison.html.tmp"
temp.write_text(page)
temp.replace(root / "partial_comparison.html")
print(len(pairs))
