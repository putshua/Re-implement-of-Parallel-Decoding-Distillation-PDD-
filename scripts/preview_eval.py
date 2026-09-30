"""Create a qualitative comparison figure from actual decoded evaluation videos."""

import argparse
from pathlib import Path
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("directory")
p.add_argument("--prompt-index", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--frames", nargs="+", type=int, default=[8, 40, 72])
p.add_argument("--methods", nargs="+", default=["teacher", "pdd_2", "pdd_4", "pdd_8"])
a = p.parse_args()
root = Path(a.directory)
fig, axes = plt.subplots(
    len(a.methods), len(a.frames), figsize=(12, 2.5 * len(a.methods)), squeeze=False
)
for row, method in enumerate(a.methods):
    path = root / f"prompt{a.prompt_index:04d}_seed{a.seed}_{method}.mp4"
    for col, frame in enumerate(a.frames):
        ax = axes[row, col]
        ax.imshow(iio.imread(path, index=frame))
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(method)
        if row == 0:
            ax.set_title(f"Frame {frame}")
fig.suptitle(f"Prompt {a.prompt_index}, seed {a.seed} | Actual decoded outputs")
fig.tight_layout()
output = root / f"preview_prompt{a.prompt_index:04d}_seed{a.seed}.png"
fig.savefig(output, dpi=140)
plt.close(fig)
print(output)
