"""Verify saved joint-training states, finite metrics and preview media."""

import argparse
import json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
root = a.directory
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
assert rows and all(r["global_batch"] == 128 for r in rows)
step = rows[-1]["step"]
folder = root / f"step_{step:06d}"
assert (folder / "COMPLETE").is_file()
cfg = json.loads((root / "config.json").read_text())
info = json.loads((folder / "joint_state.json").read_text())
assert info["step"] == step
for r in rows:
    assert torch.isfinite(torch.tensor(r["loss_mean"]))
    expected_loss = (
        sum(r[f"phase/{i}/traj_loss"] for i in range(4)) / 4 * cfg["traj_weight"]
    )
    expected_loss += (
        r["joint/dmd_weight"] * sum(r[f"phase/{i}/dmd_loss"] for i in range(4)) / 4
    )
    assert abs(r["loss_mean"] - expected_loss) < 1e-5, (r["loss_mean"], expected_loss)
    for phase in range(4):
        assert torch.isfinite(torch.tensor(r[f"phase/{phase}/traj_loss"]))
        if info["dmd_enabled"]:
            assert r[f"phase/{phase}/fake_loss"] > 0
            assert r[f"phase/{phase}/direction_rms"] > 0
states = {}
for name, heads in [("model", 128)] + ([("fake", 4)] if info["dmd_enabled"] else []):
    blob = torch.load(
        folder / f"{name}.pt", mmap=True, map_location="cpu", weights_only=True
    )
    assert blob["step"] == step
    projection = blob["student"]["backbone.proj_out.weight"]
    assert projection.shape[0] == heads and torch.isfinite(projection).all()
    counts = sorted({int(s["step"]) for s in blob["optimizer"]["state"].values()})
    expected = step if name == "model" else info["fake_updates"]
    assert counts == [expected], (name, counts, expected)
    if name == "model":
        initial_path = Path(blob["config"]["student_init"])
        if initial_path.is_dir():
            initial_path = initial_path / "model.pt"
        initial = torch.load(
            initial_path, mmap=True, map_location="cpu", weights_only=True
        )
        change = (
            (projection - initial["student"]["backbone.proj_out.weight"])
            .abs()
            .max()
            .item()
        )
        assert change > 0, "Student projection did not change"
        del initial
    else:
        change = None
    states[name] = {
        "max_projection_change_from_init": change,
        "heads": heads,
        "optimizer_steps": counts,
        "bytes": (folder / f"{name}.pt").stat().st_size,
    }
    del blob
acc = EventAccumulator(str(root / "tensorboard")).Reload()
assert "phase/0/traj_loss" in acc.Tags()["scalars"]
assert acc.Scalars("train/loss")[-1].step == step
preview = root / "fixed_prompt" / f"step_{step:06d}"
assert (preview / "COMPLETE").is_file()
assert list(preview.glob("*.mp4"))
report = {
    "steps": step,
    "global_batch": 128,
    "joint_state": info,
    "states": states,
    "tensorboard_verified": True,
    "preview_complete": True,
    "metrics": rows,
}
(root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2))
