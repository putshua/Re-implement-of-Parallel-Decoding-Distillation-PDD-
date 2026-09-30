"""Resolve restart state without loading model tensors."""

import json
import shutil
import time
from pathlib import Path


def resolve_resume(
    output, requested, world, extra_files=(), *, restart_without_checkpoint=False
):
    if requested not in ("auto", "none", "", None):
        return str(Path(requested).resolve())
    if requested != "auto":
        return None
    root = Path(output)
    candidates = []
    for folder in root.glob("step_*"):
        suffix = folder.name.removeprefix("step_")
        if not suffix.isdigit():
            continue
        required = [folder / "COMPLETE", folder / "model.pt"] + [
            folder / f"rank_{rank}.pt" for rank in range(world)
        ]
        required += [folder / name for name in extra_files]
        if all(p.is_file() for p in required) and all(
            p.stat().st_size > 0 for p in required[1:]
        ):
            candidates.append((int(suffix), folder))
    if candidates:
        return str(max(candidates)[1].resolve())
    if (root / "metrics.jsonl").exists():
        # Never turn an incompatible or damaged completed checkpoint into a fresh run.
        completed = list(root.glob("step_*/COMPLETE"))
        if restart_without_checkpoint and not completed:
            archive_uncheckpointed_run(root)
            return None
        raise FileNotFoundError(
            f"{root} has training metrics but no complete checkpoint with {world} rank files. Choose a new OUTPUT_DIR to start over; existing results were preserved."
        )
    return None


def checkpoint_due(step, every, final_step, save_first=True):
    if every < 1:
        raise ValueError("save_every must be positive")
    return (save_first and step == 1) or step % every == 0 or step == final_step


def archive_uncheckpointed_run(root):
    """Rank0 only: preserve failed-run artifacts before an explicit automatic reset.

    Logs stay in place. Move metrics LAST, so interruption of archiving remains
    detectable on the next launch. No completed checkpoint may be discarded.
    """
    root = Path(root)
    if list(root.glob("step_*/COMPLETE")):
        raise RuntimeError("Refusing to reset a run with completed checkpoints")
    latest = 0
    metrics = root / "metrics.jsonl"
    for line in metrics.read_text().splitlines():
        try:
            latest = max(latest, int(json.loads(line)["step"]))
        except (ValueError, KeyError, TypeError):
            pass
    archive = root / "restart_archives" / str(time.time_ns())
    archive.mkdir(parents=True)
    paths = [root / name for name in ("config.json", "tensorboard", "fixed_prompt")]
    paths += [p for p in root.glob("step_*") if p.is_dir()]
    paths += [metrics]
    for path in paths:
        if path.exists():
            path.replace(archive / path.name)
    event = {
        "event": "restart_without_checkpoint",
        "archive": str(archive),
        "last_logged_step": latest,
        "restart_step": 0,
        "reason": "No completed checkpoint; unsaved updates cannot be recovered",
    }
    (archive / "restart.json").write_text(json.dumps(event, indent=2) + "\n")
    print(json.dumps(event), flush=True)
    return archive


def reconcile_metrics(output, step, filename="metrics.jsonl"):
    """Archive pre-restart metrics before removing updates beyond the restored step."""
    root = Path(output)
    path = root / filename
    if not path.exists():
        return None
    kept = []
    needs_rewrite = False
    for line in path.read_text().splitlines(keepends=True):
        try:
            keep = json.loads(line)["step"] <= step
        except (ValueError, KeyError):
            keep = False
        if keep:
            kept.append(line if line.endswith("\n") else line + "\n")
        else:
            needs_rewrite = True
    if not needs_rewrite:
        return None
    stem = Path(filename).stem
    backup = root / f"{stem}_before_resume_{step:06d}_{time.time_ns()}.jsonl"
    shutil.copy2(path, backup)
    temporary = root / f"{filename}.tmp"
    temporary.write_text("".join(kept))
    temporary.replace(path)
    return str(backup)


def prune_checkpoints(output, current_step, keep_every, world, extra_files=()):
    """Rank0 only, after collective save: keep latest plus permanent milestones.

    Never prune until the replacement has all files. Ignore symlinks, future
    steps and incomplete saves; KEEP also preserves milestones across policy changes.
    """
    if keep_every < 1:
        raise ValueError("checkpoint_keep_every must be positive")
    root = Path(output)
    current = root / f"step_{current_step:06d}"
    required = [current / "model.pt"] + [current / f"rank_{r}.pt" for r in range(world)]
    required += [current / name for name in extra_files]
    if (
        current.is_symlink()
        or not (current / "COMPLETE").is_file()
        or not all(p.is_file() and p.stat().st_size > 0 for p in required)
    ):
        raise ValueError("Refusing to prune before replacement checkpoint is complete")
    if current_step % keep_every == 0:
        (current / "KEEP").touch()
    removed = []
    for folder in sorted(root.glob("step_*")):
        suffix = folder.name.removeprefix("step_")
        if folder.is_symlink() or not folder.is_dir() or not suffix.isdigit():
            continue
        step = int(suffix)
        if (
            step >= current_step
            or step % keep_every == 0
            or (folder / "KEEP").exists()
            or not (folder / "COMPLETE").is_file()
        ):
            continue
        # Atomic removal from discovery before recursive cleanup.
        garbage = root / f".obsolete_{folder.name}_{time.time_ns()}"
        folder.replace(garbage)
        shutil.rmtree(garbage)
        removed.append(folder.name)
    return removed
