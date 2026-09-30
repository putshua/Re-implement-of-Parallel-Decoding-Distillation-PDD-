"""Probe full training updates in fresh four-GPU subprocesses, retain only logs.

OOM is the only retryable failure. Other failures abort with their original log.
Test every on-policy time block, including optimizer state allocation, before acceptance.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import shlex
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--candidates", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16])
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--target-global-batch", type=int, default=256)
    p.add_argument("--memory-fraction", type=float, default=0.95)
    p.add_argument("--output", default="outputs/batch_probe")
    a = p.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    if (
        min(a.candidates + [a.gpus, a.target_global_batch]) < 1
        or not 0 < a.memory_fraction <= 1
    ):
        p.error("Positive batch/GPU counts and 0 < memory-fraction <= 1 required")
    root = Path(__file__).resolve().parents[1]
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(a.config).read_text())
    results = []
    best = None
    for batch in sorted(set(a.candidates)):
        cfg = dict(
            base,
            batch_size=batch,
            grad_accum=1,
            steps=base["num_heads"] // base["block_min"],
            save_checkpoints=False,
            output=str(out / f"b{batch}"),
        )
        cfg.pop("resume", None)
        path = out / f"b{batch}.json"
        path.write_text(json.dumps(cfg, indent=2))
        env = dict(
            os.environ,
            CONFIG=str(path),
            NPROC_PER_NODE=str(a.gpus),
            NNODES="1",
            OMP_NUM_THREADS="4",
            PYTHON_BIN=sys.executable,
        )
        for key in (
            "BATCH_SIZE",
            "GRAD_ACCUM",
            "STEPS",
            "OUTPUT",
            "RESUME",
            "NODE_RANK",
        ):
            env.pop(key, None)
        logpath = out / f"b{batch}.log"
        with logpath.open("w") as log:
            run = subprocess.run(
                ["bash", str(root / "scripts/train_pdd.sh")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        text = logpath.read_text()
        if run.returncode:
            if "out of memory" not in text.lower():
                raise RuntimeError(f"Non-OOM probe failure: {logpath}")
            results.append({"batch": batch, "status": "oom"})
            break
        rows = [
            json.loads(line)
            for line in (out / f"b{batch}" / "metrics.jsonl").read_text().splitlines()
        ]
        peak = max(x["peak_reserved_gib"] for x in rows)
        steady = rows[1:] or rows
        # Query actual smallest device capacity, not a hardcoded GPU model.
        import torch

        capacity = min(
            torch.cuda.get_device_properties(i).total_memory / 2**30
            for i in range(a.gpus)
        )
        accepted = peak <= capacity * a.memory_fraction
        results.append(
            {
                "batch": batch,
                "status": "pass" if accepted else "insufficient_headroom",
                "peak_reserved_gib": peak,
                "steady_samples_per_second": sum(x["global_batch"] for x in steady)
                / sum(x["seconds"] for x in steady),
            }
        )
        if accepted:
            best = batch
        else:
            break
    confirmations = []
    passing = [r["batch"] for r in results if r["status"] == "pass"]
    best = None
    for candidate in reversed(passing):
        accumulation = math.ceil(a.target_global_batch / (candidate * a.gpus))
        if accumulation == 1:
            best = candidate
            break
        label = f"accum_b{candidate}_a{accumulation}"
        confirmation = dict(
            base,
            batch_size=candidate,
            grad_accum=accumulation,
            steps=2,
            save_checkpoints=False,
            output=str(out / label),
        )
        confirmation.pop("resume", None)
        config_path = out / f"{label}.json"
        config_path.write_text(json.dumps(confirmation, indent=2))
        env["CONFIG"] = str(config_path)
        logfile = out / f"{label}.log"
        with logfile.open("w") as stream:
            run = subprocess.run(
                ["bash", str(root / "scripts/train_pdd.sh")],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if run.returncode:
            if "out of memory" not in logfile.read_text().lower():
                raise RuntimeError(f"Non-OOM accumulation failure: {logfile}")
            confirmations.append(
                {"batch": candidate, "grad_accum": accumulation, "status": "oom"}
            )
            continue
        rows = [
            json.loads(line)
            for line in (out / label / "metrics.jsonl").read_text().splitlines()
        ]
        peak = max(r["peak_reserved_gib"] for r in rows)
        accepted = peak <= capacity * a.memory_fraction
        confirmations.append(
            {
                "batch": candidate,
                "grad_accum": accumulation,
                "status": "pass" if accepted else "insufficient_headroom",
                "peak_reserved_gib": peak,
                "steps": len(rows),
            }
        )
        if accepted:
            best = candidate
            break
    report = {
        "accumulation_confirmations": confirmations,
        "results": results,
        "recommended_batch": best,
        "memory_fraction": a.memory_fraction,
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
    }
    if best is not None:
        accum = math.ceil(a.target_global_batch / (best * a.gpus))
        report.update(grad_accum=accum, effective_global_batch=best * a.gpus * accum)
        (out / "recommended.env").write_text(
            f"export CONFIG={shlex.quote(str(Path(a.config).resolve()))}\nexport NPROC_PER_NODE={a.gpus}\nexport BATCH_SIZE={best}\nexport GRAD_ACCUM={accum}\n"
        )
    if best is not None and os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        with (out / "recommended.env").open("a") as stream:
            stream.write(
                "export PYTORCH_CUDA_ALLOC_CONF="
                + shlex.quote(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
                + "\n"
            )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
