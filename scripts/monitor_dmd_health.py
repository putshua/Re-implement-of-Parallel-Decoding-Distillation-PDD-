"""Persist a small health report for a long-running DMD control experiment."""

import argparse
import json
import math
import time
from pathlib import Path


def inspect(output, grad_limit, loss_limit, stall_seconds, updates_file="metrics.jsonl"):
    metrics = output / updates_file
    rows = []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A writer may be appending the final line.
    bad = []
    for row in rows:
        step = row.get("step")
        keys = ["loss_mean", "grad_norm"]
        keys += [key for key in ("joint/fake_loss", "joint/fake_grad_norm") if key in row]
        for key in keys:
            value = row.get(key)
            if value is None or not math.isfinite(value):
                bad.append(f"step {step}: {key} is missing/nonfinite")
            elif key.endswith("grad_norm") and value > grad_limit:
                bad.append(f"step {step}: {key}={value:.5g} > {grad_limit}")
            elif key.endswith("loss") or key == "loss_mean":
                if abs(value) > loss_limit:
                    bad.append(f"step {step}: {key}={value:.5g} > {loss_limit}")
    age = time.time() - metrics.stat().st_mtime if metrics.exists() else None
    if age is not None and age > stall_seconds:
        bad.append(f"metrics have not advanced for {age:.0f}s")
    last = rows[-1] if rows else {}
    return {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "alert" if bad else "ok" if rows else "starting",
        "last_step": last.get("step", 0),
        "last_update_type": last.get("update_type"),
        "last_rollout_nfe": last.get("rollout_nfe"),
        "last_loss": last.get("loss_mean"),
        "last_grad_norm": last.get("grad_norm"),
        "last_fake_loss": last.get("joint/fake_loss"),
        "last_fake_grad_norm": last.get("joint/fake_grad_norm"),
        "metrics_age_seconds": age,
        "alerts": bad,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--grad-limit", type=float, default=100)
    parser.add_argument("--loss-limit", type=float, default=10)
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--updates-file", default="metrics.jsonl")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "health.json"
    while True:
        health = inspect(
            args.output, args.grad_limit, args.loss_limit, args.stall_seconds,
            args.updates_file,
        )
        temp = report.with_suffix(".json.tmp")
        temp.write_text(json.dumps(health, indent=2) + "\n")
        temp.replace(report)
        print(json.dumps(health), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
