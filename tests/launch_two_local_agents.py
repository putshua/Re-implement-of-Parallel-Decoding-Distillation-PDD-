"""Emulate two allocated nodes on four LOCAL GPUs; not an inter-node network test."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, default=29681)
    a = p.parse_args()
    processes, logs = [], []
    try:
        for agent, visible in enumerate(["0,1", "2,3"]):
            env = dict(
                os.environ,
                CONFIG=str(Path(a.config).resolve()),
                NNODES="2",
                NPROC_PER_NODE="2",
                FSDP_SHARD_SIZE="2",
                RDZV_ENDPOINT=f"127.0.0.1:{a.port}",
                RDZV_ID=a.label,
                CUDA_VISIBLE_DEVICES=visible,
                NCCL_DEBUG="WARN",
            )
            for key in [
                "NODE_RANK",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "OUTPUT",
                "BATCH_SIZE",
                "GRAD_ACCUM",
                "STEPS",
                "RESUME",
                "DRY_RUN",
            ]:
                env.pop(key, None)
            log = open(f"reports/{a.label}_agent{agent}.log", "w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    ["bash", "scripts/train_pdd.sh"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 1800
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("An agent failed; see per-agent logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Local two-agent test exceeded 30 minutes")
            time.sleep(2)
        assert all(p.returncode == 0 for p in processes)
        print("PASS: both c10d agents completed without NODE_RANK")
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
