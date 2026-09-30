import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_wan1p3b_480p_450k.sh"


class TestLauncher(unittest.TestCase):
    def test_shared_python_default(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--grad-accum 4", result.stdout)
        self.assertIn("nodes=4 x GPUs=8", result.stdout)
        self.assertIn(
            "/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python -m torch.distributed.run",
            result.stdout,
        )

    def launch(self, **changes):
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "PYTHON_BIN",
                "NNODES",
                "NPROC_PER_NODE",
                "FSDP_SHARD_SIZE",
                "BATCH_SIZE",
                "GRAD_ACCUM",
                "MASTER_ADDR",
                "RDZV_ENDPOINT",
                "RDZV_ID",
                "NODE_RANK",
                "OUTPUT",
                "OUTPUT_DIR",
                "RESUME",
                "RESUME_PATH",
                "MAX_ITER",
                "STEPS",
                "SAVE_EVERY",
            }
        }
        env.update(DRY_RUN="1", **changes)
        return subprocess.run(
            ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
        )

    def test_same_multinode_command_no_node_rank(self):
        run = self.launch(
            NNODES="2",
            NPROC_PER_NODE="8",
            FSDP_SHARD_SIZE="8",
            MASTER_ADDR="10.0.0.1",
            NODE_RANK="unused",
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        for text in [
            "--rdzv-backend c10d",
            "--rdzv-endpoint 10.0.0.1:29571",
            "--grad-accum 8",
            "distributed_strategy=fsdp",
            "fsdp_shard_size=8",
            "global_batch=128",
        ]:
            self.assertIn(text, run.stdout)
        self.assertNotIn("--node_rank", run.stdout)
        self.assertNotIn("--node-rank", run.stdout)

    def test_endpoint_aliases_and_subset_isolation(self):
        run = self.launch(
            RDZV_ENDPOINT="host-a:30201",
            RDZV_ID="test-run",
            OUTPUT_DIR="/tmp/pdd-run",
            MAX_ITER="10",
            SAVE_EVERY="5",
            BATCH_SIZE="2",
            EMBEDDING_DIR="/wrong/subset",
            PROMPT_EMBEDDINGS="/wrong.pt",
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        for text in [
            "host-a:30201",
            "--rdzv-id test-run",
            "--output /tmp/pdd-run",
            "--steps 10",
            "save_every=5",
            "--grad-accum 2",
        ]:
            self.assertIn(text, run.stdout)
        self.assertNotIn("/wrong", run.stdout)

    def test_matched_control(self):
        run = self.launch(EXPERIMENT="pdd_matched")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("pdd_matched.json", run.stdout)
        self.assertIn("pdd_matched_gbs128", run.stdout)

    def test_auto_resume_and_explicit_override(self):
        self.assertIn("--resume auto", self.launch().stdout)
        self.assertIn(
            "--resume /old/checkpoint",
            self.launch(RESUME_PATH="/old/checkpoint").stdout,
        )
        self.assertIn("--resume none", self.launch(RESUME_PATH="none").stdout)

    def test_local_and_invalid_mesh(self):
        run = self.launch(NNODES="1", NPROC_PER_NODE="4", FSDP_SHARD_SIZE="8")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--grad-accum 32", run.stdout)
        self.assertIn("phased_dmd_pdd.json", run.stdout)
        self.assertIn("--standalone", run.stdout)
        self.assertIn("effective=4", run.stdout)
        self.assertNotEqual(self.launch(FSDP_SHARD_SIZE="3").returncode, 0)
        self.assertNotEqual(self.launch(BATCH_SIZE="0").returncode, 0)
        # Real launches reject a missing rendezvous address before any data/GPU work.
        env = dict(os.environ, NNODES="2", DRY_RUN="0")
        env.pop("MASTER_ADDR", None)
        env.pop("RDZV_ENDPOINT", None)
        run = subprocess.run(
            ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("MASTER_ADDR or RDZV_ENDPOINT", run.stderr)


class Test14BLauncher(unittest.TestCase):
    def test_defaults_and_overrides(self):
        for accum in ("64", "128"):
            env = dict(
                os.environ,
                DRY_RUN="1",
                NNODES="2",
                NPROC_PER_NODE="8",
                FSDP_SHARD_SIZE="8",
                BATCH_SIZE="1",
                GRAD_ACCUM=accum,
                MASTER_ADDR="node0",
                OUTPUT_DIR="/tmp/wan14b-launch-test",
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/train_wan21_14b_480p_450k.sh")],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for expected in (
                "wan21_14b_480p_450k_16gpu.json",
                "Wan2.1-T2V-14B",
                "--batch-size 1",
                f"--grad-accum {accum}",
                f"global_batch={16 * int(accum)}",
                "--resume auto",
                "--rdzv-id pdd-wan21-14b-480p-450k",
            ):
                self.assertIn(expected, result.stdout)


class Test14BJointLauncher(unittest.TestCase):
    def run_script(self, **overrides):
        env = {k: v for k, v in os.environ.items() if k not in {
            'NNODES', 'NPROC_PER_NODE', 'FSDP_SHARD_SIZE', 'GRAD_ACCUM',
            'BATCH_SIZE', 'OUTPUT_DIR', 'OUTPUT', 'PDD_TRAIN_CONFIG', 'EXPERIMENT',
        }}
        env.update(DRY_RUN='1', **overrides)
        return subprocess.run(['bash', str(ROOT / 'scripts/train_wan21_14b_480p_phased_dmd_pdd.sh')],
                              env=env, text=True, capture_output=True)

    def test_full_shard_and_batch(self):
        run = self.run_script()
        self.assertEqual(run.returncode, 0, run.stderr)
        for text in ['nodes=2 x GPUs=8', 'shard=16 (effective=16)',
                     'global_batch=128', '--grad-accum 8',
                     'wan21_14b_480p_450k_phased_dmd_pdd.json', 'Wan2.1-T2V-14B']:
            self.assertIn(text, run.stdout)
        larger = self.run_script(GRAD_ACCUM='16')
        self.assertEqual(larger.returncode, 0, larger.stderr)
        self.assertIn('global_batch=256', larger.stdout)

    def test_no_silent_shard_reduction(self):
        run = self.run_script(NNODES='1', NPROC_PER_NODE='4')
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('exceeds total GPUs', run.stderr)
        local = self.run_script(NNODES='1', NPROC_PER_NODE='3', FSDP_SHARD_SIZE='3', GRAD_ACCUM='4')
        self.assertEqual(local.returncode, 0, local.stderr)
        self.assertIn('global_batch=12', local.stdout)
