import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestPortableLaunchers(unittest.TestCase):
    def launch(self, script, **settings):
        env = os.environ.copy()
        for key in ("NNODES", "RDZV_ENDPOINT", "RDZV_ID", "PYTHON_BIN", "MASTER_ADDR"):
            env.pop(key, None)
        env.update(CONFIG=str(ROOT / "configs/pdd_wan1p3b.example.json"),
                   DRY_RUN="1", NPROC_PER_NODE="4", PYTHON_BIN=sys.executable)
        env.update(settings)
        return subprocess.run(["bash", str(ROOT / "scripts" / script)],
                              env=env, text=True, capture_output=True)

    def test_single_node_uses_selected_environment_and_cli(self):
        result = self.launch("train_single_node.sh", BATCH_SIZE="2", GRAD_ACCUM="8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(sys.executable, result.stdout)
        self.assertIn("--nnodes 1", result.stdout)
        self.assertIn("--standalone", result.stdout)
        self.assertIn("--batch-size 2 --grad-accum 8", result.stdout)

    def test_multi_node_requires_shared_rendezvous(self):
        result = self.launch("train_multi_node.sh", NNODES="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RDZV_ENDPOINT", result.stderr)
        result = self.launch("train_multi_node.sh", NNODES="2", NPROC_PER_NODE="8",
                             RDZV_ENDPOINT="192.0.2.1:29571", RDZV_ID="portable-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--nnodes 2 --nproc_per_node 8", result.stdout)
        self.assertIn("--rdzv-endpoint 192.0.2.1:29571", result.stdout)
        self.assertIn("--rdzv-id portable-test", result.stdout)
        self.assertNotIn("--standalone", result.stdout)


if __name__ == "__main__":
    unittest.main()
