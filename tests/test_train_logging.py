import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestTrainLogging(unittest.TestCase):
    def test_console_file_stderr_child_and_exit_status(self):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(
                os.environ,
                OUTPUT=folder,
                DRY_RUN="0",
                PDD_LOGGING_ACTIVE="0",
                LOG_DIR=folder + "/logs",
            )
            code = """set -euo pipefail
source scripts/train_logging.sh
printf 'step=1 loss=0.25\\n'
printf 'error example\\n' >&2
bash -c 'source scripts/train_logging.sh; echo child-output'
exit 7
"""
            result = subprocess.run(
                ["bash", "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 7)
            files = list(Path(folder).glob("logs/*.log"))
            self.assertEqual(len(files), 1)
            text = files[0].read_text()
            for line in ("step=1 loss=0.25", "error example", "child-output"):
                self.assertEqual(text.count(line), 1)
                self.assertIn(line, result.stdout)
            # A new launch creates its own log instead of overwriting the previous one.
            subprocess.run(
                ["bash", "-c", "source scripts/train_logging.sh; echo restart"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                check=True,
            )
            self.assertEqual(len(list(Path(folder).glob("logs/*.log"))), 2)
            self.assertEqual(files[0].read_text(), text)

    def test_dry_run_does_not_create_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, OUTPUT=folder, DRY_RUN="1", PDD_LOGGING_ACTIVE="0")
            subprocess.run(
                ["bash", "-c", "source scripts/train_logging.sh"],
                cwd=ROOT,
                env=env,
                check=True,
            )
            self.assertFalse((Path(folder) / "logs").exists())
