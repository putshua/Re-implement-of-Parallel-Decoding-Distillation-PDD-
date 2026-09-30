from pathlib import Path
import tempfile
import unittest
from pdd.run_state import resolve_resume, reconcile_metrics, checkpoint_due


class TestRunState(unittest.TestCase):
    def checkpoint(self, root, step, complete=True, ranks=2):
        folder = root / f"step_{step:06d}"
        folder.mkdir()
        (folder / "model.pt").write_bytes(b"model")
        for rank in range(ranks):
            (folder / f"rank_{rank}.pt").write_bytes(b"state")
        if complete:
            (folder / "COMPLETE").touch()
        return folder

    def test_auto_fresh_and_latest_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIsNone(resolve_resume(root, "auto", 2))
            self.checkpoint(root, 25)
            latest = self.checkpoint(root, 100)
            self.checkpoint(root, 125, complete=False)
            self.checkpoint(root, 150, ranks=1)
            self.assertEqual(resolve_resume(root, "auto", 2), str(latest))
            self.assertIsNone(resolve_resume(root, "none", 2))
            self.assertEqual(
                resolve_resume(root, str(root / "step_000025"), 2),
                str(root / "step_000025"),
            )

    def test_joint_resume_requires_fake_and_joint_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.checkpoint(root, 1)
            second = self.checkpoint(root, 2)
            for name in ("fake.pt", "joint_state.json"):
                (first / name).write_bytes(b"state")
            (second / "joint_state.json").write_bytes(b"state")
            self.assertEqual(
                resolve_resume(root, "auto", 2, ["fake.pt", "joint_state.json"]),
                str(first),
            )
            (second / "fake.pt").write_bytes(b"state")
            self.assertEqual(
                resolve_resume(root, "auto", 2, ["fake.pt", "joint_state.json"]),
                str(second),
            )

    def test_no_silent_restart_if_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metrics.jsonl").write_text('{"step": 1}\n')
            with self.assertRaises(FileNotFoundError):
                resolve_resume(root, "auto", 2)

    def test_archive_metrics_beyond_checkpoint_and_partial_line(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = '{"step": 100}\n{"step": 101}\n{"step":'
            (root / "metrics.jsonl").write_text(original)
            backup = reconcile_metrics(root, 100)
            self.assertEqual(Path(backup).read_text(), original)
            self.assertEqual((root / "metrics.jsonl").read_text(), '{"step": 100}\n')
            self.assertIsNone(reconcile_metrics(root, 100))


class TestRestartBeforeFirstCheckpoint(unittest.TestCase):
    def test_archive_and_restart_preserves_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metrics.jsonl").write_text('{"step":4}\n{"step":')
            (root / "config.json").write_text("{}")
            for name in ("tensorboard", "fixed_prompt", "step_000005", "logs"):
                (root / name).mkdir()
                (root / name / "data").write_text(name)
            self.assertIsNone(
                resolve_resume(
                    root,
                    "auto",
                    16,
                    ["fake.pt", "joint_state.json"],
                    restart_without_checkpoint=True,
                )
            )
            (archive,) = (root / "restart_archives").iterdir()
            self.assertEqual(
                (archive / "metrics.jsonl").read_text(), '{"step":4}\n{"step":'
            )
            for name in ("tensorboard", "fixed_prompt", "step_000005"):
                self.assertEqual((archive / name / "data").read_text(), name)
                self.assertFalse((root / name).exists())
            self.assertTrue((root / "logs" / "data").exists())
            self.assertFalse((root / "metrics.jsonl").exists())
            self.assertIsNone(
                resolve_resume(root, "auto", 16, restart_without_checkpoint=True)
            )

    def test_damaged_or_wrong_world_completed_checkpoint_never_resets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metrics.jsonl").write_text('{"step":25}\n')
            folder = root / "step_000025"
            folder.mkdir()
            (folder / "COMPLETE").touch()
            with self.assertRaises(FileNotFoundError):
                resolve_resume(
                    root, "auto", 16, ["fake.pt"], restart_without_checkpoint=True
                )
            self.assertTrue((root / "metrics.jsonl").exists())
            self.assertFalse((root / "restart_archives").exists())

    def test_existing_complete_checkpoint_is_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metrics.jsonl").write_text('{"step":4}\n')
            folder = root / "step_000001"
            folder.mkdir()
            for name in (
                "model.pt",
                "rank_0.pt",
                "fake.pt",
                "joint_state.json",
                "COMPLETE",
            ):
                (folder / name).write_bytes(b"saved")
            self.assertEqual(
                resolve_resume(
                    root,
                    "auto",
                    1,
                    ["fake.pt", "joint_state.json"],
                    restart_without_checkpoint=True,
                ),
                str(folder),
            )
            self.assertFalse((root / "restart_archives").exists())

    def test_explicit_none_does_not_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metrics.jsonl").write_text('{"step":4}\n')
            self.assertIsNone(
                resolve_resume(root, "none", 16, restart_without_checkpoint=True)
            )
            self.assertTrue((root / "metrics.jsonl").exists())

    def test_first_periodic_final_save(self):
        self.assertEqual(
            [s for s in range(1, 13) if checkpoint_due(s, 5, 12)], [1, 5, 10, 12]
        )
        self.assertEqual(
            [s for s in range(1, 13) if checkpoint_due(s, 5, 12, False)], [5, 10, 12]
        )
