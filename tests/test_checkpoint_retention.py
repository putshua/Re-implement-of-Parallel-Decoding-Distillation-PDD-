from pathlib import Path
import tempfile
import unittest

from pdd.run_state import prune_checkpoints, resolve_resume


class TestCheckpointRetention(unittest.TestCase):
    def save(self, root, step, complete=True):
        folder = root / f"step_{step:06d}"
        folder.mkdir()
        for name in (
            "model.pt",
            "fake.pt",
            "joint_state.json",
            "rank_0.pt",
            "rank_1.pt",
        ):
            (folder / name).write_bytes(b"state")
        if complete:
            (folder / "COMPLETE").touch()
        return folder

    def test_rolling_and_milestones_auto_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for step in (5, 10, 45, 50, 55, 100, 105):
                self.save(root, step)
                prune_checkpoints(root, step, 50, 2, ["fake.pt", "joint_state.json"])
                expected = {f"step_{s:06d}" for s in (50, 100) if s <= step} | {
                    f"step_{step:06d}"
                }
                self.assertEqual({p.name for p in root.glob("step_*")}, expected)
                self.assertEqual(
                    resolve_resume(root, "auto", 2, ["fake.pt", "joint_state.json"]),
                    str(root / f"step_{step:06d}"),
                )
            self.assertTrue((root / "step_000050/KEEP").exists())

    def test_failed_replacement_preserves_previous(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = self.save(root, 5)
            self.save(root, 10, complete=False)
            with self.assertRaises(ValueError):
                prune_checkpoints(root, 10, 50, 2, ["fake.pt"])
            self.assertTrue(previous.exists())
            self.assertEqual(
                resolve_resume(root, "auto", 2, ["fake.pt"]), str(previous)
            )

    def test_missing_fake_even_with_marker_does_not_prune(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = self.save(root, 5)
            current = self.save(root, 10)
            (current / "fake.pt").unlink()
            with self.assertRaises(ValueError):
                prune_checkpoints(root, 10, 50, 2, ["fake.pt"])
            self.assertTrue(previous.exists())

    def test_incomplete_future_and_explicit_keep_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kept = self.save(root, 5)
            (kept / "KEEP").touch()
            incomplete = self.save(root, 10, False)
            self.save(root, 15)
            future = self.save(root, 20)
            prune_checkpoints(root, 15, 50, 2)
            for path in (kept, incomplete, future):
                self.assertTrue(path.exists())

    def test_external_symlink_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            outside = Path(temp) / "outside"
            outside.mkdir()
            (outside / "COMPLETE").touch()
            (root / "step_000005").symlink_to(outside, target_is_directory=True)
            self.save(root, 10)
            prune_checkpoints(root, 10, 50, 2)
            self.assertTrue((outside / "COMPLETE").exists())
            self.assertTrue((root / "step_000005").is_symlink())
