import json
from pathlib import Path
import tempfile
import unittest
from pdd.checkpoint_io import prefetch_checkpoint


class TestCheckpointPrefetch(unittest.TestCase):
    def test_sharded_dedup_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a").write_bytes(b"123")
            (root / "b").write_bytes(b"4567")
            (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "a", "y": "a", "z": "b"}})
            )
            self.assertEqual(prefetch_checkpoint(root), 7)
            (root / "b").unlink()
            with self.assertRaises(FileNotFoundError):
                prefetch_checkpoint(root)

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "diffusion_pytorch_model.safetensors"
            p.write_bytes(b"1234")
            self.assertEqual(prefetch_checkpoint(d), 4)
            p.write_bytes(b"")
            with self.assertRaises(ValueError):
                prefetch_checkpoint(d)
