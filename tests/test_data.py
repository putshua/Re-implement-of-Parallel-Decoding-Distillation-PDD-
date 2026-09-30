import io
import json
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from pdd.data import WeightedPromptCache, PromptEmbeddings


class TestData(unittest.TestCase):
    def test_weighted_zero_slots_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "weights.npy", np.array([0.0, 1.0], dtype=np.float32))
            index = {
                "version": "taxonomy_wds_repeat_weight_v1",
                "coverage_complete": True,
                "arrays": {str(root): "weights.npy"},
            }
            (root / "index.json").write_text(json.dumps(index))
            with tarfile.open(root / "shard_000000.tar", "w") as tar:
                for i in range(2):
                    stream = io.BytesIO()
                    torch.save(torch.full((512, 4096), float(i)), stream)
                    payload = stream.getvalue()
                    member = tarfile.TarInfo(f"{i:09d}.embed.pt")
                    member.size = len(payload)
                    tar.addfile(member, io.BytesIO(payload))
            iterator = iter(WeightedPromptCache(root / "index.json"))
            item = next(iterator)
            self.assertTrue(torch.equal(item["context"], torch.ones(512, 4096)))

    def test_short_final_shard_preserves_row_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = np.zeros(257, dtype=np.float32)
            weights[0] = weights[256] = 1
            np.save(root / "weights.npy", weights)
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "version": "taxonomy_wds_repeat_weight_v1",
                        "coverage_complete": True,
                        "arrays": {str(root): "weights.npy"},
                    }
                )
            )
            for shard, key in [(0, 0), (1, 256)]:
                with tarfile.open(root / f"shard_{shard:06d}.tar", "w") as tar:
                    member = tarfile.TarInfo(f"{key:09d}.embed.pt")
                    member.size = 1
                    tar.addfile(member, io.BytesIO(b"x"))
            context = torch.ones(512, 4096)
            with patch("pdd.data.torch.load", return_value=context):
                iterator = iter(WeightedPromptCache(root / "index.json", seed=42))
                tail_count = sum(
                    next(iterator)["prompt"].endswith(":256") for _ in range(32)
                )
            self.assertGreater(tail_count, 6)
            self.assertLess(tail_count, 26)

    def test_missing_cache_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "weights.npy", np.ones(1))
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "version": "taxonomy_wds_repeat_weight_v1",
                        "coverage_complete": True,
                        "arrays": {str(root / "missing"): "weights.npy"},
                    }
                )
            )
            with self.assertRaises(FileNotFoundError):
                WeightedPromptCache(root / "index.json")

    def test_prompt_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.pt"
            torch.save(
                {"prompts": ["a", "b"], "t5_text_embeddings": torch.zeros(1, 2, 3)},
                path,
            )
            with self.assertRaises(ValueError):
                PromptEmbeddings(path)


if __name__ == "__main__":
    unittest.main()
