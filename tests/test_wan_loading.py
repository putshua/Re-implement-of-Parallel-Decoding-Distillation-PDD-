import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from pdd.model import load_wan, PDDWan


class TestWanLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = dict(
            _class_name="WanModel",
            dim=16,
            num_heads=2,
            in_dim=16,
            out_dim=16,
            freq_dim=8,
            ffn_dim=32,
            num_layers=1,
            eps=1e-6,
        )
        (self.root / "config.json").write_text(json.dumps(self.cfg))
        self.state = {
            k: torch.randn(v.shape)
            for k, v in load_wan(self.root, init_only=True).state_dict().items()
        }

    def write_shards(self):
        items = list(self.state.items())
        weight_map = {}
        for i in range(2):
            name = f"model-{i}.safetensors"
            shard = dict(items[i::2])
            save_file(shard, self.root / name)
            weight_map.update({k: name for k in shard})
        (self.root / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

    @patch("pdd.model.convert_wan_transformer_to_diffusers", side_effect=lambda x: x)
    def test_sharded_assignment_dtype_and_heads(self, _):
        self.write_shards()
        model = load_wan(self.root, dtype=torch.bfloat16)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, self.state[k].bfloat16())
        student = PDDWan(model, 128)
        self.assertEqual(student.backbone.proj_out.weight.shape, (128, 64, 16))

    @patch("pdd.model.convert_wan_transformer_to_diffusers", side_effect=lambda x: x)
    def test_single_file_and_missing_key(self, _):
        file = self.root / "diffusion_pytorch_model.safetensors"
        save_file(self.state, file)
        self.assertFalse(any(p.is_meta for p in load_wan(self.root).parameters()))
        self.state.pop(next(iter(self.state)))
        save_file(self.state, file)
        with self.assertRaisesRegex(ValueError, "Missing checkpoint keys"):
            load_wan(self.root)

    @patch("pdd.model.convert_wan_transformer_to_diffusers", side_effect=lambda x: x)
    def test_index_mismatch(self, _):
        self.write_shards()
        save_file({"wrong": torch.zeros(1)}, self.root / "model-0.safetensors")
        with self.assertRaisesRegex(ValueError, "index does not match"):
            load_wan(self.root)

    def test_meta_only_never_reads_weights(self):
        with patch("pdd.model.load_file", side_effect=AssertionError("weight read")):
            model = load_wan(self.root, init_only=True, dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in model.parameters())
        )
