import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pdd.monitoring import (
    evaluation_state,
    prompt_slots,
    preview_due,
    TrainingMonitor,
    write_preview_media,
)


class TestMonitoring(unittest.TestCase):
    def test_collective_padding(self):
        for count in (1, 3, 4, 5, 9):
            slots = [list(prompt_slots(count, 4, rank)) for rank in range(4)]
            self.assertEqual(len({len(s) for s in slots}), 1)
            self.assertEqual(
                sorted(i for s in slots for i, keep in s if keep), list(range(count))
            )

    def test_schedule(self):
        self.assertTrue(preview_due(0, 25, startup=True))
        self.assertFalse(preview_due(1, 25))
        self.assertTrue(preview_due(25, 25))
        self.assertTrue(preview_due(26, 25, final=True))
        self.assertFalse(preview_due(26, 25, final=True, at_end=False))

    def test_state_restored_on_error(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        model.train()
        model[0].eval()
        model.coefficients = torch.ones(2)
        original = model.coefficients
        tr, py, npstate = (
            torch.get_rng_state(),
            random.getstate(),
            np.random.get_state(),
        )
        with self.assertRaises(ValueError):
            with evaluation_state(model, torch.device("cpu")):
                self.assertFalse(model.training)
                torch.rand(5)
                random.random()
                np.random.rand(5)
                model.coefficients = None
                raise ValueError("test")
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertIs(model.coefficients, original)
        self.assertTrue(torch.equal(torch.get_rng_state(), tr))
        self.assertEqual(random.getstate(), py)
        np.testing.assert_array_equal(np.random.get_state()[1], npstate[1])

    def test_changed_prompts_preserve_completed_preview(self):
        import json
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "prompts.pt"
            torch.save(
                {"prompts": ["RCM prompt"], "t5_text_embeddings": torch.zeros(1, 2, 4)},
                cache,
            )
            cfg = dict(
                fixed_prompt_enabled=True,
                fixed_prompt_count=1,
                fixed_prompt_decode=False,
                tensorboard_enabled=False,
                fixed_prompt_embeddings=str(cache),
                fixed_prompt_nfe=[1],
                num_heads=2,
                block_min=1,
                block_max=2,
                shift=6,
                height=16,
                width=16,
                frames=5,
            )
            old = root / "fixed_prompt" / "step_000025"
            old.mkdir(parents=True)
            (old / "manifest.json").write_text(json.dumps({"signature": "old-prompts"}))
            (old / "COMPLETE").touch()
            monitor = TrainingMonitor(cfg, root, torch.device("cpu"))
            with patch(
                "pdd.monitoring.sample", side_effect=lambda student, x, *args: x
            ) as sample:
                monitor.run_preview(torch.nn.Linear(1, 1), 25)
                monitor.run_preview(torch.nn.Linear(1, 1), 25)
                self.assertEqual(sample.call_count, 1)
            self.assertEqual(
                json.loads((old / "manifest.json").read_text())["signature"],
                "old-prompts",
            )
            new = old.with_name(old.name + "_" + monitor.signature[:12])
            self.assertTrue((new / "COMPLETE").exists())
            self.assertIn(new.name, (old.parent / "index.html").read_text())

    def test_media_and_tensorboard(self):
        from PIL import Image
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_preview_media(
                torch.rand(3, 5, 32, 32) * 2 - 1, root / "preview", preview_width=32
            )
            with Image.open(root / "preview.gif") as gif:
                self.assertEqual(gif.n_frames, 5)
            self.assertTrue((root / "preview.mp4").stat().st_size > 0)
            monitor = TrainingMonitor({}, root, torch.device("cpu"))
            record = dict(
                step=1,
                loss_mean=0.25,
                loss_rank_max=0.5,
                grad_norm=0.2,
                global_batch=8,
                seconds=2,
                samples_per_second=4,
                peak_allocated_gib=1,
                peak_reserved_gib=2,
            )
            monitor.log_step(record, 1e-5)
            monitor.close()
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertEqual(events.Scalars("train/loss")[0].value, 0.25)


if __name__ == "__main__":
    unittest.main()
