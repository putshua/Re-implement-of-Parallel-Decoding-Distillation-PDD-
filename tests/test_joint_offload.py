"""Fake optimizer checkpoint lifecycle must preserve CPU offload on resume."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from pdd.joint_training import JointTraining


class TestFakeOffload(unittest.TestCase):
    def test_save_and_restore_stage_moments_without_changing_values(self):
        joint = JointTraining.__new__(JointTraining)
        joint.enabled = True
        joint.cfg = {'optimizer_cpu_offload': True}
        joint.device = torch.device('cpu')
        joint.rank = 0
        joint.fake_updates = 5
        joint.fake = torch.nn.Linear(3, 2)
        joint.fake_optimizer = torch.optim.AdamW(joint.fake.parameters())
        joint.fake(torch.ones(1, 3)).square().mean().backward()
        joint.fake_optimizer.step()
        expected = {k: v.detach().clone() for k, v in joint.fake.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            with patch('pdd.joint_training.move_optimizer_state', wraps=__import__('pdd.distributed', fromlist=['move_optimizer_state']).move_optimizer_state) as move:
                joint.save(Path(tmp), 1)
                self.assertEqual(move.call_count, 2)
                joint.fake.weight.data.zero_()
                joint.restore(tmp, 1)
                self.assertEqual(move.call_count, 3)
            for k, v in joint.fake.state_dict().items():
                torch.testing.assert_close(v, expected[k])
            self.assertEqual(joint.fake_updates, 5)
            for state in joint.fake_optimizer.state.values():
                self.assertEqual(state['exp_avg'].device.type, 'cpu')
                self.assertEqual(state['step'].item(), 1)
