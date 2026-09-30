import unittest
from unittest.mock import patch
import torch
from pdd.core import sigma_grid
from pdd.joint_training import JointTraining


class TestFakeWarmup(unittest.TestCase):
    def trainer(self):
        j = JointTraining.__new__(JointTraining)
        j.cfg = dict(seed=42, fake_updates_per_g=4, max_grad_norm=1.,
                     log_micro_every=1, abort_grad_norm_above=100.)
        j.device = torch.device('cpu')
        j.rank, j.world = 0, 1
        j.edges, j.grid = [0, 2, 4, 6, 8], sigma_grid(8)
        j.phases, j.fake_phases = 4, 1
        j.warmup_steps, j.step_offset = 50, 50
        j.count_optimizer_updates, j.updates_per_cycle = True, 5
        j.rollout_nfes, j.dmd_endpoint_mode = [1, 2, 3, 4], 'final'
        j.enabled, j.fake_updates = True, 0
        j.student = torch.nn.Linear(2, 2)
        j.fake = torch.nn.Linear(2, 2)
        j.fake_optimizer = torch.optim.AdamW(j.fake.parameters(), lr=1e-3, foreach=False)
        j._rollout = lambda noise, context, phase, nfe: j.student(noise).detach()
        j._score_sample = lambda xs, phase: (torch.tensor(0.), torch.tensor(.5),
                                             torch.ones_like(xs), xs.detach())
        j._fake_velocity = lambda xt, t, context, phase: j.fake(xt)
        return j

    def test_fake_only_fresh_inputs_and_no_generator_gradients(self):
        j = self.trainer()
        original = {k: v.clone() for k, v in j.student.state_dict().items()}
        original_fake = j.fake.weight.detach().clone()
        contexts = []
        def next_context():
            contexts.append(1)
            return torch.zeros(2, 1, 1)
        with patch('pdd.joint_training.verify_optimizer_fp32'):
            j.begin_step(0, 2, (2, 2), next_context, warmup=True)
            first_noise = j.data[0][0].clone()
            j.begin_step(1, 2, (2, 2), next_context, warmup=True)
        self.assertEqual(j.fake_updates, 2)
        self.assertEqual(len(contexts), 4)
        self.assertEqual(len(j.fake_update_records), 1)
        self.assertTrue(j.fake_update_records[0]['warmup'])
        self.assertEqual(j.fake_update_records[0]['step'], 2)
        self.assertEqual(j.fake_update_records[0]['rollout_nfe'], 2)
        self.assertFalse(torch.equal(first_noise, j.data[0][0]))
        self.assertFalse(torch.equal(original_fake, j.fake.weight))
        self.assertTrue(torch.isfinite(j.fake.weight).all())
        for key, value in j.student.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in j.student.parameters()))

    def test_post_warmup_update_numbers_and_fake_ratio(self):
        j = self.trainer()
        j.fake_updates = 50
        j.begin_step(0, 1, (2, 2), lambda: torch.zeros(2, 1, 1))
        self.assertEqual(j.fake_updates, 54)
        self.assertEqual([r['step'] for r in j.fake_update_records], [51,52,53,54])
        self.assertEqual([r['rollout_nfe'] for r in j.fake_update_records], [3,4,1,2])
        self.assertEqual(j.g_rollout_nfe, 3)
        self.assertTrue(all(not r['warmup'] for r in j.fake_update_records))


if __name__ == '__main__':
    unittest.main()
