import unittest
import torch

from pdd.joint_training import JointTraining, cyclic_rollout_nfe
from pdd.core import sigma_grid, rollout_edges, sample


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, x, sigma, context, coefficients):
        self.calls.append((sigma.item(), coefficients.nonzero().tolist()))
        return (coefficients.sum() * self.bias).expand(x.shape[0], 1, *x.shape[1:])


class TestStandardEndpointDMD(unittest.TestCase):
    def test_four_to_one_updates_cycle_rollout_one_to_four(self):
        nfes = [1, 2, 3, 4]
        sequence = [cyclic_rollout_nfe(i, nfes) for i in range(10)]
        self.assertEqual(sequence, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2])
        self.assertEqual([sequence[i] for i in (4, 9)], [1, 2])
        self.assertEqual(rollout_edges(128, 3), [0, 42, 85, 128])
        student = TinyStudent()
        sample(student, torch.zeros(1, 1), torch.zeros(1, 1, 1), sigma_grid(8), 3)
        self.assertEqual(len(student.calls), 3)

    def test_actual_generator_loss_weight_scales_backward(self):
        from unittest.mock import patch
        grads = []
        for weight in (0.5, 2.0):
            j = JointTraining.__new__(JointTraining)
            j.cfg = dict(num_heads=8, dmd_min_gap=0.001, dmd_time_shift=2.,
                         guidance=5., skip_layer=-1, dmd_teacher_skip_layer=-1,
                         dmd_normalization="boundary", dmd_ramp_steps=0,
                         dmd_weight=weight, traj_weight=0.)
            j.edges, j.grid = [0, 2, 4, 6, 8], sigma_grid(8)
            j.student, j.device = TinyStudent(), torch.device("cpu")
            j.dmd_endpoint_mode, j.enabled = "final", True
            j.data, j.rows = [(torch.zeros(1, 2), torch.zeros(1, 1, 1), 3)], []
            j.head_rng = torch.Generator().manual_seed(5)
            j.teacher = j.negative = None
            j._fake_velocity = lambda xt, t, context, phase: torch.zeros_like(xt)
            torch.manual_seed(11)
            with patch("pdd.joint_training.guided_velocity", side_effect=lambda model, x, *args: torch.ones_like(x)):
                loss, _ = j.generator_micro(0, 0)
                (loss / 4).backward()
            grads.append(j.student.bias.grad.clone())
        self.assertGreater(grads[0].abs().item(), 0)
        torch.testing.assert_close(grads[1], grads[0] * 4)

    def test_multistep_wan_checkpointing_preserves_gradients(self):
        import copy
        from diffusers import WanTransformer3DModel
        from pdd.model import PDDWan
        torch.manual_seed(9)
        model = PDDWan(WanTransformer3DModel(
            num_attention_heads=2, attention_head_dim=16, in_channels=4,
            out_channels=4, text_dim=12, freq_dim=16, ffn_dim=48, num_layers=1,
        ), 8)
        checkpointed = copy.deepcopy(model)
        checkpointed.backbone.enable_gradient_checkpointing()
        noise = torch.randn(1, 4, 1, 4, 4)
        context = torch.randn(1, 8, 12)
        outputs = []
        for student in (model, checkpointed):
            j = JointTraining.__new__(JointTraining)
            j.cfg = {"num_heads": 8}
            j.edges, j.grid, j.student = [0, 2, 4, 6, 8], sigma_grid(8), student
            out = j._full_rollout(noise, context)
            out.square().mean().backward()
            outputs.append(out.detach())
        torch.testing.assert_close(*outputs)
        for (name, p), (_, q) in zip(model.named_parameters(), checkpointed.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, msg=name)

    def test_final_mode_rolls_all_phases(self):
        j = JointTraining.__new__(JointTraining)
        j.cfg = {"num_heads": 8, "phase_edges": [0, 2, 4, 6, 8],
                 "dmd_endpoint_mode": "final"}
        j.edges = j.cfg["phase_edges"]
        j.grid = sigma_grid(8)
        j.student = TinyStudent()
        x = torch.zeros(1, 1)
        out = j._full_rollout(x, torch.zeros(1, 1, 1))
        self.assertEqual(len(j.student.calls), 4)
        torch.testing.assert_close(out, torch.full_like(out, -1.0))
        out.sum().backward()
        torch.testing.assert_close(j.student.bias.grad, torch.tensor(-1.0))

    def test_final_mode_three_step_partition_has_gradients(self):
        j = JointTraining.__new__(JointTraining)
        j.cfg = {"num_heads": 8}
        j.edges = [0, 2, 4, 6, 8]
        j.grid = sigma_grid(8)
        j.student = TinyStudent()
        out = j._full_rollout(torch.zeros(1, 1), torch.zeros(1, 1, 1), nfe=3)
        self.assertEqual(len(j.student.calls), 3)
        torch.testing.assert_close(out, torch.full_like(out, -1.0))
        out.sum().backward()
        torch.testing.assert_close(j.student.bias.grad, torch.tensor(-1.0))

    def test_final_score_endpoint_is_clean_sample(self):
        j = JointTraining.__new__(JointTraining)
        j.cfg = {"num_heads": 8, "dmd_min_gap": 0.001,
                 "dmd_time_shift": 2.0, "dmd_endpoint_mode": "final"}
        j.grid = sigma_grid(8)
        j.device = torch.device("cpu")
        j.dmd_endpoint_mode = "final"
        xs = torch.zeros(2, 3)
        s, _, _, _ = j._score_sample(xs, 0)
        self.assertTrue(torch.allclose(s, torch.zeros_like(s)))

    def test_boundary_dmd_matches_clean_denoisers_and_descent(self):
        from pdd.phased_dmd import endpoint_dmd_loss
        x = torch.tensor([[2., 4.]], requires_grad=True)
        xt = torch.tensor([[3., 5.]])
        real = torch.tensor([[2., 2.]], requires_grad=True)
        fake = torch.tensor([[1., 1.]], requires_grad=True)
        t = 0.5
        x0_real, x0_fake = xt - t * real, xt - t * fake
        expected = (x0_fake - x0_real) / (x.detach() - x0_real).abs().mean().clamp_min(1e-5)
        loss, direction = endpoint_dmd_loss(x, real, fake, 0., t, "boundary", noisy_sample=xt)
        torch.testing.assert_close(direction, expected)
        loss.backward()
        torch.testing.assert_close(x.grad, expected / x.numel())
        self.assertIsNone(real.grad)
        self.assertIsNone(fake.grad)

    def test_boundary_phase_projects_to_same_endpoint(self):
        from pdd.phased_dmd import endpoint_dmd_loss
        x, xt = torch.ones(2, 3), torch.full((2, 3), 4.)
        real, fake = torch.ones_like(x), torch.zeros_like(x)
        _, direction = endpoint_dmd_loss(x, real, fake, 0.4, 0.8, "boundary", noisy_sample=xt)
        torch.testing.assert_close(direction, torch.full_like(x, 0.4 / 2.6))


if __name__ == "__main__":
    unittest.main()
