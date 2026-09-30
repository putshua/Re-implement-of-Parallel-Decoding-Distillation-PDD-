import unittest
import torch
from torch import nn
from pdd.core import sigma_grid, sample
from pdd.phased_dmd import (
    conditional_renoise,
    fake_score_loss,
    endpoint_dmd_loss,
    joint_phase_forward,
)


class HeadStudent(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.v = nn.Parameter(torch.linspace(0.1, 0.8, n))
        self.calls = []

    def forward(self, x, time, context, a):
        self.calls.append(torch.is_grad_enabled())
        y = a @ self.v
        if y.ndim == 1:
            y = y[None]
        if y.ndim == 2 and y.shape[0] != x.shape[0]:
            y = y.expand(x.shape[0], -1)
        return y[..., None].expand(-1, -1, x.shape[1])


class TestPhasedDMD(unittest.TestCase):
    def test_marginal_coefficients_and_clean_limit(self):
        x, e = torch.randn(3, 5), torch.randn(3, 5)
        s, t = torch.tensor([0.0, 0.3, 0.8]), torch.tensor([0.2, 0.7, 0.95])
        xt, a, b = conditional_renoise(x, s, t, e)
        torch.testing.assert_close(a[:, 0] * (1 - s), 1 - t)
        torch.testing.assert_close(
            a[:, 0].square() * s.square() + b[:, 0].square(), t.square()
        )
        torch.testing.assert_close(xt[0], 0.8 * x[0] + 0.2 * e[0])
        for endpoint, time in [(1.0, 1.0), (0.5, 0.5), (0.8, 0.2), (-0.1, 0.2)]:
            with self.assertRaises(ValueError):
                conditional_renoise(x, endpoint, time, e)

    def test_fake_clean_limit_and_detach(self):
        x = torch.randn(2, 5, requires_grad=True)
        e = torch.randn_like(x)
        v = torch.randn(2, 5, requires_grad=True)
        loss = fake_score_loss(v, x, 0.0, 0.6, e)
        torch.testing.assert_close(loss, (v - (e - x)).square().mean())
        loss.backward()
        self.assertIsNone(x.grad)
        self.assertIsNotNone(v.grad)

    def test_phase_target_matches_score_dsm(self):
        x, e = torch.randn(2, 5), torch.randn(2, 5)
        s, t = 0.65, 0.83
        xt, a, b = conditional_renoise(x, s, t, e)
        # Convert conditional Gaussian score -epsilon/b into Wan velocity.
        target_v = (t * e / b - xt) / (1 - t)
        torch.testing.assert_close(
            fake_score_loss(target_v, x, s, t, e), torch.tensor(0.0), atol=1e-11, rtol=0
        )
        self.assertGreater(fake_score_loss(e - x, x, s, t, e).item(), 0.01)

    def test_generator_sign_jacobian_and_frozen_scores(self):
        x = torch.randn(2, 5, requires_grad=True)
        real = torch.ones_like(x, requires_grad=True)
        fake = torch.zeros_like(x, requires_grad=True)
        loss, direction = endpoint_dmd_loss(x, real, fake, 0.3, 0.7)
        expected = ((1 - 0.7) / (1 - 0.3)) * (1 - 0.7) / 0.7
        torch.testing.assert_close(direction, torch.full_like(x, expected))
        loss.backward()
        torch.testing.assert_close(x.grad, direction / x.numel())
        self.assertIsNone(real.grad)
        self.assertIsNone(fake.grad)

    def test_joint_rollout_matches_sampler_and_local_gradients(self):
        model = HeadStudent(8)
        grid, edges = sigma_grid(8), [0, 2, 4, 6, 8]
        noise = torch.ones(2, 3, requires_grad=True)

        def teacher(x, t):
            self.assertFalse(torch.is_grad_enabled())
            return torch.zeros_like(x)

        mse, endpoint = joint_phase_forward(
            model, teacher, noise, None, grid, edges, 3, torch.tensor([[6], [7]])
        )
        self.assertEqual(model.calls, [False, False, False, True])
        reference = sample(model, noise.detach(), None, grid, 4)
        torch.testing.assert_close(endpoint, reference)
        (mse + endpoint.square().mean()).backward()
        torch.testing.assert_close(model.v.grad[:6], torch.zeros(6))
        self.assertGreater(model.v.grad[6:].abs().sum().item(), 0)
        self.assertIsNone(noise.grad)


if __name__ == "__main__":
    unittest.main()
