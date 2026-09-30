import unittest
import torch
from torch import nn
from pdd.core import sigma_grid, training_coefficients, pd_loss, sample
from pdd.model import PDDWan, ParallelProjection
from diffusers import WanTransformer3DModel


class TestPDD(unittest.TestCase):
    def test_grid_and_coefficients(self):
        s = sigma_grid(8, 6)
        self.assertEqual(s[0], 1)
        self.assertEqual(s[-1], 0)
        self.assertTrue((s.diff() < 0).all())
        a = training_coefficients(s, 2, 5, 2)
        self.assertEqual(a[0, 5], 1)
        torch.testing.assert_close(a[1].sum(), s[5] - s[2])
        torch.testing.assert_close(a[2].sum(), s[4] - s[2])

    def test_fusion_and_patch_order(self):
        torch.manual_seed(5)
        linear = nn.Linear(8, 12)
        head = ParallelProjection(linear, 4, 4, 3)
        with torch.no_grad():
            head.weight.add_(torch.randn_like(head.weight))
        a = torch.randn(2, 4)
        head.coefficients = a
        x = torch.randn(2, 6, 8)
        actual = head(x).reshape(2, 6, 4, 2, 3)
        explicit = torch.stack(
            [
                nn.functional.linear(x, w, b).reshape(2, 6, 4, 3)
                for w, b in zip(head.weight, head.bias)
            ],
            dim=3,
        )
        expected = torch.einsum("blpnc,qn->blpqc", explicit, a)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNotNone(head.weight.grad)

    def test_per_example_fusion_and_gradients(self):
        torch.manual_seed(7)
        head = ParallelProjection(nn.Linear(8, 12), 4, 4, 3)
        a = torch.randn(2, 3, 4)
        head.coefficients = a
        x = torch.randn(2, 5, 8)
        actual = head(x).reshape(2, 5, 4, 3, 3)
        explicit = torch.stack(
            [
                nn.functional.linear(x, w, b).reshape(2, 5, 4, 3)
                for w, b in zip(head.weight, head.bias)
            ],
            dim=3,
        )
        expected = torch.einsum("blpnc,bqn->blpqc", explicit, a)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(
            actual.square().sum(), (head.weight, head.bias), retain_graph=True
        )
        ge = torch.autograd.grad(expected.square().sum(), (head.weight, head.bias))
        for left, right in zip(ga, ge):
            torch.testing.assert_close(left, right)

    def test_per_example_midpoint_targets(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.v = nn.Parameter(torch.arange(1.0, 9.0))

            def forward(self, x, t, c, a):
                return (a @ self.v)[..., None].expand(-1, -1, x.shape[1])

        student = Student()
        grid = sigma_grid(8)
        x = torch.ones(2, 3)
        indices = torch.tensor([[2], [5]])
        queries = []

        def teacher(state, t):
            self.assertFalse(torch.is_grad_enabled())
            queries.append((state.clone(), t.clone()))
            return 2 * state + t[:, None]

        loss, next_x = pd_loss(
            student, teacher, x, None, grid, 0, indices, 1, "midpoint"
        )
        k = indices[:, 0]
        expected = torch.stack(
            [
                x[b] + (grid.diff()[:i] * student.v.detach()[:i]).sum()
                for b, i in enumerate(k)
            ]
        )
        torch.testing.assert_close(queries[0][0], expected)
        torch.testing.assert_close(queries[0][1], grid[k])
        h = grid[k + 1] - grid[k]
        torch.testing.assert_close(
            queries[1][0], expected + 0.5 * h[:, None] * (2 * expected + grid[k, None])
        )
        torch.testing.assert_close(queries[1][1], grid[k] + 0.5 * h)
        loss.backward()
        self.assertEqual(student.v.grad.nonzero().flatten().tolist(), [2, 5])
        self.assertFalse(next_x.requires_grad)
        student.zero_grad(set_to_none=True)
        queries.clear()
        indices = torch.tensor([[2, 3], [5, 6]])
        loss, _ = pd_loss(student, teacher, x, None, grid, 0, indices, 1, "euler")
        self.assertEqual(len(queries), 2)
        for j in range(2):
            torch.testing.assert_close(queries[j][1], grid[indices[:, j]])
        loss.backward()
        self.assertEqual(student.v.grad.nonzero().flatten().tolist(), [2, 3, 5, 6])

    def test_native_initialization(self):
        torch.manual_seed(0)
        model = WanTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            text_dim=12,
            freq_dim=16,
            ffn_dim=48,
            num_layers=1,
        )
        model.eval()
        x = torch.randn(1, 4, 1, 4, 4)
        context = torch.randn(1, 8, 12)
        t = torch.tensor(0.7)
        with torch.no_grad():
            base = model(x, (t * 1000).reshape(1), context, return_dict=False)[0]
        pdd = PDDWan(model, 8)
        with torch.no_grad():
            result = pdd(x, t, context, torch.eye(8))
        for i in range(8):
            torch.testing.assert_close(result[:, i], base)
        coefficients = torch.randn(2, 3, 8)
        with torch.no_grad():
            batched = pdd(
                x.expand(2, -1, -1, -1, -1), t, context.expand(2, -1, -1), coefficients
            )
        expected = base[:, None] * coefficients.sum(-1)[..., None, None, None, None]
        torch.testing.assert_close(batched, expected, atol=2e-6, rtol=2e-5)

    def test_constant_velocity_loss_sampling_stop_gradient(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.v = nn.Parameter(torch.ones(8))

            def forward(self, x, t, c, a):
                return (a @ self.v)[None, :, None].expand(x.shape[0], -1, x.shape[1])

        student = Student()
        grid = sigma_grid(8)
        x = torch.zeros(2, 3, requires_grad=True)

        def teacher(x, t):
            self.assertFalse(torch.is_grad_enabled())
            return torch.ones_like(x)

        loss, next_x = pd_loss(student, teacher, x, None, grid, 0, 3, 2, "midpoint")
        self.assertEqual(loss.item(), 0)
        self.assertFalse(next_x.requires_grad)
        torch.testing.assert_close(
            sample(student, x, None, grid, 4), -torch.ones_like(x)
        )
        torch.testing.assert_close(
            sample(student, x, None, grid, 3), -torch.ones_like(x)
        )
        with self.assertRaises(ValueError):
            sample(student, x, None, grid, 0)


class TestTeacherTargets(unittest.TestCase):
    def test_prefix_is_on_policy_and_target_detached(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.v = nn.Parameter(torch.arange(1.0, 9.0))

            def forward(self, x, t, c, a):
                return (a @ self.v)[None, :, None].expand(x.shape[0], -1, x.shape[1])

        student = Student()
        grid = sigma_grid(8)
        x = torch.ones(1, 2)
        queries = []

        def teacher(state, t):
            queries.append((state.clone(), t.clone()))
            return 2 * state + t

        loss, next_x = pd_loss(student, teacher, x, None, grid, 0, [2, 4], 1, "euler")
        for query, i in zip(queries, [2, 4]):
            expected = x + (grid.diff()[:i] * student.v.detach()[:i]).sum()
            torch.testing.assert_close(query[0], expected)
        loss.backward()
        active = student.v.grad.nonzero().flatten().tolist()
        self.assertEqual(active, [2, 4])
        self.assertFalse(next_x.requires_grad)


if __name__ == "__main__":
    unittest.main()
