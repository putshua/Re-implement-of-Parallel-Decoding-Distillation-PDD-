"""Paper Algorithms 1/3 in Wan's decreasing noise-time convention."""

import torch
from torch.nn import functional as F


def sigma_grid(n=64, shift=6.0, device="cpu"):
    if n < 1 or shift <= 0:
        raise ValueError("Positive grid size and shift required")
    s = torch.linspace(1, 0, n + 1, device=device)
    return shift * s / (1 + (shift - 1) * s)


def training_coefficients(grid, n, k, advance):
    """Selected head plus fused displacement to k and next block boundary."""
    N = len(grid) - 1
    if not 0 <= n <= k < N or not 1 <= advance <= N - n:
        raise ValueError("Invalid block indices")
    a = grid.new_zeros(3, N)
    a[0, k] = 1
    a[1, n:k] = grid.diff()[n:k]
    a[2, n : n + advance] = grid.diff()[n : n + advance]
    return a


def pd_loss(
    student,
    teacher_velocity,
    x,
    context,
    grid,
    n,
    k,
    advance,
    solver,
    *,
    detach_endpoint=True,
):
    # Tensor k is [B,targets]: each example draws its own teacher time. Scalar/list
    # forms are retained for analytic tests and represent shared target indices.
    per_example = isinstance(k, torch.Tensor)
    if per_example:
        indices = k.to(device=grid.device, dtype=torch.long)
        if indices.ndim == 1:
            indices = indices[:, None]
        if indices.ndim != 2 or indices.shape[0] != x.shape[0] or indices.shape[1] == 0:
            raise ValueError("k must have shape [batch, number_of_targets]")
        N = len(grid) - 1
        if (
            not 0 <= n < N
            or not 1 <= advance <= N - n
            or not ((indices >= n) & (indices < N)).all()
        ):
            raise ValueError("Invalid per-example block indices")
        a = grid.new_zeros(x.shape[0], 2 * indices.shape[1] + 1, N)
        rows = torch.arange(x.shape[0], device=grid.device)
        positions = torch.arange(N, device=grid.device)[None, :]
        for j in range(indices.shape[1]):
            a[rows, 2 * j, indices[:, j]] = 1
            a[:, 2 * j + 1] = (
                (positions >= n) & (positions < indices[:, j, None])
            ) * grid.diff()
        a[:, -1, n : n + advance] = grid.diff()[n : n + advance]
        target_indices = list(indices.unbind(1))
    else:
        target_indices = [k] if isinstance(k, int) else list(k)
        if not target_indices:
            raise ValueError("At least one teacher target is required")
        coeffs = [training_coefficients(grid, n, i, advance) for i in target_indices]
        a = torch.cat([c[:2] for c in coeffs] + [coeffs[0][2:]], dim=0)
    outputs = student(x, grid[n], context, a)
    targets = []
    with torch.no_grad():
        for j, i in enumerate(target_indices):
            xk = x.float() + outputs[:, 2 * j + 1].detach()
            target = teacher_velocity(xk, grid[i])
            if solver == "midpoint":
                h = grid[i + 1] - grid[i]
                spatial_h = h.reshape(-1, *([1] * (x.ndim - 1))) if per_example else h
                target = teacher_velocity(
                    xk + 0.5 * spatial_h * target, grid[i] + 0.5 * h
                )
            elif solver != "euler":
                raise ValueError(solver)
            targets.append(target.detach())
    # Joint endpoint supervision needs this branch on the student graph.
    # Teacher targets and the default retained training streams remain detached.
    next_x = x.float() + outputs[:, -1]
    loss = torch.stack(
        [F.mse_loss(outputs[:, 2 * j], target) for j, target in enumerate(targets)]
    ).mean()
    return loss, next_x.detach() if detach_endpoint else next_x


def rollout_edges(grid_size, nfe):
    """Partition every grid interval once, including non-divisor NFEs like 128/3."""
    if not 1 <= nfe <= grid_size:
        raise ValueError("NFE must be between 1 and grid size")
    return [i * grid_size // nfe for i in range(nfe + 1)]


@torch.no_grad()
def sample(student, x, context, grid, nfe):
    N = len(grid) - 1
    edges = rollout_edges(N, nfe)
    for n, end in zip(edges[:-1], edges[1:]):
        a = grid.new_zeros(1, N)
        a[0, n:end] = grid.diff()[n:end]
        x = x.float() + student(x, grid[n], context, a)[:, 0]
    return x
