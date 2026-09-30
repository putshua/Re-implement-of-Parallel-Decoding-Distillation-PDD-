"""RF-space PDD + endpoint DMD primitives used by JointTraining.

Times are noise fractions r (Wan input timestep = 1000*r), not TrigFlow angles.
Fake velocities must be conditioned on the endpoint/phase being trained.
"""

import torch
from pdd.core import pd_loss


def _times(xs, endpoint, time):
    s = torch.as_tensor(endpoint, device=xs.device, dtype=xs.dtype)
    t = torch.as_tensor(time, device=xs.device, dtype=xs.dtype)
    if s.ndim > 1 or t.ndim > 1:
        raise ValueError("Times must be scalar or [batch]")
    s, t = torch.broadcast_tensors(s, t)
    if s.ndim == 1 and s.numel() != xs.shape[0]:
        raise ValueError("Time batch does not match samples")
    if not (torch.isfinite(s).all() and torch.isfinite(t).all()):
        raise ValueError("Times must be finite")
    if not ((s >= 0) & (s < t) & (t < 1)).all():
        raise ValueError("Require 0 <= endpoint < score time < 1")
    shape = (-1,) + (1,) * (xs.ndim - 1)
    return s.reshape(shape), t.reshape(shape)


def conditional_renoise(xs, endpoint, time, noise):
    """Gaussian Markov re-noising with RF marginals alpha=1-r, sigma=r."""
    if xs.shape != noise.shape:
        raise ValueError("Noise and endpoint shapes must match")
    s, t = _times(xs, endpoint, time)
    alpha = (1 - t) / (1 - s)
    variance = t.square() - alpha.square() * s.square()
    std = variance.clamp_min(0).sqrt()
    return alpha * xs + std * noise, alpha, std


def fake_score_loss(velocity, xs, endpoint, time, noise, variance_cap=10.0):
    """Clamped subinterval velocity DSM; xs is a generated noisy endpoint.

    Implements the RF specialization of Phased-DMD Eq.13. At s=0 this
    reduces to ordinary flow matching, with weight min(1, cap*t**2).
    Only the fake velocity receives gradients.
    """
    if velocity.shape != xs.shape or noise.shape != xs.shape:
        raise ValueError("Velocity, endpoint and noise shapes must match")
    if variance_cap <= 0:
        raise ValueError("variance_cap must be positive")
    xs, noise = xs.detach(), noise.detach()
    s, t = _times(xs, endpoint, time)
    _, _, std = conditional_renoise(xs, endpoint, time, noise)
    scaled_target = (t + (1 - t) * s.square() / (1 - s).square()) * noise
    scaled_target = scaled_target - std * xs / (1 - s)
    residual = std * velocity - scaled_target
    weight = std.square().clamp_min(1e-12).reciprocal().clamp_max(variance_cap)
    return (weight * residual.square()).mean()


def endpoint_dmd_loss(
    xs,
    real_velocity,
    fake_velocity,
    endpoint,
    time,
    normalization="none",
    eps=1e-5,
    *,
    noisy_sample=None,
):
    """Detached score-difference surrogate, with explicit endpoint Jacobian.

    Returns (surrogate, detached direction). 'none' uses a*(score_fake-score_real).
    'residual' normalizes that direction by its per-sample mean absolute value;
    it is a heuristic reweighting, not the unweighted KL gradient.
    'boundary' uses the teacher reconstruction residual as the denominator
    and Euler-projects both denoisers to s. At s=0 it is normalized x0-DMD.
    Loss scalar is a surrogate magnitude, not a measured KL divergence.
    """
    if real_velocity.shape != xs.shape or fake_velocity.shape != xs.shape:
        raise ValueError("Velocity and endpoint shapes must match")
    if xs.ndim < 2 or eps <= 0:
        raise ValueError("Require batched samples and eps > 0")
    s, t = _times(xs, endpoint, time)
    with torch.no_grad():
        alpha = (1 - t) / (1 - s)
        if normalization == "boundary":
            if noisy_sample is None or noisy_sample.shape != xs.shape:
                raise ValueError(
                    "boundary normalization requires the actual noisy_sample"
                )
            # RF velocity is epsilon-x0. Euler denoising to s gives
            # x_s_hat = x_t - (t-s)*v. At s=0 this is standard x0-DMD.
            teacher_endpoint = noisy_sample.float() - (t - s) * real_velocity.float()
            denominator = (
                (xs.detach().float() - teacher_endpoint)
                .abs()
                .mean(tuple(range(1, xs.ndim)), keepdim=True)
                .clamp_min(eps)
            )
            direction = (
                (t - s) * (real_velocity.float() - fake_velocity.float()) / denominator
            )
        else:
            direction = alpha * (1 - t) / t * (real_velocity - fake_velocity)
        if normalization == "residual":
            dims = tuple(range(1, xs.ndim))
            direction = direction / direction.abs().mean(dims, keepdim=True).clamp_min(
                eps
            )
        elif normalization not in ("none", "boundary"):
            raise ValueError("normalization must be none, residual, or boundary")
        if not torch.isfinite(direction).all():
            raise FloatingPointError("Nonfinite DMD direction")
    target = (xs - direction).detach()
    return 0.5 * (xs - target).square().mean(), direction


def joint_phase_forward(
    student, teacher_velocity, noise, context, grid, edges, phase, k, solver="midpoint"
):
    """Fresh on-policy prefix, then one differentiable PDD phase + local MSE.

    Edges are head indices, e.g. [0,32,64,96,128]. Prefix is stop-gradient;
    teacher queries are stop-gradient; current phase endpoint stays on graph.
    This is local phase training of a shared backbone, not full rollout BPTT.
    k must select heads inside the current phase. Caller supplies phase-aware
    fake score and applies endpoint_dmd_loss before backward/optimizer.step.
    """
    if (
        len(edges) < 2
        or edges[0] != 0
        or edges[-1] != len(grid) - 1
        or any(a >= b for a, b in zip(edges, edges[1:]))
        or not 0 <= phase < len(edges) - 1
    ):
        raise ValueError("Invalid phase boundaries/index")
    n, end = edges[phase : phase + 2]
    k = torch.as_tensor(k, device=grid.device, dtype=torch.long)
    if not ((k >= n) & (k < end)).all():
        raise ValueError("MSE target head must be inside current phase")
    x = noise.detach()
    with torch.no_grad():
        for left, right in zip(edges[:phase], edges[1 : phase + 1]):
            coefficients = grid.new_zeros(1, len(grid) - 1)
            coefficients[0, left:right] = grid.diff()[left:right]
            x = x.float() + student(x, grid[left], context, coefficients)[:, 0]
    return pd_loss(
        student,
        teacher_velocity,
        x.detach(),
        context,
        grid,
        n,
        k,
        end - n,
        solver,
        detach_endpoint=False,
    )
