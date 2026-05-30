"""CFM inference — Euler ODE solver.

Starting from pure noise z0, integrate the learned velocity field
from τ=0 to τ=1 to produce the final trajectory.
"""

import torch


def infer_cfm_euler(
    decoder,
    action_head,
    context: torch.Tensor,
    horizon: int = 16,
    euler_steps: int = 10,
    seed: int = None,
):
    """Run CFM inference using the Euler ODE solver.

    Args:
        decoder:     trained FSD3DTransformerDecoder (§3)
        action_head: trained ActionHead (§4)
        context:     (1, 32, 128) context tensor
        horizon:     planning horizon (T)
        euler_steps: number of Euler integration steps
        seed:        optional random seed for the initial noise

    Returns:
        z_final:     (1, horizon, action_dim) — final trajectory ≈ z1
        trajectory:  list of (1, horizon, action_dim) tensors — one per Euler step
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = next(decoder.parameters()).device
    decoder.eval()
    action_head.eval()

    action_dim = action_head.action_dim
    dt = 1.0 / euler_steps
    z = torch.randn(1, horizon, action_dim, device=device)
    ctx = context.expand(1, -1, -1).to(device)

    trajectory = [z.clone()]

    for step in range(euler_steps):
        tau = torch.tensor([[step * dt]], device=device)
        latent = decoder(z, tau, ctx)
        v = action_head(latent)
        z = z + v * dt
        trajectory.append(z.clone())

    return z, trajectory
