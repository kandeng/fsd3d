"""Autoregressive inference — step-by-step generation.

Delegates to the AutoregressiveWrapper's generate() method.
"""

import torch


def infer_autoregressive(
    wrapper,
    context: torch.Tensor,
    horizon: int = 16,
    noise_step: int = 4,
    noise_sigma: float = 0.05,
    continuous_noise: bool = False,
    drift_bias: float = 0.0,
):
    """Run autoregressive inference via the wrapper's generate() method.

    Args:
        wrapper:          trained AutoregressiveWrapper
        context:         (1, 32, 128) context tensor
        horizon:         planning horizon
        noise_step:      step at which to start injecting noise
        noise_sigma:     std-dev of injected noise
        continuous_noise: inject noise at every step from noise_step onward
        drift_bias:      push x-coordinate toward 0 by this fraction after noise_step

    Returns:
        result: (1, horizon, action_dim) generated trajectory
    """
    wrapper.eval()
    device = next(wrapper.parameters()).device
    ctx = context.expand(1, -1, -1).to(device)

    return wrapper.generate(
        ctx,
        horizon=horizon,
        noise_step=noise_step,
        noise_sigma=noise_sigma,
        continuous_noise=continuous_noise,
        drift_bias=drift_bias,
    )
