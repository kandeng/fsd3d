"""AR training — Autoregressive MSE with teacher forcing.

Trains the AutoregressiveWrapper (§3 + §4 + start token) via
teacher forcing with position-MSE loss.
"""

import torch
import torch.nn.functional as F


def train_autoregressive(
    wrapper,
    z1: torch.Tensor,
    context: torch.Tensor,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    log_interval: int = 50,
):
    """Train the decoder with a conventional autoregressive MSE objective.

    Args:
        wrapper:  AutoregressiveWrapper instance
        z1:       (N, 16, 2) target plans (N ≥ 1 expert trajectories)
        context:  (1, 32, 128) context tensor
        epochs:   number of training epochs
        batch_size: mini-batch size
        lr:       learning rate
        seed:     random seed for reproducibility
        log_interval: print loss every N epochs (0 = silent)

    Returns:
        Trained wrapper (same object, modified in-place).
    """
    torch.manual_seed(seed)
    device = next(wrapper.parameters()).device
    wrapper.train()

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=lr)

    for epoch in range(epochs):
        indices = torch.randint(0, z1.size(0), (batch_size,))
        z1_batch = z1[indices].to(device)
        ctx_batch = context.expand(batch_size, -1, -1).to(device)
        tau_fixed = torch.ones(batch_size, 1, device=device)

        optimizer.zero_grad()
        pred = wrapper(z1_batch, tau_fixed, ctx_batch)

        loss = F.mse_loss(pred, z1_batch)
        loss.backward()
        optimizer.step()

        if log_interval > 0 and (epoch + 1) % log_interval == 0:
            print(f"[AR ] Epoch {epoch + 1}/{epochs}  loss={loss.item():.6f}")

    return wrapper
