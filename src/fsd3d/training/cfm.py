"""CFM training — Conditional Flow Matching with interpolation.

Trains §3 decoder + §4 ActionHead jointly via random τ interpolation.
"""

import torch
import torch.nn.functional as F


def train_cfm(
    decoder,
    action_head,
    z1: torch.Tensor,
    context: torch.Tensor,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    log_interval: int = 50,
):
    """Train the decoder + ActionHead with the Conditional Flow Matching objective.

    Args:
        decoder:     FSD3DTransformerDecoder instance (§3)
        action_head: ActionHead instance (§4)
        z1:          (N, 16, 2) target plans (N ≥ 1 expert trajectories)
        context:     (1, 32, 128) context tensor
        epochs:      number of training epochs
        batch_size:  mini-batch size
        lr:          learning rate
        seed:        random seed for reproducibility
        log_interval: print loss every N epochs (0 = silent)

    Returns:
        (decoder, action_head) — trained modules (modified in-place).
    """
    torch.manual_seed(seed)
    device = next(decoder.parameters()).device
    decoder.train()
    action_head.train()

    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(action_head.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        indices = torch.randint(0, z1.size(0), (batch_size,))
        z1_batch = z1[indices].to(device)
        ctx_batch = context.expand(batch_size, -1, -1).to(device)

        optimizer.zero_grad()
        z0 = torch.randn_like(z1_batch)
        tau = torch.rand(batch_size, 1, device=device)
        z_tau = tau.unsqueeze(2) * z1_batch + (1 - tau.unsqueeze(2)) * z0
        v_target = z1_batch - z0

        latent = decoder(z_tau, tau, ctx_batch)
        v_pred = action_head(latent)

        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if log_interval > 0 and (epoch + 1) % log_interval == 0:
            print(f"[CFM] Epoch {epoch + 1}/{epochs}  loss={loss.item():.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return decoder, action_head
