"""§5 — Action Loop: converts latent features to control commands.

A single linear projection from d_model latent space to action_dim
control space.  Zero-initialised so early predictions are near-zero,
matching the original output_head behaviour.
"""

import torch.nn as nn


class ActionHead(nn.Module):
    """§5 — Action Loop: converts latent features to control commands."""

    def __init__(self, d_model: int = 128, action_dim: int = 2):
        super().__init__()
        self.head = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.action_dim = action_dim

    def forward(self, latent):
        """Project latent features to action space.

        Args:
            latent: [B, T, d_model] — latent features from §3 decoder

        Returns:
            actions: [B, T, action_dim] — velocity vectors (CFM) or
                     coordinate predictions (AR).  Part of §5 Action Loop.
        """
        return self.head(latent)
