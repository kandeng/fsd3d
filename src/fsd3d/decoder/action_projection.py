"""§5 — Action Projection: projects z_tau into Q for the decoder.

Maps the current generative state z_tau (action-space tokens) into the
model dimension d_model, where they serve as the Query (Q) for the
§4 FSD3D Transformer Decoder's cross-attention.

Architecture diagram mapping:
  z_tau (Current Generative State) ──► Action Projection ──► Q (Query)
"""

import torch
import torch.nn as nn

from fsd3d.constants import ACTION_DIM, D_MODEL


class ActionProjection(nn.Module):
    """§5 — Action Projection: projects z_tau into Q for the decoder.

    A single linear layer that maps action-space tokens (e.g. 2D positions)
    into the model's hidden dimension, producing the Query tokens for the
    §4 transformer decoder's cross-attention mechanism.

    Args:
        action_dim: Dimension of the action space (default 2 for [x, y]).
        d_model: Transformer hidden dimension.
    """

    def __init__(self, action_dim: int = ACTION_DIM, d_model: int = D_MODEL):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.projection = nn.Linear(action_dim, d_model)

    def forward(self, z_tau: torch.Tensor) -> torch.Tensor:
        """Project z_tau into d_model space.

        Args:
            z_tau: [B, T, action_dim] — current canvas state (generative state).

        Returns:
            [B, T, d_model] — Query tokens for the §4 decoder.
        """
        return self.projection(z_tau)
