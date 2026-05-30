"""Conditioner — §2 Conditioning (stub).

Will take telemetry (position, velocity, orientation) and A* guidance
(path waypoints) and produce conditioning tokens for the decoder.

Planned architecture:
  - Telemetry embedding (linear projection of scalars)
  - A* path encoder (small Transformer or MLP)
  - Concatenation with §1 visual tokens
  - Output: (B, context_tokens, d_model) full context

Currently the context is assembled by ContextAssembler in the decoder
sub-package using static data.  When §2 is developed, the Conditioner
will replace the static assembly with learned conditioning.
"""

import torch
import torch.nn as nn


class Conditioner(nn.Module):
    """Telemetry + A* guidance conditioner (stub).

    Currently returns a zero tensor of the correct shape.  Will be
    replaced with learned conditioning when §2 is developed.
    """

    def __init__(self, d_model: int = 128, num_tokens: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

    def forward(self, telemetry: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """Produce conditioning tokens from telemetry and guidance.

        Args:
            telemetry: (B, T_telem) raw telemetry readings.
            guidance:  (B, T_guide, 2) A* waypoint coordinates.

        Returns:
            (B, num_tokens, d_model) conditioning tokens.
        """
        B = telemetry.size(0)
        return torch.zeros(B, self.num_tokens, self.d_model, device=telemetry.device)
