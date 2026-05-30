"""Domain Adapter — §1 Pilot Space (stub).

Adapts visual tokens from different data sources (3DGS, Google Earth,
real drone footage) into a shared representation before feeding into
the decoder.  Each data source plugs into this adapter via the
DataSourcePlugin interface.

Planned architecture:
  - Source-specific projection layers
  - Shared normalization
  - Output: (B, num_tokens, d_model) adapted tokens
"""

import torch
import torch.nn as nn


class DomainAdapter(nn.Module):
    """Domain adaptation layer for visual tokens (stub).

    Currently a no-op pass-through.  Will be expanded to handle
    source-specific projections when §1 is fully developed.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Adapt visual tokens from a specific domain.

        Args:
            tokens: (B, T, d_model) input visual tokens.

        Returns:
            (B, T, d_model) adapted tokens.
        """
        return tokens
