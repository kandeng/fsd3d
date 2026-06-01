"""§3 — Visual Adapter: adapts visual tokens across data sources.

Adapts visual tokens from different data sources (3DGS, Google Earth,
real drone footage) into a shared representation.  Each data source
provides its own concrete implementation via this interface.

The adapter compensates for domain shift — the same scene rendered by
3DGS vs captured by a real camera produces visually different pixels,
which the ViT encoder encodes into statistically different token
distributions.  The VisualAdapter learns to project these into a
shared space.

Architecture diagram mapping:
  §1 ViT Encoder ──► VisualAdapter ──► adapted visual tokens (B, V, d_model)
"""

import torch
import torch.nn as nn

from fsd3d.constants import D_MODEL


class VisualAdapter(nn.Module):
    """Abstract base class for adapting visual tokens across data sources.

    Subclasses implement a learned projection that maps source-specific
    visual token distributions into a shared representation space.
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Adapt visual tokens from a specific data source.

        Args:
            tokens: (B, T, d_model) input visual tokens from §1 encoder.

        Returns:
            (B, T, d_model) adapted tokens in shared representation space.
        """
        return tokens


class LinearVisualAdapter(VisualAdapter):
    """Simple linear projection adapter for visual domain shift compensation.

    A single Linear(d_model, d_model) layer that learns to project
    source-specific visual tokens into a shared space.
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__(d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project visual tokens through a learned linear layer.

        Args:
            tokens: (B, T, d_model) input visual tokens.

        Returns:
            (B, T, d_model) projected tokens.
        """
        return self.projection(tokens)
