"""Domain Adapter — §1 Pilot Space (abstract interface).

Adapts visual tokens from different data sources (3DGS, Google Earth,
real drone footage) into a shared representation before concatenation
with conditioning tokens.  Each data source provides its own concrete
implementation via this interface.

The adapter compensates for domain shift — the same scene rendered by
3DGS vs captured by a real camera produces visually different pixels,
which the ViT encoder encodes into statistically different token
distributions.  The DomainAdapter learns to project these into a
shared space.
"""

import torch
import torch.nn as nn


class DomainAdapter(nn.Module):
    """Abstract base class for domain adaptation of visual tokens.

    Subclasses implement a learned projection that maps source-specific
    visual token distributions into a shared representation space.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Adapt visual tokens from a specific domain.

        Args:
            tokens: (B, T, d_model) input visual tokens from §1 encoder.

        Returns:
            (B, T, d_model) adapted tokens in shared representation space.
        """
        return tokens


class LinearDomainAdapter(DomainAdapter):
    """Simple linear projection adapter for domain shift compensation.

    A single Linear(d_model, d_model) layer that learns to project
    source-specific visual tokens into a shared space.
    """

    def __init__(self, d_model: int = 128):
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
