"""Encoder sub-package — §1 Pilot Space.

Provides a ViT-based vision encoder and domain adapter for
converting 2D video frames into visual tokens that serve as
(K, V) context for the §3 decoder's cross-attention.
"""

from fsd3d.encoder.vit_encoder import ViTEncoder
from fsd3d.encoder.domain_adapter import DomainAdapter, LinearDomainAdapter

__all__ = ["ViTEncoder", "DomainAdapter", "LinearDomainAdapter"]
