"""Encoder sub-package — §1 Pilot Space.

Provides a ViT-based vision encoder for converting 2D video frames
into visual tokens that serve as (K, V) context for the §4 decoder's
cross-attention.

Visual adaptation (§3 Data Bridge) has been moved to
``fsd3d.data_bridge.visual_adapter``.
"""

from fsd3d.encoder.vit_encoder import ViTEncoder

# Backward-compatible re-exports (classes now live in fsd3d.data_bridge)
from fsd3d.data_bridge.visual_adapter import VisualAdapter as DomainAdapter
from fsd3d.data_bridge.visual_adapter import LinearVisualAdapter as LinearDomainAdapter

__all__ = ["ViTEncoder", "DomainAdapter", "LinearDomainAdapter"]
