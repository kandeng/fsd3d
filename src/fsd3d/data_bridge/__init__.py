"""§3 — Data Bridge: Context Normalization (The Data Bridge).

Bridges §1 Pilot Space and §2 Conditioning into the normalized context
tensor (K, V) consumed by the §4 FSD3D Transformer Decoder.

Contains:
  - VisualAdapter: adapts visual tokens across data sources (3DGS, real camera)
  - ContextNormalizer: merges + normalizes visual + conditioning tokens to (B, 32, 128)
"""

from fsd3d.data_bridge.visual_adapter import VisualAdapter, LinearVisualAdapter
from fsd3d.data_bridge.context_normalizer import ContextNormalizer

__all__ = ["VisualAdapter", "LinearVisualAdapter", "ContextNormalizer"]
