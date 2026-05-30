"""Encoder sub-package — §1 Pilot Space.

Provides a ViT-based vision encoder and domain adapter for
converting 2D video frames into 3D spatial latent tokens.

Status: Stub — to be implemented when §1 is developed.
"""

from fsd3d.encoder.vit_encoder import ViTEncoder
from fsd3d.encoder.domain_adapter import DomainAdapter

__all__ = ["ViTEncoder", "DomainAdapter"]
