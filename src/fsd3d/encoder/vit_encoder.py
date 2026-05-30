"""ViT Encoder — §1 Pilot Space (stub).

Will convert 2D video frames into a sequence of visual tokens
suitable for cross-attention in the FSD3DTransformerDecoder.

Planned architecture:
  - Patch embedding (ViT-style)
  - Positional encoding
  - Transformer encoder stack
  - Output: (B, num_patches, d_model) visual tokens
"""

import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    """Vision Transformer encoder (stub).

    Produces visual tokens from 2D image input.  Currently returns
    a zero tensor of the correct shape so downstream modules can be
    developed independently.
    """

    def __init__(self, d_model: int = 128, num_tokens: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual tokens.

        Args:
            images: (B, C, H, W) input images.

        Returns:
            (B, num_tokens, d_model) visual tokens.
        """
        B = images.size(0)
        return torch.zeros(B, self.num_tokens, self.d_model, device=images.device)
