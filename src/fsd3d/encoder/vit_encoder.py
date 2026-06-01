"""§1 — ViT Encoder: Pilot Space.

Vision Transformer encoder that converts video frames into visual tokens
serving as (K, V) context for the §4 decoder's cross-attention.

The encoder is source-agnostic — it processes (B, C*N_stack, H, W) image
tensors regardless of whether the pixels came from 3DGS, Google Earth,
or a real camera.  Domain-specific adaptation is handled by a separate
DomainAdapter module.

Architecture (mirrors the decoder but simpler — self-attention only):
  - Patch embedding (ViT-style Conv2d)
  - Learned positional encoding
  - TransformerEncoder stack (2 layers, d_model=128, nhead=4)
  - LayerNorm at output
"""

import torch
import torch.nn as nn

from fsd3d.constants import (
    D_MODEL, NHEAD, DIM_FEEDFORWARD, ENCODER_LAYERS,
    PATCH_SIZE, IMAGE_SIZE, NUM_PATCHES, NUM_FRAMES_STACK,
)


class ViTEncoder(nn.Module):
    """§1 — Pilot Space: Vision Transformer encoder.

    Converts image frames into a sequence of visual tokens that serve as
    (K, V) memory for the §4 decoder's cross-attention.

    The encoder uses self-attention only (no cross-attention, no time
    conditioning, no causal mask).  It compresses the visual scene into
    a compact token sequence that encodes spatial structure for the
    decoder to query.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_frames_stack: int = NUM_FRAMES_STACK,
        patch_size: int = PATCH_SIZE,
        image_size: int = IMAGE_SIZE,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = (image_size // patch_size) ** 2

        # Project patches into the model dimension
        total_in_channels = in_channels * num_frames_stack
        self.patch_embedding = nn.Conv2d(
            total_in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )

        # Learned positional encoding — tells the model which patch is where
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # Transformer encoder stack (Post-LN, matching decoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,       # Post-LN (matches decoder)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Final norm — output is visual tokens
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual tokens.

        Args:
            images: (B, C*N_stack, H, W) input images.
                    For 4 stacked RGB frames: (B, 12, 224, 224).

        Returns:
            (B, num_patches, d_model) visual tokens — fed as K, V into §4.
        """
        B = images.size(0)

        # 1. Patch embedding: (B, C*N_stack, H, W) → (B, d_model, H', W')
        patch_tokens = self.patch_embedding(images)
        # 2. Flatten spatial dims: (B, d_model, H', W') → (B, num_patches, d_model)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        # 3. Add learned positional encoding
        patch_tokens = patch_tokens + self.position_embedding[:, :self.num_patches, :]
        # 4. Self-attention encoder
        encoded = self.transformer(patch_tokens)
        # 5. Final norm
        visual_tokens = self.norm(encoded)
        return visual_tokens
