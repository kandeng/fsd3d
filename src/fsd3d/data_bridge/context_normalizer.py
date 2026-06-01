"""§3 — Context Normalizer: The Data Bridge.

Merges visual tokens (from §1 Pilot Space / VisualAdapter) and conditioning
tokens (from §2 Conditioning / TelemetryEncoder + PathEncoder), then
normalizes the combined sequence to exactly ``CONTEXT_TOKENS`` (32) tokens
via truncation or zero-padding.

This module is the sole owner of:
  - Source ID embedding (learned vector added to all tokens)
  - Post-merge linear projection
  - Token-length normalization (truncate / pad)

Architecture diagram mapping:
  §1 VisualAdapter ──┐
                     ├─► ContextNormalizer ──► context (B, 32, 128)
  §2 Conditioner  ───┘         │
                   Source ID ───┘
"""

import torch
import torch.nn as nn

from fsd3d.constants import D_MODEL, CONTEXT_TOKENS


class ContextNormalizer(nn.Module):
    """§3 — Context Normalizer (The Data Bridge).

    Merges outputs from §1 (visual tokens via VisualAdapter) and §2
    (conditioning tokens via Conditioner), adds a source ID embedding,
    projects through a shared linear layer, and normalizes to exactly
    ``context_tokens`` positions.

    Args:
        d_model: Transformer hidden dimension.
        context_tokens: Fixed output token count (default 32).
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        context_tokens: int = CONTEXT_TOKENS,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_tokens = context_tokens

        # Source ID embedding — distinguishes data sources in multi-domain training
        self.source_id_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Shared projection after concatenation
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        conditioning_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Merge, project, and normalize to fixed-length context.

        Args:
            visual_tokens:      (B, V, d_model) from §1 VisualAdapter.
            conditioning_tokens:(B, C, d_model) from §2 Conditioner
                                (telemetry + path tokens already concatenated).

        Returns:
            (B, context_tokens, d_model) normalized context tensor — the K, V
            memory bank for the §4 decoder's cross-attention.
        """
        B = visual_tokens.size(0)

        # 1. Concatenate §1 visual + §2 conditioning tokens
        all_tokens = torch.cat([visual_tokens, conditioning_tokens], dim=1)
        # (B, V + C, d_model)

        # 2. Add source ID embedding to all tokens
        all_tokens = all_tokens + self.source_id_embedding

        # 3. Project through shared linear
        all_tokens = self.projection(all_tokens)

        # 4. Truncate or pad to fixed context_tokens length
        T = all_tokens.size(1)
        if T >= self.context_tokens:
            context = all_tokens[:, :self.context_tokens, :]
        else:
            # Pad with zeros
            padding = torch.zeros(
                B, self.context_tokens - T, self.d_model,
                device=all_tokens.device, dtype=all_tokens.dtype,
            )
            context = torch.cat([all_tokens, padding], dim=1)

        return context
