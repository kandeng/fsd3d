"""§4 — FSD3D Transformer Decoder: Latent & Flight Generation Space.

Transformer decoder that maps (z_tau, tau, context) → (B, T, d_model)
latent features.  The model is deliberately agnostic to which paradigm
(CFM or AR) uses it — no paradigm-specific branches exist.

The output is in *latent space*, not action space.  A separate ActionHead
(§5) converts latent features to control commands.
"""

import torch
import torch.nn as nn

from fsd3d.decoder.action_projection import ActionProjection


class FSD3DTransformerDecoder(nn.Module):
    """§4 — Latent & Flight Generation Space.

    Decoder layers:
      - action_projection: ActionProjection (2 → 128)  — §5 module, used here as Q source
      - position_embedding: (1, 32, 128) learned
      - time_embedding:    Linear(1, 128)
      - transformer:       TransformerDecoder(d_model=128, nhead=4, num_layers=3,
                            dim_feedforward=512, batch_first=True)
      - norm:              LayerNorm(128)
    """

    def __init__(
        self,
        action_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        context_tokens: int = 32,
        max_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model

        # §5 Action Projection: project z_tau (action-space) → Q (d_model)
        self.action_projection = ActionProjection(action_dim, d_model)

        # Learned positional encoding — tells the model which step each token is
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Embed continuous flow time τ ∈ [0, 1]
        self.time_embedding = nn.Linear(1, d_model)

        # Core transformer decoder stack (Post-LN, as in diagram)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,       # Post-LN
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Final norm — output is in latent space (B, T, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_tau, tau, context, tgt_mask=None):
        """Forward pass — §4 Latent & Flight Generation.

        Args:
            z_tau:   [B, T, action_dim] — current canvas state (§5 z_tau)
            tau:     [B, 1]             — flow time scalar per sample
            context: [B, S, d_model]    — memory bank (K, V) from §3
            tgt_mask:[T, T] or None     — causal mask for AR mode

        Returns:
            latent:  [B, T, d_model] — latent features (§4 output)
        """
        B, T, _ = z_tau.shape

        # 1. §5 Action Projection: z_tau → Q tokens
        q_tokens = self.action_projection(z_tau)        # [B, T, d_model]

        # 2. Add learned positional encoding
        q_tokens = q_tokens + self.position_embedding[:, :T, :]  # [B, T, d_model]

        # 3. Embed flow time and broadcast across the horizon
        t_emb = self.time_embedding(tau)                # [B, d_model]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)   # [B, T, d_model]

        # 4. Add time conditioning (residual)
        q_tokens = q_tokens + t_emb                    # [B, T, d_model]

        # 5. Decode: q_tokens as tgt (Q), context as memory (K, V)
        decoded = self.transformer(
            tgt=q_tokens,
            memory=context,
            tgt_mask=tgt_mask,                          # None for CFM, causal for AR
        )                                               # [B, T, d_model]

        # 6. Final norm — output is latent (§4)
        latent = self.norm(decoded)                     # [B, T, d_model]
        return latent
