"""§2 — Conditioner: Telemetry + A* Guidance.

Produces conditioning tokens from telemetry readings and A* path waypoints.
These tokens, concatenated with §1 visual tokens, form the context memory
bank (K, V) for the §3 decoder's cross-attention.

The conditioner is source-agnostic — the TelemetryEncoder MLP and
PathEncoder transformer process generic scalars and coordinates regardless
of data source.  Domain-specific adaptation (e.g. source ID embedding)
is configured per deployment.

Architecture:
  - TelemetryEncoder: MLP (9 → d_model) — projects scalar readings
  - PathEncoder: per-point Linear + TransformerEncoder (1 layer)
  - Concatenation: visual tokens + telemetry tokens + path tokens
  - Projection: Linear(d_model, d_model)
  - Source ID embedding: learned vector added to all tokens
"""

import torch
import torch.nn as nn

from fsd3d.constants import (
    D_MODEL, NHEAD, DIM_FEEDFORWARD,
    TELEMETRY_DIM, PATH_ENCODER_LAYERS, CONTEXT_TOKENS,
)


class TelemetryEncoder(nn.Module):
    """Encode telemetry scalars into d_model-dimensional tokens.

    Telemetry is a handful of scalars (position, velocity, orientation)
    with no meaningful sequence structure — an MLP suffices.
    """

    def __init__(self, telemetry_dim: int = TELEMETRY_DIM, d_model: int = D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(telemetry_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, telemetry: torch.Tensor) -> torch.Tensor:
        """Encode telemetry readings.

        Args:
            telemetry: (B, telemetry_dim) raw telemetry readings.

        Returns:
            (B, d_model) encoded telemetry token.
        """
        return self.mlp(telemetry)


class PathEncoder(nn.Module):
    """Encode A* waypoint sequence into d_model-dimensional tokens.

    Waypoints form a spatial sequence with meaningful relationships
    (path continuity, direction changes), so a small transformer is
    appropriate.
    """

    def __init__(
        self,
        waypoint_dim: int = 3,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = PATH_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        max_waypoints: int = 64,
    ):
        super().__init__()
        self.point_projection = nn.Linear(waypoint_dim, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_waypoints, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Encode waypoint sequence.

        Args:
            waypoints: (B, N, waypoint_dim) A* waypoint coordinates.

        Returns:
            (B, N, d_model) encoded path tokens.
        """
        N = waypoints.size(1)
        tokens = self.point_projection(waypoints)
        tokens = tokens + self.position_embedding[:, :N, :]
        encoded = self.transformer(tokens)
        return self.norm(encoded)


class Conditioner(nn.Module):
    """§2 — Conditioning: Telemetry + A* Guidance.

    Assembles conditioning tokens from telemetry readings and A* path
    waypoints.  When combined with §1 visual tokens (via concatenation),
    these form the context memory bank (K, V) for the §3 decoder.
    """

    def __init__(
        self,
        telemetry_dim: int = TELEMETRY_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        path_encoder_layers: int = PATH_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        context_tokens: int = CONTEXT_TOKENS,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_tokens = context_tokens

        # Sub-encoders
        self.telemetry_encoder = TelemetryEncoder(telemetry_dim, d_model)
        self.path_encoder = PathEncoder(
            waypoint_dim=3, d_model=d_model, nhead=nhead,
            num_layers=path_encoder_layers,
            dim_feedforward=dim_feedforward,
        )

        # Shared projection after concatenation
        self.projection = nn.Linear(d_model, d_model)

        # Source ID embedding — distinguishes data sources in multi-domain training
        self.source_id_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self,
        telemetry: torch.Tensor,
        waypoints: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Produce full context tensor from telemetry, path, and visual tokens.

        Args:
            telemetry:     (B, telemetry_dim) raw telemetry readings.
            waypoints:     (B, N_wp, 3) A* waypoint coordinates.
            visual_tokens: (B, V_tokens, d_model) from §1 ViT encoder.

        Returns:
            (B, context_tokens, d_model) context tensor — compatible with §3 decoder.
        """
        B = telemetry.size(0)

        # Encode telemetry → (B, 1, d_model)
        telem_token = self.telemetry_encoder(telemetry).unsqueeze(1)

        # Encode path → (B, N_wp, d_model)
        path_tokens = self.path_encoder(waypoints)

        # Concatenate: visual + telemetry + path
        all_tokens = torch.cat([visual_tokens, telem_token, path_tokens], dim=1)
        # (B, V + 1 + N_wp, d_model)

        # Add source ID embedding to all tokens
        all_tokens = all_tokens + self.source_id_embedding

        # Project
        all_tokens = self.projection(all_tokens)

        # Truncate or pad to fixed context_tokens length
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
