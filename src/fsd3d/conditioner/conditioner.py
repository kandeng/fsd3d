"""§2 — Conditioner: Telemetry + A* Guidance.

Assembles conditioning tokens from telemetry readings and A* path waypoints.
These tokens are then merged with §1 visual tokens by the §3 ContextNormalizer
to form the context memory bank (K, V) for the §4 decoder's cross-attention.

The conditioner is source-agnostic — the TelemetryEncoder MLP and
PathEncoder transformer process generic scalars and coordinates regardless
of data source.  Domain-specific adaptation (e.g. source ID embedding)
is handled by the §3 ContextNormalizer.

Architecture:
  - TelemetryEncoder: MLP (9 → d_model) — projects scalar readings
  - PathEncoder: per-point Linear + TransformerEncoder (1 layer)
  - Both sub-encoders are imported from their own modules
  - Concatenation of telemetry + path tokens happens here (§2)
  - Source ID embedding, projection, and truncation live in §3 ContextNormalizer
"""

import torch
import torch.nn as nn

from fsd3d.constants import (
    D_MODEL, NHEAD, DIM_FEEDFORWARD,
    TELEMETRY_DIM, PATH_ENCODER_LAYERS,
)
from fsd3d.conditioner.telemetry_encoder import TelemetryEncoder
from fsd3d.conditioner.path_encoder import PathEncoder


class Conditioner(nn.Module):
    """§2 — Conditioning: Telemetry + A* Guidance.

    Assembles conditioning tokens from telemetry readings and A* path
    waypoints.  The output is fed (along with §1 visual tokens) into the
    §3 ContextNormalizer to produce the final context tensor.
    """

    def __init__(
        self,
        telemetry_dim: int = TELEMETRY_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        path_encoder_layers: int = PATH_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
    ):
        super().__init__()

        # Sub-encoders (defined in their own modules for 1:1 diagram mapping)
        self.telemetry_encoder = TelemetryEncoder(telemetry_dim, d_model)
        self.path_encoder = PathEncoder(
            waypoint_dim=3, d_model=d_model, nhead=nhead,
            num_layers=path_encoder_layers,
            dim_feedforward=dim_feedforward,
        )

    def forward(
        self,
        telemetry: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Produce conditioning tokens from telemetry and path data.

        Args:
            telemetry: (B, telemetry_dim) raw telemetry readings.
            waypoints: (B, N_wp, 3) A* waypoint coordinates.

        Returns:
            (B, 1 + N_wp, d_model) conditioning tokens — to be merged with
            §1 visual tokens by the §3 ContextNormalizer.
        """
        # Encode telemetry → (B, 1, d_model)
        telem_token = self.telemetry_encoder(telemetry).unsqueeze(1)

        # Encode path → (B, N_wp, d_model)
        path_tokens = self.path_encoder(waypoints)

        # Concatenate: telemetry + path (§2 outputs only)
        conditioning_tokens = torch.cat([telem_token, path_tokens], dim=1)
        # (B, 1 + N_wp, d_model)

        return conditioning_tokens
