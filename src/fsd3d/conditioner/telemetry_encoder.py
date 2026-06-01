"""§2 — TelemetryEncoder: encodes telemetry scalars into d_model tokens.

Telemetry is a handful of scalars (position, velocity, orientation)
with no meaningful sequence structure — an MLP suffices.

Architecture diagram mapping:
  Telemetry Data ──► TelemetryEncoder ──► (B, d_model) token
"""

import torch
import torch.nn as nn

from fsd3d.constants import TELEMETRY_DIM, D_MODEL


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
