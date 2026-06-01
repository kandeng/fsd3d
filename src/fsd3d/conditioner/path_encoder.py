"""§2 — PathEncoder: encodes A* waypoint sequences into d_model tokens.

Waypoints form a spatial sequence with meaningful relationships
(path continuity, direction changes), so a small transformer is
appropriate.

Architecture diagram mapping:
  A* Guidance ──► PathEncoder ──► (B, N_wp, d_model) path tokens
"""

import torch
import torch.nn as nn

from fsd3d.constants import D_MODEL, NHEAD, DIM_FEEDFORWARD, PATH_ENCODER_LAYERS


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
