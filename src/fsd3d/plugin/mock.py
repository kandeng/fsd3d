"""Mock plugin for testing — static Xavier context + pillar-dodge trajectories.

This is the default plugin used when no real data source (§1 + §2) is
available.  It produces the same synthetic data as the original dataset.py.
"""

import numpy as np
import torch
from math import pi

from fsd3d.plugin.base import DataSourcePlugin
from fsd3d.constants import (
    HORIZON, ACTION_DIM, CONTEXT_TOKENS, D_MODEL, TRAJECTORY_SCALE,
)

# Pillar obstacle parameters (real-space coordinates for visualization)
PILLAR_CENTER_X = 0.0
PILLAR_CENTER_Y = 5.0
PILLAR_RADIUS = 0.5


class MockPlugin(DataSourcePlugin):
    """Mock data source plugin using synthetic pillar-dodge trajectories.

    Produces static Xavier-initialized context and bimodal Y-shaped
    expert trajectories (left + right dodge around a circular pillar).
    """

    def build_context(self) -> torch.Tensor:
        """Build a static context tensor of shape (1, 32, 128).

        Represents the 3D visual tokens + Telemetry memory bank
        (output of Sections 1 & 2 of the FSD3D architecture).
        Initialised with Xavier-normal values; treated as a detached
        constant during training (no gradients).
        """
        torch.manual_seed(0)
        context = torch.nn.init.xavier_normal_(
            torch.empty(1, CONTEXT_TOKENS, D_MODEL)
        )
        return context.detach()

    def build_target_plans(self) -> torch.Tensor:
        """Build (2, 16, 2) tensor with left and right dodge expert trajectories.

        The expert path goes (in REAL-SPACE coordinates):
        - Steps 0-4:  straight up from (0, 0) to (0, 3.5)
        - Steps 5-10: semicircle dodge around pillar at (0, 5)
        - Steps 11-15: straight up from end of dodge to (0, ~10)

        Returns:
            (2, 16, 2) float32 tensor — index 0 is left dodge, index 1 is right dodge.
            Both are normalized by TRAJECTORY_SCALE.
        """
        z1_left = self._build_target_plan("left")
        z1_right = self._build_target_plan("right")
        return torch.stack([z1_left, z1_right])

    def _build_target_plan(self, direction: str = "left") -> torch.Tensor:
        """Build the (16, 2) Y-shaped Pillar Dodge maneuver target plan z1."""
        z1 = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)

        # Phase 1: straight up (steps 0-4)
        for t in range(5):
            z1[t] = [0.0, t * 3.0 / 4]

        # Phase 2: semicircle dodge (steps 5-10)
        R = 2.0  # dodge radius from pillar center
        n_curve = 6
        sign = -1 if direction == "left" else 1
        for i in range(n_curve):
            theta = -pi / 2 + (i + 1) * pi / (n_curve + 1)
            z1[5 + i] = [sign * R * np.cos(theta), PILLAR_CENTER_Y + R * np.sin(theta)]

        # Phase 3: straight up (steps 11-15)
        y_last = z1[10, 1]
        for i, t in enumerate(range(11, 16)):
            z1[t] = [0.0, y_last + (i + 1) * (10.0 - y_last) / 5]

        # Normalize for stable training
        z1 = z1 / TRAJECTORY_SCALE

        return torch.from_numpy(z1)

    def get_pillar_params(self) -> dict:
        """Return pillar obstacle parameters for collision detection."""
        return {
            "center_x": PILLAR_CENTER_X,
            "center_y": PILLAR_CENTER_Y,
            "radius": PILLAR_RADIUS,
        }
