"""Context assembly and trajectory utilities.

Assembles the context tensor [B, 32, 128] from §1 (encoder) + §2 (conditioner)
+ §3 (normalizer) outputs.  When no real encoder/conditioner is available,
falls back to a static Xavier-initialized context (via MockPlugin).

Also contains shared constants and trajectory denormalization helpers.
"""

import numpy as np
import torch
from math import pi
from typing import Optional

from fsd3d.constants import (
    HORIZON, ACTION_DIM, CONTEXT_TOKENS, D_MODEL, NUM_EXPERTS, TRAJECTORY_SCALE,
)
from fsd3d.plugin.base import DataSourcePlugin
from fsd3d.plugin.mock import MockPlugin


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
class ContextAssembler:
    """Assembles context tensor from §1 + §2 outputs.

    When a plugin is provided, delegates context generation to the plugin.
    Otherwise falls back to a static Xavier-initialized context.
    """

    def __init__(self, plugin: Optional[DataSourcePlugin] = None):
        self.plugin = plugin or MockPlugin()

    def assemble(self) -> torch.Tensor:
        """Assemble context tensor of shape (1, CONTEXT_TOKENS, D_MODEL).

        Returns:
            (1, 32, 128) context tensor (no grad)
        """
        return self.plugin.build_context()

    def build_target_plans(self) -> torch.Tensor:
        """Build (2, 16, 2) tensor with left and right dodge expert trajectories.

        Returns:
            (2, 16, 2) float32 tensor — index 0 is left dodge, index 1 is right dodge.
            Both are normalized by TRAJECTORY_SCALE.
        """
        return self.plugin.build_target_plans()

    def build_synthetic_data(self):
        """Return (z1, context) for training and inference.

        Returns:
            z1:      (2, 16, 2) target plan tensor — left and right dodge (no grad)
            context: (1, 32, 128) context tensor (no grad)
        """
        z1 = self.build_target_plans()
        context = self.assemble()
        return z1, context


# ---------------------------------------------------------------------------
# Spatial trajectory helpers (for visualisation)
# ---------------------------------------------------------------------------
def compute_spatial_trajectory(z1: np.ndarray, dt_sim: float = 0.1) -> np.ndarray:
    """Map a (T, 2) position sequence to (T, 2) spatial positions.

    Since z1 now directly represents 2D spatial positions [x, y],
    this is an identity function (returns a copy).

    Args:
        z1:     (T, 2) array — [x, y] positions (may be normalized)
        dt_sim: unused (kept for backward compatibility)

    Returns:
        (T, 2) array — [x, y] positions (copy of input)
    """
    return z1.copy()


def denormalize_trajectory(positions: np.ndarray) -> np.ndarray:
    """Convert normalized positions back to real-space coordinates.

    Args:
        positions: (T, 2) or (N, T, 2) array in normalized space

    Returns:
        Same shape array in real-space coordinates (multiplied by TRAJECTORY_SCALE)
    """
    return positions * TRAJECTORY_SCALE
