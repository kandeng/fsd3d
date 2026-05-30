"""Abstract data source plugin interface.

Each data source (3DGS, Google Earth, real drone) implements this interface
to provide context tensors, target plans, and scenario-specific data.
"""

from abc import ABC, abstractmethod
import torch


class DataSourcePlugin(ABC):
    """Abstract base class for data source plugins.

    Each data source implements this interface to provide:
      - Context tensor [1, 32, 128] (assembled from §1 + §2)
      - Target plans (expert trajectories)
      - Scenario geometry (for visualization and collision detection)
    """

    @abstractmethod
    def build_context(self) -> torch.Tensor:
        """Build context tensor of shape (1, 32, 128).

        This assembles §1 (vision tokens) + §2 (conditioning tokens)
        into the memory bank for §3's cross-attention.

        Returns:
            (1, 32, 128) context tensor (no grad)
        """

    @abstractmethod
    def build_target_plans(self) -> torch.Tensor:
        """Build expert trajectory tensor.

        Returns:
            (N, 16, 2) float32 tensor — N expert trajectories,
            normalized by TRAJECTORY_SCALE.
        """

    @abstractmethod
    def get_pillar_params(self) -> dict:
        """Return pillar obstacle parameters for collision detection.

        Returns:
            dict with keys: center_x, center_y, radius
        """
