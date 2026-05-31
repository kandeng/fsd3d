"""Voxel map — convert 3DGS scene into 3D occupancy grid for A* planning.

This is NOT part of the encoder pipeline.  The encoder learns spatial
structure implicitly through self-attention on image patches.  The
voxel map is exclusively a utility for the A* planner to navigate
the 3D scene.
"""

import numpy as np
from typing import Tuple

from fsd3d_3dgs.scene.loader import SceneLoader


class VoxelMap:
    """3D occupancy grid built from a 3DGS scene's Gaussian centers.

    Voxelize the Gaussian means into a regular 3D grid.  Mark a voxel
    as occupied if any Gaussian center falls within it AND its opacity
    exceeds a threshold.  Optionally dilate occupied voxels by a safety
    margin (drone radius).
    """

    def __init__(
        self,
        scene: SceneLoader,
        resolution: float = 0.5,
        opacity_threshold: float = 0.5,
        drone_radius: float = 0.3,
    ):
        self.scene = scene
        self.resolution = resolution
        self.opacity_threshold = opacity_threshold
        self.drone_radius = drone_radius

        self.grid: np.ndarray = np.empty(0, dtype=bool)
        self.origin: np.ndarray = np.zeros(3)
        self.grid_shape: Tuple[int, int, int] = (0, 0, 0)

    def build(self) -> "VoxelMap":
        """Build the occupancy grid from the loaded scene. Returns self."""
        assert self.scene._loaded, "Scene must be loaded first"

        means = self.scene.means.cpu().numpy()
        opacities = self.scene.opacities.cpu().numpy()

        # Filter by opacity
        mask = opacities > self.opacity_threshold
        means_filtered = means[mask]

        # Compute grid bounds with margin
        margin = self.drone_radius * 2
        self.origin = self.scene.bbox_min - margin
        bbox_max = self.scene.bbox_max + margin

        # Grid dimensions
        grid_size = np.ceil((bbox_max - self.origin) / self.resolution).astype(int)
        self.grid_shape = tuple(grid_size)
        self.grid = np.zeros(self.grid_shape, dtype=bool)

        # Mark occupied voxels
        indices = ((means_filtered - self.origin) / self.resolution).astype(int)
        # Clip to valid range
        valid = np.all((indices >= 0) & (indices < np.array(self.grid_shape)), axis=1)
        indices = indices[valid]
        self.grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # Dilate by drone radius
        if self.drone_radius > 0:
            self._dilate()

        occupied = self.grid.sum()
        total = self.grid.size
        print(f"Voxel map: {self.grid_shape}, {occupied}/{total} occupied "
              f"({100*occupied/total:.2f}%)")
        return self

    def _dilate(self):
        """Dilate occupied voxels by drone_radius to create a safety margin."""
        dilate_radius = max(1, int(np.ceil(self.drone_radius / self.resolution)))
        padded = np.pad(self.grid, dilate_radius, mode='constant', constant_values=False)

        result = np.zeros_like(padded)
        for dx in range(-dilate_radius, dilate_radius + 1):
            for dy in range(-dilate_radius, dilate_radius + 1):
                for dz in range(-dilate_radius, dilate_radius + 1):
                    if dx*dx + dy*dy + dz*dz <= dilate_radius*dilate_radius:
                        shifted = np.roll(np.roll(np.roll(padded, dx, 0), dy, 1), dz, 2)
                        result |= shifted

        # Remove padding
        self.grid = result[
            dilate_radius:-dilate_radius,
            dilate_radius:-dilate_radius,
            dilate_radius:-dilate_radius,
        ]

    def is_occupied(self, world_pos: np.ndarray) -> bool:
        """Check if a world position is in an occupied voxel."""
        idx = ((world_pos - self.origin) / self.resolution).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(self.grid_shape)):
            return True  # out of bounds = occupied (blocked)
        return self.grid[idx[0], idx[1], idx[2]]

    def world_to_grid(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid indices."""
        return ((world_pos - self.origin) / self.resolution).astype(int)

    def grid_to_world(self, grid_idx: np.ndarray) -> np.ndarray:
        """Convert grid indices to world coordinates (voxel center)."""
        return self.origin + (grid_idx + 0.5) * self.resolution
