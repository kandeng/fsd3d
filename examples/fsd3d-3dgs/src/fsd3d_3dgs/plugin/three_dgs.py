"""3DGS DataSourcePlugin — orchestrates the full §1 + §2 + §3 data pipeline.

Loads a 3DGS scene, builds a voxel map, plans an A* path, fakes telemetry,
renders frames along the path, encodes them through the §1 ViT encoder,
§2 conditioner, and §3 ContextNormalizer, and produces the context
tensor + target plans expected by the §4 decoder.

This is the concrete plugin that bridges domain-specific 3DGS data into
the source-agnostic fsd3d framework.
"""

import numpy as np
import torch
from typing import Optional

from fsd3d.plugin.base import DataSourcePlugin
from fsd3d.constants import (
    HORIZON, ACTION_DIM, CONTEXT_TOKENS, D_MODEL,
    TRAJECTORY_SCALE, NUM_FRAMES_STACK, IMAGE_SIZE,
)
from fsd3d.encoder.vit_encoder import ViTEncoder
from fsd3d.data_bridge.visual_adapter import LinearVisualAdapter
from fsd3d.conditioner.conditioner import Conditioner
from fsd3d.data_bridge.context_normalizer import ContextNormalizer

from fsd3d_3dgs.scene.loader import SceneLoader
from fsd3d_3dgs.scene.renderer import SceneRenderer
from fsd3d_3dgs.planner.voxel_map import VoxelMap
from fsd3d_3dgs.planner.astar import AStarPlanner
from fsd3d_3dgs.telemetry.faker import TelemetryFaker


class ThreeDGSPlugin(DataSourcePlugin):
    """3DGS data source plugin — full pipeline from PLY file to context tensor.

    Orchestrates:
      1. Load 3DGS scene (PLY → gsplat tensors)
      2. Build voxel map (occupancy grid for A*)
      3. Plan A* path (takeoff → cruise → landing)
      4. Fake telemetry along path (GPS/IMU/baro/orientation noise)
      5. Render frames along path (gsplat rasterization)
      6. §1 ViT encoder: frames → visual tokens
      7. **`LinearVisualAdapter`** — adapts visual tokens for domain shift
      8. **`Conditioner`** (§2) — telemetry + waypoints → conditioning tokens
      9. **`ContextNormalizer`** (§3) — visual + conditioning → context (1, 32, 128)

    The plugin also produces expert target plans from the A* path for
    training the §4 decoder.
    """

    def __init__(
        self,
        ply_path: str,
        start: np.ndarray,
        goal: np.ndarray,
        cruise_altitude: Optional[float] = None,
        max_altitude: float = 30.0,
        voxel_resolution: float = 0.5,
        dodge_offset: float = 2.0,
        device: str = "cpu",
    ):
        self.ply_path = ply_path
        self.start = start
        self.goal = goal
        self.cruise_altitude = cruise_altitude
        self.max_altitude = max_altitude
        self.voxel_resolution = voxel_resolution
        self.dodge_offset = dodge_offset
        self.device = device

        # Pipeline components (built lazily)
        self._scene: Optional[SceneLoader] = None
        self._voxel_map: Optional[VoxelMap] = None
        self._path: Optional[list] = None
        self._telemetry: Optional[np.ndarray] = None
        self._context: Optional[torch.Tensor] = None

        # Neural network modules (shared parameters for all calls)
        self._encoder = ViTEncoder().to(device)
        self._adapter = LinearVisualAdapter().to(device)
        self._conditioner = Conditioner().to(device)
        self._normalizer = ContextNormalizer().to(device)

        # Set to eval mode — no dropout, no gradient
        self._encoder.eval()
        self._adapter.eval()
        self._conditioner.eval()
        self._normalizer.eval()

    def _ensure_loaded(self):
        """Lazy-load the scene and run the full pipeline."""
        if self._scene is not None:
            return

        # 1. Load scene
        self._scene = SceneLoader(self.ply_path, device=self.device).load()

        # 2. Build voxel map
        self._voxel_map = VoxelMap(
            self._scene, resolution=self.voxel_resolution
        ).build()

        # 3. Plan A* path
        planner = AStarPlanner(self._voxel_map, max_altitude=self.max_altitude)
        self._path = planner.plan(
            self.start, self.goal, cruise_altitude=self.cruise_altitude,
        )

        # 4. Fake telemetry
        waypoints_3d = np.array([wp for wp in self._path], dtype=np.float32)
        faker = TelemetryFaker(seed=42)
        self._telemetry = faker.fake(waypoints_3d)

    def build_context(self) -> torch.Tensor:
        """Build context tensor of shape (1, 32, 128).

        Full pipeline: scene → voxels → path → telemetry → render → encode → context.
        """
        self._ensure_loaded()

        if self._context is not None:
            return self._context

        # 5. Render frames along path
        renderer = SceneRenderer(self._scene)
        waypoints_3d = np.array([wp for wp in self._path], dtype=np.float32)
        frames = renderer.render_along_path(waypoints_3d)  # (N, H, W, 3)

        # 6. Stack frames for temporal input: take groups of NUM_FRAMES_STACK
        #    For simplicity, use the last NUM_FRAMES_STACK frames repeated if needed
        N = frames.shape[0]
        if N >= NUM_FRAMES_STACK:
            # Use evenly spaced frames
            indices = np.linspace(0, N - 1, NUM_FRAMES_STACK, dtype=int)
            stacked = frames[indices]  # (4, H, W, 3)
        else:
            # Pad by repeating last frame
            stacked = torch.zeros(NUM_FRAMES_STACK, *frames.shape[1:])
            stacked[:N] = frames
            stacked[N:] = frames[-1:]

        # Convert to (1, C*N_stack, H, W) format for ViTEncoder
        # frames: (N_stack, H, W, 3) → (3, N_stack, H, W) → (3*N_stack, H, W)
        images = stacked.permute(3, 0, 1, 2).reshape(3 * NUM_FRAMES_STACK, frames.shape[1], frames.shape[2])
        images = images.unsqueeze(0).to(self.device)  # (1, C*N_stack, H, W)

        # Resize to IMAGE_SIZE if needed
        if images.shape[-1] != IMAGE_SIZE or images.shape[-2] != IMAGE_SIZE:
            images = torch.nn.functional.interpolate(
                images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False,
            )

        # 7. §1 ViT Encoder: images → visual tokens
        with torch.no_grad():
            visual_tokens = self._encoder(images)  # (1, 196, 128)
            adapted_tokens = self._adapter(visual_tokens)  # (1, 196, 128)

            # 8. §2 Conditioner: telemetry + waypoints → conditioning tokens
            telemetry_t = torch.tensor(
                self._telemetry[-1:], dtype=torch.float32, device=self.device
            )  # (1, 9) — use last telemetry reading
            # If only one reading, expand; take mean of recent readings
            # Use the last timestep's telemetry as the current state
            telemetry_t = torch.tensor(
                self._telemetry[-1:, :], dtype=torch.float32, device=self.device
            )  # (1, 9)

            # Downsample waypoints for conditioner
            from fsd3d_3dgs.planner.astar import AStarPlanner
            downsampled = AStarPlanner.downsample_waypoints(self._path, target_count=16)
            waypoints_t = torch.tensor(
                downsampled, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, 16, 3)

            conditioning_tokens = self._conditioner(telemetry_t, waypoints_t)
            # (1, 1 + 16, 128) = (1, 17, 128)

            # 9. §3 ContextNormalizer: visual + conditioning → context
            context = self._normalizer(adapted_tokens, conditioning_tokens)
            # (1, context_tokens, d_model) = (1, 32, 128)

        self._context = context.detach()
        return self._context

    def build_target_plans(self) -> torch.Tensor:
        """Build expert trajectory tensor from the A* path.

        Generates left-dodge and right-dodge variants by shifting waypoints
        perpendicular to the path direction. Falls back to the raw path
        projected to 2D if the path is too short.

        Returns:
            (N, 16, 2) float32 tensor — N expert trajectories,
            normalized by TRAJECTORY_SCALE.
        """
        self._ensure_loaded()

        # Downsample path to HORIZON waypoints
        downsampled = AStarPlanner.downsample_waypoints(self._path, target_count=HORIZON)
        # downsampled: (16, 3)

        # Project to 2D (x, y) for the decoder
        path_2d = downsampled[:, :2]  # (16, 2)

        # Compute perpendicular direction for dodge variants
        if path_2d.shape[0] >= 2:
            direction = path_2d[-1] - path_2d[0]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                perp = np.array([-direction[1], direction[0]]) / norm
            else:
                perp = np.array([1.0, 0.0])
        else:
            perp = np.array([1.0, 0.0])

        # Left and right dodge: shift waypoints perpendicular to path
        offset = self.dodge_offset
        left_2d = path_2d + perp * offset
        right_2d = path_2d - perp * offset

        # Normalize and stack
        left_norm = torch.from_numpy(left_2d.astype(np.float32)) / TRAJECTORY_SCALE
        right_norm = torch.from_numpy(right_2d.astype(np.float32)) / TRAJECTORY_SCALE

        return torch.stack([left_norm, right_norm])  # (2, 16, 2)

    def get_pillar_params(self) -> dict:
        """Analyze voxel map for the largest obstacle cluster.

        Returns:
            dict with keys: center_x, center_y, radius
        """
        self._ensure_loaded()

        if self._voxel_map is None:
            return {"center_x": 0.0, "center_y": 0.0, "radius": 0.5}

        # Find the centroid of all occupied voxels as a simple obstacle estimate
        grid = self._voxel_map.grid
        occupied_indices = np.argwhere(grid)  # (K, 3) grid indices

        if len(occupied_indices) == 0:
            return {"center_x": 0.0, "center_y": 0.0, "radius": 0.5}

        # Convert grid indices to world coordinates
        world_positions = np.array([
            self._voxel_map.grid_to_world(idx) for idx in occupied_indices
        ])

        # Use the centroid of occupied voxels as the "pillar center"
        centroid = world_positions.mean(axis=0)
        # Estimate radius as the max distance from centroid in XY
        dists = np.sqrt(np.sum((world_positions[:, :2] - centroid[:2]) ** 2, axis=1))
        radius = float(dists.max())

        return {
            "center_x": float(centroid[0]),
            "center_y": float(centroid[1]),
            "radius": radius,
        }
