"""Scene renderer — render RGB frames from 3DGS scenes via gsplat.

Uses gsplat's rasterization API to render frames from arbitrary camera
poses.  Supports batch rendering of full trajectories.
"""

import torch
import numpy as np
from typing import Optional, Tuple

from fsd3d_3dgs.scene.loader import SceneLoader


class SceneRenderer:
    """Render RGB frames from a loaded 3DGS scene using gsplat."""

    def __init__(self, scene: SceneLoader, width: int = 640, height: int = 480):
        self.scene = scene
        self.width = width
        self.height = height

    def look_at(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = np.array([0, 0, 1]),
    ) -> torch.Tensor:
        """Build a world-to-camera transformation matrix.

        Args:
            eye:    (3,) camera position in world coordinates.
            target: (3,) point the camera looks at.
            up:     (3,) up direction (default: Z-up).

        Returns:
            (4, 4) world-to-camera matrix.
        """
        eye = np.asarray(eye, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)

        # Forward (Z-axis of camera)
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Right (X-axis of camera)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Recompute up to ensure orthogonality
        up = np.cross(right, forward)

        # World-to-camera matrix (OpenGL convention: look along -Z)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[0, :3] = right
        viewmat[1, :3] = up
        viewmat[2, :3] = -forward  # negate for OpenGL convention
        viewmat[:3, 3] = -np.array([
            np.dot(right, eye),
            np.dot(up, eye),
            np.dot(-forward, eye),
        ])

        return torch.from_numpy(viewmat)

    def default_intrinsics(self, fov_deg: float = 60.0) -> torch.Tensor:
        """Build a default camera intrinsic matrix.

        Args:
            fov_deg: Horizontal field of view in degrees.

        Returns:
            (3, 3) intrinsic matrix.
        """
        fx = (self.width / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        fy = fx  # square pixels
        cx = self.width / 2.0
        cy = self.height / 2.0
        K = torch.tensor([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=torch.float32)
        return K

    def render(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
    ) -> torch.Tensor:
        """Render frames from camera poses.

        Args:
            viewmats: (C, 4, 4) world-to-camera matrices.
            Ks:       (C, 3, 3) camera intrinsic matrices.

        Returns:
            (C, H, W, 3) RGB images as float32 tensors in [0, 1].
        """
        from gsplat.rasterization import rasterization

        assert self.scene._loaded, "Scene must be loaded before rendering"

        # Add batch dimension if needed: gsplat expects (1, C, 4, 4) and (1, C, 3, 3)
        if viewmats.dim() == 3:
            viewmats = viewmats.unsqueeze(0)
        if Ks.dim() == 2:
            Ks = Ks.unsqueeze(0)

        render_colors, _, _ = rasterization(
            means=self.scene.means,
            quats=self.scene.quats,
            scales=self.scene.scales,
            opacities=self.scene.opacities,
            colors=self.scene.colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.width,
            height=self.height,
            render_mode="RGB",
        )

        # render_colors: (1, C, H, W, 3) → (C, H, W, 3)
        images = render_colors.squeeze(0)
        return images.clamp(0, 1)

    def render_along_path(
        self,
        positions: np.ndarray,
        look_direction: str = "forward",
        fov_deg: float = 60.0,
    ) -> torch.Tensor:
        """Render frames along a camera trajectory.

        Args:
            positions:     (N, 3) camera positions in world coordinates.
            look_direction: "forward" (look along velocity) or "down" (look at ground).
            fov_deg:       Horizontal field of view.

        Returns:
            (N, H, W, 3) RGB images.
        """
        N = len(positions)
        K = self.default_intrinsics(fov_deg)
        Ks = K.unsqueeze(0).expand(N, -1, -1)  # (N, 3, 3)

        viewmats = []
        for i in range(N):
            eye = positions[i]
            if look_direction == "forward" and i < N - 1:
                target = positions[i + 1]
            elif look_direction == "down":
                target = eye.copy()
                target[2] = 0  # look at ground
            else:
                target = eye + np.array([1, 0, 0])  # default forward

            vm = self.look_at(eye, target)
            viewmats.append(vm)

        viewmats = torch.stack(viewmats)  # (N, 4, 4)
        return self.render(viewmats, Ks)
