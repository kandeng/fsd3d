"""Scene loader — parse 3DGS PLY file into gsplat-ready tensors.

Reads a binary PLY file containing 3D Gaussian Splatting data (positions,
scales, SH colors, opacity, rotations) and converts it to the tensor
format expected by gsplat's rasterization API.
"""

import numpy as np
import torch
from plyfile import PlyData


class SceneLoader:
    """Load a 3DGS PLY scene and convert to gsplat-ready tensors.

    The PLY file is expected to have the standard 3DGS format with
    properties: x/y/z, scale_0/1/2, f_dc_0/1/2, opacity, rot_0/1/2/3.
    """

    def __init__(self, ply_path: str, device: str = "cpu"):
        self.ply_path = ply_path
        self.device = device
        self._loaded = False

        # Scene data (populated by load())
        self.means: torch.Tensor = torch.empty(0)
        self.quats: torch.Tensor = torch.empty(0)
        self.scales: torch.Tensor = torch.empty(0)
        self.opacities: torch.Tensor = torch.empty(0)
        self.colors: torch.Tensor = torch.empty(0)

        # Scene bounding box
        self.bbox_min: np.ndarray = np.zeros(3)
        self.bbox_max: np.ndarray = np.zeros(3)

    def load(self) -> "SceneLoader":
        """Load and parse the PLY file. Returns self for chaining."""
        plydata = PlyData.read(self.ply_path)
        vertex = plydata["vertex"]

        # Positions
        self.means = torch.tensor(
            np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1),
            dtype=torch.float32, device=self.device,
        )

        # Rotations (wxyz convention for gsplat)
        self.quats = torch.tensor(
            np.stack([vertex["rot_0"], vertex["rot_1"],
                       vertex["rot_2"], vertex["rot_3"]], axis=-1),
            dtype=torch.float32, device=self.device,
        )

        # Scales
        self.scales = torch.tensor(
            np.stack([vertex["scale_0"], vertex["scale_1"],
                       vertex["scale_2"]], axis=-1),
            dtype=torch.float32, device=self.device,
        )

        # Opacities (stored as logit; apply sigmoid)
        raw_opacity = torch.tensor(
            np.array(vertex["opacity"]), dtype=torch.float32, device=self.device,
        )
        self.opacities = torch.sigmoid(raw_opacity)

        # Colors (SH degree 0 = DC component)
        self.colors = torch.tensor(
            np.stack([vertex["f_dc_0"], vertex["f_dc_1"],
                       vertex["f_dc_2"]], axis=-1),
            dtype=torch.float32, device=self.device,
        )
        # SH DC to RGB: C_rgb = 0.5 + C_dc * 0.28209479177387814
        SH_C0 = 0.28209479177387814
        self.colors = 0.5 + self.colors * SH_C0

        # Compute bounding box
        means_np = self.means.cpu().numpy()
        self.bbox_min = means_np.min(axis=0)
        self.bbox_max = means_np.max(axis=0)

        self._loaded = True
        print(f"Loaded {self.means.shape[0]} Gaussians from {self.ply_path}")
        print(f"  Bounding box: {self.bbox_min} → {self.bbox_max}")
        return self

    def scene_info(self) -> dict:
        """Return scene metadata."""
        assert self._loaded, "Call load() first"
        return {
            "num_gaussians": self.means.shape[0],
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "bbox_size": (self.bbox_max - self.bbox_min).tolist(),
        }
