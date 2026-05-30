"""FSD3D — Flight Spatial Decoder 3D.

A modular framework for comparing Conditional Flow Matching (CFM)
against Autoregressive MSE on trajectory generation tasks.

Architecture sections:
    §1 encoder/     — Pilot Space: ViT encoder → 3D spatial latent tokens
    §2 conditioner/  — Conditioning: telemetry + A* guidance → conditioning vector
    §3 decoder/      — Latent & Flight Generation: FSD3DTransformerDecoder
    §4 decoder/      — Action Loop: ActionHead + AutoregressiveWrapper
"""

from fsd3d.constants import (
    HORIZON,
    ACTION_DIM,
    CONTEXT_TOKENS,
    D_MODEL,
    NUM_EXPERTS,
    TRAJECTORY_SCALE,
)
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.autoregressive import AutoregressiveWrapper
from fsd3d.decoder.context import (
    ContextAssembler,
    compute_spatial_trajectory,
    denormalize_trajectory,
)
from fsd3d.inference.cfm import infer_cfm_euler
from fsd3d.inference.autoregressive import infer_autoregressive
from fsd3d.training.cfm import train_cfm
from fsd3d.training.autoregressive import train_autoregressive
from fsd3d.plugin.base import DataSourcePlugin
from fsd3d.plugin.mock import MockPlugin, PILLAR_CENTER_X, PILLAR_CENTER_Y, PILLAR_RADIUS

__version__ = "0.1.0"

__all__ = [
    # §3 + §4
    "FSD3DTransformerDecoder",
    "ActionHead",
    "AutoregressiveWrapper",
    # Context
    "ContextAssembler",
    "compute_spatial_trajectory",
    "denormalize_trajectory",
    "HORIZON",
    "ACTION_DIM",
    "CONTEXT_TOKENS",
    "D_MODEL",
    "NUM_EXPERTS",
    "TRAJECTORY_SCALE",
    # Inference
    "infer_cfm_euler",
    "infer_autoregressive",
    # Training
    "train_cfm",
    "train_autoregressive",
    # Plugin
    "DataSourcePlugin",
    "MockPlugin",
    "PILLAR_CENTER_X",
    "PILLAR_CENTER_Y",
    "PILLAR_RADIUS",
]
