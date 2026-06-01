"""FSD3D — Flight Spatial Decoder 3D.

A modular framework for comparing Conditional Flow Matching (CFM)
against Autoregressive MSE on trajectory generation tasks.

Architecture sections:
    §1 encoder/     — Pilot Space: ViT encoder → 3D spatial latent tokens
    §2 conditioner/  — Conditioning: telemetry + A* guidance → conditioning vector
    §3 conditioner/  — Context Normalization: merge + normalize to 32 tokens
    §4 decoder/      — Latent & Flight Generation: FSD3DTransformerDecoder
    §5 decoder/      — Action Loop: ActionProjection + ActionHead + AutoregressiveWrapper
"""

from fsd3d.constants import (
    HORIZON,
    ACTION_DIM,
    CONTEXT_TOKENS,
    D_MODEL,
    NUM_EXPERTS,
    TRAJECTORY_SCALE,
)
from fsd3d.encoder.vit_encoder import ViTEncoder
from fsd3d.data_bridge.visual_adapter import VisualAdapter, LinearVisualAdapter
from fsd3d.data_bridge.context_normalizer import ContextNormalizer
from fsd3d.conditioner.telemetry_encoder import TelemetryEncoder
from fsd3d.conditioner.path_encoder import PathEncoder
from fsd3d.conditioner.conditioner import Conditioner
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_projection import ActionProjection
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
    # §1 Pilot Space
    "ViTEncoder",
    # §3 Data Bridge
    "VisualAdapter",
    "LinearVisualAdapter",
    "ContextNormalizer",
    # §2 Conditioning
    "TelemetryEncoder",
    "PathEncoder",
    "Conditioner",
    # §4 + §5
    "FSD3DTransformerDecoder",
    "ActionProjection",
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
