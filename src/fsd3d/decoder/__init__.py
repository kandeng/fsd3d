"""§4 + §5 — Decoder sub-package: Latent & Flight Generation + Action Loop."""

from fsd3d.constants import (
    HORIZON,
    ACTION_DIM,
    CONTEXT_TOKENS,
    D_MODEL,
    NUM_EXPERTS,
    TRAJECTORY_SCALE,
)
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_projection import ActionProjection
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.autoregressive import AutoregressiveWrapper
from fsd3d.decoder.context import (
    ContextAssembler,
    compute_spatial_trajectory,
    denormalize_trajectory,
)

__all__ = [
    "FSD3DTransformerDecoder",
    "ActionProjection",
    "ActionHead",
    "AutoregressiveWrapper",
    "ContextAssembler",
    "compute_spatial_trajectory",
    "denormalize_trajectory",
    "HORIZON",
    "ACTION_DIM",
    "CONTEXT_TOKENS",
    "D_MODEL",
    "NUM_EXPERTS",
    "TRAJECTORY_SCALE",
]
