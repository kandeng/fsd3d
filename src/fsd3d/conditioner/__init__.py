"""Conditioner sub-package — §2 Conditioning.

Produces the conditioning context vector from telemetry and
pathfinding (A*) guidance data.

Context normalization (§3 Data Bridge) has been moved to
``fsd3d.data_bridge.context_normalizer``.
"""

from fsd3d.conditioner.telemetry_encoder import TelemetryEncoder
from fsd3d.conditioner.path_encoder import PathEncoder
from fsd3d.conditioner.conditioner import Conditioner

# Backward-compatible re-export (class now lives in fsd3d.data_bridge)
from fsd3d.data_bridge.context_normalizer import ContextNormalizer

__all__ = ["TelemetryEncoder", "PathEncoder", "Conditioner", "ContextNormalizer"]
