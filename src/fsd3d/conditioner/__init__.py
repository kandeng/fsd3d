"""Conditioner sub-package — §2 Conditioning.

Produces the conditioning context vector from telemetry and
pathfinding (A*) guidance data, concatenated with visual tokens
from §1 to form the full context memory bank (K, V) for the §3 decoder.
"""

from fsd3d.conditioner.conditioner import Conditioner, TelemetryEncoder, PathEncoder

__all__ = ["Conditioner", "TelemetryEncoder", "PathEncoder"]
