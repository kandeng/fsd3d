"""Conditioner sub-package — §2 Conditioning.

Produces the conditioning context vector from telemetry and
pathfinding (A*) guidance data, concatenated with visual tokens
from §1 to form the full context for the decoder.

Status: Stub — to be implemented when §2 is developed.
"""

from fsd3d.conditioner.conditioner import Conditioner

__all__ = ["Conditioner"]
