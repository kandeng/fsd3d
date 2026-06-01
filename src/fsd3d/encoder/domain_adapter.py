"""Backward-compatible shim — classes moved to fsd3d.data_bridge.visual_adapter.

This module re-exports ``VisualAdapter`` as ``DomainAdapter`` and
``LinearVisualAdapter`` as ``LinearDomainAdapter`` for backward
compatibility.  New code should import from ``fsd3d.data_bridge`` directly.
"""

from fsd3d.data_bridge.visual_adapter import VisualAdapter as DomainAdapter
from fsd3d.data_bridge.visual_adapter import LinearVisualAdapter as LinearDomainAdapter

__all__ = ["DomainAdapter", "LinearDomainAdapter"]
