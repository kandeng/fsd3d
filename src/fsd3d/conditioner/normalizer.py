"""Backward-compatible shim — ContextNormalizer moved to fsd3d.data_bridge.context_normalizer.

New code should import from ``fsd3d.data_bridge.context_normalizer`` directly.
"""

from fsd3d.data_bridge.context_normalizer import ContextNormalizer

__all__ = ["ContextNormalizer"]
