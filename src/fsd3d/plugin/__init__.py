"""Plugin system — abstract data source interface."""

from fsd3d.plugin.base import DataSourcePlugin
from fsd3d.plugin.mock import MockPlugin

__all__ = ["DataSourcePlugin", "MockPlugin"]
