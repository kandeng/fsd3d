"""Visualization sub-package — comparison, training progression, diagram."""

from fsd3d.visualization.comparison import ComparisonVisualizer
from fsd3d.visualization.training_progression import (
    train_cfm_with_snapshots,
    train_ar_with_snapshots,
    plot_training_progression,
)
from fsd3d.visualization.diagram import generate_diagram

__all__ = [
    "ComparisonVisualizer",
    "train_cfm_with_snapshots",
    "train_ar_with_snapshots",
    "plot_training_progression",
    "generate_diagram",
]
