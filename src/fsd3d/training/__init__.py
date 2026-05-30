"""Training sub-package — CFM and AR training loops."""

from fsd3d.training.cfm import train_cfm
from fsd3d.training.autoregressive import train_autoregressive

__all__ = ["train_cfm", "train_autoregressive"]
