"""Inference sub-package — CFM Euler ODE solver and AR roll-out."""

from fsd3d.inference.cfm import infer_cfm_euler
from fsd3d.inference.autoregressive import infer_autoregressive

__all__ = ["infer_cfm_euler", "infer_autoregressive"]
