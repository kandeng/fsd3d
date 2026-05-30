"""Shared constants for the FSD3D framework.

These constants are used across multiple sub-packages (decoder, plugin, etc.)
and are defined here to avoid circular imports.
"""

# Trajectory dimensions
HORIZON = 16           # Number of timesteps in a trajectory
ACTION_DIM = 2         # [x, y] spatial positions

# Model dimensions
CONTEXT_TOKENS = 32    # Number of context tokens (§1 + §2 combined)
D_MODEL = 128         # Transformer hidden dimension
NUM_EXPERTS = 2       # Number of expert trajectories (left + right dodge)

# Normalization
TRAJECTORY_SCALE = 10.0  # Real-space y goes up to 10; normalize to [0, 1]
