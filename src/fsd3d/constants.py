"""Shared constants for the FSD3D framework.

These constants are used across multiple sub-packages (decoder, encoder,
conditioner, plugin, etc.) and are defined here to avoid circular imports.
"""

# Trajectory dimensions
HORIZON = 16           # Number of timesteps in a trajectory
ACTION_DIM = 2         # [x, y] spatial positions

# Model dimensions
CONTEXT_TOKENS = 32    # Number of context tokens (§1 + §2 combined)
D_MODEL = 128         # Transformer hidden dimension
NHEAD = 4             # Number of attention heads
DIM_FEEDFORWARD = 512  # FFN intermediate dimension
NUM_EXPERTS = 2       # Number of expert trajectories (left + right dodge)

# Decoder (§4)
DECODER_LAYERS = 3

# Encoder (§1)
PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 196
NUM_FRAMES_STACK = 4                            # frame stacking for temporal info
ENCODER_LAYERS = 2                              # fewer than decoder (compression task)

# Conditioner (§2)
TELEMETRY_DIM = 9     # x, y, z, vx, vy, vz, roll, pitch, yaw
PATH_ENCODER_LAYERS = 1

# Normalization
TRAJECTORY_SCALE = 10.0  # Real-space y goes up to 10; normalize to [0, 1]
