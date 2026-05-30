# =====================================================================
# $ pwd
#   /home/robot/fsd3d/src/fsd3d/visualization
# $ MANEUVER=0 python cfm_diffusion.py
# $ MANEUVER=1 python cfm_diffusion.py
# 
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =====================================================================
# CONFIGURATION & PARAMETERS
# =====================================================================
GRID_SIZE = 20
NUM_DOTS = 40
NUM_STEPS = 60  # Number of animation frames
DIFFUSION_NOISE_SCALE = 0.95  # Strong omnidirectional Brownian chaos

# Read from system environment variables (Defaults to "0" for the circle test)
env_maneuver = os.environ.get("MANEUVER", "0")
MANEUVER = env_maneuver.strip().replace('"', '').replace("'", "")

print(f"--- FSD3D Latent Simulation Pipeline ---")
print(f"Parsed Target Pattern Configuration: '{MANEUVER}'")
print(f"----------------------------------------")

# Fix random seed for identical particle initialization comparison
np.random.seed(42)

# Start State (tau = 0): Same random sand distribution for both pipelines
z_0 = np.random.uniform(1.0, GRID_SIZE - 1.0, size=(NUM_DOTS, 2))

# Target matrices
z_1_cfm = np.zeros_like(z_0)
z_1_diff = np.zeros_like(z_0)

CENTER_X, CENTER_Y = 10.0, 10.0
base_radius = 6.0

# =====================================================================
# TARGET DEFINITION WITH TOPOLOGICAL PRINCIPLES
# =====================================================================
for i in range(NUM_DOTS):
    start_x, start_y = z_0[i, 0], z_0[i, 1]
    
    # --- CFM Target (Optimal Transport & Quarter-Sphere Bound) ---
    if MANEUVER == "1":
        # Maneuver 1: Strict Horizontal Flow (No vertical delta)
        z_1_cfm[i, 0] = np.random.normal(loc=10.0, scale=0.15)
        z_1_cfm[i, 1] = start_y
    else:
        # Maneuver 0: Radial projection to the absolute nearest circle point
        angle_cfm = np.arctan2(start_y - CENTER_Y, start_x - CENTER_X)
        z_1_cfm[i, 0] = CENTER_X + base_radius * np.cos(angle_cfm)
        z_1_cfm[i, 1] = CENTER_Y + base_radius * np.sin(angle_cfm)
        
    # --- Diffusion Target (Blind Omnidirectional Distribution) ---
    if MANEUVER == "1":
        # Diffusion treats the vertical line as a 2D distributed column space
        z_1_diff[i, 0] = np.random.normal(loc=10.0, scale=0.15)
        z_1_diff[i, 1] = np.random.uniform(2.0, 18.0)
    else:
        # Diffusion targets a random spot anywhere on the ring circumference
        # This completely ignores its starting quadrant, forcing massive crossover pathing!
        random_angle = np.random.uniform(0, 2 * np.pi)
        z_1_diff[i, 0] = CENTER_X + base_radius * np.cos(random_angle)
        z_1_diff[i, 1] = CENTER_Y + base_radius * np.sin(random_angle)

# =====================================================================
# TRAJECTORY GENERATION ENGINE
# =====================================================================

# 1. CFM Execution: Straight, Non-intersecting Optimal Paths with Warped Timing
cfm_trajectories = []
for step in range(NUM_STEPS):
    linear_tau = step / (NUM_STEPS - 1)
    # Power schedule causes particles to quickly slide and lock down early
    warped_tau = 1.0 - (1.0 - linear_tau) ** 3.5
    
    z_tau = warped_tau * z_1_cfm + (1 - warped_tau) * z_0
    cfm_trajectories.append(z_tau)

# 2. Diffusion Execution: Turbulent 2D Brownian Motion with Path Crossover
diffusion_trajectories = []
current_z = np.copy(z_0)
diffusion_trajectories.append(np.copy(current_z))

for step in range(1, NUM_STEPS):
    tau = step / (NUM_STEPS - 1)
    dt = 1.0 / NUM_STEPS
    
    # Linear drift pulling towards the unconstrained, randomized target mapping
    pull_strength = 2.0 / (1.1 - tau)
    drift = pull_strength * (z_1_diff - current_z) * dt
    
    # Wild stochastic forces shaking particles violently across BOTH dimensions
    noise_env = np.random.normal(0, 1, size=z_0.shape)
    stochastic_kick = DIFFUSION_NOISE_SCALE * np.sqrt(dt) * noise_env * (1.0 - tau)
    
    current_z = current_z + drift + stochastic_kick
    current_z = np.clip(current_z, 0.1, GRID_SIZE - 0.1)
    diffusion_trajectories.append(np.copy(current_z))

# =====================================================================
# MATPLOTLIB COMPOSITOR & ANIMATION RUNTIME
# =====================================================================
fig, (ax_cfm, ax_diff) = plt.subplots(1, 2, figsize=(11, 5.5))
fig.suptitle(f"FSD3D Core Principle Validation (Maneuver '{MANEUVER}')", 
             fontsize=14, fontweight='bold', color='#FFFFFF')
fig.patch.set_facecolor('#111625') 

def style_axes(ax, title):
    ax.set_facecolor('#161c2e')
    ax.set_title(title, fontsize=12, color='#FFFFFF', fontweight='bold', pad=10)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(0, GRID_SIZE + 1, 5))
    ax.set_yticks(range(0, GRID_SIZE + 1, 5))
    ax.grid(True, color='#242f4c', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='#7f8c8d', labelsize=9)
    
    # Draw centerline bounds to visually check quadrant constraint enforcement
    ax.axhline(10, color='#242f4c', linestyle=':', linewidth=1)
    ax.axvline(10, color='#242f4c', linestyle=':', linewidth=1)
    
    for spine in ax.spines.values():
        spine.set_color('#242f4c')

style_axes(ax_cfm, "Conditional Flow Matching (CFM)\n[Optimal Transport • Quadrant Constrained]")
style_axes(ax_diff, "Conventional Diffusion Framework\n[Omnidirectional Chaos • Blind Crossover]")

scatter_cfm = ax_cfm.scatter([], [], color='#00ffcc', edgecolors='#004d3d', s=45, zorder=3)
scatter_diff = ax_diff.scatter([], [], color='#ffaa00', edgecolors='#4a2700', s=45, zorder=3)

text_cfm = ax_cfm.text(1, 1, '', color='#ffffff', fontsize=10, family='monospace',
                       bbox=dict(facecolor='#111625', alpha=0.8, edgecolor='none'))
text_diff = ax_diff.text(1, 1, '', color='#ffffff', fontsize=10, family='monospace',
                         bbox=dict(facecolor='#111625', alpha=0.8, edgecolor='none'))

def update(frame):
    linear_tau = frame / (NUM_STEPS - 1)
    warped_tau = 1.0 - (1.0 - linear_tau) ** 3.5
    
    scatter_cfm.set_offsets(cfm_trajectories[frame])
    text_cfm.set_text(f"Flow Time (tau) = {warped_tau:.2f}")
    
    scatter_diff.set_offsets(diffusion_trajectories[frame])
    text_diff.set_text(f"Denoise Time (t) = {1.0 - linear_tau:.2f}")
    
    return scatter_cfm, scatter_diff, text_cfm, text_diff

ani = animation.FuncAnimation(
    fig, update, frames=NUM_STEPS, interval=80, blit=True, repeat=True
)

plt.tight_layout()
plt.show()