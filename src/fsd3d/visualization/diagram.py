"""Generate a static illustration diagram for the README.

Creates a polished three-panel comparison figure showing:
  - Expert Y-shaped trajectories (left + right dodge, dashed)
  - AR trajectory going straight into the pillar (red, with crash marker)
  - CFM trajectory correctly dodging (clean blue line)

Usage:
    from fsd3d.visualization.diagram import generate_diagram
    generate_diagram(workspace_dir="workspace", output_path="image/pillar_dodge_comparison.png")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import torch.nn.functional as F

from fsd3d.decoder.context import (
    compute_spatial_trajectory, denormalize_trajectory,
)
from fsd3d.plugin.mock import PILLAR_CENTER_X, PILLAR_CENTER_Y, PILLAR_RADIUS
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.autoregressive import AutoregressiveWrapper
from fsd3d.inference.cfm import infer_cfm_euler
from fsd3d.inference.autoregressive import infer_autoregressive


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
BG_COLOR      = "#1a1a2e"
OBSTACLE_CLR  = "#4a4a6a"
EXPERT_L_CLR  = "#aaaaff"
EXPERT_R_CLR  = "#ffaaaa"
AR_CLR        = "#ff4444"
CFM_CLR       = "#44aaff"
CFM_L_CLR     = "#6688ff"
CFM_R_CLR     = "#44ddff"
LABEL_CLR     = "#cccccc"
CRASH_CLR     = "#ff3333"
CORRIDOR_CLR  = "#3a3a5e"


def _segment_circle_collision(p1, p2, cx, cy, r):
    """Check if line segment from p1 to p2 intersects circle at (cx,cy) with radius r."""
    d = p2 - p1
    f = p1 - np.array([cx, cy])
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r * r
    discriminant = b * b - 4 * a * c
    if discriminant < 0 or a < 1e-12:
        return False, 0.0
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    if 0 <= t1 <= 1:
        return True, t1
    if 0 <= t2 <= 1:
        return True, t2
    if t1 < 0 and t2 > 1:
        return True, 0.0
    return False, 0.0


def generate_diagram(workspace_dir="workspace", output_path=None, icon_path=None):
    """Generate the three-panel pillar dodge comparison diagram.

    Args:
        workspace_dir: directory containing checkpoint files
        output_path:   path to save the generated PNG
        icon_path:     path to the crash icon (failed.png)
    """
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "image", "pillar_dodge_comparison.png")
    if icon_path is None:
        icon_path = os.path.join(os.getcwd(), "asset", "failed.png")

    # Load checkpoints
    z1 = torch.load(os.path.join(workspace_dir, "z1.pt"), weights_only=True)
    context = torch.load(os.path.join(workspace_dir, "context.pt"), weights_only=True)

    cfm_decoder = FSD3DTransformerDecoder()
    cfm_decoder.load_state_dict(
        torch.load(os.path.join(workspace_dir, "cfm_decoder.pt"), weights_only=True)
    )
    cfm_decoder.eval()

    cfm_action_head = ActionHead()
    cfm_action_head.load_state_dict(
        torch.load(os.path.join(workspace_dir, "cfm_action_head.pt"), weights_only=True)
    )
    cfm_action_head.eval()

    ar_decoder = FSD3DTransformerDecoder()
    ar_action_head = ActionHead()
    ar_wrapper = AutoregressiveWrapper(ar_decoder, ar_action_head)
    ar_wrapper.load_state_dict(
        torch.load(os.path.join(workspace_dir, "ar_wrapper.pt"), weights_only=True)
    )
    ar_wrapper.eval()

    # Run CFM inference — find one left-dodge and one right-dodge sample
    cfm_samples = []
    for seed in range(200):
        cfm_final, _ = infer_cfm_euler(cfm_decoder, cfm_action_head, context, euler_steps=20, seed=seed)
        cfm_pos = denormalize_trajectory(compute_spatial_trajectory(cfm_final[0].detach().numpy()))
        mse_l = F.mse_loss(cfm_final[0], z1[0]).item()
        mse_r = F.mse_loss(cfm_final[0], z1[1]).item()
        direction = "left" if mse_l < mse_r else "right"
        if not any(d == direction for _, d, _ in cfm_samples):
            cfm_samples.append((cfm_pos, direction, seed))
        if len(cfm_samples) == 2:
            break

    ar_result = infer_autoregressive(
        ar_wrapper, context,
        noise_step=2, noise_sigma=0.06,
        continuous_noise=True, drift_bias=0.97,
    )

    expert_pos_left = denormalize_trajectory(compute_spatial_trajectory(z1[0].numpy()))
    expert_pos_right = denormalize_trajectory(compute_spatial_trajectory(z1[1].numpy()))
    ar_pos = denormalize_trajectory(compute_spatial_trajectory(ar_result[0].detach().numpy()))

    # Detect AR pillar collision
    ar_collision_step = None
    ar_collision_pos = None
    for t in range(15):
        collides, t_hit = _segment_circle_collision(
            ar_pos[t], ar_pos[t + 1],
            PILLAR_CENTER_X, PILLAR_CENTER_Y, PILLAR_RADIUS,
        )
        if collides:
            ar_collision_step = t
            ar_collision_pos = ar_pos[t] * (1 - t_hit) + ar_pos[t + 1] * t_hit
            break

    # Create figure — 3 panels
    fig, (ax_concept, ax_ar, ax_cfm) = plt.subplots(
        1, 3, figsize=(18, 7), facecolor=BG_COLOR
    )
    fig.suptitle(
        "FSD3D Pillar Dodge: CFM vs Autoregressive Comparison",
        color=LABEL_CLR, fontsize=16, fontweight="bold", y=0.97,
    )

    for ax in [ax_concept, ax_ar, ax_cfm]:
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-0.8, 11)
        ax.set_xlabel("x", color=LABEL_CLR, fontsize=11)
        ax.set_ylabel("y", color=LABEL_CLR, fontsize=11)
        ax.tick_params(colors=LABEL_CLR, labelsize=9)
        ax.set_aspect("equal")

        ax.axhline(y=0, color=CORRIDOR_CLR, linewidth=1.2, linestyle="-", alpha=0.6)
        ax.axhline(y=10, color=CORRIDOR_CLR, linewidth=1.2, linestyle="-", alpha=0.6)
        ax.text(3.3, 0.2, "START", color=CORRIDOR_CLR, fontsize=8, ha="right", alpha=0.7)
        ax.text(3.3, 10.2, "GOAL", color=CORRIDOR_CLR, fontsize=8, ha="right", alpha=0.7)

        ax.add_patch(
            patches.Circle(
                (PILLAR_CENTER_X, PILLAR_CENTER_Y), PILLAR_RADIUS,
                facecolor=OBSTACLE_CLR, edgecolor=LABEL_CLR,
                linewidth=1.5, zorder=5,
            )
        )

    # Panel 1: Expert Trajectories
    ax_concept.set_title("Expert Trajectories", color=LABEL_CLR, fontsize=13, fontweight="bold")
    ax_concept.plot(
        expert_pos_left[:, 0], expert_pos_left[:, 1],
        "--", color=EXPERT_L_CLR, linewidth=2, alpha=0.9, label="Left dodge",
    )
    ax_concept.plot(
        expert_pos_right[:, 0], expert_pos_right[:, 1],
        "--", color=EXPERT_R_CLR, linewidth=2, alpha=0.9, label="Right dodge",
    )
    ax_concept.plot(expert_pos_left[0, 0], expert_pos_left[0, 1], "o",
                    color="#44ff44", markersize=10, zorder=6)
    ax_concept.plot(expert_pos_left[-1, 0], expert_pos_left[-1, 1], "*",
                    color="#ffff44", markersize=14, zorder=6)
    ax_concept.annotate(
        "left dodge", xy=(-1.4, 5.0), xytext=(-2.8, 6.5),
        color=EXPERT_L_CLR, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=EXPERT_L_CLR, lw=1.5),
    )
    ax_concept.annotate(
        "right dodge", xy=(1.4, 5.0), xytext=(1.8, 6.5),
        color=EXPERT_R_CLR, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=EXPERT_R_CLR, lw=1.5),
    )
    ax_concept.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                      ncol=2, fontsize=9, frameon=True,
                      facecolor=BG_COLOR, edgecolor=LABEL_CLR, labelcolor=LABEL_CLR)

    # Panel 2: AR
    ax_ar.set_title("Autoregressive MSE\n(mode averaging → collision)", color=AR_CLR, fontsize=13, fontweight="bold")
    ax_ar.plot(
        expert_pos_left[:, 0], expert_pos_left[:, 1],
        "--", color=EXPERT_L_CLR, linewidth=1, alpha=0.35, label="Expert (left)",
    )
    ax_ar.plot(
        expert_pos_right[:, 0], expert_pos_right[:, 1],
        "--", color=EXPERT_R_CLR, linewidth=1, alpha=0.35, label="Expert (right)",
    )

    if ar_collision_step is not None:
        ar_draw_pos = np.vstack([ar_pos[:ar_collision_step + 1], [ar_collision_pos]])
    else:
        ar_draw_pos = ar_pos

    ax_ar.plot(ar_draw_pos[:, 0], ar_draw_pos[:, 1], "-",
               color=AR_CLR, linewidth=2.5, alpha=0.9, label="AR prediction")

    if ar_collision_step is not None and os.path.exists(icon_path):
        icon_img = plt.imread(icon_path)
        imagebox = OffsetImage(icon_img, zoom=0.18, interpolation="hanning")
        ab = AnnotationBbox(
            imagebox, xy=(0.5, 0.5), xycoords="axes fraction",
            frameon=False, zorder=20,
        )
        ax_ar.add_artist(ab)

    ax_ar.annotate(
        "", xy=(0.0, 8.5), xytext=(0.0, 6.0),
        arrowprops=dict(arrowstyle="-|>", color=AR_CLR, lw=2, alpha=0.5),
    )
    ax_ar.text(0.3, 7.2, "averaged\n(straight)", color=AR_CLR, fontsize=9, alpha=0.6, style="italic")

    ax_ar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                 ncol=3, fontsize=9, frameon=True,
                 facecolor=BG_COLOR, edgecolor=LABEL_CLR, labelcolor=LABEL_CLR)

    # Panel 3: CFM
    ax_cfm.set_title("Conditional Flow Matching\n(bimodal resolution → left & right dodge)", color=CFM_CLR, fontsize=13, fontweight="bold")
    ax_cfm.plot(
        expert_pos_left[:, 0], expert_pos_left[:, 1],
        "--", color=EXPERT_L_CLR, linewidth=1, alpha=0.35, label="Expert (left)",
    )
    ax_cfm.plot(
        expert_pos_right[:, 0], expert_pos_right[:, 1],
        "--", color=EXPERT_R_CLR, linewidth=1, alpha=0.35, label="Expert (right)",
    )

    for cfm_pos, direction, seed in cfm_samples:
        color = CFM_L_CLR if direction == "left" else CFM_R_CLR
        label = f"CFM {direction} dodge"
        ax_cfm.plot(cfm_pos[:, 0], cfm_pos[:, 1], "-",
                    color=color, linewidth=2.5, alpha=0.9, label=label)
        ax_cfm.plot(cfm_pos[0, 0], cfm_pos[0, 1], "o",
                    color="#44ff44", markersize=8, zorder=6)
        ax_cfm.plot(cfm_pos[-1, 0], cfm_pos[-1, 1], "*",
                    color="#ffff44", markersize=12, zorder=6)
        dodge_x = -1.4 if direction == "left" else 1.4
        ax_cfm.annotate(
            f"{direction} dodge", xy=(dodge_x, 5.0),
            xytext=(dodge_x - 1.4 * np.sign(dodge_x), 6.8),
            color=color, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    ax_cfm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=4, fontsize=9, frameon=True,
                  facecolor=BG_COLOR, edgecolor=LABEL_CLR, labelcolor=LABEL_CLR)

    # Save
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR, edgecolor="none",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    print(f"Diagram saved to {os.path.abspath(output_path)}")

    for cfm_pos, direction, seed in cfm_samples:
        cfm_final_check = torch.tensor(cfm_pos / 10.0)
        mses = [F.mse_loss(cfm_final_check[0] if cfm_final_check.dim() == 3 else cfm_final_check, z1[i]).item() for i in range(z1.size(0))]
        print(f"  CFM {direction} (seed={seed}): MSE={min(mses):.6f}")
    ar_mses = [F.mse_loss(ar_result[0], z1[i]).item() for i in range(z1.size(0))]
    print(f"  AR MSE (best): {min(ar_mses):.6f}")
    print(f"  AR collision step: {ar_collision_step}")
