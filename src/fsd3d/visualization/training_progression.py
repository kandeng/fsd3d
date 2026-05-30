"""Training progression visualizer for CFM and Autoregressive MSE.

Shows how the model's predictions evolve during training:
  - CFM: random dots gradually align with expert trajectories (bimodal)
  - AR:  random predictions converge to mode-averaged straight line (crash)

Trains both models from scratch, taking snapshots at key epochs,
then generates a static figure showing the training progression.

Usage:
    from fsd3d.visualization.training_progression import (
        train_cfm_with_snapshots, train_ar_with_snapshots,
        plot_training_progression,
    )
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.autoregressive import AutoregressiveWrapper
from fsd3d.decoder.context import (
    compute_spatial_trajectory, denormalize_trajectory,
)
from fsd3d.plugin.mock import PILLAR_CENTER_X, PILLAR_CENTER_Y, PILLAR_RADIUS
from fsd3d.inference.cfm import infer_cfm_euler


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
BG_COLOR     = "#1a1a2e"
OBSTACLE_CLR = "#4a4a6a"
EXPERT_CLR   = "#ffffff"
AR_CLR       = "#ff4444"
CFM_CLR      = "#44aaff"
CFM_L_CLR    = "#6688ff"
CFM_R_CLR    = "#44ddff"
LABEL_CLR    = "#cccccc"
EXPERT_L_CLR = "#aaaaff"
EXPERT_R_CLR = "#ffaaaa"
CRASH_CLR    = "#ff3333"


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


# Stage labels for progression
AR_STAGE_LABELS = {
    0:   "random",
    5:   "learning",
    15:  "vertical",
    40:  "averaging",
    100: "converged",
    200: "locked-in",
    300: "collision",
}
CFM_STAGE_LABELS = {
    0:    "random",
    50:   "learning",
    150:  "emerging",
    400:  "splitting",
    800:  "resolving",
    1500: "clean",
    3000: "converged",
}


# ---------------------------------------------------------------------------
# Snapshot training — CFM
# ---------------------------------------------------------------------------
def train_cfm_with_snapshots(
    z1, context,
    epochs=5000, batch_size=64, lr=1e-3, seed=42,
    snapshot_epochs=None,
    num_samples=4,
):
    """Train CFM and return snapshots of model predictions at key epochs.

    Returns:
        snapshots: list of (epoch, cfm_spatial_samples) where
            cfm_spatial_samples is list of (spatial_pos, direction) tuples
    """
    if snapshot_epochs is None:
        snapshot_epochs = [0, 100, 300, 700, 1500, 3000, 5000]

    torch.manual_seed(seed)
    decoder = FSD3DTransformerDecoder()
    action_head = ActionHead()
    decoder.train()
    action_head.train()

    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(action_head.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    snapshots = []

    def take_snapshot(epoch):
        decoder.eval()
        action_head.eval()
        samples = []
        for i in range(num_samples):
            with torch.no_grad():
                cfm_final, _ = infer_cfm_euler(
                    decoder, action_head, context, euler_steps=20, seed=100 + i,
                )
            cfm_spatial = denormalize_trajectory(
                compute_spatial_trajectory(cfm_final[0].detach().numpy())
            )
            mse_l = F.mse_loss(cfm_final[0], z1[0]).item()
            mse_r = F.mse_loss(cfm_final[0], z1[1]).item()
            direction = "left" if mse_l < mse_r else "right"
            samples.append((cfm_spatial, direction))
        snapshots.append((epoch, samples))
        decoder.train()
        action_head.train()

    if 0 in snapshot_epochs:
        take_snapshot(0)
        snapshot_epochs = [e for e in snapshot_epochs if e > 0]

    for epoch in range(1, epochs + 1):
        indices = torch.randint(0, z1.size(0), (batch_size,))
        z1_batch = z1[indices]
        ctx_batch = context.expand(batch_size, -1, -1)

        optimizer.zero_grad()
        z0 = torch.randn(batch_size, 16, 2)
        tau = torch.rand(batch_size, 1)
        z_tau = tau.unsqueeze(2) * z1_batch + (1 - tau.unsqueeze(2)) * z0
        v_target = z1_batch - z0

        latent = decoder(z_tau, tau, ctx_batch)
        v_pred = action_head(latent)
        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch in snapshot_epochs:
            take_snapshot(epoch)
            print(f"  [CFM] Snapshot at epoch {epoch}  loss={loss.item():.6f}")

    return snapshots


# ---------------------------------------------------------------------------
# Snapshot training — AR
# ---------------------------------------------------------------------------
def train_ar_with_snapshots(
    z1, context,
    epochs=300, batch_size=64, lr=1e-3, seed=42,
    snapshot_epochs=None,
):
    """Train AR and return snapshots of model predictions at key epochs.

    Returns:
        snapshots: list of (epoch, ar_spatial) where
            ar_spatial is a (16, 2) numpy array in real-space
    """
    if snapshot_epochs is None:
        snapshot_epochs = [0, 5, 15, 40, 100, 200, 300]

    torch.manual_seed(seed)
    decoder = FSD3DTransformerDecoder()
    action_head = ActionHead()
    wrapper = AutoregressiveWrapper(decoder, action_head)
    wrapper.train()

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=lr)

    snapshots = []

    def take_snapshot(epoch):
        wrapper.eval()
        with torch.no_grad():
            ar_result = wrapper.generate(
                context.expand(1, -1, -1), horizon=16,
                noise_step=2, noise_sigma=0.06,
                continuous_noise=True, drift_bias=0.97,
            )
        ar_spatial = denormalize_trajectory(
            compute_spatial_trajectory(ar_result[0].detach().numpy())
        )
        snapshots.append((epoch, ar_spatial))
        wrapper.train()

    if 0 in snapshot_epochs:
        take_snapshot(0)
        snapshot_epochs = [e for e in snapshot_epochs if e > 0]

    for epoch in range(1, epochs + 1):
        indices = torch.randint(0, z1.size(0), (batch_size,))
        z1_batch = z1[indices]
        ctx_batch = context.expand(batch_size, -1, -1)
        tau_fixed = torch.ones(batch_size, 1)

        optimizer.zero_grad()
        pred = wrapper(z1_batch, tau_fixed, ctx_batch)
        loss = F.mse_loss(pred, z1_batch)
        loss.backward()
        optimizer.step()

        if epoch in snapshot_epochs:
            take_snapshot(epoch)
            print(f"  [AR ] Snapshot at epoch {epoch}  loss={loss.item():.6f}")

    return snapshots


# ---------------------------------------------------------------------------
# Plot training progression
# ---------------------------------------------------------------------------
def plot_training_progression(cfm_snapshots, ar_snapshots, z1, out_path,
                              icon_path=None):
    """Generate a 2-panel figure showing training progression.

    Left panel  — AR:  random → vertical → mode-averaged straight line → crash
    Right panel — CFM: random → flow emerges → left/right mode split → clean dodge
    """

    expert_pos_left = denormalize_trajectory(compute_spatial_trajectory(z1[0].numpy()))
    expert_pos_right = denormalize_trajectory(compute_spatial_trajectory(z1[1].numpy()))
    num_stages = len(cfm_snapshots)

    fig, (ax_ar, ax_cfm) = plt.subplots(
        1, 2, figsize=(14, 8), facecolor=BG_COLOR,
    )
    fig.suptitle(
        "FSD3D Training Progression: From Random Noise to Convergence",
        color=LABEL_CLR, fontsize=14, fontweight="bold",
    )

    for ax in [ax_ar, ax_cfm]:
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 11)
        ax.set_xlabel("x", color=LABEL_CLR)
        ax.set_ylabel("y", color=LABEL_CLR)
        ax.tick_params(colors=LABEL_CLR)
        ax.set_aspect("equal")
        ax.add_patch(
            patches.Circle(
                (PILLAR_CENTER_X, PILLAR_CENTER_Y), PILLAR_RADIUS,
                facecolor=OBSTACLE_CLR, edgecolor=EXPERT_CLR, linewidth=1.5,
            )
        )
        ax.axhline(y=0, color=LABEL_CLR, linewidth=0.5, alpha=0.3)
        ax.axhline(y=10, color=LABEL_CLR, linewidth=0.5, alpha=0.3)
        ax.text(2.8, 0.2, "START", color=LABEL_CLR, fontsize=7, ha="right", alpha=0.4)
        ax.text(2.8, 10.2, "GOAL", color=LABEL_CLR, fontsize=7, ha="right", alpha=0.4)
        ax.plot(
            expert_pos_left[:, 0], expert_pos_left[:, 1],
            "--", color=EXPERT_L_CLR, linewidth=1, alpha=0.35,
        )
        ax.plot(
            expert_pos_right[:, 0], expert_pos_right[:, 1],
            "--", color=EXPERT_R_CLR, linewidth=1, alpha=0.35,
        )

    # -------------------------------------------------------------------
    # Left panel: AR training progression
    # -------------------------------------------------------------------
    ax_ar.set_title(
        "Autoregressive MSE\n"
        "random → mode averaging → collision",
        color=AR_CLR, fontsize=12, fontweight="bold",
    )

    ar_handles, ar_labels = [], []
    ar_any_collision = False
    for idx, (epoch, ar_spatial) in enumerate(ar_snapshots):
        progress = idx / max(num_stages - 1, 1)
        alpha = 0.15 + 0.75 * progress
        lw = 1.0 + 2.0 * progress
        is_last = (idx == len(ar_snapshots) - 1)

        ar_collision_step = None
        ar_collision_pos = None
        for t in range(len(ar_spatial) - 1):
            collides, t_hit = _segment_circle_collision(
                ar_spatial[t], ar_spatial[t + 1],
                PILLAR_CENTER_X, PILLAR_CENTER_Y, PILLAR_RADIUS,
            )
            if collides:
                ar_collision_step = t
                ar_collision_pos = ar_spatial[t] * (1 - t_hit) + ar_spatial[t + 1] * t_hit
                ar_spatial = np.vstack([ar_spatial[:t + 1], [ar_collision_pos]])
                ar_any_collision = True
                break

        line, = ax_ar.plot(
            ar_spatial[:, 0], ar_spatial[:, 1], "-",
            color=AR_CLR, linewidth=lw, alpha=alpha,
        )
        ax_ar.scatter(
            ar_spatial[:, 0], ar_spatial[:, 1],
            color=AR_CLR, s=8 + 12 * progress, alpha=alpha, zorder=5,
        )

        if ar_collision_pos is not None:
            ax_ar.plot(
                ar_collision_pos[0], ar_collision_pos[1], "X",
                color=CRASH_CLR, markersize=8 + 6 * progress,
                markeredgewidth=max(1, 2 * progress),
                alpha=alpha, zorder=8,
            )

        stage_label = AR_STAGE_LABELS.get(epoch, "")
        if idx % 2 == 0 or is_last:
            label_text = f"Epoch {epoch}"
            if stage_label:
                label_text += f"\n({stage_label})"
            ax_ar.annotate(
                label_text,
                xy=(ar_spatial[-1, 0], ar_spatial[-1, 1]),
                xytext=(12, 0), textcoords="offset points",
                color=AR_CLR, fontsize=7, alpha=min(alpha + 0.15, 1.0),
                fontweight="bold" if is_last else "normal",
                verticalalignment="center",
            )

        if idx == 0 or idx == len(ar_snapshots) // 2 or is_last:
            ar_handles.append(line)
            ar_labels.append(f"Epoch {epoch} ({stage_label})" if stage_label else f"Epoch {epoch}")

    if ar_any_collision and icon_path:
        icon_img = plt.imread(icon_path)
        imagebox = OffsetImage(icon_img, zoom=0.15, interpolation="hanning")
        ab = AnnotationBbox(
            imagebox, xy=(0.5, 0.5), xycoords="axes fraction",
            frameon=False, zorder=20,
        )
        ax_ar.add_artist(ab)

    ar_expert_l = ax_ar.plot([], [], "--", color=EXPERT_L_CLR, linewidth=1, alpha=0.6)[0]
    ar_expert_r = ax_ar.plot([], [], "--", color=EXPERT_R_CLR, linewidth=1, alpha=0.6)[0]
    ar_handles = [ar_expert_l, ar_expert_r] + ar_handles
    ar_labels = ["Expert (left)", "Expert (right)"] + ar_labels
    ax_ar.legend(ar_handles, ar_labels,
                 loc="upper center", bbox_to_anchor=(0.5, -0.08),
                 ncol=5, fontsize=7, frameon=True,
                 facecolor=BG_COLOR, edgecolor=LABEL_CLR, labelcolor=LABEL_CLR)

    # -------------------------------------------------------------------
    # Right panel: CFM training progression
    # -------------------------------------------------------------------
    ax_cfm.set_title(
        "Conditional Flow Matching\n"
        "random → mode split → clean dodge",
        color=CFM_CLR, fontsize=12, fontweight="bold",
    )

    cfm_handles, cfm_labels = [], []
    last_left_done = False
    last_right_done = False

    for idx, (epoch, cfm_samples) in enumerate(cfm_snapshots):
        progress = idx / max(num_stages - 1, 1)
        alpha = 0.10 + 0.70 * progress
        lw = 0.7 + 1.5 * progress
        is_last = (idx == len(cfm_snapshots) - 1)

        for sample_idx, (spatial, direction) in enumerate(cfm_samples):
            color = CFM_L_CLR if direction == "left" else CFM_R_CLR

            line, = ax_cfm.plot(
                spatial[:, 0], spatial[:, 1], "-",
                color=color, linewidth=lw, alpha=alpha,
            )
            ax_cfm.scatter(
                spatial[:, 0], spatial[:, 1],
                color=color, s=5 + 10 * progress, alpha=alpha, zorder=5,
            )

            if is_last:
                if direction == "left" and not last_left_done:
                    cfm_handles.append(line)
                    cfm_labels.append("CFM left dodge")
                    last_left_done = True
                elif direction == "right" and not last_right_done:
                    cfm_handles.append(line)
                    cfm_labels.append("CFM right dodge")
                    last_right_done = True

        stage_label = CFM_STAGE_LABELS.get(epoch, "")
        if idx % 2 == 0 or is_last:
            sample0 = cfm_samples[0][0]
            label_text = f"Epoch {epoch}"
            if stage_label:
                label_text += f"\n({stage_label})"
            ax_cfm.annotate(
                label_text,
                xy=(sample0[-1, 0], sample0[-1, 1]),
                xytext=(12, 0), textcoords="offset points",
                color=LABEL_CLR, fontsize=7, alpha=min(alpha + 0.15, 1.0),
                fontweight="bold" if is_last else "normal",
                verticalalignment="center",
            )

        if idx == 0 or idx == len(cfm_snapshots) // 2:
            cfm_handles.append(ax_cfm.plot([], [], "-", color=CFM_L_CLR, linewidth=lw)[0])
            cfm_labels.append(f"Epoch {epoch} ({stage_label})" if stage_label else f"Epoch {epoch}")

    cfm_expert_l = ax_cfm.plot([], [], "--", color=EXPERT_L_CLR, linewidth=1, alpha=0.6)[0]
    cfm_expert_r = ax_cfm.plot([], [], "--", color=EXPERT_R_CLR, linewidth=1, alpha=0.6)[0]
    cfm_handles = [cfm_expert_l, cfm_expert_r] + cfm_handles
    cfm_labels = ["Expert (left)", "Expert (right)"] + cfm_labels
    ax_cfm.legend(cfm_handles, cfm_labels,
                  loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=5, fontsize=7, frameon=True,
                  facecolor=BG_COLOR, edgecolor=LABEL_CLR, labelcolor=LABEL_CLR)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    fig.savefig(out_path, dpi=150, facecolor=BG_COLOR, edgecolor="none",
                bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"Training progression saved to {os.path.abspath(out_path)}")
