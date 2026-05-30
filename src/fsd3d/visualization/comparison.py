"""Interactive CFM vs Autoregressive side-by-side animation.

Loads trained models from workspace/ checkpoints and animates
inference results in real time using matplotlib FuncAnimation.

Usage:
    from fsd3d.visualization.comparison import ComparisonVisualizer
    viz = ComparisonVisualizer(cfm_decoder, cfm_action_head, ar_wrapper, z1, context)
    viz.run()
"""

import os
import datetime
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
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


# Noise parameters for AR inference — applied in NORMALIZED space.
AR_NOISE_SIGMA = 0.06
AR_NOISE_STEP = 2
AR_CONTINUOUS_NOISE = True
AR_DRIFT_BIAS = 0.97


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
BTN_CLR      = "#2a2a4e"
BTN_HOVER    = "#3a3a6e"
CRASH_CLR    = "#ff3333"
EXPERT_L_CLR = "#aaaaff"
EXPERT_R_CLR = "#ffaaaa"


def _segment_circle_collision(p1, p2, cx, cy, r):
    """Check if line segment from p1 to p2 intersects circle at (cx,cy) with radius r.

    Returns (collides, t_hit) where t_hit ∈ [0,1] is the first intersection parameter.
    """
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


def load_checkpoints(workspace_dir=None):
    """Load trained models and data from workspace/."""
    if workspace_dir is None:
        workspace_dir = os.path.join(os.getcwd(), "workspace")

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

    return cfm_decoder, cfm_action_head, ar_wrapper, z1, context


def log_inference(ar_wrapper, z1, context, noise_sigma, noise_step,
                  workspace_dir=None, seed=0):
    """Run AR inference with detailed per-step logging; save to workspace/."""
    if workspace_dir is None:
        workspace_dir = os.path.join(os.getcwd(), "workspace")

    torch.manual_seed(seed)
    ctx = context.expand(1, -1, -1)

    # --- Clean inference (no noise) ---
    clean_result = ar_wrapper.generate(ctx, horizon=16, noise_step=-1, noise_sigma=0.0)

    # --- Noisy inference with drift ---
    noisy_result = ar_wrapper.generate(ctx, horizon=16, noise_step=noise_step,
                                        noise_sigma=noise_sigma,
                                        continuous_noise=AR_CONTINUOUS_NOISE,
                                        drift_bias=AR_DRIFT_BIAS)

    # Find closest expert for each result
    def closest_expert_mse(result):
        mses = [F.mse_loss(result[0], z1[i]).item() for i in range(z1.size(0))]
        return min(mses), mses.index(min(mses))

    clean_mse, clean_idx = closest_expert_mse(clean_result)
    noisy_mse, noisy_idx = closest_expert_mse(noisy_result)

    # Log per-step details
    log_lines = []
    log_lines.append(f"AR Inference Log — {datetime.datetime.now().isoformat()}")
    log_lines.append(f"noise_sigma={noise_sigma}, noise_step={noise_step}")
    log_lines.append("")
    log_lines.append(f"Overall MSE — clean: {clean_mse:.6f} (expert {clean_idx}), noisy: {noisy_mse:.6f} (expert {noisy_idx})")
    log_lines.append("")

    header = f"{'step':>4}  {'dim':>3}  {'z1_gt':>8}  {'clean':>8}  {'noisy':>8}  {'|err_cl|':>10}  {'|err_no|':>10}  {'drift':>10}"
    log_lines.append(header)
    log_lines.append("-" * len(header))

    z1_ref = z1[clean_idx]
    for t in range(16):
        for d in range(2):
            gt = z1_ref[t, d].item()
            cl = clean_result[0, t, d].item()
            no = noisy_result[0, t, d].item()
            ec = abs(cl - gt)
            en = abs(no - gt)
            drift = abs(no - cl)
            log_lines.append(f"{t:4d}  {d:3d}  {gt:8.4f}  {cl:8.4f}  {no:8.4f}  {ec:10.6f}  {en:10.6f}  {drift:10.6f}")
        if t > 0:
            gt_pos = compute_spatial_trajectory(z1_ref[:t+1].numpy())[-1]
            cl_pos = compute_spatial_trajectory(clean_result[0, :t+1, :].detach().numpy())[-1]
            no_pos = compute_spatial_trajectory(noisy_result[0, :t+1, :].detach().numpy())[-1]
            log_lines.append(f"  pos gt=({gt_pos[0]:.3f},{gt_pos[1]:.3f})  clean=({cl_pos[0]:.3f},{cl_pos[1]:.3f})  noisy=({no_pos[0]:.3f},{no_pos[1]:.3f})")

    log_path = os.path.join(workspace_dir, "ar_inference_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"AR inference log saved to {log_path}")
    print(f"  Clean MSE: {clean_mse:.6f}, Noisy MSE: {noisy_mse:.6f}")

    return clean_result, noisy_result


class ComparisonVisualizer:
    """Animate CFM vs AR inference on two side-by-side subplots."""

    def __init__(self, cfm_decoder, cfm_action_head, ar_wrapper, z1, context,
                 euler_steps=20, icon_path=None):
        self.cfm_decoder = cfm_decoder
        self.cfm_action_head = cfm_action_head
        self.ar_wrapper = ar_wrapper
        self.z1 = z1
        self.context = context
        self.euler_steps = euler_steps
        self.icon_path = icon_path

        # Both expert trajectories in real space
        self.expert_pos_left = denormalize_trajectory(
            compute_spatial_trajectory(z1[0].numpy()))
        self.expert_pos_right = denormalize_trajectory(
            compute_spatial_trajectory(z1[1].numpy()))
        # AR collision state
        self.ar_collision_step = None
        self._seed_counter = 0
        self._cfm_direction_history = []
        self._setup_figure()
        self._setup_button()
        self._on_click(None)

    def _setup_figure(self):
        self.fig, (self.ax_ar, self.ax_cfm) = plt.subplots(
            1, 2, figsize=(14, 6), facecolor=BG_COLOR
        )
        self.fig.suptitle(
            "FSD3D Decoder: CFM vs Autoregressive Comparison",
            color=LABEL_CLR, fontsize=14, fontweight="bold",
        )
        plt.subplots_adjust(bottom=0.28)

        for ax, title in [
            (self.ax_ar, "Autoregressive MSE"),
            (self.ax_cfm, "Conditional Flow Matching"),
        ]:
            ax.set_facecolor(BG_COLOR)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-0.5, 11)
            ax.set_xlabel("x", color=LABEL_CLR)
            ax.set_ylabel("y", color=LABEL_CLR)
            ax.set_title(title, color=LABEL_CLR)
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
            ax.plot(
                self.expert_pos_left[:, 0], self.expert_pos_left[:, 1],
                "--", color=EXPERT_L_CLR, linewidth=1, alpha=0.5, label="Expert (left)",
            )
            ax.plot(
                self.expert_pos_right[:, 0], self.expert_pos_right[:, 1],
                "--", color=EXPERT_R_CLR, linewidth=1, alpha=0.5, label="Expert (right)",
            )

        self.ar_line, = self.ax_ar.plot(
            [], [], "-", color=AR_CLR, linewidth=2, label="AR prediction"
        )
        self.ar_label = self.ax_ar.text(
            0.02, 0.95, "", transform=self.ax_ar.transAxes,
            color=LABEL_CLR, fontsize=11, verticalalignment="top",
        )
        self.ar_crash_icon = None
        self.ax_ar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                          ncol=3, fontsize=9, frameon=True,
                          facecolor=BG_COLOR, edgecolor=LABEL_CLR,
                          labelcolor=LABEL_CLR)

        self.cfm_line, = self.ax_cfm.plot(
            [], [], "-", color=CFM_CLR, linewidth=2.2, zorder=10,
        )
        self.ax_cfm.plot([], [], "-", color=CFM_L_CLR, linewidth=2, label="CFM left dodge")
        self.ax_cfm.plot([], [], "-", color=CFM_R_CLR, linewidth=2, label="CFM right dodge")
        self.cfm_label = self.ax_cfm.text(
            0.02, 0.95, "", transform=self.ax_cfm.transAxes,
            color=LABEL_CLR, fontsize=11, verticalalignment="top",
        )
        self.ax_cfm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                            ncol=4, fontsize=9, frameon=True,
                            facecolor=BG_COLOR, edgecolor=LABEL_CLR,
                            labelcolor=LABEL_CLR)

    def _setup_button(self):
        ax_btn = self.fig.add_axes([0.35, 0.03, 0.3, 0.05])
        self.btn = Button(
            ax_btn, "Generate Trajectory",
            color=BTN_CLR, hovercolor=BTN_HOVER,
        )
        self.btn.label.set_color(LABEL_CLR)
        self.btn.on_clicked(self._on_click)

    def _on_click(self, _event):
        self.btn.set_active(False)

        GREY_HIST = "#555566"
        if hasattr(self, 'ar_spatial_full') and self.ar_spatial_full is not None:
            self.ax_ar.plot(
                self.ar_spatial_full[:, 0], self.ar_spatial_full[:, 1],
                "-", color=GREY_HIST, linewidth=1.2, alpha=0.35, zorder=1,
            )
        if hasattr(self, 'cfm_spatial_current') and self.cfm_spatial_current is not None:
            self.ax_cfm.plot(
                self.cfm_spatial_current[:, 0], self.cfm_spatial_current[:, 1],
                "-", color=GREY_HIST, linewidth=1.2, alpha=0.35, zorder=1,
            )

        if self.ar_crash_icon is not None:
            self.ar_crash_icon.remove()
            self.ar_crash_icon = None

        self.ar_collision_step = None
        self._seed_counter += 1
        base_seed = 42 + self._seed_counter * 100

        # --- AR inference ---
        self.ar_result = infer_autoregressive(
            self.ar_wrapper, self.context,
            noise_step=AR_NOISE_STEP, noise_sigma=AR_NOISE_SIGMA,
            continuous_noise=AR_CONTINUOUS_NOISE, drift_bias=AR_DRIFT_BIAS,
        )

        # --- CFM direction alternation ---
        if len(self._cfm_direction_history) >= 2 and \
           self._cfm_direction_history[-1] == self._cfm_direction_history[-2]:
            desired_direction = "right" if self._cfm_direction_history[-1] == "left" else "left"
        else:
            desired_direction = "left" if (base_seed % 2 == 0) else "right"
        self.cfm_spatial_current = None
        self.cfm_direction_current = None
        cfm_mse_best = float('inf')
        for attempt in range(50):
            cfm_seed = base_seed + attempt
            cfm_final, _ = infer_cfm_euler(
                self.cfm_decoder, self.cfm_action_head, self.context,
                euler_steps=self.euler_steps, seed=cfm_seed,
            )
            mse_l = F.mse_loss(cfm_final[0], self.z1[0]).item()
            mse_r = F.mse_loss(cfm_final[0], self.z1[1]).item()
            direction = "left" if mse_l < mse_r else "right"
            if direction == desired_direction:
                cfm_spatial = denormalize_trajectory(
                    compute_spatial_trajectory(cfm_final[0].detach().cpu().numpy())
                )
                cfm_mse = min(mse_l, mse_r)
                cfm_mse_best = cfm_mse
                self.cfm_spatial_current = cfm_spatial
                self.cfm_direction_current = direction
                break

        if self.cfm_spatial_current is None:
            cfm_seed = base_seed
            cfm_final, _ = infer_cfm_euler(
                self.cfm_decoder, self.cfm_action_head, self.context,
                euler_steps=self.euler_steps, seed=cfm_seed,
            )
            cfm_spatial = denormalize_trajectory(
                compute_spatial_trajectory(cfm_final[0].detach().cpu().numpy())
            )
            mse_l = F.mse_loss(cfm_final[0], self.z1[0]).item()
            mse_r = F.mse_loss(cfm_final[0], self.z1[1]).item()
            self.cfm_direction_current = "left" if mse_l < mse_r else "right"
            cfm_mse_best = min(mse_l, mse_r)
            self.cfm_spatial_current = cfm_spatial

        self._cfm_direction_history.append(self.cfm_direction_current)

        cfm_color = CFM_L_CLR if self.cfm_direction_current == "left" else CFM_R_CLR
        self.cfm_line.set_color(cfm_color)
        self.cfm_line.set_alpha(0.9)

        ar_mse_left = F.mse_loss(self.ar_result[0], self.z1[0]).item()
        ar_mse_right = F.mse_loss(self.ar_result[0], self.z1[1]).item()
        ar_mse = min(ar_mse_left, ar_mse_right)
        print(f"  [Click #{self._seed_counter}]  CFM MSE={cfm_mse_best:.6f} ({self.cfm_direction_current})  AR MSE={ar_mse:.6f}")

        self.ar_spatial_full = denormalize_trajectory(
            compute_spatial_trajectory(self.ar_result[0].detach().cpu().numpy())
        )

        # Detect AR pillar collision
        self.ar_collision_step = None
        self.ar_collision_pos = None
        for t in range(15):
            collides, t_hit = _segment_circle_collision(
                self.ar_spatial_full[t], self.ar_spatial_full[t + 1],
                PILLAR_CENTER_X, PILLAR_CENTER_Y, PILLAR_RADIUS,
            )
            if collides:
                self.ar_collision_step = t
                self.ar_collision_pos = (
                    self.ar_spatial_full[t] * (1 - t_hit)
                    + self.ar_spatial_full[t + 1] * t_hit
                )
                self.ar_spatial_full = np.vstack([
                    self.ar_spatial_full[:t + 1],
                    [self.ar_collision_pos],
                ])
                break

        self._start_animation()

    # Sub-frames for animation
    AR_SUB_FRAMES = 4
    CFM_SUB_FRAMES = 2
    ANIM_INTERVAL = 100

    def _start_animation(self):
        total_frames = 16 * max(self.AR_SUB_FRAMES, self.CFM_SUB_FRAMES)
        self.anim = FuncAnimation(
            self.fig, self._update_frame,
            frames=total_frames, interval=self.ANIM_INTERVAL,
            repeat=False, blit=False,
        )
        self.fig.canvas.draw_idle()

    def _update_frame(self, frame):
        ar_step = frame // self.AR_SUB_FRAMES + 1

        t_draw = min(ar_step, len(self.ar_spatial_full))
        ar_pos = self.ar_spatial_full[:t_draw]
        self.ar_line.set_data(ar_pos[:, 0], ar_pos[:, 1])

        if self.ar_collision_step is not None and ar_step >= self.ar_collision_step + 1:
            if self.ar_crash_icon is None and self.icon_path:
                icon_img = plt.imread(self.icon_path)
                imagebox = OffsetImage(icon_img, zoom=0.15, interpolation="hanning")
                self.ar_crash_icon = AnnotationBbox(
                    imagebox, xy=(0.5, 0.45), xycoords="axes fraction",
                    frameon=False, zorder=20,
                )
                self.ax_ar.add_artist(self.ar_crash_icon)
            self.ar_label.set_text(f"Step: {self.ar_collision_step + 1}/16 — CRASHED")
        else:
            self.ar_label.set_text(f"Step: {min(ar_step, 16)}/16")

        cfm_step = frame // self.CFM_SUB_FRAMES + 1
        cfm_t = min(cfm_step, 16)
        cfm_pos = self.cfm_spatial_current[:cfm_t]
        self.cfm_line.set_data(cfm_pos[:, 0], cfm_pos[:, 1])
        self.cfm_label.set_text(f"Resolved: {cfm_t}/16  ({self.cfm_direction_current} dodge)")

        total_frames = 16 * max(self.AR_SUB_FRAMES, self.CFM_SUB_FRAMES)
        if frame == total_frames - 1:
            self.btn.set_active(True)

    def run(self):
        plt.show()
