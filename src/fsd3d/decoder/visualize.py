"""Standalone visualizer: load checkpoints and launch ComparisonVisualizer.

Usage:
    cd src/fsd3d/decoder
    python visualize.py
"""

import os

from fsd3d.visualization.comparison import (
    load_checkpoints, log_inference, ComparisonVisualizer,
    AR_NOISE_SIGMA, AR_NOISE_STEP,
)

WORKSPACE = os.path.join(os.path.dirname(__file__), "workspace")
ICON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "asset", "failed.png")


def main():
    print("Loading checkpoints from workspace/ …")
    cfm_decoder, cfm_action_head, ar_wrapper, z1, context = load_checkpoints(WORKSPACE)
    print(f"  z1 shape:      {z1.shape}")
    print(f"  context shape: {context.shape}")
    print(f"  AR noise:      sigma={AR_NOISE_SIGMA}, step={AR_NOISE_STEP}")
    print()

    # Run detailed inference logging
    print("Running AR inference diagnostic …")
    log_inference(ar_wrapper, z1, context, noise_sigma=AR_NOISE_SIGMA,
                  noise_step=AR_NOISE_STEP, workspace_dir=WORKSPACE)
    print()

    # Find crash icon
    icon = ICON_PATH if os.path.exists(ICON_PATH) else None

    print("Launching visualizer …")
    viz = ComparisonVisualizer(cfm_decoder, cfm_action_head, ar_wrapper, z1, context,
                               icon_path=icon)
    viz.run()


if __name__ == "__main__":
    main()
