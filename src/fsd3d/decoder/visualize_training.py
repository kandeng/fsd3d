"""Training progression visualizer.

Usage:
    cd src/fsd3d/decoder
    python visualize_training.py
"""

import os

from fsd3d.decoder.context import ContextAssembler
from fsd3d.visualization.training_progression import (
    train_cfm_with_snapshots, train_ar_with_snapshots,
    plot_training_progression,
)

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "image")
ICON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "asset", "failed.png")


def main():
    print("Building synthetic data …")
    assembler = ContextAssembler()
    z1, context = assembler.build_synthetic_data()
    print(f"  z1 shape: {z1.shape}")
    print()

    # CFM training with snapshots
    print("=" * 60)
    print("Training CFM model (with snapshots) …")
    print("=" * 60)
    cfm_snapshots = train_cfm_with_snapshots(
        z1, context,
        epochs=3000, batch_size=64, lr=1e-3, seed=42,
        snapshot_epochs=[0, 50, 150, 400, 800, 1500, 3000],
        num_samples=4,
    )
    print()

    # AR training with snapshots
    print("=" * 60)
    print("Training AR model (with snapshots) …")
    print("=" * 60)
    ar_snapshots = train_ar_with_snapshots(
        z1, context,
        epochs=300, batch_size=64, lr=1e-3, seed=42,
        snapshot_epochs=[0, 5, 15, 40, 100, 200, 300],
    )
    print()

    # Generate figure
    os.makedirs(IMAGE_DIR, exist_ok=True)
    out_path = os.path.join(IMAGE_DIR, "training_progression.png")
    icon = ICON_PATH if os.path.exists(ICON_PATH) else None

    print("Generating training progression figure …")
    plot_training_progression(cfm_snapshots, ar_snapshots, z1, out_path,
                              icon_path=icon)


if __name__ == "__main__":
    main()
