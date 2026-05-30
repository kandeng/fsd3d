"""Generate static README illustration diagram.

Usage:
    cd src/fsd3d/decoder
    python generate_diagram.py
"""

import os

from fsd3d.visualization.diagram import generate_diagram

WORKSPACE = os.path.join(os.path.dirname(__file__), "workspace")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "image")
ICON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "asset", "failed.png")


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    out_path = os.path.join(IMAGE_DIR, "pillar_dodge_comparison.png")
    icon = ICON_PATH if os.path.exists(ICON_PATH) else None

    generate_diagram(workspace_dir=WORKSPACE, output_path=out_path, icon_path=icon)


if __name__ == "__main__":
    main()
