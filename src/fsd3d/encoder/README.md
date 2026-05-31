# §1 Encoder & §2 Conditioner

This directory contains the **§1 Pilot Space** (ViT encoder + domain adapter) and the closely related **§2 Conditioning** module. Together, they convert raw perception data into the context memory bank `(K, V)` that the §3 FSD3D Transformer Decoder queries via cross-attention.

> **Scope note**: The conditioner lives at `src/fsd3d/conditioner/`, but is documented here because §1 and §2 form a single logical pipeline — their outputs are concatenated before being consumed by §3.

---

## 1. Goal and System Role

Refer to the overall architecture diagram at [`image/fsd3d_overall_architecture.png`](../../../../image/fsd3d_overall_architecture.png):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FSD3D Architecture (simplified)                      │
│                                                                              │
│  §1 PILOT SPACE          §2 CONDITIONING         §3 GENERATION    §4 ACTION │
│  ┌──────────────┐       ┌──────────────────┐    ┌────────────┐  ┌─────────┐ │
│  │ Raw 2D Video │       │ Telemetry Data    │    │  z0 Noise  │  │ Action  │ │
│  │     │        │       │     │             │    │   (Q)      │  │ Head D  │ │
│  │     ▼        │       │     ▼             │    │     │      │  │    │    │ │
│  │ 2D-ViT      │       │ TelemetryEncoder  │    │     ▼      │  │    ▼    │ │
│  │ Encoder (E) │       │     │             │    │ FSD3D      │  │ 16×4    │ │
│  │     │        │       │ A* Guidance ──────┤    │ Transformer│  │ Horizon │ │
│  │     ▼        │       │     │  PathEncoder│    │ Decoder   │  │ Matrix  │ │
│  │ DomainAdapter│      │     ▼             │    │ (cross-attn│  │         │ │
│  │     │        │       │  Concatenation    │    │  K,V→Q)   │  │         │ │
│  │     ▼        │       │     │             │    │     │      │  │         │ │
│  └──────┼───────┘       └──────┼────────────┘    │  CFM/AR   │  │         │ │
│          │                      │                 │     │      │  │         │ │
│          └──────────┬──────────┘                 │     ▼      │  │         │ │
│                     ▼                            │   z1      │  │         │ │
│            Context as K & V ────────────────────►│           │  │         │ │
│              (1, 32, 128)                        └────────────┘  └─────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### §1 Encoder — Pilot Space

The encoder converts **raw 2D video frames** into a sequence of **visual tokens** that serve as the **Key (K) and Value (V)** inputs to the §3 decoder's cross-attention mechanism.

Key design principles:

- **Source-agnostic**: The ViT encoder processes `(B, C*N_stack, H, W)` image tensors regardless of whether the pixels came from 3DGS rendering, Google Earth, or a real camera. It has no concept of "pillar" or "obstacle" — the environment is implicitly encoded in the token sequence.
- **Self-attention only**: Unlike the decoder, the encoder uses no cross-attention, no time conditioning, and no causal mask. It compresses the visual scene through self-attention alone.
- **Mirrors decoder architecture but simpler**: 2 layers (vs. decoder's 3), same `d_model=128`, `nhead=4`, `dim_feedforward=512`, Post-LN convention.
- **Domain adaptation**: A separate `DomainAdapter` module compensates for domain shift between data sources (3DGS pixels vs. real camera pixels). This is where source-specific knowledge enters the pipeline.

### §2 Conditioner — Conditioning

The conditioner assembles conditioning tokens from two non-visual data streams:

| Input | Encoder | Rationale |
|-------|---------|-----------|
| **Telemetry** (9 scalars: x, y, z, vx, vy, vz, roll, pitch, yaw) | `TelemetryEncoder` (MLP) | A handful of scalars with no sequence structure — an MLP suffices |
| **A\* Guidance** (N waypoints × 3 coordinates) | `PathEncoder` (1-layer transformer) | Waypoints form a spatial sequence with meaningful relationships (continuity, direction changes) |

The conditioner then:
1. **Concatenates**: visual tokens (from §1) + 1 telemetry token + N path tokens
2. **Adds source ID embedding**: a learned vector added to all tokens, distinguishing data sources during multi-domain training
3. **Projects**: `Linear(d_model, d_model)` shared projection
4. **Truncates/pads** to fixed `CONTEXT_TOKENS = 32` length

The output is a `(B, 32, 128)` context tensor — this is the **K, V memory bank** for §3's cross-attention.

### Critical insight: where does Q come from?

The Query (Q) fed into §3's cross-attention does **not** come from §1 or §2. Q originates from `z_tau`, the current state of the generative process (either the noise canvas in CFM or the partially-generated sequence in AR). The decoder's `action_projection` maps `z_tau` into `d_model`-dimensional tokens that serve as Q, while the context from §1+§2 serves as K, V. The decoder is **input-agnostic** — it receives abstract numerical tensors and has no knowledge of what they physically represent.

---

## 2. File Structure and Diagram Mapping

### `src/fsd3d/encoder/` (§1 — this directory)

| File | Component in Diagram | Description |
|------|---------------------|-------------|
| `vit_encoder.py` | **2D-ViT Encoder (E)** in §1 | `ViTEncoder` — patch embedding → positional encoding → 2-layer `TransformerEncoder` → LayerNorm. Input: `(B, 12, 224, 224)` stacked frames. Output: `(B, 196, 128)` visual tokens. |
| `domain_adapter.py` | **DomainAdapter** (between §1 output and K,V) | `DomainAdapter` (abstract base, pass-through) and `LinearDomainAdapter` (learned linear projection). Compensates for domain shift between data sources. |
| `__init__.py` | — | Exports `ViTEncoder`, `DomainAdapter`, `LinearDomainAdapter`. |

### `src/fsd3d/conditioner/` (§2)

| File | Component in Diagram | Description |
|------|---------------------|-------------|
| `conditioner.py` | **TelemetryEncoder**, **PathEncoder**, **Concatenation (+)** in §2 | `TelemetryEncoder` (MLP: 9→128→128), `PathEncoder` (1-layer transformer: N×3 → N×128), `Conditioner` (concatenation + source ID + projection + truncate/pad to 32 tokens). Output: `(B, 32, 128)` context. |
| `__init__.py` | — | Exports `Conditioner`, `TelemetryEncoder`, `PathEncoder`. |

### `src/fsd3d/constants.py` (shared)

All hyperparameters shared across §1, §2, §3 are defined here to avoid circular imports:

| Constant | Value | Used by |
|----------|-------|---------|
| `D_MODEL` | 128 | §1, §2, §3 |
| `NHEAD` | 4 | §1, §2, §3 |
| `DIM_FEEDFORWARD` | 512 | §1, §2, §3 |
| `ENCODER_LAYERS` | 2 | §1 |
| `DECODER_LAYERS` | 3 | §3 |
| `CONTEXT_TOKENS` | 32 | §2, §3 |
| `PATCH_SIZE` | 16 | §1 |
| `IMAGE_SIZE` | 224 | §1 |
| `NUM_PATCHES` | 196 | §1 |
| `NUM_FRAMES_STACK` | 4 | §1 |
| `TELEMETRY_DIM` | 9 | §2 |
| `PATH_ENCODER_LAYERS` | 1 | §2 |

### `src/fsd3d/plugin/` (plugin interface)

| File | Component in Diagram | Description |
|------|---------------------|-------------|
| `base.py` | **DataSourcePlugin** interface | Abstract base class: `build_context()` → `(1, 32, 128)`, `build_target_plans()` → `(N, 16, 2)`, `get_pillar_params()` → `dict`. Domain-specific plugins (3DGS, Google Earth, real drone) implement this interface. |
| `mock.py` | — (testing) | `MockPlugin` — synthetic pillar-dodge data for decoder development without real §1/§2 inputs. |

### Data flow summary

```
Raw 2D Video ──► ViTEncoder ──► DomainAdapter ──┐
                                                  ├─► Concatenation ──► Projection ──► Context (K, V)
Telemetry ─────► TelemetryEncoder ───────────────┤                        (1, 32, 128)
                                                  │
A* Waypoints ──► PathEncoder ────────────────────┘
                                                        │
                                                        ▼
                                              FSD3D Transformer Decoder (§3)
                                              Q from z_tau, K/V from context
```

---

## 3. Installation

The encoder and conditioner are part of the `fsd3d` core package:

```bash
# Clone the repository
git clone https://github.com/<user>/fsd3d.git
cd fsd3d

# Install the core package (editable mode)
pip install -e .

# Verify installation
python -c "from fsd3d.encoder import ViTEncoder, DomainAdapter, LinearDomainAdapter; print('§1 OK')"
python -c "from fsd3d.conditioner import Conditioner, TelemetryEncoder, PathEncoder; print('§2 OK')"
```

Dependencies (installed automatically via `pyproject.toml`):
- `torch` — neural network modules
- `numpy` — numerical operations
- `matplotlib` — visualization (used by other sub-packages)

---

## 4. Usage Example: fsd3d-3dgs

The `examples/fsd3d-3dgs/` project demonstrates how to use the `fsd3d` package with 3D Gaussian Splatting (3DGS) as the data source. It implements the full pipeline: load a 3DGS scene → plan an A* path → render video frames → feed through §1 and §2 → produce context for §3.

### 4.1 Install the 3DGS example

```bash
cd examples/fsd3d-3dgs
pip install -e .
```

This pulls in additional dependencies: `gsplat`, `plyfile`, `scipy`, `opencv-python`.

### 4.2 Use §1 and §2 standalone

You can use the encoder and conditioner directly without the plugin system:

```python
import torch
from fsd3d.encoder import ViTEncoder, LinearDomainAdapter
from fsd3d.conditioner import Conditioner

# §1: Encode video frames
encoder = ViTEncoder()          # (B, 12, 224, 224) → (B, 196, 128)
adapter = LinearDomainAdapter() # (B, 196, 128) → (B, 196, 128)

images = torch.randn(1, 12, 224, 224)  # 4 stacked RGB frames
visual_tokens = encoder(images)          # (1, 196, 128)
adapted_tokens = adapter(visual_tokens)  # (1, 196, 128)

# §2: Condition on telemetry + path
conditioner = Conditioner()

telemetry = torch.randn(1, 9)    # [x, y, z, vx, vy, vz, roll, pitch, yaw]
waypoints = torch.randn(1, 16, 3) # 16 A* waypoints (x, y, z)

context = conditioner(telemetry, waypoints, adapted_tokens)
# → (1, 32, 128) — ready for §3 decoder
```

### 4.3 Use the ThreeDGSPlugin (full pipeline)

The `ThreeDGSPlugin` orchestrates the entire data pipeline automatically:

```python
import numpy as np
from fsd3d_3dgs.plugin import ThreeDGSPlugin

plugin = ThreeDGSPlugin(
    ply_path="asset/egglestone_abbey.ply",
    start=np.array([-2.0, -2.0, 0.0]),
    goal=np.array([2.0, 2.0, 0.0]),
    cruise_altitude=15.0,
    device="cuda",
)

# §1 + §2: Build context tensor
context = plugin.build_context()        # (1, 32, 128), no grad

# Expert trajectories for §3 training
plans = plugin.build_target_plans()      # (2, 16, 2), normalized

# Obstacle parameters for collision detection
pillar = plugin.get_pillar_params()      # {"center_x": ..., "center_y": ..., "radius": ...}
```

Under the hood, `ThreeDGSPlugin.build_context()` runs:

1. **`SceneLoader`** — parses the PLY file into gsplat tensors
2. **`VoxelMap`** — builds a 3D occupancy grid for A*
3. **`AStarPlanner`** — plans takeoff → cruise → landing path
4. **`TelemetryFaker`** — simulates GPS/IMU/barometer/orientation noise
5. **`SceneRenderer`** — renders RGB frames along the path via gsplat
6. **`ViTEncoder`** (§1) — frames → visual tokens `(1, 196, 128)`
7. **`LinearDomainAdapter`** — adapts visual tokens for domain shift
8. **`Conditioner`** (§2) — telemetry + waypoints + visual → context `(1, 32, 128)`

### 4.4 Train the decoder with 3DGS context

```python
from fsd3d.training.cfm import train_cfm

# Build context and plans from the plugin
context = plugin.build_context()    # (1, 32, 128)
plans = plugin.build_target_plans() # (2, 16, 2)

# Train with CFM
train_cfm(context, plans, num_epochs=5000, lr=1e-4, device="cuda")
```

### 4.5 3DGS example file structure

```
examples/fsd3d-3dgs/
├── pyproject.toml                          # Package config (depends on fsd3d)
├── asset/
│   └── egglestone_abbey.ply                # 3DGS scene (142MB, not in git)
├── src/fsd3d_3dgs/
│   ├── scene/
│   │   ├── loader.py                       # PLY → gsplat tensors
│   │   └── renderer.py                     # gsplat rasterization
│   ├── planner/
│   │   ├── voxel_map.py                    # 3D occupancy grid
│   │   └── astar.py                        # A* 3D path planner
│   ├── telemetry/
│   │   └── faker.py                        # Sensor noise simulation
│   └── plugin/
│       └── three_dgs.py                    # ThreeDGSPlugin (implements DataSourcePlugin)
└── tests/
    └── test_integration.py                 # Unit + integration tests
```

> **Note**: The voxel map and A* planner are **not** part of the encoder pipeline. The encoder learns spatial structure implicitly through self-attention on image patches. The voxel map is exclusively a utility for the A* planner to navigate the 3D scene. The encoder never sees the occupancy grid — it only sees rendered RGB frames.

---

## 5. GPU Server Tasks

Several tasks in this pipeline require a CUDA-capable GPU and cannot run on CPU-only machines. The table below summarizes which tasks need GPU, why, and how to run them.

### 5.1 Which tasks require GPU?

| Task | GPU Required? | Reason |
|------|:---:|--------|
| §1 `ViTEncoder` forward pass | No | Small model (2 layers, 196 patches) runs fine on CPU for inference |
| §2 `Conditioner` forward pass | No | MLP + 1-layer transformer, trivially small |
| §3 Decoder training (CFM/AR) | **Yes** | Backpropagation through 3-layer transformer, thousands of epochs |
| 3DGS rendering (`gsplat`) | **Yes** | CUDA-only rasterization kernel — no CPU fallback |
| PLY scene loading | No | CPU-based file parsing via `plyfile` |
| Voxel map construction | No | CPU NumPy operations |
| A* path planning | No | CPU graph search |
| Telemetry faking | No | CPU NumPy operations |
| `ThreeDGSPlugin.build_context()` | **Yes** | Calls gsplat rendering internally |
| Unit tests (no PLY) | No | Pure PyTorch on small tensors |
| Integration tests (with PLY) | **Yes** | Calls `ThreeDGSPlugin` which renders via gsplat |

### 5.2 Setup on GPU server

```bash
# 1. Clone and install
 git clone https://github.com/<user>/fsd3d.git
cd fsd3d
pip install -e .

cd examples/fsd3d-3dgs
pip install -e .    # pulls in gsplat (CUDA), plyfile, scipy, opencv-python

# 2. Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')"
python -c "import gsplat; print('gsplat OK')"

# 3. Copy the PLY asset (too large for git, ~142MB)
cp /path/to/egglestone_abbey.ply asset/
```

### 5.3 Generate video footage along A* path

Rendering hundreds of frames through gsplat requires GPU. Run this on the server:

```python
# render_flight.py
import numpy as np
import cv2
import torch
from fsd3d_3dgs.scene import SceneLoader, SceneRenderer
from fsd3d_3dgs.planner import VoxelMap, AStarPlanner

# Load scene on GPU
scene = SceneLoader("asset/egglestone_abbey.ply", device="cuda").load()

# Plan path (CPU is fine, but scene must be loaded first)
vm = VoxelMap(scene, resolution=0.5).build()
planner = AStarPlanner(vm, max_altitude=30.0)
path = planner.plan(
    start=np.array([-2.0, -2.0, 0.0]),
    goal=np.array([2.0, 2.0, 0.0]),
    cruise_altitude=15.0,
)

# Render frames along path (GPU required for gsplat)
renderer = SceneRenderer(scene, width=640, height=480)
waypoints = np.array(path, dtype=np.float32)
frames = renderer.render_along_path(waypoints)  # (N, H, W, 3)

# Save as MP4 video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("flight.mp4", fourcc, 30, (640, 480))
for i in range(frames.shape[0]):
    frame = (frames[i].cpu().numpy() * 255).astype(np.uint8)
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
video.release()
print(f"Saved {frames.shape[0]} frames to flight.mp4")
```

### 5.4 Build context and train decoder

Both `ThreeDGSPlugin.build_context()` (which renders via gsplat) and decoder training require GPU:

```python
# train_3dgs.py
import numpy as np
import torch
from fsd3d_3dgs.plugin import ThreeDGSPlugin
from fsd3d.training.cfm import train_cfm

# Build 3DGS context (renders frames via gsplat → §1 + §2 → context tensor)
plugin = ThreeDGSPlugin(
    ply_path="asset/egglestone_abbey.ply",
    start=np.array([-2.0, -2.0, 0.0]),
    goal=np.array([2.0, 2.0, 0.0]),
    device="cuda",
)
context = plugin.build_context()        # (1, 32, 128)
plans = plugin.build_target_plans()      # (2, 16, 2)

# Train §3 decoder with CFM (GPU required for backprop)
train_cfm(context, plans, num_epochs=5000, lr=1e-4, device="cuda")
```

### 5.5 Run integration tests with PLY

On the GPU server with the PLY file in place:

```bash
cd examples/fsd3d-3dgs
python -m pytest tests/ -v
# Expected: 14 unit tests + 4 PLY integration tests all pass
```

On a CPU-only machine, the 4 PLY tests are automatically skipped, and only the 14 unit tests run.
