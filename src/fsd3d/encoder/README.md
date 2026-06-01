# §1 Encoder, §2 Conditioner & §3 Data Bridge

This directory contains the **§1 Pilot Space** (ViT encoder), the **§2 Conditioning** module, and the **§3 Data Bridge** (VisualAdapter + ContextNormalizer). Together, they convert raw perception data into the context memory bank `(K, V)` that the §4 FSD3D Transformer Decoder queries via cross-attention.

> **Scope note**: The conditioner lives at `src/fsd3d/conditioner/` and the data bridge lives at `src/fsd3d/data_bridge/`, but all three are documented here because §1, §2, and §3 form a single logical pipeline — their outputs are combined by §3 before being consumed by §4.

---

## 1. Goal and System Role

![FSD3D Overall Architecture](../../../image/fsd3d_overall_architecture_02.png)

*Figure: FSD3D Architecture — Latent Flight Generation & Control Flow Blueprint. §1 (Pilot Space) produces visual tokens from raw 2D video via the ViT encoder, adapted by the VisualAdapter. §2 (Conditioning) fuses telemetry and `A*` guidance. §3 (Data Bridge) merges §1 + §2 outputs via ContextNormalizer, adds source ID embedding, and normalizes to exactly 32 tokens as K, V, while Q originates from z_tau via §5 Action Projection. §4 (Latent & Flight Generation) runs the transformer decoder with cross-attention. §5 (Action Loop) projects the decoder output to a 16×4 trajectory horizon matrix.*

### §1 Encoder — Pilot Space

The encoder converts **raw 2D video frames** into a sequence of **visual tokens** that serve as the **Key (K) and Value (V)** inputs (after §3 normalization) to the §4 decoder's cross-attention mechanism.

Key design principles:

- **Source-agnostic**: The ViT encoder processes `(B, C*N_stack, H, W)` image tensors regardless of whether the pixels came from 3DGS rendering, Google Earth, or a real camera. It has no concept of "pillar" or "obstacle" — the environment is implicitly encoded in the token sequence.
- **Self-attention only**: Unlike the decoder, the encoder uses no cross-attention, no time conditioning, and no causal mask. It compresses the visual scene through self-attention alone.
- **Mirrors decoder architecture but simpler**: 2 layers (vs. decoder's 3), same `d_model=128`, `nhead=4`, `dim_feedforward=512`, Post-LN convention.
- **Domain adaptation**: A separate `VisualAdapter` module (in `fsd3d.data_bridge`) compensates for domain shift between data sources (3DGS pixels vs. real camera pixels). This is where source-specific knowledge enters the pipeline.

### §2 Conditioner — Conditioning

The conditioner assembles conditioning tokens from two non-visual data streams:

| Input | Encoder | Rationale |
|-------|---------|----------|
| **Telemetry** (9 scalars: x, y, z, vx, vy, vz, roll, pitch, yaw) | `TelemetryEncoder` (MLP) | A handful of scalars with no sequence structure — an MLP suffices |
| **A\* Guidance** (N waypoints × 3 coordinates) | `PathEncoder` (1-layer transformer) | Waypoints form a spatial sequence with meaningful relationships (continuity, direction changes) |

The conditioner's output is a `(B, 1+N_wp, 128)` conditioning token sequence, which is then passed (along with §1 visual tokens) to the §3 ContextNormalizer in `fsd3d.data_bridge`.

### §3 Data Bridge — The Data Bridge

The Data Bridge package (`fsd3d.data_bridge`) contains two modules that bridge §1 and §2 into the §4 decoder:

**VisualAdapter** adapts visual tokens across data sources:
- Abstract base class `VisualAdapter` — pass-through by default
- `LinearVisualAdapter` — learned linear projection for domain shift compensation

**ContextNormalizer** merges and normalizes to exactly 32 tokens:

1. **Concatenates**: visual tokens (from §1 VisualAdapter) + conditioning tokens (from §2 Conditioner)
2. **Adds source ID embedding**: a learned vector added to all tokens, distinguishing data sources during multi-domain training
3. **Projects**: `Linear(d_model, d_model)` shared projection
4. **Truncates/pads** to fixed `CONTEXT_TOKENS = 32` length

The output is a `(B, 32, 128)` context tensor — this is the **K, V memory bank** for §4's cross-attention.

### Critical insight: where does Q come from?

The Query (Q) fed into §4's cross-attention does **not** come from §1, §2, or §3. Q originates from `z_tau`, the current state of the generative process (either the noise canvas in CFM or the partially-generated sequence in AR). The §5 `ActionProjection` module maps `z_tau` into `d_model`-dimensional tokens that serve as Q, while the context from §1+§2+§3 serves as K, V. The decoder is **input-agnostic** — it receives abstract numerical tensors and has no knowledge of what they physically represent.

---

## 2. File Structure and Diagram Mapping

### `src/fsd3d/encoder/` (§1 — this directory)

| File | Component in Diagram | Description |
|------|---------------------|-------------|
| `vit_encoder.py` | **2D-ViT Encoder (E)** in §1 | `ViTEncoder` — patch embedding → positional encoding → 2-layer `TransformerEncoder` → LayerNorm. Input: `(B, 12, 224, 224)` stacked frames. Output: `(B, 196, 128)` visual tokens. |
| `domain_adapter.py` | **[Shim]** | Re-exports `VisualAdapter` as `DomainAdapter` and `LinearVisualAdapter` as `LinearDomainAdapter` for backward compatibility. New code should use `fsd3d.data_bridge.visual_adapter` directly. |
| `__init__.py` | — | Exports `ViTEncoder`. Re-exports `DomainAdapter`, `LinearDomainAdapter` from data_bridge. |

### `src/fsd3d/data_bridge/` (§3 — Data Bridge)

| File | Component in Diagram | Description |
|------|---------------------|-------------|
| `visual_adapter.py` | **VisualAdapter** (between §1 output and K,V) | `VisualAdapter` (abstract base, pass-through) and `LinearVisualAdapter` (learned linear projection). Adapts visual tokens across data sources. |
| `context_normalizer.py` | **§3 Context Normalization** | `ContextNormalizer` — merges §1 visual + §2 conditioning tokens, adds source ID embedding, projects, and truncates/pads to (B, 32, 128) context. |
| `__init__.py` | — | Exports `VisualAdapter`, `LinearVisualAdapter`, `ContextNormalizer`. |

### `src/fsd3d/conditioner/` (§2)

| File | Component in Diagram | Description |
|------|---------------------|-------------|
| `telemetry_encoder.py` | **TelemetryEncoder** in §2 | `TelemetryEncoder` (MLP: 9→128→128). Encodes telemetry scalars into a d_model token. |
| `path_encoder.py` | **PathEncoder** (within §2 A* Guidance) | `PathEncoder` (1-layer transformer: N×3 → N×128). Encodes waypoint sequences. |
| `conditioner.py` | **Concatenation (+)** in §2 | `Conditioner` — orchestrates TelemetryEncoder + PathEncoder, concatenates outputs to (B, 1+N_wp, 128) conditioning tokens. |
| `normalizer.py` | **[Shim]** | Re-exports `ContextNormalizer` from `fsd3d.data_bridge` for backward compatibility. |
| `__init__.py` | — | Exports `TelemetryEncoder`, `PathEncoder`, `Conditioner`. Re-exports `ContextNormalizer` from data_bridge. |

### `src/fsd3d/constants.py` (shared)

All hyperparameters shared across §1, §2, §3, §4, §5 are defined here to avoid circular imports:

| Constant | Value | Used by |
|----------|-------|---------|
| `D_MODEL` | 128 | §1, §2, §3, §4, §5 |
| `NHEAD` | 4 | §1, §2, §4 |
| `DIM_FEEDFORWARD` | 512 | §1, §2, §4 |
| `ENCODER_LAYERS` | 2 | §1 |
| `DECODER_LAYERS` | 3 | §4 |
| `CONTEXT_TOKENS` | 32 | §3 |
| `ACTION_DIM` | 2 | §5 |
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
Raw 2D Video ──► ViTEncoder ──► VisualAdapter ──┐
                                                 ├─► ContextNormalizer (§3) ──► Context (K, V)
Telemetry ─────► TelemetryEncoder ────────────────┤    Source ID + Projection        (1, 32, 128)
                                                 │    + Truncate/Pad
A* Waypoints ──► PathEncoder ───────────────────┘
                                                       │
                                                       ▼
                                             FSD3D Transformer Decoder (§4)
                                             Q from z_tau (§5), K/V from context
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
python -c "from fsd3d.encoder import ViTEncoder; print('§1 OK')"
python -c "from fsd3d.conditioner import Conditioner, TelemetryEncoder, PathEncoder; print('§2 OK')"
python -c "from fsd3d.data_bridge import VisualAdapter, LinearVisualAdapter, ContextNormalizer; print('§3 OK')"
```

Dependencies (installed automatically via `pyproject.toml`):
- `torch` — neural network modules
- `numpy` — numerical operations
- `matplotlib` — visualization (used by other sub-packages)

---

## 4. Usage Example: fsd3d-3dgs

The `examples/fsd3d-3dgs/` project demonstrates how to use the `fsd3d` package with 3D Gaussian Splatting (3DGS) as the data source. It implements the full pipeline: load a 3DGS scene → plan an A* path → render video frames → feed through §1, §2, and §3 → produce context for §4.

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
from fsd3d.encoder import ViTEncoder
from fsd3d.data_bridge import LinearVisualAdapter, ContextNormalizer
from fsd3d.conditioner import Conditioner

# §1: Encode video frames
encoder = ViTEncoder()            # (B, 12, 224, 224) → (B, 196, 128)
adapter = LinearVisualAdapter()   # (B, 196, 128) → (B, 196, 128)

images = torch.randn(1, 12, 224, 224)  # 4 stacked RGB frames
visual_tokens = encoder(images)          # (1, 196, 128)
adapted_tokens = adapter(visual_tokens)  # (1, 196, 128)

# §2: Condition on telemetry + path
conditioner = Conditioner()

telemetry = torch.randn(1, 9)    # [x, y, z, vx, vy, vz, roll, pitch, yaw]
waypoints = torch.randn(1, 16, 3) # 16 A* waypoints (x, y, z)

conditioning_tokens = conditioner(telemetry, waypoints)
# → (1, 17, 128) — 1 telem + 16 path

# §3: Normalize to fixed-length context (from data_bridge)
normalizer = ContextNormalizer()
context = normalizer(adapted_tokens, conditioning_tokens)
# → (1, 32, 128) — ready for §4 decoder
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

# §1 + §2 + §3: Build context tensor
context = plugin.build_context()        # (1, 32, 128), no grad

# Expert trajectories for §4 training
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
7. **`LinearVisualAdapter`** (§3) — adapts visual tokens for domain shift
8. **`Conditioner`** (§2) — telemetry + waypoints → conditioning tokens `(1, 17, 128)`
9. **`ContextNormalizer`** (§3) — visual + conditioning → context `(1, 32, 128)`

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
│   └── egglestone_abbey.ply                # 3DGS scene (142MB, download from https://superspl.at/scene/67ba224d)
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
| §3 `ContextNormalizer` forward pass | No | Single Linear + truncate/pad, trivially small |
| §4 Decoder training (CFM/AR) | **Yes** | Backpropagation through 3-layer transformer, thousands of epochs |
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

# 3. Download the 3DGS scene asset (~142MB, too large for git)
wget -O asset/egglestone_abbey.ply https://superspl.at/scene/67ba224d
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

# Build 3DGS context (renders frames via gsplat → §1 + §2 + §3 → context tensor)
plugin = ThreeDGSPlugin(
    ply_path="asset/egglestone_abbey.ply",
    start=np.array([-2.0, -2.0, 0.0]),
    goal=np.array([2.0, 2.0, 0.0]),
    device="cuda",
)
context = plugin.build_context()        # (1, 32, 128)
plans = plugin.build_target_plans()      # (2, 16, 2)

# Train §4 decoder with CFM (GPU required for backprop)
train_cfm(context, plans, num_epochs=5000, lr=1e-4, device="cuda")
```

### 5.5 Run integration tests with PLY

On the GPU server with the PLY file in place:

```bash
cd examples/fsd3d-3dgs
python -m pytest tests/ -v
# Expected: 19 unit tests + 4 PLY integration tests all pass
```

On a CPU-only machine, the 4 PLY tests are automatically skipped, and only the 19 unit tests run.
