# FSD3D — Full Self Driving for Drone in 3D Space

A modular Python package implementing the FSD3D architecture for trajectory generation, comparing **Conditional Flow Matching (CFM)** against **Autoregressive MSE** on flight-path planning tasks.

## System Architecture

![Overall Architecture](image/fsd3d_overall_architecture.png)

The FSD3D pipeline processes raw video and telemetry into actionable flight trajectories through four stages:

| Section | Role | Input | Output |
|---|---|---|---|
| **§1 Pilot Space** | Encode visual perception | Raw 2D video | 3D spatial latent tokens |
| **§2 Conditioning** | Fuse guidance signals | Telemetry + A* waypoints | Concatenated conditioning vector |
| **§3 Latent & Flight Generation** | Generate a clean flight plan | Noise z₀ + context (K, V) | Latent features via CFM ODE or AR roll-out |
| **§4 Action Loop** | Project to control commands | Latent features from §3 | Trajectory horizon matrix (action-space positions) |

§1 and §2 produce the **context** (key–value memory bank). §3 and §4 form the **decoder** — the trainable core. At inference, CFM resolves the whole trajectory at once via an ODE vector field, while AR predicts step-by-step and is susceptible to compounding drift.

## File Structure

```
fsd3d/
├── pyproject.toml                  # Package config, dependencies, pytest settings
├── .gitignore
├── README.md                       # This file
├── image/                          # Architecture diagrams + generated figures
├── asset/                          # 3D model assets (FBX, GLB, PLY)
├── workspace/                      # Training checkpoints (gitignored)
└── src/
    └── fsd3d/                      # pip-installable package
        ├── __init__.py             # Public API exports
        ├── constants.py            # Shared constants (HORIZON, D_MODEL, etc.)
        ├── encoder/                # §1 — Pilot Space
        │   ├── vit_encoder.py      #   ViT encoder (stub)
        │   └── domain_adapter.py   #   Source-specific adaptation (stub)
        ├── conditioner/            # §2 — Conditioning
        │   └── conditioner.py      #   Telemetry + A* conditioner (stub)
        ├── decoder/                # §3 + §4 — Latent Generation + Action Loop
        │   ├── transformer.py      #   §3 FSD3DTransformerDecoder
        │   ├── action_head.py      #   §4 ActionHead
        │   ├── autoregressive.py   #   AutoregressiveWrapper (causal mask, start token)
        │   ├── context.py          #   ContextAssembler, trajectory utilities
        │   ├── main.py             #   Train both paradigms → save checkpoints
        │   ├── visualize.py        #   Interactive side-by-side animation
        │   ├── visualize_training.py # Training progression figure
        │   ├── generate_diagram.py #   Static README illustration
        │   ├── README.md           #   Self-testing documentation
        │   └── tests/              #   52 pytest tests
        ├── inference/              # Inference routines
        │   ├── cfm.py              #   CFM Euler ODE solver
        │   └── autoregressive.py   #   AR step-by-step roll-out
        ├── training/               # Training loops
        │   ├── cfm.py              #   CFM velocity-field training
        │   └── autoregressive.py   #   AR teacher-forcing training
        ├── plugin/                 # Data-source abstraction
        │   ├── base.py             #   DataSourcePlugin interface
        │   └── mock.py             #   MockPlugin (synthetic pillar-dodge data)
        └── visualization/          # Visualization utilities
            ├── comparison.py       #   ComparisonVisualizer (matplotlib FuncAnimation)
            ├── training_progression.py # Snapshot-based training figure
            ├── diagram.py          #   Three-panel README diagram
            └── cfm_diffusion.py    #   CFM vs. Diffusion animated comparison
```

### Architecture → Code Mapping

| Architecture Section | Package | Status |
|---|---|---|
| §1 Pilot Space | `fsd3d.encoder` | Stub — to be implemented |
| §2 Conditioning | `fsd3d.conditioner` | Stub — to be implemented |
| §3 Latent & Flight Generation | `fsd3d.decoder.transformer` | **Implemented** |
| §4 Action Loop | `fsd3d.decoder.action_head` + `fsd3d.decoder.autoregressive` | **Implemented** |

### Sub-package Descriptions

#### `decoder/` — §3 + §4: Model Definitions & Entry Points

The decoder is the trainable core of the FSD3D pipeline. It contains the model definitions and the scripts to train, visualize, and test them.

- **`transformer.py`** — `FSD3DTransformerDecoder` (§3): a cross-attention decoder that takes noise z₀ (query) and context (key–value), conditioned on a continuous flow time τ, and outputs latent features.
- **`action_head.py`** — `ActionHead` (§4): a zero-init linear head that projects latent features to velocity vectors (CFM) or position predictions (AR). Zero-initialization ensures the model starts as an identity map.
- **`autoregressive.py`** — `AutoregressiveWrapper`: wraps the decoder with a causal mask and a learned start token. In training, it uses teacher forcing (full ground-truth input). At inference, it generates step-by-step via `generate()`.
- **`context.py`** — `ContextAssembler`: builds the (1, 32, 128) context tensor from a `DataSourcePlugin`. Also provides `compute_spatial_trajectory()` and `denormalize_trajectory()` for visualization.
- **`main.py`**, **`visualize.py`**, **`visualize_training.py`**, **`generate_diagram.py`** — entry-point scripts for training and visualization.

#### `training/` — Training Loops

Contains the two competing training paradigms. Both consume the same `(z1, context)` data but differ in objective and model pairing:

- **`cfm.py`** — `train_cfm()`: trains **decoder + ActionHead** jointly. Samples random τ ∈ [0, 1], interpolates z_τ = τ·z₁ + (1−τ)·z₀, and regresses the velocity field v = z₁ − z₀. Uses `AdamW` + cosine LR schedule.
- **`autoregressive.py`** — `train_autoregressive()`: trains the **AutoregressiveWrapper** end-to-end. Uses teacher forcing — feeds full ground-truth z₁ as input with a causal mask — and minimizes position-MSE against z₁. Fixed τ = 1 (no flow-time conditioning).

#### `inference/` — Inference Routines

The public-facing API that external applications (e.g. `fsd3d-flight`) call to produce trajectories from trained models. Each function encapsulates the generation loop so the caller does not need to know the internal stepping logic:

- **`cfm.py`** — `infer_cfm_euler()`: starting from noise z₀, integrates the learned velocity field over τ ∈ [0, 1] using Euler steps. Returns the final trajectory plus the full integration path (useful for visualization of the denoising process).
- **`autoregressive.py`** — `infer_autoregressive()`: delegates to the wrapper's `generate()` method, which auto-regressively predicts one position at a time. Supports optional noise injection after a given step and drift bias for robustness testing.

## Build

```bash
pip install -e .
```

Dependencies (declared in `pyproject.toml`): `torch>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`.

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Usage

*To be documented once §1 and §2 are implemented, enabling end-to-end inference from real sensor data.*

For now, the decoder can be exercised via the entry-point scripts in `src/fsd3d/decoder/` — see [decoder/README.md](src/fsd3d/decoder/README.md).

### CFM vs. Diffusion Visualization

The `cfm_diffusion.py` script produces an animated side-by-side comparison of **Conditional Flow Matching** against a **conventional Diffusion** process, illustrating how CFM follows optimal-transport paths while diffusion suffers from omnidirectional chaos and path crossover.

From the `src/fsd3d/visualization/` directory:

```bash
# Maneuver 0: Radial projection — particles converge to the nearest point on a circle
MANEUVER=0 python cfm_diffusion.py

# Maneuver 1: Horizontal flow — particles slide along a vertical line to a target column
MANEUVER=1 python cfm_diffusion.py
```

## Testing

Each sub-module that contains tests has its own `README.md` with self-testing instructions.

| Sub-module | Tests | README |
|---|---|---|
| `decoder/` (§3 + §4) | 52 tests — architecture correctness, data profile, CFM & AR pipelines | [decoder/README.md](src/fsd3d/decoder/README.md) |

Run all tests from the project root:

```bash
python -m pytest
```

> **Note:** On systems with the ROS `launch_testing` plugin installed, prepend
> `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to suppress it:
> ```
> PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
> ```
