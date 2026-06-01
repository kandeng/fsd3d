"""Integration tests for the fsd3d-3dgs example project.

Tests are split into:
  - Unit tests (no PLY, no GPU) — always run
  - Integration tests (need PLY file) — skip if PLY not available
  - GPU tests (need CUDA + PLY) — skip if no CUDA

Run with:
    cd examples/fsd3d-3dgs
    python -m pytest tests/ -v
"""

import numpy as np
import torch
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLY_PATH = Path(__file__).resolve().parents[2] / "asset" / "egglestone_abbey.ply"
HAS_PLY = PLY_PATH.exists()

skip_if_no_ply = pytest.mark.skipif(
    not HAS_PLY, reason=f"PLY file not found: {PLY_PATH}"
)
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ===========================================================================
# Unit tests — TelemetryFaker
# ===========================================================================

class TestTelemetryFaker:
    """Unit tests for TelemetryFaker — no PLY or GPU needed."""

    def test_output_shape(self):
        """fake() returns (N, 9) telemetry."""
        from fsd3d_3dgs.telemetry.faker import TelemetryFaker

        faker = TelemetryFaker(seed=0)
        waypoints = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
        ], dtype=np.float32)
        telem = faker.fake(waypoints)
        assert telem.shape == (5, 9), f"Expected (5, 9), got {telem.shape}"

    def test_position_has_noise(self):
        """Noisy positions differ from ground truth."""
        from fsd3d_3dgs.telemetry.faker import TelemetryFaker

        faker = TelemetryFaker(seed=1)
        waypoints = np.zeros((10, 3), dtype=np.float32)
        waypoints[:, 0] = np.arange(10)  # straight line along x
        telem = faker.fake(waypoints)

        # Position columns (0:3) should NOT match ground truth exactly
        pos_diff = np.abs(telem[:, :3] - waypoints)
        assert pos_diff.max() > 0.01, "Positions should have noise"

    def test_velocity_nonzero_for_moving_path(self):
        """Velocity should be non-zero when the drone is moving."""
        from fsd3d_3dgs.telemetry.faker import TelemetryFaker

        faker = TelemetryFaker(seed=2)
        waypoints = np.zeros((10, 3), dtype=np.float32)
        waypoints[:, 0] = np.arange(10)  # moving along x
        telem = faker.fake(waypoints)

        vel = telem[:, 3:6]
        assert np.abs(vel).max() > 0.1, "Velocity should be non-zero for a moving path"

    def test_reset_clears_cumulative_state(self):
        """reset() should clear velocity bias and orientation drift."""
        from fsd3d_3dgs.telemetry.faker import TelemetryFaker

        faker = TelemetryFaker(seed=3)
        waypoints = np.zeros((20, 3), dtype=np.float32)
        waypoints[:, 0] = np.arange(20)

        _ = faker.fake(waypoints)
        # After faking, cumulative state should have been reset (reset is called at start of fake)
        # Manually set some state and verify reset clears it
        faker._vel_bias = np.array([1.0, 2.0, 3.0])
        faker._orient_drift = np.array([0.5, 0.5, 0.5])
        faker.reset()
        assert np.allclose(faker._vel_bias, 0), "vel_bias should be zero after reset"
        assert np.allclose(faker._orient_drift, 0), "orient_drift should be zero after reset"

    def test_yaw_from_velocity_direction(self):
        """Yaw should follow the velocity direction (atan2)."""
        from fsd3d_3dgs.telemetry.faker import TelemetryFaker

        faker = TelemetryFaker(seed=4, orient_drift_sigma=0.0, orient_jitter_sigma=0.0)
        # Move along +y → yaw should be ~π/2
        waypoints = np.zeros((10, 3), dtype=np.float32)
        waypoints[:, 1] = np.arange(10).astype(np.float32)
        telem = faker.fake(waypoints)

        yaw = telem[5, 8]  # yaw at mid-path
        assert abs(yaw - np.pi / 2) < 0.3, f"Yaw should be ~pi/2 for +y motion, got {yaw}"

    def test_single_waypoint(self):
        """fake() handles N=1 gracefully."""
        from fsd3d_3dgs.telemetry.faker import TelemetryFaker

        faker = TelemetryFaker(seed=5)
        waypoints = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        telem = faker.fake(waypoints)
        assert telem.shape == (1, 9)


# ===========================================================================
# Unit tests — A* planner (no PLY needed, use synthetic voxel map)
# ===========================================================================

class TestAStarPlanner:
    """A* planner tests with a synthetic voxel map."""

    def _make_simple_voxel_map(self):
        """Create a small voxel map with no obstacles."""
        from fsd3d_3dgs.planner.voxel_map import VoxelMap
        from fsd3d_3dgs.scene.loader import SceneLoader

        # Create a minimal mock SceneLoader
        scene = SceneLoader.__new__(SceneLoader)
        scene._loaded = True
        scene.device = "cpu"
        scene.means = torch.zeros(1, 3)  # single Gaussian at origin
        scene.quats = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
        scene.scales = torch.ones(1, 3) * 0.01
        scene.opacities = torch.ones(1) * 0.1  # low opacity → won't be occupied
        scene.colors = torch.ones(1, 3) * 0.5
        scene.bbox_min = np.array([-5.0, -5.0, 0.0])
        scene.bbox_max = np.array([5.0, 5.0, 10.0])

        vm = VoxelMap(scene, resolution=1.0, opacity_threshold=0.5, drone_radius=0.0)
        vm.build()
        return vm

    def test_plan_returns_waypoints(self):
        """A* should return a list of waypoints."""
        from fsd3d_3dgs.planner.astar import AStarPlanner

        vm = self._make_simple_voxel_map()
        planner = AStarPlanner(vm, max_altitude=10.0)
        path = planner.plan(
            start=np.array([-3.0, -3.0, 0.0]),
            goal=np.array([3.0, 3.0, 0.0]),
            cruise_altitude=5.0,
        )
        assert len(path) > 0, "Path should not be empty"
        # First waypoint near start, last near goal
        assert np.linalg.norm(path[0] - np.array([-3, -3, 0])) < 2.0
        assert np.linalg.norm(path[-1] - np.array([3, 3, 0])) < 2.0

    def test_downsample_waypoints(self):
        """downsample_waypoints should return (target_count, 3)."""
        from fsd3d_3dgs.planner.astar import AStarPlanner

        waypoints = [np.array([i, i, 0.0]) for i in range(50)]
        downsampled = AStarPlanner.downsample_waypoints(waypoints, target_count=16)
        assert downsampled.shape == (16, 3), f"Expected (16, 3), got {downsampled.shape}"

    def test_downsample_few_waypoints(self):
        """downsample_waypoints pads when fewer waypoints than target."""
        from fsd3d_3dgs.planner.astar import AStarPlanner

        waypoints = [np.array([0, 0, 0.0]), np.array([1, 1, 1.0])]
        downsampled = AStarPlanner.downsample_waypoints(waypoints, target_count=16)
        assert downsampled.shape == (16, 3)
        # Last points should be same as the last waypoint
        np.testing.assert_allclose(downsampled[-1], [1, 1, 1], atol=1e-6)


# ===========================================================================
# Unit tests — §1 + §2 + §3 neural modules (no PLY, no GPU)
# ===========================================================================

class TestEncoderAndConditioner:
    """Test ViTEncoder, DomainAdapter, Conditioner, and ContextNormalizer shapes."""

    def test_vit_encoder_shape(self):
        """ViTEncoder: (B, 12, 224, 224) → (B, 196, 128)."""
        from fsd3d.encoder.vit_encoder import ViTEncoder
        from fsd3d.constants import NUM_PATCHES

        encoder = ViTEncoder()
        images = torch.randn(2, 12, 224, 224)
        with torch.no_grad():
            tokens = encoder(images)
        assert tokens.shape == (2, NUM_PATCHES, 128)

    def test_linear_domain_adapter(self):
        """LinearDomainAdapter: (B, T, 128) → (B, T, 128)."""
        from fsd3d.encoder.domain_adapter import LinearDomainAdapter

        adapter = LinearDomainAdapter()
        tokens = torch.randn(2, 196, 128)
        out = adapter(tokens)
        assert out.shape == (2, 196, 128)

    def test_conditioner_shape(self):
        """§2 Conditioner produces (B, 1+N_wp, 128) conditioning tokens."""
        from fsd3d.conditioner.conditioner import Conditioner

        cond = Conditioner()
        telemetry = torch.randn(2, 9)
        waypoints = torch.randn(2, 16, 3)
        with torch.no_grad():
            conditioning = cond(telemetry, waypoints)
        assert conditioning.shape == (2, 17, 128)  # 1 telem + 16 path

    def test_context_normalizer_shape(self):
        """§3 ContextNormalizer produces (B, 32, 128) from visual + conditioning."""
        from fsd3d.conditioner.normalizer import ContextNormalizer
        from fsd3d.constants import CONTEXT_TOKENS

        normalizer = ContextNormalizer()
        visual_tokens = torch.randn(2, 196, 128)
        conditioning_tokens = torch.randn(2, 17, 128)
        with torch.no_grad():
            context = normalizer(visual_tokens, conditioning_tokens)
        assert context.shape == (2, CONTEXT_TOKENS, 128)

    def test_context_normalizer_truncation(self):
        """§3 ContextNormalizer truncates when token count > 32."""
        from fsd3d.conditioner.normalizer import ContextNormalizer
        from fsd3d.constants import CONTEXT_TOKENS

        normalizer = ContextNormalizer()
        # 196 visual + 17 conditioning = 213 tokens → truncated to 32
        visual_tokens = torch.randn(2, 196, 128)
        conditioning_tokens = torch.randn(2, 17, 128)
        with torch.no_grad():
            context = normalizer(visual_tokens, conditioning_tokens)
        assert context.shape == (2, CONTEXT_TOKENS, 128)

    def test_context_normalizer_padding(self):
        """§3 ContextNormalizer pads when token count < 32."""
        from fsd3d.conditioner.normalizer import ContextNormalizer
        from fsd3d.constants import CONTEXT_TOKENS

        normalizer = ContextNormalizer()
        # 5 visual + 3 conditioning = 8 tokens → padded to 32
        visual_tokens = torch.randn(2, 5, 128)
        conditioning_tokens = torch.randn(2, 3, 128)
        with torch.no_grad():
            context = normalizer(visual_tokens, conditioning_tokens)
        assert context.shape == (2, CONTEXT_TOKENS, 128)
        # Padded positions should be zero
        assert torch.allclose(context[:, 8:, :], torch.zeros_like(context[:, 8:, :]))

    def test_full_encoder_conditioner_normalizer_pipeline(self):
        """Full pipeline: §1 encoder + §2 conditioner + §3 normalizer → (B, 32, 128)."""
        from fsd3d.encoder.vit_encoder import ViTEncoder
        from fsd3d.encoder.domain_adapter import LinearDomainAdapter
        from fsd3d.conditioner.conditioner import Conditioner
        from fsd3d.conditioner.normalizer import ContextNormalizer
        from fsd3d.constants import CONTEXT_TOKENS

        encoder = ViTEncoder()
        adapter = LinearDomainAdapter()
        conditioner = Conditioner()
        normalizer = ContextNormalizer()

        B = 1
        images = torch.randn(B, 12, 224, 224)
        telemetry = torch.randn(B, 9)
        waypoints = torch.randn(B, 16, 3)

        with torch.no_grad():
            visual_tokens = encoder(images)
            adapted_tokens = adapter(visual_tokens)
            conditioning_tokens = conditioner(telemetry, waypoints)
            context = normalizer(adapted_tokens, conditioning_tokens)

        assert context.shape == (B, CONTEXT_TOKENS, 128)


# ===========================================================================
# Unit tests — decoder with synthetic context
# ===========================================================================

class TestDecoderWithContext:
    """Test §4 decoder + §5 ActionHead with synthetic context."""

    def test_cfm_forward(self):
        """CFM forward: decoder + action_head produce (B, 16, 2)."""
        from fsd3d.decoder.transformer import FSD3DTransformerDecoder
        from fsd3d.decoder.action_head import ActionHead
        from fsd3d.constants import HORIZON, CONTEXT_TOKENS

        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()

        B = 2
        z_tau = torch.randn(B, HORIZON, 2)
        tau = torch.rand(B, 1)
        context = torch.randn(B, CONTEXT_TOKENS, 128)

        with torch.no_grad():
            latent = decoder(z_tau, tau, context)
            actions = action_head(latent)

        assert latent.shape == (B, HORIZON, 128)
        assert actions.shape == (B, HORIZON, 2)

    def test_action_projection_shape(self):
        """§5 ActionProjection: (B, T, 2) → (B, T, 128)."""
        from fsd3d.decoder.action_projection import ActionProjection

        proj = ActionProjection()
        z_tau = torch.randn(2, 16, 2)
        q_tokens = proj(z_tau)
        assert q_tokens.shape == (2, 16, 128)

    def test_decoder_with_full_pipeline(self):
        """Full pipeline: §1 encoder → §2 conditioner → §3 normalizer → §4 decoder → §5 action_head."""
        from fsd3d.encoder.vit_encoder import ViTEncoder
        from fsd3d.encoder.domain_adapter import LinearDomainAdapter
        from fsd3d.conditioner.conditioner import Conditioner
        from fsd3d.conditioner.normalizer import ContextNormalizer
        from fsd3d.decoder.transformer import FSD3DTransformerDecoder
        from fsd3d.decoder.action_head import ActionHead
        from fsd3d.constants import HORIZON, CONTEXT_TOKENS

        encoder = ViTEncoder()
        adapter = LinearDomainAdapter()
        conditioner = Conditioner()
        normalizer = ContextNormalizer()
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()

        B = 1
        images = torch.randn(B, 12, 224, 224)
        telemetry = torch.randn(B, 9)
        waypoints = torch.randn(B, 16, 3)
        z_tau = torch.randn(B, HORIZON, 2)
        tau = torch.rand(B, 1)

        with torch.no_grad():
            visual_tokens = encoder(images)
            adapted_tokens = adapter(visual_tokens)
            conditioning_tokens = conditioner(telemetry, waypoints)
            context = normalizer(adapted_tokens, conditioning_tokens)
            latent = decoder(z_tau, tau, context)
            actions = action_head(latent)

        assert context.shape == (B, CONTEXT_TOKENS, 128)
        assert actions.shape == (B, HORIZON, 2)


# ===========================================================================
# Integration tests — need PLY file (skip if not available)
# ===========================================================================

@skip_if_no_ply
class TestThreeDGSPlugin:
    """Integration tests for ThreeDGSPlugin — requires the PLY scene file."""

    def _make_plugin(self):
        """Create a ThreeDGSPlugin with a simple flight plan."""
        from fsd3d_3dgs.plugin.three_dgs import ThreeDGSPlugin

        return ThreeDGSPlugin(
            ply_path=str(PLY_PATH),
            start=np.array([-2.0, -2.0, 0.0]),
            goal=np.array([2.0, 2.0, 0.0]),
            cruise_altitude=5.0,
            max_altitude=10.0,
            device="cpu",
        )

    def test_build_context_shape(self):
        """build_context() returns (1, 32, 128) with no grad."""
        plugin = self._make_plugin()
        context = plugin.build_context()
        assert context.shape == (1, 32, 128), f"Got {context.shape}"
        assert not context.requires_grad

    def test_build_target_plans_shape(self):
        """build_target_plans() returns (2, 16, 2) normalized tensor."""
        plugin = self._make_plugin()
        plans = plugin.build_target_plans()
        assert plans.shape == (2, 16, 2), f"Got {plans.shape}"
        # Values should be normalized (roughly in [-1, 1] range)
        assert plans.abs().max() < 5.0, "Plans should be normalized"

    def test_get_pillar_params_keys(self):
        """get_pillar_params() returns dict with required keys."""
        plugin = self._make_plugin()
        params = plugin.get_pillar_params()
        assert "center_x" in params
        assert "center_y" in params
        assert "radius" in params
        assert params["radius"] >= 0

    def test_pipeline_with_decoder(self):
        """Full pipeline: 3DGS plugin → decoder → action_head."""
        from fsd3d.decoder.transformer import FSD3DTransformerDecoder
        from fsd3d.decoder.action_head import ActionHead
        from fsd3d.constants import HORIZON

        plugin = self._make_plugin()
        context = plugin.build_context()  # (1, 32, 128)

        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()

        z_tau = torch.randn(1, HORIZON, 2)
        tau = torch.rand(1, 1)

        with torch.no_grad():
            latent = decoder(z_tau, tau, context)
            actions = action_head(latent)

        assert actions.shape == (1, HORIZON, 2)
