"""Tests for the data synthesis engine."""

import numpy as np
import torch
import pytest
from fsd3d.constants import (
    HORIZON, ACTION_DIM, CONTEXT_TOKENS, D_MODEL, TRAJECTORY_SCALE, NUM_EXPERTS,
)
from fsd3d.decoder.context import (
    ContextAssembler, compute_spatial_trajectory, denormalize_trajectory,
)
from fsd3d.plugin.mock import MockPlugin, PILLAR_CENTER_X, PILLAR_CENTER_Y, PILLAR_RADIUS


# Use a shared MockPlugin instance for data generation
_plugin = MockPlugin()


def build_target_plan(direction):
    """Build a single (16, 2) target plan via MockPlugin."""
    plans = _plugin.build_target_plans()
    return plans[0] if direction == "left" else plans[1]


def build_target_plans():
    """Build (2, 16, 2) target plans via MockPlugin."""
    return _plugin.build_target_plans()


def build_context():
    """Build (1, 32, 128) context via MockPlugin."""
    return _plugin.build_context()


def build_synthetic_data():
    """Build (z1, context) via ContextAssembler."""
    assembler = ContextAssembler()
    return assembler.build_synthetic_data()


class TestTargetPlanLeft:

    def test_z1_shape(self):
        z1 = build_target_plan("left")
        assert z1.shape == (16, 2)

    def test_z1_starts_at_origin(self):
        z1 = build_target_plan("left")
        assert z1[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert z1[0, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_z1_straight_up_before_dodge(self):
        z1 = build_target_plan("left")
        for t in range(5):
            assert abs(z1[t, 0].item()) < 1e-5, f"Step {t} x should be ≈0"

    def test_z1_dodge_goes_left(self):
        z1 = build_target_plan("left")
        for t in range(5, 11):
            assert z1[t, 0].item() < -0.01, f"Step {t} x should be negative (left dodge)"

    def test_z1_straight_up_after_dodge(self):
        z1 = build_target_plan("left")
        for t in range(11, 16):
            assert abs(z1[t, 0].item()) < 1e-5, f"Step {t} x should be ≈0"

    def test_z1_monotonically_increasing_y(self):
        z1 = build_target_plan("left")
        for t in range(1, 16):
            assert z1[t, 1].item() > z1[t - 1, 1].item(), f"Step {t} y should increase"

    def test_z1_avoids_pillar(self):
        z1 = build_target_plan("left")
        z1_real = z1.numpy() * TRAJECTORY_SCALE
        for t in range(16):
            dx = z1_real[t, 0] - PILLAR_CENTER_X
            dy = z1_real[t, 1] - PILLAR_CENTER_Y
            dist = np.sqrt(dx**2 + dy**2)
            assert dist > PILLAR_RADIUS, f"Step {t} collides with pillar (dist={dist:.3f})"

    def test_z1_normalized_range(self):
        z1 = build_target_plan("left")
        assert z1[:, 1].min().item() >= -0.01
        assert z1[:, 1].max().item() <= 1.01


class TestTargetPlanRight:

    def test_z1_shape(self):
        z1 = build_target_plan("right")
        assert z1.shape == (16, 2)

    def test_z1_starts_at_origin(self):
        z1 = build_target_plan("right")
        assert z1[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert z1[0, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_z1_straight_up_before_dodge(self):
        z1 = build_target_plan("right")
        for t in range(5):
            assert abs(z1[t, 0].item()) < 1e-5, f"Step {t} x should be ≈0"

    def test_z1_dodge_goes_right(self):
        z1 = build_target_plan("right")
        for t in range(5, 11):
            assert z1[t, 0].item() > 0.01, f"Step {t} x should be positive (right dodge)"

    def test_z1_straight_up_after_dodge(self):
        z1 = build_target_plan("right")
        for t in range(11, 16):
            assert abs(z1[t, 0].item()) < 1e-5, f"Step {t} x should be ≈0"

    def test_z1_monotonically_increasing_y(self):
        z1 = build_target_plan("right")
        for t in range(1, 16):
            assert z1[t, 1].item() > z1[t - 1, 1].item(), f"Step {t} y should increase"

    def test_z1_avoids_pillar(self):
        z1 = build_target_plan("right")
        z1_real = z1.numpy() * TRAJECTORY_SCALE
        for t in range(16):
            dx = z1_real[t, 0] - PILLAR_CENTER_X
            dy = z1_real[t, 1] - PILLAR_CENTER_Y
            dist = np.sqrt(dx**2 + dy**2)
            assert dist > PILLAR_RADIUS, f"Step {t} collides with pillar (dist={dist:.3f})"

    def test_left_right_are_mirrors(self):
        z1_left = build_target_plan("left")
        z1_right = build_target_plan("right")
        for t in range(16):
            assert z1_left[t, 0].item() == pytest.approx(-z1_right[t, 0].item(), abs=1e-5)
            assert z1_left[t, 1].item() == pytest.approx(z1_right[t, 1].item(), abs=1e-5)


class TestTargetPlans:

    def test_target_plans_shape(self):
        z1 = build_target_plans()
        assert z1.shape == (NUM_EXPERTS, 16, 2)

    def test_target_plans_contains_both(self):
        z1 = build_target_plans()
        z1_left = build_target_plan("left")
        z1_right = build_target_plan("right")
        torch.testing.assert_close(z1[0], z1_left)
        torch.testing.assert_close(z1[1], z1_right)


class TestContext:

    def test_context_shape(self):
        context = build_context()
        assert context.shape == (1, 32, 128)

    def test_context_no_grad(self):
        context = build_context()
        assert not context.requires_grad


class TestBuildSyntheticData:

    def test_returns_correct_shapes(self):
        z1, context = build_synthetic_data()
        assert z1.shape == (NUM_EXPERTS, 16, 2)
        assert context.shape == (1, 32, 128)


class TestSpatialTrajectory:

    def test_spatial_trajectory_shape(self):
        z1 = build_target_plan("left").numpy()
        pos = compute_spatial_trajectory(z1)
        assert pos.shape == (16, 2)

    def test_spatial_trajectory_starts_near_origin(self):
        z1 = build_target_plan("left").numpy()
        pos = compute_spatial_trajectory(z1)
        assert abs(pos[0, 0]) < 1e-6
        assert abs(pos[0, 1]) < 1e-6

    def test_spatial_trajectory_dodge_visible_left(self):
        z1 = build_target_plan("left").numpy()
        pos = compute_spatial_trajectory(z1)
        assert pos[7, 0] < -0.1

    def test_spatial_trajectory_dodge_visible_right(self):
        z1 = build_target_plan("right").numpy()
        pos = compute_spatial_trajectory(z1)
        assert pos[7, 0] > 0.1

    def test_denormalize_trajectory(self):
        z1 = build_target_plan("left").numpy()
        z1_real = denormalize_trajectory(z1)
        assert z1_real[-1, 1] > 9.0
        assert z1_real[7, 0] < -1.0
