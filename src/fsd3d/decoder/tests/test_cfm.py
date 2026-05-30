"""Tests for the CFM training pipeline and Euler ODE solver."""

import torch
import pytest
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.context import ContextAssembler
from fsd3d.training.cfm import train_cfm
from fsd3d.inference.cfm import infer_cfm_euler


@pytest.fixture
def synthetic_data():
    assembler = ContextAssembler()
    return assembler.build_synthetic_data()


class TestCFMTraining:

    def test_cfm_loss_decreases(self, synthetic_data):
        """Loss at epoch 50 should be less than loss at epoch 1."""
        z1, context = synthetic_data
        torch.manual_seed(42)
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        decoder.train()
        action_head.train()

        batch_size = 32
        optimizer = torch.optim.AdamW(
            list(decoder.parameters()) + list(action_head.parameters()), lr=1e-3
        )

        losses = []
        for epoch in range(60):
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
            loss = torch.nn.functional.mse_loss(v_pred, v_target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[50] < losses[0], "CFM loss should decrease over training"

    def test_interpolation_formula(self, synthetic_data):
        """z_τ = τ·z1 + (1-τ)·z0 — manual check at τ=0.5."""
        z1, _ = synthetic_data
        z0 = torch.randn_like(z1)
        tau = 0.5
        z_tau = tau * z1 + (1 - tau) * z0
        expected = 0.5 * z1 + 0.5 * z0
        torch.testing.assert_close(z_tau, expected)


class TestCFMInference:

    def test_euler_trajectory_length(self, synthetic_data):
        """Euler solver returns euler_steps + 1 trajectory snapshots."""
        z1, context = synthetic_data
        torch.manual_seed(42)
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        _, trajectory = infer_cfm_euler(decoder, action_head, context, euler_steps=10, seed=0)
        assert len(trajectory) == 11

    def test_euler_solver_convergence(self, synthetic_data):
        """After sufficient training, Euler solver should approximate one of the experts."""
        z1, context = synthetic_data
        torch.manual_seed(42)
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        decoder, action_head = train_cfm(decoder, action_head, z1, context, epochs=500, batch_size=64,
                          lr=1e-3, seed=42, log_interval=0)

        z_final, _ = infer_cfm_euler(decoder, action_head, context, euler_steps=10, seed=123)

        mses = [torch.nn.functional.mse_loss(z_final[0], z1[i]).item() for i in range(z1.size(0))]
        mse = min(mses)
        assert mse < 0.5, f"CFM MSE too high: {mse:.4f} (expected < 0.5)"

    def test_euler_output_shape(self, synthetic_data):
        z1, context = synthetic_data
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        z_final, _ = infer_cfm_euler(decoder, action_head, context, horizon=16, euler_steps=10, seed=0)
        assert z_final.shape == (1, 16, 2)
