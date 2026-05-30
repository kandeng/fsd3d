"""Tests for the Autoregressive MSE training pipeline and generation."""

import torch
import pytest
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.autoregressive import AutoregressiveWrapper
from fsd3d.decoder.context import ContextAssembler
from fsd3d.training.autoregressive import train_autoregressive
from fsd3d.inference.autoregressive import infer_autoregressive


@pytest.fixture
def synthetic_data():
    assembler = ContextAssembler()
    return assembler.build_synthetic_data()


class TestARTraining:

    def test_ar_loss_decreases(self, synthetic_data):
        """Loss at epoch 50 should be less than loss at epoch 1."""
        z1, context = synthetic_data
        torch.manual_seed(42)
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        wrapper = AutoregressiveWrapper(decoder, action_head)
        wrapper.train()

        batch_size = 32
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)

        losses = []
        for epoch in range(60):
            indices = torch.randint(0, z1.size(0), (batch_size,))
            z1_batch = z1[indices]
            ctx_batch = context.expand(batch_size, -1, -1)
            tau_fixed = torch.ones(batch_size, 1)

            optimizer.zero_grad()
            pred = wrapper(z1_batch, tau_fixed, ctx_batch)
            loss = torch.nn.functional.mse_loss(pred, z1_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[50] < losses[0], "AR loss should decrease over training"

    def test_causal_mask_no_future_leak(self, synthetic_data):
        """Output at position t does not change when future positions are altered."""
        z1, context = synthetic_data
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        wrapper = AutoregressiveWrapper(decoder, action_head)
        wrapper.eval()

        B = 2
        z1_left = z1[0]
        z1_batch = z1_left.expand(B, -1, -1)
        ctx_batch = context.expand(B, -1, -1)
        tau_fixed = torch.ones(B, 1)

        with torch.no_grad():
            out_orig = wrapper(z1_batch, tau_fixed, ctx_batch)

        z1_corrupted = z1_batch.clone()
        z1_corrupted[:, 6:, :] = 10.0

        with torch.no_grad():
            out_corr = wrapper(z1_corrupted, tau_fixed, ctx_batch)

        torch.testing.assert_close(
            out_orig[:, :6, :], out_corr[:, :6, :], atol=1e-5, rtol=1e-5
        )


class TestARInference:

    def test_start_token_shape(self):
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        wrapper = AutoregressiveWrapper(decoder, action_head)
        assert wrapper.start_token.shape == (1, 2)

    def test_generate_output_shape(self, synthetic_data):
        _, context = synthetic_data
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        wrapper = AutoregressiveWrapper(decoder, action_head)
        wrapper.eval()
        ctx = context.expand(1, -1, -1)
        result = wrapper.generate(ctx, horizon=16, noise_step=-1)
        assert result.shape == (1, 16, 2)

    def test_noise_causes_drift(self, synthetic_data):
        """AR inference with noise deviates from the clean prediction."""
        z1, context = synthetic_data

        torch.manual_seed(42)
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        wrapper = AutoregressiveWrapper(decoder, action_head)
        wrapper = train_autoregressive(wrapper, z1, context, epochs=150,
                                        batch_size=64, lr=1e-3, seed=42,
                                        log_interval=0)

        ctx = context.expand(1, -1, -1)

        torch.manual_seed(0)
        wrapper.eval()
        with torch.no_grad():
            result_clean = wrapper.generate(ctx, horizon=16, noise_step=-1, noise_sigma=0.0)

        with torch.no_grad():
            result_noisy = wrapper.generate(ctx, horizon=16, noise_step=4, noise_sigma=0.15,
                                             continuous_noise=True, drift_bias=0.0)

        drift = torch.nn.functional.mse_loss(result_noisy[0], result_clean[0]).item()
        assert drift > 1e-4, (
            f"Noisy prediction should differ from clean (drift={drift:.6f})"
        )

    def test_infer_autoregressive_wrapper(self, synthetic_data):
        """infer_autoregressive returns (1, 16, 2)."""
        z1, context = synthetic_data
        torch.manual_seed(42)
        decoder = FSD3DTransformerDecoder()
        action_head = ActionHead()
        wrapper = AutoregressiveWrapper(decoder, action_head)
        wrapper.eval()
        result = infer_autoregressive(wrapper, context, noise_step=-1, noise_sigma=0.0)
        assert result.shape == (1, 16, 2)
