"""Tests for FSD3DTransformerDecoder and AutoregressiveWrapper."""

import torch
import pytest
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.autoregressive import AutoregressiveWrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def model():
    return FSD3DTransformerDecoder()


@pytest.fixture
def action_head():
    return ActionHead()


@pytest.fixture
def wrapper(model, action_head):
    return AutoregressiveWrapper(model, action_head)


@pytest.fixture
def sample_inputs():
    B, T, S = 4, 16, 32
    z_tau = torch.randn(B, T, 2)
    tau = torch.rand(B, 1)
    context = torch.randn(B, S, 128)
    return z_tau, tau, context


# ---------------------------------------------------------------------------
# FSD3DTransformerDecoder tests
# ---------------------------------------------------------------------------
class TestFSD3DTransformerDecoder:

    def test_output_shape_cfm(self, model, sample_inputs):
        """Forward pass without causal mask produces (B, 16, d_model) latent."""
        z_tau, tau, context = sample_inputs
        out = model(z_tau, tau, context)
        assert out.shape == (4, 16, 128)

    def test_output_shape_with_causal_mask(self, model, sample_inputs):
        """Forward pass with causal mask produces (B, 16, d_model) latent."""
        z_tau, tau, context = sample_inputs
        T = z_tau.size(1)
        mask = AutoregressiveWrapper.generate_causal_mask(T)
        out = model(z_tau, tau, context, tgt_mask=mask)
        assert out.shape == (4, 16, 128)

    def test_causal_mask_blocks_future(self, model, sample_inputs):
        """Output at position t is unchanged when positions > t are zeroed."""
        z_tau, tau, context = sample_inputs
        T = z_tau.size(1)
        mask = AutoregressiveWrapper.generate_causal_mask(T)

        model.eval()
        with torch.no_grad():
            out_original = model(z_tau, tau, context, tgt_mask=mask)

            z_tau_modified = z_tau.clone()
            z_tau_modified[:, 6:, :] = 0.0

            out_modified = model(z_tau_modified, tau, context, tgt_mask=mask)

        torch.testing.assert_close(
            out_original[:, :6, :], out_modified[:, :6, :], atol=1e-5, rtol=1e-5
        )

    def test_time_embedding_effect(self, model, sample_inputs):
        """Outputs differ when τ changes — verified at the latent level."""
        z_tau, _, context = sample_inputs
        tau_0 = torch.zeros(z_tau.size(0), 1)
        tau_1 = torch.ones(z_tau.size(0), 1)

        latent_0 = model(z_tau, tau_0, context)
        latent_1 = model(z_tau, tau_1, context)

        assert not torch.allclose(latent_0, latent_1, atol=1e-5)

    def test_parameter_count(self, model):
        """Total parameters ≈ expected count for 3-layer 128-dim decoder."""
        total = sum(p.numel() for p in model.parameters())
        assert 700_000 < total < 900_000, f"Unexpected param count: {total}"


# ---------------------------------------------------------------------------
# AutoregressiveWrapper tests
# ---------------------------------------------------------------------------
class TestAutoregressiveWrapper:

    def test_start_token_shape(self, wrapper):
        assert wrapper.start_token.shape == (1, 2)

    def test_causal_mask_shape(self):
        mask = AutoregressiveWrapper.generate_causal_mask(16)
        assert mask.shape == (16, 16)

    def test_causal_mask_values(self):
        """Mask is 0 on and below diagonal, -inf above."""
        mask = AutoregressiveWrapper.generate_causal_mask(4)
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j] == 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] == float("-inf")

    def test_teacher_forcing_input_shape(self, wrapper):
        """prepare_teacher_forcing_input produces (B, T, action_dim)."""
        B, T = 4, 16
        z1 = torch.randn(B, T, 2)
        inp = wrapper.prepare_teacher_forcing_input(z1)
        assert inp.shape == (B, T, 2)

    def test_teacher_forcing_starts_with_token(self, wrapper):
        """First position of the teacher-forcing input is the start token."""
        B, T = 4, 16
        z1 = torch.randn(B, T, 2)
        inp = wrapper.prepare_teacher_forcing_input(z1)
        for b in range(B):
            torch.testing.assert_close(inp[b, 0, :], wrapper.start_token.squeeze(0))

    def test_forward_shape(self, wrapper, sample_inputs):
        """Wrapper forward pass produces (B, 16, action_dim)."""
        _, tau, context = sample_inputs
        B = tau.size(0)
        z1 = torch.randn(B, 16, 2)
        out = wrapper(z1, tau, context)
        assert out.shape == (B, 16, 2)

    def test_generate_output_shape(self, wrapper):
        """Autoregressive generate produces (1, 16, action_dim)."""
        ctx = torch.randn(1, 32, 128)
        result = wrapper.generate(ctx, horizon=16, noise_step=-1)
        assert result.shape == (1, 16, 2)


class TestActionHead:

    def test_action_head_output_shape(self, action_head):
        """ActionHead projects (B, T, d_model) → (B, T, action_dim)."""
        latent = torch.randn(4, 16, 128)
        out = action_head(latent)
        assert out.shape == (4, 16, 2)

    def test_action_head_zero_init(self, action_head):
        """Zero-initialised ActionHead produces near-zero outputs."""
        latent = torch.randn(4, 16, 128)
        out = action_head(latent)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_decoder_action_head_pipeline(self, model, action_head, sample_inputs):
        """Chaining §4 decoder → §5 ActionHead gives (B, T, action_dim)."""
        z_tau, tau, context = sample_inputs
        latent = model(z_tau, tau, context)
        actions = action_head(latent)
        assert actions.shape == (4, 16, 2)
