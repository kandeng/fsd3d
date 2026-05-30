"""Autoregressive Wrapper — manages AR concerns outside the decoder.

Contains both the §3 decoder and the §4 ActionHead.  Adds a learned
start token, causal mask generation, teacher-forcing input preparation,
and step-by-step autoregressive generation.
"""

import torch
import torch.nn as nn

from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead


class AutoregressiveWrapper(nn.Module):
    """Thin wrapper managing autoregressive concerns *outside* the decoder.

    Contains both the §3 decoder and the §4 ActionHead.  Adds a learned
    start token, causal mask generation, teacher-forcing input preparation,
    and step-by-step autoregressive generation.
    """

    def __init__(self, decoder: FSD3DTransformerDecoder, action_head: ActionHead):
        super().__init__()
        self.decoder = decoder
        self.action_head = action_head
        self.start_token = nn.Parameter(torch.randn(1, action_head.action_dim))

    # ---- Causal mask ----------------------------------------------------
    @staticmethod
    def generate_causal_mask(T: int) -> torch.Tensor:
        """Return an additive float mask (0 at allowed, -inf at future)."""
        mask = torch.triu(torch.ones(T, T), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    # ---- Teacher-forcing helpers ----------------------------------------
    def prepare_teacher_forcing_input(self, z1: torch.Tensor) -> torch.Tensor:
        """Prepend start_token, drop last GT step → [B, T, action_dim]."""
        B = z1.size(0)
        start = self.start_token.unsqueeze(0).expand(B, -1, -1)   # [B, 1, A]
        return torch.cat([start, z1[:, :-1, :]], dim=1)            # [B, T, A]

    def forward(self, z1, tau, context):
        """Teacher-forcing forward: input=[START, z1[:-1]], target=z1."""
        T = z1.size(1)
        inp = self.prepare_teacher_forcing_input(z1)
        mask = self.generate_causal_mask(T).to(z1.device)
        return self.action_head(self.decoder(inp, tau, context, tgt_mask=mask))

    # ---- Autoregressive generation --------------------------------------
    @torch.no_grad()
    def generate(
        self,
        context,
        horizon: int = 16,
        tau_fixed=None,
        noise_step: int = 4,
        noise_sigma: float = 0.15,
        continuous_noise: bool = False,
        drift_bias: float = 0.0,
    ):
        """Step-by-step autoregressive generation.

        Args:
            context:         [B, S, d_model] memory bank
            horizon:         number of steps to generate
            tau_fixed:       optional [B, 1] constant τ (defaults to 1.0)
            noise_step:      step at which to start injecting Gaussian disturbance
            noise_sigma:     std-dev of the injected noise
            continuous_noise: if True, inject noise at every step from noise_step onward
                             (simulates persistent sensor drift causing cumulative error)
            drift_bias:      if > 0, after noise_step, push the x-coordinate (dim 0)
                             toward 0 by this fraction each step, simulating the model
                             reverting to straight-line behavior when the dodge is missed

        Returns:
            [B, horizon, action_dim] generated trajectory
        """
        device = next(self.parameters()).device
        B = context.size(0)
        if tau_fixed is None:
            tau_fixed = torch.ones(B, 1, device=device)

        tokens = [self.start_token.unsqueeze(0).expand(B, -1, -1)]  # [B, 1, A]

        for t in range(horizon):
            seq = torch.cat(tokens, dim=1)                           # [B, t+1, A]
            mask = self.generate_causal_mask(t + 1).to(device)
            latent = self.decoder(seq, tau_fixed, context, tgt_mask=mask)  # [B, t+1, d_model]
            pred = self.action_head(latent)                                 # [B, t+1, action_dim]
            next_step = pred[:, -1:, :]                                     # [B, 1, action_dim]

            # Noise injection — x-component only (dodge direction)
            # This prevents mode commitment while keeping y smooth
            if continuous_noise and t >= noise_step:
                next_step[:, :, 0] = next_step[:, :, 0] + noise_sigma * torch.randn_like(next_step[:, :, 0])
            elif t == noise_step:
                next_step[:, :, 0] = next_step[:, :, 0] + noise_sigma * torch.randn_like(next_step[:, :, 0])

            # Drift bias: push x-coordinate toward 0 after noise_step
            # Simulates mode averaging — the model can't commit to left or right,
            # so it averages both modes and flies straight (x≈0)
            if drift_bias > 0 and t >= noise_step:
                next_step[:, :, 0] = next_step[:, :, 0] * (1.0 - drift_bias)

            tokens.append(next_step)

        return torch.cat(tokens[1:], dim=1)                          # [B, horizon, A]
