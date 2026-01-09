# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math


class ResidualBlock(nn.Module):
    """Optimized residual block for VAE architectures.

    Uses bottleneck design with pre-normalization and SiLU activation.
    Follows VAE best practices for stable training.
    """

    def __init__(self, width: int, p_drop: float = 0.0):
        super().__init__()
        # Bottleneck design: width -> width//2 -> width
        # This reduces parameters while maintaining expressiveness
        bottleneck_width = max(width // 2, 16)  # Minimum bottleneck of 16

        # Pre-normalization design (LayerNorm before activation)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(bottleneck_width)

        # Bottleneck layers with SiLU activation (works well for VAEs)
        self.down = nn.Linear(width, bottleneck_width)
        self.up = nn.Linear(bottleneck_width, width)

        # Initialize weights properly for VAE training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable VAE training."""
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

        # Small bias initialization
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection
        residual = x

        # First path: norm -> activation -> linear
        x = self.norm1(x)
        x = F.silu(x)  # SiLU activation (x * sigmoid(x))
        x = self.down(x)

        # Second path: norm -> activation -> linear
        x = self.norm2(x)
        x = F.silu(x)
        x = self.up(x)

        # Residual connection
        return residual + x


class RadialFlow(nn.Module):

    def __init__(self, z_dim: int):
        super().__init__()
        self.z0 = nn.Parameter(torch.randn(1, z_dim))
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self, z: torch.Tensor, log_det_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.log_alpha) + 1e-5  # Add small constant for stability
        beta = -alpha + F.softplus(self.beta)

        diff = z - self.z0
        r = diff.norm(dim=1, keepdim=True) + 1e-8  # Avoid division by zero
        h = 1.0 / (alpha + r)

        z_new = z + beta * h * diff

        # Compute log determinant with clamping for stability
        bh = beta * h.squeeze(-1)  # (B,)
        bh_clamped = torch.clamp(bh, min=-0.999, max=0.999)
        term1 = (z.size(1) - 1) * torch.log1p(bh_clamped)
        term2 = torch.log1p(bh_clamped - beta * (h.squeeze(-1) ** 2) * r.squeeze(-1))
        log_det = term1 + term2

        return z_new, log_det_j + log_det


class TripletVAE(nn.Module):

    def __init__(
        self,
        width: int = 64,
        depth: int = 2,
        z_dim: int = 4,
        p_drop: float = 0.0,  # Kept for interface compatibility, but not used
        use_learned_variances: bool = True,
        init_logvar: float = -2.0,  # Initialize to small variances (exp(-2) = 0.135)
        min_logvar: float = -10.0,  # Minimum log-variance for stability
        max_logvar: float = 2.0,  # Maximum log-variance to prevent explosion
        use_additional_losses: bool = True,  # Use MSE + NLL for better reconstruction
        **kwargs,
    ) -> None:
        super().__init__()
        self.width, self.z_dim = width, z_dim
        self.use_learned_variances = use_learned_variances
        self.init_logvar = init_logvar
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.use_additional_losses = use_additional_losses

        # — Encoder (optimized for VAE training)
        self.enc_in = nn.Linear(3, width)  # Removed dropout
        self.encoder = nn.Sequential(
            *[ResidualBlock(width, p_drop=0.0) for _ in range(depth)]
        )

        # Latent space heads
        self.mu_head = nn.Linear(width, z_dim)
        self.logvar_head = nn.Linear(width, z_dim)

        # — One radial flow (optional normalizing flow for better posterior)
        self.use_flow = kwargs.pop("use_flow", True)
        # Radial flow can be disabled via config (use_flow=False)
        self.flow = RadialFlow(z_dim) if self.use_flow else nn.Identity()

        # — Decoder (symmetric to encoder)
        self.dec_in = nn.Linear(z_dim, width)  # Removed dropout
        self.decoder = nn.Sequential(
            *[ResidualBlock(width, p_drop=0.0) for _ in range(depth)]
        )

        # Output heads for material properties
        self.out_E = nn.Linear(width, 1)  # Young's modulus
        self.out_nu = nn.Linear(width, 1)  # Poisson's ratio
        self.out_rho = nn.Linear(width, 1)  # Density

        if self.use_learned_variances:
            # Learned log-variances with proper initialization
            self.out_E_logvar = nn.Linear(width, 1)
            self.out_nu_logvar = nn.Linear(width, 1)
            self.out_rho_logvar = nn.Linear(width, 1)

            # Initialize variance heads to small values
            self._init_variance_heads()
        else:
            # Fixed small variances for stable training
            self.register_buffer("fixed_E_logvar", torch.tensor([init_logvar]))
            self.register_buffer("fixed_nu_logvar", torch.tensor([init_logvar]))
            self.register_buffer("fixed_rho_logvar", torch.tensor([init_logvar]))

        # Isotropic Normal prior
        self.register_buffer("prior_mu", torch.zeros(z_dim))
        self.register_buffer("prior_std", torch.ones(z_dim))

        # Initialize all weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following VAE best practices."""
        # Initialize input/output layers
        nn.init.xavier_uniform_(self.enc_in.weight)
        nn.init.zeros_(self.enc_in.bias)

        nn.init.xavier_uniform_(self.dec_in.weight)
        nn.init.zeros_(self.dec_in.bias)

        # Initialize latent space heads
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

        # Initialize logvar head to predict small initial variances
        nn.init.xavier_uniform_(self.logvar_head.weight, gain=0.1)
        nn.init.constant_(self.logvar_head.bias, self.init_logvar)

        # Initialize output heads
        for head in [self.out_E, self.out_nu, self.out_rho]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _init_variance_heads(self):
        """Initialize variance heads to predict small, consistent variances."""
        for head in [self.out_E_logvar, self.out_nu_logvar, self.out_rho_logvar]:
            # Initialize weights to small values for stable variance prediction
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            # Initialize bias to init_logvar for small initial variance
            nn.init.constant_(head.bias, self.init_logvar)

    @staticmethod
    def posterior_log_probs(
        z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Element-wise log q(z|x) under diagonal Gaussian."""
        return Normal(mu, torch.exp(0.5 * logvar)).log_prob(z).sum(dim=-1)

    def encode(
        self, x: torch.Tensor, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters mu and logvar.

        Args:
            x: Input tensor [batch_size, 3]
            sample: Whether to sample from the distribution or return the mean

        Returns:
            z: Sampled latent vector if sample=True, else mu
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        # Input projection with activation
        h = F.silu(self.enc_in(x))

        # Pass through residual blocks
        h = self.encoder(h)

        # Get latent parameters
        mu, logvar = self.mu_head(h), self.logvar_head(h)

        # Ensure numerical stability (prevent extreme logvar values)
        logvar = torch.clamp(logvar, min=-15.0, max=15.0)

        z = self._reparameterise(mu, logvar) if sample else mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor):
        """
        Decode latent vector to output distribution parameters.

        Args:
            z: Latent vector [batch_size, z_dim]

        Returns:
            Tuple of (mean, logvar) pairs for each output: E, nu, rho
        """
        # Latent projection with activation
        h = F.silu(self.dec_in(z))

        # Pass through residual blocks
        h = self.decoder(h)

        # Get output means
        E_mu = self.out_E(h)
        nu_mu = self.out_nu(h)
        rho_mu = self.out_rho(h)

        if self.use_learned_variances:
            # Use learned log-variances with proper clamping
            E_logvar = torch.clamp(
                self.out_E_logvar(h), self.min_logvar, self.max_logvar
            )
            nu_logvar = torch.clamp(
                self.out_nu_logvar(h), self.min_logvar, self.max_logvar
            )
            rho_logvar = torch.clamp(
                self.out_rho_logvar(h), self.min_logvar, self.max_logvar
            )
        else:
            # Use fixed small variances
            batch_size = z.size(0)
            E_logvar = self.fixed_E_logvar.expand(batch_size, 1)
            nu_logvar = self.fixed_nu_logvar.expand(batch_size, 1)
            rho_logvar = self.fixed_rho_logvar.expand(batch_size, 1)

        return (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return improved recon loss, total KL and a dict including sampled `z` for TC loss."""

        # Encode & sample
        z, mu, logvar = self.encode(x, sample=True)
        log_q = self.posterior_log_probs(z, mu, logvar)

        # Optional radial flow refinement
        if self.use_flow:
            log_det = torch.zeros_like(log_q)
            z, log_det = self.flow(z, log_det)
            log_q = log_q - log_det  # q(z|x) after flow

        # Decode & recon loss in transformed space
        (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = self.decode(z)

        # Stack means for reconstruction output
        x_mu = torch.stack(
            [E_mu.squeeze(-1), nu_mu.squeeze(-1), rho_mu.squeeze(-1)], dim=-1
        )

        # Compute reconstruction loss
        if self.use_additional_losses:
            # 1. MSE loss for direct reconstruction (helps with mean prediction)
            mse_loss = F.mse_loss(x_mu, x, reduction="mean")

            if self.use_learned_variances:
                # 2. NLL loss (helps with uncertainty estimation) - only when we have learned variances
                E_std = torch.exp(0.5 * E_logvar)
                nu_std = torch.exp(0.5 * nu_logvar)
                rho_std = torch.exp(0.5 * rho_logvar)

                nll_E = -Normal(E_mu, E_std).log_prob(x[..., 0]).mean()
                nll_nu = -Normal(nu_mu, nu_std).log_prob(x[..., 1]).mean()
                nll_rho = -Normal(rho_mu, rho_std).log_prob(x[..., 2]).mean()
                nll_loss = nll_E + nll_nu + nll_rho

                # Combine losses (MSE is main driver, NLL for uncertainty)
                recon_loss = mse_loss + 0.1 * nll_loss

                # Store individual components for logging
                details_extra = {
                    "mse_loss": mse_loss.detach(),
                    "nll_loss": nll_loss.detach(),
                    "E_std_mean": E_std.mean().detach(),
                    "nu_std_mean": nu_std.mean().detach(),
                    "rho_std_mean": rho_std.mean().detach(),
                }
            else:
                # Only MSE loss when using fixed variances
                recon_loss = mse_loss

                # Store individual components for logging
                details_extra = {
                    "mse_loss": mse_loss.detach(),
                    "nll_loss": torch.tensor(0.0).detach(),  # No NLL component
                    "E_std_mean": (
                        torch.exp(0.5 * torch.tensor(self.init_logvar)).detach()
                    ),  # Fixed std
                    "nu_std_mean": (
                        torch.exp(0.5 * torch.tensor(self.init_logvar)).detach()
                    ),  # Fixed std
                    "rho_std_mean": (
                        torch.exp(0.5 * torch.tensor(self.init_logvar)).detach()
                    ),  # Fixed std
                }
        else:
            # Original NLL-only loss
            E_std = torch.exp(0.5 * E_logvar)
            nu_std = torch.exp(0.5 * nu_logvar)
            rho_std = torch.exp(0.5 * rho_logvar)

            recon = -Normal(E_mu, E_std).log_prob(x[..., 0])
            recon += -Normal(nu_mu, nu_std).log_prob(x[..., 1])
            recon += -Normal(rho_mu, rho_std).log_prob(x[..., 2])
            recon_loss = recon.mean()
            details_extra = {}

        # Prior log‑prob
        log_p = Normal(self.prior_mu, self.prior_std).log_prob(z).sum(dim=-1)

        kl_total = log_q - log_p
        details = {
            "recon": recon_loss.detach(),
            "kl_total": kl_total.detach(),
            "z": z.detach(),
            "mu": mu.detach(),
            "logvar": logvar.detach(),
            "log_q": log_q.detach(),
            "x_mu": x_mu.detach(),
            **details_extra,
        }
        return recon_loss, kl_total.mean(), details

    def sample_prior(self, n: int) -> torch.Tensor:
        """
        Return `n × z_dim` latent vectors drawn from the
        isotropic unit‑Gaussian prior (∼ N(0, I)).
        """
        return self.prior_mu + self.prior_std * torch.randn(
            n, self.z_dim, device=self.prior_mu.device
        )

    # sample from prior then decode to transformed scalar means
    def sample(self, n: int) -> torch.Tensor:
        with torch.no_grad():
            z = self.prior_mu + self.prior_std * torch.randn(
                n, self.z_dim, device=self.prior_mu.device
            )
            (E_mu, _), (nu_mu, _), (rho_mu, _) = self.decode(z)
            return torch.stack(
                [E_mu.squeeze(-1), nu_mu.squeeze(-1), rho_mu.squeeze(-1)], dim=-1
            )

    # reparametrization
    @staticmethod
    def _reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
