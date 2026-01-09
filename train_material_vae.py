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

from __future__ import annotations
import json, math, argparse, shutil, itertools, os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal

from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    ProjectConfiguration,
    set_seed,
    DistributedDataParallelKwargs,
    TorchDynamoPlugin,
)
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from vomp.models.material_vae.beta_tc import TripletVAE
from vomp.models.material_vae.standard_vae import StandardVAE

# Set up logger
logger = get_logger(__name__)


class TensorState:
    """Simple wrapper to make tensors checkpointable with state_dict/load_state_dict."""

    def __init__(self, tensor: torch.Tensor, name: str = "tensor"):
        self.name = name
        self.tensor = tensor
        self.device = tensor.device

    def state_dict(self):
        return {self.name: self.tensor.cpu()}  # Save on CPU to avoid device issues

    def load_state_dict(self, state_dict):
        self.tensor = state_dict[self.name].to(self.device)  # Move to correct device

    def to(self, device):
        self.tensor = self.tensor.to(device)
        self.device = device
        return self

    def __getattr__(self, name):
        # Delegate attribute access to the underlying tensor
        return getattr(self.tensor, name)


class EpochTracker:
    """Tracks the current epoch for checkpointing."""

    def __init__(self, epoch: int = 0):
        self.epoch = epoch

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]

    def increment(self):
        self.epoch += 1

    def __int__(self):
        return self.epoch

    def __str__(self):
        return str(self.epoch)


def physics_aware_transform(
    E: torch.Tensor,
    nu: torch.Tensor,
    rho: torch.Tensor,
    normalization_type: str = "standard",
    nu_min: float = None,
    nu_max: float = None,
) -> torch.Tensor:
    """
    Transform material properties with physics-aware normalization.

    Args:
        E: Young's modulus (Pa)
        nu: Poisson's ratio (dimensionless)
        rho: Density (kg/m³)
        normalization_type: "standard", "minmax", "physics_bounds", or "log_minmax"
        nu_min, nu_max: Required for "standard" normalization
    """
    # Add safety check for non-positive values
    E = torch.clamp_min(E, 1e-8)
    rho = torch.clamp_min(rho, 1e-8)

    if normalization_type == "standard":
        # Original approach - may break material relationships
        if nu_min is None or nu_max is None:
            nu_min, nu_max = nu.min().item(), nu.max().item()
        yE = torch.log10(E)
        p = (nu - nu_min) / (nu_max - nu_min)
        p = p.clamp(1e-4, 1.0 - 1e-4)
        y_pr = torch.logit(p)
        y_rho = torch.log10(rho)
        return torch.stack([yE, y_pr, y_rho], -1)

    elif normalization_type == "physics_bounds":
        # Use known physical bounds for materials
        # E: 1 MPa (soft polymers) to 1000 GPa (diamond)
        E_norm = (torch.log10(E) - 6.0) / (12.0 - 6.0)  # log10(1e6) to log10(1e12)

        # ν: -1 (auxetic) to 0.5 (incompressible)
        nu_norm = (nu + 1.0) / 1.5  # [-1, 0.5] → [0, 1]

        # ρ: 0.1 kg/m³ (aerogels) to 20000 kg/m³ (tungsten)
        rho_norm = (torch.log10(rho) - (-1.0)) / (
            4.3 - (-1.0)
        )  # log10(0.1) to log10(20000)

        return torch.stack([E_norm, nu_norm, rho_norm], -1)

    elif normalization_type == "log_minmax":
        # Log transform then min-max normalize - preserves relative relationships
        log_E = torch.log10(E)
        log_rho = torch.log10(rho)

        # Min-max normalize to [0, 1] - preserves material clustering
        E_norm = (log_E - log_E.min()) / (log_E.max() - log_E.min())
        nu_norm = (nu - nu.min()) / (nu.max() - nu.min())
        rho_norm = (log_rho - log_rho.min()) / (log_rho.max() - log_rho.min())

        return torch.stack([E_norm, nu_norm, rho_norm], -1)

    elif normalization_type == "log_minmax_no_density":
        # Log transform E but not rho, then min-max normalize - hybrid approach
        log_E = torch.log10(E)

        # Min-max normalize to [0, 1] - preserves material clustering
        E_norm = (log_E - log_E.min()) / (log_E.max() - log_E.min())
        nu_norm = (nu - nu.min()) / (nu.max() - nu.min())
        rho_norm = (rho - rho.min()) / (rho.max() - rho.min())  # No log for density

        return torch.stack([E_norm, nu_norm, rho_norm], -1)

    elif normalization_type == "minmax":
        # Simple min-max normalization - preserves ALL relationships
        E_norm = (E - E.min()) / (E.max() - E.min())
        nu_norm = (nu - nu.min()) / (nu.max() - nu.min())
        rho_norm = (rho - rho.min()) / (rho.max() - rho.min())

        return torch.stack([E_norm, nu_norm, rho_norm], -1)

    else:
        raise ValueError(f"Unknown normalization_type: {normalization_type}")


def physics_aware_inverse_transform(
    y_norm: torch.Tensor,
    E_min: float,
    E_max: float,
    nu_min: float,
    nu_max: float,
    rho_min: float,
    rho_max: float,
    normalization_type: str = "standard",
    mu: torch.Tensor = None,
    std: torch.Tensor = None,
) -> torch.Tensor:
    """Inverse transform back to physical units."""

    if normalization_type == "physics_bounds":
        # Inverse of physics_bounds normalization
        log_E = y_norm[..., 0] * (12.0 - 6.0) + 6.0
        E = torch.pow(10, log_E)

        nu = y_norm[..., 1] * 1.5 - 1.0

        log_rho = y_norm[..., 2] * (4.3 - (-1.0)) + (-1.0)
        rho = torch.pow(10, log_rho)

    elif normalization_type == "log_minmax":
        # Inverse of log_minmax normalization
        log_E_min, log_E_max = torch.log10(torch.tensor(E_min)), torch.log10(
            torch.tensor(E_max)
        )
        log_rho_min, log_rho_max = torch.log10(torch.tensor(rho_min)), torch.log10(
            torch.tensor(rho_max)
        )

        log_E = y_norm[..., 0] * (log_E_max - log_E_min) + log_E_min
        E = torch.pow(10, log_E)

        nu = y_norm[..., 1] * (nu_max - nu_min) + nu_min

        log_rho = y_norm[..., 2] * (log_rho_max - log_rho_min) + log_rho_min
        rho = torch.pow(10, log_rho)

    elif normalization_type == "log_minmax_no_density":
        # Inverse of log_minmax_no_density normalization
        log_E_min, log_E_max = torch.log10(torch.tensor(E_min)), torch.log10(
            torch.tensor(E_max)
        )

        log_E = y_norm[..., 0] * (log_E_max - log_E_min) + log_E_min
        E = torch.pow(10, log_E)

        nu = y_norm[..., 1] * (nu_max - nu_min) + nu_min

        # No log for density - direct min-max inverse
        rho = y_norm[..., 2] * (rho_max - rho_min) + rho_min

    elif normalization_type == "minmax":
        # Inverse of minmax normalization
        E = y_norm[..., 0] * (E_max - E_min) + E_min
        nu = y_norm[..., 1] * (nu_max - nu_min) + nu_min
        rho = y_norm[..., 2] * (rho_max - rho_min) + rho_min

    else:
        # Standard approach (original) - requires mu and std
        if mu is None or std is None:
            raise ValueError("mu and std are required for standard normalization")
        return inverse_transform(y_norm, mu, std, nu_min, nu_max)

    return torch.stack([E, nu, rho], -1)


def forward_transform(
    E: torch.Tensor, nu: torch.Tensor, rho: torch.Tensor, nu_min: float, nu_max: float
) -> torch.Tensor:
    # Add safety check for non-positive values
    E = torch.clamp_min(E, 1e-8)
    rho = torch.clamp_min(rho, 1e-8)

    yE = torch.log10(E)

    p = (nu - nu_min) / (nu_max - nu_min)

    p = p.clamp(1e-4, 1.0 - 1e-4)
    y_pr = torch.logit(p)
    y_rho = torch.log10(rho)
    return torch.stack([yE, y_pr, y_rho], -1)


def compute_stats(
    dataset: TensorDataset, nu_min: float, nu_max: float, batch_size: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size)
    sum_, sum2, n = torch.zeros(3), torch.zeros(3), 0
    for batch in loader:
        y = batch[0]  # Since TensorDataset returns tuples, get the first element
        sum_ += y.sum(0)
        sum2 += (y**2).sum(0)
        n += y.size(0)
    mu = sum_ / n
    std = (sum2 / n - mu**2).sqrt()
    return mu, std


def inverse_transform(
    y_norm: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    nu_min: float,
    nu_max: float,
) -> torch.Tensor:
    y = y_norm * std + mu
    E = torch.pow(10, y[..., 0])
    pr = torch.sigmoid(y[..., 1]) * (nu_max - nu_min) + nu_min
    rho = torch.pow(10, y[..., 2])
    return torch.stack([E, pr, rho], -1)


@torch.no_grad()
def iwae_nll(model: nn.Module, loader: DataLoader, K: int = 50) -> float:
    """Compute IWAE negative log-likelihood estimate.

    Args:
        model: VAE model (can be TripletVAE or StandardVAE)
        loader: DataLoader for evaluation
        K: Number of importance samples

    Returns:
        Negative log-likelihood estimate (lower is better)
    """
    total_log_w, total = 0.0, 0
    device = next(model.parameters()).device

    for batch in loader:
        x_norm = batch[0] if isinstance(batch, (tuple, list)) else batch
        B = x_norm.size(0)

        # Sample from prior
        z = model.sample_prior(K * B).to(device)  # (K·B, z_dim)

        # Decode
        (E_mu, Elog_sigma_2), (pr_mu, prlog_sigma_2), (rho_mu, rholog_sigma_2) = (
            model.decode(z)
        )

        # Expand input for K samples
        x_exp = x_norm.repeat(K, 1)  # (K·B, 3)

        # Compute log p(x|z)
        log_px_z = Normal(E_mu, torch.exp(0.5 * Elog_sigma_2)).log_prob(x_exp[:, 0])
        log_px_z += Normal(pr_mu, torch.exp(0.5 * prlog_sigma_2)).log_prob(x_exp[:, 1])
        log_px_z += Normal(rho_mu, torch.exp(0.5 * rholog_sigma_2)).log_prob(
            x_exp[:, 2]
        )
        log_px_z = log_px_z.sum(-1)

        # Compute log p(z)
        log_pz = Normal(0, 1).log_prob(z).sum(-1)

        # Importance weights
        log_w = (log_px_z + log_pz).view(K, B) - math.log(K)

        total_log_w += torch.logsumexp(log_w, 0).sum()
        total += B

    return (-total_log_w / total).item()  # lower is better


def reconstruction_metrics(
    x_phys: torch.Tensor, x_recon_phys: torch.Tensor  # (B,3)
) -> Dict[str, float]:
    diff = x_recon_phys - x_phys
    rmse = diff.pow(2).mean(0).sqrt()  # (3,)
    rel = (diff.abs() / x_phys.clamp_min(1e-8)).mean(0)  # (3,)
    return {
        "rmse_E": rmse[0].item(),
        "rmse_nu": rmse[1].item(),
        "rmse_rho": rmse[2].item(),
        "rel_E": rel[0].item(),
        "rel_nu": rel[1].item(),
        "rel_rho": rel[2].item(),
    }


def compute_tc_terms(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    log_q_zx: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Return MI, TC and dimension-wise KL.

    Args:
        z:        (B, D) sampled latent codes.
        mu:       (B, D) posterior means.
        logvar:   (B, D) posterior log-variances.
        log_q_zx: (B,)   log q(z|x) for each sample (already summed over D).
    """
    B, D = z.size()

    # pairwise log prob under diagonal Gaussians
    z_exp = z.unsqueeze(1)  # (B,1,D)
    mu_exp = mu.unsqueeze(0)  # (1,B,D)
    logvar_exp = logvar.unsqueeze(0)  # (1,B,D)
    var_exp = torch.exp(logvar_exp)  # (1,B,D)

    # log q(z_i | x_j) for every pair (i,j) and every dimension d
    const = math.log(2.0 * math.pi)
    log_q_zi_xj_d = -0.5 * (
        (z_exp - mu_exp) ** 2 / var_exp + logvar_exp + const
    )  # (B,B,D)

    # Aggregated posterior log-prob log q(z_i)
    log_q_zi_xj = log_q_zi_xj_d.sum(-1)  # (B,B)
    log_qz = torch.logsumexp(log_q_zi_xj, dim=1) - math.log(B)  # (B,)

    # Marginal log-prob product \sum_d log q(z_i_d)
    log_qz_prod = torch.zeros_like(log_qz)
    log_pz_prod = torch.zeros_like(log_qz)
    for d in range(D):
        log_q_zi_xj_dim = log_q_zi_xj_d[:, :, d]  # (B,B)
        log_q_zd = torch.logsumexp(log_q_zi_xj_dim, dim=1) - math.log(B)  # (B,)
        log_qz_prod += log_q_zd
        log_pz_prod += Normal(0, 1).log_prob(z[:, d])

    # Standard normal log-prob over full latent vector
    log_pz = Normal(0, 1).log_prob(z).sum(dim=1)

    mi = torch.clamp((log_q_zx - log_qz).mean(), min=0.0)  # Mutual information
    tc = (log_qz - log_qz_prod).mean()  # Total correlation
    kl_dim = (log_qz_prod - log_pz_prod).mean()  # Dimension-wise KL
    return mi, tc, kl_dim


def rotate_checkpoints(save_dir: Path, keep_last_n: int):
    ckpts = sorted(save_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep_last_n]:
        shutil.rmtree(p, ignore_errors=True)


def make_std_dataset(
    subset: torch.utils.data.Subset, mu: torch.Tensor, std: torch.Tensor
) -> TensorDataset:
    idx = torch.tensor(subset.indices, dtype=torch.long)
    y = subset.dataset.tensors[0][idx]  # (N, 3)
    y_std = (y - mu) / std
    return TensorDataset(y_std)


def compute_perplexity(z: torch.Tensor) -> float:
    N = z.size(0)
    # pairwise distances
    diff = z.unsqueeze(1) - z.unsqueeze(0)  # (N, N, z_dim)
    dist_sq = diff.pow(2).sum(dim=-1)  # (N, N)

    # gaussian kernel (sigma=1.0)
    kernel = torch.exp(-0.5 * dist_sq)  # (N, N)

    kernel_sum = kernel.sum(dim=1, keepdim=True)  # (N, 1)
    kernel_norm = kernel / kernel_sum  # (N, N)

    entropy = -torch.sum(kernel_norm * torch.log2(kernel_norm + 1e-8)) / N
    return 2.0 ** entropy.item()


def compute_latent_statistics(model, loader: DataLoader) -> Dict[str, float]:
    """Compute statistics about the learned latent space.

    Args:
        model: VAE model
        loader: DataLoader for evaluation

    Returns:
        Dictionary of latent space statistics
    """
    all_z = []
    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch in loader:
            x_norm = batch[0] if isinstance(batch, (tuple, list)) else batch
            z, mu, logvar = model.encode(x_norm)
            all_z.append(z)
            all_mu.append(mu)
            all_logvar.append(logvar)

    all_z = torch.cat(all_z, dim=0)
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    mu_var = all_mu.var(dim=0)
    active_units = (mu_var > 0.01).sum().item()

    perplexity = compute_perplexity(all_z)

    avg_var = torch.exp(all_logvar).mean().item()

    kl_per_dim = 0.5 * (all_mu.pow(2) + torch.exp(all_logvar) - all_logvar - 1.0).mean(
        dim=0
    )

    return {
        "active_units": active_units,
        "perplexity": perplexity,
        "avg_posterior_variance": avg_var,
        "total_kl": kl_per_dim.sum().item(),
        "max_dim_kl": kl_per_dim.max().item(),
        "min_dim_kl": kl_per_dim.min().item(),
    }


def train(config: Dict[str, Any]):
    # Set seed for reproducibility
    if "seed" in config:
        set_seed(config["seed"])

    # Configure DataLoader with stateful support
    dataloader_config = DataLoaderConfiguration(
        non_blocking=True,
        use_stateful_dataloader=config.get("use_stateful_dataloader", True),
    )

    # Configure DDP if needed
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.get("find_unused_parameters", False)
    )

    # Configure torch.compile if enabled
    dynamo_plugin = None
    if config.get("compile", {}).get("enabled", False):
        compile_config = config.get("compile", {})
        dynamo_plugin = TorchDynamoPlugin(
            backend=compile_config.get("backend", "inductor"),
            mode=compile_config.get("mode", "default"),
            fullgraph=compile_config.get("fullgraph", True),
            dynamic=compile_config.get("dynamic", False),
        )

    project_dir = Path(config["project_dir"]).resolve()
    is_dry = config.get("dry_run", False)

    # Initialize accelerator with all features
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),  # "no", "fp16", "bf16"
        log_with=None if is_dry else config.get("log_with", "tensorboard"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        project_config=ProjectConfiguration(
            project_dir=str(project_dir),
            automatic_checkpoint_naming=True,
            total_limit=config.get("keep_last_checkpoints", 2),
        ),
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs] if torch.cuda.is_available() else None,
        dynamo_plugin=dynamo_plugin,
    )

    # Initialize trackers
    if accelerator.is_main_process and not is_dry:
        accelerator.init_trackers(
            project_name=config["tracker_name"],
            config={
                "batch_size": config["dataloader"]["batch_size"],
                "learning_rate": config["optimizer"]["lr"],
                "weight_decay": config["optimizer"]["weight_decay"],
                "epochs": config["epochs"],
                "grad_clip_norm": config["optimizer"]["grad_clip_norm"],
                "free_nats": config["free_nats"],
                "alpha": config.get("alpha", 1.0),
                "beta": config.get("beta", 1.0),
                "gamma": config.get("gamma", 1.0),
                "standard_vae": config.get("standard_vae", False),
                "iwae_K": config.get("iwae_K", 50),
                "mixed_precision": config.get("mixed_precision", "no"),
                "seed": config.get("seed", None),
                **config.get("model", {}),
            },
        )

    # Load and prepare data
    data_csv = Path(config["data_csv"])
    csv_path = data_csv if data_csv.is_absolute() else Path.cwd() / data_csv
    df = pd.read_csv(csv_path)
    E_t = torch.tensor(df["youngs_modulus"].values, dtype=torch.float32)
    nu_t = torch.tensor(df["poisson_ratio"].values, dtype=torch.float32)
    rho_t = torch.tensor(df["density"].values, dtype=torch.float32)

    nu_min, nu_max = nu_t.min().item(), nu_t.max().item()
    E_min, E_max = E_t.min().item(), E_t.max().item()
    rho_min, rho_max = rho_t.min().item(), rho_t.max().item()

    # Choose normalization scheme
    normalization_type = config.get("normalization_type", "standard")

    if normalization_type == "standard":
        # Original approach with standardization
        y_all = forward_transform(E_t, nu_t, rho_t, nu_min, nu_max)
        dataset_raw = TensorDataset(y_all)

        n_total = len(dataset_raw)
        n_train = int(0.9 * n_total)
        n_val = n_total - n_train
        train_raw, val_raw = random_split(dataset_raw, [n_train, n_val])

        mu, std = compute_stats(train_raw, nu_min, nu_max)
        train_ds = make_std_dataset(train_raw, mu, std)
        val_ds = make_std_dataset(val_raw, mu, std)

        # Store normalization parameters for inverse transform
        norm_params = {
            "mu": mu,
            "std": std,
            "nu_min": nu_min,
            "nu_max": nu_max,
            "normalization_type": normalization_type,
        }

        # Save normalization parameters to JSON file
        if accelerator.is_main_process:
            norm_params_path = project_dir / "normalization_params.json"

            # Convert tensors to Python floats for JSON serialization
            norm_params_json = {
                "E_min": float(E_min),
                "E_max": float(E_max),
                "nu_min": float(nu_min),
                "nu_max": float(nu_max),
                "rho_min": float(rho_min),
                "rho_max": float(rho_max),
                "normalization_type": normalization_type,
                "dataset_size": n_total,
                "train_size": n_train,
                "val_size": n_val,
                "mu": [float(x) for x in mu.tolist()],
                "std": [float(x) for x in std.tolist()],
            }

            with open(norm_params_path, "w") as f:
                json.dump(norm_params_json, f, indent=2)

            logger.info(f"Saved normalization parameters to {norm_params_path}")
    else:
        # Physics-aware normalization schemes
        y_all = physics_aware_transform(
            E_t, nu_t, rho_t, normalization_type, nu_min, nu_max
        )
        dataset_all = TensorDataset(y_all)

        n_total = len(dataset_all)
        n_train = int(0.9 * n_total)
        n_val = n_total - n_train
        train_ds, val_ds = random_split(dataset_all, [n_train, n_val])

        # Store normalization parameters for inverse transform
        norm_params = {
            "E_min": E_min,
            "E_max": E_max,
            "nu_min": nu_min,
            "nu_max": nu_max,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "normalization_type": normalization_type,
        }

        # Save normalization parameters to JSON file
        if accelerator.is_main_process:
            norm_params_path = project_dir / "normalization_params.json"

            # Convert tensors to Python floats for JSON serialization
            norm_params_json = {
                "E_min": float(E_min),
                "E_max": float(E_max),
                "nu_min": float(nu_min),
                "nu_max": float(nu_max),
                "rho_min": float(rho_min),
                "rho_max": float(rho_max),
                "normalization_type": normalization_type,
                "dataset_size": n_total,
                "train_size": n_train,
                "val_size": n_val,
            }

            # Add mu/std if using standard normalization
            if normalization_type == "standard":
                norm_params_json.update(
                    {
                        "mu": [float(x) for x in mu.tolist()],
                        "std": [float(x) for x in std.tolist()],
                    }
                )

            with open(norm_params_path, "w") as f:
                json.dump(norm_params_json, f, indent=2)

            logger.info(f"Saved normalization parameters to {norm_params_path}")

        # For compatibility with existing code, create dummy mu/std
        mu = torch.zeros(3)
        std = torch.ones(3)

    # Wrap normalization parameters for checkpointing
    if normalization_type == "standard":
        mu_state = TensorState(mu.to(accelerator.device), "mu")
        std_state = TensorState(std.to(accelerator.device), "std")
    else:
        # Store all normalization parameters
        mu_state = TensorState(
            torch.tensor([E_min, nu_min, rho_min]).to(accelerator.device), "mins"
        )
        std_state = TensorState(
            torch.tensor([E_max, nu_max, rho_max]).to(accelerator.device), "maxs"
        )

    # Log initial statistics
    if not is_dry:
        if normalization_type == "standard":
            accelerator.log(
                {f"data/mu_{i}": v.item() for i, v in enumerate(mu_state.tensor)},
                step=0,
            )
            accelerator.log(
                {f"data/std_{i}": v.item() for i, v in enumerate(std_state.tensor)},
                step=0,
            )
        else:
            accelerator.log(
                {f"data/min_{i}": v.item() for i, v in enumerate(mu_state.tensor)},
                step=0,
            )
            accelerator.log(
                {f"data/max_{i}": v.item() for i, v in enumerate(std_state.tensor)},
                step=0,
            )
        accelerator.log({"data/normalization_type": normalization_type}, step=0)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        prefetch_factor=config["dataloader"]["prefetch_factor"],
        persistent_workers=(
            config["dataloader"]["persistent_workers"]
            if config["dataloader"]["num_workers"] > 0
            else False
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        prefetch_factor=config["dataloader"]["prefetch_factor"],
        persistent_workers=(
            config["dataloader"]["persistent_workers"]
            if config["dataloader"]["num_workers"] > 0
            else False
        ),
    )

    # Choose and create model
    use_standard_vae = config.get("standard_vae", False)
    if use_standard_vae:
        model = StandardVAE(**config["model"])
    else:
        model = TripletVAE(**config["model"])

    # Store z_dim before model gets wrapped
    z_dim = model.z_dim

    # Log model information
    if accelerator.is_main_process:
        logger.info("Model hyper-parameters:")
        for k, v in config["model"].items():
            logger.info(f"  {k}: {v}")

        if use_standard_vae:
            logger.info("Using StandardVAE model")
        else:
            logger.info("Using TripletVAE model with beta-TC and radial flow")

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Total parameters: {total_params:.2f} M")

        # Log torch.compile status
        if config.get("compile", {}).get("enabled", False):
            compile_config = config.get("compile", {})
            logger.info(f"Torch compile enabled:")
            logger.info(f"  Backend: {compile_config.get('backend', 'inductor')}")
            logger.info(f"  Mode: {compile_config.get('mode', 'default')}")

        # Save model architecture summaries
        enc_path = project_dir / "encoder_summary.txt"
        dec_path = project_dir / "decoder_summary.txt"

        enc_str = str(nn.Sequential(model.enc_in, model.encoder))
        dec_str = str(nn.Sequential(model.dec_in, model.decoder))

        enc_path.write_text(enc_str)
        dec_path.write_text(dec_str)

        logger.info(f"Saved encoder summary ➜ {enc_path}")
        logger.info(f"Saved decoder summary ➜ {dec_path}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # Create learning rate scheduler if configured
    lr_scheduler = None
    if "lr_scheduler" in config and config["lr_scheduler"] is not None:
        scheduler_config = config["lr_scheduler"]
        scheduler_type = scheduler_config.get("type", "cosine")

        if scheduler_type == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["epochs"] * len(train_loader),
                eta_min=scheduler_config.get("eta_min", 0),
            )
        elif scheduler_type == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        elif scheduler_type == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=scheduler_config.get("gamma", 0.95)
            )

    # Prepare everything with accelerator
    if lr_scheduler is not None:
        model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler
        )
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

    # Register additional state for checkpointing
    accelerator.register_for_checkpointing(mu_state)
    accelerator.register_for_checkpointing(std_state)

    # Create and register epoch tracker
    epoch_tracker = EpochTracker(0)
    accelerator.register_for_checkpointing(epoch_tracker)

    # Load checkpoint if resuming
    starting_epoch = 0
    step_global = 0

    # Check for existing checkpoints in project directory
    resume_from_checkpoint = config.get("resume_from_checkpoint", None)

    if resume_from_checkpoint is None:
        # Look for existing checkpoints in the checkpoints subdirectory
        checkpoints_dir = project_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("checkpoint_*"))
            if checkpoints:
                # Sort checkpoints by number (Accelerate uses checkpoint_0, checkpoint_1, etc.)
                def get_checkpoint_number(path):
                    try:
                        return int(path.name.split("_")[1])
                    except:
                        return -1

                checkpoints = sorted(
                    checkpoints, key=get_checkpoint_number, reverse=True
                )

                # Try to find the latest valid checkpoint
                for checkpoint in checkpoints:
                    try:
                        # Check if checkpoint is valid by looking for required files
                        if (checkpoint / "model.safetensors").exists() or (
                            checkpoint / "pytorch_model.bin"
                        ).exists():
                            resume_from_checkpoint = str(checkpoint)
                            if accelerator.is_main_process:
                                logger.info(f"Found existing checkpoint: {checkpoint}")
                            break
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.warning(
                                f"Skipping potentially corrupted checkpoint {checkpoint}: {e}"
                            )

                if resume_from_checkpoint is None and checkpoints:
                    if accelerator.is_main_process:
                        logger.warning(
                            "All checkpoints appear to be corrupted, starting fresh"
                        )

    if resume_from_checkpoint:
        try:
            accelerator.load_state(resume_from_checkpoint)
            # After loading state, the epoch_tracker will have the correct epoch
            starting_epoch = epoch_tracker.epoch
            step_global = starting_epoch * len(train_loader)

            # Ensure tensor states are on the correct device after loading
            mu_state.to(accelerator.device)
            std_state.to(accelerator.device)

            # Update accelerator's iteration counter to match the epoch
            # This ensures checkpoint naming continues correctly
            if hasattr(accelerator, "project_configuration"):
                # Extract checkpoint number from the loaded checkpoint path
                checkpoint_path = Path(resume_from_checkpoint)
                if checkpoint_path.name.startswith("checkpoint_"):
                    current_checkpoint_num = int(checkpoint_path.name.split("_")[1])
                    # Set iteration to the next checkpoint number
                    accelerator.project_configuration.iteration = (
                        current_checkpoint_num + 1
                    )

                    if accelerator.is_main_process:
                        logger.info(
                            f"Loaded checkpoint_{current_checkpoint_num}, next will be checkpoint_{accelerator.project_configuration.iteration}"
                        )

            if accelerator.is_main_process:
                logger.info(
                    f"Resumed from checkpoint: {resume_from_checkpoint}, continuing from epoch {starting_epoch + 1}"
                )
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Failed to load checkpoint {resume_from_checkpoint}: {e}")
                logger.info("Starting training from scratch")
            starting_epoch = 0
            step_global = 0
            epoch_tracker.epoch = 0

    # Training loop
    for epoch in range(starting_epoch, config["epochs"]):
        model.train()

        # Standard iteration over DataLoader
        steps_per_epoch = len(train_loader)

        prog_bar = tqdm(
            train_loader,
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch+1}/{config['epochs']}",
            leave=False,
        )

        for step, batch in enumerate(prog_bar):
            if batch is None:
                continue
            x_norm = batch[0] if isinstance(batch, (tuple, list)) else batch

            with accelerator.accumulate(model):
                # Use autocast for mixed precision
                with accelerator.autocast():
                    recon, kl_total, details = model(x_norm)

                    # Different loss calculation based on model type
                    if use_standard_vae:
                        # Standard VAE loss: reconstruction + KL
                        # Apply free nats to prevent posterior collapse
                        kl_clamped = torch.clamp(kl_total, min=config["free_nats"])

                        # KL weight determination
                        if "kl_weight" in config:
                            # Use fixed KL weight from config
                            kl_weight = config["kl_weight"]
                        elif config.get("kl_annealing", False):
                            # Optional: KL annealing
                            kl_weight = min(
                                1.0,
                                (epoch + 1) / config.get("kl_annealing_epochs", 100),
                            )
                        else:
                            kl_weight = 1.0

                        # Optional: reconstruction loss scaling
                        recon_scale = config.get("recon_scale", 1.0)

                        loss = recon_scale * recon + kl_weight * kl_clamped
                    else:
                        # TripletVAE with beta-TC loss
                        z = details["z"]
                        mu_post = details["mu"]
                        logvar_post = details["logvar"]
                        log_q_zx = details["log_q"]

                        mi, tc, kl_dim = compute_tc_terms(
                            z, mu_post, logvar_post, log_q_zx
                        )
                        kl_dim = torch.clamp(kl_dim, min=config["free_nats"] * z_dim)
                        loss = (
                            recon
                            + config["gamma"] * mi
                            + config["beta"] * tc
                            + config["alpha"] * kl_dim
                        )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config["optimizer"]["grad_clip_norm"]
                    )

                optimizer.step()
                if lr_scheduler is not None and not isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    lr_scheduler.step()
                optimizer.zero_grad()

            # Log metrics
            if accelerator.sync_gradients and not is_dry:
                if use_standard_vae:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/recon": recon.item(),
                        "train/kl": kl_total.item(),
                        "train/kl_clamped": (
                            kl_clamped.item()
                            if "kl_clamped" in locals()
                            else kl_total.item()
                        ),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    # Add detailed loss components if available
                    if "mse_loss" in details:
                        log_dict.update(
                            {
                                "train/mse_loss": details["mse_loss"].item(),
                                "train/nll_loss": details["nll_loss"].item(),
                                "train/std_E": details["E_std_mean"].item(),
                                "train/std_nu": details["nu_std_mean"].item(),
                                "train/std_rho": details["rho_std_mean"].item(),
                            }
                        )
                    if config.get("kl_annealing", False):
                        log_dict["train/kl_weight"] = kl_weight
                    accelerator.log(log_dict, step=step_global)
                else:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/recon": recon.item(),
                        "train/mi": mi.item(),
                        "train/tc": tc.item(),
                        "train/kl_dim": kl_dim.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    # Add detailed loss components if available
                    if "mse_loss" in details:
                        log_dict.update(
                            {
                                "train/mse_loss": details["mse_loss"].item(),
                                "train/nll_loss": details["nll_loss"].item(),
                                "train/std_E": details["E_std_mean"].item(),
                                "train/std_nu": details["nu_std_mean"].item(),
                                "train/std_rho": details["rho_std_mean"].item(),
                            }
                        )
                    accelerator.log(log_dict, step=step_global)

            step_global += 1

            if is_dry and step_global >= 5:
                break

        if is_dry:
            break

        # Evaluation phase
        if not is_dry and (epoch + 1) % config.get("eval_interval", 1) == 0:
            model.eval()

            # Wait for all processes before evaluation
            accelerator.wait_for_everyone()

            # Unwrap model once for evaluation
            eval_model = accelerator.unwrap_model(model)

            # Calculate validation metrics
            with torch.no_grad():
                # IWAE NLL
                nll_val = iwae_nll(eval_model, val_loader, K=config.get("iwae_K", 50))

                # Gather NLL from all processes
                nll_val_tensor = torch.tensor(nll_val, device=accelerator.device)
                nll_val_gathered = accelerator.gather(nll_val_tensor)
                if accelerator.is_main_process and not is_dry:
                    nll_val_avg = nll_val_gathered.mean().item()
                    accelerator.log({"val/iwae_nll": nll_val_avg}, step=step_global)

                # Compute latent space statistics
                val_latent_stats = compute_latent_statistics(eval_model, val_loader)
                if not is_dry:
                    accelerator.log(
                        {f"val/latent/{k}": v for k, v in val_latent_stats.items()},
                        step=step_global,
                    )

                # Compute reconstruction metrics
                for split_name, loader in [
                    ("train", train_loader),
                    ("val", val_loader),
                ]:
                    errs: List[Dict[str, float]] = []
                    for batch in loader:
                        x_norm = batch[0] if isinstance(batch, (tuple, list)) else batch
                        # Deterministic reconstruction (use posterior means)
                        z_det, _, _ = eval_model.encode(x_norm, sample=False)
                        (E_mu, _), (nu_mu, _), (rho_mu, _) = eval_model.decode(z_det)
                        x_recon_norm = torch.stack(
                            [E_mu.squeeze(-1), nu_mu.squeeze(-1), rho_mu.squeeze(-1)],
                            dim=-1,
                        )

                        # Use appropriate inverse transform based on normalization type
                        if normalization_type == "standard":
                            x_phys = inverse_transform(
                                x_norm,
                                mu_state.tensor,
                                std_state.tensor,
                                nu_min,
                                nu_max,
                            )
                            x_recon_phys = inverse_transform(
                                x_recon_norm,
                                mu_state.tensor,
                                std_state.tensor,
                                nu_min,
                                nu_max,
                            )
                        else:
                            x_phys = physics_aware_inverse_transform(
                                x_norm,
                                E_min,
                                E_max,
                                nu_min,
                                nu_max,
                                rho_min,
                                rho_max,
                                normalization_type,
                            )
                            x_recon_phys = physics_aware_inverse_transform(
                                x_recon_norm,
                                E_min,
                                E_max,
                                nu_min,
                                nu_max,
                                rho_min,
                                rho_max,
                                normalization_type,
                            )

                        errs.append(reconstruction_metrics(x_phys, x_recon_phys))

                    # Aggregate metrics
                    agg = {k: sum(d[k] for d in errs) / len(errs) for k in errs[0]}

                    # Gather metrics from all processes
                    for k, v in agg.items():
                        metric_tensor = torch.tensor(v, device=accelerator.device)
                        metric_gathered = accelerator.gather(metric_tensor)
                        if accelerator.is_main_process and not is_dry:
                            metric_avg = metric_gathered.mean().item()
                            accelerator.log(
                                {f"{split_name}/{k}": metric_avg}, step=step_global
                            )

            # Save checkpoint using Accelerate's save_state
            if (epoch + 1) % config.get("save_interval", 1) == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    logger.info(f"Saving checkpoint at epoch {epoch + 1}")
                accelerator.save_state()

        # Increment epoch tracker at the end of the epoch
        epoch_tracker.increment()

    # Final checkpoint and cleanup
    accelerator.wait_for_everyone()
    if not is_dry:
        if accelerator.is_main_process:
            logger.info("Saving final checkpoint")
        accelerator.save_state()

        # Save final model in standard format
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            final_model_path = project_dir / "final_model"
            accelerator.save_model(
                unwrapped_model, final_model_path, safe_serialization=True
            )
            logger.info(f"Saved final model to {final_model_path}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate from config"
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=None,
        help="Override project directory from config",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=None,
        help="Override KL divergence weight (disables KL annealing)",
    )
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    # Apply command-line overrides
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr
        # Disable learning rate scheduler when LR is explicitly set
        cfg["lr_scheduler"] = None
        print(f"Using constant learning rate: {args.lr} (scheduler disabled)")

    if args.project_dir is not None:
        cfg["project_dir"] = args.project_dir

    if args.kl_weight is not None:
        cfg["kl_weight"] = args.kl_weight
        cfg["kl_annealing"] = False  # Disable KL annealing when weight is manually set

    train(cfg)
