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

"""Train material VAEs with configuration-selected models, data, and objectives."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
    TorchDynamoPlugin,
)
from safetensors.torch import load_file
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from vomp.models.material_vae import StandardVAE, TripletVAE

from vomp.utils.material_vae_normalization import (
    forward_transform,
    inverse_transform,
    normalize_log_minmax,
    physics_aware_inverse_transform,
)


class TensorState:
    """Normalization state compatible with existing Accelerate checkpoints."""

    def __init__(self, tensor, name):
        self.tensor, self.name = tensor, name

    def state_dict(self):
        return {self.name: self.tensor.cpu()}

    def load_state_dict(self, state):
        self.tensor = state[self.name].to(self.tensor.device)


class EpochTracker:
    def __init__(self, epoch=0):
        self.epoch = epoch

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]


def normalize(properties, bounds):
    """Apply the configured transform using the saved physical bounds."""
    mode = bounds["normalization_type"]
    if mode == "log_minmax":
        return normalize_log_minmax(properties, bounds)
    E, nu, rho = properties.unbind(-1)
    if mode == "standard":
        transformed = forward_transform(E, nu, rho, bounds["nu_min"], bounds["nu_max"])
        if "mu" in bounds:
            transformed = (
                transformed - properties.new_tensor(bounds["mu"])
            ) / properties.new_tensor(bounds["std"])
        return transformed
    if mode == "physics_bounds":
        return torch.stack(
            (
                (E.clamp_min(1e-8).log10() - 6.0) / 6.0,
                (nu + 1.0) / 1.5,
                (rho.clamp_min(1e-8).log10() + 1.0) / 5.3,
            ),
            dim=-1,
        )
    low = properties.new_tensor([bounds[f"{key}_min"] for key in ("E", "nu", "rho")])
    high = properties.new_tensor([bounds[f"{key}_max"] for key in ("E", "nu", "rho")])
    if mode == "minmax":
        return (properties - low) / (high - low)
    if mode == "log_minmax_no_density":
        return torch.stack(
            (
                (E.clamp_min(1e-8).log10() - low[0].log10())
                / (high[0].log10() - low[0].log10()),
                (nu - low[1]) / (high[1] - low[1]),
                (rho - low[2]) / (high[2] - low[2]),
            ),
            dim=-1,
        )
    raise ValueError(f"Unknown normalization_type: {mode}")


def denormalize(properties, bounds):
    args = {
        key: bounds[key]
        for key in ("E_min", "E_max", "nu_min", "nu_max", "rho_min", "rho_max")
    }
    if bounds["normalization_type"] == "standard":
        args.update(
            mu=properties.new_tensor(bounds["mu"]),
            std=properties.new_tensor(bounds["std"]),
        )
    return physics_aware_inverse_transform(
        properties, **args, normalization_type=bounds["normalization_type"]
    )


def _physical_rows(frame, label):
    columns = ("youngs_modulus", "poisson_ratio", "density")
    # Per-column conversion preserves the trained float32 normalization path.
    rows = torch.stack(
        [torch.tensor(frame[column].values, dtype=torch.float32) for column in columns],
        dim=-1,
    )
    if len(rows) == 0 or not torch.isfinite(rows).all():
        raise ValueError(f"{label} must contain finite material triplets")
    if (rows[:, (0, 2)] <= 0).any():
        raise ValueError(f"{label} requires positive Young's modulus and density")
    return rows


def load_datasets(config):
    """Keep training weights; use an explicit held-out CSV or a random 90/10 split."""
    physical = _physical_rows(pd.read_csv(config["data_csv"]), "data_csv")
    bounds = {"normalization_type": config.get("normalization_type", "standard")}
    for i, name in enumerate(("E", "nu", "rho")):
        bounds[f"{name}_min"] = physical[:, i].min().item()
        bounds[f"{name}_max"] = physical[:, i].max().item()
        if (
            bounds[f"{name}_min"] == bounds[f"{name}_max"]
            and bounds["normalization_type"] != "physics_bounds"
        ):
            raise ValueError(
                f"Non-degenerate {name} bounds are required for normalization"
            )

    explicit_validation = config.get("validation_csv")
    if explicit_validation:
        validation = pd.read_csv(explicit_validation)
        if "split" in validation:
            validation = validation.loc[
                validation["split"].fillna("").astype(str).str.strip().str.lower()
                == "validation"
            ]
        training_rows = physical
        validation_rows = _physical_rows(validation, "validation_csv")
    else:
        train_size = int(0.9 * len(physical))
        if train_size == 0:
            raise ValueError(
                "data_csv requires at least two rows for a train/validation split"
            )
        train_subset, val_subset = random_split(
            TensorDataset(physical), [train_size, len(physical) - train_size]
        )
        training_rows = physical[train_subset.indices]
        validation_rows = physical[val_subset.indices]

    if bounds["normalization_type"] == "standard":
        transformed = normalize(training_rows, bounds)
        # Match the original population statistics and chunk accumulation.
        total, squared, count = torch.zeros(3), torch.zeros(3), 0
        for (batch,) in DataLoader(TensorDataset(transformed), batch_size=1024):
            total += batch.sum(0)
            squared += batch.square().sum(0)
            count += len(batch)
        mu = total / count
        std = (squared / count - mu.square()).sqrt()
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError(
                "Standard normalization requires non-degenerate training statistics"
            )
        bounds.update(mu=mu.tolist(), std=std.tolist())

    train_y = normalize(training_rows, bounds)
    val_y = normalize(validation_rows, bounds)
    if explicit_validation:
        unique_val = np.unique(val_y.numpy(), axis=0)
        train_keys = {row.tobytes() for row in np.unique(train_y.numpy(), axis=0)}
        unique_val = unique_val[[row.tobytes() not in train_keys for row in unique_val]]
        if len(unique_val) == 0:
            raise ValueError(
                "validation_csv must contain triplets outside training data"
            )
        val_y = torch.from_numpy(unique_val)
    bounds.update(
        dataset_size=len(physical), train_size=len(train_y), val_size=len(val_y)
    )
    return TensorDataset(train_y), TensorDataset(val_y), bounds


def build_model(config):
    model_type = StandardVAE if config.get("standard_vae", False) else TripletVAE
    return model_type(**config["model"])


def make_loaders(config, train_ds, val_ds):
    options = dict(config["dataloader"])
    if options.get("num_workers", 0) == 0:
        options.pop("prefetch_factor", None)
        options["persistent_workers"] = False
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **options),
        DataLoader(val_ds, shuffle=False, drop_last=False, **options),
    )


def compute_tc_terms(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    log_q_zx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
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


def training_objective(config, recon, kl, details, *, epoch, step, steps_per_epoch):
    """Select the configured VAE objective while sharing optimization and logging."""
    if not config.get("standard_vae", False):
        mi, tc, kl_dim = compute_tc_terms(
            details["z"], details["mu"], details["logvar"], details["log_q"]
        )
        kl_dim = kl_dim.clamp_min(
            float(config.get("free_nats", 0.0)) * details["z"].shape[-1]
        )
        loss = (
            recon
            + config.get("gamma", 1.0) * mi
            + config.get("beta", 1.0) * tc
            + config.get("alpha", 1.0) * kl_dim
        )
        return loss, {"mi": mi, "tc": tc, "kl_dim": kl_dim}

    schedule = config.get(
        "kl_schedule",
        "linear_step" if config.get("kl_annealing", False) else "constant",
    )
    weight = float(config.get("kl_weight", 1.0))
    horizon = max(1, int(config.get("kl_annealing_epochs", 20)))
    if schedule == "linear_step":
        weight *= min(
            1.0, (epoch * steps_per_epoch + step + 1) / (horizon * steps_per_epoch)
        )
    elif schedule == "linear_epoch":
        weight *= min(1.0, (epoch + 1) / horizon)
    elif schedule != "constant":
        raise ValueError(f"Unknown kl_schedule: {schedule}")
    free_nats = float(config.get("free_nats", 0.0))
    mode = config.get("free_nats_mode", "floor")
    if mode not in {"floor", "excess"}:
        raise ValueError(f"Unknown free_nats_mode: {mode}")
    if free_nats > 0:
        penalty = (
            kl.clamp_min(free_nats)
            if mode == "floor"
            else (kl - free_nats).clamp_min(0)
        )
    else:
        penalty = kl
    loss = config.get("recon_scale", 1.0) * recon + weight * penalty
    return loss, {"kl_penalty": penalty, "kl_weight": recon.new_tensor(weight)}


def apply_latent_flow(model, z):
    log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    return model.flow(z, log_det) if model.use_flow else (z, log_det)


@torch.no_grad()
def iwae_nll(model, loader, K=50):
    """Validation negative log-likelihood with K posterior samples."""
    total_log_w, total = 0.0, 0
    device = next(model.parameters()).device
    for (x,) in loader:
        x = x.to(device)
        batch_size = x.size(0)
        expanded = x.unsqueeze(0).expand(K, -1, -1).reshape(K * batch_size, -1)
        z, mu, logvar = model.encode(expanded, sample=True)
        log_q = model.posterior_log_probs(z, mu, logvar)
        z, log_det = apply_latent_flow(model, z)
        log_q = log_q - log_det
        decoded = model.decode(z)
        means = torch.cat([pair[0] for pair in decoded], dim=-1)
        logvars = torch.cat([pair[1] for pair in decoded], dim=-1)
        log_px = Normal(means, torch.exp(0.5 * logvars)).log_prob(expanded).sum(-1)
        log_p = Normal(model.prior_mu, model.prior_std).log_prob(z).sum(-1)
        log_w = (log_px + log_p - log_q).view(K, batch_size)
        total_log_w += (torch.logsumexp(log_w, dim=0) - math.log(K)).sum()
        total += batch_size
    return (-total_log_w / total).item()


@torch.no_grad()
def posterior_statistics(model, loader):
    # Retain this sampled pass: removing its draws changes later training RNG.
    means, logvars, kl = [], [], []
    for (x,) in loader:
        z, mu, logvar = model.encode(x)
        log_q = model.posterior_log_probs(z, mu, logvar)
        z, log_det = apply_latent_flow(model, z)
        mu, _ = apply_latent_flow(model, mu)
        log_p = Normal(model.prior_mu, model.prior_std).log_prob(z).sum(-1)
        means.append(mu)
        logvars.append(logvar)
        kl.append(log_q - log_det - log_p)
    return {
        "active_units": (torch.cat(means).var(dim=0) > 0.01).sum().item(),
        "avg_base_posterior_variance": torch.cat(logvars).exp().mean().item(),
        "flow_kl": torch.cat(kl).mean().item(),
    }


@torch.no_grad()
def reconstruction_metrics(model, loader, bounds):
    errors = []
    for (x,) in loader:
        z, _, _ = model.encode(x, sample=False)
        z, _ = apply_latent_flow(model, z)
        recon = torch.stack([pair[0].squeeze(-1) for pair in model.decode(z)], dim=-1)
        target = denormalize(x, bounds)
        prediction = denormalize(recon, bounds)
        diff = prediction - target
        rmse = diff.square().mean(0).sqrt()
        relative = (diff.abs() / target.clamp_min(1e-8)).mean(0)
        errors.append(
            {
                **{
                    f"rmse_{k}": rmse[i].item()
                    for i, k in enumerate(("E", "nu", "rho"))
                },
                **{
                    f"rel_{k}": relative[i].item()
                    for i, k in enumerate(("E", "nu", "rho"))
                },
            }
        )
    # Preserve the original mean-of-batch-metrics reduction.
    return {k: sum(e[k] for e in errors) / len(errors) for k in errors[0]}


def evaluate(accelerator, model, train_loader, val_loader, bounds, config, step):
    model.eval()
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    # Pass order and train-loader iteration preserve the original RNG sequence.
    nll = iwae_nll(model, val_loader, K=config.get("iwae_K", 50))
    value = accelerator.gather(torch.tensor(nll, device=accelerator.device)).mean()
    if accelerator.is_main_process:
        accelerator.log({"val/iwae_nll": value.item()}, step=step)
    accelerator.log(
        {
            f"val/latent/{k}": v
            for k, v in posterior_statistics(model, val_loader).items()
        },
        step=step,
    )
    for split, loader in (("train", train_loader), ("val", val_loader)):
        for key, value in reconstruction_metrics(model, loader, bounds).items():
            gathered = accelerator.gather(
                torch.tensor(value, device=accelerator.device)
            ).mean()
            if accelerator.is_main_process:
                accelerator.log({f"{split}/{key}": gathered.item()}, step=step)


def latest_checkpoint(project_dir, has_scheduler):
    """Skip incomplete saves; Accelerate restores model, optimizer and RNG."""
    required = {
        "model.safetensors",
        "optimizer.bin",
        "custom_checkpoint_0.pkl",
        "custom_checkpoint_1.pkl",
        "custom_checkpoint_2.pkl",
    }
    if has_scheduler:
        required.add("scheduler.bin")
    checkpoints = sorted(
        (
            p
            for p in (project_dir / "checkpoints").glob("checkpoint_*")
            if p.is_dir() and p.name.removeprefix("checkpoint_").isdigit()
        ),
        key=lambda p: int(p.name.removeprefix("checkpoint_")),
        reverse=True,
    )
    for checkpoint in checkpoints:
        names = {p.name for p in checkpoint.iterdir()}
        if "pytorch_model.bin" in names:
            names.add("model.safetensors")
        if required <= names and any(n.startswith("random_states_") for n in names):
            return str(checkpoint)
    return None


def train(config):
    set_seed(config.get("seed", 42))
    project_dir = Path(config["project_dir"]).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    dry_run = config.get("dry_run", False)
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),
        log_with=None if dry_run else config.get("log_with", "tensorboard"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        project_config=ProjectConfiguration(
            project_dir=str(project_dir),
            automatic_checkpoint_naming=True,
            total_limit=config.get("keep_last_checkpoints", 4),
        ),
        dataloader_config=DataLoaderConfiguration(
            non_blocking=True,
            use_stateful_dataloader=config.get("use_stateful_dataloader", False),
        ),
        dynamo_plugin=(
            TorchDynamoPlugin(
                **{
                    key: value
                    for key, value in config["compile"].items()
                    if key != "enabled"
                }
            )
            if config.get("compile", {}).get("enabled", False)
            else None
        ),
        kwargs_handlers=(
            [
                DistributedDataParallelKwargs(
                    find_unused_parameters=config.get("find_unused_parameters", False)
                )
            ]
            if torch.cuda.is_available()
            else None
        ),
    )
    if accelerator.is_main_process and not dry_run:
        accelerator.init_trackers(
            config["tracker_name"],
            config={
                k: json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in config.items()
                if v is not None
            },
        )

    train_ds, val_ds, bounds = load_datasets(config)
    if accelerator.is_main_process:
        (project_dir / "normalization_params.json").write_text(
            json.dumps(bounds, indent=2) + "\n"
        )
        (project_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    train_loader, val_loader = make_loaders(config, train_ds, val_ds)
    model = build_model(config)
    if config.get("initial_model_path"):

        model.load_state_dict(load_file(config["initial_model_path"]), strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler_config = config.get("lr_scheduler")
    scheduler = None
    if scheduler_config is not None:
        kind = scheduler_config.get("type", "cosine")
        if kind == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("t_max_epochs", config["epochs"])
                * len(train_loader),
                eta_min=scheduler_config.get("eta_min", 0),
            )
        elif kind == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        elif kind == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=scheduler_config.get("gamma", 0.95)
            )
        else:
            raise ValueError(f"Unknown learning-rate scheduler: {kind}")
    if scheduler is None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

    # Preserve both historical checkpoint layouts: mu/std or physical min/max,
    # followed by the epoch tracker. Model/optimizer/RNG use Accelerate's format.
    if bounds["normalization_type"] == "standard":
        first = TensorState(torch.tensor(bounds["mu"], device=accelerator.device), "mu")
        second = TensorState(
            torch.tensor(bounds["std"], device=accelerator.device), "std"
        )
    else:
        first = TensorState(
            torch.tensor(
                [bounds[f"{k}_min"] for k in ("E", "nu", "rho")],
                device=accelerator.device,
            ),
            "mins",
        )
        second = TensorState(
            torch.tensor(
                [bounds[f"{k}_max"] for k in ("E", "nu", "rho")],
                device=accelerator.device,
            ),
            "maxs",
        )
    epoch_tracker = EpochTracker()
    for state in (first, second, epoch_tracker):
        accelerator.register_for_checkpointing(state)

    checkpoint = config.get("resume_from_checkpoint") or latest_checkpoint(
        project_dir, scheduler is not None
    )
    starting_epoch = 0
    if checkpoint:
        accelerator.load_state(checkpoint)
        # The directory suffix counts saves, not epochs; use the saved tracker.
        starting_epoch = epoch_tracker.epoch
        accelerator.print(f"Resuming after epoch {starting_epoch}: {checkpoint}")
    # Advance past incomplete saves too, without moving or deleting them.
    saved_indices = [
        int(p.name.removeprefix("checkpoint_"))
        for p in (project_dir / "checkpoints").glob("checkpoint_*")
        if p.is_dir() and p.name.removeprefix("checkpoint_").isdigit()
    ]
    if checkpoint and Path(checkpoint).name.removeprefix("checkpoint_").isdigit():
        saved_indices.append(int(Path(checkpoint).name.removeprefix("checkpoint_")))
    accelerator.project_configuration.iteration = max(saved_indices, default=-1) + 1
    if starting_epoch >= config["epochs"]:
        accelerator.print(f"Training already reached {starting_epoch} epochs.")
        accelerator.end_training()
        return

    step_global = starting_epoch * len(train_loader)
    for epoch in range(starting_epoch, config["epochs"]):
        model.train()
        steps_per_epoch = len(train_loader)
        batches = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']}",
            disable=not accelerator.is_main_process,
            leave=False,
        )
        for step, (x,) in enumerate(batches):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    recon, kl, details = model(x)
                    loss, objective_stats = training_objective(
                        config,
                        recon,
                        kl,
                        details,
                        epoch=epoch,
                        step=step,
                        steps_per_epoch=steps_per_epoch,
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config["optimizer"]["grad_clip_norm"]
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients and not dry_run:
                accelerator.log(
                    {
                        "train/loss": loss.item(),
                        "train/recon": recon.item(),
                        "train/kl": kl.item(),
                        **{
                            f"train/{key}": value.item()
                            for key, value in objective_stats.items()
                        },
                        "train/lr": optimizer.param_groups[0]["lr"],
                        **{
                            f"train/{k}": v.item()
                            for k, v in details.items()
                            if k in ("mse_loss", "nll_loss")
                        },
                    },
                    step=step_global,
                )
            step_global += 1
            if dry_run and step_global >= 5:
                break
        if dry_run:
            break
        epoch_tracker.epoch = epoch + 1
        if (epoch + 1) % config.get("eval_interval", 5) == 0:
            evaluate(
                accelerator,
                model,
                train_loader,
                val_loader,
                bounds,
                config,
                step_global,
            )
        if (epoch + 1) % config.get("save_interval", 5) == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state()

    accelerator.wait_for_everyone()
    if not dry_run:
        accelerator.save_state()
        if accelerator.is_main_process:
            accelerator.save_model(
                accelerator.unwrap_model(model),
                project_dir / "final_model",
                safe_serialization=True,
            )
    accelerator.end_training()
