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

import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from vomp.datasets.sparse_voxel_materials import SparseVoxelMaterials
from vomp.models.geometry_encoder import (
    ElasticGeometryEncoder,
    ElasticSLatVoxelDecoder,
)
from vomp.models.material_vae.beta_tc import TripletVAE
from vomp.utils.data_utils import recursive_to_device


def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def l2_loss(pred, target):
    return F.mse_loss(pred, target)


def find_ckpt(cfg):
    cfg["load_ckpt"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":

            files = glob.glob(os.path.join(cfg.load_dir, "ckpts", "misc_step*.pt"))
            if len(files) != 0:
                cfg.load_ckpt = max(
                    [
                        int(os.path.basename(f).split("step")[-1].split(".")[0])
                        for f in files
                    ]
                )
        elif cfg.ckpt == "none":
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def load_model(cfg):

    model_dict = {}

    model_dict["geometry_encoder"] = ElasticGeometryEncoder(
        **cfg.models.geometry_encoder.args
    ).cuda()

    if "decoder" in cfg.models:
        model_dict["decoder"] = ElasticSLatVoxelDecoder(
            **cfg.models.decoder.args
        ).cuda()

    matvae = TripletVAE(**cfg.models.matvae.args).cuda()
    matvae_loaded = False
    if cfg.get("matvae_checkpoint") is not None:
        print(f"Loading matvae checkpoint from: {cfg.matvae_checkpoint}")
        checkpoint = load_file(cfg.matvae_checkpoint)
        matvae.load_state_dict(checkpoint, strict=True)
        print("Successfully loaded matvae checkpoint")
        matvae_loaded = True

    if not matvae_loaded:
        raise RuntimeError(
            "MatVAE checkpoint was not loaded. Please ensure matvae_checkpoint is specified in the config."
        )

    matvae.eval()
    for param in matvae.parameters():
        param.requires_grad = False

    geometry_encoder_loaded = False
    if cfg.load_ckpt is not None:

        geometry_encoder_path = os.path.join(
            cfg.load_dir, "ckpts", f"geometry_encoder_step{cfg.load_ckpt:07d}.pt"
        )
        if os.path.exists(geometry_encoder_path):
            print(f"Loading geometry encoder from: {geometry_encoder_path}")
            checkpoint = torch.load(
                geometry_encoder_path, map_location="cpu", weights_only=False
            )
            model_dict["geometry_encoder"].load_state_dict(checkpoint, strict=True)
            print("Successfully loaded geometry encoder")
            geometry_encoder_loaded = True
        else:
            raise RuntimeError(
                f"Geometry encoder checkpoint not found at {geometry_encoder_path}"
            )

        if "decoder" in model_dict:
            decoder_path = os.path.join(
                cfg.load_dir, "ckpts", f"decoder_step{cfg.load_ckpt:07d}.pt"
            )
            if os.path.exists(decoder_path):
                print(f"Loading decoder from: {decoder_path}")
                checkpoint = torch.load(
                    decoder_path, map_location="cpu", weights_only=False
                )
                model_dict["decoder"].load_state_dict(checkpoint, strict=True)
                print("Successfully loaded decoder")
            else:
                print(f"Warning: Decoder checkpoint not found at {decoder_path}")

    if not geometry_encoder_loaded:
        raise RuntimeError(
            "Geometry encoder checkpoint was not loaded. Please ensure a valid checkpoint exists in the checkpoint directory."
        )

    for model in model_dict.values():
        model.eval()

    return model_dict, matvae


@torch.inference_mode()
def evaluate_model(model_dict, matvae, val_dataset, cfg):

    print(f"Evaluating model on {len(val_dataset)} samples")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=min(cfg.get("batch_size", 4), 2),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=(
            val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None
        ),
    )

    total_loss = 0.0
    total_loss_youngs = 0.0
    total_loss_poisson = 0.0
    total_loss_density = 0.0

    all_pred_materials_original = []
    all_gt_materials_original = []
    object_identifiers = []  # Store object IDs/hashes
    processed_samples = 0
    total_voxels = 0

    # Per-object tracking - collect during processing, not after
    per_object_errors = {
        "youngs": [],
        "poisson": [],
        "density": [],
        "log_density": [],
        "overall": [],
    }
    per_object_displacement_errors = {"youngs": [], "poisson": [], "density": []}
    per_object_voxel_data = {"pred_values": [], "gt_values": [], "voxel_counts": []}

    # Pre-collect object identifiers from the dataset
    # The dataset has an instances attribute with (root, instance) tuples where instance is the hash
    dataset_instances = []
    if hasattr(val_dataset, "instances"):
        dataset_instances = [instance for root, instance in val_dataset.instances]
        print(f"Found {len(dataset_instances)} instance hashes in dataset")
    else:
        print("Warning: Dataset does not have instances attribute")

    current_sample_idx = 0  # Track which sample we're processing

    loss_type = cfg.get("trainer", {}).get("args", {}).get("loss_type", "l1")
    lambda_youngs_modulus = cfg.get("lambda_youngs_modulus", 1.0)
    lambda_poissons_ratio = cfg.get("lambda_poissons_ratio", 1.0)
    lambda_density = cfg.get("lambda_density", 1.0)
    training_mode = cfg.get("training_mode", "encoder_only")

    print(f"Using loss type: {loss_type}")
    print(f"Using training mode: {training_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, data in enumerate(val_dataloader):

        data = recursive_to_device(data, device, non_blocking=True)
        feats = data["feats"]
        materials = data["materials"]

        num_voxels = feats.feats.shape[0]
        num_samples = feats.shape[0]

        # Collect object identifiers from the pre-collected dataset instances
        batch_ids = []
        for i in range(num_samples):
            if current_sample_idx < len(dataset_instances):
                batch_ids.append(dataset_instances[current_sample_idx])
                current_sample_idx += 1
            else:
                batch_ids.append(f"unknown_sample_{current_sample_idx}")
                current_sample_idx += 1

        if batch_idx == 0:  # Print once to show what identifier we're using
            print(f"Using dataset instance hashes as object identifiers")

        object_identifiers.extend(batch_ids)

        z, mean, logvar = model_dict["geometry_encoder"](
            feats, sample_posterior=False, return_raw=True
        )
        gt_materials_normalized = materials.feats

        if training_mode == "encoder_only":

            latent_2d = z.feats

            (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = matvae.decode(
                latent_2d
            )

            E_pred = E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
            nu_pred = nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
            rho_pred = rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu

        elif training_mode == "encoder_decoder_matvae":

            decoder_output = model_dict["decoder"](z)
            latent_2d = decoder_output.feats

            (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = matvae.decode(
                latent_2d
            )

            E_pred = E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
            nu_pred = nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
            rho_pred = rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu

        elif training_mode == "encoder_decoder_direct":

            decoder_output = model_dict["decoder"](z)
            material_predictions = decoder_output.feats

            E_pred = material_predictions[:, 0]
            nu_pred = material_predictions[:, 1]
            rho_pred = material_predictions[:, 2]

        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        if loss_type == "l1":
            l1_E = l1_loss(E_pred, gt_materials_normalized[:, 0])
            l1_nu = l1_loss(nu_pred, gt_materials_normalized[:, 1])
            l1_rho = l1_loss(rho_pred, gt_materials_normalized[:, 2])

            loss = (
                lambda_youngs_modulus * l1_E
                + lambda_poissons_ratio * l1_nu
                + lambda_density * l1_rho
            )
            loss_youngs = l1_E
            loss_poisson = l1_nu
            loss_density = l1_rho

        elif loss_type == "l2":
            l2_E = l2_loss(E_pred, gt_materials_normalized[:, 0])
            l2_nu = l2_loss(nu_pred, gt_materials_normalized[:, 1])
            l2_rho = l2_loss(rho_pred, gt_materials_normalized[:, 2])

            loss = (
                lambda_youngs_modulus * l2_E
                + lambda_poissons_ratio * l2_nu
                + lambda_density * l2_rho
            )
            loss_youngs = l2_E
            loss_poisson = l2_nu
            loss_density = l2_rho

        total_loss += loss.item() * num_voxels
        total_loss_youngs += loss_youngs.item() * num_voxels
        total_loss_poisson += loss_poisson.item() * num_voxels
        total_loss_density += loss_density.item() * num_voxels

        pred_materials_normalized = torch.stack([E_pred, nu_pred, rho_pred], dim=-1)

        pred_materials_original = (
            val_dataset.material_transform.destandardize_and_inverse_transform_tensor(
                pred_materials_normalized
            )
        )
        gt_materials_original = (
            val_dataset.material_transform.destandardize_and_inverse_transform_tensor(
                gt_materials_normalized
            )
        )

        all_pred_materials_original.append(pred_materials_original.cpu())
        all_gt_materials_original.append(gt_materials_original.cpu())

        # Compute per-object statistics correctly using SparseTensor structure
        batch_coords = feats.coords  # Shape: [N, 4] where first column is batch index

        # Get material values for error computation
        pred_E_batch = pred_materials_original[:, 0]
        pred_nu_batch = pred_materials_original[:, 1]
        pred_rho_batch = pred_materials_original[:, 2]

        gt_E_batch = gt_materials_original[:, 0]
        gt_nu_batch = gt_materials_original[:, 1]
        gt_rho_batch = gt_materials_original[:, 2]

        # Compute relative errors for this batch
        log_pred_E_batch = torch.log10(torch.clamp_min(pred_E_batch, 1e-8))
        log_gt_E_batch = torch.log10(torch.clamp_min(gt_E_batch, 1e-8))
        rel_error_log_E_batch = torch.abs(
            log_pred_E_batch - log_gt_E_batch
        ) / torch.abs(log_gt_E_batch)

        rel_error_nu_batch = torch.abs(pred_nu_batch - gt_nu_batch) / torch.abs(
            gt_nu_batch
        )
        rel_error_rho_batch = torch.abs(pred_rho_batch - gt_rho_batch) / torch.abs(
            gt_rho_batch
        )

        # Log-space density errors
        log_pred_rho_batch = torch.log10(torch.clamp_min(pred_rho_batch, 1e-8))
        log_gt_rho_batch = torch.log10(torch.clamp_min(gt_rho_batch, 1e-8))
        rel_error_log_rho_batch = torch.abs(
            log_pred_rho_batch - log_gt_rho_batch
        ) / torch.abs(log_gt_rho_batch)

        # Compute displacement errors for this batch
        abs_error_log_E_batch = torch.abs(
            torch.log10(torch.clamp_min(pred_E_batch, 1e-8))
            - torch.log10(torch.clamp_min(gt_E_batch, 1e-8))
        )
        abs_error_nu_batch = torch.abs(pred_nu_batch - gt_nu_batch)
        abs_error_rho_batch = torch.abs(pred_rho_batch - gt_rho_batch)

        # Process each object in the batch separately
        for obj_in_batch in range(num_samples):
            # Find voxels belonging to this object (batch index == obj_in_batch)
            obj_voxel_mask = batch_coords[:, 0] == obj_in_batch
            num_obj_voxels = obj_voxel_mask.sum().item()

            if num_obj_voxels == 0:
                continue

            # Extract relative error data for this specific object
            obj_rel_error_log_E = rel_error_log_E_batch[obj_voxel_mask].cpu()
            obj_rel_error_nu = rel_error_nu_batch[obj_voxel_mask].cpu()
            obj_rel_error_rho = rel_error_rho_batch[obj_voxel_mask].cpu()
            obj_rel_error_log_rho = rel_error_log_rho_batch[obj_voxel_mask].cpu()

            # Extract displacement error data for this specific object
            obj_abs_error_log_E = abs_error_log_E_batch[obj_voxel_mask].cpu()
            obj_abs_error_nu = abs_error_nu_batch[obj_voxel_mask].cpu()
            obj_abs_error_rho = abs_error_rho_batch[obj_voxel_mask].cpu()

            # Compute per-object relative errors
            per_object_errors["youngs"].append(obj_rel_error_log_E.mean().item())
            per_object_errors["poisson"].append(obj_rel_error_nu.mean().item())
            per_object_errors["density"].append(obj_rel_error_rho.mean().item())
            per_object_errors["log_density"].append(obj_rel_error_log_rho.mean().item())

            # Compute per-object displacement errors
            per_object_displacement_errors["youngs"].append(
                obj_abs_error_log_E.mean().item()
            )
            per_object_displacement_errors["poisson"].append(
                obj_abs_error_nu.mean().item()
            )
            per_object_displacement_errors["density"].append(
                obj_abs_error_rho.mean().item()
            )

            # Compute overall error for this object (using log-space for Young's modulus and density)
            obj_all_errors = torch.cat(
                [obj_rel_error_log_E, obj_rel_error_nu, obj_rel_error_log_rho]
            )
            per_object_errors["overall"].append(obj_all_errors.mean().item())

            # Store voxel-level data for this object
            obj_pred_values = {
                "youngs": pred_E_batch[obj_voxel_mask].cpu(),
                "poisson": pred_nu_batch[obj_voxel_mask].cpu(),
                "density": pred_rho_batch[obj_voxel_mask].cpu(),
            }
            obj_gt_values = {
                "youngs": gt_E_batch[obj_voxel_mask].cpu(),
                "poisson": gt_nu_batch[obj_voxel_mask].cpu(),
                "density": gt_rho_batch[obj_voxel_mask].cpu(),
            }

            per_object_voxel_data["pred_values"].append(obj_pred_values)
            per_object_voxel_data["gt_values"].append(obj_gt_values)
            per_object_voxel_data["voxel_counts"].append(num_obj_voxels)

        processed_samples += num_samples
        total_voxels += num_voxels

        if batch_idx % 10 == 0:
            print(f"Processed {processed_samples} samples ({total_voxels} voxels)...")

    val_metrics = {
        "val_loss": total_loss / total_voxels if total_voxels > 0 else 0.0,
        "val_loss_youngs": (
            total_loss_youngs / total_voxels if total_voxels > 0 else 0.0
        ),
        "val_loss_poisson": (
            total_loss_poisson / total_voxels if total_voxels > 0 else 0.0
        ),
        "val_loss_density": (
            total_loss_density / total_voxels if total_voxels > 0 else 0.0
        ),
        "val_samples": processed_samples,
    }

    if all_pred_materials_original and all_gt_materials_original:
        all_pred_materials = torch.cat(all_pred_materials_original, dim=0)
        all_gt_materials = torch.cat(all_gt_materials_original, dim=0)

        pred_E = all_pred_materials[:, 0]
        pred_nu = all_pred_materials[:, 1]
        pred_rho = all_pred_materials[:, 2]

        gt_E = all_gt_materials[:, 0]
        gt_nu = all_gt_materials[:, 1]
        gt_rho = all_gt_materials[:, 2]

        log_pred_E = torch.log10(torch.clamp_min(pred_E, 1e-8))
        log_gt_E = torch.log10(torch.clamp_min(gt_E, 1e-8))
        rel_error_log_E = torch.abs(log_pred_E - log_gt_E) / torch.abs(log_gt_E)
        val_metrics["val_rel_error_log_youngs"] = rel_error_log_E.mean().item()
        val_metrics["val_rel_error_log_youngs_std"] = rel_error_log_E.std().item()

        rel_error_nu = torch.abs(pred_nu - gt_nu) / torch.abs(gt_nu)
        val_metrics["val_rel_error_poisson"] = rel_error_nu.mean().item()
        val_metrics["val_rel_error_poisson_std"] = rel_error_nu.std().item()

        rel_error_rho = torch.abs(pred_rho - gt_rho) / torch.abs(gt_rho)
        val_metrics["val_rel_error_density"] = rel_error_rho.mean().item()
        val_metrics["val_rel_error_density_std"] = rel_error_rho.std().item()

        # Log-space density errors
        log_pred_rho = torch.log10(torch.clamp_min(pred_rho, 1e-8))
        log_gt_rho = torch.log10(torch.clamp_min(gt_rho, 1e-8))
        rel_error_log_rho = torch.abs(log_pred_rho - log_gt_rho) / torch.abs(log_gt_rho)
        val_metrics["val_rel_error_log_density"] = rel_error_log_rho.mean().item()
        val_metrics["val_rel_error_log_density_std"] = rel_error_log_rho.std().item()

        # Displacement errors (absolute errors)
        abs_error_E = torch.abs(pred_E - gt_E)
        abs_error_nu = torch.abs(pred_nu - gt_nu)
        abs_error_rho = torch.abs(pred_rho - gt_rho)

        # Log displacement errors for Young's modulus
        abs_error_log_E = torch.abs(log_pred_E - log_gt_E)

        val_metrics["val_abs_error_youngs"] = abs_error_E.mean().item()
        val_metrics["val_abs_error_youngs_std"] = abs_error_E.std().item()
        val_metrics["val_abs_error_log_youngs"] = abs_error_log_E.mean().item()
        val_metrics["val_abs_error_log_youngs_std"] = abs_error_log_E.std().item()

        val_metrics["val_abs_error_poisson"] = abs_error_nu.mean().item()
        val_metrics["val_abs_error_poisson_std"] = abs_error_nu.std().item()

        val_metrics["val_abs_error_density"] = abs_error_rho.mean().item()
        val_metrics["val_abs_error_density_std"] = abs_error_rho.std().item()

        all_rel_errors = torch.cat([rel_error_log_E, rel_error_nu, rel_error_log_rho])
        val_metrics["val_rel_error_overall"] = all_rel_errors.mean().item()
        val_metrics["val_rel_error_overall_std"] = all_rel_errors.std().item()

        per_object_errors_tensor = {
            "youngs": torch.tensor(per_object_errors["youngs"]),
            "poisson": torch.tensor(per_object_errors["poisson"]),
            "density": torch.tensor(per_object_errors["density"]),
            "log_density": torch.tensor(per_object_errors["log_density"]),
            "overall": torch.tensor(per_object_errors["overall"]),
        }

        val_metrics["per_object_rel_error_log_youngs_mean"] = (
            per_object_errors_tensor["youngs"].mean().item()
        )
        val_metrics["per_object_rel_error_log_youngs_std"] = (
            per_object_errors_tensor["youngs"].std().item()
        )
        val_metrics["per_object_rel_error_log_youngs_min"] = (
            per_object_errors_tensor["youngs"].min().item()
        )
        val_metrics["per_object_rel_error_log_youngs_max"] = (
            per_object_errors_tensor["youngs"].max().item()
        )

        val_metrics["per_object_rel_error_poisson_mean"] = (
            per_object_errors_tensor["poisson"].mean().item()
        )
        val_metrics["per_object_rel_error_poisson_std"] = (
            per_object_errors_tensor["poisson"].std().item()
        )
        val_metrics["per_object_rel_error_poisson_min"] = (
            per_object_errors_tensor["poisson"].min().item()
        )
        val_metrics["per_object_rel_error_poisson_max"] = (
            per_object_errors_tensor["poisson"].max().item()
        )

        val_metrics["per_object_rel_error_density_mean"] = (
            per_object_errors_tensor["density"].mean().item()
        )
        val_metrics["per_object_rel_error_density_std"] = (
            per_object_errors_tensor["density"].std().item()
        )
        val_metrics["per_object_rel_error_density_min"] = (
            per_object_errors_tensor["density"].min().item()
        )
        val_metrics["per_object_rel_error_density_max"] = (
            per_object_errors_tensor["density"].max().item()
        )

        val_metrics["per_object_rel_error_log_density_mean"] = (
            per_object_errors_tensor["log_density"].mean().item()
        )
        val_metrics["per_object_rel_error_log_density_std"] = (
            per_object_errors_tensor["log_density"].std().item()
        )
        val_metrics["per_object_rel_error_log_density_min"] = (
            per_object_errors_tensor["log_density"].min().item()
        )
        val_metrics["per_object_rel_error_log_density_max"] = (
            per_object_errors_tensor["log_density"].max().item()
        )

        val_metrics["per_object_rel_error_overall_mean"] = (
            per_object_errors_tensor["overall"].mean().item()
        )
        val_metrics["per_object_rel_error_overall_std"] = (
            per_object_errors_tensor["overall"].std().item()
        )
        val_metrics["per_object_rel_error_overall_min"] = (
            per_object_errors_tensor["overall"].min().item()
        )
        val_metrics["per_object_rel_error_overall_max"] = (
            per_object_errors_tensor["overall"].max().item()
        )

        # Add per-object displacement error statistics
        per_object_displacement_errors_tensor = {
            "youngs": torch.tensor(per_object_displacement_errors["youngs"]),
            "poisson": torch.tensor(per_object_displacement_errors["poisson"]),
            "density": torch.tensor(per_object_displacement_errors["density"]),
        }

        val_metrics["per_object_abs_error_log_youngs_mean"] = (
            per_object_displacement_errors_tensor["youngs"].mean().item()
        )
        val_metrics["per_object_abs_error_log_youngs_std"] = (
            per_object_displacement_errors_tensor["youngs"].std().item()
        )

        val_metrics["per_object_abs_error_poisson_mean"] = (
            per_object_displacement_errors_tensor["poisson"].mean().item()
        )
        val_metrics["per_object_abs_error_poisson_std"] = (
            per_object_displacement_errors_tensor["poisson"].std().item()
        )

        val_metrics["per_object_abs_error_density_mean"] = (
            per_object_displacement_errors_tensor["density"].mean().item()
        )
        val_metrics["per_object_abs_error_density_std"] = (
            per_object_displacement_errors_tensor["density"].std().item()
        )

        val_metrics["per_object_errors"] = per_object_errors
        val_metrics["per_object_displacement_errors"] = per_object_displacement_errors
        val_metrics["object_identifiers"] = object_identifiers
        val_metrics["per_object_voxel_data"] = per_object_voxel_data

    return val_metrics


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate trained geometry encoder model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file used during training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latest",
        help="Checkpoint to load (latest, none, or step number)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/", help="Data directory"
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Print only voxel-level results in LaTeX table format",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    cfg = edict()
    cfg.update(config)
    cfg.load_dir = args.checkpoint_dir
    cfg.ckpt = args.ckpt
    cfg.data_dir = args.data_dir

    print("Evaluation Config:")
    print("=" * 50)
    print(f"Config file: {args.config}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data dir: {args.data_dir}")

    cfg = find_ckpt(cfg)
    if cfg.load_ckpt is not None:
        print(f"Will load checkpoint step: {cfg.load_ckpt}")
    else:
        print("No checkpoint found to load")

    # Determine normalization parameters file path
    normalization_params_file = None
    if cfg.get("matvae_checkpoint") is not None:
        # Extract the directory containing the matvae checkpoint
        matvae_checkpoint_path = cfg.matvae_checkpoint

        # Navigate to the project directory (up from checkpoints/checkpoint_X/)
        if "checkpoints" in matvae_checkpoint_path:
            # For paths like: outputs/matvae/checkpoints/checkpoint_821/model.safetensors
            # Need to go up 3 levels: model.safetensors -> checkpoint_X -> checkpoints -> project_dir
            matvae_project_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(matvae_checkpoint_path))
            )
            normalization_params_file = os.path.join(
                matvae_project_dir, "normalization_params.json"
            )
        else:
            # For other checkpoint path formats, try to find the normalization params
            checkpoint_dir = os.path.dirname(matvae_checkpoint_path)
            normalization_params_file = os.path.join(
                checkpoint_dir, "normalization_params.json"
            )

        if os.path.exists(normalization_params_file):
            print(f"Found normalization parameters file: {normalization_params_file}")
        else:
            print(
                f"ERROR: Could not find normalization parameters file at {normalization_params_file}"
            )
            print(
                "This file is required for consistent normalization between matvae and geometry encoder training!"
            )
            print(
                "Make sure the matvae was trained with the updated train_material_vae.py that saves normalization_params.json"
            )
            raise FileNotFoundError(
                f"Required normalization parameters file not found: {normalization_params_file}"
            )
    else:
        print("ERROR: No matvae_checkpoint specified in config!")
        print(
            "The matvae_checkpoint is required to locate the normalization parameters file."
        )
        raise ValueError(
            "matvae_checkpoint must be specified in config to load normalization parameters"
        )

    val_dataset_args = cfg.dataset.args.copy()
    val_dataset_args["split"] = "test"

    if "normalization_type" in cfg.dataset:
        val_dataset_args["normalization_type"] = cfg.dataset.normalization_type
        print(f"Using material normalization type: {cfg.dataset.normalization_type}")

    # Add normalization parameters file (now mandatory)
    val_dataset_args["normalization_params_file"] = normalization_params_file

    val_dataset = SparseVoxelMaterials(**val_dataset_args)
    print(f"Loaded test dataset with {len(val_dataset)} samples")

    model_dict, matvae = load_model(cfg)

    print("\nStarting evaluation...")
    print("=" * 50)

    val_metrics = evaluate_model(model_dict, matvae, val_dataset, cfg)

    # Check if --results flag was passed for LaTeX table format output
    if args.results:
        # Print only voxel-level results in LaTeX table format
        alde_youngs = val_metrics["val_abs_error_log_youngs"]
        alde_youngs_std = val_metrics["val_abs_error_log_youngs_std"]
        alre_youngs = val_metrics["val_rel_error_log_youngs"]
        alre_youngs_std = val_metrics["val_rel_error_log_youngs_std"]
        ade_poisson = val_metrics["val_abs_error_poisson"]
        ade_poisson_std = val_metrics["val_abs_error_poisson_std"]
        are_poisson = val_metrics["val_rel_error_poisson"]
        are_poisson_std = val_metrics["val_rel_error_poisson_std"]
        ade_density = val_metrics["val_abs_error_density"]
        ade_density_std = val_metrics["val_abs_error_density_std"]
        are_density = val_metrics["val_rel_error_density"]
        are_density_std = val_metrics["val_rel_error_density_std"]

        print(
            "%.4f {\\textbf{\\scriptsize\\textcolor{gray}{($\\pm$%.2f)}}} & "
            % (alde_youngs, alde_youngs_std)
            + "%.4f {\\textbf{\\scriptsize\\textcolor{gray}{($\\pm$%.2f)}}} & "
            % (alre_youngs, alre_youngs_std)
            + "%.4f {\\textbf{\\scriptsize\\textcolor{gray}{($\\pm$%.2f)}}} & "
            % (ade_poisson, ade_poisson_std)
            + "%.4f {\\textbf{\\scriptsize\\textcolor{gray}{($\\pm$%.2f)}}} & "
            % (are_poisson, are_poisson_std)
            + "%.4f {\\textbf{\\scriptsize\\textcolor{gray}{($\\pm$%.2f)}}} & "
            % (ade_density, ade_density_std)
            + "%.4f {\\textbf{\\scriptsize\\textcolor{gray}{($\\pm$%.2f)}}}"
            % (are_density, are_density_std)
        )
        return

    print("\n" + "=" * 80)
    print("COMPREHENSIVE RELATIVE ERROR ANALYSIS")
    print("=" * 80)

    print(f"\nDataset Summary:")
    print(f"  Total samples processed: {val_metrics['val_samples']}")

    print(f"\n1. VOXEL-LEVEL RELATIVE ERRORS (computed on all voxels):")
    print("-" * 60)
    print(f"Young's modulus (log space):")
    print(
        f"  Mean: {val_metrics['val_rel_error_log_youngs']:.6f} ({val_metrics['val_rel_error_log_youngs']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['val_rel_error_log_youngs_std']:.6f} ({val_metrics['val_rel_error_log_youngs_std']*100:.2f}%)"
    )

    print(f"\nPoisson's ratio:")
    print(
        f"  Mean: {val_metrics['val_rel_error_poisson']:.6f} ({val_metrics['val_rel_error_poisson']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['val_rel_error_poisson_std']:.6f} ({val_metrics['val_rel_error_poisson_std']*100:.2f}%)"
    )

    print(f"\nDensity:")
    print(
        f"  Mean: {val_metrics['val_rel_error_density']:.6f} ({val_metrics['val_rel_error_density']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['val_rel_error_density_std']:.6f} ({val_metrics['val_rel_error_density_std']*100:.2f}%)"
    )

    print(f"\nDensity (log space):")
    print(
        f"  Mean: {val_metrics['val_rel_error_log_density']:.6f} ({val_metrics['val_rel_error_log_density']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['val_rel_error_log_density_std']:.6f} ({val_metrics['val_rel_error_log_density_std']*100:.2f}%)"
    )

    print(f"\nOverall (all properties combined):")
    print(
        f"  Mean: {val_metrics['val_rel_error_overall']:.6f} ({val_metrics['val_rel_error_overall']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['val_rel_error_overall_std']:.6f} ({val_metrics['val_rel_error_overall_std']*100:.2f}%)"
    )

    print(f"\n2. PER-OBJECT RELATIVE ERRORS (averaged per object):")
    print("-" * 60)
    print(f"Young's modulus (log space):")
    print(
        f"  Mean: {val_metrics['per_object_rel_error_log_youngs_mean']:.6f} ({val_metrics['per_object_rel_error_log_youngs_mean']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['per_object_rel_error_log_youngs_std']:.6f} ({val_metrics['per_object_rel_error_log_youngs_std']*100:.2f}%)"
    )
    print(
        f"  Min:  {val_metrics['per_object_rel_error_log_youngs_min']:.6f} ({val_metrics['per_object_rel_error_log_youngs_min']*100:.2f}%)"
    )
    print(
        f"  Max:  {val_metrics['per_object_rel_error_log_youngs_max']:.6f} ({val_metrics['per_object_rel_error_log_youngs_max']*100:.2f}%)"
    )

    print(f"\nPoisson's ratio:")
    print(
        f"  Mean: {val_metrics['per_object_rel_error_poisson_mean']:.6f} ({val_metrics['per_object_rel_error_poisson_mean']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['per_object_rel_error_poisson_std']:.6f} ({val_metrics['per_object_rel_error_poisson_std']*100:.2f}%)"
    )
    print(
        f"  Min:  {val_metrics['per_object_rel_error_poisson_min']:.6f} ({val_metrics['per_object_rel_error_poisson_min']*100:.2f}%)"
    )
    print(
        f"  Max:  {val_metrics['per_object_rel_error_poisson_max']:.6f} ({val_metrics['per_object_rel_error_poisson_max']*100:.2f}%)"
    )

    print(f"\nDensity:")
    print(
        f"  Mean: {val_metrics['per_object_rel_error_density_mean']:.6f} ({val_metrics['per_object_rel_error_density_mean']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['per_object_rel_error_density_std']:.6f} ({val_metrics['per_object_rel_error_density_std']*100:.2f}%)"
    )
    print(
        f"  Min:  {val_metrics['per_object_rel_error_density_min']:.6f} ({val_metrics['per_object_rel_error_density_min']*100:.2f}%)"
    )
    print(
        f"  Max:  {val_metrics['per_object_rel_error_density_max']:.6f} ({val_metrics['per_object_rel_error_density_max']*100:.2f}%)"
    )

    print(f"\nDensity (log space):")
    print(
        f"  Mean: {val_metrics['per_object_rel_error_log_density_mean']:.6f} ({val_metrics['per_object_rel_error_log_density_mean']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['per_object_rel_error_log_density_std']:.6f} ({val_metrics['per_object_rel_error_log_density_std']*100:.2f}%)"
    )
    print(
        f"  Min:  {val_metrics['per_object_rel_error_log_density_min']:.6f} ({val_metrics['per_object_rel_error_log_density_min']*100:.2f}%)"
    )
    print(
        f"  Max:  {val_metrics['per_object_rel_error_log_density_max']:.6f} ({val_metrics['per_object_rel_error_log_density_max']*100:.2f}%)"
    )

    print(f"\nOverall (all properties combined):")
    print(
        f"  Mean: {val_metrics['per_object_rel_error_overall_mean']:.6f} ({val_metrics['per_object_rel_error_overall_mean']*100:.2f}%)"
    )
    print(
        f"  Std:  {val_metrics['per_object_rel_error_overall_std']:.6f} ({val_metrics['per_object_rel_error_overall_std']*100:.2f}%)"
    )
    print(
        f"  Min:  {val_metrics['per_object_rel_error_overall_min']:.6f} ({val_metrics['per_object_rel_error_overall_min']*100:.2f}%)"
    )
    print(
        f"  Max:  {val_metrics['per_object_rel_error_overall_max']:.6f} ({val_metrics['per_object_rel_error_overall_max']*100:.2f}%)"
    )

    print(f"\n3. ERROR DISTRIBUTION ANALYSIS:")
    print("-" * 60)
    import numpy as np

    per_obj_errors = val_metrics["per_object_errors"]
    object_ids = val_metrics.get("object_identifiers", [])

    for prop_name, prop_key in [
        ("Young's modulus", "youngs"),
        ("Poisson's ratio", "poisson"),
        ("Density", "density"),
        ("Density (log space)", "log_density"),
    ]:
        errors = np.array(per_obj_errors[prop_key])

        p25, p50, p75, p90, p95, p99 = np.percentile(errors, [25, 50, 75, 90, 95, 99])

        print(f"\n{prop_name} per-object error percentiles:")
        print(f"  25th: {p25:.4f} ({p25*100:.2f}%)")
        print(f"  50th: {p50:.4f} ({p50*100:.2f}%)")
        print(f"  75th: {p75:.4f} ({p75*100:.2f}%)")
        print(f"  90th: {p90:.4f} ({p90*100:.2f}%)")
        print(f"  95th: {p95:.4f} ({p95*100:.2f}%)")
        print(f"  99th: {p99:.4f} ({p99*100:.2f}%)")

        worst_objects = np.argsort(errors)[-5:]
        if object_ids:
            worst_object_ids = [object_ids[i] for i in worst_objects]
            worst_object_errors = [f"{errors[i]:.4f}" for i in worst_objects]
            print(f"  Top 5 worst objects:")
            for obj_id, error in zip(worst_object_ids, worst_object_errors):
                print(f"    {obj_id}: {error} ({float(error)*100:.2f}%)")
        else:
            print(
                f"  Top 5 worst objects (indices): {worst_objects} with errors: {[f'{errors[i]:.4f}' for i in worst_objects]}"
            )

        # Show detailed voxel analysis for density (the most problematic property)
        if prop_key == "density" and "per_object_voxel_data" in val_metrics:
            voxel_data = val_metrics["per_object_voxel_data"]

            for rank, obj_idx in enumerate(worst_objects):
                obj_id = object_ids[obj_idx] if object_ids else f"object_{obj_idx}"
                error = errors[obj_idx]

                # Get the voxel data for this object
                if obj_idx < len(voxel_data["pred_values"]):
                    pred_values = voxel_data["pred_values"][obj_idx]
                    gt_values = voxel_data["gt_values"][obj_idx]
                    num_voxels = voxel_data["voxel_counts"][obj_idx]

                    print(
                        f"\n  Object {rank+1}: {obj_id} (Error: {error:.4f}, {error*100:.2f}%)"
                    )
                    print(f"    Total voxels: {num_voxels}")
                    print(f"    5 random voxels (Predicted vs Ground Truth):")

                    # Get 5 random voxel indices
                    if num_voxels > 0:
                        random_indices = np.random.choice(
                            num_voxels, min(5, num_voxels), replace=False
                        )

                        for i, voxel_idx in enumerate(random_indices):
                            pred_density = pred_values["density"][voxel_idx].item()
                            gt_density = gt_values["density"][voxel_idx].item()
                            rel_error = abs(pred_density - gt_density) / abs(gt_density)

                            print(
                                f"      Voxel {i+1}: Pred={pred_density:.2f}, GT={gt_density:.2f}, RelErr={rel_error:.4f} ({rel_error*100:.2f}%)"
                            )
                    else:
                        print("      No voxels found for this object")

    print(f"\n4. INTERPRETATION:")
    print("-" * 60)

    density_std_ratio = (
        val_metrics["per_object_rel_error_density_std"]
        / val_metrics["per_object_rel_error_density_mean"]
    )
    log_density_std_ratio = (
        val_metrics["per_object_rel_error_log_density_std"]
        / val_metrics["per_object_rel_error_log_density_mean"]
    )
    youngs_std_ratio = (
        val_metrics["per_object_rel_error_log_youngs_std"]
        / val_metrics["per_object_rel_error_log_youngs_mean"]
    )
    poisson_std_ratio = (
        val_metrics["per_object_rel_error_poisson_std"]
        / val_metrics["per_object_rel_error_poisson_mean"]
    )

    print(f"Coefficient of Variation (std/mean) across objects:")
    print(f"  Young's modulus: {youngs_std_ratio:.3f}")
    print(f"  Poisson's ratio: {poisson_std_ratio:.3f}")
    print(f"  Density: {density_std_ratio:.3f}")
    print(f"  Density (log space): {log_density_std_ratio:.3f}")

    print("\n" + "=" * 80)

    # Summary tables with displacement and relative errors
    print("\n5. SUMMARY TABLES:")
    print("=" * 80)

    # Table 1: Voxel-level averages
    print("\nTable 1: VOXEL-LEVEL AVERAGES (computed on all voxels)")
    print("-" * 100)
    print(
        f"{'Property':<20} {'Avg Displacement Error ± Std':<35} {'Avg Relative Error ± Std':<35}"
    )
    print("-" * 100)

    # Young's modulus (log space)
    log_disp_youngs = val_metrics["val_abs_error_log_youngs"]
    log_disp_youngs_std = val_metrics["val_abs_error_log_youngs_std"]
    log_rel_youngs = val_metrics["val_rel_error_log_youngs"]
    log_rel_youngs_std = val_metrics["val_rel_error_log_youngs_std"]
    print(
        f"{'Youngs Modulus':<20} {f'{log_disp_youngs:.4f}±{log_disp_youngs_std:.2f} (log)':<35} {f'{log_rel_youngs:.4f}±{log_rel_youngs_std:.2f} ({log_rel_youngs*100:.2f}%)':<35}"
    )

    # Density
    disp_density = val_metrics["val_abs_error_density"]
    disp_density_std = val_metrics["val_abs_error_density_std"]
    rel_density = val_metrics["val_rel_error_density"]
    rel_density_std = val_metrics["val_rel_error_density_std"]
    print(
        f"{'Density':<20} {f'{disp_density:.4f}±{disp_density_std:.2f}':<35} {f'{rel_density:.4f}±{rel_density_std:.2f} ({rel_density*100:.2f}%)':<35}"
    )

    # Poisson's ratio
    disp_poisson = val_metrics["val_abs_error_poisson"]
    disp_poisson_std = val_metrics["val_abs_error_poisson_std"]
    rel_poisson = val_metrics["val_rel_error_poisson"]
    rel_poisson_std = val_metrics["val_rel_error_poisson_std"]
    print(
        f"{'Poissons Ratio':<20} {f'{disp_poisson:.4f}±{disp_poisson_std:.2f}':<35} {f'{rel_poisson:.4f}±{rel_poisson_std:.2f} ({rel_poisson*100:.2f}%)':<35}"
    )

    print("-" * 100)

    # Table 2: Per-object averages
    print("\nTable 2: PER-OBJECT AVERAGES (errors averaged per voxel, then per object)")
    print("-" * 100)
    print(
        f"{'Property':<20} {'Per-Obj Displacement Error ± Std':<35} {'Per-Obj Relative Error ± Std':<35}"
    )
    print("-" * 100)

    # Young's modulus (log space) - per object
    per_obj_abs_youngs = val_metrics["per_object_abs_error_log_youngs_mean"]
    per_obj_abs_youngs_std = val_metrics["per_object_abs_error_log_youngs_std"]
    per_obj_rel_youngs = val_metrics["per_object_rel_error_log_youngs_mean"]
    per_obj_rel_youngs_std = val_metrics["per_object_rel_error_log_youngs_std"]
    print(
        f"{'Youngs Modulus':<20} {f'{per_obj_abs_youngs:.4f}±{per_obj_abs_youngs_std:.2f} (log)':<35} {f'{per_obj_rel_youngs:.4f}±{per_obj_rel_youngs_std:.2f} ({per_obj_rel_youngs*100:.2f}%)':<35}"
    )

    # Density - per object
    per_obj_abs_density = val_metrics["per_object_abs_error_density_mean"]
    per_obj_abs_density_std = val_metrics["per_object_abs_error_density_std"]
    per_obj_rel_density = val_metrics["per_object_rel_error_density_mean"]
    per_obj_rel_density_std = val_metrics["per_object_rel_error_density_std"]
    print(
        f"{'Density':<20} {f'{per_obj_abs_density:.4f}±{per_obj_abs_density_std:.2f}':<35} {f'{per_obj_rel_density:.4f}±{per_obj_rel_density_std:.2f} ({per_obj_rel_density*100:.2f}%)':<35}"
    )

    # Poisson's ratio - per object
    per_obj_abs_poisson = val_metrics["per_object_abs_error_poisson_mean"]
    per_obj_abs_poisson_std = val_metrics["per_object_abs_error_poisson_std"]
    per_obj_rel_poisson = val_metrics["per_object_rel_error_poisson_mean"]
    per_obj_rel_poisson_std = val_metrics["per_object_rel_error_poisson_std"]
    print(
        f"{'Poissons Ratio':<20} {f'{per_obj_abs_poisson:.4f}±{per_obj_abs_poisson_std:.2f}':<35} {f'{per_obj_rel_poisson:.4f}±{per_obj_rel_poisson_std:.2f} ({per_obj_rel_poisson*100:.2f}%)':<35}"
    )

    print("=" * 100)

    results_to_save = {
        k: v
        for k, v in val_metrics.items()
        if k not in ["per_object_errors", "object_identifiers", "per_object_voxel_data"]
    }
    results_file = os.path.join(args.checkpoint_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=4)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
