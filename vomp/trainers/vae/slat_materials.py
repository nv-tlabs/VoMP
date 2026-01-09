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

from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch

from ..basic import BasicTrainer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips
from ...utils.data_utils import recursive_to_device


class SLatVaeMaterialsTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE for materials.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
        lambda_kl (float): KL loss weight.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        i_eval (int): Evaluation interval.
        trellis_weights_path (str): Path to TRELLIS weights directory.
    """

    def __init__(
        self,
        *args,
        loss_type: str = "l1",
        lambda_youngs_modulus: float = 1.0,
        lambda_poissons_ratio: float = 1.0,
        lambda_density: float = 1.0,
        matvae=None,  # Add matvae as a separate parameter
        val_dataset=None,  # Add validation dataset
        i_eval: int = 1000,  # Add evaluation interval
        trellis_weights_path: str = None,  # Add path to TRELLIS weights directory
        training_mode: str = "encoder_only",  # Add training mode parameter
        **kwargs,
    ):
        self.trellis_weights_path = trellis_weights_path

        super().__init__(*args, **kwargs)

        self.loss_type = loss_type
        self.lambda_youngs_modulus = lambda_youngs_modulus
        self.lambda_poissons_ratio = lambda_poissons_ratio
        self.lambda_density = lambda_density
        self.matvae = matvae  # Store the frozen matvae model
        self.val_dataset = val_dataset  # Store validation dataset
        self.i_eval = i_eval  # Store evaluation interval
        self.training_mode = training_mode  # Store training mode

        # Validate training mode
        valid_modes = [
            "encoder_only",
            "encoder_decoder_matvae",
            "encoder_decoder_direct",
        ]
        if self.training_mode not in valid_modes:
            raise ValueError(
                f"Invalid training_mode: {self.training_mode}. Must be one of {valid_modes}"
            )

        # Validate model availability based on training mode
        if self.training_mode in ["encoder_decoder_matvae", "encoder_decoder_direct"]:
            if "decoder" not in self.models:
                raise ValueError(
                    f"Training mode '{self.training_mode}' requires a decoder model in config"
                )

        # Load TRELLIS pre-trained weights only if no checkpoint was loaded
        if self.trellis_weights_path is not None and self.step == 0:
            self._load_trellis_pretrained_weights()

    def _load_trellis_pretrained_weights(self):
        """
        Load pre-trained TRELLIS weights for the geometry encoder and decoder.
        """
        import os
        from safetensors.torch import load_file

        safetensors_path = os.path.join(
            self.trellis_weights_path, "ckpts", "slat_enc_swin8_B_64l8_fp16.safetensors"
        )

        if self.is_master:
            print(f"Loading TRELLIS pre-trained weights from: {safetensors_path}")

        # Load the safetensors file
        pretrained_state_dict = load_file(safetensors_path)

        # Load weights for geometry encoder
        self._load_pretrained_weights_for_model(
            "geometry_encoder", pretrained_state_dict
        )

        # Load weights for decoder if it exists
        if "decoder" in self.models:
            self._load_pretrained_weights_for_model("decoder", pretrained_state_dict)

    def _load_pretrained_weights_for_model(self, model_name, pretrained_state_dict):
        """
        Load pretrained weights for a specific model.

        Args:
            model_name: Name of the model ("geometry_encoder" or "decoder")
            pretrained_state_dict: Dictionary of pretrained weights
        """
        model = self.models.get(model_name)
        if model is None:
            return

        model_state_dict = model.state_dict()

        # Filter to only include parameters that match both name and shape
        filtered_state_dict = {}
        loaded_count = 0
        skipped_count = 0

        for name, param in pretrained_state_dict.items():
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    filtered_state_dict[name] = param
                    loaded_count += 1
                    if self.is_master:
                        print(f"✓ Loading {model_name}: {name} {param.shape}")
                else:
                    skipped_count += 1
                    if self.is_master:
                        print(
                            f"✗ Shape mismatch {model_name}: {name} | TRELLIS: {param.shape} | Model: {model_state_dict[name].shape}"
                        )
            else:
                skipped_count += 1
                if self.is_master:
                    print(f"✗ Not in {model_name}: {name} {param.shape}")

        # Load only the filtered weights
        model.load_state_dict(filtered_state_dict, strict=False)

        if self.is_master:
            total_pretrained = len(pretrained_state_dict)
            print(
                f"Successfully loaded {loaded_count}/{total_pretrained} parameters from FRANKENSTEIN pre-trained weights for {model_name}"
            )
            print(
                f"Skipped {skipped_count} parameters due to name/shape mismatches for {model_name}"
            )
            print(f"--- {model_name} pretrained weight loading complete ---\n")

    def training_losses(
        self,
        feats: SparseTensor,
        materials: SparseTensor,
        return_aux: bool = False,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            materials: The [N x * x 3] sparse tensor of materials.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        # Step 1: Run features through geometry encoder to get 2D latents per voxel
        z, mean, logvar = self.training_models["geometry_encoder"](
            feats, sample_posterior=False, return_raw=True
        )

        gt_materials = materials.feats  # Shape: [num_voxels, 3]

        if self.training_mode == "encoder_only":
            # Current behavior: encoder → matvae
            # Step 2: Extract the 2D latent features
            latent_2d = z.feats  # Shape: [num_voxels, 2]

            # Step 3: Run latents through material VAE decoder
            (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                self.matvae.decode(latent_2d)
            )

            # Extract predictions (remove last dimension if it exists)
            E_pred = (
                E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
            )  # Shape: [total_voxels]
            nu_pred = (
                nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
            )  # Shape: [total_voxels]
            rho_pred = (
                rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu
            )  # Shape: [total_voxels]

        elif self.training_mode == "encoder_decoder_matvae":
            # New mode: encoder → decoder → matvae
            # Step 2: Run latents through decoder to get 2D outputs
            decoder_output = self.training_models["decoder"](
                z
            )  # Should output 2D latents
            latent_2d = decoder_output.feats  # Shape: [num_voxels, 2]

            # Step 3: Run decoder output through material VAE decoder
            (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                self.matvae.decode(latent_2d)
            )

            # Extract predictions (remove last dimension if it exists)
            E_pred = (
                E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
            )  # Shape: [total_voxels]
            nu_pred = (
                nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
            )  # Shape: [total_voxels]
            rho_pred = (
                rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu
            )  # Shape: [total_voxels]

        elif self.training_mode == "encoder_decoder_direct":
            # New mode: encoder → decoder → direct 3D outputs
            # Step 2: Run latents through decoder to get 3D material outputs directly
            decoder_output = self.training_models["decoder"](
                z
            )  # Should output 3D materials
            material_predictions = decoder_output.feats  # Shape: [num_voxels, 3]

            # Extract predictions directly
            E_pred = material_predictions[:, 0]  # Shape: [total_voxels]
            nu_pred = material_predictions[:, 1]  # Shape: [total_voxels]
            rho_pred = material_predictions[:, 2]  # Shape: [total_voxels]

        # Compute reconstruction loss based on loss_type (same for all modes)
        if self.loss_type == "l1":
            l1_E = l1_loss(E_pred, gt_materials[:, 0])
            l1_nu = l1_loss(nu_pred, gt_materials[:, 1])
            l1_rho = l1_loss(rho_pred, gt_materials[:, 2])

            loss = (
                self.lambda_youngs_modulus * l1_E
                + self.lambda_poissons_ratio * l1_nu
                + self.lambda_density * l1_rho
            )

            # Individual loss components for logging
            loss_youngs = l1_E
            loss_poisson = l1_nu
            loss_density = l1_rho

        elif self.loss_type == "l2":
            l2_E = l2_loss(E_pred, gt_materials[:, 0])
            l2_nu = l2_loss(nu_pred, gt_materials[:, 1])
            l2_rho = l2_loss(rho_pred, gt_materials[:, 2])

            loss = (
                self.lambda_youngs_modulus * l2_E
                + self.lambda_poissons_ratio * l2_nu
                + self.lambda_density * l2_rho
            )

            # Individual loss components for logging
            loss_youngs = l2_E
            loss_poisson = l2_nu
            loss_density = l2_rho
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if return_aux:
            return loss, {"latent_2d": z}

        # Return in the expected format: (loss_dict, status_dict)
        loss_dict = {
            "loss": loss,
            "loss_youngs": loss_youngs,
            "loss_poisson": loss_poisson,
            "loss_density": loss_density,
        }
        status_dict = {}

        return loss_dict, status_dict

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run material property prediction on samples from the dataset.

        Args:
            num_samples: Number of samples to process
            batch_size: Batch size for processing
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing predicted and ground truth material properties
        """
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # OPTIMIZED: Use multiple workers instead of 0
            pin_memory=True,  # OPTIMIZED: Add pin_memory for faster GPU transfer
            persistent_workers=True,  # OPTIMIZED: Keep workers alive
            collate_fn=(
                self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None
            ),
        )

        # Material prediction inference
        ret_dict = {}
        all_gt_materials = []
        all_pred_materials = []
        all_latents = []

        processed_samples = 0

        for data in dataloader:
            if processed_samples >= num_samples:
                break

            # Move data to device
            feats = data["feats"].cuda() if "feats" in data else None
            materials = data["materials"].cuda() if "materials" in data else None

            if feats is None or materials is None:
                continue

            if verbose:
                print(f"Processing batch: {feats.feats.shape[0]} voxels")

            # Step 1: Run through geometry encoder (deterministic for snapshot)
            z, mean, logvar = self.models["geometry_encoder"](
                feats, sample_posterior=False, return_raw=True
            )

            gt_materials = materials.feats  # Shape: [num_voxels, 3]

            if self.training_mode == "encoder_only":
                # Current behavior: encoder → matvae
                # Step 2: Extract latents and ground truth
                latent_2d = z.feats  # Shape: [num_voxels, 2]

                # Step 3: Run latents through material VAE decoder
                (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                    self.matvae.decode(latent_2d)
                )

                # Extract predictions (remove last dimension if it exists)
                E_pred = (
                    E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
                )  # Shape: [total_voxels]
                nu_pred = (
                    nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
                )  # Shape: [total_voxels]
                rho_pred = (
                    rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu
                )  # Shape: [total_voxels]

            elif self.training_mode == "encoder_decoder_matvae":
                # New mode: encoder → decoder → matvae
                # Step 2: Run latents through decoder to get 2D outputs
                decoder_output = self.models["decoder"](z)  # Should output 2D latents
                latent_2d = decoder_output.feats  # Shape: [num_voxels, 2]

                # Step 3: Run decoder output through material VAE decoder
                (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                    self.matvae.decode(latent_2d)
                )

                # Extract predictions (remove last dimension if it exists)
                E_pred = (
                    E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
                )  # Shape: [total_voxels]
                nu_pred = (
                    nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
                )  # Shape: [total_voxels]
                rho_pred = (
                    rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu
                )  # Shape: [total_voxels]

            elif self.training_mode == "encoder_decoder_direct":
                # New mode: encoder → decoder → direct 3D outputs
                # Step 2: Run latents through decoder to get 3D material outputs directly
                decoder_output = self.models["decoder"](z)  # Should output 3D materials
                material_predictions = decoder_output.feats  # Shape: [num_voxels, 3]

                # Extract predictions directly
                E_pred = material_predictions[:, 0]  # Shape: [total_voxels]
                nu_pred = material_predictions[:, 1]  # Shape: [total_voxels]
                rho_pred = material_predictions[:, 2]  # Shape: [total_voxels]

            # Stack predictions [num_voxels, 3]
            pred_materials = torch.stack([E_pred, nu_pred, rho_pred], dim=-1)

            # Store results
            all_gt_materials.append(gt_materials.cpu())
            all_pred_materials.append(pred_materials.cpu())
            all_latents.append(z.feats.cpu())

            processed_samples += feats.feats.shape[0]

            if verbose:
                print(f"Processed {processed_samples}/{num_samples} voxels")

        # Concatenate all results
        all_gt_materials = torch.cat(all_gt_materials, dim=0)[:num_samples]
        all_pred_materials = torch.cat(all_pred_materials, dim=0)[:num_samples]

        # Return results
        ret_dict.update(
            {
                "gt_materials": {"value": all_gt_materials, "type": "tensor"},
                "pred_materials": {"value": all_pred_materials, "type": "tensor"},
            }
        )

        return ret_dict

    @torch.no_grad()
    def validate(self, num_samples: int = 1000) -> Dict:
        """
        Run validation on the validation dataset.

        Args:
            num_samples: Maximum number of samples to validate on

        Returns:
            Dictionary containing validation metrics
        """
        if self.val_dataset is None:
            return {}

        # Only run validation on master process - return empty dict on other processes
        if not self.is_master:
            return {}

        val_metrics = {}

        try:
            # Create validation dataloader without DDP sampler for consistent results
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=min(
                    self.batch_size_per_gpu, 2
                ),  # Use smaller batch size for stability
                shuffle=False,
                num_workers=4,  # OPTIMIZED: Use multiple workers instead of 0
                pin_memory=True,  # OPTIMIZED: Add pin_memory for faster GPU transfer
                persistent_workers=True,  # OPTIMIZED: Keep workers alive
                collate_fn=(
                    self.val_dataset.collate_fn
                    if hasattr(self.val_dataset, "collate_fn")
                    else None
                ),
            )

            total_loss = 0.0
            total_loss_youngs = 0.0
            total_loss_poisson = 0.0
            total_loss_density = 0.0

            all_pred_materials_original = []
            all_gt_materials_original = []
            processed_samples = 0

            for data in val_dataloader:
                # Move data to device
                data = recursive_to_device(data, self.device, non_blocking=True)
                feats = data["feats"]
                materials = data["materials"]

                # Same forward pass as training_losses
                z, mean, logvar = self.training_models["geometry_encoder"](
                    feats, sample_posterior=False, return_raw=True
                )
                gt_materials_normalized = (
                    materials.feats
                )  # Shape: [num_voxels, 3] (normalized)

                if self.training_mode == "encoder_only":
                    # Current behavior: encoder → matvae
                    latent_2d = z.feats  # Shape: [num_voxels, 2]

                    # Get predictions from material VAE
                    (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                        self.matvae.decode(latent_2d)
                    )

                    # Extract predictions (remove last dimension if it exists)
                    E_pred = (
                        E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
                    )  # Shape: [total_voxels]
                    nu_pred = (
                        nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
                    )  # Shape: [total_voxels]
                    rho_pred = (
                        rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu
                    )  # Shape: [total_voxels]

                elif self.training_mode == "encoder_decoder_matvae":
                    # New mode: encoder → decoder → matvae
                    decoder_output = self.models["decoder"](
                        z
                    )  # Should output 2D latents
                    latent_2d = decoder_output.feats  # Shape: [num_voxels, 2]

                    # Get predictions from material VAE
                    (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                        self.matvae.decode(latent_2d)
                    )

                    # Extract predictions (remove last dimension if it exists)
                    E_pred = (
                        E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
                    )  # Shape: [total_voxels]
                    nu_pred = (
                        nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
                    )  # Shape: [total_voxels]
                    rho_pred = (
                        rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu
                    )  # Shape: [total_voxels]

                elif self.training_mode == "encoder_decoder_direct":
                    # New mode: encoder → decoder → direct 3D outputs
                    decoder_output = self.models["decoder"](
                        z
                    )  # Should output 3D materials
                    material_predictions = (
                        decoder_output.feats
                    )  # Shape: [num_voxels, 3]

                    # Extract predictions directly
                    E_pred = material_predictions[:, 0]  # Shape: [total_voxels]
                    nu_pred = material_predictions[:, 1]  # Shape: [total_voxels]
                    rho_pred = material_predictions[:, 2]  # Shape: [total_voxels]

                # Compute losses (same as training_losses)
                if self.loss_type == "l1":
                    l1_E = l1_loss(E_pred, gt_materials_normalized[:, 0])
                    l1_nu = l1_loss(nu_pred, gt_materials_normalized[:, 1])
                    l1_rho = l1_loss(rho_pred, gt_materials_normalized[:, 2])

                    loss = (
                        self.lambda_youngs_modulus * l1_E
                        + self.lambda_poissons_ratio * l1_nu
                        + self.lambda_density * l1_rho
                    )
                    loss_youngs = l1_E
                    loss_poisson = l1_nu
                    loss_density = l1_rho

                elif self.loss_type == "l2":
                    l2_E = l2_loss(E_pred, gt_materials_normalized[:, 0])
                    l2_nu = l2_loss(nu_pred, gt_materials_normalized[:, 1])
                    l2_rho = l2_loss(rho_pred, gt_materials_normalized[:, 2])

                    loss = (
                        self.lambda_youngs_modulus * l2_E
                        + self.lambda_poissons_ratio * l2_nu
                        + self.lambda_density * l2_rho
                    )
                    loss_youngs = l2_E
                    loss_poisson = l2_nu
                    loss_density = l2_rho

                # Accumulate losses
                batch_size = feats.feats.shape[0]
                total_loss += loss.item() * batch_size
                total_loss_youngs += loss_youngs.item() * batch_size
                total_loss_poisson += loss_poisson.item() * batch_size
                total_loss_density += loss_density.item() * batch_size

                # Stack predictions [num_voxels, 3] (normalized)
                pred_materials_normalized = torch.stack(
                    [E_pred, nu_pred, rho_pred], dim=-1
                )

                pred_materials_original = self.val_dataset.material_transform.destandardize_and_inverse_transform_tensor(
                    pred_materials_normalized
                )
                gt_materials_original = self.val_dataset.material_transform.destandardize_and_inverse_transform_tensor(
                    gt_materials_normalized
                )

                # Store for relative error computation
                all_pred_materials_original.append(pred_materials_original.cpu())
                all_gt_materials_original.append(gt_materials_original.cpu())

                processed_samples += batch_size

            val_metrics = {
                "val_loss": (
                    total_loss / processed_samples if processed_samples > 0 else 0.0
                ),
                "val_loss_youngs": (
                    total_loss_youngs / processed_samples
                    if processed_samples > 0
                    else 0.0
                ),
                "val_loss_poisson": (
                    total_loss_poisson / processed_samples
                    if processed_samples > 0
                    else 0.0
                ),
                "val_loss_density": (
                    total_loss_density / processed_samples
                    if processed_samples > 0
                    else 0.0
                ),
                "val_samples": processed_samples,
            }

            # Compute relative errors on original scale
            if all_pred_materials_original and all_gt_materials_original:
                all_pred_materials = torch.cat(all_pred_materials_original, dim=0)
                all_gt_materials = torch.cat(all_gt_materials_original, dim=0)

                pred_E = all_pred_materials[:, 0]  # Young's modulus
                pred_nu = all_pred_materials[:, 1]  # Poisson's ratio
                pred_rho = all_pred_materials[:, 2]  # Density

                gt_E = all_gt_materials[:, 0]
                gt_nu = all_gt_materials[:, 1]
                gt_rho = all_gt_materials[:, 2]

                # Young's modulus relative error in log space
                log_pred_E = torch.log10(torch.clamp_min(pred_E, 1e-8))
                log_gt_E = torch.log10(torch.clamp_min(gt_E, 1e-8))
                rel_error_log_E = torch.abs(log_pred_E - log_gt_E) / torch.abs(log_gt_E)
                val_metrics["val_rel_error_log_youngs"] = rel_error_log_E.mean().item()

                # Poisson's ratio relative error in original space
                rel_error_nu = torch.abs(pred_nu - gt_nu) / torch.abs(gt_nu)
                val_metrics["val_rel_error_poisson"] = rel_error_nu.mean().item()

                # Density relative error in original space
                rel_error_rho = torch.abs(pred_rho - gt_rho) / torch.abs(gt_rho)
                val_metrics["val_rel_error_density"] = rel_error_rho.mean().item()

                # Also compute overall relative error (all properties combined)
                all_rel_errors = torch.cat(
                    [rel_error_log_E, rel_error_nu, rel_error_rho]
                )
                val_metrics["val_rel_error_overall"] = all_rel_errors.mean().item()

        except Exception as e:
            print(f"Error during validation: {e}")
            import traceback

            traceback.print_exc()
            val_metrics = {}

        return val_metrics
