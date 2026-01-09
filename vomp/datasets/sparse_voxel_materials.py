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
from PIL import Image
import json
import numpy as np
import pandas as pd
import torch
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from ..utils.material_transforms import MaterialPropertyTransform
from .components import StandardDatasetBase


class SparseVoxelMaterials(StandardDatasetBase):
    """
    SparseVoxelMaterials dataset.

    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        model (str): model name
        resolution (int): resolution of the data
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        compute_material_stats (bool): whether to recompute material statistics
        split (str): split filter
        normalization_type (str): type of normalization ("standard" or "log_minmax")
    """

    def __init__(
        self,
        roots: str,
        image_size: int,
        model: str = "dinov2_vitl14_reg",
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        compute_material_stats: bool = False,
        split: str = None,
        normalization_type: str = "log_minmax",
        normalization_params_file: str = None,  # Add parameter to load normalization params
    ):
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.split = split
        self.value_range = (0, 1)
        self.normalization_type = normalization_type

        self.material_transform = MaterialPropertyTransform(
            normalization_type=normalization_type
        )

        # Load normalization parameters from file (optional for direct prediction)
        if normalization_params_file is not None:
            if not os.path.exists(normalization_params_file):
                raise FileNotFoundError(
                    f"Normalization parameters file not found: {normalization_params_file}"
                )
            print(f"Loading normalization parameters from: {normalization_params_file}")
            self._load_normalization_params(normalization_params_file)
        else:
            print(
                "No normalization parameters file provided - using dataset-derived normalization for direct prediction"
            )

        super().__init__(roots)

        if compute_material_stats and normalization_type == "standard":
            print(
                "Recomputing material transform statistics for standard normalization..."
            )
            self._compute_material_stats()

    def _load_normalization_params(self, params_file: str):
        """Load normalization parameters from JSON file (saved during matvae training)."""
        import json

        with open(params_file, "r") as f:
            params = json.load(f)

        # Validate normalization type consistency
        if params.get("normalization_type") != self.normalization_type:
            raise ValueError(
                f"Normalization type mismatch! "
                f"Dataset config: {self.normalization_type}, "
                f"Params file: {params.get('normalization_type')}"
            )

        if self.normalization_type == "standard":
            if "mu" not in params or "std" not in params:
                raise ValueError(
                    "Standard normalization requires 'mu' and 'std' in params file"
                )

            self.material_transform.mu = torch.tensor(params["mu"])
            self.material_transform.std = torch.tensor(params["std"])
            self.material_transform.nu_min = params.get("nu_min", 0.0)
            self.material_transform.nu_max = params.get("nu_max", 0.5)
            self.material_transform._stats_computed = True

            print(f"Loaded standard normalization params:")
            print(f"  mu: {self.material_transform.mu}")
            print(f"  std: {self.material_transform.std}")
            print(f"  nu_min: {self.material_transform.nu_min}")
            print(f"  nu_max: {self.material_transform.nu_max}")

        elif self.normalization_type == "log_minmax":
            required_keys = ["E_min", "E_max", "nu_min", "nu_max", "rho_min", "rho_max"]
            for key in required_keys:
                if key not in params:
                    raise ValueError(
                        f"Log minmax normalization requires '{key}' in params file"
                    )

            # Convert raw values to log space (params file stores raw values, but transform expects log values)
            import math

            self.material_transform.E_min = math.log10(params["E_min"])
            self.material_transform.E_max = math.log10(params["E_max"])
            self.material_transform.nu_min = params["nu_min"]
            self.material_transform.nu_max = params["nu_max"]
            self.material_transform.rho_min = math.log10(params["rho_min"])
            self.material_transform.rho_max = math.log10(params["rho_max"])
            self.material_transform._stats_computed = True

            print(f"Loaded log minmax normalization params:")
            print(
                f"  E_min: {self.material_transform.E_min:.6f} (log10({params['E_min']:.0f} Pa))"
            )
            print(
                f"  E_max: {self.material_transform.E_max:.6f} (log10({params['E_max']:.0f} Pa))"
            )
            print(f"  nu_min: {self.material_transform.nu_min:.6f}")
            print(f"  nu_max: {self.material_transform.nu_max:.6f}")
            print(
                f"  rho_min: {self.material_transform.rho_min:.6f} (log10({params['rho_min']:.2f} kg/m³))"
            )
            print(
                f"  rho_max: {self.material_transform.rho_max:.6f} (log10({params['rho_max']:.2f} kg/m³))"
            )

        elif self.normalization_type == "log_minmax_no_density":
            required_keys = ["E_min", "E_max", "nu_min", "nu_max", "rho_min", "rho_max"]
            for key in required_keys:
                if key not in params:
                    raise ValueError(
                        f"Log minmax no density normalization requires '{key}' in params file"
                    )

            # Convert E values to log space, keep density in raw space
            import math

            self.material_transform.E_min = math.log10(params["E_min"])
            self.material_transform.E_max = math.log10(params["E_max"])
            self.material_transform.nu_min = params["nu_min"]
            self.material_transform.nu_max = params["nu_max"]
            self.material_transform.rho_min = params[
                "rho_min"
            ]  # Keep raw density values
            self.material_transform.rho_max = params[
                "rho_max"
            ]  # Keep raw density values
            self.material_transform._stats_computed = True

            print(f"Loaded log minmax no density normalization params:")
            print(
                f"  E_min: {self.material_transform.E_min:.6f} (log10({params['E_min']:.0f} Pa))"
            )
            print(
                f"  E_max: {self.material_transform.E_max:.6f} (log10({params['E_max']:.0f} Pa))"
            )
            print(f"  nu_min: {self.material_transform.nu_min:.6f}")
            print(f"  nu_max: {self.material_transform.nu_max:.6f}")
            print(f"  rho_min: {self.material_transform.rho_min:.2f} (kg/m³)")
            print(f"  rho_max: {self.material_transform.rho_max:.2f} (kg/m³)")
        else:
            raise ValueError(f"Unknown normalization_type: {self.normalization_type}")

    def _compute_material_stats(self):
        """Compute statistics for material property transforms (standard normalization only)."""
        if self.normalization_type != "standard":
            print(
                f"Warning: _compute_material_stats() called with normalization_type='{self.normalization_type}'. This method only works with 'standard' normalization."
            )
            return

        print("Computing material transform statistics for standard normalization...")

        # Create a simple dataloader for computing stats
        from torch.utils.data import DataLoader

        # Use a smaller batch size for stats computation to avoid memory issues
        stats_loader = DataLoader(
            self,
            batch_size=1,
            shuffle=False,
            collate_fn=self._stats_collate_fn,
            num_workers=0,  # Single-threaded for stats computation
        )

        # Compute stats using the transform class
        mu, std = self.material_transform.compute_stats(stats_loader)
        print(
            f"Material transform stats computed: μ={self.material_transform.mu}, σ={self.material_transform.std}"
        )

    @staticmethod
    def _stats_collate_fn(batch):
        """Special collate function for computing material statistics."""
        # Extract raw materials from batch
        all_materials = []
        for sample in batch:
            materials = sample["materials"]  # Shape: (num_voxels, 3)
            all_materials.append(materials)

        all_materials = torch.cat(all_materials, dim=0)  # Shape: (total_voxels, 3)

        # Convert to dictionary format expected by MaterialPropertyTransform
        material_properties = {
            "youngs_modulus": all_materials[:, 0],
            "poissons_ratio": all_materials[:, 1],
            "density": all_materials[:, 2],
        }

        return {"material_properties": material_properties}

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f"feature_{self.model}"]]
        stats["With features"] = len(metadata)

        if self.split is not None:
            if "split" in metadata.columns:
                metadata = metadata[metadata["split"] == self.split]
                stats[f"Split '{self.split}'"] = len(metadata)
            else:
                print(
                    f"Warning: 'split' column not found in metadata, ignoring split filter"
                )

        # metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
        # stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        return metadata, stats

    def _get_image(self, root, instance):
        with open(os.path.join(root, "renders", instance, "transforms.json")) as f:
            metadata = json.load(f)
        n_views = len(metadata["frames"])
        view = np.random.randint(n_views)
        metadata = metadata["frames"][view]
        fov = metadata["camera_angle_x"]
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(fov), torch.tensor(fov)
        )
        c2w = torch.tensor(metadata["transform_matrix"])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, "renders", instance, metadata["file_path"])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        alpha = alpha.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0

        return {
            "image": image,
            "alpha": alpha,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }

    def _get_feat(self, root, instance):
        DATA_RESOLUTION = 64  # Must match the resolution used during feature extraction for this dataset
        feats_path = os.path.join(root, "features", self.model, f"{instance}.npz")
        feats = np.load(feats_path, allow_pickle=True)
        coords = torch.tensor(feats["indices"]).int()
        feats = torch.tensor(feats["patchtokens"]).float()

        materials_path = os.path.join(root, "voxels", f"{instance}_with_materials.npz")
        materials_data = np.load(materials_path, allow_pickle=True)
        voxel_data = materials_data["voxel_data"]

        # Extract material properties: [youngs_modulus, poissons_ratio, density]
        materials = np.column_stack(
            [
                voxel_data["youngs_modulus"],
                voxel_data["poissons_ratio"],
                voxel_data["density"],
            ]
        )
        materials = torch.tensor(materials).float()  # Shape: (voxels, 3)

        if self.resolution != DATA_RESOLUTION:
            factor = DATA_RESOLUTION // self.resolution
            original_coords = coords.clone()
            original_materials = materials.clone()

            coords = coords // factor
            coords, idx = coords.unique(return_inverse=True, dim=0)
            feats = torch.scatter_reduce(
                torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce="mean",
            )

            # Nearest neighbor interpolation for materials
            # For each downsampled voxel, find the nearest original voxel
            downsampled_centers = (
                coords.float() * factor + factor / 2
            )  # Centers of downsampled voxels
            original_coords_float = original_coords.float()

            # Find nearest neighbor for each downsampled voxel
            nearest_materials = []
            for center in downsampled_centers:
                # Calculate distances to all original voxels
                distances = torch.norm(original_coords_float - center, dim=1)
                nearest_idx = torch.argmin(distances)
                nearest_materials.append(original_materials[nearest_idx])

            materials = torch.stack(nearest_materials)

        # Per-object voxel sampling: limit each object to max_num_voxels
        num_voxels = coords.shape[0]
        if num_voxels > self.max_num_voxels:
            # Randomly sample max_num_voxels from this object
            indices = torch.randperm(num_voxels)[: self.max_num_voxels]
            coords = coords[indices]
            feats = feats[indices]
            materials = materials[indices]

        return {
            "coords": coords,
            "feats": feats,
            "materials": materials,
        }

    def collate_fn(self, batch):
        pack = {}
        coords = []
        for i, b in enumerate(batch):
            coords.append(
                torch.cat(
                    [
                        torch.full((b["coords"].shape[0], 1), i, dtype=torch.int32),
                        b["coords"],
                    ],
                    dim=-1,
                )
            )
        coords = torch.cat(coords)
        feats = torch.cat([b["feats"] for b in batch])
        raw_materials = torch.cat([b["materials"] for b in batch])

        # Note: Per-object voxel sampling is now handled in _get_feat()
        # No cross-batch sampling needed since each object is already limited to max_num_voxels

        pack["feats"] = SparseTensor(
            coords=coords,
            feats=feats,
        )

        # Apply material transforms if stats have been computed
        if self.material_transform._stats_computed:
            # Convert to dictionary format
            material_dict = {
                "youngs_modulus": raw_materials[:, 0],
                "poissons_ratio": raw_materials[:, 1],
                "density": raw_materials[:, 2],
            }

            # Apply forward transform and standardization
            transformed_dict = (
                self.material_transform.forward_transform_and_standardize(material_dict)
            )

            # Convert back to tensor format
            transformed_materials = torch.stack(
                [
                    transformed_dict["youngs_modulus"],
                    transformed_dict["poissons_ratio"],
                    transformed_dict["density"],
                ],
                dim=-1,
            )

            pack["materials"] = SparseTensor(
                coords=coords,
                feats=transformed_materials,
            )
        else:
            # If stats not computed, just store raw materials
            pack["materials"] = SparseTensor(
                coords=coords,
                feats=raw_materials,
            )

        # pack["image"] = torch.stack([b["image"] for b in batch])
        # pack["alpha"] = torch.stack([b["alpha"] for b in batch])
        # pack["extrinsics"] = torch.stack([b["extrinsics"] for b in batch])
        # pack["intrinsics"] = torch.stack([b["intrinsics"] for b in batch])

        return pack

    def get_instance(self, root, instance):
        # image = self._get_image(root, instance)
        feat = self._get_feat(root, instance)
        return {
            # **image,
            **feat,
        }
