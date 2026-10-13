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
import hashlib
from itertools import permutations, product
from PIL import Image
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ..modules.sparse.basic import SparseTensor
from ..utils.material_transforms import MaterialPropertyTransform
from .components import StandardDatasetBase
import utils3d.torch


def _cube_grid_rotations():
    rotations = []
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int8)
            matrix[np.arange(3), axes] = signs
            if round(np.linalg.det(matrix)) == 1:
                rotations.append((axes, signs))
    return tuple(rotations)


_CUBE_GRID_ROTATIONS = _cube_grid_rotations()


def _random_grid_rotation(coords, resolution):
    axes, signs = _CUBE_GRID_ROTATIONS[
        torch.randint(len(_CUBE_GRID_ROTATIONS), ()).item()
    ]
    rotated = coords[:, list(axes)].clone()
    for axis, sign in enumerate(signs):
        if sign < 0:
            rotated[:, axis] = resolution - 1 - rotated[:, axis]
    return rotated


def _segment_balanced_random_rows(segment_ids, budget, generator=None):
    unique_segments, labels, counts = np.unique(
        np.asarray(segment_ids).astype(str), return_inverse=True, return_counts=True
    )
    if len(unique_segments) > budget:
        raise ValueError(
            f"cannot represent {len(unique_segments)} segments with {budget} voxels"
        )
    base = max(1, min(64, budget // len(unique_segments)))
    selected = torch.zeros(len(labels), dtype=torch.bool)
    for label, count in enumerate(counts):
        rows = torch.from_numpy(np.flatnonzero(labels == label))
        take = min(int(count), base)
        selected[rows[torch.randperm(len(rows), generator=generator)[:take]]] = True
    remaining = torch.nonzero(~selected, as_tuple=True)[0]
    take = budget - int(selected.sum())
    selected[
        remaining[torch.randperm(len(remaining), generator=generator)[:take]]
    ] = True
    return torch.nonzero(selected, as_tuple=True)[0][
        torch.randperm(budget, generator=generator)
    ]


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
        feature_noise_std (float): Gaussian noise scale as a fraction of each
            object's DINO feature standard deviation
        archive_format (str): "benchmark" for row-aligned material archives or
            "mapped" for archives with explicit feature-to-voxel row indices
        segment_balanced_sampling (bool): retain small material segments when
            subsampling; defaults to enabled for mapped archives
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
        random_grid_rotation: bool = False,
        feature_noise_std: float = 0.0,
        archive_format: str = "benchmark",
        segment_balanced_sampling: bool = None,
    ):
        if archive_format not in {"benchmark", "mapped"}:
            raise ValueError("archive_format must be 'benchmark' or 'mapped'")
        self.archive_format = archive_format
        self.segment_balanced_sampling = (
            archive_format == "mapped"
            if segment_balanced_sampling is None
            else segment_balanced_sampling
        )
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        if resolution <= 0 or 64 % resolution != 0:
            raise ValueError("resolution must be a positive divisor of 64")
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        if max_num_voxels <= 0:
            raise ValueError("max_num_voxels must be positive")
        self.split = split
        self.value_range = (0, 1)
        self.normalization_type = normalization_type
        self.random_grid_rotation = random_grid_rotation
        if not np.isfinite(feature_noise_std) or feature_noise_std < 0:
            raise ValueError("feature_noise_std must be finite and non-negative")
        self.feature_noise_std = float(feature_noise_std)

        self.material_transform = MaterialPropertyTransform(
            normalization_type=normalization_type
        )

        # Load normalization parameters from file (optional for direct prediction)
        if normalization_params_file is not None:
            if not os.path.exists(normalization_params_file):
                raise FileNotFoundError(f"Normalization parameters file not found: {normalization_params_file}")
            print(f"Loading normalization parameters from: {normalization_params_file}")
            self._load_normalization_params(normalization_params_file)
        else:
            print("No normalization parameters file provided - using dataset-derived normalization for direct prediction")

        super().__init__(roots)

        if compute_material_stats and normalization_type == "standard":
            print(
                "Recomputing material transform statistics for standard normalization..."
            )
            self._compute_material_stats()

    def _load_normalization_params(self, params_file: str):
        """Load normalization parameters from JSON file (saved during matvae training)."""
        with open(params_file, 'r') as f:
            params = json.load(f)

        # Validate normalization type consistency
        if params.get("normalization_type") != self.normalization_type:
            raise ValueError(
                f"Normalization type mismatch! "
                f"Dataset config: {self.normalization_type}, "
                f"Params file: {params.get('normalization_type')}"
            )

        self.material_transform = MaterialPropertyTransform.from_params(
            params, self.normalization_type
        )
        print(f"Loaded {self.normalization_type} normalization params from {params_file}")

    def _compute_material_stats(self):
        """Compute statistics for material property transforms (standard normalization only)."""
        if self.normalization_type != "standard":
            print(
                f"Warning: _compute_material_stats() called with normalization_type='{self.normalization_type}'. This method only works with 'standard' normalization."
            )
            return

        print("Computing material transform statistics for standard normalization...")

        # Create a simple dataloader for computing stats

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
        with np.load(feats_path, allow_pickle=False) as feature_archive:
            indices = feature_archive["indices"]
            features = feature_archive["patchtokens"]
            voxel_rows = (
                feature_archive["voxel_row_indices"]
                if self.archive_format == "mapped"
                else None
            )
            if self.archive_format == "benchmark" and "voxel_row_indices" in feature_archive:
                raise ValueError(
                    f"Mapped features require archive_format='mapped': {feats_path}"
                )

        if (
            indices.ndim != 2
            or indices.shape[1] != 3
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= DATA_RESOLUTION)
            or features.ndim != 2
            or features.shape[0] != len(indices)
            or not len(indices)
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"Invalid feature arrays in {feats_path}")
        coords = torch.tensor(indices).int()
        feats = torch.tensor(features).float()

        materials_path = os.path.join(root, "voxels", f"{instance}_with_materials.npz")
        with np.load(
            materials_path, allow_pickle=self.archive_format == "benchmark"
        ) as materials_data:
            voxel_data = materials_data["voxel_data"]
            grid_resolution = (
                np.asarray(materials_data["grid_resolution"])
                if self.archive_format == "mapped"
                else None
            )

        if self.archive_format == "mapped":
            if (
                grid_resolution.shape != ()
                or grid_resolution.item() != DATA_RESOLUTION
                or voxel_rows.shape != (len(coords),)
                or not np.issubdtype(voxel_rows.dtype, np.integer)
                or len(np.unique(voxel_rows)) != len(voxel_rows)
                or np.any(voxel_rows < 0)
                or np.any(voxel_rows >= len(voxel_data))
            ):
                raise ValueError(f"Invalid feature-to-voxel mapping in {feats_path}")
            voxel_data = voxel_data[voxel_rows.astype(np.int64)]
            material_positions = np.column_stack(
                (voxel_data["x"], voxel_data["y"], voxel_data["z"])
            )
            material_coords = np.floor(
                (material_positions.astype(np.float64) + 0.5) * DATA_RESOLUTION
            ).astype(np.int64)
            expected_positions = (
                (material_coords.astype(np.float64) + 0.5) / DATA_RESOLUTION - 0.5
            ).astype(np.float32)
            if (
                np.any(material_coords < 0)
                or np.any(material_coords >= DATA_RESOLUTION)
                or not np.array_equal(material_positions, expected_positions)
                or not np.array_equal(indices, material_coords)
                or len(np.unique(indices, axis=0)) != len(indices)
            ):
                raise ValueError(
                    f"Feature and material rows are not exact voxel centers for {instance}"
                )
            segment_ids = voxel_data["segment_id"].astype(str)
        else:
            if len(voxel_data) != len(coords):
                raise ValueError(f"Benchmark feature and material rows do not align for {instance}")
            segment_ids = (
                voxel_data["segment_id"].astype(str)
                if "segment_id" in (voxel_data.dtype.names or ())
                else np.full(len(voxel_data), "object")
            )

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
                include_self=False,
            )

            # Nearest neighbor interpolation for materials
            # For each downsampled voxel, find the nearest original voxel
            downsampled_centers = (
                coords.float() * factor + (factor - 1) / 2
            )  # Centers of downsampled voxels
            original_coords_float = original_coords.float()

            # Find nearest neighbor for each downsampled voxel
            nearest_materials = []
            nearest_segments = []
            for center in downsampled_centers:
                # Calculate distances to all original voxels
                distances = torch.norm(original_coords_float - center, dim=1)
                nearest_idx = torch.argmin(distances)
                nearest_materials.append(original_materials[nearest_idx])
                nearest_segments.append(segment_ids[int(nearest_idx)])

            materials = torch.stack(nearest_materials)
            segment_ids = np.asarray(nearest_segments)

        # Per-object voxel sampling: limit each object to max_num_voxels
        num_voxels = coords.shape[0]
        evaluation_split = self.split in {"val", "test", "rotated_val", "rotated_test"}
        if num_voxels > self.max_num_voxels:
            generator = None
            if evaluation_split:
                seed_bytes = hashlib.sha256(
                    f"{root}\0{instance}".encode("utf-8")
                ).digest()[:8]
                generator = torch.Generator().manual_seed(
                    int.from_bytes(seed_bytes, "little") % (2**63 - 1)
                )
            indices = (
                _segment_balanced_random_rows(segment_ids, self.max_num_voxels, generator)
                if self.segment_balanced_sampling
                else torch.randperm(num_voxels, generator=generator)[:self.max_num_voxels]
            )
            coords = coords[indices]
            feats = feats[indices]
            materials = materials[indices]
            segment_ids = segment_ids[indices.numpy()]

        if self.feature_noise_std > 0 and not evaluation_split:
            noise_std = self.feature_noise_std * feats.std(unbiased=False)
            feats.add_(torch.randn_like(feats), alpha=float(noise_std))

        if self.random_grid_rotation and not evaluation_split:
            coords = _random_grid_rotation(coords, self.resolution)

        _, segment_ids = np.unique(segment_ids, return_inverse=True)

        return {
            "coords": coords,
            "feats": feats,
            "materials": materials,
            "segment_ids": torch.from_numpy(segment_ids).long(),
        }

    def collate_fn(self, batch):
        pack = {}
        coords = []
        segment_ids = []
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
            segment_ids.append(b["segment_ids"])
        coords = torch.cat(coords)
        feats = torch.cat([b["feats"] for b in batch])
        raw_materials = torch.cat([b["materials"] for b in batch])
        segment_ids = torch.cat(segment_ids).float().unsqueeze(-1)

        pack["feats"] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        pack["segment_ids"] = SparseTensor(
            coords=coords,
            feats=segment_ids,
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

    def __getitem__(self, index):
        root, instance = self.instances[index]
        return self.get_instance(root, instance)

    def get_instance(self, root, instance):
        # image = self._get_image(root, instance)
        feat = self._get_feat(root, instance)
        return {
            # **image,
            **feat,
        }
