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

"""Material property transforms for normalization and denormalization."""

import torch
from typing import Dict, Tuple, Optional


class MaterialPropertyTransform:
    """Handles forward and inverse transforms for material properties.

    This class provides consistent normalization for material properties:
    - Standard mode: log transform + standardization
    - Log minmax mode: log transform + min-max normalization
    """

    def __init__(
        self,
        nu_min: float = 0.0,
        nu_max: float = 0.5,
        normalization_type: str = "standard",
    ):
        """Initialize transform with Poisson's ratio bounds.

        Args:
            nu_min: Minimum Poisson's ratio (default: 0.0)
            nu_max: Maximum Poisson's ratio (default: 0.5)
            normalization_type: Type of normalization ("standard" or "log_minmax")
        """
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.normalization_type = normalization_type
        self._stats_computed = False

        # For standard normalization
        self.mu = None
        self.std = None

        # For log minmax normalization
        self.E_min = None
        self.E_max = None
        self.rho_min = None
        self.rho_max = None

    def forward_transform(
        self, properties: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply forward transform to material properties.

        Args:
            properties: Dictionary with keys 'youngs_modulus', 'poissons_ratio', 'density'

        Returns:
            Transformed properties dictionary
        """
        if self.normalization_type == "log_minmax":
            return self._log_minmax_transform(properties)
        elif self.normalization_type == "log_minmax_no_density":
            return self._log_minmax_no_density_transform(properties)
        else:
            return self._standard_transform(properties)

    def _standard_transform(
        self, properties: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply standard log/logit transform."""
        transformed = {}

        # Young's modulus - log transform
        E = properties["youngs_modulus"]
        E_clamped = torch.clamp_min(E, 1e-8)
        transformed["youngs_modulus"] = torch.log10(E_clamped)

        # Poisson's ratio - logit transform with bounds
        nu = properties["poissons_ratio"]
        # Normalize to [0, 1] range based on physical bounds
        p = (nu - self.nu_min) / (self.nu_max - self.nu_min)
        p_clamped = torch.clamp(p, 1e-4, 1.0 - 1e-4)
        transformed["poissons_ratio"] = torch.logit(p_clamped)

        # Density - log transform
        rho = properties["density"]
        rho_clamped = torch.clamp_min(rho, 1e-8)
        transformed["density"] = torch.log10(rho_clamped)

        return transformed

    def _log_minmax_transform(
        self, properties: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply log minmax transform."""
        if not self._stats_computed:
            raise ValueError("Must compute stats before log minmax transform")

        transformed = {}

        # Young's modulus - log transform then min-max normalize
        E = properties["youngs_modulus"]
        E_clamped = torch.clamp_min(E, 1e-8)
        log_E = torch.log10(E_clamped)
        transformed["youngs_modulus"] = (log_E - self.E_min) / (self.E_max - self.E_min)

        # Poisson's ratio - min-max normalize directly
        nu = properties["poissons_ratio"]
        transformed["poissons_ratio"] = (nu - self.nu_min) / (self.nu_max - self.nu_min)

        # Density - log transform then min-max normalize
        rho = properties["density"]
        rho_clamped = torch.clamp_min(rho, 1e-8)
        log_rho = torch.log10(rho_clamped)
        transformed["density"] = (log_rho - self.rho_min) / (
            self.rho_max - self.rho_min
        )

        return transformed

    def _log_minmax_no_density_transform(
        self, properties: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply log minmax transform without log for density."""
        if not self._stats_computed:
            raise ValueError(
                "Must compute stats before log minmax no density transform"
            )

        transformed = {}

        # Young's modulus - log transform then min-max normalize
        E = properties["youngs_modulus"]
        E_clamped = torch.clamp_min(E, 1e-8)
        log_E = torch.log10(E_clamped)
        transformed["youngs_modulus"] = (log_E - self.E_min) / (self.E_max - self.E_min)

        # Poisson's ratio - min-max normalize directly
        nu = properties["poissons_ratio"]
        transformed["poissons_ratio"] = (nu - self.nu_min) / (self.nu_max - self.nu_min)

        # Density - min-max normalize WITHOUT log transform
        rho = properties["density"]
        transformed["density"] = (rho - self.rho_min) / (self.rho_max - self.rho_min)

        return transformed

    def forward_transform_and_standardize(
        self, properties: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply forward transform and standardization in one step.

        Args:
            properties: Dictionary with material properties

        Returns:
            Transformed and standardized properties
        """
        if not self._stats_computed:
            raise ValueError("Must compute stats before standardization")

        if (
            self.normalization_type == "log_minmax"
            or self.normalization_type == "log_minmax_no_density"
        ):
            # For log minmax variants, forward transform already includes normalization
            return self.forward_transform(properties)
        else:
            # Standard approach: forward transform + standardization
            # First apply forward transform
            transformed = self.forward_transform(properties)

            # Then standardize each property individually
            standardized = {}
            standardized["youngs_modulus"] = (
                transformed["youngs_modulus"] - self.mu[0]
            ) / self.std[0]
            standardized["poissons_ratio"] = (
                transformed["poissons_ratio"] - self.mu[1]
            ) / self.std[1]
            standardized["density"] = (transformed["density"] - self.mu[2]) / self.std[
                2
            ]

            return standardized

    def inverse_transform(
        self, transformed: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply inverse transform to get back original scale.

        Args:
            transformed: Dictionary with transformed properties

        Returns:
            Properties in original scale
        """
        if self.normalization_type == "log_minmax":
            return self._log_minmax_inverse_transform(transformed)
        elif self.normalization_type == "log_minmax_no_density":
            return self._log_minmax_no_density_inverse_transform(transformed)
        else:
            return self._standard_inverse_transform(transformed)

    def _standard_inverse_transform(
        self, transformed: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply standard inverse transform."""
        original = {}

        # Young's modulus - exp transform
        original["youngs_modulus"] = torch.pow(10, transformed["youngs_modulus"])

        # Poisson's ratio - inverse logit and rescale
        p = torch.sigmoid(transformed["poissons_ratio"])
        original["poissons_ratio"] = p * (self.nu_max - self.nu_min) + self.nu_min

        # Density - exp transform
        original["density"] = torch.pow(10, transformed["density"])

        return original

    def _log_minmax_inverse_transform(
        self, transformed: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply log minmax inverse transform."""
        if not self._stats_computed:
            raise ValueError("Must compute stats before inverse transform")

        original = {}

        # Young's modulus - inverse min-max then exp transform
        log_E = transformed["youngs_modulus"] * (self.E_max - self.E_min) + self.E_min
        original["youngs_modulus"] = torch.pow(10, log_E)

        # Poisson's ratio - inverse min-max
        original["poissons_ratio"] = (
            transformed["poissons_ratio"] * (self.nu_max - self.nu_min) + self.nu_min
        )

        # Density - inverse min-max then exp transform
        log_rho = transformed["density"] * (self.rho_max - self.rho_min) + self.rho_min
        original["density"] = torch.pow(10, log_rho)

        return original

    def _log_minmax_no_density_inverse_transform(
        self, transformed: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply log minmax inverse transform without log for density."""
        if not self._stats_computed:
            raise ValueError("Must compute stats before inverse transform")

        original = {}

        # Young's modulus - inverse min-max then exp transform
        log_E = transformed["youngs_modulus"] * (self.E_max - self.E_min) + self.E_min
        original["youngs_modulus"] = torch.pow(10, log_E)

        # Poisson's ratio - inverse min-max
        original["poissons_ratio"] = (
            transformed["poissons_ratio"] * (self.nu_max - self.nu_min) + self.nu_min
        )

        # Density - inverse min-max WITHOUT exp transform
        original["density"] = (
            transformed["density"] * (self.rho_max - self.rho_min) + self.rho_min
        )

        return original

    def compute_stats(self, dataset_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute statistics for normalization.

        Args:
            dataset_loader: DataLoader to compute statistics from

        Returns:
            mean and std tensors (for standard mode) or min/max tensors (for log_minmax mode)
        """
        if self.normalization_type == "log_minmax":
            return self._compute_log_minmax_stats(dataset_loader)
        elif self.normalization_type == "log_minmax_no_density":
            return self._compute_log_minmax_no_density_stats(dataset_loader)
        else:
            return self._compute_standard_stats(dataset_loader)

    def _compute_standard_stats(
        self, dataset_loader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std for standardization on transformed values."""
        sum_ = torch.zeros(3)
        sum2 = torch.zeros(3)
        n = 0

        for batch in dataset_loader:
            if "material_properties" in batch:
                props = batch["material_properties"]
                # Apply forward transform
                transformed = self._standard_transform(props)

                # Stack into tensor [E, nu, rho]
                stacked = torch.stack(
                    [
                        transformed["youngs_modulus"].flatten(),
                        transformed["poissons_ratio"].flatten(),
                        transformed["density"].flatten(),
                    ],
                    dim=-1,
                )

                sum_ += stacked.sum(0)
                sum2 += (stacked**2).sum(0)
                n += stacked.shape[0]

        self.mu = sum_ / n
        self.std = torch.sqrt(sum2 / n - self.mu**2)
        self._stats_computed = True

        return self.mu, self.std

    def _compute_log_minmax_stats(
        self, dataset_loader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute min/max values for log minmax normalization."""
        log_E_values = []
        nu_values = []
        log_rho_values = []

        for batch in dataset_loader:
            if "material_properties" in batch:
                props = batch["material_properties"]

                # Young's modulus - log transform
                E = props["youngs_modulus"]
                E_clamped = torch.clamp_min(E, 1e-8)
                log_E_values.append(torch.log10(E_clamped).flatten())

                # Poisson's ratio - use directly
                nu_values.append(props["poissons_ratio"].flatten())

                # Density - log transform
                rho = props["density"]
                rho_clamped = torch.clamp_min(rho, 1e-8)
                log_rho_values.append(torch.log10(rho_clamped).flatten())

        # Concatenate all values
        all_log_E = torch.cat(log_E_values)
        all_nu = torch.cat(nu_values)
        all_log_rho = torch.cat(log_rho_values)

        # Compute min/max
        self.E_min = all_log_E.min().item()
        self.E_max = all_log_E.max().item()
        self.nu_min = all_nu.min().item()
        self.nu_max = all_nu.max().item()
        self.rho_min = all_log_rho.min().item()
        self.rho_max = all_log_rho.max().item()

        self._stats_computed = True

        # Return min/max as tensors for compatibility
        min_vals = torch.tensor([self.E_min, self.nu_min, self.rho_min])
        max_vals = torch.tensor([self.E_max, self.nu_max, self.rho_max])

        return min_vals, max_vals

    def _compute_log_minmax_no_density_stats(
        self, dataset_loader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute min/max values for log minmax no density normalization."""
        log_E_values = []
        nu_values = []
        rho_values = []

        for batch in dataset_loader:
            if "material_properties" in batch:
                props = batch["material_properties"]

                # Young's modulus - log transform
                E = props["youngs_modulus"]
                E_clamped = torch.clamp_min(E, 1e-8)
                log_E_values.append(torch.log10(E_clamped).flatten())

                # Poisson's ratio - use directly
                nu_values.append(props["poissons_ratio"].flatten())

                # Density - use directly WITHOUT log transform
                rho_values.append(props["density"].flatten())

        # Concatenate all values
        all_log_E = torch.cat(log_E_values)
        all_nu = torch.cat(nu_values)
        all_rho = torch.cat(rho_values)

        # Compute min/max
        self.E_min = all_log_E.min().item()
        self.E_max = all_log_E.max().item()
        self.nu_min = all_nu.min().item()
        self.nu_max = all_nu.max().item()
        self.rho_min = all_rho.min().item()
        self.rho_max = all_rho.max().item()

        self._stats_computed = True

        # Return min/max as tensors for compatibility
        min_vals = torch.tensor([self.E_min, self.nu_min, self.rho_min])
        max_vals = torch.tensor([self.E_max, self.nu_max, self.rho_max])

        return min_vals, max_vals

    def standardize(
        self, properties: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply forward transform and standardization.

        Args:
            properties: Dictionary with material properties

        Returns:
            Standardized properties
        """
        return self.forward_transform_and_standardize(properties)

    def destandardize(
        self, standardized: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remove standardization and apply inverse transform.

        Args:
            standardized: Dictionary with standardized properties

        Returns:
            Properties in original scale
        """
        return self.destandardize_and_inverse_transform(standardized)

    def destandardize_and_inverse_transform(
        self, standardized: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remove standardization and apply inverse transform in one step.

        Args:
            standardized: Dictionary with standardized properties

        Returns:
            Properties in original scale
        """
        if not self._stats_computed:
            raise ValueError("Must compute stats before destandardization")

        if (
            self.normalization_type == "log_minmax"
            or self.normalization_type == "log_minmax_no_density"
        ):
            # For log minmax variants, no standardization step - direct inverse transform
            return self.inverse_transform(standardized)
        else:
            # Standard approach: destandardize then inverse transform
            # First remove standardization
            transformed = {}
            transformed["youngs_modulus"] = (
                standardized["youngs_modulus"] * self.std[0] + self.mu[0]
            )
            transformed["poissons_ratio"] = (
                standardized["poissons_ratio"] * self.std[1] + self.mu[1]
            )
            transformed["density"] = standardized["density"] * self.std[2] + self.mu[2]

            # Then apply inverse transform
            return self.inverse_transform(transformed)

    def state_dict(self):
        """Get state dict for saving."""
        state = {
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "normalization_type": self.normalization_type,
            "mu": self.mu,
            "std": self.std,
            "_stats_computed": self._stats_computed,
        }

        if (
            self.normalization_type == "log_minmax"
            or self.normalization_type == "log_minmax_no_density"
        ):
            state.update(
                {
                    "E_min": self.E_min,
                    "E_max": self.E_max,
                    "rho_min": self.rho_min,
                    "rho_max": self.rho_max,
                }
            )

        return state

    def load_state_dict(self, state_dict):
        """Load from state dict."""
        self.nu_min = state_dict["nu_min"]
        self.nu_max = state_dict["nu_max"]
        self.normalization_type = state_dict.get("normalization_type", "standard")
        self.mu = state_dict.get("mu")
        self.std = state_dict.get("std")
        self._stats_computed = state_dict["_stats_computed"]

        if (
            self.normalization_type == "log_minmax"
            or self.normalization_type == "log_minmax_no_density"
        ):
            self.E_min = state_dict.get("E_min")
            self.E_max = state_dict.get("E_max")
            self.rho_min = state_dict.get("rho_min")
            self.rho_max = state_dict.get("rho_max")

    def forward_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward transform to a tensor of shape (..., 3).

        Args:
            tensor: Tensor with last dimension [E, nu, rho]

        Returns:
            Transformed tensor
        """
        shape = tensor.shape
        tensor_flat = tensor.view(-1, 3)

        # Create dict from tensor
        props = {
            "youngs_modulus": tensor_flat[:, 0],
            "poissons_ratio": tensor_flat[:, 1],
            "density": tensor_flat[:, 2],
        }

        # Transform
        transformed = self.forward_transform(props)

        # Convert back to tensor
        result = torch.stack(
            [
                transformed["youngs_modulus"],
                transformed["poissons_ratio"],
                transformed["density"],
            ],
            dim=-1,
        )

        return result.view(shape)

    def standardize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Standardize a tensor that's already been forward transformed.

        Args:
            tensor: Tensor of shape (..., 3) with transformed values

        Returns:
            Standardized tensor
        """
        if not self._stats_computed:
            raise ValueError("Must compute stats before standardization")

        if (
            self.normalization_type == "log_minmax"
            or self.normalization_type == "log_minmax_no_density"
        ):
            # For log minmax variants, already normalized in forward transform
            return tensor
        else:
            # Standard approach: standardize using mu and std
            # Ensure mu and std are on the same device as tensor
            mu = self.mu.to(tensor.device)
            std = self.std.to(tensor.device)

            # Standardize each channel
            standardized = (tensor - mu) / std
            return standardized

    def forward_transform_and_standardize_tensor(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Apply forward transform and standardization to a tensor.

        Args:
            tensor: Tensor with last dimension [E, nu, rho] in original scale

        Returns:
            Transformed and standardized tensor
        """
        transformed = self.forward_transform_tensor(tensor)
        return self.standardize_tensor(transformed)

    def destandardize_and_inverse_transform_tensor(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Remove standardization and apply inverse transform to a tensor.

        Args:
            tensor: Standardized tensor of shape (..., 3)

        Returns:
            Tensor in original scale
        """
        if not self._stats_computed:
            raise ValueError("Must compute stats before destandardization")

        if (
            self.normalization_type == "log_minmax"
            or self.normalization_type == "log_minmax_no_density"
        ):
            # For log minmax variants, no standardization step - direct inverse transform
            shape = tensor.shape
            tensor_flat = tensor.view(-1, 3)

            props = {
                "youngs_modulus": tensor_flat[:, 0],
                "poissons_ratio": tensor_flat[:, 1],
                "density": tensor_flat[:, 2],
            }

            original = self.inverse_transform(props)

            result = torch.stack(
                [
                    original["youngs_modulus"],
                    original["poissons_ratio"],
                    original["density"],
                ],
                dim=-1,
            )

            return result.view(shape)
        else:
            # Standard approach: destandardize then inverse transform
            # Ensure mu and std are on the same device as tensor
            mu = self.mu.to(tensor.device)
            std = self.std.to(tensor.device)

            # Destandardize
            transformed = tensor * std + mu

            # Apply inverse transform
            shape = transformed.shape
            transformed_flat = transformed.view(-1, 3)

            props = {
                "youngs_modulus": transformed_flat[:, 0],
                "poissons_ratio": transformed_flat[:, 1],
                "density": transformed_flat[:, 2],
            }

            original = self.inverse_transform(props)

            result = torch.stack(
                [
                    original["youngs_modulus"],
                    original["poissons_ratio"],
                    original["density"],
                ],
                dim=-1,
            )

            return result.view(shape)
