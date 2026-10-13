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

"""Material transforms shared by MatVAE training and visualization scripts."""

from __future__ import annotations

import torch


def forward_components(
    E, nu, rho, nu_min, nu_max, normalization_type="standard",
    E_min=None, E_max=None, rho_min=None, rho_max=None,
):
    """Shared arithmetic; logarithmic modes receive bounds already in log space."""
    yE = torch.log10(torch.clamp_min(E, 1e-8))
    if normalization_type == "standard":
        p = (nu - nu_min) / (nu_max - nu_min)
        y_nu = torch.logit(p.clamp(1e-4, 1.0 - 1e-4))
    else:
        yE = (yE - E_min) / (E_max - E_min)
        y_nu = (nu - nu_min) / (nu_max - nu_min)
    if normalization_type == "log_minmax_no_density":
        y_rho = rho
    else:
        y_rho = torch.log10(torch.clamp_min(rho, 1e-8))
    if normalization_type != "standard":
        y_rho = (y_rho - rho_min) / (rho_max - rho_min)
    return yE, y_nu, y_rho


def inverse_components(
    yE, y_nu, y_rho, nu_min, nu_max, normalization_type="standard",
    E_min=None, E_max=None, rho_min=None, rho_max=None,
):
    """Shared inverse arithmetic with precomputed, caller-precision bounds."""
    if normalization_type == "standard":
        nu = torch.sigmoid(y_nu) * (nu_max - nu_min) + nu_min
    else:
        yE = yE * (E_max - E_min) + E_min
        nu = y_nu * (nu_max - nu_min) + nu_min
        y_rho = y_rho * (rho_max - rho_min) + rho_min
    E = yE if normalization_type == "minmax" else torch.pow(10, yE)
    rho = (
        y_rho if normalization_type in {"minmax", "log_minmax_no_density"}
        else torch.pow(10, y_rho)
    )
    return E, nu, rho


def normalize_log_minmax(properties, bounds):
    """MatVAE training path: retain tensor-dtype/device logarithms of raw bounds."""
    E, nu, rho = properties.unbind(-1)
    E_min, E_max = E.new_tensor(bounds["E_min"]), E.new_tensor(bounds["E_max"])
    rho_min = rho.new_tensor(bounds["rho_min"])
    rho_max = rho.new_tensor(bounds["rho_max"])
    return torch.stack(forward_components(
        E, nu, rho, bounds["nu_min"], bounds["nu_max"], "log_minmax",
        E_min.log10(), E_max.log10(), rho_min.log10(), rho_max.log10(),
    ), dim=-1)


def physics_aware_transform_inference(
    E, nu, rho, normalization_type, E_min, E_max, nu_min, nu_max, rho_min, rho_max
):
    """Legacy MatVAE evaluation path: retain default-tensor bound rounding/clipping."""
    if normalization_type not in {"log_minmax", "log_minmax_no_density"}:
        raise ValueError(f"Unknown normalization_type: {normalization_type}")
    E_min, E_max = torch.log10(torch.tensor(E_min)), torch.log10(torch.tensor(E_max))
    if normalization_type == "log_minmax":
        rho_min, rho_max = torch.log10(torch.tensor(rho_min)), torch.log10(torch.tensor(rho_max))
    # These evaluation scripts also clamp density in the no-density-log mode;
    # MaterialPropertyTransform deliberately retains its original unclamped path.
    rho = torch.clamp_min(rho, 1e-8)
    return torch.stack(forward_components(
        E, nu, rho, nu_min, nu_max, normalization_type,
        E_min, E_max, rho_min, rho_max,
    ), -1)


def forward_transform(
    E: torch.Tensor, nu: torch.Tensor, rho: torch.Tensor, nu_min: float, nu_max: float
) -> torch.Tensor:
    """Apply the original log/logit material transform."""
    return torch.stack(forward_components(E, nu, rho, nu_min, nu_max), -1)


def inverse_transform(
    y_norm: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    nu_min: float,
    nu_max: float,
) -> torch.Tensor:
    """Undo standardization and the original log/logit transform."""
    y = y_norm * std + mu
    return torch.stack(inverse_components(*y.unbind(-1), nu_min, nu_max), -1)


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
    """Restore physical units using the normalization saved with the model."""
    if normalization_type == "physics_bounds":
        E_min, E_max = 6.0, 12.0
        nu_min, nu_max = -1.0, 0.5
        rho_min, rho_max = -1.0, 4.3
        normalization_type = "log_minmax"
    elif normalization_type == "log_minmax":
        E_min, E_max = torch.log10(torch.tensor(E_min)), torch.log10(
            torch.tensor(E_max)
        )
        rho_min, rho_max = torch.log10(torch.tensor(rho_min)), torch.log10(
            torch.tensor(rho_max)
        )
    elif normalization_type == "log_minmax_no_density":
        E_min, E_max = torch.log10(torch.tensor(E_min)), torch.log10(
            torch.tensor(E_max)
        )
    elif normalization_type != "minmax":
        if mu is None or std is None:
            raise ValueError("mu and std are required for standard normalization")
        return inverse_transform(y_norm, mu, std, nu_min, nu_max)

    return torch.stack(inverse_components(
        *y_norm.unbind(-1), nu_min, nu_max, normalization_type,
        E_min, E_max, rho_min, rho_max,
    ), -1)
