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

import numpy as np
import warp as wp
from vomp.inference.utils import MaterialUpsampler


def load_material_data(npz_path: str):
    """
    Load material data from npz file.

    Args:
        npz_path: Path to .npz file containing voxel_data

    Returns:
        tuple: (voxel_coords, voxel_materials) where:
            - voxel_coords: (N, 3) array of voxel positions
            - voxel_materials: (N, 3) array of [E, nu, rho] per voxel
    """
    data = np.load(npz_path)
    voxel_data = data["voxel_data"]

    voxel_coords = np.stack([voxel_data["x"], voxel_data["y"], voxel_data["z"]], axis=1)

    voxel_materials = np.stack(
        [
            voxel_data["youngs_modulus"],
            voxel_data["poissons_ratio"],
            voxel_data["density"],
        ],
        axis=1,
    )

    print(f"Loaded material data:")
    print(f"  Voxels: {voxel_coords.shape[0]}")
    print(
        f"  Young's modulus range: [{voxel_materials[:, 0].min():.2e}, {voxel_materials[:, 0].max():.2e}]"
    )
    print(
        f"  Poisson's ratio range: [{voxel_materials[:, 1].min():.3f}, {voxel_materials[:, 1].max():.3f}]"
    )
    print(
        f"  Density range: [{voxel_materials[:, 2].min():.2f}, {voxel_materials[:, 2].max():.2f}]"
    )

    return voxel_coords, voxel_materials


def apply_spatially_varying_materials(sim, npz_path: str, k_neighbors: int = 1):
    voxel_coords, voxel_materials = load_material_data(npz_path)

    # Create upsampler
    upsampler = MaterialUpsampler(voxel_coords, voxel_materials)

    # Get vertex positions from the mesh
    node_positions = sim.lame_field.space.node_positions()
    query_points = node_positions.numpy()  # Convert warp array to numpy

    print(f"\nInterpolating materials to {query_points.shape[0]} mesh vertices...")

    interpolated_materials, distances = upsampler.interpolate(
        query_points, k=k_neighbors
    )

    youngs_modulus_per_vertex = interpolated_materials[:, 0]  # (N,) array
    poisson_ratio_per_vertex = interpolated_materials[:, 1]  # (N,) array
    density_per_vertex = interpolated_materials[:, 2]  # (N,) array

    # Convert Young's modulus and Poisson's ratio to Lame parameters
    # lame[0] = lambda = E * nu / ((1 + nu) * (1 - 2*nu))
    # lame[1] = mu = E / (2 * (1 + nu))
    lame_lambda = (youngs_modulus_per_vertex * poisson_ratio_per_vertex) / (
        (1.0 + poisson_ratio_per_vertex) * (1.0 - 2.0 * poisson_ratio_per_vertex)
    )
    lame_mu = youngs_modulus_per_vertex / (2.0 * (1.0 + poisson_ratio_per_vertex))

    # Stack into (N, 2) array for lame parameters [lambda, mu]
    lame_params = np.stack([lame_lambda, lame_mu], axis=1)

    # Directly set the lame field values (don't use scale_lame_field as it would multiply)
    # The lame_field.dof_values is a warp array of wp.vec2
    sim.lame_field.dof_values.assign(wp.array(lame_params, dtype=wp.vec2))

    return {
        "youngs_modulus": youngs_modulus_per_vertex,
        "poisson_ratio": poisson_ratio_per_vertex,
        "density": density_per_vertex,
        "interpolation_distances": distances,
    }


def visualize_material_distribution(sim, material_stats: dict, output_path: str = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Young's modulus histogram
    axes[0].hist(
        material_stats["youngs_modulus"], bins=50, alpha=0.7, edgecolor="black"
    )
    axes[0].set_xlabel("Young's Modulus (Pa)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Young's Modulus Distribution")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Poisson's ratio histogram
    axes[1].hist(
        material_stats["poisson_ratio"],
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color="orange",
    )
    axes[1].set_xlabel("Poisson's Ratio")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Poisson's Ratio Distribution")
    axes[1].grid(True, alpha=0.3)

    # Density histogram
    axes[2].hist(
        material_stats["density"], bins=50, alpha=0.7, edgecolor="black", color="green"
    )
    axes[2].set_xlabel("Density (kg/m³)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Density Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
