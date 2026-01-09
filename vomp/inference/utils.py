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

"""
Utility classes and functions for material property inference.

This module provides supporting utilities including lazy model loading
and material property upsampling capabilities.
"""

import os
from typing import Optional, Union, Tuple, Dict, Any

import torch
import numpy as np
import trimesh
from torchvision import transforms
from scipy.spatial import cKDTree


class LazyLoadDino:
    """
    Lazy-loading DINO model wrapper with shared class-level storage.

    Prevents multiple model instantiations and reduces memory overhead
    by loading models only when accessed and sharing across instances.
    """

    __model__: Optional[torch.nn.Module] = None
    __transform__: Optional[transforms.Compose] = None
    __n_patch__: Optional[int] = None
    __device__: Optional[torch.device] = None
    __use_trt__: bool = False
    __cuda_stream__: Optional[torch.cuda.Stream] = None

    def __init__(
        self,
        model_name: str = "dinov2_vitl14_reg",
        device: Union[str, torch.device, None] = None,
        use_trt: bool = False,
    ):
        """
        Initialize LazyLoadDino with model name and device.

        Args:
            model_name (str): Name of the DINO model to load
            device (Union[str, torch.device, None]): Device to load the model on.
                If None, defaults to "cuda" if available, else "cpu"
            use_trt (bool): Whether to use TensorRT acceleration for inference (significantly faster).
                Requires torch-tensorrt package to be installed.
        """
        self.model_name = model_name
        self.use_trt = use_trt
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

    @classmethod
    def _load_model_and_transform(
        cls, model_name: str, device: torch.device, use_trt: bool = False
    ) -> None:
        """Load the DINO model and setup transforms if not already loaded"""
        if cls.__model__ is None:
            print(f"Loading DINO model: {model_name} on device: {device}...")
            cls.__model__ = torch.hub.load("facebookresearch/dinov2", model_name)
            cls.__model__.eval().to(device)
            cls.__device__ = device
            cls.__use_trt__ = use_trt

            cls.__transform__ = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            cls.__n_patch__ = 518 // 14

            print(f"DINO model loaded successfully. Patch size: {cls.__n_patch__}")

            # Apply TensorRT compilation if requested
            if use_trt:
                cls._compile_with_tensorrt()

    @classmethod
    def _compile_with_tensorrt(cls) -> None:
        """Compile the loaded DINO model with TensorRT acceleration."""
        try:
            import torch_tensorrt
        except ImportError:
            print(
                "\n❌ Error: torch-tensorrt package is required for TensorRT acceleration."
            )
            print("Please install it using:")
            print(
                "pip install -U torch-tensorrt --no-deps --index-url https://download.pytorch.org/whl/cu118"
            )
            print("\nExiting...")
            exit(1)

        print("Compiling DINO model with TensorRT...")
        try:
            original_model_name = "dinov2_vitl14_reg"

            # Create a non-default CUDA stream to avoid synchronization warnings
            if cls.__device__.type == "cuda":
                cls.__cuda_stream__ = torch.cuda.Stream()

            # Create sample inputs with different batch sizes that will be used
            sample_inputs = [
                torch.randn(1, 3, 518, 518).to(cls.__device__),  # Single image
                torch.randn(4, 3, 518, 518).to(cls.__device__),
                torch.randn(8, 3, 518, 518).to(cls.__device__),
            ]

            # Compile with TensorRT
            print("Compiling with TensorRT backend...")
            cls.__model__ = torch.compile(cls.__model__, backend="tensorrt")

            print("Warming up with different batch sizes...")
            with torch.inference_mode():
                for i, sample_input in enumerate(sample_inputs):
                    print(f"  Warming up batch size {sample_input.shape[0]}...")
                    try:
                        _ = cls.__model__(sample_input)
                        print(
                            f"  ✓ Batch size {sample_input.shape[0]} compiled successfully"
                        )
                    except Exception as e:
                        print(f"  ⚠ Batch size {sample_input.shape[0]} failed: {e}")
                        continue

            print("✓ DINO model compiled with TensorRT successfully!")

        except Exception as e:
            print(f"❌ TensorRT compilation failed: {e}")
            print("Try to load without `use_trt`...")

    def get_model(self) -> torch.nn.Module:
        """
        Get the loaded DINO model, loading it if necessary.

        Returns:
            torch.nn.Module: The loaded DINO model
        """
        self._load_model_and_transform(self.model_name, self._device, self.use_trt)
        return self.__model__

    def get_transform(self) -> transforms.Compose:
        """
        Get the image transform, loading the model if necessary.

        Returns:
            torchvision.transforms.Compose: The image transform pipeline
        """
        self._load_model_and_transform(self.model_name, self._device, self.use_trt)
        return self.__transform__

    @property
    def n_patch(self) -> int:
        """
        Get the patch size, loading the model if necessary.

        Returns:
            int: Number of patches per side
        """
        self._load_model_and_transform(self.model_name, self._device, self.use_trt)
        return self.__n_patch__

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded"""
        return self.__model__ is not None

    @property
    def device(self) -> torch.device:
        """Get the device the model will be/is loaded on"""
        return self._device

    @classmethod
    def clear_model(cls) -> None:
        """Clear the loaded model to free memory"""
        if cls.__model__ is not None:
            del cls.__model__
            cls.__model__ = None
            cls.__transform__ = None
            cls.__n_patch__ = None
            cls.__device__ = None
            cls.__use_trt__ = False
            cls.__cuda_stream__ = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print("DINO model cleared from memory")


class MaterialUpsampler:
    """
    Nearest-neighbor material property interpolation for spatial upsampling.

    Efficiently interpolates material properties from sparse voxel data to arbitrary
    3D query points using scipy's cKDTree for fast spatial lookups.
    """

    def __init__(self, voxel_coords: np.ndarray, voxel_materials: np.ndarray):
        """
        Initialize the upsampler with voxel data.

        Args:
            voxel_coords: Voxel coordinates (N, 3) in world space
            voxel_materials: Material properties per voxel (N, 3) [E, nu, rho]
        """
        self.voxel_coords = np.asarray(voxel_coords, dtype=np.float32)
        self.voxel_materials = np.asarray(voxel_materials, dtype=np.float32)

        # Build KDTree for fast nearest neighbor search

        self.tree = cKDTree(self.voxel_coords)

    def interpolate(
        self, query_points: np.ndarray, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate material properties to query points via k-nearest neighbors.

        Args:
            query_points: Target coordinates (M, 3) for interpolation
            k: Number of neighbors for interpolation (1=NN, >1=weighted average)

        Returns:
            Tuple of (interpolated_materials, distances):
            - interpolated_materials: (M, 3) [E, nu, rho] at query points
            - distances: (M,) distance to nearest neighbor

        Raises:
            ValueError: If query_points shape is not (N, 3)
        """
        query_points = np.asarray(query_points, dtype=np.float32)

        if query_points.shape[1] != 3:
            raise ValueError(f"Query points must be (N, 3), got {query_points.shape}")

        # Find nearest neighbors
        distances, indices = self.tree.query(query_points, k=k)

        if k == 1:
            # Simple nearest neighbor
            interpolated_materials = self.voxel_materials[indices]
        else:
            # Distance-weighted average of k nearest neighbors
            weights = 1.0 / (
                distances + 1e-10
            )  # Add small epsilon to avoid division by zero
            weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize weights

            # Weighted average of materials
            interpolated_materials = np.sum(
                self.voxel_materials[indices] * weights[:, :, np.newaxis], axis=1
            )
            distances = distances[:, 0]  # Return distance to closest neighbor

        return interpolated_materials, distances

    def interpolate_to_gaussians(
        self, gaussian_model: "Gaussian"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to interpolate to Gaussian splat centers.

        Args:
            gaussian_model: Gaussian splat model

        Returns:
            Tuple of (interpolated_materials, distances)
        """
        # Get Gaussian positions
        gaussian_positions = gaussian_model.get_xyz.detach().cpu().numpy()

        return self.interpolate(gaussian_positions)

    def save_results(
        self,
        query_points: np.ndarray,
        materials: np.ndarray,
        distances: np.ndarray,
        output_path: str,
        format: str = "npz",
    ):
        """
        Save interpolation results to file.

        Args:
            query_points: Query coordinates (M, 3)
            materials: Interpolated materials (M, 3)
            distances: Distances to nearest neighbors (M,)
            output_path: Output file path
            format: File format ("npz" or "pth")
        """

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        if format.lower() == "npz":
            # Create structured array
            num_points = len(query_points)
            dtype = [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("youngs_modulus", "<f4"),
                ("poissons_ratio", "<f4"),
                ("density", "<f4"),
                ("nn_distance", "<f4"),
                ("segment_id", "<U32"),
            ]

            data = np.zeros(num_points, dtype=dtype)
            data["x"] = query_points[:, 0].astype(np.float32)
            data["y"] = query_points[:, 1].astype(np.float32)
            data["z"] = query_points[:, 2].astype(np.float32)
            data["youngs_modulus"] = materials[:, 0].astype(np.float32)
            data["poissons_ratio"] = materials[:, 1].astype(np.float32)
            data["density"] = materials[:, 2].astype(np.float32)
            data["nn_distance"] = distances.astype(np.float32)
            data["segment_id"] = "interpolated_material"

            np.savez_compressed(output_path, voxel_data=data)

        elif format.lower() == "pth":
            # Save as PyTorch tensors
            data = {
                "x": torch.from_numpy(query_points[:, 0]),
                "y": torch.from_numpy(query_points[:, 1]),
                "z": torch.from_numpy(query_points[:, 2]),
                "youngs_modulus": torch.from_numpy(materials[:, 0]),
                "poissons_ratio": torch.from_numpy(materials[:, 1]),
                "density": torch.from_numpy(materials[:, 2]),
                "nn_distance": torch.from_numpy(distances),
                "query_points": torch.from_numpy(query_points),
                "materials": torch.from_numpy(materials),
            }
            torch.save(data, output_path)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'pth'")

    @property
    def num_voxels(self) -> int:
        """Get number of voxels"""
        return len(self.voxel_coords)

    @property
    def material_stats(self) -> Dict[str, Any]:
        """Get material property statistics"""
        return {
            "youngs_modulus": {
                "mean": self.voxel_materials[:, 0].mean(),
                "std": self.voxel_materials[:, 0].std(),
            },
            "poisson_ratio": {
                "mean": self.voxel_materials[:, 1].mean(),
                "std": self.voxel_materials[:, 1].std(),
            },
            "density": {
                "mean": self.voxel_materials[:, 2].mean(),
                "std": self.voxel_materials[:, 2].std(),
            },
        }


def save_materials(
    materials_dict: Dict[str, np.ndarray],
    output_path: str,
    format: str = "npz",
) -> None:
    """
    Save materials to file in specified format.

    This utility function saves material property dictionaries to disk in various formats.
    It handles coordinate detection automatically and creates appropriate structured arrays.

    Args:
        materials_dict: Dictionary containing material properties with keys:
            - 'youngs_modulus': Young's modulus values
            - 'poisson_ratio': Poisson's ratio values
            - 'density': Density values
            - Coordinates: Either 'voxel_coords_world' or 'splat_coords_world'
        output_path: Output file path
        format: File format ("npz" or "pth")

    Raises:
        ValueError: If no material data found or unsupported format

    Examples:
        >>> from vomp.inference.utils import save_materials
        >>> save_materials(results, "materials.npz")
        >>> save_materials(results, "materials.pth", format="pth")
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    if format.lower() == "npz":
        # Save as compressed numpy archive
        if "voxel_data" in materials_dict:
            # Already in structured array format
            np.savez_compressed(output_path, **materials_dict)
        else:
            # Convert to structured array format
            num_points = len(materials_dict.get("youngs_modulus", []))
            if num_points == 0:
                raise ValueError("No material data to save")

            # Create structured array identical to existing pipeline
            dtype = [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("youngs_modulus", "<f4"),
                ("poissons_ratio", "<f4"),
                ("density", "<f4"),
                ("segment_id", "<U32"),
            ]
            voxel_data = np.zeros(num_points, dtype=dtype)

            # Detect coordinates - prefer query coords, fallback to voxel coords
            if "query_coords_world" in materials_dict:
                coords = materials_dict["query_coords_world"]
                segment_id = "query_point_material"
            elif "voxel_coords_world" in materials_dict:
                coords = materials_dict["voxel_coords_world"]
                segment_id = "voxel_material"
            else:
                raise ValueError(
                    "No coordinate data found. Expected 'query_coords_world' or 'voxel_coords_world'"
                )

            # Fill coordinate data
            voxel_data["x"] = coords[:, 0].astype(np.float32)
            voxel_data["y"] = coords[:, 1].astype(np.float32)
            voxel_data["z"] = coords[:, 2].astype(np.float32)

            # Fill material properties
            voxel_data["youngs_modulus"] = materials_dict["youngs_modulus"].astype(
                np.float32
            )
            voxel_data["poissons_ratio"] = materials_dict["poisson_ratio"].astype(
                np.float32
            )
            voxel_data["density"] = materials_dict["density"].astype(np.float32)
            voxel_data["segment_id"] = segment_id

            np.savez_compressed(output_path, voxel_data=voxel_data)

    elif format.lower() == "pth":
        # Save as PyTorch tensors
        torch_dict = {}
        for key, value in materials_dict.items():
            if isinstance(value, np.ndarray):
                torch_dict[key] = torch.from_numpy(value)
            else:
                torch_dict[key] = value
        torch.save(torch_dict, output_path)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'pth'")

    print(f"Saved materials to: {output_path}")


def load_materials(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load materials from file.

    Args:
        file_path: Path to materials file (.npz or .pth)

    Returns:
        Dictionary containing material properties
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Materials file not found: {file_path}")

    if file_path.endswith(".npz"):
        data = np.load(file_path)
        if "voxel_data" in data:
            # Structured array format
            voxel_data = data["voxel_data"]
            return {
                "x": voxel_data["x"],
                "y": voxel_data["y"],
                "z": voxel_data["z"],
                "youngs_modulus": voxel_data["youngs_modulus"],
                "poisson_ratio": voxel_data["poissons_ratio"],
                "density": voxel_data["density"],
                "coords_world": np.column_stack(
                    [voxel_data["x"], voxel_data["y"], voxel_data["z"]]
                ),
            }
        else:
            # Direct dictionary format
            return dict(data)

    elif file_path.endswith(".pth"):
        # PyTorch format
        data = torch.load(file_path, map_location="cpu")
        return {
            k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }

    else:
        raise ValueError(
            f"Unsupported file format. Expected .npz or .pth, got: {file_path}"
        )


def get_mesh_transform_params(mesh_path: str) -> Tuple[np.ndarray, float]:
    """
    Compute normalization parameters for a mesh.

    Computes the center and scale used to normalize a mesh to [-0.5, 0.5] range.
    This matches the normalization used in mesh voxelization.

    Args:
        mesh_path: Path to mesh file (supports OBJ, PLY, STL, etc.)

    Returns:
        Tuple of (center, scale) where:
            - center: 3D center point of the mesh bounding box (np.ndarray)
            - scale: Maximum dimension of the bounding box (float)
    """
    mesh = trimesh.load(mesh_path)
    if hasattr(mesh, "vertices"):
        mesh_vertices = mesh.vertices
    else:
        vertices_list = []
        for geometry in mesh.geometry.values():
            if hasattr(geometry, "vertices"):
                vertices_list.append(geometry.vertices)
        if not vertices_list:
            raise ValueError("No vertices found in mesh")
        mesh_vertices = np.vstack(vertices_list)

    # Compute transformation parameters
    center = (mesh_vertices.min(axis=0) + mesh_vertices.max(axis=0)) / 2
    scale = (mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)).max()

    return center, scale


def normalize_coords(
    coords: np.ndarray, center: np.ndarray, scale: float, clip: bool = True
) -> np.ndarray:
    """
    Normalize coordinates to [-0.5, 0.5] range.

    Args:
        coords: Input coordinates (N, 3)
        center: Center point for normalization (3,)
        scale: Scale factor for normalization (scalar)
        clip: Whether to clip to [-0.5 + eps, 0.5 - eps] (default: True)

    Returns:
        Normalized coordinates (N, 3)
    """
    normalized = (coords - center) / scale
    if clip:
        normalized = np.clip(normalized, -0.5 + 1e-6, 0.5 - 1e-6)
    return normalized


def denormalize_coords(
    coords: np.ndarray, center: np.ndarray, scale: float
) -> np.ndarray:
    """
    Denormalize coordinates from [-0.5, 0.5] range back to original scale.

    Args:
        coords: Normalized coordinates (N, 3)
        center: Center point used for normalization (3,)
        scale: Scale factor used for normalization (scalar)

    Returns:
        Denormalized coordinates (N, 3) in original scale
    """
    return coords * scale + center
