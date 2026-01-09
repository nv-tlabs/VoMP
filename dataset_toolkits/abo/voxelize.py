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
import copy
import sys
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d
import trimesh
import tempfile
import shutil

# Add current directory to path to import dataset modules
sys.path.insert(0, os.path.dirname(__file__))

import ABO500 as dataset_utils


def voxelize_mesh(
    vertices, faces, voxel_size=1 / 64, center_scale=None, max_voxels=None
):
    """
    Voxelize a mesh represented by vertices and faces using volumetric voxelization.

    Args:
        vertices (numpy.ndarray): Array of vertices
        faces (numpy.ndarray): Array of faces
        voxel_size (float): Size of each voxel
        center_scale (tuple): Optional center and scale for normalization
        max_voxels (int): Maximum number of voxels to return (will subsample if exceeded)

    Returns:
        tuple: (voxel_centers, voxel_grid) - center coordinates of voxels and Trimesh voxel grid
    """
    # Create a Trimesh mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Normalize the mesh to [-0.5, 0.5] range
    vertices = mesh.vertices.copy()

    if center_scale is None:
        vertices_min = np.min(vertices, axis=0)
        vertices_max = np.max(vertices, axis=0)
        center = (vertices_min + vertices_max) / 2
        scale = np.max(vertices_max - vertices_min)
    else:
        center, scale = center_scale

    vertices = (vertices - center) / scale
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)

    # Update mesh with normalized vertices
    mesh.vertices = vertices

    # Create volumetric voxel grid using Trimesh
    voxel_grid = mesh.voxelized(pitch=voxel_size).fill()

    # Get voxel centers from the filled voxel grid
    voxel_centers = voxel_grid.points

    # Subsample if we have too many voxels
    if max_voxels is not None and len(voxel_centers) > max_voxels:
        print(f"Subsampling voxels: {len(voxel_centers):,} -> {max_voxels:,}")
        # Use random sampling to maintain spatial distribution
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(voxel_centers), max_voxels, replace=False)
        voxel_centers = voxel_centers[indices]

    return voxel_centers, voxel_grid


def load_glb_mesh(glb_path):
    """
    Load a GLB file and extract mesh data.

    Args:
        glb_path (str): Path to the GLB file

    Returns:
        tuple: (vertices, faces) - mesh vertices and faces
    """
    try:
        # Load the GLB file using trimesh
        mesh = trimesh.load(glb_path)

        # Handle different mesh types
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, combine all meshes
            combined_mesh = trimesh.util.concatenate(
                [
                    geometry
                    for geometry in mesh.geometry.values()
                    if isinstance(geometry, trimesh.Trimesh)
                ]
            )
            if combined_mesh is None:
                raise ValueError("No valid meshes found in GLB file")
            mesh = combined_mesh
        elif not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("GLB file does not contain a valid mesh")

        # Ensure the mesh has faces
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")

        return mesh.vertices, mesh.faces

    except Exception as e:
        print(f"Error loading GLB file {glb_path}: {e}")
        return None, None


def voxelize_glb(glb_path, sha256, output_dir, max_voxels=None):
    """
    Voxelize a GLB file and save the result.

    Args:
        glb_path (str): Path to the GLB file
        sha256 (str): SHA256 hash of the file
        output_dir (str): Directory to save the voxelized data
        max_voxels (int): Maximum number of voxels to generate

    Returns:
        dict: Result dictionary with processing info
    """
    try:
        # Load the GLB mesh
        vertices, faces = load_glb_mesh(glb_path)

        if vertices is None or faces is None:
            print(f"Failed to load mesh from {glb_path}")
            return {"sha256": sha256, "voxelized": False, "num_voxels": 0}

        print(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")

        # Voxelize the mesh
        voxel_centers, voxel_grid = voxelize_mesh(
            vertices, faces, max_voxels=max_voxels
        )

        if len(voxel_centers) == 0:
            print(f"No voxels generated for {sha256}")
            return {"sha256": sha256, "voxelized": False, "num_voxels": 0}

        # Save voxel centers as PLY file
        ply_output_path = os.path.join(output_dir, "voxels", f"{sha256}.ply")
        save_ply(ply_output_path, voxel_centers)

        print(f"Voxelized {sha256}: {len(voxel_centers)} voxels")

        return {"sha256": sha256, "voxelized": True, "num_voxels": len(voxel_centers)}

    except Exception as e:
        print(f"Error voxelizing {glb_path}: {e}")
        import traceback

        traceback.print_exc()
        return {"sha256": sha256, "voxelized": False, "num_voxels": 0}


def save_ply(filename, points):
    """
    Save points as a PLY file.

    Args:
        filename (str): Output filename
        points (numpy.ndarray): Array of 3D points
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def _voxelize(file_path, sha256, output_dir=None, max_voxels=None):
    """Wrapper function for voxelization."""
    return voxelize_glb(file_path, sha256, output_dir, max_voxels=max_voxels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelize ABO 500 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing metadata and where to save voxelized data",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Specific instances to process (comma-separated or file path)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force voxelization even if already processed",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N objects"
    )
    parser.add_argument(
        "--max_voxels",
        type=int,
        default=70000,
        help="Maximum number of voxels per asset",
    )

    args = parser.parse_args()
    opt = edict(vars(args))

    # Create voxels directory
    os.makedirs(os.path.join(opt.output_dir, "voxels"), exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(opt.output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise ValueError(f"metadata.csv not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Filter instances if specified
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]
    else:
        # Only process objects that haven't been voxelized yet
        if "voxelized" in metadata.columns and not opt.force:
            metadata = metadata[metadata["voxelized"] == False]

    # Apply distributed processing
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    # Apply limit if specified
    if opt.limit is not None:
        metadata = metadata.head(opt.limit)

    print(f"Processing {len(metadata)} objects with max_voxels={opt.max_voxels:,}...")

    # Track already processed objects
    records = []

    # Filter out objects that are already processed
    if not opt.force:
        for sha256 in copy.copy(metadata["sha256"].values):
            ply_path = os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
            if os.path.exists(ply_path):
                try:
                    pts = utils3d.io.read_ply(ply_path)[0]
                    records.append(
                        {"sha256": sha256, "voxelized": True, "num_voxels": len(pts)}
                    )
                    metadata = metadata[metadata["sha256"] != sha256]
                except:
                    # If file is corrupted, re-process it
                    pass

    # Process remaining objects
    if len(metadata) > 0:
        func = partial(_voxelize, output_dir=opt.output_dir, max_voxels=opt.max_voxels)
        voxelized = dataset_utils.foreach_instance(
            metadata,
            opt.output_dir,
            func,
            max_workers=opt.max_workers,
            desc="Voxelizing",
        )

        # Combine results
        if len(records) > 0:
            voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])

        # Save results
        voxelized.to_csv(
            os.path.join(opt.output_dir, f"voxelized_{opt.rank}.csv"), index=False
        )

        print(f"Voxelization complete. Results saved to voxelized_{opt.rank}.csv")
    else:
        print("No objects to process.")
