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
import importlib
import argparse
import pandas as pd
import json
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d
import numpy as np
import tempfile
import shutil
from pxr import Usd, UsdGeom, Gf, UsdShade
import re
import trimesh

sys.path.append(os.path.dirname(__file__))
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.common import (
    extract_materials_from_usd,
)

# For fuzzy fallback using material ranges CSV
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    load_material_ranges,
    find_reference_materials,
    parse_numerical_range_str,
)

# SimReady helper to parse material names into opacity/material_type/semantic_usage
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.simready import (
    parse_material_name,
)

DEFAULT_MATERIAL_PROPS = {
    "youngs_modulus": 1e6,  # Default Young's modulus in Pa
    "poisson_ratio": 0.3,  # Default Poisson ratio
    "density": 1000.0,  # Default density in kg/m^3
    "friction": 0.5,  # Default friction coefficient
    "dynamic_friction": 0.5,  # Default dynamic friction
    "static_friction": 0.5,  # Default static friction
    "restitution": 0.3,  # Default restitution
}

MATERIAL_DATASET_PATH = "datasets/raw/material_annotations.json"


def load_material_dataset(material_dataset_path):
    """
    Load the material dataset JSON file.

    Args:
        material_dataset_path (str): Path to the material dataset JSON

    Returns:
        list: List of material objects from the JSON file
    """
    try:
        with open(material_dataset_path, "r") as f:
            material_data = json.load(f)

        # Return the raw list from the JSON file
        return material_data
    except Exception as e:
        print(f"Error loading material dataset: {str(e)}")
        return []


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
    # Calculate pitch (voxel size) and use voxelized method with fill for volumetric voxelization
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


def extract_and_voxelize_segments(
    usd_file_path, sha256, output_dir, material_dataset_path, max_voxels=None
):
    """
    Extract each segment from a 'zag_middle' USD model, voxelize them,
    and save combined voxel data with material properties.

    Args:
        usd_file_path (str): Path to the USD file
        output_dir (str): Directory to save extracted segments
        material_dataset_path (str): Path to the material dataset JSON
        max_voxels (int): Maximum number of voxels per asset (will subsample if exceeded)

    Returns:
        bool: Success or failure
    """
    if not os.path.exists(usd_file_path):
        print(f"Error: USD file not found at {usd_file_path}")
        return False

    # Load material dataset
    material_lookup = load_material_dataset(material_dataset_path)
    if not material_lookup:
        print("Warning: Material dataset could not be loaded or is empty.")

    # Load reference material ranges once for fuzzy fallback
    material_db = load_material_ranges()

    # Extract the object name from the file path - use basename without extensions
    object_name = os.path.splitext(os.path.basename(usd_file_path))[0]

    # Remove common suffixes
    object_name = (
        object_name.replace("_inst_base", "").replace("_inst", "").replace("_base", "")
    )

    # Normalize name: strip "sm_" prefix (SimReady naming) and trailing variant indices like "_01"
    original_lower = object_name.lower()

    # Prepare list of candidate names starting with the full original
    object_name_candidates = [original_lower]

    # If name starts with 'sm_', also add the version without it
    if original_lower.startswith("sm_"):
        stripped = original_lower[3:]
        object_name_candidates.append(stripped)
    else:
        stripped = original_lower

    # Variantless (remove numeric suffix) from both versions
    variantless = re.sub(r"_[0-9]{1,2}$", "", stripped)
    if variantless not in object_name_candidates:
        object_name_candidates.append(variantless)

    # Store for later use in fuzzy fallback
    object_name_lower = original_lower

    # DEBUG: Print the extracted object name
    # print(f"DEBUG: Extracted object name: {object_name}")
    # print(f"DEBUG: USD file path: {usd_file_path}")

    try:
        # Load USD stage
        # Do NOT modify the path here – keep the original _inst_base if present

        # We no longer rely on extract_materials_from_usd for segment names; instead we
        # directly construct segment keys from mesh paths and match them against the
        # JSON dataset. This avoids the extra USD-parsing pass used in VLM annotation.

        # DEBUG: Print all segments found
        # (Removed: dependency on material_result)

        stage = Usd.Stage.Open(usd_file_path)
        if not stage:
            print(f"Error: Failed to open USD stage from {usd_file_path}")
            return False

        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            print("DEBUG: No default prim found, using root")
            default_prim = stage.GetPrimAtPath("/")

        # Find all mesh prims
        mesh_prims = []
        for prim in Usd.PrimRange(default_prim):
            if prim.GetTypeName() == "Mesh":
                mesh_prims.append(prim)
                # DEBUG: Print each mesh found
                # print(f"DEBUG: Found mesh: {prim.GetPath()}")

        if not mesh_prims:
            print(f"Error: No meshes found in {usd_file_path}")
            return False

        # Compute global bounding box across all meshes to preserve relative positions
        all_vertices = []
        for prim in mesh_prims:
            mesh_tmp = UsdGeom.Mesh(prim)
            pts_attr = mesh_tmp.GetPointsAttr()
            if pts_attr and pts_attr.Get():
                points_local = pts_attr.Get()
                # Get world transform for this mesh
                xformable = UsdGeom.Xformable(prim)
                world_transform = xformable.ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                # Transform points to world space
                verts_tmp = []
                for p in points_local:
                    p_world = world_transform.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                    verts_tmp.append(
                        [float(p_world[0]), float(p_world[1]), float(p_world[2])]
                    )
                verts_tmp = np.array(verts_tmp, dtype=np.float32)
                all_vertices.append(verts_tmp)

        if len(all_vertices) == 0:
            print("DEBUG: No vertices found in any mesh")
            return False

        all_vertices_concat = np.vstack(all_vertices)
        global_min = np.min(all_vertices_concat, axis=0)
        global_max = np.max(all_vertices_concat, axis=0)
        global_center = (global_min + global_max) / 2
        global_scale = np.max(global_max - global_min)
        global_center_scale = (global_center, global_scale)

        # Store voxels and properties for all segments
        all_voxel_centers = []
        all_material_props = []
        all_segment_ids = []

        # Process each mesh as a potential segment
        for mesh_prim in mesh_prims:
            mesh = UsdGeom.Mesh(mesh_prim)
            mesh_name = mesh_prim.GetName()

            # Attempt to get bound material name for this mesh
            material_name = None
            try:
                binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
                direct_binding = binding_api.GetDirectBinding()
                if direct_binding and direct_binding.GetMaterial():
                    material_name = direct_binding.GetMaterial().GetPath().name
                else:
                    # Try collection bindings
                    for col_binding in binding_api.GetCollectionBindings():
                        mat = col_binding.GetMaterial()
                        if mat:
                            material_name = mat.GetPath().name
                            break
            except Exception:
                material_name = None

            # Get the full path for more accurate segment identification
            full_path = str(mesh_prim.GetPath())
            path_parts = full_path.split("/")
            # Skip the root and "World" prefixes
            parent_parts = [p for p in path_parts[2:-1] if p]

            # Construct segment key that matches the naming convention in common.py
            segment_key = mesh_name
            if parent_parts:
                segment_key = "_".join(parent_parts + [mesh_name])

            # Lowercase version for case-insensitive lookups
            segment_key_lower = segment_key.lower()

            # print(f"DEBUG: Processing segment key: {segment_key}")

            # Get vertices
            points_attr = mesh.GetPointsAttr()
            if not points_attr or not points_attr.Get():
                print(f"DEBUG: No vertices in mesh {segment_key}, skipping")
                continue

            points_local = points_attr.Get()
            # Get world transform for this mesh
            xformable = UsdGeom.Xformable(mesh_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            # Transform points to world space
            vertices = []
            for p in points_local:
                p_world = world_transform.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                vertices.append(
                    [float(p_world[0]), float(p_world[1]), float(p_world[2])]
                )
            vertices = np.array(vertices, dtype=np.float32)

            # Get face information
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

            if not face_vertex_counts or not face_vertex_indices:
                print(f"DEBUG: No faces in mesh {segment_key}, skipping")
                continue

            # Extract faces
            faces = []
            idx = 0
            for count in face_vertex_counts:
                if count == 3:
                    # Triangle
                    face = [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                    faces.append(face)
                elif count == 4:
                    # Triangulate quad into two triangles
                    face1 = [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                    face2 = [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 2],
                        face_vertex_indices[idx + 3],
                    ]
                    faces.append(face1)
                    faces.append(face2)
                idx += count

            faces = np.array(faces, dtype=np.int32)

            # We no longer consult VLM-extracted segments – rely purely on JSON file

            # Check for this segment key in the material_lookup
            material_props = DEFAULT_MATERIAL_PROPS.copy()
            segment_found = False
            ref_applied = False

            # Loop through objects to find one with matching object_name (case-insensitive)
            for obj in material_lookup:
                if str(obj.get("object_name", "")).lower() in object_name_candidates:
                    obj_segments = obj.get("segments", {})

                    # Deterministic keys to try (mirrors extract_materials_from_usd logic)
                    candidate_keys = [segment_key, mesh_name, f"{mesh_name}_whole"]
                    if material_name:
                        candidate_keys.append(material_name)

                        # Build SimReady-style key if possible
                        try:
                            opacity, mat_type, semantic_usage = parse_material_name(
                                material_name
                            )
                            if semantic_usage:
                                simready_key = f"opaque__{mat_type}__{semantic_usage}"
                                candidate_keys.append(simready_key)
                            # Default fallback naming
                            if (
                                "default__" not in material_name.lower()
                                and mat_type
                                and object_name
                            ):
                                candidate_keys.append(
                                    f"default__{mat_type}__{object_name}"
                                )
                        except Exception:
                            pass

                    segment_data = None
                    for key in candidate_keys:
                        # Try exact match first
                        if key in obj_segments:
                            segment_data = obj_segments[key]
                            segment_found = True
                            break
                        # Case-insensitive match
                        for seg_name, seg_info in obj_segments.items():
                            if seg_name.lower() == key.lower():
                                segment_data = seg_info
                                segment_found = True
                                break
                        if segment_found:
                            break

                    # After candidate key checks, attempt prefix fallbacks if still not found
                    if not segment_found:
                        # Case-insensitive prefix check between segment_key / mesh_name and JSON keys
                        for seg_name, seg_info in obj_segments.items():
                            seg_low = seg_name.lower()
                            if (
                                seg_low.startswith(segment_key_lower)
                                or segment_key_lower.startswith(seg_low)
                                or seg_low.startswith(mesh_name.lower())
                                or (
                                    material_name
                                    and seg_low.startswith(material_name.lower())
                                )
                            ):
                                segment_data = seg_info
                                segment_found = True
                                break

                    # Additional prefix2 fallback (first two tokens)
                    if not segment_found:
                        mesh_tokens = mesh_name.split("_")
                        if len(mesh_tokens) >= 2:
                            prefix2 = "_".join(mesh_tokens[:2]).lower()
                            for seg_name, seg_info in obj_segments.items():
                                if seg_name.lower().startswith(prefix2):
                                    segment_data = seg_info
                                    segment_found = True
                                    break

                    if segment_found and segment_data is not None:
                        # Extract material properties directly
                        material_props["density"] = float(
                            segment_data.get(
                                "density", DEFAULT_MATERIAL_PROPS["density"]
                            )
                        )
                        material_props["youngs_modulus"] = float(
                            segment_data.get(
                                "youngs_modulus",
                                DEFAULT_MATERIAL_PROPS["youngs_modulus"],
                            )
                        )
                        material_props["poisson_ratio"] = float(
                            segment_data.get(
                                "poissons_ratio",
                                DEFAULT_MATERIAL_PROPS["poisson_ratio"],
                            )
                        )

                        # Optional properties (ignored if missing)
                    else:
                        # If no segment found, keep default material properties
                        pass

                    # Break out after processing the matching object
                    break

            if not segment_found:
                print(f"DEBUG: Using DEFAULT material properties (no segment match)")
                print("Object name: ", object_name)
                print("Segment key: ", segment_key)

                # Fuzzy fallback: use bound material name to fetch approximate values
                if material_name:
                    refs = find_reference_materials(
                        material_db, material_name, max_matches=1
                    )
                    if refs:
                        ref = refs[0]
                        try:
                            y_min, y_max = parse_numerical_range_str(ref.get("youngs"))
                            material_props["youngs_modulus"] = (
                                (y_min + y_max) / 2 if y_max else y_min
                            ) * 1e9  # convert GPa to Pa
                        except Exception:
                            pass
                        try:
                            p_min, p_max = parse_numerical_range_str(ref.get("poisson"))
                            material_props["poisson_ratio"] = (
                                (p_min + p_max) / 2 if p_max else p_min
                            )
                        except Exception:
                            pass
                        try:
                            d_min, d_max = parse_numerical_range_str(ref.get("density"))
                            material_props["density"] = (
                                (d_min + d_max) / 2 if d_max else d_min
                            )
                        except Exception:
                            pass
                        ref_applied = True
                        print(
                            f"DEBUG: Fuzzy fallback applied using reference material '{ref.get('name')}' for material '{material_name}'"
                        )

                # If still not applied, try fuzzy match on object name and segment key
                if not ref_applied:
                    extra_candidates = [segment_key_lower, mesh_name.lower()]
                    for cand in extra_candidates:
                        # strict containment search only (no fuzzy)
                        refs = [
                            m
                            for m in material_db
                            if cand in m["name"].lower() or m["name"].lower() in cand
                        ][:1]
                        if refs:
                            ref = refs[0]
                            try:
                                y_min, y_max = parse_numerical_range_str(
                                    ref.get("youngs")
                                )
                                material_props["youngs_modulus"] = (
                                    (y_min + y_max) / 2 if y_max else y_min
                                ) * 1e9
                            except Exception:
                                pass
                            try:
                                p_min, p_max = parse_numerical_range_str(
                                    ref.get("poisson")
                                )
                                material_props["poisson_ratio"] = (
                                    (p_min + p_max) / 2 if p_max else p_min
                                )
                            except Exception:
                                pass
                            try:
                                d_min, d_max = parse_numerical_range_str(
                                    ref.get("density")
                                )
                                material_props["density"] = (
                                    (d_min + d_max) / 2 if d_max else d_min
                                )
                            except Exception:
                                pass
                            ref_applied = True
                            print(
                                f"DEBUG: Fuzzy fallback applied using reference material '{ref.get('name')}' for candidate '{cand}'"
                            )
                            break

            # If we still have neither a direct match nor a reference fallback, skip this mesh
            if not segment_found and not ref_applied:
                print(f"DEBUG: Skipping mesh {segment_key} – no material data found")
                continue

            # Voxelize the mesh
            voxel_centers, _ = voxelize_mesh(
                vertices, faces, center_scale=global_center_scale, max_voxels=max_voxels
            )
            # print(f"DEBUG: Segment {segment_key} generated {len(voxel_centers)} voxels")

            # Add voxels and properties to the combined arrays
            if len(voxel_centers) > 0:
                all_voxel_centers.append(voxel_centers)

                # Create array of material properties for all voxels in this segment
                segment_props = np.tile(
                    [
                        material_props["youngs_modulus"],
                        material_props["poisson_ratio"],
                        material_props["density"],
                    ],
                    (len(voxel_centers), 1),
                )
                all_material_props.append(segment_props)

                # Use segment_key directly as the segment ID
                all_segment_ids.extend([segment_key] * len(voxel_centers))

        # Combine all voxels and properties
        if all_voxel_centers:
            combined_voxel_centers = np.vstack(all_voxel_centers)
            combined_material_props = np.vstack(all_material_props)

            # Apply global max_voxels limit if specified and exceeded
            if max_voxels is not None and len(combined_voxel_centers) > max_voxels:
                print(
                    f"Global subsampling: {len(combined_voxel_centers):,} -> {max_voxels:,} voxels"
                )
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(
                    len(combined_voxel_centers), max_voxels, replace=False
                )
                combined_voxel_centers = combined_voxel_centers[indices]
                combined_material_props = combined_material_props[indices]
                all_segment_ids = [all_segment_ids[i] for i in indices]

            # print(f"DEBUG: Combined model has {len(combined_voxel_centers)} voxels")

            ply_output_path = os.path.join(output_dir, "voxels", f"{sha256}.ply")
            save_ply(ply_output_path, combined_voxel_centers)
            # print(f"DEBUG: Saved voxel centers to {ply_output_path}")

            ply_positions = utils3d.io.read_ply(ply_output_path)[0]
            ply_positions = ply_positions.astype(np.float32)
            indices = ((ply_positions + 0.5) * 64).astype(np.int64)
            indices = np.clip(indices, 0, 63)
            discretized_positions = indices.astype(np.float32) / 64.0 - 0.5
            # print(f"DEBUG: Materials using same count as PLY: {len(discretized_positions)} voxels")
            # Save voxel centers and material properties as a single NPZ file
            voxel_data = np.zeros(
                len(discretized_positions),
                dtype=[
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("youngs_modulus", "<f4"),
                    ("poissons_ratio", "<f4"),
                    ("density", "<f4"),
                    ("segment_id", "<U32"),
                ],
            )

            voxel_data["x"] = discretized_positions[:, 0]
            voxel_data["y"] = discretized_positions[:, 1]
            voxel_data["z"] = discretized_positions[:, 2]
            voxel_data["youngs_modulus"] = combined_material_props[:, 0]
            voxel_data["poissons_ratio"] = combined_material_props[:, 1]
            voxel_data["density"] = combined_material_props[:, 2]
            voxel_data["segment_id"] = all_segment_ids

            # Save in the same format as the reference files
            npz_output_path = os.path.join(
                output_dir, "voxels", f"{sha256}_with_materials.npz"
            )
            np.savez(npz_output_path, voxel_data=voxel_data)
            # print(f"DEBUG: Saved combined voxel data to {npz_output_path}")

            return len(combined_voxel_centers)
        else:
            print("No valid segments found for voxelization")
            return False

    except Exception as e:
        print(f"Error processing USD file: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def save_ply(filename, points):
    """
    Save points as a PLY file.

    Args:
        filename (str): Output filename
        points (numpy.ndarray): Array of 3D points
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def _voxelize(file, sha256, dataset=None, output_dir=None, max_voxels=None):
    num_voxels = extract_and_voxelize_segments(
        file, sha256, output_dir, MATERIAL_DATASET_PATH, max_voxels=max_voxels
    )
    return {"sha256": sha256, "voxelized": True, "num_voxels": num_voxels}


if __name__ == "__main__":
    # Use unified dataset utilities that work across all subsets
    from dataset_toolkits.datasets import allmats as dataset_utils

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the metadata"
    )
    parser.add_argument(
        "--instances", type=str, default=None, help="Instances to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force voxelization even if the object has already been voxelized",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, process only the first N objects after filtering",
    )
    parser.add_argument(
        "--max_voxels",
        type=int,
        default=70000,
        help="Maximum number of voxels per asset (will subsample if exceeded). Based on analysis of successfully processed assets.",
    )
    # Parse all CLI arguments (no leading dataset label needed)
    opt = parser.parse_args(sys.argv[1:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, "voxels"), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))

    if opt.instances is None:
        if "rendered" not in metadata.columns:
            raise ValueError(
                'metadata.csv does not have "rendered" column, please run "build_metadata.py" first'
            )
        metadata = metadata[metadata["rendered"] == True]
        if "voxelized" in metadata.columns:
            if not opt.force:
                metadata = metadata[metadata["voxelized"] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    # Initialize list to hold stats of previously voxelised objects
    records = []

    # Apply limit if requested
    if opt.get("limit") is not None:
        metadata = metadata.head(opt.limit)

    print(f"Processing {len(metadata)} objects with max_voxels={opt.max_voxels:,}...")

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "voxels"), exist_ok=True)

    # filter out objects that are already processed
    if not opt.force:
        for sha256 in copy.copy(metadata["sha256"].values):
            if os.path.exists(os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")):
                pts = utils3d.io.read_ply(
                    os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
                )[0]
                records.append(
                    {"sha256": sha256, "voxelized": True, "num_voxels": len(pts)}
                )
                metadata = metadata[metadata["sha256"] != sha256]

    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir, max_voxels=opt.max_voxels)
    voxelized = dataset_utils.foreach_instance(
        metadata, opt.output_dir, func, max_workers=opt.max_workers, desc="Voxelizing"
    )
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(
        os.path.join(opt.output_dir, f"voxelized_{opt.rank}.csv"), index=False
    )
