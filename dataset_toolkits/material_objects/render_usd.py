#!/usr/bin/env python3
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
Clean USD rendering pipeline.

This script extracts meshes and textures directly from USD files,
similar to how Omniverse exports meshes. It does NOT search the filesystem
for textures - all texture paths come from the USD shaders themselves.

For vegetation datasets that use MDL materials, it parses the MDL files
to extract texture references.
"""

import os
import sys
import json
import re
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from subprocess import call, DEVNULL

import numpy as np
import pandas as pd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import sphere_hammersley_sequence

BLENDER_LINK = (
    "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz"
)
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"


class USDMaterialExtractor:
    """
    Extracts materials and textures directly from USD files.

    This class reads shader inputs from USD prims and resolves texture paths
    relative to the USD file. For MDL materials (used in vegetation), it
    parses the MDL files to extract texture references.
    """

    # MDL texture patterns (for vegetation)
    MDL_TEXTURE_PATTERNS = [
        r'diffuse_texture:\s*texture_2d\("([^"]+)"',
        r'normalmap_texture:\s*texture_2d\("([^"]+)"',
        r'reflectionroughness_texture:\s*texture_2d\("([^"]+)"',
        r'metallic_texture:\s*texture_2d\("([^"]+)"',
        r'ORM_texture:\s*texture_2d\("([^"]+)"',
    ]

    def __init__(self, usd_path: str, verbose: bool = False):
        self.usd_path = Path(usd_path).resolve()
        self.usd_dir = self.usd_path.parent
        self.verbose = verbose
        self.stage = None

        # Extracted data
        self.materials = {}  # material_path -> {input_name: texture_path}
        self.meshes = {}  # mesh_path -> {name, material, vertices, faces, uvs}
        self.mesh_materials = {}  # mesh_path -> material_path

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _resolve_texture_path(self, texture_path: str) -> Optional[Path]:
        texture_path = texture_path.strip("@")

        # Handle UDIM textures - replace <UDIM> with first available tile
        if "<UDIM>" in texture_path:
            # Try common UDIM tile numbers
            for udim in ["1001", "1002", "1003", "1004"]:
                resolved = self._resolve_texture_path(
                    texture_path.replace("<UDIM>", udim)
                )
                if resolved:
                    return resolved
            return None

        # Already absolute
        if Path(texture_path).is_absolute():
            p = Path(texture_path)
            return p if p.exists() else None

        # Try relative to USD directory
        candidates = [
            self.usd_dir / texture_path,
            self.usd_dir / "textures" / Path(texture_path).name,
            self.usd_dir / "Textures" / Path(texture_path).name,
            self.usd_dir / ".." / texture_path,
            self.usd_dir / ".." / "textures" / Path(texture_path).name,
            self.usd_dir / ".." / "materials" / "textures" / Path(texture_path).name,
        ]

        for p in candidates:
            if p.exists():
                return p.resolve()

        # Fuzzy matching: look for files containing the texture name
        texture_name = Path(texture_path).stem  # e.g., "Iron_BaseColor"
        texture_ext = Path(texture_path).suffix  # e.g., ".png"

        # Search in Textures folders
        search_dirs = [
            self.usd_dir / "Textures",
            self.usd_dir / "textures",
            self.usd_dir / ".." / "Textures",
            self.usd_dir / ".." / "textures",
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for f in search_dir.iterdir():
                    # Check if the file contains the texture name (fuzzy match)
                    if (
                        texture_name in f.stem
                        and f.suffix.lower() == texture_ext.lower()
                    ):
                        self._log(f"    (fuzzy match: {texture_name} -> {f.name})")
                        return f.resolve()

        return None

    def _categorize_input(self, name: str) -> str:
        name_lower = name.lower()

        # Check for texture/color type based on common patterns
        if any(
            x in name_lower for x in ["diffuse", "albedo", "basecolor", "base_color"]
        ):
            return "diffuse"
        elif any(x in name_lower for x in ["normal", "bump"]):
            return "normal"
        elif any(x in name_lower for x in ["rough"]):
            return "roughness"
        elif any(x in name_lower for x in ["metal"]):
            return "metallic"
        elif any(x in name_lower for x in ["orm", "occlusion"]):
            return "orm"
        elif any(x in name_lower for x in ["opacity", "alpha"]):
            return "opacity"
        else:
            return name  # Use original name if no match

    def _find_fallback_textures(self, material_name: str) -> Dict[str, str]:
        textures = {}

        # Search in Textures folders
        search_dirs = [
            self.usd_dir / "Textures",
            self.usd_dir / "textures",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Find all unique texture prefixes (e.g., BlueRug from BlueRug_BaseColor.png)
            texture_files = list(search_dir.glob("*.png")) + list(
                search_dir.glob("*.jpg")
            )
            if not texture_files:
                continue

            # Group by prefix (before _BaseColor, _N, _R, etc.)
            prefixes = set()
            for f in texture_files:
                stem = f.stem
                for suffix in [
                    "_BaseColor",
                    "_basecolor",
                    "_A",
                    "_albedo",
                    "_diffuse",
                    "_N",
                    "_Normal",
                    "_normal",
                    "_R",
                    "_Roughness",
                    "_roughness",
                ]:
                    if suffix in stem:
                        prefix = stem.split(suffix)[0]
                        prefixes.add(prefix)
                        break

            # Use the first available texture set
            if prefixes:
                prefix = sorted(prefixes)[0]  # Pick first alphabetically
                self._log(f"    (fallback: using {prefix}_* textures)")

                # Find matching textures
                for f in texture_files:
                    if f.stem.startswith(prefix):
                        stem_lower = f.stem.lower()
                        if any(
                            x in stem_lower
                            for x in ["basecolor", "_a", "albedo", "diffuse"]
                        ):
                            textures["diffuse"] = str(f.resolve())
                            self._log(f"    ✓ fallback diffuse: {f.name}")
                        elif any(x in stem_lower for x in ["normal", "_n"]):
                            textures["normal"] = str(f.resolve())
                            self._log(f"    ✓ fallback normal: {f.name}")
                        elif any(x in stem_lower for x in ["rough", "_r"]):
                            textures["roughness"] = str(f.resolve())
                            self._log(f"    ✓ fallback roughness: {f.name}")

                if textures:
                    return textures

        return textures

    def _extract_textures_from_shader(self, shader: UsdShade.Shader) -> Dict[str, any]:
        result = {}

        for shader_input in shader.GetInputs():
            val = shader_input.Get()
            if val is None:
                continue

            input_name = shader_input.GetBaseName()
            category = self._categorize_input(input_name)

            # Texture path (AssetPath)
            if isinstance(val, Sdf.AssetPath) and val.path:
                texture_path = val.path.strip("@")
                resolved = self._resolve_texture_path(texture_path)
                if resolved:
                    result[category] = str(resolved)
                    self._log(f"    ✓ {input_name} -> {category}: {resolved.name}")
                else:
                    self._log(f"    ✗ {input_name}: {texture_path} (not resolved)")

            # Color value (Vec3)
            elif (
                hasattr(val, "__len__")
                and len(val) == 3
                and "color" in input_name.lower()
            ):
                result[f"{category}_color"] = (
                    float(val[0]),
                    float(val[1]),
                    float(val[2]),
                )
                self._log(
                    f"    ✓ {input_name} -> {category}_color: ({val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f})"
                )

        return result

    def _extract_textures_from_mdl(self, mdl_path: Path) -> Dict[str, str]:
        textures = {}

        if not mdl_path.exists():
            return textures

        try:
            content = mdl_path.read_text()

            # Parse texture references
            type_mapping = {
                "diffuse_texture": "diffuse",
                "normalmap_texture": "normal",
                "reflectionroughness_texture": "roughness",
                "metallic_texture": "metallic",
                "ORM_texture": "orm",
            }

            for tex_type, canonical_name in type_mapping.items():
                pattern = rf'{tex_type}:\s*texture_2d\("([^"]+)"'
                match = re.search(pattern, content)
                if match:
                    rel_path = match.group(1)
                    # MDL paths are relative to the MDL file location
                    resolved = self._resolve_texture_path(rel_path)
                    if not resolved:
                        # Try relative to MDL file directory
                        mdl_dir = mdl_path.parent
                        candidates = [
                            mdl_dir / rel_path,
                            mdl_dir / "textures" / Path(rel_path).name,
                        ]
                        for c in candidates:
                            if c.exists():
                                resolved = c.resolve()
                                break

                    if resolved:
                        textures[canonical_name] = str(resolved)
                        self._log(f"    ✓ {canonical_name}: {resolved.name} (from MDL)")
                    else:
                        self._log(
                            f"    ✗ {canonical_name}: {rel_path} (MDL, not resolved)"
                        )

        except Exception as e:
            self._log(f"    Error parsing MDL {mdl_path}: {e}")

        return textures

    def _find_mdl_for_material(self, material_prim: Usd.Prim) -> Optional[Path]:
        for child in material_prim.GetChildren():
            if child.GetTypeName() == "Shader":
                # Check for MDL source asset
                mdl_attr = child.GetAttribute("info:mdl:sourceAsset")
                if mdl_attr and mdl_attr.Get():
                    mdl_path_val = mdl_attr.Get()
                    if isinstance(mdl_path_val, Sdf.AssetPath) and mdl_path_val.path:
                        mdl_rel = mdl_path_val.path.strip("@")

                        # Try to resolve MDL path
                        candidates = [
                            self.usd_dir / mdl_rel,
                            self.usd_dir / "materials" / Path(mdl_rel).name,
                            self.usd_dir / ".." / "materials" / Path(mdl_rel).name,
                        ]

                        for c in candidates:
                            if c.exists():
                                return c.resolve()

        return None

    def _get_geomsubset_bindings(
        self, mesh_prim: Usd.Prim
    ) -> Dict[str, Tuple[str, List[int]]]:
        bindings = {}

        for child in mesh_prim.GetChildren():
            if child.GetTypeName() == "GeomSubset":
                subset_name = child.GetName()

                # Get face indices for this subset
                indices_attr = child.GetAttribute("indices")
                face_indices = (
                    list(indices_attr.Get())
                    if indices_attr and indices_attr.Get()
                    else []
                )

                # Get material binding
                mat_path = None
                binding_rel = child.GetRelationship("material:binding")
                if binding_rel:
                    targets = binding_rel.GetTargets()
                    if targets:
                        mat_path = str(targets[0])

                if mat_path:
                    bindings[subset_name] = (mat_path, face_indices)

        return bindings

    def extract(self) -> bool:
        try:
            self.stage = Usd.Stage.Open(str(self.usd_path))
        except Exception as e:
            print(f"ERROR: Could not open USD: {self.usd_path}")
            print(f"  {e}")
            return False

        if not self.stage:
            return False

        self._log(f"\n=== Extracting from: {self.usd_path.name} ===")

        # Step 1: Find all materials and extract textures
        self._log("\n--- Materials ---")
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Material":
                mat_path = str(prim.GetPath())
                self._log(f"\nMaterial: {prim.GetName()}")

                textures = {}

                # Try extracting from shader inputs
                for child in prim.GetChildren():
                    if child.GetTypeName() == "Shader":
                        shader = UsdShade.Shader(child)
                        textures.update(self._extract_textures_from_shader(shader))

                # If no textures found, try MDL
                if not textures:
                    mdl_path = self._find_mdl_for_material(prim)
                    if mdl_path:
                        self._log(f"  Using MDL: {mdl_path.name}")
                        textures = self._extract_textures_from_mdl(mdl_path)

                # Fallback: if still no textures, search Textures folder for any available
                if not textures:
                    textures = self._find_fallback_textures(prim.GetName())

                self.materials[mat_path] = textures

        # Step 2: Find all meshes and their material bindings
        self._log("\n--- Meshes ---")
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                mesh_path = str(prim.GetPath())
                mesh_name = prim.GetName()

                # Get direct material binding first
                binding_api = UsdShade.MaterialBindingAPI(prim)
                bound_material = binding_api.ComputeBoundMaterial()[0]
                mat_path = str(bound_material.GetPath()) if bound_material else None

                # Check for GeomSubset bindings (per-face materials)
                geomsubset_bindings = self._get_geomsubset_bindings(prim)

                # If no direct binding but has GeomSubsets, use first one as default
                if not mat_path and geomsubset_bindings:
                    first_subset = list(geomsubset_bindings.values())[0]
                    mat_path = first_subset[0]

                self.mesh_materials[mesh_path] = mat_path

                # Get mesh geometry
                mesh = UsdGeom.Mesh(prim)
                points_local = mesh.GetPointsAttr().Get()
                face_counts = mesh.GetFaceVertexCountsAttr().Get()
                face_indices = mesh.GetFaceVertexIndicesAttr().Get()

                # Apply world transform to vertices
                xformable = UsdGeom.Xformable(prim)
                world_transform = xformable.ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )

                # Transform points to world space
                points = []
                if points_local:
                    for p in points_local:
                        # Apply 4x4 transform matrix to point
                        p_world = world_transform.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                        points.append(Gf.Vec3f(p_world[0], p_world[1], p_world[2]))
                else:
                    points = None

                # Get UVs and check interpolation
                uvs = None
                uv_interpolation = None
                uv_indices = None
                for primvar_name in ["st", "uvs", "uv", "UVMap", "texCoords"]:
                    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(primvar_name)
                    if primvar and primvar.Get():
                        uvs = primvar.Get()
                        uv_interpolation = primvar.GetInterpolation()
                        # For indexed primvars, get the indices
                        if primvar.IsIndexed():
                            uv_indices = primvar.GetIndices()
                        break

                self.meshes[mesh_path] = {
                    "name": mesh_name,
                    "material": mat_path,
                    "points": points,
                    "face_counts": face_counts,
                    "face_indices": face_indices,
                    "uvs": uvs,
                    "uv_interpolation": uv_interpolation,
                    "uv_indices": uv_indices,
                    "geomsubsets": geomsubset_bindings,  # Store GeomSubset bindings
                }

                has_tex = bool(self.materials.get(mat_path))
                if geomsubset_bindings:
                    has_tex = any(
                        self.materials.get(m) for m, _ in geomsubset_bindings.values()
                    )

                status = "✓" if has_tex else "○"

                if geomsubset_bindings:
                    self._log(
                        f"  {status} {mesh_name} (GeomSubsets: {len(geomsubset_bindings)})"
                    )
                    for subset_name, (sub_mat, _) in geomsubset_bindings.items():
                        sub_mat_name = sub_mat.split("/")[-1] if sub_mat else "none"
                        self._log(f"      {subset_name} -> {sub_mat_name}")
                else:
                    self._log(
                        f"  {status} {mesh_name} -> {mat_path or '(no material)'}"
                    )

        if not self.meshes:
            self._log("WARNING: No meshes found in USD file")
            return False

        has_valid_mesh = any(
            mesh_data.get("points") for mesh_data in self.meshes.values()
        )
        if not has_valid_mesh:
            self._log("WARNING: No meshes with valid geometry found")
            return False

        return True

    def export_obj(
        self, output_dir: Path, normalize: bool = True
    ) -> Tuple[Optional[Path], Optional[Path]]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        obj_path = output_dir / "model.obj"
        mtl_path = output_dir / "model.mtl"

        # Collect all vertices, faces, UVs
        all_vertices = []
        all_faces = []
        all_uvs = []
        face_materials = []
        vertex_offset = 0
        uv_offset = 0

        material_list = []  # List of unique materials used
        material_map = {}  # material_path -> index

        for mesh_path, mesh_data in self.meshes.items():
            if not mesh_data["points"]:
                continue

            points = mesh_data["points"]
            face_counts = mesh_data["face_counts"]
            face_indices = mesh_data["face_indices"]
            uvs = mesh_data["uvs"]
            uv_interpolation = mesh_data.get("uv_interpolation")
            uv_indices = mesh_data.get("uv_indices")
            mat_path = mesh_data["material"]
            geomsubsets = mesh_data.get("geomsubsets", {})

            # Add vertices
            for p in points:
                all_vertices.append((float(p[0]), float(p[1]), float(p[2])))

            # Add UVs - handle different interpolation modes
            mesh_uv_offset = len(all_uvs)
            if uvs:
                for uv in uvs:
                    all_uvs.append((float(uv[0]), float(uv[1])))

            # Track UV mapping for this mesh
            # For faceVarying, we need to map face-vertex index to UV index
            mesh_data["_uv_offset"] = mesh_uv_offset
            mesh_data["_has_uvs"] = uvs is not None and len(uvs) > 0

            # Build face-to-material mapping for GeomSubsets
            face_to_subset_mat = {}
            if geomsubsets:
                for subset_name, (
                    sub_mat_path,
                    sub_face_indices,
                ) in geomsubsets.items():
                    for face_idx in sub_face_indices:
                        face_to_subset_mat[face_idx] = sub_mat_path
                    # Ensure this material is in our list
                    if sub_mat_path and sub_mat_path not in material_map:
                        material_map[sub_mat_path] = len(material_list)
                        material_list.append(sub_mat_path)

            # Track default material
            if mat_path and mat_path not in material_map:
                material_map[mat_path] = len(material_list)
                material_list.append(mat_path)

            # Add faces with proper UV indexing
            idx = 0
            face_num = 0
            face_vertex_idx = 0  # Running index for faceVarying UVs

            for count in face_counts:
                # Determine material for this face
                if face_num in face_to_subset_mat:
                    face_mat = face_to_subset_mat[face_num]
                else:
                    face_mat = mat_path

                mat_idx = material_map.get(face_mat, 0) if face_mat else 0

                # Determine UV indices based on interpolation mode
                def get_uv_idx(local_vert_idx, fv_offset):
                    if not mesh_data["_has_uvs"]:
                        return None
                    vertex_idx = face_indices[local_vert_idx]

                    if uv_interpolation == "faceVarying":
                        if uv_indices is not None:
                            # Indexed faceVarying: indices are per face-vertex
                            return mesh_uv_offset + int(uv_indices[fv_offset])
                        else:
                            # Non-indexed faceVarying: sequential per face-vertex
                            return mesh_uv_offset + fv_offset
                    else:
                        # vertex interpolation
                        if uv_indices is not None:
                            # Indexed vertex: indices are per-vertex
                            return mesh_uv_offset + int(uv_indices[vertex_idx])
                        else:
                            # Non-indexed vertex: UV index matches vertex index
                            return mesh_uv_offset + vertex_idx

                if count == 3:
                    v_indices = [
                        face_indices[idx] + vertex_offset,
                        face_indices[idx + 1] + vertex_offset,
                        face_indices[idx + 2] + vertex_offset,
                    ]
                    uv_idxs = [
                        get_uv_idx(idx, face_vertex_idx),
                        get_uv_idx(idx + 1, face_vertex_idx + 1),
                        get_uv_idx(idx + 2, face_vertex_idx + 2),
                    ]
                    all_faces.append((v_indices, uv_idxs))
                    face_materials.append(mat_idx)
                    face_vertex_idx += 3
                elif count == 4:
                    # Triangulate quad
                    v_indices1 = [
                        face_indices[idx] + vertex_offset,
                        face_indices[idx + 1] + vertex_offset,
                        face_indices[idx + 2] + vertex_offset,
                    ]
                    v_indices2 = [
                        face_indices[idx] + vertex_offset,
                        face_indices[idx + 2] + vertex_offset,
                        face_indices[idx + 3] + vertex_offset,
                    ]
                    uv_idxs1 = [
                        get_uv_idx(idx, face_vertex_idx),
                        get_uv_idx(idx + 1, face_vertex_idx + 1),
                        get_uv_idx(idx + 2, face_vertex_idx + 2),
                    ]
                    uv_idxs2 = [
                        get_uv_idx(idx, face_vertex_idx),
                        get_uv_idx(idx + 2, face_vertex_idx + 2),
                        get_uv_idx(idx + 3, face_vertex_idx + 3),
                    ]
                    all_faces.append((v_indices1, uv_idxs1))
                    all_faces.append((v_indices2, uv_idxs2))
                    face_materials.append(mat_idx)
                    face_materials.append(mat_idx)
                    face_vertex_idx += 4
                else:
                    # Skip n-gons
                    face_vertex_idx += count

                idx += count
                face_num += 1

            vertex_offset += len(points)

        if not all_vertices:
            return None, None

        # Normalize vertices to fit in [-0.5, 0.5]^3 centered at origin
        # Use margin factor to ensure object fits fully in camera frame at all angles
        MARGIN_FACTOR = 0.85  # Scale to 85% of unit cube to leave padding

        if normalize and all_vertices:
            # Compute bounding box
            xs = [v[0] for v in all_vertices]
            ys = [v[1] for v in all_vertices]
            zs = [v[2] for v in all_vertices]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)

            # Compute center and scale
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_z = (min_z + max_z) / 2

            extent_x = max_x - min_x
            extent_y = max_y - min_y
            extent_z = max_z - min_z
            max_extent = max(extent_x, extent_y, extent_z)

            if max_extent > 0:
                # Scale to fit in unit cube, with margin for camera framing
                scale = MARGIN_FACTOR / max_extent
            else:
                scale = 1.0

            # Apply normalization: center then scale
            all_vertices = [
                (
                    (v[0] - center_x) * scale,
                    (v[1] - center_y) * scale,
                    (v[2] - center_z) * scale,
                )
                for v in all_vertices
            ]

            self._log(f"\nNormalization applied:")
            self._log(
                f"  Original bounds: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}], Z[{min_z:.2f}, {max_z:.2f}]"
            )
            self._log(f"  Scale factor: {scale:.6f} (with {MARGIN_FACTOR:.0%} margin)")
            self._log(
                f"  Center offset: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})"
            )

        # Copy textures and write MTL
        with open(mtl_path, "w") as f:
            for mat_path in material_list:
                mat_name = mat_path.split("/")[-1] if mat_path else "default_material"
                textures = self.materials.get(mat_path, {})

                f.write(f"newmtl {mat_name}\n")
                f.write("Ka 0.2 0.2 0.2\n")

                # Use diffuse color constant if available, otherwise default gray
                if "diffuse_color" in textures:
                    color = textures["diffuse_color"]
                    f.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
                    self._log(
                        f"  Material {mat_name}: using diffuse color ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})"
                    )
                else:
                    f.write("Kd 0.8 0.8 0.8\n")

                f.write("Ks 0.2 0.2 0.2\n")
                f.write("Ns 50.0\n")
                f.write("d 1.0\n")
                f.write("illum 2\n")

                for tex_type, tex_value in textures.items():
                    # Skip color constants (they're tuples, not paths)
                    if isinstance(tex_value, tuple):
                        continue

                    tex_path = tex_value
                    if os.path.exists(tex_path):
                        # Copy texture to output dir
                        tex_name = os.path.basename(tex_path)
                        dest = output_dir / tex_name
                        if not dest.exists():
                            shutil.copy2(tex_path, dest)

                        # Write to MTL
                        if tex_type == "diffuse":
                            f.write(f"map_Kd {tex_name}\n")
                        elif tex_type == "normal":
                            f.write(f"map_Bump {tex_name}\n")
                        elif tex_type == "roughness":
                            f.write(f"map_Ns {tex_name}\n")
                        elif tex_type == "metallic":
                            f.write(f"map_Ks {tex_name}\n")

                f.write("\n")

        # Write OBJ
        with open(obj_path, "w") as f:
            f.write(f"mtllib model.mtl\n\n")

            for v in all_vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            f.write("\n")
            for uv in all_uvs:
                f.write(f"vt {uv[0]} {uv[1]}\n")

            f.write("\n")

            # Group faces by material
            mat_faces = defaultdict(list)
            for i, face_data in enumerate(all_faces):
                mat_faces[face_materials[i]].append(face_data)

            for mat_idx, faces in mat_faces.items():
                mat_path = (
                    material_list[mat_idx] if mat_idx < len(material_list) else None
                )
                mat_name = mat_path.split("/")[-1] if mat_path else "default_material"

                f.write(f"usemtl {mat_name}\n")
                for face_data in faces:
                    v_indices, uv_indices = face_data
                    # OBJ indices are 1-based
                    if uv_indices[0] is not None:
                        # Include UV indices
                        f.write(
                            f"f {v_indices[0]+1}/{uv_indices[0]+1} {v_indices[1]+1}/{uv_indices[1]+1} {v_indices[2]+1}/{uv_indices[2]+1}\n"
                        )
                    else:
                        # No UVs, just vertex indices
                        f.write(
                            f"f {v_indices[0]+1} {v_indices[1]+1} {v_indices[2]+1}\n"
                        )
                f.write("\n")

        return obj_path, mtl_path


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system("sudo apt-get update")
        os.system(
            "sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6"
        )
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(
            f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}"
        )


def render_usd(
    usd_path: str,
    output_dir: str,
    num_views: int = 150,
    resolution: int = 512,
    verbose: bool = False,
) -> bool:
    os.makedirs(output_dir, exist_ok=True)

    # Extract mesh and materials from USD
    extractor = USDMaterialExtractor(usd_path, verbose=verbose)
    if not extractor.extract():
        print(f"Failed to extract from USD: {usd_path}")
        return False

    # Export to OBJ + MTL
    temp_dir = tempfile.mkdtemp()
    try:
        # Don't normalize - let Blender's normalize_scene() handle it
        obj_path, mtl_path = extractor.export_obj(Path(temp_dir), normalize=False)
        if not obj_path:
            print(f"Failed to export OBJ from USD: {usd_path}")
            return False

        if verbose:
            print(f"\nExported to: {obj_path}")
            # List textures
            textures = list(Path(temp_dir).glob("*.png")) + list(
                Path(temp_dir).glob("*.tga")
            )
            if textures:
                print(f"Textures copied: {len(textures)}")
                for t in textures:
                    print(f"  - {t.name}")

        # Generate camera views
        yaws = []
        pitchs = []
        offset = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset)
            yaws.append(y)
            pitchs.append(p)
        # Radius 2.5 ensures object corners fit in frame:
        # Object diagonal at 0.866 from center, visible range at radius=2.5, FOV=40° is ±0.91
        radius = [2.1] * num_views
        fov = [40 / 180 * np.pi] * num_views
        views = [
            {"yaw": y, "pitch": p, "radius": r, "fov": f}
            for y, p, r, f in zip(yaws, pitchs, radius, fov)
        ]

        # Call Blender
        blender_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "dataset_toolkits",
            "blender_script",
            "render.py",
        )

        args = [
            BLENDER_PATH,
            "-b",
            "-P",
            blender_script,
            "--",
            "--views",
            json.dumps(views),
            "--object",
            str(obj_path),
            "--resolution",
            str(resolution),
            "--output_folder",
            output_dir,
            "--engine",
            "CYCLES",
            "--save_mesh",
            "--use_gpu",  # Enable GPU acceleration
        ]

        if verbose:
            print(f"\nRunning Blender...")

        call(
            args,
            stdout=DEVNULL if not verbose else None,
            stderr=DEVNULL if not verbose else None,
        )

        success = os.path.exists(os.path.join(output_dir, "transforms.json"))
        return success

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _render_worker(
    file_path: str,
    sha256: str,
    dataset: str,
    output_dir: str,
    num_views: int,
    quiet: bool,
) -> Optional[Dict]:
    output_folder = os.path.join(output_dir, "renders", sha256)

    # Skip if already rendered
    if os.path.exists(os.path.join(output_folder, "transforms.json")):
        return {"sha256": sha256, "rendered": True}

    success = render_usd(
        file_path,
        output_folder,
        num_views=num_views,
        resolution=512,
        verbose=not quiet,
    )

    if success:
        return {"sha256": sha256, "rendered": True}
    else:
        if not quiet:
            print(f"Failed to render: {file_path}")
        return None


def main_batch():
    import importlib
    import copy
    from functools import partial
    from easydict import EasyDict as edict

    # First argument is dataset type (e.g., "allmats")
    dataset_utils = importlib.import_module(f"dataset_toolkits.datasets.{sys.argv[1]}")

    parser = argparse.ArgumentParser(
        description="Batch render USD files with proper texture extraction"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save renders"
    )
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Instances to process (comma-separated or file path)",
    )
    parser.add_argument(
        "--num_views", type=int, default=150, help="Number of views to render"
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Worker rank for distributed processing"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total workers for distributed processing",
    )
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    # Add dataset-specific args
    dataset_utils.add_args(parser)

    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, "renders"), exist_ok=True)

    # Install blender
    if not opt.quiet:
        print("Checking blender...", flush=True)
    _install_blender()

    # Get file list from metadata
    metadata_path = os.path.join(opt.output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise ValueError(f"metadata.csv not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    if opt.instances is None:
        metadata = metadata[metadata["local_path"].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[
                metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score
            ]
        if "rendered" in metadata.columns:
            metadata = metadata[metadata["rendered"] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    # Distributed processing slice
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # Filter already processed
    for sha256 in copy.copy(metadata["sha256"].values):
        if os.path.exists(
            os.path.join(opt.output_dir, "renders", sha256, "transforms.json")
        ):
            records.append({"sha256": sha256, "rendered": True})
            metadata = metadata[metadata["sha256"] != sha256]

    print(f"Processing {len(metadata)} objects (rank {opt.rank}/{opt.world_size})...")

    # Process objects
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    results = []
    with ThreadPoolExecutor(max_workers=opt.max_workers) as executor:
        futures = []
        for _, row in metadata.iterrows():
            sha256 = row["sha256"]
            local_path = row["local_path"]
            dataset = row.get("dataset", "unknown")

            futures.append(
                executor.submit(
                    _render_worker,
                    local_path,
                    sha256,
                    dataset,
                    opt.output_dir,
                    opt.num_views,
                    opt.quiet,
                )
            )

        for future in tqdm(futures, desc="Rendering", disable=opt.quiet):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                if not opt.quiet:
                    print(f"Error in worker: {e}")

    # Save results
    rendered = pd.concat(
        [pd.DataFrame.from_records(results), pd.DataFrame.from_records(records)]
    )
    rendered.to_csv(
        os.path.join(opt.output_dir, f"rendered_{opt.rank}.csv"), index=False
    )

    print(f"Done! Rendered {len(results)} objects.")


def main_single():
    parser = argparse.ArgumentParser(
        description="Render a single USD file with proper texture extraction"
    )
    parser.add_argument("usd_file", help="Path to USD file")
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        help="Output directory (default: /tmp/render_<filename>)",
    )
    parser.add_argument(
        "--num_views", type=int, default=150, help="Number of views to render"
    )
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed logs"
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only extract materials (don't render)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.usd_file):
        print(f"ERROR: USD file not found: {args.usd_file}")
        sys.exit(1)

    if args.extract_only:
        extractor = USDMaterialExtractor(args.usd_file, verbose=True)
        extractor.extract()

        print("\n=== SUMMARY ===")
        print(f"Meshes: {len(extractor.meshes)}")
        print(f"Materials: {len(extractor.materials)}")

        total_textures = sum(len(t) for t in extractor.materials.values())
        print(f"Total textures: {total_textures}")

        for mat_path, textures in extractor.materials.items():
            if textures:
                mat_name = mat_path.split("/")[-1] if mat_path else "unknown"
                print(f"\n{mat_name}:")
                for tex_type, tex_path in textures.items():
                    print(f"  {tex_type}: {os.path.basename(tex_path)}")
    else:
        _install_blender()

        output_dir = args.output_dir
        if not output_dir:
            filename = Path(args.usd_file).stem
            output_dir = f"/tmp/render_{filename}"

        success = render_usd(
            args.usd_file,
            output_dir,
            num_views=args.num_views,
            resolution=args.resolution,
            verbose=args.verbose,
        )

        if success:
            print(f"\n✓ Rendered to: {output_dir}")
        else:
            print(f"\n✗ Rendering failed")
            sys.exit(1)


if __name__ == "__main__":
    # Check if first arg is a dataset type (batch mode) or a file (single mode)
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and not os.path.exists(sys.argv[1])
    ):
        # Batch mode: first arg is dataset type like "allmats"
        main_batch()
    else:
        # Single file mode
        main_single()
