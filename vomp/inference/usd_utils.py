"""
USD utilities for SimReady format USD files.
"""

import os
import glob
import numpy as np
import shutil
import tempfile
import traceback
from typing import Tuple, List, Dict, Optional
from pxr import Usd, UsdGeom, UsdShade


def find_textures_in_directory(model_dir: str) -> Dict[str, str]:
    """Find texture files in the model directory and subdirectories."""
    textures = {}
    if not os.path.exists(model_dir):
        return textures

    prop_name = os.path.basename(model_dir)
    parent_dir = os.path.dirname(model_dir)

    texture_dirs = [
        model_dir,
        os.path.join(model_dir, "textures"),
        os.path.join(model_dir, "Textures"),
        os.path.join(parent_dir, "textures"),
        os.path.join(parent_dir, "Textures"),
    ]

    texture_files = []
    for texture_dir in texture_dirs:
        if not os.path.exists(texture_dir):
            continue
        texture_files.extend(
            glob.glob(os.path.join(texture_dir, "**/*.png"), recursive=True)
        )
        texture_files.extend(
            glob.glob(os.path.join(texture_dir, "**/*.jpg"), recursive=True)
        )

    texture_candidates = {
        "diffuse": [],
        "normal": [],
        "roughness": [],
        "metallic": [],
        "specular": [],
        "bump": [],
    }

    for texture_path in texture_files:
        filename = os.path.basename(texture_path).lower()

        if any(term in filename for term in ["diffuse", "color", "albedo", "base"]):
            texture_candidates["diffuse"].append(texture_path)
        if any(term in filename for term in ["normal", "nrm"]):
            texture_candidates["normal"].append(texture_path)
        if any(term in filename for term in ["rough", "roughness"]):
            texture_candidates["roughness"].append(texture_path)
        if any(term in filename for term in ["metal", "metallic"]):
            texture_candidates["metallic"].append(texture_path)
        if any(term in filename for term in ["spec", "specular"]):
            texture_candidates["specular"].append(texture_path)
        if any(term in filename for term in ["bump", "height"]):
            texture_candidates["bump"].append(texture_path)

    for tex_type, candidates in texture_candidates.items():
        if candidates:
            textures[tex_type] = candidates[0]

    return textures


def extract_material_info(mesh_prim, stage) -> Dict:
    """Extract material information from a USD mesh prim."""
    material_info = {
        "name": f"material_{mesh_prim.GetName()}",
        "diffuse_color": [0.8, 0.8, 0.8],
        "specular_color": [0.2, 0.2, 0.2],
        "roughness": 0.5,
        "metallic": 0.0,
        "textures": {},
    }

    model_dir = os.path.dirname(stage.GetRootLayer().realPath)
    material_info["textures"] = find_textures_in_directory(model_dir)

    return material_info


def load_mesh_from_usd(usd_file_path: str, temp_dir: Optional[str] = None):
    """
    Load and merge all meshes from a USD file with texture information.
    """
    if not os.path.exists(usd_file_path):
        print(f"Error: USD file not found at {usd_file_path}")
        return None, None, None, None, None

    try:
        stage = Usd.Stage.Open(usd_file_path)
        if not stage:
            print(f"Error: Failed to open USD stage from {usd_file_path}")
            return None, None, None, None, None

        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            default_prim = stage.GetPrimAtPath("/")

        mesh_prims = []
        for prim in Usd.PrimRange(default_prim):
            if prim.GetTypeName() == "Mesh":
                mesh_prims.append(prim)

        if not mesh_prims:
            print(f"Error: No meshes found in {usd_file_path}")
            return None, None, None, None, None

        all_vertices = []
        all_faces = []
        all_materials = []
        all_face_materials = []
        all_uvs = []  # Flat list
        vertex_offset = 0
        uv_offset = 0  # Only used when has_uvs=True

        for mesh_prim in mesh_prims:
            mesh = UsdGeom.Mesh(mesh_prim)

            points_attr = mesh.GetPointsAttr()
            if not points_attr or not points_attr.Get():
                continue

            verts = points_attr.Get()
            vertices = np.array([(v[0], v[1], v[2]) for v in verts], dtype=np.float32)

            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

            if not face_vertex_counts or not face_vertex_indices:
                continue

            has_uvs = False
            mesh_uvs = []

            for primvar in [
                UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st"),
                UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("uvs"),
                UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("uv"),
                UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("UVMap"),
                UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("texCoords"),
            ]:
                if primvar and primvar.Get():
                    uv_data = primvar.Get()
                    interp = primvar.GetInterpolation()

                    if interp == UsdGeom.Tokens.vertex:
                        mesh_uvs = np.array(
                            [(uv[0], uv[1]) for uv in uv_data], dtype=np.float32
                        )
                        has_uvs = True
                        break
                    elif interp == UsdGeom.Tokens.faceVarying:
                        mesh_uvs = np.array(
                            [(uv[0], uv[1]) for uv in uv_data], dtype=np.float32
                        )
                        has_uvs = True
                        break

            if not has_uvs:
                mesh_uvs = np.zeros((len(vertices), 2), dtype=np.float32)

            # Extract material
            try:
                material_info = extract_material_info(mesh_prim, stage)
                all_materials.append(material_info)
                material_index = len(all_materials) - 1
            except Exception as e:
                print(f"Warning: Error extracting material: {e}")
                material_info = {
                    "name": f"material_{mesh_prim.GetName()}",
                    "diffuse_color": [0.8, 0.8, 0.8],
                    "specular_color": [0.2, 0.2, 0.2],
                    "roughness": 0.5,
                    "metallic": 0.0,
                    "textures": {},
                }
                all_materials.append(material_info)
                material_index = len(all_materials) - 1

            faces = []
            face_materials = []

            idx = 0
            for count in face_vertex_counts:
                if count == 3:
                    face = [
                        face_vertex_indices[idx] + vertex_offset,
                        face_vertex_indices[idx + 1] + vertex_offset,
                        face_vertex_indices[idx + 2] + vertex_offset,
                    ]
                    faces.append(face)
                    face_materials.append(material_index)

                elif count == 4:
                    face1 = [
                        face_vertex_indices[idx] + vertex_offset,
                        face_vertex_indices[idx + 1] + vertex_offset,
                        face_vertex_indices[idx + 2] + vertex_offset,
                    ]
                    face2 = [
                        face_vertex_indices[idx] + vertex_offset,
                        face_vertex_indices[idx + 2] + vertex_offset,
                        face_vertex_indices[idx + 3] + vertex_offset,
                    ]
                    faces.append(face1)
                    faces.append(face2)
                    face_materials.append(material_index)
                    face_materials.append(material_index)

                idx += count

            all_vertices.append(vertices)
            all_faces.extend(faces)
            all_face_materials.extend(face_materials)
            all_uvs.extend(mesh_uvs)  # EXTEND not append!

            vertex_offset += len(vertices)
            if has_uvs:
                uv_offset += len(mesh_uvs)  # Only increment when has_uvs

        if not all_vertices or not all_faces:
            print(f"Error: No valid meshes found in {usd_file_path}")
            return None, None, None, None, None

        merged_vertices = np.vstack(all_vertices)
        merged_faces = np.array(all_faces, dtype=np.int32)

        # Copy textures
        if temp_dir:
            for material in all_materials:
                new_textures = {}
                for tex_type, tex_path in material["textures"].items():
                    if os.path.exists(tex_path):
                        tex_name = os.path.basename(tex_path)
                        dest_path = os.path.join(temp_dir, tex_name)
                        shutil.copy2(tex_path, dest_path)
                        new_textures[tex_type] = dest_path
                material["textures"] = new_textures

        # Return flat UV list (not nested)
        return merged_vertices, merged_faces, all_materials, all_face_materials, all_uvs

    except Exception as e:
        print(f"Error loading USD file: {str(e)}")

        traceback.print_exc()
        return None, None, None, None, None


def save_obj_with_materials(
    vertices, faces, materials, face_materials, obj_file_path, uvs=None
):
    """
    Save vertices, faces and materials as OBJ and MTL files.
    """
    mtl_file_path = obj_file_path.replace(".obj", ".mtl")
    mtl_filename = os.path.basename(mtl_file_path)

    # Write MTL file
    with open(mtl_file_path, "w") as f:
        for i, material in enumerate(materials):
            f.write(f"newmtl {material['name']}\n")
            f.write(f"Ns 90.0\n")
            f.write(f"Ka 1.0 1.0 1.0\n")

            diffuse = material["diffuse_color"]
            f.write(f"Kd {diffuse[0]} {diffuse[1]} {diffuse[2]}\n")

            specular = material["specular_color"]
            f.write(f"Ks {specular[0]} {specular[1]} {specular[2]}\n")

            f.write(f"Ns {(1.0 - material['roughness']) * 100.0}\n")
            f.write(f"d 1.0\n")
            f.write(f"illum 2\n")

            for tex_type, tex_path in material["textures"].items():
                if not os.path.exists(tex_path):
                    continue

                tex_rel_path = os.path.basename(tex_path)
                if tex_type == "diffuse":
                    f.write(f"map_Kd {tex_rel_path}\n")
                elif tex_type == "normal":
                    f.write(f"map_Bump {tex_rel_path}\n")
                elif tex_type == "roughness":
                    f.write(f"map_Ns {tex_rel_path}\n")
                elif tex_type == "metallic" or tex_type == "specular":
                    f.write(f"map_Ks {tex_rel_path}\n")
                elif tex_type == "bump":
                    f.write(f"map_bump {tex_rel_path}\n")

            f.write("\n")

    # Write OBJ file
    with open(obj_file_path, "w") as f:
        f.write(f"mtllib {mtl_filename}\n\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        f.write("\n")

        # Write UVs
        have_uvs = uvs is not None and len(uvs) > 0
        if have_uvs:
            for uv in uvs:
                f.write(f"vt {uv[0]} {uv[1]}\n")
        else:
            for _ in range(len(vertices)):
                f.write(f"vt 0.0 0.0\n")

        f.write("\n")

        # Write normals
        for _ in range(len(vertices)):
            f.write(f"vn 0.0 0.0 1.0\n")

        f.write("\n")

        # Group faces by material
        material_to_faces = {}
        for i, face in enumerate(faces):
            mat_idx = face_materials[i]
            if mat_idx not in material_to_faces:
                material_to_faces[mat_idx] = []
            material_to_faces[mat_idx].append((i, face))

        # Write faces
        for mat_idx, mat_faces in material_to_faces.items():
            if mat_idx < len(materials):
                material_name = materials[mat_idx]["name"]
                f.write(f"usemtl {material_name}\n")

                for face_idx, face in mat_faces:
                    f.write(
                        f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n"
                    )


def convert_usd_to_obj(
    usd_path: str,
    output_obj_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Convert a USD file to OBJ format with materials.

    Args:
        usd_path: Path to input USD file
        output_obj_path: Optional path for output OBJ (if None, creates temp file)

    Returns:
        Tuple of (obj_path, temp_dir)
        temp_dir is the temporary directory created (or None if output_obj_path was specified)
    """
    # Create temp directory for OBJ and textures
    if output_obj_path:
        temp_dir = None
        obj_path = output_obj_path
        texture_dir = os.path.dirname(obj_path)
    else:
        temp_dir = tempfile.mkdtemp(prefix="vomp_usd_")
        obj_path = os.path.join(temp_dir, "model.obj")
        texture_dir = temp_dir

    # Load USD and convert
    vertices, faces, materials, face_materials, uvs = load_mesh_from_usd(
        usd_path, temp_dir=texture_dir
    )

    if vertices is None:
        raise ValueError(f"Failed to load USD file: {usd_path}")

    # Save as OBJ with materials
    save_obj_with_materials(
        vertices, faces, materials, face_materials, obj_path, uvs=uvs
    )

    return obj_path, temp_dir
