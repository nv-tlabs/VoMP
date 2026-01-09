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
import os
import pathlib
from multiprocessing import Pool, cpu_count
from uipc import view, Vector3, Transform, AngleAxis, builtin
from uipc.geometry import (
    SimplicialComplexIO,
    label_surface,
    ground,
    tetrahedralize,
    tetmesh,
)
from uipc.constitution import StableNeoHookean, ElasticModuli
from uipc.unit import kPa, GPa
from scipy.spatial import cKDTree


def compute_barycentric_coordinates(point, tet_vertices):
    """
    Compute barycentric coordinates of a point with respect to a tetrahedron.
    Returns coordinates [lambda1, lambda2, lambda3, lambda4] where the point = sum(lambda_i * v_i)
    For points inside the tetrahedron, all lambdas are in [0, 1] and sum to 1.
    For points outside, some lambdas may be negative or >1, but they still sum to 1.
    """
    v1, v2, v3, v4 = tet_vertices

    # Set up linear system: point - v4 = A * [lambda1, lambda2, lambda3]^T
    # where A = [v1-v4, v2-v4, v3-v4]
    A = np.column_stack([v1 - v4, v2 - v4, v3 - v4])
    b = point - v4

    try:
        det_A = np.linalg.det(A)

        # Check for degenerate tetrahedra (volume too small)
        # Volume of tetrahedron = |det(A)| / 6
        if abs(det_A) < 1e-15:
            # Degenerate tetrahedron - return centroid coordinates
            return np.array([0.25, 0.25, 0.25, 0.25])

        # Check condition number to avoid numerical instability
        cond_A = np.linalg.cond(A)
        if cond_A > 1e10:
            # Poorly conditioned - return centroid coordinates
            return np.array([0.25, 0.25, 0.25, 0.25])

        # Solve for barycentric coordinates
        lambdas_123 = np.linalg.solve(A, b)
        lambda4 = 1.0 - np.sum(lambdas_123)

        coords = np.array([lambdas_123[0], lambdas_123[1], lambdas_123[2], lambda4])

        # Ensure coordinates are finite
        if not np.all(np.isfinite(coords)):
            return np.array([0.25, 0.25, 0.25, 0.25])

        # For points significantly outside the tetrahedron, clamp to reasonable range
        # This prevents extreme extrapolation that can cause visual artifacts
        max_abs_coord = np.max(np.abs(coords))
        if max_abs_coord > 3.0:
            # Scale down to reasonable range while maintaining relative proportions
            coords = coords * (3.0 / max_abs_coord)
            # Renormalize to ensure sum = 1
            coords = coords / np.sum(coords)

        return coords

    except np.linalg.LinAlgError:
        # Singular matrix or other linear algebra error
        return np.array([0.25, 0.25, 0.25, 0.25])


def point_in_tetrahedron(point, tet_vertices, tolerance=1e-8):
    """
    Check if a point is inside a tetrahedron using barycentric coordinates.
    A point is inside if all barycentric coordinates are non-negative (within tolerance).

    Args:
        point: 3D point coordinates
        tet_vertices: Array of 4 vertices defining the tetrahedron
        tolerance: Small negative value allowed for numerical precision (default 1e-8)

    Returns:
        (is_inside, bary_coords): Boolean indicating if inside, and the barycentric coordinates
    """
    bary_coords = compute_barycentric_coordinates(point, tet_vertices)

    # Point is inside if all coordinates are >= -tolerance
    # We use a small tolerance to account for numerical precision
    is_inside = np.all(bary_coords >= -tolerance) and np.all(
        bary_coords <= 1.0 + tolerance
    )

    return is_inside, bary_coords


def _process_vertex_batch(args):
    """
    Worker function for multiprocessing
    """
    vertices_batch, tet_vertices_list, tet_centroids, start_idx = args

    # Build KDTree for this worker (each process needs its own)
    tet_tree = cKDTree(tet_centroids)

    batch_embeddings = []
    batch_stats = {"inside_count": 0, "outside_count": 0, "distances": []}

    for i, visual_vertex in enumerate(vertices_batch):
        tet_idx, bary_coords, distance = find_containing_tetrahedron_fast_optimized(
            visual_vertex,
            tet_vertices_list,
            tet_tree,
            fallback_closest=True,
            max_check=500,
        )

        if tet_idx >= 0:
            batch_embeddings.append((tet_idx, bary_coords))

            if distance == 0.0:
                batch_stats["inside_count"] += 1
            else:
                batch_stats["outside_count"] += 1
                batch_stats["distances"].append(distance)
        else:
            # Fallback to closest centroid method
            centroid_distances = np.linalg.norm(tet_centroids - visual_vertex, axis=1)
            closest_tet_idx = np.argmin(centroid_distances)

            bary_coords = compute_barycentric_coordinates(
                visual_vertex, tet_vertices_list[closest_tet_idx]
            )
            batch_embeddings.append((closest_tet_idx, bary_coords))

            batch_stats["outside_count"] += 1
            batch_stats["distances"].append(centroid_distances[closest_tet_idx])

    return start_idx, batch_embeddings, batch_stats


def find_containing_tetrahedron_fast_optimized(
    point, tet_vertices_list, tet_tree, fallback_closest=True, max_check=500
):
    """
    Optimized version using KDTree for fast spatial queries
    """
    # Query the KDTree for nearest tetrahedron centroids
    distances, indices = tet_tree.query(point, k=min(max_check, len(tet_vertices_list)))

    # Handle single point case
    if np.isscalar(distances):
        distances = [distances]
        indices = [indices]

    best_tet_idx = -1
    best_bary_coords = None
    min_negative_weight = float("inf")

    for dist, tet_idx in zip(distances, indices):
        tet_vertices = tet_vertices_list[tet_idx]
        is_inside, bary_coords = point_in_tetrahedron(point, tet_vertices)

        if is_inside:
            return tet_idx, bary_coords, 0.0

        if fallback_closest:
            max_negative = np.min(bary_coords)
            if max_negative > min_negative_weight:
                min_negative_weight = max_negative
                best_tet_idx = tet_idx
                best_bary_coords = bary_coords

    if fallback_closest and best_tet_idx >= 0:
        return best_tet_idx, best_bary_coords, abs(min_negative_weight)

    return -1, None, float("inf")


def find_containing_tetrahedron_fast(
    point, tet_vertices_list, tet_centroids, fallback_closest=True, max_check=500
):

    distances_to_centroids = np.array(
        [np.linalg.norm(point - centroid) for centroid in tet_centroids]
    )
    sorted_indices = np.argsort(distances_to_centroids)

    best_tet_idx = -1
    best_bary_coords = None
    min_negative_weight = float("inf")

    num_to_check = min(max_check, len(sorted_indices))

    for i in range(num_to_check):
        tet_idx = sorted_indices[i]
        tet_vertices = tet_vertices_list[tet_idx]
        is_inside, bary_coords = point_in_tetrahedron(point, tet_vertices)

        if is_inside:

            return tet_idx, bary_coords, 0.0

        if fallback_closest:

            max_negative = np.min(bary_coords)
            if max_negative > min_negative_weight:
                min_negative_weight = max_negative
                best_tet_idx = tet_idx
                best_bary_coords = bary_coords

    if fallback_closest and best_tet_idx >= 0:
        return best_tet_idx, best_bary_coords, abs(min_negative_weight)

    return -1, None, float("inf")


def find_containing_tetrahedron(point, tet_vertices_list, fallback_closest=True):
    best_tet_idx = -1
    best_bary_coords = None
    min_negative_weight = float("inf")

    for tet_idx, tet_vertices in enumerate(tet_vertices_list):
        is_inside, bary_coords = point_in_tetrahedron(point, tet_vertices)

        if is_inside:

            return tet_idx, bary_coords, 0.0

        if fallback_closest:

            max_negative = np.min(bary_coords)
            if max_negative > min_negative_weight:
                min_negative_weight = max_negative
                best_tet_idx = tet_idx
                best_bary_coords = bary_coords

    if fallback_closest and best_tet_idx >= 0:
        return best_tet_idx, best_bary_coords, abs(min_negative_weight)

    return -1, None, float("inf")


def embed_visual_mesh_in_physics_tets(
    visual_vertices,
    physics_mesh,
    use_multiprocessing=True,
    n_processes=None,
    use_original_method=False,
):
    print(f"Embedding {len(visual_vertices)} visual vertices in physics tetrahedra...")

    physics_positions = physics_mesh.positions().view()
    physics_tets = physics_mesh.tetrahedra()
    tet_connectivity = view(physics_tets.topo())

    physics_pos_array = []
    for i in range(len(physics_positions)):
        pos = physics_positions[i]
        if hasattr(pos, "tolist"):
            pos_list = pos.tolist()
            if isinstance(pos_list[0], list):
                physics_pos_array.append(
                    [pos_list[0][0], pos_list[1][0], pos_list[2][0]]
                )
            else:
                physics_pos_array.append(pos_list)
        else:
            physics_pos_array.append([float(pos[0]), float(pos[1]), float(pos[2])])

    physics_pos_array = np.array(physics_pos_array, dtype=np.float32)

    tet_vertices_list = []
    tet_centroids = []

    for i in range(physics_tets.size()):
        tet_verts = tet_connectivity[i]

        tet_positions = physics_pos_array[tet_verts]

        if tet_positions.shape != (4, 3):
            tet_positions = tet_positions.reshape(4, 3)

        tet_vertices_list.append(tet_positions)

        centroid = np.mean(tet_positions, axis=0)
        tet_centroids.append(centroid)

    print(
        f"Physics mesh has {len(physics_pos_array)} vertices and {len(tet_vertices_list)} tetrahedra"
    )

    tet_centroids = np.array(tet_centroids)

    embeddings = []
    stats = {
        "inside_count": 0,
        "outside_count": 0,
        "max_distance": 0.0,
        "avg_distance": 0.0,
    }
    distances = []

    if use_original_method:
        # Use the exact original method for comparison
        print("Using original method for maximum accuracy...")
        for i, visual_vertex in enumerate(visual_vertices):
            tet_idx, bary_coords, distance = find_containing_tetrahedron_fast(
                visual_vertex, tet_vertices_list, tet_centroids, fallback_closest=True
            )

            if tet_idx >= 0:
                embeddings.append((tet_idx, bary_coords))

                if distance == 0.0:
                    stats["inside_count"] += 1
                else:
                    stats["outside_count"] += 1
                    distances.append(distance)
            else:
                # Fallback to closest centroid method
                centroid_distances = [
                    np.linalg.norm(visual_vertex - centroid)
                    for centroid in tet_centroids
                ]
                closest_tet_idx = np.argmin(centroid_distances)

                bary_coords = compute_barycentric_coordinates(
                    visual_vertex, tet_vertices_list[closest_tet_idx]
                )
                embeddings.append((closest_tet_idx, bary_coords))

                stats["outside_count"] += 1
                distances.append(centroid_distances[closest_tet_idx])

            if (i + 1) % 1000 == 0:
                print(f"Embedded {i + 1}/{len(visual_vertices)} vertices...")

    elif use_multiprocessing and len(visual_vertices) > 1000:
        # Use multiprocessing for large datasets
        if n_processes is None:
            n_processes = min(cpu_count(), 16)  # Cap at 8 to avoid memory issues

        print(f"Using multiprocessing with {n_processes} processes...")

        # Split vertices into batches for each process
        batch_size = max(
            100, len(visual_vertices) // (n_processes * 4)
        )  # 4 batches per process
        batches = []

        for i in range(0, len(visual_vertices), batch_size):
            end_idx = min(i + batch_size, len(visual_vertices))
            batch = visual_vertices[i:end_idx]
            batches.append((batch, tet_vertices_list, tet_centroids, i))

        print(
            f"Processing {len(visual_vertices)} vertices in {len(batches)} batches..."
        )

        # Process batches in parallel
        with Pool(n_processes) as pool:
            results = pool.map(_process_vertex_batch, batches)

        # Combine results
        embeddings = [None] * len(visual_vertices)
        for start_idx, batch_embeddings, batch_stats in results:
            for i, embedding in enumerate(batch_embeddings):
                embeddings[start_idx + i] = embedding

            stats["inside_count"] += batch_stats["inside_count"]
            stats["outside_count"] += batch_stats["outside_count"]
            distances.extend(batch_stats["distances"])

        print("Multiprocessing completed!")

    else:
        # Use single-threaded processing for smaller datasets or when disabled
        print("Building spatial index for fast tetrahedron lookup...")
        tet_tree = cKDTree(tet_centroids)
        print("Spatial index built successfully!")

        # Process in batches for better memory management
        batch_size = 1000
        print(
            f"Processing {len(visual_vertices)} vertices in batches of {batch_size}..."
        )

        for batch_start in range(0, len(visual_vertices), batch_size):
            batch_end = min(batch_start + batch_size, len(visual_vertices))
            batch_vertices = visual_vertices[batch_start:batch_end]

            for i, visual_vertex in enumerate(batch_vertices):
                global_i = batch_start + i

                tet_idx, bary_coords, distance = (
                    find_containing_tetrahedron_fast_optimized(
                        visual_vertex,
                        tet_vertices_list,
                        tet_tree,
                        fallback_closest=True,
                        max_check=500,
                    )
                )

                if tet_idx >= 0:
                    embeddings.append((tet_idx, bary_coords))

                    if distance == 0.0:
                        stats["inside_count"] += 1
                    else:
                        stats["outside_count"] += 1
                        distances.append(distance)
                else:
                    # Fallback to closest centroid method
                    centroid_distances = np.linalg.norm(
                        tet_centroids - visual_vertex, axis=1
                    )
                    closest_tet_idx = np.argmin(centroid_distances)

                    bary_coords = compute_barycentric_coordinates(
                        visual_vertex, tet_vertices_list[closest_tet_idx]
                    )
                    embeddings.append((closest_tet_idx, bary_coords))

                    stats["outside_count"] += 1
                    distances.append(centroid_distances[closest_tet_idx])

                if (global_i + 1) % 1000 == 0:
                    print(f"Embedded {global_i + 1}/{len(visual_vertices)} vertices...")

    if distances:
        stats["max_distance"] = np.max(distances)
        stats["avg_distance"] = np.mean(distances)

    print(f"Embedding completed:")
    print(f"  Inside tetrahedra: {stats['inside_count']}")
    print(f"  Outside tetrahedra: {stats['outside_count']}")
    print(f"  Max distance to boundary: {stats['max_distance']:.6f}")
    print(f"  Average distance to boundary: {stats['avg_distance']:.6f}")

    return embeddings, stats


def load_visual_mesh(
    file_path,
    scale=1.0,
    pre_transform=None,
    normalize_like_blender=True,
    material_file=None,
):
    if pre_transform is None:
        pre_transform = Transform.Identity()
        pre_transform.scale(scale)

    # Extract scale from pre_transform to apply after normalization
    if pre_transform is not None:
        # Extract scale component from transform matrix more robustly
        transform_matrix = pre_transform.matrix()
        # Calculate scale as the length of the first column vector (assumes uniform scaling)
        scale_vector = np.array(
            [transform_matrix[0][0], transform_matrix[1][0], transform_matrix[2][0]]
        )
        transform_scale = float(np.linalg.norm(scale_vector))
    else:
        transform_scale = scale

    if file_path.lower().endswith(".obj") and normalize_like_blender:
        print(f"Loading OBJ with normalization: {file_path}")

        # Load OBJ manually to preserve vertex-UV mapping and material assignments
        vertices = []
        uv_coords = []
        faces = []
        face_uvs = []  # UV indices for each face vertex
        face_materials = []  # Material name for each face
        current_material = None
        mtl_lib = None

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("mtllib "):
                    # Material library reference
                    mtl_lib = line.split()[1]
                elif line.startswith("usemtl "):
                    # Material usage - affects subsequent faces
                    current_material = line.split()[1]
                elif line.startswith("v "):
                    # Vertex: v x y z
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vt "):
                    # UV coordinate: vt u v [w]
                    parts = line.split()
                    uv_coords.append(
                        [float(parts[1]), float(parts[2])]
                    )  # Only use u,v (ignore w)
                elif line.startswith("f "):
                    # Face: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 [v4/vt4/vn4]
                    parts = line.split()[1:]  # Skip 'f'
                    face_vertices = []
                    face_uv_indices = []

                    for part in parts:
                        indices = part.split("/")
                        # OBJ indices are 1-based, convert to 0-based
                        vertex_idx = int(indices[0]) - 1
                        uv_idx = (
                            int(indices[1]) - 1
                            if len(indices) > 1 and indices[1]
                            else vertex_idx
                        )

                        face_vertices.append(vertex_idx)
                        face_uv_indices.append(uv_idx)

                    # Convert quad faces to triangles if needed
                    if len(face_vertices) == 3:
                        faces.append(face_vertices)
                        face_uvs.append(face_uv_indices)
                        face_materials.append(current_material)
                    elif len(face_vertices) == 4:
                        # Split quad into two triangles
                        faces.append(
                            [face_vertices[0], face_vertices[1], face_vertices[2]]
                        )
                        faces.append(
                            [face_vertices[0], face_vertices[2], face_vertices[3]]
                        )
                        face_uvs.append(
                            [face_uv_indices[0], face_uv_indices[1], face_uv_indices[2]]
                        )
                        face_uvs.append(
                            [face_uv_indices[0], face_uv_indices[2], face_uv_indices[3]]
                        )
                        face_materials.append(current_material)
                        face_materials.append(current_material)

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        uv_coords = np.array(uv_coords, dtype=np.float32) if uv_coords else None

        print(
            f"Loaded OBJ: {len(vertices)} vertices, {len(faces)} faces, {len(uv_coords) if uv_coords is not None else 0} UVs"
        )
        if mtl_lib:
            print(f"Material library: {mtl_lib}")

        # Count materials used
        unique_materials = set(mat for mat in face_materials if mat is not None)
        if unique_materials:
            print(f"Materials used: {', '.join(unique_materials)}")

        # NEW APPROACH: Normalize then transform (instead of normalize with embedded scale)
        # This matches how SimplicalComplexIO works
        print(f"Applying normalization to match voxel coordinate space")

        # Step 1: Compute original bbox
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_size = bbox_max - bbox_min

        print(f"Original bounding box: center={bbox_center}, size={bbox_size}")

        # Step 2: Normalize to unit size centered at origin
        max_dimension = np.max(bbox_size)
        if max_dimension > 0:
            # First center at origin
            vertices = vertices - bbox_center
            # Then scale to unit size
            vertices = vertices / max_dimension
            print(f"Normalized to unit size (scale={1.0/max_dimension:.6f})")

        # Step 3: Now apply the full pre_transform (just like SimplicalComplexIO does)
        if pre_transform is not None:
            transform_matrix = np.array(pre_transform.matrix())
            vertices_homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
            vertices_transformed = (transform_matrix @ vertices_homogeneous.T).T
            vertices = vertices_transformed[:, :3].astype(np.float32)
            print(f"Applied full pre_transform (scale={transform_scale:.6f})")

        final_bbox_min = np.min(vertices, axis=0)
        final_bbox_max = np.max(vertices, axis=0)
        final_center = (final_bbox_min + final_bbox_max) / 2.0
        final_size = final_bbox_max - final_bbox_min
        print(f"Final bounding box: center={final_center}, size={final_size}")

        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix="trellis_visual_mesh_")
        temp_obj_path = os.path.join(temp_dir, "normalized_mesh.obj")

        original_dir = os.path.dirname(file_path)
        original_name = os.path.splitext(os.path.basename(file_path))[0]

        mtl_path = os.path.join(original_dir, f"{original_name}.mtl")
        if not os.path.exists(mtl_path):

            mtl_path = os.path.join(original_dir, "new_mesh.mtl")

        if os.path.exists(mtl_path):

            temp_mtl_path = os.path.join(temp_dir, "normalized_mesh.mtl")
            shutil.copy2(mtl_path, temp_mtl_path)

            for file in os.listdir(original_dir):
                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
                ):
                    shutil.copy2(os.path.join(original_dir, file), temp_dir)
                    print(f"Copied texture: {file}")

        with open(temp_obj_path, "w") as f:
            if os.path.exists(mtl_path):
                f.write("mtllib normalized_mesh.mtl\n\n")

            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")

            if uv_coords is not None:
                for uv in uv_coords:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                f.write("\n")

            for i in range(len(vertices)):
                f.write(f"vn 0.0 0.0 1.0\n")
            f.write("\n")

            if os.path.exists(mtl_path):
                f.write("usemtl material_0\n")

            for face in faces:
                if uv_coords is not None:
                    f.write(
                        f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n"
                    )
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"Saved normalized mesh with original textures to: {temp_obj_path}")

        return vertices, faces, uv_coords, face_uvs, face_materials, mtl_lib

    elif file_path.lower().endswith(".obj") and not normalize_like_blender:
        print(f"Loading OBJ without normalization: {file_path}")

        # Load OBJ manually and apply pre_transform directly
        vertices = []
        uv_coords = []
        faces = []
        face_uvs = []
        face_materials = []
        current_material = None
        mtl_lib = None

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("mtllib "):
                    mtl_lib = line.split()[1]
                elif line.startswith("usemtl "):
                    current_material = line.split()[1]
                elif line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vt "):
                    parts = line.split()
                    uv_coords.append([float(parts[1]), float(parts[2])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    face_vertices = []
                    face_uv_indices = []

                    for part in parts:
                        indices = part.split("/")
                        vertex_idx = int(indices[0]) - 1
                        uv_idx = (
                            int(indices[1]) - 1
                            if len(indices) > 1 and indices[1]
                            else vertex_idx
                        )

                        face_vertices.append(vertex_idx)
                        face_uv_indices.append(uv_idx)

                    if len(face_vertices) == 3:
                        faces.append(face_vertices)
                        face_uvs.append(face_uv_indices)
                        face_materials.append(current_material)
                    elif len(face_vertices) == 4:
                        faces.append(
                            [face_vertices[0], face_vertices[1], face_vertices[2]]
                        )
                        faces.append(
                            [face_vertices[0], face_vertices[2], face_vertices[3]]
                        )
                        face_uvs.append(
                            [face_uv_indices[0], face_uv_indices[1], face_uv_indices[2]]
                        )
                        face_uvs.append(
                            [face_uv_indices[0], face_uv_indices[2], face_uv_indices[3]]
                        )
                        face_materials.append(current_material)
                        face_materials.append(current_material)

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        uv_coords = np.array(uv_coords, dtype=np.float32) if uv_coords else None

        print(
            f"Original OBJ range: X=[{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}], Y=[{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}], Z=[{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]"
        )

        # Apply pre_transform directly without normalization
        if pre_transform is not None:
            transform_matrix = np.array(pre_transform.matrix())
            vertices_homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
            vertices_transformed = (transform_matrix @ vertices_homogeneous.T).T
            vertices = vertices_transformed[:, :3].astype(np.float32)
            print(f"Applied pre_transform to OBJ mesh")
            print(
                f"Transformed range: X=[{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}], Y=[{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}], Z=[{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]"
            )

        print(
            f"Loaded OBJ without normalization: {len(vertices)} vertices, {len(faces)} faces"
        )
        return vertices, faces, uv_coords, face_uvs, face_materials, mtl_lib

    io = SimplicialComplexIO(pre_transform)
    surface_mesh = io.read(file_path)

    positions = surface_mesh.positions().view()

    vertices = []
    for i in range(len(positions)):
        pos = positions[i]
        if hasattr(pos, "tolist"):
            pos_list = pos.tolist()
            if isinstance(pos_list[0], list):
                vertices.append([pos_list[0][0], pos_list[1][0], pos_list[2][0]])
            else:
                vertices.append(pos_list)
        else:
            vertices.append([float(pos[0]), float(pos[1]), float(pos[2])])

    vertices = np.array(vertices, dtype=np.float32)

    # Report the range after loading with pre_transform
    print(f"Loaded {file_path} with pre_transform applied")
    print(
        f"Range: X=[{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}], Y=[{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}], Z=[{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]"
    )

    if normalize_like_blender and not file_path.lower().endswith(".obj"):
        print(f"Applying blender-style normalization with scale {transform_scale}")

        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_size = bbox_max - bbox_min

        print(f"Original bounding box: {bbox_min} to {bbox_max}")
        print(f"Original size: {bbox_size}")

        max_dimension = np.max(bbox_size)
        if max_dimension > 0:
            # Normalize to unit size, then apply the desired scale
            scale_factor = transform_scale / max_dimension
            vertices = vertices * scale_factor
            print(
                f"Applied scale factor: {scale_factor} (normalization factor: {1.0/max_dimension}, desired scale: {transform_scale})"
            )

        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        offset = -(bbox_min + bbox_max) / 2.0
        vertices = vertices + offset
        print(f"Applied center offset: {offset}")

        print(f"Applied Blender-style normalization with scale {transform_scale}")

        final_bbox_min = np.min(vertices, axis=0)
        final_bbox_max = np.max(vertices, axis=0)
        print(f"Final bounding box: {final_bbox_min} to {final_bbox_max}")
    else:
        print(
            f"No additional normalization applied (normalize_like_blender={normalize_like_blender})"
        )

    triangles = surface_mesh.triangles()
    faces = []
    if triangles.size() > 0:
        tri_topo = view(triangles.topo())
        for i in range(triangles.size()):
            face = tri_topo[i]

            if hasattr(face, "tolist"):
                face_list = face.tolist()
                if isinstance(face_list[0], list):
                    faces.append([face_list[0][0], face_list[1][0], face_list[2][0]])
                else:
                    faces.append(face_list)
            else:
                faces.append([int(face[0]), int(face[1]), int(face[2])])

    faces = np.array(faces, dtype=np.int32)

    print(f"Loaded visual mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")

    return (
        vertices,
        faces,
        None,
        None,
        None,
        None,
    )  # No UV coordinates, face materials, or MTL lib for non-OBJ files


def write_visual_meshes_obj(
    output_path, visual_meshes, visual_mesh_objects, current_visual_vertices=None
):
    if not visual_meshes:
        return

    output_dir = os.path.dirname(output_path)
    import glob
    import shutil

    # Collect per-object material information
    object_materials = {}
    texture_patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]

    # Process each object's materials and textures
    for i, (
        original_vertices,
        faces,
        centroid_offset,
        physics_obj_index,
        uv_coords,
        face_uvs,
        material_dir,
        obj_name,
        face_materials,
        mtl_lib,
    ) in enumerate(visual_meshes):
        object_materials[i] = {
            "materials": {},  # Will hold material_name -> material_properties
            "face_materials": face_materials,
            "obj_name": obj_name,
        }

        if material_dir and os.path.exists(material_dir):
            print(
                f"Processing materials for object '{obj_name}' from directory: {material_dir}"
            )

            # Find and read MTL file
            mtl_files = glob.glob(os.path.join(material_dir, "*.mtl"))
            if mtl_files:
                mtl_file = mtl_files[0]
                print(
                    f"Reading material properties for {obj_name} from: {os.path.basename(mtl_file)}"
                )

                # Parse MTL file to extract materials
                try:
                    current_material = None
                    with open(mtl_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("newmtl "):
                                current_material = line.split()[1]
                                # Create unique material name for this object
                                unique_material_name = (
                                    f"{obj_name}_{i}_{current_material}"
                                )
                                object_materials[i]["materials"][current_material] = {
                                    "unique_name": unique_material_name,
                                    "properties": [f"newmtl {unique_material_name}"],
                                }
                            elif current_material and (
                                line.startswith("Ka ")
                                or line.startswith("Kd ")
                                or line.startswith("Ks ")
                                or line.startswith("Ns ")
                                or line.startswith("d ")
                                or line.startswith("Tr ")
                                or line.startswith("illum ")
                                or line.startswith("Ni ")
                            ):
                                # Store material properties
                                object_materials[i]["materials"][current_material][
                                    "properties"
                                ].append(line)
                            elif current_material and line.startswith("map_"):
                                # Handle texture maps
                                parts = line.split()
                                if len(parts) >= 2:
                                    map_type = parts[0]
                                    original_texture = parts[1]
                                    texture_basename = os.path.basename(
                                        original_texture
                                    )

                                    # Create unique texture name for this object
                                    unique_texture_name = (
                                        f"{obj_name}_{i}_{texture_basename}"
                                    )

                                    # Copy texture file
                                    original_texture_path = os.path.join(
                                        material_dir, texture_basename
                                    )
                                    if os.path.exists(original_texture_path):
                                        dest_file = os.path.join(
                                            output_dir, unique_texture_name
                                        )
                                        if not os.path.exists(dest_file):
                                            shutil.copy2(
                                                original_texture_path, dest_file
                                            )
                                            print(
                                                f"Copied texture for {obj_name}: {texture_basename} -> {unique_texture_name}"
                                            )

                                    # Store updated texture reference
                                    updated_line = f"{map_type} {unique_texture_name}"
                                    object_materials[i]["materials"][current_material][
                                        "properties"
                                    ].append(updated_line)

                except Exception as e:
                    print(f"Warning: Could not read MTL file for {obj_name}: {e}")

            # Also copy any loose texture files not referenced in MTL
            for pattern in texture_patterns:
                for texture_file in glob.glob(os.path.join(material_dir, pattern)):
                    texture_basename = os.path.basename(texture_file)
                    unique_texture_name = f"{obj_name}_{i}_{texture_basename}"
                    dest_file = os.path.join(output_dir, unique_texture_name)

                    if not os.path.exists(dest_file):
                        shutil.copy2(texture_file, dest_file)
                        print(
                            f"Copied additional texture for {obj_name}: {texture_basename} -> {unique_texture_name}"
                        )

        # If no materials were found, create a default material
        if not object_materials[i]["materials"]:
            default_material_name = f"{obj_name}_{i}_default"
            object_materials[i]["materials"]["default"] = {
                "unique_name": default_material_name,
                "properties": [
                    f"newmtl {default_material_name}",
                    "Ka 0.2 0.2 0.2",
                    "Kd 0.8 0.8 0.8",
                    "Ks 0.0 0.0 0.0",
                    "Ns 0.0",
                ],
            }
            print(f"No materials found for object '{obj_name}', using default material")

    # Write the combined MTL file
    if object_materials:
        mtl_path = os.path.join(output_dir, "surface_mesh.mtl")
        with open(mtl_path, "w") as mtl_f:
            for i, obj_materials in object_materials.items():
                obj_name = obj_materials["obj_name"]
                for material_name, material_info in obj_materials["materials"].items():
                    # Write material properties
                    for prop in material_info["properties"]:
                        mtl_f.write(prop + "\n")
                    mtl_f.write("\n")
        print(
            f"Created combined MTL file with materials from {len(object_materials)} objects: {mtl_path}"
        )

    # Write the OBJ file
    with open(output_path, "w") as f:
        # Write MTL library reference
        if object_materials:
            f.write("mtllib surface_mesh.mtl\n\n")

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Collect all vertices and faces first
        for i, (
            original_vertices,
            faces,
            centroid_offset,
            physics_obj_index,
            uv_coords,
            face_uvs,
            material_dir,
            obj_name,
            face_materials,
            mtl_lib,
        ) in enumerate(visual_meshes):

            if current_visual_vertices and i < len(current_visual_vertices):
                vertices_to_write = current_visual_vertices[i]
            else:
                vertices_to_write = original_vertices + centroid_offset

            all_vertices.extend(vertices_to_write)

            for face in faces:
                adjusted_face = [
                    face[0] + vertex_offset,
                    face[1] + vertex_offset,
                    face[2] + vertex_offset,
                ]
                all_faces.append(adjusted_face)

            vertex_offset += len(vertices_to_write)

        # Write vertices
        for vertex in all_vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        f.write("\n")

        # Collect UV coordinates from all meshes and track offsets
        all_uvs = []
        mesh_uv_offsets = []

        for i, (
            original_vertices,
            faces,
            centroid_offset,
            physics_obj_index,
            uv_coords,
            face_uvs,
            material_dir,
            obj_name,
            face_materials,
            mtl_lib,
        ) in enumerate(visual_meshes):
            mesh_uv_offsets.append(len(all_uvs))  # Remember where this mesh's UVs start

            if uv_coords is not None and len(uv_coords) > 0:
                all_uvs.extend(uv_coords)
            else:
                # Add dummy UVs for meshes without UV coordinates
                num_vertices = (
                    len(current_visual_vertices[i])
                    if current_visual_vertices and i < len(current_visual_vertices)
                    else len(original_vertices)
                )
                all_uvs.extend([(0.5, 0.5)] * num_vertices)

        # Write UV coordinates
        for uv in all_uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n")

        # Write normals
        for i in range(len(all_vertices)):
            f.write(f"vn 0.0 0.0 1.0\n")
        f.write("\n")

        # Write faces with per-object materials, preserving original material assignments
        vertex_offset = 0
        face_idx = 0

        for mesh_i, (
            original_vertices,
            faces,
            centroid_offset,
            physics_obj_index,
            uv_coords,
            face_uvs,
            material_dir,
            obj_name,
            face_materials,
            mtl_lib,
        ) in enumerate(visual_meshes):

            num_vertices = (
                len(current_visual_vertices[mesh_i])
                if current_visual_vertices and mesh_i < len(current_visual_vertices)
                else len(original_vertices)
            )
            uv_offset = mesh_uv_offsets[mesh_i]

            # Group faces by material to minimize usemtl calls
            material_face_groups = {}

            for face_local_idx in range(len(faces)):
                face_global_idx = face_idx + face_local_idx

                # Determine material for this face
                if (
                    face_materials
                    and face_local_idx < len(face_materials)
                    and face_materials[face_local_idx]
                ):
                    original_material = face_materials[face_local_idx]
                    if (
                        mesh_i in object_materials
                        and original_material in object_materials[mesh_i]["materials"]
                    ):
                        material_name = object_materials[mesh_i]["materials"][
                            original_material
                        ]["unique_name"]
                    else:
                        # Fallback if material not found
                        material_name = f"{obj_name}_{mesh_i}_default"
                else:
                    # Use default material if no material specified
                    if mesh_i in object_materials:
                        materials_list = list(
                            object_materials[mesh_i]["materials"].keys()
                        )
                        if materials_list:
                            first_material = materials_list[0]
                            material_name = object_materials[mesh_i]["materials"][
                                first_material
                            ]["unique_name"]
                        else:
                            material_name = f"{obj_name}_{mesh_i}_default"
                    else:
                        material_name = f"{obj_name}_{mesh_i}_default"

                if material_name not in material_face_groups:
                    material_face_groups[material_name] = []
                material_face_groups[material_name].append(face_local_idx)

            # Write faces grouped by material
            for material_name, face_indices in material_face_groups.items():
                f.write(f"usemtl {material_name}\n")
                print(
                    f"Using material '{material_name}' for {len(face_indices)} faces of object '{obj_name}'"
                )

                for face_local_idx in face_indices:
                    face = all_faces[
                        face_idx + face_local_idx
                    ]  # Global face indices (already offset)

                    if face_uvs is not None and face_local_idx < len(face_uvs):
                        # Use the original UV mapping
                        uv_indices = face_uvs[face_local_idx]
                        f.write(
                            f"f {face[0]+1}/{uv_indices[0]+1+uv_offset}/{face[0]+1} {face[1]+1}/{uv_indices[1]+1+uv_offset}/{face[1]+1} {face[2]+1}/{uv_indices[2]+1+uv_offset}/{face[2]+1}\n"
                        )
                    else:
                        # Fallback: assume vertex index = UV index
                        f.write(
                            f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n"
                        )

            face_idx += len(faces)
            vertex_offset += num_vertices

        print(
            f"Exported {len(visual_meshes)} visual meshes with per-object multi-material textures: {len(all_vertices)} vertices, {len(all_faces)} faces to {output_path}"
        )


class MeshProcessor:

    def __init__(self, scene):
        self.scene = scene
        self.snh = StableNeoHookean()
        self.default_element = scene.contact_tabular().default_element()

    def build_tet_mesh(self, voxel_centers, voxel_size):

        half_size = voxel_size / 2.0

        n_voxels = len(voxel_centers)
        vertices = np.zeros((n_voxels * 8, 3), dtype=np.float32)

        offsets = np.array(
            [
                [-half_size, -half_size, -half_size],
                [half_size, -half_size, -half_size],
                [half_size, -half_size, half_size],
                [-half_size, -half_size, half_size],
                [-half_size, half_size, -half_size],
                [half_size, half_size, -half_size],
                [half_size, half_size, half_size],
                [-half_size, half_size, half_size],
            ]
        )

        for i, center in enumerate(voxel_centers):

            for j in range(8):
                vertices[i * 8 + j] = center + offsets[j]

        tets = np.zeros((n_voxels * 5, 4), dtype=np.int32)

        tet_patterns = [
            [0, 1, 2, 4],
            [1, 2, 4, 5],
            [2, 4, 5, 6],
            [0, 2, 3, 4],
            [2, 3, 4, 7],
        ]

        for i in range(n_voxels):
            base_vertex = i * 8
            for j, pattern in enumerate(tet_patterns):

                tet = [base_vertex + v for v in pattern]
                tets[i * 5 + j] = tet

        print(
            f"Created tetrahedral mesh with {len(vertices)} vertices and {len(tets)} tetrahedra"
        )
        return vertices, tets

    def build_tet_mesh_from_voxels(self, voxel_centers):
        from scipy.spatial import Delaunay

        # Estimate actual scale from mesh dimensions (voxel centers are already transformed)
        mesh_size = np.max(voxel_centers, axis=0) - np.min(voxel_centers, axis=0)
        avg_mesh_dimension = np.mean(mesh_size)

        # Scale-aware jitter: make jitter proportional to the mesh size
        jitter_amount = max(
            0.01 * avg_mesh_dimension / 10.0, 1e-6
        )  # Minimum jitter to avoid zero
        print(f"Mesh dimensions: {mesh_size}, avg: {avg_mesh_dimension:.6f}")
        print(f"Using jitter amount: {jitter_amount:.6f}")
        np.random.seed(42)
        jitter = np.random.normal(0, jitter_amount, voxel_centers.shape)
        voxel_centers_jittered = voxel_centers + jitter

        tri = Delaunay(voxel_centers_jittered)

        vertices = voxel_centers_jittered.astype(np.float64)

        tets = tri.simplices.astype(np.int32)

        # Scale-aware volume threshold: use a threshold relative to mesh size
        # Typical tetrahedron volume ~ (edge_length)^3, so threshold ~ (avg_dimension/N^(1/3))^3
        # where N is number of points
        typical_edge_length = avg_mesh_dimension / (len(voxel_centers) ** (1 / 3))
        volume_threshold = max(
            (typical_edge_length**3) * 1e-6, 1e-12
        )  # Minimum threshold
        print(
            f"Using volume threshold: {volume_threshold:.2e} (typical edge length: {typical_edge_length:.6f})"
        )

        valid_tets = []
        volumes = []
        negative_count = 0

        for tet in tets:

            v0, v1, v2, v3 = vertices[tet]

            mat = np.array([v1 - v0, v2 - v0, v3 - v0])
            signed_volume = np.linalg.det(mat) / 6.0
            volumes.append(abs(signed_volume))

            if abs(signed_volume) > volume_threshold:

                if signed_volume < 0:
                    tet = [tet[0], tet[2], tet[1], tet[3]]
                    negative_count += 1

                valid_tets.append(tet)

        valid_tets = np.array(valid_tets, dtype=np.int32)

        if volumes:
            print(
                f"Volume statistics: min={np.min(volumes):.2e}, max={np.max(volumes):.2e}, median={np.median(volumes):.2e}"
            )
            print(f"Negative volumes corrected: {negative_count}")

        print(
            f"Created tetrahedral mesh with {len(vertices)} vertices and {len(valid_tets)} tetrahedra"
        )
        print(f"Filtered out {len(tets) - len(valid_tets)} degenerate tetrahedra")

        return vertices, valid_tets

    def load_voxel_mesh(
        self,
        voxel_path,
        voxel_size=1.0,
        scale=1.0,
        pre_transform=None,
        max_voxels=15000,
    ):
        if pre_transform is None:
            pre_transform = Transform.Identity()
            pre_transform.scale(scale)

        io = SimplicialComplexIO(pre_transform)
        voxel_cloud = io.read(voxel_path)

        voxel_positions = voxel_cloud.positions().view()

        voxel_centers = []
        for i in range(len(voxel_positions)):
            pos = voxel_positions[i]

            if hasattr(pos, "tolist"):
                pos_list = pos.tolist()

                if isinstance(pos_list[0], list):
                    voxel_centers.append(
                        [pos_list[0][0], pos_list[1][0], pos_list[2][0]]
                    )
                else:
                    voxel_centers.append(pos_list)
            else:
                voxel_centers.append([float(pos[0]), float(pos[1]), float(pos[2])])

        voxel_centers = np.array(voxel_centers, dtype=np.float32)

        print(f"Loaded {len(voxel_centers)} voxel centers from {voxel_path}")
        print(f"Voxel centers shape: {voxel_centers.shape}")

        if max_voxels is None or max_voxels >= len(voxel_centers):

            voxel_centers_subset = voxel_centers
            stride = 1
            print(
                f"Using ALL {len(voxel_centers_subset)} voxels for maximum resolution"
            )
        else:

            target_voxels = min(max_voxels, len(voxel_centers))
            stride = max(1, len(voxel_centers) // target_voxels)
            voxel_centers_subset = voxel_centers[::stride]

            print(
                f"Using {len(voxel_centers_subset)} voxels for tetrahedral mesh (stride={stride})"
            )
            print(
                f"Resolution: {len(voxel_centers_subset)}/{len(voxel_centers)} = {100*len(voxel_centers_subset)/len(voxel_centers):.1f}%"
            )

        vertices, tets = self.build_tet_mesh_from_voxels(voxel_centers_subset)

        print(f"Vertices shape: {vertices.shape}, Tets shape: {tets.shape}")

        tet_mesh = tetmesh(vertices, tets)

        label_surface(tet_mesh)

        return tet_mesh

    def load_mesh(self, file_path, scale=1.0, pre_transform=None):
        if pre_transform is None:
            pre_transform = Transform.Identity()
            pre_transform.scale(scale)

        io = SimplicialComplexIO(pre_transform)
        surface_mesh = io.read(file_path)

        label_surface(surface_mesh)

        tet_mesh = tetrahedralize(surface_mesh)

        label_surface(tet_mesh)

        return tet_mesh

    def load_material_properties_from_npz(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Material properties file not found: {npz_path}")

        try:
            data = np.load(npz_path)

            if "voxel_data" not in data:
                raise ValueError("NPZ file must contain 'voxel_data' field")

            voxel_data = data["voxel_data"]

            voxel_positions = np.column_stack(
                [voxel_data["x"], voxel_data["y"], voxel_data["z"]]
            )

            material_props = {
                "positions": voxel_positions,
                "youngs_modulus": voxel_data["youngs_modulus"],
                "poisson_ratio": voxel_data["poissons_ratio"],
                "density": voxel_data["density"],
            }

            print(f"Loaded {len(voxel_data)} voxel material properties from {npz_path}")
            return material_props

        except Exception as e:
            raise RuntimeError(
                f"Error loading material properties from {npz_path}: {e}"
            )

    def assign_per_tetrahedron_materials(
        self, mesh, material_props, scale=1.0, rotation=None, translation=None
    ):
        from scipy.spatial import cKDTree

        tetrahedra = mesh.tetrahedra()
        vertices = mesh.vertices()

        num_tetrahedra = tetrahedra.size()
        num_vertices = vertices.size()

        print(
            f"Assigning materials to {num_tetrahedra} tetrahedra based on {len(material_props['positions'])} voxels"
        )

        vertex_positions = mesh.positions().view()
        tet_connectivity = view(tetrahedra.topo())

        tet_centroids = []
        for i in range(num_tetrahedra):
            tet_verts = tet_connectivity[i]

            v0 = np.array(vertex_positions[tet_verts[0]]).flatten()
            v1 = np.array(vertex_positions[tet_verts[1]]).flatten()
            v2 = np.array(vertex_positions[tet_verts[2]]).flatten()
            v3 = np.array(vertex_positions[tet_verts[3]]).flatten()

            centroid = (v0 + v1 + v2 + v3) / 4.0
            tet_centroids.append(centroid)

        tet_centroids = np.array(tet_centroids)
        print(f"Tetrahedron centroids shape: {tet_centroids.shape}")

        # Transform material voxel positions to match mesh transformation
        transformed_voxel_positions = material_props["positions"].copy()

        # 1. Apply scaling
        if scale != 1.0:
            print(f"Scaling material voxel positions by factor: {scale}")
            transformed_voxel_positions *= scale

        # 2. Apply rotation if specified
        if rotation is not None:
            import math

            print(f"Rotating material voxel positions by: {rotation} degrees")

            # Create rotation matrix - same order as mesh rotation (Z, Y, X)
            rotation_matrix = np.eye(3)

            # Z rotation
            if rotation[2] != 0:
                angle_rad = math.radians(rotation[2])
                cos_z, sin_z = math.cos(angle_rad), math.sin(angle_rad)
                rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
                rotation_matrix = rotation_matrix @ rot_z

            # Y rotation
            if rotation[1] != 0:
                angle_rad = math.radians(rotation[1])
                cos_y, sin_y = math.cos(angle_rad), math.sin(angle_rad)
                rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
                rotation_matrix = rotation_matrix @ rot_y

            # X rotation
            if rotation[0] != 0:
                angle_rad = math.radians(rotation[0])
                cos_x, sin_x = math.cos(angle_rad), math.sin(angle_rad)
                rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
                rotation_matrix = rotation_matrix @ rot_x

            # Apply rotation to voxel positions
            transformed_voxel_positions = (
                rotation_matrix @ transformed_voxel_positions.T
            ).T

        # 3. Apply translation if specified
        if translation is not None:
            print(f"Translating material voxel positions by: {translation}")
            transformed_voxel_positions += np.array(translation)

        print(
            f"Material voxel positions transformed from {np.mean(material_props['positions'], axis=0)} to {np.mean(transformed_voxel_positions, axis=0)} (centroid)"
        )

        voxel_tree = cKDTree(transformed_voxel_positions)

        distances, closest_voxel_indices = voxel_tree.query(tet_centroids)

        mu_values = []
        lambda_values = []

        for i in range(num_tetrahedra):
            voxel_idx = closest_voxel_indices[i]

            E = float(material_props["youngs_modulus"][voxel_idx])
            nu = float(material_props["poisson_ratio"][voxel_idx])

            mu = E / (2.0 * (1.0 + nu))
            lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

            mu_values.append(mu)
            lambda_values.append(lam)

        mu_attr = mesh.tetrahedra().find("mu")
        if not mu_attr:
            mu_attr = mesh.tetrahedra().create("mu", 0.0)

        lambda_attr = mesh.tetrahedra().find("lambda")
        if not lambda_attr:
            lambda_attr = mesh.tetrahedra().create("lambda", 0.0)

        mu_view = view(mu_attr)
        lambda_view = view(lambda_attr)

        for i in range(num_tetrahedra):
            mu_view[i] = mu_values[i]
            lambda_view[i] = lambda_values[i]

        mass_density_attr = mesh.vertices().find("mass_density")
        if not mass_density_attr:
            mass_density_attr = mesh.vertices().create("mass_density", 1000.0)

        mass_density_view = view(mass_density_attr)

        vertex_coords = []
        for i in range(num_vertices):
            vertex_coords.append(np.array(vertex_positions[i]).flatten())
        vertex_coords = np.array(vertex_coords)

        _, closest_vertex_voxels = voxel_tree.query(vertex_coords)

        for i in range(num_vertices):
            voxel_idx = closest_vertex_voxels[i]
            rho = float(material_props["density"][voxel_idx])
            mass_density_view[i] = rho

        print(
            f"Successfully assigned per-tetrahedron materials to {num_tetrahedra} tetrahedra"
        )

        material_counts = {}
        for i in range(num_tetrahedra):
            voxel_idx = closest_voxel_indices[i]
            E = material_props["youngs_modulus"][voxel_idx]
            if E not in material_counts:
                material_counts[E] = 0
            material_counts[E] += 1

        print(f"VERIFICATION: Per-tetrahedron material distribution:")
        for E, count in material_counts.items():
            print(f"  {count} tetrahedra with E = {E:.0f} Pa")
        print(
            f"VERIFICATION: Total tetrahedra: {sum(material_counts.values())} (should equal {num_tetrahedra})"
        )

        print(f"Applied Lamé parameters range:")
        print(
            f"  μ (shear modulus): {np.min(mu_values):.0f} - {np.max(mu_values):.0f} Pa"
        )
        print(
            f"  λ (first Lamé parameter): {np.min(lambda_values):.0f} - {np.max(lambda_values):.0f} Pa"
        )

    def apply_material_properties(
        self,
        mesh,
        youngs_modulus=1e4,
        poisson_ratio=0.45,
        density=500,
        material_file=None,
        scale=1.0,
        rotation=None,
        translation=None,
    ):

        if material_file is not None:
            print(f"Loading heterogeneous material properties from: {material_file}")
            material_props = self.load_material_properties_from_npz(material_file)

            representative_E = float(np.mean(material_props["youngs_modulus"]))
            representative_nu = float(np.mean(material_props["poisson_ratio"]))
            representative_rho = float(np.mean(material_props["density"]))

            print(
                f"Applying constitutive model with representative values: E={representative_E:.0f} Pa, ν={representative_nu:.3f}, ρ={representative_rho:.0f} kg/m³"
            )
            moduli = ElasticModuli.youngs_poisson(representative_E, representative_nu)
            self.snh.apply_to(mesh, moduli=moduli, mass_density=representative_rho)

            print(
                "Overriding uniform values with per-tetrahedron heterogeneous materials..."
            )
            self.assign_per_tetrahedron_materials(
                mesh, material_props, scale, rotation, translation
            )
            print(
                "Per-tetrahedron mu/lambda attributes now override uniform values during simulation"
            )
        else:

            print(
                f"Applying uniform material properties: E={youngs_modulus:.0f} Pa, ν={poisson_ratio:.3f}, ρ={density:.0f} kg/m³"
            )
            moduli = ElasticModuli.youngs_poisson(youngs_modulus, poisson_ratio)
            self.snh.apply_to(mesh, moduli=moduli, mass_density=density)

        self.default_element.apply_to(mesh)

    def create_simple_tet_mesh(self):

        Vs = np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [-np.sqrt(3) / 2, 0, -0.5],
                [np.sqrt(3) / 2, 0, -0.5],
            ],
            dtype=np.float64,
        )
        Ts = np.array([[0, 1, 2, 3]], dtype=np.int32)

        tet_mesh = tetmesh(Vs, Ts)

        label_surface(tet_mesh)

        return tet_mesh

    def create_objects_from_config(self, objects_config):
        all_meshes = []

        for i, obj_config in enumerate(objects_config):

            if obj_config.get("type") == "voxel":
                voxel_path = obj_config["voxel_path"]
                voxel_size = obj_config.get("voxel_size", 1.0)
                scale = obj_config.get("scale", 1.0)
                max_voxels = obj_config.get("max_voxels", 15000)

                mesh = self.load_voxel_mesh(
                    voxel_path, voxel_size, scale, max_voxels=max_voxels
                )
            elif obj_config.get("type") == "simple_tet":
                mesh = self.create_simple_tet_mesh()
            elif obj_config.get("type") == "msh":
                msh_path = obj_config["msh_path"]
                scale = obj_config.get("scale", 1.0)

                mesh = self.load_msh_file(msh_path, scale)
            else:

                mesh = self.load_mesh(obj_config["path"], obj_config.get("scale", 1.0))

            # Extract transformation parameters from config for material assignment
            mesh_scale = obj_config.get("scale", 1.0)
            mesh_rotation = obj_config.get("rotation", None)
            mesh_translation = obj_config.get("translation", None)

            material = obj_config.get("material", {})
            material_file = material.get("file", None)

            # Apply material properties with transformation parameters
            # so materials are oriented correctly before mesh transformation
            self.apply_material_properties(
                mesh,
                material.get("youngs_modulus", 1e4),
                material.get("poisson_ratio", 0.45),
                material.get("density", 500),
                material_file,
                mesh_scale,
                mesh_rotation,
                mesh_translation,
            )

            # Apply boundary conditions if specified in config
            if obj_config.get("apply_boundary_conditions", False):
                fix_percentage = obj_config.get("boundary_fix_percentage", 0.1)
                self.apply_fixed_boundary_conditions(mesh, fix_percentage)
            elif obj_config.get("apply_frame_boundary_conditions", False):
                frame_positions_file = obj_config.get(
                    "frame_positions_file", "assets/armchair/frame_positions_for_bc.npz"
                )
                proximity_threshold = obj_config.get("frame_proximity_threshold", 0.1)
                self.apply_frame_boundary_conditions(
                    mesh, frame_positions_file, proximity_threshold
                )

            transform = Transform.Identity()
            if "translation" in obj_config:
                t = obj_config["translation"]
                translation_vector = (
                    Vector3.UnitX() * t[0]
                    + Vector3.UnitY() * t[1]
                    + Vector3.UnitZ() * t[2]
                )
                transform.translate(translation_vector)
            if "rotation" in obj_config:
                r = obj_config["rotation"]
                # Convert degrees to radians and apply rotations in order: Z, Y, X
                import math

                if r[2] != 0:
                    angle_rad = math.radians(r[2])
                    transform.rotate(AngleAxis(angle_rad, Vector3.UnitZ()))
                if r[1] != 0:
                    angle_rad = math.radians(r[1])
                    transform.rotate(AngleAxis(angle_rad, Vector3.UnitY()))
                if r[0] != 0:
                    angle_rad = math.radians(r[0])
                    transform.rotate(AngleAxis(angle_rad, Vector3.UnitX()))

            view(mesh.transforms())[0] = transform.matrix()

            obj_name = obj_config.get("name", f"object_{i+1}")
            scene_object = self.scene.objects().create(obj_name)
            scene_object.geometries().create(mesh)

            all_meshes.append((mesh, obj_config))

        return all_meshes

    def create_ground(self, height=0.0):
        ground_object = self.scene.objects().create("ground")
        g = ground(height)
        ground_object.geometries().create(g)
        return ground_object

    def load_msh_file(self, msh_path, scale=1.0, pre_transform=None):
        """
        Load a tetrahedral mesh directly from an MSH file (Gmsh format)
        """
        if not os.path.exists(msh_path):
            raise FileNotFoundError(f"MSH file not found: {msh_path}")

        if pre_transform is None:
            pre_transform = Transform.Identity()
            pre_transform.scale(scale)

        print(f"Loading tetrahedral mesh from MSH file: {msh_path}")

        # Load the MSH file using SimplicialComplexIO
        io = SimplicialComplexIO(pre_transform)
        tet_mesh = io.read(msh_path)

        # Label surface triangles for rendering
        label_surface(tet_mesh)

        # Get mesh statistics
        vertices = tet_mesh.vertices()
        tetrahedra = tet_mesh.tetrahedra()
        triangles = tet_mesh.triangles()

        print(
            f"Loaded MSH mesh: {vertices.size()} vertices, {tetrahedra.size()} tetrahedra, {triangles.size()} triangles"
        )
        print(f"MSH file loaded successfully from {msh_path}")

        return tet_mesh

    def apply_fixed_boundary_conditions(self, mesh, fix_lower_percentage=0.1):
        """
        Apply boundary conditions to fix the lower percentage of vertices to their current positions.

        Args:
            mesh: The tetrahedral mesh to apply boundary conditions to
            fix_lower_percentage: Percentage of lowest vertices to fix (default: 0.1 for 10%)
        """
        print(
            f"Applying boundary conditions to fix lower {fix_lower_percentage*100:.1f}% of vertices..."
        )

        # Get vertex positions
        vertex_positions = mesh.positions().view()
        num_vertices = len(vertex_positions)

        # Extract Z coordinates (height) for all vertices
        z_coords = []
        for i in range(num_vertices):
            pos = vertex_positions[i]
            if hasattr(pos, "tolist"):
                pos_list = pos.tolist()
                if isinstance(pos_list[0], list):
                    z_coords.append(pos_list[2][0])  # Z coordinate
                else:
                    z_coords.append(pos_list[2])  # Z coordinate
            else:
                z_coords.append(float(pos[2]))  # Z coordinate

        z_coords = np.array(z_coords)

        # Find the threshold Z coordinate for the lower percentage of vertices
        z_threshold = np.percentile(z_coords, fix_lower_percentage * 100)

        # Find vertices below or at the threshold
        fixed_vertex_indices = np.where(z_coords <= z_threshold)[0]

        print(f"Z-coordinate range: {np.min(z_coords):.6f} to {np.max(z_coords):.6f}")
        print(f"Threshold Z-coordinate: {z_threshold:.6f}")
        print(
            f"Fixing {len(fixed_vertex_indices)} vertices out of {num_vertices} total vertices"
        )

        # Get or create the is_fixed attribute
        is_fixed_attr = mesh.vertices().find(builtin.is_fixed)
        if not is_fixed_attr:
            is_fixed_attr = mesh.vertices().create(builtin.is_fixed, 0)

        # Set the is_fixed attribute for the selected vertices
        is_fixed_view = view(is_fixed_attr)
        for vertex_idx in fixed_vertex_indices:
            is_fixed_view[vertex_idx] = 1

        print(
            f"Successfully applied boundary conditions to {len(fixed_vertex_indices)} vertices"
        )

        return fixed_vertex_indices

    def apply_frame_boundary_conditions(
        self, mesh, frame_positions_file, proximity_threshold=0.1
    ):
        """
        Apply boundary conditions to fix vertices near frame positions.

        Args:
            mesh: The tetrahedral mesh to apply boundary conditions to
            frame_positions_file: NPZ file containing frame_positions array
            proximity_threshold: Distance threshold for considering vertices as frame (default: 0.1)
        """
        print(
            f"Applying frame-based boundary conditions with proximity threshold {proximity_threshold}..."
        )

        # Load frame positions
        frame_data = np.load(frame_positions_file)
        frame_positions = frame_data["frame_positions"]
        print(f"Loaded {len(frame_positions)} frame voxel positions")

        # Get vertex positions
        vertex_positions = mesh.positions().view()
        num_vertices = len(vertex_positions)

        # Convert vertex positions to numpy array
        vertex_coords = []
        for i in range(num_vertices):
            pos = vertex_positions[i]
            if hasattr(pos, "tolist"):
                pos_list = pos.tolist()
                if isinstance(pos_list[0], list):
                    vertex_coords.append(
                        [pos_list[0][0], pos_list[1][0], pos_list[2][0]]
                    )
                else:
                    vertex_coords.append(pos_list)
            else:
                vertex_coords.append([float(pos[0]), float(pos[1]), float(pos[2])])

        vertex_coords = np.array(vertex_coords)

        # Find vertices close to frame positions using KDTree for efficiency
        from scipy.spatial import cKDTree

        frame_tree = cKDTree(frame_positions)

        # Query distances from each vertex to nearest frame position
        distances, closest_frame_indices = frame_tree.query(vertex_coords)

        # Find vertices within proximity threshold of frame positions
        frame_vertex_indices = np.where(distances <= proximity_threshold)[0]

        print(
            f"Found {len(frame_vertex_indices)} vertices within {proximity_threshold} units of frame positions"
        )
        print(
            f"Fixing {len(frame_vertex_indices)}/{num_vertices} vertices ({100*len(frame_vertex_indices)/num_vertices:.1f}%)"
        )

        # Get or create the is_fixed attribute
        is_fixed_attr = mesh.vertices().find(builtin.is_fixed)
        if not is_fixed_attr:
            is_fixed_attr = mesh.vertices().create(builtin.is_fixed, 0)

        # Set the is_fixed attribute for frame vertices
        is_fixed_view = view(is_fixed_attr)
        for vertex_idx in frame_vertex_indices:
            is_fixed_view[vertex_idx] = 1

        print(
            f"Successfully applied frame-based boundary conditions to {len(frame_vertex_indices)} vertices"
        )

        return frame_vertex_indices
