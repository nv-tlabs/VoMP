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

"""Create solid, material-labelled 64^3 voxels from normalized meshes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import igl
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

GRID_RESOLUTION = 64
GRID_SHAPE = (GRID_RESOLUTION,) * 3
GRID_MINIMUM = -0.5
GRID_MAXIMUM = 0.5
VOXEL_SIZE = 1.0 / GRID_RESOLUTION
BOUNDARY_EPSILON = 1.0e-6
WINDING_PROXY_THRESHOLD = 2_000_000
MAX_WINDING_PROXY_TRIANGLES = 4_000_000
WINDING_PROXY_RESOLUTIONS = (512, 256, 128)


class VoxelizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterialRecord:
    dataset: str
    segment_keys: tuple[str, ...]
    properties: np.ndarray
    fingerprint: str


@dataclass(frozen=True)
class RenderMesh:
    render_dir: Path
    mesh_path: Path
    mesh_sha256: str
    segment_keys: tuple[str, ...]
    segment_keys_sha256: str
    source_entrypoint: str
    dataset: str
    vertex_count: int
    triangle_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_source_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return str(path.resolve())


def _material_fingerprint(
    segment_keys: tuple[str, ...], properties: np.ndarray
) -> str:
    payload = [
        [
            key,
            float(properties[index, 0]),
            float(properties[index, 1]),
            float(properties[index, 2]),
        ]
        for index, key in enumerate(segment_keys)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_material_annotations(path: str | Path) -> dict[str, MaterialRecord]:
    annotation_path = Path(path).expanduser().resolve()
    with annotation_path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise VoxelizationError("material_annotations.json must contain a list")

    result: dict[str, MaterialRecord] = {}
    for object_record in records:
        if not isinstance(object_record, Mapping):
            raise VoxelizationError("material annotation object is malformed")
        file_path = object_record.get("file_path")
        dataset = object_record.get("dataset_type")
        segments = object_record.get("segments")
        if (
            not isinstance(file_path, str)
            or not file_path
            or not isinstance(dataset, str)
            or not dataset
            or not isinstance(segments, Mapping)
            or not segments
        ):
            raise VoxelizationError("material annotation object is incomplete")

        canonical_path = _canonical_source_path(file_path)
        if canonical_path in result:
            raise VoxelizationError(
                f"duplicate material annotation path: {canonical_path}"
            )

        segment_keys = tuple(sorted(segments))
        properties = np.empty((len(segment_keys), 3), dtype=np.float64)
        for index, key in enumerate(segment_keys):
            segment = segments[key]
            if not isinstance(key, str) or not key or not isinstance(segment, Mapping):
                raise VoxelizationError(
                    f"invalid material segment in {canonical_path}"
                )
            values = (
                segment.get("youngs_modulus"),
                segment.get("poissons_ratio"),
                segment.get("density"),
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise VoxelizationError(
                    f"segment {key!r} has incomplete material properties"
                )
            youngs_modulus, poissons_ratio, density = map(float, values)
            if (
                youngs_modulus <= 0.0
                or not 0.0 <= poissons_ratio <= 0.5
                or density <= 0.0
            ):
                raise VoxelizationError(
                    f"segment {key!r} has invalid material properties"
                )
            properties[index] = (youngs_modulus, poissons_ratio, density)
            if not np.isfinite(properties[index].astype(np.float32)).all():
                raise VoxelizationError(
                    f"segment {key!r} properties exceed float32 range"
                )

        result[canonical_path] = MaterialRecord(
            dataset=dataset,
            segment_keys=segment_keys,
            properties=properties,
            fingerprint=_material_fingerprint(segment_keys, properties),
        )
    return result


def _load_render_mesh_info(
    output_dir: str | Path, sha256: str, source_path: str, dataset: str
) -> RenderMesh:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise VoxelizationError(f"invalid asset hash: {sha256!r}")
    render_dir = Path(output_dir).expanduser().resolve() / "renders" / sha256
    manifest_path = render_dir / "transforms.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    mesh = manifest.get("mesh")
    normalization = manifest.get("normalization")
    settings = manifest.get("render_settings")
    if (
        manifest.get("aabb")
        != [[GRID_MINIMUM, GRID_MINIMUM, GRID_MINIMUM], [GRID_MAXIMUM] * 3]
        or not isinstance(normalization, Mapping)
        or not isinstance(settings, Mapping)
        or settings.get("resolution") != 512
        or settings.get("num_views") != 150
        or not isinstance(mesh, Mapping)
    ):
        raise VoxelizationError(f"render mesh metadata is invalid for {sha256}")

    if "augmentation" in manifest:
        augmentation = manifest.get("augmentation")
        rotation = np.asarray(normalization.get("rotation_matrix"), dtype=np.float64)
        rotation_center = np.asarray(
            normalization.get("rotation_center"), dtype=np.float64
        )
        rotation_scale = normalization.get("rotation_scale")
        quaternion = np.asarray(
            augmentation.get("quaternion_wxyz")
            if isinstance(augmentation, Mapping)
            else None,
            dtype=np.float64,
        )
        if (
            not isinstance(augmentation, Mapping)
            or not isinstance(augmentation.get("base_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", augmentation["base_sha256"])
            or augmentation.get("index") not in {1, 2}
            or rotation.shape != (3, 3)
            or rotation_center.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(rotation_center).all()
            or not np.isfinite(quaternion).all()
            or not math.isclose(
                float(np.linalg.norm(quaternion)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            or isinstance(rotation_scale, bool)
            or not isinstance(rotation_scale, (int, float))
            or not math.isfinite(float(rotation_scale))
            or float(rotation_scale) <= 0.0
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
            or not math.isclose(
                float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-10
            )
            or not np.allclose(
                rotation,
                np.asarray(augmentation.get("rotation_matrix"), dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise VoxelizationError(f"augmentation transform is invalid for {sha256}")

        w, x, y, z = quaternion
        quaternion_rotation = np.asarray(
            (
                (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )
        base_sha256 = augmentation["base_sha256"]
        augmentation_index = augmentation["index"]
        expected_sha256 = hashlib.sha256(
            f"{base_sha256}\0offline_rotation\0{augmentation_index}".encode("ascii")
        ).hexdigest()
        base_manifest_path = (
            Path(output_dir).expanduser().resolve()
            / "renders"
            / base_sha256
            / "transforms.json"
        )
        with base_manifest_path.open("r", encoding="utf-8") as stream:
            base_manifest = json.load(stream)
        if (
            sha256 != expected_sha256
            or not np.allclose(
                rotation, quaternion_rotation, rtol=0.0, atol=1.0e-12
            )
            or augmentation.get("canonical_mesh_sha256")
            != base_manifest.get("mesh", {}).get("sha256")
            or augmentation.get("canonical_input_fingerprint")
            != base_manifest.get("input_fingerprint")
        ):
            raise VoxelizationError(f"augmentation metadata is stale for {sha256}")

    segment_keys = mesh.get("segment_keys")
    vertex_count = mesh.get("vertex_count")
    triangle_count = mesh.get("triangle_count")
    segment_keys_sha256 = mesh.get("segment_keys_sha256")
    mesh_sha256 = mesh.get("sha256")
    if (
        mesh.get("file_path") != "mesh.ply"
        or not isinstance(segment_keys, list)
        or not segment_keys
        or segment_keys != sorted(segment_keys)
        or len(set(segment_keys)) != len(segment_keys)
        or not all(isinstance(key, str) and key for key in segment_keys)
        or type(vertex_count) is not int
        or vertex_count < 3
        or type(triangle_count) is not int
        or triangle_count < 1
        or not isinstance(segment_keys_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", segment_keys_sha256)
        or not isinstance(mesh_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", mesh_sha256)
    ):
        raise VoxelizationError(f"normalized mesh record is invalid for {sha256}")

    expected_keys_sha256 = hashlib.sha256(
        json.dumps(
            segment_keys, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if expected_keys_sha256 != segment_keys_sha256:
        raise VoxelizationError(f"segment-key digest is invalid for {sha256}")

    canonical_source = _canonical_source_path(source_path)
    manifest_source = manifest.get("source_entrypoint")
    source_sha256 = manifest.get("source_entrypoint_sha256")
    if (
        not isinstance(manifest_source, str)
        or _canonical_source_path(manifest_source) != canonical_source
        or not isinstance(source_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
        or _sha256_file(Path(canonical_source)) != source_sha256
    ):
        raise VoxelizationError(f"render source identity is invalid for {sha256}")

    render_mesh = RenderMesh(
        render_dir=render_dir,
        mesh_path=render_dir / "mesh.ply",
        mesh_sha256=mesh_sha256,
        segment_keys=tuple(segment_keys),
        segment_keys_sha256=segment_keys_sha256,
        source_entrypoint=canonical_source,
        dataset=dataset,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
    )
    _validate_render_mesh_file(render_mesh)
    return render_mesh


def _materials_for_asset(
    render_mesh: RenderMesh,
    annotations: Mapping[str, MaterialRecord],
) -> MaterialRecord:
    material = annotations.get(render_mesh.source_entrypoint)
    if material is None:
        raise VoxelizationError(
            f"no material annotation for {render_mesh.source_entrypoint}"
        )
    if material.dataset != render_mesh.dataset:
        raise VoxelizationError(
            f"annotation dataset disagrees for {render_mesh.source_entrypoint}"
        )
    material_indices = {key: index for index, key in enumerate(material.segment_keys)}
    missing = [key for key in render_mesh.segment_keys if key not in material_indices]
    if missing:
        raise VoxelizationError(
            f"annotation segments disagree for {render_mesh.source_entrypoint}; "
            f"missing={missing}"
        )
    properties = material.properties[
        [material_indices[key] for key in render_mesh.segment_keys]
    ]
    return MaterialRecord(
        dataset=material.dataset,
        segment_keys=render_mesh.segment_keys,
        properties=properties,
        fingerprint=_material_fingerprint(render_mesh.segment_keys, properties),
    )


def _render_mesh_layout(render_mesh: RenderMesh) -> tuple[bytes, np.dtype]:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment exact composed render geometry in normalized object coordinates\n"
        f"comment segment_keys_sha256 {render_mesh.segment_keys_sha256}\n"
        f"element vertex {render_mesh.vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {render_mesh.triangle_count}\n"
        "property list uchar int vertex_indices\n"
        "property int segment_index\n"
        "end_header\n"
    ).encode("ascii")
    face_dtype = np.dtype(
        [
            ("count", "u1"),
            ("indices", "<i4", (3,)),
            ("segment_index", "<i4"),
        ]
    )
    return header, face_dtype


def _validate_render_mesh_file(render_mesh: RenderMesh) -> None:
    expected_header, face_dtype = _render_mesh_layout(render_mesh)
    header_size = len(expected_header)
    expected_size = (
        header_size
        + render_mesh.vertex_count * 12
        + render_mesh.triangle_count * face_dtype.itemsize
    )
    with render_mesh.mesh_path.open("rb") as stream:
        header = stream.read(header_size)
    if (
        header != expected_header
        or render_mesh.mesh_path.stat().st_size != expected_size
        or _sha256_file(render_mesh.mesh_path) != render_mesh.mesh_sha256
    ):
        raise VoxelizationError(f"mesh PLY does not match its manifest: {render_mesh.mesh_path}")

    vertices = np.memmap(
        render_mesh.mesh_path,
        mode="r",
        dtype="<f4",
        offset=header_size,
        shape=(render_mesh.vertex_count, 3),
    )
    faces = np.memmap(
        render_mesh.mesh_path,
        mode="r",
        dtype=face_dtype,
        offset=header_size + render_mesh.vertex_count * 12,
        shape=(render_mesh.triangle_count,),
    )
    valid = (
        np.isfinite(vertices).all()
        and not np.any(vertices < GRID_MINIMUM - 1.0e-6)
        and not np.any(vertices > GRID_MAXIMUM + 1.0e-6)
        and not np.any(faces["count"] != 3)
        and int(faces["indices"].min()) >= 0
        and int(faces["indices"].max()) < render_mesh.vertex_count
        and np.array_equal(
            np.unique(faces["segment_index"]),
            np.arange(len(render_mesh.segment_keys), dtype=np.int32),
        )
    )
    if not valid:
        raise VoxelizationError(f"mesh PLY payload is invalid: {render_mesh.mesh_path}")
    del faces
    del vertices


def _read_render_mesh(
    render_mesh: RenderMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_header, face_dtype = _render_mesh_layout(render_mesh)
    header_size = len(expected_header)
    vertices = np.memmap(
        render_mesh.mesh_path,
        mode="r",
        dtype="<f4",
        offset=header_size,
        shape=(render_mesh.vertex_count, 3),
    )
    faces = np.memmap(
        render_mesh.mesh_path,
        mode="r",
        dtype=face_dtype,
        offset=header_size + render_mesh.vertex_count * 12,
        shape=(render_mesh.triangle_count,),
    )
    vertex_array = np.asarray(vertices, dtype=np.float64).copy()
    triangle_array = np.asarray(faces["indices"], dtype=np.int64).copy()
    segment_array = np.asarray(faces["segment_index"], dtype=np.int32).copy()
    del faces
    del vertices
    return vertex_array, triangle_array, segment_array


def _indices_to_centers(indices: np.ndarray) -> np.ndarray:
    return (
        (np.asarray(indices, dtype=np.float64) + 0.5) / GRID_RESOLUTION
        + GRID_MINIMUM
    )


def _surface_flat_indices(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if not len(triangles):
        return np.empty(0, dtype=np.int64)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=VOXEL_SIZE,
        min_bound=np.full(3, GRID_MINIMUM, dtype=np.float64),
        max_bound=np.full(3, GRID_MAXIMUM, dtype=np.float64),
    )
    indices = np.asarray(
        [voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.int64
    )
    if not len(indices):
        return np.empty(0, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise VoxelizationError("Open3D returned malformed voxel indices")
    if np.any(indices < 0) or np.any(indices >= GRID_RESOLUTION):
        raise VoxelizationError("Open3D returned an out-of-range voxel index")
    return np.unique(np.ravel_multi_index(indices.T, GRID_SHAPE))


def _quantized_winding_proxy(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Cheaper inside tests for meshes with > 2 million triangles.
    for resolution in WINDING_PROXY_RESOLUTIONS:
        quantized = np.rint(
            (vertices - GRID_MINIMUM) * resolution
        ).astype(np.int64)
        quantized = np.clip(quantized, 0, resolution)
        vertex_keys = np.ravel_multi_index(
            quantized.T, (resolution + 1,) * 3
        )
        unique_keys, inverse = np.unique(vertex_keys, return_inverse=True)
        proxy_triangles = inverse[triangles]
        nondegenerate = (
            (proxy_triangles[:, 0] != proxy_triangles[:, 1])
            & (proxy_triangles[:, 1] != proxy_triangles[:, 2])
            & (proxy_triangles[:, 2] != proxy_triangles[:, 0])
        )
        proxy_triangles = np.unique(proxy_triangles[nondegenerate], axis=0)
        proxy_indices = np.column_stack(
            np.unravel_index(unique_keys, (resolution + 1,) * 3)
        )
        proxy_vertices = (
            proxy_indices.astype(np.float64) / resolution + GRID_MINIMUM
        )
        if (
            len(proxy_triangles) <= MAX_WINDING_PROXY_TRIANGLES
            or resolution == WINDING_PROXY_RESOLUTIONS[-1]
        ):
            if not len(proxy_triangles):
                raise VoxelizationError("winding proxy contains no triangles")
            return proxy_vertices, proxy_triangles.astype(np.int64, copy=False)
    raise AssertionError("unreachable winding proxy resolution")


def _solid_voxel_indices(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    # Keep boundary vertices from falling just outside the voxel grid.
    clamped = np.clip(
        vertices,
        GRID_MINIMUM + BOUNDARY_EPSILON,
        GRID_MAXIMUM - BOUNDARY_EPSILON,
    )
    surface_flat = _surface_flat_indices(clamped, triangles)
    if not len(surface_flat):
        raise VoxelizationError("render mesh produced no surface voxels")

    all_flat = np.arange(GRID_RESOLUTION**3, dtype=np.int64)
    all_indices = np.column_stack(np.unravel_index(all_flat, GRID_SHAPE))
    all_centers = _indices_to_centers(all_indices)
    winding_vertices, winding_triangles = vertices, triangles
    if len(triangles) > WINDING_PROXY_THRESHOLD:
        winding_vertices, winding_triangles = _quantized_winding_proxy(
            vertices, triangles
        )
    winding = np.asarray(
        igl.fast_winding_number(
            winding_vertices.astype(np.float64, copy=False),
            winding_triangles.astype(np.int64, copy=False),
            all_centers,
        ),
        dtype=np.float64,
    ).reshape(-1)
    if winding.shape != (GRID_RESOLUTION**3,) or not np.isfinite(winding).all():
        raise VoxelizationError("generalized winding numbers are invalid")

    surface_mask = np.zeros(GRID_RESOLUTION**3, dtype=bool)
    surface_mask[surface_flat] = True
    # Keep thin surfaces even when the inside test misses them.
    occupied_mask = surface_mask | (np.abs(winding) > 0.5)
    occupied_flat = np.flatnonzero(occupied_mask)
    occupied_indices = np.column_stack(
        np.unravel_index(occupied_flat, GRID_SHAPE)
    ).astype(np.uint8)
    occupied_surface_mask = surface_mask[occupied_flat]
    statistics = {
        "surface_voxel_count": int(surface_mask.sum()),
        "interior_voxel_count": int(np.count_nonzero(occupied_mask & ~surface_mask)),
        "winding_ambiguity_count": int(
            np.count_nonzero(
                (np.abs(winding) >= 0.45)
                & (np.abs(winding) <= 0.55)
                & ~surface_mask
            )
        ),
    }
    return occupied_indices, occupied_surface_mask, clamped, statistics


def _nearest_segment_labels(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_segments: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32, copy=False)),
        o3d.core.Tensor(triangles.astype(np.uint32, copy=False)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    primitive_blocks = []
    for start in range(0, len(centers), 65536):
        query = o3d.core.Tensor(
            centers[start : start + 65536].astype(np.float32, copy=False)
        )
        result = scene.compute_closest_points(query, nthreads=1)
        primitive_blocks.append(result["primitive_ids"].numpy().astype(np.int64))
    primitive_ids = np.concatenate(primitive_blocks)
    if (
        primitive_ids.shape != (len(centers),)
        or np.any(primitive_ids < 0)
        or np.any(primitive_ids >= len(triangles))
    ):
        raise VoxelizationError("nearest-triangle query returned invalid face IDs")
    return triangle_segments[primitive_ids]


def _force_segment_coverage(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_segments: np.ndarray,
    occupied_indices: np.ndarray,
    labels: np.ndarray,
    segment_count: int,
) -> int:
    # Tiny segments can lose every voxel to a nearby larger segment.
    counts = np.bincount(labels, minlength=segment_count)
    missing = np.flatnonzero(counts == 0)
    if not len(missing):
        return 0

    occupied_flat = np.ravel_multi_index(occupied_indices.T, GRID_SHAPE)
    row_for_flat = np.full(GRID_RESOLUTION**3, -1, dtype=np.int32)
    row_for_flat[occupied_flat] = np.arange(len(occupied_flat), dtype=np.int32)

    def surface_rows(segment_index: int) -> np.ndarray:
        segment_triangles = triangles[triangle_segments == segment_index]
        referenced, inverse = np.unique(
            segment_triangles.reshape(-1), return_inverse=True
        )
        local_triangles = inverse.reshape((-1, 3))
        surface_flat = _surface_flat_indices(vertices[referenced], local_triangles)
        rows = row_for_flat[surface_flat]
        return np.unique(rows[rows >= 0]).astype(np.int32, copy=False)

    def find_matching(rows_by_segment: list[np.ndarray]) -> np.ndarray:
        indptr = np.concatenate(
            (
                [0],
                np.cumsum(
                    [len(rows) for rows in rows_by_segment], dtype=np.int64
                ),
            )
        )
        graph = csr_matrix(
            (
                np.ones(indptr[-1], dtype=bool),
                np.concatenate(rows_by_segment),
                indptr,
            ),
            shape=(segment_count, len(labels)),
        )
        return maximum_bipartite_matching(graph, perm_type="column")

    def expanded_rows(rows: np.ndarray, radius: int) -> np.ndarray:
        coordinates = occupied_indices[rows].astype(np.int16, copy=False)
        neighbors = []
        for x_offset in range(-radius, radius + 1):
            for y_offset in range(-radius, radius + 1):
                for z_offset in range(-radius, radius + 1):
                    shifted = coordinates + (x_offset, y_offset, z_offset)
                    valid = np.all(
                        (shifted >= 0) & (shifted < GRID_RESOLUTION), axis=1
                    )
                    if np.any(valid):
                        flat = np.ravel_multi_index(shifted[valid].T, GRID_SHAPE)
                        neighbors.append(row_for_flat[flat])
        expanded = np.concatenate(neighbors)
        return np.unique(expanded[expanded >= 0]).astype(np.int32, copy=False)

    missing_set = set(missing)
    candidate_rows = []
    for segment_index in range(segment_count):
        if segment_index not in missing_set:
            rows = np.flatnonzero(labels == segment_index)
        else:
            rows = surface_rows(segment_index)
        if not len(rows):
            raise VoxelizationError(
                f"segment {segment_index} is below the 64^3 grid resolution"
            )
        candidate_rows.append(rows.astype(np.int32, copy=False))

    # Reserve a different voxel for each segment so they don't overwrite each other.
    matched_rows = find_matching(candidate_rows)
    if matched_rows.shape != (segment_count,) or np.any(matched_rows < 0):
        exact_missing_rows = {
            int(index): surface_rows(int(index)) for index in missing
        }
        for radius in (1, 2):
            candidate_rows = [
                (
                    expanded_rows(exact_missing_rows[segment_index], radius)
                    if segment_index in missing_set
                    else np.flatnonzero(labels == segment_index).astype(
                        np.int32, copy=False
                    )
                )
                for segment_index in range(segment_count)
            ]
            matched_rows = find_matching(candidate_rows)
            if matched_rows.shape == (segment_count,) and np.all(matched_rows >= 0):
                break
        else:
            raise VoxelizationError(
                "cannot represent every segment within two 64^3 cells of its surface"
            )
    forced = int(
        np.count_nonzero(
            labels[matched_rows] != np.arange(segment_count, dtype=labels.dtype)
        )
    )
    labels[matched_rows] = np.arange(segment_count, dtype=labels.dtype)
    if not np.array_equal(
        np.unique(labels), np.arange(segment_count, dtype=labels.dtype)
    ):
        raise VoxelizationError("voxel segment coverage is incomplete")
    return forced


def _read_voxel_ply(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        prefix = stream.read(4096)
    marker = b"end_header\n"
    marker_offset = prefix.find(marker)
    if marker_offset < 0:
        raise VoxelizationError(f"voxel PLY header is incomplete: {path}")
    header_size = marker_offset + len(marker)
    match = re.search(rb"element vertex ([0-9]+)\n", prefix[:header_size])
    if match is None:
        raise VoxelizationError(f"voxel PLY has no vertex count: {path}")
    count = int(match.group(1))
    expected_header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment solid 64^3 voxel centers in normalized render coordinates\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    if prefix[:header_size] != expected_header or path.stat().st_size != header_size + count * 12:
        raise VoxelizationError(f"voxel PLY layout is invalid: {path}")
    points = np.memmap(path, mode="r", dtype="<f4", offset=header_size, shape=(count, 3))
    result = np.asarray(points).copy()
    del points
    return result


def _write_voxel_ply(path: Path, centers: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment solid 64^3 voxel centers in normalized render coordinates\n"
        f"element vertex {len(centers)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        np.asarray(centers, dtype="<f4").tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())


def _npz_scalar(archive: Any, key: str) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise VoxelizationError(f"NPZ field {key!r} is not scalar")
    return value.item()


def _validate_voxel_pair(
    ply_path: Path,
    npz_path: Path,
    render_mesh: RenderMesh,
    material: MaterialRecord,
) -> int:
    points = _read_voxel_ply(ply_path)
    if not 0 < len(points) <= GRID_RESOLUTION**3 or not np.isfinite(points).all():
        raise VoxelizationError("voxel coordinates are empty or invalid")

    scaled = (points.astype(np.float64) - GRID_MINIMUM) * GRID_RESOLUTION
    indices = np.floor(scaled).astype(np.int64)
    expected_points = _indices_to_centers(indices).astype(np.float32)
    if (
        np.any(indices < 0)
        or np.any(indices >= GRID_RESOLUTION)
        or not np.array_equal(points, expected_points)
    ):
        raise VoxelizationError("voxel coordinates are not exact TRELLIS cell centers")
    flat = np.ravel_multi_index(indices.T, GRID_SHAPE)
    if len(flat) > 1 and np.any(np.diff(flat) <= 0):
        raise VoxelizationError("voxel coordinates are duplicated or not ordered")

    required_archive_fields = {
        "voxel_data",
        "is_surface",
        "grid_resolution",
        "render_mesh_sha256",
        "segment_keys_sha256",
        "material_properties_sha256",
        "voxel_ply_sha256",
        "surface_voxel_count",
        "interior_voxel_count",
        "winding_ambiguity_count",
        "forced_segment_voxel_count",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        if not required_archive_fields <= set(archive.files):
            raise VoxelizationError("material NPZ is missing source fields")
        voxel_data = archive["voxel_data"]
        surface_mask = archive["is_surface"]
        provenance = {
            key: _npz_scalar(archive, key)
            for key in required_archive_fields - {"voxel_data", "is_surface"}
        }

    expected_names = (
        "x",
        "y",
        "z",
        "youngs_modulus",
        "poissons_ratio",
        "density",
        "segment_id",
    )
    if (
        voxel_data.ndim != 1
        or len(voxel_data) != len(points)
        or voxel_data.dtype.names != expected_names
        or any(voxel_data.dtype.fields[name][0] != np.dtype("<f4") for name in expected_names[:6])
        or voxel_data.dtype.fields["segment_id"][0].kind != "U"
    ):
        raise VoxelizationError("material voxel_data schema is invalid")

    npz_points = np.column_stack(
        (voxel_data["x"], voxel_data["y"], voxel_data["z"])
    )
    if not np.array_equal(npz_points, points):
        raise VoxelizationError("PLY and material NPZ coordinates are not aligned")
    if (
        surface_mask.dtype != np.dtype(bool)
        or surface_mask.shape != (len(points),)
        or int(surface_mask.sum()) != provenance["surface_voxel_count"]
    ):
        raise VoxelizationError("material NPZ surface mask is invalid")

    if (
        provenance["grid_resolution"] != GRID_RESOLUTION
        or provenance["render_mesh_sha256"] != render_mesh.mesh_sha256
        or provenance["segment_keys_sha256"] != render_mesh.segment_keys_sha256
        or provenance["material_properties_sha256"] != material.fingerprint
        or provenance["voxel_ply_sha256"] != _sha256_file(ply_path)
        or type(provenance["surface_voxel_count"]) is not int
        or type(provenance["interior_voxel_count"]) is not int
        or type(provenance["winding_ambiguity_count"]) is not int
        or type(provenance["forced_segment_voxel_count"]) is not int
        or provenance["surface_voxel_count"] <= 0
        or provenance["interior_voxel_count"] < 0
        or provenance["surface_voxel_count"]
        + provenance["interior_voxel_count"]
        != len(points)
        or provenance["winding_ambiguity_count"] < 0
        or provenance["forced_segment_voxel_count"] < 0
    ):
        raise VoxelizationError("material NPZ metadata is invalid or stale")

    segment_ids = voxel_data["segment_id"].astype(str)
    if set(np.unique(segment_ids)) != set(render_mesh.segment_keys):
        raise VoxelizationError("material NPZ does not represent every exact segment")
    for index, key in enumerate(render_mesh.segment_keys):
        mask = segment_ids == key
        expected = material.properties[index].astype(np.float32)
        if (
            not np.all(voxel_data["youngs_modulus"][mask] == expected[0])
            or not np.all(voxel_data["poissons_ratio"][mask] == expected[1])
            or not np.all(voxel_data["density"][mask] == expected[2])
        ):
            raise VoxelizationError(f"material properties disagree for segment {key!r}")
    return len(points)


def voxel_outputs_are_complete(
    output_dir: str | Path,
    sha256: str,
    source_path: str,
    dataset: str,
    annotations: Mapping[str, MaterialRecord] | None = None,
) -> tuple[bool, int]:
    try:
        render_mesh = _load_render_mesh_info(output_dir, sha256, source_path, dataset)
        if annotations is None:
            annotations = load_material_annotations(
                Path(output_dir) / "material_annotations.json"
            )
        material = _materials_for_asset(render_mesh, annotations)
        voxel_dir = Path(output_dir).expanduser().resolve() / "voxels"
        count = _validate_voxel_pair(
            voxel_dir / f"{sha256}.ply",
            voxel_dir / f"{sha256}_with_materials.npz",
            render_mesh,
            material,
        )
        return True, count
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        KeyError,
        zipfile.BadZipFile,
        VoxelizationError,
    ):
        return False, 0


def _write_outputs(
    output_dir: Path,
    sha256: str,
    centers: np.ndarray,
    labels: np.ndarray,
    surface_mask: np.ndarray,
    statistics: Mapping[str, int],
    forced_segment_count: int,
    render_mesh: RenderMesh,
    material: MaterialRecord,
) -> int:
    voxel_dir = output_dir / "voxels"
    voxel_dir.mkdir(parents=True, exist_ok=True)
    ply_path = voxel_dir / f"{sha256}.ply"
    npz_path = voxel_dir / f"{sha256}_with_materials.npz"
    ply_descriptor, ply_name = tempfile.mkstemp(
        prefix=f".{sha256}.", suffix=".ply.part", dir=voxel_dir
    )
    npz_descriptor, npz_name = tempfile.mkstemp(
        prefix=f".{sha256}.", suffix=".npz.part", dir=voxel_dir
    )
    os.close(ply_descriptor)
    os.close(npz_descriptor)
    temporary_ply = Path(ply_name)
    temporary_npz = Path(npz_name)
    try:
        _write_voxel_ply(temporary_ply, centers)
        segment_width = max(len(key) for key in render_mesh.segment_keys)
        voxel_data = np.empty(
            len(centers),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("youngs_modulus", "<f4"),
                ("poissons_ratio", "<f4"),
                ("density", "<f4"),
                ("segment_id", f"<U{segment_width}"),
            ],
        )
        voxel_data["x"] = centers[:, 0]
        voxel_data["y"] = centers[:, 1]
        voxel_data["z"] = centers[:, 2]
        voxel_data["youngs_modulus"] = material.properties[labels, 0]
        voxel_data["poissons_ratio"] = material.properties[labels, 1]
        voxel_data["density"] = material.properties[labels, 2]
        voxel_data["segment_id"] = np.asarray(render_mesh.segment_keys)[labels]

        with temporary_npz.open("wb") as stream:
            np.savez_compressed(
                stream,
                voxel_data=voxel_data,
                is_surface=np.asarray(surface_mask, dtype=bool),
                grid_resolution=np.asarray(GRID_RESOLUTION, dtype=np.int16),
                render_mesh_sha256=np.asarray(render_mesh.mesh_sha256),
                segment_keys_sha256=np.asarray(render_mesh.segment_keys_sha256),
                material_properties_sha256=np.asarray(material.fingerprint),
                voxel_ply_sha256=np.asarray(_sha256_file(temporary_ply)),
                surface_voxel_count=np.asarray(
                    statistics["surface_voxel_count"], dtype=np.int32
                ),
                interior_voxel_count=np.asarray(
                    statistics["interior_voxel_count"], dtype=np.int32
                ),
                winding_ambiguity_count=np.asarray(
                    statistics["winding_ambiguity_count"], dtype=np.int32
                ),
                forced_segment_voxel_count=np.asarray(
                    forced_segment_count, dtype=np.int32
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())

        count = _validate_voxel_pair(
            temporary_ply, temporary_npz, render_mesh, material
        )
        os.replace(temporary_ply, ply_path)
        os.replace(temporary_npz, npz_path)
        directory = os.open(voxel_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _validate_voxel_pair(ply_path, npz_path, render_mesh, material)
        return count
    finally:
        temporary_ply.unlink(missing_ok=True)
        temporary_npz.unlink(missing_ok=True)


def _voxelize_one(
    row: Mapping[str, Any],
    output_dir: Path,
    annotations: Mapping[str, MaterialRecord],
    force: bool,
) -> dict[str, Any]:
    sha256 = str(row["sha256"])
    source_path = str(row["local_path"])
    dataset = str(row["dataset"])
    render_mesh = _load_render_mesh_info(
        output_dir, sha256, source_path, dataset
    )
    material = _materials_for_asset(render_mesh, annotations)
    ply_path = output_dir / "voxels" / f"{sha256}.ply"
    npz_path = output_dir / "voxels" / f"{sha256}_with_materials.npz"
    if not force and ply_path.is_file() and npz_path.is_file():
        try:
            count = _validate_voxel_pair(
                ply_path, npz_path, render_mesh, material
            )
            return {"sha256": sha256, "voxelized": True, "num_voxels": count}
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            zipfile.BadZipFile,
            VoxelizationError,
        ):
            pass

    vertices, triangles, triangle_segments = _read_render_mesh(render_mesh)
    occupied_indices, surface_mask, clamped_vertices, statistics = (
        _solid_voxel_indices(vertices, triangles)
    )
    centers = _indices_to_centers(occupied_indices).astype(np.float32)
    labels = _nearest_segment_labels(
        clamped_vertices,
        triangles,
        triangle_segments,
        centers,
    )
    forced_segment_count = _force_segment_coverage(
        clamped_vertices,
        triangles,
        triangle_segments,
        occupied_indices,
        labels,
        len(render_mesh.segment_keys),
    )
    count = _write_outputs(
        output_dir,
        sha256,
        centers,
        labels,
        surface_mask,
        statistics,
        forced_segment_count,
        render_mesh,
        material,
    )
    return {
        "sha256": sha256,
        "voxelized": True,
        "num_voxels": count,
    }


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pd.DataFrame.from_records(
            records, columns=["sha256", "voxelized", "num_voxels"]
        ).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_instances(value: str) -> set[str]:
    if "," in value:
        values = value.split(",")
    else:
        path = Path(value)
        values = (
            path.read_text(encoding="utf-8").splitlines()
            if path.is_file()
            else [value]
        )
    result = {item.strip() for item in values if item.strip()}
    if not result:
        raise VoxelizationError("--instances did not identify any assets")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize exact normalized render meshes into solid, material-labelled "
            "64^3 occupancy grids."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_file")
    parser.add_argument("--instances")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument(
        "--max_voxels",
        type=int,
        help="Deprecated and ignored; solid grids are never subsampled.",
    )
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    if (
        arguments.world_size <= 0
        or not 0 <= arguments.rank < arguments.world_size
        or arguments.max_workers <= 0
        or arguments.limit is not None
        and arguments.limit <= 0
    ):
        raise VoxelizationError("invalid rank, world size, worker count, or limit")

    output_dir = Path(arguments.output_dir).expanduser().resolve()
    metadata_path = (
        Path(arguments.metadata_file).expanduser().resolve()
        if arguments.metadata_file
        else output_dir / "metadata.csv"
    )
    annotation_path = output_dir / "material_annotations.json"
    if not metadata_path.is_file():
        raise VoxelizationError(f"metadata.csv is missing: {metadata_path}")
    if not annotation_path.is_file():
        raise VoxelizationError(
            f"material_annotations.json is missing: {annotation_path}"
        )

    metadata = pd.read_csv(metadata_path)
    required_columns = {"sha256", "local_path", "dataset"}
    if not required_columns <= set(metadata.columns):
        raise VoxelizationError(
            f"metadata.csv is missing columns: {sorted(required_columns - set(metadata.columns))}"
        )
    if arguments.instances:
        instances = _parse_instances(arguments.instances)
        metadata = metadata[metadata["sha256"].astype(str).isin(instances)]
        found = set(metadata["sha256"].astype(str))
        if found != instances:
            raise VoxelizationError(
                f"unknown instance hashes: {sorted(instances - found)}"
            )
    elif not arguments.metadata_file:
        rendered_hashes = {
            path.parent.name
            for path in (output_dir / "renders").glob("*/transforms.json")
        }
        metadata = metadata[metadata["sha256"].astype(str).isin(rendered_hashes)]

    metadata = metadata.sort_values("sha256", kind="stable").reset_index(drop=True)
    metadata = metadata.iloc[arguments.rank :: arguments.world_size]
    if arguments.limit is not None:
        metadata = metadata.head(arguments.limit)
    rows = metadata.to_dict("records")
    annotations = load_material_annotations(annotation_path)
    (output_dir / "voxels").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=arguments.max_workers) as executor:
        future_rows = {
            executor.submit(
                _voxelize_one,
                row,
                output_dir,
                annotations,
                arguments.force,
            ): row
            for row in rows
        }
        for future in tqdm(
            as_completed(future_rows),
            total=len(future_rows),
            desc=f"Voxelizing shard {arguments.rank}",
        ):
            row = future_rows[future]
            try:
                records.append(future.result())
            except Exception as exc:
                sha256 = str(row["sha256"])
                failures.append((sha256, exc))
                records.append(
                    {"sha256": sha256, "voxelized": False, "num_voxels": 0}
                )
                tqdm.write(
                    f"FAILED {sha256} {row['local_path']}: "
                    f"{type(exc).__name__}: {exc}"
                )

    records.sort(key=lambda record: record["sha256"])
    record_tag = f"{metadata_path.stem}_" if arguments.metadata_file else ""
    record_path = output_dir / f"voxelized_{record_tag}{arguments.rank}.csv"
    _write_records(record_path, records)
    succeeded = sum(record["voxelized"] is True for record in records)
    print(
        f"Shard {arguments.rank}: {succeeded} complete, {len(failures)} failed; "
        f"results: {record_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
