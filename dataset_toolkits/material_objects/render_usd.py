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

"""Render material objects from 150 viewpoints using the VLM renderer's USD loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_toolkits.material_objects.camera import (
    CAMERA_FOV,
    CAMERA_FRAME_FRACTION,
    CAMERA_NOMINAL_RADIUS,
    CAMERA_SENSOR_SIZE_MM,
    DEFAULT_NUM_VIEWS,
    RESOLUTION,
    _camera_frame,
    _camera_radius,
    _camera_samples,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.common import (
    extract_materials_from_usd as extract_common_materials,
    extract_simready_materials_from_usd as extract_simready_materials,
)
from dataset_toolkits.material_objects.vlm_annotations.asset_finder import DEFAULT_DATASET_ROOTS
from dataset_toolkits.material_objects.vlm_annotations import segment_textured_renderer as textured
import bpy
from mathutils import Matrix, Vector
from dataset_toolkits.material_objects.vlm_annotations.segment_visual_inputs import (
    build_asset_geometry_payloads,
)
from dataset_toolkits.material_objects.vlm_annotations.annotation_input_policy import (
    asset_annotation_exclusion,
)
from dataset_toolkits.datasets.allmats import get_metadata as get_asset_inventory


CYCLES_SAMPLES = 128
ALPHA_MASK_SAMPLES = 8
WORLD_COLOR = (0.18, 0.18, 0.18)
WORLD_STRENGTH = 0.8
AREA_LIGHT_STRENGTH = 40.0
AREA_LIGHT_SIZE = 4.0


class RenderError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_file_records(
    source: Path,
    geometry_payloads: Mapping[str, Any],
    plans: Mapping[str, Any],
) -> list[dict[str, str]]:
    records: dict[str, str] = {}

    def add(path_value: Any, sha256_value: Any) -> None:
        if not isinstance(path_value, str) or not isinstance(sha256_value, str):
            return
        path = Path(path_value).expanduser()
        if not path.is_absolute() or not path.is_file():
            return
        path = path.resolve()
        previous = records.get(str(path))
        if previous is not None:
            if previous != sha256_value:
                raise RenderError(f"conflicting input hashes for {path}")
            return
        actual = _sha256_file(path)
        if actual != sha256_value:
            raise RenderError(f"input hash changed while planning render: {path}")
        records[str(path)] = sha256_value

    records[str(source.resolve())] = _sha256_file(source)
    for payload in geometry_payloads.values():
        for record in payload.provenance.get("used_layers") or ():
            if isinstance(record, Mapping):
                add(record.get("real_path"), record.get("content_sha256"))
    for plan in plans.values():
        for texture in plan.textures:
            if texture.tiles:
                for tile in texture.tiles:
                    add(tile.path, tile.file_sha256)
            else:
                add(texture.path, texture.file_sha256)

        def walk(value: Any) -> None:
            if isinstance(value, Mapping):
                for path_key, hash_key in (
                    ("resolved_source_path", "source_sha256"),
                    ("resolved_path", "file_sha256"),
                    ("real_path", "content_sha256"),
                    ("path", "file_sha256"),
                ):
                    add(value.get(path_key), value.get(hash_key))
                for child in value.values():
                    walk(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    walk(child)

        walk(plan.model_parameters)
    return [
        {"path": path, "sha256": records[path]}
        for path in sorted(records)
    ]


def _manifest_inputs_are_current(manifest: Mapping[str, Any]) -> bool:
    records = manifest.get("input_files")
    if not isinstance(records, list) or not records:
        return False
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            return False
        path_value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            return False
        path = Path(path_value)
        if path_value in seen or not path.is_absolute() or not path.is_file():
            return False
        seen.add(path_value)
        if _sha256_file(path) != expected:
            return False
    return True


def _extract_asset(source: Path, dataset: str) -> dict[str, Any]:
    if dataset == "simready":
        result = extract_simready_materials(
            str(source), source.parent.name, str(source.parent)
        )
    else:
        result = extract_common_materials(str(source), dataset)
    if not isinstance(result, dict) or not result.get("segments"):
        raise RenderError(f"material extraction returned no segments for {source}")
    return result


def _asset_key(source: Path, dataset: str) -> str | None:
    root = DEFAULT_DATASET_ROOTS.get(dataset)
    if root is None:
        return None
    try:
        relative = source.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    return f"{dataset}/{relative}"


def _visible_bounds(
    segments: Mapping[str, Any], geometry_payloads: Mapping[str, Any]
) -> tuple[np.ndarray, float, np.ndarray]:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    occurrence_points: list[np.ndarray] = []

    for key in sorted(segments):
        if not textured._render_content(segments[key]):
            continue
        payload = geometry_payloads[key]
        points = np.asarray(payload.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            continue
        for transform in payload.occurrence_transforms:
            transformed = textured._transform_points(points, transform)
            minimum = np.minimum(minimum, transformed.min(axis=0))
            maximum = np.maximum(maximum, transformed.max(axis=0))
            occurrence_points.append(transformed)

    if not occurrence_points or not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise RenderError("asset has no finite visible render geometry")
    extent = maximum - minimum
    scale = float(extent.max())
    if not math.isfinite(scale) or scale <= 1.0e-12:
        raise RenderError("asset visible bounds are degenerate")
    center = (minimum + maximum) * 0.5
    normalized_points = (np.concatenate(occurrence_points, axis=0) - center) / scale
    return center, scale, normalized_points


def _write_normalized_mesh(
    output_dir: Path,
    renderables: list[Any],
    segments: Mapping[str, Any],
    center: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    segment_keys = sorted(
        key for key, segment in segments.items() if textured._render_content(segment)
    )
    segment_keys_sha256 = hashlib.sha256(
        json.dumps(
            segment_keys, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    segment_indices = {key: index for index, key in enumerate(segment_keys)}
    vertex_blocks = []
    triangle_blocks = []
    triangle_segment_blocks = []
    vertex_offset = 0
    for renderable in renderables:
        if len(renderable.face_segments) != len(renderable.mesh.polygons):
            raise RenderError("render mesh face ownership is incomplete")
        renderable.mesh.calc_loop_triangles()
        visible_triangles = []
        visible_triangle_segments = []
        for triangle in renderable.mesh.loop_triangles:
            owner = renderable.face_segments[triangle.polygon_index]
            if not textured._render_content(segments[owner]):
                continue
            visible_triangles.append(
                tuple(int(value) for value in triangle.vertices)
            )
            visible_triangle_segments.append(segment_indices[owner])
        triangles = np.asarray(visible_triangles, dtype=np.int64)
        if not len(triangles):
            continue
        triangles = triangles.reshape((-1, 3))
        triangle_segments = np.asarray(
            visible_triangle_segments, dtype="<i4"
        )
        referenced = np.unique(triangles)
        remap = np.full(len(renderable.mesh.vertices), -1, dtype=np.int64)
        remap[referenced] = np.arange(len(referenced), dtype=np.int64)
        local = np.empty((len(renderable.mesh.vertices), 3), dtype=np.float64)
        renderable.mesh.vertices.foreach_get("co", local.ravel())
        local = local[referenced]
        homogeneous = np.concatenate(
            (local, np.ones((len(local), 1), dtype=np.float64)), axis=1
        )
        for obj in renderable.objects:
            if vertex_offset + len(referenced) > np.iinfo(np.int32).max:
                raise RenderError("normalized render mesh exceeds the PLY index range")
            matrix = np.asarray(obj.matrix_world, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise RenderError("render occurrence has an invalid world transform")
            determinant = float(np.linalg.det(matrix[:3, :3]))
            if not math.isfinite(determinant) or abs(determinant) < 1.0e-15:
                raise RenderError("render occurrence has a singular world transform")
            world = homogeneous @ matrix.T
            weights = world[:, 3:4]
            if np.any(np.abs(weights) < 1.0e-14):
                raise RenderError("render occurrence maps geometry to w=0")
            normalized = (world[:, :3] / weights - center) / scale
            if not np.isfinite(normalized).all():
                raise RenderError("normalized render mesh contains non-finite vertices")
            if np.any(normalized < -0.500001) or np.any(normalized > 0.500001):
                raise RenderError("normalized render mesh exceeds its unit-cube bounds")
            vertex_blocks.append(np.asarray(normalized, dtype="<f4"))
            occurrence_triangles = remap[triangles] + vertex_offset
            if determinant < 0.0:
                occurrence_triangles = occurrence_triangles[:, (0, 2, 1)]
            triangle_blocks.append(np.asarray(occurrence_triangles, dtype="<i4"))
            triangle_segment_blocks.append(triangle_segments)
            vertex_offset += len(referenced)

    if not vertex_blocks or not triangle_blocks or not triangle_segment_blocks:
        raise RenderError("asset has no visible triangles to export")
    vertices = np.concatenate(vertex_blocks, axis=0)
    triangles = np.concatenate(triangle_blocks, axis=0)
    triangle_segments = np.concatenate(triangle_segment_blocks, axis=0)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment exact composed render geometry in normalized object coordinates\n"
        f"comment segment_keys_sha256 {segment_keys_sha256}\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\n"
        "property int segment_index\n"
        "end_header\n"
    ).encode("ascii")
    face_records = np.empty(
        len(triangles),
        dtype=np.dtype(
            [
                ("count", "u1"),
                ("indices", "<i4", (3,)),
                ("segment_index", "<i4"),
            ]
        ),
    )
    face_records["count"] = 3
    face_records["indices"] = triangles
    face_records["segment_index"] = triangle_segments
    temporary = output_dir / ".mesh.ply.part"
    with temporary.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
        face_records.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    destination = output_dir / "mesh.ply"
    os.replace(temporary, destination)
    record = {
        "file_path": "mesh.ply",
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "segment_keys": segment_keys,
        "segment_keys_sha256": segment_keys_sha256,
        "sha256": _sha256_file(destination),
    }
    _validate_normalized_mesh(destination, record)
    return record


def _validate_normalized_mesh(path: Path, record: Mapping[str, Any]) -> None:
    vertex_count = record.get("vertex_count")
    triangle_count = record.get("triangle_count")
    segment_keys = record.get("segment_keys")
    if (
        record.get("file_path") != "mesh.ply"
        or type(vertex_count) is not int
        or vertex_count < 3
        or type(triangle_count) is not int
        or triangle_count < 1
        or not isinstance(segment_keys, list)
        or not segment_keys
        or len(set(segment_keys)) != len(segment_keys)
        or not all(isinstance(key, str) and key for key in segment_keys)
        or segment_keys != sorted(segment_keys)
        or not isinstance(record.get("segment_keys_sha256"), str)
        or len(record["segment_keys_sha256"]) != 64
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
    ):
        raise RenderError("normalized mesh manifest is invalid")
    with path.open("rb") as stream:
        header = stream.read(65536)
    marker = b"end_header\n"
    marker_offset = header.find(marker)
    if marker_offset < 0:
        raise RenderError("normalized mesh has no complete PLY header")
    header_size = marker_offset + len(marker)
    segment_keys_sha256 = hashlib.sha256(
        json.dumps(
            segment_keys, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if record["segment_keys_sha256"] != segment_keys_sha256:
        raise RenderError("normalized mesh segment-key hash is invalid")
    expected_header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment exact composed render geometry in normalized object coordinates\n"
        f"comment segment_keys_sha256 {segment_keys_sha256}\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {triangle_count}\n"
        "property list uchar int vertex_indices\n"
        "property int segment_index\n"
        "end_header\n"
    ).encode("ascii")
    if header[:header_size] != expected_header:
        raise RenderError("normalized mesh PLY header does not match its manifest")
    expected_size = header_size + vertex_count * 12 + triangle_count * 17
    if path.stat().st_size != expected_size:
        raise RenderError("normalized mesh PLY payload has an invalid size")
    vertices = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=header_size,
        shape=(vertex_count, 3),
    )
    face_dtype = np.dtype(
        [
            ("count", "u1"),
            ("indices", "<i4", (3,)),
            ("segment_index", "<i4"),
        ]
    )
    faces = np.memmap(
        path,
        mode="r",
        dtype=face_dtype,
        offset=header_size + vertex_count * 12,
        shape=(triangle_count,),
    )
    if (
        not np.isfinite(vertices).all()
        or np.any(vertices < -0.500001)
        or np.any(vertices > 0.500001)
        or np.any(faces["count"] != 3)
        or int(faces["indices"].min()) < 0
        or int(faces["indices"].max()) >= vertex_count
        or int(faces["segment_index"].min()) < 0
        or int(faces["segment_index"].max()) >= len(segment_keys)
        or not np.array_equal(
            np.unique(faces["segment_index"]),
            np.arange(len(segment_keys), dtype=np.int32),
        )
    ):
        raise RenderError("normalized mesh PLY payload is invalid")
    del faces
    del vertices
    if _sha256_file(path) != record["sha256"]:
        raise RenderError("normalized mesh PLY hash does not match its manifest")


def _transparent_png_statistics(path: Path, size: int) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGBA" or image.size != (size, size):
            raise RenderError(
                f"invalid render {path}: {image.format}, {image.mode}, {image.size}"
            )
        rgba = np.asarray(image, dtype=np.uint8)
    alpha = rgba[:, :, 3]
    foreground = alpha > 0
    foreground_count = int(foreground.sum())
    border = np.concatenate(
        (alpha[0], alpha[-1], alpha[1:-1, 0], alpha[1:-1, -1])
    )
    if int(border.max()) != 0:
        raise RenderError(f"rendered geometry touches the image border: {path}")
    if foreground_count == 0:
        return {
            "foreground_pixels": 0,
            "alpha_bbox": None,
            "mean_foreground_rgb": None,
        }
    ys, xs = np.nonzero(foreground)
    foreground_rgb = rgba[:, :, :3][foreground]
    return {
        "foreground_pixels": foreground_count,
        "alpha_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "mean_foreground_rgb": [
            float(value) for value in foreground_rgb.mean(axis=0) / 255.0
        ],
    }


def _geometric_alpha_material(
    bpy: Any,
    plan: Any,
    image_cache: dict[tuple[str, str, str], Any],
    uv_aliases: Mapping[str, str],
) -> Any:
    material = bpy.data.materials.new(
        f"multiview_geometric_alpha_{plan.provenance_sha256[:12]}"
    )
    material.use_nodes = True
    material.use_backface_culling = not plan.double_sided
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    sockets = textured._texture_socket_map(
        bpy,
        nodes,
        links,
        plan,
        image_cache,
        None,
        uv_aliases,
        False,
    )
    opacity: Any
    if "opacity" in sockets:
        opacity = sockets["opacity"][1]
        if (
            plan.material_model == "principled"
            and not plan.omnipbr_controls
            and plan.opacity != 1.0
        ):
            opacity = textured._math_node(
                nodes, "MULTIPLY", opacity, plan.opacity
            )
        threshold = max(plan.opacity_threshold, 1.0e-6)
        opacity = textured._math_node(
            nodes,
            "SUBTRACT",
            1.0,
            textured._math_node(nodes, "LESS_THAN", opacity, threshold),
        )
    elif plan.material_model == "principled" and plan.display_opacity_attribute:
        attribute = nodes.new("ShaderNodeAttribute")
        attribute.attribute_name = plan.display_opacity_attribute
        threshold = max(plan.opacity_threshold, 1.0e-6)
        opacity = textured._math_node(
            nodes,
            "SUBTRACT",
            1.0,
            textured._math_node(
                nodes, "LESS_THAN", attribute.outputs["Fac"], threshold
            ),
        )
    else:
        if plan.opacity_threshold > 0.0:
            opacity = float(plan.opacity >= plan.opacity_threshold)
        else:
            opacity = float(plan.opacity > 0.0)

    surface = textured._opacity_shader(
        nodes, links, emission.outputs["Emission"], opacity, 0.0
    )
    surface = textured._one_sided_shader(
        nodes, links, surface, plan.double_sided
    )
    links.new(surface, output.inputs["Surface"])
    return material


def _geometric_alpha_segments(
    segments: Mapping[str, Any], plans: Mapping[str, Any]
) -> set[str]:
    result = set()
    for key, plan in plans.items():
        if not textured._render_content(segments[key]):
            continue
        has_opacity_texture = any(
            texture.role == "opacity" for texture in plan.textures
        )
        has_dynamic_opacity = has_opacity_texture or bool(
            plan.display_opacity_attribute
        )
        has_visible_opacity = has_dynamic_opacity or plan.opacity > 0.0
        has_transmission_texture = any(
            texture.role == "transmission_weight" for texture in plan.textures
        )
        has_dynamic_transmission = (
            plan.material_model == "omnisurface"
            and bool(plan.model_parameters.get("transmission_enabled"))
            and has_transmission_texture
        )
        if (
            (
                has_visible_opacity
                and (
                    plan.material_model == "omniglass"
                    or plan.transmission_weight > 0.0
                    or has_dynamic_transmission
                )
            )
            or (
                0.0 < plan.opacity < 1.0
                and plan.opacity_threshold == 0.0
                and not has_opacity_texture
                and not plan.display_opacity_attribute
            )
        ):
            result.add(key)
    return result


def _prepare_geometric_alpha_states(
    renderables: list[Any], segment_keys: set[str], materials: Mapping[str, Any]
) -> list[tuple[Any, tuple[int, ...], bool]]:
    states = []
    for renderable in renderables:
        mask_indices_by_segment = {}
        for key in sorted(renderable.segment_keys & segment_keys):
            mask_indices_by_segment[key] = len(renderable.mesh.materials)
            renderable.mesh.materials.append(materials[key])
        mask_indices = tuple(
            mask_indices_by_segment.get(owner, renderable.transparent_index)
            for owner in renderable.face_segments
        )
        visible = any(owner in segment_keys for owner in renderable.face_segments)
        states.append((renderable, mask_indices, visible))
    return states


def _set_whole_render_state(
    renderables: list[Any], segments: Mapping[str, Any]
) -> None:
    for renderable in renderables:
        if any(textured._render_content(segments[key]) for key in renderable.segment_keys):
            renderable.whole_state()
        else:
            for obj in renderable.objects:
                obj.hide_render = True


def _render_geometric_alpha(
    bpy: Any,
    scene: Any,
    states: list[tuple[Any, tuple[int, ...], bool]],
    destination: Path,
) -> None:
    for renderable, indices, visible in states:
        for polygon, material_index in zip(renderable.mesh.polygons, indices):
            polygon.material_index = material_index
        for obj in renderable.objects:
            obj.hide_render = not visible
    bpy.context.view_layer.update()
    previous_samples = scene.cycles.samples
    previous_denoising = scene.cycles.use_denoising
    previous_transparent_bounces = scene.cycles.transparent_max_bounces
    try:
        scene.cycles.samples = ALPHA_MASK_SAMPLES
        scene.cycles.use_denoising = False
        scene.cycles.transparent_max_bounces = 64
        scene.render.filepath = str(destination)
        result = bpy.ops.render.render(scene=scene.name, write_still=True)
        if result != {"FINISHED"}:
            raise RenderError(f"geometric-alpha render did not finish: {result!r}")
    finally:
        scene.cycles.samples = previous_samples
        scene.cycles.use_denoising = previous_denoising
        scene.cycles.transparent_max_bounces = previous_transparent_bounces


def _render_transmissive_backdrop(
    bpy: Any,
    scene: Any,
    renderables: list[Any],
    segments: Mapping[str, Any],
    destination: Path,
) -> None:
    _set_whole_render_state(renderables, segments)
    bpy.context.view_layer.update()
    previous_transparency = scene.render.film_transparent
    try:
        scene.render.film_transparent = False
        scene.render.filepath = str(destination)
        result = bpy.ops.render.render(scene=scene.name, write_still=True)
        if result != {"FINISHED"}:
            raise RenderError(f"transmissive-backdrop render did not finish: {result!r}")
    finally:
        scene.render.film_transparent = previous_transparency


def _apply_geometric_alpha(
    beauty_path: Path,
    backdrop_path: Path,
    mask_path: Path,
    size: int,
) -> None:
    with Image.open(beauty_path) as image:
        image.load()
        beauty = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    with Image.open(backdrop_path) as image:
        image.load()
        backdrop = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    with Image.open(mask_path) as image:
        image.load()
        mask = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if (
        beauty.shape != (size, size, 4)
        or backdrop.shape != beauty.shape
        or mask.shape != beauty.shape
    ):
        raise RenderError("beauty, backdrop, and geometric-alpha renders do not align")
    # Keep glass in the foreground mask even when its render alpha is low.
    selected = mask[:, :, 3] > beauty[:, :, 3]
    beauty[:, :, :3][selected] = backdrop[:, :, :3][selected]
    beauty[:, :, 3] = np.maximum(beauty[:, :, 3], mask[:, :, 3])
    combined = beauty_path.with_name(beauty_path.stem + ".combined.png")
    Image.fromarray(beauty, "RGBA").save(combined, format="PNG", compress_level=3)
    os.replace(combined, beauty_path)
    backdrop_path.unlink()
    mask_path.unlink()


def _configure_multiview_scene(bpy: Any, scene: Any) -> dict[str, Any]:
    runtime = textured._configure_cycles_gpu(bpy, scene)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.render.use_persistent_data = True

    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*WORLD_COLOR, 1.0)
    background.inputs["Strength"].default_value = WORLD_STRENGTH
    runtime["cycles_samples"] = CYCLES_SAMPLES
    return runtime


def _create_camera_and_lights(
    bpy: Any,
    collection: Any,
    center: np.ndarray,
    raw_scale: float,
):
    camera_data = bpy.data.cameras.new("multiview_camera")
    camera_data.type = "PERSP"
    camera_data.sensor_width = CAMERA_SENSOR_SIZE_MM
    camera_data.sensor_height = CAMERA_SENSOR_SIZE_MM
    camera_data.lens = 0.5 * CAMERA_SENSOR_SIZE_MM / math.tan(CAMERA_FOV * 0.5)
    camera = bpy.data.objects.new("multiview_camera", camera_data)
    collection.objects.link(camera)

    lights = []
    axes = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    for index, axis in enumerate(axes):
        light_data = bpy.data.lights.new(f"uniform_area_{index}", "AREA")
        light_data.shape = "DISK"
        light_data.energy = AREA_LIGHT_STRENGTH * raw_scale * raw_scale
        light_data.size = AREA_LIGHT_SIZE * raw_scale
        light_data.use_shadow = False
        light = bpy.data.objects.new(f"uniform_area_{index}", light_data)
        collection.objects.link(light)
        location = center + np.asarray(axis, dtype=np.float64) * raw_scale * 3.0
        light.location = tuple(float(value) for value in location)
        light.rotation_euler = Vector(tuple(center - location)).to_track_quat("-Z", "Y").to_euler()
        lights.append(light)
    return camera, lights


def _render_asset_in_process(
    source: Path,
    dataset: str,
    asset_id: str,
    output_dir: Path,
    num_views: int,
    size: int,
) -> None:
    extraction = _extract_asset(source, dataset)
    geometry_payloads = build_asset_geometry_payloads(source, extraction)
    segments, materials, payload_hashes = textured._validate_payloads(
        extraction, geometry_payloads
    )
    subdivision_level = textured._asset_catmull_subdivision_level(geometry_payloads)
    up_axes = {
        str(payload.provenance.get("stage_up_axis") or "").upper()
        for payload in geometry_payloads.values()
    }
    if len(up_axes) != 1 or not up_axes <= {"Y", "Z"}:
        raise RenderError(f"inconsistent stage up axes: {sorted(up_axes)!r}")
    up_axis = next(iter(up_axes))
    plans = {
        key: textured._build_material_plan(segments[key], payload, materials)
        for key, payload in geometry_payloads.items()
    }
    uv_aliases = {key: textured._uv_aliases(plan) for key, plan in plans.items()}
    center, raw_scale, normalized_points = _visible_bounds(
        segments, geometry_payloads
    )

    if tuple(bpy.app.version[:2]) != (4, 2):
        raise RenderError(f"expected bpy 4.2.x, found {tuple(bpy.app.version)}")

    before = textured._snapshot_data_names(bpy)
    previous_scene = bpy.context.window.scene if bpy.context.window else None
    try:
        scene, collection = textured._new_scene(bpy, size)
        if bpy.context.window:
            bpy.context.window.scene = scene
        runtime = _configure_multiview_scene(bpy, scene)
        image_cache: dict[tuple[str, str, str], Any] = {}
        blender_materials = {
            key: textured._build_blender_material(
                bpy, plan, image_cache, uv_aliases=uv_aliases[key]
            )
            for key, plan in plans.items()
        }
        transparent = textured._transparent_material(bpy)
        renderables = textured._build_asset_renderables(
            bpy,
            Matrix,
            collection,
            segments,
            geometry_payloads,
            plans,
            blender_materials,
            transparent,
            uv_aliases,
            subdivision_level,
        )
        mesh_record = _write_normalized_mesh(
            output_dir,
            renderables,
            segments,
            center,
            raw_scale,
        )
        _set_whole_render_state(renderables, segments)
        geometric_alpha_segments = _geometric_alpha_segments(segments, plans)
        geometric_alpha_materials = {
            key: _geometric_alpha_material(
                bpy,
                plans[key],
                image_cache,
                uv_aliases[key],
            )
            for key in geometric_alpha_segments
        }
        geometric_alpha_states = (
            _prepare_geometric_alpha_states(
                renderables,
                geometric_alpha_segments,
                geometric_alpha_materials,
            )
            if geometric_alpha_segments
            else []
        )

        camera, _ = _create_camera_and_lights(bpy, collection, center, raw_scale)
        scene.camera = camera
        camera.data.clip_start = max(raw_scale * 1.0e-4, 1.0e-6)
        camera.data.clip_end = max(raw_scale * 8.0, camera.data.clip_start * 1000.0)

        frames = []
        frame_statistics = []
        for index, yaw, pitch, direction in _camera_samples(
            asset_id, num_views, up_axis
        ):
            horizontal, vertical = _camera_frame(direction, up_axis)
            radius = _camera_radius(
                normalized_points, direction, horizontal, vertical
            )
            raw_location = center + direction * radius * raw_scale
            camera.matrix_world = Matrix(
                (
                    (horizontal[0], vertical[0], direction[0], raw_location[0]),
                    (horizontal[1], vertical[1], direction[1], raw_location[1]),
                    (horizontal[2], vertical[2], direction[2], raw_location[2]),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
            bpy.context.view_layer.update()

            filename = f"{index:03d}.png"
            destination = output_dir / filename
            temporary = output_dir / f".{index:03d}.part.png"
            scene.render.filepath = str(temporary)
            result = bpy.ops.render.render(scene=scene.name, write_still=True)
            if result != {"FINISHED"}:
                raise RenderError(f"Cycles did not finish view {index}: {result!r}")
            if geometric_alpha_states:
                backdrop_temporary = output_dir / f".{index:03d}.backdrop.part.png"
                alpha_temporary = output_dir / f".{index:03d}.alpha.part.png"
                try:
                    _render_transmissive_backdrop(
                        bpy,
                        scene,
                        renderables,
                        segments,
                        backdrop_temporary,
                    )
                    _render_geometric_alpha(
                        bpy,
                        scene,
                        geometric_alpha_states,
                        alpha_temporary,
                    )
                    _apply_geometric_alpha(
                        temporary,
                        backdrop_temporary,
                        alpha_temporary,
                        size,
                    )
                finally:
                    _set_whole_render_state(renderables, segments)
                    bpy.context.view_layer.update()
                    backdrop_temporary.unlink(missing_ok=True)
                    alpha_temporary.unlink(missing_ok=True)
            statistics = _transparent_png_statistics(temporary, size)
            os.replace(temporary, destination)
            frame_statistics.append(statistics)

            normalized_location = (raw_location - center) / raw_scale
            normalized_c2w = [
                [
                    float(horizontal[0]),
                    float(vertical[0]),
                    float(direction[0]),
                    float(normalized_location[0]),
                ],
                [
                    float(horizontal[1]),
                    float(vertical[1]),
                    float(direction[1]),
                    float(normalized_location[1]),
                ],
                [
                    float(horizontal[2]),
                    float(vertical[2]),
                    float(direction[2]),
                    float(normalized_location[2]),
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
            frames.append(
                {
                    "file_path": filename,
                    "camera_angle_x": CAMERA_FOV,
                    "transform_matrix": normalized_c2w,
                    "yaw": yaw,
                    "pitch": pitch,
                    "radius": radius,
                }
            )

        input_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "entrypoint": _sha256_file(source),
                    "geometry": payload_hashes,
                    "materials": {
                        key: plans[key].provenance_sha256 for key in sorted(plans)
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        input_files = _input_file_records(source, geometry_payloads, plans)
        visible_statistics = [
            record
            for record in frame_statistics
            if record["mean_foreground_rgb"] is not None
        ]
        if not visible_statistics:
            raise RenderError("asset is transparent from every sampled viewpoint")
        focal = 0.5 / math.tan(CAMERA_FOV * 0.5)
        manifest = {
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "scale": 1.0 / raw_scale,
            "offset": [float(value) for value in -center / raw_scale],
            "normalization": {
                "source_center": [float(value) for value in center],
                "source_scale": raw_scale,
            },
            "mesh": mesh_record,
            "camera_intrinsics": {
                "normalized": [
                    [focal, 0.0, 0.5],
                    [0.0, focal, 0.5],
                    [0.0, 0.0, 1.0],
                ],
                "pixels": [
                    [focal * size, 0.0, size * 0.5],
                    [0.0, focal * size, size * 0.5],
                    [0.0, 0.0, 1.0],
                ],
            },
            "render_settings": {
                "resolution": size,
                "num_views": num_views,
                "engine": "CYCLES",
                "blender_version": list(bpy.app.version),
                "gpu_backend": runtime["render_backend"],
                "gpu_device": list(runtime["render_device"]),
                "cycles_samples": CYCLES_SAMPLES,
                "adaptive_sampling": False,
                "denoising": True,
                "pixel_filter_type": "BOX",
                "pixel_filter_width": 1.0,
                "diffuse_bounces": 1,
                "glossy_bounces": 1,
                "transparent_bounces": 3,
                "transmission_bounces": 3,
                "camera_fov_degrees": 40.0,
                "camera_nominal_radius": CAMERA_NOMINAL_RADIUS,
                "camera_fit_fraction": CAMERA_FRAME_FRACTION,
                "transparent_background": True,
                "alpha_mask_samples": ALPHA_MASK_SAMPLES,
                "geometric_alpha_segment_count": len(geometric_alpha_segments),
                "color_transform": "Filmic",
                "world_color": list(WORLD_COLOR),
                "world_strength": WORLD_STRENGTH,
                "area_light_count": 6,
                "area_light_strength": AREA_LIGHT_STRENGTH,
                "area_light_size": AREA_LIGHT_SIZE,
                "light_shadows": False,
            },
            "source_entrypoint": str(source),
            "source_entrypoint_sha256": _sha256_file(source),
            "input_fingerprint": input_fingerprint,
            "input_files": input_files,
            "stage_up_axis": up_axis,
            "foreground_rgb_range": {
                "minimum_mean": [
                    min(record["mean_foreground_rgb"][channel] for record in visible_statistics)
                    for channel in range(3)
                ],
                "maximum_mean": [
                    max(record["mean_foreground_rgb"][channel] for record in visible_statistics)
                    for channel in range(3)
                ],
            },
            "empty_frame_count": len(frame_statistics) - len(visible_statistics),
            "frames": frames,
        }
        temporary_manifest = output_dir / ".transforms.json.part"
        with temporary_manifest.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_manifest, output_dir / "transforms.json")
    finally:
        if bpy.context.window and previous_scene is not None:
            bpy.context.window.scene = previous_scene
        textured._cleanup_new_data(bpy, before)


def _validate_complete_render(
    output_dir: Path, num_views: int, size: int, *, decode: bool
) -> dict[str, Any]:
    manifest_path = output_dir / "transforms.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    settings = manifest.get("render_settings")
    frames = manifest.get("frames")
    mesh_record = manifest.get("mesh")
    if (
        not isinstance(settings, dict)
        or settings.get("resolution") != size
        or settings.get("num_views") != num_views
        or settings.get("engine") != "CYCLES"
        or settings.get("cycles_samples") != CYCLES_SAMPLES
        or settings.get("adaptive_sampling") is not False
        or settings.get("denoising") is not True
        or settings.get("pixel_filter_type") != "BOX"
        or settings.get("pixel_filter_width") != 1.0
        or settings.get("diffuse_bounces") != 1
        or settings.get("glossy_bounces") != 1
        or settings.get("transparent_bounces") != 3
        or settings.get("transmission_bounces") != 3
        or settings.get("camera_fov_degrees") != 40.0
        or settings.get("camera_nominal_radius") != CAMERA_NOMINAL_RADIUS
        or settings.get("camera_fit_fraction") != CAMERA_FRAME_FRACTION
        or settings.get("transparent_background") is not True
        or settings.get("alpha_mask_samples") != ALPHA_MASK_SAMPLES
        or settings.get("color_transform") != "Filmic"
        or settings.get("world_color") != list(WORLD_COLOR)
        or settings.get("world_strength") != WORLD_STRENGTH
        or settings.get("area_light_count") != 6
        or settings.get("area_light_strength") != AREA_LIGHT_STRENGTH
        or settings.get("area_light_size") != AREA_LIGHT_SIZE
        or settings.get("light_shadows") is not False
        or not isinstance(mesh_record, dict)
        or not isinstance(frames, list)
        or len(frames) != num_views
    ):
        raise RenderError("render manifest does not match the active renderer")

    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict):
        raise RenderError("render manifest has no normalization parameters")
    center = np.asarray(normalization.get("source_center"), dtype=np.float64)
    source_scale = float(normalization.get("source_scale"))
    offset = np.asarray(manifest.get("offset"), dtype=np.float64)
    scale = float(manifest.get("scale"))
    if (
        manifest.get("aabb") != [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        or center.shape != (3,)
        or offset.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(offset).all()
        or not math.isfinite(source_scale)
        or source_scale <= 0.0
        or not math.isfinite(scale)
        or not math.isclose(scale, 1.0 / source_scale, rel_tol=0.0, abs_tol=1.0e-12)
        or not np.allclose(
            offset,
            -center / source_scale,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise RenderError("render manifest normalization values are inconsistent")
    _validate_normalized_mesh(output_dir / "mesh.ply", mesh_record)

    expected_paths = [f"{index:03d}.png" for index in range(num_views)]
    if [frame.get("file_path") for frame in frames] != expected_paths:
        raise RenderError("render manifest has missing or reordered frame paths")
    for frame, relative_path in zip(frames, expected_paths):
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise RenderError(f"invalid camera matrix for {relative_path}")
        if not math.isclose(
            float(frame.get("camera_angle_x")), CAMERA_FOV, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RenderError(f"invalid camera FOV for {relative_path}")
        image_path = output_dir / relative_path
        if not image_path.is_file() or image_path.stat().st_size < 128:
            raise RenderError(f"missing rendered frame {image_path}")
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGBA" or image.size != (size, size):
                raise RenderError(f"invalid PNG frame {image_path}")
            if decode:
                image.load()
            else:
                image.verify()
    return manifest


def _publish_render(staging: Path, destination: Path, num_views: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    completion_marker = destination / "transforms.json"
    if completion_marker.exists():
        completion_marker.unlink()
    os.replace(staging / "mesh.ply", destination / "mesh.ply")
    for index in range(num_views):
        filename = f"{index:03d}.png"
        os.replace(staging / filename, destination / filename)
    os.replace(staging / "transforms.json", completion_marker)
    directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _render_one(
    row: Mapping[str, Any],
    output_root: Path,
    num_views: int,
    size: int,
    force: bool,
    quiet: bool,
) -> dict[str, Any]:
    asset_id = str(row["sha256"])
    dataset = str(row.get("dataset") or "simready")
    source = Path(str(row["local_path"])).expanduser().resolve()
    if not source.is_file():
        raise RenderError(f"source asset is missing: {source}")
    key = _asset_key(source, dataset)
    if key is not None and asset_annotation_exclusion(key) is not None:
        return {"sha256": asset_id, "rendered": False, "dataset": dataset, "skipped": True}

    destination = output_root / "renders" / asset_id
    if not force and (destination / "transforms.json").is_file():
        try:
            manifest = _validate_complete_render(
                destination, num_views, size, decode=False
            )
            if (
                manifest.get("source_entrypoint_sha256") == _sha256_file(source)
                and _manifest_inputs_are_current(manifest)
            ):
                return {"sha256": asset_id, "rendered": True, "dataset": dataset}
        except (AttributeError, OSError, TypeError, ValueError, RenderError):
            pass

    staging_parent = output_root / "renders"
    staging = Path(tempfile.mkdtemp(prefix=f".{asset_id}.render.", dir=staging_parent))
    try:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source",
            str(source),
            "--dataset",
            dataset,
            "--asset-id",
            asset_id,
            "--worker-output",
            str(staging),
            "--num_views",
            str(num_views),
            "--resolution",
            str(size),
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE if quiet else None,
            stderr=subprocess.STDOUT if quiet else None,
        )
        if completed.returncode:
            detail = (completed.stdout or "")[-12000:].strip()
            raise RenderError(
                f"renderer process exited {completed.returncode} for {source}"
                + (f"\n{detail}" if detail else "")
            )
        _validate_complete_render(staging, num_views, size, decode=True)
        _publish_render(staging, destination, num_views)
        _validate_complete_render(destination, num_views, size, decode=False)
        return {"sha256": asset_id, "rendered": True, "dataset": dataset}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _run_batch(arguments: argparse.Namespace) -> int:
    if arguments.dataset_type not in {
        "allmats",
        "simready",
        "commercial",
        "residential",
        "vegetation",
    }:
        raise RenderError(f"unknown dataset type {arguments.dataset_type!r}")
    if arguments.num_views <= 0 or arguments.resolution != RESOLUTION:
        raise RenderError("num_views must be positive and resolution must be 512")
    if arguments.max_workers != 1:
        raise RenderError("exact Cycles rendering requires max_workers=1 per GPU shard")
    if not 0 <= arguments.rank < arguments.world_size:
        raise RenderError("rank must be in [0, world_size)")

    output_root = Path(arguments.output_dir).expanduser().resolve()
    (output_root / "renders").mkdir(parents=True, exist_ok=True)
    included = (
        "simready,commercial,residential,vegetation"
        if arguments.dataset_type == "allmats"
        else arguments.dataset_type
    )
    metadata = get_asset_inventory(
        output_dir=None,
        include_datasets=included,
        skip_split=True,
    )
    if arguments.instances:
        instance_path = Path(arguments.instances)
        instances = (
            instance_path.read_text(encoding="utf-8").splitlines()
            if instance_path.is_file()
            else arguments.instances.split(",")
        )
        metadata = metadata[metadata["sha256"].astype(str).isin(instances)]

    metadata = metadata.sort_values("sha256", kind="stable").reset_index(drop=True)
    metadata = metadata.iloc[arguments.rank :: arguments.world_size]
    rows = metadata.to_dict("records")
    records = []
    failures = []
    for row in tqdm(rows, desc=f"Rendering shard {arguments.rank}"):
        try:
            record = _render_one(
                row,
                output_root,
                arguments.num_views,
                arguments.resolution,
                arguments.force,
                arguments.quiet,
            )
        except Exception as exc:
            record = {
                "sha256": str(row["sha256"]),
                "rendered": False,
                "dataset": str(row.get("dataset") or "simready"),
            }
            failures.append((row, exc))
            tqdm.write(
                f"FAILED {row['sha256']} {row['local_path']}: "
                f"{type(exc).__name__}: {exc}"
            )
        records.append(record)

    records.sort(key=lambda record: record["sha256"])
    result_path = output_root / f"rendered_{arguments.rank}.csv"
    pd.DataFrame.from_records(records).to_csv(result_path, index=False)
    succeeded = sum(record["rendered"] is True for record in records)
    skipped = sum(record.get("skipped") is True for record in records)
    print(
        f"Shard {arguments.rank}: {succeeded} complete, {skipped} intentionally skipped, "
        f"{len(failures)} failed; results: {result_path}"
    )
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render all material USD assets with exact textures and cameras."
    )
    parser.add_argument("dataset_type", nargs="?")
    parser.add_argument("--output_dir")
    parser.add_argument("--num_views", type=int, default=DEFAULT_NUM_VIEWS)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--instances")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument("--asset-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.worker:
        if not all(
            (arguments.source, arguments.dataset, arguments.asset_id, arguments.worker_output)
        ):
            raise RenderError("worker arguments are incomplete")
        _render_asset_in_process(
            Path(arguments.source).resolve(),
            arguments.dataset,
            arguments.asset_id,
            Path(arguments.worker_output).resolve(),
            arguments.num_views,
            arguments.resolution,
        )
        return 0
    if not arguments.dataset_type or not arguments.output_dir:
        raise RenderError("dataset_type and --output_dir are required")
    return _run_batch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
