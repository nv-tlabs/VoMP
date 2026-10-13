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

"""Training render/voxel preprocessing shared by the mesh API and figure runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import shutil
import zipfile

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_toolkits.dino import render_hash
from dataset_toolkits.material_objects import render_usd
from dataset_toolkits.material_objects.camera import DEFAULT_CAMERA_SEED, resolve_camera_id
import bpy
from mathutils import Matrix
from plyfile import PlyData
from dataset_toolkits.material_objects.vlm_annotations import segment_textured_renderer as textured
from dataset_toolkits.material_objects.voxelize import (
    _force_segment_coverage,
    _indices_to_centers,
    _nearest_segment_labels,
    _solid_voxel_indices,
    _write_voxel_ply,
)


def render_mesh_in_process(
    mesh_path: Path, output_dir: Path, asset_id: str, dataset: str = "simready"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mesh_path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
        render_usd._render_asset_in_process(
            mesh_path.resolve(), dataset, asset_id, output_dir,
            render_usd.DEFAULT_NUM_VIEWS, render_usd.RESOLUTION,
        )
        manifest = json.loads((output_dir / "transforms.json").read_text())
        mesh = PlyData.read(str(output_dir / "mesh.ply"))
        np.savez_compressed(
            output_dir / "normalized_geometry.npz",
            vertices=np.column_stack([mesh["vertex"][axis] for axis in "xyz"]),
            triangles=np.stack(mesh["face"]["vertex_indices"]).astype(np.int32),
            triangle_segments=np.asarray(mesh["face"]["segment_index"], dtype=np.int32),
            segment_keys=np.asarray(manifest["mesh"]["segment_keys"]),
        )
        _stamp_render_source(mesh_path, output_dir, asset_id)
        return

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = render_usd.RESOLUTION
    scene.render.resolution_y = render_usd.RESOLUTION
    scene.render.resolution_percentage = 100
    if scene.world is None:
        scene.world = bpy.data.worlds.new("preprocessing_world")
    scene.world.use_nodes = True
    runtime = render_usd._configure_multiview_scene(bpy, scene)

    extension = mesh_path.suffix.lower()
    filepath = str(mesh_path.resolve())
    if extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=filepath, import_shading="NORMALS")
    elif extension == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif extension == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    elif extension == ".stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    else:
        raise ValueError(f"Unsupported mesh format: {extension}")
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    mesh_objects = [obj for obj in scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"Asset contains no mesh geometry: {mesh_path}")
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            transmission = node.inputs.get("Transmission Weight")
            alpha = node.inputs.get("Alpha")
            if (transmission and (transmission.is_linked or transmission.default_value > 0)) or (
                alpha and not alpha.is_linked and 0 < alpha.default_value < 1
            ):
                raise ValueError(
                    "Transmissive/semitransparent mesh materials need the exact USD "
                    "renderer's geometric-alpha handling. Supply the source USD."
                )

    vertices_blocks: list[np.ndarray] = []
    triangle_blocks: list[np.ndarray] = []
    vertex_offset = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in mesh_objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
            local = np.asarray([tuple(vertex.co) for vertex in mesh.vertices])
            homogeneous = np.column_stack((local, np.ones(len(local))))
            world = (homogeneous @ matrix.T)[:, :3]
            triangles = np.asarray(
                [tuple(triangle.vertices) for triangle in mesh.loop_triangles],
                dtype=np.int64,
            )
            # Mirrored transforms flip face winding, so swap it back.
            if np.linalg.det(matrix[:3, :3]) < 0:
                triangles = triangles[:, (0, 2, 1)]
            vertices_blocks.append(world)
            triangle_blocks.append(triangles + vertex_offset)
            vertex_offset += len(world)
        finally:
            evaluated.to_mesh_clear()

    raw_vertices = np.concatenate(vertices_blocks).astype(np.float64)
    triangles = np.concatenate(triangle_blocks).astype(np.int64)
    minimum = raw_vertices.min(axis=0)
    maximum = raw_vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    source_scale = float((maximum - minimum).max())
    if not math.isfinite(source_scale) or source_scale <= 1.0e-12:
        raise RuntimeError(f"mesh bounds are degenerate: {mesh_path}")
    vertices = (raw_vertices - center) / source_scale
    if (
        not np.isfinite(vertices).all()
        or vertices.min() < -0.50001
        or vertices.max() > 0.50001
    ):
        raise RuntimeError(f"normalized mesh is outside the unit cube: {mesh_path}")
    np.savez_compressed(
        output_dir / "normalized_geometry.npz",
        vertices=vertices.astype(np.float32),
        triangles=triangles.astype(np.int32),
    )

    camera, _ = render_usd._create_camera_and_lights(
        bpy, scene.collection, center, source_scale
    )
    scene.camera = camera
    camera.data.clip_start = max(source_scale * 1.0e-4, 1.0e-6)
    camera.data.clip_end = max(
        source_scale * 8.0, camera.data.clip_start * 1000.0
    )

    frames = []
    for index, yaw, pitch, direction in render_usd._camera_samples(
        asset_id, render_usd.DEFAULT_NUM_VIEWS, "Z"
    ):
        horizontal, vertical = textured._camera_frame(direction, "Z")
        radius = render_usd._camera_radius(
            vertices, direction, horizontal, vertical
        )
        raw_location = center + direction * radius * source_scale
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
        temporary = output_dir / f".{index:03d}.part.png"
        destination = output_dir / filename
        scene.render.filepath = str(temporary)
        if bpy.ops.render.render(scene=scene.name, write_still=True) != {"FINISHED"}:
            raise RuntimeError(f"Cycles failed to render {mesh_path} view {index}")
        render_usd._transparent_png_statistics(
            temporary, render_usd.RESOLUTION
        )
        os.replace(temporary, destination)
        normalized_location = (raw_location - center) / source_scale
        frames.append(
            {
                "file_path": filename,
                "camera_angle_x": render_usd.CAMERA_FOV,
                "transform_matrix": [
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
                ],
                "yaw": yaw,
                "pitch": pitch,
                "radius": radius,
            }
        )

    with (output_dir / "transforms.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                "scale": 1.0 / source_scale,
                "offset": [float(value) for value in -center / source_scale],
                "normalization": {
                    "source_center": [float(value) for value in center],
                    "source_scale": source_scale,
                },
                "frames": frames,
                "render_settings": {
                    "num_views": render_usd.DEFAULT_NUM_VIEWS,
                    "resolution": render_usd.RESOLUTION,
                    "engine": "CYCLES",
                    "blender_version": list(bpy.app.version),
                    "gpu_backend": runtime.get("render_backend"),
                    "gpu_device": runtime.get("render_device"),
                    "cycles_samples": render_usd.CYCLES_SAMPLES,
                    "adaptive_sampling": False,
                    "denoising": True,
                    "pixel_filter_type": "BOX",
                    "pixel_filter_width": 1.0,
                    "diffuse_bounces": 1,
                    "glossy_bounces": 1,
                    "transparent_bounces": 3,
                    "transmission_bounces": 3,
                    "camera_fov_degrees": math.degrees(render_usd.CAMERA_FOV),
                    "camera_nominal_radius": render_usd.CAMERA_NOMINAL_RADIUS,
                    "camera_fit_fraction": render_usd.CAMERA_FRAME_FRACTION,
                    "transparent_background": True,
                    "color_transform": "Filmic",
                    "world_color": list(render_usd.WORLD_COLOR),
                    "world_strength": render_usd.WORLD_STRENGTH,
                    "area_light_count": 6,
                    "area_light_strength": render_usd.AREA_LIGHT_STRENGTH,
                    "area_light_size": render_usd.AREA_LIGHT_SIZE,
                    "light_shadows": False,
                },
                "stage_up_axis": "Z",
            },
            stream,
            indent=2,
        )
    _stamp_render_source(mesh_path, output_dir, asset_id)


def _stamp_render_source(source: Path, render_dir: Path, asset_id: str) -> None:
    manifest_path = render_dir / "transforms.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        inference_source=str(source.resolve()),
        inference_source_sha256=sha256_file(source),
        inference_asset_id=asset_id,
        inference_render_contract=render_contract(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampling_seed(seed: int | None = None) -> int:
    """Use the fixed inference seed unless the caller supplies an override."""
    return int(seed) if seed is not None else DEFAULT_CAMERA_SEED


def render_contract() -> str:
    paths = (
        Path(__file__), Path(__file__).with_name("render_usd.py"),
        Path(__file__).with_name("camera.py"),
        Path(__file__).parent / "vlm_annotations" / "segment_textured_renderer.py",
    )
    return hashlib.sha256("".join(sha256_file(path) for path in paths).encode()).hexdigest()


def validate_render_outputs(render_dir: Path) -> None:
    geometry_path = render_dir / "normalized_geometry.npz"
    with np.load(geometry_path) as archive:
        vertices = archive["vertices"]
        triangles = archive["triangles"]
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not len(triangles)
        or not np.isfinite(vertices).all()
        or np.any(triangles < 0)
        or np.any(triangles >= len(vertices))
    ):
        raise ValueError("rendered normalized geometry is invalid")

    with (render_dir / "transforms.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    frames = manifest.get("frames")
    if manifest.get("inference_render_contract") != render_contract():
        raise ValueError("Render code changed; regenerate cached renders")
    settings = manifest.get("render_settings") or {}
    if tuple(settings.get("blender_version", ())[:2]) != (4, 2):
        raise ValueError("Training rendering requires Blender 4.2.x")
    expected_settings = {
        "num_views": render_usd.DEFAULT_NUM_VIEWS,
        "resolution": render_usd.RESOLUTION,
        "cycles_samples": render_usd.CYCLES_SAMPLES,
        "engine": "CYCLES",
        "adaptive_sampling": False,
        "denoising": True,
        "pixel_filter_type": "BOX",
        "pixel_filter_width": 1.0,
        "diffuse_bounces": 1,
        "glossy_bounces": 1,
        "transparent_bounces": 3,
        "transmission_bounces": 3,
        "transparent_background": True,
        "color_transform": "Filmic",
        "light_shadows": False,
        "camera_fov_degrees": math.degrees(render_usd.CAMERA_FOV),
        "camera_nominal_radius": render_usd.CAMERA_NOMINAL_RADIUS,
        "camera_fit_fraction": render_usd.CAMERA_FRAME_FRACTION,
        "world_color": list(render_usd.WORLD_COLOR),
        "world_strength": render_usd.WORLD_STRENGTH,
        "area_light_count": 6,
        "area_light_strength": render_usd.AREA_LIGHT_STRENGTH,
        "area_light_size": render_usd.AREA_LIGHT_SIZE,
    }
    if any(settings.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("render manifest settings do not match training preprocessing")
    expected_names = [
        f"{index:03d}.png" for index in range(render_usd.DEFAULT_NUM_VIEWS)
    ]
    if not isinstance(frames, list) or [frame.get("file_path") for frame in frames] != expected_names:
        raise ValueError("render manifest does not contain all 150 ordered views")
    for frame, name in zip(frames, expected_names):
        transform = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"render manifest has an invalid camera transform: {name}")
        image_path = render_dir / name
        with Image.open(image_path) as image:
            if image.mode != "RGBA" or image.size != (
                render_usd.RESOLUTION,
                render_usd.RESOLUTION,
            ):
                raise ValueError(f"render has the wrong format: {image_path}")
            image.verify()


def _self_contained_glb(source: Path) -> bool:
    """Only cache GLBs whose buffers and images are embedded in the source file."""
    if source.suffix.lower() != ".glb":
        return False
    try:
        with source.open("rb") as stream:
            magic, version, size = struct.unpack("<III", stream.read(12))
            length, kind = struct.unpack("<II", stream.read(8))
            if (magic, version, kind) != (0x46546C67, 2, 0x4E4F534A):
                return False
            if size != source.stat().st_size or not 0 < length <= size - 20:
                return False
            document = json.loads(stream.read(length))
        return all(
            isinstance(document.get(group, []), list)
            and all(
                isinstance(item, dict)
                and ("uri" not in item or (
                    isinstance(item["uri"], str) and item["uri"].lower().startswith("data:")
                ))
                for item in document.get(group, [])
            )
            for group in ("buffers", "images")
        )
    except (OSError, ValueError, AttributeError, struct.error):
        return False


def _write_json_atomic(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_mesh(
    source: str | Path,
    output_dir: str | Path,
    *,
    blender_path: str | None = None,
    seed: int | None = None,
    dataset: str = "simready",
    force: bool = False,
) -> list[dict]:
    """Render with Blender 4.2, reusing verified embedded-GLB renders unless forced."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_id = resolve_camera_id(seed=seed)
    destination = output_dir / "renders"
    cache_path = output_dir / "render_cache.json"
    cache_key = None
    if _self_contained_glb(source):
        cache_key = {
            "source": str(source), "source_sha256": sha256_file(source),
            "asset_id": asset_id, "sampling_seed": sampling_seed(seed),
            "dataset": dataset, "render_contract": render_contract(),
        }
    if cache_key is not None and not force:
        try:
            cached = json.loads(cache_path.read_text())
            if cached["key"] == cache_key:
                validate_render_outputs(destination)
                manifest = json.loads((destination / "transforms.json").read_text())
                frames = manifest["frames"]
                if (
                    cached["render_sha256"] == render_hash(destination, frames)
                    and cached["geometry_sha256"] == sha256_file(destination / "normalized_geometry.npz")
                ):
                    _write_json_atomic(output_dir / "renders_metadata.json", frames)
                    print(f"Reusing validated training renders from {destination}")
                    return frames
        except (OSError, ValueError, KeyError, TypeError, AttributeError,
                SyntaxError, EOFError, zipfile.BadZipFile):
            pass
    staging = Path(tempfile.mkdtemp(prefix=".mesh_render.", dir=output_dir))
    command = [sys.executable, str(Path(__file__).resolve())]
    if blender_path:
        command = [
            str(Path(blender_path).expanduser()), "-b", "-P",
            str(Path(__file__).resolve()), "--",
        ]
    command += ["--source", str(source), "--output-dir", str(staging),
                "--asset-id", asset_id, "--dataset", dataset]
    try:
        log_path = output_dir / "render.log"
        with log_path.open("w") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            validate_render_outputs(staging)
        except Exception as exc:
            detail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
            raise RuntimeError(
                f"Training rendering failed (exit {completed.returncode}): {exc}\n{detail}"
            ) from exc
        if completed.returncode:
            print(f"Blender exited {completed.returncode} after all renders validated")
        destination.mkdir(exist_ok=True)
        for path in staging.iterdir():
            os.replace(path, destination / path.name)
        manifest_path = destination / "transforms.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sampling_seed"] = sampling_seed(seed)
        _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(output_dir / "renders_metadata.json", manifest["frames"])
        if cache_key is not None:
            if (
                manifest.get("inference_source") != cache_key["source"]
                or manifest.get("inference_source_sha256") != cache_key["source_sha256"]
                or manifest.get("inference_asset_id") != cache_key["asset_id"]
                or manifest.get("inference_render_contract") != cache_key["render_contract"]
                or sha256_file(source) != cache_key["source_sha256"]
            ):
                raise RuntimeError("Render inputs changed while rendering; retry with a fresh output_dir")
            _write_json_atomic(cache_path, {
                "key": cache_key,
                "render_sha256": render_hash(destination, manifest["frames"]),
                "geometry_sha256": sha256_file(destination / "normalized_geometry.npz"),
            })
        return manifest["frames"]
    finally:
        shutil.rmtree(staging)


def load_render_geometry(source: str | Path, output_dir: str | Path):
    render_dir = Path(output_dir) / "renders"
    manifest = json.loads((render_dir / "transforms.json").read_text())
    source = Path(source).expanduser().resolve()
    if manifest.get("inference_source") != str(source) or manifest.get(
        "inference_source_sha256"
    ) != sha256_file(source):
        raise ValueError("Render geometry belongs to a different or changed source; render again")
    with np.load(render_dir / "normalized_geometry.npz", allow_pickle=False) as archive:
        geometry = {key: archive[key] for key in archive.files}
    return geometry, manifest


def voxelize_render_geometry(source: str | Path, output_dir: str | Path) -> np.ndarray:
    geometry, _ = load_render_geometry(source, output_dir)
    vertices = geometry["vertices"].astype(np.float64)
    triangles = geometry["triangles"].astype(np.int64)
    indices, surface, clamped, _ = _solid_voxel_indices(vertices, triangles)
    centers = _indices_to_centers(indices).astype(np.float32)
    if "triangle_segments" in geometry:
        triangle_segments = geometry["triangle_segments"]
        segment_keys = geometry["segment_keys"]
        labels = _nearest_segment_labels(clamped, triangles, triangle_segments, centers)
        _force_segment_coverage(
            clamped, triangles, triangle_segments, indices, labels, len(segment_keys)
        )
        segment_ids = segment_keys[labels]
        segment_order = segment_keys
    else:
        segment_ids = np.full(len(indices), "mesh")
        segment_order = np.asarray(["mesh"])
    voxel_dir = Path(output_dir) / "voxels"
    voxel_dir.mkdir(parents=True, exist_ok=True)
    _write_voxel_ply(voxel_dir / "voxels.ply", centers)
    np.savez_compressed(
        voxel_dir / "voxel_metadata.npz", indices=indices,
        surface_mask=surface, segment_ids=segment_ids, segment_order=segment_order,
    )
    return centers


def feature_segment_ids(output_dir: str | Path, indices: np.ndarray) -> np.ndarray:
    """Look up labels after DINO sampling without depending on its disk cache."""
    with np.load(Path(output_dir) / "voxels" / "voxel_metadata.npz") as archive:
        full_indices = archive["indices"]
        labels = archive["segment_ids"]
    full_flat = np.ravel_multi_index(full_indices.T, (64, 64, 64))
    selected_flat = np.ravel_multi_index(np.asarray(indices).T, (64, 64, 64))
    order = np.argsort(full_flat)
    rows = np.searchsorted(full_flat[order], selected_flat)
    if np.any(rows >= len(order)) or not np.array_equal(full_flat[order[rows]], selected_flat):
        raise ValueError("Feature coordinates do not match the rendered mesh voxels")
    return labels[order[rows]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--dataset", default="simready")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    arguments = parser.parse_args(argv)
    render_mesh_in_process(
        Path(arguments.source), Path(arguments.output_dir), arguments.asset_id,
        arguments.dataset,
    )
