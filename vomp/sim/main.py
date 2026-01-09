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

import json
import sys
import os
import pathlib
import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view, Vector3, Transform, Logger, AngleAxis
from uipc.core import Engine, World, Scene, SceneIO
from uipc.gui import SceneGUI
from uipc.unit import kPa, GPa
from uipc.backend import SceneVisitor

from vomp.sim.meshes import (
    MeshProcessor,
    load_visual_mesh,
    write_visual_meshes_obj,
    embed_visual_mesh_in_physics_tets,
)


def create_output_dir(output_path):
    output_dir = pathlib.Path(output_path).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def setup_scene_from_config(config):
    scene_config = Scene.default_config()

    sim_params = config.get("simulation", {})
    scene_config["dt"] = sim_params.get("dt", 0.02)

    gravity = sim_params.get("gravity", [0.0, -9.8, 0.0])
    scene_config["gravity"] = [[gravity[0]], [gravity[1]], [gravity[2]]]

    contact_params = sim_params.get("contact", {})
    scene_config["contact"]["friction"]["enable"] = contact_params.get(
        "friction_enable", False
    )
    scene_config["contact"]["d_hat"] = contact_params.get("d_hat", 0.01)

    return Scene(scene_config)


def setup_contact_model(scene, config):
    contact_config = config.get("contact_model", {})
    friction = contact_config.get("friction", 0.5)
    contact_resistance = contact_config.get("contact_resistance", 1.0) * GPa

    scene.contact_tabular().default_model(friction, contact_resistance)


def run_simulation(config_path):

    with open(config_path, "r") as f:
        config = json.load(f)

    output_dir = create_output_dir(config["output"]["directory"])

    log_level = config.get("logging", {}).get("level", "warn")
    if log_level.lower() == "info":
        Logger.set_level(Logger.Level.Info)
    elif log_level.lower() == "debug":
        Logger.set_level(Logger.Level.Debug)
    else:
        Logger.set_level(Logger.Level.Warn)

    engine_type = config.get("engine", {}).get("type", "cuda")
    engine = Engine(engine_type, output_dir)
    world = World(engine)

    scene = setup_scene_from_config(config)
    setup_contact_model(scene, config)

    mesh_processor = MeshProcessor(scene)

    objects_config = config.get("objects", [])
    all_meshes = mesh_processor.create_objects_from_config(objects_config)

    ground_config = config.get("ground", {})
    if ground_config.get("enable", True):
        ground_height = ground_config.get("height", 0.0)
        mesh_processor.create_ground(ground_height)

    world.init(scene)

    sio = SceneIO(scene)

    save_meshes = config["output"].get("save_meshes", True)
    obj_output_dir = None
    if save_meshes:
        obj_output_dir = os.path.join(output_dir, "surface_meshes")
        os.makedirs(obj_output_dir, exist_ok=True)

    show_gui = config.get("gui", {}).get("enable", True)

    if show_gui:
        run_with_gui(world, scene, sio, obj_output_dir, save_meshes, all_meshes, config)
    else:
        run_headless(world, scene, sio, obj_output_dir, save_meshes, config)


def run_with_gui(world, scene, sio, obj_output_dir, save_meshes, all_meshes, config):

    def compute_visual_vertices_from_barycentric(
        physics_obj_index, embeddings, physics_positions
    ):
        """
        Compute deformed visual vertex positions using barycentric coordinates.
        This interpolates the visual mesh vertices based on the physics mesh deformation.
        """
        physics_mesh = all_meshes[physics_obj_index][0]
        physics_tets = physics_mesh.tetrahedra()
        tet_connectivity = view(physics_tets.topo())

        deformed_visual_vertices = np.zeros((len(embeddings), 3), dtype=np.float32)

        # Track warnings to avoid spamming console
        warning_counts = {
            "invalid_tet": 0,
            "invalid_coords": 0,
            "extreme_coords": 0,
            "nan_inf": 0,
        }

        for j, (tet_idx, bary_coords) in enumerate(embeddings):
            # Validate tetrahedron index
            if tet_idx < 0 or tet_idx >= len(tet_connectivity):
                warning_counts["invalid_tet"] += 1
                continue

            tet_verts = tet_connectivity[tet_idx]
            tet_positions = physics_positions[tet_verts]

            # Ensure correct shape
            if tet_positions.shape != (4, 3):
                tet_positions = tet_positions.reshape(4, 3)

            if bary_coords.ndim > 1:
                bary_coords = bary_coords.flatten()

            # Validate barycentric coordinates
            if len(bary_coords) != 4:
                warning_counts["invalid_coords"] += 1
                bary_coords = np.array([0.25, 0.25, 0.25, 0.25])

            # Check for NaN or infinite values first
            if np.any(np.isnan(bary_coords)) or np.any(np.isinf(bary_coords)):
                warning_counts["nan_inf"] += 1
                bary_coords = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                # Check for extreme values and renormalize if needed
                max_abs_coord = np.max(np.abs(bary_coords))
                if max_abs_coord > 3.0:
                    warning_counts["extreme_coords"] += 1
                    # Scale down to reasonable range
                    bary_coords = bary_coords * (3.0 / max_abs_coord)

                # Ensure sum is approximately 1.0
                coord_sum = np.sum(bary_coords)
                if abs(coord_sum) > 1e-10:
                    bary_coords = bary_coords / coord_sum
                else:
                    bary_coords = np.array([0.25, 0.25, 0.25, 0.25])

            # Compute interpolated position
            deformed_pos = np.sum(bary_coords[:, np.newaxis] * tet_positions, axis=0)
            deformed_visual_vertices[j] = deformed_pos

        # Report warnings summary (only if there are warnings)
        if any(warning_counts.values()):
            print(
                f"Barycentric interpolation warnings: {sum(warning_counts.values())} total issues"
            )
            if warning_counts["invalid_tet"] > 0:
                print(
                    f"  - {warning_counts['invalid_tet']} invalid tetrahedron indices"
                )
            if warning_counts["invalid_coords"] > 0:
                print(
                    f"  - {warning_counts['invalid_coords']} invalid coordinate lengths"
                )
            if warning_counts["extreme_coords"] > 0:
                print(
                    f"  - {warning_counts['extreme_coords']} extreme coordinates (renormalized)"
                )
            if warning_counts["nan_inf"] > 0:
                print(f"  - {warning_counts['nan_inf']} NaN/Inf coordinates")

        return deformed_visual_vertices

    sgui = SceneGUI(scene)

    ps.init()
    ground_height = config.get("ground", {}).get("height", 0.0)
    ps.set_ground_plane_height(ground_height)

    tri_surf, line_surf, point_surf = sgui.register()
    if tri_surf:
        tri_surf.set_edge_width(1)

    visual_meshes = []
    visual_mesh_objects = []
    physics_initial_positions = []
    physics_current_positions = []
    visual_barycentric_embeddings = []
    scene_object_names = []
    current_visual_vertices = []

    objects_config = config.get("objects", [])
    for i, (physics_mesh, obj_config) in enumerate(
        zip([m[0] for m in all_meshes], objects_config)
    ):

        physics_positions = physics_mesh.positions().view()
        physics_pos_array = []
        for j in range(len(physics_positions)):
            pos = physics_positions[j]
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
        physics_initial_positions.append(physics_pos_array.copy())
        physics_current_positions.append(physics_pos_array.copy())

        obj_name = obj_config.get("name", f"object_{i+1}")
        scene_object_names.append(obj_name)

    has_visual_meshes = any(
        obj_config.get("visual_mesh") for obj_config in objects_config
    )

    if has_visual_meshes:
        print("Loading visual meshes from config...")

        for physics_obj_index, (physics_mesh, obj_config) in enumerate(
            zip([m[0] for m in all_meshes], objects_config)
        ):
            visual_mesh_path = obj_config.get("visual_mesh")

            if not visual_mesh_path:
                print(
                    f"Warning: No visual mesh specified for object {physics_obj_index+1} ({obj_config.get('name', 'unnamed')})"
                )
                continue

            if not os.path.exists(visual_mesh_path):
                print(
                    f"Warning: Visual mesh file {visual_mesh_path} not found for object {physics_obj_index+1}"
                )
                continue

            print(
                f"Loading visual mesh for object {physics_obj_index+1} from {visual_mesh_path}"
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

            visual_scale = obj_config.get("scale", 1.0)
            visual_transform = Transform.Identity()
            visual_transform.scale(visual_scale)
            visual_transform = transform * visual_transform

            # Load visual mesh with the same normalization as specified in config
            # This ensures visual mesh scale matches physics mesh scale
            material = obj_config.get("material", {})
            material_file = material.get("file", None)
            normalize_visual = obj_config.get("normalize_visual_mesh", True)

            # Apply the transform - normalization will happen inside if enabled
            vertices, faces, uv_coords, face_uvs, face_materials, mtl_lib = (
                load_visual_mesh(
                    visual_mesh_path,
                    1.0,
                    visual_transform,
                    normalize_like_blender=normalize_visual,
                    material_file=material_file,
                )
            )

            physics_pos_array = physics_initial_positions[physics_obj_index]

            print(
                f"Physics object {physics_obj_index+1}: {len(physics_pos_array)} physics vertices"
            )

            print(
                f"Debug: Physics positions (world space) - X: [{np.min(physics_pos_array[:, 0]):.3f}, {np.max(physics_pos_array[:, 0]):.3f}], Y: [{np.min(physics_pos_array[:, 1]):.3f}, {np.max(physics_pos_array[:, 1]):.3f}], Z: [{np.min(physics_pos_array[:, 2]):.3f}, {np.max(physics_pos_array[:, 2]):.3f}]"
            )
            print(
                f"Debug: Visual mesh (after transform) - X: [{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}], Y: [{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}], Z: [{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]"
            )

            # Check alignment between visual and physics meshes
            physics_centroid = np.mean(physics_pos_array, axis=0)
            visual_centroid = np.mean(vertices, axis=0)
            physics_size = np.max(physics_pos_array, axis=0) - np.min(
                physics_pos_array, axis=0
            )
            visual_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)

            print(f"Debug: Physics centroid: {physics_centroid}, size: {physics_size}")
            print(f"Debug: Visual centroid: {visual_centroid}, size: {visual_size}")

            # Check if sizes match (within 10% - they should match if properly normalized/transformed)
            size_ratio = np.linalg.norm(visual_size) / np.linalg.norm(physics_size)
            print(f"Debug: Size ratio (visual/physics): {size_ratio:.3f}")

            centroid_offset = physics_centroid - visual_centroid
            offset_magnitude = np.linalg.norm(centroid_offset)
            relative_offset = offset_magnitude / np.linalg.norm(physics_size)

            # Decision logic for alignment:
            # 1. If sizes match well (0.9 < ratio < 1.1), apply centroid alignment
            # 2. If sizes don't match, user needs to fix the normalization/scale settings
            if 0.9 < size_ratio < 1.1:
                if relative_offset > 0.001:  # More than 0.1% offset
                    print(
                        f"Sizes match, applying centroid alignment (offset: {centroid_offset}, relative: {relative_offset:.4f})"
                    )
                    aligned_visual_vertices = vertices + centroid_offset
                else:
                    print(
                        f"Sizes and centroids match perfectly (offset: {relative_offset:.4f})"
                    )
                    aligned_visual_vertices = vertices
                    centroid_offset = np.array([0.0, 0.0, 0.0])
            else:
                print(
                    f"WARNING: Size mismatch detected! Visual and physics meshes have different scales."
                )
                print(
                    f"  This usually means normalize_visual_mesh setting is incorrect."
                )
                print(
                    f"  Physics size: {np.linalg.norm(physics_size):.3f}, Visual size: {np.linalg.norm(visual_size):.3f}"
                )
                print(
                    f"  Attempting centroid alignment anyway, but results may be poor."
                )
                aligned_visual_vertices = vertices + centroid_offset

            print(
                f"Final visual range: X=[{np.min(aligned_visual_vertices[:, 0]):.3f}, {np.max(aligned_visual_vertices[:, 0]):.3f}], Y=[{np.min(aligned_visual_vertices[:, 1]):.3f}, {np.max(aligned_visual_vertices[:, 1]):.3f}], Z=[{np.min(aligned_visual_vertices[:, 2]):.3f}, {np.max(aligned_visual_vertices[:, 2]):.3f}]"
            )

            print(
                f"Computing barycentric embedding for visual mesh {physics_obj_index+1}..."
            )

            # Get embedding optimization settings from config
            embedding_config = config.get("embedding", {})
            use_multiprocessing = embedding_config.get("use_multiprocessing", True)
            n_processes = embedding_config.get("n_processes", None)
            use_original_method = embedding_config.get("use_original_method", False)

            embeddings, embedding_stats = embed_visual_mesh_in_physics_tets(
                aligned_visual_vertices,
                physics_mesh,
                use_multiprocessing,
                n_processes,
                use_original_method,
            )
            visual_barycentric_embeddings.append(embeddings)

            print(
                f"Visual mesh {physics_obj_index+1} embedding: "
                f"{embedding_stats['inside_count']} inside, "
                f"{embedding_stats['outside_count']} outside, "
                f"max distance = {embedding_stats['max_distance']:.4f}"
            )

            # Verify alignment after embedding
            aligned_centroid = np.mean(aligned_visual_vertices, axis=0)
            print(
                f"Debug: Final aligned visual centroid: [{aligned_centroid[0]:.3f}, {aligned_centroid[1]:.3f}, {aligned_centroid[2]:.3f}]"
            )

            # Sanity check: compute distance between closest visual vertex and physics vertex
            from scipy.spatial import cKDTree

            physics_tree = cKDTree(physics_pos_array)
            sample_visual_vertices = aligned_visual_vertices[
                :: max(1, len(aligned_visual_vertices) // 100)
            ]  # Sample 100 points
            sample_distances, _ = physics_tree.query(sample_visual_vertices)
            avg_distance = np.mean(sample_distances)
            max_distance = np.max(sample_distances)
            print(
                f"Debug: Visual to Physics mesh distances - Avg: {avg_distance:.4f}, Max: {max_distance:.4f}"
            )

            initial_visual_vertices = compute_visual_vertices_from_barycentric(
                physics_obj_index, embeddings, physics_pos_array
            )

            visual_mesh_name = f"visual_mesh_{physics_obj_index+1}"
            visual_mesh = ps.register_surface_mesh(
                visual_mesh_name,
                initial_visual_vertices,
                faces,
                edge_width=1.0,
                color=(0.2, 0.8, 0.2),
            )
            visual_mesh.set_transparency(0.7)

            # Extract material directory from the visual mesh path or material file
            material_dir = None
            if material_file:
                material_dir = os.path.dirname(material_file)
            elif visual_mesh_path:
                material_dir = os.path.dirname(visual_mesh_path)

            visual_meshes.append(
                (
                    vertices,
                    faces,
                    centroid_offset,
                    physics_obj_index,
                    uv_coords,
                    face_uvs,
                    material_dir,
                    obj_config.get("name", f"object_{physics_obj_index+1}"),
                    face_materials,
                    mtl_lib,
                )
            )
            visual_mesh_objects.append(visual_mesh)

            current_visual_vertices.append(initial_visual_vertices.copy())

            print(
                f"Registered visual mesh {visual_mesh_name} with {len(vertices)} vertices (maps to physics object {physics_obj_index})"
            )
    else:
        print("No visual meshes specified in config")

    if save_meshes and visual_meshes:
        initial_obj_path = os.path.join(
            obj_output_dir, f"scene_surface_{world.frame():04d}.obj"
        )
        write_visual_meshes_obj(
            initial_obj_path,
            visual_meshes,
            visual_mesh_objects,
            current_visual_vertices,
        )

    run_simulation = config.get("simulation", {}).get("auto_start", False)
    frame_count = 0
    max_frames = config.get("simulation", {}).get("max_frames", 1000)

    def update_physics_positions():
        try:

            scene_visitor = SceneVisitor(scene)
            geo_slots = scene_visitor.geometries()

            geometry_index = 0
            for geo_slot in geo_slots:
                if geometry_index >= len(physics_initial_positions):
                    break

                geo = geo_slot.geometry()

                if hasattr(geo, "positions"):

                    current_positions = geo.positions().view()

                    current_pos_array = []
                    for i in range(len(current_positions)):
                        pos = current_positions[i]
                        if hasattr(pos, "tolist"):
                            pos_list = pos.tolist()
                            if isinstance(pos_list[0], list):
                                current_pos_array.append(
                                    [pos_list[0][0], pos_list[1][0], pos_list[2][0]]
                                )
                            else:
                                current_pos_array.append(pos_list)
                        else:
                            current_pos_array.append(
                                [float(pos[0]), float(pos[1]), float(pos[2])]
                            )

                    current_pos_array = np.array(current_pos_array, dtype=np.float32)

                    if frame_count % 20 == 0:
                        print(
                            f"Debug: Geometry {geometry_index+1}: Retrieved {len(current_pos_array)} vertices"
                        )

                    expected_vertex_count = len(
                        physics_initial_positions[geometry_index]
                    )
                    if len(current_pos_array) == expected_vertex_count:

                        displacement = (
                            current_pos_array
                            - physics_initial_positions[geometry_index]
                        )
                        displacement_magnitude = np.linalg.norm(displacement, axis=1)
                        max_displacement = np.max(displacement_magnitude)
                        avg_displacement = np.mean(displacement_magnitude)

                        if frame_count % 20 == 0:
                            print(
                                f"Debug: Physics object {geometry_index+1} - Max displacement: {max_displacement:.6f}, Avg displacement: {avg_displacement:.6f}"
                            )
                            print(
                                f"Debug: Current avg Y: {np.mean(current_pos_array[:, 1]):.4f}, Initial avg Y: {np.mean(physics_initial_positions[geometry_index][:, 1]):.4f}"
                            )

                        physics_current_positions[geometry_index] = (
                            current_pos_array.copy()
                        )
                    else:
                        if frame_count % 20 == 0:
                            print(
                                f"Debug: Vertex count mismatch for object {geometry_index+1} (expected {expected_vertex_count}, got {len(current_pos_array)})"
                            )

                    geometry_index += 1

        except Exception as e:
            if frame_count % 20 == 0:
                print(
                    f"Debug: Error getting current positions from geometry slots: {e}"
                )

    def update_visual_meshes():
        for i, (
            vertices,
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
            if i < len(visual_barycentric_embeddings) and physics_obj_index < len(
                physics_current_positions
            ):

                current_physics = physics_current_positions[physics_obj_index]
                embeddings = visual_barycentric_embeddings[i]

                deformed_visual_vertices = compute_visual_vertices_from_barycentric(
                    physics_obj_index, embeddings, current_physics
                )

                if frame_count % 20 == 0:
                    initial_visual_aligned = vertices + centroid_offset
                    displacement = deformed_visual_vertices - initial_visual_aligned
                    displacement_magnitude = np.linalg.norm(displacement, axis=1)
                    max_displacement = np.max(displacement_magnitude)
                    avg_displacement = np.mean(displacement_magnitude)

                    print(
                        f"Debug: Visual mesh {i+1} (physics obj {physics_obj_index+1}) - Max displacement: {max_displacement:.6f}, Avg displacement: {avg_displacement:.6f}"
                    )

                visual_mesh_objects[i].update_vertex_positions(deformed_visual_vertices)

                current_visual_vertices[i] = deformed_visual_vertices.copy()

    def on_update():
        nonlocal run_simulation, frame_count

        if imgui.Button("Start/Stop Simulation"):
            run_simulation = not run_simulation

        imgui.SameLine()
        if imgui.Button("Reset"):

            mesh_processor = MeshProcessor(scene)
            objects_config = config.get("objects", [])
            for i, (mesh, obj_config) in enumerate(
                zip([m[0] for m in all_meshes], objects_config)
            ):
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

            for i, (
                vertices,
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
                if i < len(visual_barycentric_embeddings) and physics_obj_index < len(
                    physics_initial_positions
                ):

                    initial_physics = physics_initial_positions[physics_obj_index]
                    embeddings = visual_barycentric_embeddings[i]

                    reset_visual_vertices = compute_visual_vertices_from_barycentric(
                        physics_obj_index, embeddings, initial_physics
                    )

                    visual_mesh_objects[i].update_vertex_positions(
                        reset_visual_vertices
                    )

                    current_visual_vertices[i] = reset_visual_vertices.copy()

            for i in range(len(physics_current_positions)):
                physics_current_positions[i] = physics_initial_positions[i].copy()

            world.init(scene)
            frame_count = 0

            if save_meshes:
                reset_obj_path = os.path.join(
                    obj_output_dir, f"scene_surface_{world.frame():04d}.obj"
                )

                if visual_meshes:
                    write_visual_meshes_obj(
                        reset_obj_path,
                        visual_meshes,
                        visual_mesh_objects,
                        current_visual_vertices,
                    )
                else:
                    sio.write_surface(reset_obj_path)

        imgui.Text(f"Frame: {frame_count}")
        imgui.Text(f"Max Frames: {max_frames}")
        imgui.Text(f"Time: {frame_count * config['simulation'].get('dt', 0.02):.2f}s")
        imgui.Text(f"Objects: {len(all_meshes)}")
        imgui.Text(f"World Frame: {world.frame()}")
        imgui.Text(f"Visual Meshes: {len(visual_meshes)}")

        if run_simulation and frame_count < max_frames:
            world.advance()
            world.retrieve()
            sgui.update()

            update_physics_positions()

            update_visual_meshes()

            frame_count += 1

            if save_meshes:
                step_obj_path = os.path.join(
                    obj_output_dir, f"scene_surface_{world.frame():04d}.obj"
                )

                if visual_meshes:
                    write_visual_meshes_obj(
                        step_obj_path,
                        visual_meshes,
                        visual_mesh_objects,
                        current_visual_vertices,
                    )
                else:
                    sio.write_surface(step_obj_path)

        if frame_count >= max_frames:
            run_simulation = False

    ps.set_user_callback(on_update)
    ps.show()


def run_headless(world, scene, sio, obj_output_dir, save_meshes, config):
    max_frames = config.get("simulation", {}).get("max_frames", 100)

    for frame in range(max_frames):
        world.advance()
        world.retrieve()

        if save_meshes:
            step_obj_path = os.path.join(
                obj_output_dir, f"scene_surface_{world.frame():04d}.obj"
            )

            sio.write_surface(step_obj_path)

        if frame % 10 == 0:
            print(f"Frame {frame}/{max_frames}")

    print(f"Simulation completed: {max_frames} frames")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.json>")
        print("  config.json: JSON configuration file for the simulation")
        return

    config_path = sys.argv[1]
    run_simulation(config_path)


if __name__ == "__main__":
    main()
