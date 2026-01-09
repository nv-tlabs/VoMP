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

import argparse
import time
import numpy as np
import os
import sys

import warp as wp
import warp.fem as fem

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from vomp.fem.simulations.cube_fall import GroundCollidingSim


from vomp.fem.fem_examples.mfem.softbody_sim import run_softbody_sim
from vomp.fem.fem_examples.mfem.softbody_sim import ClassicFEM


from vomp.fem.simulations.object_simulation import (
    load_usd_mesh,
    merge_meshes,
    voxelize_mesh,
    build_tet_mesh,
)


def compute_usd_bounds(file_path):
    """Return axis-aligned bounding box (min, max) of all meshes inside a USD file.

    Parameters
    ----------
    file_path : str
        Path to a .usd or .usda file containing Mesh prims.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (min_bounds, max_bounds) each shape (3,), dtype float32
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise RuntimeError(
            "pxr module not found. Please install the USD Python bindings (pip install usd-core)."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"USD file '{file_path}' not found")

    stage = Usd.Stage.Open(file_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage '{file_path}'")

    bb_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    bb_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            if not points_attr.HasAuthoredValue():
                continue
            verts = points_attr.Get()
            if verts is None:
                continue
            pts = np.asarray([(v[0], v[1], v[2]) for v in verts], dtype=np.float32)
            bb_min = np.minimum(bb_min, pts.min(axis=0))
            bb_max = np.maximum(bb_max, pts.max(axis=0))

    if np.any(bb_min == np.inf):
        raise RuntimeError("No Mesh prims with point data found in USD file")

    return bb_min, bb_max


def normalize_meshes(meshes, target_size=1.0, center=True):
    """Normalize meshes to fit within a cube of target_size.

    Parameters
    ----------
    meshes : list
        List of (vertices, faces, name) tuples
    target_size : float
        Target size for the largest dimension of the bounding box
    center : bool
        Whether to center the meshes at origin

    Returns
    -------
    list
        Normalized meshes with same structure as input
    """

    all_vertices = []
    for vtx, _, _ in meshes:
        all_vertices.append(vtx)

    if not all_vertices:
        return meshes

    all_verts_concat = np.concatenate(all_vertices, axis=0)
    min_bounds = all_verts_concat.min(axis=0)
    max_bounds = all_verts_concat.max(axis=0)

    center_point = (min_bounds + max_bounds) / 2.0
    size = max_bounds - min_bounds
    max_size = size.max()

    if max_size < 1e-6:
        print("Warning: Object has near-zero size, skipping normalization")
        return meshes

    scale = target_size / max_size

    normalized_meshes = []
    for vtx, faces, name in meshes:
        new_vtx = vtx.copy()
        if center:
            new_vtx -= center_point
        new_vtx *= scale
        normalized_meshes.append((new_vtx, faces, name))

    print(f"Normalized object: scale={scale:.4f}, original_size={max_size:.4f}")
    return normalized_meshes


def main():
    wp.init()

    parser = argparse.ArgumentParser(
        description="Simple drop simulation of a USD mesh using sparse voxels"
    )
    parser.add_argument(
        "--usd_file",
        type=str,
        required=True,
        help="Path to the USD file to drop",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Size of voxels used for sparse voxel simulation",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Polyscope visualisation",
    )

    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize the object to fit within a target size",
    )
    parser.add_argument(
        "--target_size",
        type=float,
        default=1.0,
        help="Target size for normalization (largest dimension)",
    )
    parser.add_argument(
        "--center_object",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center the object at origin during normalization",
    )

    parser.add_argument(
        "--youngs",
        type=float,
        nargs="+",
        default=[1e4],
        help="Young's modulus per mesh in order",
    )
    parser.add_argument(
        "--poissons",
        type=float,
        nargs="+",
        default=[0.45],
        help="Poisson ratio per mesh in order",
    )
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=[500.0],
        help="Density per mesh in order (not used yet)",
    )

    GroundCollidingSim.add_parser_arguments(parser)

    parser.set_defaults(
        n_newton=10,
        n_frames=50,
        young_modulus=1e4,
        poisson_ratio=0.45,
        density=500.0,
        gravity=9.81,
        ground=True,
        ground_height=0.0,
        collision_radius=0.05,
        dt=0.02,
    )

    args = parser.parse_args()

    usd_meshes = load_usd_mesh(args.usd_file)

    print("Meshes found (index : name):")
    for idx, (_, _, name) in enumerate(usd_meshes):
        print(f"  [{idx}] {name}")

    n_mesh = len(usd_meshes)

    if args.normalize:
        usd_meshes = normalize_meshes(usd_meshes, args.target_size, args.center_object)

        args.voxel_size = args.voxel_size * args.target_size

    def _match_list(lst, fill):
        if len(lst) >= n_mesh:
            return lst[:n_mesh]
        else:
            return lst + [fill] * (n_mesh - len(lst))

    young_list = _match_list(args.youngs, args.youngs[-1])
    pois_list = _match_list(args.poissons, args.poissons[-1])
    dens_list = _match_list(args.densities, args.densities[-1])

    voxel_centers_all = []
    mesh_indices = []

    for m_idx, (vtx, faces, _name) in enumerate(usd_meshes):
        vc, _ = voxelize_mesh(vtx, faces, voxel_size=args.voxel_size)
        if vc is not None and len(vc):
            voxel_centers_all.append(vc)
            mesh_indices.append(np.full(len(vc), m_idx, dtype=np.int32))

    if not voxel_centers_all:
        raise RuntimeError("Voxelisation produced no voxels – cannot proceed")

    voxel_centers = np.concatenate(voxel_centers_all, axis=0)
    voxel_mesh_idx = np.concatenate(mesh_indices, axis=0)

    voxel_centers = voxel_centers[:, [0, 2, 1]]
    voxel_centers[:, 2] *= -1.0

    min_y = voxel_centers[:, 1].min()
    lift_amount = 0.5 - min_y
    voxel_centers[:, 1] += lift_amount

    print(f"Total voxel centers: {len(voxel_centers)}")

    pts = wp.array(voxel_centers, dtype=wp.vec3)

    volume = wp.Volume.allocate_by_voxels(pts, voxel_size=args.voxel_size)
    geo = fem.Nanogrid(volume)

    print(f"Created sparse nanogrid with {geo.cell_count()} cells")

    sim = GroundCollidingSim(geo, active_cells=None, args=args)
    sim.init_displacement_space()
    sim.init_strain_spaces()
    sim.init_collision_detector()

    def lame_from_E_nu(E, nu):
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu

    print(f"\n=== MATERIAL ASSIGNMENT DEBUG ===")
    print(f"Number of simulation cells: {geo.cell_count()}")
    print(f"Number of original voxels: {len(voxel_centers)}")
    print(f"Young's modulus values: {young_list}")
    print(
        f"Mesh indices per voxel: min={voxel_mesh_idx.min()}, max={voxel_mesh_idx.max()}"
    )

    from scipy.spatial import KDTree

    voxel_tree = KDTree(voxel_centers)

    n_cells = geo.cell_count()

    print("Getting cell positions from volume structure...")

    volume_voxels = wp.zeros(shape=(volume.get_voxel_count(),), dtype=wp.vec3i)
    volume.get_voxels(volume_voxels)
    volume_voxels_np = volume_voxels.numpy()

    print(f"Volume created {len(volume_voxels_np)} voxels")
    print(f"Volume voxel size: {args.voxel_size}")

    cell_positions = volume_voxels_np.astype(np.float32) * args.voxel_size

    cell_positions += args.voxel_size * 0.5

    print(
        f"Cell positions range: min={cell_positions.min(axis=0)}, max={cell_positions.max(axis=0)}"
    )
    print(
        f"Original voxel range: min={voxel_centers.min(axis=0)}, max={voxel_centers.max(axis=0)}"
    )

    if len(cell_positions) != n_cells:
        print(
            f"WARNING: Mismatch between volume voxels ({len(cell_positions)}) and nanogrid cells ({n_cells})"
        )

        node_positions = sim.u_field.space.node_positions().numpy()
        cell_node_indices = sim.u_field.space.topology.element_node_indices().numpy()

        cell_positions = np.zeros((n_cells, 3), dtype=np.float32)
        for cell_idx in range(n_cells):
            node_indices = cell_node_indices[cell_idx]
            cell_nodes = node_positions[node_indices]
            cell_positions[cell_idx] = np.mean(cell_nodes, axis=0)
        print("Using fallback method: computing cell centers from node positions")

    lame_np = np.zeros((n_cells, 2), dtype=np.float32)

    material_counts = np.zeros(len(young_list), dtype=int)

    max_distance = 0.0
    total_distance = 0.0

    print("Assigning materials to cells...")

    for cell_idx in range(min(n_cells, len(cell_positions))):
        cell_center = cell_positions[cell_idx]

        distance, closest_voxel_idx = voxel_tree.query(cell_center)
        mesh_idx = voxel_mesh_idx[closest_voxel_idx]

        lam, mu = lame_from_E_nu(young_list[mesh_idx], pois_list[mesh_idx])
        lame_np[cell_idx, 0] = lam
        lame_np[cell_idx, 1] = mu

        material_counts[mesh_idx] += 1
        max_distance = max(max_distance, distance)
        total_distance += distance

        if cell_idx < 5:
            print(
                f"Cell {cell_idx}: center={cell_center}, closest_voxel={closest_voxel_idx}, "
                f"distance={distance:.4f}, mesh_idx={mesh_idx}, E={young_list[mesh_idx]:.0e}"
            )

    if len(cell_positions) < n_cells:
        print(
            f"WARNING: {n_cells - len(cell_positions)} cells may not have been assigned materials"
        )

        default_lam, default_mu = lame_from_E_nu(young_list[0], pois_list[0])
        for cell_idx in range(len(cell_positions), n_cells):
            lame_np[cell_idx, 0] = default_lam
            lame_np[cell_idx, 1] = default_mu

    print(f"\n=== MATERIAL ASSIGNMENT RESULTS ===")
    print(f"Maximum distance to closest voxel: {max_distance:.4f}")
    print(f"Average distance to closest voxel: {total_distance / n_cells:.4f}")
    print(f"Material distribution: {material_counts}")

    zero_lam = np.sum(lame_np[:, 0] == 0)
    zero_mu = np.sum(lame_np[:, 1] == 0)
    if zero_lam > 0 or zero_mu > 0:
        print(
            f"WARNING: Found {zero_lam} cells with zero lambda, {zero_mu} cells with zero mu"
        )

    print(f"Lame field expects {sim.lame_field.dof_values.shape[0]} values")
    print(f"We have {len(lame_np)} material assignments")

    if sim.lame_field.dof_values.shape[0] != len(lame_np):
        print(
            f"ERROR: Field size mismatch! Expected {sim.lame_field.dof_values.shape[0]}, got {len(lame_np)}"
        )
        print("This suggests the material field structure doesn't match the geometry")

        if len(lame_np) > sim.lame_field.dof_values.shape[0]:
            print("Truncating material array to match field size")
            lame_np = lame_np[: sim.lame_field.dof_values.shape[0]]
        else:
            print("Extending material array to match field size")

            additional_entries = sim.lame_field.dof_values.shape[0] - len(lame_np)
            last_lam, last_mu = lame_from_E_nu(young_list[-1], pois_list[-1])
            last_material = np.array([last_lam, last_mu], dtype=np.float32)
            extension = np.tile(last_material, (additional_entries, 1))
            lame_np = np.vstack([lame_np, extension])

    sim.lame_field.dof_values.assign(wp.array(lame_np, dtype=wp.vec2))

    E_cells = (lame_np[:, 1] * (3 * lame_np[:, 0] + 2 * lame_np[:, 1])) / (
        lame_np[:, 0] + lame_np[:, 1] + 1e-9
    )
    max_E = float(E_cells.max())
    min_E = float(E_cells.min())
    print(f"Final Young's modulus range: {min_E:.3e} – {max_E:.3e} Pa")
    print(f"Material assignment completed successfully!\n")

    E_cells = (lame_np[:, 1] * (3 * lame_np[:, 0] + 2 * lame_np[:, 1])) / (
        lame_np[:, 0] + lame_np[:, 1] + 1e-9
    )
    max_E = float(E_cells.max())
    min_E = float(E_cells.min())
    print(f"Young's modulus range across cells: {min_E:.3e} – {max_E:.3e}")

    sim.typical_stiffness = max(
        args.density * args.gravity * sim.typical_length,
        min(max_E, args.density * sim.typical_length**2 / (args.dt**2)),
    )

    sim.set_boundary_condition(boundary_projector_form=None)

    merged_all_vertices, merged_all_faces = merge_meshes(usd_meshes)

    surf_vertices = merged_all_vertices.copy()

    rot_surf = surf_vertices.copy()
    rot_surf[:, 1], rot_surf[:, 2] = surf_vertices[:, 2], -surf_vertices[:, 1]
    rot_surf[:, 1] += lift_amount

    surf_vertices = rot_surf
    surf_faces = merged_all_faces.astype(np.int32)

    max_valid = surf_vertices.shape[0] - 1
    surf_faces = np.where(surf_faces > max_valid, max_valid, surf_faces)

    recorded = []

    def _recorder(pos):
        recorded.append(pos.copy())

    run_softbody_sim(sim, ui=False, frame_callback=_recorder)

    if not args.ui:
        return

    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_ground_plane_mode("tile")
    ps.set_ground_plane_height(0.0)

    surf_mesh = ps.register_surface_mesh(
        "usd_mesh",
        surf_vertices,
        surf_faces,
        edge_width=1.0,
    )

    sim_nodes = recorded[0]

    lame_values = sim.lame_field.dof_values.numpy()

    E_cells = (lame_values[:, 1] * (3 * lame_values[:, 0] + 2 * lame_values[:, 1])) / (
        lame_values[:, 0] + lame_values[:, 1] + 1e-9
    )

    print(
        f"Young's modulus range for visualization: {E_cells.min():.2e} - {E_cells.max():.2e} Pa"
    )

    cell_node_indices = sim.u_field.space.topology.element_node_indices().numpy()

    print(
        f"Debug: E_cells shape: {E_cells.shape}, cell_node_indices shape: {cell_node_indices.shape}"
    )

    node_to_E = np.zeros(sim_nodes.shape[0])

    n_cells_to_process = min(len(E_cells), cell_node_indices.shape[0])
    print(f"Processing {n_cells_to_process} cells")

    for cell_idx in range(n_cells_to_process):
        node_indices = cell_node_indices[cell_idx]
        node_to_E[node_indices] = E_cells[cell_idx]

    E_min, E_max = E_cells.min(), E_cells.max()

    E_log = np.log10(node_to_E + 1e-9)
    E_log_min, E_log_max = np.log10(E_min), np.log10(E_max)
    E_normalized = (E_log - E_log_min) / (E_log_max - E_log_min + 1e-9)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    colormap = cm.get_cmap("viridis")
    colors = colormap(E_normalized)[:, :3]

    print(f"Colored {len(sim_nodes)} physics voxels based on Young's modulus")
    print(
        f"Color mapping (log scale): Blue = {E_min:.2e} Pa (soft), Yellow = {E_max:.2e} Pa (stiff)"
    )
    print(f"Log range: {E_log_min:.2f} - {E_log_max:.2f}")

    unique_colors = np.unique(E_normalized)
    print(f"Number of unique color values: {len(unique_colors)}")
    if len(unique_colors) <= 10:
        print(f"Color values: {unique_colors}")

    physics_voxels = ps.register_point_cloud(
        "physics_voxels",
        sim_nodes,
        radius=args.voxel_size * 0.3,
    )
    physics_voxels.add_color_quantity("youngs_modulus", colors, enabled=True)
    physics_voxels.set_enabled(True)

    try:

        if hasattr(geo, "cell_positions"):
            cell_centers = geo.cell_positions().numpy()
            volume_viz = ps.register_point_cloud(
                "volume_cells",
                cell_centers,
                radius=args.voxel_size * 0.4,
            )
            volume_viz.set_color((0.3, 0.8, 0.3))
            volume_viz.set_enabled(False)
    except:

        pass

    from scipy.spatial import KDTree

    kdtree = KDTree(recorded[0])

    surf_to_node = [kdtree.query(v)[1] for v in surf_vertices]

    current = [0]
    play = [False]
    last = [time.time()]
    fps = [20]

    surf_enabled = [True]
    physics_enabled = [True]

    def _ui():
        changed, val = psim.SliderInt("frame", current[0], 0, len(recorded) - 1)
        if changed:
            current[0] = val
            _update_frame(val)

        changed_fps, new_fps = psim.SliderInt("fps", fps[0], 1, 60)
        if changed_fps:
            fps[0] = new_fps

        if psim.Button("Play" if not play[0] else "Pause"):
            play[0] = not play[0]
            last[0] = time.time()

        if psim.Button("Toggle Surface"):
            surf_enabled[0] = not surf_enabled[0]
            surf_mesh.set_enabled(surf_enabled[0])

        if psim.Button("Toggle Physics Voxels"):
            physics_enabled[0] = not physics_enabled[0]
            physics_voxels.set_enabled(physics_enabled[0])

        if play[0] and time.time() - last[0] > 1.0 / fps[0]:
            current[0] = (current[0] + 1) % len(recorded)
            _update_frame(current[0])
            last[0] = time.time()

    def _update_frame(idx):
        disp = recorded[idx] - recorded[0]

        surf_mesh.update_vertex_positions(surf_vertices + disp[surf_to_node])

        physics_voxels.update_point_positions(recorded[idx])

    ps.set_user_callback(_ui)
    ps.show()


if __name__ == "__main__":
    main()
