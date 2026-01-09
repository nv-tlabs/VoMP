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


from vomp.fem.fem_examples.mfem.collisions import CollisionHandler


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


def build_simulation_grid(tet_vertices: np.ndarray, res: int, pad_ratio: float = 0.05):
    """Return a Grid3D covering the tetrahedral mesh plus padding.

    Also returns the bounds (lo, hi) arrays.
    """
    min_bounds = tet_vertices.min(axis=0)
    max_bounds = tet_vertices.max(axis=0)

    pad = (max_bounds - min_bounds) * pad_ratio
    min_bounds -= pad
    max_bounds += pad

    grid = fem.Grid3D(
        res=wp.vec3i(res, res, res),
        bounds_lo=wp.vec3(*min_bounds),
        bounds_hi=wp.vec3(*max_bounds),
    )
    return grid, min_bounds, max_bounds


class ObjectCollidingSim(GroundCollidingSim):
    """Extends GroundCollidingSim to handle collisions with kinematic objects."""

    def __init__(self, geo, active_cells, args, kinematic_meshes=None):
        super().__init__(geo, active_cells, args)
        self.kinematic_meshes = kinematic_meshes or []

    def init_collision_detector(self):
        """Create a CollisionHandler with both ground and object collision support."""

        node_pos = self.u_field.space.node_positions()

        pic_qp = fem.PicQuadrature(fem.Cells(self.geo), node_pos)
        pic_qp.domain = self.u_test.domain

        self.collision_handler = CollisionHandler(
            kinematic_meshes=self.kinematic_meshes,
            cp_cell_indices=pic_qp.cell_indices,
            cp_cell_coords=pic_qp.particle_coords,
            sim=self,
        )

        self._cp_quadrature = pic_qp


def create_dropping_cube_mesh(
    center, size, time_step, current_frame, drop_height=2.0, drop_speed=2.0
):
    """Create a warp mesh for a cube that drops from above.

    Parameters:
    -----------
    center : tuple
        (x, z) center position of the cube
    size : float
        Side length of the cube
    time_step : float
        Simulation time step
    current_frame : int
        Current frame number
    drop_height : float
        Initial height above the object
    drop_speed : float
        Falling speed

    Returns:
    --------
    wp.Mesh : A warp mesh representing the cube
    """
    half_size = size / 2.0

    current_time = current_frame * time_step
    y_pos = drop_height - drop_speed * current_time
    y_pos = max(y_pos, half_size)

    vertices = np.array(
        [
            [center[0] - half_size, y_pos - half_size, center[1] - half_size],
            [center[0] + half_size, y_pos - half_size, center[1] - half_size],
            [center[0] - half_size, y_pos + half_size, center[1] - half_size],
            [center[0] + half_size, y_pos + half_size, center[1] - half_size],
            [center[0] - half_size, y_pos - half_size, center[1] + half_size],
            [center[0] + half_size, y_pos - half_size, center[1] + half_size],
            [center[0] - half_size, y_pos + half_size, center[1] + half_size],
            [center[0] + half_size, y_pos + half_size, center[1] + half_size],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ],
        dtype=np.int32,
    ).flatten()

    velocities = np.zeros_like(vertices)
    if y_pos > half_size:
        velocities[:, 1] = -drop_speed

    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(indices, dtype=int),
        velocities=wp.array(velocities, dtype=wp.vec3),
    )

    return mesh


def create_cube_tetrahedra(vertices):
    """Create tetrahedral elements for a cube given its 8 vertices.

    Parameters:
    -----------
    vertices : np.ndarray
        Array of shape (8, 3) containing the cube vertices

    Returns:
    --------
    np.ndarray : Array of shape (n_tets, 4) containing tetrahedral connectivity
    """
    # Subdivide cube into 5 tetrahedra
    # This is a standard subdivision that ensures compatibility
    tets = np.array(
        [[0, 1, 2, 4], [1, 3, 2, 7], [1, 5, 4, 7], [2, 6, 4, 7], [1, 2, 4, 7]],
        dtype=np.int32,
    )

    return tets


def get_cube_position(
    center, size, time_step, current_frame, drop_height=2.0, drop_speed=2.0
):
    """Calculate the current position of the cube.

    Returns:
    --------
    tuple : (y_position, vertices_array)
    """
    half_size = size / 2.0

    current_time = current_frame * time_step
    y_pos = drop_height - drop_speed * current_time
    y_pos = max(y_pos, half_size)

    vertices = np.array(
        [
            [center[0] - half_size, y_pos - half_size, center[1] - half_size],
            [center[0] + half_size, y_pos - half_size, center[1] - half_size],
            [center[0] - half_size, y_pos + half_size, center[1] - half_size],
            [center[0] + half_size, y_pos + half_size, center[1] - half_size],
            [center[0] - half_size, y_pos - half_size, center[1] + half_size],
            [center[0] + half_size, y_pos - half_size, center[1] + half_size],
            [center[0] - half_size, y_pos + half_size, center[1] + half_size],
            [center[0] + half_size, y_pos + half_size, center[1] + half_size],
        ],
        dtype=np.float32,
    )

    return y_pos, vertices


def update_cube_mesh_position(
    mesh, center, size, time_step, current_frame, drop_height=2.0, drop_speed=2.0
):
    """Update the position of an existing cube mesh.

    Parameters:
    -----------
    mesh : wp.Mesh
        The mesh to update
    center : tuple
        (x, z) center position of the cube
    size : float
        Side length of the cube
    time_step : float
        Simulation time step
    current_frame : int
        Current frame number
    drop_height : float
        Initial height above the object
    drop_speed : float
        Falling speed
    """
    half_size = size / 2.0

    current_time = current_frame * time_step
    y_pos = drop_height - drop_speed * current_time
    y_pos = max(y_pos, half_size)

    vertices = np.array(
        [
            [center[0] - half_size, y_pos - half_size, center[1] - half_size],
            [center[0] + half_size, y_pos - half_size, center[1] - half_size],
            [center[0] - half_size, y_pos + half_size, center[1] - half_size],
            [center[0] + half_size, y_pos + half_size, center[1] - half_size],
            [center[0] - half_size, y_pos - half_size, center[1] + half_size],
            [center[0] + half_size, y_pos - half_size, center[1] + half_size],
            [center[0] - half_size, y_pos + half_size, center[1] + half_size],
            [center[0] + half_size, y_pos + half_size, center[1] + half_size],
        ],
        dtype=np.float32,
    )

    velocities = np.zeros_like(vertices)
    if y_pos > half_size:
        velocities[:, 1] = -drop_speed

    wp.copy(mesh.points, wp.array(vertices, dtype=wp.vec3))
    wp.copy(mesh.velocities, wp.array(velocities, dtype=wp.vec3))

    mesh.refit()


class MultiObjectSim:
    """Manages multiple FEM objects with collision detection between them."""

    def __init__(self, args):
        self.args = args
        self.sims = []
        self.meshes = []
        self.kinematic_meshes = []

    def add_fem_object(
        self,
        tet_vertices,
        tet_elements,
        active_cells,
        young_modulus,
        poisson_ratio,
        density,
        initial_velocity=None,
        material_map=None,
    ):
        """Add a FEM object to the simulation.

        Args:
            material_map: Optional dict mapping cell indices to (young_modulus, poisson_ratio) tuples
        """
        # Create grid for this object
        geo, bounds_lo, bounds_hi = build_simulation_grid(
            tet_vertices, self.args.resolution
        )

        # Create simulation for this object
        sim = GroundCollidingSim(
            geo,
            wp.array(active_cells, dtype=wp.int32),
            self.args,
        )

        # Set default material properties
        sim.args.young_modulus = young_modulus
        sim.args.poisson_ratio = poisson_ratio
        sim.args.density = density

        sim.init_displacement_space()
        sim.init_strain_spaces()

        # Apply material map if provided
        if material_map is not None:

            def lame_from_E_nu(E, nu):
                lam = E * nu / ((1 + nu) * (1 - 2 * nu))
                mu = E / (2 * (1 + nu))
                return lam, mu

            # Default Lame parameters
            lam0, mu0 = lame_from_E_nu(young_modulus, poisson_ratio)
            lame_np = np.full((geo.cell_count(), 2), [lam0, mu0], dtype=np.float32)

            # Apply specific materials to cells
            for cell_idx, (E, nu) in material_map.items():
                if cell_idx < geo.cell_count():
                    lam, mu = lame_from_E_nu(E, nu)
                    lame_np[cell_idx] = [lam, mu]

            sim.lame_field.dof_values.assign(wp.array(lame_np, dtype=wp.vec2))

        # Store initial positions
        sim.initial_positions = sim.u_field.space.node_positions().numpy().copy()

        # Set initial velocity if provided
        if initial_velocity is not None:
            node_positions = sim.u_field.space.node_positions().numpy()
            velocities = np.zeros_like(node_positions)

            # Apply velocity to all nodes
            velocities[:] = initial_velocity

            if hasattr(sim, "v_field"):
                sim.v_field.dof_values.assign(wp.array(velocities, dtype=wp.vec3))

        self.sims.append(sim)

        # Create mesh for collision detection
        mesh_points = wp.array(tet_vertices, dtype=wp.vec3)
        mesh_velocities = wp.zeros_like(mesh_points)

        # Simple cube faces for collision mesh
        if len(tet_vertices) >= 8:
            faces = np.array(
                [
                    [0, 2, 1],
                    [1, 2, 3],
                    [4, 5, 6],
                    [5, 7, 6],
                    [0, 1, 4],
                    [1, 5, 4],
                    [2, 6, 3],
                    [3, 6, 7],
                    [0, 4, 2],
                    [2, 4, 6],
                    [1, 3, 5],
                    [3, 7, 5],
                ],
                dtype=np.int32,
            ).flatten()
        else:
            faces = np.array([0, 1, 2], dtype=np.int32)  # Dummy face

        mesh = wp.Mesh(
            points=mesh_points,
            indices=wp.array(faces, dtype=int),
            velocities=mesh_velocities,
        )

        self.meshes.append(mesh)

        return len(self.sims) - 1

    def init_simulations(self):
        """Initialize all simulations with collision detection."""
        for i, sim in enumerate(self.sims):
            # Each sim sees other sims' meshes as kinematic objects
            other_meshes = [self.meshes[j] for j in range(len(self.meshes)) if j != i]

            sim.init_collision_detector()

            # If the sim has collision handler, update it with other meshes
            if hasattr(sim, "collision_handler") and hasattr(
                sim.collision_handler, "kinematic_meshes"
            ):
                sim.collision_handler.kinematic_meshes = other_meshes

            sim.set_boundary_condition(boundary_projector_form=None)
            sim.init_constant_forms()
            sim.project_constant_forms()

    def update_collision_meshes(self):
        """Update collision meshes based on current deformations."""
        for i, (sim, mesh) in enumerate(zip(self.sims, self.meshes)):
            # Get current positions
            positions = sim.u_field.space.node_positions().numpy()
            displacements = sim.u_field.dof_values.numpy()
            current_positions = positions + displacements

            # Update mesh points (only first 8 vertices for cube)
            n_update = min(8, len(current_positions))
            mesh.points.numpy()[:n_update] = current_positions[:n_update]

            # Update velocities if available
            if hasattr(sim, "v_field"):
                velocities = sim.v_field.dof_values.numpy()
                mesh.velocities.numpy()[:n_update] = velocities[:n_update]

            mesh.refit()

    def run_frame(self):
        """Run one frame of simulation for all objects."""
        # Update collision meshes before running physics
        self.update_collision_meshes()

        # Run physics for each object
        for sim in self.sims:
            sim.run_frame()

    def get_positions(self):
        """Get current positions for all objects."""
        all_positions = []
        for sim in self.sims:
            positions = sim.u_field.space.node_positions().numpy()
            displacements = sim.u_field.dof_values.numpy()
            current_positions = positions + displacements
            all_positions.append(current_positions)
        return all_positions


def main():
    wp.init()

    parser = argparse.ArgumentParser(description="Simple drop simulation of a USD mesh")
    parser.add_argument(
        "--usd_file",
        type=str,
        required=True,
        help="Path to the USD file to drop",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=20,
        help="Grid resolution per axis for the FEM discretisation",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.03,
        help="Size of voxels used to approximate the USD mesh",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Polyscope visualisation",
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

    parser.add_argument(
        "--cube_size",
        type=float,
        default=0.3,
        help="Size of the dropping cube",
    )
    parser.add_argument(
        "--cube_drop_height",
        type=float,
        default=1.2,
        help="Initial height of the cube above the object",
    )
    parser.add_argument(
        "--cube_drop_speed",
        type=float,
        default=1.5,
        help="Falling speed of the cube",
    )
    parser.add_argument(
        "--cube_x",
        type=float,
        default=0.0,
        help="X position of the cube center",
    )
    parser.add_argument(
        "--cube_z",
        type=float,
        default=0.0,
        help="Z position of the cube center",
    )
    parser.add_argument(
        "--cube_young",
        type=float,
        default=5e3,
        help="Young's modulus of the cube",
    )
    parser.add_argument(
        "--cube_poisson",
        type=float,
        default=0.3,
        help="Poisson ratio of the cube",
    )
    parser.add_argument(
        "--cube_density",
        type=float,
        default=1000.0,
        help="Density of the cube",
    )

    parser.set_defaults(
        n_newton=10,
        n_frames=250,
        young_modulus=1e4,
        poisson_ratio=0.45,
        density=500.0,
        gravity=9.81,
        ground=True,
        ground_height=0.0,
        collision_radius=0.01,
        dt=0.05,
    )

    args = parser.parse_args()

    usd_meshes = load_usd_mesh(args.usd_file)

    print("Meshes found (index : name):")
    for idx, (_, _, name) in enumerate(usd_meshes):
        print(f"  [{idx}] {name}")

    n_mesh = len(usd_meshes)

    def _match_list(lst, fill):
        if len(lst) >= n_mesh:
            return lst[:n_mesh]
        else:
            return lst + [fill] * (n_mesh - len(lst))

    young_list = _match_list(args.youngs, args.youngs[-1])
    pois_list = _match_list(args.poissons, args.poissons[-1])
    dens_list = _match_list(args.densities, args.densities[-1])

    print("\nMaterial properties assigned:")
    for idx, (_, _, name) in enumerate(usd_meshes):
        if idx < len(young_list):
            print(f"  [{idx}] {name}: E={young_list[idx]:.2e} Pa, nu={pois_list[idx]}")

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

    # Create voxels for the cube
    cube_center = (args.cube_x, args.cube_z)
    cube_voxel_size = (
        args.voxel_size
    )  # Use same voxel size as the object for consistency

    # Calculate initial cube position
    initial_cube_y = args.cube_drop_height

    # Ensure cube starts above the object with minimum separation
    object_max_y = voxel_centers[:, 1].max()
    min_separation = 0.2  # Minimum 20cm separation
    min_cube_y = object_max_y + args.cube_size / 2.0 + min_separation

    if initial_cube_y < min_cube_y:
        print(
            f"Warning: Adjusting cube height from {initial_cube_y:.3f} to {min_cube_y:.3f} to ensure separation"
        )
        initial_cube_y = min_cube_y

    # Generate cube voxel centers
    cube_voxels = []
    half_size = args.cube_size / 2.0
    n_voxels_per_side = max(2, int(args.cube_size / cube_voxel_size))
    actual_voxel_size = args.cube_size / n_voxels_per_side

    for i in range(n_voxels_per_side):
        for j in range(n_voxels_per_side):
            for k in range(n_voxels_per_side):
                x = cube_center[0] - half_size + (i + 0.5) * actual_voxel_size
                y = initial_cube_y - half_size + (j + 0.5) * actual_voxel_size
                z = cube_center[1] - half_size + (k + 0.5) * actual_voxel_size
                cube_voxels.append([x, y, z])

    cube_voxels = np.array(cube_voxels, dtype=np.float32)
    n_cube_voxels = len(cube_voxels)

    # Combine object and cube voxels
    all_voxel_centers = np.vstack([voxel_centers, cube_voxels])

    # Create mesh indices for the cube (use the next index after existing meshes)
    cube_mesh_idx = n_mesh
    all_voxel_mesh_idx = np.concatenate(
        [voxel_mesh_idx, np.full(n_cube_voxels, cube_mesh_idx, dtype=np.int32)]
    )

    # Build tetrahedral meshes separately to avoid connections
    object_tet_vertices, object_tet_elements = build_tet_mesh(
        voxel_centers, args.voxel_size
    )
    cube_tet_vertices_local, cube_tet_elements_local = build_tet_mesh(
        cube_voxels, args.voxel_size
    )

    # Create multi-object simulation
    multi_sim = MultiObjectSim(args)

    # Create material map for object based on voxel mesh indices
    object_material_map = {}

    # Map voxels to cells in the object grid
    object_geo, bounds_lo, bounds_hi = build_simulation_grid(
        object_tet_vertices, args.resolution
    )
    res = args.resolution

    # Convert bounds to numpy arrays
    bounds_lo_np = np.array(
        [bounds_lo[0], bounds_lo[1], bounds_lo[2]], dtype=np.float32
    )
    bounds_hi_np = np.array(
        [bounds_hi[0], bounds_hi[1], bounds_hi[2]], dtype=np.float32
    )

    for v_idx, v in enumerate(object_tet_vertices):
        # Calculate cell index for this vertex
        rel = (v - bounds_lo_np) / (bounds_hi_np - bounds_lo_np)
        cx = min(res - 1, max(0, int(rel[0] * res)))
        cy = min(res - 1, max(0, int(rel[1] * res)))
        cz = min(res - 1, max(0, int(rel[2] * res)))
        cell_idx = cx + cy * res + cz * res * res

        # Get mesh index for this vertex
        voxel_idx = v_idx // 8
        if voxel_idx < len(voxel_mesh_idx):
            m_idx = voxel_mesh_idx[voxel_idx]
            if m_idx < len(young_list):
                E = young_list[m_idx]
                nu = pois_list[m_idx]
                object_material_map[cell_idx] = (E, nu)

    print(
        f"Material map created with {len(object_material_map)} unique cell assignments"
    )
    for m_idx in range(len(young_list)):
        count = sum(
            1 for _, (E, _) in object_material_map.items() if E == young_list[m_idx]
        )
        print(f"  Mesh {m_idx}: {count} cells with E={young_list[m_idx]:.2e}")

    # Add object to simulation
    object_cells = np.arange(len(voxel_centers) * 8)  # 8 cells per voxel
    object_idx = multi_sim.add_fem_object(
        object_tet_vertices,
        object_tet_elements,
        object_cells,
        young_modulus=young_list[0],
        poisson_ratio=pois_list[0],
        density=dens_list[0],
        initial_velocity=None,  # Object starts at rest
        material_map=object_material_map,
    )

    # Add cube to simulation
    cube_cells = np.arange(len(cube_voxels) * 8)
    cube_initial_velocity = np.array([0.0, -args.cube_drop_speed, 0.0])
    cube_idx = multi_sim.add_fem_object(
        cube_tet_vertices_local,
        cube_tet_elements_local,
        cube_cells,
        young_modulus=args.cube_young,
        poisson_ratio=args.cube_poisson,
        density=args.cube_density,
        initial_velocity=cube_initial_velocity,
        material_map=None,  # No material map for cube
    )

    # Initialize all simulations with collision detection
    multi_sim.init_simulations()

    print(
        f"Object simulation: {len(object_tet_vertices)} vertices, {len(object_tet_elements)} tets"
    )
    print(
        f"Cube simulation: {len(cube_tet_vertices_local)} vertices, {len(cube_tet_elements_local)} tets"
    )

    # For compatibility with visualization, combine vertices
    tet_vertices = np.vstack([object_tet_vertices, cube_tet_vertices_local])
    tet_elements = np.vstack(
        [object_tet_elements, cube_tet_elements_local + len(object_tet_vertices)]
    )

    n_object_vertices = len(object_tet_vertices)
    n_cube_vertices = len(cube_tet_vertices_local)

    # Debug: Check initial positions
    print(
        f"Object voxels Y range: {voxel_centers[:, 1].min():.3f} to {voxel_centers[:, 1].max():.3f}"
    )
    print(
        f"Cube voxels Y range: {cube_voxels[:, 1].min():.3f} to {cube_voxels[:, 1].max():.3f}"
    )
    print(
        f"Initial separation: {cube_voxels[:, 1].min() - voxel_centers[:, 1].max():.3f}"
    )

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

    # Record initial positions from both simulations
    initial_positions = multi_sim.get_positions()
    all_initial_pos = np.vstack(initial_positions)
    recorded.append(all_initial_pos)

    for frame in range(args.n_frames):

        multi_sim.run_frame()

        # Get positions from all objects
        current_positions = multi_sim.get_positions()
        all_positions = np.vstack(current_positions)
        recorded.append(all_positions.copy())

        # Calculate total energy
        total_energy = 0.0
        for i, sim in enumerate(multi_sim.sims):
            energy = float(sim.evaluate_energy()[0])
            total_energy += energy

        if frame % 10 == 0:
            print(f"Frame {frame+1}/{args.n_frames} total energy: {total_energy:.2e}")

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
    surf_mesh.set_smooth_shade(True)
    surf_mesh.set_edge_width(0.5)
    surf_mesh.set_edge_color((0.1, 0.1, 0.1))

    # Create cube surface mesh vertices and faces
    cube_surf_vertices = cube_tet_vertices_local[
        :8
    ].copy()  # First 8 vertices form the cube
    cube_surf_faces = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],  # Bottom
            [4, 5, 6],
            [5, 7, 6],  # Top
            [0, 1, 4],
            [1, 5, 4],  # Front
            [2, 6, 3],
            [3, 6, 7],  # Back
            [0, 4, 2],
            [2, 4, 6],  # Left
            [1, 3, 5],
            [3, 7, 5],  # Right
        ],
        dtype=np.int32,
    )

    # Find the actual cube surface vertices by looking for vertices in the cube region
    cube_vertex_mask = (
        (cube_tet_vertices_local[:, 0] >= cube_center[0] - args.cube_size / 2 - 0.05)
        & (cube_tet_vertices_local[:, 0] <= cube_center[0] + args.cube_size / 2 + 0.05)
        & (cube_tet_vertices_local[:, 1] >= initial_cube_y - args.cube_size / 2 - 0.05)
        & (cube_tet_vertices_local[:, 1] <= initial_cube_y + args.cube_size / 2 + 0.05)
        & (cube_tet_vertices_local[:, 2] >= cube_center[1] - args.cube_size / 2 - 0.05)
        & (cube_tet_vertices_local[:, 2] <= cube_center[1] + args.cube_size / 2 + 0.05)
    )

    # Get the outer vertices of the cube
    cube_region_vertices = cube_tet_vertices_local[cube_vertex_mask]
    if len(cube_region_vertices) >= 8:
        # Find the 8 corner vertices
        min_x, max_x = (
            cube_region_vertices[:, 0].min(),
            cube_region_vertices[:, 0].max(),
        )
        min_y, max_y = (
            cube_region_vertices[:, 1].min(),
            cube_region_vertices[:, 1].max(),
        )
        min_z, max_z = (
            cube_region_vertices[:, 2].min(),
            cube_region_vertices[:, 2].max(),
        )

        # Create the 8 corner vertices
        cube_surf_vertices = np.array(
            [
                [min_x, min_y, min_z],  # 0
                [max_x, min_y, min_z],  # 1
                [min_x, max_y, min_z],  # 2
                [max_x, max_y, min_z],  # 3
                [min_x, min_y, max_z],  # 4
                [max_x, min_y, max_z],  # 5
                [min_x, max_y, max_z],  # 6
                [max_x, max_y, max_z],  # 7
            ],
            dtype=np.float32,
        )
    else:
        # Fallback: use the cube tet vertices directly
        print(f"Warning: Using first 8 cube tet vertices for surface mesh")
        cube_surf_vertices = cube_tet_vertices_local[:8].copy()

    # Register cube surface mesh
    cube_surf_mesh = ps.register_surface_mesh(
        "cube_surface",
        cube_surf_vertices,
        cube_surf_faces,
        edge_width=1.0,
    )
    cube_surf_mesh.set_smooth_shade(True)
    cube_surf_mesh.set_color((1.0, 0.2, 0.2))
    cube_surf_mesh.set_edge_width(0.5)
    cube_surf_mesh.set_edge_color((0.5, 0.1, 0.1))

    # Register object tetrahedral mesh
    object_mesh = ps.register_volume_mesh(
        "object_tet",
        object_tet_vertices,
        tets=object_tet_elements,
        edge_width=1.0,
    )

    # Enable the physics mesh and make it semi-transparent
    object_mesh.set_enabled(True)
    object_mesh.set_transparency(0.3)
    object_mesh.set_color((0.2, 0.8, 0.2))  # Green color for physics mesh
    object_mesh.set_edge_width(2.0)
    object_mesh.set_edge_color((0.0, 0.5, 0.0))

    # Register cube tetrahedral mesh
    cube_tet_mesh = ps.register_volume_mesh(
        "cube_physics",
        cube_tet_vertices_local,
        tets=cube_tet_elements_local,
        edge_width=1.0,
    )
    cube_tet_mesh.set_enabled(True)
    cube_tet_mesh.set_transparency(0.3)
    cube_tet_mesh.set_color((0.8, 0.2, 0.2))  # Red color for cube physics mesh
    cube_tet_mesh.set_edge_width(2.0)
    cube_tet_mesh.set_edge_color((0.5, 0.0, 0.0))

    from scipy.spatial import KDTree

    kdtree = KDTree(recorded[0])

    # Map object tet vertices to simulation nodes
    object_tet_to_node = [kdtree.query(v)[1] for v in object_tet_vertices]
    cube_tet_to_node = [kdtree.query(v)[1] for v in cube_tet_vertices_local]
    surf_to_node = [kdtree.query(v)[1] for v in surf_vertices]

    # Map cube surface vertices to nodes
    cube_surf_to_node = [kdtree.query(v)[1] for v in cube_surf_vertices]

    current = [0]
    play = [False]
    last = [time.time()]
    fps = [20]
    show_physics_mesh = [True]
    show_surface_mesh = [True]
    show_cube_physics = [True]
    show_cube_surface = [True]

    def _ui():
        psim.Text("Mesh Visibility")

        psim.Text("Object Meshes:")
        changed_physics, val_physics = psim.Checkbox(
            "Show Object Physics Mesh", show_physics_mesh[0]
        )
        if changed_physics:
            show_physics_mesh[0] = val_physics
            object_mesh.set_enabled(val_physics)

        changed_surface, val_surface = psim.Checkbox(
            "Show Object Surface Mesh", show_surface_mesh[0]
        )
        if changed_surface:
            show_surface_mesh[0] = val_surface
            surf_mesh.set_enabled(val_surface)

        psim.Text("Cube Meshes:")
        changed_cube_physics, val_cube_physics = psim.Checkbox(
            "Show Cube Physics Mesh", show_cube_physics[0]
        )
        if changed_cube_physics:
            show_cube_physics[0] = val_cube_physics
            cube_tet_mesh.set_enabled(val_cube_physics)

        changed_cube_surface, val_cube_surface = psim.Checkbox(
            "Show Cube Surface Mesh", show_cube_surface[0]
        )
        if changed_cube_surface:
            show_cube_surface[0] = val_cube_surface
            cube_surf_mesh.set_enabled(val_cube_surface)

        psim.Separator()
        psim.Text("Transparency Settings")

        if show_physics_mesh[0]:
            changed_transparency, val_transparency = psim.SliderFloat(
                "Object Physics Mesh Transparency",
                object_mesh.get_transparency(),
                0.0,
                1.0,
            )
            if changed_transparency:
                object_mesh.set_transparency(val_transparency)

        if show_cube_physics[0]:
            changed_cube_transparency, val_cube_transparency = psim.SliderFloat(
                "Cube Physics Mesh Transparency",
                cube_tet_mesh.get_transparency(),
                0.0,
                1.0,
            )
            if changed_cube_transparency:
                cube_tet_mesh.set_transparency(val_cube_transparency)

        psim.Separator()
        psim.Text("Animation Controls")

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

        if play[0] and time.time() - last[0] > 1.0 / fps[0]:
            current[0] = (current[0] + 1) % len(recorded)
            _update_frame(current[0])
            last[0] = time.time()

    def _update_frame(idx):
        disp = recorded[idx] - recorded[0]

        # Update object tetrahedral mesh
        object_mesh.update_vertex_positions(
            object_tet_vertices + disp[object_tet_to_node]
        )

        # Update cube tetrahedral mesh
        cube_tet_mesh.update_vertex_positions(
            cube_tet_vertices_local + disp[cube_tet_to_node]
        )

        # Update cube surface mesh
        cube_surf_mesh.update_vertex_positions(
            cube_surf_vertices + disp[cube_surf_to_node]
        )

        # Update surface mesh
        surf_mesh.update_vertex_positions(surf_vertices + disp[surf_to_node])

    ps.set_user_callback(_ui)
    ps.show()


if __name__ == "__main__":
    main()
