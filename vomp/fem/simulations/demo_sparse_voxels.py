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

import warp as wp
import warp.fem as fem
from warp.fem import Domain, Sample, Field
from warp.fem import integrand, normal

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vomp.fem.fem_examples.mfem.softbody_sim import ClassicFEM, run_softbody_sim
from vomp.fem.fem_examples.mfem.mfem_3d import MFEM_RS_F, MFEM_sF_S
from vomp.fem.fem_examples.mfem.collisions import CollisionHandler

import warp.examples.fem.utils as fem_example_utils

# Demo app


class GroundCollidingSim(ClassicFEM):
    """Classic FEM with ground collision handling using existing CollisionHandler."""

    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        # inherit all ClassicFEM args plus collision-related ones
        ClassicFEM.add_parser_arguments(parser)
        CollisionHandler.add_parser_arguments(parser)

    # -------------------------------------------------------------
    # Collision-specific utilities
    # -------------------------------------------------------------
    def init_collision_detector(self):
        """Create a CollisionHandler sampling one particle per mesh node."""
        # One contact particle per velocity node
        node_pos = self.u_field.space.node_positions()
        # Pic quadrature over cells at node positions
        pic_qp = fem.PicQuadrature(fem.Cells(self.geo), node_pos)
        pic_qp.domain = self.u_test.domain  # ensure the same domain as tests

        self.collision_handler = CollisionHandler(
            kinematic_meshes=[],
            cp_cell_indices=pic_qp.cell_indices,
            cp_cell_coords=pic_qp.particle_coords,
            sim=self,
        )

        # store for later reuse
        self._cp_quadrature = pic_qp

    # -------------------------------------------------------------
    # Overrides hooking collision terms into Newton solver
    # -------------------------------------------------------------
    def compute_initial_guess(self):
        self.du_field.dof_values.zero_()
        self.collision_handler.detect_collisions(self.dt)

    def evaluate_energy(self):
        E, C = super().evaluate_energy()
        E = self.collision_handler.add_collision_energy(E)
        return E, C

    def newton_lhs(self):
        lhs = super().newton_lhs()
        self.collision_handler.add_collision_hessian(lhs)
        fem.dirichlet.project_system_matrix(lhs, self.v_bd_matrix)
        return lhs

    def newton_rhs(self, tape=None):
        rhs = super().newton_rhs(tape)
        self.collision_handler.add_collision_forces(rhs)
        self._filter_forces(rhs, tape=tape)
        return rhs

    def prepare_newton_step(self, tape=None):
        self.collision_handler.prepare_newton_step(self.dt)
        return super().prepare_newton_step(tape)


def load_voxels_from_npz(
    npz_path, scale=1.0, offset=(0.0, 0.0, 0.0), rotate_to_vertical=True
):
    """
    Load voxel centers from an NPZ file.

    Note: Voxels are organized in a 64^3 grid structure with coordinates
    typically ranging from [-0.5, 0.5] in each dimension.
    This gives a grid spacing of 1.0/64 = 0.015625 units per voxel.

    Args:
        npz_path: Path to the NPZ file containing voxel data
        scale: Scaling factor for voxel positions
        offset: Offset to apply to voxel positions (x, y, z)
        rotate_to_vertical: Whether to rotate voxels to vertical orientation

    Returns:
        wp.array of voxel positions as vec3
        dict containing material properties per voxel
        float: recommended voxel size for simulation
    """
    try:
        data = np.load(npz_path)
        voxel_data = data["voxel_data"]

        # Extract positions
        positions = np.column_stack([voxel_data["x"], voxel_data["y"], voxel_data["z"]])

        # Rotate to vertical orientation if requested (swap Y and Z axes)
        if rotate_to_vertical:
            positions = positions[:, [0, 2, 1]]  # X, Z, Y -> X, Y, Z (Z becomes Y)
            print("Applied rotation: swapped Y and Z axes to make voxels vertical")

        # Apply scaling and offset
        positions = positions * scale
        positions[:, 0] += offset[0]
        positions[:, 1] += offset[1]
        positions[:, 2] += offset[2]

        # Convert to warp array
        pts = wp.array(positions, dtype=wp.vec3)

        # Extract material properties
        materials = {
            "youngs_modulus": voxel_data["youngs_modulus"],
            "poissons_ratio": voxel_data["poissons_ratio"],
            "density": voxel_data["density"],
            "segment_id": voxel_data["segment_id"],
        }

        # Calculate grid spacing (assuming 64^3 grid in [-0.5, 0.5] range)
        grid_size = 64
        theoretical_spacing = 1.0 / grid_size  # 0.015625 in normalized space

        # The actual voxel size in simulation space should account for scaling
        recommended_voxel_size = theoretical_spacing * scale

        print(f"Loaded {len(positions)} voxels from {npz_path}")
        print(f"Voxels from 64^3 grid (theoretical max: {grid_size**3} voxels)")
        print(f"Original grid spacing: {theoretical_spacing:.6f}")
        print(f"Scaling factor: {scale}")
        print(f"Recommended voxel size for simulation: {recommended_voxel_size:.6f}")
        print(
            f"Position bounds: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
            f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
            f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]"
        )
        print(f"Unique materials: {len(np.unique(materials['segment_id']))}")

        return pts, materials, recommended_voxel_size

    except Exception as e:
        print(f"Error loading NPZ file {npz_path}: {e}")
        raise


def assign_spatially_varying_materials(sim, voxel_positions, materials, voxel_size):
    """
    Assign spatially varying material properties to simulation cells based on voxel data.

    Args:
        sim: The simulation object
        voxel_positions: numpy array of voxel positions (N, 3)
        materials: dict containing material properties per voxel
        voxel_size: size of voxels for volume computation

    Returns:
        E_cells: numpy array of Young's modulus values per cell (for visualization)
    """
    from scipy.spatial import KDTree

    print(f"\n=== SPATIALLY VARYING MATERIAL ASSIGNMENT ===")

    # Build KD-tree for fast nearest neighbor search
    voxel_tree = KDTree(voxel_positions)

    # Get simulation cell count and positions
    n_cells = sim.geo.cell_count()
    print(f"Number of simulation cells: {n_cells}")
    print(f"Number of material voxels: {len(voxel_positions)}")

    # Compute cell centers from node positions
    node_positions = sim.u_field.space.node_positions().numpy()
    cell_node_indices = sim.u_field.space.topology.element_node_indices().numpy()

    cell_positions = np.zeros((n_cells, 3), dtype=np.float32)
    for cell_idx in range(n_cells):
        node_indices = cell_node_indices[cell_idx]
        cell_nodes = node_positions[node_indices]
        cell_positions[cell_idx] = np.mean(cell_nodes, axis=0)

    print(
        f"Cell positions range: min={cell_positions.min(axis=0)}, max={cell_positions.max(axis=0)}"
    )
    print("Computing cell centers from node positions")

    # Function to convert Young's modulus and Poisson ratio to Lamé parameters
    def lame_from_E_nu(E, nu):
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu

    # Initialize arrays for material properties
    lame_np = np.zeros((n_cells, 2), dtype=np.float32)
    density_np = np.zeros(n_cells, dtype=np.float32)

    # Extract material property arrays
    youngs_modulus = materials["youngs_modulus"]
    poissons_ratio = materials["poissons_ratio"]
    density = materials["density"]

    # Statistics tracking
    max_distance = 0.0
    total_distance = 0.0

    print("Assigning materials to simulation cells...")

    # Assign material properties to each simulation cell
    for cell_idx in range(min(n_cells, len(cell_positions))):
        cell_center = cell_positions[cell_idx]

        # Find closest voxel
        distance, closest_voxel_idx = voxel_tree.query(cell_center)

        # Get material properties for this voxel
        E = youngs_modulus[closest_voxel_idx]
        nu = poissons_ratio[closest_voxel_idx]
        rho = density[closest_voxel_idx]

        E /= 2e5

        # Convert to Lamé parameters
        lam, mu = lame_from_E_nu(E, nu)
        lame_np[cell_idx, 0] = lam
        lame_np[cell_idx, 1] = mu
        density_np[cell_idx] = rho

        # Update statistics
        max_distance = max(max_distance, distance)
        total_distance += distance

        # Debug output for first few cells
        if cell_idx < 5:
            print(
                f"Cell {cell_idx}: center={cell_center}, closest_voxel={closest_voxel_idx}, "
                f"distance={distance:.4f}, E={E:.2e}, nu={nu:.3f}, rho={rho:.1f}"
            )

    # Handle remaining cells if any
    if len(cell_positions) < n_cells:
        print(
            f"WARNING: {n_cells - len(cell_positions)} cells using default material properties"
        )
        # Use default values for remaining cells
        default_E = youngs_modulus.mean()
        default_nu = poissons_ratio.mean()
        default_rho = density.mean()
        default_lam, default_mu = lame_from_E_nu(default_E, default_nu)

        for cell_idx in range(len(cell_positions), n_cells):
            lame_np[cell_idx, 0] = default_lam
            lame_np[cell_idx, 1] = default_mu
            density_np[cell_idx] = default_rho

    # Verify field size compatibility
    print(f"Lame field expects {sim.lame_field.dof_values.shape[0]} values")
    print(f"We have {len(lame_np)} material assignments")

    if sim.lame_field.dof_values.shape[0] != len(lame_np):
        print(
            f"ERROR: Field size mismatch! Expected {sim.lame_field.dof_values.shape[0]}, got {len(lame_np)}"
        )

        if len(lame_np) > sim.lame_field.dof_values.shape[0]:
            print("Truncating material arrays to match field size")
            lame_np = lame_np[: sim.lame_field.dof_values.shape[0]]
            density_np = density_np[: sim.lame_field.dof_values.shape[0]]
        else:
            print("Extending material arrays to match field size")
            additional_entries = sim.lame_field.dof_values.shape[0] - len(lame_np)

            # Use mean values for extension
            mean_lam = lame_np[:, 0].mean()
            mean_mu = lame_np[:, 1].mean()
            mean_rho = density_np.mean()

            lame_extension = np.tile([mean_lam, mean_mu], (additional_entries, 1))
            density_extension = np.full(additional_entries, mean_rho)

            lame_np = np.vstack([lame_np, lame_extension])
            density_np = np.hstack([density_np, density_extension])

    # Assign to simulation
    sim.lame_field.dof_values.assign(wp.array(lame_np, dtype=wp.vec2))

    # Store density array for potential future use (currently not used in simulation)
    sim._voxel_density = density_np

    # Compute final statistics
    E_cells = (lame_np[:, 1] * (3 * lame_np[:, 0] + 2 * lame_np[:, 1])) / (
        lame_np[:, 0] + lame_np[:, 1] + 1e-9
    )

    print(f"\n=== MATERIAL ASSIGNMENT RESULTS ===")
    print(f"Maximum distance to closest voxel: {max_distance:.4f}")
    print(f"Average distance to closest voxel: {total_distance / n_cells:.4f}")
    print(f"Final Young's modulus range: {E_cells.min():.2e} - {E_cells.max():.2e} Pa")
    print(
        f"Final Poisson's ratio range: {poissons_ratio.min():.3f} - {poissons_ratio.max():.3f}"
    )
    print(f"Final density range: {density_np.min():.1f} - {density_np.max():.1f} kg/m³")

    # Check for potential issues
    zero_lam = np.sum(lame_np[:, 0] == 0)
    zero_mu = np.sum(lame_np[:, 1] == 0)
    if zero_lam > 0 or zero_mu > 0:
        print(
            f"WARNING: Found {zero_lam} cells with zero lambda, {zero_mu} cells with zero mu"
        )

    print("Spatially varying material assignment completed successfully!\n")

    return E_cells  # Return for visualization purposes


def load_ply_mesh(
    ply_path, scale=1.0, offset=(0.0, 0.0, 0.0), flip_y=False, flip_z=False
):
    """
    Load a PLY mesh file for visualization.

    Args:
        ply_path: Path to the PLY file
        scale: Scaling factor for mesh positions
        offset: Offset to apply to mesh positions (x, y, z)
        flip_y: Whether to flip Y axis
        flip_z: Whether to flip Z axis

    Returns:
        vertices: np.array of vertex positions
        faces: np.array of face indices
    """
    try:
        mesh = trimesh.load_mesh(ply_path)

        # Get vertices and faces
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()

        # Apply coordinate transformations
        if flip_y:
            vertices[:, 1] = -vertices[:, 1]
        if flip_z:
            vertices[:, 2] = -vertices[:, 2]

        # Apply scaling and offset
        vertices = vertices * scale
        vertices[:, 0] += offset[0]
        vertices[:, 1] += offset[1]
        vertices[:, 2] += offset[2]

        print(f"Loaded PLY mesh from {ply_path}")
        print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
        print(f"Applied transformations: flip_y={flip_y}, flip_z={flip_z}")
        print(
            f"Vertex bounds: X[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}], "
            f"Y[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}], "
            f"Z[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]"
        )

        return vertices, faces

    except Exception as e:
        print(f"Error loading PLY file {ply_path}: {e}")
        raise


def compute_barycentric_mapping(ply_vertices, voxel_positions):
    """
    Compute barycentric mapping from PLY mesh vertices to voxel tetrahedral mesh.

    Args:
        ply_vertices: PLY mesh vertex positions (N, 3)
        voxel_positions: Voxel mesh vertex positions (M, 3)

    Returns:
        mapping_data: Dictionary containing mapping information for deformation
    """
    print("Computing barycentric mapping from PLY mesh to voxel mesh...")

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(voxel_positions)

    # For each PLY vertex, find closest voxel vertices for interpolation
    ply_to_voxel_mapping = []

    for i, ply_vertex in enumerate(ply_vertices):
        # Find k nearest voxel vertices
        k = min(4, len(voxel_positions))  # Use up to 4 nearest neighbors
        distances, indices = tree.query(ply_vertex, k=k)

        # Compute inverse distance weights (avoid division by zero)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / np.sum(weights)  # Normalize weights

        ply_to_voxel_mapping.append(
            {
                "indices": indices,
                "weights": weights,
            }
        )

    print(f"Computed mapping for {len(ply_vertices)} PLY vertices to voxel mesh")

    return {
        "ply_to_voxel": ply_to_voxel_mapping,
        "original_ply_vertices": ply_vertices.copy(),
        "original_voxel_positions": voxel_positions.copy(),
    }


def deform_ply_mesh(mapping_data, current_voxel_positions):
    """
    Deform PLY mesh based on current voxel positions using precomputed mapping.

    Args:
        mapping_data: Mapping data from compute_barycentric_mapping
        current_voxel_positions: Current deformed voxel positions (M, 3)

    Returns:
        deformed_ply_vertices: Deformed PLY mesh vertices (N, 3)
    """
    original_ply = mapping_data["original_ply_vertices"]
    original_voxel = mapping_data["original_voxel_positions"]
    ply_to_voxel = mapping_data["ply_to_voxel"]

    deformed_ply = np.zeros_like(original_ply)

    for i, mapping in enumerate(ply_to_voxel):
        indices = mapping["indices"]
        weights = mapping["weights"]

        # Compute weighted displacement from original to current voxel positions
        original_voxel_cluster = original_voxel[indices]
        current_voxel_cluster = current_voxel_positions[indices]

        # Displacement of each voxel vertex
        displacements = current_voxel_cluster - original_voxel_cluster

        # Weighted average displacement
        avg_displacement = np.sum(weights[:, np.newaxis] * displacements, axis=0)

        # Apply displacement to PLY vertex
        deformed_ply[i] = original_ply[i] + avg_displacement

    return deformed_ply


if __name__ == "__main__":
    # wp.config.verify_cuda = True
    # wp.config.verify_fp = True
    wp.init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_file", "-f", type=str, help="Path to NPZ file containing voxel data"
    )
    parser.add_argument(
        "--ply_file", "-p", type=str, help="Path to PLY file for visualization mesh"
    )
    parser.add_argument(
        "--voxel_size",
        "-dx",
        type=float,
        default=0.015625,
        help="Voxel size for simulation. When loading NPZ files, this will be auto-calculated based on scale unless explicitly overridden.",
    )
    parser.add_argument(
        "--scale", type=float, default=2.0, help="Scaling factor for voxel positions"
    )
    parser.add_argument(
        "--offset_y",
        type=float,
        default=1.5,
        help="Y offset to lift object above ground",
    )
    parser.add_argument(
        "--flip_ply_y",
        action="store_true",
        help="Don't flip Y axis of PLY mesh (Y axis is flipped by default to match voxel coordinates)",
    )
    parser.add_argument(
        "--flip_ply_z", action="store_true", help="Flip Z axis of PLY mesh"
    )
    parser.add_argument(
        "--no_rotate_voxels",
        action="store_true",
        help="Don't rotate voxels to vertical orientation",
    )
    parser.add_argument(
        "--static_reference",
        action="store_true",
        help="Keep PLY mesh static (don't deform with voxels)",
    )
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
    GroundCollidingSim.add_parser_arguments(parser)
    args = parser.parse_args()

    # Force 100 frames for the headless simulation
    args.n_frames = 50

    # Increase Newton iterations for better convergence
    args.n_newton = 10

    # Load PLY mesh for visualization if provided
    ply_vertices = None
    ply_faces = None
    if args.ply_file:
        ply_vertices, ply_faces = load_ply_mesh(
            args.ply_file,
            scale=args.scale,
            offset=(0.0, args.offset_y, 0.0),
            flip_y=not args.flip_ply_y,  # Flip Y by default to match voxel coordinate system
            flip_z=args.flip_ply_z,
        )

    if args.npz_file:
        # Load voxels from NPZ file
        pts, materials, recommended_voxel_size = load_voxels_from_npz(
            args.npz_file,
            scale=args.scale,
            offset=(0.0, args.offset_y, 0.0),
            rotate_to_vertical=not args.no_rotate_voxels,
        )

        # Use recommended voxel size unless user specified a custom one
        if args.voxel_size == 0.015625:  # Default value, use recommended
            voxel_size = recommended_voxel_size
            print(f"Using recommended voxel size: {voxel_size:.6f}")
        else:  # User specified custom voxel size
            voxel_size = args.voxel_size
            print(f"Using user-specified voxel size: {voxel_size:.6f}")
    else:
        # Fallback to original random sphere generation
        n_pts = 1000
        print("No NPZ file specified, generating random sphere points...")

        # generate random points on a sphere
        pts = np.random.rand(n_pts, 3) * 2.0 - 1.0
        pts /= 2.0 * np.linalg.norm(pts, axis=1)[:, None] + 0.001
        pts[:, 1] += 1.5
        pts = wp.array(pts, dtype=wp.vec3)
        materials = None
        voxel_size = args.voxel_size

    volume = wp.Volume.allocate_by_voxels(pts, voxel_size=voxel_size)
    geo = fem.Nanogrid(volume)

    sim = GroundCollidingSim(geo, active_cells=None, args=args)
    sim.init_displacement_space()
    sim.init_strain_spaces()
    sim.init_collision_detector()

    sim.set_boundary_condition(
        boundary_projector_form=None,
    )

    # Assign spatially varying materials if NPZ file was loaded
    E_cells = None
    if args.npz_file and materials is not None:
        voxel_positions_np = pts.numpy()
        E_cells = assign_spatially_varying_materials(
            sim, voxel_positions_np, materials, voxel_size
        )

        # Update typical stiffness based on actual material properties
        max_E = float(E_cells.max())
        min_E = float(E_cells.min())
        # Use a moderate stiffness to balance stability and deformation
        sim.typical_stiffness = max(
            args.density * args.gravity * sim.typical_length,
            min(
                min_E * 0.01, args.density * sim.typical_length**2 / (args.dt**2)
            ),  # Use 1% of min stiffness
        )
        print(
            f"Updated typical stiffness to: {sim.typical_stiffness:.2e} based on material range {min_E:.2e} - {max_E:.2e}"
        )
        print(f"Using reduced stiffness (1% of minimum) for better deformation")

    if args.ui:
        # First run the simulation headless and record all frames
        print(f"Running simulation for {args.n_frames} frames...")
        recorded_frames = []

        def frame_recorder(displaced_pos):
            recorded_frames.append(displaced_pos.copy())

        # Run headless simulation
        run_softbody_sim(sim, ui=False, frame_callback=frame_recorder)

        print(f"Simulation complete! Recorded {len(recorded_frames)} frames.")

        # Compute PLY-to-voxel mapping if both meshes are available
        mapping_data = None
        if ply_vertices is not None and not args.static_reference:
            print("Computing mesh deformation mapping...")
            initial_voxel_positions = recorded_frames[0]
            mapping_data = compute_barycentric_mapping(
                ply_vertices, initial_voxel_positions
            )
            print("Mesh deformation mapping complete!")

        print("Starting playback visualization...")

        # Now set up the UI for playback
        import polyscope as ps
        import polyscope.imgui as psim
        import time

        ps.init()
        ps.set_ground_plane_mode("tile")
        ps.set_ground_plane_height(0.0)

        # Get voxel mesh connectivity
        try:
            hexes = sim.u_field.space.node_hexes()
        except AttributeError:
            hexes = None

        if hexes is None:
            try:
                tets = sim.u_field.space.node_tets()
            except AttributeError:
                tets = None
        else:
            tets = None

        # Always register the voxel mesh (this is what actually deforms)
        voxel_mesh = ps.register_volume_mesh(
            "voxel_simulation",
            recorded_frames[0],
            hexes=hexes,
            tets=tets,
            edge_width=1.0,
            enabled=True,  # Always show by default
        )

        # Register PLY mesh if provided
        ply_mesh = None
        deformed_ply_frames = []
        if ply_vertices is not None and ply_faces is not None:
            if mapping_data is not None:
                # Compute deformed PLY mesh for all frames
                print("Computing deformed PLY mesh for all frames...")
                for i, voxel_frame in enumerate(recorded_frames):
                    deformed_ply = deform_ply_mesh(mapping_data, voxel_frame)
                    deformed_ply_frames.append(deformed_ply)
                    if i % 10 == 0:
                        print(f"Processed frame {i+1}/{len(recorded_frames)}")

                print("Registering deformable PLY mesh")
                ply_mesh = ps.register_surface_mesh(
                    "deforming_mesh", deformed_ply_frames[0], ply_faces, enabled=True
                )
            else:
                # Static PLY mesh
                print("Registering static PLY mesh")
                ply_mesh = ps.register_surface_mesh(
                    "reference_mesh", ply_vertices, ply_faces, enabled=True
                )
                # Make it semi-transparent to see the voxels
                ply_mesh.set_transparency(0.5)

        # Playback controls
        current_frame = [0]
        is_playing = [False]
        last_time = [time.time()]
        playback_speed = [1.0]
        show_reference = [ply_mesh is not None]
        show_voxels = [True]

        def ui_callback():
            # Frame controls
            psim.TextUnformatted(
                f"Frame: {current_frame[0] + 1} / {len(recorded_frames)}"
            )

            # Slider for manual frame control
            changed, new_frame = psim.SliderInt(
                "##frame", current_frame[0], 0, len(recorded_frames) - 1
            )
            if changed:
                current_frame[0] = new_frame
                voxel_mesh.update_vertex_positions(recorded_frames[current_frame[0]])

                # Update PLY mesh if it's deformable
                if deformed_ply_frames:
                    ply_mesh.update_vertex_positions(
                        deformed_ply_frames[current_frame[0]]
                    )

            # Play/Pause button
            if psim.Button("Play" if not is_playing[0] else "Pause"):
                is_playing[0] = not is_playing[0]
                last_time[0] = time.time()

            psim.SameLine()

            # Reset button
            if psim.Button("Reset"):
                current_frame[0] = 0
                is_playing[0] = False
                voxel_mesh.update_vertex_positions(recorded_frames[0])
                if deformed_ply_frames:
                    ply_mesh.update_vertex_positions(deformed_ply_frames[0])

            # Speed control
            changed, new_speed = psim.SliderFloat("Speed", playback_speed[0], 0.1, 5.0)
            if changed:
                playback_speed[0] = new_speed

            # Toggle mesh visibility
            if ply_mesh is not None:
                changed, new_show = psim.Checkbox(
                    "Show Surface Mesh", show_reference[0]
                )
                if changed:
                    show_reference[0] = new_show
                    ply_mesh.set_enabled(show_reference[0])

            # Toggle voxel visibility
            changed, new_show = psim.Checkbox("Show Voxel Mesh", show_voxels[0])
            if changed:
                show_voxels[0] = new_show
                voxel_mesh.set_enabled(show_voxels[0])

            # Display mesh deformation info
            if mapping_data is not None:
                psim.TextUnformatted("Surface mesh: DEFORMING with voxels")
            elif ply_mesh is not None:
                psim.TextUnformatted("Surface mesh: STATIC reference")

            # Display material property information (if available)
            if E_cells is not None:
                psim.Separator()
                psim.TextUnformatted("Material Properties:")
                psim.TextUnformatted(
                    f"Young's Modulus: {E_cells.min():.2e} - {E_cells.max():.2e} Pa"
                )
                if hasattr(sim, "_voxel_density"):
                    density_range = sim._voxel_density
                    psim.TextUnformatted(
                        f"Density: {density_range.min():.1f} - {density_range.max():.1f} kg/m³"
                    )
                psim.TextUnformatted(
                    f"Material variation: {E_cells.max()/E_cells.min():.1f}x stiffness range"
                )

            # Auto-advance frames if playing
            if is_playing[0]:
                current_time = time.time()
                dt = current_time - last_time[0]
                if dt >= (1.0 / 30.0) / playback_speed[0]:  # 30 FPS base rate
                    current_frame[0] = (current_frame[0] + 1) % len(recorded_frames)
                    voxel_mesh.update_vertex_positions(
                        recorded_frames[current_frame[0]]
                    )
                    if deformed_ply_frames:
                        ply_mesh.update_vertex_positions(
                            deformed_ply_frames[current_frame[0]]
                        )
                    last_time[0] = current_time

        ps.set_user_callback(ui_callback)
        ps.show()
    else:
        # Run headless only
        run_softbody_sim(sim, ui=False)
