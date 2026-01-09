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

import warp as wp
import warp.fem as fem
from warp.fem import Field, Sample, Domain, integrand

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-use utilities and kernels from the existing demo_3d script
from vomp.fem.fem_examples.mfem.demo_3d import (
    material_fraction_form,
    mark_active,
)
from vomp.fem.fem_examples.mfem.softbody_sim import (
    ClassicFEM,
    run_softbody_sim,
)

# Optional MFEM variants
from vomp.fem.fem_examples.mfem.mfem_3d import MFEM_RS_F, MFEM_sF_S

from vomp.fem.fem_examples.mfem.collisions import CollisionHandler


def build_geometry(res: int):
    """Create a unit cube grid with origin at (0,0,0)."""
    return fem.Grid3D(res=wp.vec3i(res), bounds_lo=wp.vec3(0.0, 0.0, 0.0))


def main():
    wp.init()

    # ------------------------------------------------------------------
    # 2.  Parse generic runtime/simulator-specific arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution", type=int, default=10, help="Grid resolution per axis"
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Polyscope viewer",
    )
    GroundCollidingSim.add_parser_arguments(parser)  # include collision args
    # Provide more Newton iterations by default for stability
    parser.set_defaults(
        n_newton=10,
        n_frames=100,
        # Softer material so deformation is visible
        young_modulus=1e4,  # Pa
        poisson_ratio=0.45,
        density=500.0,
        # Collision / ground defaults
        ground=True,
        ground_height=0.0,
        collision_radius=0.05,
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 3.  Build geometry and mark all cells as "active" material
    # ------------------------------------------------------------------
    # Start above ground but not too high   y ∈ [2 , 3]
    geo = fem.Grid3D(
        res=wp.vec3i(args.resolution),
        bounds_lo=wp.vec3(0.0, 2.0, 0.0),
        bounds_hi=wp.vec3(1.0, 3.0, 1.0),
    )

    active_cells = None  # simulate all cells without masking

    # ------------------------------------------------------------------
    # 4.  Create simulator and set boundary conditions (fixed bottom)
    # ------------------------------------------------------------------
    # Use GroundCollidingSim with collision handling
    sim_class = GroundCollidingSim
    sim = sim_class(geo, active_cells, args)
    sim.init_displacement_space()
    sim.init_strain_spaces()

    # set up collision detector
    sim.init_collision_detector()

    # No displacement boundary conditions – cube can move freely
    sim.set_boundary_condition(boundary_projector_form=None)

    # ------------------------------------------------------------------
    # 5.  Run the simulation (headless or with Polyscope)
    # ------------------------------------------------------------------
    # Re-enable Polyscope ground plane that run_softbody_sim disables
    def _viewer_init():
        import polyscope as ps

        ps.set_ground_plane_mode("tile")

    if args.ui:
        # ---------------------------------------------
        # First, run the simulation head-less and store every frame
        # ---------------------------------------------
        recorded_frames = []

        def _recorder(pos):
            recorded_frames.append(pos.copy())

        run_softbody_sim(sim, ui=False, frame_callback=_recorder)

        # ---------------------------------------------
        # Build Polyscope viewer with slider
        # ---------------------------------------------
        import polyscope.imgui as psim
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("tile")  # plane at y = 0 matches ground_height
        ps.set_ground_plane_height(0.0)

        # connectivity: try hexes first, else tets
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

        mesh = ps.register_volume_mesh(
            "cube",
            recorded_frames[0],
            hexes=hexes,
            tets=tets,
            edge_width=1.0,
        )

        current_idx = [0]
        play = [False]
        last_t = [time.time()]

        def _slider_callback():
            changed, new_val = psim.SliderInt(
                "frame", current_idx[0], 0, len(recorded_frames) - 1
            )
            if changed:
                current_idx[0] = new_val
                mesh.update_vertex_positions(recorded_frames[new_val])

            if psim.Button("Play" if not play[0] else "Pause"):
                play[0] = not play[0]
                last_t[0] = time.time()

            # advance automatically at ~20 fps
            if play[0] and (time.time() - last_t[0] > 0.05):
                current_idx[0] = (current_idx[0] + 1) % len(recorded_frames)
                mesh.update_vertex_positions(recorded_frames[current_idx[0]])
                last_t[0] = time.time()

        ps.set_user_callback(_slider_callback)
        ps.show()
    else:
        run_softbody_sim(sim, ui=False)


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


if __name__ == "__main__":
    main()
