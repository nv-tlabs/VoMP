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

from fem_examples.mfem.softbody_sim import ClassicFEM, run_softbody_sim
from fem_examples.mfem.variable_density import ClassicFEMWithDensity
from fem_examples.mfem.collisions import CollisionHandler

from fem_examples.mfem.mfem_3d import MFEM_RS_F, MFEM_sF_S

import warp.examples.fem.utils as fem_example_utils
import meshio
import numpy as np

from material_loader import (
    apply_spatially_varying_materials,
    visualize_material_distribution,
    load_material_data,
)
from vomp.inference.utils import MaterialUpsampler


@wp.func
def material_fraction(x: wp.vec3):
    return 1.0
    # return wp.select(wp.length(x - wp.vec3(0.5, 1.0, 0.875)) > 0.2, 0.0, 1.0)


@integrand
def material_fraction_form(s: Sample, domain: Domain, phi: Field):
    return material_fraction(domain(s)) * phi(s)


@wp.kernel
def mark_active(fraction: wp.array(dtype=wp.float64), active: wp.array(dtype=int)):
    active[wp.tid()] = int(wp.nonzero(fraction[wp.tid()]))


@integrand
def clamped_edge(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    """Dirichlet boundary condition projector (fixed vertices selection)"""

    clamped = float(0.0)

    # Single clamped edge
    if s.qp_index < 10:
        clamped = 1.0

    return wp.dot(u(s), v(s)) * clamped


@integrand
def clamped_right(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    """Dirichlet boundary condition projector (fixed vertices selection)"""

    pos = domain(s)
    clamped = float(0.0)

    # clamped right sides
    clamped = 0.0  # wp.select(pos[0] < 1.0, 1.0, 0.0)

    return wp.dot(u(s), v(s)) * clamped


@integrand
def clamped_sides(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    """Dirichlet boundary condition projector (fixed vertices selection)"""

    nor = normal(domain, s)
    clamped = float(0.0)

    # clamped vertical sides
    clamped = wp.abs(nor[0])

    return wp.dot(u(s), v(s)) * clamped


@integrand
def boundary_displacement_form(
    s: Sample,
    domain: Domain,
    v: Field,
    displacement: float,
):
    """Prescribed displacement"""

    # opposed to normal
    nor = normal(domain, s)

    # vertical sides only
    clamped = wp.abs(nor[0])

    return -displacement * wp.dot(nor, v(s)) * clamped


class ClassicFEMWithFloorAndBC(ClassicFEMWithDensity):
    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        ClassicFEM.add_parser_arguments(parser)
        CollisionHandler.add_parser_arguments(parser)

    def compute_initial_guess(self):
        self.du_field.dof_values.zero_()
        self.collision_handler.detect_collisions(self.dt)

    def evaluate_energy(self):
        E_p, c_r = super().evaluate_energy()
        E_p = self.collision_handler.add_collision_energy(E_p)

        return E_p, c_r

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

    def init_collision_detector(
        self,
        vtx_quadrature: fem.PicQuadrature,
    ):
        self.collision_handler = CollisionHandler(
            [], vtx_quadrature.cell_indices, vtx_quadrature.particle_coords, self
        )


if __name__ == "__main__":
    # wp.config.verify_cuda = True
    # wp.config.verify_fp = True
    wp.init()

    class_parser = argparse.ArgumentParser()
    class_parser.add_argument(
        "--variant", "-v", choices=["mfem", "classic", "trusty"], default="classic"
    )
    class_args, remaining_args = class_parser.parse_known_args()

    if class_args.variant == "mfem":
        sim_class = MFEM_RS_F
    elif class_args.variant == "trusty":
        sim_class = MFEM_sF_S
    else:
        sim_class = ClassicFEMWithFloorAndBC

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True, help="Path to .msh file")
    parser.add_argument(
        "--materials",
        type=str,
        default=None,
        help="Path to .npz material file (optional)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=1,
        help="Number of neighbors for material interpolation",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=20,
        help="Resolution for collision radius calculation",
    )
    parser.add_argument("--displacement", type=float, default=0.0)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction)
    parser.add_argument("--clamping", type=str, default="right")
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center and scale mesh to a consistent size.",
    )
    parser.add_argument(
        "--normalize-size",
        type=float,
        default=1.0,
        help="Target size for the largest mesh dimension after normalization.",
    )

    sim_class.add_parser_arguments(parser)

    args = parser.parse_args(remaining_args)
    args.ground_height = -1.0
    args.collision_radius = 0.5 / args.resolution
    args.up_axis = 1
    args.young_modulus = 10000.0
    args.density = 500.0
    args.poisson_ratio = 0.45

    # Load tetmesh from file
    print(f"Loading mesh from: {args.mesh}")
    msh = meshio.read(args.mesh, file_format="gmsh")
    points_np = msh.points.astype(np.float32)
    if args.normalize:
        bbox_min = points_np.min(axis=0)
        bbox_max = points_np.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        max_extent = float(np.max(bbox_max - bbox_min))
        scale = (args.normalize_size / max_extent) if max_extent > 1e-12 else 1.0
        points_np = (points_np - center) * scale
        print(f"Normalized mesh: center ~ 0, max dimension -> {args.normalize_size}")
    pos = wp.array(points_np, dtype=wp.vec3f)
    assert (
        msh.cells[0].type == "tetra"
    ), f"Expected tetra cells, got {msh.cells[0].type}"
    tets = wp.array(msh.cells[0].data, dtype=wp.int32)
    pos.requires_grad = True
    geo = fem.Tetmesh(positions=pos, tet_vertex_indices=tets, build_bvh=True)

    vtx_quadrature = fem.PicQuadrature(fem.Cells(geo), pos)

    print(f"Mesh loaded: {pos.shape[0]} vertices, {tets.shape[0]} tetrahedra")

    # identify cells with > 0 material fraction
    fraction_space = fem.make_polynomial_space(geo, dtype=float, degree=0)
    fraction_test = fem.make_test(fraction_space)
    fraction = fem.integrate(material_fraction_form, fields={"phi": fraction_test})
    active_cells = wp.array(dtype=int, shape=fraction.shape)
    wp.launch(mark_active, dim=fraction.shape, inputs=[fraction, active_cells])

    sim = sim_class(geo, active_cells, args)
    sim.init_displacement_space(None)
    sim.init_strain_spaces()
    sim.init_collision_detector(vtx_quadrature)

    if args.materials:
        print(f"\n{'='*60}")
        print("Applying spatially varying material properties...")
        print(f"{'='*60}")
        material_stats = apply_spatially_varying_materials(
            sim, args.materials, k_neighbors=args.k_neighbors
        )

        # spatially varying density at velocity nodes (correct DOF locations)
        voxel_coords, voxel_materials = load_material_data(args.materials)
        upsampler = MaterialUpsampler(voxel_coords, voxel_materials)
        vel_node_pos = sim.u_field.space.node_positions().numpy()
        mats_u, _ = upsampler.interpolate(vel_node_pos, k=args.k_neighbors)
        density_u = mats_u[:, 2]
        if hasattr(sim, "set_density_from_array"):
            sim.set_density_from_array(density_u)
            print(
                f"Assigned spatial density to {density_u.shape[0]} velocity nodes:"
                f" min={density_u.min():.2f}, max={density_u.max():.2f}"
            )
    else:
        print(f"\nUsing uniform material properties:")
        print(f"  Young's modulus: {args.young_modulus}")
        print(f"  Poisson's ratio: {args.poisson_ratio}")
        print(f"  Density: {args.density}")

    if args.clamping == "sides":
        boundary_projector_form = clamped_sides
    elif args.clamping == "edge":
        boundary_projector_form = clamped_edge
    else:
        boundary_projector_form = clamped_right

    sim.set_boundary_condition(
        boundary_projector_form=boundary_projector_form,
        boundary_displacement_form=boundary_displacement_form,
        boundary_displacement_args={
            "displacement": args.displacement / max(1, args.n_frames)
        },
    )

    run_softbody_sim(sim, ui=args.ui)
