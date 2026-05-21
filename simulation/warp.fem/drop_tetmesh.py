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
from fem_examples.mfem.collisions import CollisionHandler, CollisionPotential
from fem_examples.mfem.mfem_3d import MFEM_RS_F, MFEM_sF_S

import meshio
import numpy as np

from material_loader import (
    apply_spatially_varying_materials,
    visualize_material_distribution,
    load_material_data,
)
from vomp.inference.utils import MaterialUpsampler


running = False
def _run_with_ground_ui(sim):
    import polyscope as ps
    import polyscope.imgui as psim

    active_cells = None if sim.cells is None else sim.cells.array.numpy()

    try:
        hexes = sim.u_field.space.node_hexes()
        if active_cells is not None:
            hex_per_cell = len(hexes) // sim.geo.cell_count()
            selected = np.broadcast_to(
                (active_cells * hex_per_cell).reshape(-1, 1),
                shape=(len(active_cells), hex_per_cell),
            ) + np.broadcast_to(
                np.arange(hex_per_cell).reshape(1, -1),
                shape=(len(active_cells), hex_per_cell),
            )
            hexes = hexes[selected.flatten()]
    except AttributeError:
        hexes = None

    if hexes is None:
        try:
            tets = sim.u_field.space.node_tets()
            if active_cells is not None:
                tet_per_cell = len(tets) // sim.geo.cell_count()
                selected = np.broadcast_to(
                    (active_cells * tet_per_cell).reshape(-1, 1),
                    shape=(len(active_cells), tet_per_cell),
                ) + np.broadcast_to(
                    np.arange(tet_per_cell).reshape(1, -1),
                    shape=(len(active_cells), tet_per_cell),
                )
                tets = tets[selected.flatten()]
        except AttributeError:
            tets = None
    else:
        tets = None

    ps.init()
    ps.set_ground_plane_height(sim.args.ground_height)

    node_pos = sim.u_field.space.node_positions().numpy()
    ps_vol = ps.register_volume_mesh(
        "volume mesh", node_pos, hexes=hexes, tets=tets, edge_width=1.0
    )
    ps.register_volume_mesh(
        "reference mesh",
        node_pos,
        hexes=hexes,
        tets=tets,
        edge_width=1.0,
        enabled=False,
    )

    sim.init_constant_forms()
    sim.project_constant_forms()
    sim.cur_frame = 0

    active_indices = sim.u_field.space_partition.space_node_indices().numpy()

    def callback():
        global running
        _, running = psim.Checkbox("Running", running)
        if not running:
            return

        sim.cur_frame += 1
        if sim.args.n_frames >= 0 and sim.cur_frame > sim.args.n_frames:
            return

        with wp.ScopedTimer(f"--- Frame --- {sim.cur_frame}", synchronize=True):
            sim.run_frame()

        displaced_pos = sim.u_field.space.node_positions().numpy()
        displaced_pos[active_indices] += sim.u_field.dof_values.numpy()
        ps_vol.update_vertex_positions(displaced_pos)

    ps.set_user_callback(callback)
    ps.show()


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
    clamped = 0.0  # wp.where(pos[0] < 1.0, 0.0, 1.0)

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

    nor = normal(domain, s)

    clamped = wp.abs(nor[0])

    return -displacement * wp.dot(nor, v(s)) * clamped


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
        sim_class = ClassicFEM

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
    CollisionHandler.add_parser_arguments(parser)

    args = parser.parse_args(remaining_args)
    args.ground_height = -1.0
    args.collision_radius = 0.5 / args.resolution
    args.up_axis = 1
    args.young_modulus = 10000.0
    args.density = 500.0
    args.poisson_ratio = 0.45
    args.dt = 0.1
    args.gravity = 10.0
    args.cg_tol = 1.0e-8
    args.cg_iters = 1000

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

    fraction_space = fem.make_polynomial_space(geo, dtype=float, degree=0)
    fraction_test = fem.make_test(fraction_space)
    fraction = fem.integrate(material_fraction_form, fields={"phi": fraction_test})
    active_cells = wp.array(dtype=int, shape=fraction.shape)
    wp.launch(mark_active, dim=fraction.shape, inputs=[fraction, active_cells])

    sim = sim_class(geo, active_cells, args)
    sim.init_displacement_space()
    sim.init_strain_spaces()

    collision_handler = CollisionHandler(
        kinematic_meshes=[],
        cp_cell_indices=vtx_quadrature.cell_indices,
        cp_cell_coords=vtx_quadrature.particle_coords,
    )
    sim.add_energy_potential(CollisionPotential(sim, collision_handler))

    if args.materials:
        print(f"\n{'='*60}")
        print("Applying spatially varying material properties...")
        print(f"{'='*60}")
        material_stats = apply_spatially_varying_materials(
            sim, args.materials, k_neighbors=args.k_neighbors
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

    if args.ui:
        _run_with_ground_ui(sim)
    else:
        run_softbody_sim(sim, ui=False)
