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
from fem_examples.mfem.mfem_3d import MFEM_RS_F, MFEM_sF_S

import warp.examples.fem.utils as fem_example_utils

# Demo ap


@wp.func
def material_fraction(x: wp.vec3):
    # arbitrary notch in the grid
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
    clamped = wp.where(pos[0] < 1.0, 0.0, 1.0)

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


if __name__ == "__main__":
    # wp.config.verify_cuda = True
    # wp.config.verify_fp = True
    wp.init()

    class_parser = argparse.ArgumentParser()
    class_parser.add_argument(
        "--variant", "-v", choices=["mfem", "classic", "trusty"], default="mfem"
    )
    class_args, remaining_args = class_parser.parse_known_args()

    if class_args.variant == "mfem":
        sim_class = MFEM_RS_F
    elif class_args.variant == "trusty":
        sim_class = MFEM_sF_S
    else:
        sim_class = ClassicFEM

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--displacement", type=float, default=0.0)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction)
    parser.add_argument("--clamping", type=str, default="right")
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
    sim_class.add_parser_arguments(parser)
    args = parser.parse_args(remaining_args)

    if args.grid:
        geo = fem.Grid3D(
            res=wp.vec3i(args.resolution), bounds_lo=wp.vec3(0.0, 0.75, 0.75)
        )
    else:
        pos, tets = fem_example_utils.gen_tetmesh(
            res=wp.vec3i(args.resolution), bounds_lo=wp.vec3(0.0, 0.75, 0.75)
        )
        pos.requires_grad = True
        geo = fem.Tetmesh(positions=pos, tet_vertex_indices=tets)

    # identify cells with > 0 material fraction
    fraction_space = fem.make_polynomial_space(geo, dtype=float, degree=0)
    fraction_test = fem.make_test(fraction_space)
    fraction = fem.integrate(material_fraction_form, fields={"phi": fraction_test})
    active_cells = wp.array(dtype=int, shape=fraction.shape)
    wp.launch(mark_active, dim=fraction.shape, inputs=[fraction, active_cells])

    sim = sim_class(geo, active_cells, args)
    sim.init_displacement_space()
    sim.init_strain_spaces()

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
