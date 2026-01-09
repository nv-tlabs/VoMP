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

import math
import argparse

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

from fem_examples.mfem.mfem_3d import MFEM_RS_F
from fem_examples.mfem.softbody_sim import run_softbody_sim

import polyscope as ps


@fem.integrand
def boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """Dirichlet boundary condition projector (fixed vertices selection)"""

    nor = fem.normal(domain, s)
    pos = domain(s)
    clamped = float(0.0)

    # Single clamped point
    # if s.qp_index < 10:
    #  clamped = 1.0

    # clamped vertical sides
    # clamped = wp.abs(nor[0])

    # clamped right sides
    clamped = wp.where(pos[0] < 1.0, 0.0, 1.0)

    return wp.dot(u(s), v(s)) * clamped


@fem.integrand
def refinement_field(s: fem.Sample, stress: fem.Field, max_p: float):
    p = wp.abs(wp.trace(stress(s))) / max_p
    return wp.max(0.0, 1.0 - p)


if __name__ == "__main__":
    # wp.config.verify_cuda = True
    # wp.config.verify_fp = True
    wp.init()

    # sim_class = ClassicFEM
    sim_class = MFEM_RS_F

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--displacement", type=float, default=0.0)
    sim_class.add_parser_arguments(parser)
    args = parser.parse_args()

    args.grid = True

    vol1 = fem_example_utils.gen_volume(
        # res=wp.vec3i(args.resolution), bounds_lo=wp.vec3(0.0, 0.5, 0.875)
        res=wp.vec3i(args.resolution),
        bounds_lo=wp.vec3(0.0, 0.75, 0.75),
    )
    coarse_grid = fem.Nanogrid(vol1)

    ref_field = fem.make_polynomial_space(coarse_grid, degree=3).make_field()
    ref_field.dof_values.fill_(1.0)

    def frame_callback(displaced_pos):
        sim.interpolate_constraint_field(strain=False)

        vol_mesh = ps.get_volume_mesh("volume mesh")
        stress = sim.interpolated_constraint_field.dof_values.numpy()
        stress_n = np.abs(np.trace(stress, axis1=1, axis2=2))
        max_stress = np.max(stress_n)
        print(max_stress)
        print(stress_n.shape)
        vol_mesh.add_scalar_quantity("stress", stress_n, enabled=True)

    for k in range(4):
        geo = fem.adaptive_nanogrid_from_field(
            coarse_grid=vol1, level_count=4, refinement_field=ref_field, grading="face"
        )

        sim = sim_class(geo, active_cells=None, args=args)
        sim.init_displacement_space()
        sim.init_strain_spaces()

        sim.set_boundary_condition(
            boundary_projector_form=boundary_projector_form,
        )

        run_softbody_sim(sim, frame_callback=frame_callback)

        sim.interpolate_constraint_field(strain=False)
        stress_field = fem.NonconformingField(
            fem.Cells(coarse_grid),
            sim.interpolated_constraint_field,
            background=wp.mat33f(100.0),
        )
        fem.interpolate(
            refinement_field,
            dest=ref_field,
            fields={"stress": stress_field},
            values={"max_p": 50.0},
        )

        # pc = ps.register_point_cloud("ref", ref_field.space.node_positions().numpy())
        # c.add_scalar_quantity("ref", ref_field.dof_values.numpy(), enabled=True)
        # ps.show()
