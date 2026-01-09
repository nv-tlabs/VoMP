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

"""
This example solves the Navier-Stokes equations for a velocity field using semi-Lagrangian advection:

Re Du / dt - Div[ 2 D(u) ] = 0   # conservation of momentum
div u = 0                        # incompressibility
with u = 0 on the domain boundary, except at the channel outlet

Time integration is performed using a BDF2 scheme as described in Section 2.5 of this document :
https://membres-ljk.imag.fr/Pierre.Saramito/rheolef/rheolef.pdf

"""

import argparse

import numpy as np

import warp as wp
import warp.fem as fem

from warp.fem.utils import array_axpy

from warp.examples.fem.utils import SaddleSystem, bsr_solve_saddle, Plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import meshio


@fem.integrand
def inertia_form(s: fem.Sample, u: fem.Field, v: fem.Field, dt: float):
    return wp.dot(u(s), v(s)) / dt


@fem.integrand
def viscosity_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return 2.0 * wp.ddot(fem.D(u, s), fem.D(v, s))


@fem.integrand
def bdf2_viscosity_and_inertia_form(
    s: fem.Sample, u: fem.Field, v: fem.Field, dt: float, Re: float
):
    return 1.5 * Re * inertia_form(s, u, v, dt) + viscosity_form(s, u, v)


@fem.integrand
def bdf2_transported_inertia_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    u_prev: fem.Field,
    v: fem.Field,
    dt: float,
    Re: float,
):
    pos = domain(s)

    vel = u(s)
    vel_prev = u_prev(s)

    vel_star = 2.0 * vel - vel_prev

    conv_pos = pos - 1.0 * vel_star * dt
    conv_s = fem.lookup(domain, conv_pos, s)
    conv_vel = wp.select(conv_pos[0] >= 16.0, u(conv_s), wp.vec2(0.0))

    conv_pos_prev = pos - 2.0 * vel_star * dt
    conv_s_prev = fem.lookup(domain, conv_pos_prev, s)
    conv_vel_prev = wp.select(
        conv_pos_prev[0] >= 16.0, u_prev(conv_s_prev), wp.vec2(0.0)
    )

    return 0.5 * Re * wp.dot(4.0 * conv_vel - conv_vel_prev, v(s)) / dt


# Incompressibility
@fem.integrand
def div_form(
    s: fem.Sample,
    u: fem.Field,
    q: fem.Field,
):
    return -q(s) * fem.div(u, s)


# Vorticity field, for visualization
@fem.integrand
def curl_field(
    s: fem.Sample,
    u: fem.Field,
):
    return fem.curl(u, s)


# No-slip boundary condition on walls, free velocity at outlet
@fem.integrand
def u_boundary_projector_form(
    s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field
):
    x = fem.position(domain, s)

    if x[0] >= 16.0:
        return 0.0

    if x[0] <= 0.0:
        return wp.dot(u(s), v(s))

    n = fem.normal(domain, s)
    return wp.dot(n, u(s)) * wp.dot(n, v(s))

    # if x[1] <= 0.0 or x[1] >= 4.0:
    #    n = fem.normal(domain, s)
    #    return wp.dot(n, u(s)) * wp.dot(n, v(s))

    # return wp.dot(u(s), v(s))


@fem.integrand
def u_boundary_value_form(
    s: fem.Sample, domain: fem.Domain, v: fem.Field, in_velocity: float
):
    x = fem.position(domain, s)

    if x[0] <= 0.0:
        return in_velocity * v(s)[0]

    return 0.0


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("-n", "--num_frames", type=int, default=1000)
    parser.add_argument("--Re", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.025)

    def __init__(self, stage=None, quiet=False, args=None, **kwargs):
        if args is None:
            # Read args from kwargs, add default arg values from parser
            args = argparse.Namespace(**kwargs)
            args = Example.parser.parse_args(args=[], namespace=args)
        self._args = args
        self._quiet = quiet

        self.sim_dt = args.dt
        self.current_frame = 0

        # Read mesh and create FEM geometry
        mesh: meshio.Mesh = meshio.read(args.mesh)
        positions = wp.array(mesh.points[:, :2], dtype=wp.vec2)
        tri_vidx = wp.array(mesh.cells_dict["triangle"], dtype=int)
        self._geo = fem.Trimesh2D(
            tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True
        )

        # Velocity/pressure function spaces: P_k / P_{k-1}
        u_space = fem.make_polynomial_space(
            self._geo,
            degree=args.degree,
            dtype=wp.vec2,
            element_basis=fem.ElementBasis.LAGRANGE,
        )
        p_space = fem.make_polynomial_space(
            self._geo,
            degree=args.degree - 1,
            dtype=float,
            element_basis=fem.ElementBasis.LAGRANGE,
        )

        # Displacement and displacement delta (i.e. velocity * dt) fields
        self.u_field = u_space.make_field()
        self._u_prev_field = u_space.make_field()
        # Pressure field
        self._p_field = p_space.make_field()

        # Assemble constant matrices

        # Inertia and elasticity terms
        domain = fem.Cells(geometry=self._geo)
        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)

        u_matrix = fem.integrate(
            bdf2_viscosity_and_inertia_form,
            fields={"u": u_trial, "v": u_test},
            values={
                "dt": args.dt,
                "Re": self._args.Re,
            },
        )

        # Incompressbility constraint
        p_test = fem.make_test(space=p_space, domain=domain)
        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

        # Boundary conditions
        boundary = fem.BoundarySides(self._geo)

        # Build our projectors
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_bd_projector = fem.integrate(
            u_boundary_projector_form,
            fields={
                "u": u_bd_trial,
                "v": u_bd_test,
            },
            nodal=True,
        )
        u_bd_value = fem.integrate(
            u_boundary_value_form,
            fields={
                "v": u_bd_test,
            },
            values={"in_velocity": 1.0},
            nodal=True,
            output_dtype=wp.vec2d,
        )
        fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value)

        # Project velocity matrix and boundary right hand side
        u_bd_rhs = wp.zeros_like(u_bd_value)
        fem.project_linear_system(
            u_matrix, u_bd_rhs, u_bd_projector, u_bd_value, normalize_projector=False
        )

        # Project pressure boundary right-hand-side
        div_bd_rhs = -div_matrix @ u_bd_value

        # Project divergence matrix
        div_matrix -= div_matrix @ u_bd_projector

        # Assemble saddle system
        self._saddle_system = SaddleSystem(u_matrix, div_matrix)

        # Save boundary projector and prescribed values for later
        self._u_bd_projector = u_bd_projector
        self._u_bd_rhs = u_bd_rhs
        self._u_test = u_test
        self._div_bd_rhs = div_bd_rhs

        # For visualization only
        curl_space = fem.make_polynomial_space(
            self._geo,
            degree=args.degree,
            dtype=float,
        )
        self.curl_field = curl_space.make_field()

    def update(self):
        # Use Lobatto quadrature for stability of semi-Lagrangian advection
        quadrature = fem.RegularQuadrature(
            self._u_test.domain,
            order=2 * args.degree,
            family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE,
        )

        u_rhs = fem.integrate(
            bdf2_transported_inertia_form,
            quadrature=quadrature,
            fields={
                "u": self.u_field,
                "u_prev": self._u_prev_field,
                "v": self._u_test,
            },
            values={"dt": self.sim_dt, "Re": self._args.Re},
            output_dtype=wp.vec2d,
        )

        # Apply boundary conditions
        # u_rhs = (I - P) * u_rhs + u_bd_rhs
        wp.sparse.bsr_mv(self._u_bd_projector, x=u_rhs, y=u_rhs, alpha=-1.0, beta=1.0)
        array_axpy(x=self._u_bd_rhs, y=u_rhs, alpha=1.0, beta=1.0)

        p_rhs = self._div_bd_rhs

        x_u = wp.empty_like(u_rhs)
        x_p = wp.empty_like(p_rhs)
        wp.utils.array_cast(out_array=x_u, in_array=self.u_field.dof_values)
        wp.utils.array_cast(out_array=x_p, in_array=self._p_field.dof_values)

        bsr_solve_saddle(
            saddle_system=self._saddle_system,
            tol=1.0e-8,
            x_u=x_u,
            x_p=x_p,
            b_u=u_rhs,
            b_p=p_rhs,
            quiet=True,
            method="bicgstab",
        )

        # Extract result
        wp.copy(src=self.u_field.dof_values, dest=self._u_prev_field.dof_values)
        wp.utils.array_cast(in_array=x_u, out_array=self.u_field.dof_values)
        wp.utils.array_cast(in_array=x_p, out_array=self._p_field.dof_values)

        self.current_frame += 1

        return np.max(np.abs(self.u_field.dof_values.numpy()))

    def render(self):
        # Save curl data for visualization
        fem.interpolate(curl_field, dest=self.curl_field, fields={"u": self.u_field})


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    args = Example.parser.parse_args()

    example = Example(args=args)
    plot = Plot()

    for k in range(args.num_frames):
        max_vel = example.update()

        if example.current_frame % 10 == 1:
            example.render()
            plot.add_field("curl", example.curl_field)

        print(f"Done frame {k} with maximum velocity = {max_vel}")

    # plot.add_field("velocity_final", example.u_field)
    plot.plot(
        options={
            "velocity_final": {"streamlines": {"density": 2}},
            "curl": {
                "contours": {"levels": [-10, -5, -2, -1, 1, 2, 5, 10]},
                "clim": (-10, 10.0),
            },
        }
    )
