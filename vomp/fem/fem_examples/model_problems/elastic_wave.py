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
This example solves an elastic wave propagation problem for a displacement field u:

rho d2 u / dt2 - Div[ sigma + phi(t) Id ] = 0   # conservation of momentum
sigma = mu * eps + lambda * Trace(eps) * Id     # stress-strain relationship
eps = D(u) := 1/2 (grad u + (grad u)^T)         # strain tensor
with u.n = 0 on the domain boundary

with phi(t) a time-varying source term


After integrating against test functions and using integration by part for the stress terms,
we can express a backwards Euler integrator as follow:

m(du^k, v) + k(du^k, v) = m(du^{k-1}, v) - k(u^{k+1}, v) + f(t, v)  for all v
du^k.n = 0 on the domain boundary
u^k = u^{k+1} + du^K

with:
m(u, v) = int{ u.v / dt^2 }
k(u, v) = int{ sigma(D(u)) : D(v) }  with sigma(eps) the strain-stress relationship
f(t, v) = int{ phi(t) div(v) } the source term

This yields a linear system (M + K) du = rhs, which is then projected to satisfy boudary conditions
and solved with conjugate gradient

"""

import argparse

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils


import meshio


# Corresponds to sigma(eps) strain-stress relationship
@wp.func
def strain_stress_relationship(strain: wp.mat22, lame_lambda: float, lame_mu: float):
    return 2.0 * lame_mu * strain + lame_lambda * wp.trace(strain) * wp.identity(
        n=2, dtype=float
    )


# Corresponds to k(u, v) bilinear form
@fem.integrand
def stress_form(
    s: fem.Sample, u: fem.Field, v: fem.Field, lame_lambda: float, lame_mu: float
):
    strain = fem.D(u, s)
    stress = strain_stress_relationship(strain, lame_lambda, lame_mu)
    return wp.ddot(fem.D(v, s), stress)


# Corresponds to m(u, v) bilinear form
@fem.integrand
def inertia_form(s: fem.Sample, u: fem.Field, v: fem.Field, rho: float, dt: float):
    return wp.dot(u(s), v(s)) * rho / (dt * dt)


# Corresponds to phi(t) source term
@wp.func
def source_term(x: wp.vec2, t: float, dt: float, dx: float):
    src = wp.sin(t / (20.0 * dt))

    r_sq = wp.length_sq(x) / (4.0 * dx * dx)
    return src * wp.max(0.0, 1.0 - r_sq)


# Sum of all rhs terms
@fem.integrand
def rhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    du: fem.Field,
    v: fem.Field,
    rho: float,
    t: float,
    dt: float,
    dx: float,
    x0: wp.vec2,
    lame_lambda: float,
    lame_mu: float,
):
    return (
        inertia_form(s, du, v, rho, dt)
        - stress_form(s, u, v, lame_lambda, lame_mu)
        + lame_lambda * source_term(domain(s) - x0, t, dt, dx) * fem.div(v, s)
    )


# Boundary normal field, for free-slip boundary condition.
# Used to average the discontinuous per-side normals at nodes
@fem.integrand
def normal_field(s: fem.Sample, domain: fem.Domain):
    return fem.normal(domain, s)


# Free-slip (reflecting) boundary condition
@fem.integrand
def boundary_projector_form(
    s: fem.Sample, u: fem.Field, v: fem.Field, normal: fem.Field
):
    n = normal(s)
    return wp.dot(u(s), n) * wp.dot(v(s), n)


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--resolution", type=int, default=25)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("-n", "--n_frames", type=int, default=250)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--young_modulus", type=float, default=100.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.95)
    parser.add_argument("--dt", type=float, default=0.01)

    def __init__(self, stage=None, quiet=False, args=None, **kwargs):
        if args is None:
            # Read args from kwargs, add default arg values from parser
            args = argparse.Namespace(**kwargs)
            args = Example.parser.parse_args(args=[], namespace=args)
        self._args = args
        self._quiet = quiet

        self.current_time = 0.0

        # Read mesh and create FEM geometry
        mesh: meshio.Mesh = meshio.read(args.mesh)
        positions = wp.array(mesh.points[:, :2], dtype=wp.vec2)
        tri_vidx = wp.array(mesh.cells_dict["triangle"], dtype=int)
        self._geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)

        # Convert Young/Poisson to Lame coefficients
        young = args.young_modulus
        poisson = args.poisson_ratio
        self.lame_lambda = young / (1.0 + poisson) * poisson / (1.0 - poisson)
        self.lame_mu = 0.5 * young / (1.0 + poisson)

        # Displacement function space
        u_space = fem.make_polynomial_space(
            self._geo,
            degree=args.degree,
            dtype=wp.vec2,
            element_basis=fem.ElementBasis.LAGRANGE,
        )

        # Displacement and displacement delta (i.e. velocity * dt) fields
        self.u_field = u_space.make_field()
        self._du_field = u_space.make_field()

        # Assemble constant matrices

        # Inertia and elasticity terms
        domain = fem.Cells(geometry=self._geo)
        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)

        inertia_matrix = fem.integrate(
            inertia_form,
            fields={"u": u_trial, "v": u_test},
            values={
                "dt": args.dt,
                "rho": self._args.density,
            },
        )
        elasticity_matrix = fem.integrate(
            stress_form,
            fields={"u": u_trial, "v": u_test},
            values={"lame_lambda": self.lame_lambda, "lame_mu": self.lame_mu},
        )

        self._system_matrix = elasticity_matrix + inertia_matrix

        # Free-slip boundary conditions: zero in normal direction
        boundary = fem.BoundarySides(self._geo)

        # First we need a continuous normal field; interpolate per-side normals to a continuous field
        vertex_normal_field = u_space.make_field()
        fem.interpolate(
            normal_field,
            dest=fem.make_restriction(vertex_normal_field, domain=boundary),
        )

        # Now build our projector
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_bd_projector = fem.integrate(
            boundary_projector_form,
            fields={
                "u": u_bd_trial,
                "v": u_bd_test,
                "normal": vertex_normal_field.trace(),
            },
            nodal=True,
        )
        fem.dirichlet.normalize_dirichlet_projector(u_bd_projector)
        self._boundary_projector = u_bd_projector

        self._projected_system_matrix = wp.sparse.bsr_copy(self._system_matrix)
        fem.dirichlet.project_system_matrix(
            projector_matrix=self._boundary_projector,
            system_matrix=self._projected_system_matrix,
        )

        # Save test function, we'll reuse for rhs
        self._u_test = u_test

    def update(self):
        # Assemble time-varying righ-hand-side
        u_rhs = fem.integrate(
            rhs_form,
            fields={"du": self._du_field, "u": self.u_field, "v": self._u_test},
            values={
                "rho": self._args.density,
                "dt": self._args.dt,
                "t": self.current_time,
                "dx": 0.1,
                "x0": wp.vec2(0.75, 0.75),
                "lame_lambda": self.lame_lambda,
                "lame_mu": self.lame_mu,
            },
            output_dtype=wp.vec2d,
        )

        # Enforce boundary condition
        fem.dirichlet.project_system_rhs(
            system_matrix=self._system_matrix,
            projector_matrix=self._boundary_projector,
            system_rhs=u_rhs,
        )

        # Solve with CG
        x = wp.zeros_like(u_rhs)
        err, n_iter = fem_example_utils.bsr_cg(
            self._projected_system_matrix, b=u_rhs, x=x, tol=1.0e-16, quiet=self._quiet
        )

        # Extract result and add displacement delta to displacement
        wp.utils.array_cast(in_array=x, out_array=self._du_field.dof_values)
        fem.utils.array_axpy(x=self._du_field.dof_values, y=self.u_field.dof_values)

        self.current_time += self._args.dt

        return err, n_iter

    def render(self):
        self.renderer.add_field("solution", self.u_field)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})  # To speed-up compilation

    args = Example.parser.parse_args()

    sim = Example(args=args, quiet=True)

    plot = fem_example_utils.Plot()

    for f in range(sim._args.n_frames):
        err, niter = sim.update()
        print(f"Done frame {f} with CG residual {err} after {niter} iterations")
        plot.add_field("u", sim.u_field)

    plot.plot({"u": {"displacement": {}}})
