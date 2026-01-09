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

import numpy as np

import warp as wp
import warp.fem as fem
from warp.fem import Domain, Sample, Field

from fem_examples.mfem.softbody_sim import (
    SoftbodySim,
    ClassicFEM,
    dg_penalty_form,
    external_forces_form,
    VolumetricForces,
)


@fem.integrand
def inertia_form_density(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
    rho: Field,
    dt: float,
):
    """<rho/dt^2 u, v> with spatially varying density."""

    u_rhs = rho(s) * u(s) / (dt * dt)
    return wp.dot(u_rhs, v(s))


@fem.integrand
def displacement_rhs_form_density(
    s: Sample,
    domain: Domain,
    u: Field,
    u_prev: Field,
    v: Field,
    rho: Field,
    gravity: wp.vec3,
    dt: float,
):
    """<rho/dt^2 u_prev, v> - <rho/dt^2 u, v> + <rho g, v> with spatially varying density."""

    f = (
        inertia_form_density(s, domain, u_prev, v, rho, dt)
        - inertia_form_density(s, domain, u, v, rho, dt)
        + rho(s) * wp.dot(gravity, v(s))
    )
    return f


@fem.integrand
def kinetic_potential_energy_density(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
    rho: Field,
    gravity: wp.vec3,
    dt: float,
    forces: VolumetricForces,
):
    """
    kinetic+potential energy with spatially varying density.

    0.5 * <rho/dt^2 |u - v|^2> - <rho g, u>
    """

    du = u(s) - v(s)
    kinetic = 0.5 * rho(s) * wp.dot(du, du) / (dt * dt)
    potential = -rho(s) * wp.dot(gravity, u(s))
    # Subtract external forces potential energy contribution to match base behavior
    E = kinetic + potential
    E -= external_forces_form(s, domain, u, forces)
    return E


class SoftbodySimDensity(SoftbodySim):
    """Extension of SoftbodySim that supports spatially varying density.

    Adds a scalar density field `rho_field` defined on the same collocated basis as velocity.
    Overrides mass assembly, RHS, and energy to use the density field.
    """

    def init_displacement_space(self, side_subdomain: fem.Domain):
        super().init_displacement_space(side_subdomain)

        # Collocated scalar field for per-vertex density
        rho_space = fem.make_collocated_function_space(self._vel_basis, dtype=float)
        self.rho_field = rho_space.make_field(
            space_partition=self.u_field.space_partition
        )
        self.rho_field.dof_values.fill_(self.args.density)

    def set_density_from_array(self, density_per_vertex: np.ndarray):
        """Assign per-vertex density values (shape must match velocity DOFs)."""

        if density_per_vertex.ndim != 1:
            raise ValueError(
                "density_per_vertex must be a 1D array of length = node_count"
            )

        node_count = self.u_test.space_partition.node_count()
        if density_per_vertex.shape[0] != node_count:
            raise ValueError(
                f"Expected density array of length {node_count}, got {density_per_vertex.shape[0]}"
            )

        self.rho_field.dof_values = wp.array(density_per_vertex, dtype=float)

    def init_constant_forms(self):
        args = self.args

        self.update_force_weight()

        if args.matrix_free:
            self.A = None
            return

        if self.args.lumped_mass:
            self.A = fem.integrate(
                inertia_form_density,
                fields={"u": self.u_trial, "v": self.u_test, "rho": self.rho_field},
                values={"dt": self.dt},
                output_dtype=float,
                nodal=True,
            )
        else:
            self.A = fem.integrate(
                inertia_form_density,
                fields={"u": self.u_trial, "v": self.u_test, "rho": self.rho_field},
                values={"dt": self.dt},
                output_dtype=float,
                quadrature=self.vel_quadrature,
            )

        if self.side_quadrature is not None:
            self.A += fem.integrate(
                dg_penalty_form,
                fields={"u": self.u_side_trial, "v": self.u_side_test},
                values={"k": self.typical_stiffness * self.args.dg_jump_pen},
                quadrature=self.side_quadrature,
                output_dtype=float,
            )

        if self._penalty_lhs_form:
            self.A += fem.integrate(
                self._penalty_lhs_form,
                fields={"u": self.u_trial, "v": self.u_test},
                output_dtype=float,
                quadrature=self.vel_quadrature,
            )

        self.A.nnz_sync()

    def constraint_free_rhs(self, dt=None, with_external_forces=True, tape=None):
        gravity = self.gravity if with_external_forces else wp.vec3(0.0)

        with_gradient = tape is not None
        rhs_tape = wp.Tape() if tape is None else tape
        rhs = wp.zeros(
            dtype=wp.vec3,
            requires_grad=with_gradient,
            shape=self.u_test.space_partition.node_count(),
        )

        with rhs_tape:
            fem.integrate(
                displacement_rhs_form_density,
                fields={
                    "u": self.du_field,
                    "u_prev": self.du_prev,
                    "v": self.u_test,
                    "rho": self.rho_field,
                },
                values={"dt": self._step_dt(), "gravity": gravity},
                output=rhs,
                quadrature=self.vel_quadrature,
                kernel_options={"enable_backward": True},
            )

            if self.side_quadrature is not None:
                fem.integrate(
                    dg_penalty_form,
                    fields={"u": self.u_field.trace(), "v": self.u_side_test},
                    values={"k": -self.typical_stiffness * self.args.dg_jump_pen},
                    quadrature=self.side_quadrature,
                    output=rhs,
                    add=True,
                    kernel_options={"enable_backward": True},
                )

            if self._penalty_rhs_form:
                fem.integrate(
                    self._penalty_rhs_form,
                    fields={"u": self.u_field, "v": self.u_test},
                    output=rhs,
                    add=True,
                    quadrature=self.vel_quadrature,
                    kernel_options={"enable_backward": True},
                )

        if with_external_forces and self.forces.count > 0:
            fem.integrate(
                external_forces_form,
                fields={"v": self.u_test},
                values={"forces": self.forces},
                output_dtype=wp.vec3,
                quadrature=self.vel_quadrature,
                kernel_options={"enable_backward": False},
                output=rhs,
                add=True,
            )

        return rhs

    def evaluate_energy(self, E_u=None, cr=None):
        E_u = fem.integrate(
            kinetic_potential_energy_density,
            quadrature=self.vel_quadrature,
            fields={"u": self.du_field, "v": self.du_prev, "rho": self.rho_field},
            values={
                "dt": self._step_dt(),
                "gravity": self.gravity,
                "forces": self.forces,
            },
            output=E_u,
        )

        if self.side_quadrature is not None:
            if wp.types.is_array(E_u):
                Eu_dg = wp.empty_like(E_u)
            else:
                Eu_dg = None

            Eu_dg = fem.integrate(
                dg_penalty_form,
                fields={"u": self.u_field.trace(), "v": self.u_field.trace()},
                values={"k": self.typical_stiffness * self.args.dg_jump_pen},
                quadrature=self.side_quadrature,
                output=Eu_dg,
            )

            if wp.types.is_array(E_u):
                fem.utils.array_axpy(y=E_u, x=Eu_dg)
            else:
                E_u += Eu_dg

        return E_u, 0.0


class ClassicFEMWithDensity(SoftbodySimDensity, ClassicFEM):
    """ClassicFEM variant that supports spatially varying density via `rho_field`."""

    pass
