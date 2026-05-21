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
import weakref
from typing import List, Optional, TextIO

import numpy as np

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.examples.fem.utils import bsr_cg
from warp.fem import Domain, Field, Sample
from warp.fem.utils import array_axpy
from warp.optim.linear import LinearOperator

from .elastic_models import (
    hooke_energy,
    hooke_hessian,
    hooke_stress,
    snh_energy,
    snh_hessian_proj_analytic,
    snh_stress,
    symmetric_strain,
    symmetric_strain_delta,
)
from .linalg import diff_bsr_mv, project_system_matrix, normalize_dirichlet_projector
from .linesearch import (
    LineSearch,
    LineSearchNaiveCriterion,
    LineSearchUnconstrainedArmijoCriterion,
)

wp.set_module_options({"enable_backward": False})
wp.set_module_options({"max_unroll": 4})
wp.set_module_options({"fast_math": True})


@fem.integrand
def defgrad(u: fem.Field, s: fem.Sample):
    return fem.grad(u, s) + wp.identity(n=3, dtype=float)


@fem.integrand
def defgrad_avg(u: fem.Field, s: fem.Sample):
    return fem.grad_average(u, s) + wp.identity(n=3, dtype=float)


@fem.integrand
def inertia_form(s: Sample, domain: Domain, u: Field, v: Field, rho: Field, dt: float):
    """<rho/dt^2 u, v>"""

    u_rhs = rho(s) * u(s) / (dt * dt)
    return wp.dot(u_rhs, v(s))


@fem.integrand
def dg_penalty_form(s: Sample, domain: Domain, u: Field, v: Field, k: float):
    ju = fem.jump(u, s)
    jv = fem.jump(v, s)

    return wp.dot(ju, jv) * k * fem.measure_ratio(domain, s)


@fem.integrand
def displacement_rhs_form(
    s: Sample,
    domain: Domain,
    u: Field,
    u_prev: Field,
    v: Field,
    rho: Field,
    gravity: wp.vec3,
    dt: float,
):
    """<rho/dt^2 u, v> + <rho g, v>"""
    f = (
        inertia_form(s, domain, u_prev, v, rho, dt)
        - inertia_form(s, domain, u, v, rho, dt)
        + rho(s) * wp.dot(gravity, v(s))
    )

    return f


@fem.integrand
def kinetic_potential_energy(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
    rho: Field,
    dt: float,
    gravity: wp.vec3,
):
    du = u(s)
    dv = v(s)

    E = rho(s) * (0.5 * wp.dot(du - dv, du - dv) / (dt * dt) - wp.dot(du, gravity))

    return E


@wp.kernel(enable_backward=True)
def scale_lame(
    lame_out: wp.array(dtype=wp.vec2),
    lame_ref: wp.vec2,
    scale: wp.array(dtype=float),
):
    i = wp.tid()
    lame_out[i] = lame_ref * scale[i]


class DisplacementPotential:
    """Base class for additional potentials that depend only on the displacement field"""

    def __init__(self, sim):
        self.sim = weakref.proxy(sim)

    def prepare_newton_step(self, dt, tape):
        pass

    def prepare_frame(self, dt):
        pass

    def init_constant_forms(self):
        pass

    def add_energy(self, E_u: wp.array):
        pass

    def add_hessian(self, lhs: sp.BsrMatrix):
        pass

    def add_forces(self, rhs: wp.array, _tape):
        pass


class SoftbodySim:
    def __init__(self, geo: fem.Geometry, active_cells: Optional[wp.array], args):
        self.args = args
        self.geo = geo

        if self.has_discontinuities() and not self.supports_discontinuities():
            raise TypeError(
                f"Simulator of type {type(self)} does not support discontinuities"
            )

        if self.args.matrix_free and not self.supports_matrix_free():
            raise TypeError(
                f"Simulator of type {type(self)} does not support matrix-free solves"
            )

        # Check if we're simming a subset of the geomtry or the whole thing
        self.geo_partition = fem.Cells(geo).geometry_partition
        self.cells: Optional[wp.array] = None

        if active_cells is not None:
            geo_partition = fem.ExplicitGeometryPartition(geo, cell_mask=active_cells)
            if geo_partition.cell_count() < geo.cell_count():
                self.geo_partition = geo_partition
                self.cells = self.geo_partition._cells

        if not self.args.quiet:
            print(f"Active cells: {self.geo_partition.cell_count()}")

        self.dt = args.dt

        self.up_axis = np.zeros(3)
        self.up_axis[self.args.up_axis] = 1.0

        self.gravity = -self.args.gravity * self.up_axis

        young = args.young_modulus
        poisson = args.poisson_ratio
        self.lame_ref = wp.vec2(
            young / (1.0 + poisson) * np.array([poisson / (1.0 - 2.0 * poisson), 0.5])
        )

        # Typical dimensions, useful to scale numerical stiffness terms
        self.typical_length = 1.0
        self.typical_stiffness = max(
            args.density * args.gravity * self.typical_length,
            min(
                args.young_modulus,  # handle no-gravity, quasistatic case
                args.density
                * self.typical_length**2
                / (args.dt**2),  # handle no-gravity, dynamic case
            ),
        )

        self._ls: LineSearch = LineSearchNaiveCriterion(self)

        self._init_displacement_basis()
        self.energy_potentials: List[DisplacementPotential] = []

        self._collision_projector_form: Optional[fem.Integrand] = None
        self._collision_projector_args = {}

        self.log: Optional[TextIO] = None

    def add_energy_potential(self, potential: DisplacementPotential):
        self.energy_potentials.append(potential)

    def has_discontinuities(self):
        return isinstance(self.geo, fem.AdaptiveNanogrid)

    def supports_discontinuities(self):
        return False

    def supports_matrix_free(self):
        return False

    def _init_displacement_basis(self):
        element_basis = (
            fem.ElementBasis.SERENDIPITY
            if self.args.serendipity
            else fem.ElementBasis.LAGRANGE
        )
        self._vel_basis = fem.make_polynomial_basis_space(
            self.geo,
            degree=self.args.degree,
            element_basis=element_basis,
            discontinuous=self.args.discontinuous,
        )

    def set_displacement_basis(self, basis: Optional[fem.BasisSpace] = None):
        if basis is None:
            self._init_displacement_basis()
        else:
            self._vel_basis = basis

    def init_displacement_space(self, side_subdomain: Optional[fem.Domain] = None):
        args = self.args

        u_space = fem.make_collocated_function_space(self._vel_basis, dtype=wp.vec3)
        u_space_partition = fem.make_space_partition(
            space_topology=self._vel_basis.topology,
            geometry_partition=self.geo_partition,
            with_halo=False,
        )

        # Defines some fields over our function spaces
        self.u_field = u_space.make_field(
            space_partition=u_space_partition
        )  # displacement
        self.du_field = u_space.make_field(
            space_partition=u_space_partition
        )  # displacement delta
        self.du_prev = u_space.make_field(
            space_partition=u_space_partition
        )  # displacement delta
        self.force_field = u_space.make_field(
            space_partition=u_space_partition
        )  # total force field -- for collision filtering

        # Since our spaces are constant, we can also predefine the test/trial functions that we will need for integration
        self.u_trial = fem.make_trial(space=u_space, space_partition=u_space_partition)
        self.u_test = fem.make_test(space=u_space, space_partition=u_space_partition)

        self.vel_quadrature = fem.RegularQuadrature(
            self.u_test.domain, order=2 * args.degree
        )

        # DG style integration on sides for discontinuous elements
        if self.has_discontinuities():
            if side_subdomain is None:
                sides = fem.Sides(self.geo)
            else:
                sides = side_subdomain

            self.u_side_trial = fem.make_trial(
                space=u_space, space_partition=u_space_partition, domain=sides
            )
            self.u_side_test = fem.make_test(
                space=u_space, space_partition=u_space_partition, domain=sides
            )

            self.side_quadrature = fem.RegularQuadrature(
                self.u_side_test.domain, order=2 * args.degree
            )
        else:
            self.side_quadrature = None

        # Create material parameters space with same basis as deformation field
        vertex_basis = fem.make_polynomial_basis_space(self.geo)
        density_space = fem.make_collocated_function_space(vertex_basis, dtype=float)
        lame_space = fem.make_collocated_function_space(vertex_basis, dtype=wp.vec2)

        self.density_field = density_space.make_field()
        self.density_field.dof_values.fill_(self.args.density)
        self.lame_field = lame_space.make_field()
        self.lame_field.dof_values.fill_(self.lame_ref)

        # For interpolating the stress/strain fields back to velocity space
        interpolated_constraint_space = fem.make_collocated_function_space(
            self._vel_basis, dtype=wp.mat33
        )
        self.interpolated_constraint_field = interpolated_constraint_space.make_field(
            space_partition=u_space_partition
        )

        self._mass_space = fem.make_collocated_function_space(
            self._vel_basis, dtype=float
        )
        self._mass = None

    def set_boundary_condition(
        self,
        boundary_projector_form,
        boundary_displacement_form=None,
        boundary_displacement_args=None,
    ):
        u_space = self.u_field.space

        # Displacement boundary conditions
        boundary = fem.BoundarySides(self.geo_partition)

        u_bd_test = fem.make_test(
            space=u_space,
            space_partition=self.u_test.space_partition,
            domain=boundary,
        )
        u_bd_trial = fem.make_trial(
            space=u_space,
            space_partition=self.u_test.space_partition,
            domain=boundary,
        )
        self.v_bd_rhs = None
        if boundary_displacement_form is not None:
            self.v_bd_rhs = fem.integrate(
                boundary_displacement_form,
                fields={"v": u_bd_test},
                values=boundary_displacement_args or {},
                assembly="nodal",
                output_dtype=wp.vec3f,
            )
        if boundary_projector_form is not None:
            self.v_bd_matrix = fem.integrate(
                boundary_projector_form,
                fields={"u": u_bd_trial, "v": u_bd_test},
                assembly="nodal",
                output_dtype=float,
            )
        else:
            self.v_bd_matrix = sp.bsr_zeros(
                self.u_trial.space_partition.node_count(),
                self.u_trial.space_partition.node_count(),
                block_type=wp.mat33,
            )
        normalize_dirichlet_projector(self.v_bd_matrix, self.v_bd_rhs)

    def set_fixed_points_condition(
        self,
        fixed_points_projector_form,
        fixed_point_projector_args=None,
    ):
        self.v_bd_rhs = None
        self.v_bd_matrix = fem.integrate(
            fixed_points_projector_form,
            fields={"u": self.u_trial, "v": self.u_test, "u_cur": self.u_field},
            values=fixed_point_projector_args or {},
            assembly="nodal",
            output_dtype=float,
        )

        self.v_bd_rhs = None
        normalize_dirichlet_projector(self.v_bd_matrix, self.v_bd_rhs)

    def set_fixed_points_displacement(
        self,
        fixed_points_displacement_field=None,
        fixed_points_displacement_args=None,
    ):
        bd_field = self.u_test.space.make_field(
            space_partition=self.u_test.space_partition
        )
        fem.interpolate(
            fixed_points_displacement_field,
            dest=bd_field,
            fields={"u_cur": self.u_field},
            values=fixed_points_displacement_args or {},
        )

        self.v_bd_rhs = bd_field.dof_values

    def set_collision_projector(
        self, collision_projector_form, collision_projector_args=None
    ):
        """Projector to be evaluated at each newton iteration for active-set style collisions"""
        self._collision_projector_form = collision_projector_form
        self._collision_projector_args = collision_projector_args or {}

    def init_constant_forms(self):
        args = self.args

        if args.matrix_free:
            self.A = None
        else:
            if self.args.lumped_mass:
                self.A = fem.integrate(
                    inertia_form,
                    fields={
                        "u": self.u_trial,
                        "v": self.u_test,
                        "rho": self.density_field,
                    },
                    values={"dt": self.dt},
                    output_dtype=float,
                    assembly="nodal",
                )
            else:
                self.A = fem.integrate(
                    inertia_form,
                    fields={
                        "u": self.u_trial,
                        "v": self.u_test,
                        "rho": self.density_field,
                    },
                    values={"dt": self.dt},
                    output_dtype=float,
                    quadrature=self.vel_quadrature,
                )

            if self.side_quadrature is not None and self.args.dg_jump_pen > 0.0:
                self.A += fem.integrate(
                    dg_penalty_form,
                    fields={"u": self.u_side_trial, "v": self.u_side_test},
                    values={"k": self.typical_stiffness * self.args.dg_jump_pen},
                    quadrature=self.side_quadrature,
                    output_dtype=float,
                )

            self.A.nnz_sync()

        for potential in self.energy_potentials:
            potential.init_constant_forms()

    def project_constant_forms(self):
        pass

    def constraint_free_rhs(self, dt=None, with_external_forces=True, tape=None):
        gravity = self.gravity if with_external_forces else wp.vec3(0.0)

        # Quasi-quasistatic: normal dt in lhs (trust region), large dt in rhs (quasistatic)

        with_gradient = tape is not None
        rhs_tape = wp.Tape() if tape is None else tape
        rhs = wp.zeros(
            dtype=wp.vec3,
            requires_grad=with_gradient,
            shape=self.u_test.space_partition.node_count(),
        )

        with rhs_tape:
            fem.integrate(
                displacement_rhs_form,
                output=rhs,
                quadrature=self.vel_quadrature,
                kernel_options={"enable_backward": True},
                fields={
                    "u": self.du_field,
                    "u_prev": self.du_prev,
                    "v": self.u_test,
                    "rho": self.density_field,
                },
                values={
                    "dt": self._step_dt(),
                    "gravity": gravity,
                },
            )

            if self.side_quadrature is not None and self.args.dg_jump_pen > 0.0:
                fem.integrate(
                    dg_penalty_form,
                    fields={"u": self.u_field.trace(), "v": self.u_side_test},
                    values={"k": -self.typical_stiffness * self.args.dg_jump_pen},
                    quadrature=self.side_quadrature,
                    output=rhs,
                    add=True,
                    kernel_options={"enable_backward": True},
                )

        if with_external_forces:
            for potential in self.energy_potentials:
                potential.add_forces(rhs, rhs_tape)

        return rhs

    def constraint_free_lhs(self):
        lhs = sp.bsr_copy(self.A)

        for potential in self.energy_potentials:
            potential.add_hessian(lhs)

        return lhs

    def run_frame(self):
        (self.du_field, self.du_prev) = (self.du_prev, self.du_field)

        if self.args.quasi_quasistatic:
            self.du_prev.dof_values.zero_()

        self.prepare_frame()

        tol = self.args.newton_tol**2

        def host_read(tup):
            return (x[:1].numpy()[0] if isinstance(x, wp.array) else x for x in tup)

        E_cur, C_cur = host_read(self.evaluate_energy())
        cumulative_time = 0.0

        if not self.args.quiet:
            print(f"Newton initial guess: E={E_cur}, Cr={C_cur}")
        if self.log:
            mean_displ = np.mean(
                np.linalg.norm(self.du_field.dof_values.numpy(), axis=1)
            )
            print(
                "\t".join(
                    str(x)
                    for x in (0, E_cur, C_cur, mean_displ, 0.0, 0.0, cumulative_time)
                ),
                file=self.log,
            )

        for k in range(self.args.n_newton):
            with wp.ScopedTimer(f"Iter {k}", print=False) as timer:
                E_ref, C_ref = E_cur, C_cur
                self.checkpoint_newton_values()

                self.prepare_newton_step()
                rhs = self.newton_rhs()
                lhs = self.newton_lhs()
                delta_fields = self.solve_newton_system(lhs, rhs)

                self.apply_newton_deltas(delta_fields)
                E_cur, C_cur = host_read(self.evaluate_energy())

                ddu = delta_fields[0]
                step_size = wp.utils.array_inner(ddu, ddu) / (1 + ddu.shape[0])

                # linear model
                self._ls.build_linear_model(lhs, rhs, delta_fields)

                # Line search
                alpha = 1.0
                for _j in range(self.args.n_backtrack):
                    if self._ls.accept(alpha, E_cur, C_cur, E_ref, C_ref):
                        break

                    alpha = 0.5 * alpha
                    self.apply_newton_deltas(delta_fields, alpha=alpha)
                    E_cur, C_cur = host_read(self.evaluate_energy())

                if not self.args.quiet:
                    print(
                        f"Newton iter {k}: E={E_cur}, Cr={C_cur}, step size {np.sqrt(step_size)}, alpha={alpha}"
                    )

            cumulative_time += timer.elapsed
            if self.log:
                mean_displ = np.mean(
                    np.linalg.norm(self.du_field.dof_values.numpy(), axis=1)
                )
                print(
                    "\t".join(
                        str(x)
                        for x in (
                            k + 1,
                            E_cur,
                            C_cur,
                            mean_displ,
                            step_size,
                            alpha,
                            cumulative_time,
                        )
                    ),
                    file=self.log,
                )

            if step_size < tol:
                break

    def prepare_newton_step(self, tape=None):
        for potential in self.energy_potentials:
            potential.prepare_newton_step(self.dt, tape)

    def prepare_frame(self):
        self.compute_initial_guess()
        for potential in self.energy_potentials:
            potential.prepare_frame(self.dt)

    def checkpoint_newton_values(self):
        self._u_cur = wp.clone(self.u_field.dof_values)
        self._du_cur = wp.clone(self.du_field.dof_values)

    def apply_newton_deltas(self, delta_fields, alpha=1.0):
        # Restore checkpoint
        wp.copy(src=self._u_cur, dest=self.u_field.dof_values)
        wp.copy(src=self._du_cur, dest=self.du_field.dof_values)

        # Add to total displacement
        if alpha == 0.0:
            return

        delta_du = delta_fields[0]
        array_axpy(x=delta_du, y=self.u_field.dof_values, alpha=alpha)
        array_axpy(x=delta_du, y=self.du_field.dof_values, alpha=alpha)

    def _step_dt(self):
        # In fake quasistatic mode, use a large timestep for the rhs computation
        # Note that self.dt is still use to compute lhs (inertia matrix)
        return 1.0e6 if self.args.quasi_quasistatic else self.dt

    def evaluate_energy(self, E_u=None, cr=None):
        if E_u is None:
            E_u = wp.zeros(shape=(1,), dtype=float)

        E_u = fem.integrate(
            kinetic_potential_energy,
            quadrature=self.vel_quadrature,
            fields={"u": self.du_field, "v": self.du_prev, "rho": self.density_field},
            values={
                "dt": self._step_dt(),
                "gravity": self.gravity,
            },
            output=E_u,
            add=True,
        )

        if self.side_quadrature is not None and self.args.dg_jump_pen > 0.0:
            fem.integrate(
                dg_penalty_form,
                fields={"u": self.u_field.trace(), "v": self.u_field.trace()},
                values={"k": 0.5 * self.typical_stiffness * self.args.dg_jump_pen},
                quadrature=self.side_quadrature,
                output=E_u,
                add=True,
            )

        for potential in self.energy_potentials:
            potential.add_energy(E_u)

        return E_u, cr

    def _filter_forces(self, u_rhs, tape, temporary_store=None):
        if self._collision_projector_form is not None:
            # update collision projector, if required
            self.force_field.dof_values = u_rhs

            self.v_bd_matrix = fem.integrate(
                self._collision_projector_form,
                fields={
                    "u": self.u_trial,
                    "v": self.u_test,
                    "u_cur": self.u_field,
                    "f": self.force_field,
                },
                values=self._collision_projector_args,
                assembly="nodal",
                output_dtype=float,
            )
            normalize_dirichlet_projector(self.v_bd_matrix)
            self.project_constant_forms()

        orig_rhs = fem.borrow_temporary_like(u_rhs, temporary_store)
        orig_rhs.array.assign(u_rhs)

        proj_tape = wp.Tape() if tape is None else tape
        with proj_tape:
            diff_bsr_mv(
                A=self.v_bd_matrix,
                x=orig_rhs.array,
                y=u_rhs,
                alpha=-1.0,
                beta=1.0,
                self_adjoint=True,
            )

    def compute_initial_guess(self):
        # Start from last frame pose, known good state, plus dirichlet BC

        if self.v_bd_rhs is None:
            self.du_field.dof_values.zero_()
        else:
            diff_bsr_mv(self.v_bd_matrix, self.v_bd_rhs, self.du_field.dof_values)
            array_axpy(x=self.du_field.dof_values, y=self.u_field.dof_values)

    def scale_lame_field(self, stiffness_scale_array: wp.array):
        """Sets the lame field as a per-vertex multiplier times the reference lame values"""
        wp.launch(
            scale_lame,
            dim=self.lame_field.dof_values.shape[0],
            inputs=[
                self.lame_field.dof_values,
                self.lame_ref,
                stiffness_scale_array,
            ],
        )

    def reset_fields(self):
        self.u_field.dof_values.zero_()

    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--degree", type=int, default=1)
        parser.add_argument(
            "--serendipity", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument("-n", "--n_frames", type=int, default=-1)
        parser.add_argument("--n_newton", type=int, default=2)
        parser.add_argument("--newton_tol", type=float, default=1.0e-4)
        parser.add_argument("--cg_tol", type=float, default=1.0e-6)
        parser.add_argument("--cg_iters", type=float, default=250)
        parser.add_argument("--n_backtrack", type=int, default=4)
        parser.add_argument("--young_modulus", type=float, default=1000.0)
        parser.add_argument("--poisson_ratio", type=float, default=0.4)
        parser.add_argument("--gravity", type=float, default=1.0)
        parser.add_argument("--up_axis", "-up", type=int, default=1)
        parser.add_argument("--density", type=float, default=1.0)
        parser.add_argument("--dt", type=float, default=0.1)
        parser.add_argument(
            "--quasi_quasistatic", "-qqs", action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "-nh", "--neo_hookean", action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "-dg", "--discontinuous", action=argparse.BooleanOptionalAction
        )
        parser.add_argument("--dg_jump_pen", type=float, default=1.0)
        parser.add_argument("-ss", "--step_size", type=float, default=0.001)
        parser.add_argument(
            "--quiet", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument("--lumped_mass", action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--fp64", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument(
            "--matrix_free", action=argparse.BooleanOptionalAction, default=False
        )


class ClassicFEM(SoftbodySim):
    def __init__(self, geo: fem.Geometry, active_cells: wp.array, args):
        super().__init__(geo, active_cells, args)

        self._ls = LineSearchUnconstrainedArmijoCriterion(self)

        self._make_elasticity_forms()

    def _make_elasticity_forms(self):
        if self.args.neo_hookean:
            self.elastic_energy = ClassicFEM.nh_elastic_energy
            self.elastic_forces = ClassicFEM.nh_elastic_forces
            self.elasticity_hessian = ClassicFEM.nh_elasticity_hessian
            self.stress_field = ClassicFEM.nh_stress_field
        else:
            self.elastic_energy = ClassicFEM.cr_elastic_energy
            self.elastic_forces = ClassicFEM.cr_elastic_forces
            self.elasticity_hessian = ClassicFEM.cr_elasticity_hessian
            self.elastic_energy_dg_sip = ClassicFEM.cr_dg_elastic_energy
            self.elastic_forces_dg_sip = ClassicFEM.cr_dg_elastic_forces
            self.elasticity_hessian_dg_sip = ClassicFEM.cr_dg_elasticity_hessian
            self.stress_field = ClassicFEM.cr_stress_field

        # cached polar decomposition
        self._svd_U = None
        self._svd_sig = None
        self._svd_V = None
        self._svd_sides_U = None
        self._svd_sides_sig = None
        self._svd_sides_V = None

    def _elasticity_form_arguments(self):
        if self.args.neo_hookean:
            return {}

        # cached polar decomposition
        return {"Us": self._svd_U, "Vs": self._svd_V, "sigs": self._svd_sig}

    def _sides_elasticity_form_arguments(self):
        if self.args.neo_hookean:
            return {}

        # cached polar decomposition
        return {
            "Us": self._svd_sides_U,
            "Vs": self._svd_sides_V,
            "sigs": self._svd_sides_sig,
        }

    def supports_discontinuities(self):
        return not self.args.neo_hookean

    def supports_matrix_free(self):
        return not self.has_discontinuities()

    def init_strain_spaces(self):
        self.elasticity_quadrature = self.vel_quadrature
        self.constraint_field = self.interpolated_constraint_field
        self._constraint_field_restriction = fem.make_restriction(
            self.constraint_field, space_restriction=self.u_test.space_restriction
        )

    def set_strain_basis(self, strain_basis: fem.BasisSpace):
        pass

    def prepare_newton_step(self, tape: wp.Tape = None):
        super().prepare_newton_step(tape=tape)

        if not self.args.neo_hookean:
            if tape is not None:
                with tape:
                    self._cache_polar_decomposition()
            else:
                self._cache_polar_decomposition()

    def evaluate_energy(self, E_u=None):
        E_u, c_r = super().evaluate_energy(E_u=E_u)

        fem.integrate(
            self.elastic_energy,
            quadrature=self.elasticity_quadrature,
            fields={"u_cur": self.u_field, "lame": self.lame_field},
            output=E_u,
            add=True,
        )

        if self.side_quadrature is not None:
            fem.integrate(
                self.elastic_energy_dg_sip,
                quadrature=self.side_quadrature,
                fields={
                    "u_cur": self.u_field.trace(),
                    "lame": self.lame_field.trace(),
                },
                add=True,
                output=E_u,
            )

        return E_u, c_r

    def newton_lhs(self):
        if self.args.matrix_free:
            return None

        u_matrix = fem.integrate(
            self.elasticity_hessian,
            quadrature=self.elasticity_quadrature,
            fields={
                "u_cur": self.u_field,
                "u": self.u_trial,
                "v": self.u_test,
                "lame": self.lame_field,
            },
            values=self._elasticity_form_arguments(),
            output_dtype=float,
        )

        if self.side_quadrature is not None:
            fem.integrate(
                self.elasticity_hessian_dg_sip,
                quadrature=self.side_quadrature,
                fields={
                    "u_cur": self.u_field.trace(),
                    "u": self.u_side_trial,
                    "v": self.u_side_test,
                    "lame": self.lame_field.trace(),
                },
                values=self._sides_elasticity_form_arguments(),
                add=True,
                output=u_matrix,
            )

        u_matrix += self.constraint_free_lhs()
        project_system_matrix(u_matrix, self.v_bd_matrix)

        return u_matrix

    def newton_rhs(self, tape: wp.Tape = None):
        u_rhs = self.constraint_free_rhs(tape=tape)

        rhs_tape = wp.Tape() if tape is None else tape
        with rhs_tape:
            fem.integrate(
                self.elastic_forces,
                quadrature=self.elasticity_quadrature,
                fields={
                    "u_cur": self.u_field,
                    "v": self.u_test,
                    "lame": self.lame_field,
                },
                values=self._elasticity_form_arguments(),
                output=u_rhs,
                add=True,
                kernel_options={"enable_backward": True},
            )

            if self.side_quadrature is not None:
                fem.integrate(
                    self.elastic_forces_dg_sip,
                    quadrature=self.side_quadrature,
                    fields={
                        "u_cur": self.u_field.trace(),
                        "v": self.u_side_test,
                        "lame": self.lame_field.trace(),
                    },
                    values=self._sides_elasticity_form_arguments(),
                    add=True,
                    output=u_rhs,
                )

        self._minus_dE_du = wp.clone(u_rhs, requires_grad=False)

        self._filter_forces(u_rhs, tape=tape)

        return u_rhs

    def _cache_polar_decomposition(self, requires_grad: bool = False):
        # precompute polar decomposition
        qp_count = self.elasticity_quadrature.total_point_count()
        self._svd_U = wp.empty(
            dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad
        )
        self._svd_sig = wp.empty(
            dtype=wp.vec3, shape=qp_count, requires_grad=requires_grad
        )
        self._svd_V = wp.empty(
            dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad
        )

        fem.interpolate(
            ClassicFEM._polar_decomposition,
            quadrature=self.elasticity_quadrature,
            fields={"u": self.u_field},
            values={"Us": self._svd_U, "Vs": self._svd_V, "sigs": self._svd_sig},
        )

        if self.side_quadrature is not None:
            # polar decompostion on side quadrature points
            qp_count = self.side_quadrature.total_point_count()
            self._svd_sides_U = wp.empty(
                dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad
            )
            self._svd_sides_sig = wp.empty(
                dtype=wp.vec3, shape=qp_count, requires_grad=requires_grad
            )
            self._svd_sides_V = wp.empty(
                dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad
            )

            fem.interpolate(
                ClassicFEM._polar_decomposition,
                quadrature=self.side_quadrature,
                fields={"u": self.u_field.trace()},
                values={
                    "Us": self._svd_sides_U,
                    "Vs": self._svd_sides_V,
                    "sigs": self._svd_sides_sig,
                },
            )

    @staticmethod
    def _solve_fp64(lhs, rhs, res, maxiters, tol=None):
        lhs64 = sp.bsr_copy(lhs, scalar_type=wp.float64)
        rhs64 = wp.empty(shape=rhs.shape, dtype=wp.vec3d, device=rhs.device)
        wp.utils.array_cast(in_array=rhs, out_array=rhs64)

        res64 = wp.zeros_like(rhs64)
        bsr_cg(
            A=lhs64,
            b=rhs64,
            x=res64,
            quiet=True,
            tol=tol,
            max_iters=maxiters,
        )

        wp.utils.array_cast(in_array=res64, out_array=res)

        return res

    def solve_newton_system(self, lhs, rhs):
        if self.args.fp64:
            res = wp.empty_like(rhs)
            ClassicFEM._solve_fp64(
                lhs, rhs, res, maxiters=self.args.cg_iters, tol=self.args.cg_tol
            )
            return (res,)

        if lhs is None:
            lhs = self._make_matrix_free_newton_linear_operator()
            use_diag_precond = False
        else:
            use_diag_precond = True

        res = wp.zeros_like(rhs)
        bsr_cg(
            A=lhs,
            b=rhs,
            x=res,
            quiet=True,
            tol=self.args.cg_tol,
            max_iters=self.args.cg_iters,
            use_diag_precond=use_diag_precond,
        )
        return (res,)

    def record_adjoint(self, tape):
        # The forward Newton is finding a root of rhs(q, p) = 0 with q = (u, S, R, lambda)
        # so drhs/dp = drhs/dq dq/dp + drhs/dp = 0
        # [- drhs/dq] dq/dp = drhs/dp
        # lhs dq/dp = drhs/dp

        self.prepare_newton_step(tape=tape)
        rhs = self.newton_rhs(tape=tape)
        lhs = self.newton_lhs()

        def solve_backward():
            adj_res = self.u_field.dof_values.grad
            ClassicFEM._solve_fp64(lhs, adj_res, rhs.grad, maxiters=self.args.cg_iters)

        tape.record_func(
            solve_backward,
            arrays=[
                self.u_field.dof_values,
                rhs,
            ],
        )

        # So we can compute stress-based losses
        with tape:
            self.interpolate_constraint_field()

    def interpolate_constraint_field(self, strain=False):
        field = self.strain_field if strain else self.stress_field

        fem.interpolate(
            field,
            fields={
                "u_cur": self.u_field,
                "lame": self.lame_field,
            },
            kernel_options={"enable_backward": True},
            dest=self._constraint_field_restriction,
        )

    @fem.integrand
    def nh_elastic_energy(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        return snh_energy(F, lame(s))

    @fem.integrand
    def nh_elastic_forces(s: Sample, u_cur: Field, v: Field, lame: Field):
        F = defgrad(u_cur, s)
        tau = fem.grad(v, s)

        return -wp.ddot(tau, snh_stress(F, lame(s)))

    @fem.integrand
    def nh_stress_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        return snh_stress(F, lame(s))

    @fem.integrand
    def nh_elasticity_hessian(s: Sample, u_cur: Field, u: Field, v: Field, lame: Field):
        F_s = defgrad(u_cur, s)
        tau_s = fem.grad(v, s)
        sig_s = fem.grad(u, s)
        lame_s = lame(s)

        return snh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s)

    @fem.integrand
    def cr_elastic_energy(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        S = symmetric_strain(F)
        return hooke_energy(S, lame(s))

    @fem.integrand
    def cr_elastic_forces(
        s: Sample,
        u_cur: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S = symmetric_strain(sig, V)
        tau = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
        return -wp.ddot(tau, hooke_stress(S, lame(s)))

    @fem.integrand
    def cr_stress_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        S = symmetric_strain(F)
        return hooke_stress(S, lame(s))

    @fem.integrand
    def cr_elasticity_hessian(
        s: Sample,
        u_cur: Field,
        u: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = symmetric_strain(sig, V)
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
        sig_s = symmetric_strain_delta(U, sig, V, fem.grad(u, s))
        lame_s = lame(s)

        return hooke_hessian(S_s, tau_s, sig_s, lame_s)

    @fem.integrand
    def strain_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        return symmetric_strain(F)

    @fem.integrand
    def _polar_decomposition(
        s: fem.Sample,
        u: fem.Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        F = defgrad_avg(u, s)

        U = wp.mat33()
        D = wp.vec3()
        V = wp.mat33()
        wp.svd3(F, U, D, V)

        Us[s.qp_index] = U
        sigs[s.qp_index] = D
        Vs[s.qp_index] = V

    def _make_matrix_free_newton_linear_operator(self):
        x_field = self.u_field.space.make_field(
            space_partition=self.u_field.space_partition
        )

        temporary_store = fem.TemporaryStore()

        if self.args.neo_hookean:

            def matvec(
                x: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float
            ):
                """Compute z = alpha * A @ x + beta * y"""
                wp.copy(src=x, dest=x_field.dof_values)
                fem.integrate(
                    self._nh_matrix_free_lhs_form,
                    quadrature=self.elasticity_quadrature,
                    fields={
                        "u_cur": self.u_field,
                        "u": x_field,
                        "v": self.u_test,
                        "lame": self.lame_field,
                        "rho": self.density_field,
                    },
                    values={"dt": self.dt},
                    output=z,
                    temporary_store=temporary_store,
                )

                self._filter_forces(z, tape=None, temporary_store=temporary_store)
                fem.linalg.array_axpy(x=y, y=z, alpha=beta, beta=alpha)

        else:

            def matvec(
                x: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float
            ):
                """Compute z = alpha * A @ x + beta * y"""
                wp.copy(src=x, dest=x_field.dof_values)

                fem.integrate(
                    self._cr_matrix_free_lhs_form,
                    quadrature=self.elasticity_quadrature,
                    fields={
                        "u": x_field,
                        "v": self.u_test,
                        "lame": self.lame_field,
                        "rho": self.density_field,
                    },
                    values={
                        "dt": self.dt,
                        **self._elasticity_form_arguments(),
                    },
                    output=z,
                    temporary_store=temporary_store,
                )

                self._filter_forces(z, tape=None, temporary_store=temporary_store)
                fem.linalg.array_axpy(x=y, y=z, alpha=beta, beta=alpha)

        # dry run to make sure temporary_store is populated
        matvec(
            x=wp.zeros_like(x_field.dof_values),
            y=wp.zeros_like(x_field.dof_values),
            z=wp.zeros_like(x_field.dof_values),
            alpha=0.0,
            beta=0.0,
        )

        n = x_field.dof_values.shape[0] * 3
        linop = LinearOperator(
            shape=(n, n),
            dtype=float,
            device=x_field.dof_values.device,
            matvec=matvec,
        )
        linop._field = x_field  # prevent garbage collection
        return linop

    @fem.integrand
    def _cr_matrix_free_lhs_form(
        s: Sample,
        domain: Domain,
        u: Field,
        v: Field,
        lame: Field,
        rho: float,
        dt: float,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = symmetric_strain(sig, V)
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad(u, s))
        sig_s = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
        lame_s = lame(s)

        return hooke_hessian(S_s, tau_s, sig_s, lame_s) + inertia_form(
            s, domain, u, v, rho, dt
        )

    @fem.integrand
    def _nh_matrix_free_lhs_form(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        u: Field,
        v: Field,
        lame: Field,
        rho: float,
        dt: float,
    ):
        F_s = defgrad(u_cur, s)
        tau_s = fem.grad(u, s)
        sig_s = fem.grad(v, s)
        lame_s = lame(s)

        return snh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s) + inertia_form(
            s, domain, u, v, rho, dt
        )

    @fem.integrand
    def cr_dg_elastic_energy(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        lame: Field,
    ):
        F_s = defgrad_avg(u_cur, s)

        U = wp.mat33()
        sig = wp.vec3()
        V = wp.mat33()
        wp.svd3(F_s, U, sig, V)

        S_s = symmetric_strain(sig, V)
        lame_s = lame(s)

        normal = fem.normal(domain, s)
        jump_u = fem.outer(fem.jump(u_cur, s), normal)
        R_jump_u = symmetric_strain_delta(U, sig, V, jump_u)

        return 0.5 * fem.measure_ratio(domain, s) * wp.ddot(jump_u, jump_u) * lame_s[
            0
        ] - wp.ddot(R_jump_u, hooke_stress(S_s, lame_s))

    @fem.integrand
    def cr_dg_elastic_forces(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = symmetric_strain(sig, V)
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad_average(v, s))
        lame_s = lame(s)

        normal = fem.normal(domain, s)
        jump_u = fem.outer(fem.jump(u_cur, s), normal)
        jump_v = fem.outer(fem.jump(v, s), normal)
        R_jump_u = symmetric_strain_delta(U, sig, V, jump_u)
        R_jump_v = symmetric_strain_delta(U, sig, V, jump_v)

        return -(
            fem.measure_ratio(domain, s) * wp.ddot(jump_u, jump_v) * lame_s[0]
            - hooke_hessian(S_s, R_jump_u, tau_s, lame_s)
            - wp.ddot(R_jump_v, hooke_stress(S_s, lame_s))
        )

    @fem.integrand
    def cr_dg_elasticity_hessian(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        u: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = wp.mat33()
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad_average(v, s))
        sig_s = symmetric_strain_delta(U, sig, V, fem.grad_average(u, s))
        lame_s = lame(s)

        normal = fem.normal(domain, s)
        jump_u = fem.outer(fem.jump(u, s), normal)
        jump_v = fem.outer(fem.jump(v, s), normal)
        R_jump_u = symmetric_strain_delta(U, sig, V, jump_u)
        R_jump_v = symmetric_strain_delta(U, sig, V, jump_v)

        return (
            fem.measure_ratio(domain, s) * wp.ddot(jump_u, jump_v) * lame_s[0]
            - hooke_hessian(S_s, R_jump_u, tau_s, lame_s)
            - hooke_hessian(S_s, R_jump_v, sig_s, lame_s)
        )


def run_softbody_sim(
    sim: SoftbodySim,
    init_callback=None,
    frame_callback=None,
    ui=True,
    log=None,
    shutdown=False,
):
    if log is not None:
        with open(log, "w") as log_f:
            sim.log = log_f
            run_softbody_sim(sim, init_callback, frame_callback, ui=ui, log=None)
            sim.log = None
        return

    if not ui:
        sim.init_constant_forms()
        sim.project_constant_forms()

        sim.cur_frame = 0
        if init_callback:
            init_callback()

        active_indices = sim.u_field.space_partition.space_node_indices().numpy()

        for frame in range(sim.args.n_frames):
            sim.cur_frame = frame + 1
            with wp.ScopedTimer(f"--- Frame --- {sim.cur_frame}", synchronize=True):
                sim.run_frame()

            displaced_pos = sim.u_field.space.node_positions().numpy()
            displaced_pos[active_indices] += sim.u_field.dof_values.numpy()

            if frame_callback:
                frame_callback(displaced_pos)
        return

    import polyscope as ps

    active_cells = None if sim.cells is None else sim.cells.array.numpy()

    try:
        hexes = sim.u_field.space.node_hexes()

        if active_cells is not None:
            hex_per_cell = len(hexes) // sim.geo.cell_count()
            selected_hexes = np.broadcast_to(
                (active_cells * hex_per_cell).reshape(len(active_cells), 1),
                shape=(len(active_cells), hex_per_cell),
            )
            selected_hexes = selected_hexes + np.broadcast_to(
                np.arange(hex_per_cell).reshape(1, hex_per_cell),
                shape=(len(active_cells), hex_per_cell),
            )

            hexes = hexes[selected_hexes.flatten()]

    except AttributeError:
        hexes = None

    if hexes is None:
        try:
            tets = sim.u_field.space.node_tets()

            if active_cells is not None:
                tet_per_cell = len(tets) // sim.geo.cell_count()
                selected_tets = np.broadcast_to(
                    (active_cells * tet_per_cell).reshape(len(active_cells), 1),
                    shape=(len(active_cells), tet_per_cell),
                )
                selected_tets = selected_tets + np.broadcast_to(
                    np.arange(tet_per_cell).reshape(1, tet_per_cell),
                    shape=(len(active_cells), tet_per_cell),
                )

                tets = tets[selected_tets.flatten()]

        except AttributeError:
            tets = None
    else:
        tets = None

    ps.init()
    ps.set_ground_plane_mode(mode_str="none")

    node_pos = sim.u_field.space.node_positions().numpy()

    ps_vol = ps.register_volume_mesh(
        "volume mesh", node_pos, hexes=hexes, tets=tets, edge_width=1.0
    )

    sim.init_constant_forms()
    sim.project_constant_forms()
    sim.cur_frame = 0

    if init_callback:
        init_callback()

    active_indices = sim.u_field.space_partition.space_node_indices().numpy()

    def callback():
        sim.cur_frame = sim.cur_frame + 1
        if sim.args.n_frames >= 0 and sim.cur_frame > sim.args.n_frames:
            if shutdown:
                ps.unshow()
            return

        with wp.ScopedTimer(f"--- Frame --- {sim.cur_frame}", synchronize=True):
            sim.run_frame()

        displaced_pos = sim.u_field.space.node_positions().numpy()
        displaced_pos[active_indices] += sim.u_field.dof_values.numpy()
        ps_vol.update_vertex_positions(displaced_pos)

        if frame_callback:
            frame_callback(displaced_pos)

        # ps.screenshot()

    ps.set_user_callback(callback)
    # ps.look_at(target=(0.5, 0.5, 0.5), camera_location=(0.5, 0.5, 2.5))
    ps.show()
