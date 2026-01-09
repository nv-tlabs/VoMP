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

from typing import Optional
import argparse
import numpy as np

import warp as wp
import warp.sparse as sp
import warp.fem as fem

from warp.fem import Domain, Sample, Field
from warp.fem.utils import array_axpy
from warp.optim.linear import LinearOperator

from warp.examples.fem.utils import bsr_cg
import potpourri3d as pp3d

from fem_examples.mfem.elastic_models import (
    symmetric_strain,
    symmetric_strain_delta,
    snh_energy,
    snh_stress,
    snh_hessian_proj_analytic,
    nh_energy,
    nh_stress,
    nh_hessian_proj_analytic,
    hooke_energy,
    hooke_stress,
    hooke_hessian,
)
from fem_examples.mfem.linalg import diff_bsr_mv

wp.set_module_options({"enable_backward": False})
wp.set_module_options({"max_unroll": 4})
wp.set_module_options({"fast_math": True})

import torch
import warp.sparse as wps


def _wp_bsr_to_torch_bsr(mat: wps.BsrMatrix):  # pragma: no cover
    r"""Converts a sparse warp BSR matrix (or CSR matrix) to a sparse torch matrix.

    Args:
        mat (wps.BsrMatrix): A sparse warp BSR matrix of shape :math:`(m, n)` with block shape :math:`(b, b)`.

    Returns:
        torch.sparse_bsr_tensor: A sparse torch matrix of shape :math:`(m, n)` with block shape :math:`(b, b)`.
    """
    mat.nnz_sync()

    if mat.block_shape == (1, 1):
        ctor = torch.sparse_csr_tensor
        values = wp.to_torch(mat.values[: mat.nnz]).flatten()
    else:
        ctor = torch.sparse_bsr_tensor
        values = wp.to_torch(mat.values[: mat.nnz])
    torch_weights = ctor(
        crow_indices=wp.to_torch(mat.offsets[: mat.nrow + 1]),
        col_indices=wp.to_torch(mat.columns[: mat.nnz]),
        # values=wp.to_torch(mat.values[: mat.nnz]),
        values=values,
        size=mat.shape,
    )
    return torch_weights


@fem.integrand
def defgrad(u: fem.Field, s: fem.Sample):
    return fem.grad(u, s) + wp.identity(n=3, dtype=float)


@fem.integrand
def inertia_form(s: Sample, domain: Domain, u: Field, v: Field, rho: float, dt: float):
    """<rho/dt^2 u, v>"""

    u_rhs = rho * u(s) / (dt * dt)
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
    rho: float,
    gravity: wp.vec3,
    dt: float,
):
    """<rho/dt^2 u, v> + <rho g, v>"""
    f = (
        inertia_form(s, domain, u_prev, v, rho, dt)
        - inertia_form(s, domain, u, v, rho, dt)
        + rho * wp.dot(gravity, v(s))
    )

    return f


@wp.struct
class VolumetricForces:
    count: int
    centers: wp.array(dtype=wp.vec3)
    radii: wp.array(dtype=float)
    forces: wp.array(dtype=wp.vec3)
    tot_weight: wp.array(dtype=float)


@wp.func
def force_weight(x: wp.vec3, forces: VolumetricForces, force_index: int):
    r = wp.min(
        wp.length(x - forces.centers[force_index])
        / (forces.radii[force_index] + 1.0e-7),
        1.0,
    )
    r2 = r * r
    return 2.0 * r * r2 - 3.0 * r2 + 1.0  # cubic spline


@fem.integrand
def force_weight_form(
    s: Sample, domain: Domain, forces: VolumetricForces, force_index: int
):
    return force_weight(domain(s), forces, force_index)


@fem.integrand
def force_action(x: wp.vec3, forces: VolumetricForces, force_index: int, vec: wp.vec3):
    # action of a force over a vector
    return wp.where(
        forces.tot_weight[force_index] >= 1.0e-6,
        wp.dot(forces.forces[force_index], vec)
        * force_weight(x, forces, force_index)
        / forces.tot_weight[force_index],
        0.0,
    )


@fem.integrand
def external_forces_form(s: Sample, domain: Domain, v: Field, forces: VolumetricForces):
    f = float(0.0)
    x = domain(s)
    for fi in range(forces.count):
        f += force_action(x, forces, fi, v(s))
    return f


@fem.integrand
def kinetic_potential_energy(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
    rho: float,
    dt: float,
    gravity: wp.vec3,
    forces: VolumetricForces,
):
    du = u(s)
    dv = v(s)

    E = rho * (0.5 * wp.dot(du - dv, du - dv) / (dt * dt) - wp.dot(du, gravity))
    E -= external_forces_form(s, domain, u, forces)

    return E


@fem.integrand
def geo_def_grad_field(s: Sample, domain: Domain):
    return fem.deformation_gradient(domain, s)


@wp.kernel(enable_backward=True)
def scale_lame(
    lame_out: wp.array(dtype=wp.vec2),
    lame_ref: wp.vec2,
    scale: wp.array(dtype=float),
):
    i = wp.tid()
    lame_out[i] = lame_ref * scale[i]


running = False


class LineSearchNaiveCriterion:
    def __init__(self, sim):
        self.penalty = sim.args.young_modulus

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        pass

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        f_cur = E_cur + self.penalty * C_cur
        f_ref = E_ref + self.penalty * C_ref
        return f_cur <= f_ref


class LineSearchUnconstrainedArmijoCriterion:
    def __init__(self, sim):
        self.armijo_coeff = 0.0001

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        (delta_u,) = delta_fields

        m = -wp.utils.array_inner(delta_u, sim._minus_dE_du.view(delta_u.dtype))
        self.m = m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        return E_cur <= E_ref + self.armijo_coeff * alpha * self.m


class SoftbodySim:
    def __init__(self, geo: fem.Geometry, active_cells: Optional[wp.array], args):
        self.args = args

        self.geo = geo

        if active_cells is None:
            self._geo_partition = fem.geometry.WholeGeometryPartition(geo)
        else:
            self._geo_partition = fem.ExplicitGeometryPartition(
                geo, cell_mask=active_cells
            )
            if self._geo_partition.cell_count() == geo.cell_count():
                self._geo_partition = fem.geometry.WholeGeometryPartition(geo)

        if isinstance(self._geo_partition, fem.ExplicitGeometryPartition):
            self.cells = self._geo_partition._cells
        else:
            self.cells = None

        if self.has_discontinuities() and not self.supports_discontinuities():
            raise TypeError(
                f"Simulator of type {type(self)} does not support discontinuities"
            )

        if self.args.matrix_free and not self.supports_matrix_free():
            raise TypeError(
                f"Simulator of type {type(self)} does not support matrix-free solves"
            )

        if not self.args.quiet:
            print(f"Active cells: {self._geo_partition.cell_count()}")

        self.dt = args.dt

        self.up_axis = np.zeros(3)
        self.up_axis[self.args.up_axis] = 1.0

        self.gravity = -self.args.gravity * self.up_axis

        self.forces = VolumetricForces()
        self.forces.count = 0
        self.forces.forces = wp.zeros(shape=(0,), dtype=wp.vec3)
        self.forces.radii = wp.zeros(shape=(0,), dtype=float)
        self.forces.centers = wp.zeros(shape=(0,), dtype=wp.vec3)
        self.forces.tot_weight = wp.zeros(shape=(0,), dtype=float)

        young = args.young_modulus
        poisson = args.poisson_ratio
        self.lame_ref = wp.vec2(
            young / (1.0 + poisson) * np.array([poisson / (1.0 - 2.0 * poisson), 0.5])
        )
        print(f"young: {young}")
        print(f"poisson: {poisson}")
        print(f"lame_ref: {np.array(self.lame_ref)}")

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

        self._ls = LineSearchNaiveCriterion(self)
        self._init_displacement_basis()

        self._collision_projector_form = None
        self._penalty_lhs_form = None
        self._penalty_rhs_form = None

        self.log = None

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

    def set_displacement_basis(self, basis: fem.BasisSpace = None):
        if basis is None:
            self._init_displacement_basis()
        else:
            self._vel_basis = basis

    def init_displacement_space(self, side_subdomain: fem.Domain):
        args = self.args

        u_space = fem.make_collocated_function_space(self._vel_basis, dtype=wp.vec3)
        u_space_partition = fem.make_space_partition(
            space_topology=self._vel_basis.topology,
            geometry_partition=self._geo_partition,
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
        lame_space = fem.make_polynomial_space(self.geo, dtype=wp.vec2)

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
        boundary_projector_args={},
        boundary_displacement_form=None,
        boundary_displacement_args={},
    ):
        u_space = self.u_field.space

        # Displacement boundary conditions
        boundary = fem.BoundarySides(self._geo_partition)

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
                values=boundary_displacement_args,
                nodal=True,
                output_dtype=wp.vec3f,
            )
        if boundary_projector_form is not None:
            self.v_bd_matrix = fem.integrate(
                boundary_projector_form,
                fields={"u": u_bd_trial, "v": u_bd_test},
                values=boundary_projector_args,
                nodal=True,
                output_dtype=float,
            )
        else:
            self.v_bd_matrix = sp.bsr_zeros(
                self.u_trial.space_partition.node_count(),
                self.u_trial.space_partition.node_count(),
                block_type=wp.mat33,
            )
        fem.normalize_dirichlet_projector(self.v_bd_matrix, self.v_bd_rhs)

    def set_fixed_points_condition(
        self,
        fixed_points_projector_form,
        fixed_point_projector_args={},
    ):
        self.v_bd_rhs = None
        self.v_bd_matrix = fem.integrate(
            fixed_points_projector_form,
            fields={"u": self.u_trial, "v": self.u_test, "u_cur": self.u_field},
            values=fixed_point_projector_args,
            nodal=True,
            output_dtype=float,
        )

        self.v_bd_rhs = None
        fem.normalize_dirichlet_projector(self.v_bd_matrix, self.v_bd_rhs)

    def set_fixed_points_displacement(
        self,
        fixed_points_displacement_field=None,
        fixed_points_displacement_args={},
    ):
        bd_field = self.u_test.space.make_field(
            space_partition=self.u_test.space_partition
        )
        fem.interpolate(
            fixed_points_displacement_field,
            dest=bd_field,
            fields={"u_cur": self.u_field},
            values=fixed_points_displacement_args,
        )

        self.v_bd_rhs = bd_field.dof_values

    def init_constant_forms(self):
        args = self.args

        self.update_force_weight()

        if args.matrix_free:
            self.A = None
            return

        if self.args.lumped_mass:
            self.A = fem.integrate(
                inertia_form,
                fields={"u": self.u_trial, "v": self.u_test},
                values={"rho": args.density, "dt": self.dt},
                output_dtype=float,
                nodal=True,
            )
        else:
            self.A = fem.integrate(
                inertia_form,
                fields={"u": self.u_trial, "v": self.u_test},
                values={"rho": args.density, "dt": self.dt},
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

    def project_constant_forms(self):
        if self.A is None:
            return

        self.A_proj = sp.bsr_copy(self.A)
        fem.dirichlet.project_system_matrix(self.A_proj, self.v_bd_matrix)
        self.A_proj.nnz_sync()

    def update_force_weight(self):
        self.forces.tot_weight = wp.empty(shape=self.forces.count, dtype=wp.float32)
        for fi in range(self.forces.count):
            wi = self.forces.tot_weight[fi : fi + 1]
            fem.integrate(
                force_weight_form,
                quadrature=self.vel_quadrature,
                values={
                    "force_index": fi,
                    "forces": self.forces,
                },
                output=wi,
                accumulate_dtype=wp.float32,
            )

    def constraint_free_rhs(self, dt=None, with_external_forces=True, tape=None):
        args = self.args

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
                fields={"u": self.du_field, "u_prev": self.du_prev, "v": self.u_test},
                values={"rho": args.density, "dt": self._step_dt(), "gravity": gravity},
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
            # NOT differentiating with respect to external forces
            # Those are assumed to not depend on the geometry
            fem.integrate(
                external_forces_form,
                fields={"v": self.u_test},
                values={
                    "forces": self.forces,
                },
                output_dtype=wp.vec3,
                quadrature=self.vel_quadrature,
                kernel_options={"enable_backward": False},
                output=rhs,
                add=True,
            )

        return rhs

    def run_frame(self):
        (self.du_field, self.du_prev) = (self.du_prev, self.du_field)

        if self.args.quasi_quasistatic:
            self.du_prev.dof_values.zero_()

        self.compute_initial_guess()

        tol = self.args.newton_tol**2

        E_cur, C_cur = self.evaluate_energy()
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
                E_cur, C_cur = self.evaluate_energy()

                ddu = delta_fields[0]
                step_size = wp.utils.array_inner(ddu, ddu) / (1 + ddu.shape[0])

                # linear model
                self._ls.build_linear_model(self, lhs, rhs, delta_fields)

                # Line search
                alpha = 1.0
                for j in range(self.args.n_backtrack):
                    if self._ls.accept(alpha, E_cur, C_cur, E_ref, C_ref):
                        break

                    alpha = 0.5 * alpha
                    self.apply_newton_deltas(delta_fields, alpha=alpha)
                    E_cur, C_cur = self.evaluate_energy()

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
        pass

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
        E_u = fem.integrate(
            kinetic_potential_energy,
            quadrature=self.vel_quadrature,
            fields={"u": self.du_field, "v": self.du_prev},
            values={
                "rho": self.args.density,
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

    def _filter_forces(self, u_rhs, tape):
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
                nodal=True,
                output_dtype=float,
            )
            fem.normalize_dirichlet_projector(self.v_bd_matrix)
            self.project_constant_forms()

        proj_tape = wp.Tape() if tape is None else tape
        with proj_tape:
            diff_bsr_mv(
                A=self.v_bd_matrix,
                x=u_rhs,
                y=u_rhs,
                alpha=-1.0,
                beta=1.0,
                self_adjoint=True,
            )

    def compute_initial_guess(self):
        # Self-advect
        self.du_field.dof_values.zero_()
        rhs = self.constraint_free_rhs(with_external_forces=False)

        fem.dirichlet.project_system_rhs(self.A, rhs, self.v_bd_matrix, self.v_bd_rhs)

        bsr_cg(self.A_proj, b=rhs, x=self.du_field.dof_values, quiet=True)

        array_axpy(x=self.du_field.dof_values, y=self.u_field.dof_values)

    def scale_lame_field(self, stiffness_scale_array: wp.array):
        # make a copy of the original field for differentiability
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
        parser.add_argument("--cg_tol", type=float, default=1.0e-8)
        parser.add_argument("--cg_iters", type=float, default=1000)
        parser.add_argument("--n_backtrack", type=int, default=4)
        parser.add_argument("--young_modulus", type=float, default=250.0)
        parser.add_argument("--poisson_ratio", type=float, default=0.4)
        parser.add_argument("--gravity", type=float, default=10.0)
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
            self.stress_field = ClassicFEM.cr_stress_field

    def supports_matrix_free(self):
        return True

    def evaluate_energy(self):
        E_u, c_r = super().evaluate_energy()

        E_e = fem.integrate(
            self.elastic_energy,
            quadrature=self.elasticity_quadrature,
            fields={"u_cur": self.u_field, "lame": self.lame_field},
        )

        E_tot = E_u + E_e

        return E_tot, c_r

    def init_strain_spaces(self):
        self.elasticity_quadrature = self.vel_quadrature
        self.constraint_field = self.interpolated_constraint_field
        self._constraint_field_restriction = fem.make_restriction(
            self.constraint_field, space_restriction=self.u_test.space_restriction
        )

    def set_strain_basis(self, strain_basis: fem.BasisSpace):
        pass

    def compute_initial_guess(self):
        # Start from last frame pose, known good state
        self.du_field.dof_values.zero_()

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
            output_dtype=float,
        )

        u_matrix += self.A
        fem.dirichlet.project_system_matrix(u_matrix, self.v_bd_matrix)

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
                output=u_rhs,
                add=True,
                kernel_options={"enable_backward": True},
            )

        self._minus_dE_du = wp.clone(u_rhs, requires_grad=False)

        self._filter_forces(u_rhs, tape=tape)

        return u_rhs

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
        # return snh_energy(F, lame(s))
        return nh_energy(F, lame(s))

    @fem.integrand
    def nh_elastic_forces(s: Sample, u_cur: Field, v: Field, lame: Field):
        F = defgrad(u_cur, s)
        tau = fem.grad(v, s)

        # return -wp.ddot(tau, snh_stress(F, lame(s)))
        return -wp.ddot(tau, nh_stress(F, lame(s)))

    @fem.integrand
    def nh_stress_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        # return snh_stress(F, lame(s))
        return nh_stress(F, lame(s))

    @fem.integrand
    def nh_elasticity_hessian(s: Sample, u_cur: Field, u: Field, v: Field, lame: Field):
        F_s = defgrad(u_cur, s)
        tau_s = fem.grad(v, s)
        sig_s = fem.grad(u, s)
        lame_s = lame(s)

        # return snh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s)
        return nh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s)

    @fem.integrand
    def cr_elastic_energy(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        S = symmetric_strain(F)
        return hooke_energy(S, lame(s))

    @fem.integrand
    def cr_elastic_forces(s: Sample, u_cur: Field, v: Field, lame: Field):
        F = defgrad(u_cur, s)
        S = symmetric_strain(F)
        tau = symmetric_strain_delta(F, fem.grad(v, s))
        return -wp.ddot(tau, hooke_stress(S, lame(s)))

    @fem.integrand
    def cr_stress_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        S = symmetric_strain(F)
        return hooke_stress(S, lame(s))

    @fem.integrand
    def cr_elasticity_hessian(s: Sample, u_cur: Field, u: Field, v: Field, lame: Field):
        F_s = defgrad(u_cur, s)

        U = wp.mat33()
        sig = wp.vec3()
        V = wp.mat33()
        wp.svd3(F_s, U, sig, V)

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
        F = defgrad(u, s)

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
                    },
                    values={"rho": self.args.density, "dt": self.dt},
                    output=z,
                )

                self._filter_forces(z, tape=None)
                fem.linalg.array_axpy(x=y, y=z, alpha=beta, beta=alpha)

        else:
            # precompute polar decomposition, constant over CG iterations
            U = wp.empty(
                dtype=wp.mat33, shape=self.elasticity_quadrature.total_point_count()
            )
            sig = wp.empty(
                dtype=wp.vec3, shape=self.elasticity_quadrature.total_point_count()
            )
            V = wp.empty(
                dtype=wp.mat33, shape=self.elasticity_quadrature.total_point_count()
            )

            fem.interpolate(
                ClassicFEM._polar_decomposition,
                quadrature=self.elasticity_quadrature,
                fields={"u": self.u_field},
                values={"Us": U, "Vs": V, "sigs": sig},
            )

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
                    },
                    values={
                        "rho": self.args.density,
                        "dt": self.dt,
                        "Us": U,
                        "Vs": V,
                        "sigs": sig,
                    },
                    output=z,
                )

                self._filter_forces(z, tape=None)
                fem.linalg.array_axpy(x=y, y=z, alpha=beta, beta=alpha)

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

        # return snh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s) + inertia_form(
        return nh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s) + inertia_form(
            s, domain, u, v, rho, dt
        )


def run_softbody_sim(
    sim: SoftbodySim,
    init_callback=None,
    frame_callback=None,
    ui=True,
    log=None,
    shutdown=False,
    output_path=None,
    output_save_volume=True,
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
    import polyscope.imgui as psim

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
    if sim.args.ground:
        ps.set_ground_plane_height(sim.args.ground_height)
    else:
        ps.set_ground_plane_mode(mode_str="none")

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

    if init_callback:
        init_callback()

    active_indices = sim.u_field.space_partition.space_node_indices().numpy()

    faces = sim.geo._face_vertex_indices.numpy()
    bd_faces = faces[sim.geo._boundary_face_indices.numpy()]
    surface_pos = sim.geo.positions.numpy()

    if output_path is not None:
        import os

        os.makedirs(output_path, exist_ok=True)
        if output_save_volume:
            pp3d.write_mesh(
                node_pos, tets, output_path + "/frame_" + str(sim.cur_frame) + ".obj"
            )
        else:
            pp3d.write_mesh(
                surface_pos,
                bd_faces,
                output_path + "/frame_" + str(sim.cur_frame) + ".obj",
            )

    def callback():
        global running
        changed, running = psim.Checkbox("Running", running)
        if not running:
            return

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

        if output_path is not None:
            if output_save_volume:
                # volume mesh
                pp3d.write_mesh(
                    displaced_pos,
                    tets,
                    output_path + "/frame_" + str(sim.cur_frame) + ".obj",
                )
            else:
                # surface mesh
                surface_pos = sim.geo.positions.numpy()
                surface_pos += sim.u_field.dof_values.numpy()[: surface_pos.shape[0]]
                pp3d.write_mesh(
                    surface_pos,
                    bd_faces,
                    output_path + "/frame_" + str(sim.cur_frame) + ".obj",
                )

        # ps.screenshot()

    ps.set_user_callback(callback)
    # ps.look_at(target=(0.5, 0.5, 0.5), camera_location=(0.5, 0.5, 2.5))
    ps.show()
