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
import math
import gc
from typing import Any, Optional

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem import Domain, Field, Sample
from warp.fem.utils import array_axpy

from fem_examples.mfem.linalg import MFEMSystem
from fem_examples.mfem.softbody_sim import (
    SoftbodySim,
    defgrad,
)

from fem_examples.mfem.elastic_models import hooke_energy, hooke_stress, hooke_hessian
from fem_examples.mfem.elastic_models import (
    symmetric_strain,
    symmetric_strain_delta,
    snh_energy,
    snh_stress,
    snh_hessian_proj,
)

wp.set_module_options({"enable_backward": False})
wp.set_module_options({"max_unroll": 4})
wp.set_module_options({"fast_math": True})

Scalar = wp.float32
vec3s = wp.vec(length=3, dtype=Scalar)
vec6s = wp.vec(length=6, dtype=Scalar)
vec9s = wp.vec(length=9, dtype=Scalar)

_SQRT_2 = wp.constant(math.sqrt(2.0))
_SQRT_1_2 = wp.constant(math.sqrt(1.0 / 2.0))


class FullTensorMapper(fem.DofMapper):
    """Orthonormal isomorphism from R^{n (n+1)} to nxn symmetric tensors,
    using usual L2 norm for vectors and half Frobenius norm, (tau : tau)/2 for tensors.
    """

    def __init__(self):
        self.value_dtype = wp.mat33
        self.DOF_SIZE = wp.constant(9)
        self.dof_dtype = vec9s

    def __str__(self):
        return f"_{self.DOF_SIZE}"

    @wp.func
    def dof_to_value(dof: vec9s):
        a = _SQRT_2 * dof[0]
        b = _SQRT_2 * dof[1]
        c = _SQRT_2 * dof[2]
        d = dof[3]
        e = dof[4]
        f = dof[5]

        ka = dof[6]
        kb = dof[7]
        kc = dof[8]
        return wp.mat33(
            a,
            f - kc,
            e + kb,
            f + kc,
            b,
            d - ka,
            e - kb,
            d + ka,
            c,
        )

    @wp.func
    def value_to_dof(val: wp.mat33):
        a = _SQRT_1_2 * val[0, 0]
        b = _SQRT_1_2 * val[1, 1]
        c = _SQRT_1_2 * val[2, 2]

        d = 0.5 * (val[2, 1] + val[1, 2])
        e = 0.5 * (val[0, 2] + val[2, 0])
        f = 0.5 * (val[1, 0] + val[0, 1])

        ka = 0.5 * (val[2, 1] - val[1, 2])
        kb = 0.5 * (val[0, 2] - val[2, 0])
        kc = 0.5 * (val[1, 0] - val[0, 1])

        return vec9s(a, b, c, d, e, f, ka, kb, kc)


@wp.func
def rotation_matrix(rot_vec: wp.vec3):
    quat = wp.quat_from_axis_angle(wp.normalize(rot_vec), wp.length(rot_vec))
    return wp.quat_to_matrix(quat)


@wp.kernel
def apply_rotation_delta(
    r_vec: wp.array(dtype=wp.vec3), dR: wp.array(dtype=wp.vec3), alpha: float
):
    i = wp.tid()

    Q = wp.quat_from_axis_angle(wp.normalize(r_vec[i]), wp.length(r_vec[i]))
    omega = dR[i] * alpha

    dQ = wp.quat(omega, 0.0)
    Q = wp.normalize(Q + 0.5 * Q * dQ)

    axis = wp.vec3()
    angle = float(0)
    wp.quat_to_axis_angle(Q, axis, angle)
    r_vec[i] = axis * angle


@fem.integrand
def tensor_mass_form(s: Sample, sig: Field, tau: Field):
    """
    Mass form over tensor space
       sig : tau
    """
    return wp.ddot(sig(s), tau(s))


class LineSearchMeritCriterion:
    # Numeric Optimization, chapter 15.4

    def __init__(self, sim: SoftbodySim):
        self.armijo_coeff = 0.0001

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        delta_u, dS, dR, dLambda = delta_fields

        c_k = rhs[3]
        c_k_normalized = wp.empty_like(c_k)

        wp.launch(
            self._normalize_c_k,
            inputs=[c_k, c_k_normalized, sim._stiffness_field.dof_values],
            dim=c_k.shape,
        )

        delta_ck = lhs._B @ delta_u
        sp.bsr_mv(A=lhs._Cs, x=dS, y=delta_ck, alpha=-1.0, beta=1.0)

        if lhs._Cr is not None:
            sp.bsr_mv(A=lhs._Cr, x=dR, y=delta_ck, alpha=-1.0, beta=1.0)

        m = wp.utils.array_inner(dS, sim._dE_dS.view(dS.dtype)) - wp.utils.array_inner(
            delta_u, sim._minus_dE_du.view(delta_u.dtype)
        ) * wp.utils.array_inner(c_k_normalized, delta_ck.view(c_k_normalized.dtype))

        self.m = m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        f_cur = E_cur + C_cur
        f_ref = E_ref + C_ref

        return f_cur <= f_ref + self.armijo_coeff * alpha * self.m

    @wp.kernel
    def _normalize_c_k(
        c_k: wp.array(dtype=Any),
        c_k_norm: wp.array(dtype=Any),
        scale: wp.array(dtype=float),
    ):
        i = wp.tid()
        c_k_norm[i] = wp.normalize(c_k[i]) * scale[i]


class LineSearchMultiObjCriterion:
    # Line Search Filter Methods for Nonlinear Programming: Motivation and Global Convergence
    # 2005, SIAM Journal on Optimization 16(1):1-31

    def __init__(self, sim: SoftbodySim):
        # constraint decrease
        E_scale = sim.typical_stiffness / sim.lame_ref[1]
        self.gamma_theta = 0.75
        self.gamma_f = 0.1 * E_scale

        # switching rule
        self.s_theta = 1.5
        self.s_rho = 2.5 * self.s_theta
        self.delta = 0.01 * E_scale ** (self.s_theta / self.s_rho)

        self.armijo_coeff = 0.0001

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        delta_u, dS, dR, dLambda = delta_fields
        m = wp.utils.array_inner(dS, sim._dE_dS.view(dS.dtype)) - wp.utils.array_inner(
            delta_u, sim._minus_dE_du.view(delta_u.dtype)
        )
        self.m = m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        if (
            self.m < 0.0
            and (-self.m) ** self.s_rho * alpha > self.delta * C_ref**self.s_theta
        ):
            return E_cur <= E_ref + self.armijo_coeff * alpha * self.m

        return C_cur <= (1.0 - self.gamma_theta) * C_ref or (
            E_cur <= E_ref - self.gamma_f * C_ref
        )


class LineSearchLagrangianArmijoCriterion:
    # Unconstrained line-search based on Lagrangian

    def __init__(self, sim: SoftbodySim):
        self.armijo_coeff = 0.0001

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        delta_u, dS, dR, dLambda = delta_fields

        m = wp.utils.array_inner(dS, sim._dE_dS.view(dS.dtype)) - wp.utils.array_inner(
            delta_u, sim._minus_dE_du.view(delta_u.dtype)
        )

        c_k = rhs[3]
        delta_ck = lhs._B @ delta_u
        sp.bsr_mv(A=lhs._Cs, x=dS, y=delta_ck, alpha=-1.0, beta=1.0)
        if lhs._Cr is not None:
            sp.bsr_mv(A=lhs._Cr, x=dR, y=delta_ck, alpha=-1.0, beta=1.0)

        c_m = wp.utils.array_inner(c_k, dLambda.view(c_k.dtype)) + wp.utils.array_inner(
            delta_ck, sim.constraint_field.dof_values.view(delta_ck.dtype)
        )
        self.m = m - c_m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        return E_cur + C_cur <= E_ref + C_ref + self.armijo_coeff * alpha * self.m


class MFEM(SoftbodySim):
    def __init__(self, geo: fem.Geometry, active_cells: wp.array, args):
        super().__init__(geo, active_cells, args)

        self._make_elasticity_forms()
        self._init_strain_basis()

        self._lagrangian_constraint_energy = False
        if self.args.line_search == "merit":
            self._ls = LineSearchMeritCriterion(self)
        elif self.args.line_search == "lagrangian":
            self._lagrangian_constraint_energy = True
            self._ls = LineSearchLagrangianArmijoCriterion(self)
        else:
            self._ls = LineSearchMultiObjCriterion(self)

        # Temp storage for energy cuda graph
        self._E = wp.empty(3, dtype=wp.float64)
        self._E_pinned = wp.empty_like(self._E, device="cpu", pinned=True)
        self._E_graph = None

    def set_strain_basis(self, strain_basis: Optional[fem.BasisSpace]):
        if strain_basis is None:
            self._init_strain_basis()
        else:
            self._strain_basis = strain_basis

    def _make_elasticity_forms(self):
        if self.args.neo_hookean:
            self.elastic_energy_form = MFEM.nh_elasticity_energy_form
            self.elastic_gradient_form = MFEM.nh_elasticity_gradient_form
            self.elastic_hessian_form = MFEM.nh_elasticity_hessian_form
        else:
            self.elastic_energy_form = MFEM.hooke_elasticity_energy_form
            self.elastic_gradient_form = MFEM.hooke_elasticity_gradient_form
            self.elastic_hessian_form = MFEM.hooke_elasticity_hessian_form

    def _init_strain_basis(self):
        if isinstance(self.geo.reference_cell(), fem.geometry.element.Cube):
            strain_degree = self.args.degree
            strain_basis = fem.ElementBasis.LAGRANGE
            strain_poly = fem.Polynomial.GAUSS_LEGENDRE
        else:
            strain_degree = self.args.degree - 1
            strain_basis = fem.ElementBasis.NONCONFORMING_POLYNOMIAL
            strain_poly = None

        self._strain_basis = fem.make_polynomial_basis_space(
            self.geo,
            degree=strain_degree,
            discontinuous=True,
            element_basis=strain_basis,
            family=strain_poly,
        )

    def init_strain_spaces(self, constraint_dof_mapper: fem.DofMapper):
        sym_space = fem.make_collocated_function_space(
            self._strain_basis,
            dof_mapper=fem.SymmetricTensorMapper(
                wp.mat33, mapping=fem.SymmetricTensorMapper.Mapping.DB16
            ),
        )

        # Function spaces for piecewise-constant per-element rotations and rotation vectors
        rot_space = fem.make_collocated_function_space(
            self._strain_basis,
            dtype=wp.vec3,
        )
        skew_space = fem.make_collocated_function_space(
            self._strain_basis,
            dof_mapper=fem.SkewSymmetricTensorMapper(wp.mat33),
        )
        constraint_space = fem.make_collocated_function_space(
            self._strain_basis,
            dof_mapper=constraint_dof_mapper,
        )

        strain_space_partition = fem.make_space_partition(
            space_topology=self._strain_basis.topology,
            geometry_partition=self._geo_partition,
            with_halo=False,
        )

        # Defines some fields over our function spaces
        self.S = sym_space.make_field(
            space_partition=strain_space_partition
        )  # Rotated symmetric train
        self.S.dof_values.fill_(
            sym_space.dof_mapper.value_to_dof(
                wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            )
        )  # initialize with identity

        self.R = rot_space.make_field(
            space_partition=strain_space_partition
        )  # Rotation

        self.constraint_field = constraint_space.make_field(
            space_partition=strain_space_partition
        )

        # Since our spaces are constant, we can also predefine the test/trial functions that we will need for integration
        domain = self.u_test.domain
        self.sym_test = fem.make_test(
            space=sym_space, space_partition=strain_space_partition, domain=domain
        )
        self.sym_trial = fem.make_trial(
            space=sym_space, space_partition=strain_space_partition, domain=domain
        )

        self.skew_test = fem.make_test(
            space=skew_space, space_partition=strain_space_partition, domain=domain
        )
        self.skew_trial = fem.make_trial(
            space=skew_space, space_partition=strain_space_partition, domain=domain
        )

        self.constraint_test = fem.make_test(
            space=constraint_space, space_partition=strain_space_partition
        )

        # self.strain_quadrature = fem.RegularQuadrature(self.sym_test.domain, order=2 * sym_space.degree)
        # self.elasticity_quadrature = fem.RegularQuadrature(self.sym_test.domain, order=2 * sym_space.degree)
        self.strain_quadrature = fem.NodalQuadrature(
            self.sym_test.domain, space=sym_space
        )
        self.elasticity_quadrature = fem.NodalQuadrature(
            self.sym_test.domain, space=sym_space
        )

        self._stiffness_field = fem.make_collocated_function_space(
            self._strain_basis, dtype=float
        ).make_field()
        fem.interpolate(
            MFEM._typical_stiffness_field,
            fields={"lame": self.lame_field},
            dest=self._stiffness_field,
        )

    def init_constant_forms(self):
        super().init_constant_forms()

        # Temp storage so we can use cuda graphs (forward pass only)
        self._u_rhs = wp.empty_like(self.u_field.dof_values, requires_grad=False)
        self._f = wp.empty_like(self.S.dof_values, requires_grad=False)
        self._w = wp.empty_like(self.R.dof_values, requires_grad=False)
        self._c_k = wp.empty_like(self.constraint_field.dof_values, requires_grad=False)
        self._rhs_graph = None

        # For line search
        self._minus_dE_du = wp.empty_like(self._u_rhs)
        self._dE_dS = wp.empty_like(self._f)
        self._dE_dR = wp.empty_like(self._w)

        self._schur_work_arrays = None

    def reset_fields(self):
        super().reset_fields()

        self.S.dof_values.fill_(
            self.S.space.dof_mapper.value_to_dof(
                wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            )
        )  # initialize with identity
        self.R.dof_values.zero_()
        self.constraint_field.dof_values.zero_()

    def checkpoint_newton_values(self):
        super().checkpoint_newton_values()

        self._R_cur = wp.clone(self.R.dof_values)
        self._S_cur = wp.clone(self.S.dof_values)
        self._lbd_cur = wp.clone(self.constraint_field.dof_values)

    def apply_newton_deltas(self, delta_fields, alpha=1.0):
        super().apply_newton_deltas(delta_fields, alpha=alpha)

        wp.copy(src=self._S_cur, dest=self.S.dof_values)
        wp.copy(src=self._lbd_cur, dest=self.constraint_field.dof_values)

        _, dS, dR, dLambda = delta_fields

        array_axpy(x=dS, y=self.S.dof_values, alpha=alpha)

        array_axpy(
            x=dLambda.view(dtype=self.constraint_field.dof_values.dtype),
            y=self.constraint_field.dof_values,
            alpha=alpha,
        )

        self._apply_rotation_delta(dR, alpha)

    def _apply_rotation_delta(self, dR, alpha):
        wp.copy(src=self._R_cur, dest=self.R.dof_values)
        wp.launch(
            apply_rotation_delta, dim=dR.shape[0], inputs=[self.R.dof_values, dR, alpha]
        )

    def _evaluate_energy(self, E_u, E_e, c_r, lagrangian=False):
        super().evaluate_energy(E_u=E_u)

        fem.integrate(
            self.elastic_energy_form,
            quadrature=self.elasticity_quadrature,
            fields={"S": self.S, "lame": self.lame_field},
            output=E_e,
        )

        if lagrangian:
            fem.integrate(
                self.constraint_form,
                quadrature=self.elasticity_quadrature,
                fields={
                    "u": self.u_field,
                    "sig": self.S,
                    "R": self.R,
                    "tau": self.constraint_field,
                },
                kernel_options={"enable_backward": True},
                output=c_r,
            )
        else:
            c_k = wp.empty_like(self.constraint_field.dof_values, requires_grad=False)
            self._evaluate_constraint_residual(out=c_k)
            c_k_norm = wp.empty(shape=c_k.shape, dtype=wp.float64)
            wp.launch(
                MFEM.constraint_norm,
                dim=c_k.shape,
                inputs=[c_k, c_k_norm, self._stiffness_field.dof_values],
            )
            wp.utils.array_sum(c_k_norm, out=c_r)

    def evaluate_energy(self):
        E_u = self._E[0:1]
        E_e = self._E[1:2]
        c_r = self._E[2:3]

        lagrangian = self._lagrangian_constraint_energy

        if (
            self.args.cuda_graphs
            and self.__class__._ENERGY_MODULES_LOADED
            and self._E_graph is None
        ):
            try:
                gc.collect(0)
                gc.disable()
                with wp.ScopedCapture(force_module_load=False) as capture:
                    self._evaluate_energy(E_u, E_e, c_r, lagrangian=lagrangian)
                    wp.copy(src=self._E, dest=self._E_pinned)
                    gc.collect(0)
                gc.enable()
                self._E_graph = capture.graph
            except Exception as err:
                print("Energy graph capture failed", err)

        if self._E_graph is None:
            self._evaluate_energy(E_u, E_e, c_r, lagrangian=lagrangian)
            wp.copy(src=self._E, dest=self._E_pinned)
            self.__class__._ENERGY_MODULES_LOADED = True
        else:
            wp.capture_launch(self._E_graph)

        wp.synchronize_stream()

        E = self._E_pinned.numpy()
        E_tot = E[0] + E[1]
        c_r = -E[2] if lagrangian else E[2]

        return E_tot, c_r

    def solve_newton_system(self, lhs, rhs):
        if self.args.fp64:
            lhs = lhs.cast(wp.float64)

        if self._schur_work_arrays is None:
            self._schur_work_arrays = sp.bsr_mm_work_arrays()
            reuse_topology = False
        else:
            reuse_topology = True

        return lhs.solve_schur(
            rhs,
            max_iters=self.args.cg_iters,
            tol=self.args.cg_tol,
            work_arrays=self._schur_work_arrays,
            reuse_topology=reuse_topology,
        )

    def newton_rhs(self, tape: wp.Tape = None):
        with_gradient = tape is not None

        u_rhs = self.constraint_free_rhs(tape=tape)

        if with_gradient:
            c_k = wp.empty_like(self.constraint_field.dof_values, requires_grad=True)
            f = wp.empty_like(self.S.dof_values, requires_grad=True)
            w = wp.empty_like(self.R.dof_values, requires_grad=True)

            with tape:
                self._assemble_rhs(u_rhs, f, w, c_k)
        else:
            wp.copy(src=u_rhs, dest=self._u_rhs)
            u_rhs = self._u_rhs
            f = self._f
            w = self._w
            c_k = self._c_k

            if (
                self.args.cuda_graphs
                and self.__class__._RHS_MODULES_LOADED
                and self._rhs_graph is None
            ):
                try:
                    gc.collect(0)
                    gc.disable()
                    with wp.ScopedCapture(force_module_load=False) as capture:
                        self._assemble_rhs(
                            u_rhs,
                            f,
                            w,
                            c_k,
                            minus_dE_dU=self._minus_dE_du,
                            dE_dS=self._dE_dS,
                            dE_dR=self._dE_dR,
                        )
                        gc.collect(0)
                    gc.enable()
                    self._rhs_graph = capture.graph
                except Exception as err:
                    print("RHS CAPTURE FAILED", err)

            if self._rhs_graph is None:
                self._assemble_rhs(
                    u_rhs,
                    f,
                    w,
                    c_k,
                    minus_dE_dU=self._minus_dE_du,
                    dE_dS=self._dE_dS,
                    dE_dR=self._dE_dR,
                )
                self.__class__._RHS_MODULES_LOADED = True
            else:
                wp.capture_launch(self._rhs_graph)

        # Displacement boundary condition -- Filter u rhs
        self._filter_forces(u_rhs, tape=tape)

        return u_rhs, f, w, c_k

    def record_adjoint(self, tape):
        # The forward Newton is finding a root of rhs(q, p) = 0 with q = (u, S, R, lambda)
        # so drhs/dp = drhs/dq dq/dp + drhs/dp = 0
        # [- drhs/dq] dq/dp = drhs/dp
        # lhs dq/dp = drhs/dp

        self.prepare_newton_step(tape)
        rhs = self.newton_rhs(tape=tape)
        lhs = self.newton_lhs()

        def solve_backward():
            adj_res = (
                self.u_field.dof_values.grad,
                *(wp.zeros_like(field) for field in rhs[1:-1]),
                self.constraint_field.dof_values.grad,
            )
            delta_du, dS, dR, dLambda = lhs.cast(wp.float64).solve_schur(
                adj_res, max_iters=self.args.cg_iters
            )

            u_rhs, f, w_lambda, c_k = rhs
            wp.copy(src=delta_du, dest=u_rhs.grad)
            wp.copy(src=dLambda, dest=c_k.grad)
            array_axpy(dS, f.grad, alpha=-1.0, beta=0.0)
            array_axpy(dR, w_lambda.grad, alpha=-1.0, beta=0.0)

        tape.record_func(
            solve_backward,
            arrays=[
                self.u_field.dof_values,
                self.constraint_field.dof_values,
                *rhs,
            ],
        )

    def interpolate_constraint_field(self, strain=False):
        tau = fem.make_test(
            self.interpolated_constraint_field.space,
            space_restriction=self.u_test.space_restriction,
        )

        if strain:
            # interpolate strain instead of stress field
            fem.integrate(
                tensor_mass_form,
                quadrature=self.strain_quadrature,
                fields={"sig": self.S, "tau": tau},
                output=self.interpolated_constraint_field.dof_values,
                kernel_options={"enable_backward": True},
            )
        else:
            fem.integrate(
                tensor_mass_form,
                quadrature=self.strain_quadrature,
                fields={"sig": self.constraint_field, "tau": tau},
                output=self.interpolated_constraint_field.dof_values,
                kernel_options={"enable_backward": True},
            )

        # Scale by inverse mass

        if self._mass is None:
            mass_test = fem.make_test(
                self._mass_space, space_partition=self.u_test.space_partition
            )
            self._mass = fem.integrate(
                self.mass_form,
                quadrature=self.strain_quadrature,
                fields={"p": mass_test},
                output_dtype=wp.float32,
            )

        wp.launch(
            self.scale_interpolated_quantity,
            dim=self._mass.shape,
            inputs=[self.interpolated_constraint_field.dof_values, self._mass],
        )

    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        super(MFEM, MFEM).add_parser_arguments(parser)

        parser.add_argument(
            "--cuda-graphs", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--line-search",
            "-ls",
            choices=["merit", "mobj", "lagrangian"],
            default="merit",
        )

    @fem.integrand
    def mass_form(s: Sample, p: Field):
        return p(s)

    @wp.kernel
    def scale_interpolated_quantity(
        qtt: wp.array(dtype=wp.mat33), mass: wp.array(dtype=float)
    ):
        i = wp.tid()
        qtt[i] = qtt[i] / mass[i]

    @fem.integrand
    def _typical_stiffness_field(s: Sample, lame: Field):
        return wp.min(lame(s))

    @fem.integrand
    def hooke_elasticity_hessian_form(
        s: Sample, domain: Domain, S: Field, tau: Field, sig: Field, lame: Field
    ):
        return hooke_hessian(S(s), tau(s), sig(s), lame(s))

    @fem.integrand
    def hooke_elasticity_gradient_form(
        s: Sample, domain: Domain, tau: Field, S: Field, lame: Field
    ):
        return wp.ddot(tau(s), hooke_stress(S(s), lame(s)))

    @fem.integrand
    def hooke_elasticity_energy_form(s: Sample, domain: Domain, S: Field, lame: Field):
        return hooke_energy(S(s), lame(s))

    @fem.integrand
    def nh_elasticity_hessian_form(
        s: Sample, domain: Domain, S: Field, tau: Field, sig: Field, lame: Field
    ):
        return snh_hessian_proj(S(s), tau(s), sig(s), lame(s))

    @fem.integrand
    def nh_elasticity_gradient_form(
        s: Sample, domain: Domain, tau: Field, S: Field, lame: Field
    ):
        return wp.ddot(tau(s), snh_stress(S(s), lame(s)))

    @fem.integrand
    def nh_elasticity_energy_form(s: Sample, domain: Domain, S: Field, lame: Field):
        return snh_energy(S(s), lame(s))

    @wp.kernel
    def constraint_norm(
        C: wp.array(dtype=Any),
        C_norm: wp.array(dtype=wp.float64),
        scale: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        Ci = C[i]
        C_norm[i] = wp.float64(wp.sqrt(0.5 * wp.dot(Ci, Ci)) * scale[i])


class MFEM_RS_F(MFEM):
    """RS = F variant"""

    _RHS_MODULES_LOADED = False
    _ENERGY_MODULES_LOADED = False

    def init_strain_spaces(self):
        super().init_strain_spaces(constraint_dof_mapper=FullTensorMapper())

        self._pen_field = fem.make_collocated_function_space(
            self._strain_basis,
            dtype=wp.vec2,
        ).make_field(space_partition=self.sym_test.space_partition)
        self._pen_field_restr = fem.make_restriction(
            self._pen_field, space_restriction=self.sym_test.space_restriction
        )

    def supports_discontinuities(self):
        return True

    def init_constant_forms(self):
        super().init_constant_forms()

        if self.has_discontinuities():
            self._constraint_side_test = fem.make_test(
                self.constraint_test.space,
                space_partition=self.constraint_test.space_partition,
                domain=fem.Sides(self.geo),
            )

        # Displacement gradient matrix
        self.B = fem.integrate(
            self.dispgrad_form,
            fields={"tau": self.constraint_test, "u": self.u_trial},
            output_dtype=float,
            quadrature=self.strain_quadrature,
            kernel_options={"enable_backward": True},
        )

        if self.has_discontinuities():
            self.B += fem.integrate(
                self.dispgrad_side_form,
                fields={"tau": self._constraint_side_test, "u": self.u_side_trial},
                output_dtype=float,
                quadrature=self.side_quadrature,
            )
        self.B.nnz_sync()

        # Temp storage for lhs cuda graph
        self._lhs = None
        self._lhs_graph = None

    def project_constant_forms(self):
        super().project_constant_forms()

        self.B_proj = sp.bsr_copy(self.B)
        sp.bsr_mm(x=self.B, y=self.v_bd_matrix, z=self.B_proj, alpha=-1.0, beta=1.0)
        self.B_proj.nnz_sync()

        self.Bt_proj = sp.bsr_transposed(self.B_proj)
        self.Bt_proj.nnz_sync()

    def prepare_newton_step(self, tape: Optional[wp.Tape] = None):
        # Update penalization (no contribution to derivatives)
        backward_step = tape is not None
        fem.interpolate(
            self.penalization_field,
            dest=self._pen_field_restr,
            fields={
                "S": self.S,
                "R": self.R,
                "stress": self.constraint_field,
                "lame": self.lame_field,
            },
            values={
                "rot_compliance": 0.0 if backward_step else self.args.rot_compliance,
                "typ_stiff": self.typical_stiffness * self.args.constraint_pen,
            },
        )

    def newton_lhs(self):
        if self.args.cuda_graphs and self._lhs is not None and self._lhs_graph is None:
            try:
                gc.collect(0)
                gc.disable()
                with wp.ScopedCapture(force_module_load=False) as capture:
                    self._assemble_lhs(self._lhs)
                    gc.collect(0)
                gc.enable()
                self._lhs_graph = capture.graph
            except Exception as err:
                print("LHS capture failed", err)

        if self._lhs_graph is None:
            if self._lhs is None:
                self._lhs = self._assemble_lhs()
                self._lhs._H_pen = sp.bsr_copy(self._lhs._H)
            else:
                self._assemble_lhs(self._lhs)
        else:
            wp.capture_launch(self._lhs_graph)

        self._lhs._A = self.A_proj
        self._lhs._B = self.B_proj
        self._lhs._Bt = self.Bt_proj

        return self._lhs

    def _assemble_lhs(self, lhs: MFEMSystem = None):
        W_skew = fem.integrate(
            MFEM_RS_F.rot_penalization,
            fields={
                "tau": self.skew_test,
                "sig": self.skew_trial,
                "S": self.S,
                "pen": self._pen_field,
            },
            nodal=True,
            output=lhs._W if lhs else None,
            output_dtype=float,
        )

        # Grad of rotated strain w.r.t R, S
        CSk = fem.integrate(
            self.rotated_strain_form,
            nodal=True,
            fields={"sig": self.sym_trial, "R": self.R, "tau": self.constraint_test},
            output=lhs._Cs if lhs else None,
            output_dtype=float,
            kernel_options={"enable_backward": True},
        )
        CRk = fem.integrate(
            self.incremental_strain_rotation_form,
            nodal=True,
            fields={
                "sig": self.S,
                "dR": self.skew_trial,
                "R": self.R,
                "tau": self.constraint_test,
            },
            output=lhs._Cr if lhs else None,
            output_dtype=float,
            kernel_options={"enable_backward": True},
        )

        # Elasticity -- use nodal integration so that H is block diagonal
        H = fem.integrate(
            self.elastic_hessian_form,
            nodal=True,
            fields={
                "S": self.S,
                "sig": self.sym_trial,
                "tau": self.sym_test,
                "lame": self.lame_field,
            },
            output=lhs._H if lhs else None,
            output_dtype=float,
        )
        H_pen = fem.integrate(
            MFEM_RS_F.strain_penalization,
            fields={
                "tau": self.sym_test,
                "sig": self.sym_trial,
                "pen": self._pen_field,
            },
            nodal=True,
            output=lhs._H_pen if lhs else None,
            output_dtype=float,
        )
        fem.utils.array_axpy(x=H_pen.values, y=H.values)

        return MFEMSystem(
            self.A_proj, H, W_skew, self.B_proj, CSk, CRk, Bt=self.Bt_proj
        )

    def _evaluate_constraint_residual(self, out):
        fem.integrate(
            self.constraint_form,
            nodal=True,
            fields={
                "u": self.u_field,
                "tau": self.constraint_test,
                "sig": self.S,
                "R": self.R,
            },
            kernel_options={"enable_backward": True},
            output=out,
        )

        if self.has_discontinuities():
            fem.integrate(
                self.dispgrad_side_form,
                quadrature=self.side_quadrature,
                fields={
                    "u": self.u_field.trace(),
                    "tau": self._constraint_side_test,
                },
                kernel_options={"enable_backward": True},
                output=out,
                add=True,
            )

    def _assemble_rhs(self, u_rhs, f, w, c_k, minus_dE_dU=None, dE_dS=None, dE_dR=None):
        if minus_dE_dU:
            wp.copy(src=u_rhs, dest=minus_dE_dU)

        # Add current stresses
        fem.integrate(
            self.dispgrad_form,
            quadrature=self.strain_quadrature,
            fields={"u": self.u_test, "tau": self.constraint_field},
            kernel_options={"enable_backward": True},
            output=u_rhs,
            add=True,
        )

        if self.has_discontinuities():
            fem.integrate(
                self.dispgrad_side_form,
                quadrature=self.side_quadrature,
                fields={
                    "u": self.u_side_test,
                    "tau": self.constraint_field.trace(),
                },
                kernel_options={"enable_backward": True},
                output=u_rhs,
                add=True,
            )

        # c_k -- constraint residual (Fk - RS)
        self._evaluate_constraint_residual(out=c_k)

        # Other primal variables:
        # Symmetric and skew-symmetric strains

        # Elastic stress + Lagrange multiplier
        fem.integrate(
            self.elastic_gradient_form,
            nodal=True,
            fields={"S": self.S, "tau": self.sym_test, "lame": self.lame_field},
            output=f,
            kernel_options={"enable_backward": True},
        )

        if dE_dS:
            wp.copy(src=f, dest=dE_dS)

        fem.integrate(
            self.strain_penalization_rhs,
            nodal=True,
            fields={
                "sig": self.sym_test,
                "u": self.u_field,
                "R": self.R,
                "S": self.S,
                "pen": self._pen_field,
            },
            output=f,
            add=True,
            kernel_options={"enable_backward": True},
        )

        fem.integrate(
            self.rotated_strain_form,
            nodal=True,
            fields={
                "sig": self.sym_test,
                "R": self.R,
                "tau": self.constraint_field,
            },
            output=f,
            add=True,
            kernel_options={"enable_backward": True},
        )

        # Rotational stress

        fem.integrate(
            self.rot_penalization_rhs,
            nodal=True,
            fields={
                "sig": self.skew_test,
                "u": self.u_field,
                "R": self.R,
                "S": self.S,
                "pen": self._pen_field,
            },
            output=w,
            kernel_options={"enable_backward": True},
        )

        if dE_dR:
            dE_dR.zero_()

        fem.integrate(
            self.incremental_strain_rotation_form,
            nodal=True,
            fields={
                "sig": self.S,
                "dR": self.skew_test,
                "R": self.R,
                "tau": self.constraint_field,
            },
            output=w,
            add=True,
            kernel_options={"enable_backward": True},
        )

    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        super(MFEM_RS_F, MFEM_RS_F).add_parser_arguments(parser)

        parser.add_argument("--constraint_pen", type=float, default=0.01)
        parser.add_argument("--rot_compliance", type=float, default=0.1)

    @fem.integrand
    def constraint_form(
        domain: Domain, s: Sample, u: Field, tau: Field, R: Field, sig: Field
    ):
        C = defgrad(u, s) - rotation_matrix(R(s)) * sig(s)
        return wp.ddot(C, tau(s))

    @fem.integrand
    def dispgrad_form(
        domain: Domain,
        s: Sample,
        u: Field,
        tau: Field,
    ):
        """
        Displacement gradient form
        grad(u) : tau^T
        """
        return wp.ddot(tau(s), fem.grad(u, s))

    @fem.integrand
    def dispgrad_side_form(
        domain: Domain,
        s: Sample,
        u: Field,
        tau: Field,
    ):
        """
        Displacement gradient form
        grad(u) : tau^T
        """
        grad_h = -wp.outer(fem.jump(u, s), fem.normal(domain, s))
        return wp.ddot(tau(s), grad_h)

    @fem.integrand
    def rotated_strain_form(
        s: Sample, domain: Domain, R: Field, sig: Field, tau: Field
    ):
        """
        Form expressing variation of rotated deformation gradient with rotation increment
        R S : tau^T
        """
        return wp.ddot(rotation_matrix(R(s)) * sig(s), tau(s))

    @fem.integrand
    def incremental_strain_rotation_form(
        s: Sample, domain: Domain, R: Field, dR: Field, sig: Field, tau: Field
    ):
        """
        Form expressing variation of rotated deformation gradient with rotation increment
        R dR S : tau^T
        """
        return wp.ddot(rotation_matrix(R(s)) * dR(s) * sig(s), tau(s))

    @fem.integrand
    def rot_penalization_rhs(
        s: Sample,
        sig: Field,
        u: Field,
        R: Field,
        S: Field,
        pen: Field,
    ):
        S_s = S(s)
        R_s = rotation_matrix(R(s))
        F_s = defgrad(u, s)
        C_s = F_s - R_s * S_s
        return -pen(s)[0] * wp.ddot(R_s * sig(s) * S_s, C_s)

    @fem.integrand
    def rot_penalization(
        s: Sample,
        sig: Field,
        tau: Field,
        S: Field,
        pen: Field,
    ):
        S_s = S(s)
        return pen(s)[0] * wp.ddot(sig(s) * S_s, tau(s) * S_s) + pen(s)[1] * wp.ddot(
            sig(s), tau(s)
        )

    @fem.integrand
    def strain_penalization_rhs(
        s: Sample, sig: Field, u: Field, R: Field, S: Field, pen: Field
    ):
        S_s = S(s)
        R_s = rotation_matrix(R(s))
        F_s = defgrad(u, s)
        C_s = F_s - R_s * S_s
        return -pen(s)[0] * wp.ddot(R_s * sig(s), C_s)

    @fem.integrand
    def strain_penalization(
        s: Sample,
        sig: Field,
        tau: Field,
        pen: Field,
    ):
        return pen(s)[0] * wp.ddot(sig(s), tau(s))

    @fem.integrand
    def penalization_field(
        s: Sample,
        S: Field,
        R: Field,
        stress: Field,
        lame: Field,
        rot_compliance: float,
        typ_stiff: float,
    ):
        rot = rotation_matrix(R(s))
        strain = S(s)
        sym_stress = wp.transpose(rot) * stress(s)

        skew_stress = sym_stress - wp.transpose(sym_stress)
        lbd_pen = wp.sqrt(wp.ddot(skew_stress, skew_stress)) * wp.sqrt(
            wp.ddot(strain, strain)
        )

        return wp.vec2(typ_stiff, rot_compliance * lbd_pen)


class MFEM_sF_S(MFEM):
    """s(F) = S variant (Trusty SigAsia22)"""

    _RHS_MODULES_LOADED = False
    _ENERGY_MODULES_LOADED = False

    def init_strain_spaces(self):
        super().init_strain_spaces(
            constraint_dof_mapper=fem.SymmetricTensorMapper(
                dtype=wp.mat33, mapping=fem.SymmetricTensorMapper.Mapping.DB16
            ),
        )

    def init_constant_forms(self):
        super().init_constant_forms()

        self.C = fem.integrate(
            tensor_mass_form,
            nodal=True,
            fields={"sig": self.sym_trial, "tau": self.constraint_test},
            output_dtype=float,
            kernel_options={"enable_backward": True},
        )

    def _apply_rotation_delta(self, dR, alpha):
        pass

    def newton_lhs(self):
        # Grad of rotated strain w.r.t R, S

        Bk_proj = fem.integrate(
            self.rotated_dispgrad_form,
            fields={
                "u_cur": self.u_field,
                "tau": self.constraint_test,
                "u": self.u_trial,
            },
            output_dtype=float,
            quadrature=self.strain_quadrature,
            kernel_options={"enable_backward": True},
        )

        sp.bsr_mm(x=Bk_proj, y=self.v_bd_matrix, z=Bk_proj, alpha=-1.0, beta=1.0)

        # Elasticity -- use nodal integration so that H is block diagonal
        H = fem.integrate(
            self.elastic_hessian_form,
            nodal=True,
            fields={
                "S": self.S,
                "sig": self.sym_trial,
                "tau": self.sym_test,
                "lame": self.lame_field,
            },
            output_dtype=float,
        )

        return MFEMSystem(self.A_proj, H, None, Bk_proj, self.C, None)

    def _evaluate_constraint_residual(self, out):
        fem.integrate(
            self.constraint_form,
            nodal=True,
            fields={
                "u": self.u_field,
                "tau": self.constraint_test,
                "sig": self.S,
                "R": self.R,  # unused
            },
            kernel_options={"enable_backward": True},
            output=out,
        )

    def _assemble_rhs(self, u_rhs, f, w, c_k, minus_dE_dU=None, dE_dS=None, dE_dR=None):
        if minus_dE_dU:
            wp.copy(src=u_rhs, dest=minus_dE_dU)

        # Add current stresses
        fem.integrate(
            self.rotated_dispgrad_form,
            quadrature=self.strain_quadrature,
            fields={
                "u_cur": self.u_field,
                "u": self.u_test,
                "tau": self.constraint_field,
            },
            kernel_options={"enable_backward": True},
            output=u_rhs,
            add=True,
        )

        # constraint residual
        self._evaluate_constraint_residual(out=c_k)

        # Symmetric strain

        # Elastic stress + Lagrange multiplier
        fem.integrate(
            self.elastic_gradient_form,
            nodal=True,
            fields={"S": self.S, "tau": self.sym_test, "lame": self.lame_field},
            output=f,
            kernel_options={"enable_backward": True},
        )

        if dE_dS:
            wp.copy(src=f, dest=dE_dS)

        fem.integrate(
            tensor_mass_form,
            nodal=True,
            fields={
                "sig": self.sym_test,
                "tau": self.constraint_field,
            },
            output=f,
            add=True,
            kernel_options={"enable_backward": True},
        )

    @fem.integrand
    def constraint_form(
        domain: Domain, s: Sample, u: Field, tau: Field, sig: Field, R: Field
    ):
        C = symmetric_strain(defgrad(u, s)) - sig(s)
        return wp.ddot(C, tau(s))

    @fem.integrand
    def rotated_dispgrad_form(
        domain: Domain,
        s: Sample,
        u_cur: Field,
        u: Field,
        tau: Field,
    ):
        """
        Rotated deformation gradient form
        dS : tau
        """
        F = defgrad(u_cur, s)
        dF = fem.grad(u, s)
        return wp.ddot(symmetric_strain_delta(F, dF), tau(s))
