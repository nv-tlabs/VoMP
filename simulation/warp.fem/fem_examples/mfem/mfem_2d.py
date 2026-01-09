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

import numpy as np

from warp.fem import Domain, Sample, Field
from warp.fem import normal, integrand, grad
import warp.fem as fem

from warp.fem.utils import array_axpy
from warp.sparse import bsr_transposed, bsr_mm, bsr_axpy, bsr_mv, bsr_copy

import warp.examples.fem.utils as fem_example_utils

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import math

_SQRT_2 = wp.constant(math.sqrt(2.0))
_SQRT_1_2 = wp.constant(math.sqrt(1.0 / 2.0))


class FullTensorMapper(fem.DofMapper):
    """Orthonormal isomorphism from R^{n (n+1)} to nxn symmetric tensors,
    using usual L2 norm for vectors and half Frobenius norm, (tau : tau)/2 for tensors.
    """

    def __init__(self, dtype: type):
        self.value_dtype = dtype
        self.DOF_SIZE = wp.constant(4)
        self.dof_dtype = wp.vec4

    def __str__(self):
        return f"_{self.DOF_SIZE}"

    @wp.func
    def dof_to_value(dof: wp.vec4):
        a = _SQRT_2 * dof[0]
        b = _SQRT_2 * dof[1]
        c = dof[2]
        d = dof[3]
        return wp.mat22(a, c - d, c + d, b)

    @wp.func
    def value_to_dof(val: wp.mat22):
        a = _SQRT_1_2 * val[0, 0]
        b = _SQRT_1_2 * val[1, 1]
        c = 0.5 * (val[0, 1] + val[1, 0])
        d = 0.5 * (val[1, 0] - val[0, 1])
        return wp.vec4(a, b, c, d)


@wp.func
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(
        n=2, dtype=float
    )


@integrand
def linear_elasticity_hessian_form(
    s: Sample, S: Field, tau: Field, sig: Field, lame: wp.vec2
):
    return wp.ddot(hooke_stress(sig(s), lame), tau(s))


@integrand
def linear_elasticity_gradient_form(s: Sample, tau: Field, S: Field, lame: wp.vec2):
    return wp.ddot(hooke_stress(S(s) - wp.identity(n=2, dtype=float), lame), tau(s))


@integrand
def linear_elasticity_energy(s: Sample, S: Field, lame: wp.vec2):
    strain = S(s) - wp.identity(n=2, dtype=float)
    return 0.5 * wp.ddot(strain, hooke_stress(strain, lame))


@wp.func
def nh_parameters_from_lame(lame: wp.vec2):
    """Parameters such that for small strains model behaves according to Hooke's law"""
    mu_nh = lame[1]
    lambda_nh = lame[0] + lame[1]

    return mu_nh, lambda_nh


@wp.func
def nh_energy(F: wp.mat22, lame: wp.vec2):
    J = wp.determinant(F)
    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    gamma = 1.0 + mu_nh / lambda_nh

    return 0.5 * lambda_nh * (J - gamma) * (J - gamma) + 0.5 * mu_nh * wp.ddot(F, F)


@wp.func
def nh_stress(F: wp.mat22, lame: wp.vec2):
    J = wp.determinant(F)
    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    gamma = 1.0 + mu_nh / lambda_nh

    dJ_dF = wp.mat22(F[1, 1], -F[1, 0], -F[0, 1], F[0, 0])
    return mu_nh * F + (lambda_nh * (J - gamma)) * dJ_dF


@integrand
def nh_elasticity_hessian_form(
    s: Sample, S: Field, tau: Field, sig: Field, lame: wp.vec2
):
    tau_s = tau(s)
    sig_s = sig(s)

    F_s = S(s)
    dJ_dF = wp.mat22(F_s[1, 1], -F_s[1, 0], -F_s[0, 1], F_s[0, 0])

    mu_nh, lambda_nh = nh_parameters_from_lame(lame)

    dpsi_dpsi = mu_nh * wp.ddot(tau_s, sig_s) + lambda_nh * wp.ddot(
        dJ_dF, tau_s
    ) * wp.ddot(dJ_dF, sig_s)

    # SPD projection of (J - gamma) d2J_dS2
    gamma = 1.0 + mu_nh / lambda_nh
    J = wp.determinant(F_s)

    d2J_dF_pos = wp.mat22(
        sig_s[1, 1] + sig_s[0, 0],
        sig_s[0, 1] - sig_s[1, 0],
        sig_s[1, 0] - sig_s[0, 1],
        sig_s[0, 0] + sig_s[1, 1],
    )
    d2J_dF_neg = wp.mat22(
        sig_s[1, 1] - sig_s[0, 0],
        -sig_s[0, 1] - sig_s[1, 0],
        -sig_s[1, 0] - sig_s[0, 1],
        sig_s[0, 0] - sig_s[1, 1],
    )

    d2J_dF = wp.min(0.5 * lambda_nh * (J - gamma), mu_nh) * d2J_dF_neg
    d2J_dF += wp.max(0.5 * lambda_nh * (J - gamma), -mu_nh) * d2J_dF_pos

    return dpsi_dpsi + wp.ddot(d2J_dF, tau_s)


@integrand
def nh_elasticity_gradient_form(s: Sample, tau: Field, S: Field, lame: wp.vec2):
    return wp.ddot(tau(s), nh_stress(S(s), lame))


@integrand
def nh_elasticity_energy(s: Sample, S: Field, lame: wp.vec2):
    return nh_energy(S(s), lame)


@integrand
def tensor_mass_form(s: Sample, sig: Field, tau: Field):
    """
    Mass form over tensor space
       sig : tau
    """
    return wp.ddot(sig(s), tau(s))


@integrand
def boundary_projector_form(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    """Dirichlet boundary condition projector (fixed vertices selection)"""

    nor = normal(domain, s)
    clamped = float(0.0)

    # Single clamped point
    if s.qp_index == 0:
        clamped = 1.0

    # clamped vertical sides
    # clamped = wp.abs(nor[0])

    # clamped right sides
    # clamped = wp.max(0.0, nor[0])

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
    return -displacement * wp.dot(nor, v(s))


@integrand
def inertia_form(s: Sample, u: Field, v: Field, rho: float, dt: float):
    """<rho/dt^2 u, v>"""

    u_rhs = rho * u(s) / (dt * dt)
    return wp.dot(u_rhs, v(s))


@integrand
def dg_penalty_form(s: Sample, domain: Domain, u: Field, v: Field, k: float):
    ju = fem.jump(u, s)
    jv = fem.jump(v, s)

    return wp.dot(ju, jv) * k / 10.0 * fem.measure_ratio(domain, s)


@integrand
def displacement_rhs_form(
    s: Sample, u_cur: Field, u: Field, v: Field, rho: float, gravity: wp.vec2, dt: float
):
    """<rho/dt^2 u, v> + <rho g, v>"""

    return (
        inertia_form(s, u, v, rho, dt)
        - inertia_form(s, u_cur, v, rho, dt)
        + rho * wp.dot(gravity, v(s))
    )


@integrand
def kinetic_potential_energy(
    s: Sample, u: Field, v: Field, rho: float, dt: float, gravity: wp.vec2
):
    du = u(s)
    dv = v(s)
    return rho * (0.5 * wp.dot(du - dv, du - dv) / (dt * dt) - wp.dot(du, gravity))


@wp.func
def rotation_matrix(angle: float):
    # return wp.identity(n=2, dtype=wp.float32)

    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.mat22(c, -s, s, c)


@wp.kernel
def apply_rotation_delta(
    R: wp.array(dtype=float), dR: wp.array(dtype=float), alpha: float
):
    i = wp.tid()
    R[i] = R[i] + dR[i] * alpha


class MFEM:
    def __init__(self, args):
        self.args = args

        if args.grid:
            self.geo = fem.Grid2D(
                res=wp.vec2i(args.resolution), bounds_lo=wp.vec2(0.0, 0.75)
            )
        else:
            positions, tri_vidx = fem_example_utils.gen_trimesh(
                res=wp.vec2i(args.resolution), bounds_lo=wp.vec2(0.0, 0.75)
            )
            self.geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)

        print("Cell area", 0.25 / self.geo.cell_count())

        # Strain-stress matrix
        young = args.young_modulus
        poisson = args.poisson_ratio
        self.lame = wp.vec2(
            young / (1.0 + poisson) * np.array([poisson / (1.0 - poisson), 0.5])
        )

        self.dt = args.dt
        self.gravity = wp.vec2(0.0, -args.gravity)
        self.rot_stiff = 1.0 / args.rot_compliance

        if args.grid:
            self.strain_degree = args.degree
            self.rot_degree = args.degree
            self.strain_basis = fem.ElementBasis.LAGRANGE
        else:
            self.strain_degree = args.degree - 1
            self.rot_degree = args.degree - 1
            self.strain_basis = fem.ElementBasis.NONCONFORMING_POLYNOMIAL

        self.strain_poly = fem.Polynomial.GAUSS_LEGENDRE

        if args.neo_hookean:
            self.elastic_energy_form = nh_elasticity_energy
            self.elastic_gradient_form = nh_elasticity_gradient_form
            self.elastic_hessian_form = nh_elasticity_hessian_form
        else:
            self.elastic_energy_form = linear_elasticity_energy
            self.elastic_gradient_form = linear_elasticity_gradient_form
            self.elastic_hessian_form = linear_elasticity_hessian_form

    def init_vel_space(self):
        # Function spaces -- Q_k for displacement, Q_{k-1}d for stress
        u_space = fem.make_polynomial_space(
            self.geo,
            degree=args.degree,
            dtype=wp.vec2,
            discontinuous=False,
            element_basis=(
                fem.ElementBasis.SERENDIPITY if self.args.serendipity else None
            ),
        )

        # Defines some fields over our function spaces
        self.u_field = u_space.make_field()  # displacement
        self.du_field = u_space.make_field()  # displacement delta
        self.du_prev = u_space.make_field()  # displacement delta

        # Since our spaces are constant, we can also predefine the test/trial functions that we will need for integration
        domain = fem.Cells(self.geo)
        self.u_trial = fem.make_trial(space=u_space, domain=domain)
        self.u_test = fem.make_test(space=u_space, domain=domain)

        sides = fem.Sides(self.geo)
        self.u_side_trial = fem.make_trial(space=u_space, domain=sides)
        self.u_side_test = fem.make_test(space=u_space, domain=sides)

    def init_strain_spaces(self):
        args = self.args

        # Store stress degrees of freedom as symmetric tensors (3 dof) rather than full 2x2 matrices
        sym_space = fem.make_polynomial_space(
            self.geo,
            degree=self.strain_degree,
            dof_mapper=fem.SymmetricTensorMapper(wp.mat22),
            discontinuous=True,
            element_basis=self.strain_basis,
            family=self.strain_poly,
        )

        # Function spaces for piecewise-constant per-element rotations and rotation vectors
        rot_space = fem.make_polynomial_space(
            self.geo,
            degree=self.rot_degree,
            discontinuous=True,
            dtype=float,
            element_basis=self.strain_basis,
            family=self.strain_poly,
        )
        skew_space = fem.make_polynomial_space(
            self.geo,
            degree=self.rot_degree,
            dof_mapper=fem.SkewSymmetricTensorMapper(wp.mat22),
            discontinuous=True,
            element_basis=self.strain_basis,
            family=self.strain_poly,
        )

        # Defines some fields over our function spaces

        self.S = sym_space.make_field()  # Rotated symmetric train
        self.S.dof_values.fill_(
            sym_space.dof_mapper.value_to_dof(wp.mat22(1.0, 0.0, 0.0, 1.0))
        )  # initialize with identity

        self.R = rot_space.make_field()  # Rotation

        # Since our spaces are constant, we can also predefine the test/trial functions that we will need for integration
        domain = fem.Cells(self.geo)
        self.sym_test = fem.make_test(space=sym_space, domain=domain)
        self.sym_trial = fem.make_trial(
            space=sym_space,
            space_partition=self.sym_test.space_partition,
            domain=domain,
        )

        if skew_space.degree == sym_space.degree:
            self.skew_test = fem.make_test(
                space=skew_space,
                space_partition=self.sym_test.space_partition,
                domain=domain,
            )
            self.skew_trial = fem.make_trial(
                space=skew_space,
                space_partition=self.sym_test.space_partition,
                domain=domain,
            )
        else:
            self.skew_test = fem.make_test(space=skew_space, domain=domain)
            self.skew_trial = fem.make_trial(space=skew_space, domain=domain)

        self.quadrature = fem.RegularQuadrature(domain, order=2 * args.degree)

    def init_boundary_conditions(self):
        u_space = self.u_field.space

        # Displacement boundary conditions
        # For simplicity, assume constant per-frame displacement
        boundary = fem.BoundarySides(self.geo)
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        self.v_bd_rhs = fem.integrate(
            boundary_displacement_form,
            fields={"v": u_bd_test},
            values={"displacement": args.displacement / args.n_frames},
            nodal=True,
            output_dtype=wp.vec2f,
        )
        self.v_bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": u_bd_trial, "v": u_bd_test},
            nodal=True,
            output_dtype=float,
        )
        fem.normalize_dirichlet_projector(self.v_bd_matrix, self.v_bd_rhs)

    def init_constant_forms(self):
        self.A = fem.integrate(
            inertia_form,
            fields={"u": self.u_trial, "v": self.u_test},
            values={"rho": args.density, "dt": self.dt},
            output_dtype=float,
        ) + fem.integrate(
            dg_penalty_form,
            fields={"u": self.u_side_trial, "v": self.u_side_test},
            values={"k": self.lame[0]},
            output_dtype=float,
        )

        self.Ci = fem.integrate(
            tensor_mass_form,
            fields={"tau": self.constraint_test, "sig": self.constraint_trial},
            nodal=True,
            output_dtype=float,
        )
        fem_example_utils.invert_diagonal_bsr_matrix(self.Ci)

    def run_frame(self):
        (self.du_field, self.du_prev) = (self.du_prev, self.du_field)

        self.compute_initial_guess()

        tol = 1.0e-8

        for k in range(self.args.n_newton):
            E_ref = self.evaluate_energy()

            ddu, dR, dS = self.newton_iter(k)
            self.apply_newton_deltas(ddu, dR, dS)

            step_size = wp.utils.array_inner(ddu, ddu) / (1 + ddu.shape[0])

            # Line search
            alpha = 1.0
            for j in range(self.args.n_backtrack):
                E_cur = self.evaluate_energy()
                if E_cur < E_ref:
                    break

                alpha = 0.5 * alpha
                self.apply_newton_deltas(ddu, dR, dS, alpha=-alpha)

            print(f"Newton iter {k}: step size {step_size}, alpha={alpha}")

            if step_size < tol:
                break

    def assemble_constraint_free_system(self, with_external_forces=True):
        gravity = self.gravity if with_external_forces else wp.vec2(0.0)

        l = fem.integrate(
            displacement_rhs_form,
            fields={"u_cur": self.du_field, "u": self.du_prev, "v": self.u_test},
            values={"rho": args.density, "dt": self.dt, "gravity": gravity},
            output_dtype=wp.vec2,
        )

        pen_rhs = fem.integrate(
            dg_penalty_form,
            fields={"u": self.u_field.trace(), "v": self.u_side_test},
            values={"k": self.lame[0]},
            output_dtype=wp.vec2,
        )
        fem.utils.array_axpy(x=pen_rhs, y=l, alpha=-1, beta=1)

        return self.A, l

    def apply_newton_deltas(self, delta_du, dR, dS, alpha=1.0):
        # Add to total displacement
        array_axpy(x=delta_du, y=self.u_field.dof_values, alpha=alpha)
        array_axpy(x=delta_du, y=self.du_field.dof_values, alpha=alpha)

        array_axpy(x=dS, y=self.S.dof_values, alpha=alpha)

        # Apply rotation delta
        wp.launch(
            kernel=apply_rotation_delta,
            dim=self.R.space.node_count(),
            inputs=[self.R.dof_values, dR, alpha],
        )

    def evaluate_energy(self, include_constraint_residual=True):
        E_e = fem.integrate(
            self.elastic_energy_form,
            quadrature=fem.RegularQuadrature(
                fem.Cells(self.geo), order=2 * self.args.degree
            ),
            fields={"S": self.S},
            values={"lame": self.lame},
        )
        E_u = fem.integrate(
            kinetic_potential_energy,
            quadrature=self.quadrature,
            fields={"u": self.du_field, "v": self.du_prev},
            values={"rho": self.args.density, "dt": self.dt, "gravity": self.gravity},
        )
        E_pen = fem.integrate(
            dg_penalty_form,
            domain=fem.Sides(self.geo),
            fields={"u": self.u_field.trace(), "v": self.u_field.trace()},
            values={"k": self.lame[0]},
        )

        E_tot = E_u + E_e + E_pen

        if include_constraint_residual:
            ck_field = self.evaluate_ck()

            ck_field.dof_values = self.Ci @ ck_field.dof_values

            c_r = (
                fem.integrate(
                    tensor_mass_form,
                    quadrature=self.quadrature,
                    fields={"sig": ck_field, "tau": ck_field},
                )
                * self.lame[0]
            )
            E_tot += c_r

        return E_tot

    def compute_initial_guess(self):
        # Self-advect
        A, l = self.assemble_constraint_free_system(with_external_forces=False)
        u_rhs = l
        u_matrix = bsr_copy(self.A)

        fem.project_linear_system(
            u_matrix, u_rhs, self.v_bd_matrix, self.v_bd_rhs, normalize_projector=False
        )
        fem_example_utils.bsr_cg(
            u_matrix, b=u_rhs, x=self.du_field.dof_values, quiet=True
        )
        array_axpy(x=self.du_field.dof_values, y=self.u_field.dof_values)


class MFEM_S_RF(MFEM):
    """S = RF variant"""

    def __init__(self, args):
        super().__init__(args)

    def init_constant_forms(self):
        super().init_constant_forms()

        self.W = fem.integrate(
            tensor_mass_form,
            fields={"tau": self.sym_test, "sig": self.sym_trial},
            nodal=True,
            output_dtype=float,
        )
        self.W_inv = bsr_copy(self.W)
        fem_example_utils.invert_diagonal_bsr_matrix(self.W_inv)

        self.W_skew = fem.integrate(
            tensor_mass_form,
            fields={"tau": self.skew_test, "sig": self.skew_trial},
            nodal=True,
            output_dtype=float,
        )
        self.W_skew_inv = bsr_copy(self.W_skew)
        fem_example_utils.invert_diagonal_bsr_matrix(self.W_skew_inv)

    def newton_iter(self, k: int):
        # argmin_u,S,R argmax lambda V(u, u) + psi(S) + c(dR) + w(S, lambda) - b(R, u, lambda)
        #
        # with lambda = lambda_sym + lambda_skew
        #
        # a(dU, v)  - b(R, lambda, v)    = l(v) - a(u, v            forall v in R^3
        # w(dS, tau) - b(R, dU, tau)     = b(R, u, tau) - w(S, tau) forall tau in sym(3x3)
        # -b(R, dU, tau) - b(dR, u, tau)  = b(R, u, tau)            forall tau in skew(3x3)
        # h(S; dS, tau) + w(tau, lambda) = -f(S; tau)               forall tau in sym(3x3)
        # w(dR, tau) - b(tau, u, lambda) = 0                        forall tau in skew(3x3)
        #
        # a(u, v) = int( rho/dt <u, v> )
        # l(v) = int( <(rho/dt v^0 + rho g), v> )
        #
        # b(R, u, tau) = int ( R (I + grad u) : tau)
        # w(sig, tau) = int ( sig : tau)
        #
        # H(S; sig, tau) = int ( sig : (d2 psi/dS2)(S) : tau )
        # f(S; tau) = int ( (dpsi / dS)(S) : tau )
        #
        # Notes:
        # In general there should also be a b(dR, u, tau) term for tau symmetric,
        #   which will be zero if the deviatoric part of RF is zero
        # w(dR, tau) is artificial inertia on dR (Tikhonov regularization)

        # Unconstrained dynamics
        A, l = self.assemble_constraint_free_system()

        u_rhs = l

        bsr_mv(A=self.A, x=self.du_field.dof_values, y=u_rhs, alpha=-1.0, beta=1.0)

        u_matrix = bsr_copy(self.A)

        # Rotated deformation gradient RFk and gradient matrix RBk
        RFk = fem.integrate(
            self.rotated_defgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_field, "R": self.R, "tau": self.sym_test},
            output_dtype=wp.vec3,
        )
        RBk = fem.integrate(
            self.rotated_dispgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_trial, "R": self.R, "tau": self.sym_test},
            output_dtype=float,
        )

        # Optional -- reset S as symmetric part of RFk
        self.S.dof_values = self.W_inv @ RFk

        Sk = fem.integrate(
            tensor_mass_form,
            # quadrature=self.quadrature,
            nodal=True,
            fields={"sig": self.S, "tau": self.sym_test},
            output_dtype=wp.vec3,
        )

        # c_k -- sym part of constraint residual (RFk - WS)
        c_k = RFk
        array_axpy(Sk, c_k, alpha=-1.0, beta=1.0)

        # Elasticity
        H = fem.integrate(
            self.elastic_hessian_form,
            quadrature=self.quadrature,
            fields={"S": self.S, "sig": self.sym_trial, "tau": self.sym_test},
            values={"lame": self.lame},
            output_dtype=float,
        )
        f = fem.integrate(
            self.elastic_gradient_form,
            quadrature=self.quadrature,
            fields={"S": self.S, "tau": self.sym_test},
            values={"lame": self.lame},
            output_dtype=wp.vec3,
        )

        # Schur complements

        # lambda_rhs = H W^-1 c_k + f
        lambda_rhs = f
        bsr_mv(A=H, x=self.W_inv @ c_k, y=lambda_rhs, alpha=1.0, beta=1.0)

        WiRBk = self.W_inv @ RBk
        WiRBk_T = WiRBk.transpose()

        bsr_mv(A=WiRBk_T, x=lambda_rhs, y=u_rhs, alpha=-1.0, beta=1.0)

        u_matrix += WiRBk_T @ H @ WiRBk

        # Rotation
        RFk_skew = fem.integrate(
            self.rotated_defgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_field, "R": self.R, "tau": self.skew_test},
            output_dtype=float,
        )

        RBk_skew = fem.integrate(
            self.rotated_dispgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_trial, "R": self.R, "tau": self.skew_test},
            output_dtype=float,
        )

        # print(
        #    "RES ",
        #    wp.utils.array_inner(RFk_skew, bsr_mv(self.W_skew_inv, RFk_skew).view(dtype=RFk_skew.dtype)),
        #    wp.utils.array_inner(c_k, bsr_mv(self.W_inv, c_k).view(dtype=c_k.dtype)),
        # )

        Ck = fem.integrate(
            self.defgrad_incremental_rotation_form,
            fields={
                "u": self.u_field,
                "dR": self.skew_trial,
                "R": self.R,
                "tau": self.skew_test,
            },
            nodal=True,
            output_dtype=float,
        )
        WiCt = self.rot_stiff * self.W_skew_inv @ Ck.transpose()

        CWiCt_inv = Ck @ WiCt
        fem_example_utils.invert_diagonal_bsr_matrix(CWiCt_inv)

        RBk_skew_TCi = RBk_skew.transpose() @ CWiCt_inv
        u_matrix += RBk_skew_TCi @ RBk_skew
        bsr_mv(A=RBk_skew_TCi, x=RFk_skew, y=u_rhs, alpha=-1.0, beta=1.0)

        # Enforce boundary conditions
        du_bd_rhs = wp.clone(self.v_bd_rhs)
        bsr_mv(
            A=self.v_bd_matrix,
            x=self.du_field.dof_values,
            y=du_bd_rhs,
            alpha=-1.0,
            beta=1.0,
        )
        fem.project_linear_system(
            u_matrix, u_rhs, self.v_bd_matrix, du_bd_rhs, normalize_projector=False
        )

        delta_du = wp.zeros_like(u_rhs)
        fem_example_utils.bsr_cg(
            u_matrix, b=u_rhs, x=delta_du, tol=1.0e-16, max_iters=250, quiet=True
        )

        # update S
        # -RBk du  + W   dS       =  c_k
        # W  dS = c_k + RBK du
        bsr_mv(A=RBk, x=delta_du, y=c_k, alpha=1.0, beta=1.0)
        dS = self.W_inv @ c_k

        # update R
        #
        # -RBk_skew du - CWiCt dlambda = RFk
        # cW dR - Ct dlambda = 0
        bsr_mv(A=RBk_skew, x=delta_du, y=RFk_skew, alpha=1.0, beta=1.0)
        lambda_skew = -CWiCt_inv @ RFk_skew
        dR = WiCt @ lambda_skew

        return delta_du, dR, dS

    @integrand
    def constraint_residual(
        s: Sample,
        S: Field,
        R: Field,
        u: Field,
    ):
        """
        Constraint residual
        """
        RFs = rotation_matrix(R(s)) * (grad(u, s) + wp.identity(n=2, dtype=float))
        c = S(s) - RFs

        return 0.5 * wp.ddot(c, c)

    @integrand
    def rotated_dispgrad_form(
        s: Sample,
        R: Field,
        u: Field,
        tau: Field,
    ):
        """
        Rotated displacement gradient form
        R grad(u) : tau
        """
        return wp.ddot(wp.transpose(tau(s)), rotation_matrix(R(s)) * grad(u, s))

    @integrand
    def rotated_defgrad_form(
        s: Sample,
        R: Field,
        u: Field,
        tau: Field,
    ):
        """
        Rotated deformation gradient form
        R (I + grad(u)) : tau
        """
        return wp.ddot(
            wp.transpose(tau(s)),
            rotation_matrix(R(s)) * (wp.identity(n=2, dtype=float) + grad(u, s)),
        )

    @integrand
    def defgrad_incremental_rotation_form(
        s: Sample, R: Field, dR: Field, u: Field, tau: Field
    ):
        """
        Form expressing variation of rotated deformation gradient with rotation increment
        R dR grad(u) : tau
        """
        return wp.ddot(
            rotation_matrix(R(s))
            * dR(s)
            * (wp.identity(n=2, dtype=float) + grad(u, s)),
            wp.transpose(tau(s)),
        )


class MFEM_RS_F(MFEM):
    """RS = F variant"""

    def __init__(self, args):
        super().__init__(args)

        # self.rot_stiff = self.rot_stiff / self.lame[0]

    def init_strain_spaces(self):
        super().init_strain_spaces()

        constraint_space = fem.make_polynomial_space(
            self.geo,
            degree=self.strain_degree,
            # dtype=wp.mat22,
            dof_mapper=FullTensorMapper(wp.mat22),
            discontinuous=True,
            element_basis=self.strain_basis,
            family=self.strain_poly,
        )

        self.constraint_test = fem.make_test(
            space=constraint_space, space_partition=self.sym_test.space_partition
        )
        self.constraint_trial = fem.make_trial(
            space=constraint_space, space_partition=self.sym_test.space_partition
        )

    def init_constant_forms(self):
        super().init_constant_forms()

        self.quadrature = fem.RegularQuadrature(
            fem.Cells(self.geo), order=2 * (args.degree - 1)
        )

        u_sides = fem.make_trial(self.u_trial.space, domain=fem.Sides(self.geo))
        tau_sides = fem.make_test(
            self.constraint_test.space, domain=fem.Sides(self.geo)
        )

        self.B = fem.integrate(
            self.dispgrad_form,
            fields={"tau": self.constraint_test, "u": self.u_trial},
            output_dtype=float,
        ) + fem.integrate(
            self.dispgrad_side_form,
            fields={"tau": tau_sides, "u": u_sides},
            output_dtype=float,
        )

        self.Bt = bsr_transposed(self.B)

        self.W_skew = fem.integrate(
            tensor_mass_form,
            fields={"tau": self.skew_test, "sig": self.skew_trial},
            nodal=True,
            output_dtype=float,
        )
        self.W_skew_inv = bsr_copy(self.W_skew)
        fem_example_utils.invert_diagonal_bsr_matrix(self.W_skew_inv)

    def evaluate_ck(self):
        tau_sides = fem.make_test(
            self.constraint_test.space, domain=fem.Sides(self.geo)
        )

        Fk = fem.integrate(
            self.defgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_field, "tau": self.constraint_test},
            output_dtype=wp.vec4,
        )
        fem.utils.array_axpy(
            y=Fk,
            x=fem.integrate(
                self.dispgrad_side_form,
                # quadrature=self.quadrature,
                fields={"u": self.u_field.trace(), "tau": tau_sides},
                output_dtype=wp.vec4,
            ),
        )

        # Rotated deformation gradient RFk and gradient matrix RBk
        RSk = fem.integrate(
            self.rotated_strain_form,
            quadrature=self.quadrature,
            fields={"sig": self.S, "R": self.R, "tau": self.constraint_test},
            output_dtype=wp.vec4,
        )

        # c_k -- constraint residual (Fk - RS)
        c_k = Fk
        array_axpy(x=RSk, y=c_k, alpha=-1.0, beta=1.0)

        ck_field = self.constraint_test.space.make_field()
        ck_field.dof_values = c_k
        return ck_field

    def newton_iter(self, k: int):
        # Unconstrained dynamics
        A, l = self.assemble_constraint_free_system()

        u_rhs = l
        # bsr_mv(A=self.A, x=self.du_field.dof_values, y=u_rhs, alpha=-1.0, beta=1.0)

        u_matrix = bsr_copy(self.A)

        c_k = self.evaluate_ck().dof_values

        # Grad of rotated strain w.r.t R, S
        CSk = fem.integrate(
            self.rotated_strain_form,
            nodal=True,
            fields={"sig": self.sym_trial, "R": self.R, "tau": self.constraint_test},
            output_dtype=float,
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
            output_dtype=float,
        )

        # Elasticity -- use nodal integration so that H is block diagonal
        H = fem.integrate(
            self.elastic_hessian_form,
            nodal=True,
            fields={"S": self.S, "sig": self.sym_trial, "tau": self.sym_test},
            values={"lame": self.lame},
            output_dtype=float,
        )
        f = fem.integrate(
            self.elastic_gradient_form,
            quadrature=self.quadrature,
            fields={"S": self.S, "tau": self.sym_test},
            values={"lame": self.lame},
            output_dtype=wp.vec3,
        )

        H_inv = bsr_copy(H)
        wp.launch(
            kernel=self.invert_hessian_blocks, dim=H_inv.nnz, inputs=[H_inv.values]
        )

        # Schur complements

        CSkHi = CSk @ H_inv
        CSt = CSk.transpose()

        CHiCt = CSkHi @ CSt

        # lambda_rhs = c_k + CS H^-1 f
        lambda_rhs = c_k
        bsr_mv(A=CSkHi, x=f, y=lambda_rhs, alpha=1.0, beta=1.0)

        CRkWi = CRk @ (self.W_skew_inv * self.rot_stiff)
        CRt = CRk.transpose()

        CWiCt = CRkWi @ CRt

        CHiCt_inv = bsr_copy(CHiCt, scalar_type=wp.float64) + bsr_copy(
            CWiCt, scalar_type=wp.float64
        )
        wp.launch(
            kernel=self.invert_schur_blocks,
            dim=CHiCt_inv.nnz,
            inputs=[CHiCt_inv.values],
        )
        CHiCt_inv = bsr_copy(CHiCt_inv, scalar_type=wp.float32)

        BtCHiCt_inv = self.Bt @ CHiCt_inv

        bsr_mm(x=BtCHiCt_inv, y=self.B, z=u_matrix, alpha=1.0, beta=1.0)
        bsr_mv(A=BtCHiCt_inv, x=lambda_rhs, y=u_rhs, alpha=-1.0, beta=1.0)

        # Enforce boundary conditions
        du_bd_rhs = wp.clone(self.v_bd_rhs)
        bsr_mv(
            A=self.v_bd_matrix,
            x=self.du_field.dof_values,
            y=du_bd_rhs,
            alpha=-1.0,
            beta=1.0,
        )
        fem.project_linear_system(
            u_matrix, u_rhs, self.v_bd_matrix, du_bd_rhs, normalize_projector=False
        )

        delta_du = wp.zeros_like(u_rhs)
        fem_example_utils.bsr_cg(
            u_matrix, b=u_rhs, x=delta_du, tol=1.0e-16, max_iters=250, quiet=True
        )

        # get back lambda
        # -B du  -ChiC lambda = lambda_k

        bsr_mv(A=self.B, x=delta_du, y=lambda_rhs, alpha=1.0, beta=1.0)
        lambda_k = -CHiCt_inv @ lambda_rhs

        # update S
        # H dS + CSkT lambda = - f

        bsr_mv(A=CSt, x=lambda_k, y=f, alpha=1.0, beta=1.0)
        dS = -H_inv @ f

        # update R
        r = (CRt @ lambda_k).view(dtype=float)
        dR = (-self.rot_stiff * self.W_skew_inv) @ r

        return delta_du, dR, dS

    @integrand
    def constraint_residual(
        domain: Domain,
        s: Sample,
        S: Field,
        R: Field,
        u: Field,
    ):
        """
        Constraint residual
        """
        c = rotation_matrix(R(s)) * S(s) - (grad(u, s) + wp.identity(n=2, dtype=float))

        return 0.5 * wp.ddot(c, c)

    @integrand
    def dispgrad_form(
        domain: Domain,
        s: Sample,
        u: Field,
        tau: Field,
    ):
        """
        Displacement gradient form
        grad(u) : tau
        """
        return wp.ddot(tau(s), grad(u, s))

    @integrand
    def defgrad_form(
        domain: Domain,
        s: Sample,
        u: Field,
        tau: Field,
    ):
        """
        Deformation gradient form
        (I + grad(u)) : tau
        """
        return wp.ddot(tau(s), (wp.identity(n=2, dtype=float) + grad(u, s)))

    @integrand
    def dispgrad_side_form(
        domain: Domain,
        s: Sample,
        u: Field,
        tau: Field,
    ):
        """
        Displacement gradient form
        grad(u) : tau
        """
        grad_h = -wp.outer(fem.jump(u, s), fem.normal(domain, s))
        return wp.ddot(tau(s), grad_h)

    @integrand
    def rotated_strain_form(
        s: Sample, domain: Domain, R: Field, sig: Field, tau: Field
    ):
        """
        Form expressing variation of rotated deformation gradient with rotation increment
        R S : tau
        """
        return wp.ddot(rotation_matrix(R(s)) * sig(s), tau(s))

    @integrand
    def incremental_strain_rotation_form(
        s: Sample, domain: Domain, R: Field, dR: Field, sig: Field, tau: Field
    ):
        """
        Form expressing variation of rotated deformation gradient with rotation increment
        R dR S : tau
        """
        return wp.ddot(rotation_matrix(R(s)) * dR(s) * sig(s), tau(s))

    @wp.kernel
    def invert_hessian_blocks(values: wp.array(dtype=(wp.mat33))):
        i = wp.tid()
        values[i] = wp.inverse(values[i])

    @wp.kernel
    def invert_schur_blocks(values: wp.array(dtype=(wp.mat44d))):
        i = wp.tid()
        values[i] = fem.utils.inverse_qr(values[i])

    # @wp.kernel
    # def pseudo_inverse_S(
    #     values: wp.array(dtype=(wp.mat(shape=(4, 3), dtype=wp.float32)))
    # ):
    #     i = wp.tid()

    #     v = values[i]
    #     values[i] = v * wp.inverse(wp.transpose(v) * v)

    # @wp.kernel
    # def pseudo_inverse_R(
    #     values: wp.array(dtype=(wp.mat(shape=(4, 1), dtype=wp.float32)))
    # ):
    #     i = wp.tid()

    #     v = values[i]
    #     values[i] = v / wp.ddot(v, v)

    # @wp.kernel
    # def add_coupling(
    #     CHCi: wp.array(dtype=(wp.mat(shape=(4, 4), dtype=wp.float32))),
    #     CS: wp.array(dtype=(wp.mat(shape=(4, 3), dtype=wp.float32))),
    #     CR: wp.array(dtype=(wp.mat(shape=(4, 1), dtype=wp.float32))),
    #     W: wp.array(dtype=float),
    #     rot_stiff: float
    # ):
    #     i = wp.tid()

    #     cr = CR[i]
    #     cwc = cr * wp.transpose(cr) * rot_stiff / W[i]

    #     if i == 0:
    #         print(cwc)

    #     #cs = CS[i]
    #     #pis = cs * wp.transpose(cs) / wp.ddot(cs, cs)

    #     #CHCi[i] -= wp.ddot(pis, cwc) * pis
    #
    @wp.kernel
    def compute_CHiCT_inv(
        CHCi: wp.array(dtype=(wp.mat(shape=(4, 4), dtype=wp.float32))),
        CS: wp.array(dtype=(wp.mat(shape=(4, 3), dtype=wp.float32))),
        CR: wp.array(dtype=(wp.mat(shape=(4, 1), dtype=wp.float32))),
        H: wp.array(dtype=(wp.mat(shape=(3, 3), dtype=wp.float32))),
        W: wp.array(dtype=float),
        rot_stiff: float,
    ):
        i = wp.tid()

        cr = CR[i]
        cs = CS[i]

        pis = 3.0 * cs * wp.transpose(cs) / wp.ddot(cs, cs)
        pir = wp.identity(n=4, dtype=wp.float32) - pis

        cs_mp = cs * wp.inverse(wp.transpose(cs) * cs)
        cihci = cs_mp * H[i] * wp.transpose(cs_mp)

        cr_s = pis * cr
        cr_r = cr - cr_s

        wi = rot_stiff / W[i]

        Ar = wi * cr_s * wp.transpose(cr_s)
        A = cs * wp.inverse(H[i]) * wp.transpose(cs) + Ar
        C = wi * cr_r * wp.transpose(cr_r)
        B = wi * cr_r * wp.transpose(cr_s)

        # A_mp = cihci - cihci * Ar * cihci
        A_mp = pis * wp.inverse(A + pir) * pis

        cr_mp = cr_r / wp.ddot(cr_r, cr_r)
        C_mp = cr_mp * W[i] / rot_stiff * wp.transpose(cr_mp)

        Ai = A_mp + A_mp * (wp.transpose(B) * C_mp * B) * A_mp
        Ci = C_mp + C_mp * B * A_mp * wp.transpose(B) * C_mp

        # AS = A - wp.transpose(B) * C_mp * B
        # Ai = pis * wp.inverse(AS + pir) * pis

        # CSsh = C - B * A_mp * wp.transpose(B)
        # Ci = pir * wp.inverse(CSsh + pis) * pir

        CHCi[i] = Ai + Ci - Ai * wp.transpose(B) * C_mp - Ci * B * A_mp

        CHC = A + C + B + wp.transpose(B)
        # CHCi[i] = wp.inverse(CHC)

        if i == 0:
            wp.print(CHCi[i] * CHC)


class MFEM_S_RF_v2(MFEM):
    """S = RF variant v2"""

    def __init__(self, args):
        super().__init__(args)

        # self.rot_stiff = self.rot_stiff / self.lame[0]

        self.strain_form = tensor_mass_form
        self.constraint_residual = MFEM_S_RF.constraint_residual

    def init_strain_spaces(self):
        super().init_strain_spaces()

        constraint_space = fem.make_polynomial_space(
            self.geo,
            degree=self.strain_degree,
            dof_mapper=FullTensorMapper(wp.mat22),
            discontinuous=True,
            element_basis=self.strain_basis,
            family=self.strain_poly,
        )

        self.constraint_test = fem.make_test(
            space=constraint_space, space_partition=self.sym_test.space_partition
        )

    def init_constant_forms(self):
        super().init_constant_forms()

        self.quadrature = fem.RegularQuadrature(
            fem.Cells(self.geo), order=2 * (args.degree - 1)
        )

        self.W_skew = fem.integrate(
            tensor_mass_form,
            fields={"tau": self.skew_test, "sig": self.skew_trial},
            nodal=True,
            output_dtype=float,
        )
        self.W_skew_inv = bsr_copy(self.W_skew)
        fem_example_utils.invert_diagonal_bsr_matrix(self.W_skew_inv)

    def newton_iter(self, k: int):
        # Unconstrained dynamics
        A, l = self.assemble_constraint_free_system()

        u_rhs = l
        bsr_mv(A=self.A, x=self.du_field.dof_values, y=u_rhs, alpha=-1.0, beta=1.0)

        u_matrix = bsr_copy(self.A)

        # Deformation gradient Fl
        RFk = fem.integrate(
            MFEM_S_RF.rotated_defgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_field, "R": self.R, "tau": self.constraint_test},
            output_dtype=wp.vec4,
        )
        RBk = fem.integrate(
            MFEM_S_RF.rotated_dispgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_trial, "R": self.R, "tau": self.constraint_test},
            output_dtype=float,
        )
        RBk_T = bsr_transposed(RBk)

        # Rotated deformation gradient RFk and gradient matrix RBk
        Sk = fem.integrate(
            self.strain_form,
            # quadrature=self.quadrature,
            nodal=True,
            fields={"sig": self.S, "tau": self.constraint_test},
            output_dtype=wp.vec4,
        )

        # c_k -- constraint residual (RFk - S)
        c_k = RFk
        array_axpy(x=Sk, y=c_k, alpha=-1.0, beta=1.0)

        # Elasticity -- use nodal integration so that H is block diagonal
        H = fem.integrate(
            self.elastic_hessian_form,
            nodal=True,
            fields={"S": self.S, "sig": self.sym_trial, "tau": self.sym_test},
            values={"lame": self.lame},
            output_dtype=float,
        )
        f = fem.integrate(
            self.elastic_gradient_form,
            quadrature=self.quadrature,
            fields={"S": self.S, "tau": self.sym_test},
            values={"lame": self.lame},
            output_dtype=wp.vec3,
        )

        # Grad of rotated strain w.r.t R, S
        CSk = fem.integrate(
            self.strain_form,
            nodal=True,
            fields={"sig": self.sym_trial, "tau": self.constraint_test},
            output_dtype=float,
        )
        CRk = fem.integrate(
            MFEM_S_RF.defgrad_incremental_rotation_form,
            fields={
                "u": self.u_field,
                "dR": self.skew_trial,
                "R": self.R,
                "tau": self.constraint_test,
            },
            nodal=True,
            output_dtype=float,
        )

        H_inv = bsr_copy(H)
        wp.launch(
            kernel=self.invert_hessian_blocks, dim=H_inv.nnz, inputs=[H_inv.values]
        )

        # Schur complements

        CSkHi = bsr_mm(CSk, H_inv)
        CSt = bsr_transposed(CSk)

        CHiCt = bsr_mm(CSkHi, CSt)

        # lambda_rhs = c_k + CS H^-1 f
        lambda_rhs = c_k
        bsr_mv(A=CSkHi, x=f, y=lambda_rhs, alpha=1.0, beta=1.0)

        CRkWi = bsr_mm(CRk, self.W_skew_inv, alpha=self.rot_stiff)
        CRt = bsr_transposed(CRk)

        CWiCt = bsr_mm(CRkWi, CRt, alpha=1.0, beta=1.0)

        CHiCt_inv = bsr_axpy(
            bsr_copy(CHiCt, scalar_type=wp.float64),
            bsr_copy(CWiCt, scalar_type=wp.float64),
        )
        wp.launch(
            kernel=self.invert_schur_blocks,
            dim=CHiCt_inv.nnz,
            inputs=[CHiCt_inv.values],
        )
        CHiCt_inv = bsr_copy(CHiCt_inv, scalar_type=wp.float32)

        BtCHiCt_inv = bsr_mm(RBk_T, CHiCt_inv)

        bsr_mm(x=BtCHiCt_inv, y=RBk, z=u_matrix, alpha=1.0, beta=1.0)
        bsr_mv(A=BtCHiCt_inv, x=lambda_rhs, y=u_rhs, alpha=-1.0, beta=1.0)

        # Enforce boundary conditions
        du_bd_rhs = wp.clone(self.v_bd_rhs)
        bsr_mv(
            A=self.v_bd_matrix,
            x=self.du_field.dof_values,
            y=du_bd_rhs,
            alpha=-1.0,
            beta=1.0,
        )
        fem.project_linear_system(
            u_matrix, u_rhs, self.v_bd_matrix, du_bd_rhs, normalize_projector=False
        )

        delta_du = wp.zeros_like(u_rhs)
        fem_example_utils.bsr_cg(
            u_matrix, b=u_rhs, x=delta_du, tol=1.0e-16, max_iters=250, quiet=True
        )

        # get back lambda
        # -B du  -ChiC lambda = lambda_k

        bsr_mv(A=RBk, x=delta_du, y=lambda_rhs, alpha=1.0, beta=1.0)
        lambda_k = bsr_mv(CHiCt_inv, lambda_rhs, alpha=-1.0, beta=0.0)

        # update S
        # H dS + CSkT lambda = - f

        bsr_mv(A=CSt, x=lambda_k, y=f, alpha=1.0, beta=1.0)
        dS = bsr_mv(A=H_inv, x=f, alpha=-1.0, beta=0.0)

        # update R
        r = bsr_mv(A=CRt, x=lambda_k).view(dtype=float)
        dR = bsr_mv(A=self.W_skew_inv, x=r, alpha=self.rot_stiff, beta=0.0)

        return delta_du, dR, dS

    @wp.kernel
    def invert_hessian_blocks(values: wp.array(dtype=(wp.mat33))):
        i = wp.tid()
        values[i] = wp.inverse(values[i])

    @wp.kernel
    def invert_schur_blocks(values: wp.array(dtype=(wp.mat44d))):
        i = wp.tid()
        values[i] = fem.utils.inverse_qr(values[i])


class MFEM_S_F(MFEM):
    """S = F variant"""

    def __init__(self, args):
        super().__init__(args)

    def init_strain_spaces(self):
        super().init_strain_spaces()

        constraint_space = fem.make_polynomial_space(
            self.geo,
            degree=self.strain_degree,
            dtype=wp.mat22,
            discontinuous=True,
            element_basis=self.strain_basis,
            family=self.strain_poly,
        )

        self.constraint_test = fem.make_test(space=constraint_space)
        self.constraint_trial = fem.make_trial(space=constraint_space)

        self.S = constraint_space.make_field()
        self.S.dof_values.fill_(
            wp.mat22(1.0, 0.0, 0.0, 1.0)
        )  # initialize with identity

    def init_constant_forms(self):
        super().init_constant_forms()

        self.quadrature = fem.RegularQuadrature(
            fem.Cells(self.geo), order=2 * args.degree - 1
        )

        self.B = fem.integrate(
            self.dispgrad_form,
            fields={"tau": self.constraint_test, "u": self.u_trial},
            output_dtype=float,
        )

        self.W = fem.integrate(
            tensor_mass_form,
            fields={"tau": self.constraint_test, "sig": self.constraint_trial},
            nodal=True,
            output_dtype=float,
        )
        self.W_inv = bsr_copy(self.W)
        fem_example_utils.invert_diagonal_bsr_matrix(self.W_inv)

        self.WiB = self.W_inv @ self.B
        self.BtWi = self.WiB.transpose()

    def newton_iter(self, k: int):
        # Unconstrained dynamics
        A, l = self.assemble_constraint_free_system()

        u_rhs = l
        bsr_mv(A=self.A, x=self.du_field.dof_values, y=u_rhs, alpha=-1.0, beta=1.0)

        u_matrix = bsr_copy(self.A)

        # Deformation gradient Fl
        Fk = fem.integrate(
            self.defgrad_form,
            quadrature=self.quadrature,
            fields={"u": self.u_field, "tau": self.constraint_test},
            output_dtype=wp.vec4,
        )

        # bsr_mv(self.W_inv, x=Fk, y=self.S.dof_values.view(dtype=wp.vec4), alpha=1.0, beta=0.0)

        # Rotated deformation gradient RFk and gradient matrix RBk
        Sk = fem.integrate(
            tensor_mass_form,
            quadrature=self.quadrature,
            fields={"sig": self.S, "tau": self.constraint_test},
            output_dtype=wp.vec4,
        )

        # c_k -- constraint residual (Fk - S)
        c_k = Fk
        array_axpy(x=Sk, y=c_k, alpha=-1.0, beta=1.0)

        # Elasticity -- use nodal integration so that H is block diagonal
        H = fem.integrate(
            self.elastic_hessian_form,
            quadrature=self.quadrature,
            fields={
                "S": self.S,
                "sig": self.constraint_test,
                "tau": self.constraint_trial,
            },
            values={"lame": self.lame},
            output_dtype=float,
        )
        f = fem.integrate(
            self.elastic_gradient_form,
            quadrature=self.quadrature,
            fields={"S": self.S, "tau": self.constraint_test},
            values={"lame": self.lame},
            output_dtype=wp.vec4,
        )

        # Schur complement

        # lambda_rhs = H W^-1 c_k + f
        lambda_rhs = f
        bsr_mv(A=H, x=bsr_mv(self.W_inv, c_k), y=lambda_rhs, alpha=1.0, beta=1.0)

        bsr_mv(A=self.BtWi, x=lambda_rhs, y=u_rhs, alpha=-1.0, beta=1.0)
        bsr_mm(x=self.BtWi, y=bsr_mm(H, self.WiB), z=u_matrix, alpha=1.0, beta=1.0)

        # Enforce boundary conditions
        du_bd_rhs = wp.clone(self.v_bd_rhs)
        bsr_mv(
            A=self.v_bd_matrix,
            x=self.du_field.dof_values,
            y=du_bd_rhs,
            alpha=-1.0,
            beta=1.0,
        )
        fem.project_linear_system(
            u_matrix, u_rhs, self.v_bd_matrix, du_bd_rhs, normalize_projector=False
        )

        delta_du = wp.zeros_like(u_rhs)
        fem_example_utils.bsr_cg(
            u_matrix, b=u_rhs, x=delta_du, tol=1.0e-16, max_iters=250, quiet=True
        )

        # update S
        # -Bk du  + W   dS       =  c_k
        # W  dS = c_k + RBK du
        bsr_mv(A=self.B, x=delta_du, y=c_k, alpha=1.0, beta=1.0)
        dS = bsr_mv(A=self.W_inv, x=c_k, alpha=1.0, beta=0.0)
        dS = dS.view(dtype=wp.mat22)

        dR = self.skew_test.space.make_field().dof_values

        return delta_du, dR, dS

    @integrand
    def constraint_residual(
        s: Sample,
        S: Field,
        R: Field,
        u: Field,
    ):
        """
        Constraint residual
        """
        c = S(s) - (grad(u, s) + wp.identity(n=2, dtype=float))
        return 0.5 * wp.ddot(c, c)

    @integrand
    def dispgrad_form(
        s: Sample,
        u: Field,
        tau: Field,
    ):
        """
        Displacement gradient form
        grad(u) : tau
        """
        return wp.ddot(tau(s), grad(u, s))

    @integrand
    def defgrad_form(
        s: Sample,
        u: Field,
        tau: Field,
    ):
        """
        Deformation gradient form
        (I + grad(u)) : tau
        """
        return wp.ddot(tau(s), (wp.identity(n=2, dtype=float) + grad(u, s)))


if __name__ == "__main__":
    # wp.config.verify_cuda = True
    # wp.config.verify_fp = True
    wp.init()
    wp.set_module_options({"enable_backward": False})
    wp.set_module_options({"max_unroll": 2})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--serendipity", action="store_true", default=False)
    parser.add_argument("--displacement", type=float, default=0.0)
    parser.add_argument("-n", "--n_frames", type=int, default=25)
    parser.add_argument("--n_newton", type=int, default=2)
    parser.add_argument("--n_backtrack", type=int, default=4)
    parser.add_argument("--young_modulus", type=float, default=100.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.5)
    parser.add_argument("--gravity", type=float, default=10.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--rot_compliance", type=float, default=0.001)
    parser.add_argument("-v", "--variant", type=str, default="rs_f")
    parser.add_argument("--grid", action="store_true", default=False)
    parser.add_argument("-nh", "--neo_hookean", action="store_true", default=False)
    args = parser.parse_args()

    if args.variant == "rs_f":
        sim = MFEM_RS_F(args)
    elif args.variant == "s_rf":
        sim = MFEM_S_RF(args)
    elif args.variant == "s_rf2":
        sim = MFEM_S_RF_v2(args)
    elif args.variant == "s_f":
        sim = MFEM_S_F(args)
    else:
        raise ValueError(f"Invalid variant: {args.variant}")

    sim.init_vel_space()
    sim.init_strain_spaces()
    sim.init_boundary_conditions()
    sim.init_constant_forms()

    plot = fem_example_utils.Plot()
    plot.add_field("u", sim.u_field)

    for f in range(args.n_frames):
        # TODO currently performing a single Newton iteration

        print(f"--- Frame {f} ---")
        sim.run_frame()
        plot.add_field("u", sim.u_field)

    plot.plot(
        {
            "u": {
                "displacement": {},
                "xlim": (-0.5, 1.5),
                "ylim": (-0.25, 1.25),
            },
        },
        backend="matplotlib",
    )
