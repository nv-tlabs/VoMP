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

import weakref
from typing import Any

import warp as wp
import warp.sparse as sp


class LineSearch:
    def __init__(self, sim):
        self.sim = weakref.proxy(sim)

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        pass

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        pass


class LineSearchNaiveCriterion(LineSearch):
    def __init__(self, sim):
        super().__init__(sim)
        self.penalty = sim.args.young_modulus

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        pass

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        f_cur = E_cur + self.penalty * C_cur
        f_ref = E_ref + self.penalty * C_ref
        return f_cur <= f_ref


class LineSearchUnconstrainedArmijoCriterion(LineSearch):
    def __init__(self, sim):
        super().__init__(sim)
        self.armijo_coeff = 0.0001

    def build_linear_model(self, lhs, rhs, delta_fields):
        (delta_u,) = delta_fields

        m = -wp.utils.array_inner(delta_u, self.sim._minus_dE_du.view(delta_u.dtype))
        self.m = m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        return E_cur <= E_ref + self.armijo_coeff * alpha * self.m


class LineSearchMeritCriterion(LineSearch):
    # Numeric Optimization, chapter 15.4

    def __init__(self, sim):
        super().__init__(sim)
        self.armijo_coeff = 0.0001

    def build_linear_model(self, lhs, rhs, delta_fields):
        delta_u, dS, dR, dLambda = delta_fields

        c_k = rhs[3]
        c_k_normalized = wp.empty_like(c_k)

        wp.launch(
            self._normalize_c_k,
            inputs=[c_k, c_k_normalized, self.sim._stiffness_field.dof_values],
            dim=c_k.shape,
        )

        delta_ck = lhs._B @ delta_u
        sp.bsr_mv(A=lhs._Cs, x=dS, y=delta_ck, alpha=-1.0, beta=1.0)

        if lhs._Cr is not None:
            sp.bsr_mv(A=lhs._Cr, x=dR, y=delta_ck, alpha=-1.0, beta=1.0)

        m = wp.utils.array_inner(dS, self.sim._dE_dS.view(dS.dtype)) - wp.utils.array_inner(
            delta_u, self.sim._minus_dE_du.view(delta_u.dtype)
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


class LineSearchMultiObjCriterion(LineSearch):
    # Line Search Filter Methods for Nonlinear Programming: Motivation and Global Convergence
    # 2005, SIAM Journal on Optimization 16(1):1-31

    def __init__(self, sim):
        super().__init__(sim)
        # constraint decrease
        E_scale = sim.typical_stiffness / sim.lame_ref[1]
        self.gamma_theta = 0.75
        self.gamma_f = 0.1 * E_scale

        # switching rule
        self.s_theta = 1.5
        self.s_rho = 2.5 * self.s_theta
        self.delta = 0.01 * E_scale ** (self.s_theta / self.s_rho)

        self.armijo_coeff = 0.0001

    def build_linear_model(self, lhs, rhs, delta_fields):
        delta_u, dS, dR, dLambda = delta_fields
        m = wp.utils.array_inner(dS, self.sim._dE_dS.view(dS.dtype)) - wp.utils.array_inner(
            delta_u, self.sim._minus_dE_du.view(delta_u.dtype)
        )
        self.m = m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        if self.m < 0.0 and (-self.m) ** self.s_rho * alpha > self.delta * C_ref**self.s_theta:
            return E_cur <= E_ref + self.armijo_coeff * alpha * self.m

        return C_cur <= (1.0 - self.gamma_theta) * C_ref or (E_cur <= E_ref - self.gamma_f * C_ref)


class LineSearchLagrangianArmijoCriterion(LineSearch):
    # Unconstrained line-search based on Lagrangian

    def __init__(self, sim):
        super().__init__(sim)
        self.armijo_coeff = 0.0001

    def build_linear_model(self, lhs, rhs, delta_fields):
        delta_u, dS, dR, dLambda = delta_fields

        m = wp.utils.array_inner(dS, self.sim._dE_dS.view(dS.dtype)) - wp.utils.array_inner(
            delta_u, self.sim._minus_dE_du.view(delta_u.dtype)
        )

        c_k = rhs[3]
        delta_ck = lhs._B @ delta_u
        sp.bsr_mv(A=lhs._Cs, x=dS, y=delta_ck, alpha=-1.0, beta=1.0)
        if lhs._Cr is not None:
            sp.bsr_mv(A=lhs._Cr, x=dR, y=delta_ck, alpha=-1.0, beta=1.0)

        c_m = wp.utils.array_inner(c_k, dLambda.view(c_k.dtype)) + wp.utils.array_inner(
            delta_ck, self.sim.constraint_field.dof_values.view(delta_ck.dtype)
        )
        self.m = m - c_m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        return E_cur + C_cur <= E_ref + C_ref + self.armijo_coeff * alpha * self.m
