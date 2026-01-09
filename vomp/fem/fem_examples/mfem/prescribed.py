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

from typing import Optional, Dict, Any

import warp as wp
import warp.fem as fem

from .softbody_sim import SoftbodySim


class QPBasedImplicitField(fem.field.GeometryField):
    """Same as fem.ImplicitField, but passes QP index instead of grid-space position"""

    def __init__(
        self,
        domain: fem.GeometryDomain,
        func: wp.Function,
        values: Optional[Dict[str, Any]] = None,
        degree=0,
    ):
        self.domain = domain
        self._degree = degree

        if not isinstance(func, wp.Function):
            raise ValueError(
                "Implicit field function must be a warp Function (decorated with `wp.func`)"
            )

        self._func = func

        argspec = fem.integrand(func.func).argspec
        arg_types = argspec.annotations

        qp_arg_type = arg_types.pop(argspec.args[0]) if arg_types else None
        if not qp_arg_type or not wp.types.types_equal(
            qp_arg_type,
            int,
            match_generic=True,
        ):
            raise ValueError(
                f"QP-based Implicit field function '{func.func.__name__}' must accept an index as its first argument"
            )

        pos_arg_type = arg_types.pop(argspec.args[1]) if arg_types else None
        if not pos_arg_type or not wp.types.types_equal(
            pos_arg_type,
            wp.vec3,
            match_generic=True,
        ):
            raise ValueError(
                f"QP-based Implicit field function '{func.func.__name__}' must accept a position as its second argument"
            )

        self.EvalArg = fem.cache.get_argument_struct(arg_types)
        self.values = values

        self.ElementEvalArg = self._make_element_eval_arg()
        self.eval_degree = self._make_eval_degree()

        self.eval_inner = self._make_eval_func(func)
        self.eval_outer = self.eval_inner

    @property
    def values(self):
        return self._func_arg

    @values.setter
    def values(self, v):
        self._func_arg = fem.cache.populate_argument_struct(
            self.EvalArg, v, self._func.func.__name__
        )

    @property
    def geometry(self):
        return self.domain.geometry

    @property
    def element_kind(self):
        return self.domain.element_kind

    def eval_arg_value(self, device):
        return self._func_arg

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def name(self) -> str:
        return f"Implicit_{self.domain.name}_{self.degree}_{self.EvalArg.key}"

    def _make_eval_func(self, func):
        if func is None:
            return None

        @fem.cache.dynamic_func(
            suffix=f"{self.name}_{func.key}",
            code_transformers=[
                fem.cache.ExpandStarredArgumentStruct({"args.eval_arg": self.EvalArg})
            ],
        )
        def eval_inner(args: self.ElementEvalArg, s: fem.Sample):
            return func(
                s.qp_index,
                self.domain.element_position(args.elt_arg, s),
                *args.eval_arg,
            )

        return eval_inner

    def _make_element_eval_arg(self):
        @fem.cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.domain.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_degree(self):
        ORDER = wp.constant(self._degree)

        @fem.cache.dynamic_func(suffix=self.name)
        def degree(args: self.ElementEvalArg):
            return ORDER

        return degree


class PrescribedMotion:
    def __init__(self, sim: SoftbodySim, quadrature: fem.PicQuadrature):
        self.sim = sim
        self.set_quadrature(quadrature)
        self._prescribed_pos_field = None
        self._prescribed_pos_weight_field = None

    def set_quadrature(self, quadrature: fem.PicQuadrature):
        self.quadrature = quadrature

    def set_prescribed_positions(
        self, pos_field: fem.field.GeometryField, weight_field: fem.field.GeometryField
    ):
        # for driving objects kinematically
        self._prescribed_pos_field = pos_field
        self._prescribed_pos_weight_field = weight_field

    def add_energy(self, E: float):
        if self._prescribed_pos_field:
            E += fem.integrate(
                prescribed_position_energy_form,
                quadrature=self.quadrature,
                fields={
                    "u_cur": self.sim.u_field,
                    "stiffness": self._prescribed_pos_weight_field,
                    "target": self._prescribed_pos_field,
                },
            )

        return E

    def add_hessian(self, lhs: wp.array):
        if self._prescribed_pos_field:
            z = fem.integrate(
                prescribed_position_lhs_form,
                quadrature=self.quadrature,
                fields={
                    "u": self.sim.u_trial,
                    "v": self.sim.u_test,
                    "stiffness": self._prescribed_pos_weight_field,
                },
                output_dtype=float,
            )
            lhs += z

        return lhs

    def add_forces(self, rhs: wp.array):
        if self._prescribed_pos_field:
            fem.integrate(
                prescribed_position_rhs_form,
                quadrature=self.quadrature,
                fields={
                    "u_cur": self.sim.u_field,
                    "v": self.sim.u_test,
                    "stiffness": self._prescribed_pos_weight_field,
                    "target": self._prescribed_pos_field,
                },
                output=rhs,
                add=True,
            )

        return rhs


@fem.integrand
def prescribed_position_lhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
    stiffness: fem.Field,
):
    u_displ = u(s)
    v_displ = v(s)

    return stiffness(s) * wp.dot(u_displ, v_displ)


@fem.integrand
def prescribed_position_rhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    v: fem.Field,
    stiffness: fem.Field,
    target: fem.Field,
):
    pos = u_cur(s) + domain(s)
    v_displ = v(s)
    target_pos = target(s)
    return stiffness(s) * wp.dot(target_pos - pos, v_displ)


@fem.integrand
def prescribed_position_energy_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    stiffness: fem.Field,
    target: fem.Field,
):
    pos = u_cur(s) + domain(s)
    target_pos = target(s)
    return 0.5 * stiffness(s) * wp.length_sq(pos - target_pos)
