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


class ProductShapeFunction(fem.space.shape.CubeShapeFunction):
    """Shape function defined as the product of two shapes
    N(x) = sum_i sum_j N0_i(x) N1_j(x)

    N0 is used to define the node topology (connectivity between cells),
    while N1 nodes are duplicated at each N0 node
    """

    def __init__(
        self,
        shape0: fem.space.ShapeFunction,
        shape1: fem.space.ShapeFunction,
        shape1_duplicates: int = 1,
    ):
        self._shape0 = shape0
        self._shape1 = shape1
        self._duplicates = shape1_duplicates

        assert shape0.value == fem.space.ShapeFunction.Value.Scalar

        self.ORDER = self._shape0.ORDER * self._shape1.ORDER
        duplicated_shape1_nodes = self._duplicates * self._shape1.NODES_PER_ELEMENT

        self.NODES_PER_ELEMENT = (
            self._shape0.NODES_PER_ELEMENT * duplicated_shape1_nodes
        )

        if isinstance(shape0, fem.space.shape.CubeShapeFunction):
            self.VERTEX_NODE_COUNT = (
                self._shape0.VERTEX_NODE_COUNT * duplicated_shape1_nodes
            )
            self.EDGE_NODE_COUNT = (
                self._shape0.EDGE_NODE_COUNT * duplicated_shape1_nodes
            )
            self.FACE_NODE_COUNT = (
                self._shape0.FACE_NODE_COUNT * duplicated_shape1_nodes
            )
            self.INTERIOR_NODE_COUNT = (
                self._shape0.INTERIOR_NODE_COUNT * duplicated_shape1_nodes
            )

        self.node_type_and_type_index = self._get_node_type_and_type_index()
        self.split_node_indices = self._get_split_node_indices()

    @property
    def value(self) -> fem.space.ShapeFunction.Value:
        return self._shape1.value

    @property
    def name(self):
        return f"{self._shape0.name}x{self._shape1.name}x{self._duplicates}"

    def _get_split_node_indices(self):
        N1_DUPLICATED_NODES = self._duplicates * self._shape1.NODES_PER_ELEMENT
        N_DUPLICATES = self._duplicates

        @fem.cache.dynamic_func(suffix=self.name)
        def split_node_indices(
            node_index_in_elt: int,
        ):
            n0_index = node_index_in_elt // N1_DUPLICATED_NODES
            n1_dup_index = node_index_in_elt - N1_DUPLICATED_NODES * n0_index
            n1_index = n1_dup_index // N_DUPLICATES

            return n0_index, n1_index

        return split_node_indices

    def _get_node_type_and_type_index(self):
        N1_DUPLICATED_NODES = self._duplicates * self._shape1.NODES_PER_ELEMENT

        @fem.cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            n0_index = node_index_in_elt // N1_DUPLICATED_NODES
            n1_dup_index = node_index_in_elt - N1_DUPLICATED_NODES * n0_index
            node_type, type_instance, type_index = (
                self._shape0.node_type_and_type_index(n0_index)
            )

            return (
                node_type,
                type_instance,
                (type_index * N1_DUPLICATED_NODES + n1_dup_index),
            )

        return node_type_and_index

    def make_element_inner_weight(self):
        n0_weight = self._shape0.make_element_inner_weight()
        n0_coords = self._shape0.make_node_coords_in_element()
        n1_weight = self._shape1.make_element_inner_weight()

        @fem.cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: fem.Coords,
            node_index_in_elt: int,
        ):
            n0_index, n1_index = self.split_node_indices(node_index_in_elt)
            n1_coords = coords + wp.vec3(0.5) - n0_coords(n0_index)

            clamped_coords = wp.min(wp.max(coords, wp.vec3(0.0)), wp.vec3(1.0))

            return n0_weight(clamped_coords, n0_index) * n1_weight(n1_coords, n1_index)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        n0_weight = self._shape0.make_element_inner_weight()
        n1_weight = self._shape1.make_element_inner_weight()
        n0_grad = self._shape0.make_element_inner_weight_gradient()
        n1_grad = self._shape1.make_element_inner_weight_gradient()
        n0_coords = self._shape0.make_node_coords_in_element()

        @fem.cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: fem.Coords,
            node_index_in_elt: int,
        ):
            n0_index, n1_index = self.split_node_indices(node_index_in_elt)
            n1_coords = coords + wp.vec3(0.5) - n0_coords(n0_index)

            clamped_coords = wp.min(wp.max(coords, wp.vec3(0.0)), wp.vec3(1.0))
            n0_grad_clamped = n0_grad(clamped_coords, n0_index)
            # Fix gradient for out-of cell coordinates
            for k in range(3):
                if coords[k] < 0 or coords[k] > 1:
                    n0_grad_clamped[k] = 0.0

            return n0_weight(clamped_coords, n0_index) * n1_grad(
                n1_coords, n1_index
            ) + n0_grad_clamped * n1_weight(n1_coords, n1_index)

        return element_inner_weight_gradient

    # boilerplate for nodal integration
    # (not used here)

    def make_node_coords_in_element(self):
        N1_DUPLICATED_NODES = self._shape1.NODES_PER_ELEMENT * self._duplicates
        n0_coords = self._shape0.make_node_coords_in_element()

        @fem.cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            n0_index = node_index_in_elt // N1_DUPLICATED_NODES
            return n0_coords(n0_index)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        N1_DUPLICATED_NODES = self._shape1.NODES_PER_ELEMENT * self._duplicates
        n0_weight = self._shape0.make_node_quadrature_weight()

        @fem.cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            n0_index = node_index_in_element // N1_DUPLICATED_NODES
            return n0_weight(n0_index) / float(N1_DUPLICATED_NODES)

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        N1_DUPLICATED_NODES = self._shape1.NODES_PER_ELEMENT * self._duplicates
        n0_trace_weight = self._shape0.make_trace_node_quadrature_weight()

        @fem.cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(node_index_in_element: int):
            n0_index = node_index_in_element // N1_DUPLICATED_NODES
            return n0_trace_weight(n0_index) / float(N1_DUPLICATED_NODES)

        return trace_node_quadrature_weight


class SmoothStepShapeFunction(fem.space.shape.CubeTripolynomialShapeFunctions):
    """Like trilinear, but with smooth step instead"""

    def __init__(
        self,
    ):
        super().__init__(degree=1, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    @property
    def name(self):
        return "SmoothStep"

    @wp.func
    def _smoothstep(x: float):
        t = 1.0 - wp.abs(x)
        # t2 = t * t
        # return 3.0 * t2 - 2.0 * t * t2
        # t = wp.abs(x)
        t3 = t * t * t
        return t3 * (t * (t * 6.0 - 15.0) + 10.0)

    @wp.func
    def _smoothstep_grad(x: float):
        t = wp.abs(x)
        # t2 = t * t
        # g = 6.0 * (t - t2)

        g = t * t * (3.0 * (t * (t * 6.0 - 15.0) + 10.0) + t * (12.0 * t - 15.0))

        return -wp.sign(x) * g

    def make_element_inner_weight(self):
        @fem.cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: fem.Coords,
            node_index_in_elt: int,
        ):
            v = self._vertex_coords_f(node_index_in_elt)
            off = coords - v

            wx = SmoothStepShapeFunction._smoothstep(off[0])
            wy = SmoothStepShapeFunction._smoothstep(off[1])
            wz = SmoothStepShapeFunction._smoothstep(off[2])

            return wx * wy * wz

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        @fem.cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: fem.Coords,
            node_index_in_elt: int,
        ):
            v = self._vertex_coords_f(node_index_in_elt)
            off = coords - v

            wx = SmoothStepShapeFunction._smoothstep(off[0])
            wy = SmoothStepShapeFunction._smoothstep(off[1])
            wz = SmoothStepShapeFunction._smoothstep(off[2])

            dx = SmoothStepShapeFunction._smoothstep_grad(off[0])
            dy = SmoothStepShapeFunction._smoothstep_grad(off[1])
            dz = SmoothStepShapeFunction._smoothstep_grad(off[2])

            return wp.vec3(dx * wy * wz, dy * wz * wx, dz * wx * wy)

        return element_inner_weight_gradient


class DuplicatedBasisSpace(fem.BasisSpace):
    """
    Basis space with duplicated nodes weighted by shape functions
    presampled at each quadrature point
    """

    @wp.struct
    class BasisArg:
        weights: wp.array3d(dtype=float)
        weight_gradients: wp.array3d(dtype=wp.vec3)
        subset_indices: wp.array(dtype=int)

    def __init__(
        self,
        topology: fem.SpaceTopology,
        shape: ProductShapeFunction,
        duplicate_count: int,
    ):
        super().__init__(topology)
        self._shape = shape
        self._duplicate_count = duplicate_count

        self._weights = None
        self._weight_gradients = None
        self._subset_indices = None

        self.ORDER = self._shape.ORDER

    @property
    def value(self) -> fem.space.shape.ShapeFunction.Value:
        return self._shape.value

    @property
    def name(self):
        return f"{self.topology.name}_{self._shape.name}"

    def basis_arg_value(self, device):
        args = self.BasisArg()
        args.weights = self._weights.to(device)
        args.weight_gradients = self._weight_gradients.to(device)
        args.subset_indices = (
            None if self._subset_indices is None else self._subset_indices.to(device)
        )

        return args

    def set_cached_qp_weights_and_gradients(
        self, weights: wp.array, weight_gradients: wp.array
    ):
        if not weights:
            weights = self._weights
        if not weight_gradients:
            weight_gradients = self._weight_gradients

        self._weights, prev_weights = weights, self._weights
        self._weight_gradients, prev_gradients = (
            weight_gradients,
            self._weight_gradients,
        )
        return prev_weights, prev_gradients

    def set_subset_indices(self, subset_indices: wp.array):
        self._subset_indices, prev_indices = subset_indices, self._subset_indices
        return prev_indices

    def make_element_inner_weight(self):
        shape_element_inner_weight = self._shape.make_element_inner_weight()
        DUPLICATE_COUNT = self._duplicate_count
        VERTEX_NODE_COUNT = DUPLICATE_COUNT * self._shape._shape1.NODES_PER_ELEMENT

        @fem.cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: fem.ElementIndex,
            coords: fem.Coords,
            node_index_in_elt: int,
            qp_index: fem.QuadraturePointIndex,
        ):
            duplicate_idx = node_index_in_elt % DUPLICATE_COUNT

            vertex_idx = node_index_in_elt // VERTEX_NODE_COUNT

            if basis_arg.subset_indices.shape[0] > 0:
                qp_index = basis_arg.subset_indices[qp_index]

            return basis_arg.weights[
                qp_index, vertex_idx, duplicate_idx
            ] * shape_element_inner_weight(coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        shape_element_inner_weight_gradient = (
            self._shape.make_element_inner_weight_gradient()
        )
        shape_element_inner_weight = self._shape.make_element_inner_weight()
        DUPLICATE_COUNT = self._duplicate_count
        VERTEX_NODE_COUNT = DUPLICATE_COUNT * self._shape._shape1.NODES_PER_ELEMENT

        @fem.cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: fem.ElementIndex,
            coords: fem.Coords,
            node_index_in_elt: int,
            qp_index: fem.QuadraturePointIndex,
        ):
            duplicate_idx = node_index_in_elt % DUPLICATE_COUNT

            vertex_idx = node_index_in_elt // VERTEX_NODE_COUNT

            if basis_arg.subset_indices.shape[0] > 0:
                qp_index = basis_arg.subset_indices[qp_index]

            return basis_arg.weights[
                qp_index, vertex_idx, duplicate_idx
            ] * shape_element_inner_weight_gradient(
                coords, node_index_in_elt
            ) + basis_arg.weight_gradients[
                qp_index, vertex_idx, duplicate_idx
            ] * shape_element_inner_weight(
                coords, node_index_in_elt
            )

        return element_inner_weight_gradient

    # Disable nodal integration

    def make_node_coords_in_element(self):
        return None

    def make_node_quadrature_weight(self):
        return None

    def make_trace_node_quadrature_weight(self, trace_basis):
        return None


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

        pos_arg_type = arg_types.pop(argspec.args[0]) if arg_types else None
        if not pos_arg_type or not wp.types.types_equal(
            pos_arg_type,
            int,
            match_generic=True,
        ):
            raise ValueError(
                f"QP-based Implicit field function '{func.func.__name__}' must accept an index as its first argument"
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
            return func(s.qp_index, *args.eval_arg)

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
