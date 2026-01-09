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

from typing import NamedTuple, Optional, Callable, List, Dict, Any
import gc

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.optim.linear import LinearOperator

import argparse
import polyscope as ps

from fem_examples.mfem.softbody_sim import ClassicFEM, defgrad
from fem_examples.simplicits.qp_basis_space import (
    DuplicatedBasisSpace,
    QPBasedImplicitField,
    ProductShapeFunction,
    SmoothStepShapeFunction,
)
from fem_examples.simplicits.collisions import (
    detect_ground_collisions,
    detect_particle_collisions,
    detect_mesh_collisions,
    compute_collision_bounds,
    collision_energy,
    collision_gradient_and_hessian,
    apply_collision_bounds,
)
from fem_examples.simplicits.linalg import (
    bsr_coarsen_aligned,
    bsr_mul_diag,
    create_batched_cholesky_kernel,
    create_batched_cholesky_solve_kernel,
    dense_chol_batched,
    dense_chol_subs_batched,
)


import warp.examples.fem.utils as fem_example_utils


class SparseBlendedSim(ClassicFEM):
    def __init__(
        self, geo: fem.Geometry, active_cells: wp.array, n_duplicates: int, args
    ):
        self._n_duplicates = n_duplicates

        self.warp_meshes = []

        self._prescribed_pos_weight_field = None
        self._prescribed_pos_field = None
        self._tiled_lhs = None

        super().__init__(geo, active_cells, args)

    @property
    def domain(self):
        return self.u_test.domain

    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        ClassicFEM.add_parser_arguments(parser)
        parser.add_argument(
            "--weight_degree",
            type=int,
            default=1,
            help="Degree of grid polynomial basis function (1 means trilinear, 2 triquadratic, etc)",
        )
        parser.add_argument(
            "--collision_stiffness",
            "-ck",
            type=float,
            default=1.0e3,
            help="Multiplier for collision force/energy",
        )
        parser.add_argument(
            "--collision_radius",
            "-cr",
            type=float,
            default=0.1,
            help="Radius of interaction for collision particles",
        )
        parser.add_argument(
            "--collision_detection_ratio",
            "-cd",
            type=float,
            default=1.25,
            help="Multiplier of collision radius for detection",
        )
        parser.add_argument(
            "--collision_barrier_ratio",
            "-cp",
            type=float,
            default=0.5,
            help="Fraction of collision radius that is non-penetrable",
        )
        parser.add_argument(
            "--self_immunity_radius_ratio",
            "-cs",
            type=float,
            default=4.0,
            help="Ignore self-collision for particles that were within this ratio at rest",
        )
        parser.add_argument("--friction", "-mu", type=float, default=0.2)
        parser.add_argument(
            "--friction_reg",
            "-mur",
            type=float,
            default=0.1,
            help="Regularization coefficient for IPC friction",
        )
        parser.add_argument(
            "--friction_fluid",
            "-nu",
            type=float,
            default=0.01,
            help="Additional fluid friction ratio to convexify things",
        )
        parser.add_argument(
            "--bounds",
            action=argparse.BooleanOptionalAction,
            help="Enforce non-penetration collision bounds",
        )
        parser.add_argument(
            "--admm_iterations",
            "-admm",
            type=int,
            default=0,
            help="Use ADMM solver instead of IPC. Discouraged.",
        )
        parser.add_argument(
            "--dual-grid",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Force running on primal grid instead of dual grid",
        )
        parser.add_argument(
            "--smoothstep",
            action=argparse.BooleanOptionalAction,
            help="Use tri-smoothstep grid functions instead of tri-polynomial",
        )
        parser.add_argument(
            "--precond_reg",
            "-preg",
            type=float,
            default=0.0001,
            help="preconditioner regularization",
        )
        parser.add_argument(
            "--direct",
            action=argparse.BooleanOptionalAction,
            help="Use (dense) direct solver for newton systems",
        )
        parser.add_argument(
            "--tiles",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use tile-based kernels. Requires recent warp built with libmathdx",
        )
        parser.add_argument(
            "--off-diagonal",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Include off-diagonal terms in contact matrix",
        )
        parser.add_argument(
            "--ground",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Do ground collisions",
        )
        parser.add_argument(
            "--ground_height",
            type=float,
            default=0.0,
            help="Ground height",
        )

    def set_quadrature(
        self,
        cell_indices,
        cell_coords,
        measures: wp.array,
        qp_obj_ids: wp.array,
        qp_stiffness_scales: wp.array,
        qp_node_weights: wp.array = None,
        qp_node_gradients: wp.array = None,
    ):
        self.cell_indices = cell_indices
        self.cell_coords = cell_coords

        quadrature = fem.PicQuadrature(
            domain=self.vel_quadrature.domain,
            positions=(self.cell_indices, self.cell_coords),
            measures=measures,
        )

        self.set_cached_basis_qp_weights(qp_node_weights, qp_node_gradients)

        self.vel_quadrature = quadrature
        self.elasticity_quadrature = quadrature

        self.qp_obj_ids = qp_obj_ids
        self.qp_stiff_scale = qp_stiffness_scales

        self.lame_field = QPBasedImplicitField(
            quadrature.domain,
            _lame_field,
            values={
                "lame_ref": self.lame_ref,
                "qp_stiff_scale": self.qp_stiff_scale,
            },
        )

    def set_cached_basis_qp_weights(
        self,
        qp_node_weights: wp.array = None,
        qp_node_gradients: wp.array = None,
    ):
        if isinstance(self._vel_basis, DuplicatedBasisSpace):
            return self._vel_basis.set_cached_qp_weights_and_gradients(
                qp_node_weights, qp_node_gradients
            )

        return None, None

    def _init_displacement_basis(self):
        # weights

        if self.args.smoothstep:
            assert self.args.weight_degree == 1
            shape0 = SmoothStepShapeFunction()
        else:
            shape0 = fem.space.shape.get_shape_function(
                self.geo.reference_cell(),
                self.geo.dimension,
                degree=self.args.weight_degree,
                element_basis=fem.ElementBasis.LAGRANGE,
            )

        # handles
        shape1 = fem.space.shape.get_shape_function(
            self.geo.reference_cell(),
            self.geo.dimension,
            self.args.degree,
            fem.ElementBasis.NONCONFORMING_POLYNOMIAL,
        )
        self._handle_size = shape1.NODES_PER_ELEMENT

        shape = ProductShapeFunction(
            shape0, shape1, shape1_duplicates=self._n_duplicates
        )

        if self.args.weight_degree == 0:
            topology = fem.space.topology.RegularDiscontinuousSpaceTopology(
                self.geo, shape.NODES_PER_ELEMENT
            )
        elif isinstance(self.geo.base, fem.Grid3D):
            topology = fem.space.make_grid_3d_space_topology(self.geo, shape)
        else:
            topology = fem.space.make_hexmesh_space_topology(self.geo, shape)

        if self._n_duplicates > 1:
            basis_space = DuplicatedBasisSpace(
                topology, shape, duplicate_count=self._n_duplicates
            )
        else:
            basis_space = fem.space.ShapeBasisSpace(topology, shape)

        self.set_displacement_basis(basis_space)

        self._use_tiles = self.args.tiles
        if self._use_tiles:
            self._tile_size = self._handle_size * 3 * self._n_duplicates
            self._use_tiles = self._tile_size > 6 and self._tile_size < 90

    def compute_initial_guess(self):
        self.du_field.dof_values.zero_()
        self.detect_collisions()

    def set_prescribed_positions(self, pos_field, weight_field):
        # for driving objects kinematically
        self._prescribed_pos_field = pos_field
        self._prescribed_pos_weight_field = weight_field

    def evaluate_energy(self):
        E_p, c_r = super().evaluate_energy()

        if self._prescribed_pos_field:
            E_p += fem.integrate(
                prescribed_position_energy_form,
                quadrature=self.vel_quadrature,
                fields={
                    "u_cur": self.u_field,
                    "stiffness": self._prescribed_pos_weight_field,
                    "target": self._prescribed_pos_field,
                },
            )

        if self.n_contact > 0:
            self._sample_cp_displacement(self.du_field, dest=self.cp_du)
            col_energies = wp.empty(self.n_contact, dtype=float)
            wp.launch(
                collision_energy,
                dim=self.n_contact,
                inputs=[
                    self.args.collision_radius,
                    self.args.collision_barrier_ratio,
                    self.args.friction,
                    self.args.dt * self.args.friction_reg,
                    self.args.friction_fluid * self.args.friction_reg,
                    self.cp_du,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b,
                    col_energies,
                ],
            )
            E_col = self._collision_stiffness * wp.utils.array_sum(col_energies)

            E_p += E_col

        return E_p, c_r

    def newton_lhs(self):
        lhs = super().newton_lhs()

        if self._prescribed_pos_field:
            z = fem.integrate(
                prescribed_position_lhs_form,
                quadrature=self.vel_quadrature,
                fields={
                    "u": self.u_trial,
                    "v": self.u_test,
                    "stiffness": self._prescribed_pos_weight_field,
                },
                output_dtype=float,
            )
            lhs += z

        # coarsen if requested
        if self._use_tiles:
            lhs = bsr_coarsen_aligned(
                lhs,
                block_shape=(self._tile_size, self._tile_size),
                coarse=self._tiled_lhs,
            )
            self._tiled_lhs = lhs

        # contacts
        if self.n_contact > 0 and not self.args.admm_iterations:
            H = self._collision_jacobian
            Ht = self._collision_jacobian_t
            wp.launch(
                bsr_mul_diag,
                dim=(Ht.nnz, Ht.block_shape[0]),
                inputs=[Ht.scalar_values, Ht.columns, self._col_energy_hessian],
            )

            sp.bsr_mm(
                x=Ht,
                y=H,
                z=lhs,
                alpha=self._collision_stiffness,
                beta=1.0,
                masked=not self.args.off_diagonal,
                work_arrays=self._HtH_work_arrays,
            )

        return lhs

    def newton_rhs(self, tape=None):
        rhs = super().newton_rhs(tape)

        if self._prescribed_pos_field:
            fem.integrate(
                prescribed_position_rhs_form,
                quadrature=self.vel_quadrature,
                fields={
                    "u_cur": self.u_field,
                    "v": self.u_test,
                    "stiffness": self._prescribed_pos_weight_field,
                    "target": self._prescribed_pos_field,
                },
                output=rhs,
                add=True,
            )

        # contacts
        if self.n_contact > 0 and not self.args.admm_iterations:
            sp.bsr_mv(
                A=self._collision_jacobian_t,
                x=self._col_energy_gradients,
                y=rhs,
                alpha=-self._collision_stiffness,
                beta=1.0,
            )

        return rhs

    def solve_newton_system(self, lhs, rhs):
        # from warp.tests.test_sparse import _bsr_to_dense
        # A = _bsr_to_dense(lhs)
        # print("COND::", np.linalg.cond(A))

        gc.collect(0)

        if self.n_contact > 0 and self.args.admm_iterations > 0:
            return self._solve_admm(lhs, rhs)

        if self.args.direct:
            # those imports should not be there
            # but ideally this file would be torch-free...
            import torch
            from simplicits import bsr_to_torch

            lhs = bsr_to_torch(lhs).to_dense()
            rhs = wp.to_torch(rhs).flatten()
            res = torch.linalg.solve(lhs, rhs)
            res = wp.from_torch(res.reshape(-1, 3), dtype=wp.vec3)
        else:
            # return super().solve_newton_system(lhs, rhs)

            res = wp.zeros_like(rhs)

            gc.disable()

            P = self._build_preconditioner(lhs)

            fem_example_utils.bsr_cg(
                A=lhs,
                b=rhs,
                x=res,
                quiet=True,
                M=P,
                use_diag_precond=False,
                tol=self.args.cg_tol,
                max_iters=self.args.cg_iters,
            )

            gc.enable()

        self._dof_bounds = wp.ones(res.shape, dtype=float)
        if self.n_contact > 0 and self.args.bounds:
            self._compute_bounds(res)

        return (res,)

    def _compute_bounds(self, delta_du: wp.array):
        # compute position increment for each quadrature point
        delta_du_field = fem.make_discrete_field(
            space=self.u_field.space, space_partition=self.u_field.space_partition
        )
        delta_du_field.dof_values = delta_du
        cp_delta_du = wp.empty_like(self.cp_du)
        self._sample_cp_displacement(delta_du_field, dest=cp_delta_du)

        # delta_gap = self._collision_jacobian @ delta_du

        wp.launch(
            compute_collision_bounds,
            dim=self.n_contact,
            inputs=[
                self.args.collision_radius,
                self.args.collision_barrier_ratio,
                self.cp_du,
                self.collision_kinematic_gaps,
                self.collision_normals,
                self.collision_indices_a,
                self.collision_indices_b,
                cp_delta_du,
                self._collision_jacobian_a.offsets,
                self._collision_jacobian_a.columns,
                self._collision_jacobian_b.offsets,
                self._collision_jacobian_b.columns,
                self._dof_bounds,
            ],
        )

    def apply_newton_deltas(self, delta_fields, alpha=1.0):
        # Restore checkpoint
        wp.copy(src=self._u_cur, dest=self.u_field.dof_values)
        wp.copy(src=self._du_cur, dest=self.du_field.dof_values)

        delta_du = delta_fields[0]

        wp.launch(
            apply_collision_bounds,
            dim=delta_du.shape,
            inputs=[
                delta_du,
                alpha,
                self._dof_bounds,
                self.u_field.dof_values,
                self.du_field.dof_values,
            ],
        )

    def _solve_admm(self, lhs, rhs):
        k = self._collision_stiffness
        H = self._collision_jacobian
        Ht = self._collision_jacobian.transpose()
        n_contact = self.n_contact

        lhs += k * (Ht @ H)
        P = self._build_preconditioner(lhs)

        gap_zero = self.collision_kinematic_gaps[:n_contact]
        sp.bsr_mv(A=H, x=self.du_field.dof_values, y=gap_zero, alpha=1.0, beta=1.0)
        gap = wp.clone(gap_zero)

        tau = wp.clone(gap)
        lbd = wp.zeros_like(gap_zero)

        rhs_i = wp.zeros_like(rhs)
        ddu = wp.zeros_like(rhs)

        for it in range(self.args.admm_iterations):
            # rhs_i = rhs + k Ht( gap - v + ldb)

            wp.copy(src=rhs, dest=rhs_i)
            sp.bsr_mv(A=Ht, x=gap, y=rhs_i, alpha=k, beta=1.0)
            sp.bsr_mv(A=Ht, x=tau, y=rhs_i, alpha=-k, beta=1.0)
            sp.bsr_mv(A=Ht, x=lbd, y=rhs_i, alpha=k, beta=1.0)

            fem_example_utils.bsr_cg(
                A=lhs,
                b=rhs_i,
                x=ddu,
                quiet=True,
                M=P,
                tol=self.args.cg_tol,
                max_iters=self.args.cg_iters,
            )

            # update gap = H ddu + gap_0
            wp.copy(src=gap_zero, dest=gap)
            sp.bsr_mv(A=H, x=ddu, y=gap, alpha=1.0, beta=1.0)

            # project on contacts
            wp.launch(
                SparseBlendedSim._solve_admm_contacts,
                dim=n_contact,
                inputs=[
                    self.args.collision_radius,
                    self.args.collision_barrier_ratio,
                    self.args.friction,
                    self.collision_normals,
                    gap_zero,
                    gap,
                    tau,
                    lbd,
                ],
            )

            print(it, np.linalg.norm(gap.numpy() - tau.numpy()))

        if self.args.bounds:
            self._enforce_bounds(ddu)

        return (ddu,)

    @wp.kernel
    def _solve_admm_contacts(
        rc: float,
        rp_ratio: float,
        mu: float,
        normals: wp.array(dtype=wp.vec3),
        gaps0: wp.array(dtype=wp.vec3),
        gaps: wp.array(dtype=wp.vec3),
        taus: wp.array(dtype=wp.vec3),
        lbds: wp.array(dtype=wp.vec3),
    ):
        c = wp.tid()
        nor = normals[c]

        # rp = wp.min(rc * rp_ratio, wp.max(0.0, wp.dot(gaps0[c], nor)))
        rp = rc * rp_ratio

        gap = gaps[c] - rp * nor
        guess = gap - lbds[c]

        # solve coulomb
        un = wp.dot(guess, nor)
        if un < 0.0:
            guess -= un * nor
            uTn = wp.length_sq(guess)
            alpha = mu * un
            if uTn <= alpha * alpha:
                guess = wp.vec3(0.0)
            else:
                guess *= 1.0 + mu * un / wp.sqrt(uTn)

        taus[c] = guess + rp * nor
        lbds[c] += guess - gap

    def _build_preconditioner(self, lhs):
        typical_inertia = self.typical_length**3 * self.args.density / self.args.dt**2
        p_reg = self.args.precond_reg * typical_inertia
        if p_reg < 0:
            # disable preconditioner
            return None

        # Block-diagonal preconditioner with handle-sized blocks (12x12)
        use_tiles = self._use_tiles
        if use_tiles:
            block_size = self._tile_size
        else:
            block_size = self._handle_size * 3

        block_type = wp.mat((block_size, block_size), dtype=float)

        P_coarse = sp.bsr_diag(
            rows_of_blocks=lhs.shape[0] // block_size,
            block_type=block_type,
        )
        sp.bsr_assign(src=lhs, dest=P_coarse, masked=True)

        if use_tiles:

            def _as_float_array(x):
                return wp.array(
                    ptr=x.ptr, shape=(x.shape[0] * 3), device=x.device, dtype=float
                )

            has_tile_chol = "tile_cholesky" in wp.context.builtin_functions

            if not has_tile_chol:
                P_values = P_coarse.scalar_values.reshape(
                    (P_coarse.nrow, block_size * block_size)
                )
                L = wp.empty_like(P_values)

                wp.launch(
                    dense_chol_batched,
                    dim=(P_coarse.nrow),
                    inputs=[block_size, p_reg, P_values, L],
                )

                def P_inv_mv(x, y, z, alpha, beta):
                    # for cg, y = z, alpha = 1, beta = 0
                    x = _as_float_array(x).reshape((P_coarse.nrow, block_size))
                    z = _as_float_array(z).reshape((P_coarse.nrow, block_size))
                    wp.launch(
                        dense_chol_subs_batched,
                        dim=(P_coarse.nrow),
                        inputs=[x, z, L],
                    )

                return LinearOperator(
                    P_coarse.shape, P_coarse.dtype, P_coarse.device, P_inv_mv
                )

            P_values = P_coarse.scalar_values
            tile_chol = create_batched_cholesky_kernel(block_size)
            tile_solve = create_batched_cholesky_solve_kernel(block_size)

            BLOCK_DIM = 128

            wp.launch(
                tile_chol,
                dim=(P_coarse.nrow, BLOCK_DIM),
                inputs=[P_values, p_reg],
                block_dim=BLOCK_DIM,
            )

            def P_inv_mv(x, y, z, alpha, beta):
                # for cg, y = z, alpha = 1, beta = 0
                x = _as_float_array(x)
                z = _as_float_array(z)
                wp.launch(
                    tile_solve,
                    dim=(P_coarse.nrow, BLOCK_DIM),
                    inputs=[P_values, x, z],
                    block_dim=BLOCK_DIM,
                )

            return LinearOperator(
                P_coarse.shape, P_coarse.dtype, P_coarse.device, P_inv_mv
            )

        P_coarse += p_reg * sp.bsr_diag(
            block_type(np.eye(block_size)),
            rows_of_blocks=P_coarse.nrow,
            cols_of_blocks=P_coarse.ncol,
        )
        fem_example_utils.invert_diagonal_bsr_matrix(P_coarse)

        return sp.bsr_copy(P_coarse, block_shape=lhs.block_shape)

    def prepare_newton_step(self, tape=None):
        self.detect_collisions()
        self.build_collision_jacobian()

        # compute per-contact forces and hessian

        n_contact = self.n_contact
        if n_contact > 0 and not self.args.admm_iterations:
            self._col_energy_gradients = wp.empty(n_contact, dtype=wp.vec3)
            self._col_energy_hessian = wp.empty(n_contact, dtype=wp.mat33)

            wp.launch(
                collision_gradient_and_hessian,
                dim=n_contact,
                inputs=[
                    self.args.collision_radius,
                    self.args.collision_barrier_ratio,
                    self.args.friction,
                    self.args.dt * self.args.friction_reg,
                    self.args.friction_fluid * self.args.friction_reg,
                    self.cp_du,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b,
                    self._col_energy_gradients,
                    self._col_energy_hessian,
                ],
            )

        return super().prepare_newton_step(tape)

    def init_collision_detector(
        self,
        kinematic_meshes: List[wp.Mesh],
        cell_indices,
        cell_coords,
        cp_obj_ids: wp.array,
        cp_node_weights: wp.array = None,
        cp_node_gradients: wp.array = None,
    ):
        self.warp_meshes = kinematic_meshes

        self.cp_cell_indices = cell_indices
        self.cp_cell_coords = cell_coords

        n_cp = self.cp_cell_indices.shape[0]

        self.collision_quadrature = fem.PicQuadrature(
            domain=self.vel_quadrature.domain,
            positions=(self.cp_cell_indices, self.cp_cell_coords),
            measures=wp.ones(n_cp, dtype=float),
        )

        self.cp_obj_ids = cp_obj_ids
        self._cp_node_weights = cp_node_weights
        self._cp_node_gradients = cp_node_gradients
        self.n_contact = 0

        self._hashgrid = wp.HashGrid(128, 128, 128)
        n_cp = self.collision_quadrature.total_point_count()

        self.cp_du = wp.empty(n_cp, dtype=wp.vec3)
        self.cp_rest_pos = self.cp_world_position()

        max_contacts = 10 * n_cp
        self.collision_indices_a = wp.empty(max_contacts, dtype=int)
        self.collision_indices_b = wp.empty(max_contacts, dtype=int)
        self.collision_normals = wp.empty(max_contacts, dtype=wp.vec3)
        self.collision_kinematic_gaps = wp.empty(max_contacts, dtype=wp.vec3)

        jac_cols = self.u_field.space_partition.node_count()
        self._collision_jacobian_a = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
        self._collision_jacobian_b = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)

        if self._use_tiles:
            jac_cols = (jac_cols * 3) // self._tile_size
            self._collision_jacobian = sp.bsr_zeros(
                0, jac_cols, block_type=wp.mat((3, self._tile_size), dtype=float)
            )
            self._collision_jacobian_t = sp.bsr_zeros(
                jac_cols, 0, block_type=wp.mat((self._tile_size, 3), dtype=float)
            )
        else:
            self._collision_jacobian = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
            self._collision_jacobian_t = sp.bsr_zeros(jac_cols, 0, block_type=wp.mat33)

        self._HtH_work_arrays = sp.bsr_mm_work_arrays()
        self._HbHa_work_arrays = sp.bsr_axpy_work_arrays()

        # auto-scale with mass, but keep backward compat with old default
        self._collision_stiffness = (
            self.args.collision_stiffness * self.args.density / 1000.0
        )

    def qp_world_position(self):
        qp_pic = self.vel_quadrature
        qp_cur_pos = wp.empty(qp_pic.total_point_count(), dtype=wp.vec3)
        fem.interpolate(
            world_position,
            fields={"u": self.u_field},
            dest=qp_cur_pos,
            quadrature=qp_pic,
        )

        return qp_cur_pos

    def cp_world_position(self):
        cp_pic = self.collision_quadrature
        cp_cur_pos = wp.empty_like(self.cp_du)
        with ScopedCachedBasisWeights(
            self, self._cp_node_weights, self._cp_node_gradients
        ):
            fem.interpolate(
                world_position,
                fields={"u": self.u_field},
                dest=cp_cur_pos,
                quadrature=cp_pic,
            )

        return cp_cur_pos

    def _sample_cp_displacement(self, du_field, dest):
        cp_pic = self.collision_quadrature
        with ScopedCachedBasisWeights(
            self, self._cp_node_weights, self._cp_node_gradients
        ):
            fem.interpolate(
                du_field,
                dest=dest,
                quadrature=cp_pic,
            )

    def detect_collisions(self):
        cp_pic = self.collision_quadrature
        n_cp = cp_pic.total_point_count()

        cp_cur_pos = self.cp_world_position()
        self._sample_cp_displacement(self.du_field, self.cp_du)

        count = wp.zeros(1, dtype=int)
        max_contacts = self.collision_normals.shape[0]

        indices_a = self.collision_indices_a
        indices_b = self.collision_indices_b
        normals = self.collision_normals
        kinematic_gaps = self.collision_kinematic_gaps

        collision_radius = (
            self.args.collision_radius * self.args.collision_detection_ratio
        )

        if self.args.ground:
            ground_height = self.args.ground_height
            wp.launch(
                detect_ground_collisions,
                dim=n_cp,
                inputs=[
                    max_contacts,
                    self.args.up_axis,
                    cp_cur_pos,
                    self.cp_du,
                    collision_radius,
                    ground_height,
                    count,
                    normals,
                    kinematic_gaps,
                    indices_a,
                    indices_b,
                ],
            )

        self_collision_immune_radius = (
            self.args.self_immunity_radius_ratio * collision_radius
        )

        self._hashgrid.build(cp_cur_pos, radius=2.0 * collision_radius)
        wp.launch(
            detect_particle_collisions,
            dim=n_cp,
            inputs=[
                max_contacts,
                self._hashgrid.id,
                2.0 * collision_radius,
                self_collision_immune_radius,
                cp_cur_pos,
                self.cp_rest_pos,
                self.cp_du,
                self.cp_obj_ids,
                count,
                normals,
                kinematic_gaps,
                indices_a,
                indices_b,
            ],
        )

        if self.warp_meshes:
            mesh_ids = wp.array([mesh.id for mesh in self.warp_meshes], dtype=wp.uint64)
            wp.launch(
                detect_mesh_collisions,
                dim=(len(mesh_ids), n_cp),
                inputs=[
                    max_contacts,
                    mesh_ids,
                    cp_cur_pos,
                    self.cp_du,
                    collision_radius,
                    count,
                    normals,
                    kinematic_gaps,
                    indices_a,
                    indices_b,
                ],
            )

        self.n_contact = int(count.numpy()[0])

        if self.n_contact > max_contacts:
            print("Warning: contact buffer size exceeded, some have bee ignored")
            self.n_contact = max_contacts

    def build_collision_jacobian(self):
        n_contact = self.n_contact

        # Build collision jacobian
        # (derivative of collision gap `pos_a - pos_b` w.r.t. degrees of freedom)

        if n_contact == 0:
            return

        a_cells = wp.empty(n_contact, dtype=int)
        a_coords = wp.empty(n_contact, dtype=wp.vec3)
        b_cells = wp.empty(n_contact, dtype=int)
        b_coords = wp.empty(n_contact, dtype=wp.vec3)
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.cp_cell_indices,
                self.cp_cell_coords,
                self.collision_indices_a,
                a_cells,
                a_coords,
            ],
        )
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.cp_cell_indices,
                self.cp_cell_coords,
                self.collision_indices_b,
                b_cells,
                b_coords,
            ],
        )

        measures = wp.ones(n_contact, dtype=float)

        a_contact_pic = fem.PicQuadrature(
            self.collision_quadrature.domain,
            positions=(a_cells, a_coords),
            measures=measures,
        )
        b_contact_pic = fem.PicQuadrature(
            self.collision_quadrature.domain,
            positions=(b_cells, b_coords),
            measures=measures,
        )
        u_trial = fem.make_trial(
            self.u_field.space, space_partition=self.u_field.space_partition
        )

        with ScopedCachedBasisWeights(
            self, self._cp_node_weights, self._cp_node_gradients
        ):
            if isinstance(self._vel_basis, DuplicatedBasisSpace):
                self._vel_basis.set_subset_indices(self.collision_indices_a)

            sp.bsr_set_zero(
                self._collision_jacobian_a,
                n_contact,
                self.u_field.space_partition.node_count(),
            )
            fem.interpolate(
                u_trial,
                quadrature=a_contact_pic,
                dest=self._collision_jacobian_a,
                bsr_options={"prune_numerical_zeros": False},
            )

            sp.bsr_set_zero(
                self._collision_jacobian_b,
                n_contact,
                self.u_field.space_partition.node_count(),
            )
            if isinstance(self._vel_basis, DuplicatedBasisSpace):
                self._vel_basis.set_subset_indices(self.collision_indices_b)
            fem.interpolate(
                u_trial,
                quadrature=b_contact_pic,
                dest=self._collision_jacobian_b,
                bsr_options={"prune_numerical_zeros": False},
            )

            if isinstance(self._vel_basis, DuplicatedBasisSpace):
                self._vel_basis.set_subset_indices(None)

        gc.collect(1)
        self._collision_jacobian_a.nnz_sync()
        self._collision_jacobian_b.nnz_sync()

        if self._use_tiles:
            H = bsr_coarsen_aligned(
                self._collision_jacobian_a,
                block_shape=(3, self._tile_size),
                coarse=self._collision_jacobian,
            )
            del self._collision_jacobian_a.values
            Hb = bsr_coarsen_aligned(
                self._collision_jacobian_b,
                block_shape=(3, self._tile_size),
            )
            del self._collision_jacobian_b.values
            sp.bsr_axpy(
                x=Hb,
                y=H,
                alpha=-1,
                beta=1,
                work_arrays=self._HbHa_work_arrays,
            )
            del Hb

            self._collision_jacobian = H

        else:
            sp.bsr_assign(self._collision_jacobian, src=self._collision_jacobian_a)
            del self._collision_jacobian_a.values
            sp.bsr_axpy(
                x=self._collision_jacobian_b,
                y=self._collision_jacobian,
                alpha=-1,
                beta=1,
                work_arrays=self._HbHa_work_arrays,
            )
            del self._collision_jacobian_b.values

        # we no longer need a,b, values, just the topology
        # garbag-collect value arrays
        self._collision_jacobian_a.values = wp.empty(0, dtype=wp.mat33)
        self._collision_jacobian_b.values = wp.empty(0, dtype=wp.mat33)

        gc.collect(0)
        self._collision_jacobian.nnz_sync()
        sp.bsr_set_transpose(
            dest=self._collision_jacobian_t, src=self._collision_jacobian
        )


class ScopedCachedBasisWeights:
    def __init__(self, sim: SparseBlendedSim, weights, grad_weights):
        self._sim = sim
        self._weights = weights
        self._grad_weights = grad_weights

    def __enter__(self):
        self._weights, self._grad_weights = self._sim.set_cached_basis_qp_weights(
            self._weights, self._grad_weights
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self._weights, self._grad_weights = self._sim.set_cached_basis_qp_weights(
            self._weights, self._grad_weights
        )


@wp.func
def _lame_field(
    qp_index: int,
    lame_ref: wp.vec2,
    qp_stiff_scale: wp.array(dtype=float),
):
    return lame_ref * qp_stiff_scale[qp_index]


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
    pos = world_position(s, domain, u_cur)
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
    pos = world_position(s, domain, u_cur)
    target_pos = target(s)
    return 0.5 * stiffness(s) * wp.length_sq(pos - target_pos)


@wp.kernel
def gather_cell_coordinates(
    qp_cells: wp.array(dtype=int),
    qp_coords: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    cells: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    qp = indices[i]

    if qp == fem.NULL_QP_INDEX:
        cells[i] = fem.NULL_ELEMENT_INDEX
    else:
        cells[i] = qp_cells[qp]
        coords[i] = qp_coords[qp]


@fem.integrand
def world_position(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
):
    return domain(s) + u(s)


class SparseBlendedScene:
    def __init__(self, args):
        self.args = args
        self.objects = []
        self.kinematic_meshes = []

        self._obj_vertex_offsets = []

        self._vertices = None
        self._vertex_Fs = None
        self._vertex_quadrature = None
        self._vtx_weights = None
        self._vtx_weight_gradients = None

        # fem.set_default_temporary_store(fem.TemporaryStore())

    class Object(NamedTuple):
        origin: wp.vec3
        rotation: wp.quat
        scale: float
        qps: Optional[wp.array] = None
        cps: Optional[wp.array] = None
        vertices: Optional[wp.array] = None
        vertex_Fs: Optional[wp.array] = None
        weight_eval_fn: Optional[Callable] = None
        stiffness_eval_fn: Optional[Callable] = None
        input_bbox: np.array = None
        vol_fraction: float = 1.0

    def add_object(
        self,
        origin,
        rotation=None,
        scale=None,
        qps=None,
        cps=None,
        vertices=None,
        input_bbox=None,
        weight_eval_fn=None,
        stiffness_eval_fn=None,
        vol_fraction=1.0,
    ):
        if vertices is None:
            vertices = np.empty(shape=(0, 3))
        if rotation is None:
            rotation = wp.quat_identity(dtype=float)

        if scale is None:
            scale = wp.vec3(1.0)
        else:
            scale = wp.vec3(scale)

        if qps is None:
            qps = np.random.rand(self.args.n_qp, 3)

        if cps is None:
            cps = qps

        if input_bbox is None:
            all_pts = np.vstack((qps, cps, vertices))
            input_bbox = np.array([np.min(all_pts, axis=0), np.max(all_pts, axis=0)])

        scale = wp.cw_mul(scale, wp.vec3(input_bbox[1] - input_bbox[0]))

        if stiffness_eval_fn is None:
            stiffness_eval_fn = lambda pts: np.ones(pts.shape[0])

        self.objects.append(
            SparseBlendedScene.Object(
                origin=origin,
                rotation=rotation,
                scale=scale,
                qps=qps,
                cps=cps,
                weight_eval_fn=weight_eval_fn,
                stiffness_eval_fn=stiffness_eval_fn,
                vertices=vertices,
                vertex_Fs=np.empty((vertices.shape[0], 3, 3)),
                input_bbox=input_bbox,
                vol_fraction=vol_fraction,
            )
        )

    def add_kinematic_mesh(self, mesh: wp.Mesh):
        self.kinematic_meshes.append(mesh)

    @wp.kernel
    def _world_position(
        vtx_obj_ids: wp.array(dtype=int),
        vtx_pos: wp.array(dtype=wp.vec3),
        obj_pos: wp.array(dtype=wp.vec3),
        obj_rot: wp.array(dtype=wp.quatf),
        obj_scale: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        oid = vtx_obj_ids[i]
        loc_pos = vtx_pos[i]
        vtx_pos[i] = (
            wp.quat_rotate(obj_rot[oid], wp.cw_mul(loc_pos, obj_scale[oid]))
            + obj_pos[oid]
        )

    @fem.integrand
    def _assign_to_cells(
        domain: fem.Domain,
        s: fem.Sample,
        points: wp.array(dtype=wp.vec3),
        point_obj_ids: wp.array(dtype=int),
        cell_obj_ids: wp.array(dtype=int),
        cell_coords: wp.array(dtype=wp.vec3),
        cell_indices: wp.array(dtype=int),
    ):
        pos = points[s.qp_index]
        obj_id = point_obj_ids[s.qp_index]
        max_dist = 1.0
        s_proj = fem.lookup(domain, pos, max_dist, cell_obj_ids, obj_id)

        cell_indices[s.qp_index] = s_proj.element_index
        cell_coords[s.qp_index] = fem.element_coordinates(
            domain, s_proj.element_index, pos
        )

    def _merged_world_space_points(self, per_object_points: List[np.array]):
        n_obj = len(self.objects)

        points = np.vstack(per_object_points)
        points = wp.array(points, dtype=wp.vec3)

        obj_n_points = np.array([pts.shape[0] for pts in per_object_points])
        obj_ids = wp.array(np.arange(n_obj).repeat(obj_n_points), dtype=int)
        obj_offsets = np.concatenate(([0], np.cumsum(obj_n_points)), dtype=int)

        self._transform_points(points, obj_ids)

        return points, obj_ids, obj_offsets

    def _transform_points(self, points, obj_ids):
        obj_pos = wp.array([obj.origin for obj in self.objects], dtype=wp.vec3)
        obj_scales = wp.array([obj.scale for obj in self.objects], dtype=wp.vec3)
        obj_rot = wp.array([obj.rotation for obj in self.objects], dtype=wp.quatf)

        wp.launch(
            SparseBlendedScene._world_position,
            dim=points.shape,
            inputs=[obj_ids, points, obj_pos, obj_rot, obj_scales],
        )

    def _gather_world_space_points(self, points_lambda: Callable):
        points = []
        for obj in self.objects:
            obj_extent = obj.input_bbox[1] - obj.input_bbox[0]
            points.append((points_lambda(obj) - obj.input_bbox[0]) / obj_extent)

        return self._merged_world_space_points(points)

    def _build_grid(self):
        res = wp.vec3i(self.args.res)

        grid_vtx = []
        grid_cells = []

        vtx_obj_ids = []
        cell_obj_ids = []

        # Build object grids
        if self.args.dual_grid:
            bounds_lo = wp.vec3(0.5 / (self.args.res + 1))
            bounds_hi = wp.vec3(1.0 - 0.5 / (self.args.res + 1))
        else:
            bounds_lo = wp.vec3(0.0)
            bounds_hi = wp.vec3(1.0)

        obj_vtx_offset = 0
        for oid, obj in enumerate(self.objects):
            obj_grid_vtx, obj_grid_cells = fem_example_utils.gen_hexmesh(
                res, bounds_lo, bounds_hi
            )
            obj_grid_vtx = obj_grid_vtx.numpy()
            obj_grid_cells = obj_grid_cells.numpy()

            obj_grid_cells += obj_vtx_offset

            grid_vtx.append(obj_grid_vtx)
            grid_cells.append(obj_grid_cells)

            cell_obj_ids.append(np.full(obj_grid_cells.shape[0], fill_value=oid))
            obj_vtx_offset += obj_grid_vtx.shape[0]

        grid_cells = wp.array(np.vstack(grid_cells), dtype=int)
        cell_obj_ids = wp.array(np.concatenate(cell_obj_ids), dtype=int)

        grid_vtx, vtx_obj_ids, vtx_obj_offsets = self._merged_world_space_points(
            grid_vtx
        )

        geo = fem.Hexmesh(
            grid_cells, grid_vtx, assume_parallelepiped_cells=True, build_bvh=True
        )

        return geo, cell_obj_ids, vtx_obj_ids, vtx_obj_offsets

    def _embed_points(self, geo, cell_obj_ids, points, point_obj_ids):
        domain = fem.Cells(geo)

        qp_cell_indices = wp.empty_like(point_obj_ids)
        qp_cell_coords = wp.empty_like(points)
        fem.interpolate(
            SparseBlendedScene._assign_to_cells,
            domain=domain,
            dim=len(points),
            values={
                "points": points,
                "point_obj_ids": point_obj_ids,
                "cell_obj_ids": cell_obj_ids,
                "cell_coords": qp_cell_coords,
                "cell_indices": qp_cell_indices,
            },
        )

        return qp_cell_indices, qp_cell_coords

    def make_sim(self) -> SparseBlendedSim:
        n_obj = len(self.objects)
        if n_obj == 0:
            return None

        geo, cell_obj_ids, grid_node_obj_ids, obj_grid_node_offsets = self._build_grid()

        # quadrature points
        qps, qp_obj_ids, self._obj_qp_offsets = self._gather_world_space_points(
            lambda obj: obj.qps
        )
        qp_cell_indices, qp_cell_coords = self._embed_points(
            geo, cell_obj_ids, qps, qp_obj_ids
        )
        (
            weights,
            weight_gradients,
        ) = self._eval_point_weights(
            geo, qps, qp_cell_indices, qp_cell_coords, qp_obj_ids, self._obj_qp_offsets
        )

        n_duplicates = weights.shape[-1] if weights is not None else 1
        print(f"Using {n_duplicates} handles-per-cell")

        sim = SparseBlendedSim(
            geo=geo,
            args=self.args,
            active_cells=None,
            n_duplicates=n_duplicates,
        )
        sim.init_displacement_space()
        sim.init_strain_spaces()

        # Use our own quadrature points instead of regular ones
        obj_qp_count = self._obj_qp_offsets[1:] - self._obj_qp_offsets[:-1]
        qp_measures = (
            np.array([obj.vol_fraction for obj in self.objects]) / obj_qp_count
        )
        self.qp_measures = wp.array(
            np.repeat(qp_measures, obj_qp_count),
            dtype=float,
        )

        qp_stiffness_scales = wp.ones(qps.shape[0], dtype=float)
        self._eval_point_stiffness(geo, qps, self._obj_qp_offsets, qp_stiffness_scales)

        sim.set_quadrature(
            qp_cell_indices,
            qp_cell_coords,
            self.qp_measures,
            qp_obj_ids,
            qp_stiffness_scales,
            weights,
            weight_gradients,
        )

        # Disable fixed points
        sim.set_boundary_condition(boundary_projector_form=None)

        # collision particles
        cps, cp_obj_ids, self._obj_cp_offsets = self._gather_world_space_points(
            lambda obj: obj.cps
        )
        cp_cell_indices, cp_cell_coords = self._embed_points(
            geo, cell_obj_ids, cps, cp_obj_ids
        )
        (
            cp_weights,
            cp_weight_gradients,
        ) = self._eval_point_weights(
            geo, cps, cp_cell_indices, cp_cell_coords, cp_obj_ids, self._obj_cp_offsets
        )

        sim.init_collision_detector(
            self.kinematic_meshes,
            cp_cell_indices,
            cp_cell_coords,
            cp_obj_ids,
            cp_weights,
            cp_weight_gradients,
        )

        # visualization points
        # Do the same thing as for quadrature points:
        # assign to cells, evaluate wieghts and gradients, create quadrature

        self._vertices, self._vtx_obj_ids, self._obj_vtx_offsets = (
            self._gather_world_space_points(lambda obj: obj.vertices)
        )
        vtx_cell_indices, vtx_cell_coords = self._embed_points(
            geo, cell_obj_ids, self._vertices, self._vtx_obj_ids
        )
        (
            self._vtx_weights,
            self._vtx_weight_gradients,
        ) = self._eval_point_weights(
            geo,
            self._vertices,
            vtx_cell_indices,
            vtx_cell_coords,
            self._vtx_obj_ids,
            self._obj_vtx_offsets,
        )
        measures = wp.ones(vtx_cell_indices.shape[0], dtype=float)
        self._vertex_quadrature = fem.PicQuadrature(
            fem.Cells(geo),
            positions=(vtx_cell_indices, vtx_cell_coords),
            measures=measures,
        )
        self._vertex_Fs = wp.zeros((self._vertices.shape[0]), dtype=wp.mat33)

        # grid vertices, again for visualization

        grid_cell_indices, grid_cell_coords = self._embed_points(
            geo, cell_obj_ids, geo.positions, grid_node_obj_ids
        )

        (
            self._grid_weights,
            self._grid_weight_gradients,
        ) = self._eval_point_weights(
            geo,
            geo.positions,
            grid_cell_indices,
            grid_cell_coords,
            grid_node_obj_ids,
            obj_grid_node_offsets,
        )

        measures = wp.ones(grid_cell_indices.shape[0], dtype=float)
        self._grid_quadrature = fem.PicQuadrature(
            fem.Cells(geo),
            positions=(grid_cell_indices, grid_cell_coords),
            measures=measures,
        )

        return sim

    def cell_node_indices(self, sim: SparseBlendedSim):
        # Save cell node indices; useful for feature network evaluation
        cell_node_indices = sim.u_field.space.topology.element_node_indices()
        node_size = sim._handle_size * sim._n_duplicates
        node_count = cell_node_indices.shape[1] // node_size

        cell_node_indices_3d = wp.empty(
            (cell_node_indices.shape[0], node_count), dtype=wp.vec3i
        )

        wp.launch(
            _extract_3d_node_indices,
            dim=cell_node_indices_3d.shape,
            inputs=[
                node_size,
                sim.geo.res,
                cell_node_indices,
                cell_node_indices_3d,
            ],
        )

        return cell_node_indices_3d

    def _eval_point_weights(
        self,
        geo: fem.Grid3D,
        points: wp.array,
        cell_indices: wp.array,
        cell_coords: wp.array,
        pt_obj_ids: wp.array,
        obj_pt_offsets: np.array,
    ):
        n_obj = len(self.objects)

        weights = None
        weight_gradients = None

        if self.objects[0].weight_eval_fn is None:
            return weights, weight_gradients

        cells_per_obj = ((geo.res[0] + n_obj - 1) // n_obj) * geo.res[1] * geo.res[2]
        obj_cell_indices = wp.empty_like(cell_indices)
        wp.launch(
            _compute_obj_cell_indices,
            dim=cell_indices.shape,
            inputs=[cell_indices, cells_per_obj, pt_obj_ids, obj_cell_indices],
        )

        cell_size = np.array(geo.cell_size)

        for k, obj in enumerate(self.objects):
            pts_slice = slice(int(obj_pt_offsets[k]), int(obj_pt_offsets[k + 1]))
            if pts_slice.start == pts_slice.stop:
                continue

            grad_scale = cell_size * (obj.input_bbox[1] - obj.input_bbox[0])
            obj_weights, obj_weight_gradients = obj.weight_eval_fn(
                points[pts_slice],
                obj_cell_indices[pts_slice],
                cell_coords[pts_slice],
                grad_scale,
            )

            if weights is None:
                weights = wp.empty(
                    (points.shape[0], *obj_weights.shape[1:]), dtype=float
                )
                weight_gradients = wp.empty(weights.shape, dtype=wp.vec3)

            weights[pts_slice].assign(obj_weights)
            weight_gradients[pts_slice].assign(obj_weight_gradients)

        return (
            weights,
            weight_gradients,
        )

    def _eval_point_stiffness(
        self,
        geo: fem.Grid3D,
        points: wp.array,
        obj_pt_offsets: np.array,
        pt_stiffness: wp.array,
    ):
        for k, obj in enumerate(self.objects):
            if not obj.stiffness_eval_fn:
                continue

            pts_slice = slice(int(obj_pt_offsets[k]), int(obj_pt_offsets[k + 1]))
            if pts_slice.start == pts_slice.stop:
                continue
            pt_stiffness[pts_slice].assign(obj.stiffness_eval_fn(points[pts_slice]))

    def update_object_vertices(self, sim: SparseBlendedSim):
        # Update shared vertex array

        with ScopedCachedBasisWeights(
            sim, self._vtx_weights, self._vtx_weight_gradients
        ):
            fem.interpolate(
                world_position,
                fields={"u": sim.u_field},
                dest=self._vertices,
                quadrature=self._vertex_quadrature,
            )

            fem.interpolate(
                defgrad,
                fields={"u": sim.u_field},
                dest=self._vertex_Fs,
                quadrature=self._vertex_quadrature,
            )

        # Copy to individual objects
        vertices_np = self._vertices.numpy()
        Fs_np = self._vertex_Fs.numpy()
        for k, obj in enumerate(self.objects):
            vtx_beg = self._obj_vtx_offsets[k]
            vtx_end = self._obj_vtx_offsets[k + 1]
            np.copyto(dst=obj.vertices, src=vertices_np[vtx_beg:vtx_end])
            np.copyto(dst=obj.vertex_Fs, src=Fs_np[vtx_beg:vtx_end])

    def update_grid_nodes(self, sim: SparseBlendedSim):
        # Update shared vertex array

        with ScopedCachedBasisWeights(
            sim, self._grid_weights, self._grid_weight_gradients
        ):
            fem.interpolate(
                world_position,
                fields={"u": sim.u_field},
                dest=self.grid_nodes,
                quadrature=self._grid_quadrature,
            )


@wp.kernel
def _compute_obj_cell_indices(
    cell_indices: wp.array(dtype=int),
    cells_per_obj: int,
    pt_obj_ids: wp.array(dtype=int),
    obj_cell_indices: wp.array(dtype=int),
):
    i = wp.tid()
    obj_cell_indices[i] = cell_indices[i] - cells_per_obj * pt_obj_ids[i]


@wp.kernel
def _extract_3d_node_indices(
    node_size: int,
    grid_res: wp.vec3i,
    cell_node_indices: wp.array2d(dtype=int),
    cell_node_indices_3d: wp.array2d(dtype=wp.vec3i),
):
    cell, k = wp.tid()

    node_idx = cell_node_indices[cell, k * node_size] // node_size

    strides = wp.vec2i((grid_res[1] + 1) * (grid_res[2] + 1), (grid_res[2] + 1))
    nx = node_idx // strides[0]
    ny = (node_idx - strides[0] * nx) // strides[1]
    nz = node_idx - strides[0] * nx - strides[1] * ny

    cell_node_indices_3d[cell, k] = wp.vec3i(nx, ny, nz)


def _analytical_weights(args):
    n_nodes = 8

    def cosine_weights(pts, cell_indices, cell_coords, world_space_grad_scale):
        pts = pts.numpy()

        ones = np.ones(pts.shape[:-1])

        freq = 2.0 * np.pi * (args.res + 1)

        weights = np.stack(
            (
                ones,
                np.cos(pts[..., 0] * freq),
                np.cos(pts[..., 1] * freq),
                np.cos(pts[..., 2] * freq),
            ),
            axis=1,
        )

        grads = np.zeros((*weights.shape, 3))
        grads[..., 1, 0] = (
            -freq * np.sin(pts[..., 0] * freq) * world_space_grad_scale[0]
        )
        grads[..., 2, 1] = (
            -freq * np.sin(pts[..., 1] * freq) * world_space_grad_scale[1]
        )
        grads[..., 3, 2] = (
            -freq * np.sin(pts[..., 2] * freq) * world_space_grad_scale[2]
        )

        weights = np.broadcast_to(
            weights[np.newaxis, ...], shape=(n_nodes, *weights.shape)
        )
        grads = np.broadcast_to(grads[np.newaxis, ...], shape=(n_nodes, *grads.shape))

        weights = np.transpose(weights, axes=(1, 0, 2))
        grads = np.transpose(grads, axes=(1, 0, 2, 3))

        weights = np.ascontiguousarray(weights)
        grads = np.ascontiguousarray(grads)

        return weights, grads

    return cosine_weights if args.cosine else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    SparseBlendedSim.add_parser_arguments(parser)
    parser.add_argument("--res", type=int, default=1, help="Grid resolution")
    parser.add_argument(
        "--n_qp", type=int, default=256, help="Number of quadrature points per object"
    )

    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)
    parser.add_argument("--screenshot", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--cosine",
        action=argparse.BooleanOptionalAction,
        help="Use cosine basis functions (emulates beural basis)",
    )
    args = parser.parse_args()

    scene = SparseBlendedScene(args)

    weight_eval_fn = _analytical_weights(args)

    # scene.add_object(wp.vec3(0.0, -0.25, 0.0))

    scene.add_object(
        wp.vec3(0.0, 1.0, 0.0),
        scale=wp.vec3(1.5, 1.0, 0.75),
        # rotation=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.8),
        weight_eval_fn=weight_eval_fn,
        stiffness_eval_fn=lambda pts: np.where(pts.numpy()[:, 1] < 0.5, 0.1, 1.0),
    )
    # scene.add_object(wp.vec3(1.4, 1.5, 0.0), weight_eval_fn=weight_eval_fn)
    scene.add_object(wp.vec3(0.5, 3.5, 0.0), scale=0.75, weight_eval_fn=weight_eval_fn)
    scene.add_object(wp.vec3(0.5, 5.25, 0.0), weight_eval_fn=weight_eval_fn)
    scene.add_object(wp.vec3(0.5, 6.5, 0.0), weight_eval_fn=weight_eval_fn)

    scene.add_object(
        wp.vec3(0.5, 8.75, 0.0),
        qps=np.random.rand(args.n_qp, 3) * 2.0 + 2.0,
        weight_eval_fn=weight_eval_fn,
    )

    sim = scene.make_sim()

    # fix first object
    qp_target_pos = sim.qp_world_position()
    qp_start_pos = qp_target_pos.numpy()

    @wp.func
    def prescribed_pos(qp_index: int, qp_target_pos: wp.array(dtype=wp.vec3)):
        return qp_target_pos[qp_index]

    @wp.func
    def prescribed_pos_weight(qp_index: int, k: float, n_qp: int):
        return wp.where(qp_index < n_qp, k, 0.0)

    sim.set_prescribed_positions(
        pos_field=QPBasedImplicitField(
            sim.domain, prescribed_pos, values={"qp_target_pos": qp_target_pos}
        ),
        weight_field=QPBasedImplicitField(
            sim.domain,
            prescribed_pos_weight,
            values={"n_qp": args.n_qp, "k": args.collision_stiffness},
        ),
    )

    # animate

    sim.init_constant_forms()
    sim.project_constant_forms()

    if args.headless:
        for sim.cur_frame in range(sim.args.n_frames):
            with wp.ScopedTimer(f"--- Frame --- {sim.cur_frame}", synchronize=True):
                sim.run_frame()

    else:
        sim.cur_frame = 0

        ps.init()
        ps.set_ground_plane_height(0.0)

        qpoints = ps.register_point_cloud(
            "qp", sim.qp_world_position().numpy(), enabled=False
        )

        cpoints = ps.register_point_cloud("cp", sim.cp_world_position().numpy())
        cpoints.set_radius(args.collision_radius, relative=False)

        n_obj = len(scene.objects)
        colors = np.broadcast_to(
            np.random.rand(n_obj, 1, 3),
            shape=(n_obj, args.n_qp, 3),
        ).reshape(-1, 3)

        qpoints.add_color_quantity("index", colors, enabled=True)
        cpoints.add_color_quantity("index", colors, enabled=True)

        def callback():
            sim.cur_frame = sim.cur_frame + 1
            if sim.args.n_frames >= 0 and sim.cur_frame > sim.args.n_frames:
                return

            # animate target points
            qp_target_pos.assign(
                qp_start_pos + np.array([0, 0, np.sin(sim.cur_frame / 10)])[np.newaxis]
            )

            with wp.ScopedTimer(f"--- Frame --- {sim.cur_frame}", synchronize=True):
                sim.run_frame()

            # sample displacement at quadrature points
            qpoints.update_point_positions(sim.qp_world_position().numpy())
            cpoints.update_point_positions(sim.cp_world_position().numpy())

            if args.screenshot:
                ps.screenshot()

        ps.set_user_callback(callback)
        ps.look_at(target=(1.5, 1.5, 1.0), camera_location=(0.0, 3.0, 7))
        ps.show()
