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

import gc
import math
from typing import Optional
import warp as wp
import warp.fem as fem
import warp.sparse as sp

from warp.fem.utils import symmetric_eigenvalues_qr
from warp.utils import array_inner

DELASSUS_DIAG_CUTOFF = wp.constant(1.0e-6)

vec6 = wp.vec(length=6, dtype=wp.float32)
mat66 = wp.mat(shape=(6, 6), dtype=wp.float32)
mat63 = wp.mat(shape=(6, 3), dtype=wp.float32)
mat36 = wp.mat(shape=(3, 6), dtype=wp.float32)


@wp.kernel
def compute_delassus_diagonal(
    split_mass: wp.bool,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    inv_volume: wp.array(dtype=float),
    stress_strain_matrices: wp.array(dtype=mat66),
    transposed_strain_mat_offsets: wp.array(dtype=int),
    strain_rhs: wp.array(dtype=vec6),
    stress: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
):
    tau_i = wp.tid()
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    diag_block = stress_strain_matrices[tau_i]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]

        mass_ratio = wp.where(
            split_mass,
            value_if_true=float(
                transposed_strain_mat_offsets[u_i + 1]
                - transposed_strain_mat_offsets[u_i]
            ),
            value_if_false=1.0,
        )

        b_val = strain_mat_values[b]
        inv_frac = inv_volume[u_i] * mass_ratio

        diag_block += (b_val * inv_frac) @ wp.transpose(b_val)

    if wp.trace(diag_block) > DELASSUS_DIAG_CUTOFF:
        for k in range(1, 6):
            # Remove shear-divergence coupling
            # (current implementation of solve_coulomb_aniso normal and tangential responses are independent)
            diag_block[0, k] = 0.0
            diag_block[k, 0] = 0.0

        diag, ev = symmetric_eigenvalues_qr(diag_block, 1.0e-12)
        if not (wp.ddot(ev, ev) < 1.0e16 and wp.length_sq(diag) < 1.0e16):
            # wp.print(diag_block)
            # wp.print(diag)
            diag = wp.get_diag(diag_block)
            ev = wp.identity(n=6, dtype=float)

        # Disable null modes -- e.g. from velocity boundary conditions
        for k in range(0, 6):
            if diag[k] < DELASSUS_DIAG_CUTOFF:
                diag[k] = 1.0
                ev[k] = vec6(0.0)

        delassus_diagonal[tau_i] = diag
        delassus_rotation[tau_i] = wp.transpose(ev) @ wp.diag(
            wp.cw_div(vec6(1.0), diag)
        )

        # Apply rotation to contact data
        nor = ev * vec6(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        delassus_normal[tau_i] = nor

        strain_rhs[tau_i] = ev * strain_rhs[tau_i]
        stress[tau_i] = wp.cw_mul(ev * stress[tau_i], diag)

        for b in range(block_beg, block_end):
            strain_mat_values[b] = ev * strain_mat_values[b]

        stress_strain_matrices[tau_i] = (
            ev * stress_strain_matrices[tau_i] * delassus_rotation[tau_i]
        )
    else:
        # Not a valid constraint, disable
        delassus_diagonal[tau_i] = vec6(1.0)
        delassus_normal[tau_i] = vec6(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        delassus_rotation[tau_i] = wp.identity(n=6, dtype=float)
        stress[tau_i] = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        strain_rhs[tau_i] = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        stress_strain_matrices[tau_i] = mat66(0.0)


@wp.kernel
def scale_transposed_strain_mat(
    tr_strain_mat_offsets: wp.array(dtype=int),
    tr_strain_mat_columns: wp.array(dtype=int),
    tr_strain_mat_values: wp.array(dtype=mat36),
    inv_volume: wp.array(dtype=float),
    delassus_rotation: wp.array(dtype=mat66),
):
    u_i = wp.tid()
    block_beg = tr_strain_mat_offsets[u_i]
    block_end = tr_strain_mat_offsets[u_i + 1]

    for b in range(block_beg, block_end):
        tau_i = tr_strain_mat_columns[b]

        tr_strain_mat_values[b] = (
            inv_volume[u_i] * tr_strain_mat_values[b] @ delassus_rotation[tau_i]
        )


@wp.kernel
def postprocess_stress(
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
    stress_strain_matrices: wp.array(dtype=mat66),
    stress: wp.array(dtype=vec6),
    elastic_strain: wp.array(dtype=vec6),
    delta_stress: wp.array(dtype=vec6),
):
    i = wp.tid()
    rot = delassus_rotation[i]
    diag = delassus_diagonal[i]

    loc_stress = stress[i]
    loc_strain = -(stress_strain_matrices[i] * loc_stress)

    stress[i] = rot * loc_stress
    elastic_strain[i] = rot * wp.cw_mul(loc_strain, diag)
    delta_stress[i] = wp.cw_div(loc_stress, diag)


@wp.func
def eval_sliding_residual(alpha: float, D: vec6, b_T: vec6):
    d_alpha = D + vec6(alpha)

    r_alpha = wp.cw_div(b_T, d_alpha)
    dr_dalpha = -wp.cw_div(r_alpha, d_alpha)

    f = wp.dot(r_alpha, r_alpha) - 1.0
    df_dalpha = 2.0 * wp.dot(r_alpha, dr_dalpha)
    return f, df_dalpha


@wp.func
def solve_sliding_aniso(D: vec6, b_T: vec6, yield_stress: float):
    if yield_stress <= 0.0:
        return b_T

    # Viscous shear opposite to tangential stress, zero divergence
    # find alpha, r_t,  mu_rn, (D + alpha/(mu r_n) I) r_t + b_t = 0, |r_t| = mu r_n
    # find alpha,  |(D mu r_n + alpha I)^{-1} b_t|^2 = 1.0

    mu_rn = yield_stress
    Dmu_rn = D * mu_rn

    alpha_0 = wp.length(b_T)
    alpha_max = alpha_0 - wp.min(Dmu_rn)
    alpha_min = wp.max(0.0, alpha_0 - wp.max(Dmu_rn))

    # We're looking for the root of an hyperbola, approach using Newton from the left
    alpha_cur = alpha_min

    if alpha_max - alpha_min > DELASSUS_DIAG_CUTOFF:
        for k in range(24):
            f_cur, df_dalpha = eval_sliding_residual(alpha_cur, Dmu_rn, b_T)

            alpha_next = wp.clamp(alpha_cur - f_cur / df_dalpha, alpha_min, alpha_max)
            alpha_cur = alpha_next

    u_T = wp.cw_div(b_T * alpha_cur, Dmu_rn + vec6(alpha_cur))

    # Sanity checkI
    # r_T_sol = -u_T * mu_rn / alpha_cur
    # err = wp.length(r_T_sol) - mu_rn
    # if err > 1.4:
    #     f_cur, df_dalpha = eval_sliding_residual(alpha_cur, Dmu_rn, b_T)
    #     wp.printf("%d %f %f %f %f %f \n", k, wp.length(r_T_sol), f_cur, mu_rn, alpha_cur / alpha_min, alpha_cur / alpha_max)
    #     #wp.print(D)
    #     #wp.print(b)
    #     #wp.print(mu_dyn)

    return u_T


@wp.func
def solve_coulomb_aniso(
    D: vec6,
    b: vec6,
    nor: vec6,
    unilateral: wp.bool,
    mu_st: float,
    mu_dyn: float,
    yield_stress: wp.vec3,
):
    # Note: this assumes that D.nor = lambda nor
    # i.e. nor should be along one canonical axis
    # (solve_sliding aniso would get a lot more complex otherwise as normal and tangential
    # responses become interlinked)

    # Positive divergence, zero stress
    b_N = wp.dot(b, nor)
    if unilateral and b_N >= 0.0:
        return b

    # Static friction, zero shear
    r_0 = -wp.cw_div(b, D)
    r_N0 = wp.dot(r_0, nor)
    r_T = r_0 - r_N0 * nor

    r_N = wp.clamp(r_N0, yield_stress[1], yield_stress[2])
    u_N = (r_N - r_N0) * wp.cw_mul(nor, D)

    mu_rn = wp.max(mu_st * r_N, yield_stress[0])
    mu_rn_sq = mu_rn * mu_rn
    if mu_rn >= 0 and wp.dot(r_T, r_T) <= mu_rn_sq:
        return u_N

    mu_rn = wp.max(mu_dyn * r_N, yield_stress[0])
    b_T = b - b_N * nor
    return u_N + solve_sliding_aniso(D, b_T, mu_rn)


@wp.func
def solve_local_stress(
    tau_i: int,
    unilateral: wp.bool,
    friction_coeff: float,
    D: vec6,
    yield_stress: wp.array(dtype=wp.vec3),
    stress_strain_matrices: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    delassus_normal: wp.array(dtype=vec6),
    strain_rhs: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
    stress: wp.array(dtype=vec6),
    delta_correction: wp.array(dtype=vec6),
):
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    tau = strain_rhs[tau_i]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        tau += strain_mat_values[b] * velocities[u_i]

    nor = delassus_normal[tau_i]
    cur_stress = stress[tau_i]

    # substract elastic strain
    # this is the one thing that spearates elasticity from simple modification
    # of the the Delassus operator
    tau += stress_strain_matrices[tau_i] * cur_stress

    tau_new = solve_coulomb_aniso(
        D,
        tau - cur_stress,
        nor,
        unilateral,
        friction_coeff,
        friction_coeff,
        yield_stress[tau_i],
    )

    delta_stress = tau_new - tau

    delta_correction[tau_i] = delta_stress
    stress[tau_i] = cur_stress + delta_stress


@wp.kernel
def solve_local_stress_jacobi(
    unilateral: wp.bool,
    friction_coeff: float,
    yield_stress: wp.array(dtype=wp.vec3),
    stress_strain_matrices: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    strain_rhs: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
    stress: wp.array(dtype=vec6),
    delta_correction: wp.array(dtype=vec6),
):
    tau_i = wp.tid()
    D = delassus_diagonal[tau_i]

    solve_local_stress(
        tau_i,
        unilateral,
        friction_coeff,
        D,
        yield_stress,
        stress_strain_matrices,
        strain_mat_offsets,
        strain_mat_columns,
        strain_mat_values,
        delassus_normal,
        strain_rhs,
        velocities,
        stress,
        delta_correction,
    )


@wp.func
def apply_stress_delta_gs(
    tau_i: int,
    D: vec6,
    delta_stress: vec6,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    inv_mass_matrix: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
):
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        delta_vel = inv_mass_matrix[u_i] @ (
            wp.cw_div(delta_stress, D) @ strain_mat_values[b]
        )
        velocities[u_i] += delta_vel


@wp.kernel
def apply_stress_gs(
    color_offset: int,
    color_indices: wp.array(dtype=int),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    inv_mass_matrix: wp.array(dtype=float),
    stress: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
):
    tau_i = color_indices[wp.tid() + color_offset]

    D = delassus_diagonal[tau_i]
    cur_stress = stress[tau_i]

    apply_stress_delta_gs(
        tau_i,
        D,
        cur_stress,
        strain_mat_offsets,
        strain_mat_columns,
        strain_mat_values,
        inv_mass_matrix,
        velocities,
    )


@wp.kernel
def solve_local_stress_gs(
    color_offset: int,
    unilateral: wp.bool,
    friction_coeff: float,
    color_indices: wp.array(dtype=int),
    yield_stress: wp.array(dtype=wp.vec3),
    stress_strain_matrices: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    inv_mass_matrix: wp.array(dtype=float),
    strain_rhs: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
    stress: wp.array(dtype=vec6),
    delta_correction: wp.array(dtype=vec6),
):
    tau_i = color_indices[wp.tid() + color_offset]

    D = delassus_diagonal[tau_i]
    solve_local_stress(
        tau_i,
        unilateral,
        friction_coeff,
        D,
        yield_stress,
        stress_strain_matrices,
        strain_mat_offsets,
        strain_mat_columns,
        strain_mat_values,
        delassus_normal,
        strain_rhs,
        velocities,
        stress,
        delta_correction,
    )

    apply_stress_delta_gs(
        tau_i,
        D,
        delta_correction[tau_i],
        strain_mat_offsets,
        strain_mat_columns,
        strain_mat_values,
        inv_mass_matrix,
        velocities,
    )


@wp.kernel
def apply_collider_impulse(
    collider_impulse: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    collider_inv_mass: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    collider_velocities: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    velocities[i] += inv_mass[i] * collider_impulse[i]
    collider_velocities[i] -= collider_inv_mass[i] * collider_impulse[i]


@wp.func
def solve_coulomb_isotropic(
    mu: float,
    nor: wp.vec3,
    u: wp.vec3,
):
    u_n = wp.dot(u, nor)
    if u_n < 0.0:
        u -= u_n * nor
        tau = wp.length_sq(u)
        alpha = mu * u_n
        if tau <= alpha * alpha:
            u = wp.vec3(0.0)
        else:
            u *= 1.0 + mu * u_n / wp.sqrt(tau)

    return u


@wp.kernel
def solve_nodal_friction(
    inv_mass: wp.array(dtype=float),
    collider_friction: wp.array(dtype=float),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_inv_mass: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    collider_velocities: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        return

    n = collider_normals[i]
    u0 = velocities[i] - collider_velocities[i]

    w = inv_mass[i] + collider_inv_mass[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - impulse[i] * w)

    delta_u = u - u0
    delta_impulse = delta_u / w

    impulse[i] += delta_impulse
    velocities[i] += inv_mass[i] * delta_impulse
    collider_velocities[i] -= collider_inv_mass[i] * delta_impulse


def solve_rheology(
    unilateral: bool,
    friction_coeff: float,
    max_iterations: int,
    tolerance: float,
    strain_mat: sp.BsrMatrix,
    transposed_strain_mat: sp.BsrMatrix,
    inv_volume,
    yield_stress,
    stress_strain_matrices,
    strain_rhs,
    stress,
    velocity,
    collider_friction,
    collider_normals,
    collider_velocities,
    collider_inv_mass,
    collider_impulse,
    color_offsets,
    color_indices: wp.array = None,
    rigidity_mat: Optional[sp.BsrMatrix] = None,
    temporary_store: Optional[fem.TemporaryStore] = None,
):
    delta_stress = fem.borrow_temporary_like(stress, temporary_store)

    delassus_rotation = fem.borrow_temporary(
        temporary_store, shape=stress.shape, dtype=mat66
    )
    delassus_diagonal = fem.borrow_temporary(
        temporary_store, shape=stress.shape, dtype=vec6
    )
    delassus_normal = fem.borrow_temporary(
        temporary_store, shape=stress.shape, dtype=vec6
    )

    color_count = 0 if color_offsets is None else len(color_offsets) - 1
    gs = color_count > 0
    split_mass = not gs

    #  Build transposed matrix
    if not gs:
        sp.bsr_set_transpose(dest=transposed_strain_mat, src=strain_mat)

    # Compute and factorize diagonal blacks, rotate strain matrix to diagonal basis
    wp.launch(
        kernel=compute_delassus_diagonal,
        dim=stress.shape[0],
        inputs=[
            split_mass,
            strain_mat.offsets,
            strain_mat.columns,
            strain_mat.values,
            inv_volume,
            stress_strain_matrices,
            transposed_strain_mat.offsets,
            strain_rhs,
            stress,
            delassus_rotation.array,
            delassus_diagonal.array,
            delassus_normal.array,
        ],
    )

    if rigidity_mat is not None:
        prev_collider_velocity = fem.borrow_temporary_like(
            collider_velocities, temporary_store
        )
        wp.copy(dest=prev_collider_velocity.array, src=collider_velocities)

    def apply_rigidity_matrix():
        # Apply rigidity matrix
        if rigidity_mat is not None:
            # velocity delta
            fem.utils.array_axpy(
                y=prev_collider_velocity.array,
                x=collider_velocities,
                alpha=1.0,
                beta=-1.0,
            )
            # rigidity contribution to new velocity
            sp.bsr_mv(
                A=rigidity_mat,
                x=prev_collider_velocity.array,
                y=collider_velocities,
                alpha=1.0,
                beta=1.0,
            )
            # save for next iterations
            wp.copy(dest=prev_collider_velocity.array, src=collider_velocities)

    if gs:
        # Apply initial guess
        apply_stress_launch = wp.launch(
            kernel=apply_stress_gs,
            dim=1,
            inputs=[
                0,
                color_indices,
                strain_mat.offsets,
                strain_mat.columns,
                strain_mat.values,
                delassus_diagonal.array,
                inv_volume,
                stress,
                velocity,
            ],
            block_dim=64,
            record_cmd=True,
        )

        for k in range(color_count):
            apply_stress_launch.set_param_at_index(0, color_offsets[k])
            apply_stress_launch.set_dim((int(color_offsets[k + 1] - color_offsets[k]),))
            apply_stress_launch.launch()

    else:
        # Apply local scaling and rotations to transposed strain matrix
        wp.launch(
            kernel=scale_transposed_strain_mat,
            dim=inv_volume.shape[0],
            inputs=[
                transposed_strain_mat.offsets,
                transposed_strain_mat.columns,
                transposed_strain_mat.values,
                inv_volume,
                delassus_rotation.array,
            ],
        )

        # Apply initial guess
        sp.bsr_mv(A=transposed_strain_mat, x=stress, y=velocity, alpha=1.0, beta=1.0)

    # Apply initial contact guess
    wp.launch(
        kernel=apply_collider_impulse,
        dim=collider_impulse.shape[0],
        inputs=[
            collider_impulse,
            inv_volume,
            collider_inv_mass,
            velocity,
            collider_velocities,
        ],
    )
    apply_rigidity_matrix()

    def solve_collider():
        wp.launch(
            kernel=solve_nodal_friction,
            dim=collider_impulse.shape[0],
            inputs=[
                inv_volume,
                collider_friction,
                collider_normals,
                collider_inv_mass,
                velocity,
                collider_velocities,
                collider_impulse,
            ],
        )
        apply_rigidity_matrix()

    # Gauss-Seidel solve
    def do_gs_batch(num_iterations):
        solve_local_launch = wp.launch(
            kernel=solve_local_stress_gs,
            dim=1,
            inputs=[
                0,
                unilateral,
                friction_coeff * math.sqrt(3.0 / 2.0),
                color_indices,
                yield_stress,
                stress_strain_matrices,
                strain_mat.offsets,
                strain_mat.columns,
                strain_mat.values,
                delassus_diagonal.array,
                delassus_normal.array,
                inv_volume,
                strain_rhs,
                velocity,
                stress,
                delta_stress.array,
            ],
            block_dim=64,
            record_cmd=True,
        )

        for i in range(num_iterations):
            for k in range(color_count):
                solve_local_launch.set_param_at_index(0, color_offsets[k])
                solve_local_launch.set_dim(
                    (int(color_offsets[k + 1] - color_offsets[k]),)
                )
                solve_local_launch.launch()

            solve_collider()

    # Jacobi solve
    def do_jacobi_batch(num_iterations):
        solve_local_launch = wp.launch(
            kernel=solve_local_stress_jacobi,
            dim=stress.shape[0],
            inputs=[
                unilateral,
                friction_coeff * math.sqrt(3.0 / 2.0),
                yield_stress,
                stress_strain_matrices,
                strain_mat.offsets,
                strain_mat.columns,
                strain_mat.values,
                delassus_diagonal.array,
                delassus_normal.array,
                strain_rhs,
                velocity,
                stress,
                delta_stress.array,
            ],
            record_cmd=True,
        )
        for i in range(num_iterations):
            solve_local_launch.launch()
            # Add jacobi delta
            sp.bsr_mv(
                A=transposed_strain_mat,
                x=delta_stress.array,
                y=velocity,
                alpha=1.0,
                beta=1.0,
            )
            solve_collider()

    solve_granularity = 25 if gs else 50
    use_graph = True
    solve_graph = None

    do_batch = do_gs_batch if gs else do_jacobi_batch

    if use_graph:
        gc.collect(0)
        gc.disable()
        wp.capture_begin(force_module_load=False)
        do_batch(solve_granularity)
        gc.collect(0)
        solve_graph = wp.capture_end()
        gc.enable()

    for batch in range(max_iterations // solve_granularity):
        if solve_graph is None:
            do_batch(solve_granularity)
        else:
            wp.capture_launch(solve_graph)

        res = math.sqrt(array_inner(delta_stress.array, delta_stress.array)) / (
            1 + stress.shape[0]
        )
        print(
            f"{'Gauss-Seidel' if gs else 'Jacobi'} iterations #{(batch+1) * solve_granularity} \t res(l2)={res}"
        )
        if res < tolerance:
            break

    wp.launch(
        kernel=postprocess_stress,
        dim=stress.shape[0],
        inputs=[
            delassus_rotation.array,
            delassus_diagonal.array,
            stress_strain_matrices,
            stress,
            strain_rhs,
            delta_stress.array,
        ],
    )
