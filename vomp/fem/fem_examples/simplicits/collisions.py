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

import warp as wp
import warp.fem as fem

from warp.sim.collide import triangle_closest_point


@wp.kernel
def detect_mesh_collisions(
    max_contacts: int,
    mesh_ids: wp.array(dtype=wp.uint64),
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    m, tid = wp.tid()
    mesh_id = mesh_ids[m]

    x = pos_cur[tid]

    lower = x - wp.vec3(radius)
    upper = x + wp.vec3(radius)

    query = wp.mesh_query_aabb(mesh_id, lower, upper)

    mesh = wp.mesh_get(mesh_id)

    face_index = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face_index):
        t1 = mesh.indices[face_index * 3 + 0]
        t2 = mesh.indices[face_index * 3 + 1]
        t3 = mesh.indices[face_index * 3 + 2]

        u1 = mesh.points[t1]
        u2 = mesh.points[t2]
        u3 = mesh.points[t3]
        p, bary, feature_type = triangle_closest_point(u1, u2, u3, x)

        delta = x - p
        dist = wp.length(delta)  # * sign
        penetration = radius - dist

        if penetration > 0.0:
            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts:
                return

            if dist < 0.00001:
                n = wp.mesh_eval_face_normal(mesh_id, face_index)
            else:
                n = wp.normalize(delta)  # * sign
            normals[idx] = n

            v = wp.mesh_eval_velocity(mesh_id, face_index, bary[1], bary[2])

            kinematic_gap = (dist - wp.dot(du_cur[tid], n)) * n - v
            kinematic_gaps[idx] = kinematic_gap
            indices_a[idx] = tid
            indices_b[idx] = fem.NULL_QP_INDEX


@wp.kernel
def detect_ground_collisions(
    max_contacts: int,
    up_axis: int,
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    ground_height: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    i = wp.tid()
    x = pos_cur[i]

    if x[up_axis] < ground_height + radius:
        idx = wp.atomic_add(count, 0, 1)
        if idx >= max_contacts:
            return

        nor = wp.vec3(0.0)
        nor[up_axis] = 1.0

        normals[idx] = nor
        kinematic_gaps[idx] = (wp.dot(x - du_cur[i], nor) - ground_height) * nor
        indices_a[idx] = i
        indices_b[idx] = fem.NULL_QP_INDEX


@wp.kernel
def detect_particle_collisions(
    max_contacts: int,
    grid: wp.uint64,
    radius: float,
    self_collision_immune_radius: float,
    pos_cur: wp.array(dtype=wp.vec3),
    pos_rest: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    qp_obj_ids: wp.array(dtype=int),
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    tid = wp.tid()

    idx_a = wp.hash_grid_point_id(grid, tid)
    obj_a = qp_obj_ids[idx_a]
    pos_a = pos_cur[idx_a]

    for idx_b in wp.hash_grid_query(grid, pos_a, radius):
        if idx_a >= idx_b:
            continue  # symmetric

        obj_b = qp_obj_ids[idx_b]
        if (
            obj_a == obj_b
            and wp.length_sq(pos_rest[idx_a] - pos_rest[idx_b])
            < self_collision_immune_radius
        ):
            continue  # no self collisions

        pos_b = pos_cur[idx_b]
        d = wp.length(pos_a - pos_b)
        if d <= radius:
            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts:
                return

            nor = wp.normalize(pos_a - pos_b)
            normals[idx] = nor
            kinematic_gaps[idx] = (
                wp.dot(pos_a - pos_b - (du_cur[idx_a] - du_cur[idx_b]), nor) * nor
            )
            indices_a[idx] = idx_a
            indices_b[idx] = idx_b


@wp.func
def collision_offset(
    c: int,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    idx_a = indices_a[c]
    idx_b = indices_b[c]

    offset = du_cur[idx_a] + kinematic_gaps[c]
    if idx_b != fem.NULL_QP_INDEX:
        offset -= du_cur[idx_b]

    return offset


@wp.func
def collision_target_distance(
    c: int,
    radius: float,
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    return wp.where(indices_b[c] == fem.NULL_ELEMENT_INDEX, 1.0, 2.0) * radius


@wp.kernel
def collision_energy(
    radius: float,
    barrier_distance_ratio: float,
    mu: float,
    dt: float,
    nu: float,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    energies: wp.array(dtype=float),
):
    c = wp.tid()

    offset = collision_offset(c, du_cur, kinematic_gaps, indices_a, indices_b)
    rc = collision_target_distance(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    # Check its within radiuses.
    if rp_ratio < d_hat and d_hat <= 1.0:
        d_min_l_squared = (d_hat - 1.0) * (
            d_hat - 1.0
        )  # quadratic term ensures energy is 0 when d = rc
        E = -d_min_l_squared * wp.log(
            d_hat - rp_ratio
        )  # log barrier term. inf when two rp's overlaps.

        # friction energy
        dc = d_hat - 1.0
        dp = d_hat - rp_ratio
        barrier = 2.0 * wp.log(dp)

        dE_d_hat = -dc * (barrier + dc / dp)
        vt = (offset - d * nor) / dt  # tangential velocity
        vt_norm = wp.length(vt)

        mu_fn = -mu * dE_d_hat / rc  # yield force

        E += (
            mu_fn
            * dt
            * (
                0.5 * nu * vt_norm * vt_norm
                + wp.where(
                    vt_norm < 1.0,
                    vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
                    vt_norm - 1.0 / 3.0,
                )
            )
        )

    else:
        E = 0.0

    energies[c] = E


@wp.kernel
def collision_gradient_and_hessian(
    radius: float,
    barrier_distance_ratio: float,
    mu: float,
    dt: float,
    nu: float,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    gradient: wp.array(dtype=wp.vec3),
    hessian: wp.array(dtype=wp.mat33),
):
    c = wp.tid()

    offset = collision_offset(c, du_cur, kinematic_gaps, indices_a, indices_b)
    rc = collision_target_distance(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    if rp_ratio < d_hat and d_hat <= 1.0:
        dc = d_hat - 1.0
        dp = d_hat - rp_ratio
        barrier = 2.0 * wp.log(dp)

        dE_d_hat = -dc * (barrier + dc / dp)
        gradient[c] = dE_d_hat / rc * nor

        dbarrier_d_hat = 2.0 / dp
        ddcdp_d_hat = (dp - dc) / (dp * dp)

        d2E_d_hat2 = -(barrier + dc / dp) - dc * (dbarrier_d_hat + ddcdp_d_hat)
        hessian[c] = d2E_d_hat2 / (rc * rc) * wp.outer(nor, nor)

        # friction

        vt = (offset - d * nor) / dt  # tangential velocity
        vt_norm = wp.length(vt)
        vt_dir = wp.normalize(vt)  # avoids dealing with 0

        mu_fn = -mu * dE_d_hat / rc  # yield force

        f1_over_vt_norm = wp.where(vt_norm < 1.0, 2.0 - vt_norm, 1.0 / vt_norm)
        gradient[c] += mu_fn * (f1_over_vt_norm + nu) * vt

        # regularization such that f / H dt <= k v (penalizes friction switching dir)
        friction_slip_reg = 0.1
        df1_d_vtn = wp.max(
            2.0 * (1.0 - vt_norm),
            friction_slip_reg / (0.5 * friction_slip_reg + vt_norm),
        )

        vt_perp = wp.cross(vt_dir, nor)
        hessian[c] += (
            mu_fn
            / dt
            * (
                (df1_d_vtn + nu) * wp.outer(vt_dir, vt_dir)
                + (f1_over_vt_norm + nu) * wp.outer(vt_perp, vt_perp)
            )
        )

    else:
        gradient[c] = wp.vec3(0.0)
        hessian[c] = wp.mat33(0.0)


@wp.kernel
def compute_collision_bounds(
    radius: float,
    barrier_distance_ratio: float,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    delta_du: wp.array(dtype=wp.vec3),
    jacobian_a_offsets: wp.array(dtype=int),
    jacobian_a_columns: wp.array(dtype=int),
    jacobian_b_offsets: wp.array(dtype=int),
    jacobian_b_columns: wp.array(dtype=int),
    dof_t_max: wp.array(dtype=float),
):
    c = wp.tid()

    # Distance delta
    nor = normals[c]

    idx_a = indices_a[c]
    idx_b = indices_b[c]

    delta_d_a = wp.dot(nor, delta_du[idx_a])
    if idx_b == fem.NULL_QP_INDEX:
        delta_d_b = 0.0
    else:
        delta_d_b = -wp.dot(nor, delta_du[idx_b])

    # Current distance
    offset = collision_offset(c, du_cur, kinematic_gaps, indices_a, indices_b)
    rc = collision_target_distance(c, radius, indices_a, indices_b)
    rp = barrier_distance_ratio * rc
    gap_cur = rp - wp.dot(offset, nor)

    if gap_cur >= 0.0:
        # Missed due to too large timestep. Can't do anything now
        return

    MAX_PROGRESS = 0.75
    max_delta_d = 0.5 * MAX_PROGRESS * gap_cur

    # local bounds
    if delta_d_a < 0.0:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_a, 0.0, 1.0)
        if t_max < 1.0:
            dof_beg = jacobian_a_offsets[c]
            dof_end = jacobian_a_offsets[c + 1]
            for dof in range(dof_beg, dof_end):
                wp.atomic_min(dof_t_max, jacobian_a_columns[dof], t_max)

    if delta_d_b < 0.0:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_b, 0.0, 1.0)
        if t_max < 1.0:
            dof_beg = jacobian_b_offsets[c]
            dof_end = jacobian_b_offsets[c + 1]
            for dof in range(dof_beg, dof_end):
                wp.atomic_min(dof_t_max, jacobian_b_columns[dof], t_max)


@wp.kernel
def apply_collision_bounds(
    delta_du: wp.array(dtype=wp.vec3),
    alpha: float,
    dof_t_max: wp.array(dtype=float),
    du: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    dui = wp.min(alpha, dof_t_max[i]) * delta_du[i]

    du[i] += dui
    u[i] += dui
