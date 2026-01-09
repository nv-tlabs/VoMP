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

from typing import List
import argparse
from enum import Enum

import numpy as np

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.sim import Model, State


from fem_examples.mpm.solve_rheology import solve_rheology, solve_coulomb_isotropic

vec6 = wp.vec(length=6, dtype=wp.float32)
mat66 = wp.mat(shape=(6, 6), dtype=wp.float32)
mat63 = wp.mat(shape=(6, 3), dtype=wp.float32)
mat36 = wp.mat(shape=(3, 6), dtype=wp.float32)

VOLUME_CUTOFF = wp.constant(1.0e-4)
COLLIDER_EXTRAPOLATION_DISTANCE = wp.constant(0.25)
COLLIDER_PROJECTION_THRESHOLD = wp.constant(0.5)

INF_MASS = wp.constant(1.0e12)


_FLOOR_ID = -1
_NULL_COLLIDER_ID = -2
_FLOOR_FRICTION = 1.0
_DEFAULT_FRICTION = 0.0
_DEFAULT_THICKNESS = 0.5

SMALL_STRAINS = True


@wp.struct
class Collider:
    meshes: wp.array(dtype=wp.uint64)
    thicknesses: wp.array(dtype=float)
    projection_threshold: wp.array(dtype=float)
    friction: wp.array(dtype=float)
    masses: wp.array(dtype=float)
    query_max_dist: float
    floor_y: float
    floor_normal: wp.vec3


@wp.func
def collision_sdf(x: wp.vec3, collider: Collider):
    floor_sdf = wp.dot(x, collider.floor_normal) - collider.floor_y

    min_sdf = floor_sdf
    sdf_grad = collider.floor_normal
    sdf_vel = wp.vec3(0.0)
    collider_id = int(_FLOOR_ID)

    for m in range(collider.meshes.shape[0]):
        mesh = collider.meshes[m]

        # Union of mesh-based sdf and floor
        query = wp.mesh_query_point_sign_normal(mesh, x, collider.query_max_dist)

        if query.result:
            cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

            offset = x - cp
            d = wp.length(offset) * query.sign
            sdf = d - collider.thicknesses[m]

            if sdf < min_sdf:
                min_sdf = sdf
                if wp.abs(d) < 0.0001:
                    sdf_grad = wp.mesh_eval_face_normal(mesh, query.face)
                else:
                    sdf_grad = wp.normalize(offset) * query.sign

                sdf_vel = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
                collider_id = m

    return min_sdf, sdf_grad, sdf_vel, collider_id


@wp.func
def collision_is_active(sdf: float, voxel_size: float):
    return sdf < COLLIDER_EXTRAPOLATION_DISTANCE * voxel_size


@fem.integrand
def collider_volume(
    s: fem.Sample,
    domain: fem.Domain,
    collider: Collider,
    voxel_size: float,
    volumes: wp.array(dtype=float),
    cells: wp.array(dtype=int),
):
    x = domain(s)

    sdf, sdf_gradient, sdf_vel, collider_id = collision_sdf(x, collider)
    bc_active = collision_is_active(sdf, voxel_size)

    if bc_active:
        cells[s.element_index] = 1
        if collider_id >= 0:
            wp.atomic_add(volumes, collider_id, fem.measure(domain, s) * s.qp_weight)


@wp.func
def collider_friction_coefficient(collider_id: int, collider: Collider):
    if collider_id == _FLOOR_ID:
        return _FLOOR_FRICTION
    return collider.friction[collider_id]


@wp.func
def collider_density(
    collider_id: int, collider: Collider, collider_volumes: wp.array(dtype=float)
):
    if collider_id == _FLOOR_ID:
        return INF_MASS
    return collider.masses[collider_id] / collider_volumes[collider_id]


@wp.func
def collider_projection_threshold(collider_id: int, collider: Collider):
    if collider_id == _FLOOR_ID:
        return COLLIDER_PROJECTION_THRESHOLD
    return collider.projection_threshold[collider_id]


@wp.func
def collider_is_dynamic(collider_id: int, collider: Collider):
    if collider_id == _FLOOR_ID:
        return False
    return collider.masses[collider_id] < INF_MASS


@fem.integrand
def integrate_fraction(
    s: fem.Sample, phi: fem.Field, domain: fem.Domain, inv_cell_volume: float
):
    return phi(s) * inv_cell_volume


@fem.integrand
def integrate_collider_fraction(
    s: fem.Sample,
    domain: fem.Domain,
    phi: fem.Field,
    collider: Collider,
    inv_cell_volume: float,
):
    # Ignore space inside collider
    sdf, sdf_gradient, sdf_vel, _id = collision_sdf(domain(s), collider)
    return phi(s) * wp.where(sdf <= 0.0, inv_cell_volume, 0.0)


@fem.integrand
def integrate_velocity(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    dt: float,
    gravity: wp.vec3,
    inv_cell_volume: float,
):
    # APIC velocity prediction
    node_offset = domain(fem.at_node(u, s)) - domain(s)
    vel_apic = velocities[s.qp_index] + velocity_gradients[s.qp_index] * node_offset
    vel_adv = vel_apic + dt * gravity

    return wp.dot(u(s), vel_adv) * inv_cell_volume


@fem.integrand
def update_particles(
    s: fem.Sample,
    grid_vel: fem.Field,
    grid_strain: fem.Field,
    grid_strain_delta: fem.Field,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    transform: wp.array(dtype=wp.mat33),
    transform_prev: wp.array(dtype=wp.mat33),
    vel: wp.array(dtype=wp.vec3),
    vel_grad: wp.array(dtype=wp.mat33),
    elastic_strain_prev: wp.array(dtype=wp.mat33),
    elastic_strain: wp.array(dtype=wp.mat33),
):
    # Advect particles and project if necessary

    p_vel = grid_vel(s)
    p_vel_grad = fem.grad(grid_vel, s)

    pos_adv = pos_prev[s.qp_index] + dt * p_vel

    flip = 0.95
    strain = grid_strain(s)
    strain_delta = grid_strain_delta(s)
    # strain_delta = dt * fem.D(grid_vel, s)

    strain = strain * (1.0 - flip) + flip * (
        strain_delta + elastic_strain_prev[s.qp_index]
    )

    if SMALL_STRAINS:
        # Jaumann convected derivative
        skew = 0.5 * (p_vel_grad - wp.transpose(p_vel_grad))
        strain += dt * (skew * strain - strain * skew)
        strain = 0.5 * (strain + wp.transpose(strain))

    pos[s.qp_index] = pos_adv
    vel[s.qp_index] = p_vel
    vel_grad[s.qp_index] = p_vel_grad

    F_prev = transform_prev[s.qp_index]
    # dX1/dx = dX1/dX0 dX0/dx
    F = F_prev + dt * p_vel_grad @ F_prev

    # F = F_prev + dt * (p_vel_grad @ F_prev + F_prev @ p_vel_grad)

    # clamp eigenvalues of F
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, S, V)
    S = wp.max(wp.min(S, wp.vec3(2.0)), wp.vec3(0.25))
    F = U @ wp.diag(S) @ wp.transpose(V)

    # F = wp.identity(n = 3, dtype=float) + dt*fem.grad(grid_vel, s)
    transform[s.qp_index] = F

    elastic_strain[s.qp_index] = strain


@wp.kernel
def project_outside_collider(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    collider: Collider,
    voxel_size: float,
    dt: float,
):
    i = wp.tid()
    pos_adv = positions[i]
    p_vel = velocities[i]

    # project outside of collider
    sdf, sdf_gradient, sdf_vel, collider_id = collision_sdf(pos_adv, collider)

    sdf_end = (
        sdf
        - wp.dot(sdf_vel, sdf_gradient) * dt
        + collider_projection_threshold(collider_id, collider) * voxel_size
    )
    if sdf_end < 0:
        # remove normal vel
        friction = collider_friction_coefficient(collider_id, collider)
        delta_vel = (
            solve_coulomb_isotropic(friction, sdf_gradient, p_vel - sdf_vel)
            + sdf_vel
            - p_vel
        )

        p_vel += delta_vel
        pos_adv += delta_vel * dt

        # project out
        pos_adv -= (
            wp.min(0.0, sdf_end + dt * wp.dot(delta_vel, sdf_gradient)) * sdf_gradient
        )  # delta_vel * dt

        positions[i] = pos_adv
        velocities[i] = p_vel

        # kill velocity gradient (could maybe keep rigid)
        velocity_gradients[i] = wp.mat33(0.0)


@fem.integrand
def collider_velocity(
    s: fem.Sample,
    domain: fem.Domain,
    particle_density: float,
    collider: Collider,
    voxel_size: float,
    node_volume: wp.array(dtype=float),
    collider_volumes: wp.array(dtype=float),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_friction: wp.array(dtype=float),
    collider_ids: wp.array(dtype=int),
    collider_impulse: wp.array(dtype=wp.vec3),
    collider_inv_mass_matrix: wp.array(dtype=float),
):
    x = domain(s)
    sdf, sdf_gradient, sdf_vel, collider_id = collision_sdf(x, collider)
    bc_active = collision_is_active(sdf, voxel_size)

    if not bc_active:
        collider_normals[s.qp_index] = wp.vec3(0.0)
        collider_friction[s.qp_index] = -1.0
        collider_inv_mass_matrix[s.qp_index] = 0.0
        collider_ids[s.qp_index] = _NULL_COLLIDER_ID
        collider_impulse[s.qp_index] = wp.vec3(0.0)
        return wp.vec3(0.0)

    if collider_is_dynamic(collider_id, collider):
        bc_vol = node_volume[s.qp_index]
        bc_density = collider_density(collider_id, collider, collider_volumes)
        bc_mass = bc_vol * bc_density
        collider_inv_mass_matrix[s.qp_index] = particle_density / bc_mass
    else:
        collider_inv_mass_matrix[s.qp_index] = 0.0

    collider_ids[s.qp_index] = collider_id
    collider_normals[s.qp_index] = sdf_gradient
    collider_friction[s.qp_index] = collider_friction_coefficient(collider_id, collider)

    return sdf_vel


@fem.integrand
def free_velocity(
    s: fem.Sample,
    velocity_int: wp.array(dtype=wp.vec3),
    particle_volume: wp.array(dtype=float),
    inv_mass_matrix: wp.array(dtype=float),
):
    pvol = particle_volume[s.qp_index]
    inv_particle_volume = 1.0 / wp.max(pvol, VOLUME_CUTOFF)

    vel = velocity_int[s.qp_index] * inv_particle_volume
    inv_mass_matrix[s.qp_index] = inv_particle_volume

    return vel


@fem.integrand
def strain_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
    elastic_strain: wp.array(dtype=vec6),
    rotation: wp.array(dtype=wp.quatf),
    dt: float,
    domain: fem.Domain,
    inv_cell_volume: float,
):
    # get polar decomposition at strain node
    tau_index = fem.operator.node_index(tau, s)
    R = wp.quat_to_matrix(rotation[tau_index])
    dS = wp.transpose(R) @ fem.grad(u, s) @ R

    S = fem.SymmetricTensorMapper.dof_to_value_3d(elastic_strain[tau_index])
    dS += dS @ S

    return wp.ddot(dS, tau(s)) * (dt * inv_cell_volume)


@fem.integrand
def integrate_elastic_strain(
    s: fem.Sample,
    elastic_strains: wp.array(dtype=wp.mat33),
    tau: fem.Field,
    domain: fem.Domain,
    inv_cell_volume: float,
):
    return wp.ddot(elastic_strains[s.qp_index], tau(s)) * inv_cell_volume


@wp.kernel
def add_unilateral_strain_offset(
    max_fraction: float,
    compliance: float,
    particle_volume: wp.array(dtype=float),
    collider_volume: wp.array(dtype=float),
    node_volume: wp.array(dtype=float),
    prev_symmetric_strain: wp.array(dtype=vec6),
    int_symmetric_strain: wp.array(dtype=vec6),
):
    i = wp.tid()

    spherical_part = (
        max_fraction * (node_volume[i] - collider_volume[i]) - particle_volume[i]
    )
    spherical_part = wp.max(spherical_part, 0.0)

    strain_offset = spherical_part / 3.0 * wp.identity(n=3, dtype=float)
    strain_offset += strain_offset * fem.SymmetricTensorMapper.dof_to_value_3d(
        prev_symmetric_strain[i]
    )

    int_symmetric_strain[i] += fem.SymmetricTensorMapper.value_to_dof_3d(strain_offset)


@wp.func
def polar_decomposition(F: wp.mat33):
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)

    Vt = wp.transpose(V)
    R = U * Vt
    S = V * wp.diag(sig) * Vt

    return R, S


@wp.kernel
def scale_yield_stress_and_stress_matrices(
    yield_stress: wp.vec3,
    stress_strain_mat: mat66,
    particle_volume: wp.array(dtype=float),
    node_volume: wp.array(dtype=float),
    scaled_yield_stress: wp.array(dtype=wp.vec3),
    scaled_mat: wp.array(dtype=mat66),
):
    i = wp.tid()

    # Option 1: constitutive relation with particle stressa
    pvol = wp.max(particle_volume[i], VOLUME_CUTOFF)

    scaled_yield_stress[i] = yield_stress
    scaled_mat[i] = stress_strain_mat * pvol

    # Option 2: constitutive relation with Cauchy stress
    # the stress that we compute is actually scaled by 1/fraction,
    # so do the same thing for the yield stress

    # fraction = wp.clamp(particle_volume[i] / node_volume[i], VOLUME_CUTOFF, 1.0)
    # scaled_yield_stress[i] = yield_stress / fraction

    # # we solve want eps = K cauchy, but we solve tau, sig,
    # # with tau = V eps and phi sig = cauchy
    # # so tau = V K cauchy = (V phi K) sig
    # scaled_mat[i] = stress_strain_mat * fraction * wp.max(particle_volume[i], VOLUME_CUTOFF)


@wp.kernel
def elastic_strain_rotation(
    int_elastic_strain: wp.array(dtype=wp.mat33),
    particle_volume: wp.array(dtype=float),
    strain_rotation: wp.array(dtype=wp.quatf),
    int_symmetric_strain: wp.array(dtype=vec6),
    symmetric_strain: wp.array(dtype=vec6),
):
    i = wp.tid()

    Id = wp.identity(n=3, dtype=float)
    V = particle_volume[i]
    dF = int_elastic_strain[i]

    F = dF / wp.max(V, VOLUME_CUTOFF) + Id
    R, S = polar_decomposition(F)

    strain_rotation[i] = wp.quat_from_matrix(R)

    int_symmetric_strain[i] = fem.SymmetricTensorMapper.value_to_dof_3d(
        wp.transpose(R) @ (dF + V * Id) - V * Id
    )
    if SMALL_STRAINS:
        symmetric_strain[i] = vec6(0.0)
    else:
        symmetric_strain[i] = fem.SymmetricTensorMapper.value_to_dof_3d(S - Id)


@wp.kernel
def compute_elastic_strain_delta(
    particle_volume: wp.array(dtype=float),
    strain_rotation: wp.array(dtype=wp.quatf),
    sym_elastic_strain: wp.array(dtype=vec6),
    full_strain: wp.array(dtype=wp.mat33),
    elastic_strain: wp.array(dtype=wp.mat33),
    elastic_strain_delta: wp.array(dtype=wp.mat33),
):
    i = wp.tid()

    V_inv = 1.0 / wp.max(particle_volume[i], VOLUME_CUTOFF)
    dFe_prev = elastic_strain[i] * V_inv
    dSe = fem.SymmetricTensorMapper.dof_to_value_3d(sym_elastic_strain[i]) * V_inv

    Id = wp.identity(n=3, dtype=float)
    R = wp.quat_to_matrix(strain_rotation[i])

    if SMALL_STRAINS:
        dR = wp.mat33(0.0)
    else:
        RtdF = full_strain[i] * V_inv
        dR = 0.5 * (RtdF - wp.transpose(RtdF))

    dFe = R @ (Id + dSe + dR) - Id

    elastic_strain_delta[i] = dFe - dFe_prev
    elastic_strain[i] = dFe

    # elastic_strain_delta[i] = R @ (Id + dS - S) @ S
    # elastic_strain_delta[i] = R @ dS


@wp.kernel
def fill_collider_rigidity_matrices(
    node_positions: wp.array(dtype=wp.vec3),
    collider_volumes: wp.array(dtype=float),
    node_volumes: wp.array(dtype=float),
    collider: Collider,
    voxel_size: float,
    collider_ids: wp.array(dtype=int),
    collider_coms: wp.array(dtype=wp.vec3),
    collider_inv_inertia: wp.array(dtype=wp.mat33),
    J_rows: wp.array(dtype=int),
    J_cols: wp.array(dtype=int),
    J_values: wp.array(dtype=wp.mat33),
    IJtm_values: wp.array(dtype=wp.mat33),
    non_rigid_diagonal: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    x = node_positions[i]

    collider_id = collider_ids[i]
    bc_active = collider_id != _NULL_COLLIDER_ID

    cvol = voxel_size * voxel_size * voxel_size

    if bc_active and collider_is_dynamic(collider_id, collider):
        J_rows[2 * i] = i
        J_rows[2 * i + 1] = i
        J_cols[2 * i] = 2 * collider_id
        J_cols[2 * i + 1] = 2 * collider_id + 1

        W = wp.skew(collider_coms[collider_id] - x)
        I = wp.identity(n=3, dtype=float)
        J_values[2 * i] = W
        J_values[2 * i + 1] = I

        bc_mass = (
            node_volumes[i]
            * cvol
            * collider_density(collider_id, collider, collider_volumes)
        )

        IJtm_values[2 * i] = -bc_mass * collider_inv_inertia[collider_id] * W
        IJtm_values[2 * i + 1] = bc_mass / collider.masses[collider_id] * I

        non_rigid_diagonal[i] = -I

    else:
        J_cols[2 * i] = -1
        J_cols[2 * i + 1] = -1
        J_rows[2 * i] = -1
        J_rows[2 * i + 1] = -1

        non_rigid_diagonal[i] = wp.mat33(0.0)


@wp.kernel
def extract_rotation_and_scales(
    transform: wp.array(dtype=wp.mat33),
    rotation: wp.array(dtype=wp.vec4h),
    scales: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    A = transform[i]

    Q = wp.mat33()
    R = wp.mat33()
    wp.qr3(A, Q, R)

    q = wp.quat_from_matrix(Q)

    rotation[i] = wp.vec4h(wp.vec4(q[0], q[1], q[2], q[3]))
    scales[i] = wp.vec3(wp.length(R[0]), wp.length(R[1]), wp.length(R[2]))


@wp.kernel
def sample_grains(
    particles: wp.array(dtype=wp.vec3),
    radius: float,
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    rng = wp.rand_init(pid * positions.shape[1] + k)

    pos_loc = (
        2.0
        * wp.vec3(wp.randf(rng) - 0.5, wp.randf(rng) - 0.5, wp.randf(rng) - 0.5)
        * radius
    )
    positions[pid, k] = particles[pid] + pos_loc


@wp.kernel
def transform_grains(
    particle_pos_prev: wp.array(dtype=wp.vec3),
    particle_transform_prev: wp.array(dtype=wp.mat33),
    particle_pos: wp.array(dtype=wp.vec3),
    particle_transform: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    pos_adv = positions[pid, k]

    p_pos = particle_pos[pid]
    p_frame = particle_transform[pid]
    p_pos_prev = particle_pos_prev[pid]
    p_frame_prev = particle_transform_prev[pid]

    pos_loc = wp.inverse(p_frame_prev) @ (pos_adv - p_pos_prev)

    p_pos_adv = p_frame @ pos_loc + p_pos
    positions[pid, k] = p_pos_adv


@fem.integrand
def advect_grains(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    dt: float,
    positions: wp.array(dtype=wp.vec3),
):
    x = domain(s)
    vel = grid_vel(s)
    pos_adv = x + dt * vel
    positions[s.qp_index] = pos_adv


@wp.kernel
def advect_grains_from_particles(
    dt: float,
    particle_pos_prev: wp.array(dtype=wp.vec3),
    particle_pos: wp.array(dtype=wp.vec3),
    particle_vel_grad: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    p_pos = particle_pos[pid]
    p_pos_prev = particle_pos_prev[pid]

    pos_loc = positions[pid, k] - p_pos_prev

    p_vel_grad = particle_vel_grad[pid]

    displ = dt * p_vel_grad * pos_loc + (p_pos - p_pos_prev)
    positions[pid, k] += displ


@wp.kernel
def project_grains(
    radius: float,
    particle_pos: wp.array(dtype=wp.vec3),
    particle_frames: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    pos_adv = positions[pid, k]

    p_pos = particle_pos[pid]
    p_frame = particle_frames[pid]

    # keep within source particle
    # pos_loc = wp.inverse(p_frame) @ (pos_adv - p_pos)
    # dist = wp.max(wp.abs(pos_loc))
    # if dist > radius:
    #     pos_loc = pos_loc / dist * radius
    # p_pos_adv = p_frame @ pos_loc + p_pos

    p_frame = (radius * radius) * p_frame * wp.transpose(p_frame)
    pos_loc = pos_adv - p_pos
    vn = wp.max(1.0, wp.dot(pos_loc, wp.inverse(p_frame) * pos_loc))
    p_pos_adv = pos_loc / wp.sqrt(vn) + p_pos

    positions[pid, k] = p_pos_adv


@wp.kernel
def pad_voxels(
    particle_q: wp.array(dtype=wp.vec3i), padded_q: wp.array4d(dtype=wp.vec3i)
):
    pid = wp.tid()

    for i in range(3):
        for j in range(3):
            for k in range(3):
                padded_q[pid, i, j, k] = particle_q[pid] + wp.vec3i(i - 1, j - 1, k - 1)


@wp.func
def positive_mod3(x: int):
    return (x % 3 + 3) % 3


@wp.kernel
def node_color_27_stencil(
    voxels: wp.array2d(dtype=int),
    colors: wp.array(dtype=int),
    color_indices: wp.array(dtype=int),
):
    pid = wp.tid()

    c = voxels[pid]
    colors[pid] = (
        positive_mod3(c[0]) * 9 + positive_mod3(c[1]) * 3 + positive_mod3(c[2])
    )
    color_indices[pid] = pid


@wp.kernel
def node_color_8_stencil(
    voxels: wp.array2d(dtype=int),
    colors: wp.array(dtype=int),
    color_indices: wp.array(dtype=int),
):
    pid = wp.tid()

    c = voxels[pid]
    colors[pid] = ((c[0] & 1) << 2) + ((c[1] & 1) << 1) + (c[2] & 1)
    color_indices[pid] = pid


def allocate_by_voxels(particle_q, voxel_size, padded=True):
    volume = wp.Volume.allocate_by_voxels(
        voxel_points=particle_q.flatten(),
        voxel_size=voxel_size,
    )
    if not padded:
        return volume

    voxels = wp.empty((volume.get_voxel_count(),), dtype=wp.vec3i)
    volume.get_voxels(voxels)

    padded_voxels = wp.zeros((voxels.shape[0], 3, 3, 3), dtype=wp.vec3i)
    wp.launch(pad_voxels, voxels.shape[0], (voxels, padded_voxels))

    volume = wp.Volume.allocate_by_voxels(
        voxel_points=padded_voxels.flatten(),
        voxel_size=voxel_size,
    )

    return volume


class MPMIntegrator:
    def __init__(
        self,
        args,
        model: Model,
        colliders: List[wp.Mesh] = None,
        collider_thicknesses: List[float] = None,
        collider_projection_threshold: List[float] = None,
        collider_masses: List[float] = None,
        collider_friction: List[float] = None,
    ):
        self.density = args.density
        self.friction_coeff = args.friction
        self.yield_stresses = wp.vec3(
            args.yield_stress,
            -args.stretching_yield_stress,
            args.compression_yield_stress,
        )

        self.unilateral = args.unilateral
        self.max_fraction = args.max_fraction

        self.max_iterations = args.max_iters
        self.tolerance = args.tol

        self.voxel_size = args.voxel_size
        self.degree = 1 if args.unilateral else 0

        self.pad_grid = args.pad_grid
        self.coloring = args.gs

        self.compliance = args.compliance
        poisson = args.poisson
        lame = 1.0 / (1.0 + poisson) * np.array([poisson / (1.0 - 2.0 * poisson), 0.5])
        K = args.compliance
        self.stress_strain_mat = mat66(K / (2.0 * lame[1]) * np.eye(6))
        # self.stress_strain_mat = mat66(K * np.zeros((6, 6)))
        self.stress_strain_mat[0, 0] = K / (2.0 * lame[1] + 3.0 * lame[0])

        collider = Collider()
        collider.meshes = wp.array(
            [collider.id for collider in colliders], dtype=wp.uint64
        )
        collider.thicknesses = (
            wp.full(
                len(collider.meshes), _DEFAULT_THICKNESS * self.voxel_size, dtype=float
            )
            if collider_thicknesses is None
            else wp.array(collider_thicknesses, dtype=float)
        )
        collider.friction = (
            wp.full(len(collider.meshes), _DEFAULT_FRICTION, dtype=float)
            if collider_friction is None
            else wp.array([bc for bc in collider_friction], dtype=float)
        )
        collider.masses = (
            wp.full(len(collider.meshes), INF_MASS, dtype=float)
            if collider_masses is None
            else wp.array(collider_masses, dtype=float)
        )
        collider.projection_threshold = (
            wp.full(len(collider.meshes), COLLIDER_PROJECTION_THRESHOLD, dtype=float)
            if collider_projection_threshold is None
            else wp.array(collider_projection_threshold, dtype=float)
        )
        collider.query_max_dist = 4.0 * self.voxel_size
        collider.floor_y = model.ground_plane.numpy()[3] if model.ground else -1.0e8
        collider.floor_normal = wp.vec3(model.ground_plane.numpy()[:3])
        self._has_compliant_bodies = (
            len(collider.masses) > 0 and np.min(collider.masses.numpy()) < INF_MASS
        )

        self.collider_coms = wp.zeros(len(collider.meshes), dtype=wp.vec3)
        self.collider_inv_inertia = wp.zeros(len(collider.meshes), dtype=wp.mat33)

        # collider.floor_normal = wp.vec3(0.0)
        # collider.floor_normal[model.up_axis] = 1
        self.collider = collider

        # Warp.sim model
        self.model = model
        print("Particle count:", self.model.particle_count)

        self.stress_field = None
        self.velocity_field = None
        self.impulse_field = None

        self._collider_ids = None
        self._strain_matrix = sp.bsr_zeros(0, 0, mat63)
        self._transposed_strain_matrix = sp.bsr_zeros(0, 0, mat36)

        self.temporary_store = fem.TemporaryStore()
        fem.set_default_temporary_store(self.temporary_store)

    @staticmethod
    def enrich_state(state: State):
        state.particle_qd_grad = wp.zeros(state.particle_qd.shape[0], dtype=wp.mat33)
        state.particle_elastic_strain = wp.zeros(
            state.particle_qd.shape[0], dtype=wp.mat33
        )
        state.particle_transform = wp.empty(state.particle_qd.shape[0], dtype=wp.mat33)
        state.particle_transform.fill_(wp.mat33(np.eye(3)))

    @staticmethod
    def add_parser_arguments(parser):
        parser.add_argument("--density", type=float, default=1.0)
        parser.add_argument("--max_fraction", type=float, default=1.0)

        parser.add_argument("--compliance", type=float, default=0.0)
        parser.add_argument("--poisson", type=float, default=0.3)
        parser.add_argument("--friction", type=float, default=0.48)
        parser.add_argument("--yield_stress", "-ys", type=float, default=0.0)
        parser.add_argument(
            "--compression_yield_stress", "-cys", type=float, default=1.0e8
        )
        parser.add_argument(
            "--stretching_yield_stress", "-sys", type=float, default=1.0e8
        )
        parser.add_argument(
            "--unilateral", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--pad_grid", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument("--gs", action=argparse.BooleanOptionalAction, default=True)

        parser.add_argument("--max_iters", type=int, default=250)
        parser.add_argument("--tol", type=float, default=1.0e-5)
        parser.add_argument("--voxel_size", type=float, default=1.0)

    def simulate(
        self, state_0: State, state_1: State, dt: float, project_outside: bool = True
    ):
        fined_grained_timers = True
        timers_use_nvtx = True

        with wp.ScopedTimer(
            "Allocate grid",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            volume = allocate_by_voxels(
                state_0.particle_q, self.voxel_size, padded=self.pad_grid
            )
            grid = fem.Nanogrid(volume)

        domain = fem.Cells(grid)
        inv_cell_volume = 1.0 / self.voxel_size**3

        # Define function spaces: linear (Q1) for velocity and volume fraction,
        # piecewise-constant for pressure
        velocity_basis = fem.make_polynomial_basis_space(grid, degree=1)
        strain_basis = fem.make_polynomial_basis_space(
            grid,
            self.degree,
            # discontinuous=True,
            # element_basis=fem.ElementBasis.NONCONFORMING_POLYNOMIAL,
        )

        velocity_space = fem.make_collocated_function_space(
            velocity_basis, dtype=wp.vec3
        )
        fraction_space = fem.make_collocated_function_space(velocity_basis, dtype=float)
        full_strain_space = fem.make_collocated_function_space(
            strain_basis, dtype=wp.mat33
        )
        rotation_space = fem.make_collocated_function_space(
            strain_basis, dtype=wp.quatf
        )
        sym_strain_space = fem.make_collocated_function_space(
            strain_basis,
            dof_mapper=fem.SymmetricTensorMapper(
                dtype=wp.mat33, mapping=fem.SymmetricTensorMapper.Mapping.DB16
            ),
        )
        divergence_space = fem.make_collocated_function_space(strain_basis, dtype=float)

        with wp.ScopedTimer(
            "Create fields",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            velocity_test = fem.make_test(velocity_space)
            velocity_trial = fem.make_trial(velocity_space)
            fraction_test = fem.make_test(
                fraction_space,
                space_restriction=velocity_test.space_restriction,
            )

            sym_strain_test = fem.make_test(sym_strain_space)
            full_strain_test = fem.make_test(
                full_strain_space, space_restriction=sym_strain_test.space_restriction
            )

            velocity_field = velocity_space.make_field()
            self.velocity_field = velocity_field

            impulse_field = velocity_space.make_field()
            if self.impulse_field is not None:
                prev_impulse_field = fem.NonconformingField(domain, self.impulse_field)
                fem.interpolate(prev_impulse_field, dest=impulse_field)
            self.impulse_field = impulse_field

            # Interpolate previous stress
            stress_field = sym_strain_space.make_field()
            if self.stress_field is not None:
                prev_stress_field = fem.NonconformingField(domain, self.stress_field)
                fem.interpolate(prev_stress_field, dest=stress_field)
            self.stress_field = stress_field

            elastic_strain_field = full_strain_space.make_field()
            elastic_strain_delta_field = full_strain_space.make_field()
            int_symmetric_strain_field = sym_strain_space.make_field()
            elastic_rotation_field = rotation_space.make_field()

            collider_velocity_field = velocity_space.make_field()

        # Bin particles to grid cells
        with wp.ScopedTimer(
            "Bin particles",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            pic = fem.PicQuadrature(
                domain=domain,
                positions=state_0.particle_q,
                measures=self.model.particle_mass,
            )

        vel_node_count = velocity_space.node_count()
        strain_node_count = sym_strain_space.node_count()

        # Velocity right-hand side
        with wp.ScopedTimer(
            "Free velocity",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            velocity_int = fem.integrate(
                integrate_velocity,
                quadrature=pic,
                fields={"u": velocity_test},
                values={
                    "velocities": state_0.particle_qd,
                    "velocity_gradients": state_0.particle_qd_grad,
                    "dt": dt,
                    "gravity": self.model.gravity,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=wp.vec3,
            )
            particle_volume = fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": fraction_test},
                values={"inv_cell_volume": inv_cell_volume},
                output_dtype=float,
            )

            inv_mass_matrix = fem.borrow_temporary(
                self.temporary_store, shape=(vel_node_count,), dtype=float
            )

            fem.interpolate(
                free_velocity,
                dest=fem.make_restriction(
                    velocity_field, space_restriction=velocity_test.space_restriction
                ),
                values={
                    "velocity_int": velocity_int,
                    "particle_volume": particle_volume,
                    "inv_mass_matrix": inv_mass_matrix.array,
                },
            )

        with wp.ScopedTimer(
            "collider",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            # Accumulate collider volume so we can distribute mass
            # and record cells with collider to build subdomain

            collider_total_volumes = fem.borrow_temporary(
                self.temporary_store, shape=(self.collider.meshes.shape[0]), dtype=float
            )
            collider_total_volumes.array.zero_()

            collider_cells = fem.borrow_temporary(
                self.temporary_store, shape=(domain.element_count(),), dtype=int
            )
            collider_cells.array.zero_()

            collider_quadrature_order = self.degree + 1
            collider_quadrature = fem.RegularQuadrature(
                domain=domain,
                order=collider_quadrature_order,
                family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE,
            )
            fem.interpolate(
                collider_volume,
                quadrature=collider_quadrature,
                values={
                    "collider": self.collider,
                    "volumes": collider_total_volumes.array,
                    "cells": collider_cells.array,
                    "voxel_size": self.voxel_size,
                },
            )

            node_volume = fem.integrate(
                integrate_fraction,
                fields={"phi": fraction_test},
                values={"inv_cell_volume": inv_cell_volume},
                output_dtype=float,
            )

            collider_normal = fem.borrow_temporary(
                self.temporary_store, shape=(vel_node_count,), dtype=wp.vec3
            )
            collider_friction = fem.borrow_temporary(
                self.temporary_store, shape=(vel_node_count,), dtype=float
            )
            self._collider_ids = wp.empty(vel_node_count, dtype=int)

            collider_inv_mass_matrix = fem.borrow_temporary(
                self.temporary_store, shape=(vel_node_count,), dtype=float
            )

            fem.interpolate(
                collider_velocity,
                dest=fem.make_restriction(
                    collider_velocity_field,
                    space_restriction=velocity_test.space_restriction,
                ),
                values={
                    "particle_density": self.density,
                    "node_volume": node_volume,
                    "voxel_size": self.voxel_size,
                    "collider_volumes": collider_total_volumes.array,
                    "collider": self.collider,
                    "collider_inv_mass_matrix": collider_inv_mass_matrix.array,
                    "collider_normals": collider_normal.array,
                    "collider_friction": collider_friction.array,
                    "collider_ids": self._collider_ids,
                    "collider_impulse": self.impulse_field.dof_values,
                },
            )

        with wp.ScopedTimer(
            "Rigidity",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            rigidity_matrix = self.build_rigidity_matrix(
                collider_total_volumes.array, node_volume
            )

        # Scale plastic / elastic parameters with volume fraction
        with wp.ScopedTimer(
            "Parameter scaling",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            divergence_test = fem.make_test(
                divergence_space,
                space_restriction=sym_strain_test.space_restriction,
            )

            particle_volume = fem.borrow_temporary(
                self.temporary_store, shape=strain_node_count, dtype=float
            )
            node_volume = fem.borrow_temporary(
                self.temporary_store, shape=strain_node_count, dtype=float
            )
            node_collider_volume = fem.borrow_temporary(
                self.temporary_store, shape=strain_node_count, dtype=float
            )

            fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": divergence_test},
                values={"inv_cell_volume": inv_cell_volume},
                output=particle_volume.array,
            )
            fem.integrate(
                integrate_fraction,
                fields={"phi": divergence_test},
                values={"inv_cell_volume": inv_cell_volume},
                output=node_volume.array,
            )
            fem.integrate(
                integrate_collider_fraction,
                quadrature=fem.RegularQuadrature(
                    domain=domain,
                    order=collider_quadrature_order,
                    family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE,
                ),
                fields={"phi": divergence_test},
                values={
                    "collider": self.collider,
                    "inv_cell_volume": inv_cell_volume,
                },
                output=node_collider_volume.array,
            )

            scaled_yield_stress = fem.borrow_temporary(
                self.temporary_store, shape=strain_node_count, dtype=wp.vec3
            )
            scaled_stress_strain_mat = fem.borrow_temporary(
                self.temporary_store, shape=strain_node_count, dtype=mat66
            )

            wp.launch(
                kernel=scale_yield_stress_and_stress_matrices,
                dim=strain_node_count,
                inputs=[
                    self.yield_stresses / self.density,
                    self.stress_strain_mat,
                    particle_volume.array,
                    node_volume.array,
                    scaled_yield_stress.array,
                    scaled_stress_strain_mat.array,
                ],
            )

        with wp.ScopedTimer(
            "Strain rhs",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            fem.integrate(
                integrate_elastic_strain,
                quadrature=pic,
                fields={"tau": full_strain_test},
                values={
                    "elastic_strains": state_0.particle_elastic_strain,
                    "inv_cell_volume": inv_cell_volume,
                },
                output=elastic_strain_field.dof_values,
            )

            prev_symmetric_strain = wp.empty_like(int_symmetric_strain_field.dof_values)
            wp.launch(
                elastic_strain_rotation,
                dim=strain_node_count,
                inputs=[
                    elastic_strain_field.dof_values,
                    particle_volume.array,
                    elastic_rotation_field.dof_values,
                    int_symmetric_strain_field.dof_values,
                    prev_symmetric_strain,
                ],
            )

        # Strain matrix
        with wp.ScopedTimer(
            "Strain matrix",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            sp.bsr_set_zero(
                self._strain_matrix,
                rows_of_blocks=strain_node_count,
                cols_of_blocks=vel_node_count,
            )
            fem.integrate(
                strain_form,
                quadrature=pic,
                fields={
                    "u": velocity_trial,
                    "tau": sym_strain_test,
                },
                values={
                    "dt": dt,
                    "inv_cell_volume": inv_cell_volume,
                    "elastic_strain": prev_symmetric_strain,
                    "rotation": elastic_rotation_field.dof_values,
                },
                output_dtype=float,
                output=self._strain_matrix,
            )
            # self._strain_matrix.nnz_sync()

            self._strain_matrix.nnz_sync()

        if self.unilateral:
            with wp.ScopedTimer(
                "Offset",
                active=fined_grained_timers,
                use_nvtx=timers_use_nvtx,
                synchronize=True,
            ):
                wp.launch(
                    add_unilateral_strain_offset,
                    dim=strain_node_count,
                    inputs=[
                        self.max_fraction,
                        self.compliance,
                        particle_volume.array,
                        node_collider_volume.array,
                        node_volume.array,
                        prev_symmetric_strain,
                        int_symmetric_strain_field.dof_values,
                    ],
                )

        with wp.ScopedTimer(
            "Strain solve",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            color_offsets, color_indices = self._compute_coloring(
                grid, strain_node_count
            )

            solve_rheology(
                self.unilateral,
                self.friction_coeff,
                self.max_iterations,
                self.tolerance,
                self._strain_matrix,
                self._transposed_strain_matrix,
                inv_mass_matrix.array,
                scaled_yield_stress.array,
                scaled_stress_strain_mat.array,
                int_symmetric_strain_field.dof_values,
                stress_field.dof_values,
                velocity_field.dof_values,
                collider_friction.array,
                collider_normal.array,
                collider_velocity_field.dof_values,
                collider_inv_mass_matrix.array,
                self.impulse_field.dof_values,
                color_offsets=color_offsets,
                color_indices=None if color_indices is None else color_indices.array,
                rigidity_mat=rigidity_matrix,
                temporary_store=self.temporary_store,
            )

            if color_indices is not None:
                color_indices.release()

            full_strain = fem.integrate(
                strain_form,
                quadrature=pic,
                fields={
                    "u": velocity_field,
                    "tau": full_strain_test,
                },
                values={
                    "dt": dt,
                    "inv_cell_volume": inv_cell_volume,
                    "elastic_strain": prev_symmetric_strain,
                    "rotation": elastic_rotation_field.dof_values,
                },
                output_dtype=wp.mat33,
            )

            wp.launch(
                compute_elastic_strain_delta,
                dim=strain_node_count,
                inputs=[
                    particle_volume.array,
                    elastic_rotation_field.dof_values,
                    int_symmetric_strain_field.dof_values,
                    full_strain,
                    elastic_strain_field.dof_values,
                    elastic_strain_delta_field.dof_values,
                ],
            )

        # (A)PIC advection
        with wp.ScopedTimer(
            "Advection",
            active=fined_grained_timers,
            use_nvtx=timers_use_nvtx,
            synchronize=True,
        ):
            fem.interpolate(
                update_particles,
                quadrature=pic,
                values={
                    "pos": state_1.particle_q,
                    "pos_prev": state_0.particle_q,
                    "vel": state_1.particle_qd,
                    "vel_grad": state_1.particle_qd_grad,
                    "transform": state_1.particle_transform,
                    "transform_prev": state_0.particle_transform,
                    "dt": dt,
                    "elastic_strain_prev": state_0.particle_elastic_strain,
                    "elastic_strain": state_1.particle_elastic_strain,
                },
                fields={
                    "grid_vel": velocity_field,
                    "grid_strain": elastic_strain_field,
                    "grid_strain_delta": elastic_strain_delta_field,
                },
            )

            if project_outside:
                self.project_outside(state_1, dt)

    def build_rigidity_matrix(self, collider_volumes, node_volumes):
        if not self._has_compliant_bodies:
            return None

        vel_node_count = self.velocity_field.space.node_count()
        collider_count = self.collider.meshes.shape[0]

        J_rows = wp.empty(vel_node_count * 2, dtype=int)
        J_cols = wp.empty(vel_node_count * 2, dtype=int)
        J_values = wp.empty(vel_node_count * 2, dtype=wp.mat33)
        IJtm_values = wp.empty(vel_node_count * 2, dtype=wp.mat33)
        Iphi_diag = wp.empty(vel_node_count, dtype=wp.mat33)

        with wp.ScopedTimer("Fill rigidity matrix", synchronize=True, active=False):
            wp.launch(
                fill_collider_rigidity_matrices,
                dim=vel_node_count,
                inputs=[
                    self.velocity_field.space.node_positions(),
                    collider_volumes,
                    node_volumes,
                    self.collider,
                    self.voxel_size,
                    self._collider_ids,
                    self.collider_coms,
                    self.collider_inv_inertia,
                    J_rows,
                    J_cols,
                    J_values,
                    IJtm_values,
                    Iphi_diag,
                ],
            )

        with wp.ScopedTimer("Build rigidity matrix", synchronize=True, active=False):
            J = sp.bsr_from_triplets(
                rows_of_blocks=vel_node_count,
                cols_of_blocks=2 * collider_count,
                rows=J_rows,
                columns=J_cols,
                values=J_values,
            )

            IJtm = sp.bsr_from_triplets(
                cols_of_blocks=vel_node_count,
                rows_of_blocks=2 * collider_count,
                columns=J_rows,
                rows=J_cols,
                values=IJtm_values,
            )

        with wp.ScopedTimer("Assemble rigidity matrix", synchronize=True, active=False):
            Iphi = sp.bsr_diag(Iphi_diag)
            Iphi = sp.bsr_from_triplets(
                rows_of_blocks=Iphi.nrow,
                cols_of_blocks=Iphi.ncol,
                columns=Iphi.columns,
                rows=Iphi.uncompress_rows(),
                values=Iphi.values,
            )
            rigid = Iphi + J @ IJtm
            rigid.nnz_sync()

        return rigid

    def collect_collider_impulses(self):
        x = self.velocity_field.space.node_positions()

        collider_impulse = wp.zeros_like(self.impulse_field.dof_values)
        cell_volume = self.voxel_size**3
        fem.utils.array_axpy(
            y=collider_impulse,
            x=self.impulse_field.dof_values,
            alpha=-self.density * cell_volume,
            beta=0.0,
        )

        return self._collider_ids, collider_impulse, x

    def project_outside(self, state: State, dt: float):
        wp.launch(
            project_outside_collider,
            dim=state.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.particle_qd_grad,
                self.collider,
                self.voxel_size,
                dt,
            ],
        )

    def extract_rotation_and_scales(self, state: State):
        particle_transform_rotation = wp.empty(state.particle_count, dtype=wp.vec4h)
        particle_transform_scales = wp.empty(state.particle_count, dtype=wp.vec3)

        wp.launch(
            extract_rotation_and_scales,
            dim=state.particle_count,
            inputs=[
                state.particle_transform,
                particle_transform_rotation,
                particle_transform_scales,
            ],
        )

        return particle_transform_rotation, particle_transform_scales

    def sample_grains(
        self, state: State, particle_radius: float, grains_per_particle: int
    ):
        grains = wp.empty((state.particle_count, grains_per_particle), dtype=wp.vec3)

        wp.launch(
            sample_grains,
            dim=grains.shape,
            inputs=[
                state.particle_q,
                particle_radius,
                grains,
            ],
        )

        return grains

    def update_grains(
        self,
        state_prev: State,
        state: State,
        grains: wp.array,
        particle_radius: float,
        dt: float,
    ):
        if self.velocity_field is None:
            return

        grain_pos = grains.flatten()
        domain = fem.Cells(self.velocity_field.space.geometry)
        grain_pic = fem.PicQuadrature(domain, positions=grain_pos)

        # wp.launch(
        #     transform_grains,
        #     dim=grains.shape,
        #     inputs=[
        #         state_prev.particle_q,
        #         state_prev.particle_transform,
        #         state.particle_q,
        #         state.particle_transform,
        #         grains,
        #     ],
        # )

        wp.launch(
            advect_grains_from_particles,
            dim=grains.shape,
            inputs=[
                dt,
                state_prev.particle_q,
                state.particle_q,
                state.particle_qd_grad,
                grains,
            ],
        )

        fem.interpolate(
            advect_grains,
            quadrature=grain_pic,
            values={
                "dt": dt,
                "positions": grain_pos,
            },
            fields={
                "grid_vel": self.velocity_field,
            },
        )

        wp.launch(
            project_grains,
            dim=grains.shape,
            inputs=[
                particle_radius,
                state.particle_q,
                state.particle_transform,
                grains,
            ],
        )

    def _compute_coloring(self, grid, strain_node_count):
        if not self.coloring:
            return None, None

        colors = fem.borrow_temporary(
            self.temporary_store, shape=strain_node_count * 2, dtype=int
        )
        color_indices = fem.borrow_temporary(
            self.temporary_store, shape=strain_node_count * 2, dtype=int
        )

        if self.degree == 1:
            voxels = grid.vertex_grid.get_voxels()
            wp.launch(
                node_color_27_stencil,
                dim=strain_node_count,
                inputs=[voxels, colors.array, color_indices.array],
            )
        else:
            voxels = grid.cell_grid.get_voxels()
            wp.launch(
                node_color_8_stencil,
                dim=strain_node_count,
                inputs=[voxels, colors.array, color_indices.array],
            )

        wp.utils.radix_sort_pairs(
            keys=colors.array,
            values=color_indices.array,
            count=strain_node_count,
        )

        unique_colors = colors.array[strain_node_count:]
        color_node_counts = color_indices.array[strain_node_count:]
        color_count = wp.utils.runlength_encode(
            colors.array,
            run_values=unique_colors,
            run_lengths=color_node_counts,
            value_count=strain_node_count,
        )

        color_offsets = np.concatenate(
            [[0], np.cumsum(color_node_counts[:color_count].numpy())]
        )

        return color_offsets, color_indices
