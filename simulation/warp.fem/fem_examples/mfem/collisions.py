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
from typing import Any, List
from weakref import ReferenceType

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem.geometry.closest_point import project_on_tri_at_origin

from .softbody_sim import DisplacementPotential, SoftbodySim


class CollisionHandler:
    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--collision_stiffness",
            "-ck",
            type=float,
            default=1.0,
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
            default=2.0,
            help="Multiplier of collision radius for detection",
        )
        parser.add_argument("--friction", "-mu", type=float, default=0.2)
        parser.add_argument(
            "--friction_reg",
            "-mur",
            type=float,
            default=0.1,
            help="Regularization coefficient for friction",
        )
        parser.add_argument(
            "--friction_fluid",
            "-nu",
            type=float,
            default=0.01,
            help="Additional fluid friction ratio to convexify things",
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

    def __init__(
        self,
        kinematic_meshes: List[wp.Mesh],
        cp_cell_indices,
        cp_cell_coords,
    ):
        self.warp_meshes = kinematic_meshes
        self.cp_cell_indices = cp_cell_indices
        self.cp_cell_coords = cp_cell_coords

        self.collision_quadrature = None
        self.n_contact = 0

    def init_collision_detector(self, sim: ReferenceType[SoftbodySim]):
        self.args = sim.args
        self.sim = sim  # weakref

        n_cp = self.cp_cell_indices.shape[0]
        collision_quadrature = fem.PicQuadrature(
            domain=sim.vel_quadrature.domain,
            positions=(self.cp_cell_indices, self.cp_cell_coords),
            measures=wp.ones(n_cp, dtype=float),
        )
        self.set_collision_quadrature(collision_quadrature)
        self.n_contact = 0

        max_contacts = 10 * self.cp_cell_indices.shape[0]
        self.collision_indices_a = wp.empty(max_contacts, dtype=int)
        self.collision_indices_b = wp.empty(max_contacts, dtype=int)
        self.collision_normals = wp.empty(max_contacts, dtype=wp.vec3)
        self.collision_kinematic_gaps = wp.empty(max_contacts, dtype=wp.vec3)

        jac_cols = sim.u_field.space_partition.node_count()
        self._collision_jacobian_a = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
        self._collision_jacobian_b = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)

        self._collision_jacobian = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
        self._collision_jacobian_t = sp.bsr_zeros(jac_cols, 0, block_type=wp.mat33)

        self._HtH_work_arrays = sp.bsr_mm_work_arrays()
        self._HbHa_work_arrays = sp.bsr_axpy_work_arrays()

        self._collision_stiffness = self.args.collision_stiffness * self.args.density / n_cp

    def set_collision_quadrature(self, quadrature: fem.PicQuadrature):
        self.collision_quadrature = quadrature

    def add_collision_energy(self, E: wp.array):
        if self.n_contact == 0:
            return E

        cp_du = self._sample_cp_displacement(self.sim.du_field)
        col_energies = wp.empty(self.n_contact, dtype=float)
        wp.launch(
            collision_energy,
            dim=self.n_contact,
            inputs=[
                self.args.collision_radius,
                self.args.friction,
                self.args.dt * self.args.friction_reg,
                self.args.friction_fluid * self.args.friction_reg,
                cp_du,
                self.collision_kinematic_gaps,
                self.collision_normals,
                self.collision_indices_a,
                self.collision_indices_b,
                col_energies,
            ],
        )
        Ec = wp.empty_like(E)
        wp.utils.array_sum(col_energies, out=Ec)

        fem.utils.array_axpy(x=Ec, y=E, alpha=self._collision_stiffness, beta=1.0)

        return E

    def add_collision_hessian(self, lhs: wp.array):
        # contacts
        if self.n_contact == 0:
            return lhs

        H = self._collision_jacobian
        Ht = self._collision_jacobian_t
        wp.launch(
            bsr_mul_diag,
            dim=(Ht.nnz_sync(), Ht.block_shape[0]),
            inputs=[Ht.scalar_values, Ht.columns, self._col_energy_hessian],
        )

        sp.bsr_mm(
            x=Ht,
            y=H,
            z=lhs,
            alpha=self._collision_stiffness,
            beta=1.0,
            work_arrays=self._HtH_work_arrays,
        )

        return lhs

    def add_collision_forces(self, rhs: wp.array):
        if self.n_contact == 0:
            return rhs

        # contacts
        sp.bsr_mv(
            A=self._collision_jacobian_t,
            x=self._col_energy_gradients,
            y=rhs,
            alpha=-self._collision_stiffness,
            beta=1.0,
        )

        return rhs

    def prepare_newton_step(self, dt: float):
        self.detect_collisions(dt)
        self.build_collision_jacobian()

        # compute per-contact forces and hessian
        n_contact = self.n_contact
        if n_contact > 0:
            self._col_energy_gradients = wp.empty(n_contact, dtype=wp.vec3)
            self._col_energy_hessian = wp.empty(n_contact, dtype=wp.mat33)
            cp_du = self._sample_cp_displacement(self.sim.du_field)

            wp.launch(
                collision_gradient_and_hessian,
                dim=n_contact,
                inputs=[
                    self.args.collision_radius,
                    self.args.friction,
                    dt * self.args.friction_reg,
                    self.args.friction_fluid * self.args.friction_reg,
                    cp_du,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b,
                    self._col_energy_gradients,
                    self._col_energy_hessian,
                ],
            )

    def cp_world_position(self, dest=None):
        cp_pic = self.collision_quadrature

        if cp_pic is None:
            if dest is None:
                dest = wp.array([], dtype=wp.vec3)
            else:
                dest.assign([])
            return dest

        if dest is None:
            dest = wp.empty(cp_pic.total_point_count(), dtype=wp.vec3)
        fem.interpolate(
            world_position,
            fields={"u": self.sim.u_field},
            dest=dest,
            quadrature=cp_pic,
        )

        return dest

    def _sample_cp_displacement(self, du_field, dest=None):
        cp_pic = self.collision_quadrature
        if dest is None:
            dest = wp.empty(cp_pic.total_point_count(), dtype=wp.vec3)
        fem.interpolate(
            du_field,
            dest=dest,
            quadrature=cp_pic,
        )
        return dest

    def detect_collisions(self, dt):
        max_contacts = self.collision_normals.shape[0]

        count = wp.zeros(1, dtype=int)
        indices_a = self.collision_indices_a
        indices_b = self.collision_indices_b
        normals = self.collision_normals
        kinematic_gaps = self.collision_kinematic_gaps

        self.run_collision_detectors(
            dt,
            count,
            indices_a,
            indices_b,
            normals,
            kinematic_gaps,
        )

        self.n_contact = int(count.numpy()[0])

        if self.n_contact > max_contacts:
            print("Warning: contact buffer size exceeded, some have bee ignored")
            self.n_contact = max_contacts

    def run_collision_detectors(
        self,
        dt,
        count,
        indices_a,
        indices_b,
        normals,
        kinematic_gaps,
    ):
        cp_pic = self.collision_quadrature
        n_cp = cp_pic.total_point_count()
        max_contacts = self.collision_normals.shape[0]

        cp_cur_pos = self.cp_world_position()
        cp_du = self._sample_cp_displacement(self.sim.du_field)

        collision_radius = self.args.collision_radius * self.args.collision_detection_ratio

        if self.args.ground:
            ground_height = self.args.ground_height
            wp.launch(
                detect_ground_collisions,
                dim=n_cp,
                inputs=[
                    max_contacts,
                    self.args.up_axis,
                    cp_cur_pos,
                    cp_du,
                    collision_radius,
                    ground_height,
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
                    dt,
                    mesh_ids,
                    cp_cur_pos,
                    cp_du,
                    collision_radius,
                    count,
                    normals,
                    kinematic_gaps,
                    indices_a,
                    indices_b,
                ],
            )

    def build_collision_quadratures(self):
        n_contact = self.n_contact
        a_cells = wp.empty(n_contact, dtype=int)
        a_coords = wp.empty(n_contact, dtype=wp.vec3)
        b_cells = wp.empty(n_contact, dtype=int)
        b_coords = wp.empty(n_contact, dtype=wp.vec3)
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.collision_quadrature.cell_indices,
                self.collision_quadrature.particle_coords,
                self.collision_indices_a,
                a_cells,
                a_coords,
            ],
        )
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.collision_quadrature.cell_indices,
                self.collision_quadrature.particle_coords,
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

        return a_contact_pic, b_contact_pic

    def build_collision_jacobian(self):
        n_contact = self.n_contact

        # Build collision jacobian
        # (derivative of collision gap `pos_a - pos_b` w.r.t. degrees of freedom)

        if n_contact == 0:
            return

        a_contact_pic, b_contact_pic = self.build_collision_quadratures()

        u_trial = fem.make_trial(self.sim.u_field.space, space_partition=self.sim.u_field.space_partition)

        sp.bsr_set_zero(
            self._collision_jacobian_a,
            n_contact,
            self.sim.u_field.space_partition.node_count(),
        )
        fem.interpolate(
            u_trial,
            quadrature=a_contact_pic,
            dest=self._collision_jacobian_a,
            kernel_options={"enable_backward": False},
        )

        sp.bsr_set_zero(
            self._collision_jacobian_b,
            n_contact,
            self.sim.u_field.space_partition.node_count(),
        )
        fem.interpolate(
            u_trial,
            quadrature=b_contact_pic,
            dest=self._collision_jacobian_b,
            kernel_options={"enable_backward": False},
        )

        self._collision_jacobian_a.nnz_sync()
        self._collision_jacobian_b.nnz_sync()

        sp.bsr_assign(self._collision_jacobian, src=self._collision_jacobian_a)
        sp.bsr_axpy(
            x=self._collision_jacobian_b,
            y=self._collision_jacobian,
            alpha=-1,
            beta=1,
            work_arrays=self._HbHa_work_arrays,
        )

        sp.bsr_set_transpose(dest=self._collision_jacobian_t, src=self._collision_jacobian)


class MeshSelfCollisionHandler(CollisionHandler):
    def __init__(
        self,
        vtx_quadrature: fem.PicQuadrature,
        tri_mesh: wp.Mesh,
    ):
        super().__init__([], vtx_quadrature.cell_indices, vtx_quadrature.particle_coords)

        self.tri_vtx_quadrature = vtx_quadrature
        self.vtx_rest_pos = wp.clone(tri_mesh.points)
        self.tri_mesh = tri_mesh

    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        CollisionHandler.add_parser_arguments(parser)
        parser.add_argument(
            "--self_immunity_radius_ratio",
            "-cs",
            type=float,
            default=4.0,
            help="Ignore self-collision for particles that were within this ratio at rest",
        )

    def run_collision_detectors(
        self,
        dt,
        count,
        indices_a,
        indices_b,
        normals,
        kinematic_gaps,
    ):
        self.set_collision_quadrature(self.tri_vtx_quadrature)

        super().run_collision_detectors(
            dt,
            count,
            indices_a,
            indices_b,
            normals,
            kinematic_gaps,
        )
        self.cp_world_position(dest=self.tri_mesh.points)
        self.tri_mesh.refit()

        cp_du = self._sample_cp_displacement(self.sim.du_field)

        n_cp = cp_du.shape[0]
        max_contacts = self.collision_normals.shape[0]

        collision_radius = self.args.collision_radius * self.args.collision_detection_ratio

        start_contacts = count.numpy()[0]
        pos_b = wp.empty(indices_b.shape, dtype=wp.vec3)

        wp.launch(
            detect_mesh_self_collisions,
            dim=(n_cp),
            inputs=[
                start_contacts,
                max_contacts,
                dt,
                self.args.self_immunity_radius_ratio,
                self.tri_mesh.id,
                self.vtx_rest_pos,
                cp_du,
                collision_radius,
                count,
                normals,
                kinematic_gaps,
                indices_a,
                indices_b,
                pos_b,
            ],
        )
        self_contacts = int(min(max_contacts, count.numpy()[0]) - start_contacts)

        if self_contacts > 0:
            contact_points = wp.empty(n_cp + self_contacts, dtype=wp.vec3)
            wp.copy(contact_points[:n_cp], self.vtx_rest_pos)
            wp.copy(contact_points[n_cp:], pos_b[:self_contacts])

            quadrature = fem.PicQuadrature(
                fem.Cells(self.sim.geo),
                contact_points,
                max_dist=self.sim.typical_length,
            )
            self.set_collision_quadrature(quadrature)


class CollisionPotential(DisplacementPotential):
    def __init__(self, sim: SoftbodySim, collision_handler: CollisionHandler):
        super().__init__(sim)
        self.collision_handler = collision_handler

    def prepare_frame(self, dt):
        self.collision_handler.detect_collisions(dt)

    def add_energy(self, E_u):
        self.collision_handler.add_collision_energy(E_u)

    def add_hessian(self, lhs: sp.BsrMatrix):
        self.collision_handler.add_collision_hessian(lhs)

    def add_forces(self, rhs: wp.array, tape: wp.Tape = None):
        self.collision_handler.add_collision_forces(rhs)

    def prepare_newton_step(self, dt, tape=None):
        self.collision_handler.prepare_newton_step(dt)

    def init_constant_forms(self):
        self.collision_handler.init_collision_detector(self.sim)


@wp.kernel
def bsr_mul_diag(
    Bt_values: wp.array3d(dtype=float),
    Bt_columns: wp.array(dtype=int),
    C_values: wp.array(dtype=Any),
):
    i, r = wp.tid()
    col = Bt_columns[i]

    C = C_values[col]

    Btr = Bt_values[i, r]
    BtC = wp.vec3(Btr[0], Btr[1], Btr[2]) @ C
    for k in range(3):
        Btr[k] = BtC[k]


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

        nor = wp.vec3()
        nor[up_axis] = 1.0

        normals[idx] = nor
        kinematic_gaps[idx] = (wp.dot(x - du_cur[i], nor) - ground_height) * nor
        indices_a[idx] = i
        indices_b[idx] = fem.NULL_QP_INDEX


@wp.kernel
def detect_mesh_collisions(
    max_contacts: int,
    dt: float,
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

    query = wp.mesh_query_point(mesh_id, x, radius)

    if query.result:
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)

        delta = x - cp
        dist = wp.length(delta) * query.sign

        if dist < radius:
            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts:
                return

            if dist < 0.00001:
                n = wp.mesh_eval_face_normal(mesh_id, query.face)
            else:
                n = wp.normalize(delta) * query.sign
            normals[idx] = n

            v = wp.mesh_eval_velocity(mesh_id, query.face, query.u, query.v)

            kinematic_gap = (dist - wp.dot(du_cur[tid], n)) * n - v * dt
            kinematic_gaps[idx] = kinematic_gap
            indices_a[idx] = tid
            indices_b[idx] = fem.NULL_QP_INDEX


@wp.kernel
def detect_mesh_self_collisions(
    cur_contacts: int,
    max_contacts: int,
    dt: float,
    self_immunity_ratio: float,
    mesh_id: wp.uint64,
    mesh_rest_pos: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    pos_b: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    mesh = wp.mesh_get(mesh_id)

    x = mesh.points[tid]

    lower = x - wp.vec3(radius)
    upper = x + wp.vec3(radius)

    query = wp.mesh_query_aabb(mesh_id, lower, upper)

    face_index = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face_index):
        t0 = mesh.indices[3 * face_index + 0]
        t1 = mesh.indices[3 * face_index + 1]
        t2 = mesh.indices[3 * face_index + 2]
        if tid == t0 or tid == t1 or tid == t2:
            # Fast self collision
            continue

        u1 = mesh.points[t0]
        u2 = mesh.points[t1]
        u3 = mesh.points[t2]

        d, bary = project_on_tri_at_origin(x - u1, u2 - u1, u3 - u1)
        if wp.max(bary) >= 1.0 or wp.min(bary) <= 0.0:
            # point not in interior, ignore
            continue

        cp = bary[0] * u1 + bary[1] * u2 + bary[2] * u3
        delta = x - cp

        face_nor = wp.mesh_eval_face_normal(mesh_id, face_index)
        sign = wp.where(wp.dot(delta, face_nor) > 0.0, 1.0, -1.0)

        dist = wp.length(delta) * sign

        if dist < radius:
            # discard self-collisions of points that were very close at rest
            rp0 = mesh_rest_pos[t0]
            rp1 = mesh_rest_pos[t1]
            rp2 = mesh_rest_pos[t2]
            xb_rest = bary[0] * rp0 + bary[1] * rp1 + bary[2] * rp2
            xa_rest = mesh_rest_pos[tid]
            if wp.length(xb_rest - xa_rest) < self_immunity_ratio * radius:
                continue

            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts:
                return

            if dist < 0.00001:
                n = face_nor
            else:
                n = wp.normalize(delta) * sign
            normals[idx] = n

            du0 = du_cur[t0]
            du1 = du_cur[t1]
            du2 = du_cur[t2]
            du = du_cur[tid] - du0 * bary[0] - du1 * bary[1] - du2 * bary[2]

            kinematic_gap = (dist - wp.dot(du, n)) * n
            kinematic_gaps[idx] = kinematic_gap
            indices_a[idx] = tid
            indices_b[idx] = mesh.points.shape[0] + idx - cur_contacts
            pos_b[idx - cur_contacts] = xb_rest


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

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    stick = wp.where(d_hat < 1.0, 1.0, 0.0)
    gap = d_hat - 1.0
    E = 0.5 * stick * gap * gap

    vt = (offset - d * nor) / dt  # tangential velocity
    vt_norm = wp.length(vt)

    mu_fn = -mu * wp.min(0.0, gap) / rc  # yield force

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

    energies[c] = E


@wp.kernel
def collision_gradient_and_hessian(
    radius: float,
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

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    stick = wp.where(d_hat < 1.0, 1.0, 0.0)

    dE_d_hat = d_hat - 1.0
    gradient[c] = dE_d_hat * stick / rc * nor
    hessian[c] = wp.outer(nor, nor) * stick / (rc * rc)

    vt = (offset - d * nor) / dt  # tangential velocity
    vt_norm = wp.length(vt)
    vt_dir = wp.normalize(vt)  # avoids dealing with 0

    mu_fn = -mu * wp.min(0.0, dE_d_hat) / rc  # yield force

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
        mu_fn / dt * ((df1_d_vtn + nu) * wp.outer(vt_dir, vt_dir) + (f1_over_vt_norm + nu) * wp.outer(vt_perp, vt_perp))
    )


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
def world_position(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    return domain(s) + u(s)
