# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
from vomp.fem.scripts.example_cutting.embedded_sim_utils import (
    flexicubes_from_sdf_grid,
    sim_from_flexicubes,
    surface_positions,
)
from kaolin.io import import_mesh

from vomp.fem.fem_examples.mfem.softbody_sim import ClassicFEM
from vomp.fem.fem_examples.mfem.collisions import CollisionHandler

import warp as wp
import warp.fem as fem
from warp.sim.collide import triangle_closest_point, TRI_CONTACT_FEATURE_FACE_INTERIOR

torch.cuda.set_per_process_memory_fraction(0.5)


def load_mesh(path):
    """Load and normalize an obj mesh from path"""

    mesh = import_mesh(path, triangulate=True).cuda()
    # normalize to [-1, 1]
    half_bbox = (
        0.5 * torch.min(mesh.vertices, dim=0)[0],
        0.5 * torch.max(mesh.vertices, dim=0)[0],
    )
    normalized_vertices = (mesh.vertices - half_bbox[0] - half_bbox[1]) / torch.max(
        half_bbox[1] - half_bbox[0] + 0.001
    )
    return wp.Mesh(
        wp.from_torch(normalized_vertices, dtype=wp.vec3),
        wp.from_torch(mesh.faces.flatten().to(torch.int32)),
        support_winding_number=True,
    )


@fem.integrand
def fixed_points_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    u: fem.Field,
    v: fem.Field,
):
    """Dirichlet boundary condition projector

    Here we simply clamp points near Z boundaries
    """

    y = domain(s)

    clamped = wp.where(wp.abs(y[1]) > 0.9, 1.0, 0.0)

    return wp.dot(u(s), v(s)) * clamped


@wp.kernel
def world_to_rest_pose_kernel(
    mesh: wp.uint64,
    rest_points: wp.array(dtype=wp.vec3),
    pos: wp.vec3,
    out: wp.array(dtype=wp.vec3),
):
    """
    Converts a point on the deformed surface to its rest-pose counterpart
    """

    max_dist = 1.0
    query = wp.mesh_query_point_no_sign(mesh, pos, max_dist)

    if query.result:
        faces = wp.mesh_get(mesh).indices
        v0 = rest_points[faces[3 * query.face + 0]]
        v1 = rest_points[faces[3 * query.face + 1]]
        v2 = rest_points[faces[3 * query.face + 2]]

        p = v0 + query.u * (v1 - v0) + query.v * (v2 - v0)

    else:
        p = pos

    out[0] = p


@wp.kernel
def mesh_sdf_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    sdf: wp.array(dtype=float),
):
    """Builds an SDF using mesh closest-point queries"""

    i = wp.tid()
    pos = points[i]

    max_dist = 1.0
    query = wp.mesh_query_point_sign_winding_number(mesh, pos, max_dist)

    if query.result:
        mesh_pos = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        sdf[i] = query.sign * wp.length(pos - mesh_pos)
    else:
        sdf[i] = 1.0


@wp.kernel
def detect_self_collisions(
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

        cp, bary, feature_type = triangle_closest_point(u1, u2, u3, x)
        if feature_type != TRI_CONTACT_FEATURE_FACE_INTERIOR:
            continue

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


class SelfCollisionHandler(CollisionHandler):
    def __init__(
        self,
        vtx_quadrature: fem.PicQuadrature,
        tri_mesh: wp.Mesh,
        sim: ClassicFEM,
    ):
        super().__init__(
            [], vtx_quadrature.cell_indices, vtx_quadrature.particle_coords, sim
        )

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

        collision_radius = (
            self.args.collision_radius * self.args.collision_detection_ratio
        )

        start_contacts = count.numpy()[0]
        pos_b = wp.empty(indices_b.shape, dtype=wp.vec3)

        wp.launch(
            detect_self_collisions,
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

            quadrature = fem.PicQuadrature(fem.Cells(self.sim.geo.base), contact_points)
            quadrature.domain = self.collision_quadrature.domain
            self.set_collision_quadrature(quadrature)


class SelfCollidingSim(ClassicFEM):
    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        ClassicFEM.add_parser_arguments(parser)
        SelfCollisionHandler.add_parser_arguments(parser)

    def compute_initial_guess(self):
        self.du_field.dof_values.zero_()
        self.collision_handler.detect_collisions(self.dt)

    def evaluate_energy(self):
        E_p, c_r = super().evaluate_energy()

        E_p = self.collision_handler.add_collision_energy(E_p)

        return E_p, c_r

    def newton_lhs(self):
        lhs = super().newton_lhs()
        self.collision_handler.add_collision_hessian(lhs)
        fem.dirichlet.project_system_matrix(lhs, self.v_bd_matrix)

        return lhs

    def newton_rhs(self, tape=None):
        rhs = super().newton_rhs(tape)
        self.collision_handler.add_collision_forces(rhs)
        self._filter_forces(rhs, tape=tape)
        return rhs

    def prepare_newton_step(self, tape=None):
        self.collision_handler.prepare_newton_step(self.dt)

        return super().prepare_newton_step(tape)

    def init_collision_detector(
        self,
        vtx_quadrature: fem.PicQuadrature,
        tri_mesh: wp.Mesh,
    ):
        self.collision_handler = SelfCollisionHandler(vtx_quadrature, tri_mesh, self)


class Clay:
    """Utility struct for storing simulation state"""

    def __init__(self, geo):
        self.geo = geo

        self.sim = None
        self.tri_mesh = None
        self.tri_vtx_quadrature = None
        self.rest_points = None

    def create_sim(self, flexicubes):
        if self.sim is not None:
            # save previous displacement and velocity
            prev_displacement_field = self.sim.u_field.space.make_field()
            prev_velocity_field = self.sim.du_field.space.make_field()
            fem.interpolate(self.sim.u_field, dest=prev_displacement_field)
            fem.interpolate(self.sim.du_field, dest=prev_velocity_field)
            prev_displacement = prev_displacement_field.dof_values
            prev_velocity = prev_velocity_field.dof_values
        else:
            prev_displacement = None
            prev_velocity = None

        # (Re)create simulation
        self.sim = sim_from_flexicubes(SelfCollidingSim, flexicubes, geo, args)
        self.sim.set_fixed_points_condition(
            fixed_points_projector_form,
        )
        self.sim.init_constant_forms()
        self.sim.project_constant_forms()

        # Interpolate back previous displacement
        if prev_displacement is not None:
            prev_displacement_field = self.sim.u_field.space.make_field()
            prev_displacement_field.dof_values = prev_displacement
            prev_velocity_field = self.sim.du_field.space.make_field()
            prev_velocity_field.dof_values = prev_velocity
            fem.interpolate(prev_displacement_field, dest=self.sim.u_field)
            fem.interpolate(prev_velocity_field, dest=self.sim.du_field)

        # Embed triangle mesh
        tri_vertices = flexicubes["tri_vertices"]
        tri_faces = wp.array(flexicubes["tri_faces"], dtype=int).flatten()
        tri_vtx_pos = wp.array(tri_vertices, dtype=wp.vec3)

        self.tri_mesh = wp.Mesh(tri_vtx_pos, tri_faces)
        self.rest_points = wp.clone(tri_vtx_pos)
        self.surface_vtx_quadrature = fem.PicQuadrature(fem.Cells(geo), tri_vtx_pos)
        self.surface_vtx_quadrature.domain = self.sim.u_test.domain

        self.sim.init_collision_detector(self.surface_vtx_quadrature, self.tri_mesh)

    def world_to_rest_pos(self, world_pos):
        tri_mesh = self.tri_mesh

        rest_pos = wp.empty(1, dtype=wp.vec3)
        tri_mesh.refit()
        wp.launch(
            world_to_rest_pose_kernel,
            dim=1,
            inputs=[tri_mesh.id, self.rest_points, world_pos, rest_pos],
        )
        return rest_pos


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})
    wp.set_module_options({"fast_math": True})

    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--quadrature_model", "-qm", default=None)
    parser.add_argument("--force_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=64)

    SelfCollidingSim.add_parser_arguments(parser)
    args = parser.parse_args()

    # fall back to full-cell quadrature if neural model not provided
    args.reg_qp = 2 if args.quadrature_model is None else 0
    args.clip = False

    args.ground_height = -1.0
    args.collision_radius = 0.5 / args.resolution

    res = args.resolution
    geo = fem.Grid3D(res=wp.vec3i(res), bounds_lo=wp.vec3(-1), bounds_hi=wp.vec3(1))

    # sample mesh SDF on grid nodes
    source_mesh = load_mesh(args.mesh)
    grid_node_pos = fem.make_polynomial_space(geo).node_positions()
    grid_sdf = wp.empty(grid_node_pos.shape[0], dtype=float)
    wp.launch(
        mesh_sdf_kernel,
        dim=grid_node_pos.shape,
        inputs=[source_mesh.id, grid_node_pos, grid_sdf],
    )
    grid_sdf = wp.to_torch(grid_sdf)
    grid_node_pos = wp.to_torch(grid_node_pos)

    # Create
    flexicubes = flexicubes_from_sdf_grid(
        res, pos=grid_node_pos, sdf=grid_sdf, sdf_grad_func=None
    )

    # Create simulation
    clay = Clay(geo)
    clay.create_sim(flexicubes)

    # Add hooks for displaying surface and run sim

    def init_surface(flexicubes):
        tri_vertices = flexicubes["tri_vertices"]
        tri_faces = flexicubes["tri_faces"]

        surface = ps.register_surface_mesh("surf", tri_vertices, tri_faces)
        surface.set_edge_width(1.0)

    prev_world_pos = None

    # user interface callback
    def callback():
        global grid_sdf, prev_world_pos, force_center_quadrature

        io = psim.GetIO()

        if io.KeyMods in (1, psim.ImGuiModFlags_Ctrl):
            # ctrl + mouse: update SDF values

            if io.MouseDown[0]:
                sign = -1.0  # left-mouse, add material
            elif io.MouseDown[1]:
                sign = 1.0  # right-mouse, remove material
            else:
                sign = 0.0

            if sign != 0.0:
                screen_coords = io.MousePos

                # Convert clicked position to rest pose
                world_pos = ps.screen_coords_to_world_position(screen_coords)
                if np.all(np.isfinite(world_pos)):
                    prev_world_pos = world_pos
                elif prev_world_pos is not None:
                    world_pos = prev_world_pos

                rest_pos = wp.to_torch(clay.world_to_rest_pos(world_pos))

                # locally update sdf values
                delta_pos_sq = torch.sum(
                    (grid_node_pos - rest_pos) * (grid_node_pos - rest_pos), dim=1
                )
                delta_sdf = torch.exp(-0.25 * delta_pos_sq * res * res)

                grid_sdf += 50.0 * sign / res * delta_sdf

                # rebuilds flexicubes structure and recreate sim
                flexicubes = flexicubes_from_sdf_grid(res, grid_sdf, grid_node_pos)
                clay.create_sim(flexicubes)
                init_surface(flexicubes)

                io.WantCaptureMouse = True

        sim = clay.sim

        # run one frame of simulation
        sim.run_frame()

        # Interpolate deformation back to vertices
        fem.interpolate(
            surface_positions,
            quadrature=clay.surface_vtx_quadrature,
            dest=clay.tri_mesh.points,
            fields={"displacement": sim.u_field},
        )
        surf_mesh = ps.get_surface_mesh("surf")
        surf_mesh.update_vertex_positions(clay.tri_mesh.points.numpy())

        ps.register_point_cloud(
            "CP",
            sim.collision_handler.cp_world_position().numpy()[
                clay.tri_mesh.points.shape[0] :
            ],
        )

        # Dynamic picking force
        # (shift + click)

        if io.KeyMods in (2, psim.ImGuiModFlags_Shift):  # shift
            if io.MouseClicked[0]:
                screen_coords = io.MousePos
                world_ray = ps.screen_coords_to_world_ray(screen_coords)
                world_pos = ps.screen_coords_to_world_position(screen_coords)

                if np.all(np.isfinite(world_pos)):
                    rest_pos = clay.world_to_rest_pos(world_pos)

                    # update force application point
                    sim.forces.count = 1
                    sim.forces.forces.zero_()
                    sim.forces.centers = rest_pos
                    sim.update_force_weight()

                    # embed force center so we can move it with the sim
                    force_center_quadrature = fem.PicQuadrature(
                        fem.Cells(geo),
                        rest_pos,
                    )
                    force_center_quadrature._domain = sim.u_test.domain

                else:
                    sim.forces.count = 0

            elif sim.forces.count > 0:
                screen_coords = io.MousePos
                world_ray = ps.screen_coords_to_world_ray(screen_coords)

                # interpolate current position of force application center
                force_center_position = wp.empty(shape=(1,), dtype=wp.vec3)
                fem.interpolate(
                    surface_positions,
                    quadrature=force_center_quadrature,
                    dest=force_center_position,
                    fields={"displacement": sim.u_field},
                )
                deformed_force_center = force_center_position.numpy()[0]

                # update picking force direction
                ray_dir = world_ray / np.linalg.norm(world_ray)
                ray_orig = ps.get_view_camera_parameters().get_position()
                perp = ray_orig - deformed_force_center
                perp -= np.dot(perp, ray_dir) * ray_dir

                sim.forces.forces = wp.array([perp * args.force_scale], dtype=wp.vec3)

                # force line visualization
                ps.get_curve_network("force_line").update_node_positions(
                    np.array([deformed_force_center, deformed_force_center + perp])
                )
                ps.get_curve_network("force_line").set_enabled(True)

            if io.MouseReleased[0]:
                sim.forces.count = 0
                ps.get_curve_network("force_line").set_enabled(False)

            io.WantCaptureMouse = sim.forces.count > 0

    ps.init()

    ps.set_ground_plane_mode(mode_str="none")
    ps.register_curve_network(
        "force_line",
        nodes=np.zeros((2, 3)),
        edges=np.array([[0, 1]]),
        enabled=False,
    )

    init_surface(flexicubes)

    # ps.set_build_default_gui_panels(False)
    ps.set_user_callback(callback)
    ps.show()
