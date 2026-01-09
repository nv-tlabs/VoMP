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
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim.render
from warp.sim import Model, State


from fem_examples.mpm.mpm_integrator import MPMIntegrator

import pyvista


class Example:
    def __init__(self, args, collider: wp.Volume, stage=None):
        builder = wp.sim.ModelBuilder()
        Example.emit_particles(builder, args)

        builder.set_ground_plane(
            offset=(
                np.min(collider.points.numpy()[:, 1])
                if collider
                else np.min(builder.particle_q[:, 1])
            )
        )

        model: Model = builder.finalize()

        model.gravity = wp.vec3(args.gravity)

        self.frame_dt = 1.0 / args.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.model = model
        self.state_0: State = model.state()
        self.state_1: State = model.state()

        self.sim_time = 0.0
        self.integrator = MPMIntegrator(args, model, [collider] if collider else [])

        self.integrator.enrich_state(self.state_0)
        self.integrator.enrich_state(self.state_1)

        self.particle_radius = self.integrator.voxel_size / 6

        if args.grains:
            self.grains = self.integrator.sample_grains(
                self.state_0,
                particle_radius=self.particle_radius,
                grains_per_particle=10,
            )
        else:
            self.grains = None

        if stage is not None:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage)
        else:
            self.renderer = None

    @staticmethod
    def emit_particles(builder: wp.sim.ModelBuilder, args):
        max_fraction = args.max_fraction
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array(args.emit_lo)
        particle_hi = np.array(args.emit_hi)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        Example._spawn_particles(
            builder, particle_res, particle_lo, particle_hi, max_fraction
        )

    @staticmethod
    def _spawn_particles(
        builder: wp.sim.ModelBuilder, res, bounds_lo, bounds_hi, packing_fraction
    ):
        Nx = res[0]
        Ny = res[1]
        Nz = res[2]

        px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
        py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
        pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)

        radius = np.max(cell_size) * 0.5
        volume = np.prod(cell_volume) * packing_fraction

        points += 2.0 * radius * (np.random.rand(*points.shape) - 0.5)
        vel = np.zeros_like(points)

        builder.particle_q = points
        builder.particle_qd = vel
        builder.particle_mass = np.full(points.shape[0], volume)
        builder.particle_radius = np.full(points.shape[0], radius)
        builder.particle_flags = np.zeros(points.shape[0], dtype=int)

    @staticmethod
    def add_parser_arguments(parser):
        parser.add_argument("--emit_lo", type=float, nargs=3, default=[-10, 15, -10])
        parser.add_argument("--emit_hi", type=float, nargs=3, default=[10, 35, 10])
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, -10, 0])
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=1)
        parser.add_argument(
            "--grains", action=argparse.BooleanOptionalAction, default=False
        )

        MPMIntegrator.add_parser_arguments(parser)

    def update(self, frame_index):
        with wp.ScopedTimer(f"simulate {frame_index}", active=True, synchronize=True):
            for _s in range(self.sim_substeps):
                self.integrator.simulate(
                    self.state_0, self.state_1, self.sim_dt, project_outside=False
                )
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, plotter: pyvista.Plotter):
        with wp.ScopedTimer("render", active=True, synchronize=True):
            time = self.sim_time

            if self.renderer is not None:
                self.renderer.begin_frame(time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()

            if plotter is not None:
                points = self.state_0.particle_q.numpy()
                vel = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)

                if self.grains is None:
                    plotter.add_points(
                        points,
                        name="particles",
                        style="points",
                        render_points_as_spheres=True,
                        scalars=vel,
                        show_scalar_bar=False,
                    )
                else:
                    self.integrator.update_grains(
                        self.state_1,
                        self.state_0,
                        self.grains,
                        self.particle_radius,
                        self.sim_dt,
                    )
                    plotter.add_points(
                        self.grains.flatten().numpy(),
                        name="grains",
                        render_points_as_spheres=True,
                        # color="b",
                        point_size=2,
                    )

                # if self.velocity_field:
                #     field = self.velocity_field
                #     cells, types = field.space.vtk_cells()
                #     node_pos = field.space.node_positions().numpy()

                #     grid = pyvista.UnstructuredGrid(cells, types, node_pos)
                #     # grid.point_data["v"] = field.dof_values.numpy()

                #     plotter.add_mesh(
                #         grid,
                #         name="grid",
                #         show_edges=True,
                #         use_transparency=True,
                #         opacity=0.75,
                #     )
        self.sim_time += self.frame_dt


@wp.kernel
def _fill_triangle_indices(
    face_offsets: wp.array(dtype=int),
    face_vertex_indices: wp.array(dtype=int),
    tri_vertex_indices: wp.array(dtype=int),
):
    fid = wp.tid()

    if fid == 0:
        beg = 0
    else:
        beg = face_offsets[fid - 1]
    end = face_offsets[fid]

    for t in range(beg, end - 2):
        tri_index = t - 2 * fid
        tri_vertex_indices[3 * tri_index + 0] = face_vertex_indices[beg]
        tri_vertex_indices[3 * tri_index + 1] = face_vertex_indices[t + 1]
        tri_vertex_indices[3 * tri_index + 2] = face_vertex_indices[t + 2]


def mesh_triangle_indices(face_index_counts, face_indices):
    face_count = len(face_index_counts)

    face_offsets = np.cumsum(face_index_counts)
    tot_index_count = int(face_offsets[-1])

    tri_count = tot_index_count - 2 * face_count
    tri_index_count = 3 * tri_count

    face_offsets = wp.array(face_offsets, dtype=int)
    face_indices = wp.array(face_indices, dtype=int)

    tri_indices = wp.empty(tri_index_count, dtype=int)

    wp.launch(
        kernel=_fill_triangle_indices,
        dim=face_count,
        inputs=[face_offsets, face_indices, tri_indices],
    )

    return tri_indices


def load_collider_mesh(stage_path, prim_path):
    # Create collider mesh
    from pxr import Usd, UsdGeom

    collider_stage = Usd.Stage.Open(stage_path)
    usd_mesh = UsdGeom.Mesh(collider_stage.GetPrimAtPath(prim_path))
    usd_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
    usd_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())

    collider_points = wp.array(usd_mesh.GetPointsAttr().Get(), dtype=wp.vec3)
    collider_indices = mesh_triangle_indices(usd_counts, usd_indices)
    return wp.Mesh(collider_points, collider_indices)


def _create_collider_mesh(kind: str):
    if kind == "rocks":
        collider_stage_path = os.path.join(
            warp.examples.get_asset_directory(), "rocks.usd"
        )
        collider_prim_path = "/root/rocks"
        return load_collider_mesh(collider_stage_path, collider_prim_path)

    if kind == "wedge":
        cube_faces = np.array(
            [
                [0, 2, 6, 4],
                [1, 5, 7, 3],
                [0, 4, 5, 1],
                [2, 3, 7, 6],
                [0, 1, 3, 2],
                [4, 6, 7, 5],
            ]
        )

        # Generate cube vertex positions and rotate them by 45 degrees along z
        cube_points = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        cube_points = (cube_points * [10, 10, 25]) @ np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        cube_points = cube_points + np.array([-9, 0, -12])

        cube_indices = mesh_triangle_indices(np.full(6, 4), cube_faces.flatten())

        return wp.Mesh(
            wp.array(cube_points, dtype=wp.vec3), wp.array(cube_indices, dtype=int)
        )

    return None


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": False})
    wp.set_module_options({"fast_math": True})
    wp.set_module_options({"max_unroll": 2})
    wp.config.fast_math = True
    # wp.verify_cuda = True

    wp.init()

    parser = argparse.ArgumentParser()

    Example.add_parser_arguments(parser)

    parser.add_argument("--stage_path", type=str, default=None)
    parser.add_argument("--collider", choices=["rocks", "wedge", ""], default=None)
    parser.add_argument("--frame_count", type=int, default=-1)

    args = parser.parse_args()

    collider_mesh = _create_collider_mesh(args.collider)
    example = Example(args, collider=collider_mesh, stage=args.stage_path)

    plotter = pyvista.Plotter()
    plotter.set_background(color="white")
    example.render(plotter)

    # Add mesh for visualization
    if collider_mesh is not None:
        try:
            usd_points = collider_mesh.points.numpy()
            usd_counts = np.full(collider_mesh.indices.shape[0] // 3, 3)
            usd_indices = collider_mesh.indices.numpy()

            offsets = np.cumsum(usd_counts)
            ranges = np.array([offsets - usd_counts, offsets]).T
            faces = np.concatenate(
                [
                    [count] + list(usd_indices[beg:end])
                    for (count, (beg, end)) in zip(usd_counts, ranges)
                ]
            )
            ref_geom = pyvista.PolyData(usd_points, faces)
            plotter.add_mesh(ref_geom)
        except Exception:
            pass

    plotter.view_xy()
    cpos = plotter.camera_position
    plotter.camera_position = [
        (cpos[0][0], cpos[0][1], 1.5 * cpos[0][2]),
        cpos[1],
        cpos[2],
    ]

    plotter.show(interactive_update=True)

    frame = 0
    while not plotter.iren.interactor.GetDone():
        if args.frame_count < 0 or frame <= args.frame_count:
            frame += 1
            example.update(frame)
            example.render(plotter)
            plotter.screenshot(f"screenshot_{frame:04}.png")
        plotter.update()

    if example.renderer is not None:
        example.renderer.save()
