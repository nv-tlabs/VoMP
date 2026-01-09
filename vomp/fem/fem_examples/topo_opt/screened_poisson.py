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

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import polyscope as ps
import torch
import trimesh
from icosphere import icosphere
from largesteps.geometry import compute_matrix
from largesteps.parameterize import from_differential, to_differential
from pyremesh import remesh_botsch

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@fem.integrand
def screened_diffusion_form(
    s: fem.Sample, u: fem.Field, v: fem.Field, sigma: float, nu: float
):
    return sigma * u(s) * v(s) + nu * wp.dot(
        fem.grad(u, s),
        fem.grad(v, s),
    )


@fem.integrand
def boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    return u(s) * v(s)


@fem.integrand
def interior_penalty_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
    strength: float,
):
    return strength * u(s) * v(s)


@fem.integrand
def sample_solution(s: fem.Sample, u: fem.Field):
    return u(s)


@wp.kernel
def loss_fn(
    target: wp.array(dtype=float),
    samples: wp.array(dtype=float),
    loss: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    diff = target[i] - samples[i]
    wp.atomic_add(loss, 0, wp.float64(diff * diff))


@wp.kernel
def gen_face_samples(
    mesh_faces: wp.array2d(dtype=int),
    mesh_vertices: wp.array(dtype=wp.vec3),
    qp_coords: wp.array(dtype=wp.vec3),
    qp_weights: wp.array(dtype=float),
    points: wp.array2d(dtype=wp.vec3),
    point_measures: wp.array2d(dtype=float),
):
    i, j = wp.tid()

    v0 = mesh_vertices[mesh_faces[i, 0]]
    v1 = mesh_vertices[mesh_faces[i, 1]]
    v2 = mesh_vertices[mesh_faces[i, 2]]
    points[i, j] = qp_coords[j][0] * v0 + qp_coords[j][1] * v1 + qp_coords[j][2] * v2

    area = 0.5 * wp.length(wp.cross(v1 - v0, v2 - v0))
    point_measures[i, j] = qp_weights[j] * area


def remesh(vertices, faces, h):
    v = vertices.numpy()
    f = faces.numpy()
    v_new, f_new = remesh_botsch(v.astype(np.double), f.astype(np.int32), 5, h, True)

    return wp.array(v_new, dtype=wp.vec3, requires_grad=True), wp.array(
        f_new, dtype=int
    )


def largesteps_matrix(vertices, faces, smoothing):
    with torch.no_grad():
        M = compute_matrix(
            wp.to_torch(vertices, requires_grad=False),
            wp.to_torch(faces, requires_grad=False),
            lambda_=smoothing,
            # alpha=0.75,
        )
        return M


def parametrize(vertices, params, M):
    with torch.no_grad():
        u = wp.from_torch(
            to_differential(L=M, v=wp.to_torch(vertices, requires_grad=False)),
            requires_grad=False,
            dtype=wp.vec3,
        )
        wp.copy(src=u, dest=params)


def unparametrize(params, vertices, M):
    with torch.no_grad():
        u = wp.from_torch(
            from_differential(L=M, u=wp.to_torch(params, requires_grad=False)),
            requires_grad=False,
            dtype=wp.vec3,
        )
        wp.copy(src=u, dest=vertices)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


class Example:
    def __init__(
        self,
        target_path,
        degree=2,
        resolution=10,
        serendipity=False,
        viscosity=2.0,
        screening=1.0,
        boundary_compliance=0.001,
        sampling_coord=0.33,
        smoothing=10.0,
    ):
        self._iter = 0

        self._viscosity = viscosity
        self._screening = screening

        self._boundary_compliance = boundary_compliance

        self._emission_value = 1.0
        self._smoothing = smoothing
        self._h = 0.5 / resolution  # target edge length for remeshing

        # resolution of sampling planees
        sampling_res = 64  # 2 * resolution

        sphere_rad = sampling_coord / 2.0  # radius of initial sphere

        bc_order = (
            2 + 2 * degree
        )  # quadrature order for integrating BC over mesh triangles

        # Initial guess (sphere)
        vertices, faces = icosphere(int(np.log2(resolution)))
        vertices *= sphere_rad

        # Target mesh (ellipse at different resolution)
        target = as_mesh(trimesh.load(target_path))

        vmin, vmax = np.min(target.vertices, axis=0), np.max(target.vertices, axis=0)
        scale = sampling_coord / np.max(vmax - vmin)
        target.vertices = target.vertices - (vmax + vmin) / 2  # Center mesh on origin
        target.vertices = target.vertices * scale * 1.9

        # sampling planes
        X, Y = np.meshgrid(
            np.linspace(-1, 1, sampling_res),
            np.linspace(-1, 1, sampling_res),
        )
        Z = np.ones_like(X)

        P0 = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        P1 = np.stack((X, Y, -Z), axis=-1).reshape(-1, 3)
        P2 = np.stack((Z, X, Y), axis=-1).reshape(-1, 3)
        P3 = np.stack((-Z, X, Y), axis=-1).reshape(-1, 3)
        P4 = np.stack((Y, Z, X), axis=-1).reshape(-1, 3)
        P5 = np.stack((Y, -Z, X), axis=-1).reshape(-1, 3)

        P = np.vstack((P0, P1, P2, P3, P4, P5))
        P[:, 0] *= 0.33 * sampling_coord + 0.67 * scale * (vmax[0] - vmin[0])
        P[:, 1] *= 0.33 * sampling_coord + 0.67 * scale * (vmax[1] - vmin[1])
        P[:, 2] *= 0.33 * sampling_coord + 0.67 * scale * (vmax[2] - vmin[2])

        # per-face BC evaluation points
        # define per-triangle quadrature for non-conforming boundary condition
        coords, weights = fem.geometry.element.Triangle().instantiate_quadrature(
            order=bc_order, family=fem.Polynomial.GAUSS_LEGENDRE
        )

        # Renderer
        ps.register_surface_mesh("surface", vertices, faces)
        ps.register_surface_mesh("target", target.vertices, target.faces)
        ps.register_point_cloud("samples", points=P, radius=0.0025)

        # Move to warp arrays

        self._proj_pos = wp.array(P, dtype=wp.vec3)

        self._qp_coords = wp.array(coords, dtype=wp.vec3)
        self._qp_weights = wp.array(weights, dtype=float)

        self._vertices = wp.array(vertices, dtype=wp.vec3, requires_grad=True)
        self._faces = wp.array(faces, dtype=int)

        self._target_vertices = wp.array(
            target.vertices, dtype=wp.vec3, requires_grad=True
        )
        self._target_faces = wp.array(target.faces, dtype=int)

        # Init sim

        res = wp.vec3i(resolution, resolution, resolution)
        self._geo = fem.Grid3D(
            res=res,
            bounds_lo=wp.vec3(-1.0, -1.0, -1.0),
            bounds_hi=wp.vec3(1.0, 1.0, 1.0),
        )

        # Function space
        element_basis = fem.ElementBasis.SERENDIPITY if serendipity else None
        self._scalar_space = fem.make_polynomial_space(
            self._geo, degree=degree, element_basis=element_basis
        )

        print(
            f"Cell count: {self._geo.cell_count()}, total nodes: {self._scalar_space.node_count()}"
        )

        # Scalar field over our function space
        self._scalar_field: fem.DiscreteField = self._scalar_space.make_field()
        self._scalar_field.dof_values.requires_grad = True

        self._emission_field: fem.DiscreteField = self._scalar_space.make_field()
        self._emission_field.dof_values.fill_(self._emission_value)

        self._init_constant_forms()
        self._setup_target()

    def _init_constant_forms(self):
        geo = self._geo
        domain = fem.Cells(geometry=geo)

        # Hard Dirichlet BC on exterior boundary
        boundary = fem.BoundarySides(geo)

        bd_test = fem.make_test(space=self._scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=self._scalar_space, domain=boundary)
        self._bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": bd_trial, "v": bd_test},
            nodal=True,
            output_dtype=float,
        )
        fem.dirichlet.normalize_dirichlet_projector(self._bd_matrix)

        # Diffusion form
        self._test = fem.make_test(space=self._scalar_space, domain=domain)
        self._trial = fem.make_trial(space=self._scalar_space, domain=domain)

        self._poisson_matrix = fem.integrate(
            screened_diffusion_form,
            fields={"u": self._trial, "v": self._test},
            values={"nu": self._viscosity, "sigma": self._screening},
            output_dtype=float,
        )

        # Points at which solution is sampled
        self._sample_pic = fem.PicQuadrature(
            domain=self._test.domain, positions=self._proj_pos
        )
        self._sampled_values = wp.array(
            dtype=float,
            shape=(self._sample_pic.total_point_count()),
            requires_grad=True,
        )

    def _evaluate_forward(
        self, mesh_vertices, mesh_faces, solution_field, solution_samples, tape=None
    ):
        domain = self._test.domain

        with_gradient = tape is not None
        if not with_gradient:
            tape = wp.Tape()

        # Generate points over mesh triangles
        point_shape = (mesh_faces.shape[0], self._qp_coords.shape[0])
        points = wp.empty(point_shape, dtype=wp.vec3, requires_grad=True)
        point_measures = wp.empty(point_shape, dtype=float, requires_grad=True)
        with tape:
            wp.launch(
                gen_face_samples,
                point_shape,
                inputs=[
                    mesh_faces,
                    mesh_vertices,
                    self._qp_coords,
                    self._qp_weights,
                    points,
                    point_measures,
                ],
            )

            points = points.flatten()
            point_measures = point_measures.flatten()
            pic = fem.PicQuadrature(
                domain=domain,
                positions=points,
                measures=point_measures,
                requires_grad=True,
            )

        # Integrate weak BC over points to get left- and-right-hand-side
        pen_matrix = fem.integrate(
            interior_penalty_form,
            quadrature=pic,
            fields={"u": self._trial, "v": self._test},
            values={
                "strength": 1.0 / self._boundary_compliance,
            },
            output_dtype=float,
        )
        lhs = pen_matrix
        lhs += self._poisson_matrix

        rhs = wp.array(
            shape=self._scalar_space.node_count(),
            dtype=wp.float32,
            requires_grad=with_gradient,
        )
        with tape:
            fem.integrate(
                interior_penalty_form,
                quadrature=pic,
                fields={"u": self._emission_field, "v": self._test},
                values={
                    "strength": 1.0 / self._boundary_compliance,
                },
                output=rhs,
            )
        fem.project_linear_system(lhs, rhs, self._bd_matrix, normalize_projector=False)

        # CG solve
        fem_example_utils.bsr_cg(
            lhs, b=rhs, x=solution_field.dof_values, tol=1.0e-6, quiet=True
        )

        if with_gradient:
            # Register CG for backward pass
            def backward_cg():
                fem_example_utils.bsr_cg(
                    lhs, b=solution_field.dof_values.grad, x=rhs.grad, quiet=True
                )

            tape.record_func(backward_cg, arrays=[solution_field.dof_values, rhs])

        # Interpolate solution on sampling planes
        with tape:
            fem.interpolate(
                sample_solution,
                dest=solution_samples,
                fields={"u": solution_field},
                quadrature=self._sample_pic,
            )

    def _reset_optimizer(self):
        self._vertices, self._faces = remesh(self._vertices, self._faces, h=self._h)

        self._laplacian = largesteps_matrix(
            self._vertices, self._faces, self._smoothing
        )
        self._vertices_param = wp.empty_like(self._vertices)
        parametrize(self._vertices, self._vertices_param, self._laplacian)

        self._adam = wp.optim.Adam(
            params=[self._vertices_param], lr=0.005 * (0.99**self._iter)
        )

    def _setup_target(self):
        target_field = self._scalar_space.make_field()
        self._target_values = wp.array(
            dtype=float, shape=(self._sample_pic.total_point_count())
        )
        self._evaluate_forward(
            self._target_vertices, self._target_faces, target_field, self._target_values
        )

        ps.get_point_cloud("samples").add_scalar_quantity(
            "target", self._target_values.numpy()
        )

    def step(self):
        # Log2 schedule for remeshing
        if (1 << int(np.round(np.log2(self._iter + 1)))) == self._iter + 1:
            self._reset_optimizer()

        tape = wp.Tape()

        # Go from smoothing param space to 3d vertex space, and register same op on grad tape
        unparametrize(self._vertices_param, self._vertices, self._laplacian)

        def unparametrize_grad():
            unparametrize(
                self._vertices.grad, self._vertices_param.grad, self._laplacian
            )

        tape.record_func(
            unparametrize_grad, arrays=[self._vertices_param, self._vertices]
        )

        # Solve diffusion eq on get value on sampling planes
        self._evaluate_forward(
            self._vertices,
            self._faces,
            self._scalar_field,
            self._sampled_values,
            tape=tape,
        )

        # Evaluate loss and do backprop step
        loss = wp.zeros(shape=(1,), dtype=wp.float64, requires_grad=True)
        with tape:
            wp.launch(
                loss_fn,
                dim=self._sampled_values.shape[0],
                inputs=[self._target_values, self._sampled_values, loss],
            )

        print(f"Loss at iteration {self._iter}: {loss.numpy()[0]}")

        tape.backward(loss)
        self._adam.step([self._vertices.grad])

        # Zero-out gradients for next step
        tape.zero()

        self._iter += 1

    def render(self):
        ps.register_surface_mesh("surface", self._vertices.numpy(), self._faces.numpy())
        ps.get_point_cloud("samples").add_scalar_quantity(
            "value",
            self._sampled_values.numpy(),
            enabled=True if self._iter <= 1 else None,
        )
        ps.get_point_cloud("samples").add_scalar_quantity(
            "diff",
            np.abs(self._sampled_values.numpy() - self._target_values.numpy()),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("target_path")
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution.")
    parser.add_argument(
        "--degree", type=int, default=2, help="Polynomial degree of shape functions."
    )
    parser.add_argument(
        "--serendipity",
        action="store_true",
        default=True,
        help="Use Serendipity basis functions.",
    )
    parser.add_argument(
        "--viscosity", type=float, default=2.0, help="Fluid viscosity parameter."
    )
    parser.add_argument(
        "--screening", type=float, default=1.0, help="Screening parameter."
    )
    parser.add_argument(
        "--smoothing", type=float, default=10, help="Smoothing parameter."
    )
    parser.add_argument(
        "--boundary_compliance",
        type=float,
        default=0.001,
        help="Dirichlet boundary condition compliance.",
    )
    parser.add_argument(
        "--num_iters", type=int, default=250, help="Number of iterations"
    )

    args = parser.parse_args()

    ps.init()

    with wp.ScopedDevice(args.device):
        example = Example(
            target_path=args.target_path,
            degree=args.degree,
            resolution=args.resolution,
            serendipity=args.serendipity,
            viscosity=args.viscosity,
            screening=args.screening,
            boundary_compliance=args.boundary_compliance,
            smoothing=args.smoothing,
        )

        def ps_callback():
            if example._iter > args.num_iters:
                return

            example.step()
            example.render()

            # pseudo turntable
            t = 0.025 * example._iter
            c = np.cos(t)
            s = np.sin(t)
            r = 1.5
            ps.look_at(camera_location=(r * c, 0.25, r * s), target=(0.0, 0.0, 0.0))

            ps.screenshot()

        ps.set_ground_plane_mode("none")
        ps.set_user_callback(ps_callback)

        ps.show()
