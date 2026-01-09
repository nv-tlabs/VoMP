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

import numpy as np

import warp as wp
import warp.fem as fem

import warp.examples.fem.utils as fem_example_utils

mu_0 = wp.constant(np.pi * 4.0e-7)  # Vacuum magnetic permeability
mu_c = wp.constant(1.25e-6)  # Copper magnetic permeability
mu_i = wp.constant(6.0e-3)  # Iron magnetic permeability
J_0 = wp.constant(5.0e6)  # Current density

# Mesh

res = 32
degree = 1

R = wp.constant(2.0)  # domain radius
coil_height = 0.25
coil_internal_radius = wp.constant(0.3)
coil_external_radius = wp.constant(0.4)

core_height = 1.0
core_radius = wp.constant(0.2)

# positions, tet_vidx = fem_example_utils.gen_tetmesh(
#     bounds_lo=wp.vec3(-R, -R, -R),
#     bounds_hi=wp.vec3(R, R, R),
#     res=wp.vec3i(res, res, res),
# )
positions, hex_vidx = fem_example_utils.gen_hexmesh(
    bounds_lo=wp.vec3(-R, -R, -R),
    bounds_hi=wp.vec3(R, R, R),
    res=wp.vec3i(res, res, res),
)


@wp.kernel
def cylinderify(pos: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p = pos[i]

    pxz = wp.vec3(p[0], 0.0, p[2])
    pos[i] = wp.max(wp.abs(pxz)) * wp.normalize(pxz) + wp.vec3(0.0, p[1], 0.0)


wp.launch(cylinderify, dim=positions.shape, inputs=[positions])

# geo = fem.Tetmesh(tet_vertex_indices=tet_vidx, positions=positions)
geo = fem.Hexmesh(hex_vertex_indices=hex_vidx, positions=positions)
# geo = fem.Grid3D(
#     bounds_lo=wp.vec3(-R, -R, -R),
#     bounds_hi=wp.vec3(R, R, R),
#     res=wp.vec3i(res, res, res),
# )

v_space = fem.make_polynomial_space(
    geo, degree=degree, element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND, dtype=wp.vec3
)


@wp.func
def mu(pos: wp.vec3):
    x = wp.abs(pos[0])
    y = wp.abs(pos[1])
    z = wp.abs(pos[2])

    r = wp.sqrt(x * x + z * z)

    if r <= core_radius:
        return wp.select(y < core_height, mu_0, mu_i)

    if r >= coil_internal_radius and r <= coil_external_radius:
        return wp.select(y < coil_height, mu_0, mu_c)

    return mu_0


@wp.func
def j(pos: wp.vec3):
    x = pos[0]
    y = wp.abs(pos[1])
    z = pos[2]

    r = wp.sqrt(x * x + z * z)

    return wp.select(
        y < coil_height and r >= coil_internal_radius and r <= coil_external_radius,
        wp.vec3(0.0),
        wp.vec3(z, 0.0, -x) * J_0 / r,
    )


@fem.integrand
def curl_curl_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    return wp.dot(fem.curl(u, s), fem.curl(v, s)) / mu(domain(s))


@fem.integrand
def j_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    return wp.dot(j(domain(s)), v(s))


@fem.integrand
def bd_proj_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    nor = fem.normal(domain, s)
    u_s = u(s)
    v_s = v(s)
    u_t = u_s - wp.dot(u_s, nor) * nor
    v_t = v_s - wp.dot(v_s, nor) * nor

    return wp.dot(u_t, v_t)


domain = fem.Cells(geo)
u = fem.make_trial(space=v_space, domain=domain)
v = fem.make_test(space=v_space, domain=domain)

quadrature = fem.RegularQuadrature(domain, order=2 * degree)

lhs = fem.integrate(
    curl_curl_form,
    quadrature=quadrature,
    fields={"u": u, "v": v},
    output_dtype=float,
    assembly="generic",
)
rhs = fem.integrate(
    j_form,
    quadrature=quadrature,
    fields={"v": v},
    output_dtype=v_space.dof_dtype,
    assembly="generic",
)


# Dirichlet BC
boundary = fem.BoundarySides(geo)
u_bd = fem.make_trial(space=v_space, domain=boundary)
v_bd = fem.make_test(space=v_space, domain=boundary)
dirichlet_bd_proj = fem.integrate(
    bd_proj_form, fields={"u": u_bd, "v": v_bd}, nodal=True, output_dtype=float
)
fem.project_linear_system(lhs, rhs, dirichlet_bd_proj)


x = wp.zeros_like(rhs)
fem_example_utils.bsr_cg(
    lhs, b=rhs, x=x, tol=1.0e-3, quiet=False, method="cr", max_iters=1000
)

# make sure result is exactly zero outisde of circle
wp.sparse.bsr_mv(dirichlet_bd_proj, x=x, y=x, alpha=-1.0, beta=1.0)

renderer = fem_example_utils.Plot()
u_field = v_space.make_field()
wp.utils.array_cast(in_array=x, out_array=u_field.dof_values)


@fem.integrand
def norm_expr(s: fem.Sample, u: fem.Field):
    return wp.length(u(s))


@fem.integrand
def curl_expr(s: fem.Sample, u: fem.Field):
    return fem.curl(u, s)


A_norm_space = fem.make_polynomial_space(
    geo,
    degree=degree,
    element_basis=fem.ElementBasis.LAGRANGE,
    discontinuous=False,
    dtype=float,
)
A_norm = A_norm_space.make_field()


B_space = fem.make_polynomial_space(
    geo, degree=degree, element_basis=fem.ElementBasis.LAGRANGE, dtype=wp.vec3
)
B = B_space.make_field()

fem.interpolate(norm_expr, dest=A_norm, fields={"u": u_field})
fem.interpolate(curl_expr, dest=B, fields={"u": u_field})

renderer.add_field("A", A_norm)
renderer.add_field("B", B)

renderer.plot(
    {"A": {"contours": {}}, "B": {"streamlines": {}}, "density": 20},
)
