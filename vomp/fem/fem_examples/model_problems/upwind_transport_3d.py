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

###########################################################################
# Example Convection Diffusion DG
#
# This example simulates a convection-diffusion PDE using Discontinuous
# Galerkin with upwind transport and Symmetric Interior Penalty
#
# D phi / dt - nu d2 phi / dx^2 = 0
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@fem.integrand
def inertia_form(s: fem.Sample, phi: fem.Field, psi: fem.Field, dt: float):
    return phi(s) * psi(s) / dt


@wp.func
def velocity(pos: wp.vec3, ang_vel: float):
    center = wp.vec3(0.5)
    offset = pos - center
    return wp.vec3(offset[1], -offset[0], 0.0) * ang_vel


@fem.integrand
def initial_condition(s: fem.Sample, domain: fem.Domain):
    x = domain(s)
    return wp.cos(10.0 * x[0]) * wp.sin(10.0 * x[1])


# Standard transport term, on cells' interior
@fem.integrand
def transport_form(
    s: fem.Sample, domain: fem.Domain, phi: fem.Field, psi: fem.Field, ang_vel: float
):
    pos = domain(s)
    vel = velocity(pos, ang_vel)

    return psi(s) * wp.dot(fem.grad(phi, s), vel)


# Upwind flux, on cell sides
@fem.integrand
def upwind_transport_form(
    s: fem.Sample, domain: fem.Domain, phi: fem.Field, psi: fem.Field, ang_vel: float
):
    pos = domain(s)
    vel = velocity(pos, ang_vel)
    vel_n = wp.dot(vel, fem.normal(domain, s))

    if vel_n <= 0.0 and (
        wp.max(pos) >= 0.9999 or wp.min(pos) <= 0.0001
    ):  # boundary side
        return phi(s) * (-psi(s) * vel_n + 0.5 * psi(s) * wp.abs(vel_n))

    # interior side
    return fem.jump(phi, s) * (
        -fem.average(psi, s) * vel_n + 0.5 * fem.jump(psi, s) * wp.abs(vel_n)
    )


@wp.func
def refinement_field(x: wp.vec3):
    return wp.length(x - wp.vec3(0.5)) * 3.0


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=50,
        ang_vel=1.0,
        level_count=5,
    ):
        self._quiet = quiet

        self.sim_dt = 1.0 / (ang_vel * 50)
        self.current_frame = 0

        # geo = fem.Grid3D(res=wp.vec3i(resolution))

        # vtx, hexes = fem_example_utils.gen_hexmesh(res=wp.vec3i(resolution))
        # geo = fem.Hexmesh(hexes, vtx)

        vol = fem_example_utils.gen_volume(res=wp.vec3i(resolution))
        # geo = fem.Nanogrid(vol)

        refinement = fem.ImplicitField(
            domain=fem.Cells(fem.Nanogrid(vol)), func=refinement_field
        )
        geo = fem.adaptivity.adaptive_nanogrid_from_field(
            vol, level_count, refinement_field=refinement, grading="none"
        )

        domain = fem.Cells(geometry=geo)
        sides = fem.Sides(geo)
        scalar_space = fem.make_polynomial_space(
            geo,
            discontinuous=True,
            degree=degree,
            family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE,
        )

        # Assemble transport, diffusion and inertia matrices

        self._test = fem.make_test(space=scalar_space, domain=domain)
        trial = fem.make_trial(space=scalar_space, domain=domain)

        matrix_inertia = fem.integrate(
            inertia_form,
            fields={"phi": trial, "psi": self._test},
            values={"dt": self.sim_dt},
        )

        matrix_transport = fem.integrate(
            transport_form,
            fields={"phi": trial, "psi": self._test},
            values={"ang_vel": ang_vel},
        )

        side_test = fem.make_test(space=scalar_space, domain=sides)
        side_trial = fem.make_trial(space=scalar_space, domain=sides)

        matrix_transport += fem.integrate(
            upwind_transport_form,
            fields={"phi": side_trial, "psi": side_test},
            values={"ang_vel": ang_vel},
        )

        self._matrix = matrix_inertia + matrix_transport

        # Initial condition
        self._phi_field = scalar_space.make_field()
        fem.interpolate(initial_condition, dest=self._phi_field)

        self.renderer = fem_example_utils.Plot()
        self.renderer.add_field("phi", self._phi_field)

    def step(self):
        self.current_frame += 1

        rhs = fem.integrate(
            inertia_form,
            fields={"phi": self._phi_field, "psi": self._test},
            values={"dt": self.sim_dt},
        )

        phi = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(
            self._matrix, b=rhs, x=phi, method="bicgstab", quiet=self._quiet
        )

        wp.utils.array_cast(in_array=phi, out_array=self._phi_field.dof_values)

    def render(self):
        self.renderer.begin_frame(time=self.current_frame * self.sim_dt)
        self.renderer.add_field("phi", self._phi_field)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument(
        "--degree", type=int, default=2, help="Polynomial degree of shape functions."
    )
    parser.add_argument(
        "--level_count", type=int, default=4, help="Number of refinement levels."
    )
    parser.add_argument(
        "--num_frames", type=int, default=100, help="Total number of frames."
    )
    parser.add_argument("--ang_vel", type=float, default=1.0, help="Angular velocity.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            degree=args.degree,
            resolution=args.resolution,
            ang_vel=args.ang_vel,
            level_count=args.level_count,
        )

        for k in range(args.num_frames):
            print(f"Frame {k}:")
            example.step()
            example.render()

        if not args.headless:
            example.renderer.plot()
