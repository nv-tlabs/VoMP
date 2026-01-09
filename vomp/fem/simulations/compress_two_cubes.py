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
import time
import copy
import json
import os

import warp as wp
import warp.fem as fem
from warp.fem import Field, Sample, Domain, integrand, normal

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vomp.fem.fem_examples.mfem.softbody_sim import ClassicFEM, run_softbody_sim

# -----------------------------------------------------------------------------
# Geometry helper
# -----------------------------------------------------------------------------


def build_cube_geo(res, offset, height=0.0):
    lo = wp.vec3(offset[0], height, offset[1])
    hi = wp.vec3(offset[0] + 1.0, height + 1.0, offset[1] + 1.0)
    return fem.Grid3D(res=wp.vec3i(res), bounds_lo=lo, bounds_hi=hi)


# -----------------------------------------------------------------------------
# Boundary projectors
# -----------------------------------------------------------------------------


@integrand
def clamp_bottom_face(s: Sample, domain: Domain, u: Field, v: Field):
    """Fix bottom nodes (normal.y < 0)."""
    nor = normal(domain, s)
    clamped = wp.where(nor[1] < 0.0, 1.0, 0.0)
    return wp.dot(u(s), v(s)) * clamped


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    wp.init()

    parser = argparse.ArgumentParser(
        description="Compress each cube vertically by pushing on its top face"
    )
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument(
        "--spacing", type=float, default=1.5, help="spacing between cube centres"
    )
    parser.add_argument(
        "--youngs",
        type=float,
        nargs=2,
        default=[1e4, 3e4],
        help="Young's modulus for cube0 cube1",
    )
    parser.add_argument(
        "--poissons",
        type=float,
        nargs=2,
        default=[0.45, 0.45],
        help="Poisson ratio for cube0 cube1",
    )
    parser.add_argument(
        "--densities",
        type=float,
        nargs=2,
        default=[1000.0, 1000.0],
        help="Density for cube0 cube1",
    )
    parser.add_argument(
        "--force",
        type=float,
        default=1e3,
        help="Magnitude of downward compressive force (N)",
    )
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ramp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ramp force over frames",
    )

    # generic FEM args
    ClassicFEM.add_parser_arguments(parser)
    parser.set_defaults(
        n_frames=100,
        gravity=0.0,
        quasi_quasistatic=True,
        young_modulus=1e4,
        poisson_ratio=0.45,
        density=1000.0,
    )

    args = parser.parse_args()

    offsets = [(0.0, 0.0), (args.spacing, 0.0)]
    sims = []
    force_sims = []  # list of sims for force ramping

    for idx, offs in enumerate(offsets):
        geo = build_cube_geo(args.resolution, offs)

        local_args = copy.copy(args)
        local_args.young_modulus = args.youngs[idx]
        local_args.poisson_ratio = args.poissons[idx]
        local_args.density = args.densities[idx]

        sim = ClassicFEM(geo, None, local_args)
        sim.init_displacement_space()
        sim.init_strain_spaces()

        # Clamp bottom face for vertical compression
        sim.set_boundary_condition(boundary_projector_form=clamp_bottom_face)

        # Force applied at top face centre downward (-Y)
        top_center = wp.vec3(offs[0] + 0.5, 1.0, offs[1] + 0.5)

        sim.forces.count = 1
        sim.forces.centers = wp.array([top_center], dtype=wp.vec3)
        sim.forces.radii = wp.array([0.6], dtype=float)
        sim.forces.forces = wp.array([wp.vec3(0.0, -args.force, 0.0)], dtype=wp.vec3)
        sim.update_force_weight()

        sims.append(sim)
        force_sims.append(sim)

        sim.init_constant_forms()
        sim.project_constant_forms()

    recorded = [[] for _ in sims]

    for i, sim in enumerate(sims):
        recorded[i].append(sim.u_field.space.node_positions().numpy())

    n_frames = args.n_frames if args.n_frames > 0 else 1

    for frame in range(n_frames):
        if args.ramp:
            scale = float(frame + 1) / float(n_frames)
            for sim in force_sims:
                sim.forces.forces = wp.array(
                    [wp.vec3(0.0, -args.force * scale, 0.0)], dtype=wp.vec3
                )
                sim.update_force_weight()

        for i, sim in enumerate(sims):
            sim.run_frame()
            pos = (
                sim.u_field.space.node_positions().numpy()
                + sim.u_field.dof_values.numpy()
            )
            recorded[i].append(pos)

        # per-frame energy report matching stretch_two_cubes.py
        en_frame = [float(sim.evaluate_energy()[0]) for sim in sims]
        print(f"Frame {frame+1}/{n_frames} energies: {en_frame}")

    # Compute and print potential energies for both cubes
    energies = []
    for sim in sims:
        E, _ = sim.evaluate_energy()
        energies.append(float(E))

    print("Potential energies (left_cube, right_cube):", energies)

    if args.ui:
        import polyscope as ps
        import polyscope.imgui as psim

        ps.init()
        ps.set_ground_plane_mode("shadow_only")

        # Create cam directory if it doesn't exist
        os.makedirs("cam", exist_ok=True)

        # Load camera view if it exists
        cam_file = "cam/compress_two_cubes.json"
        if os.path.exists(cam_file):
            try:
                with open(cam_file, "r") as f:
                    view_data = json.load(f)
                view_json = json.dumps(view_data)
                ps.set_view_from_json(view_json)
                print(f"Loaded camera view from {cam_file}")
            except Exception as e:
                print(f"Error loading camera view: {e}")

        meshes = []
        for i, sim in enumerate(sims):
            try:
                hexes = sim.u_field.space.node_hexes()
            except AttributeError:
                hexes = None
            if hexes is None:
                try:
                    tets = sim.u_field.space.node_tets()
                except AttributeError:
                    tets = None
            else:
                tets = None

            mesh = ps.register_volume_mesh(
                f"cube_{i}",
                recorded[i][0],
                hexes=hexes,
                tets=tets,
                edge_width=0.0,
                transparency=0.6,
            )
            meshes.append(mesh)

        current = [0]
        play = [False]
        last = [time.time()]

        def _ui():
            changed, val = psim.SliderInt("frame", current[0], 0, n_frames)
            if changed:
                current[0] = val
                for m, rec in zip(meshes, recorded):
                    m.update_vertex_positions(rec[val])
            if psim.Button("Play" if not play[0] else "Pause"):
                play[0] = not play[0]
                last[0] = time.time()

            # Camera capture button
            if psim.Button("Capture Camera View"):
                # Get current camera view as JSON
                view_json = ps.get_view_as_json()
                # Simple filename
                filename = "cam/compress_two_cubes.json"
                # Save to file
                with open(filename, "w") as f:
                    json.dump(json.loads(view_json), f, indent=2)
                print(f"Camera view saved to {filename}")

            # Load camera view dropdown
            if os.path.exists("cam"):
                cam_files = [f for f in os.listdir("cam") if f.endswith(".json")]
                if cam_files:
                    psim.Text("Load Camera View:")
                    for cam_file in sorted(cam_files):
                        if psim.Button(f"Load {cam_file}"):
                            try:
                                with open(f"cam/{cam_file}", "r") as f:
                                    view_data = json.load(f)
                                view_json = json.dumps(view_data)
                                ps.set_view_from_json(view_json)
                                print(f"Loaded camera view from {cam_file}")
                            except Exception as e:
                                print(f"Error loading camera view: {e}")

            if play[0] and time.time() - last[0] > 0.05:
                current[0] = (current[0] + 1) % (n_frames + 1)
                for m, rec in zip(meshes, recorded):
                    m.update_vertex_positions(rec[current[0]])
                last[0] = time.time()

        ps.set_user_callback(_ui)
        ps.show()
    else:
        # Headless mode - save screenshots
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("shadow_only")

        # Create output directory
        output_dir = "outputs/compress_two_cubes"
        os.makedirs(output_dir, exist_ok=True)

        # Load camera view if it exists
        cam_file = "cam/compress_two_cubes.json"
        if os.path.exists(cam_file):
            try:
                with open(cam_file, "r") as f:
                    view_data = json.load(f)
                view_json = json.dumps(view_data)
                ps.set_view_from_json(view_json)
                print(f"Loaded camera view from {cam_file}")
            except Exception as e:
                print(f"Error loading camera view: {e}")

        # Register meshes
        meshes = []
        for i, sim in enumerate(sims):
            try:
                hexes = sim.u_field.space.node_hexes()
            except AttributeError:
                hexes = None
            if hexes is None:
                try:
                    tets = sim.u_field.space.node_tets()
                except AttributeError:
                    tets = None
            else:
                tets = None

            mesh = ps.register_volume_mesh(
                f"cube_{i}",
                recorded[i][0],
                hexes=hexes,
                tets=tets,
                edge_width=0.0,
                transparency=0.6,
            )
            meshes.append(mesh)

        # Save 10 evenly distributed frames
        screenshot_frames = [int(i * n_frames / 9) for i in range(10)]
        print(f"Saving screenshots for frames: {screenshot_frames}")

        for frame_idx in screenshot_frames:
            # Update mesh positions for this frame
            for m, rec in zip(meshes, recorded):
                m.update_vertex_positions(rec[frame_idx])

            # Take screenshot
            filename = f"{output_dir}/frame_{frame_idx:06d}.png"
            ps.screenshot(filename)
            print(f"Screenshot saved: {filename}")

        print(f"All screenshots saved to {output_dir}/")


if __name__ == "__main__":
    main()
