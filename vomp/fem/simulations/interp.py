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
import numpy as np
import warp as wp
import warp.fem as fem
from vomp.fem.simulations.cube_fall import GroundCollidingSim, run_softbody_sim


def load_material_interpolation_data():
    """Load material interpolation data from JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "material_interpolation_results.json")

    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def build_cube_geo(res, x_offset, height=2.0, width=1.0):
    """Return a Grid3D geometry for one cube at x_offset."""
    lo = wp.vec3(x_offset, height, 0.0)
    hi = wp.vec3(x_offset + width, height + width, width)
    return fem.Grid3D(res=wp.vec3i(res), bounds_lo=lo, bounds_hi=hi)


def write_material_properties_summary(
    output_dir, args, material_1, material_2, cube_materials
):
    """Write a detailed summary of material properties and color coding to a text file."""
    summary_file = os.path.join(
        output_dir, f"material_properties_summary_pair_{args.pair}.txt"
    )

    with open(summary_file, "w") as f:
        f.write("MATERIAL INTERPOLATION SIMULATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Material Pair {args.pair}:\n")
        f.write(f"  Base Material 1: {material_1['name']}\n")
        f.write(f"  Base Material 2: {material_2['name']}\n")
        f.write(f"  Simulation Frames: {args.n_frames}\n")
        f.write(f"  Cube Spacing: {args.spacing} units\n")
        f.write(f"  Drop Height: {args.drop_height} units\n\n")

        f.write("CUBE ARRANGEMENT (Left to Right):\n")
        f.write("-" * 50 + "\n")

        for i, mat in enumerate(cube_materials):
            position = i + 1

            if i == 0:
                color = "Green"
                material_type = "Original Material 1"
            elif i == 4:
                color = "Yellow"
                material_type = "Original Material 2"
            else:
                color = f"Blue (intensity {0.3 + 0.5 * (i-1)/2:.1f})"
                interp_factor = [0.25, 0.5, 0.75][i - 1]
                material_type = f"Interpolated ({interp_factor:.2f} factor)"

            f.write(f"Position {position}: {mat['name']}\n")
            f.write(f"  Material Type: {material_type}\n")
            f.write(f"  Visualization Color: {color}\n")
            f.write(f"  Young's Modulus: {mat['youngs_modulus_gpa']:.6f} GPa\n")
            f.write(f"  Poisson Ratio: {mat['poisson_ratio']:.6f}\n")
            f.write(f"  Density: {mat['density']:.2f} kg/m³\n")
            f.write(
                f"  Young's Modulus (Pa): {mat['youngs_modulus_gpa'] * 1e9:.0f} Pa\n"
            )
            f.write("\n")

        f.write("INTERPOLATION DETAILS:\n")
        f.write("-" * 50 + "\n")
        f.write(
            "The middle three cubes have properties interpolated between the two base materials:\n"
        )
        f.write("  • Position 2: 25% interpolation (75% Material 1 + 25% Material 2)\n")
        f.write("  • Position 3: 50% interpolation (50% Material 1 + 50% Material 2)\n")
        f.write(
            "  • Position 4: 75% interpolation (25% Material 1 + 75% Material 2)\n\n"
        )

        f.write("EXPECTED BEHAVIOR:\n")
        f.write("-" * 50 + "\n")
        f.write(
            "Each cube falls freely under gravity with its specific material properties.\n"
        )
        f.write(
            "Differences in Young's modulus, Poisson ratio, and density will cause:\n"
        )
        f.write("  • Different deformation patterns during impact\n")
        f.write("  • Varying bounce characteristics\n")
        f.write("  • Different settling behavior\n")
        f.write("  • Distinct collision responses\n\n")

        f.write("MATERIAL PROPERTY RANGES:\n")
        f.write("-" * 50 + "\n")
        youngs_values = [mat["youngs_modulus_gpa"] for mat in cube_materials]
        poisson_values = [mat["poisson_ratio"] for mat in cube_materials]
        density_values = [mat["density"] for mat in cube_materials]

        f.write(
            f"Young's Modulus Range: {min(youngs_values):.6f} - {max(youngs_values):.6f} GPa\n"
        )
        f.write(
            f"Poisson Ratio Range: {min(poisson_values):.6f} - {max(poisson_values):.6f}\n"
        )
        f.write(
            f"Density Range: {min(density_values):.2f} - {max(density_values):.2f} kg/m³\n\n"
        )

        f.write("COLOR LEGEND:\n")
        f.write("-" * 50 + "\n")
        f.write("  🟢 Green (Position 1): Original Material 1\n")
        f.write("  🔵 Dark Blue (Position 2): 25% Interpolated\n")
        f.write("  🔵 Medium Blue (Position 3): 50% Interpolated\n")
        f.write("  🔵 Light Blue (Position 4): 75% Interpolated\n")
        f.write("  🟡 Yellow (Position 5): Original Material 2\n")

    return summary_file


def main():
    wp.init()

    parser = argparse.ArgumentParser(
        description="5 cubes in a row with interpolated material properties"
    )
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument(
        "--spacing", type=float, default=1.5, help="spacing between cube centers"
    )
    parser.add_argument(
        "--drop_height", type=float, default=2.5, help="initial height of cubes"
    )
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--pair",
        type=int,
        default=1,
        help="Material pair to use (1 or 2 from JSON file)",
    )

    GroundCollidingSim.add_parser_arguments(parser)

    parser.set_defaults(
        n_frames=200,
        gravity=9.81,
        quasi_quasistatic=False,
        young_modulus=1e4,
        poisson_ratio=0.45,
        density=1000.0,
        ground=True,
        ground_height=0.0,
        collision_radius=0.05,
        n_newton=100,
    )

    args = parser.parse_args()

    material_data = load_material_interpolation_data()

    pair_data = None
    for pair in material_data:
        if pair["pair_index"] == args.pair:
            pair_data = pair
            break

    if pair_data is None:
        available_pairs = [p["pair_index"] for p in material_data]
        raise ValueError(
            f"Material pair {args.pair} not found. Available pairs: {available_pairs}"
        )

    material_1 = pair_data["material_1"]
    material_2 = pair_data["material_2"]
    interpolated = pair_data["interpolated_materials"]

    cube_materials = [
        {
            "name": f"{material_1['name']}",
            "youngs_modulus_gpa": material_1["youngs_modulus_gpa"],
            "poisson_ratio": material_1["poisson_ratio"],
            "density": material_1["density"],
            "is_fixed": False,
        },
        {
            "name": f"Interpolated 0.25",
            "youngs_modulus_gpa": interpolated[0]["target_properties"][
                "youngs_modulus_gpa"
            ],
            "poisson_ratio": interpolated[0]["target_properties"]["poisson_ratio"],
            "density": interpolated[0]["target_properties"]["density"],
            "is_fixed": False,
        },
        {
            "name": f"Interpolated 0.5",
            "youngs_modulus_gpa": interpolated[1]["target_properties"][
                "youngs_modulus_gpa"
            ],
            "poisson_ratio": interpolated[1]["target_properties"]["poisson_ratio"],
            "density": interpolated[1]["target_properties"]["density"],
            "is_fixed": False,
        },
        {
            "name": f"Interpolated 0.75",
            "youngs_modulus_gpa": interpolated[2]["target_properties"][
                "youngs_modulus_gpa"
            ],
            "poisson_ratio": interpolated[2]["target_properties"]["poisson_ratio"],
            "density": interpolated[2]["target_properties"]["density"],
            "is_fixed": False,
        },
        {
            "name": f"{material_2['name']}",
            "youngs_modulus_gpa": material_2["youngs_modulus_gpa"],
            "poisson_ratio": material_2["poisson_ratio"],
            "density": material_2["density"],
            "is_fixed": False,
        },
    ]

    print(f"Using material pair {args.pair}:")
    print(f"Material 1: {material_1['name']}")
    print(f"Material 2: {material_2['name']}")
    print(f"Simulation set to run for {args.n_frames} frames")
    print(f"Creating 5 cubes in a row with material properties:")
    print()

    for i, mat in enumerate(cube_materials):
        cube_type = "ORIGINAL" if i == 0 or i == 4 else "INTERPOLATED"
        print(f"Cube {i+1}: {mat['name']} ({cube_type})")
        print(f"  Young's modulus: {mat['youngs_modulus_gpa']:.6f} GPa")
        print(f"  Poisson ratio: {mat['poisson_ratio']:.6f}")
        print(f"  Density: {mat['density']:.2f} kg/m³")
        print()

    sims = []

    for i, mat in enumerate(cube_materials):
        x_offset = i * args.spacing
        geo = build_cube_geo(args.resolution, x_offset, height=args.drop_height)

        local_args = copy.copy(args)
        local_args.young_modulus = mat["youngs_modulus_gpa"] * 1e9
        local_args.poisson_ratio = mat["poisson_ratio"]
        local_args.density = mat["density"]

        sim = GroundCollidingSim(geo, None, local_args)
        sim.init_displacement_space()
        sim.init_strain_spaces()
        sim.init_collision_detector()

        sim.set_boundary_condition(boundary_projector_form=None)

        sim.init_constant_forms()
        sim.project_constant_forms()

        sims.append(sim)

    n_frames = args.n_frames if hasattr(args, "n_frames") and args.n_frames > 0 else 200
    recorded = [[] for _ in sims]

    for i, sim in enumerate(sims):
        recorded[i].append(
            sim.u_field.space.node_positions().numpy() + sim.u_field.dof_values.numpy()
        )

    print(f"Running simulation for {n_frames} frames...")
    for frame in range(n_frames):
        for i, sim in enumerate(sims):
            sim.run_frame()
            recorded[i].append(
                sim.u_field.space.node_positions().numpy()
                + sim.u_field.dof_values.numpy()
            )

        if frame % 25 == 0:
            print(f"Frame {frame+1}/{n_frames}")

    if args.ui:
        import polyscope as ps
        import polyscope.imgui as psim

        ps.init()
        ps.set_window_size(1920, 1080)
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height(0.0)

        os.makedirs("cam", exist_ok=True)

        cam_file = "cam/interp_cubes_fall.json"
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
        for i, (sim, mat) in enumerate(zip(sims, cube_materials)):
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

            transparency = 0.7 if i == 0 or i == 4 else 0.6

            m = ps.register_volume_mesh(
                f"cube_{i}_{mat['name'].replace(' ', '_')}",
                recorded[i][0],
                hexes=hexes,
                tets=tets,
                edge_width=0.0,
                transparency=transparency,
            )

            if i == 0 or i == 4:
                if i == 0:
                    m.set_color((0.2, 0.8, 0.2))
                else:
                    m.set_color((0.8, 0.8, 0.2))
            else:

                interp_factor = (i - 1) / 2
                blue_intensity = 0.3 + 0.5 * interp_factor
                m.set_color((0.2, 0.2, blue_intensity))

            meshes.append(m)

        current = [0]
        play = [False]
        last = [time.time()]

        def _ui():
            changed, val = psim.SliderInt("frame", current[0], 0, n_frames - 1)
            if changed:
                current[0] = val
                for m, rec in zip(meshes, recorded):
                    m.update_vertex_positions(rec[val])
            if psim.Button("Play" if not play[0] else "Pause"):
                play[0] = not play[0]
                last[0] = time.time()

            if psim.Button("Capture Camera View"):
                view_json = ps.get_view_as_json()
                filename = "cam/interp_cubes_fall.json"
                with open(filename, "w") as f:
                    json.dump(json.loads(view_json), f, indent=2)
                print(f"Camera view saved to {filename}")

            if os.path.exists("cam/interp_cubes_fall.json"):
                if psim.Button("Load Camera View"):
                    try:
                        with open("cam/interp_cubes_fall.json", "r") as f:
                            view_data = json.load(f)
                        view_json = json.dumps(view_data)
                        ps.set_view_from_json(view_json)
                        print("Camera view loaded")
                    except Exception as e:
                        print(f"Error loading camera view: {e}")

            psim.Text("Material Properties:")
            psim.Text(
                f"Pair: {args.pair} ({material_1['name']} → {material_2['name']})"
            )
            psim.Text(f"Frames: {n_frames}")
            psim.Text("Cube colors: Green = Original, Blue = Interpolated")
            psim.Separator()

            for i, mat in enumerate(cube_materials):
                status = "ORIGINAL" if i == 0 or i == 4 else "INTERPOLATED"
                psim.Text(f"Cube {i+1}: {status}")
                psim.Text(f"  YM: {mat['youngs_modulus_gpa']:.3f} GPa")
                psim.Text(f"  ν: {mat['poisson_ratio']:.3f}")
                psim.Text(f"  ρ: {mat['density']:.0f} kg/m³")
                if i < len(cube_materials) - 1:
                    psim.Separator()

            if play[0] and time.time() - last[0] > 0.05:
                current[0] = (current[0] + 1) % n_frames
                for m, rec in zip(meshes, recorded):
                    m.update_vertex_positions(rec[current[0]])
                last[0] = time.time()

        ps.set_user_callback(_ui)
        ps.show()

        summary_dir = "outputs/interp_cubes_fall"
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = write_material_properties_summary(
            summary_dir, args, material_1, material_2, cube_materials
        )
        print(f"Material properties summary saved to: {summary_file}")
    else:

        import polyscope as ps
        import subprocess
        import glob

        print(f"\nRunning headless mode for material pair {args.pair}")
        print(f"Materials: {material_1['name']} → {material_2['name']}")
        print(f"Total frames: {n_frames}")

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height(0.0)

        output_dir = f"outputs/interp_cubes_fall/pair_{args.pair}"
        frames_dir = f"{output_dir}/frames"
        os.makedirs(frames_dir, exist_ok=True)

        cam_file = "cam/interp_cubes_fall.json"
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
        for i, (sim, mat) in enumerate(zip(sims, cube_materials)):
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

            transparency = 0.7 if i == 0 or i == 4 else 0.6
            m = ps.register_volume_mesh(
                f"cube_{i}_{mat['name'].replace(' ', '_')}",
                recorded[i][0],
                hexes=hexes,
                tets=tets,
                edge_width=0.0,
                transparency=transparency,
            )

            if i == 0 or i == 4:
                if i == 0:
                    m.set_color((0.2, 0.8, 0.2))
                else:
                    m.set_color((0.8, 0.8, 0.2))
            else:

                interp_factor = (i - 1) / 2
                blue_intensity = 0.3 + 0.5 * interp_factor
                m.set_color((0.2, 0.2, blue_intensity))

            meshes.append(m)

        screenshot_frames = [0, 8, 16, 20, 25, 199]
        screenshot_frames = [f for f in screenshot_frames if f < n_frames]
        print(
            f"Saving {len(screenshot_frames)} screenshots at frames: {screenshot_frames}"
        )

        for i, frame_idx in enumerate(screenshot_frames):
            for m, rec in zip(meshes, recorded):
                m.update_vertex_positions(rec[frame_idx])

            filename = f"{output_dir}/frame_{frame_idx:06d}.png"
            ps.screenshot(filename)
            print(f"Screenshot saved: {filename}")

        print(f"Saving all {n_frames} frames for video creation...")

        for frame_idx in range(n_frames):
            for m, rec in zip(meshes, recorded):
                m.update_vertex_positions(rec[frame_idx])

            filename = f"{frames_dir}/frame_{frame_idx:06d}.png"
            ps.screenshot(filename)

            if frame_idx % 25 == 0:
                print(f"  Frame {frame_idx+1}/{n_frames}")

        video_filename = f"{output_dir}/animation_pair_{args.pair}.mp4"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "20",
            "-i",
            f"{frames_dir}/frame_%06d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            video_filename,
        ]

        try:
            print(f"Creating video: {video_filename}")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"Video created successfully: {video_filename}")

            frame_files = glob.glob(f"{frames_dir}/*.png")
            for frame_file in frame_files:
                os.remove(frame_file)
            os.rmdir(frames_dir)
            print(f"Cleaned up {len(frame_files)} frame images")

        except subprocess.CalledProcessError as e:
            print(f"Error creating video with ffmpeg: {e}")
            print(f"Frame images preserved in: {frames_dir}")
        except FileNotFoundError:
            print("ffmpeg not found. Install ffmpeg to create videos.")
            print(f"Frame images saved in: {frames_dir}")

        print(f"\nOutputs saved to {output_dir}/")
        print(f"Material pair {args.pair}: {material_1['name']} → {material_2['name']}")
        print(f"Simulation completed: {n_frames} frames")
        print("Screenshots saved:")
        print(
            f"  - {len(screenshot_frames)} screenshots at frames: 0, 8, 16, 20, 25, 199"
        )
        print("Animation video created (if ffmpeg available)")
        print("Color coding:")
        print("  - Green cube (pos 1): Original material 1")
        print("  - Blue cubes (pos 2-4): Interpolated materials")
        print("  - Yellow cube (pos 5): Original material 2")
        print("All cubes fall freely with their respective material properties")

        summary_file = write_material_properties_summary(
            output_dir, args, material_1, material_2, cube_materials
        )
        print(f"Material properties summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
