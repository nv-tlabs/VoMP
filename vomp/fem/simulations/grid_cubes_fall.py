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
from vomp.fem.simulations.material_grids_generated import (
    get_material_properties_setting_1,
    get_material_properties_setting_2,
    get_material_properties_setting_3,
)


def build_cube_geo(res, offset, height=2.0):
    """Return a Grid3D geometry for one cube centred at offset (x,z)."""
    lo = wp.vec3(offset[0], height, offset[1])
    hi = wp.vec3(offset[0] + 1.0, height + 1.0, offset[1] + 1.0)
    return fem.Grid3D(res=wp.vec3i(res), bounds_lo=lo, bounds_hi=hi)


def main():
    wp.init()

    parser = argparse.ArgumentParser(
        description="Grid of falling cubes with different material properties"
    )
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--grid_n", type=int, default=5, help="NxN cubes")
    parser.add_argument(
        "--spacing", type=float, default=1.5, help="center spacing between cubes"
    )
    parser.add_argument(
        "--drop_height", type=float, default=2.5, help="initial height of cubes"
    )
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)

    # Material property setting
    parser.add_argument(
        "--setting",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Material property setting: 1 (YM=1e4), 2 (YM=1e5), 3 (YM=1e6)",
    )

    # View selection for headless mode
    parser.add_argument(
        "--view",
        type=int,
        choices=[1, 2],
        help="Camera view to use in headless mode: 1 or 2. If not specified, uses all available views.",
    )

    # Add GroundCollidingSim arguments
    GroundCollidingSim.add_parser_arguments(parser)

    # Set some defaults suitable for falling cubes
    parser.set_defaults(
        n_frames=200,
        gravity=9.81,
        quasi_quasistatic=False,
        young_modulus=1e4,  # will be overridden per cube
        poisson_ratio=0.45,  # will be overridden per cube
        density=1000.0,  # will be overridden per cube
        # Collision / ground defaults - important for proper collision!
        ground=True,
        ground_height=0.0,
        collision_radius=0.05,
        n_newton=10,  # More iterations for stability with collision
    )

    args = parser.parse_args()

    # Get pre-computed material properties for the selected setting
    setting_functions = {
        1: get_material_properties_setting_1,
        2: get_material_properties_setting_2,
        3: get_material_properties_setting_3,
    }

    materials_grid = setting_functions[args.setting]()

    # Convert 2D grid to flat arrays
    n_cubes = args.grid_n * args.grid_n
    youngs = np.zeros(n_cubes)
    poissons = np.zeros(n_cubes)
    densities = np.zeros(n_cubes)

    cube_idx = 0
    for iz in range(args.grid_n):
        for ix in range(args.grid_n):
            ym, poisson, density = materials_grid[iz][ix]
            youngs[cube_idx] = ym
            poissons[cube_idx] = poisson
            densities[cube_idx] = density
            cube_idx += 1

    print(f"Using pre-computed material property setting {args.setting}")
    print(f"Creating {n_cubes} cubes with interpolated material properties:")
    print(
        f"Middle cube (setting {args.setting}): Young's modulus = {[5e4, 5e5, 1e6][args.setting-1]:.0e}"
    )
    print(f"Young's modulus range: {youngs.min():.0e} - {youngs.max():.0e}")
    print(f"Poisson ratio range: {poissons.min():.3f} - {poissons.max():.3f}")
    print(f"Density range: {densities.min():.0f} - {densities.max():.0f}")
    print()

    sims = []
    cube_idx = 0

    for ix in range(args.grid_n):
        for iz in range(args.grid_n):
            offset = (ix * args.spacing, iz * args.spacing)
            geo = build_cube_geo(args.resolution, offset, height=args.drop_height)

            # Create local args with specific material properties for this cube
            local_args = copy.copy(args)
            local_args.young_modulus = youngs[cube_idx]
            local_args.poisson_ratio = poissons[cube_idx]
            local_args.density = densities[cube_idx]

            sim = GroundCollidingSim(geo, None, local_args)
            sim.init_displacement_space()
            sim.init_strain_spaces()
            sim.init_collision_detector()
            sim.set_boundary_condition(boundary_projector_form=None)

            # allocate constant matrices needed by Newton solver
            sim.init_constant_forms()
            sim.project_constant_forms()

            sims.append(sim)
            cube_idx += 1

    # record frames for all sims
    n_frames = args.n_frames if hasattr(args, "n_frames") and args.n_frames > 0 else 100
    recorded = [[] for _ in sims]

    # record initial rest pose
    for i, sim in enumerate(sims):
        recorded[i].append(
            sim.u_field.space.node_positions().numpy() + sim.u_field.dof_values.numpy()
        )

    # Run simulation
    for frame in range(n_frames):
        for i, sim in enumerate(sims):
            sim.run_frame()
            recorded[i].append(
                sim.u_field.space.node_positions().numpy()
                + sim.u_field.dof_values.numpy()
            )

        # Print progress every 20 frames
        if frame % 20 == 0:
            print(f"Frame {frame+1}/{n_frames}")

    if args.ui:
        import polyscope as ps
        import polyscope.imgui as psim

        ps.init()
        ps.set_window_size(1920, 1080)
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height(0.0)

        # Create cam directory if it doesn't exist
        os.makedirs("cam", exist_ok=True)

        # Load camera view if it exists - try view1 first, then fallback to old format
        cam_file = "cam/grid_cubes_fall_view1.json"
        if not os.path.exists(cam_file):
            cam_file = "cam/grid_cubes_fall.json"

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
            m = ps.register_volume_mesh(
                f"cube_{i}",
                recorded[i][0],
                hexes=hexes,
                tets=tets,
                edge_width=0.0,
                transparency=0.6,
            )
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

            # Two camera capture buttons
            if psim.Button("Capture Camera View 1"):
                # Get current camera view as JSON
                view_json = ps.get_view_as_json()
                # Save to first camera file
                filename = "cam/grid_cubes_fall_view1.json"
                # Save to file
                with open(filename, "w") as f:
                    json.dump(json.loads(view_json), f, indent=2)
                print(f"Camera view 1 saved to {filename}")

            psim.SameLine()
            if psim.Button("Capture Camera View 2"):
                # Get current camera view as JSON
                view_json = ps.get_view_as_json()
                # Save to second camera file
                filename = "cam/grid_cubes_fall_view2.json"
                # Save to file
                with open(filename, "w") as f:
                    json.dump(json.loads(view_json), f, indent=2)
                print(f"Camera view 2 saved to {filename}")

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

            # Material properties display
            psim.Text("Material Properties:")
            psim.Text(
                f"Setting: {args.setting} (Middle cube YM = {[5e4, 5e5, 1e6][args.setting-1]:.0e})"
            )
            psim.Text(f"Young's modulus: {youngs.min():.0e} - {youngs.max():.0e}")
            psim.Text(f"Poisson ratio: {poissons.min():.3f} - {poissons.max():.3f}")
            psim.Text(f"Density: {densities.min():.0f} - {densities.max():.0f}")
            psim.Text(f"Total cubes: {len(sims)}")
            psim.Text("Properties interpolated from middle cube outward")

            if play[0] and time.time() - last[0] > 0.05:
                current[0] = (current[0] + 1) % n_frames
                for m, rec in zip(meshes, recorded):
                    m.update_vertex_positions(rec[current[0]])
                last[0] = time.time()

        ps.set_user_callback(_ui)
        ps.show()
    else:
        # Headless mode - save screenshots
        import polyscope as ps
        import subprocess
        import glob

        print(f"\nRunning headless mode for material setting {args.setting}")
        print(f"Middle cube Young's modulus: {[5e4, 5e5, 1e6][args.setting-1]:.0e} Pa")

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height(0.0)

        # Create output directory with setting subfolder
        output_dir = f"outputs/grid_cubes_fall/setting_{args.setting}"
        os.makedirs(output_dir, exist_ok=True)

        # Check which camera views exist and filter by --view argument if specified
        all_camera_views = []
        for view_name in ["view1", "view2"]:
            cam_file = f"cam/grid_cubes_fall_{view_name}.json"
            if os.path.exists(cam_file):
                all_camera_views.append((view_name, cam_file))

        # Fallback to old format if no new views exist
        if not all_camera_views:
            cam_file = "cam/grid_cubes_fall.json"
            if os.path.exists(cam_file):
                all_camera_views.append(("default", cam_file))

        # Filter views based on --view argument
        if args.view is not None:
            view_name = f"view{args.view}"
            cam_file = f"cam/grid_cubes_fall_{view_name}.json"
            if os.path.exists(cam_file):
                camera_views = [(view_name, cam_file)]
                print(f"Using specified view {args.view}")
            else:
                print(
                    f"Warning: View {args.view} not found ({cam_file}). Available views:"
                )
                for name, file in all_camera_views:
                    print(f"  - {name}: {file}")
                camera_views = []
        else:
            camera_views = all_camera_views

        if not camera_views:
            print("No camera view files found. Using default camera.")
            camera_views.append(("default", None))

        # Register meshes once
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

            m = ps.register_volume_mesh(
                f"cube_{i}",
                recorded[i][0],
                hexes=hexes,
                tets=tets,
                edge_width=0.0,
                transparency=0.6,
            )
            meshes.append(m)

        # Process each camera view
        for view_name, cam_file in camera_views:
            print(f"\nProcessing camera view: {view_name}")

            # Load camera view
            if cam_file and os.path.exists(cam_file):
                try:
                    with open(cam_file, "r") as f:
                        view_data = json.load(f)
                    view_json = json.dumps(view_data)
                    ps.set_view_from_json(view_json)
                    print(f"Loaded camera view from {cam_file}")
                except Exception as e:
                    print(f"Error loading camera view: {e}")

            # Create subdirectory for this view
            view_output_dir = f"{output_dir}/{view_name}"
            frames_dir = f"{view_output_dir}/frames"
            os.makedirs(frames_dir, exist_ok=True)

            # Save 5 screenshots at specific frames
            screenshot_frames = [0, 20, 30, 50, 199]
            # Ensure all frames are within bounds
            screenshot_frames = [f for f in screenshot_frames if f < n_frames]
            print(
                f"Saving {len(screenshot_frames)} key screenshots for {view_name} at frames: {screenshot_frames}"
            )

            if len(screenshot_frames) < 5:
                print(
                    f"Warning: Only {len(screenshot_frames)} frames available (n_frames={n_frames}). Consider increasing --n_frames to at least 200."
                )

            for i, frame_idx in enumerate(screenshot_frames):
                # Update mesh positions for this frame
                for m, rec in zip(meshes, recorded):
                    m.update_vertex_positions(rec[frame_idx])

                # Take screenshot
                filename = (
                    f"{view_output_dir}/key_frame_{i+1:02d}_frame_{frame_idx:06d}.png"
                )
                ps.screenshot(filename)
                print(f"Key screenshot saved: {filename}")

            # Save all frames for video creation
            print(f"Saving all {n_frames} frames for video creation...")

            for frame_idx in range(n_frames):
                # Update mesh positions for this frame
                for m, rec in zip(meshes, recorded):
                    m.update_vertex_positions(rec[frame_idx])

                # Take screenshot for video
                filename = f"{frames_dir}/frame_{frame_idx:06d}.png"
                ps.screenshot(filename)

                # Progress indicator
                if frame_idx % 10 == 0:
                    print(f"  Frame {frame_idx+1}/{n_frames}")

            # Create video using ffmpeg
            video_filename = f"{view_output_dir}/animation_{view_name}.mp4"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # -y to overwrite output file
                "-framerate",
                "20",  # 20 FPS
                "-i",
                f"{frames_dir}/frame_%06d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # High quality
                video_filename,
            ]

            try:
                print(f"Creating video: {video_filename}")
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Video created successfully: {video_filename}")

                # Delete frame images after successful video creation
                frame_files = glob.glob(f"{frames_dir}/*.png")
                for frame_file in frame_files:
                    os.remove(frame_file)
                os.rmdir(frames_dir)
                print(f"Cleaned up {len(frame_files)} frame images")

            except subprocess.CalledProcessError as e:
                print(f"Error creating video with ffmpeg: {e}")
                print(f"Frame images preserved in: {frames_dir}")
            except FileNotFoundError:
                print("ffmpeg not found. Please install ffmpeg to create videos.")
                print(f"Frame images saved in: {frames_dir}")
                print("You can manually create the video with:")
                print(
                    f"  ffmpeg -framerate 20 -i {frames_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {video_filename}"
                )

        print(f"\nAll outputs saved to {output_dir}/")
        print(
            f"Material setting {args.setting}: {[5e4, 5e5, 1e6][args.setting-1]:.0e} Pa Young's modulus"
        )
        if args.view:
            print(f"Processed view {args.view} with {n_frames} frames")
        else:
            print(
                f"Processed {len(camera_views)} camera view(s) with {n_frames} frames each"
            )
        if camera_views:
            print("Each view contains:")
            print("  - Key frame screenshots at frames: 0, 20, 30, 50, 199")
            print("  - Full animation video (if ffmpeg available)")
            print(
                f"Structure: outputs/grid_cubes_fall/setting_{args.setting}/[view_name]/"
            )

        # Show which views were processed
        if camera_views:
            print("Views processed:")
            for view_name, cam_file in camera_views:
                print(f"  - {view_name}: {cam_file if cam_file else 'default camera'}")


if __name__ == "__main__":
    main()
