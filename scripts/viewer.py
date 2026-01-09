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
import polyscope as ps
import argparse
import sys
import os
import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase


def load_npz_file(filename):
    """Load an npz file and extract point cloud data."""
    try:
        # Load the npz file
        npz_data = np.load(filename)
        print(f"Successfully loaded {filename}")
        print(f"Available arrays in the file: {list(npz_data.files)}")

        # Dictionary to store all point clouds found
        point_clouds = {}

        # Iterate through all arrays in the npz file
        for array_name in npz_data.files:
            array_data = npz_data[array_name]
            print(
                f"\nArray '{array_name}': shape={array_data.shape}, dtype={array_data.dtype}"
            )

            # Check if this is a structured array with coordinate fields
            if array_data.dtype.names is not None:
                print(f"  Structured array with fields: {array_data.dtype.names}")

                # Look for coordinate fields
                coord_fields = []
                if "x" in array_data.dtype.names and "y" in array_data.dtype.names:
                    coord_fields = ["x", "y"]
                    if "z" in array_data.dtype.names:
                        coord_fields.append("z")

                    # Extract coordinates
                    if len(coord_fields) == 2:
                        # 2D coordinates
                        points = np.column_stack(
                            [array_data[field] for field in coord_fields]
                        )
                        # Add z=0 to make 3D
                        points_3d = np.column_stack([points, np.zeros(points.shape[0])])
                        point_clouds[array_name] = points_3d.astype(np.float32)
                        print(
                            f"  -> Extracted 2D coordinates from fields {coord_fields} (converted to 3D)"
                        )
                    else:
                        # 3D coordinates
                        points = np.column_stack(
                            [array_data[field] for field in coord_fields]
                        )
                        point_clouds[array_name] = points.astype(np.float32)
                        print(
                            f"  -> Extracted 3D coordinates from fields {coord_fields}"
                        )

                    print(f"  -> Found {points.shape[0]} points")

                    # Store additional scalar data as attributes
                    additional_data = {}
                    for field_name in array_data.dtype.names:
                        if field_name not in coord_fields:
                            field_data = array_data[field_name]
                            # Only add numeric fields as scalar quantities
                            if np.issubdtype(field_data.dtype, np.number):
                                additional_data[field_name] = field_data
                                print(
                                    f"    -> Will add '{field_name}' as scalar quantity"
                                )

                    # Store additional data with the points
                    if additional_data:
                        point_clouds[array_name + "_metadata"] = additional_data
                else:
                    print(f"  -> No coordinate fields (x, y) found in structured array")

            # Check if this could be a regular point cloud (should be Nx2 or Nx3)
            elif len(array_data.shape) == 2:
                if array_data.shape[1] == 2:
                    # 2D points - add z=0 to make them 3D
                    points_3d = np.column_stack(
                        [array_data, np.zeros(array_data.shape[0])]
                    )
                    point_clouds[array_name] = points_3d.astype(np.float32)
                    print(f"  -> Added as 2D point cloud (converted to 3D)")
                elif array_data.shape[1] == 3:
                    # 3D points
                    point_clouds[array_name] = array_data.astype(np.float32)
                    print(f"  -> Added as 3D point cloud")
                elif array_data.shape[1] > 3:
                    # Use only first 3 columns as XYZ coordinates
                    points_3d = array_data[:, :3]
                    point_clouds[array_name] = points_3d.astype(np.float32)
                    print(f"  -> Using first 3 columns as 3D point cloud")
                else:
                    print(
                        f"  -> Skipped (only {array_data.shape[1]} column(s), need at least 2)"
                    )
            elif len(array_data.shape) == 1 and len(array_data) >= 2:
                # 1D array with at least 2 elements - treat as single point
                if len(array_data) == 2:
                    # 2D point
                    point_clouds[array_name] = np.array(
                        [[array_data[0], array_data[1], 0.0]], dtype=np.float32
                    )
                    print(f"  -> Added as single 2D point (converted to 3D)")
                elif len(array_data) >= 3:
                    # 3D point (use first 3 elements)
                    point_clouds[array_name] = np.array(
                        [[array_data[0], array_data[1], array_data[2]]],
                        dtype=np.float32,
                    )
                    print(f"  -> Added as single 3D point")
            else:
                print(f"  -> Skipped (incompatible shape for point cloud)")

        npz_data.close()

        actual_point_clouds_data = []
        for name, data in point_clouds.items():
            if (
                not name.endswith("_metadata")
                and isinstance(data, np.ndarray)
                and len(data.shape) == 2
            ):
                actual_point_clouds_data.append(data)

        if actual_point_clouds_data:
            # Compute overall bounding box across all point clouds
            all_points = np.vstack(actual_point_clouds_data)
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            coord_range = max_coords - min_coords
            max_range = np.max(coord_range)

            print(f"\nOriginal coordinate ranges:")
            print(
                f"  Min: [{min_coords[0]:.6f}, {min_coords[1]:.6f}, {min_coords[2]:.6f}]"
            )
            print(
                f"  Max: [{max_coords[0]:.6f}, {max_coords[1]:.6f}, {max_coords[2]:.6f}]"
            )
            print(f"  Max range: {max_range:.6f}")

            # Normalize to [-0.5, 0.5] range to match the "ours" method scale
            # This ensures camera views are consistent across methods
            if max_range > 1e-10:  # Avoid division by zero
                print(f"  -> Normalizing coordinates to [-0.5, 0.5] range")

                # Calculate center and normalize around it
                center = (min_coords + max_coords) / 2

                # Normalize all point clouds to [-0.5, 0.5] range
                for name, data in point_clouds.items():
                    if (
                        not name.endswith("_metadata")
                        and isinstance(data, np.ndarray)
                        and len(data.shape) == 2
                    ):
                        # Center the data and scale to [-0.5, 0.5]
                        centered_data = data - center
                        normalized_data = (
                            centered_data / max_range
                        )  # Scale to [-0.5, 0.5] approximately
                        point_clouds[name] = normalized_data.astype(np.float32)

                # Print new ranges to verify
                all_normalized_points = np.vstack(
                    [
                        data
                        for name, data in point_clouds.items()
                        if not name.endswith("_metadata")
                        and isinstance(data, np.ndarray)
                        and len(data.shape) == 2
                    ]
                )
                new_min = np.min(all_normalized_points, axis=0)
                new_max = np.max(all_normalized_points, axis=0)
                new_range = np.max(new_max - new_min)
                print(
                    f"  New Min: [{new_min[0]:.6f}, {new_min[1]:.6f}, {new_min[2]:.6f}]"
                )
                print(
                    f"  New Max: [{new_max[0]:.6f}, {new_max[1]:.6f}, {new_max[2]:.6f}]"
                )
                print(f"  New max range: {new_range:.6f}")
            else:
                print(f"  -> Coordinates have no range, skipping normalization")

        return point_clouds

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file '{filename}': {str(e)}")
        return None


def create_matplotlib_legend(data, property_name, colormap="viridis", filename=None):
    """Create a matplotlib-style colorbar legend and save it."""
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)

    # Remove the main axis
    ax.remove()

    # Create colorbar
    cmap = plt.cm.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

    # Create colorbar axis
    cbar_ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="horizontal")
    # Remove labels - no text on the colorbar
    cb.set_label("")  # Empty label

    # Remove tick labels if you want completely clean
    # cb.set_ticks([])  # Uncomment this line to also remove tick marks/numbers

    # No title - clean colorbar only
    # plt.suptitle("", fontsize=16, fontweight="bold")  # Removed title

    if filename is None:
        filename = f"{property_name}_colorbar_legend.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved colorbar legend: {filename}")
    return filename


def visualize_points(
    point_clouds,
    show_individual=True,
    show_combined=True,
    camera_file=None,
    headless=False,
):
    """Visualize point clouds using polyscope."""

    # Initialize polyscope
    ps.init()

    # Set some nice viewing options
    ps.set_up_dir("z_up")  # Z axis pointing up
    ps.set_ground_plane_mode("shadow_only")  # Show shadows on ground plane

    # Load camera view if provided
    if camera_file:
        try:
            if os.path.exists(camera_file):
                with open(camera_file, "r") as f:
                    camera_json_str = f.read()  # Read the JSON string directly
                ps.set_view_from_json(camera_json_str)
                print(f"Loaded camera view from: {camera_file}")
            else:
                print(
                    f"Warning: Camera file '{camera_file}' not found, using default view"
                )
        except Exception as e:
            print(f"Warning: Failed to load camera file '{camera_file}': {str(e)}")
            print("Using default camera view")

    if not point_clouds:
        print("No valid point clouds found to visualize.")
        return

    # Separate point cloud data from metadata
    actual_point_clouds = {}
    metadata_dict = {}

    for name, data in point_clouds.items():
        if name.endswith("_metadata"):
            # This is metadata for another point cloud
            original_name = name[:-9]  # Remove '_metadata' suffix
            metadata_dict[original_name] = data
        else:
            # This is actual point cloud data
            actual_point_clouds[name] = data

    if not actual_point_clouds:
        print("No valid point clouds found to visualize.")
        return

    print(f"\nVisualizing {len(actual_point_clouds)} point cloud(s)...")

    # Properties to create separate visualizations for
    # Handle different naming conventions across methods
    target_properties = ["youngs_modulus", "poissons_ratio", "density"]

    # Create mapping for alternative field names (Phys4DGen/PUGS use different names)
    field_name_mapping = {
        "youngs_modulus": ["youngs_modulus", "young_modulus"],
        "poissons_ratio": ["poissons_ratio", "poisson_ratio"],
        "density": ["density"],
    }

    # Create separate point clouds for each target property
    for cloud_name, points in actual_point_clouds.items():
        if cloud_name in metadata_dict:
            metadata = metadata_dict[cloud_name]

            for prop_name in target_properties:
                # Find the actual field name in the metadata (handle different naming conventions)
                actual_field_name = None
                prop_data = None

                for possible_name in field_name_mapping[prop_name]:
                    if possible_name in metadata:
                        actual_field_name = possible_name
                        prop_data = metadata[possible_name]
                        break

                if prop_data is not None:
                    # Register point cloud for this property
                    ps_cloud = ps.register_point_cloud(
                        f"{prop_name}_visualization", points
                    )

                    # Add the property as a scalar quantity with viridis colormap
                    ps_cloud.add_scalar_quantity(
                        prop_name, prop_data, enabled=True, cmap="viridis"
                    )

                    # Hide the point cloud initially (we'll show them one by one)
                    ps_cloud.set_enabled(False)

                    print(
                        f"  Created visualization for '{prop_name}' (field: '{actual_field_name}'): {points.shape[0]} points"
                    )

                    # Create matplotlib legend for this property (only create legend images)
                    create_matplotlib_legend(prop_data, prop_name, "viridis")
                else:
                    print(
                        f"  Warning: Property '{prop_name}' not found in metadata (tried: {field_name_mapping[prop_name]})"
                    )

    # Enable all point clouds for interactive exploration
    for prop_name in target_properties:
        cloud_name = f"{prop_name}_visualization"
        if ps.has_point_cloud(cloud_name):
            ps.get_point_cloud(cloud_name).set_enabled(True)

    # Add a slice plane for exploring internal structure
    slice_plane = None
    if actual_point_clouds:
        # Calculate center point of all data
        all_points = np.vstack(list(actual_point_clouds.values()))
        center = np.mean(all_points, axis=0)

        # Add slice plane through the center
        slice_plane = ps.add_scene_slice_plane()
        slice_plane.set_pose(
            center, (1.0, 0.0, 0.0)
        )  # Start with X-normal plane through center
        slice_plane.set_draw_plane(True)  # Show the semi-transparent plane
        slice_plane.set_draw_widget(True)  # Show interactive controls

        print(f"Added slice plane at center: {center}")
        print(f"  - Use the widget to drag and rotate the slice plane")
        print(f"  - Or use View -> Slice Planes in the GUI to control it")

    # Define callback functions for UI buttons
    def capture_current_property():
        """Capture and save the currently visible property visualizations."""
        screenshots_taken = []

        # Find which properties are currently enabled
        enabled_properties = []
        for prop_name in target_properties:
            cloud_name = f"{prop_name}_visualization"
            if (
                ps.has_point_cloud(cloud_name)
                and ps.get_point_cloud(cloud_name).is_enabled()
            ):
                enabled_properties.append(prop_name)

        if not enabled_properties:
            print("No property visualizations are currently visible!")
            return

        # Take screenshot of current view
        screenshot_filename = f"captured_properties.png"
        ps.screenshot(screenshot_filename)
        screenshots_taken.append(screenshot_filename)

        # Save camera view as JSON
        camera_json_str = ps.get_view_as_json()
        camera_filename = "camera.json"
        with open(camera_filename, "w") as f:
            f.write(camera_json_str)  # Write the JSON string directly from polyscope

        print(
            f"Captured screenshot of properties {enabled_properties}: {screenshot_filename}"
        )
        print(f"Saved camera view: {camera_filename}")
        return screenshots_taken

    def capture_all_properties():
        """Capture and save each property individually at current camera position."""
        screenshots_taken = []

        # Save camera view as JSON (once, since all screenshots use same camera position)
        camera_json_str = ps.get_view_as_json()
        camera_filename = "camera.json"
        with open(camera_filename, "w") as f:
            f.write(camera_json_str)  # Write the JSON string directly from polyscope

        # Store original slice plane state
        original_slice_active = None
        original_draw_plane = None
        original_draw_widget = None
        if slice_plane is not None:
            original_slice_active = slice_plane.get_active()
            original_draw_plane = slice_plane.get_draw_plane()
            original_draw_widget = slice_plane.get_draw_widget()

        # First pass: Capture images WITHOUT slice plane effects
        if slice_plane is not None:
            slice_plane.set_active(False)  # Disable slice plane effects completely

        for prop_name in target_properties:
            cloud_name = f"{prop_name}_visualization"
            if ps.has_point_cloud(cloud_name):
                # Hide all point clouds first
                for other_prop in target_properties:
                    other_cloud_name = f"{other_prop}_visualization"
                    if ps.has_point_cloud(other_cloud_name):
                        ps.get_point_cloud(other_cloud_name).set_enabled(False)

                # Show only this property's point cloud
                ps.get_point_cloud(cloud_name).set_enabled(True)

                # Take screenshot without slice plane
                screenshot_filename = f"{prop_name}_no_slice.png"
                ps.screenshot(screenshot_filename)
                screenshots_taken.append(screenshot_filename)
                print(f"Captured {prop_name} (no slice): {screenshot_filename}")

        # Second pass: Capture images WITH slice plane effects (but hide visuals)
        if slice_plane is not None:
            slice_plane.set_active(True)  # Enable slice plane effects
            slice_plane.set_draw_plane(False)  # Hide the plane visual
            slice_plane.set_draw_widget(False)  # Hide the widget

        for prop_name in target_properties:
            cloud_name = f"{prop_name}_visualization"
            if ps.has_point_cloud(cloud_name):
                # Hide all point clouds first
                for other_prop in target_properties:
                    other_cloud_name = f"{other_prop}_visualization"
                    if ps.has_point_cloud(other_cloud_name):
                        ps.get_point_cloud(other_cloud_name).set_enabled(False)

                # Show only this property's point cloud
                ps.get_point_cloud(cloud_name).set_enabled(True)

                # Take screenshot with slice plane effects (but visuals hidden)
                screenshot_filename = f"{prop_name}_with_slice.png"
                ps.screenshot(screenshot_filename)
                screenshots_taken.append(screenshot_filename)
                print(f"Captured {prop_name} (with slice): {screenshot_filename}")

        # Restore original slice plane state
        if slice_plane is not None:
            slice_plane.set_active(original_slice_active)
            slice_plane.set_draw_plane(original_draw_plane)
            slice_plane.set_draw_widget(original_draw_widget)

        # Re-enable all point clouds
        for prop_name in target_properties:
            cloud_name = f"{prop_name}_visualization"
            if ps.has_point_cloud(cloud_name):
                ps.get_point_cloud(cloud_name).set_enabled(True)

        print(
            f"Captured {len(screenshots_taken)} property visualizations (6 total: 3 without slice, 3 with slice)"
        )
        print(f"Saved camera view: {camera_filename}")
        return screenshots_taken

    if headless:
        # Headless mode: automatically capture images and exit
        print(f"\nRunning in headless mode...")

        # Print basic statistics
        total_points = sum(points.shape[0] for points in actual_point_clouds.values())
        print(f"Total points loaded: {total_points}")

        if camera_file:
            print(f"Using camera view from: {camera_file}")
        else:
            print(f"Using default camera view")

        print(f"Capturing 6 images...")
        capture_all_properties()

        print(f"Headless mode complete. Generated files:")
        for prop_name in target_properties:
            print(f"  - {prop_name}_no_slice.png")
            print(f"  - {prop_name}_with_slice.png")
            print(f"  - {prop_name}_colorbar_legend.png")
        print(f"  - camera.json")

    else:
        # Interactive mode
        # Add UI callbacks
        def callback():
            if ps.imgui.Button("Capture Current View"):
                capture_current_property()

            if ps.imgui.Button("Capture All Properties"):
                capture_all_properties()

        ps.set_user_callback(callback)

        # Print statistics
        total_points = sum(points.shape[0] for points in actual_point_clouds.values())
        print(f"\nTotal points loaded: {total_points}")

        # Compute and print bounding box
        if total_points > 0:
            all_points = np.vstack(list(actual_point_clouds.values()))
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            print(f"Bounding box: min={min_coords}, max={max_coords}")

        print(f"\nLegend files created:")
        for prop_name in target_properties:
            print(f"  - {prop_name}_colorbar_legend.png (matplotlib colorbar)")

        print(f"\nUse the buttons in the UI to capture visualizations:")
        print(
            f"  - 'Capture Current View': Save current view with visible properties + camera.json"
        )
        print(
            f"  - 'Capture All Properties': Save 6 images (3 without slice, 3 with slice) + camera.json"
        )

        print(f"\nInteractive features:")
        print(f"  - Toggle point cloud visibility in the left panel")
        print(f"  - Drag the slice plane widget to explore internal structure")
        print(f"  - Use View -> Slice Planes in the menu for more slice plane controls")

        if camera_file:
            print(f"\nStarted with camera view from: {camera_file}")
        else:
            print(
                f"\nStarted with default camera view (use --cam <file> to load a saved view)"
            )

        ps.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize point clouds from an NPZ file using polyscope"
    )
    parser.add_argument("filename", help="Path to the NPZ file to load")
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Don't show individual point clouds with different colors",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Don't show combined view of all point clouds",
    )
    parser.add_argument(
        "--cam",
        type=str,
        help="Path to camera.json file to load initial camera view from",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode: automatically capture 6 images and exit (no GUI)",
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' does not exist.")
        sys.exit(1)

    # Load the NPZ file
    point_clouds = load_npz_file(args.filename)

    if point_clouds is None:
        sys.exit(1)

    if not point_clouds:
        print("No valid point clouds found in the file.")
        sys.exit(1)

    # Visualize the points
    show_individual = not args.no_individual
    show_combined = not args.no_combined

    visualize_points(
        point_clouds, show_individual, show_combined, args.cam, args.headless
    )


if __name__ == "__main__":
    main()
