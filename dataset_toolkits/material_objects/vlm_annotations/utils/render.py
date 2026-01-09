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
from PIL import Image
import polyscope as ps


def create_sphere(radius=1.0, resolution=64):
    """
    Create a sphere mesh with the given radius and resolution.
    Returns vertices, faces, and UV coordinates.
    """
    vertices = []
    uvs = []

    for i in range(resolution + 1):
        theta = i * np.pi / resolution
        for j in range(resolution * 2):
            phi = j * 2 * np.pi / (resolution * 2)

            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            vertices.append([x, y, z])

            u = phi / (2 * np.pi)
            v = 1 - theta / np.pi
            uvs.append([u, v])

    faces = []
    for i in range(resolution):
        for j in range(resolution * 2):
            next_j = (j + 1) % (resolution * 2)

            p1 = i * (resolution * 2) + j
            p2 = i * (resolution * 2) + next_j
            p3 = (i + 1) * (resolution * 2) + j
            p4 = (i + 1) * (resolution * 2) + next_j

            if i > 0:
                faces.append([p1, p2, p3])
            if i < resolution - 1:
                faces.append([p2, p4, p3])

    return np.array(vertices), np.array(faces), np.array(uvs)


def apply_texture_to_vertices(vertices, uvs, texture_path):
    """
    Apply texture from file to vertices based on UV coordinates.
    """
    # Load the texture image
    img = Image.open(texture_path)
    img = img.convert("RGB")
    texture = np.array(img) / 255.0

    height, width, _ = texture.shape
    colors = np.zeros((len(vertices), 3))

    for i, uv in enumerate(uvs):
        x = int(uv[0] * (width - 1))
        y = int((1 - uv[1]) * (height - 1))
        colors[i] = texture[y, x]

    return colors


def apply_normal_mapped_colors(vertices, faces, uvs, normal_path, base_colors):
    """
    Apply normal mapping effect to the base colors.
    """
    # Load the normal map
    img = Image.open(normal_path)
    img = img.convert("RGB")
    normal_texture = np.array(img) / 255.0

    height, width, _ = normal_texture.shape

    # Convert normal map from [0,1] range to [-1,1] range
    normals_from_map = normal_texture * 2.0 - 1.0

    # Simple lighting setup (directional light from above-right)
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    ambient = 0.3  # Ambient light intensity
    diffuse = 0.7  # Diffuse light intensity

    # Apply normal-based lighting
    adjusted_colors = base_colors.copy()
    for i, uv in enumerate(uvs):
        x = int(uv[0] * (width - 1))
        y = int((1 - uv[1]) * (height - 1))

        # Extract normal from texture
        normal = normals_from_map[y, x]
        normal = normal / np.linalg.norm(normal)

        # Calculate lighting factor using normal
        diffuse_factor = max(0, np.dot(normal, light_dir))
        light_factor = ambient + diffuse * diffuse_factor

        # Apply lighting to base color
        adjusted_colors[i] = base_colors[i] * light_factor

        # Ensure values stay in [0,1] range
        adjusted_colors[i] = np.clip(adjusted_colors[i], 0, 1)

    return adjusted_colors


def render_sphere_with_texture(textures_dict, output_path=None):
    """
    Render a sphere with the given textures.
    Returns the RGB numpy array of the rendered image.

    Args:
        textures_dict (dict): Dictionary of texture paths by type
        output_path (str, optional): Path to save the output image if desired

    Returns:
        numpy.ndarray: RGB image array of the rendered sphere
    """
    # Create sphere mesh
    vertices, faces, uvs = create_sphere(radius=1.0, resolution=64)

    # Register the sphere mesh with Polyscope
    ps_mesh = ps.register_surface_mesh("sphere", vertices, faces)

    try:
        # Apply textures
        if "albedo" in textures_dict:
            base_colors = apply_texture_to_vertices(
                vertices, uvs, textures_dict["albedo"]
            )
            final_colors = base_colors.copy()

            # Apply normal map if available
            if "normal" in textures_dict:
                final_colors = apply_normal_mapped_colors(
                    vertices, faces, uvs, textures_dict["normal"], final_colors
                )

            # Add the final material color to the mesh
            ps_mesh.add_color_quantity(
                "material", final_colors, defined_on="vertices", enabled=True
            )
        else:
            # Fallback to a simple colored material
            ps_mesh.add_color_quantity(
                "default",
                np.ones((len(vertices), 3)) * np.array([0.7, 0.7, 0.7]),
                defined_on="vertices",
                enabled=True,
            )

    except Exception as e:
        print(f"Error processing textures: {str(e)}")
        ps_mesh.add_color_quantity(
            "default",
            np.ones((len(vertices), 3)) * np.array([0.7, 0.7, 0.7]),
            defined_on="vertices",
            enabled=True,
        )

    # Set up camera and rendering settings
    ps.reset_camera_to_home_view()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")

    ps_mesh.set_smooth_shade(True)

    # Capture screenshot as buffer (RGBA)
    rgba_buffer = ps.screenshot_to_buffer(transparent_bg=False)

    # Convert RGBA to RGB
    h, w, _ = rgba_buffer.shape
    rgb_buffer = np.zeros((h, w, 3), dtype=np.uint8)

    # Just take the RGB channels from RGBA
    rgb_buffer = rgba_buffer[:, :, :3]

    # Save to file if output path is provided
    if output_path:
        # Save the RGB buffer to file
        Image.fromarray(rgb_buffer).save(output_path)

    # Clean up (remove the mesh from polyscope)
    ps.remove_surface_mesh("sphere")

    return rgb_buffer


def process_rgba_to_rgb(image_path):
    """
    Convert RGBA image to RGB, making fully transparent pixels white.
    """
    try:
        # Open the image
        img = Image.open(image_path)

        # Check if the image has an alpha channel
        if img.mode == "RGBA":
            # Create a white background
            background = Image.new("RGB", img.size, (255, 255, 255))

            # Paste the image on the background using itself as mask
            background.paste(img, mask=img.split()[3])

            # Save the result to the same path
            background.save(image_path.replace(".png", "_rgb.png"))
            return image_path.replace(".png", "_rgb.png")
        elif img.mode != "RGB":
            # Convert any other mode to RGB
            img = img.convert("RGB")
            img.save(image_path.replace(".png", "_rgb.png"))
            return image_path.replace(".png", "_rgb.png")
        else:
            # Already RGB, return the original path
            return image_path

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
