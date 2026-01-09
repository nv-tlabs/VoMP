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

"""
Replicator-based rendering for VoMP material inference.

This module provides an alternative to Blender rendering using NVIDIA Isaac Sim's
Replicator for photorealistic rendering of 3D assets.
"""

import os
import json
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Union
import numpy as np

# Preset RTX configurations
RTX_PRESETS = {
    "fast": {
        "/rtx/rendermode": "RayTracedLighting",  # Real-time ray tracing
        "/rtx/post/backgroundZeroAlpha/enabled": True,
        "/rtx/post/backgroundZeroAlpha/backgroundComposite": False,
        "/rtx/post/backgroundZeroAlpha/outputAlphaInComposite": True,
    },
    "path_tracing": {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 256,
        "/rtx/pathtracing/totalSpp": 256,
        "/rtx/pathtracing/maxBounces": 8,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 8,
        "/rtx/post/backgroundZeroAlpha/enabled": True,
        "/rtx/post/backgroundZeroAlpha/backgroundComposite": False,
        "/rtx/post/backgroundZeroAlpha/outputAlphaInComposite": True,
        "/rtx/pathtracing/fireflyFilter/enable": True,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
    },
}


def generate_replicator_script(
    asset_path: str,
    output_dir: str,
    num_views: int,
    resolution: tuple,
    yaws: List[float],
    pitchs: List[float],
    radius_list: List[float],
    fov_list: List[float],
    rtx_settings: Dict[str, Any],
    light_intensity: float = 1000.0,
    normalize_object: bool = True,
) -> str:
    """
    Generate a Replicator rendering script for Isaac Sim execution.

    Args:
        asset_path: Path to USD asset file
        output_dir: Directory to save renders
        num_views: Number of views to render
        resolution: Tuple of (width, height)
        yaws: List of yaw angles
        pitchs: List of pitch angles
        radius_list: List of camera radii
        fov_list: List of FOV values
        rtx_settings: RTX renderer settings dictionary
        light_intensity: Sphere light intensity
        normalize_object: Whether to normalize object to [-0.5, 0.5]

    Returns:
        String containing the complete Python script
    """
    script = f'''"""
Auto-generated Replicator rendering script for VoMP.
Generated for asset: {os.path.basename(asset_path)}
"""

import os
import omni.replicator.core as rep
import omni.replicator.core.functional as F
from PIL import Image
import carb
import numpy as np
import math
import omni.usd
from pxr import UsdGeom, Gf, Usd
import json

# Configuration
RESOLUTION = {resolution}
ASSET_PATH = "{asset_path}"
OUTPUT_DIR = "{output_dir}"
NUM_CAMERAS = {num_views}
LIGHT_INTENSITY = {light_intensity}
NORMALIZE_OBJECT = {normalize_object}

# Camera parameters (pre-computed)
YAWS = {yaws}
PITCHS = {pitchs}
RADIUS_LIST = {radius_list}
FOV_LIST = {fov_list}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
renders_dir = os.path.join(OUTPUT_DIR, "renders")
os.makedirs(renders_dir, exist_ok=True)

# RTX Settings
RTX_SETTINGS = {rtx_settings}

print("Configuring RTX renderer...")
for setting_path, value in RTX_SETTINGS.items():
    carb.settings.get_settings().set(setting_path, value)
print("✓ RTX settings applied")

# Set stage Z-Up
rep.settings.set_stage_up_axis("Z")
rep.settings.set_stage_meters_per_unit(1.0)


def normalize_asset_to_unit_box(asset_prim_path, target_scale=0.98):
    """Normalize asset to fit within [-0.5, 0.5] coordinate space."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(asset_prim_path)
    
    if not prim:
        print(f"Warning: Could not find prim at {{asset_prim_path}}")
        return 1.0, (0, 0, 0)
    
    # Compute bounding box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.GetRange()
    
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    
    center = (min_point + max_point) / 2.0
    extent_vec = max_point - min_point
    max_extent = max(extent_vec[0], extent_vec[1], extent_vec[2])
    
    if max_extent == 0:
        print("Warning: Object has zero extent")
        return 1.0, (0, 0, 0)
    
    scale_factor = target_scale / max_extent
    
    print(f"Object normalization:")
    print(f"  Max extent: {{max_extent:.3f}}")
    print(f"  Scale factor: {{scale_factor:.6f}}")
    
    return scale_factor, center


# Import asset
print(f"Loading asset: {{ASSET_PATH}}")
asset = F.create.reference(ASSET_PATH)

# Normalize if requested
if NORMALIZE_OBJECT:
    stage = omni.usd.get_context().get_stage()
    asset_prim_path = asset.GetPath()
    
    scale_factor, center = normalize_asset_to_unit_box(asset_prim_path, target_scale=0.98)
    
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(asset_prim_path))
    xformable.ClearXformOpOrder()
    
    scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale_factor, scale_factor, scale_factor))
    
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3f(-center[0] * scale_factor, -center[1] * scale_factor, -center[2] * scale_factor))
    
    print("✓ Asset normalized to [-0.5, 0.5]")

# Setup annotator
anno = rep.annotators.get("rgb")

# Store frame metadata
frames_metadata = []


async def main():
    """Render all camera views."""
    print(f"\\nRendering {{NUM_CAMERAS}} views...")
    
    for i in range(NUM_CAMERAS):
        yaw = YAWS[i]
        pitch = PITCHS[i]
        radius = RADIUS_LIST[i]
        fov = FOV_LIST[i]
        
        # Convert spherical to Cartesian
        x = radius * math.cos(pitch) * math.cos(yaw)
        y = radius * math.cos(pitch) * math.sin(yaw)
        z = radius * math.sin(pitch)
        
        # Create camera
        camera = F.create.camera(position=[x, y, z], look_at=asset)
        
        # Add light
        light = F.create.sphere_light(parent=camera, intensity=LIGHT_INTENSITY)
        
        # Create render product
        render_product = rep.create.render_product(camera, RESOLUTION)
        anno.attach(render_product)
        
        # Render
        await rep.orchestrator.step_async()
        
        # Save image
        filename = f"{{i:04d}}.png"
        output_path = os.path.join(renders_dir, filename)
        Image.fromarray(anno.get_data()).save(output_path)
        
        # Create camera transform matrix for metadata
        # Camera to world transform
        cam_pos = np.array([x, y, z])
        
        # Look-at transformation (simplified)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross([0, 0, 1], forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross([0, 1, 0], forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        # Build camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        
        # Convert to NeRF convention (flip Y and Z)
        c2w[:3, 1:3] *= -1
        
        frame_metadata = {{
            "file_path": filename,
            "transform_matrix": c2w.tolist(),
            "camera_angle_x": np.radians(fov),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "radius": float(radius),
        }}
        frames_metadata.append(frame_metadata)
        
        print(f"  Rendered view {{i+1}}/{{NUM_CAMERAS}}")
        
        # Clean up
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(camera.GetPath()):
            stage.RemovePrim(camera.GetPath())
        if stage.GetPrimAtPath(light.GetPath()):
            stage.RemovePrim(light.GetPath())
    
    # Save transforms.json
    transforms_data = {{
        "camera_angle_x": np.radians(FOV_LIST[0]) if FOV_LIST else 0.0,
        "frames": frames_metadata,
    }}
    
    transforms_path = os.path.join(renders_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(transforms_data, f, indent=2)
    
    # Save metadata for pipeline compatibility
    metadata_path = os.path.join(OUTPUT_DIR, "renders_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(frames_metadata, f, indent=2)
    
    print(f"\\n✓ All {{NUM_CAMERAS}} renders completed!")
    print(f"✓ Output: {{OUTPUT_DIR}}")
    
    # Exit
    carb.settings.get_settings().set("/app/file/ignoreUnsavedOnExit", True)
    omni.kit.app.get_app().post_quit()


import asyncio
asyncio.ensure_future(main())
'''
    return script


def render_with_replicator(
    asset_path: str,
    output_dir: str,
    num_views: int,
    yaws: List[float],
    pitchs: List[float],
    radius_list: List[float],
    fov_list: List[float],
    isaac_sim_path: str,
    resolution: tuple = (1024, 1024),
    render_mode: str = "path_tracing",
    rtx_settings_override: Optional[Dict[str, Any]] = None,
    light_intensity: float = 1000.0,
    normalize_object: bool = True,
) -> List[Dict[str, Any]]:
    """
    Render views using Isaac Sim Replicator in headless mode.

    Args:
        asset_path: Path to USD asset file (.usd, .usda, .usdc)
        output_dir: Directory to save rendered images
        num_views: Number of camera views to render
        yaws: List of yaw angles (azimuth) for camera positions
        pitchs: List of pitch angles (elevation) for camera positions
        radius_list: List of camera distances from object
        fov_list: List of field-of-view values (degrees)
        isaac_sim_path: Path to Isaac Sim executable (isaac-sim.sh)
        resolution: Tuple of (width, height) for rendered images
        render_mode: Rendering quality preset - "fast" or "path_tracing"
        rtx_settings_override: Optional dict to override RTX settings
        light_intensity: Sphere light intensity value
        normalize_object: Whether to normalize object to [-0.5, 0.5]

    Returns:
        List of frame metadata dictionaries with camera transforms

    Raises:
        FileNotFoundError: If Isaac Sim executable not found
        RuntimeError: If rendering fails

    Example:
        >>> yaws, pitchs, radii, fovs = sample_camera_views(150)
        >>> frames = render_with_replicator(
        ...     asset_path="model.usd",
        ...     output_dir="./renders",
        ...     num_views=150,
        ...     yaws=yaws, pitchs=pitchs,
        ...     radius_list=radii, fov_list=fovs,
        ...     isaac_sim_path="~/isaac-sim/isaac-sim.sh",
        ...     render_mode="path_tracing"
        ... )
    """
    # Validate Isaac Sim path
    isaac_sim_path = os.path.expanduser(isaac_sim_path)
    if not os.path.exists(isaac_sim_path):
        raise FileNotFoundError(
            f"Isaac Sim executable not found at: {isaac_sim_path}\n"
            f"Please provide the correct path to isaac-sim.sh"
        )

    # Get RTX settings
    if render_mode not in RTX_PRESETS:
        raise ValueError(
            f"Invalid render_mode: {render_mode}. Must be 'fast' or 'path_tracing'"
        )

    rtx_settings = RTX_PRESETS[render_mode].copy()

    # Apply user overrides
    if rtx_settings_override:
        rtx_settings.update(rtx_settings_override)

    print(f"=== Replicator Rendering with Isaac Sim ===")
    print(f"Render mode: {render_mode}")
    print(f"Asset: {os.path.basename(asset_path)}")
    print(f"Views: {num_views}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")

    # Generate rendering script
    script_content = generate_replicator_script(
        asset_path=os.path.abspath(asset_path),
        output_dir=os.path.abspath(output_dir),
        num_views=num_views,
        resolution=resolution,
        yaws=yaws,
        pitchs=pitchs,
        radius_list=radius_list,
        fov_list=fov_list,
        rtx_settings=rtx_settings,
        light_intensity=light_intensity,
        normalize_object=normalize_object,
    )

    # Create temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=output_dir
    ) as f:
        script_path = f.name
        f.write(script_content)

    print(f"Generated rendering script: {script_path}")

    try:
        # Execute Isaac Sim in headless mode
        print(f"Launching Isaac Sim in headless mode...")
        cmd = [isaac_sim_path, "--no-window", "--exec", script_path]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"Isaac Sim stderr:\n{result.stderr}")
            raise RuntimeError(
                f"Isaac Sim rendering failed with return code {result.returncode}"
            )

        print(f"✓ Isaac Sim rendering completed")

        # Load and return metadata
        metadata_path = os.path.join(output_dir, "renders_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                frames_metadata = json.load(f)
            print(f"✓ Loaded metadata for {len(frames_metadata)} frames")
            return frames_metadata
        else:
            raise RuntimeError(
                f"Metadata file not found: {metadata_path}\n"
                f"Isaac Sim may not have completed successfully"
            )

    finally:
        # Clean up temporary script
        if os.path.exists(script_path):
            os.remove(script_path)
            print(f"Cleaned up temporary script")
