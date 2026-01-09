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

import os
import sys
import glob

from pxr import Usd, UsdShade
import logging
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    round_float_to_2dp,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    SIMREADY_ASSET_CLASS_MAPPING as ASSET_CLASS_MAPPING,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    SIMREADY_MATERIALS_DIR as PHYSICS_MATERIALS_DIR,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    SIMREADY_PROPS_DIR as PROPS_DIR,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.render import (
    render_sphere_with_texture,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.vlm import (
    analyze_material_with_vlm,
    parse_vlm_properties,
)
import re
from tqdm import tqdm

PROMPTS = {
    "few_shot_examples": (
        """
Example 1:
Material: metal
Opacity: opaque
Density: 7800 kg/m^3
Dynamic friction: 0.3
Static friction: 0.4
Restitution: 0.3
Usage: structural component

Analysis: 
Step 1: Based on the images, this appears to be a standard structural steel with a matte gray finish.
Step 2: The surface has medium roughness with some subtle texture visible in the reflection pattern.
Step 3: The physical properties (density, friction values, restitution) are consistent with carbon steel.
Step 4: Considering the usage and measured properties:
   - High stiffness (Young's modulus ~200 GPa) based on typical steel values
   - Medium Poisson's ratio typical of metals
   - High density matching the measured 7800 kg/m^3

Young's modulus: 2.0e11 Pa
Poisson's ratio: 0.29
Density: 7800 kg/m^3

Example 2:
Material: plastic
Opacity: opaque
Density: 950 kg/m^3
Dynamic friction: 0.25
Static friction: 0.35
Restitution: 0.6
Usage: household container

Analysis:
Step 1: The material shows the characteristic smooth, uniform appearance of a consumer plastic.
Step 2: It has moderate gloss with some translucency and a slight texture.
Step 3: The physical properties (medium-low density, moderate friction, higher restitution) match polypropylene.
Step 4: Based on these observations and measurements:
   - Medium-low stiffness typical of polyolefin plastics
   - Higher Poisson's ratio indicating good lateral deformation
   - Density matching the measured 950 kg/m^3

Young's modulus: 1.3e9 Pa
Poisson's ratio: 0.42
Density: 950 kg/m^3

Example 3:
Material: fabric
Opacity: opaque
Density: 300 kg/m^3
Dynamic friction: 0.55
Static friction: 0.75
Restitution: 0.2
Usage: furniture covering

Analysis:
Step 1: The material shows a woven textile structure with visible fibers.
Step 2: The surface has significant texture with a matte appearance and no specular highlights.
Step 3: The physical properties (low density, high friction, low restitution) match a woven textile.
Step 4: Based on these observations and measurements:
   - Low stiffness as expected for flexible textiles
   - Medium-high Poisson's ratio from the woven structure
   - Density matching the measured 300 kg/m^3

Young's modulus: 1.2e8 Pa
Poisson's ratio: 0.38
Density: 300 kg/m^3

Example 4:
Material: organic
Opacity: opaque
Density: 400 kg/m^3
Dynamic friction: 0.45
Static friction: 0.65
Restitution: 0.15
Usage: decorative element

Analysis:
Step 1: This is an organic material with the characteristic structure of natural fibers.
Step 2: The surface shows a natural pattern, matte finish, and relatively brittle structure.
Step 3: The physical properties (low density, moderate-high friction, low restitution) align with plant-based materials.
Step 4: Considering similar organic materials and the measured properties:
   - Low-medium stiffness in the fiber direction
   - Medium Poisson's ratio reflecting the fibrous structure
   - Density matching the measured 400 kg/m^3

Young's modulus: 2.5e9 Pa
Poisson's ratio: 0.30
Density: 400 kg/m^3
"""
    ),
    "query_prompt": (
        """
Based on the provided images and context information, analyze the material properties.
Note: The material segment might be internal to the object and not visible from the outside.

Respond using EXACTLY the following format (do not deviate from this structure):

Analysis: 
Step 1: Identify the material class/type based on visual appearance
Step 2: Describe the surface characteristics (texture, reflectivity, color)
Step 3: Determine the specific material subtype considering its physical properties
Step 4: Reason through each property estimate based on visual and measured data

Young's modulus: <value in scientific notation> Pa
Poisson's ratio: <single decimal value between 0.0 and 0.5>
Density: <value in scientific notation> kg/m^3

Critical Instructions:
1. You MUST provide numerical estimates for ALL materials, including organic or unusual materials
2. For natural materials like leaves, wood, or paper, provide estimates based on similar materials with known properties
3. Never use "N/A", "unknown", or any non-numeric responses for the material properties
4. For Poisson's ratio, provide a simple decimal number (like 0.3 or 0.42)
5. Each property should be on its own line with exact formatting shown above
"""
    ),
}


def make_user_prompt(
    material_type,
    opacity,
    density,
    dynamic_friction,
    static_friction,
    restitution,
    semantic_usage,
    has_texture_sphere=True,
):
    intro_text = (
        """
You are a materials science expert analyzing two images:
1. A photo of the full object (showing how the material appears in context).
2. A sphere with the material's texture (showing color/roughness/reflectivity in isolation).

Using both images and the information below, identify the real-world material and estimate its mechanical properties.
"""
        if has_texture_sphere
        else """
You are a materials science expert analyzing an image of the full object (showing how the material appears in context).

Using this image and the information below, identify the real-world material and estimate its mechanical properties.
"""
    )

    return f"""{intro_text}
Material context:
  * Material type: {material_type}
  * Opacity: {opacity}
  * Density: {density if density is not None else 'unknown'} kg/m^3
  * Dynamic friction: {dynamic_friction if dynamic_friction is not None else 'unknown'}
  * Static friction: {static_friction if static_friction is not None else 'unknown'}
  * Restitution: {restitution if restitution is not None else 'unknown'}
  * Usage: {semantic_usage}

Your task is to provide three specific properties:
1. Young's modulus (in Pa using scientific notation)
2. Poisson's ratio (a value between 0.0 and 0.5)
3. Density (in kg/m^3 using scientific notation)
"""


# Use the centralized parser function from vlm.py instead
parse_vlm_output = parse_vlm_properties


def list_simready_objects():
    """
    List all available props in the SimReady props directory.
    """
    if not os.path.isdir(PROPS_DIR):
        print(f"Error: SimReady props directory not found at {PROPS_DIR}")
        sys.exit(1)

    print("\nAvailable props:")
    for prop_dir in sorted(os.listdir(PROPS_DIR)):
        if os.path.isdir(os.path.join(PROPS_DIR, prop_dir)):
            print(f"  - {prop_dir}")
    print()


def get_usd_file_from_prop_dir(prop_dir):
    inst_base_files = glob.glob(os.path.join(prop_dir, "*_inst_base.usd"))
    if inst_base_files:
        return inst_base_files[0]

    base_files = glob.glob(os.path.join(prop_dir, "*_base.usd"))
    if base_files:
        return base_files[0]

    usd_files = glob.glob(os.path.join(prop_dir, "*.usd"))
    if usd_files:
        return usd_files[0]

    print(f"Error: No USD file found in {prop_dir}")
    sys.exit(1)


def load_physics_materials():
    """
    Load physics material properties from USD files.

    Returns:
        dict: Dictionary mapping material types to physics properties
    """
    physics_materials = {}

    # Check if physics materials directory exists
    if not os.path.isdir(PHYSICS_MATERIALS_DIR):
        print(
            f"Warning: Physics materials directory not found at {PHYSICS_MATERIALS_DIR}"
        )
        print("Physics properties will not be available.")
        return physics_materials

    # Load each physics material file
    for usda_file in glob.glob(os.path.join(PHYSICS_MATERIALS_DIR, "physics_*.usda")):
        material_name = (
            os.path.basename(usda_file).replace("physics_", "").replace(".usda", "")
        )

        try:
            stage = Usd.Stage.Open(usda_file)
            if not stage:
                print(f"Warning: Could not open physics material file {usda_file}")
                continue

            # Find material prims with PhysicsMaterialAPI
            material_prims = []
            for prim in stage.Traverse():
                # Check if the prim is a Material and has PhysicsMaterialAPI applied
                if prim.IsA(UsdShade.Material) or "PhysicsMaterialAPI" in [
                    s.GetName() for s in prim.GetAppliedSchemas()
                ]:
                    material_prims.append(prim)

            if not material_prims:
                # Try to find any prim with physics attributes as a fallback
                for prim in stage.Traverse():
                    if prim.HasAttribute("physics:density") or prim.HasAttribute(
                        "physics:dynamicFriction"
                    ):
                        material_prims.append(prim)
                        break

            if not material_prims:
                print(f"Warning: No physics material found in {usda_file}")
                continue

            material_prim = material_prims[0]

            # Extract physics properties
            density = (
                material_prim.GetAttribute("physics:density").Get()
                if material_prim.HasAttribute("physics:density")
                else None
            )
            dynamic_friction = (
                material_prim.GetAttribute("physics:dynamicFriction").Get()
                if material_prim.HasAttribute("physics:dynamicFriction")
                else None
            )
            static_friction = (
                material_prim.GetAttribute("physics:staticFriction").Get()
                if material_prim.HasAttribute("physics:staticFriction")
                else None
            )
            restitution = (
                material_prim.GetAttribute("physics:restitution").Get()
                if material_prim.HasAttribute("physics:restitution")
                else None
            )

            # Round values to 2 decimal places
            physics_materials[material_name] = {
                "density": round_float_to_2dp(density),
                "dynamic_friction": round_float_to_2dp(dynamic_friction),
                "static_friction": round_float_to_2dp(static_friction),
                "restitution": round_float_to_2dp(restitution),
            }

            # print(f"Loaded physics material '{material_name}' with properties: density={round_float_to_2dp(density)}, dynamic_friction={round_float_to_2dp(dynamic_friction)}, static_friction={round_float_to_2dp(static_friction)}, restitution={round_float_to_2dp(restitution)}")

        except Exception as e:
            print(f"Error loading physics material {usda_file}: {str(e)}")
            import traceback

            traceback.print_exc()

    return physics_materials


def parse_material_name(material_name):
    """
    Parse a material name according to the SimReady convention.
    Format: opacity__material-type__specific-usage

    Args:
        material_name (str): Material name

    Returns:
        tuple: (opacity, material_type, semantic_usage)
    """
    parts = material_name.split("__")

    if len(parts) >= 3:
        opacity = parts[0]
        material_type = parts[1]
        semantic_usage = parts[2]
    elif len(parts) == 2:
        opacity = parts[0]
        material_type = parts[1]
        semantic_usage = ""
    else:
        # Default values if the name doesn't follow the convention
        opacity = "unknown"
        material_type = material_name
        semantic_usage = ""

    return opacity, material_type, semantic_usage


def find_textures_for_material(prop_dir, mesh_name, material_type):
    """
    Find textures associated with a mesh in the prop directory.
    This function is used as a fallback when USD-based texture extraction doesn't find textures.

    Args:
        prop_dir (str): Path to the prop directory
        mesh_name (str): Name of the mesh segment (can be empty for general search)
        material_type (str): Type of material (e.g., "metal", "fabric")

    Returns:
        dict: Dictionary of texture paths by type (albedo, normal, orm)
    """
    textures = {}

    # Handle special debug case for problematic objects
    special_debug = "appleseed_coffeetable" in prop_dir

    if special_debug:
        logging.info(
            f"SPECIAL DEBUG TEXTURES: Finding textures for mesh '{mesh_name}', material type '{material_type}' in {prop_dir}"
        )

    # Check for texture directories, case-insensitive
    texture_directories = []

    # Directory patterns to check (both lowercase and uppercase first letter)
    dir_patterns = [
        os.path.join(prop_dir, "textures"),
        os.path.join(prop_dir, "Textures"),
        os.path.join(prop_dir, "materials", "textures"),
        os.path.join(prop_dir, "materials", "Textures"),
        os.path.join(prop_dir, "Materials", "textures"),
        os.path.join(prop_dir, "Materials", "Textures"),
    ]

    # Check each pattern
    for dir_path in dir_patterns:
        if os.path.isdir(dir_path):
            texture_directories.append(dir_path)

            # Check for material-specific subdirectories
            for subdir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.isdir(subdir_path) and not subdir.startswith("."):
                    texture_directories.append(subdir_path)

    if special_debug:
        logging.info(
            f"SPECIAL DEBUG TEXTURES: Found {len(texture_directories)} texture directories"
        )

    # Try looking in parent directory if no texture directories found
    if not texture_directories:
        parent_dir = os.path.dirname(prop_dir)
        parent_textures_dir_patterns = [
            os.path.join(parent_dir, "textures"),
            os.path.join(parent_dir, "Textures"),
        ]

        for dir_path in parent_textures_dir_patterns:
            if os.path.isdir(dir_path):
                texture_directories.append(dir_path)

    if not texture_directories:
        logging.warning(f"No textures directory found for {prop_dir}")
        return {}

    # Extract prop name and mesh parts for texture matching
    prop_name = os.path.basename(os.path.normpath(prop_dir))

    # Extract the base name without version/variant (e.g., "reflectivetape" from "reflectivetape2m5cm_i01")
    base_prop_name = prop_name.split("_")[0].lower()

    # Extract material info from mesh name for better targeting
    mesh_parts = []
    if mesh_name:
        # Split by camel case or underscores
        import re

        mesh_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", mesh_name)
        mesh_parts = [part.lower() for part in mesh_parts]

        # Also add common prefixes
        if mesh_name.startswith("SM_"):
            mesh_parts.append("sm")
        if "_" in mesh_name:
            for part in mesh_name.split("_"):
                if part.strip():
                    mesh_parts.append(part.lower())

    # Different possible naming patterns
    patterns = []

    # If mesh name is provided, add specific patterns first
    if mesh_name:
        # Direct segment match
        patterns.append(f"{mesh_name}_*")
        patterns.append(f"{mesh_name}*_*")
        # Prop name + segment
        patterns.append(f"{prop_name}_{mesh_name}_*")
        patterns.append(f"{prop_name}_{mesh_name}*_*")

    # More generic patterns
    patterns.extend(
        [
            # Material type
            f"*{material_type}*_*",
            # T_ prefix patterns
            f"T_{base_prop_name}*_*",
            f"T_*{base_prop_name}*_*",
            f"T_*{material_type}*_*",
            # Capitalized T_ patterns for case-insensitive matching
            f"T_{base_prop_name.capitalize()}*_*",
            f"T_{prop_name.capitalize()}*_*",
            f"T_{material_type.capitalize()}*_*",
            # By material type
            f"*{material_type}*.*",
            f"{material_type}*.*",
            # Direct prop name
            f"{prop_name}*.*",
            f"*{base_prop_name}*.*",
            # For materials in subdirectories
            f"T_*_Albedo.png",
            f"T_*_Normal.png",
            f"T_*_ORM.png",
            f"T_*_Roughness.png",
            f"T_*_Metallic.png",
            # Single-letter suffix patterns (e.g., _A.png for Albedo, _N.png for Normal)
            f"*_A.png",  # Albedo
            f"*_N.png",  # Normal
            f"*_R.png",  # Roughness
            f"*_M.png",  # Metallic
            f"*_O.png",  # Occlusion
            # Simplified match based on mesh name parts
            f"*{mesh_name.split('_')[-1] if mesh_name and '_' in mesh_name else ''}*.*",
        ]
    )

    # Add mesh part-based patterns
    for part in mesh_parts:
        if len(part) > 2:  # Skip very short parts
            patterns.append(f"*{part}*.*")

    # For appleseed_coffeetable, add more specific patterns
    if "appleseed_coffeetable" in prop_dir:
        patterns.extend(
            [
                "Appleseed_CoffeeTable*.*",
                "*CoffeeTable*.*",
                "*Coffee*.*",
                "*Table*.*",
                "*coffee*.*",
                "*table*.*",
            ]
        )

        # Add material-specific patterns based on mesh name
        if mesh_name and "Top" in mesh_name:
            patterns.extend(
                ["*Wood*.*", "*top*.*", "*Top*.*"]  # Wood textures for the top
            )
        elif mesh_name and "Leg" in mesh_name:
            patterns.extend(
                ["*Metal*.*", "*leg*.*", "*Leg*.*"]  # Metal textures for the legs
            )

    # Search in all texture directories
    found_textures = []
    for textures_dir in texture_directories:
        all_png_files = glob.glob(os.path.join(textures_dir, "*.png"))

        # Skip hidden directories and files
        all_png_files = [
            f for f in all_png_files if not os.path.basename(f).startswith(".")
        ]

        # Log for directories with .png files
        if all_png_files:
            logging.info(f"Found {len(all_png_files)} PNG files in {textures_dir}")

        # If there are only a few textures in the directory, they're likely for this prop
        if len(all_png_files) <= 10:
            found_textures.extend(all_png_files)
        else:
            # Try specific patterns first
            for pattern in patterns[:-1]:  # Skip the last (wildcard) pattern initially
                matches = glob.glob(os.path.join(textures_dir, pattern))
                if special_debug and matches:
                    logging.info(
                        f"SPECIAL DEBUG TEXTURES: Pattern '{pattern}' matched {len(matches)} files"
                    )
                if matches:
                    found_textures.extend(matches)

            # If no matches, try the generic pattern
            if not found_textures and patterns:
                found_textures = glob.glob(os.path.join(textures_dir, patterns[-1]))
                if special_debug and found_textures:
                    logging.info(
                        f"SPECIAL DEBUG TEXTURES: Generic pattern matched {len(found_textures)} files"
                    )

    if special_debug:
        logging.info(
            f"SPECIAL DEBUG TEXTURES: Found {len(found_textures)} texture files total"
        )
        for texture in found_textures[:5]:  # Log up to 5 textures
            logging.info(f"  - {os.path.basename(texture)}")
        if len(found_textures) > 5:
            logging.info(f"  ... and {len(found_textures) - 5} more")

    # Map texture types, case-insensitive
    for texture_path in found_textures:
        texture_name = os.path.basename(texture_path)
        lower_texture_name = texture_name.lower()

        # Handle BaseColor/Color/Albedo textures
        if any(
            term in lower_texture_name for term in ["_basecolor", "_color", "_albedo"]
        ) or lower_texture_name.endswith("_a.png"):
            textures["albedo"] = texture_path

        # Handle Normal maps
        elif any(
            term in lower_texture_name for term in ["_normal", "_nrm"]
        ) or lower_texture_name.endswith("_n.png"):
            textures["normal"] = texture_path

        # Handle Roughness maps
        elif any(
            term in lower_texture_name for term in ["_roughness", "_rgh"]
        ) or lower_texture_name.endswith("_r.png"):
            textures["roughness"] = texture_path

        # Handle Metallic maps
        elif any(
            term in lower_texture_name for term in ["_metallic", "_mtl"]
        ) or lower_texture_name.endswith("_m.png"):
            textures["metallic"] = texture_path

        # Handle ORM (combined Occlusion/Roughness/Metallic) maps
        elif any(term in lower_texture_name for term in ["_orm", "_arm"]):
            textures["orm"] = texture_path

        # Handle Emissive maps
        elif any(
            term in lower_texture_name for term in ["_emissive", "_emission"]
        ) or lower_texture_name.endswith("_e.png"):
            textures["emissive"] = texture_path

    # Log what we found
    if textures:
        logging.info(
            f"Found textures for {prop_name}/{mesh_name if mesh_name else 'general'}: {', '.join(textures.keys())}"
        )

    return textures


def extract_materials_from_usd(usd_file_path, prop_name=None, prop_dir=None):
    """
    Extract material information from a USD file with improved texture handling.
    First tries USD-based texture extraction, then falls back to directory searching.

    Args:
        usd_file_path (str): Path to the USD file
        prop_name (str, optional): Name of the prop (directory name)
        prop_dir (str, optional): Path to the prop directory

    Returns:
        dict: Dictionary mapping segment names to material info
    """
    if not os.path.exists(usd_file_path):
        logging.error(f"USD file not found at {usd_file_path}")
        return {"object_name": prop_name, "segments": {}}

    try:
        # Load physics materials
        physics_materials = load_physics_materials()

        # Open the USD stage
        stage = Usd.Stage.Open(usd_file_path)
        if not stage:
            logging.error(f"Failed to open USD stage from {usd_file_path}")
            return {"object_name": prop_name, "segments": {}}

        # Get the default prim
        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            default_prim = stage.GetPrimAtPath("/")

        # Create the result dictionary with prop name, dataset type, and category
        category = ASSET_CLASS_MAPPING.get(prop_name, "unknown")
        result = {
            "object_name": prop_name,
            "category": category,
            "dataset_type": "simready",
            "file_path": usd_file_path,
            "segments": {},
        }

        # Flag to track if we found any materials
        found_materials = False

        # Find all mesh prims
        mesh_prims = []
        for prim in Usd.PrimRange(default_prim):
            if prim.GetTypeName() == "Mesh":
                mesh_prims.append(prim)

        if not mesh_prims:
            logging.warning(f"No mesh prims found in {usd_file_path}")

        # Process each mesh prim
        for prim in mesh_prims:
            mesh_name = prim.GetName()

            # Get the full path for consistent segment key construction
            full_path = str(prim.GetPath())
            path_parts = full_path.split("/")
            # Skip the root and default prim if needed
            parent_parts = [p for p in path_parts[2:-1] if p]

            # Construct segment key that matches the common.py convention
            segment_key = mesh_name
            if parent_parts:
                segment_key = "_".join(parent_parts + [mesh_name])

            # Get material binding
            material_binding = UsdShade.MaterialBindingAPI(prim)
            bound_material = None

            if material_binding:
                bound_material = material_binding.GetDirectBinding().GetMaterial()
                if not bound_material:
                    # Try to get bound material through collections
                    for binding in material_binding.GetCollectionBindings():
                        bound_material = binding.GetMaterial()
                        if bound_material:
                            break

            # If material is found, extract its properties
            if bound_material:
                found_materials = True

                # Get material name
                material_name = bound_material.GetPath().name

                # Parse material name
                opacity, material_type, semantic_usage = parse_material_name(
                    material_name
                )

                # Get physics properties for this material type if available
                physics_props = physics_materials.get(material_type, {})

                # First try USD-based texture extraction
                textures = {}
                if prop_dir:
                    # Find the surface shader for this material
                    surface_output = bound_material.GetSurfaceOutput()
                    if surface_output.HasConnectedSource():
                        connected_source = surface_output.GetConnectedSource()
                        if connected_source[0]:
                            shader_prim = connected_source[0].GetPrim()
                            textures = extract_textures_from_shader(
                                shader_prim, prop_dir
                            )

                    # If no textures found via USD, fall back to directory searching
                    if not textures:
                        logging.info(
                            f"No USD textures found for {segment_key}, falling back to directory search"
                        )
                        textures = find_textures_for_material(
                            prop_dir, mesh_name, material_type
                        )

                # Create material info dictionary with rounded values
                result["segments"][segment_key] = {
                    "name": material_name,
                    "opacity": opacity,
                    "material_type": material_type,
                    "semantic_usage": semantic_usage,
                    "density": round_float_to_2dp(physics_props.get("density")),
                    "dynamic_friction": round_float_to_2dp(
                        physics_props.get("dynamic_friction")
                    ),
                    "static_friction": round_float_to_2dp(
                        physics_props.get("static_friction")
                    ),
                    "restitution": round_float_to_2dp(physics_props.get("restitution")),
                    "textures": textures,
                }

        # If no materials were found but we have meshes, create default entries with textures
        if not found_materials and mesh_prims and prop_dir:
            logging.info(
                f"No material bindings found for {prop_name}, creating default materials"
            )

            # Try to infer material type from prop name
            material_type = "metal"  # Default
            if "wood" in prop_name.lower():
                material_type = "wood"
            elif "metal" in prop_name.lower() or "aluminum" in prop_name.lower():
                material_type = "metal"
            elif "plastic" in prop_name.lower():
                material_type = "plastic"
            elif "fabric" in prop_name.lower() or "cloth" in prop_name.lower():
                material_type = "fabric"

            # Get physics properties for this material type
            physics_props = physics_materials.get(material_type, {})

            # Process each mesh
            for prim in mesh_prims:
                mesh_name = prim.GetName()

                # Get the full path for consistent segment key construction
                full_path = str(prim.GetPath())
                path_parts = full_path.split("/")
                # Skip the root and default prim if needed
                parent_parts = [p for p in path_parts[2:-1] if p]

                # Construct segment key that matches the common.py convention
                segment_key = mesh_name
                if parent_parts:
                    segment_key = "_".join(parent_parts + [mesh_name])

                # Find textures for this mesh using directory search (since no USD materials)
                textures = find_textures_for_material(
                    prop_dir, mesh_name, material_type
                )

                # If no textures found for this specific mesh, try more general search
                if not textures:
                    textures = find_textures_for_material(prop_dir, "", material_type)

                # Create a default material info entry
                if textures:
                    default_material_name = f"default__{material_type}__{prop_name}"

                    result["segments"][segment_key] = {
                        "name": default_material_name,
                        "opacity": "opaque",
                        "material_type": material_type,
                        "semantic_usage": prop_name,
                        "density": round_float_to_2dp(physics_props.get("density")),
                        "dynamic_friction": round_float_to_2dp(
                            physics_props.get("dynamic_friction")
                        ),
                        "static_friction": round_float_to_2dp(
                            physics_props.get("static_friction")
                        ),
                        "restitution": round_float_to_2dp(
                            physics_props.get("restitution")
                        ),
                        "textures": textures,
                    }

                    found_materials = True

        # Handle case where we couldn't find any materials or textures
        if not found_materials:
            logging.warning(f"No materials or textures found for {prop_name}")

            # Create a minimal stub entry
            if mesh_prims:
                # Just use the first mesh
                prim = mesh_prims[0]
                mesh_name = prim.GetName()

                # Get the full path for consistent segment key construction
                full_path = str(prim.GetPath())
                path_parts = full_path.split("/")
                # Skip the root and default prim if needed
                parent_parts = [p for p in path_parts[2:-1] if p]

                # Construct segment key that matches the common.py convention
                segment_key = mesh_name
                if parent_parts:
                    segment_key = "_".join(parent_parts + [mesh_name])

                result["segments"][segment_key] = {
                    "name": f"unknown_material",
                    "opacity": "opaque",
                    "material_type": "unknown",
                    "semantic_usage": "",
                    "density": None,
                    "dynamic_friction": None,
                    "static_friction": None,
                    "restitution": None,
                    "textures": {},
                }

        # Ensure category and dataset_type present (in case older function call didn't inject)
        if "category" not in result:
            result["category"] = ASSET_CLASS_MAPPING.get(prop_name, "unknown")
        if "dataset_type" not in result:
            result["dataset_type"] = "simready"

        return result

    except Exception as e:
        logging.error(f"Error extracting materials for {prop_name}: {str(e)}")
        import traceback

        logging.error(traceback.format_exc())
        return {"object_name": prop_name, "segments": {}}


def extract_textures_from_shader(shader_prim, prop_dir):
    """
    Extract texture information directly from a shader prim.

    Args:
        shader_prim: USD shader prim
        prop_dir: Path to prop directory for resolving relative paths

    Returns:
        dict: Dictionary of texture paths by type
    """
    textures = {}

    if not shader_prim.IsA(UsdShade.Shader):
        return textures

    shader = UsdShade.Shader(shader_prim)

    # Map of USD shader input names to our texture type names
    texture_input_mapping = {
        "diffuse_texture": "albedo",
        "albedo_texture": "albedo",
        "basecolor_texture": "albedo",
        "normalmap_texture": "normal",
        "normal_texture": "normal",
        "reflectionroughness_texture": "roughness",
        "roughness_texture": "roughness",
        "metallic_texture": "metallic",
        "orm_texture": "orm",
        "emissive_texture": "emissive",
        "opacity_texture": "opacity",
    }

    # Get all shader inputs
    shader_inputs = shader.GetInputs()

    for shader_input in shader_inputs:
        input_name = shader_input.GetBaseName()

        # Check if this input corresponds to a texture
        texture_type = texture_input_mapping.get(input_name)
        if texture_type:
            # Get the connected source
            source_info = shader_input.GetConnectedSource()
            if source_info[0]:  # If connected
                source_prim = source_info[0].GetPrim()
                if source_prim.IsA(UsdShade.Shader):
                    # This is a texture node - get the file path
                    texture_shader = UsdShade.Shader(source_prim)
                    file_input = texture_shader.GetInput("file")
                    if file_input:
                        texture_path = file_input.Get()
                        if texture_path:
                            # Convert to absolute path if relative
                            if not os.path.isabs(texture_path):
                                texture_path = os.path.join(prop_dir, texture_path)
                            textures[texture_type] = texture_path

        # Also check direct file inputs (some shaders may have direct file paths)
        elif input_name == "file" or input_name.endswith("_file"):
            texture_path = shader_input.Get()
            if texture_path:
                # Try to infer texture type from filename
                lower_name = os.path.basename(texture_path).lower()
                if (
                    "albedo" in lower_name
                    or "basecolor" in lower_name
                    or "diffuse" in lower_name
                ):
                    texture_type = "albedo"
                elif "normal" in lower_name:
                    texture_type = "normal"
                elif "roughness" in lower_name:
                    texture_type = "roughness"
                elif "metallic" in lower_name:
                    texture_type = "metallic"
                elif "orm" in lower_name:
                    texture_type = "orm"
                elif "emissive" in lower_name:
                    texture_type = "emissive"
                else:
                    texture_type = "albedo"  # Default to albedo

                # Convert to absolute path if relative
                if not os.path.isabs(texture_path):
                    texture_path = os.path.join(prop_dir, texture_path)
                textures[texture_type] = texture_path

    return textures


def extract_materials_from_usd_improved(usd_file_path, prop_name=None, prop_dir=None):
    """
    Extract material information from a USD file with improved texture handling.
    Uses direct USD shader attribute extraction instead of directory searching.

    Args:
        usd_file_path (str): Path to the USD file
        prop_name (str, optional): Name of the prop (directory name)
        prop_dir (str, optional): Path to the prop directory

    Returns:
        dict: Dictionary mapping segment names to material info
    """
    if not os.path.exists(usd_file_path):
        logging.error(f"USD file not found at {usd_file_path}")
        return {"object_name": prop_name, "segments": {}}

    try:
        # Load physics materials
        physics_materials = load_physics_materials()

        # Open the USD stage
        stage = Usd.Stage.Open(usd_file_path)
        if not stage:
            logging.error(f"Failed to open USD stage from {usd_file_path}")
            return {"object_name": prop_name, "segments": {}}

        # Get the default prim
        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            default_prim = stage.GetPrimAtPath("/")

        # Create the result dictionary with prop name, dataset type, and category
        category = ASSET_CLASS_MAPPING.get(prop_name, "unknown")
        result = {
            "object_name": prop_name,
            "category": category,
            "dataset_type": "simready",
            "segments": {},
        }

        # Flag to track if we found any materials
        found_materials = False

        # Find all mesh prims
        mesh_prims = []
        for prim in Usd.PrimRange(default_prim):
            if prim.GetTypeName() == "Mesh":
                mesh_prims.append(prim)

        if not mesh_prims:
            logging.warning(f"No mesh prims found in {usd_file_path}")

        # Process each mesh prim
        for prim in mesh_prims:
            mesh_name = prim.GetName()

            # Get the full path for consistent segment key construction
            full_path = str(prim.GetPath())
            path_parts = full_path.split("/")
            # Skip the root and default prim if needed
            parent_parts = [p for p in path_parts[2:-1] if p]

            # Construct segment key that matches the common.py convention
            segment_key = mesh_name
            if parent_parts:
                segment_key = "_".join(parent_parts + [mesh_name])

            # Get material binding
            material_binding = UsdShade.MaterialBindingAPI(prim)
            bound_material = None

            if material_binding:
                bound_material = material_binding.GetDirectBinding().GetMaterial()
                if not bound_material:
                    # Try to get bound material through collections
                    material_found = False
                    for binding in material_binding.GetCollectionBindings():
                        bound_material = binding.GetMaterial()
                        if bound_material:
                            material_found = True
                            break

            # If material is found, extract its properties
            if bound_material:
                found_materials = True

                # Get material name
                material_name = bound_material.GetPath().name

                # Parse material name
                opacity, material_type, semantic_usage = parse_material_name(
                    material_name
                )

                # Get physics properties for this material type if available
                physics_props = physics_materials.get(material_type, {})

                # Find textures for this material
                textures = {}
                if prop_dir:
                    textures = find_textures_for_material(
                        prop_dir, mesh_name, material_type
                    )

                # Create material info dictionary with rounded values
                result["segments"][segment_key] = {
                    "name": material_name,
                    "opacity": opacity,
                    "material_type": material_type,
                    "semantic_usage": semantic_usage,
                    "density": round_float_to_2dp(physics_props.get("density")),
                    "dynamic_friction": round_float_to_2dp(
                        physics_props.get("dynamic_friction")
                    ),
                    "static_friction": round_float_to_2dp(
                        physics_props.get("static_friction")
                    ),
                    "restitution": round_float_to_2dp(physics_props.get("restitution")),
                    "textures": textures,
                }

        # If no materials were found but we have meshes, create default entries with textures
        if not found_materials and mesh_prims and prop_dir:
            logging.info(
                f"No material bindings found for {prop_name}, creating default materials"
            )

            # Try to infer material type from prop name
            material_type = "metal"  # Default
            if "wood" in prop_name.lower():
                material_type = "wood"
            elif "metal" in prop_name.lower() or "aluminum" in prop_name.lower():
                material_type = "metal"
            elif "plastic" in prop_name.lower():
                material_type = "plastic"
            elif "fabric" in prop_name.lower() or "cloth" in prop_name.lower():
                material_type = "fabric"

            # Get physics properties for this material type
            physics_props = physics_materials.get(material_type, {})

            # Process each mesh
            for prim in mesh_prims:
                mesh_name = prim.GetName()

                # Get the full path for consistent segment key construction
                full_path = str(prim.GetPath())
                path_parts = full_path.split("/")
                # Skip the root and default prim if needed
                parent_parts = [p for p in path_parts[2:-1] if p]

                # Construct segment key that matches the common.py convention
                segment_key = mesh_name
                if parent_parts:
                    segment_key = "_".join(parent_parts + [mesh_name])

                # Find textures for this mesh
                textures = find_textures_for_material(
                    prop_dir, mesh_name, material_type
                )

                # If no textures found for this specific mesh, try more general search
                if not textures:
                    textures = find_textures_for_material(prop_dir, "", material_type)

                # Create a default material info entry
                if textures:
                    default_material_name = f"default__{material_type}__{prop_name}"

                    result["segments"][segment_key] = {
                        "name": default_material_name,
                        "opacity": "opaque",
                        "material_type": material_type,
                        "semantic_usage": prop_name,
                        "density": round_float_to_2dp(physics_props.get("density")),
                        "dynamic_friction": round_float_to_2dp(
                            physics_props.get("dynamic_friction")
                        ),
                        "static_friction": round_float_to_2dp(
                            physics_props.get("static_friction")
                        ),
                        "restitution": round_float_to_2dp(
                            physics_props.get("restitution")
                        ),
                        "textures": textures,
                    }

                    found_materials = True

        # Handle case where we couldn't find any materials or textures
        if not found_materials:
            logging.warning(f"No materials or textures found for {prop_name}")

            # Create a minimal stub entry
            if mesh_prims:
                # Just use the first mesh
                prim = mesh_prims[0]
                mesh_name = prim.GetName()

                # Get the full path for consistent segment key construction
                full_path = str(prim.GetPath())
                path_parts = full_path.split("/")
                # Skip the root and default prim if needed
                parent_parts = [p for p in path_parts[2:-1] if p]

                # Construct segment key that matches the common.py convention
                segment_key = mesh_name
                if parent_parts:
                    segment_key = "_".join(parent_parts + [mesh_name])

                result["segments"][segment_key] = {
                    "name": f"unknown_material",
                    "opacity": "opaque",
                    "material_type": "unknown",
                    "semantic_usage": "",
                    "density": None,
                    "dynamic_friction": None,
                    "static_friction": None,
                    "restitution": None,
                    "textures": {},
                }

        # Ensure category and dataset_type present (in case older function call didn't inject)
        if "category" not in result:
            result["category"] = ASSET_CLASS_MAPPING.get(prop_name, "unknown")
        if "dataset_type" not in result:
            result["dataset_type"] = "simready"

        return result

    except Exception as e:
        logging.error(f"Error extracting materials for {prop_name}: {str(e)}")
        import traceback

        logging.error(traceback.format_exc())
        return {"object_name": prop_name, "segments": {}}


def process_simready_objects(
    vlm_model,
    vlm_processor,
    limit=None,
    processed_objects=None,
    output_file=None,
    existing_results=None,
):
    OBJECTS_TO_PROCESS = [
        d for d in os.listdir(PROPS_DIR) if os.path.isdir(os.path.join(PROPS_DIR, d))
    ]
    OBJECTS_TO_PROCESS.sort()

    processed_objects = processed_objects or set()
    existing_results = existing_results or []

    # Initialize statistics
    all_results = []
    success_count = 0
    failed_props = []
    total_segments = 0
    unique_materials = set()
    materials_per_prop = {}
    total_rendered_segments = 0
    total_vlm_segments = 0

    # Statistics for texture availability
    segments_with_albedo = 0
    segments_without_albedo = 0
    segments_with_thumbnail_only = 0

    if limit:
        OBJECTS_TO_PROCESS = OBJECTS_TO_PROCESS[:limit]

    for prop_idx, prop_name in enumerate(
        tqdm(OBJECTS_TO_PROCESS, desc="Processing SimReady objects")
    ):
        # Skip objects that have already been processed
        if prop_name in processed_objects:
            logging.info(f"Skipping already processed object: {prop_name}")

            # Find its results in existing_results to count in statistics
            for result in existing_results:
                if result.get("object_name") == prop_name:
                    all_results.append(result)
                    success_count += 1
                    total_segments += len(result.get("segments", {}))
                    break

            continue

        try:
            full_prop_dir = os.path.join(PROPS_DIR, prop_name)

            if not os.path.isdir(full_prop_dir):
                logging.error(f"Prop directory not found at {full_prop_dir}")
                failed_props.append(prop_name)
                continue

            # Find a USD file in the prop directory
            try:
                usd_file = get_usd_file_from_prop_dir(full_prop_dir)
                logging.info(
                    f"Found USD file for {prop_name}: {os.path.basename(usd_file)}"
                )
            except:
                logging.error(f"Could not find USD file for {prop_name}")
                failed_props.append(prop_name)
                continue

            # Extract material information
            materials_dict = extract_materials_from_usd(
                usd_file, prop_name, full_prop_dir
            )

            # Track statistics
            segments = materials_dict.get("segments", {})
            total_segments += len(segments)

            # Count unique materials for this prop
            prop_materials = set()
            for segment_key, segment_info in segments.items():
                unique_materials.add(segment_info["name"])
                prop_materials.add(segment_info["name"])

            # Record materials per prop
            if len(segments) > 0:
                materials_per_prop[prop_name] = len(prop_materials)

            # Determine thumbnail path from SimReady structure
            thumb_path = os.path.join(
                full_prop_dir,
                ".thumbs",
                "256x256",
                f"{prop_name}.usd.png",
            )
            has_thumbnail = os.path.exists(thumb_path)

            if not has_thumbnail:
                logging.warning(f"No thumbnail found for {prop_name} at {thumb_path}")
                # Try to find any thumbnail in the .thumbs directory
                thumb_dir = os.path.join(full_prop_dir, ".thumbs", "256x256")
                if os.path.exists(thumb_dir):
                    thumb_files = [
                        f for f in os.listdir(thumb_dir) if f.endswith(".png")
                    ]
                    if thumb_files:
                        thumb_path = os.path.join(thumb_dir, thumb_files[0])
                        has_thumbnail = True
                        logging.info(f"Found alternative thumbnail: {thumb_path}")

            # Add to combined results if we have segments
            os.makedirs("/tmp/vlm", exist_ok=True)
            if len(segments) > 0:
                # Process every segment with VLM
                if vlm_model:
                    for segment_key, segment_info in segments.items():
                        textures = segment_info.get("textures", {})

                        # Log texture information for diagnostics
                        logging.info(
                            f"Segment {segment_key} has textures: {list(textures.keys())}"
                        )

                        has_albedo = "albedo" in textures
                        if has_albedo:
                            # Has albedo texture - render sphere and use with thumbnail
                            segments_with_albedo += 1
                            logging.info(
                                f"Rendering texture sphere for {prop_name}, segment {segment_key}"
                            )

                            # Set up file path for this segment's rendered sphere
                            segment_render_path = (
                                f"/tmp/vlm/texture_sphere_{prop_name}_{segment_key}.png"
                            )

                            try:
                                rgb_buffer = render_sphere_with_texture(
                                    textures, segment_render_path
                                )
                                logging.info(f"RGB buffer shape: {rgb_buffer.shape}")
                            except Exception as e:
                                logging.error(
                                    f"Error rendering texture for {segment_key}: {str(e)}"
                                )
                                segment_render_path = None
                        else:
                            # No albedo texture - just use thumbnail
                            segments_without_albedo += 1
                            segment_render_path = None
                            logging.info(
                                f"No albedo texture for {prop_name}, segment {segment_key}. Using thumbnail only."
                            )

                        # Always try to process with VLM, even if no texture
                        try:
                            # If we have a thumbnail but no texture, still run VLM with just the thumbnail
                            if not has_albedo and has_thumbnail:
                                segments_with_thumbnail_only += 1
                                logging.info(
                                    f"Using thumbnail only for {prop_name}, segment {segment_key}"
                                )

                            # Don't run VLM if we have neither texture nor thumbnail
                            if not segment_render_path and not has_thumbnail:
                                logging.warning(
                                    f"Skipping VLM for {segment_key} - no texture or thumbnail available"
                                )
                                continue

                            part1 = make_user_prompt(
                                segment_info["material_type"],
                                segment_info["opacity"],
                                segment_info["density"],
                                segment_info["dynamic_friction"],
                                segment_info["static_friction"],
                                segment_info["restitution"],
                                segment_info["semantic_usage"],
                                has_texture_sphere=segment_render_path is not None,
                            )

                            # Store the custom prompt in material_info
                            segment_info["user_prompt"] = part1

                            # Debug: Log the prompt type based on texture availability
                            if segment_render_path is not None:
                                logging.info(
                                    f"Using prompt WITH texture sphere for {prop_name}, segment {segment_key}"
                                )
                            else:
                                logging.info(
                                    f"Using prompt WITHOUT texture sphere for {prop_name}, segment {segment_key}"
                                )
                                logging.info(
                                    f"PROMPT: {part1[:100]}..."
                                )  # Print just the beginning of the prompt

                            vlm_analysis = analyze_material_with_vlm(
                                segment_render_path,  # This can be None, in which case only thumbnail is used
                                segment_info,
                                vlm_model,
                                vlm_processor,
                                thumbnail_path=thumb_path,
                                dataset_name="simready",
                                PROMPTS=PROMPTS,
                                make_user_prompt=make_user_prompt,
                                parse_vlm_output=parse_vlm_output,
                            )

                            # Add VLM analysis to segment info directly
                            if vlm_analysis and "error" not in vlm_analysis:
                                # Update the segment info with VLM-derived properties
                                segment_info["vlm_analysis"] = vlm_analysis.get(
                                    "vlm_analysis"
                                )

                                if vlm_analysis.get("youngs_modulus") is not None:
                                    segment_info["youngs_modulus"] = vlm_analysis.get(
                                        "youngs_modulus"
                                    )

                                if vlm_analysis.get("poissons_ratio") is not None:
                                    segment_info["poissons_ratio"] = vlm_analysis.get(
                                        "poissons_ratio"
                                    )

                                if vlm_analysis.get("density") is not None:
                                    segment_info["density"] = vlm_analysis.get(
                                        "density"
                                    )

                                total_vlm_segments += 1
                                logging.info(
                                    f"VLM analysis successful for {segment_key}:"
                                )
                                logging.info(
                                    f"  Young's modulus: {vlm_analysis.get('youngs_modulus')}"
                                )
                                logging.info(
                                    f"  Poisson's ratio: {vlm_analysis.get('poissons_ratio')}"
                                )
                                if vlm_analysis.get("density") is not None:
                                    logging.info(
                                        f"  Density: {vlm_analysis.get('density')}"
                                    )
                            else:
                                logging.error(
                                    f"VLM analysis failed for {segment_key}: {vlm_analysis.get('error', 'Unknown error')}"
                                )
                        except Exception as e:
                            import traceback

                            logging.error(
                                f"Error during VLM analysis for {segment_key}: {str(e)}"
                            )
                            logging.error(traceback.format_exc())

                        total_rendered_segments += 1

                all_results.append(materials_dict)
                success_count += 1

                # Incremental save after each object if output file is provided
                if output_file:
                    try:
                        with open(output_file, "w") as f:
                            import json
                            from dataset_toolkits.material_objects.vlm_annotations.data_subsets.common import (
                                UsdJsonEncoder,
                            )

                            # Debug save contents
                            logging.info(
                                f"Saving checkpoint with {len(all_results)} objects"
                            )

                            # Ensure result types are JSON serializable
                            for idx, item in enumerate(all_results):
                                if "segments" in item:
                                    for seg_key, seg_info in item["segments"].items():
                                        if "textures" in seg_info and isinstance(
                                            seg_info["textures"], dict
                                        ):
                                            # Convert any non-serializable texture paths to strings
                                            serializable_textures = {}
                                            for tex_type, tex_path in seg_info[
                                                "textures"
                                            ].items():
                                                serializable_textures[tex_type] = str(
                                                    tex_path
                                                )
                                            seg_info["textures"] = serializable_textures

                            json.dump(all_results, f, indent=4, cls=UsdJsonEncoder)

                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {str(e)}")
                        import traceback

                        logging.error(traceback.format_exc())
            else:
                logging.warning(f"No segments extracted for {prop_name}")
                failed_props.append(prop_name)

        except Exception as e:
            import traceback

            logging.error(f"Error processing {prop_name}: {str(e)}")
            logging.error(traceback.format_exc())
            failed_props.append(prop_name)

    # Log texture statistics
    logging.info("Texture Statistics:")
    logging.info(f"  Total segments processed: {total_segments}")
    logging.info(f"  Segments with albedo textures: {segments_with_albedo}")
    logging.info(f"  Segments without albedo textures: {segments_without_albedo}")
    logging.info(f"  Segments with thumbnail only: {segments_with_thumbnail_only}")
    logging.info(f"  Total VLM analyses completed: {total_vlm_segments}")

    return (
        all_results,
        len(OBJECTS_TO_PROCESS),
        success_count,
        failed_props,
        total_segments,
        total_rendered_segments,
        total_vlm_segments,
        list(unique_materials),
        materials_per_prop,
    )
