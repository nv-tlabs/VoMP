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
import json
import glob
from functools import lru_cache
import traceback
import argparse
import numpy as np
import logging
import copy
from PIL import Image
from tqdm import tqdm
from pxr import Usd, UsdShade, Sdf, Ar
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    COMMERCIAL_BASE_DIR,
    RESIDENTIAL_BASE_DIR,
    VEGETATION_BASE_DIR,
    SIMREADY_ASSET_CLASS_MAPPING as ASSET_CLASS_MAPPING,
    SIMREADY_MATERIALS_DIR as PHYSICS_MATERIALS_DIR,
    round_float_to_2dp,
)
from dataset_toolkits.material_objects.vlm_annotations.usd_segment_reader import (
    authored_perspective_camera_hint,
    build_material_record,
    collect_material_texture_evidence,
    enumerate_material_regions,
    material_record_extraction_errors,
    material_record_id,
    region_metadata,
    select_resolved_textures,
    stage_color_management_metadata,
    stage_composition_errors,
)
import datetime
import uuid
import shutil


class UsdJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)


def find_textures_for_material(object_dir, texture_path):
    """
    Find textures referenced by a material in a USD file.

    Args:
        object_dir (str): Directory containing the USD file
        texture_path (str): Texture path from the USD file

    Returns:
        dict: Dictionary mapping texture types to full paths
    """
    if not texture_path:
        return {}

    # Convert Sdf.AssetPath to string if needed
    if (
        hasattr(texture_path, "__class__")
        and texture_path.__class__.__name__ == "AssetPath"
    ):
        texture_path = str(texture_path)

    # Handle absolute paths
    if os.path.isabs(texture_path):
        if os.path.exists(texture_path):
            return {determine_texture_type(texture_path): texture_path}
        return {}

    # Handle relative paths with various prefixes
    clean_path = texture_path.replace("@", "").replace("./", "")

    # Try direct path
    direct_path = os.path.join(object_dir, clean_path)
    if os.path.exists(direct_path):
        return {determine_texture_type(direct_path): direct_path}

    # Try common texture directories
    texture_dirs = []
    for texture_dir_name in [
        "textures",
        "Textures",
        "materials/textures",
        "Materials/Textures",
    ]:
        texture_dir = os.path.join(object_dir, texture_dir_name)
        if os.path.isdir(texture_dir):
            texture_dirs.append(texture_dir)

    # Look in parent directory if object_dir doesn't have textures
    if not texture_dirs:
        parent_dir = os.path.dirname(object_dir)
        for texture_dir_name in [
            "textures",
            "Textures",
            "materials/textures",
            "Materials/Textures",
        ]:
            texture_dir = os.path.join(parent_dir, texture_dir_name)
            if os.path.isdir(texture_dir):
                texture_dirs.append(texture_dir)

    # Check for texture in each texture directory
    for texture_dir in texture_dirs:
        texture_file = os.path.join(texture_dir, os.path.basename(clean_path))
        if os.path.exists(texture_file):
            return {determine_texture_type(texture_file): texture_file}

    return {}


def determine_texture_type(texture_path):
    """
    Determine the type of texture based on its filename.

    Args:
        texture_path (str): Path to texture file

    Returns:
        str: Texture type (albedo, normal, roughness, metallic, orm)
    """
    filename = os.path.basename(texture_path).lower()

    # Check for common texture type indicators in filename
    if any(
        term in filename
        for term in ["albedo", "basecolor", "color", "_a.", "_a_", "_diffuse", "_diff"]
    ):
        return "albedo"
    elif any(term in filename for term in ["normal", "nrm", "_n.", "_n_"]):
        return "normal"
    elif any(term in filename for term in ["roughness", "rough", "_r.", "_r_"]):
        return "roughness"
    elif any(term in filename for term in ["metallic", "metal", "_m.", "_m_"]):
        return "metallic"
    elif any(term in filename for term in ["orm", "arm", "occlusion"]):
        return "orm"
    elif any(term in filename for term in ["emissive", "emission", "_e."]):
        return "emissive"
    elif any(term in filename for term in ["opacity", "transparent", "alpha"]):
        return "opacity"
    elif any(term in filename for term in ["specular", "spec", "_s."]):
        return "specular"
    elif any(term in filename for term in ["displacement", "height", "bump"]):
        return "displacement"

    # If no specific type is identified, make an educated guess based on file extension
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".tga", ".tif", ".tiff"]:
        return "albedo"  # Default to albedo for unrecognized image files

    return "unknown"


def copy_texture_to_output(
    texture_path, output_dir, object_name, material_name, texture_type
):
    """
    Copy a texture file to the output directory with a standardized name.

    Args:
        texture_path (str): Source texture path
        output_dir (str): Output directory
        object_name (str): Name of the object
        material_name (str): Name of the material
        texture_type (str): Type of texture

    Returns:
        str: Path to the copied texture file
    """
    if not os.path.exists(texture_path):
        return None

    # Create output subdirectory for this object if it doesn't exist
    object_output_dir = os.path.join(output_dir, object_name)
    os.makedirs(object_output_dir, exist_ok=True)

    # Create standardized output filename
    texture_ext = os.path.splitext(texture_path)[1]
    output_filename = f"{material_name}_{texture_type}{texture_ext}"
    output_path = os.path.join(object_output_dir, output_filename)

    try:
        # Copy the texture file

        shutil.copy2(texture_path, output_path)
        return output_path
    except Exception as e:
        logging.error(f"Error copying texture {texture_path}: {str(e)}")
        return None


def extract_material_from_shader(shader_prim, object_dir, dataset_type=None):
    """
    Extract material properties and textures from a shader prim.

    Args:
        shader_prim (UsdShade.Shader): Shader prim
        object_dir (str): Directory containing the USD file
        dataset_type (str, optional): Type of dataset (commercial, residential, vegetation)

    Returns:
        dict: Dictionary with material properties and textures
    """
    material_info = {"textures": {}}

    # Create a shader object from the prim
    shader = UsdShade.Shader(shader_prim)
    if not shader:
        logging.warning(f"Failed to create shader from {shader_prim.GetPath()}")
        return material_info

    # Get material name from shader path
    shader_path = str(shader_prim.GetPath())
    material_name = None
    if "/Looks/" in shader_path:
        material_name = shader_path.split("/Looks/")[1].split("/")[0]

    logging.info(f"Processing shader for material: {material_name}")

    # For vegetation materials, try to find matching textures by material name
    if dataset_type == "vegetation" and material_name:
        # Find the materials/textures directory
        object_dir_parts = object_dir.split(os.sep)
        trees_dir = None
        for i in range(len(object_dir_parts)):
            if object_dir_parts[i] == "Trees":
                trees_dir = os.sep.join(object_dir_parts[: i + 1])
                break

        if trees_dir:
            textures_dir = os.path.join(trees_dir, "materials", "textures")
            if os.path.exists(textures_dir):
                material_name_lower = material_name.lower()
                material_parts = material_name_lower.replace("_", " ").split()

                # Get all texture files in the directory
                texture_files = [
                    f
                    for f in os.listdir(textures_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
                ]

                # Track potential matches for different texture types
                texture_matches = {
                    "diffuse": [],
                    "normal": [],
                    "roughness": [],
                    "metallic": [],
                    "orm": [],
                }

                # Categorize material into types
                material_categories = {
                    "bark": [
                        "bark",
                        "trunk",
                        "wood",
                        "tree",
                        "log",
                        "stump",
                        "stem",
                        "branch",
                        "twig",
                    ],
                    "leaf": ["leaf", "leaves", "foliage", "needle", "needles", "frond"],
                    "flower": [
                        "flower",
                        "flowers",
                        "petal",
                        "petals",
                        "bloom",
                        "blossom",
                    ],
                    "fruit": [
                        "fruit",
                        "fruits",
                        "berry",
                        "berries",
                        "seed",
                        "seeds",
                        "cone",
                        "cones",
                    ],
                    "grass": [
                        "grass",
                        "grasses",
                        "reed",
                        "reeds",
                        "sedge",
                        "rush",
                        "blade",
                    ],
                }

                # Find all applicable categories
                material_types = []
                for category, keywords in material_categories.items():
                    if any(keyword in material_name_lower for keyword in keywords):
                        material_types.append(category)

                # If we couldn't determine a category from material name, try from object name
                if not material_types:
                    object_name = os.path.splitext(os.path.basename(object_dir))[
                        0
                    ].lower()
                    for category, keywords in material_categories.items():
                        if any(keyword in object_name for keyword in keywords):
                            material_types.append(category)

                # Still no category? Add generic fallbacks
                if not material_types:
                    # Default to bark for most vegetation models
                    material_types = ["bark"]

                logging.info(
                    f"Material categories for {material_name}: {material_types}"
                )

                # Scoring function for texture relevance to material name
                def score_texture_for_material(texture_name, texture_type):
                    score = 0
                    texture_name_lower = texture_name.lower()

                    # Direct material name match (highest priority)
                    if material_name_lower in texture_name_lower:
                        score += 200

                    # Match individual parts of material name
                    for part in material_parts:
                        if len(part) > 2 and part in texture_name_lower:
                            score += 50

                    # Match material categories
                    for material_type in material_types:
                        # Match exact category name
                        if material_type in texture_name_lower:
                            score += 100

                        # Match keywords for this category
                        for keyword in material_categories.get(material_type, []):
                            if keyword in texture_name_lower:
                                score += 40

                    # Correct type suffix
                    type_suffixes = {
                        "diffuse": [
                            "basecolor",
                            "albedo",
                            "color",
                            "diffuse",
                            "_bc",
                            "_a",
                            "_d",
                        ],
                        "normal": ["normal", "nrm", "_n", "nor"],
                        "roughness": ["roughness", "rough", "_r", "rgh"],
                        "metallic": ["metallic", "metal", "_m", "mtl"],
                        "orm": ["orm", "arm", "occlusion"],
                    }

                    for suffix in type_suffixes.get(texture_type, []):
                        if suffix in texture_name_lower:
                            score += 40

                    # Boost score for more specific matches (longer texture names probably more specific)
                    if len(texture_name_lower) > 15:
                        score += 10

                    # Exact matches for specific materials
                    if material_name_lower == "bark" and "bark" in texture_name_lower:
                        score += 50
                    elif (
                        material_name_lower == "leaves" and "leaf" in texture_name_lower
                    ):
                        score += 50
                    elif (
                        material_name_lower == "needle"
                        and "needle" in texture_name_lower
                    ):
                        score += 50
                    elif (
                        "trunk" in material_name_lower and "bark" in texture_name_lower
                    ):
                        score += 30

                    return score

                # Process each texture file
                for texture_file in texture_files:
                    # Determine texture type
                    texture_type = determine_texture_type(texture_file)

                    # Don't process "unknown" textures
                    if texture_type == "unknown":
                        continue

                    # Score this texture for this material
                    score = score_texture_for_material(texture_file, texture_type)

                    # If it's a good match (score > 0), add to potential matches
                    if score > 0:
                        # Convert diffuse type to match our expected naming
                        if texture_type in ["albedo", "basecolor", "color"]:
                            texture_type = "diffuse"

                        # Add to matches with score
                        if texture_type in texture_matches:
                            texture_matches[texture_type].append((texture_file, score))

                # Sort matches by score and select the best for each type
                for texture_type, matches in texture_matches.items():
                    if matches:
                        # Sort by score (highest first)
                        matches.sort(key=lambda x: x[1], reverse=True)
                        best_match = matches[0][0]

                        # Add to material info
                        texture_path = os.path.join(textures_dir, best_match)
                        material_info["textures"][texture_type] = texture_path
                        logging.info(
                            f"Found {texture_type} texture for {material_name}: {best_match}"
                        )

                # If we still don't have textures, use fallbacks from generic categories
                if not any(material_info["textures"].values()):
                    logging.info(
                        f"No direct texture matches found for {material_name}, trying category fallbacks"
                    )

                    # Key textures we need
                    needed_types = ["diffuse", "normal", "roughness"]

                    # Generic fallbacks by category
                    fallbacks = {
                        "bark": {
                            "diffuse": "pinebark1_basecolor.png",
                            "normal": "pinebark1_normal.png",
                            "roughness": "pinebark1_roughness.png",
                        },
                        "leaf": {
                            "diffuse": "oakleaves1_basecolor.png",
                            "normal": "oakleaves1_normal.png",
                            "roughness": "oakleaves1_roughness.png",
                        },
                        "flower": {
                            "diffuse": "goldenchain_flowers_basecolor.png",
                            "normal": "goldenchain_flowers_normal.png",
                            "roughness": "goldenchain_flowers_roughness.png",
                        },
                        "grass": {
                            "diffuse": "ashleaves1_basecolor.png",
                            "normal": "ashleaves1_normal.png",
                            "roughness": "ashleaves1_roughness.png",
                        },
                        "needle": {
                            "diffuse": "spruceneedles_basecolor.png",
                            "normal": "spruceneedles_normal.png",
                            "roughness": "spruceneedles_roughness.png",
                        },
                    }

                    # Try each category we matched
                    for material_type in material_types:
                        if material_type in fallbacks:
                            for texture_type in needed_types:
                                if texture_type not in material_info[
                                    "textures"
                                ] and fallbacks[material_type].get(texture_type):
                                    fallback_file = fallbacks[material_type][
                                        texture_type
                                    ]
                                    fallback_path = os.path.join(
                                        textures_dir, fallback_file
                                    )
                                    if os.path.exists(fallback_path):
                                        material_info["textures"][
                                            texture_type
                                        ] = fallback_path
                                        logging.info(
                                            f"Using fallback {texture_type} texture for {material_name}: {fallback_file}"
                                        )

                    # If still missing textures, use bark as an ultimate fallback (most common)
                    for texture_type in needed_types:
                        if texture_type not in material_info["textures"]:
                            fallback_file = fallbacks["bark"][texture_type]
                            fallback_path = os.path.join(textures_dir, fallback_file)
                            if os.path.exists(fallback_path):
                                material_info["textures"][texture_type] = fallback_path
                                logging.info(
                                    f"Using ultimate fallback {texture_type} texture for {material_name}: {fallback_file}"
                                )

    # Check for shader attributes
    inputs_to_check = [
        # Common texture inputs
        "diffuse_color_texture",
        "inputs:diffuse_color_texture",
        "normalmap_texture",
        "inputs:normalmap_texture",
        "reflectionroughness_texture",
        "inputs:reflectionroughness_texture",
        "diffusecolor_texture",
        "inputs:diffusecolor_texture",
        "normal_texture",
        "inputs:normal_texture",
        "roughness_texture",
        "inputs:roughness_texture",
        # Common material constants
        "diffuse_color_constant",
        "inputs:diffuse_color_constant",
        "reflection_roughness_constant",
        "inputs:reflection_roughness_constant",
        "metallic_constant",
        "inputs:metallic_constant",
        "opacity_constant",
        "inputs:opacity_constant",
        "emissive_color_constant",
        "inputs:emissive_color_constant",
    ]

    # Process each input attribute
    for input_name in inputs_to_check:
        # Remove "inputs:" prefix if present
        input_name_clean = input_name.replace("inputs:", "")

        # Try to get the input
        shader_input = shader.GetInput(input_name_clean)
        if not shader_input:
            continue

        # Get the value
        value = shader_input.Get()
        if value is None:
            continue

        # Format input name to standard form
        standard_name = input_name_clean.lower()

        # Check if this is a texture input
        if "texture" in standard_name:
            # Determine texture type
            if "normal" in standard_name:
                texture_type = "normal"
            elif "rough" in standard_name:
                texture_type = "roughness"
            elif "diffuse" in standard_name or "color" in standard_name:
                texture_type = "diffuse"
            elif "specular" in standard_name:
                texture_type = "specular"
            elif "metallic" in standard_name:
                texture_type = "metallic"
            elif "opacity" in standard_name:
                texture_type = "opacity"
            elif "emissive" in standard_name:
                texture_type = "emissive"
            else:
                texture_type = "other"

            # Handle asset path values
            if isinstance(value, Sdf.AssetPath):
                texture_path = value.resolvedPath
                if not texture_path:
                    # Try to resolve relative path
                    rel_path = value.path
                    if rel_path.startswith("./"):
                        rel_path = rel_path[2:]
                    texture_path = os.path.join(object_dir, rel_path)

                if os.path.exists(texture_path):
                    if (
                        dataset_type == "vegetation"
                        and texture_type in material_info["textures"]
                    ):
                        logging.info(
                            f"Keeping already found {texture_type} texture for {material_name}"
                        )
                    else:
                        material_info["textures"][texture_type] = texture_path

            # For vegetation, try to find exact textures by material name
            if (
                dataset_type == "vegetation"
                and not material_info["textures"].get(texture_type)
                and material_name
            ):
                logging.info(
                    f"Looking for exact vegetation texture: {texture_type} for {material_name}"
                )

                # Find the materials/textures directory
                object_dir_parts = object_dir.split(os.sep)
                trees_dir = None
                for i in range(len(object_dir_parts)):
                    if object_dir_parts[i] == "Trees":
                        trees_dir = os.sep.join(object_dir_parts[: i + 1])
                        break

                if trees_dir:
                    materials_dir = os.path.join(trees_dir, "materials")
                    textures_dir = os.path.join(materials_dir, "textures")

                    logging.info(f"Looking for textures in: {textures_dir}")

                    if os.path.exists(textures_dir):
                        # Look for textures with material name
                        material_name_lower = material_name.lower()

                        # Build specific patterns for this material name
                        specific_patterns = [
                            f"{material_name_lower}_{texture_type}.png",
                            f"{material_name_lower.replace('_', '')}_{texture_type}.png",
                        ]

                        # Try alternate texture type names for diffuse
                        if texture_type == "diffuse":
                            specific_patterns.extend(
                                [
                                    f"{material_name_lower}_basecolor.png",
                                    f"{material_name_lower.replace('_', '')}_basecolor.png",
                                    f"{material_name_lower}_albedo.png",
                                    f"{material_name_lower.replace('_', '')}_albedo.png",
                                ]
                            )

                        # Search for exact matches only
                        for pattern in specific_patterns:
                            potential_file = os.path.join(textures_dir, pattern)
                            if os.path.exists(potential_file):
                                logging.info(
                                    f"Found exact vegetation texture: {os.path.basename(potential_file)}"
                                )
                                material_info["textures"][texture_type] = potential_file
                                break

                        # If exact match not found, try partial matches
                        if not material_info["textures"].get(texture_type):
                            for file in os.listdir(textures_dir):
                                file_lower = file.lower()
                                if (
                                    file_lower.endswith(".png")
                                    and material_name_lower in file_lower
                                ):
                                    # Check for texture type in filename
                                    if texture_type in file_lower or (
                                        texture_type == "diffuse"
                                        and "basecolor" in file_lower
                                    ):
                                        full_path = os.path.join(textures_dir, file)
                                        logging.info(
                                            f"Found related vegetation texture: {file}"
                                        )
                                        material_info["textures"][
                                            texture_type
                                        ] = full_path
                                        break
        else:
            # Handle non-texture attributes
            material_info[standard_name] = value

    return material_info


def apply_generic_textures_to_segments(
    segments, object_name, object_dir, output_textures_dir=None
):
    """
    Apply generic textures to mesh segments that don't have textures.

    Args:
        segments (dict): Segments dictionary to update
        object_name (str): Name of the object
        object_dir (str): Directory containing the USD file
        output_textures_dir (str, optional): Directory to save extracted textures

    Returns:
        dict: Updated segments dictionary
    """
    # Skip if no segments
    if not segments:
        return segments

    # Find the materials/textures directory
    object_dir_parts = object_dir.split(os.sep)
    trees_dir = None
    shrub_dir = None
    debris_dir = None

    # Look for Trees directory
    for i in range(len(object_dir_parts)):
        if object_dir_parts[i] == "Trees":
            trees_dir = os.sep.join(object_dir_parts[: i + 1])
            break

    # Look for Shrub directory
    for i in range(len(object_dir_parts)):
        if object_dir_parts[i] == "Shrub":
            shrub_dir = os.sep.join(object_dir_parts[: i + 1])
            break

    # Look for Debris directory
    for i in range(len(object_dir_parts)):
        if object_dir_parts[i] == "Debris":
            debris_dir = os.sep.join(object_dir_parts[: i + 1])
            break

    # Set up textures directory based on dataset subdirectory found
    if trees_dir:
        textures_dir = os.path.join(trees_dir, "materials", "textures")
    elif shrub_dir:
        textures_dir = os.path.join(shrub_dir, "materials", "textures")
    elif debris_dir:
        textures_dir = os.path.join(debris_dir, "materials", "textures")
    else:
        # Check for Plant_Tropical directory
        tropical_dir = None
        for i in range(len(object_dir_parts)):
            if object_dir_parts[i] == "Plant_Tropical":
                tropical_dir = os.sep.join(object_dir_parts[: i + 1])
                break

        if tropical_dir:
            textures_dir = os.path.join(tropical_dir, "materials", "textures")
        else:
            # Try looking for material textures directory in current location
            textures_dir = os.path.join(object_dir, "materials", "textures")
            if not os.path.exists(textures_dir):
                # Go up one directory and look there
                parent_dir = os.path.dirname(object_dir)
                textures_dir = os.path.join(parent_dir, "materials", "textures")
                if not os.path.exists(textures_dir):
                    # Try root vegetation directory as a last resort
                    veg_root = None
                    for i in range(len(object_dir_parts)):
                        if object_dir_parts[i] == "vegetation":
                            veg_root = os.sep.join(object_dir_parts[: i + 1])
                            break
                    if veg_root:
                        textures_dir = os.path.join(veg_root, "materials", "textures")

    # If no textures directory found, return segments unchanged
    if not os.path.exists(textures_dir):
        return segments

    # Categorize object by name
    object_name_lower = object_name.lower()
    object_categories = []

    # Common categories
    category_keywords = {
        "tree": [
            "tree",
            "pine",
            "oak",
            "maple",
            "birch",
            "cedar",
            "ash",
            "spruce",
            "poplar",
            "aspen",
            "beech",
            "dogwood",
            "cypress",
            "hemlock",
        ],
        "palm": ["palm", "frond"],
        "flower": ["flower", "bloom", "blossom", "rose", "tulip", "lily"],
        "grass": [
            "grass",
            "reed",
            "sedge",
            "fern",
            "bamboo",
            "pampas",
            "fountain",
            "switchgrass",
        ],
        "bush": [
            "bush",
            "shrub",
            "boxwood",
            "barberry",
            "lilac",
            "lupin",
            "daphne",
            "forsythia",
            "vibernum",
            "rhododendron",
        ],
    }

    # Determine categories
    for category, keywords in category_keywords.items():
        if any(keyword in object_name_lower for keyword in keywords):
            object_categories.append(category)

    if not object_categories:
        # Default to tree if no other category matched
        object_categories = ["tree"]

    # Define generic texture sets for each category and part
    generic_textures = {
        "tree": {
            "bark": {
                "diffuse": "pinebark1_basecolor.png",
                "normal": "pinebark1_normal.png",
                "roughness": "pinebark1_roughness.png",
            },
            "leaf": {
                "diffuse": "oakleaves1_basecolor.png",
                "normal": "oakleaves1_normal.png",
                "roughness": "oakleaves1_roughness.png",
            },
        },
        "palm": {
            "bark": {
                "diffuse": "bark1_basecolor.png",
                "normal": "bark1_normal.png",
                "roughness": "bark1_roughness.png",
            },
            "leaf": {
                "diffuse": "palmleaves_mat_basecolor.png",
                "normal": "palmleaves_mat_normal.png",
                "roughness": "palmleaves_mat_roughness.png",
            },
        },
        "flower": {
            "stem": {
                "diffuse": "bark2_basecolor.png",
                "normal": "bark2_normal.png",
                "roughness": "bark2_roughness.png",
            },
            "petal": {
                "diffuse": "goldenchain_flowers_basecolor.png",
                "normal": "goldenchain_flowers_normal.png",
                "roughness": "goldenchain_flowers_roughness.png",
            },
        },
        "grass": {
            "blade": {
                "diffuse": "ashleaves1_basecolor.png",
                "normal": "ashleaves1_normal.png",
                "roughness": "ashleaves1_roughness.png",
            }
        },
        "bush": {
            "branch": {
                "diffuse": "bark3_basecolor.png",
                "normal": "bark3_normal.png",
                "roughness": "bark3_roughness.png",
            },
            "leaf": {
                "diffuse": "dogwood_leaf_basecolor.png",
                "normal": "dogwood_leaf_normal.png",
                "roughness": "dogwood_leaf_roughness.png",
            },
        },
    }

    # Special material name to texture mappings for problematic cases
    special_material_textures = {
        # Special material names
        "Lupin_m": {
            "diffuse": "lupin_basecolor.png",
            "normal": "lupin_normal.png",
            "roughness": "lupin_roughness.png",
        },
        "Dagger_M": {
            "diffuse": "plantatlas1_basecolor.png",
            "normal": "plantatlas1_normal.png",
            "roughness": "plantatlas1_roughness.png",
        },
        "bark3": {
            "diffuse": "bark3_basecolor.png",
            "normal": "bark3_normal.png",
            "roughness": "bark3_roughness.png",
        },
        "Pampas_flower": {
            "diffuse": "pampas_flower.png",
            "normal": "fanpalm_normal.png",  # Fallback normal map
            "roughness": "fanpalm_roughness.png",  # Fallback roughness map
        },
        "FountainGrass": {
            "diffuse": "fountaingrass_basecolor.png",
            "normal": "pampas_grass_normal.png",
            "roughness": "pampas_grass.png",
        },
        "TreeBark_01": {
            "diffuse": "tree_bark_03_diff_2k.png",
            "normal": "bark1_normal.png",
            "roughness": "sycamorebark2_roughness.png",
        },
        "Barberry": {
            "diffuse": "barberry_basecolor.png",
            "normal": "bark3_normal.png",  # Fallback
            "roughness": "bark3_roughness.png",  # Fallback
        },
        "Century_m": {
            "diffuse": "century_m_basecolor.png",
            "normal": "Century_m_Normal.png",
            "roughness": "Century_m_Roughness.png",
        },
        "Rhododendron": {
            "diffuse": "rhododendron_basecolor.png",
            "normal": "rhododendron_normal.png",
            "roughness": "rhododendron_roughness.png",
        },
        # Add more problematic materials
        "Burning_Bush": {
            "diffuse": "burningbush_leaf_basecolor.png",
            "normal": "burningbush_leaf_normal.png",
            "roughness": "burningbush_leaf_roughness.png",
        },
        "Cedar_Shrub": {
            "diffuse": "pinebark1_basecolor.png",
            "normal": "pinebark1_normal.png",
            "roughness": "pinebark1_roughness.png",
        },
        "Japanese_Flame": {
            "diffuse": "japaneseflame_basecolor.png",
            "normal": "japaneseflame_normal.png",
            "roughness": "japaneseflame_roughness.png",
        },
        "Honey_Myrtle": {
            "diffuse": "honeymyrtle_basecolor.png",
            "normal": "hollyprivet_normal.png",  # Fallback
            "roughness": "hollyprivet_roughness.png",  # Fallback
        },
        "Hurricane_Palm_bark_Mat": {
            "diffuse": "hurricanepalm_bark_basecolor.png",
            "normal": "hurricanepalm_bark_normal.png",
            "roughness": "hurricanepalm_bark_roughness.png",
        },
        "Australian_Fern_leaves_Mat": {
            "diffuse": "australianfern_leaves_basecolor.png",
            "normal": "australianfern_leaves_normal.png",
            "roughness": "australianfern_leaves_roughness.png",
        },
        "Australian_Fern_trunk": {
            "diffuse": "australianfern_trunk_basecolor.png",
            "normal": "australianfern_trunk_normal.png",
            "roughness": "australianfern_trunk_roughness.png",
        },
        "Agave_mat": {
            "diffuse": "agave_basecolor.png",
            "normal": "agave_normal.png",
            "roughness": "Agave_Roughness.png",
        },
        "Bamboo_leaf_Mat1": {
            "diffuse": "bambooleaf_basecolor.png",
            "normal": "bambooleaf_normal.png",
            "roughness": "bambooleaf_roughness.png",
        },
        "Bamboo_shoot_Mat1": {
            "diffuse": "bambooshoot_basecolor.png",
            "normal": "bambooshoot_normal.png",
            "roughness": "bambooshoot_roughness.png",
        },
        "CraneLily_mat": {
            "diffuse": "cranelily_basecolor.png",
            "normal": "cranelily_normal.png",
            "roughness": "cranelily_roughness.png",
        },
        "CraneLily_mat_2": {
            "diffuse": "cranelily_basecolor.png",
            "normal": "cranelily_normal.png",
            "roughness": "cranelily_roughness.png",
        },
        "CraneLily_mat_3": {
            "diffuse": "cranelily_basecolor.png",
            "normal": "cranelily_normal.png",
            "roughness": "cranelily_roughness.png",
        },
        "GrassPalm_bark": {
            "diffuse": "grasspalm_bark_basecolor.png",
            "normal": "grasspalm_bark_normal.png",
            "roughness": "grasspalm_bark_roughness.png",
        },
        "GrassPalm_leaves": {
            "diffuse": "grasspalm_leaves_basecolor.png",
            "normal": "grasspalm_leaves_normal.png",
            "roughness": "grasspalm_leaves_roughness.png",
        },
    }

    # First try to apply special material textures based on material name in each segment
    for segment_key, segment_info in segments.items():
        if segment_info is None:
            continue

        # Skip segments that already have textures
        if segment_info.get("textures") and len(segment_info["textures"]) > 0:
            continue

        # Initialize textures dict if needed
        if "textures" not in segment_info:
            segment_info["textures"] = {}

        # Get material name
        material_name = segment_info.get("name", "")

        # Check for special material name mapping
        if material_name in special_material_textures:
            for texture_type, texture_file in special_material_textures[
                material_name
            ].items():
                texture_path = os.path.join(textures_dir, texture_file)
                if os.path.exists(texture_path):
                    segment_info["textures"][texture_type] = texture_path

                    # Copy texture if needed
                    if output_textures_dir:
                        copied_path = copy_texture_to_output(
                            texture_path,
                            output_textures_dir,
                            object_name,
                            material_name,
                            texture_type,
                        )
                        if copied_path:
                            segment_info["textures"][
                                f"{texture_type}_copied"
                            ] = copied_path

        # If we found specific textures for this segment, continue to next segment
        if segment_info.get("textures") and len(segment_info["textures"]) > 0:
            continue

        # Apply category-based textures if specific ones weren't found
        material_type = segment_info.get("material_type", "")
        segment_type = "leaf"  # Default

        # Determine segment type
        if material_type in ["bark", "trunk", "stem", "branch", "stalk"]:
            segment_type = (
                "bark" if "bark" in generic_textures[object_categories[0]] else "branch"
            )
        elif material_type in ["leaf", "leaves", "foliage", "needle", "frond"]:
            segment_type = "leaf"
        elif material_type in ["petal", "flower", "bloom", "blossom"]:
            segment_type = "petal"
        elif material_type in ["blade", "grass"]:
            segment_type = "blade"

        # Get the right texture set based on object category and segment type
        for category in object_categories:
            if (
                category in generic_textures
                and segment_type in generic_textures[category]
            ):
                texture_set = generic_textures[category][segment_type]

                # Apply textures from set
                for texture_type, texture_file in texture_set.items():
                    texture_path = os.path.join(textures_dir, texture_file)
                    if os.path.exists(texture_path):
                        segment_info["textures"][texture_type] = texture_path

                        # Copy texture if needed
                        if output_textures_dir:
                            copied_path = copy_texture_to_output(
                                texture_path,
                                output_textures_dir,
                                object_name,
                                material_name or segment_key,
                                texture_type,
                            )
                            if copied_path:
                                segment_info["textures"][
                                    f"{texture_type}_copied"
                                ] = copied_path

                # Break once we found a suitable texture set
                if segment_info.get("textures") and len(segment_info["textures"]) > 0:
                    break

        # If we still don't have textures, try to find them by looking for any textures that might match
        if not segment_info.get("textures") or len(segment_info["textures"]) == 0:
            # Try to find any textures that might match by name
            object_dir_lower = object_dir.lower()
            material_name_lower = material_name.lower() if material_name else ""
            segment_key_lower = segment_key.lower()
            object_name_lower = object_name.lower()

            # Look in the textures directory for matching textures
            for texture_file in os.listdir(textures_dir):
                texture_lower = texture_file.lower()

                # Try to find matches by object name, material name, or segment key
                if (
                    object_name_lower in texture_lower
                    or material_name_lower in texture_lower
                    or segment_key_lower in texture_lower
                ):

                    # Determine texture type
                    texture_type = None
                    if "basecolor" in texture_lower or "diffuse" in texture_lower:
                        texture_type = "diffuse"
                    elif "normal" in texture_lower:
                        texture_type = "normal"
                    elif "roughness" in texture_lower:
                        texture_type = "roughness"

                    if texture_type:
                        texture_path = os.path.join(textures_dir, texture_file)
                        segment_info["textures"][texture_type] = texture_path

                        # Copy texture if needed
                        if output_textures_dir:
                            copied_path = copy_texture_to_output(
                                texture_path,
                                output_textures_dir,
                                object_name,
                                material_name or segment_key,
                                texture_type,
                            )
                            if copied_path:
                                segment_info["textures"][
                                    f"{texture_type}_copied"
                                ] = copied_path

    # If still missing textures, apply default textures
    for segment_key, segment_info in segments.items():
        if segment_info is None:
            continue

        if not segment_info.get("textures"):
            segment_info["textures"] = {}

        # Check if we're missing any texture types
        missing_types = []
        for texture_type in ["diffuse", "normal", "roughness"]:
            if texture_type not in segment_info["textures"]:
                missing_types.append(texture_type)

        if not missing_types:
            continue

        # Determine segment type again
        material_type = segment_info.get("material_type", "")
        segment_type = "leaf"  # Default

        if material_type in ["bark", "trunk", "stem", "branch", "stalk"]:
            segment_type = (
                "bark" if "bark" in generic_textures[object_categories[0]] else "branch"
            )
        elif material_type in ["leaf", "leaves", "foliage", "needle", "frond"]:
            segment_type = "leaf"
        elif material_type in ["petal", "flower", "bloom", "blossom"]:
            segment_type = "petal"
        elif material_type in ["blade", "grass"]:
            segment_type = "blade"

        # Apply default textures from the first applicable category
        for category in object_categories:
            if (
                category in generic_textures
                and segment_type in generic_textures[category]
            ):
                for texture_type in missing_types:
                    if texture_type in generic_textures[category][segment_type]:
                        texture_file = generic_textures[category][segment_type][
                            texture_type
                        ]
                        texture_path = os.path.join(textures_dir, texture_file)

                        if os.path.exists(texture_path):
                            segment_info["textures"][texture_type] = texture_path

                            # Copy texture if needed
                            if output_textures_dir:
                                copied_path = copy_texture_to_output(
                                    texture_path,
                                    output_textures_dir,
                                    object_name,
                                    segment_info.get("name", segment_key),
                                    texture_type,
                                )
                                if copied_path:
                                    segment_info["textures"][
                                        f"{texture_type}_copied"
                                    ] = copied_path

                # Break once we've applied textures from a category
                if all(
                    texture_type in segment_info["textures"]
                    for texture_type in missing_types
                ):
                    break

    return segments


def extract_materials_from_usd(
    usd_file_path, dataset_type=None, output_textures_dir=None
):
    """Extract exact, path-addressable material regions from a USD asset.

    Includes nested instance proxies and inherited/collection bindings,
    limits subsets to the ``materialBind`` family, and assigns every mesh face
    to exactly one output region. Texture values come from authored shader inputs.
    """

    logging.info("Extracting USD material regions from %s", usd_file_path)
    result = {
        "object_name": os.path.splitext(os.path.basename(usd_file_path))[0],
        "dataset_type": dataset_type,
        "file_path": str(usd_file_path),
        "date_processed": datetime.datetime.now().isoformat(),
        "segments": {},
        "materials": {},
        "segment_inventory": {},
    }

    try:
        stage = Usd.Stage.Open(str(usd_file_path), load=Usd.Stage.LoadAll)
        if not stage:
            raise RuntimeError("Usd.Stage.Open returned no stage")
    except Exception as exc:
        logging.error("Could not open USD file %s: %s", usd_file_path, exc)
        result["segment_inventory"] = {
            "errors": [f"stage open failed: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "warning_records": [],
            "mesh_paths": [],
            "empty_mesh_paths": [],
            "invalid_mesh_paths": [],
            "hole_faces": [],
            "point_instancers": [],
        }
        return result

    result.update(stage_color_management_metadata(stage))
    camera_hint = authored_perspective_camera_hint(stage)
    if camera_hint is not None:
        result["authored_perspective_camera_hint"] = camera_hint

    inventory = enumerate_material_regions(stage)
    inventory.errors.extend(
        error
        for error in stage_composition_errors(stage)
        if error not in inventory.errors
    )
    result["segment_inventory"] = {
        "errors": inventory.errors,
        "warnings": inventory.warnings,
        "warning_records": inventory.warning_records,
        "mesh_paths": inventory.mesh_paths,
        "empty_mesh_paths": inventory.empty_mesh_paths,
        "invalid_mesh_paths": inventory.invalid_mesh_paths,
        "hole_faces": inventory.hole_faces,
        "point_instancers": inventory.point_instancers,
    }

    material_cache = {}
    for region in inventory.regions:
        material = region.material
        material_name = material.GetPrim().GetName() if material else "unknown_material"
        for purpose_material in region.materials_by_binding_purpose.values():
            if not purpose_material:
                continue
            purpose_record_id = material_record_id(purpose_material)
            if purpose_record_id in material_cache:
                continue
            record = build_material_record(
                purpose_material, root_layer=stage.GetRootLayer()
            )
            if record:
                material_cache.setdefault(record["material_record_id"], record)
                for error in material_record_extraction_errors(record):
                    if error not in inventory.errors:
                        inventory.errors.append(error)
                for warning in record.get("annotation_warnings", []):
                    if warning not in inventory.warning_records:
                        inventory.warning_records.append(warning)
                    message = str(warning.get("message") or warning.get("code"))
                    if message and message not in inventory.warnings:
                        inventory.warnings.append(message)
        metadata = region_metadata(region)
        selected_record_id = metadata["material_record_id"]
        selected_record = (
            material_cache.get(selected_record_id) if selected_record_id else None
        )
        textures = (
            selected_record["selected_surface_textures"] if selected_record else {}
        )
        selected_texture_evidence = (
            selected_record["selected_surface_texture_evidence"]
            if selected_record
            else {}
        )
        texture_candidates = (
            selected_record["surface_texture_candidates"] if selected_record else {}
        )
        unresolved_textures = (
            selected_record["unresolved_surface_textures"] if selected_record else []
        )
        texture_role_conflicts = (
            selected_record["surface_texture_role_conflicts"]
            if selected_record
            else []
        )
        texture_ambiguities = (
            selected_record["surface_texture_ambiguities"]
            if selected_record
            else []
        )
        unknown_texture_roles = (
            selected_record["surface_unknown_texture_roles"]
            if selected_record
            else []
        )
        texture_fallbacks = selected_record["texture_fallbacks"] if selected_record else []
        source_repairs = selected_record["source_repairs"] if selected_record else []
        annotation_warnings = (
            selected_record["annotation_warnings"]
            if selected_record
            else []
        )

        selected_textures = dict(textures)
        if output_textures_dir:
            for texture_type, texture_path in textures.items():
                copied_path = copy_texture_to_output(
                    texture_path,
                    output_textures_dir,
                    result["object_name"],
                    material_name,
                    texture_type,
                )
                if copied_path:
                    selected_textures[f"{texture_type}_copied"] = copied_path

        segment_info = {
            "name": material_name,
            "material_type": material_name,
            "semantic_usage": region.semantic_usage,
            "textures": selected_textures,
            "selected_surface_texture_evidence": selected_texture_evidence,
            "texture_candidates": texture_candidates,
            "unresolved_textures": [],
            "texture_role_conflicts": [],
            "texture_ambiguities": [],
            "unknown_texture_roles": [],
            "appearance_evidence_warnings": annotation_warnings,
            "unresolved_texture_evidence": unresolved_textures,
            "texture_role_conflict_evidence": texture_role_conflicts,
            "texture_ambiguity_evidence": texture_ambiguities,
            "unknown_texture_role_evidence": unknown_texture_roles,
            "texture_fallbacks": texture_fallbacks,
            "source_repairs": source_repairs,
        }
        segment_info.update(metadata)
        result["segments"][region.key] = segment_info

    result["materials"] = material_cache

    logging.info(
        "Extracted %d exact material regions across %d meshes from %s",
        len(result["segments"]),
        len(inventory.mesh_paths),
        usd_file_path,
    )
    return result


@lru_cache(maxsize=1)
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


        except Exception as e:
            print(f"Error loading physics material {usda_file}: {str(e)}")

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
        semantic_usage = "__".join(parts[2:])
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


def extract_simready_materials_from_usd(usd_file_path, prop_name=None, prop_dir=None):
    """Extract every exact SimReady mesh-face material region.

    Effective USD bindings are computed for each region, so inherited and
    collection bindings cannot cause a mesh to disappear.  Only textures
    authored on the connected material graph are returned; directory-wide
    texture guessing is intentionally excluded.
    """

    object_name = prop_name or os.path.basename(os.path.dirname(usd_file_path))
    result = {
        "object_name": object_name,
        "category": ASSET_CLASS_MAPPING.get(object_name, "unknown"),
        "dataset_type": "simready",
        "file_path": str(usd_file_path),
        "segments": {},
        "materials": {},
        "segment_inventory": {},
    }
    if not os.path.isfile(usd_file_path):
        result["segment_inventory"] = {
            "errors": [f"USD entry point is missing: {usd_file_path}"],
            "warnings": [],
            "warning_records": [],
            "mesh_paths": [],
            "empty_mesh_paths": [],
            "invalid_mesh_paths": [],
            "hole_faces": [],
            "point_instancers": [],
        }
        return result

    try:
        stage = Usd.Stage.Open(str(usd_file_path), load=Usd.Stage.LoadAll)
        if not stage:
            raise RuntimeError("Usd.Stage.Open returned no stage")
    except Exception as exc:
        result["segment_inventory"] = {
            "errors": [f"stage open failed: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "warning_records": [],
            "mesh_paths": [],
            "empty_mesh_paths": [],
            "invalid_mesh_paths": [],
            "hole_faces": [],
            "point_instancers": [],
        }
        return result

    result.update(stage_color_management_metadata(stage))
    camera_hint = authored_perspective_camera_hint(stage)
    if camera_hint is not None:
        result["authored_perspective_camera_hint"] = camera_hint

    inventory = enumerate_material_regions(stage)
    inventory.errors.extend(
        error
        for error in stage_composition_errors(stage)
        if error not in inventory.errors
    )
    result["segment_inventory"] = {
        "errors": inventory.errors,
        "warnings": inventory.warnings,
        "warning_records": inventory.warning_records,
        "mesh_paths": inventory.mesh_paths,
        "empty_mesh_paths": inventory.empty_mesh_paths,
        "invalid_mesh_paths": inventory.invalid_mesh_paths,
        "hole_faces": inventory.hole_faces,
        "point_instancers": inventory.point_instancers,
    }
    physics_materials = load_physics_materials()
    material_cache = {}

    for region in inventory.regions:
        material = region.material
        material_name = material.GetPrim().GetName() if material else "unknown_material"
        opacity, material_type, material_usage = parse_material_name(material_name)
        physics_props = physics_materials.get(material_type, {})

        for purpose_material in region.materials_by_binding_purpose.values():
            if not purpose_material:
                continue
            purpose_record_id = material_record_id(purpose_material)
            if purpose_record_id in material_cache:
                continue
            record = build_material_record(
                purpose_material, root_layer=stage.GetRootLayer()
            )
            if record:
                material_cache.setdefault(record["material_record_id"], record)
                for error in material_record_extraction_errors(record):
                    if error not in inventory.errors:
                        inventory.errors.append(error)
                for warning in record.get("annotation_warnings", []):
                    if warning not in inventory.warning_records:
                        inventory.warning_records.append(warning)
                    message = str(warning.get("message") or warning.get("code"))
                    if message and message not in inventory.warnings:
                        inventory.warnings.append(message)
        metadata = region_metadata(region)
        selected_record_id = metadata["material_record_id"]
        selected_record = (
            material_cache.get(selected_record_id) if selected_record_id else None
        )
        textures = (
            selected_record["selected_surface_textures"] if selected_record else {}
        )
        selected_texture_evidence = (
            selected_record["selected_surface_texture_evidence"]
            if selected_record
            else {}
        )
        texture_candidates = (
            selected_record["surface_texture_candidates"] if selected_record else {}
        )
        unresolved_textures = (
            selected_record["unresolved_surface_textures"] if selected_record else []
        )
        texture_role_conflicts = (
            selected_record["surface_texture_role_conflicts"]
            if selected_record
            else []
        )
        texture_ambiguities = (
            selected_record["surface_texture_ambiguities"]
            if selected_record
            else []
        )
        unknown_texture_roles = (
            selected_record["surface_unknown_texture_roles"]
            if selected_record
            else []
        )
        texture_fallbacks = selected_record["texture_fallbacks"] if selected_record else []
        source_repairs = selected_record["source_repairs"] if selected_record else []
        annotation_warnings = (
            selected_record["annotation_warnings"]
            if selected_record
            else []
        )

        segment_info = {
            "name": material_name,
            "opacity": opacity,
            "material_type": material_type,
            "material_usage": material_usage or None,
            "semantic_usage": material_usage or region.semantic_usage,
            "source_physics": {
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
            },
            "textures": dict(textures),
            "selected_surface_texture_evidence": selected_texture_evidence,
            "texture_candidates": texture_candidates,
            "unresolved_textures": [],
            "texture_role_conflicts": [],
            "texture_ambiguities": [],
            "unknown_texture_roles": [],
            "appearance_evidence_warnings": annotation_warnings,
            "unresolved_texture_evidence": unresolved_textures,
            "texture_role_conflict_evidence": texture_role_conflicts,
            "texture_ambiguity_evidence": texture_ambiguities,
            "unknown_texture_role_evidence": unknown_texture_roles,
            "texture_fallbacks": texture_fallbacks,
            "source_repairs": source_repairs,
        }
        segment_info.update(metadata)
        result["segments"][region.key] = segment_info

    result["materials"] = material_cache

    logging.info(
        "Extracted %d exact SimReady regions across %d meshes from %s",
        len(result["segments"]),
        len(inventory.mesh_paths),
        usd_file_path,
    )
    return result
