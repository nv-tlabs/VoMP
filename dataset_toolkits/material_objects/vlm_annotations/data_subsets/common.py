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
import argparse
import numpy as np
import logging
import copy
from PIL import Image
from tqdm import tqdm
from pxr import Usd, UsdGeom, UsdShade, Sdf, Ar
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    COMMERCIAL_BASE_DIR,
    RESIDENTIAL_BASE_DIR,
    VEGETATION_BASE_DIR,
)
import datetime
import uuid


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
        import shutil

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
                    # If we already found a texture through our material name matching,
                    # don't override it for vegetation materials
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
    """
    Extract material information from a USD file with improved handling of material bindings.

    Args:
        usd_file_path: Path to the USD file
        dataset_type: Type of dataset (residential, commercial, etc.)

    Returns:
        Dictionary with material information
    """
    logging.info(f"Extracting materials from {usd_file_path}")
    result = {
        "object_name": os.path.splitext(os.path.basename(usd_file_path))[0],
        "dataset_type": dataset_type,
        "file_path": usd_file_path,
        "date_processed": datetime.datetime.now().isoformat(),
        "segments": {},
    }

    # Open the USD stage
    try:
        stage = Usd.Stage.Open(usd_file_path)
        if not stage:
            logging.error(f"Could not open USD file: {usd_file_path}")
            return None
    except Exception as e:
        logging.error(f"Error opening USD file {usd_file_path}: {str(e)}")
        return None

    # Track all materials we find in the stage
    all_materials = {}

    # First pass: collect all materials and their properties
    logging.info("First pass: collecting all materials")
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            material_path = str(prim.GetPath())
            material_name = prim.GetName()

            # Store material info with default values
            all_materials[material_path] = {
                "name": material_name,
                "material_type": material_name,  # Default to name
                "textures": {},
            }

            # Process material's shaders to find textures
            # Correctly get all the shader prims in this material
            shader_prims = []
            for child_prim in Usd.PrimRange(prim):
                if child_prim.IsA(UsdShade.Shader):
                    shader_prims.append(child_prim)

            for shader_prim in shader_prims:
                shader = UsdShade.Shader(shader_prim)
                if not shader:
                    continue

                # Inspect shader inputs for textures
                for input in shader.GetInputs():
                    input_name = input.GetBaseName()

                    # Check if this input has a connected source that's an asset
                    if input.HasConnectedSource():
                        source = input.GetConnectedSource()
                        if source:
                            source_shader, source_output, _ = source
                            source_prim = source_shader.GetPrim()

                            # Check if the source is a texture
                            if source_prim.IsA(UsdShade.Shader):
                                source_shader_id = UsdShade.Shader(
                                    source_prim
                                ).GetShaderId()
                                if "texture" in str(source_shader_id).lower():
                                    # Try to find the file asset path
                                    for source_input in UsdShade.Shader(
                                        source_prim
                                    ).GetInputs():
                                        if source_input.GetBaseName() in [
                                            "file",
                                            "filename",
                                            "filePath",
                                            "varname",
                                        ]:
                                            asset_path = source_input.Get()
                                            if asset_path:
                                                # Determine texture type from connection patterns
                                                tex_type = "unknown"
                                                if (
                                                    "diffuse" in input_name.lower()
                                                    or "albedo" in input_name.lower()
                                                    or "color" in input_name.lower()
                                                ):
                                                    tex_type = "diffuse"
                                                elif "normal" in input_name.lower():
                                                    tex_type = "normal"
                                                elif "roughness" in input_name.lower():
                                                    tex_type = "roughness"
                                                elif "metallic" in input_name.lower():
                                                    tex_type = "metallic"
                                                elif "specular" in input_name.lower():
                                                    tex_type = "specular"
                                                elif (
                                                    "displacement" in input_name.lower()
                                                ):
                                                    tex_type = "displacement"

                                                # Store texture path
                                                logging.info(
                                                    f"Found texture: {tex_type} = {asset_path} for material {material_name}"
                                                )
                                                all_materials[material_path][
                                                    "textures"
                                                ][tex_type] = str(asset_path)

                    # Direct asset inputs (not connected through other shaders)
                    elif input.GetTypeName() == "asset":
                        asset_path = input.Get()
                        if asset_path:
                            # Determine texture type from input name
                            tex_type = "unknown"
                            if (
                                "diffuse" in input_name.lower()
                                or "albedo" in input_name.lower()
                                or "color" in input_name.lower()
                            ):
                                tex_type = "diffuse"
                            elif "normal" in input_name.lower():
                                tex_type = "normal"
                            elif "roughness" in input_name.lower():
                                tex_type = "roughness"
                            elif "metallic" in input_name.lower():
                                tex_type = "metallic"
                            elif "specular" in input_name.lower():
                                tex_type = "specular"
                            elif "displacement" in input_name.lower():
                                tex_type = "displacement"

                            # Store texture path
                            logging.info(
                                f"Found direct texture: {tex_type} = {asset_path} for material {material_name}"
                            )
                            all_materials[material_path]["textures"][tex_type] = str(
                                asset_path
                            )

    # Second pass: find all material bindings
    logging.info("Second pass: finding material bindings")

    # Process meshes and their subsets
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh_name = prim.GetName()
            logging.info(f"Processing mesh: {mesh_name}")

            # First check direct binding on the mesh
            binding_api = UsdShade.MaterialBindingAPI(prim)
            direct_binding = binding_api.GetDirectBinding()
            direct_material = None

            if direct_binding.GetMaterial():
                direct_material = direct_binding.GetMaterial()
                material_path = str(direct_material.GetPath())
                logging.info(f"  Found direct material binding: {material_path}")

                if material_path in all_materials:
                    # Create segment for the whole mesh
                    segment_key = f"{mesh_name}_whole"
                    material_info = all_materials[material_path].copy()
                    material_info["semantic_usage"] = mesh_name

                    result["segments"][segment_key] = material_info
                    logging.info(
                        f"  Created segment {segment_key} with material {material_path}"
                    )

            # Then check GeomSubsets - these are more specific material assignments
            imageable = UsdGeom.Imageable(prim)
            subsets = UsdGeom.Subset.GetGeomSubsets(imageable)

            if subsets:
                logging.info(f"  Found {len(subsets)} geom subsets for {mesh_name}")
                for subset in subsets:
                    subset_prim = subset.GetPrim()
                    subset_name = subset_prim.GetName()
                    family = (
                        subset.GetFamilyNameAttr().Get()
                        if subset.GetFamilyNameAttr()
                        else "unknown"
                    )

                    logging.info(
                        f"  Processing subset: {subset_name} (Family: {family})"
                    )

                    # Check material binding on subset
                    subset_binding_api = UsdShade.MaterialBindingAPI(subset_prim)
                    subset_direct_binding = subset_binding_api.GetDirectBinding()

                    if subset_direct_binding.GetMaterial():
                        subset_material = subset_direct_binding.GetMaterial()
                        subset_material_path = str(subset_material.GetPath())
                        logging.info(
                            f"    Found subset material binding: {subset_material_path}"
                        )

                        if subset_material_path in all_materials:
                            # Create segment for this subset
                            segment_key = subset_name
                            material_info = all_materials[subset_material_path].copy()
                            material_info["semantic_usage"] = subset_name

                            result["segments"][segment_key] = material_info
                            logging.info(
                                f"    Created segment {segment_key} with material {subset_material_path}"
                            )

            # If no subsets but we have a direct material, use that
            if not subsets and direct_material:
                material_path = str(direct_material.GetPath())

                if material_path in all_materials:
                    # Create segment for the whole mesh
                    segment_key = mesh_name
                    material_info = all_materials[material_path].copy()
                    material_info["semantic_usage"] = mesh_name

                    result["segments"][segment_key] = material_info
                    logging.info(
                        f"  No subsets, created segment {segment_key} with material {material_path}"
                    )

    # Final check - make sure we have segments
    if not result["segments"]:
        logging.warning(f"No material segments found in {usd_file_path}")

        # Last resort - add all materials as segments
        for material_path, material_info in all_materials.items():
            material_name = material_info["name"]
            segment_key = f"material_{material_name}"

            result["segments"][segment_key] = material_info.copy()
            result["segments"][segment_key]["semantic_usage"] = material_name

            logging.info(
                f"Added material {material_name} as segment {segment_key} (last resort)"
            )

    logging.info(
        f"Extracted {len(result['segments'])} material segments from {usd_file_path}"
    )
    return result
