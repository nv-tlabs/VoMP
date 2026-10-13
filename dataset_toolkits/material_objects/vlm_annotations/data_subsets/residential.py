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

from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    RESIDENTIAL_BASE_DIR,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.render import (
    render_sphere_with_texture,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.vlm import (
    analyze_material_with_vlm,
    parse_vlm_properties,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.common import (
    extract_materials_from_usd,
)
from dataset_toolkits.material_objects.vlm_annotations.vlm_input_builder import (
    make_common_user_prompt,
)
import re
from tqdm import tqdm
import os
import logging
import copy
import traceback
import json
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.common import UsdJsonEncoder

PROMPTS = {
    "few_shot_examples": (
        """
Text-only examples (no example images are supplied for these cases):
- These cases deliberately span compliant foams, elastomers, plastics, wood, and metals. They illustrate scale; they are not lookup bins or class defaults.
- Do not copy a value merely because a broad label such as metal, plastic, fabric, or organic is similar. Identify the material subtype, physical form, and exact segment usage first.
- Estimate room-temperature, dry, small-strain effective properties for the bulk material represented by the exact segment. For porous materials, density means apparent bulk density, not solid-polymer or fiber density.
- For layered objects, annotate the isolated segment: a cushion core is bulk foam, whereas an explicitly isolated upholstery cover is textile or leather. For anisotropic materials, the required scalar modulus is an effective approximation for the loading direction most consistent with the segment's usage.

Example 1:
Material: flexible open-cell polyurethane foam
Usage: soft seat-cushion core
Object name: SeatCushionCore

Analysis:
Step 1: The named segment is the compliant cushion core, not a thin upholstery cover and not solid polyurethane.
Step 2: Open-cell foam is a porous cellular structure, so its effective bulk stiffness and apparent density are far below those of the polymer forming the cell walls.
Step 3: Flexible seat foam is nonlinear and rate dependent; use a representative small-vibration linearized modulus for this requested scalar label.
Step 4: A soft 65 kg/m^3 seat foam is represented by about 0.20 MPa Young's modulus and a Poisson's ratio of 0.25.

Young's modulus: 2.0e5 Pa
Poisson's ratio: 0.25
Density: 6.5e1 kg/m^3

Example 2:
Material: 6061-T6 aluminum alloy
Usage: lightweight rigid frame member
Object name: AluminumFrame

Analysis:
Step 1: The stated alloy and lightweight frame usage identify aluminum rather than treating every metal-colored segment as steel.
Step 2: Surface color or finish alone cannot determine a metal alloy, so subtype and structural usage are decisive here.
Step 3: Wrought 6061-T6 is a stiff, low-density metal at room temperature, but is roughly three times less stiff and dense than steel.
Step 4: Representative elastic constants are about 68.3 GPa Young's modulus, 0.33 Poisson's ratio, and 2700 kg/m^3 density.

Young's modulus: 6.83e10 Pa
Poisson's ratio: 0.33
Density: 2.70e3 kg/m^3

Example 3:
Material: cured Sylgard 184 PDMS silicone rubber
Usage: soft vibration-isolation pad
Object name: SiliconeIsolationPad

Analysis:
Step 1: The exact segment is a cured silicone elastomer pad, not a rigid thermoplastic or a cellular cushion foam.
Step 2: A solid elastomer is much denser than foam but remains highly compliant and nearly incompressible.
Step 3: Silicone is nonlinear at large strain, so use a representative small-strain effective modulus and keep Poisson's ratio just below 0.5.
Step 4: Typical cured Sylgard 184 values are about 2 MPa Young's modulus, 0.495 Poisson's ratio, and 1030 kg/m^3 density.

Young's modulus: 2.0e6 Pa
Poisson's ratio: 0.495
Density: 1.03e3 kg/m^3

Example 4:
Material: structural carbon steel
Usage: load-bearing beam
Object name: CarbonSteelBeam

Analysis:
Step 1: The stated structural-carbon-steel subtype and beam usage identify an iron-based structural alloy rather than a generic metallic finish.
Step 2: Geometry and usage support a load-bearing metal, while the subtype distinguishes it from aluminum, brass, or a thin plated polymer.
Step 3: Carbon steel has high elastic stiffness and density at room temperature.
Step 4: Representative structural-steel values are about 200 GPa Young's modulus, 0.30 Poisson's ratio, and 7850 kg/m^3 density.

Young's modulus: 2.0e11 Pa
Poisson's ratio: 0.30
Density: 7.85e3 kg/m^3

Example 5:
Material: clear coastal Douglas-fir wood at approximately 12 percent moisture
Usage: furniture rail loaded primarily along the grain
Object name: DouglasFirRail

Analysis:
Step 1: This is a solid wood member with a specified species, moisture condition, and primary loading direction, not a generic organic-material bin.
Step 2: Wood is orthotropic; its longitudinal stiffness is much larger than its radial or tangential stiffness.
Step 3: Because this rail is used along the grain, the requested scalar should approximate the longitudinal small-strain modulus rather than an arbitrary isotropic wood value.
Step 4: Clear coastal Douglas-fir near 12 percent moisture is represented by about 13.4 GPa longitudinal modulus, 0.292 longitudinal-radial Poisson's ratio, and 540 kg/m^3 density.

Young's modulus: 1.34e10 Pa
Poisson's ratio: 0.292
Density: 5.4e2 kg/m^3

Example 6:
Material: unfilled injection-molded polypropylene
Usage: household container wall
Object name: PolypropyleneContainer

Analysis:
Step 1: The named segment is an unfilled molded polypropylene wall, not an elastomer, foam, or fiber-reinforced engineering plastic.
Step 2: The smooth consumer-plastic form and container usage are consistent with a low-density semicrystalline thermoplastic.
Step 3: The precise polypropylene subtype supports a gigapascal-scale modulus, well above rubber but far below wood or metal.
Step 4: A representative unfilled grade has about 1.2 GPa Young's modulus, 0.40 Poisson's ratio, and 905 kg/m^3 density.

Young's modulus: 1.2e9 Pa
Poisson's ratio: 0.40
Density: 9.05e2 kg/m^3
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
Step 3: Determine the specific material subtype considering its usage
Step 4: Reason through each property estimate based on visual and contextual clues

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
    semantic_usage,
    object_name,
    has_texture_sphere=True,
):
    del has_texture_sphere
    return make_common_user_prompt(
        material_type,
        semantic_usage,
        object_name,
    )


# Use the centralized parser function from vlm.py instead
parse_vlm_output = parse_vlm_properties


def list_residential_objects():
    """
    List all available residential objects in the residential directory.
    """
    usd_files = []
    print("\nAvailable residential objects:")
    for root, _, files in os.walk(RESIDENTIAL_BASE_DIR):
        for file in files:
            if file.endswith(".usd") and not os.path.basename(root).startswith("."):
                usd_files.append(os.path.join(root, file))
                print(f"  - {os.path.basename(root)}/{file}")
    print()


def process_residential(
    vlm_model,
    vlm_processor,
    limit=None,
    processed_objects=None,
    output_file=None,
    existing_results=None,
):
    usd_files = []
    for root, _, files in os.walk(RESIDENTIAL_BASE_DIR):
        for file in files:
            if file.endswith(".usd") and not os.path.basename(root).startswith("."):
                usd_files.append(os.path.join(root, file))

    logging.info(f"Found {len(usd_files)} USD files in residential dataset")

    # Initialize tracking sets and results
    processed_objects = set() if processed_objects is None else processed_objects
    existing_results = [] if existing_results is None else existing_results

    # Build a set of already processed object names from existing_results
    existing_object_names = {
        result.get("object_name")
        for result in existing_results
        if "object_name" in result
    }
    logging.info(
        f"Found {len(existing_object_names)} already processed objects in existing results"
    )

    # Add names from existing_results to processed_objects to avoid reprocessing
    processed_objects.update(existing_object_names)

    # Create a copy of existing_results to avoid modifying the original
    all_results = copy.deepcopy(existing_results)

    usd_files.sort()

    if limit and limit > 0:
        usd_files = usd_files[:limit]

    success_count = 0
    failed_objects = []
    total_segments = 0
    unique_materials = set()
    materials_per_object = {}
    total_rendered_segments = 0
    total_vlm_segments = 0

    # Count total segments from existing results
    for result in existing_results:
        total_segments += len(result.get("segments", {}))

    # Statistics for texture availability
    segments_with_texture = 0
    segments_without_texture = 0
    segments_with_thumbnail_only = 0

    # Track processed files to avoid duplicates from the same directory
    processed_files = set()

    for usd_file in tqdm(usd_files, desc=f"Processing residential dataset"):
        # Extract object name from path
        object_name = os.path.basename(os.path.dirname(usd_file))

        # Skip if we already processed this exact file
        if usd_file in processed_files:
            continue

        # Skip objects that have already been processed
        if object_name in processed_objects:
            logging.info(f"Skipping already processed object: {object_name}")
            continue

        try:
            directory = os.path.dirname(usd_file)

            # Extract material information
            result = extract_materials_from_usd(usd_file, "residential")

            if result:
                # Add to processed_files to avoid duplicates
                processed_files.add(usd_file)

                # Track statistics
                segments = result.get("segments", {})
                total_segments += len(segments)

                # Remove object_name and note fields from segments
                for segment_key, segment_info in segments.items():
                    if "object_name" in segment_info:
                        del segment_info["object_name"]
                    if "note" in segment_info:
                        del segment_info["note"]

                # Count unique materials for this object
                object_materials = set()
                for segment_name, segment_info in segments.items():
                    material_name = segment_info.get("material_type", "unknown")
                    unique_materials.add(material_name)
                    object_materials.add(material_name)

                # Record materials per object
                if len(segments) > 0:
                    materials_per_object[object_name] = len(object_materials)

                # Get thumbnail path if available
                thumb_path = None
                # For residential dataset, thumbnails are in .thumbs/256x256 directory
                thumb_dir = os.path.join(
                    os.path.dirname(usd_file), ".thumbs", "256x256"
                )

                has_thumbnail = False
                if os.path.exists(thumb_dir):
                    # Try to find a thumbnail matching the USD filename
                    usd_filename = os.path.basename(usd_file)
                    thumb_candidates = [
                        # Regular thumbnail
                        os.path.join(thumb_dir, f"{usd_filename}.png"),
                        # Auto-generated thumbnail
                        os.path.join(thumb_dir, f"{usd_filename}.auto.png"),
                    ]

                    for candidate in thumb_candidates:
                        if os.path.exists(candidate):
                            thumb_path = candidate
                            has_thumbnail = True
                            logging.info(f"Found thumbnail: {thumb_path}")
                            break

                # Process VLM for all segments if VLM model is provided
                os.makedirs("/tmp/vlm", exist_ok=True)

                if vlm_model and len(segments) > 0:
                    for segment_key, segment_info in segments.items():
                        textures = segment_info.get("textures", {})

                        # Log texture information for diagnostics
                        logging.info(
                            f"Segment {segment_key} has textures: {list(textures.keys())}"
                        )

                        # Check if we have either a normal or roughness texture for rendering
                        has_texture = (
                            "normal" in textures
                            or "roughness" in textures
                            or "diffuse" in textures
                        )
                        if has_texture:
                            # Has texture - render sphere and use with thumbnail
                            segments_with_texture += 1
                            logging.info(
                                f"Rendering texture sphere for {object_name}, segment {segment_key}"
                            )

                            # Set up file path for this segment's rendered sphere
                            segment_render_path = f"/tmp/vlm/texture_sphere_{object_name}_{segment_key}.png"

                            # Render the textured sphere
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
                            # No texture - just use thumbnail
                            segments_without_texture += 1
                            segment_render_path = None
                            logging.info(
                                f"No texture for {object_name}, segment {segment_key}. Using thumbnail only."
                            )

                        # Always try to process with VLM, even if no texture
                        try:
                            # If we have a thumbnail but no texture, still run VLM with just the thumbnail
                            if not has_texture and has_thumbnail:
                                segments_with_thumbnail_only += 1
                                logging.info(
                                    f"Using thumbnail only for {object_name}, segment {segment_key}"
                                )

                            # Don't run VLM if we have neither texture nor thumbnail
                            if not segment_render_path and not has_thumbnail:
                                logging.warning(
                                    f"Skipping VLM for {segment_key} - no texture or thumbnail available"
                                )
                                continue

                            # Set semantic usage to segment name but don't store in segment data
                            semantic_usage = segment_key
                            temp_object_name = object_name

                            # Create custom prompt based on texture availability
                            custom_prompt = make_user_prompt(
                                segment_info["material_type"],
                                semantic_usage,
                                temp_object_name,
                                has_texture_sphere=segment_render_path is not None,
                            )

                            # Store the custom prompt in material_info but not object_name
                            segment_info["user_prompt"] = custom_prompt

                            # Debug: Log the prompt type based on texture availability
                            if segment_render_path is not None:
                                logging.info(
                                    f"Using prompt WITH texture sphere for {object_name}, segment {segment_key}"
                                )
                            else:
                                logging.info(
                                    f"Using prompt WITHOUT texture sphere for {object_name}, segment {segment_key}"
                                )
                                logging.info(
                                    f"PROMPT: {custom_prompt[:100]}..."
                                )  # Print just the beginning of the prompt

                            # Create a temporary segment_info with object_name for VLM but don't save to result
                            temp_segment_info = segment_info.copy()
                            temp_segment_info["semantic_usage"] = semantic_usage
                            temp_segment_info["object_name"] = temp_object_name

                            vlm_analysis = analyze_material_with_vlm(
                                segment_render_path,  # This can be None, in which case only thumbnail is used
                                temp_segment_info,  # Use temporary copy with object_name
                                vlm_model,
                                vlm_processor,
                                thumbnail_path=thumb_path,
                                dataset_name="residential",
                                PROMPTS=PROMPTS,
                                make_user_prompt=make_user_prompt,
                                parse_vlm_output=parse_vlm_output,
                            )

                            # Add VLM analysis to segment info
                            if vlm_analysis and "error" not in vlm_analysis:
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
                                logging.info(
                                    f"  Density: {vlm_analysis.get('density')}"
                                )
                            else:
                                logging.error(
                                    f"VLM analysis failed for {segment_key}: {vlm_analysis.get('error', 'Unknown error')}"
                                )
                        except Exception as e:
                            logging.error(
                                f"Error during VLM analysis for {segment_key}: {str(e)}"
                            )
                            logging.error(traceback.format_exc())

                        total_rendered_segments += 1

                all_results.append(result)  # Add to our local copy of results
                processed_objects.add(object_name)  # Mark as processed

                # Incremental save after each object if output file is provided
                if output_file:
                    try:
                        with open(output_file, "w") as f:
                            # Debug save contents
                            logging.info(
                                f"Saving checkpoint with {len(all_results)} objects"
                            )

                            # Ensure result types are JSON serializable
                            for idx, item in enumerate(all_results):
                                if "segments" in item:
                                    for seg_key, seg_info in item["segments"].items():
                                        # Remove object_name and note fields if they exist
                                        if "object_name" in seg_info:
                                            del seg_info["object_name"]
                                        if "note" in seg_info:
                                            del seg_info["note"]

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

                            # Try to serialize to a string first to check for issues
                            try:
                                json_str = json.dumps(
                                    all_results, cls=UsdJsonEncoder, indent=4
                                )
                                logging.info(
                                    f"JSON serialization successful, string length: {len(json_str)}"
                                )

                                # Now write to file
                                f.write(json_str)

                            except Exception as json_err:
                                logging.error(
                                    f"JSON serialization error: {str(json_err)}"
                                )
                                # Try to identify problematic objects
                                for i, item in enumerate(all_results):
                                    try:
                                        json.dumps(item, cls=UsdJsonEncoder)
                                    except Exception as e:
                                        logging.error(
                                            f"Error serializing object {i}: {str(e)}"
                                        )
                                raise json_err  # Re-raise to be caught by outer exception handler

                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {str(e)}")

                        logging.error(traceback.format_exc())

                success_count += 1
            else:
                logging.warning(f"No material information extracted for {usd_file}")
                failed_objects.append(object_name)
        except Exception as e:
            logging.error(f"Error processing {usd_file}: {str(e)}")
            logging.error(traceback.format_exc())
            failed_objects.append(os.path.basename(os.path.dirname(usd_file)))

    # Convert materials_per_object to list format for consistency with simready
    materials_per_object_list = []
    for obj_name, count in materials_per_object.items():
        materials_per_object_list.append(obj_name)

    # Log texture statistics
    logging.info("Texture Statistics:")
    logging.info(f"  Total segments processed: {total_segments}")
    logging.info(f"  Segments with textures: {segments_with_texture}")
    logging.info(f"  Segments without textures: {segments_without_texture}")
    logging.info(f"  Segments with thumbnail only: {segments_with_thumbnail_only}")
    logging.info(f"  Total VLM analyses completed: {total_vlm_segments}")

    return (
        all_results,
        len(usd_files),
        success_count,
        failed_objects,
        total_segments,
        total_rendered_segments,
        total_vlm_segments,
        list(unique_materials),
        materials_per_object_list,
    )
