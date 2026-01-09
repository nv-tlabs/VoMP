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
    VEGETATION_BASE_DIR,
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
    UsdJsonEncoder,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.residential import (
    PROMPTS,
    make_user_prompt,
)
import re
from tqdm import tqdm
import os
import logging
import copy

# Use the centralized parser function
parse_vlm_output = parse_vlm_properties


def list_vegetation_objects():
    """
    List all available vegetation objects in the vegetation directory.
    """
    usd_files = []
    print("\nAvailable vegetation objects:")
    for root, _, files in os.walk(VEGETATION_BASE_DIR):
        for file in files:
            if file.endswith(".usd") and not os.path.basename(root).startswith("."):
                usd_files.append(os.path.join(root, file))
                print(f"  - {os.path.basename(root)}/{file}")
    print()


def process_vegetation(
    vlm_model,
    vlm_processor,
    limit=None,
    processed_objects=None,
    output_file=None,
    existing_results=None,
):
    usd_files = []
    for root, _, files in os.walk(VEGETATION_BASE_DIR):
        for file in files:
            if file.endswith(".usd") and not os.path.basename(root).startswith("."):
                usd_files.append(os.path.join(root, file))

    logging.info(f"Found {len(usd_files)} USD files in vegetation dataset")

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

    for usd_file in tqdm(usd_files, desc=f"Processing vegetation dataset"):
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
            result = extract_materials_from_usd(usd_file, "vegetation")

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

                # Process VLM for each segment if VLM model is provided
                os.makedirs("/tmp/vlm", exist_ok=True)

                if vlm_model and len(segments) > 0:
                    for segment_name, segment_info in segments.items():
                        textures = segment_info.get("textures", {})

                        # Log texture information for diagnostics
                        logging.info(
                            f"Segment {segment_name} has textures: {list(textures.keys())}"
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
                                f"Rendering texture sphere for {object_name}, segment {segment_name}"
                            )

                            # Set up file path for this segment's rendered sphere
                            segment_render_path = f"/tmp/vlm/texture_sphere_{object_name}_{segment_name}.png"

                            # Render the textured sphere
                            try:
                                rgb_buffer = render_sphere_with_texture(
                                    textures, segment_render_path
                                )
                                logging.info(f"RGB buffer shape: {rgb_buffer.shape}")
                            except Exception as e:
                                logging.error(
                                    f"Error rendering texture for {segment_name}: {str(e)}"
                                )
                                segment_render_path = None
                        else:
                            # No texture - just use thumbnail
                            segments_without_texture += 1
                            segment_render_path = None
                            logging.info(
                                f"No texture for {object_name}, segment {segment_name}. Using thumbnail only."
                            )

                        try:
                            # Get thumbnail path if available
                            thumb_path = None
                            # For vegetation dataset, thumbnails are in .thumbs/256x256 directory
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

                            # If we have a thumbnail but no texture, still run VLM with just the thumbnail
                            if not has_texture and has_thumbnail:
                                segments_with_thumbnail_only += 1
                                logging.info(
                                    f"Using thumbnail only for {object_name}, segment {segment_name}"
                                )

                            # Don't run VLM if we have neither texture nor thumbnail
                            if not segment_render_path and not has_thumbnail:
                                logging.warning(
                                    f"Skipping VLM for {segment_name} - no texture or thumbnail available"
                                )
                                continue

                            # Set semantic usage to segment name but don't store in segment data
                            semantic_usage = segment_name
                            temp_object_name = object_name

                            # Create custom prompt based on texture availability
                            custom_prompt = make_user_prompt(
                                segment_info["material_type"],
                                semantic_usage,
                                temp_object_name,
                                has_texture_sphere=segment_render_path is not None,
                            )

                            # Store the prompt for VLM but not object_name
                            segment_info["user_prompt"] = custom_prompt

                            # Debug: Log the prompt type based on texture availability
                            if segment_render_path is not None:
                                logging.info(
                                    f"Using prompt WITH texture sphere for {object_name}, segment {segment_name}"
                                )
                            else:
                                logging.info(
                                    f"Using prompt WITHOUT texture sphere for {object_name}, segment {segment_name}"
                                )
                                logging.info(
                                    f"PROMPT: {custom_prompt[:100]}..."
                                )  # Print just the beginning of the prompt

                            # Create a temporary segment_info with object_name for VLM but don't save to result
                            temp_segment_info = segment_info.copy()
                            temp_segment_info["semantic_usage"] = semantic_usage
                            temp_segment_info["object_name"] = temp_object_name

                            vlm_analysis = analyze_material_with_vlm(
                                segment_render_path,
                                temp_segment_info,  # Use temporary copy with object_name
                                vlm_model,
                                vlm_processor,
                                thumbnail_path=thumb_path,
                                dataset_name="vegetation",
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
                                logging.info(f"VLM analysis successful:")
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
                                    f"VLM analysis failed: {vlm_analysis.get('error', 'Unknown error')}"
                                )
                        except Exception as e:
                            import traceback

                            logging.error(f"Error during VLM analysis: {str(e)}")
                            logging.error(traceback.format_exc())

                        total_rendered_segments += 1

                all_results.append(result)  # Add to our local copy of results
                processed_objects.add(object_name)  # Mark as processed

                # Incremental save after each object if output file is provided
                if output_file:
                    try:
                        with open(output_file, "w") as f:
                            import json

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

                            json.dump(all_results, f, indent=4, cls=UsdJsonEncoder)

                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {str(e)}")
                        import traceback

                        logging.error(traceback.format_exc())

                success_count += 1
            else:
                logging.warning(f"No material information extracted for {usd_file}")
                failed_objects.append(object_name)
        except Exception as e:
            import traceback

            logging.error(f"Error processing {usd_file}: {str(e)}")
            logging.error(traceback.format_exc())
            failed_objects.append(os.path.basename(os.path.dirname(usd_file)))

    # Log texture statistics
    logging.info("Texture Statistics:")
    logging.info(f"  Total segments processed: {total_segments}")
    logging.info(f"  Segments with textures: {segments_with_texture}")
    logging.info(f"  Segments without textures: {segments_without_texture}")
    logging.info(f"  Segments with thumbnail only: {segments_with_thumbnail_only}")
    logging.info(f"  Total VLM analyses completed: {total_vlm_segments}")

    # Convert materials_per_object to list format for consistency with simready
    materials_per_object_list = []
    for obj_name, count in materials_per_object.items():
        materials_per_object_list.append(obj_name)

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
