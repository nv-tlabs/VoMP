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
import sys
import logging
import os
import json
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.simready import (
    list_simready_objects,
    extract_materials_from_usd as simready_extract_materials,
    get_usd_file_from_prop_dir as simready_get_usd_file,
    PROPS_DIR as SIMREADY_PROPS_DIR,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.residential import (
    list_residential_objects,
    make_user_prompt as residential_make_user_prompt,
    PROMPTS as RESIDENTIAL_PROMPTS,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.commercial import (
    list_commercial_objects,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.vegetation import (
    list_vegetation_objects,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    COMMERCIAL_BASE_DIR,
    RESIDENTIAL_BASE_DIR,
    VEGETATION_BASE_DIR,
    load_material_ranges,
    find_reference_materials,
    parse_numerical_range_str,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.vlm import (
    load_vlm_model,
    analyze_material_with_vlm,
    parse_vlm_properties,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.render import (
    render_sphere_with_texture,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.common import (
    UsdJsonEncoder,
    extract_materials_from_usd as common_extract_materials,
)
import polyscope as ps
import copy
from tqdm import tqdm
import shutil

ps.set_allow_headless_backends(True)  # enable headless backends
ps.init()


def load_existing_results(output_file):
    """
    Load existing results from a JSON file.

    Args:
        output_file (str): Path to the JSON file

    Returns:
        tuple: (results list, set of already processed object names, set of already processed file paths)
    """
    try:
        if (
            os.path.exists(output_file) and os.path.getsize(output_file) > 10
        ):  # Ensure file has some content
            with open(output_file, "r") as f:
                existing_results = json.load(f)

                if not isinstance(existing_results, list):
                    logging.warning(
                        f"Expected list in results file, found {type(existing_results)}"
                    )
                    existing_results = []

                # Create a set of already processed object names for efficient lookup
                processed_objects = set()
                processed_file_paths = set()  # Also track file paths

                for result in existing_results:
                    obj_name = result.get("object_name", "")
                    file_path = result.get("file_path", "")

                    if obj_name:
                        processed_objects.add(obj_name)
                    if file_path:
                        processed_file_paths.add(file_path)

                logging.info(
                    f"Loaded {len(existing_results)} existing results from {output_file}"
                )
                logging.info(
                    f"Found {len(processed_objects)} already processed objects and {len(processed_file_paths)} file paths"
                )

                # Debug - print each processed object
                if processed_objects:
                    logging.info(f"Processed objects: {', '.join(processed_objects)}")

                return existing_results, processed_objects, processed_file_paths
    except Exception as e:
        logging.error(f"Error loading existing results: {str(e)}")

    return [], set(), set()


def append_result_to_file(result, all_results, output_file):
    """
    Append a newly processed result to the output file.

    Args:
        result: The new result to append
        all_results: The full list of results (includes the new result)
        output_file: Path to the output file

    Returns:
        bool: Whether the append was successful
    """
    try:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4, cls=UsdJsonEncoder)

        # Verify file was written successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 10:
            logging.info(
                f"Updated file with new object: {result.get('object_name', 'unknown')}"
            )
            logging.info(
                f"File size: {os.path.getsize(output_file)} bytes, contains {len(all_results)} objects"
            )
            return True
        else:
            logging.error(f"Failed to write updated file - file empty or missing")
            return False
    except Exception as e:
        logging.error(f"Error appending result: {str(e)}")
        import traceback

        logging.error(traceback.format_exc())
        return False


def analyze_material_with_retry(
    segment_key,
    segment_render_path,
    segment_info,
    vlm_model,
    vlm_processor,
    thumb_path,
    dataset_name,
    prompts,
    make_user_prompt_func,
    parse_vlm_output_func,
    custom_prompt=None,
    temp_segment_info=None,
):
    """
    Analyze a material with VLM, with retry logic to ensure we get all necessary values.

    Args:
        segment_key: Key of the segment being analyzed
        segment_render_path: Path to the rendered sphere image
        segment_info: Info about the segment
        vlm_model: VLM model for analysis
        vlm_processor: VLM processor for analysis
        thumb_path: Path to thumbnail image
        dataset_name: Name of dataset
        prompts: Prompt templates to use
        make_user_prompt_func: Function to generate user prompt
        parse_vlm_output_func: Function to parse VLM output
        custom_prompt: Optional custom prompt already generated
        temp_segment_info: Optional temporary segment info

    Returns:
        Segment info with analysis results added
    """
    # First attempt at VLM analysis
    vlm_analysis = analyze_material_with_vlm(
        segment_render_path,
        segment_info if temp_segment_info is None else temp_segment_info,
        vlm_model,
        vlm_processor,
        thumbnail_path=thumb_path,
        dataset_name=dataset_name,
        PROMPTS=prompts,
        make_user_prompt=make_user_prompt_func,
        parse_vlm_output=parse_vlm_output_func,
    )

    # The base prompt to use for retries
    base_prompt = (
        custom_prompt
        if custom_prompt is not None
        else segment_info.get("user_prompt", "")
    )

    # Check if we got successful analysis
    if vlm_analysis:
        # Add VLM analysis to segment info
        segment_info["vlm_analysis"] = vlm_analysis.get("vlm_analysis")

        # Check for missing values
        missing_values = []
        if vlm_analysis.get("youngs_modulus") is not None:
            segment_info["youngs_modulus"] = vlm_analysis.get("youngs_modulus")
        else:
            missing_values.append("Young's modulus")

        if vlm_analysis.get("poissons_ratio") is not None:
            segment_info["poissons_ratio"] = vlm_analysis.get("poissons_ratio")
        else:
            missing_values.append("Poisson's ratio")

        if vlm_analysis.get("density") is not None:
            segment_info["density"] = vlm_analysis.get("density")
        else:
            missing_values.append("density")

        # If any values are missing, retry with stricter instructions
        if missing_values and len(missing_values) > 0:
            logging.warning(
                f"Missing values for {segment_key}: {', '.join(missing_values)}. Retrying with stricter instructions."
            )

            # Create a stronger prompt emphasizing the need for these values
            stricter_prompt = (
                base_prompt
                + "\n\nIMPORTANT: You MUST provide numerical values for Young's modulus, Poisson's ratio, and density. Do not use 'unknown' or any non-numeric values."
            )

            # Make a temporary copy with the stricter prompt
            temp_segment_info_strict = (temp_segment_info or segment_info).copy()
            temp_segment_info_strict["user_prompt"] = stricter_prompt

            # Retry with stricter instructions
            stricter_prompts = copy.deepcopy(prompts)
            stricter_prompts["query_prompt"] = (
                prompts["query_prompt"]
                + "\n\nCRITICAL: You MUST provide NUMERICAL estimates for ALL properties regardless of uncertainty. Make your best estimate even if uncertain."
            )

            vlm_analysis_retry = analyze_material_with_vlm(
                segment_render_path,
                temp_segment_info_strict,
                vlm_model,
                vlm_processor,
                thumbnail_path=thumb_path,
                dataset_name=dataset_name,
                PROMPTS=stricter_prompts,
                make_user_prompt=make_user_prompt_func,
                parse_vlm_output=parse_vlm_output_func,
            )

            if vlm_analysis_retry and "error" not in vlm_analysis_retry:
                # Update missing values from retry
                for missing_value in missing_values:
                    if (
                        missing_value == "Young's modulus"
                        and vlm_analysis_retry.get("youngs_modulus") is not None
                    ):
                        segment_info["youngs_modulus"] = vlm_analysis_retry.get(
                            "youngs_modulus"
                        )
                        logging.info(
                            f"Successfully extracted Young's modulus on retry: {vlm_analysis_retry.get('youngs_modulus')}"
                        )

                    if (
                        missing_value == "Poisson's ratio"
                        and vlm_analysis_retry.get("poissons_ratio") is not None
                    ):
                        segment_info["poissons_ratio"] = vlm_analysis_retry.get(
                            "poissons_ratio"
                        )
                        logging.info(
                            f"Successfully extracted Poisson's ratio on retry: {vlm_analysis_retry.get('poissons_ratio')}"
                        )

                    if (
                        missing_value == "density"
                        and vlm_analysis_retry.get("density") is not None
                    ):
                        segment_info["density"] = vlm_analysis_retry.get("density")
                        logging.info(
                            f"Successfully extracted density on retry: {vlm_analysis_retry.get('density')}"
                        )

                # Add the retry analysis to segment info
                segment_info["vlm_analysis_retry"] = vlm_analysis_retry.get(
                    "vlm_analysis"
                )
            else:
                logging.error(f"VLM retry analysis failed for {segment_key}")

        logging.info(f"VLM analysis successful for {segment_key}:")
        logging.info(
            f"  Young's modulus: {segment_info.get('youngs_modulus', 'Not extracted')}"
        )
        logging.info(
            f"  Poisson's ratio: {segment_info.get('poissons_ratio', 'Not extracted')}"
        )
        logging.info(f"  Density: {segment_info.get('density', 'Not extracted')}")
    else:
        # First attempt failed completely
        logging.error(
            f"VLM analysis failed for {segment_key}: {vlm_analysis.get('error', 'Unknown error')}"
        )

        # Try again with stricter instructions
        logging.warning(
            f"First VLM analysis failed for {segment_key}. Retrying with stricter instructions."
        )

        # Create a stronger prompt emphasizing the need for a successful analysis
        stricter_prompt = (
            base_prompt
            + "\n\nIMPORTANT: You MUST provide numerical values for ALL properties. Make your best estimate even with uncertainty."
        )

        # Make a temporary copy with the stricter prompt
        temp_segment_info_strict = (temp_segment_info or segment_info).copy()
        temp_segment_info_strict["user_prompt"] = stricter_prompt

        # Retry with stricter instructions
        stricter_prompts = copy.deepcopy(prompts)
        stricter_prompts["query_prompt"] = (
            prompts["query_prompt"]
            + "\n\nCRITICAL: You MUST provide NUMERICAL estimates for ALL properties regardless of uncertainty. Make your best estimate even if uncertain."
        )

        vlm_analysis_retry = analyze_material_with_vlm(
            segment_render_path,
            temp_segment_info_strict,
            vlm_model,
            vlm_processor,
            thumbnail_path=thumb_path,
            dataset_name=dataset_name,
            PROMPTS=stricter_prompts,
            make_user_prompt=make_user_prompt_func,
            parse_vlm_output=parse_vlm_output_func,
        )

        if vlm_analysis_retry and "error" not in vlm_analysis_retry:
            # Use the retry values
            segment_info["vlm_analysis"] = vlm_analysis_retry.get("vlm_analysis")

            if vlm_analysis_retry.get("youngs_modulus") is not None:
                segment_info["youngs_modulus"] = vlm_analysis_retry.get(
                    "youngs_modulus"
                )

            if vlm_analysis_retry.get("poissons_ratio") is not None:
                segment_info["poissons_ratio"] = vlm_analysis_retry.get(
                    "poissons_ratio"
                )

            if vlm_analysis_retry.get("density") is not None:
                segment_info["density"] = vlm_analysis_retry.get("density")

            logging.info(f"VLM retry analysis successful for {segment_key}:")
            logging.info(
                f"  Young's modulus: {segment_info.get('youngs_modulus', 'Not extracted')}"
            )
            logging.info(
                f"  Poisson's ratio: {segment_info.get('poissons_ratio', 'Not extracted')}"
            )
            logging.info(f"  Density: {segment_info.get('density', 'Not extracted')}")
        else:
            logging.error(f"VLM retry analysis also failed for {segment_key}")

    # Determine if we have extracted all required properties
    required_props = ["youngs_modulus", "poissons_ratio", "density"]
    have_all_props = all(segment_info.get(p) is not None for p in required_props)

    # If any property is still missing, fall back to reference CSV values
    if not have_all_props:
        try:
            material_type = segment_info.get("material_type", "")

            # Load material database and find closest reference
            material_db = load_material_ranges()
            reference_materials = find_reference_materials(
                material_db, material_type, max_matches=1
            )

            if reference_materials:
                ref = reference_materials[0]

                # Young's modulus (range is in GPa -> convert to Pa)
                if segment_info.get("youngs_modulus") is None:
                    try:
                        y_min, y_max = parse_numerical_range_str(ref["youngs"])
                        ym_gpa = (y_min + y_max) / 2 if y_min != y_max else y_min
                        segment_info["youngs_modulus"] = ym_gpa * 1e9  # Pa
                    except Exception as e:
                        logging.error(
                            f"Error parsing Young's modulus range for fallback: {e}"
                        )

                # Poisson's ratio
                if segment_info.get("poissons_ratio") is None:
                    try:
                        p_min, p_max = parse_numerical_range_str(ref["poisson"])
                        segment_info["poissons_ratio"] = (
                            (p_min + p_max) / 2 if p_min != p_max else p_min
                        )
                    except Exception as e:
                        logging.error(
                            f"Error parsing Poisson's ratio range for fallback: {e}"
                        )

                # Density (already in kg/m^3)
                if segment_info.get("density") is None:
                    try:
                        d_min, d_max = parse_numerical_range_str(ref["density"])
                        segment_info["density"] = (
                            (d_min + d_max) / 2 if d_min != d_max else d_min
                        )
                    except Exception as e:
                        logging.error(f"Error parsing density range for fallback: {e}")

                # Annotate analysis field
                fallback_note = f"FALLBACK: Used reference material '{ref['name']}' values from CSV."
                segment_info["vlm_analysis"] = (
                    segment_info.get("vlm_analysis", "") + "\n" + fallback_note
                ).strip()

                logging.warning(
                    f"{segment_key}: Filled missing properties using reference material '{ref['name']}'."
                )
            else:
                logging.warning(
                    f"{segment_key}: No reference material match found for type '{material_type}'. Unable to fill missing properties."
                )
        except Exception as e:
            logging.error(
                f"{segment_key}: Error while applying reference material fallback – {e}"
            )

    # Final success evaluation after fallback
    have_all_props = all(segment_info.get(p) is not None for p in required_props)

    return segment_info, have_all_props


def process_object(
    usd_file,
    dataset_type,
    object_name,
    vlm_model,
    vlm_processor,
    extract_func,
    all_results,
    output_file,
    processed_objects,
):
    """
    Process a single object and update results.

    Args:
        usd_file: Path to the USD filengc.nvidia.com
        tuple: (success, result_dict, segments_count, rendered_segments_count, vlm_segments_count, unique_materials)
    """
    # Skip if already processed
    if object_name in processed_objects:
        logging.info(f"Skipping already processed object: {object_name}")
        return False, None, 0, 0, 0, set()

    try:
        # Extract material information
        result = extract_func(usd_file, dataset_type)

        if not result or not result.get("segments", {}):
            logging.warning(f"No material information extracted for {usd_file}")
            return False, None, 0, 0, 0, set()

        # Process every segment with VLM
        segments = result.get("segments", {})
        segment_count = len(segments)
        rendered_count = 0
        vlm_count = 0
        unique_materials = set()

        if vlm_model and segment_count > 0:
            # Process VLM for each segment - implementation varies by dataset
            # This would be dataset-specific code...
            pass

        # Add to results
        all_results.append(result)
        processed_objects.add(object_name)

        # Write incremental update
        if output_file:
            append_result_to_file(result, all_results, output_file)

        return True, result, segment_count, rendered_count, vlm_count, unique_materials

    except Exception as e:
        logging.error(f"Error processing {usd_file}: {str(e)}")
        import traceback

        logging.error(traceback.format_exc())
        return False, None, 0, 0, 0, set()


def process_dataset(
    dataset_type,
    vlm_model,
    vlm_processor,
    limit=None,
    output_file=None,
    dry_run=False,
    force_reprocess=False,
    existing_results=None,
    processed_objects=None,
    processed_file_paths=None,
):
    """
    Process all objects in a dataset.

    Args:
        dataset_type: Type of dataset (simready, residential, commercial, vegetation)
        vlm_model: VLM model for analysis
        vlm_processor: VLM processor for analysis
        limit: Maximum number of objects to process
        output_file: Path to output file for incremental updates
        dry_run: If True, only process a few objects
        force_reprocess: If True, reprocess objects even if they exist in output_file
        existing_results: List of existing results to start with
        processed_objects: Set of already processed object names
        processed_file_paths: Set of already processed file paths

    Returns:
        Tuple of results and statistics
    """
    # Initialize tracking sets and results
    processed_objects = set() if processed_objects is None else processed_objects
    processed_file_paths = (
        set() if processed_file_paths is None else processed_file_paths
    )
    existing_results = [] if existing_results is None else existing_results

    # Build a set of already processed object names from existing_results
    existing_object_names = {
        result.get("object_name")
        for result in existing_results
        if "object_name" in result
    }
    existing_file_paths = {
        result.get("file_path") for result in existing_results if "file_path" in result
    }

    logging.info(
        f"Found {len(existing_object_names)} already processed objects in existing results"
    )
    logging.info(
        f"Found {len(existing_file_paths)} already processed file paths in existing results"
    )

    # If not forcing reprocess, add names from existing_results to processed_objects
    if not force_reprocess:
        processed_objects.update(existing_object_names)
        processed_file_paths.update(existing_file_paths)

    # Create a copy of existing_results to avoid modifying the original
    all_results = copy.deepcopy(existing_results)

    # Get the list of objects to process based on dataset type
    objects_to_process = []

    if dataset_type == "simready":
        # For SimReady, get the list of prop directories
        if not os.path.isdir(SIMREADY_PROPS_DIR):
            logging.error(f"SimReady props directory not found at {SIMREADY_PROPS_DIR}")
            return all_results, 0, 0, [], 0, 0, 0, [], {}

        objects_to_process = [
            d
            for d in os.listdir(SIMREADY_PROPS_DIR)
            if os.path.isdir(os.path.join(SIMREADY_PROPS_DIR, d))
        ]
        objects_to_process.sort()
    elif dataset_type == "residential":
        # For Residential, get the list of USD files
        objects_to_process = []
        for root, _, files in os.walk(RESIDENTIAL_BASE_DIR):
            for file in files:
                if file.endswith(".usd") and not os.path.basename(root).startswith("."):
                    objects_to_process.append(
                        (
                            os.path.join(root, file),
                            os.path.basename(os.path.dirname(os.path.join(root, file))),
                        )
                    )
    elif dataset_type == "commercial":
        # For Commercial, get the list of USD files
        objects_to_process = []
        for root, _, files in os.walk(COMMERCIAL_BASE_DIR):
            for file in files:
                if file.endswith(".usd") and not os.path.basename(root).startswith("."):
                    objects_to_process.append(
                        (
                            os.path.join(root, file),
                            os.path.basename(os.path.dirname(os.path.join(root, file))),
                        )
                    )
    elif dataset_type == "vegetation":
        # For Vegetation, get the list of USD files
        objects_to_process = []
        for root, _, files in os.walk(VEGETATION_BASE_DIR):
            for file in files:
                if file.endswith(".usd") and not os.path.basename(root).startswith("."):
                    objects_to_process.append(
                        (
                            os.path.join(root, file),
                            os.path.basename(os.path.dirname(os.path.join(root, file))),
                        )
                    )
    else:
        logging.error(f"Unknown dataset type: {dataset_type}")
        return all_results, 0, 0, [], 0, 0, 0, [], {}

    # Limit the number of objects to process if specified
    if limit and limit > 0:
        if isinstance(objects_to_process[0], tuple):
            # For non-SimReady datasets, objects_to_process contains tuples of (path, name)
            logging.info(
                f"Limiting to {limit} objects out of {len(objects_to_process)}"
            )
            objects_to_process = objects_to_process[:limit]
        else:
            # For SimReady, objects_to_process contains just names
            logging.info(
                f"Limiting to {limit} objects out of {len(objects_to_process)}"
            )
            objects_to_process = objects_to_process[:limit]

    # Filter out already processed objects before we even start
    objects_to_process_filtered = []
    for obj_info in objects_to_process:
        if dataset_type == "simready":
            # For SimReady, obj_info is the prop name
            prop_name = obj_info
            object_name = prop_name

            # Skip if already processed and not forcing reprocessing
            if object_name in processed_objects and not force_reprocess:
                logging.info(f"Filtering out already processed object: {object_name}")
                continue
        else:
            # For other datasets, obj_info is a tuple of (path, name)
            usd_file, object_name = obj_info

            # Skip if already processed by file path or name and not forcing reprocessing
            if usd_file in processed_file_paths and not force_reprocess:
                logging.info(f"Filtering out already processed file: {usd_file}")
                continue

            if object_name in processed_objects and not force_reprocess:
                logging.info(f"Filtering out already processed object: {object_name}")
                continue

        # If we get here, add the object to the filtered list
        objects_to_process_filtered.append(obj_info)

    # Update the objects to process list
    logging.info(
        f"Filtered {len(objects_to_process) - len(objects_to_process_filtered)} already processed objects"
    )
    logging.info(f"Processing {len(objects_to_process_filtered)} remaining objects")
    objects_to_process = objects_to_process_filtered

    # Initialize statistics
    success_count = 0
    failed_objects = []
    total_segments = 0
    total_rendered_segments = 0
    total_vlm_segments = 0
    unique_materials = set()
    materials_per_object = {}

    # Count total segments from existing results
    for result in existing_results:
        segments = result.get("segments", {})
        total_segments += len(segments)

        # Count unique materials
        for segment_key, segment_info in segments.items():
            material_name = segment_info.get("name", "")
            if material_name:
                unique_materials.add(material_name)

    # Statistics for texture availability
    segments_with_texture = 0
    segments_without_texture = 0
    segments_with_thumbnail_only = 0
    segments_text_only_vlm = 0  # New: segments with no images but VLM analysis

    # Track processed files to avoid duplicates
    processed_files = set()

    # Process each object
    for obj_idx, obj_info in enumerate(
        tqdm(objects_to_process, desc=f"Processing {dataset_type} dataset")
    ):
        if dataset_type == "simready":
            # For SimReady, obj_info is the prop name
            prop_name = obj_info
            object_name = prop_name

            try:
                # Get the full prop directory path
                full_prop_dir = os.path.join(SIMREADY_PROPS_DIR, prop_name)

                if not os.path.isdir(full_prop_dir):
                    logging.error(f"Prop directory not found at {full_prop_dir}")
                    failed_objects.append(prop_name)
                    continue

                # Find a USD file in the prop directory
                try:
                    usd_file = simready_get_usd_file(full_prop_dir)
                    logging.info(
                        f"Found USD file for {prop_name}: {os.path.basename(usd_file)}"
                    )
                except:
                    logging.error(f"Could not find USD file for {prop_name}")
                    failed_objects.append(prop_name)
                    continue

                # Extract material information
                materials_dict = simready_extract_materials(
                    usd_file, prop_name, full_prop_dir
                )

                # Ensure file_path is properly stored
                if "file_path" not in materials_dict or not materials_dict["file_path"]:
                    materials_dict["file_path"] = usd_file

                # Add to tracking
                processed_file_paths.add(usd_file)

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
                    materials_per_object[prop_name] = len(prop_materials)

                # Determine thumbnail path from SimReady structure
                thumb_path = os.path.join(
                    full_prop_dir,
                    ".thumbs",
                    "256x256",
                    f"{prop_name}.usd.png",
                )
                has_thumbnail = os.path.exists(thumb_path)

                if not has_thumbnail:
                    logging.warning(
                        f"No thumbnail found for {prop_name} at {thumb_path}"
                    )
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
                                segments_with_texture += 1
                                logging.info(
                                    f"Rendering texture sphere for {prop_name}, segment {segment_key}"
                                )

                                # Set up file path for this segment's rendered sphere
                                segment_render_path = f"/tmp/vlm/texture_sphere_{prop_name}_{segment_key}.png"

                                try:
                                    rgb_buffer = render_sphere_with_texture(
                                        textures, segment_render_path
                                    )
                                    logging.info(
                                        f"RGB buffer shape: {rgb_buffer.shape}"
                                    )
                                except Exception as e:
                                    logging.error(
                                        f"Error rendering texture for {segment_key}: {str(e)}"
                                    )
                                    segment_render_path = None
                            else:
                                # No albedo texture - just use thumbnail
                                segments_without_texture += 1
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

                                # Run VLM even if we have neither texture nor thumbnail (text-only analysis)
                                if not segment_render_path and not has_thumbnail:
                                    segments_text_only_vlm += 1
                                    logging.info(
                                        f"Running text-only VLM analysis for {segment_key} - no texture or thumbnail available"
                                    )

                                # Using the make_user_prompt from simready.py
                                from dataset_toolkits.material_objects.vlm_annotations.data_subsets.simready import (
                                    make_user_prompt as simready_make_user_prompt,
                                    PROMPTS as SIMREADY_PROMPTS,
                                    parse_vlm_output as simready_parse_vlm,
                                )

                                part1 = simready_make_user_prompt(
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
                                elif has_thumbnail:
                                    logging.info(
                                        f"Using prompt WITH thumbnail only for {prop_name}, segment {segment_key}"
                                    )
                                else:
                                    logging.info(
                                        f"Using TEXT-ONLY prompt for {prop_name}, segment {segment_key}"
                                    )
                                    logging.info(
                                        f"PROMPT: {part1[:100]}..."
                                    )  # Print just the beginning of the prompt

                                segment_info, vlm_analysis_success = (
                                    analyze_material_with_retry(
                                        segment_key,
                                        segment_render_path,
                                        segment_info,
                                        vlm_model,
                                        vlm_processor,
                                        thumb_path if has_thumbnail else None,
                                        "simready",
                                        SIMREADY_PROMPTS,
                                        simready_make_user_prompt,
                                        simready_parse_vlm,
                                        part1,
                                        None,
                                    )
                                )

                                # Add VLM analysis to segment info
                                if vlm_analysis_success:
                                    segment_info["vlm_analysis"] = segment_info[
                                        "vlm_analysis"
                                    ]
                                    total_vlm_segments += 1
                                    logging.info(
                                        f"VLM analysis successful for {segment_key}:"
                                    )
                                    logging.info(
                                        f"  Young's modulus: {segment_info.get('youngs_modulus', 'Not extracted')}"
                                    )
                                    logging.info(
                                        f"  Poisson's ratio: {segment_info.get('poissons_ratio', 'Not extracted')}"
                                    )
                                    logging.info(
                                        f"  Density: {segment_info.get('density', 'Not extracted')}"
                                    )
                                else:
                                    logging.error(
                                        f"VLM analysis failed for {segment_key}"
                                    )
                            except Exception as e:
                                import traceback

                                logging.error(
                                    f"Error during VLM analysis for {segment_key}: {str(e)}"
                                )
                                logging.error(traceback.format_exc())

                            total_rendered_segments += 1

                    all_results.append(materials_dict)
                    processed_objects.add(object_name)
                    success_count += 1

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
                                        for seg_key, seg_info in item[
                                            "segments"
                                        ].items():
                                            # Remove user_prompt field if it exists
                                            if "user_prompt" in seg_info:
                                                del seg_info["user_prompt"]

                                            if "textures" in seg_info and isinstance(
                                                seg_info["textures"], dict
                                            ):
                                                # Convert any non-serializable texture paths to strings
                                                serializable_textures = {}
                                                for tex_type, tex_path in seg_info[
                                                    "textures"
                                                ].items():
                                                    serializable_textures[tex_type] = (
                                                        str(tex_path)
                                                    )
                                                seg_info["textures"] = (
                                                    serializable_textures
                                                )

                                json.dump(all_results, f, indent=4, cls=UsdJsonEncoder)

                        except Exception as e:
                            logging.error(f"Error saving checkpoint: {str(e)}")
                            import traceback

                            logging.error(traceback.format_exc())
                else:
                    logging.warning(f"No segments extracted for {prop_name}")
                    failed_objects.append(prop_name)
            except Exception as e:
                import traceback

                logging.error(f"Error processing {prop_name}: {str(e)}")
                logging.error(traceback.format_exc())
                failed_objects.append(prop_name)
        else:
            # For other datasets, obj_info is a tuple of (path, name)
            usd_file, object_name = obj_info

            try:
                directory = os.path.dirname(usd_file)

                # Extract material information
                result = common_extract_materials(usd_file, dataset_type)

                if result:
                    # Ensure file_path is stored in the result to improve future matching
                    if "file_path" not in result or not result["file_path"]:
                        result["file_path"] = usd_file

                    # Add to processed tracking
                    processed_file_paths.add(usd_file)
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
                    # Thumbnails are in .thumbs/256x256 directory
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
                                    logging.info(
                                        f"RGB buffer shape: {rgb_buffer.shape}"
                                    )
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

                                # Run VLM even if we have neither texture nor thumbnail (text-only analysis)
                                if not segment_render_path and not has_thumbnail:
                                    segments_text_only_vlm += 1
                                    logging.info(
                                        f"Running text-only VLM analysis for {segment_key} - no texture or thumbnail available"
                                    )

                                # Set semantic usage to segment name but don't store in segment data
                                semantic_usage = segment_key
                                temp_object_name = object_name

                                # Create custom prompt based on texture availability
                                custom_prompt = residential_make_user_prompt(
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
                                elif has_thumbnail:
                                    logging.info(
                                        f"Using prompt WITH thumbnail only for {object_name}, segment {segment_key}"
                                    )
                                else:
                                    logging.info(
                                        f"Using TEXT-ONLY prompt for {object_name}, segment {segment_key}"
                                    )
                                    logging.info(
                                        f"PROMPT: {custom_prompt[:100]}..."
                                    )  # Print just the beginning of the prompt

                                # Create a temporary segment_info with object_name for VLM but don't save to result
                                temp_segment_info = segment_info.copy()
                                temp_segment_info["semantic_usage"] = semantic_usage
                                temp_segment_info["object_name"] = temp_object_name

                                segment_info, vlm_analysis_success = (
                                    analyze_material_with_retry(
                                        segment_key,
                                        segment_render_path,
                                        segment_info,
                                        vlm_model,
                                        vlm_processor,
                                        thumb_path if has_thumbnail else None,
                                        dataset_type,
                                        RESIDENTIAL_PROMPTS,
                                        residential_make_user_prompt,
                                        parse_vlm_properties,
                                        custom_prompt,
                                        temp_segment_info,
                                    )
                                )

                                # Add VLM analysis to segment info
                                if vlm_analysis_success:
                                    segment_info["vlm_analysis"] = segment_info[
                                        "vlm_analysis"
                                    ]
                                    total_vlm_segments += 1
                                    logging.info(
                                        f"VLM analysis successful for {segment_key}:"
                                    )
                                    logging.info(
                                        f"  Young's modulus: {segment_info.get('youngs_modulus', 'Not extracted')}"
                                    )
                                    logging.info(
                                        f"  Poisson's ratio: {segment_info.get('poissons_ratio', 'Not extracted')}"
                                    )
                                    logging.info(
                                        f"  Density: {segment_info.get('density', 'Not extracted')}"
                                    )
                                else:
                                    logging.error(
                                        f"VLM analysis failed for {segment_key}"
                                    )
                            except Exception as e:
                                import traceback

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
                                        for seg_key, seg_info in item[
                                            "segments"
                                        ].items():
                                            # Remove user_prompt field if it exists
                                            if "user_prompt" in seg_info:
                                                del seg_info["user_prompt"]

                                            if "textures" in seg_info and isinstance(
                                                seg_info["textures"], dict
                                            ):
                                                # Convert any non-serializable texture paths to strings
                                                serializable_textures = {}
                                                for tex_type, tex_path in seg_info[
                                                    "textures"
                                                ].items():
                                                    serializable_textures[tex_type] = (
                                                        str(tex_path)
                                                    )
                                                seg_info["textures"] = (
                                                    serializable_textures
                                                )

                                if dataset_type in ["residential", "vegetation"]:
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
                                else:
                                    # Dump to file for other datasets
                                    json.dump(
                                        all_results, f, indent=4, cls=UsdJsonEncoder
                                    )

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
                failed_objects.append(object_name)

    # Log texture statistics
    logging.info("Texture Statistics:")
    logging.info(f"  Total segments processed: {total_segments}")
    logging.info(f"  Segments with textures: {segments_with_texture}")
    logging.info(f"  Segments without textures: {segments_without_texture}")
    logging.info(f"  Segments with thumbnail only: {segments_with_thumbnail_only}")
    logging.info(f"  Segments with text-only VLM: {segments_text_only_vlm}")
    logging.info(f"  Total VLM analyses completed: {total_vlm_segments}")

    return (
        all_results,
        len(objects_to_process),
        success_count,
        failed_objects,
        total_segments,
        total_rendered_segments,
        total_vlm_segments,
        list(unique_materials),
        materials_per_object,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract material information from datasets"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["simready", "residential", "commercial", "vegetation"],
        help="Name of the dataset to process",
    )
    parser.add_argument("--list", action="store_true", help="List all available props")
    parser.add_argument("--dry-run", action="store_true", help="Dry run the script")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging to console"
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON file to save combined material information"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of objects even if they exist in the output file",
    )
    parser.add_argument(
        "--model",
        default="qwen",
        help="Model to use for VLM analysis. Options: 'qwen' (default), 'gemini-2.0-flash', 'gemini-2.5-pro', etc.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for Gemini models. Required when using Gemini models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of objects to process per dataset. Useful for testing.",
    )
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Validate model and API key
    if args.model.startswith("gemini") and not args.api_key:
        logging.error(
            "API key is required when using Gemini models. Use --api-key to provide it."
        )
        sys.exit(1)

    if args.list:
        for dataset in args.dataset:
            if dataset == "simready":
                print(f"Listing SimReady objects")
                list_simready_objects()
            elif dataset == "residential":
                print(f"Listing Residential objects")
                list_residential_objects()
            elif dataset == "commercial":
                print(f"Listing Commercial objects")
                list_commercial_objects()
            elif dataset == "vegetation":
                print(f"Listing Vegetation objects")
                list_vegetation_objects()
        sys.exit()

    # Load existing results if output file is specified
    all_results = []
    processed_objects = set()
    processed_file_paths = set()
    if args.output and not args.force_reprocess:
        all_results, processed_objects, processed_file_paths = load_existing_results(
            args.output
        )

    print("Loading VLM model...")
    vlm_model, vlm_processor = load_vlm_model(args.model, args.api_key)

    # Initialize statistics tracking dictionaries
    stats_dict = {
        "num_objects": {},
        "success_count": {},
        "failed_props": {},
        "total_segments": {},
        "total_rendered_segments": {},
        "total_vlm_segments": {},
        "unique_materials": set(),
        "materials_per_prop": {},
    }

    # Process each dataset with the shared all_results and processed_objects
    for dataset in args.dataset:
        print(f"Processing {dataset} objects...")

        (
            all_results,
            dataset_num_objects,
            dataset_success_count,
            dataset_failed_props,
            dataset_total_segments,
            dataset_total_rendered_segments,
            dataset_total_vlm_segments,
            dataset_unique_materials,
            dataset_materials_per_prop,
        ) = process_dataset(
            dataset,
            vlm_model,
            vlm_processor,
            limit=args.limit if args.limit else (2 if args.dry_run else None),
            output_file=args.output,
            dry_run=args.dry_run,
            force_reprocess=args.force_reprocess,
            existing_results=all_results,
            processed_objects=processed_objects,
            processed_file_paths=processed_file_paths,
        )

        # Update stats
        stats_dict["num_objects"][dataset] = dataset_num_objects
        stats_dict["success_count"][dataset] = dataset_success_count
        stats_dict["failed_props"][dataset] = dataset_failed_props
        stats_dict["total_segments"][dataset] = dataset_total_segments
        stats_dict["total_rendered_segments"][dataset] = dataset_total_rendered_segments
        stats_dict["total_vlm_segments"][dataset] = dataset_total_vlm_segments
        stats_dict["unique_materials"].update(dataset_unique_materials)
        stats_dict["materials_per_prop"][dataset] = dataset_materials_per_prop

    # Final logging
    if args.verbose:
        # Calculate total objects processed across all datasets
        total_objects = sum(stats_dict["num_objects"].values())
        print(f"Total objects processed: {total_objects}")

        # Print dataset-specific stats
        processed_datasets = []
        for dataset in args.dataset:
            if dataset in stats_dict["num_objects"]:
                dataset_upper = dataset.upper()
                processed_datasets.append(dataset_upper)
                print(f"    {dataset_upper}: {stats_dict['num_objects'][dataset]}")

        # Calculate success percentages
        total_success = sum(stats_dict["success_count"].values())
        if total_objects > 0:
            total_success_pct = total_success / total_objects * 100
        else:
            total_success_pct = 0
        print(f"Successfully processed: {total_success} ({total_success_pct:.1f}%)")

        # Print success percentage per dataset
        for dataset in processed_datasets:
            dataset_lower = dataset.lower()
            if (
                dataset_lower in stats_dict["success_count"]
                and dataset_lower in stats_dict["num_objects"]
                and stats_dict["num_objects"][dataset_lower] > 0
            ):
                success_pct = (
                    stats_dict["success_count"][dataset_lower]
                    / stats_dict["num_objects"][dataset_lower]
                    * 100
                )
                print(f"    {dataset}: {success_pct:.1f}%")
            elif dataset_lower in processed_datasets:
                print(f"    {dataset}: 0.0%")

        # Print segment stats
        print(f"Total segments: {sum(stats_dict['total_segments'].values())}")
        print(
            f"Total rendered segments: {sum(stats_dict['total_rendered_segments'].values())}"
        )
        print(f"Total VLM segments: {sum(stats_dict['total_vlm_segments'].values())}")
        print(f"Unique materials: {len(stats_dict['unique_materials'])}")

        # Cleanup
        if os.path.exists("/tmp/vlm"):
            shutil.rmtree("/tmp/vlm")
