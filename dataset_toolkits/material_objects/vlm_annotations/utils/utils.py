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

import torch
import numpy as np
import os
import csv
import difflib
import logging
from qwen_vl_utils import process_vision_info
from typing import Tuple, List, Dict
from pathlib import Path

from dataset_toolkits.datasets.simready import (
    get_asset_class_mapping as get_simready_asset_class_mapping,
)

# Resolve path to material_ranges.csv relative to this file's location
UTILS_FILE_PATH = Path(__file__).resolve()
# utils.py is in dataset_toolkits/material_objects/vlm_annotations/utils/
# WORKSPACE_ROOT is 4 levels up from 'utils'
WORKSPACE_ROOT = UTILS_FILE_PATH.parents[4]
MATERIAL_RANGES_CSV_ABSOLUTE = (
    WORKSPACE_ROOT / "datasets" / "latent_space" / "material_ranges_old.csv"
)

# Keep the old constant for now if anything else relies on it by name, but new logic uses _ABSOLUTE
MATERIAL_RANGES_CSV = "datasets/latent_space/material_ranges_old.csv"

BASE_DIR = "datasets/raw"

SIMREADY_BASE_DIR = os.path.join(BASE_DIR, "simready")
SIMREADY_PROPS_DIR = os.path.join(SIMREADY_BASE_DIR, "common_assets", "props")
SIMREADY_MATERIALS_DIR = os.path.join(SIMREADY_BASE_DIR, "materials", "physics")
SIMREADY_ASSET_INFO_PATH = os.path.join(SIMREADY_BASE_DIR, "asset_info.json")
SIMREADY_ASSET_CLASS_MAPPING = get_simready_asset_class_mapping(
    SIMREADY_ASSET_INFO_PATH
)

RESIDENTIAL_BASE_DIR = os.path.join(BASE_DIR, "residential")

COMMERCIAL_BASE_DIR = os.path.join(BASE_DIR, "commercial")

VEGETATION_BASE_DIR = os.path.join(BASE_DIR, "vegetation")


def set_seed(seed: int = 42) -> None:
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def load_material_ranges(csv_path=None):
    if csv_path is None:
        csv_path = MATERIAL_RANGES_CSV_ABSOLUTE

    # Ensure csv_path is a string for os.path.exists if it's a Path object
    csv_path_str = str(csv_path)

    if not os.path.exists(csv_path_str):
        logging.warning(f"Material ranges CSV file not found at {csv_path_str}")
        return []

    material_ranges = []

    with open(csv_path_str, "r", newline="", encoding="utf-8") as f:
        # Read the first line to get/clean the header
        header_line = f.readline().strip()
        if header_line.startswith('"') and header_line.endswith('"'):
            header_line = header_line[
                1:-1
            ]  # Remove surrounding quotes from the whole line

        fieldnames = [name.strip() for name in header_line.split(",")]

        # Use the cleaned fieldnames for DictReader
        # The rest of the file (f) is now positioned after the header line
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for i, row in enumerate(reader):
            try:
                # Normalize keys for easier usage
                material_ranges.append(
                    {
                        "name": row["Material Name"].strip(),
                        "youngs": row["Young's Modulus Range [GPa]"].strip(),
                        "poisson": row["Poisson's Ratio Range"].strip(),
                        "density": row["Density Range"].strip(),
                    }
                )
            except KeyError as e:
                logging.error(
                    f"KeyError processing row {i+1} in {csv_path_str}: {e}. Row data: {row}. Expected headers: {fieldnames}"
                )
                # Optionally skip this row or raise the error
                continue
            except Exception as e:
                logging.error(
                    f"Error processing row {i+1} in {csv_path_str}: {e}. Row data: {row}"
                )
                continue
    return material_ranges


def find_reference_materials(material_db, material_query, max_matches=3):
    """Return a list of reference materials whose names fuzzy-match the query."""

    material_query = (material_query or "").lower()
    if not material_query:
        return []

    names = [m["name"] for m in material_db]

    close_names = difflib.get_close_matches(
        material_query, names, n=max_matches, cutoff=0.4
    )

    refs = [m for m in material_db if m["name"] in close_names]
    if not refs:
        refs = [m for m in material_db if material_query in m["name"].lower()][
            :max_matches
        ]
    return refs


def round_float_to_2dp(value):
    if isinstance(value, float):
        return round(value, 2)
    return value


def parse_numerical_range_str(range_str: str) -> Tuple[float, float]:
    """
    Parses a string that can be a single number or a range in brackets.
    Examples: "2.5", "[2.5]", "[2.5, 3.5]"
    Returns a tuple (min_val, max_val).
    For a single number, min_val will be equal to max_val.
    """
    cleaned_str = str(range_str).strip()  # Ensure input is string
    if cleaned_str.startswith("[") and cleaned_str.endswith("]"):
        cleaned_str = cleaned_str[1:-1].strip()

    if not cleaned_str:  # Handle empty string or "[]"
        raise ValueError(
            f"Input string for range parsing is empty or invalid: '{range_str}'"
        )

    parts = [part.strip() for part in cleaned_str.split(",")]

    try:
        if len(parts) == 1:
            if not parts[0]:  # Handles "[]" which becomes ""
                raise ValueError(
                    f"Input string for range parsing is empty or invalid after stripping brackets: '{range_str}'"
                )
            val = float(parts[0])
            return val, val
        elif len(parts) == 2:
            low = float(parts[0])
            high = float(parts[1])
            # Optional: ensure low <= high.
            # if low > high:
            #     # Consider logging a warning or raising an error if min > max
            #     return high, low # Or handle as error
            return low, high
        else:
            raise ValueError(
                f"Invalid range string format: '{range_str}'. Expected 1 or 2 parts after splitting by comma, got {len(parts)}."
            )
    except ValueError as e:
        # Re-raise with more context
        raise ValueError(
            f"Error parsing numerical range from string '{range_str}': {e}"
        )
