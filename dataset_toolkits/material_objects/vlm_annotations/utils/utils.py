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
import re
import unicodedata
from typing import Tuple, List, Dict
from pathlib import Path

from dataset_toolkits.datasets.simready import (
    get_asset_class_mapping as get_simready_asset_class_mapping,
)

# Resolve path to material_ranges.csv relative to this file's location
UTILS_FILE_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = UTILS_FILE_PATH.parents[4]
MATERIAL_RANGES_CSV = (
    WORKSPACE_ROOT / "datasets" / "latent_space" / "material_ranges.csv"
)

BASE_DIR = str(WORKSPACE_ROOT / "datasets" / "raw")

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
        csv_path = MATERIAL_RANGES_CSV

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

        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            material_ranges.append(
                {
                    "name": row["Material Name"].strip(),
                    "youngs": row["Young's Modulus Range [GPa]"].strip(),
                    "poisson": row["Poisson's Ratio Range"].strip(),
                    "density": row["Density Range"].strip(),
                }
            )
    return material_ranges


_CAMEL_WORD_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"
    r"|(?<=[A-Z])(?=[A-Z][a-z])"
    r"|(?<=[A-Za-z])(?=[0-9])"
    r"|(?<=[0-9])(?=[A-Za-z])"
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

_MATERIAL_LABEL_WRAPPERS = frozenset(
    {"m", "mat", "material", "omni", "pbr", "surface"}
)

_EXACT_REFERENCE_ALIASES = {
    "abs": "abs acrylonitrile butadiene styrene",
    "aluminum": "aluminium",
    "aluminum alloy": "aluminium alloy",
    "clear glass": "glass",
    "dark glass": "glass",
    "eps": "polystyrene foam eps",
    "frosted glass": "glass",
    "hard rubber": "rubber hard",
    "hdpe": "high density polyethylene hdpe",
    "ldpe": "low density polyethylene ldpe",
    "mdf": "medium density fiberboard mdf",
    "pc": "polycarbonate pc",
    "peek": "peek polyetheretherketone",
    "pet": "polyethylene terephthalate pet",
    "pla": "pla polylactic acid",
    "plastic": "plastics",
    "plexiglas": "plexiglass",
    "pmma": "pmma acrylic",
    "pom": "acetal pom",
    "pp": "polypropylene pp",
    "ptfe": "ptfe teflon",
    "pvc": "polyvinyl chloride pvc",
    "rough glass": "glass",
    "soft rubber": "rubber soft",
    "stainless": "stainless steel",
    "tinted glass": "glass",
    "pu foam": "polyurethane foam",
}

_MATERIAL_TOKEN_ALIASES = {
    "aluminium alloy": "aluminium alloy",
    "aluminum alloy": "aluminium alloy",
    "beryllium copper": "beryllium copper",
    "carbon steel": "carbon steel",
    "cast iron": "cast iron",
    "copper beryllium": "copper beryllium",
    "epoxy resin": "epoxy resin",
    "float glass": "float glass",
    "mild steel": "mild steel",
    "stainless steel": "stainless steel",
    "steel stainless": "stainless steel",
    "aluminium": "aluminium",
    "aluminum": "aluminium",
    "acrylic": "acrylic",
    "basalt": "basalt",
    "concrete": "concrete",
    "cork": "cork",
    "granite": "granite",
    "marble": "marble",
    "neoprene": "neoprene",
    "nylon": "nylon",
    "polycarbonate": "polycarbonate",
    "polyethylene": "polyethylene",
    "polypropylene": "polypropylene",
    "polystyrene": "polystyrene",
    "stainless": "stainless steel",
    "steel": "steel",
}

_DISABLED_GENERIC_REFERENCE_NAMES = frozenset({"rubber"})

def _normalize_material_name(value):
    """Normalize a material label without using approximate spelling."""

    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = _CAMEL_WORD_BOUNDARY.sub(" ", text).casefold()
    return " ".join(_NON_ALPHANUMERIC.sub(" ", text).split())


def _without_material_label_wrappers(normalized_query):
    return " ".join(
        token
        for token in normalized_query.split()
        if token not in _MATERIAL_LABEL_WRAPPERS
    )


def _contains_token_phrase(query_tokens, phrase_tokens):
    phrase_length = len(phrase_tokens)
    if not phrase_length or phrase_length > len(query_tokens):
        return False
    return any(
        query_tokens[start : start + phrase_length] == phrase_tokens
        for start in range(len(query_tokens) - phrase_length + 1)
    )


def _reference_range_bounds(value):
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    parts = [part.strip() for part in text.split(",")]
    if len(parts) == 1:
        lower = upper = float(parts[0])
    elif len(parts) == 2:
        lower, upper = (float(part) for part in parts)
    else:
        raise ValueError(f"invalid reference range: {value!r}")
    return (lower, upper) if lower <= upper else (upper, lower)


def _format_reference_range(lower, upper):
    low_text = f"{lower:.15g}"
    high_text = f"{upper:.15g}"
    return low_text if lower == upper else f"[{low_text}, {high_text}]"


def _merge_duplicate_reference_rows(rows):
    """Combine duplicate CSV names into a conservative numeric envelope."""

    if not rows:
        return None
    merged = dict(rows[0])
    if len(rows) == 1:
        return merged
    for field in ("youngs", "poisson", "density"):
        bounds = [_reference_range_bounds(row[field]) for row in rows]
        merged[field] = _format_reference_range(
            min(lower for lower, _ in bounds),
            max(upper for _, upper in bounds),
        )
    return merged


def find_reference_material_match(material_db, material_query):
    """Return one conservative reference for a material-name hint.

    Matching proceeds from exact normalized CSV names, to explicit aliases,
    then to an allowlist of unambiguous material words embedded in authored
    material labels.  It intentionally performs no edit-distance matching:
    returning no reference is safer than attaching an unrelated physical
    property range to a segment.
    """

    normalized_query = _normalize_material_name(material_query)
    if not normalized_query:
        return None

    by_normalized_name = {}
    for material in material_db or []:
        if not isinstance(material, dict):
            continue
        normalized_name = _normalize_material_name(material.get("name"))
        if normalized_name:
            by_normalized_name.setdefault(normalized_name, []).append(material)

    def resolve(normalized_name):
        if normalized_name in _DISABLED_GENERIC_REFERENCE_NAMES:
            return None
        rows = by_normalized_name.get(normalized_name)
        return _merge_duplicate_reference_rows(rows) if rows else None

    # The complete authored label names a CSV material exactly.
    match = resolve(normalized_query)
    if match:
        return match

    unwrapped_query = _without_material_label_wrappers(normalized_query)
    if unwrapped_query != normalized_query:
        match = resolve(unwrapped_query)
        if match:
            return match

    for candidate in (normalized_query, unwrapped_query):
        alias_target = _EXACT_REFERENCE_ALIASES.get(candidate)
        if alias_target:
            match = resolve(alias_target)
            if match:
                return match

    query_tokens = unwrapped_query.split()
    token_matches = [
        (alias, target)
        for alias, target in _MATERIAL_TOKEN_ALIASES.items()
        if _contains_token_phrase(query_tokens, alias.split())
    ]
    if not token_matches:
        return None

    longest = max(len(alias.split()) for alias, _ in token_matches)
    strongest = sorted(
        (alias, target)
        for alias, target in token_matches
        if len(alias.split()) == longest
    )
    targets = {target for _, target in strongest}
    if len(targets) != 1:
        return None
    return resolve(strongest[0][1])


def find_reference_materials(material_db, material_query, max_matches=3):
    """Return reference materials using the original voxelization matcher."""

    material_query = (material_query or "").lower()
    if not material_query:
        return []
    names = [material["name"] for material in material_db]
    close_names = difflib.get_close_matches(
        material_query, names, n=max_matches, cutoff=0.4
    )
    matches = [material for material in material_db if material["name"] in close_names]
    if matches:
        return matches
    return [
        material
        for material in material_db
        if material_query in material["name"].lower()
    ][:max_matches]


def round_float_to_2dp(value):
    if isinstance(value, float):
        return round(value, 2)
    return value


def parse_numerical_range_str(range_str: str) -> Tuple[float, float]:
    """Parse a single number or a two-number bracketed range."""

    cleaned_str = str(range_str).strip()
    if cleaned_str.startswith("[") and cleaned_str.endswith("]"):
        cleaned_str = cleaned_str[1:-1].strip()
    parts = [part.strip() for part in cleaned_str.split(",")]
    if len(parts) not in {1, 2} or not all(parts):
        raise ValueError(f"Invalid numerical range: {range_str!r}")
    values = tuple(float(part) for part in parts)
    return (values[0], values[0]) if len(values) == 1 else values


_VLM_MATERIAL_RANGES = load_material_ranges()


def format_reference_material_hint(material_query):
    """Return a short optional range for a confidently matched material name."""

    material = find_reference_material_match(_VLM_MATERIAL_RANGES, material_query)
    if material is None:
        return ""

    def display_range(value):
        lower, upper = parse_numerical_range_str(value)
        return f"{lower:g}" if lower == upper else f"{lower:g}-{upper:g}"

    return (
        "\nPossible reference values, only if they agree with the images and "
        "selected part:\n"
        f"- {material['name']}: Young's modulus "
        f"{display_range(material['youngs'])} GPa; Poisson's ratio "
        f"{display_range(material['poisson'])}; density "
        f"{display_range(material['density'])} kg/m^3.\n"
    )
