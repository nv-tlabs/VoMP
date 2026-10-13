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

"""Parse material properties from VLM responses."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from dataset_toolkits.material_objects.vlm_annotations.vlm_input_builder import (
    valid_vlm_property_value,
)


def raw_vlm_output_sha256(output_text: str) -> str:
    """Hash the exact UTF-8 response text stored in an attempt record."""

    if not isinstance(output_text, str):
        raise TypeError("raw VLM output must be a string")
    return hashlib.sha256(output_text.encode("utf-8")).hexdigest()


def parse_vlm_properties(output_text: str) -> dict[str, Any]:
    """Read Young's modulus, Poisson's ratio, and density from a response."""

    if not isinstance(output_text, str):
        raise TypeError("VLM output must be a string")

    analysis_lines = []
    capturing_analysis = False
    for line in output_text.split("\n"):
        line = line.strip()
        if line.lower().startswith("analysis:"):
            capturing_analysis = True
            analysis_lines.append(line.split(":", 1)[1].strip())
        elif capturing_analysis and (
            line.lower().startswith("young")
            or line.lower().startswith("poisson")
            or line.lower().startswith("density")
        ):
            capturing_analysis = False
        elif capturing_analysis and line:
            analysis_lines.append(line)
    analysis = "\n".join(analysis_lines)

    youngs_modulus = None
    poissons_ratio = None
    density = None

    try:
        modulus_lines = re.findall(
            r"^[ \t]*young(?:['’])?s\s+modulus\b.*$",
            output_text,
            re.IGNORECASE | re.MULTILINE,
        )
        modulus_pattern = re.compile(
            r"^\s*young(?:['’])?s\s+modulus\s*[:\-]?\s*"
            r"(?P<value>[+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?"
            r"(?:[eE][+\-]?\d+)?)\s*"
            r"(?P<unit>GPa|MPa|kPa|Pa)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        modulus_match = (
            modulus_pattern.fullmatch(modulus_lines[0])
            if len(modulus_lines) == 1
            else None
        )
        if len(modulus_lines) > 1:
            logging.warning(
                "Rejecting Young's modulus because the VLM emitted %d property lines",
                len(modulus_lines),
            )
        if modulus_match:
            numeric = float(modulus_match.group("value").replace(",", ""))
            multiplier = {
                "pa": 1.0,
                "kpa": 1.0e3,
                "mpa": 1.0e6,
                "gpa": 1.0e9,
            }[modulus_match.group("unit").casefold()]
            youngs_modulus = numeric * multiplier
            logging.info(
                "Extracted Young's modulus with explicit %s unit: %s Pa",
                modulus_match.group("unit"),
                youngs_modulus,
            )
    except Exception as exc:
        logging.error("Young's modulus extraction failed: %s", exc)

    try:
        ratio_lines = re.findall(
            r"^[ \t]*poisson(?:['’])?s[ \t]+ratio\b.*$",
            output_text,
            re.IGNORECASE | re.MULTILINE,
        )
        ratio_pattern = re.compile(
            r"^[ \t]*poisson(?:['’])?s[ \t]+ratio[ \t]*:[ \t]*"
            r"(?P<value>[0-9]+(?:\.[0-9]+)?)[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        )
        ratio_match = (
            ratio_pattern.fullmatch(ratio_lines[0])
            if len(ratio_lines) == 1
            else None
        )
        if len(ratio_lines) > 1:
            logging.warning(
                "Rejecting Poisson's ratio because the VLM emitted %d property lines",
                len(ratio_lines),
            )
        if ratio_match:
            poissons_ratio = float(ratio_match.group("value"))
            logging.info("Extracted Poisson's ratio: %s", poissons_ratio)
    except Exception as exc:
        logging.error("Poisson's ratio extraction failed: %s", exc)

    try:
        density_lines = re.findall(
            r"^[ \t]*density\b.*$",
            output_text,
            re.IGNORECASE | re.MULTILINE,
        )
        density_pattern = re.compile(
            r"^\s*density\s*[:\-]?\s*"
            r"(?P<value>[+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?"
            r"(?:[eE][+\-]?\d+)?)\s*"
            r"(?P<unit>kg\s*/\s*m(?:\^?3|³)|g\s*/\s*cm(?:\^?3|³))"
            r"\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        density_match = (
            density_pattern.fullmatch(density_lines[0])
            if len(density_lines) == 1
            else None
        )
        if len(density_lines) > 1:
            logging.warning(
                "Rejecting density because the VLM emitted %d property lines",
                len(density_lines),
            )
        if density_match:
            numeric = float(density_match.group("value").replace(",", ""))
            normalized_unit = re.sub(
                r"\s+", "", density_match.group("unit").casefold()
            )
            multiplier = 1000.0 if normalized_unit.startswith("g/") else 1.0
            density = numeric * multiplier
            logging.info(
                "Extracted density with explicit %s unit: %s kg/m^3",
                density_match.group("unit"),
                density,
            )
    except Exception as exc:
        logging.error("Density extraction failed: %s", exc)

    parsed_values = {
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "density": density,
    }
    for property_name, value in parsed_values.items():
        if value is not None and not valid_vlm_property_value(property_name, value):
            logging.warning(
                "Invalid %s value in VLM output: %r",
                property_name,
                value,
            )
            parsed_values[property_name] = None

    missing_properties = [
        display_name
        for property_name, display_name in (
            ("youngs_modulus", "Young's modulus"),
            ("poissons_ratio", "Poisson's ratio"),
            ("density", "density"),
        )
        if parsed_values[property_name] is None
    ]
    if missing_properties:
        logging.warning(
            "Missing properties in VLM output: %s", ", ".join(missing_properties)
        )
    else:
        logging.info(
            "All properties extracted: Young's modulus=%s, Poisson's ratio=%s, "
            "Density=%s",
            parsed_values["youngs_modulus"],
            parsed_values["poissons_ratio"],
            parsed_values["density"],
        )

    return {
        "vlm_analysis": analysis,
        **parsed_values,
        "raw_vlm_output": output_text,
        "raw_vlm_output_sha256": raw_vlm_output_sha256(output_text),
    }
