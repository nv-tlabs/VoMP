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
import json
import csv
import random
from pathlib import Path
from typing import Tuple, Set, List
import math
from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    parse_numerical_range_str,
)


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Generate a materials.csv file from material_ranges.csv in the provided directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing material_ranges.csv",
    )
    args = parser.parse_args()
    directory = Path(args.directory).expanduser().resolve()

    if not directory.is_dir():
        parser.error(f"Provided path '{directory}' is not a directory.")

    return directory


def read_dataset(json_path: Path):
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find '{json_path}'. Ensure the directory contains the file."
        )


def extract_unique_rows(
    dataset, unique_triplets: Set[Tuple[float, float, float]] | None = None
) -> tuple[list, Set[Tuple[float, float, float]]]:
    if unique_triplets is None:
        unique_triplets = set()

    rows: List[Tuple[str, float, float, float]] = []

    for obj in dataset:
        segments = obj.get("segments", {})
        for seg_key, seg_data in segments.items():
            try:
                youngs = float(seg_data["youngs_modulus"])
                poisson = float(seg_data["poissons_ratio"])
                density = float(seg_data["density"])
            except (KeyError, ValueError, TypeError):
                continue

            if youngs <= 0 or youngs > 1e13:
                print(
                    f"WARNING: Skipping material with invalid Young's modulus: {youngs}"
                )
                continue

            if poisson < -1.0 or poisson > 0.5:
                print(
                    f"WARNING: Skipping material with invalid Poisson's ratio: {poisson}"
                )
                continue

            if density <= 0 or density > 50000:
                print(f"WARNING: Skipping material with invalid density: {density}")
                continue

            triplet = (youngs, poisson, density)
            if triplet in unique_triplets:
                continue

            unique_triplets.add(triplet)
            material_name = seg_data.get("name", seg_key)
            rows.append((material_name, youngs, poisson, density))

    return rows, unique_triplets


def sample_ranges(
    csv_path: Path,
    unique_triplets: Set[Tuple[float, float, float]],
    min_samples_per_material: int = 100,
    max_samples_per_material: int = 2500,
    target_total_samples: int = 100_000,
) -> list:
    if not csv_path.exists():

        return []

    parsed_rows: list[dict] = []
    dynamic_indices: list[int] = []

    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")

    for idx, line in enumerate(lines[1:], 0):

        parts = []
        current = ""
        in_brackets = False
        for char in line.strip() + ",":
            if char == "," and not in_brackets:
                parts.append(current)
                current = ""
            else:
                if char == "[":
                    in_brackets = True
                elif char == "]":
                    in_brackets = False
                current += char

        if len(parts) < 4:
            print(f"WARNING: Line {idx+1} has incorrect format: {line.strip()}")
            continue

        material_name = parts[0].strip().strip('"')

        y_range_str = parts[1].strip().strip('"')
        p_range_str = parts[2].strip().strip('"')
        d_range_str = parts[3].strip().strip('"')

        try:
            y_low, y_high = parse_numerical_range_str(y_range_str)
            p_low, p_high = parse_numerical_range_str(p_range_str)
            d_low, d_high = parse_numerical_range_str(d_range_str)
        except ValueError as e:
            print(
                f"WARNING: Error parsing ranges for {material_name} on line {idx+1}: {e} - Skipping material."
            )
            continue

        y_low *= 1e9
        y_high *= 1e9

        y_low = max(1e6, min(y_low, 1e13))
        y_high = max(y_low, min(y_high, 1e13))

        p_low = max(-0.999, min(p_low, 0.499))
        p_high = max(p_low, min(p_high, 0.499))

        d_low = max(10.0, min(d_low, 50000.0))
        d_high = max(d_low, min(d_high, 50000.0))

        y_has_range = abs(y_high - y_low) > 1e-6
        p_has_range = abs(p_high - p_low) > 1e-6
        d_has_range = abs(d_high - d_low) > 1e-6

        has_range = y_has_range or p_has_range or d_has_range

        y_width = max(y_high - y_low, 1.0) if y_has_range else 1.0
        p_width = max(p_high - p_low, 0.001) if p_has_range else 0.001
        d_width = max(d_high - d_low, 1.0) if d_has_range else 1.0

        y_width_norm = y_width / 1e9

        volume = y_width_norm * p_width * d_width

        if has_range:
            dynamic_indices.append(idx)

        parsed_rows.append(
            {
                "material_name": material_name,
                "y_low": y_low,
                "y_high": y_high,
                "p_low": p_low,
                "p_high": p_high,
                "d_low": d_low,
                "d_high": d_high,
                "has_range": has_range,
                "y_has_range": y_has_range,
                "p_has_range": p_has_range,
                "d_has_range": d_has_range,
                "volume": volume,
            }
        )

    if not parsed_rows:
        return []

    print(
        f"Found {len(dynamic_indices)} materials with ranges out of {len(parsed_rows)} total"
    )

    fixed_count = len(parsed_rows) - len(dynamic_indices)
    print(f"Number of materials with fixed values: {fixed_count}")

    if dynamic_indices:
        print("\nExample materials with ranges:")
        for i in range(min(5, len(dynamic_indices))):
            idx = dynamic_indices[i]
            info = parsed_rows[idx]
            ranges_info = []
            if info["y_has_range"]:
                ranges_info.append(
                    f"Young's: {info['y_low']/1e9:.3f}-{info['y_high']/1e9:.3f} GPa"
                )
            if info["p_has_range"]:
                ranges_info.append(
                    f"Poisson's: {info['p_low']:.3f}-{info['p_high']:.3f}"
                )
            if info["d_has_range"]:
                ranges_info.append(
                    f"Density: {info['d_low']:.1f}-{info['d_high']:.1f} kg/m³"
                )

            print(
                f"  {info['material_name']}: {', '.join(ranges_info)} (volume: {info['volume']:.4f})"
            )

    total_volume = sum(parsed_rows[idx]["volume"] for idx in dynamic_indices)
    print(f"\nTotal parameter space volume: {total_volume:.4f}")

    volume_scale_factor = 13.0

    samples_per_material = {}

    for idx in dynamic_indices:
        volume_ratio = parsed_rows[idx]["volume"] / total_volume
        proportional_samples = max(
            math.ceil(target_total_samples * volume_ratio * volume_scale_factor),
            min_samples_per_material,
        )

        samples_per_material[idx] = min(proportional_samples, max_samples_per_material)

    fixed_total = 0

    dynamic_total = sum(samples_per_material.values())
    total_planned = dynamic_total + fixed_total

    print(f"\nSampling strategy (scaled by {volume_scale_factor}x):")
    print(f"  Minimum samples per material with ranges: {min_samples_per_material}")
    print(f"  Maximum samples per material: {max_samples_per_material}")
    print(f"  Planned total samples: {total_planned}")

    sorted_materials = sorted(
        [
            (
                idx,
                parsed_rows[idx]["material_name"],
                parsed_rows[idx]["volume"],
                samples_per_material.get(idx, 1) if idx in dynamic_indices else 1,
            )
            for idx in range(len(parsed_rows))
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    print("\nTop 15 highest volume materials:")
    for idx, name, volume, samples in sorted_materials[:15]:
        if idx in dynamic_indices:
            volume_percent = volume / total_volume * 100
            print(
                f"  {name}: volume {volume:.4f} ({volume_percent:.2f}%), {samples} samples"
            )
        else:
            print(f"  {name}: fixed values, 1 sample")

    rows: list[Tuple[str, float, float, float]] = []

    def _add_triplet(material: str, y: float, p: float, d: float):

        if y <= 0 or y > 1e13:
            return False
        if p < -1.0 or p > 0.5:
            return False
        if d <= 0 or d > 50000:
            return False

        triplet = (y, p, d)
        if triplet in unique_triplets:
            return False
        unique_triplets.add(triplet)
        rows.append((material, y, p, d))
        return True

    total_generated = 0
    duplicate_avoidance_failures = 0

    for idx, info in enumerate(parsed_rows):
        if not info["has_range"]:
            name = info["material_name"]
            y_val = info["y_low"]
            p_val = info["p_low"]
            d_val = info["d_low"]

            if _add_triplet(name, y_val, p_val, d_val):
                total_generated += 1

    print(f"Added {total_generated} materials with fixed values")

    for idx in dynamic_indices:
        info = parsed_rows[idx]
        name = info["material_name"]
        y_low, y_high = info["y_low"], info["y_high"]
        p_low, p_high = info["p_low"], info["p_high"]
        d_low, d_high = info["d_low"], info["d_high"]

        required = samples_per_material.get(idx, 0)

        report_progress = required > 100

        attempts = 0
        generated = 0

        max_attempts = required * 50

        if info["volume"] > 10.0:
            max_attempts *= 2

        if report_progress:
            print(
                f"Generating {required} samples for {name} (volume: {info['volume']:.4f})"
            )

        while generated < required and attempts < max_attempts:
            attempts += 1

            y_val = random.uniform(y_low, y_high) if info["y_has_range"] else y_low
            p_val = random.uniform(p_low, p_high) if info["p_has_range"] else p_low
            d_val = random.uniform(d_low, d_high) if info["d_has_range"] else d_low

            y_val = round(y_val, 10)
            p_val = round(p_val, 10)
            d_val = round(d_val, 10)

            if _add_triplet(name, y_val, p_val, d_val):
                generated += 1
                total_generated += 1
            else:

                duplicate_avoidance_failures += 1

            if report_progress and generated > 0 and generated % 100 == 0:
                print(f"  Generated {generated}/{required} samples for {name}")

        if required > 0 and report_progress:

            success_rate = (generated / attempts) * 100 if attempts > 0 else 0
            print(
                f"Material {name}: Generated {generated}/{required} samples after {attempts} attempts (success rate: {success_rate:.1f}%)"
            )

    print(f"Successfully generated {len(rows)} unique material property combinations")
    print(
        f"Duplicate avoidance prevented {duplicate_avoidance_failures} potential duplicates"
    )
    return rows


def write_csv(rows: list, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["material_name", "youngs_modulus", "poisson_ratio", "density"])
        for row in rows:
            writer.writerow(row)


def main():
    directory = parse_args()
    csv_path = directory / "materials.csv"

    print("Generating materials data from ranges only (skipping JSON file)...")

    unique_triplets = set()

    ranges_csv_path = directory / "material_ranges.csv"
    if not ranges_csv_path.exists():
        print(f"ERROR: material_ranges.csv not found at {ranges_csv_path}")
        return

    sampled_rows = sample_ranges(ranges_csv_path, unique_triplets)

    write_csv(sampled_rows, csv_path)

    print(
        f"materials.csv generated with {len(sampled_rows)} unique rows at '{csv_path}'."
    )
    print("All data generated from material_ranges.csv with validation applied.")


if __name__ == "__main__":
    main()
