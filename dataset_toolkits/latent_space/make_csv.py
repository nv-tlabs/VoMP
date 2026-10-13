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


def parse_numerical_range_str(range_str: str) -> Tuple[float, float]:
    cleaned = str(range_str).strip().removeprefix("[").removesuffix("]")
    values = tuple(float(part.strip()) for part in cleaned.split(","))
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values
    raise ValueError(f"Invalid numerical range: {range_str!r}")


def parse_args() -> Path:
    """Parse command-line arguments and return the target directory path."""
    parser = argparse.ArgumentParser(
        description="Generate new MatVAE material CSVs from material_ranges.csv."
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
    """Read and return the content of material_dataset.json."""
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
    """Extract unique rows (by physical property triplet) from the dataset.

    Returns a tuple of (rows, unique_triplets).
    Rows are tuples in the order:
        (material_name, youngs_modulus, poisson_ratio, density)
    """
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

            # Validate physical properties are in reasonable ranges
            if (
                youngs <= 0 or youngs > 1e13
            ):  # Young's modulus should be positive and < 10 TPa
                print(
                    f"WARNING: Skipping material with invalid Young's modulus: {youngs}"
                )
                continue

            if poisson <= -1.0 or poisson >= 0.5:  # Stable isotropic limits are open
                print(
                    f"WARNING: Skipping material with invalid Poisson's ratio: {poisson}"
                )
                continue

            if (
                density <= 0 or density > 50000
            ):  # Density should be positive and < 50,000 kg/m³
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
    target_total_samples: int = 100_000,
) -> list:
    """Sample property triplets from ranges defined in material_ranges.csv.

    Keeps interpolated triplets unique using *unique_triplets*. Valid source
    anchors may be repeated to meet a training-distribution quota without
    inventing unsupported combinations.
    Returns list of rows: (material_name, youngs_modulus, poisson_ratio, density)
    """
    if not csv_path.exists():

        return []

    parsed_rows: list[dict] = []
    dynamic_indices: list[int] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))

    for idx, source_row in enumerate(source_rows):
        material_name = source_row["Material Name"].strip()
        y_range_str = source_row["Young's Modulus Range [GPa]"]
        p_range_str = source_row["Poisson's Ratio Range"]
        d_range_str = source_row["Density Range"]

        y_low, y_high = parse_numerical_range_str(y_range_str)
        p_low, p_high = parse_numerical_range_str(p_range_str)
        d_low, d_high = parse_numerical_range_str(d_range_str)

        # Apply GPa conversion for Young's modulus
        # Assuming input values in CSV for Young's Modulus are in GPa
        y_low *= 1e9
        y_high *= 1e9

        # Fail on bad source data instead of silently clamping it into a different
        # material. These are the thermodynamic bounds for a stable isotropic
        # linear-elastic triplet, plus generous dataset ceilings.
        if not 0.0 < y_low <= y_high <= 1e13:
            raise ValueError(
                f"Invalid Young's-modulus range for {material_name!r} on line {idx + 2}: "
                f"{y_low} to {y_high} Pa"
            )
        if not -1.0 < p_low <= p_high < 0.5:
            raise ValueError(
                f"Invalid Poisson-ratio range for {material_name!r} on line {idx + 2}: "
                f"{p_low} to {p_high}"
            )
        if not 0.0 < d_low <= d_high <= 50000.0:
            raise ValueError(
                f"Invalid density range for {material_name!r} on line {idx + 2}: "
                f"{d_low} to {d_high} kg/m^3"
            )

        y_has_range = (
            abs(y_high - y_low) > 1e-6
        )  # Using a small epsilon for float comparison
        p_has_range = abs(p_high - p_low) > 1e-6
        d_has_range = abs(d_high - d_low) > 1e-6

        varying_properties = sum((y_has_range, p_has_range, d_has_range))

        # A one-property interval can be sampled without inventing covariance.
        # When a source reports two or three marginal intervals but no paired
        # observations, retain its central representative triplet instead of
        # fabricating a correlation between the properties.
        has_range = varying_properties == 1
        if varying_properties > 1:
            y_low = y_high = (y_low + y_high) / 2.0
            p_low = p_high = (p_low + p_high) / 2.0
            d_low = d_high = (d_low + d_high) / 2.0
            y_has_range = p_has_range = d_has_range = False

        if has_range:
            dynamic_indices.append(len(parsed_rows))

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
                "family": source_row.get("Material Family", "unknown"),
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

            print(f"  {info['material_name']}: {', '.join(ranges_info)}")

    rows: list[Tuple[str, float, float, float]] = []
    fixed_anchors: list[tuple[Tuple[str, float, float, float], str]] = []

    def _add_triplet(material: str, y: float, p: float, d: float):
        """Helper to add a triplet if it is globally unique."""
        # Validate physical properties before adding
        if y <= 0 or y > 1e13:
            return False
        if p <= -1.0 or p >= 0.5:
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
                fixed_anchors.append((rows[-1], info["family"]))

    print(f"Added {total_generated} materials with fixed values")

    # do some derivations to reserve more rows for propotyionally lower log(E) ranges
    reference_counts = (
        1000,
        1000,
        1000,
        4000,
        7000,
        8000,
        9000,
        12000,
        23000,
        16000,
        17000,
        1000,
    )
    bin_targets = [
        round(target_total_samples * count / 100_000) for count in reference_counts
    ]
    bin_targets[8] += target_total_samples - sum(bin_targets)
    youngs_bands = [(10.0**power, 10.0 ** (power + 1)) for power in range(1, 13)]

    def _youngs_band(value: float) -> int:
        for band, (low, high) in enumerate(youngs_bands):
            if low <= value < high or (
                band == len(youngs_bands) - 1 and low <= value <= high
            ):
                return band
        raise ValueError(f"Young's modulus is outside sampling bands: {value}")

    fixed_per_band = [0] * len(youngs_bands)
    for _, y_val, _, _ in rows:
        fixed_per_band[_youngs_band(y_val)] += 1

    max_samples_per_range_per_band = 512
    sampling_jobs = []
    repeat_jobs = []

    def _queue_anchor_repeats(band: int, required: int, eligible: list):
        anchors = [
            (row, family)
            for row, family in fixed_anchors
            if _youngs_band(row[1]) == band
        ]
        for idx, sample_low, sample_high in eligible:
            info = parsed_rows[idx]
            y_val = (
                math.sqrt(sample_low * sample_high)
                if sample_low < sample_high
                else sample_low
            )
            p_val = (info["p_low"] + info["p_high"]) / 2.0
            d_val = math.sqrt(info["d_low"] * info["d_high"])
            anchors.append(
                (
                    (info["material_name"], y_val, p_val, d_val),
                    info["family"],
                )
            )

        if not anchors:
            band_low, band_high = youngs_bands[band]
            raise RuntimeError(
                f"No sourced material covers {band_low:.0e}-{band_high:.0e} Pa"
            )

        unique_anchors_by_family = {}
        seen_triplets = set()
        for row, family in anchors:
            triplet = row[1:]
            if triplet not in seen_triplets:
                seen_triplets.add(triplet)
                unique_anchors_by_family.setdefault(family, []).append(row)

        families = sorted(unique_anchors_by_family)
        family_samples, family_extra = divmod(required, len(families))
        for family_position, family in enumerate(families):
            family_required = family_samples + (family_position < family_extra)
            family_anchors = unique_anchors_by_family[family]
            anchor_samples, anchor_extra = divmod(
                family_required, len(family_anchors)
            )
            for anchor_position, row in enumerate(family_anchors):
                count = anchor_samples + (anchor_position < anchor_extra)
                if count:
                    repeat_jobs.append((row, count))

    for band, ((band_low, band_high), target) in enumerate(
        zip(youngs_bands, bin_targets)
    ):
        if fixed_per_band[band] > target:
            raise RuntimeError(
                f"Young's-modulus band {band_low:.0e}-{band_high:.0e} Pa has "
                f"{fixed_per_band[band]} fixed source points but a target of only {target}"
            )
        needed = target - fixed_per_band[band]
        if needed == 0:
            continue

        eligible = []
        for idx in dynamic_indices:
            info = parsed_rows[idx]
            if info["y_has_range"]:
                sample_low = max(info["y_low"], band_low)
                sample_high = min(info["y_high"], band_high)
                if sample_low >= sample_high:
                    continue
                if sample_high == band_high:
                    sample_high = band_high * (1.0 - 1e-10)
            elif _youngs_band(info["y_low"]) == band:
                sample_low = sample_high = info["y_low"]
            else:
                continue
            eligible.append((idx, sample_low, sample_high))

        if not eligible:
            _queue_anchor_repeats(band, needed, eligible)
            continue

        by_family = {}
        for entry in eligible:
            family = parsed_rows[entry[0]]["family"]
            by_family.setdefault(family, []).append(entry)

        families = sorted(by_family)
        family_samples, family_extra = divmod(needed, len(families))
        allocated = 0
        for family_position, family in enumerate(families):
            family_target = family_samples + (family_position < family_extra)
            records = by_family[family]
            record_samples, record_extra = divmod(family_target, len(records))
            for record_position, (idx, sample_low, sample_high) in enumerate(records):
                required = record_samples + (record_position < record_extra)
                required = min(required, max_samples_per_range_per_band)
                if required:
                    sampling_jobs.append((idx, sample_low, sample_high, required))
                    allocated += required

        if allocated < needed:
            _queue_anchor_repeats(band, needed - allocated, eligible)

    print("Sampling targets by Young's-modulus decade:")
    for power, target, fixed in zip(range(1, 13), bin_targets, fixed_per_band):
        print(f"  1e{power}-1e{power + 1} Pa: {target} rows ({fixed} fixed)")

    for idx, sample_y_low, sample_y_high, required in sampling_jobs:
        info = parsed_rows[idx]
        name = info["material_name"]
        y_low, y_high = info["y_low"], info["y_high"]
        p_low, p_high = info["p_low"], info["p_high"]
        d_low, d_high = info["d_low"], info["d_high"]

        attempts = 0
        generated = 0

        max_attempts = required * 50

        while generated < required and attempts < max_attempts:
            attempts += 1

            if attempts == 1:
                y_val = sample_y_low
            elif attempts == 2:
                y_val = sample_y_high
            elif sample_y_low < sample_y_high:
                y_val = 10 ** random.uniform(
                    math.log10(sample_y_low), math.log10(sample_y_high)
                )
            else:
                y_val = sample_y_low

            if info["p_has_range"]:
                if attempts == 1:
                    p_val = p_low
                elif attempts == 2:
                    p_val = p_high
                else:
                    p_val = random.uniform(p_low, p_high)
            else:
                p_val = p_low

            if info["d_has_range"]:
                if attempts == 1:
                    d_val = d_low
                elif attempts == 2:
                    d_val = d_high
                else:
                    d_val = 10 ** random.uniform(
                        math.log10(d_low), math.log10(d_high)
                    )
            else:
                d_val = d_low

            y_val = round(y_val, 10)
            p_val = round(p_val, 10)
            d_val = round(d_val, 10)

            if _add_triplet(name, y_val, p_val, d_val):
                generated += 1
                total_generated += 1
            else:
                duplicate_avoidance_failures += 1

        if generated != required:
            print(f"WARNING: Generated only {generated}/{required} samples for {name}")

    repeated_rows = 0
    for row, required in repeat_jobs:
        rows.extend([row] * required)
        repeated_rows += required
        total_generated += required

    if len(rows) != target_total_samples:
        raise RuntimeError(
            f"Generated {len(rows)} rows instead of the requested {target_total_samples}"
        )

    unique_count = len({(y, p, d) for _, y, p, d in rows})
    print(
        f"Successfully generated {len(rows)} training rows from {unique_count} "
        "source-supported property combinations"
    )
    print(f"Repeated {repeated_rows} valid anchor rows instead of inventing correlations")
    print(
        f"Duplicate avoidance prevented {duplicate_avoidance_failures} potential duplicates"
    )
    return rows


def write_csv(rows: list, csv_path: Path):
    """Write rows to a CSV file with given path including headers."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["material_name", "youngs_modulus", "poisson_ratio", "density"])
        for row in rows:
            writer.writerow(row)


def main():
    directory = parse_args()
    csv_path = directory / "materials.csv"
    final_csv_path = directory / "materials_filtered.csv"


    # Start with empty unique triplets set
    unique_triplets = set()

    # Only sample from ranges
    ranges_csv_path = directory / "material_ranges.csv"
    if not ranges_csv_path.exists():
        print(f"ERROR: material_ranges.csv not found at {ranges_csv_path}")
        return

    random.seed(42)
    sampled_rows = sample_ranges(ranges_csv_path, unique_triplets)

    write_csv(sampled_rows, csv_path)
    write_csv(sampled_rows, final_csv_path)

    print(
        f"Generated {len(sampled_rows)} training rows in '{csv_path}' and '{final_csv_path}'."
    )

if __name__ == "__main__":
    main()
