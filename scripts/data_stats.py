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

import pandas as pd
import numpy as np
import os
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Add paths to import material annotations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_material_annotations():
    """Load material annotations JSON file"""
    json_path = "datasets/raw/material_annotations.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return []


def get_material_properties_stats(segments):
    """Extract material properties for statistical analysis"""
    densities = []
    youngs_moduli = []
    poissons_ratios = []

    for segment_name, segment_data in segments.items():
        density = segment_data.get("density")
        youngs_modulus = segment_data.get("youngs_modulus")
        poissons_ratio = segment_data.get("poissons_ratio")

        if density is not None:
            densities.append(density)
        if youngs_modulus is not None:
            youngs_moduli.append(youngs_modulus)
        if poissons_ratio is not None:
            poissons_ratios.append(poissons_ratio)

    return densities, youngs_moduli, poissons_ratios


def print_property_stats(values, property_name, unit=""):
    """Print comprehensive statistics for a material property"""
    if not values:
        print(f"  {property_name}: No data available")
        return

    values = np.array(values)
    print(f"  {property_name} ({unit}):")
    print(f"    Count: {len(values)}")
    print(f"    Min: {np.min(values):.6e}")
    print(f"    Max: {np.max(values):.6e}")
    print(f"    Mean: {np.mean(values):.6e}")
    print(f"    Median: {np.median(values):.6e}")
    print(f"    Std Dev: {np.std(values):.6e}")
    print(f"    25th Percentile: {np.percentile(values, 25):.6e}")
    print(f"    75th Percentile: {np.percentile(values, 75):.6e}")
    print(f"    95th Percentile: {np.percentile(values, 95):.6e}")
    print(f"    99th Percentile: {np.percentile(values, 99):.6e}")

    # Check for outliers (values beyond 3 standard deviations)
    mean_val = np.mean(values)
    std_val = np.std(values)
    outliers = values[
        (values < mean_val - 3 * std_val) | (values > mean_val + 3 * std_val)
    ]
    print(f"    Outliers (±3σ): {len(outliers)} ({len(outliers)/len(values)*100:.1f}%)")

    # Unique values count
    unique_values = len(np.unique(values))
    print(f"    Unique Values: {unique_values}")
    print()


def get_material_triplets(segments):
    """Extract material triplets from segments"""
    triplets = []
    for segment_name, segment_data in segments.items():
        density = segment_data.get("density")
        youngs_modulus = segment_data.get("youngs_modulus")
        poissons_ratio = segment_data.get("poissons_ratio")
        if all(x is not None for x in [density, youngs_modulus, poissons_ratio]):
            triplets.append((density, youngs_modulus, poissons_ratio))
    return triplets


def get_material_types(segments):
    """Extract material types from segments"""
    return [
        segment_data.get("material_type", "unknown")
        for segment_data in segments.values()
    ]


def count_points_in_material_npz(file_path):
    """Count total points in a material NPZ file"""
    try:
        data = np.load(file_path)
        if len(data.files) > 0:
            first_key = data.files[0]
            return data[first_key].shape[0]
    except Exception:
        pass
    return 0


def print_stats():
    # Load all datasets metadata (stored in simready directory)
    metadata_df = pd.read_csv("datasets/simready/metadata.csv")

    # Load material annotations
    material_annotations = load_material_annotations()

    # Create lookup for material annotations by file path
    material_lookup = {}
    for entry in material_annotations:
        file_path = entry.get("file_path", "")
        if file_path:
            material_lookup[file_path] = entry

    # Filter for objects that are rendered, have features, and are voxelized
    filtered_df = metadata_df[
        (metadata_df["rendered"] == True)
        & (metadata_df["feature_dinov2_vitl14_reg"] == True)
        & (metadata_df["voxelized"] == True)
    ].copy()

    print("=== DATASET STATISTICS ===")
    print()

    # Basic object counts
    total_objects = len(filtered_df)
    print(f"Total Objects: {total_objects}")
    print()

    # Objects per dataset
    print("Objects per Dataset:")
    dataset_counts = filtered_df["dataset"].value_counts()
    for dataset, count in dataset_counts.items():
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {dataset}: {count} ({pct:.1f}%)")
    print()

    # Train/Val/Test splits - Total
    print("Train/Val/Test Split - Total:")
    split_counts = filtered_df["split"].value_counts()
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {split}: {count} ({pct:.1f}%)")
    print()

    # Train/Val/Test splits per dataset
    print("Train/Val/Test Split per Dataset:")
    for dataset in dataset_counts.index:
        dataset_df = filtered_df[filtered_df["dataset"] == dataset]
        dataset_split_counts = dataset_df["split"].value_counts()
        print(f"  {dataset}:")
        for split in ["train", "val", "test"]:
            count = dataset_split_counts.get(split, 0)
            pct = (count / len(dataset_df) * 100) if len(dataset_df) > 0 else 0
            print(f"    {split}: {count} ({pct:.1f}%)")
    print()

    # Segment statistics
    total_segments = 0
    segments_per_dataset = defaultdict(int)
    material_triplets = []
    triplets_per_dataset = defaultdict(list)
    material_types_all = []
    material_types_per_dataset = defaultdict(list)

    # Material properties for statistical analysis
    all_densities = []
    all_youngs_moduli = []
    all_poissons_ratios = []

    objects_with_materials = 0

    for _, row in filtered_df.iterrows():
        file_path = row["local_path"]
        dataset = row["dataset"]

        # Get material annotation for this object
        material_entry = material_lookup.get(file_path)
        if material_entry:
            objects_with_materials += 1
            segments = material_entry.get("segments", {})
            num_segments = len(segments)
            total_segments += num_segments
            segments_per_dataset[dataset] += num_segments

            # Extract material triplets
            triplets = get_material_triplets(segments)
            material_triplets.extend(triplets)
            triplets_per_dataset[dataset].extend(triplets)

            # Extract material types
            mat_types = get_material_types(segments)
            material_types_all.extend(mat_types)
            material_types_per_dataset[dataset].extend(mat_types)

            # Extract material properties for statistical analysis
            densities, youngs_moduli, poissons_ratios = get_material_properties_stats(
                segments
            )
            all_densities.extend(densities)
            all_youngs_moduli.extend(youngs_moduli)
            all_poissons_ratios.extend(poissons_ratios)

    print(f"Total Segments: {total_segments}")
    segments_per_object = (
        total_segments / objects_with_materials if objects_with_materials > 0 else 0
    )
    print(f"Average Segments per Object: {segments_per_object:.2f}")
    print()

    print("Segments per Dataset:")
    for dataset, count in segments_per_dataset.items():
        pct = (count / total_segments * 100) if total_segments > 0 else 0
        print(f"  {dataset}: {count} ({pct:.1f}%)")
    print()

    # Points in material NPZ files
    total_points = 0
    points_per_dataset = defaultdict(int)
    points_per_object = []
    points_per_object_per_dataset = defaultdict(list)
    material_npz_dir = "datasets/simready/voxels"

    for _, row in filtered_df.iterrows():
        sha256 = row["sha256"]
        dataset = row["dataset"]
        material_file = f"{sha256}_with_materials.npz"
        material_path = os.path.join(material_npz_dir, material_file)

        if os.path.exists(material_path):
            points = count_points_in_material_npz(material_path)
            total_points += points
            points_per_dataset[dataset] += points
            points_per_object.append(points)
            points_per_object_per_dataset[dataset].append(points)

    print(f"Total Voxels in Material NPZ: {total_points}")
    if points_per_object:
        avg_points = np.mean(points_per_object)
        std_points = np.std(points_per_object)
        print(f"Average Voxels per Object: {avg_points:.0f} (±{std_points:.0f})")
        print(f"Min Voxels per Object: {min(points_per_object)}")
        print(f"Max Voxels per Object: {max(points_per_object)}")
    print()

    print("Total Voxels per Dataset:")
    for dataset, points in points_per_dataset.items():
        pct = (points / total_points * 100) if total_points > 0 else 0
        print(f"  {dataset}: {points} ({pct:.1f}%)")
    print()

    print("Average Voxels per Object per Dataset:")
    for dataset in dataset_counts.index:
        if (
            dataset in points_per_object_per_dataset
            and points_per_object_per_dataset[dataset]
        ):
            avg_voxels = np.mean(points_per_object_per_dataset[dataset])
            print(f"  {dataset}: {avg_voxels:.0f}")
        else:
            print(f"  {dataset}: 0")
    print()

    # Unique material triplets
    unique_triplets = set(material_triplets)
    print(f"Total Unique Material Triplets: {len(unique_triplets)}")
    print()

    print("Unique Material Triplets per Dataset:")
    for dataset, triplets in triplets_per_dataset.items():
        unique_dataset_triplets = set(triplets)
        print(f"  {dataset}: {len(unique_dataset_triplets)}")
    print()

    # Material Properties Statistical Analysis
    print("Material Properties Statistical Analysis:")
    print_property_stats(all_densities, "Density", "kg/m³")
    print_property_stats(all_youngs_moduli, "Young's Modulus", "Pa")
    print_property_stats(all_poissons_ratios, "Poisson's Ratio", "dimensionless")

    # Material type statistics
    print("Material Type Counts (Top 10):")
    material_type_counts = Counter(material_types_all)
    for material_type, count in material_type_counts.most_common(10):
        pct = (count / len(material_types_all) * 100) if material_types_all else 0
        print(f"  {material_type}: {count} ({pct:.1f}%)")

    print(f"\nTotal Unique Material Types: {len(material_type_counts)}")
    print()

    # Object class statistics
    print("Object Class Counts (All):")
    class_counts = filtered_df["class"].value_counts()
    for obj_class, count in class_counts.items():
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {obj_class}: {count} ({pct:.1f}%)")

    print(f"\nTotal Unique Object Classes: {len(class_counts)}")
    print()

    # Dataset-specific material type analysis
    print("Material Types per Dataset (Top 5 each):")
    for dataset in dataset_counts.index:
        dataset_materials = material_types_per_dataset[dataset]
        if dataset_materials:
            dataset_material_counts = Counter(dataset_materials)
            print(f"  {dataset}:")
            for material_type, count in dataset_material_counts.most_common(5):
                pct = (count / len(dataset_materials) * 100) if dataset_materials else 0
                print(f"    {material_type}: {count} ({pct:.1f}%)")
    print()

    # Generate LaTeX table
    print("LaTeX Table:")
    print("\\begin{tabular}{lrrrrrrrr}")
    print("\\toprule")
    print("\\rowcolor{blue!15}")
    print(
        "Dataset & Total Objects & Segments (\\%) & Voxels (\\%) & Avg. Segments/Object & Avg. Voxels/Object \\\\"
    )
    print("\\midrule")

    # Calculate statistics per dataset for the table
    table_data = []
    total_objects_all = 0
    total_segments_all = 0
    total_voxels_all = 0
    all_segments_per_obj = []
    all_voxels_per_obj = []

    for dataset in sorted(dataset_counts.index):
        dataset_df = filtered_df[filtered_df["dataset"] == dataset]
        dataset_objects = len(dataset_df)
        dataset_segments = segments_per_dataset[dataset]
        dataset_voxels = points_per_dataset[dataset]

        # Calculate segments per object for this dataset
        dataset_segments_per_obj = []
        for _, row in dataset_df.iterrows():
            file_path = row["local_path"]
            material_entry = material_lookup.get(file_path)
            if material_entry:
                segments = material_entry.get("segments", {})
                dataset_segments_per_obj.append(len(segments))

        # Calculate voxels per object for this dataset
        dataset_voxels_per_obj = points_per_object_per_dataset[dataset]

        # Store for overall totals
        total_objects_all += dataset_objects
        total_segments_all += dataset_segments
        total_voxels_all += dataset_voxels
        all_segments_per_obj.extend(dataset_segments_per_obj)
        all_voxels_per_obj.extend(dataset_voxels_per_obj)

        # Calculate averages and std devs
        avg_segments = (
            np.mean(dataset_segments_per_obj) if dataset_segments_per_obj else 0
        )
        std_segments = (
            np.std(dataset_segments_per_obj) if dataset_segments_per_obj else 0
        )
        avg_voxels = np.mean(dataset_voxels_per_obj) if dataset_voxels_per_obj else 0
        std_voxels = np.std(dataset_voxels_per_obj) if dataset_voxels_per_obj else 0

        table_data.append(
            {
                "dataset": dataset,
                "objects": dataset_objects,
                "segments": dataset_segments,
                "voxels": dataset_voxels,
                "avg_segments": avg_segments,
                "std_segments": std_segments,
                "avg_voxels": avg_voxels,
                "std_voxels": std_voxels,
            }
        )

    # Print each dataset row
    for data in table_data:
        segments_pct = (
            (data["segments"] / total_segments_all * 100)
            if total_segments_all > 0
            else 0
        )
        voxels_pct = (
            (data["voxels"] / total_voxels_all * 100) if total_voxels_all > 0 else 0
        )

        print(
            f"{data['dataset']} & {data['objects']} & {data['segments']} ({segments_pct:.1f}) & {data['voxels']:,} ({voxels_pct:.1f}) & {data['avg_segments']:.2f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${data['std_segments']:.2f})}}}}}} & {data['avg_voxels']:,.0f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${data['std_voxels']:,.0f})}}}}}} \\\\"
        )

    print("\\midrule")

    # Calculate overall totals
    overall_avg_segments = np.mean(all_segments_per_obj) if all_segments_per_obj else 0
    overall_std_segments = np.std(all_segments_per_obj) if all_segments_per_obj else 0
    overall_avg_voxels = np.mean(all_voxels_per_obj) if all_voxels_per_obj else 0
    overall_std_voxels = np.std(all_voxels_per_obj) if all_voxels_per_obj else 0

    print(
        f"\\textbf{{Total}} & {total_objects_all} & {total_segments_all} (100.0) & {total_voxels_all:,} (100.0) & {overall_avg_segments:.2f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${overall_std_segments:.2f})}}}}}} & {overall_avg_voxels:,.0f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${overall_std_voxels:,.0f})}}}}}} \\\\"
    )

    print("\\bottomrule")
    print("\\end{tabular}")
    print()

    # Generate LaTeX table by splits
    print("LaTeX Table by Splits:")
    print("\\begin{tabular}{lrrrrrrrr}")
    print("\\toprule")
    print("\\rowcolor{blue!15}")
    print(
        "Dataset & Total Objects & Segments (\\%) & Voxels (\\%) & Avg. Segments/Object & Avg. Voxels/Object \\\\"
    )
    print("\\midrule")

    # Calculate statistics per split for the table
    split_table_data = []
    split_names = ["train", "val", "test"]  # Use 'val' as it appears in the data

    for split in split_names:
        split_df = filtered_df[filtered_df["split"] == split]
        split_objects = len(split_df)

        # Calculate segments for this split
        split_segments = 0
        split_segments_per_obj = []
        for _, row in split_df.iterrows():
            file_path = row["local_path"]
            material_entry = material_lookup.get(file_path)
            if material_entry:
                segments = material_entry.get("segments", {})
                num_segments = len(segments)
                split_segments += num_segments
                split_segments_per_obj.append(num_segments)

        # Calculate voxels for this split
        split_voxels = 0
        split_voxels_per_obj = []
        for _, row in split_df.iterrows():
            sha256 = row["sha256"]
            material_file = f"{sha256}_with_materials.npz"
            material_path = os.path.join(material_npz_dir, material_file)

            if os.path.exists(material_path):
                points = count_points_in_material_npz(material_path)
                split_voxels += points
                split_voxels_per_obj.append(points)

        # Calculate averages and std devs
        avg_segments = np.mean(split_segments_per_obj) if split_segments_per_obj else 0
        std_segments = np.std(split_segments_per_obj) if split_segments_per_obj else 0
        avg_voxels = np.mean(split_voxels_per_obj) if split_voxels_per_obj else 0
        std_voxels = np.std(split_voxels_per_obj) if split_voxels_per_obj else 0

        # Convert 'val' to 'validation' for display
        display_split = "validation" if split == "val" else split

        split_table_data.append(
            {
                "split": display_split,
                "objects": split_objects,
                "segments": split_segments,
                "voxels": split_voxels,
                "avg_segments": avg_segments,
                "std_segments": std_segments,
                "avg_voxels": avg_voxels,
                "std_voxels": std_voxels,
            }
        )

    # Print each split row
    for data in split_table_data:
        segments_pct = (
            (data["segments"] / total_segments_all * 100)
            if total_segments_all > 0
            else 0
        )
        voxels_pct = (
            (data["voxels"] / total_voxels_all * 100) if total_voxels_all > 0 else 0
        )

        print(
            f"{data['split']} & {data['objects']} & {data['segments']} ({segments_pct:.1f}) & {data['voxels']:,} ({voxels_pct:.1f}) & {data['avg_segments']:.2f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${data['std_segments']:.2f})}}}}}} & {data['avg_voxels']:,.0f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${data['std_voxels']:,.0f})}}}}}} \\\\"
        )

    print("\\midrule")

    # Total row (same as before)
    print(
        f"\\textbf{{Total}} & {total_objects_all} & {total_segments_all} (100.0) & {total_voxels_all:,} (100.0) & {overall_avg_segments:.2f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${overall_std_segments:.2f})}}}}}} & {overall_avg_voxels:,.0f} {{\\textbf{{\\scriptsize\\textcolor{{gray}}{{($\\pm${overall_std_voxels:,.0f})}}}}}} \\\\"
    )

    print("\\bottomrule")
    print("\\end{tabular}")
    print()


if __name__ == "__main__":
    print_stats()
