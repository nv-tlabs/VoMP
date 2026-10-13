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
import json
import pandas as pd
import numpy as np
import hashlib
import random
from pathlib import Path


from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    SIMREADY_PROPS_DIR,
    COMMERCIAL_BASE_DIR,
    RESIDENTIAL_BASE_DIR,
    VEGETATION_BASE_DIR,
    SIMREADY_ASSET_CLASS_MAPPING,
    SIMREADY_ASSET_INFO_PATH,
)
from dataset_toolkits.material_objects.vlm_annotations.annotation_input_policy import (
    asset_annotation_exclusion,
)
from dataset_toolkits.material_objects.vlm_annotations.asset_finder import (
    discover_assets,
)
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def add_args(parser):
    """Add dataset-specific arguments to the parser."""
    parser.add_argument(
        "--simready_dir",
        type=str,
        default=SIMREADY_PROPS_DIR,
        help="Path to the SimReady props directory",
    )
    parser.add_argument(
        "--commercial_dir",
        type=str,
        default=COMMERCIAL_BASE_DIR,
        help="Path to the Commercial models directory",
    )
    parser.add_argument(
        "--residential_dir",
        type=str,
        default=RESIDENTIAL_BASE_DIR,
        help="Path to the Residential models directory",
    )
    parser.add_argument(
        "--vegetation_dir",
        type=str,
        default=VEGETATION_BASE_DIR,
        help="Path to the Vegetation models directory",
    )
    parser.add_argument(
        "--asset_info_path",
        type=str,
        default=SIMREADY_ASSET_INFO_PATH,
        help="Path to the SimReady asset_info.json file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--default_class",
        type=str,
        default="unknown",
        help="Default class label to use when class information is not available",
    )
    parser.add_argument(
        "--include_datasets",
        type=str,
        default="simready,commercial,residential,vegetation",
        help="Comma-separated list of datasets to include",
    )


def split_dataset(metadata, seed=42):
    """
    Split dataset into train, validation, and test sets with 80-10-10 ratio,
    ensuring proportional class distribution across all material datasets.

    Args:
        metadata (pandas.DataFrame): DataFrame with metadata
        seed (int): Random seed for reproducibility

    Returns:
        pandas.DataFrame: Updated DataFrame with split information
    """

    np.random.seed(seed)
    random.seed(seed)

    metadata_copy = metadata.copy()

    metadata_copy["split"] = "train"

    classes = metadata_copy["class"].unique()

    large_classes = []
    small_classes = []

    for cls in classes:

        cls_indices = metadata_copy[metadata_copy["class"] == cls].index.tolist()
        if len(cls_indices) >= 10:
            large_classes.append(cls)
        else:
            small_classes.append(cls)

    for cls in large_classes:
        cls_indices = metadata_copy[metadata_copy["class"] == cls].index.tolist()
        random.shuffle(cls_indices)

        n_samples = len(cls_indices)
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)

        train_indices = cls_indices[:n_train]
        val_indices = cls_indices[n_train : n_train + n_val]
        test_indices = cls_indices[n_train + n_val :]

        metadata_copy.loc[train_indices, "split"] = "train"
        metadata_copy.loc[val_indices, "split"] = "val"
        metadata_copy.loc[test_indices, "split"] = "test"

    total_samples = len(metadata_copy)
    goal_train = int(0.8 * total_samples)
    goal_val = int(0.1 * total_samples)
    goal_test = total_samples - goal_train - goal_val

    current_train = (metadata_copy["split"] == "train").sum()
    current_val = (metadata_copy["split"] == "val").sum()
    current_test = (metadata_copy["split"] == "test").sum()

    small_indices = []
    for cls in small_classes:
        cls_indices = metadata_copy[metadata_copy["class"] == cls].index.tolist()
        small_indices.extend(cls_indices)

    random.shuffle(small_indices)

    need_train = max(0, goal_train - current_train)
    need_val = max(0, goal_val - current_val)
    need_test = max(0, goal_test - current_test)

    idx = 0
    while idx < len(small_indices):
        if need_train > 0:
            metadata_copy.loc[small_indices[idx], "split"] = "train"
            need_train -= 1
            idx += 1
        elif need_val > 0:
            metadata_copy.loc[small_indices[idx], "split"] = "val"
            need_val -= 1
            idx += 1
        elif need_test > 0:
            metadata_copy.loc[small_indices[idx], "split"] = "test"
            need_test -= 1
            idx += 1
        else:

            metadata_copy.loc[small_indices[idx:], "split"] = "train"
            break

    train_count = (metadata_copy["split"] == "train").sum()
    val_count = (metadata_copy["split"] == "val").sum()
    test_count = (metadata_copy["split"] == "test").sum()

    print(
        f"Dataset split: Train: {train_count} ({train_count/len(metadata_copy)*100:.1f}%), "
        f"Val: {val_count} ({val_count/len(metadata_copy)*100:.1f}%), "
        f"Test: {test_count} ({test_count/len(metadata_copy)*100:.1f}%)"
    )

    if small_classes:
        print("\nSmall class distribution across splits:")
        for cls in small_classes:
            cls_data = metadata_copy[metadata_copy["class"] == cls]
            cls_train = (cls_data["split"] == "train").sum()
            cls_val = (cls_data["split"] == "val").sum()
            cls_test = (cls_data["split"] == "test").sum()
            cls_total = len(cls_data)
            print(
                f"  - {cls} (total {cls_total}): Train: {cls_train}, Val: {cls_val}, Test: {cls_test}"
            )

    return metadata_copy


def get_simready_metadata(simready_dir, asset_info_path, default_class="unknown"):
    """
    Extract metadata from SimReady dataset.

    Args:
        simready_dir (str): Path to SimReady props directory
        asset_info_path (str): Path to asset_info.json file
        default_class (str): Default class to use if class information is not available

    Returns:
        list: List of metadata dictionaries
    """

    asset_class_mapping = SIMREADY_ASSET_CLASS_MAPPING

    if not asset_class_mapping and asset_info_path and os.path.exists(asset_info_path):
        try:
            with open(asset_info_path, "r") as f:
                asset_info = json.load(f)

            asset_class_mapping = {}
            for asset in asset_info:
                simple_name = asset.get("Simple Name")
                if simple_name and "Labels" in asset and "Class" in asset["Labels"]:
                    asset_class_mapping[simple_name] = asset["Labels"]["Class"]

            print(f"Loaded class information for {len(asset_class_mapping)} assets")
        except Exception as e:
            print(f"Error loading asset info: {e}")

    metadata = []
    props_root = Path(simready_dir).resolve()
    dataset_root = props_root.parent.parent
    for asset in discover_assets("simready", dataset_root):
        prop_name = asset.object_name
        sha256 = hashlib.sha256(prop_name.encode()).hexdigest()
        prop_class = asset_class_mapping.get(prop_name, default_class)
        metadata.append(
            {
                "sha256": sha256,
                "local_path": str(asset.path),
                "original_name": prop_name,
                "aesthetic_score": 1.0,
                "rendered": False,
                "class": prop_class,
                "dataset": "simready",
            }
        )

    return metadata


def _get_tree_metadata(dataset, dataset_dir, default_class):
    metadata = []
    for asset in discover_assets(dataset, dataset_dir):
        if asset_annotation_exclusion(asset.key) is not None:
            print(f"Skipping scene-level asset: {asset.relative_path}")
            continue
        object_name = asset.path.parent.name
        filename = asset.path.name
        sha256 = hashlib.sha256(f"{object_name}_{filename}".encode()).hexdigest()
        metadata.append(
            {
                "sha256": sha256,
                "local_path": str(asset.path),
                "original_name": f"{object_name}/{filename}",
                "aesthetic_score": 1.0,
                "rendered": False,
                "class": default_class,
                "dataset": dataset,
            }
        )
    return metadata


def get_commercial_metadata(commercial_dir, default_class="commercial"):
    return _get_tree_metadata("commercial", commercial_dir, default_class)


def get_residential_metadata(residential_dir, default_class="residential"):
    return _get_tree_metadata("residential", residential_dir, default_class)


def get_vegetation_metadata(vegetation_dir, default_class="vegetation"):
    return _get_tree_metadata("vegetation", vegetation_dir, default_class)


def get_metadata(
    simready_dir=None,
    commercial_dir=None,
    residential_dir=None,
    vegetation_dir=None,
    output_dir=None,
    asset_info_path=None,
    include_datasets="simready,commercial,residential,vegetation",
    seed=42,
    default_class="unknown",
    skip_split=False,
    **kwargs,
):
    """
    Extract metadata from all material datasets.

    Args:
        simready_dir (str): Path to SimReady props directory
        commercial_dir (str): Path to Commercial models directory
        residential_dir (str): Path to Residential models directory
        vegetation_dir (str): Path to Vegetation models directory
        output_dir (str): Output directory to save metadata
        asset_info_path (str): Path to asset_info.json file for SimReady
        include_datasets (str): Comma-separated list of datasets to include
        seed (int): Random seed for reproducibility
        default_class (str): Default class to use if class information is not available
        skip_split (bool): If True, skip dataset splitting

    Returns:
        pandas.DataFrame: Metadata dataframe with all datasets combined
    """

    set_seeds(seed)

    if simready_dir is None:
        simready_dir = SIMREADY_PROPS_DIR
    if commercial_dir is None:
        commercial_dir = COMMERCIAL_BASE_DIR
    if residential_dir is None:
        residential_dir = RESIDENTIAL_BASE_DIR
    if vegetation_dir is None:
        vegetation_dir = VEGETATION_BASE_DIR
    if asset_info_path is None:
        asset_info_path = SIMREADY_ASSET_INFO_PATH

    datasets = [d.strip() for d in include_datasets.split(",")]

    metadata = []

    if "simready" in datasets:
        print(f"Processing SimReady dataset from {simready_dir}")
        simready_metadata = get_simready_metadata(
            simready_dir, asset_info_path, default_class
        )
        metadata.extend(simready_metadata)
        print(f"Added {len(simready_metadata)} items from SimReady dataset")

    if "commercial" in datasets:
        print(f"Processing Commercial dataset from {commercial_dir}")
        commercial_metadata = get_commercial_metadata(commercial_dir)
        metadata.extend(commercial_metadata)
        print(f"Added {len(commercial_metadata)} items from Commercial dataset")

    if "residential" in datasets:
        print(f"Processing Residential dataset from {residential_dir}")
        residential_metadata = get_residential_metadata(residential_dir)
        metadata.extend(residential_metadata)
        print(f"Added {len(residential_metadata)} items from Residential dataset")

    if "vegetation" in datasets:
        print(f"Processing Vegetation dataset from {vegetation_dir}")
        vegetation_metadata = get_vegetation_metadata(vegetation_dir)
        metadata.extend(vegetation_metadata)
        print(f"Added {len(vegetation_metadata)} items from Vegetation dataset")

    df = pd.DataFrame(metadata)

    if df.empty:
        print("Warning: No metadata collected from any dataset")
        return df

    class_counts = df["class"].value_counts()
    print("\nClass distribution in combined dataset:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} ({count/len(df)*100:.1f}%)")

    dataset_counts = df["dataset"].value_counts()
    print("\nDataset distribution:")
    for dataset_name, count in dataset_counts.items():
        print(f"  - {dataset_name}: {count} ({count/len(df)*100:.1f}%)")

    if not skip_split:
        df = split_dataset(df, seed=seed)
    else:
        print("Skipping dataset splitting as requested")
        df["split"] = "train"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

        splits_dir = os.path.join(output_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)

        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            if not split_df.empty:
                split_df.to_csv(os.path.join(splits_dir, f"{split}.csv"), index=False)

        class_stats = df.groupby(["class", "split"]).size().unstack(fill_value=0)
        class_stats.to_csv(os.path.join(output_dir, "class_distribution.csv"))

        dataset_stats = df.groupby(["dataset", "split"]).size().unstack(fill_value=0)
        dataset_stats.to_csv(os.path.join(output_dir, "dataset_distribution.csv"))

    return df


def foreach_instance(metadata, output_dir, func, max_workers=8, desc="Processing"):
    """
    Process each instance in the combined dataset.

    Args:
        metadata (pandas.DataFrame): DataFrame with metadata
        output_dir (str): Output directory
        func (function): Function to apply to each instance
        max_workers (int): Number of parallel workers
        desc (str): Description for progress bar

    Returns:
        pandas.DataFrame: DataFrame with processing results
    """

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, row in metadata.iterrows():
            sha256 = row["sha256"]
            local_path = row["local_path"]
            dataset = row.get("dataset", "unknown")

            futures.append(executor.submit(func, local_path, sha256, dataset))

        for future in tqdm(futures, desc=desc, total=len(futures)):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error in worker: {e}")

    return pd.DataFrame.from_records(results)
