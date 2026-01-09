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
import sys
import argparse
import pandas as pd
from easydict import EasyDict as edict

# Add current directory to path to import dataset modules
sys.path.insert(0, os.path.dirname(__file__))

# Import the local ABO500 module directly
import ABO500 as dataset_utils


def main():
    parser = argparse.ArgumentParser(description="Build metadata for ABO 500 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the metadata and processed files",
    )

    # Add dataset-specific arguments
    dataset_utils.add_args(parser)

    args = parser.parse_args()
    opt = edict(vars(args))

    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)

    # Get metadata
    print("Building metadata for ABO 500 dataset...")
    metadata = dataset_utils.get_metadata(**opt)

    # Add default columns for tracking processing status
    metadata["rendered"] = False
    metadata["voxelized"] = False
    metadata["feature_dinov2_vitl14_reg"] = False

    # Check for existing processed files and update flags
    for idx, row in metadata.iterrows():
        sha256 = row["sha256"]

        # Check if voxel file exists
        voxel_path = os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
        if os.path.exists(voxel_path):
            metadata.at[idx, "voxelized"] = True

        # Check if render file exists (transforms.json)
        render_path = os.path.join(opt.output_dir, "renders", sha256, "transforms.json")
        if os.path.exists(render_path):
            metadata.at[idx, "rendered"] = True

        # Check if feature file exists
        feature_path = os.path.join(
            opt.output_dir, "features", "dinov2_vitl14_reg", f"{sha256}.npz"
        )
        if os.path.exists(feature_path):
            metadata.at[idx, "feature_dinov2_vitl14_reg"] = True

    # Save metadata
    metadata_path = os.path.join(opt.output_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    print(f"Metadata saved to {metadata_path}")
    print(f"Total objects: {len(metadata)}")
    print(f"Objects by split:")
    if "split" in metadata.columns:
        print(metadata["split"].value_counts())

    # Also save a summary file
    summary = {
        "total_objects": len(metadata),
        "dataset": "ABO500",
        "splits": (
            metadata["split"].value_counts().to_dict()
            if "split" in metadata.columns
            else {}
        ),
        "output_dir": opt.output_dir,
    }

    import json

    with open(os.path.join(opt.output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Dataset summary saved to dataset_summary.json")


if __name__ == "__main__":
    main()
