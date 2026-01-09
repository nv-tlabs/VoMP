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
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--abo_500_dir",
        type=str,
        default="/home/rdagli/code/datasets/abo_500",
        help="Path to the ABO 500 dataset directory",
    )
    parser.add_argument(
        "--abo_3d_dir",
        type=str,
        default="/home/rdagli/code/datasets/abo-3dmodels/3dmodels",
        help="Path to the ABO 3D models directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N objects for testing",
    )


def get_file_hash(file_path):
    """Get SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_metadata(abo_500_dir, abo_3d_dir, split="all", limit=None, **kwargs):
    """Get metadata for ABO 500 dataset."""
    splits_path = os.path.join(abo_500_dir, "splits.json")

    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file not found at {splits_path}")

    with open(splits_path, "r") as f:
        splits_data = json.load(f)

    if split == "all":
        object_ids = splits_data["train"] + splits_data["val"] + splits_data["test"]
    else:
        object_ids = splits_data[split]

    # Apply limit if specified
    if limit is not None:
        object_ids = object_ids[:limit]

    print(f"Processing {len(object_ids)} objects from {split} split")

    # Create metadata records
    metadata_records = []
    missing_files = []

    for object_id in tqdm(object_ids, desc="Building metadata"):
        # Extract base ID (remove suffix after underscore if present)
        base_id = object_id.split("_")[0]

        # Search for GLB file - try multiple patterns and locations
        glb_path = None

        # Pattern 1: Try with base_id in the directory based on first character
        first_char = base_id[0]
        candidate_path = os.path.join(
            abo_3d_dir, "original", first_char, f"{base_id}.glb"
        )
        if os.path.exists(candidate_path):
            glb_path = candidate_path
        else:
            # Pattern 2: Try with full object_id (without underscore splitting)
            first_char_full = object_id[0]
            candidate_path = os.path.join(
                abo_3d_dir, "original", first_char_full, f"{object_id}.glb"
            )
            if os.path.exists(candidate_path):
                glb_path = candidate_path
            else:
                # Pattern 3: Search in all directories for the base_id
                for dir_name in os.listdir(os.path.join(abo_3d_dir, "original")):
                    dir_path = os.path.join(abo_3d_dir, "original", dir_name)
                    if os.path.isdir(dir_path):
                        candidate_path = os.path.join(dir_path, f"{base_id}.glb")
                        if os.path.exists(candidate_path):
                            glb_path = candidate_path
                            break
                        # Also try the full object_id
                        candidate_path = os.path.join(dir_path, f"{object_id}.glb")
                        if os.path.exists(candidate_path):
                            glb_path = candidate_path
                            break

        if glb_path and os.path.exists(glb_path):
            # Get file hash
            try:
                sha256 = get_file_hash(glb_path)
                metadata_records.append(
                    {
                        "object_id": object_id,
                        "sha256": sha256,
                        "local_path": glb_path,
                        "file_type": "glb",
                        "split": split,
                        "dataset": "ABO500",
                    }
                )
            except Exception as e:
                print(f"Error processing {object_id}: {e}")
                missing_files.append(object_id)
        else:
            print(
                f"Warning: GLB file not found for {object_id} (tried base_id: {base_id})"
            )
            missing_files.append(object_id)

    if missing_files:
        print(f"Warning: {len(missing_files)} objects have missing GLB files")

    metadata = pd.DataFrame(metadata_records)
    return metadata


def download(metadata, output_dir, **kwargs):
    """For ABO 500, files are already downloaded, so just return local paths."""
    download_records = []

    for _, row in metadata.iterrows():
        download_records.append(
            {"sha256": row["sha256"], "local_path": row["local_path"]}
        )

    return pd.DataFrame(download_records)


def foreach_instance(
    metadata, output_dir, func, max_workers=None, desc="Processing objects"
) -> pd.DataFrame:
    """Process each instance in the metadata."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    # Convert to list of records
    metadata_records = metadata.to_dict("records")

    # Processing objects
    records = []
    max_workers = max_workers or os.cpu_count()

    try:
        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(total=len(metadata_records), desc=desc) as pbar,
        ):

            def worker(metadatum):
                try:
                    local_path = metadatum["local_path"]
                    sha256 = metadatum["sha256"]
                    record = func(local_path, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()

            executor.map(worker, metadata_records)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")

    return pd.DataFrame.from_records(records)
