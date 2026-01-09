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
import shutil
import sys
import time
import importlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
import utils3d


def get_first_directory(path):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                return entry.name
    return None


def need_process(key):
    return key in opt.field or opt.field == ["all"]


if __name__ == "__main__":
    dataset_utils = importlib.import_module(f"dataset_toolkits.datasets.{sys.argv[1]}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the metadata"
    )
    parser.add_argument(
        "--field",
        type=str,
        default="all",
        help="Fields to process, separated by commas",
    )
    parser.add_argument(
        "--from_file",
        action="store_true",
        help="Build metadata from file instead of from records of processings."
        + "Useful when some processing fail to generate records but file already exists.",
    )
    parser.add_argument(
        "--force_update_class_split",
        action="store_true",
        help="Force updating class and split information even if metadata file exists",
    )
    parser.add_argument(
        "--skip_class_split_on_error",
        action="store_true",
        help="Skip updating class and split if an error occurs, instead of failing",
    )
    dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "merged_records"), exist_ok=True)

    opt.field = opt.field.split(",")

    timestamp = str(int(time.time()))

    # Check if metadata file exists
    metadata_exists = os.path.exists(os.path.join(opt.output_dir, "metadata.csv"))

    # Load or create metadata
    if metadata_exists:
        print("Loading previous metadata...")
        metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))

        # Check if class and split information needs to be updated
        requires_class_update = (
            "class" not in metadata.columns or opt.force_update_class_split
        )
        requires_split_update = (
            "split" not in metadata.columns or opt.force_update_class_split
        )

        if requires_class_update or requires_split_update:
            # Generate fresh metadata with class and split information
            print("Updating class and split information...")
            try:
                fresh_metadata = dataset_utils.get_metadata(**opt)

                # Set index on sha256 for both DataFrames
                metadata.set_index("sha256", inplace=True)
                fresh_metadata.set_index("sha256", inplace=True)

                # Update class information if needed
                if requires_class_update and "class" in fresh_metadata.columns:
                    if "class" not in metadata.columns:
                        metadata["class"] = "unknown"
                    metadata.update(fresh_metadata[["class"]])

                # Update split information if needed
                if requires_split_update and "split" in fresh_metadata.columns:
                    if "split" not in metadata.columns:
                        metadata["split"] = "train"  # Default value
                    metadata.update(fresh_metadata[["split"]])
            except Exception as e:
                if opt.skip_class_split_on_error:
                    print(f"Warning: Error updating class and split information: {e}")
                    print("Continuing with existing metadata...")
                    if "class" not in metadata.columns:
                        metadata["class"] = "unknown"
                    if "split" not in metadata.columns:
                        metadata["split"] = "train"
                    metadata.set_index("sha256", inplace=True)
                else:
                    raise e
        else:
            metadata.set_index("sha256", inplace=True)
    else:
        # Create new metadata with all required information
        print("Creating new metadata...")
        try:
            metadata = dataset_utils.get_metadata(**opt)
            metadata.set_index("sha256", inplace=True)
        except Exception as e:
            if opt.skip_class_split_on_error:
                print(
                    f"Warning: Error creating metadata with class and split information: {e}"
                )
                print("Creating basic metadata without class and split information...")
                metadata = dataset_utils.get_metadata(skip_split=True, **opt)
                metadata.set_index("sha256", inplace=True)
                if "class" not in metadata.columns:
                    metadata["class"] = "unknown"
                if "split" not in metadata.columns:
                    metadata["split"] = "train"
            else:
                raise e

    # merge downloaded
    df_files = [
        f
        for f in os.listdir(opt.output_dir)
        if f.startswith("downloaded_") and f.endswith(".csv")
    ]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        if "local_path" in metadata.columns:
            metadata.update(df, overwrite=True)
        else:
            metadata = metadata.join(df, on="sha256", how="left")
        for f in df_files:
            shutil.move(
                os.path.join(opt.output_dir, f),
                os.path.join(opt.output_dir, "merged_records", f"{timestamp}_{f}"),
            )

    # detect models
    image_models = []
    if os.path.exists(os.path.join(opt.output_dir, "features")):
        image_models = os.listdir(os.path.join(opt.output_dir, "features"))
    latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, "latents")):
        latent_models = os.listdir(os.path.join(opt.output_dir, "latents"))
    ss_latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, "ss_latents")):
        ss_latent_models = os.listdir(os.path.join(opt.output_dir, "ss_latents"))
    print(f"Image models: {image_models}")
    print(f"Latent models: {latent_models}")
    print(f"Sparse Structure latent models: {ss_latent_models}")

    if "rendered" not in metadata.columns:
        metadata["rendered"] = [False] * len(metadata)
    if "voxelized" not in metadata.columns:
        metadata["voxelized"] = [False] * len(metadata)
    if "num_voxels" not in metadata.columns:
        metadata["num_voxels"] = [0] * len(metadata)
    if "cond_rendered" not in metadata.columns:
        metadata["cond_rendered"] = [False] * len(metadata)
    for model in image_models:
        if f"feature_{model}" not in metadata.columns:
            metadata[f"feature_{model}"] = [False] * len(metadata)
    for model in latent_models:
        if f"latent_{model}" not in metadata.columns:
            metadata[f"latent_{model}"] = [False] * len(metadata)
    for model in ss_latent_models:
        if f"ss_latent_{model}" not in metadata.columns:
            metadata[f"ss_latent_{model}"] = [False] * len(metadata)

    # merge rendered
    df_files = [
        f
        for f in os.listdir(opt.output_dir)
        if f.startswith("rendered_") and f.endswith(".csv")
    ]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(
                os.path.join(opt.output_dir, f),
                os.path.join(opt.output_dir, "merged_records", f"{timestamp}_{f}"),
            )

    # merge voxelized
    df_files = [
        f
        for f in os.listdir(opt.output_dir)
        if f.startswith("voxelized_") and f.endswith(".csv")
    ]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(
                os.path.join(opt.output_dir, f),
                os.path.join(opt.output_dir, "merged_records", f"{timestamp}_{f}"),
            )

    # merge cond_rendered
    df_files = [
        f
        for f in os.listdir(opt.output_dir)
        if f.startswith("cond_rendered_") and f.endswith(".csv")
    ]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index("sha256", inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(
                os.path.join(opt.output_dir, f),
                os.path.join(opt.output_dir, "merged_records", f"{timestamp}_{f}"),
            )

    # merge features
    for model in image_models:
        df_files = [
            f
            for f in os.listdir(opt.output_dir)
            if f.startswith(f"feature_{model}_") and f.endswith(".csv")
        ]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index("sha256", inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(
                    os.path.join(opt.output_dir, f),
                    os.path.join(opt.output_dir, "merged_records", f"{timestamp}_{f}"),
                )

    # merge latents
    for model in latent_models:
        df_files = [
            f
            for f in os.listdir(opt.output_dir)
            if f.startswith(f"latent_{model}_") and f.endswith(".csv")
        ]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index("sha256", inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(
                    os.path.join(opt.output_dir, f),
                    os.path.join(opt.output_dir, "merged_records", f"{timestamp}_{f}"),
                )

    # merge sparse structure latents
    for model in ss_latent_models:
        df_files = [
            f
            for f in os.listdir(opt.output_dir)
            if f.startswith(f"ss_latent_{model}_") and f.endswith(".csv")
        ]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index("sha256", inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(
                    os.path.join(opt.output_dir, f),
                    os.path.join(opt.output_dir, "merged_records", f"{timestamp}_{f}"),
                )

    # build metadata from files
    if opt.from_file:
        with (
            ThreadPoolExecutor(max_workers=os.cpu_count()) as executor,
            tqdm(total=len(metadata), desc="Building metadata") as pbar,
        ):

            def worker(sha256):
                try:
                    if (
                        need_process("rendered")
                        and metadata.loc[sha256, "rendered"] == False
                        and os.path.exists(
                            os.path.join(
                                opt.output_dir, "renders", sha256, "transforms.json"
                            )
                        )
                    ):
                        metadata.loc[sha256, "rendered"] = True
                    if (
                        need_process("voxelized")
                        and metadata.loc[sha256, "rendered"] == True
                        and metadata.loc[sha256, "voxelized"] == False
                        and os.path.exists(
                            os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
                        )
                    ):
                        try:
                            pts = utils3d.io.read_ply(
                                os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
                            )[0]
                            metadata.loc[sha256, "voxelized"] = True
                            metadata.loc[sha256, "num_voxels"] = len(pts)
                        except Exception as e:
                            pass
                    if (
                        need_process("cond_rendered")
                        and metadata.loc[sha256, "cond_rendered"] == False
                        and os.path.exists(
                            os.path.join(
                                opt.output_dir,
                                "renders_cond",
                                sha256,
                                "transforms.json",
                            )
                        )
                    ):
                        metadata.loc[sha256, "cond_rendered"] = True
                    for model in image_models:
                        if (
                            need_process(f"feature_{model}")
                            and metadata.loc[sha256, f"feature_{model}"] == False
                            and metadata.loc[sha256, "rendered"] == True
                            and metadata.loc[sha256, "voxelized"] == True
                            and os.path.exists(
                                os.path.join(
                                    opt.output_dir, "features", model, f"{sha256}.npz"
                                )
                            )
                        ):
                            metadata.loc[sha256, f"feature_{model}"] = True
                    for model in latent_models:
                        if (
                            need_process(f"latent_{model}")
                            and metadata.loc[sha256, f"latent_{model}"] == False
                            and metadata.loc[sha256, "rendered"] == True
                            and metadata.loc[sha256, "voxelized"] == True
                            and os.path.exists(
                                os.path.join(
                                    opt.output_dir, "latents", model, f"{sha256}.npz"
                                )
                            )
                        ):
                            metadata.loc[sha256, f"latent_{model}"] = True
                    for model in ss_latent_models:
                        if (
                            need_process(f"ss_latent_{model}")
                            and metadata.loc[sha256, f"ss_latent_{model}"] == False
                            and metadata.loc[sha256, "voxelized"] == True
                            and os.path.exists(
                                os.path.join(
                                    opt.output_dir, "ss_latents", model, f"{sha256}.npz"
                                )
                            )
                        ):
                            metadata.loc[sha256, f"ss_latent_{model}"] = True
                    pbar.update()
                except Exception as e:
                    print(f"Error processing {sha256}: {e}")
                    pbar.update()

            executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    # Save dataset splits if we have split information
    if "split" in metadata.columns:
        os.makedirs(os.path.join(opt.output_dir, "splits"), exist_ok=True)
        # Reset index to include sha256 in the exported files
        metadata_export = metadata.reset_index()
        for split in ["train", "val", "test"]:
            split_df = metadata_export[metadata_export["split"] == split]
            if not split_df.empty:
                split_df.to_csv(
                    os.path.join(opt.output_dir, "splits", f"{split}.csv"), index=False
                )

    # statistics
    metadata.to_csv(os.path.join(opt.output_dir, "metadata.csv"))
    num_downloaded = (
        metadata["local_path"].count() if "local_path" in metadata.columns else 0
    )

    # If from_file is True, update metadata to reflect actual files on disk before writing statistics
    if opt.from_file:
        print("Updating metadata to reflect actual files on disk...")
        for model in image_models:
            for sha256 in metadata.index:
                actual_exists = os.path.exists(
                    os.path.join(opt.output_dir, "features", model, f"{sha256}.npz")
                )
                metadata.loc[sha256, f"feature_{model}"] = actual_exists

        for model in latent_models:
            for sha256 in metadata.index:
                actual_exists = os.path.exists(
                    os.path.join(opt.output_dir, "latents", model, f"{sha256}.npz")
                )
                metadata.loc[sha256, f"latent_{model}"] = actual_exists

        for model in ss_latent_models:
            for sha256 in metadata.index:
                actual_exists = os.path.exists(
                    os.path.join(opt.output_dir, "ss_latents", model, f"{sha256}.npz")
                )
                metadata.loc[sha256, f"ss_latent_{model}"] = actual_exists

        # Save updated metadata
        metadata.to_csv(os.path.join(opt.output_dir, "metadata.csv"))

    with open(os.path.join(opt.output_dir, "statistics.txt"), "w") as f:
        f.write("Statistics:\n")
        f.write(f"  - Number of assets: {len(metadata)}\n")
        f.write(f"  - Number of assets downloaded: {num_downloaded}\n")
        f.write(f'  - Number of assets rendered: {metadata["rendered"].sum()}\n')
        f.write(f'  - Number of assets voxelized: {metadata["voxelized"].sum()}\n')
        if len(image_models) != 0:
            f.write(f"  - Number of assets with image features extracted:\n")
            for model in image_models:
                # Always use metadata counts since they're now accurate when from_file=True
                f.write(f'    - {model}: {metadata[f"feature_{model}"].sum()}\n')
        if len(latent_models) != 0:
            f.write(f"  - Number of assets with latents extracted:\n")
            for model in latent_models:
                f.write(f'    - {model}: {metadata[f"latent_{model}"].sum()}\n')
        if len(ss_latent_models) != 0:
            f.write(f"  - Number of assets with sparse structure latents extracted:\n")
            for model in ss_latent_models:
                f.write(f'    - {model}: {metadata[f"ss_latent_{model}"].sum()}\n')

        # Only report captions if the column exists (it may not for Gaussian splats)
        if "captions" in metadata.columns:
            f.write(
                f'  - Number of assets with captions: {metadata["captions"].count()}\n'
            )
        else:
            f.write(
                f"  - Number of assets with captions: N/A (no caption data available)\n"
            )

        f.write(
            f'  - Number of assets with image conditions: {metadata["cond_rendered"].sum()}\n'
        )

        # Add class distribution statistics
        if "class" in metadata.columns:
            f.write("\nClass distribution:\n")
            class_counts = metadata["class"].value_counts()
            for class_name, count in class_counts.items():
                f.write(f"  - {class_name}: {count} ({count/len(metadata)*100:.1f}%)\n")

        # Add split statistics if split column exists
        if "split" in metadata.columns:
            f.write("\nDataset splits:\n")
            split_counts = metadata["split"].value_counts()
            for split_name, count in split_counts.items():
                f.write(f"  - {split_name}: {count} ({count/len(metadata)*100:.1f}%)\n")

            # Add class distribution per split if both columns exist
            if "class" in metadata.columns:
                f.write("\nClass distribution per split:\n")
                # Reset index to allow cross-tabulation
                metadata_reset = metadata.reset_index()
                # For each split, show class distribution
                for split_name in ["train", "val", "test"]:
                    if split_name in split_counts:
                        f.write(f"  {split_name.upper()}:\n")
                        split_data = metadata_reset[
                            metadata_reset["split"] == split_name
                        ]
                        class_in_split = split_data["class"].value_counts()
                        for class_name, count in class_in_split.items():
                            f.write(
                                f"    - {class_name}: {count} ({count/len(split_data)*100:.1f}%)\n"
                            )

    with open(os.path.join(opt.output_dir, "statistics.txt"), "r") as f:
        print(f.read())
