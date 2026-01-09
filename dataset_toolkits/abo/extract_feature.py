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
import copy
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image

# Add current directory to path to import dataset modules
sys.path.insert(0, os.path.dirname(__file__))
import ABO500 as dataset_utils

torch.set_grad_enabled(False)


def get_data(frames, sha256, output_dir):
    """
    Load and preprocess rendered images for feature extraction.

    Args:
        frames (list): List of frame data from transforms.json
        sha256 (str): SHA256 hash of the object
        output_dir (str): Output directory containing renders

    Returns:
        list: List of processed image data
    """
    valid_data = []

    for view in frames:
        image_path = os.path.join(output_dir, "renders", sha256, view["file_path"])
        try:
            # Check if file exists before trying to open it
            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_path} not found, skipping")
                continue

            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        try:
            # Resize and normalize image
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]  # Apply alpha channel
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Extract camera parameters
            c2w = torch.tensor(view["transform_matrix"])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view["camera_angle_x"]
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(
                torch.tensor(fov), torch.tensor(fov)
            )

            valid_data.append(
                {"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics}
            )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    if len(valid_data) == 0:
        print(f"Warning: No valid images found for {sha256}")
    else:
        print(f"Loaded {len(valid_data)}/{len(frames)} valid images for {sha256}")

    return valid_data


def extract_features(
    file_path,
    sha256,
    output_dir=None,
    model=None,
    transform=None,
    batch_size=16,
    feature_name="dinov2_vitl14_reg",
):
    """
    Extract features for a single object.

    Args:
        file_path (str): Path to the GLB file (not used directly, but needed for interface)
        sha256 (str): SHA256 hash of the object
        output_dir (str): Output directory
        model: Pre-loaded feature extraction model
        transform: Image transformation pipeline
        batch_size (int): Batch size for processing
        feature_name (str): Name of the feature extraction method

    Returns:
        dict: Result dictionary with processing info
    """
    try:
        # Load transforms.json
        transforms_path = os.path.join(output_dir, "renders", sha256, "transforms.json")
        if not os.path.exists(transforms_path):
            print(f"transforms.json not found for {sha256}")
            return {"sha256": sha256, f"feature_{feature_name}": False}

        with open(transforms_path, "r") as f:
            metadata_json = json.load(f)

        frames = metadata_json["frames"]
        data = get_data(frames, sha256, output_dir)

        if len(data) == 0:
            print(f"Skipping {sha256}: no valid image data")
            return {"sha256": sha256, f"feature_{feature_name}": False}

        # Apply transform to images
        for datum in data:
            datum["image"] = transform(datum["image"])

        # Load voxel positions
        voxel_path = os.path.join(output_dir, "voxels", f"{sha256}.ply")
        if not os.path.exists(voxel_path):
            print(f"Voxel file not found for {sha256}")
            return {"sha256": sha256, f"feature_{feature_name}": False}

        positions = utils3d.io.read_ply(voxel_path)[0]
        positions = torch.from_numpy(positions).float().cuda()
        indices = ((positions + 0.5) * 64).long()
        # Clamp indices to valid range [0, 63] to handle floating point precision issues
        indices = torch.clamp(indices, 0, 63)

        n_views = len(data)
        n_patch = 518 // 14
        pack = {
            "indices": indices.cpu().numpy().astype(np.uint8),
        }

        patchtokens_lst = []
        uv_lst = []

        # Process in batches
        for i in range(0, n_views, batch_size):
            batch_data = data[i : i + batch_size]
            bs = len(batch_data)
            batch_images = torch.stack([d["image"] for d in batch_data]).cuda()
            batch_extrinsics = torch.stack([d["extrinsics"] for d in batch_data]).cuda()
            batch_intrinsics = torch.stack([d["intrinsics"] for d in batch_data]).cuda()

            # Extract features using the model
            features = model(batch_images, is_training=True)

            # Project 3D positions to 2D
            uv = (
                utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[
                    0
                ]
                * 2
                - 1
            )

            # Extract patch tokens
            patchtokens = (
                features["x_prenorm"][:, model.num_register_tokens + 1 :]
                .permute(0, 2, 1)
                .reshape(bs, 1024, n_patch, n_patch)
            )
            patchtokens_lst.append(patchtokens)
            uv_lst.append(uv)

        patchtokens = torch.cat(patchtokens_lst, dim=0)
        uv = torch.cat(uv_lst, dim=0)

        # Sample features at voxel positions
        pack["patchtokens"] = (
            F.grid_sample(
                patchtokens,
                uv.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(2)
            .permute(0, 2, 1)
            .cpu()
            .numpy()
        )
        pack["patchtokens"] = np.mean(pack["patchtokens"], axis=0).astype(np.float16)

        # Save features
        save_path = os.path.join(output_dir, "features", feature_name, f"{sha256}.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **pack)

        return {"sha256": sha256, f"feature_{feature_name}": True}

    except Exception as e:
        print(f"Error processing {sha256}: {e}")
        import traceback

        traceback.print_exc()
        return {"sha256": sha256, f"feature_{feature_name}": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for ABO 500 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing metadata and where to save features",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vitl14_reg",
        help="Feature extraction model",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Specific instances to process (comma-separated or file path)",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force feature extraction even if already processed",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N objects"
    )

    args = parser.parse_args()
    opt = edict(vars(args))

    feature_name = opt.model

    # Create features directory
    os.makedirs(os.path.join(opt.output_dir, "features", feature_name), exist_ok=True)

    # Load model
    print(f"Loading model: {opt.model}")
    dinov2_model = torch.hub.load("facebookresearch/dinov2", opt.model)
    dinov2_model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load metadata
    metadata_path = os.path.join(opt.output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise ValueError(f"metadata.csv not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Filter instances if specified
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]
    else:
        # Only process objects that have been rendered and voxelized
        if "rendered" in metadata.columns:
            metadata = metadata[metadata["rendered"] == True]
        if "voxelized" in metadata.columns:
            metadata = metadata[metadata["voxelized"] == True]

        # Only process objects that haven't had features extracted yet
        if f"feature_{feature_name}" in metadata.columns and not opt.force:
            metadata = metadata[metadata[f"feature_{feature_name}"] == False]

    # Apply distributed processing
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    # Apply limit if specified
    if opt.limit is not None:
        metadata = metadata.head(opt.limit)

    print(f"Processing {len(metadata)} objects...")

    # Track already processed objects
    records = []
    sha256s = list(metadata["sha256"].values)

    # Filter out objects that are already processed
    if not opt.force:
        for sha256 in copy.copy(sha256s):
            feature_path = os.path.join(
                opt.output_dir, "features", feature_name, f"{sha256}.npz"
            )
            if os.path.exists(feature_path):
                records.append({"sha256": sha256, f"feature_{feature_name}": True})
                sha256s.remove(sha256)

    # Filter out objects that don't have required prerequisite files
    initial_count = len(sha256s)
    filtered_sha256s = []

    for sha256 in sha256s:
        # Check for voxel file
        voxel_path = os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
        if not os.path.exists(voxel_path):
            print(f"Skipping {sha256}: voxel file not found")
            continue

        # Check for transforms.json
        transforms_path = os.path.join(
            opt.output_dir, "renders", sha256, "transforms.json"
        )
        if not os.path.exists(transforms_path):
            print(f"Skipping {sha256}: transforms.json not found")
            continue

        filtered_sha256s.append(sha256)

    sha256s = filtered_sha256s
    print(
        f"Filtered from {initial_count} to {len(sha256s)} objects with required files"
    )

    # Extract features for remaining objects
    if len(sha256s) > 0:
        for sha256 in tqdm(sha256s, desc="Extracting features"):
            # Get the file path (not used directly but needed for interface consistency)
            file_path = metadata[metadata["sha256"] == sha256]["local_path"].iloc[0]

            result = extract_features(
                file_path=file_path,
                sha256=sha256,
                output_dir=opt.output_dir,
                model=dinov2_model,
                transform=transform,
                batch_size=opt.batch_size,
                feature_name=feature_name,
            )

            if result is not None:
                records.append(result)

    # Save results
    if len(records) > 0:
        results_df = pd.DataFrame.from_records(records)
        results_df.to_csv(
            os.path.join(opt.output_dir, f"feature_{feature_name}_{opt.rank}.csv"),
            index=False,
        )
        print(
            f"Feature extraction complete. Results saved to feature_{feature_name}_{opt.rank}.csv"
        )
    else:
        print("No objects processed.")
