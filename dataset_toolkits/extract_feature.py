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
import importlib
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

torch.set_grad_enabled(False)


def get_data(frames, sha256):
    valid_data = []

    for view in frames:
        image_path = os.path.join(opt.output_dir, "renders", sha256, view["file_path"])
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
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the metadata"
    )
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vitl14_reg",
        help="Feature extraction model",
    )
    parser.add_argument(
        "--instances", type=str, default=None, help="Instances to process"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force feature extraction even if feature files already exist",
    )
    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, "features", feature_name), exist_ok=True)

    # load model
    dinov2_model = torch.hub.load("facebookresearch/dinov2", opt.model)
    dinov2_model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    n_patch = 518 // 14

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))
    else:
        raise ValueError("metadata.csv not found")
    if opt.instances is not None:
        with open(opt.instances, "r") as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata["sha256"].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[
                metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score
            ]
        if f"feature_{feature_name}" in metadata.columns and not opt.force:
            metadata = metadata[metadata[f"feature_{feature_name}"] == False]
        metadata = metadata[metadata["voxelized"] == True]
        metadata = metadata[metadata["rendered"] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata["sha256"].values)
    if not opt.force:
        for sha256 in copy.copy(sha256s):
            if os.path.exists(
                os.path.join(opt.output_dir, "features", feature_name, f"{sha256}.npz")
            ):
                records.append({"sha256": sha256, f"feature_{feature_name}": True})
                sha256s.remove(sha256)
    else:
        print(
            f"Force mode enabled. Processing all {len(sha256s)} objects regardless of existing features."
        )

    # filter out objects that don't have voxel files
    initial_count = len(sha256s)
    sha256s_with_voxels = []
    for sha256 in sha256s:
        voxel_path = os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
        if os.path.exists(voxel_path):
            sha256s_with_voxels.append(sha256)
        else:
            print(f"Skipping {sha256}: voxel file not found at {voxel_path}")

    sha256s = sha256s_with_voxels
    print(f"Filtered from {initial_count} to {len(sha256s)} objects with voxel files")

    # extract features
    for sha256 in tqdm(sha256s, desc="Extracting features"):
        try:
            # Load data
            with open(
                os.path.join(opt.output_dir, "renders", sha256, "transforms.json"),
                "r",
            ) as f:
                metadata_json = json.load(f)
            frames = metadata_json["frames"]
            data = get_data(frames, sha256)

            if len(data) == 0:
                print(f"Skipping {sha256}: no valid image data")
                continue

            # Apply transform to images
            for datum in data:
                datum["image"] = transform(datum["image"])

            # Load positions
            positions = utils3d.io.read_ply(
                os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
            )[0]
            positions = torch.from_numpy(positions).float().cuda()
            indices = ((positions + 0.5) * 64).long()
            # Clamp indices to valid range [0, 63] to handle floating point precision issues
            indices = torch.clamp(indices, 0, 63)

            n_views = len(data)
            N = positions.shape[0]
            pack = {
                "indices": indices.cpu().numpy().astype(np.uint8),
            }

            patchtokens_lst = []
            uv_lst = []

            # Process in batches
            for i in range(0, n_views, opt.batch_size):
                batch_data = data[i : i + opt.batch_size]
                bs = len(batch_data)
                batch_images = torch.stack([d["image"] for d in batch_data]).cuda()
                batch_extrinsics = torch.stack(
                    [d["extrinsics"] for d in batch_data]
                ).cuda()
                batch_intrinsics = torch.stack(
                    [d["intrinsics"] for d in batch_data]
                ).cuda()
                features = dinov2_model(batch_images, is_training=True)
                uv = (
                    utils3d.torch.project_cv(
                        positions, batch_extrinsics, batch_intrinsics
                    )[0]
                    * 2
                    - 1
                )
                patchtokens = (
                    features["x_prenorm"][:, dinov2_model.num_register_tokens + 1 :]
                    .permute(0, 2, 1)
                    .reshape(bs, 1024, n_patch, n_patch)
                )
                patchtokens_lst.append(patchtokens)
                uv_lst.append(uv)

            patchtokens = torch.cat(patchtokens_lst, dim=0)
            uv = torch.cat(uv_lst, dim=0)

            # Save features
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
            pack["patchtokens"] = np.mean(pack["patchtokens"], axis=0).astype(
                np.float16
            )
            save_path = os.path.join(
                opt.output_dir, "features", feature_name, f"{sha256}.npz"
            )
            np.savez_compressed(save_path, **pack)
            records.append({"sha256": sha256, f"feature_{feature_name}": True})

        except Exception as e:
            print(f"Error processing {sha256}: {e}")
            continue

    records = pd.DataFrame.from_records(records)
    records.to_csv(
        os.path.join(opt.output_dir, f"feature_{feature_name}_{opt.rank}.csv"),
        index=False,
    )
