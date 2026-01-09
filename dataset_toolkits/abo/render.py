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
import copy
import sys
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np

# Add current directory to path to import dataset modules
sys.path.insert(0, os.path.dirname(__file__))
import ABO500 as dataset_utils

# Import from the existing render.py utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import sphere_hammersley_sequence

BLENDER_LINK = (
    "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz"
)
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"


def _install_blender():
    """Install Blender if not already installed."""
    if not os.path.exists(BLENDER_PATH):
        os.system("sudo apt-get update")
        os.system(
            "sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6"
        )
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(
            f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}"
        )


def _render_glb(file_path, sha256, output_dir, num_views):
    """
    Render a GLB file from multiple viewpoints.

    Args:
        file_path (str): Path to the GLB file
        sha256 (str): SHA256 hash of the file
        output_dir (str): Directory to save renders
        num_views (int): Number of viewpoints to render

    Returns:
        dict: Result dictionary with rendering info
    """
    # Convert to absolute path to avoid issues with relative paths
    output_dir = os.path.abspath(output_dir)
    output_folder = os.path.join(output_dir, "renders", sha256)

    # Build camera parameters {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)

    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [
        {"yaw": y, "pitch": p, "radius": r, "fov": f}
        for y, p, r, f in zip(yaws, pitchs, radius, fov)
    ]

    # Construct Blender command
    blender_script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "blender_script", "render.py"
    )

    args = [
        BLENDER_PATH,
        "-b",
        "-P",
        blender_script_path,
        "--",
        "--views",
        json.dumps(views),
        "--object",
        os.path.expanduser(file_path),
        "--resolution",
        "512",
        "--output_folder",
        output_folder,
        "--engine",
        "CYCLES",
        "--save_mesh",
    ]

    try:
        # Execute Blender rendering
        result = call(args, stdout=DEVNULL, stderr=DEVNULL)

        # Check if rendering was successful
        if result == 0 and os.path.exists(
            os.path.join(output_folder, "transforms.json")
        ):
            return {"sha256": sha256, "rendered": True}
        else:
            print(f"Rendering failed for {sha256}")
            return {"sha256": sha256, "rendered": False}

    except Exception as e:
        print(f"Error rendering {file_path}: {e}")
        return {"sha256": sha256, "rendered": False}


def _render(file_path, sha256, output_dir=None, num_views=150):
    """Wrapper function for rendering."""
    return _render_glb(file_path, sha256, output_dir, num_views)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render ABO 500 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing metadata and where to save renders",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Specific instances to process (comma-separated or file path)",
    )
    parser.add_argument(
        "--num_views", type=int, default=150, help="Number of views to render"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rendering even if already processed"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N objects"
    )

    args = parser.parse_args()
    opt = edict(vars(args))

    # Create renders directory
    os.makedirs(os.path.join(opt.output_dir, "renders"), exist_ok=True)

    # Install Blender if needed
    print("Checking Blender installation...", flush=True)
    _install_blender()

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
        # Only process objects that have valid local paths
        metadata = metadata[metadata["local_path"].notna()]

        # Only process objects that haven't been rendered yet
        if "rendered" in metadata.columns and not opt.force:
            metadata = metadata[metadata["rendered"] == False]

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

    # Filter out objects that are already processed
    if not opt.force:
        for sha256 in copy.copy(metadata["sha256"].values):
            transforms_path = os.path.join(
                opt.output_dir, "renders", sha256, "transforms.json"
            )
            if os.path.exists(transforms_path):
                records.append({"sha256": sha256, "rendered": True})
                metadata = metadata[metadata["sha256"] != sha256]

    # Process remaining objects
    if len(metadata) > 0:
        func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views)
        rendered = dataset_utils.foreach_instance(
            metadata,
            opt.output_dir,
            func,
            max_workers=opt.max_workers,
            desc="Rendering objects",
        )

        # Combine results
        if len(records) > 0:
            rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])

        # Save results
        rendered.to_csv(
            os.path.join(opt.output_dir, f"rendered_{opt.rank}.csv"), index=False
        )

        print(f"Rendering complete. Results saved to rendered_{opt.rank}.csv")
    else:
        print("No objects to process.")
