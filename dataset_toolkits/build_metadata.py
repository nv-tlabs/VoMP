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
import hashlib
import shutil
import sys
import time
import argparse
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
import utils3d

from dataset_toolkits.datasets import ABO, allmats, simready
from dataset_toolkits.dino import (
    DINO_FEATURE_DIM,
    DINO_MODEL,
    DINO_REPOSITORY,
    render_hash,
)
from dataset_toolkits.voxel_sampling import sample_voxel_rows
from dataset_toolkits.material_objects.voxelize import (
    load_material_annotations,
    voxel_outputs_are_complete,
)


def get_first_directory(path):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                return entry.name
    return None


def need_process(key):
    return key in opt.field or opt.field == ["all"]


def feature_file_is_complete(output_dir, model, sha256):
    feature_path = os.path.join(output_dir, "features", model, f"{sha256}.npz")
    material_path = os.path.join(
        output_dir, "voxels", f"{sha256}_with_materials.npz"
    )
    try:
        if model != DINO_MODEL:
            return False
        with np.load(feature_path, allow_pickle=False) as feature_archive:
            if not {
                "indices",
                "patchtokens",
                "voxel_row_indices",
                "model_name",
                "model_repository",
                "render_sha256",
            } <= set(feature_archive.files):
                return False
            indices = feature_archive["indices"]
            patchtokens = feature_archive["patchtokens"]
            voxel_rows = feature_archive["voxel_row_indices"]
            stored_model = feature_archive["model_name"]
            stored_repository = feature_archive["model_repository"]
            stored_render = feature_archive["render_sha256"]
        with np.load(material_path, allow_pickle=False) as material_archive:
            voxel_data = material_archive["voxel_data"]
            surface_mask = material_archive["is_surface"]
        render_dir = os.path.join(output_dir, "renders", sha256)
        with open(
            os.path.join(render_dir, "transforms.json"), "r", encoding="utf-8"
        ) as stream:
            render_metadata = json.load(stream)
        segment_order = render_metadata["mesh"]["segment_keys"]
        expected_render = render_hash(render_dir, render_metadata["frames"])

        full_count = len(voxel_data)
        if full_count == 0:
            return False
        expected_count = min(full_count, 72_000)
        if (
            indices.dtype != np.dtype(np.uint8)
            or indices.shape != (expected_count, 3)
            or patchtokens.dtype != np.dtype(np.float16)
            or patchtokens.ndim != 2
            or patchtokens.shape[0] != expected_count
            or patchtokens.shape[1] != DINO_FEATURE_DIM
            or not np.isfinite(patchtokens).all()
            or voxel_rows.dtype != np.dtype(np.int32)
            or voxel_rows.shape != (expected_count,)
            or np.any(voxel_rows < 0)
            or np.any(voxel_rows >= full_count)
            or expected_count > 1
            and np.any(np.diff(voxel_rows) <= 0)
            or stored_model.shape != ()
            or str(stored_model.item()) != model
            or stored_repository.shape != ()
            or str(stored_repository.item()) != DINO_REPOSITORY
            or stored_render.shape != ()
            or str(stored_render.item()) != expected_render
        ):
            return False
        if full_count <= 72_000 and not np.array_equal(
            voxel_rows, np.arange(full_count, dtype=np.int32)
        ):
            return False

        full_coords = np.floor(
            (
                np.column_stack(
                    (voxel_data["x"], voxel_data["y"], voxel_data["z"])
                ).astype(np.float64)
                + 0.5
            )
            * 64
        ).astype(np.int64)
        if not np.array_equal(indices, full_coords[voxel_rows].astype(np.uint8)):
            return False

        expected_rows = sample_voxel_rows(
            full_coords,
            voxel_data["segment_id"],
            surface_mask,
            segment_order,
        )
        return bool(np.array_equal(voxel_rows, expected_rows.astype(np.int32)))
    except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def normalized_mesh_is_complete(render_dir, record):
    if not isinstance(record, dict):
        return False
    vertex_count = record.get("vertex_count")
    triangle_count = record.get("triangle_count")
    segment_keys = record.get("segment_keys")
    segment_keys_sha256 = record.get("segment_keys_sha256")
    expected_sha256 = record.get("sha256")
    if (
        record.get("file_path") != "mesh.ply"
        or type(vertex_count) is not int
        or vertex_count < 3
        or type(triangle_count) is not int
        or triangle_count < 1
        or not isinstance(segment_keys, list)
        or not segment_keys
        or len(set(segment_keys)) != len(segment_keys)
        or not all(isinstance(key, str) and key for key in segment_keys)
        or segment_keys != sorted(segment_keys)
        or not isinstance(segment_keys_sha256, str)
        or len(segment_keys_sha256) != 64
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        return False
    path = os.path.join(render_dir, "mesh.ply")
    try:
        with open(path, "rb") as stream:
            header = stream.read(65536)
        marker = b"end_header\n"
        marker_offset = header.find(marker)
        if marker_offset < 0:
            return False
        header_size = marker_offset + len(marker)
        expected_segment_keys_sha256 = hashlib.sha256(
            json.dumps(
                segment_keys, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if segment_keys_sha256 != expected_segment_keys_sha256:
            return False
        expected_header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment exact composed render geometry in normalized object coordinates\n"
            f"comment segment_keys_sha256 {segment_keys_sha256}\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {triangle_count}\n"
            "property list uchar int vertex_indices\n"
            "property int segment_index\n"
            "end_header\n"
        ).encode("ascii")
        expected_size = header_size + vertex_count * 12 + triangle_count * 17
        if header[:header_size] != expected_header or os.path.getsize(path) != expected_size:
            return False
        face_dtype = np.dtype(
            [
                ("count", "u1"),
                ("indices", "<i4", (3,)),
                ("segment_index", "<i4"),
            ]
        )
        vertices = np.memmap(
            path,
            mode="r",
            dtype="<f4",
            offset=header_size,
            shape=(vertex_count, 3),
        )
        faces = np.memmap(
            path,
            mode="r",
            dtype=face_dtype,
            offset=header_size + vertex_count * 12,
            shape=(triangle_count,),
        )
        valid_payload = (
            np.isfinite(vertices).all()
            and not np.any(vertices < -0.500001)
            and not np.any(vertices > 0.500001)
            and not np.any(faces["count"] != 3)
            and int(faces["indices"].min()) >= 0
            and int(faces["indices"].max()) < vertex_count
            and np.array_equal(
                np.unique(faces["segment_index"]),
                np.arange(len(segment_keys), dtype=np.int32),
            )
        )
        del faces
        del vertices
        if not valid_payload:
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == expected_sha256
    except OSError:
        return False


def render_is_complete(
    render_dir, expected_views=150, expected_resolution=512, strict=False
):
    manifest_path = os.path.join(render_dir, "transforms.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            return False
        frames = manifest["frames"]
        if not isinstance(frames, list) or not frames:
            return False
        if strict:
            settings = manifest.get("render_settings")
            mesh = manifest.get("mesh")
            normalization = manifest.get("normalization")
            if (
                not isinstance(settings, dict)
                or settings.get("resolution") != expected_resolution
                or settings.get("num_views") != expected_views
                or settings.get("transparent_background") is not True
                or not isinstance(normalization, dict)
                or not isinstance(mesh, dict)
                or not normalized_mesh_is_complete(render_dir, mesh)
                or len(frames) != expected_views
            ):
                return False
        root = os.path.realpath(render_dir)
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                return False
            relative_path = frame.get("file_path")
            if not isinstance(relative_path, str) or not relative_path:
                return False
            if strict and relative_path != f"{index:03d}.png":
                return False
            path = os.path.realpath(os.path.join(root, relative_path))
            if os.path.commonpath((root, path)) != root or not os.path.isfile(path):
                return False
            if not strict:
                continue
            with open(path, "rb") as stream:
                header = stream.read(26)
                stream.seek(-12, os.SEEK_END)
                footer = stream.read(12)
            if (
                header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[12:16] != b"IHDR"
                or int.from_bytes(header[16:20], "big") != expected_resolution
                or int.from_bytes(header[20:24], "big") != expected_resolution
                or header[24:26] != b"\x08\x06"
                or footer[4:8] != b"IEND"
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


if __name__ == "__main__":
    dataset_utils = {"ABO": ABO, "allmats": allmats, "simready": simready}[sys.argv[1]]

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

        # Include newly added assets and keep the processing status of existing ones.
        if sys.argv[1] == "allmats":
            fresh_options = dict(opt)
            fresh_options["output_dir"] = None
            fresh_metadata = dataset_utils.get_metadata(**fresh_options)
            metadata.set_index("sha256", inplace=True)
            fresh_metadata.set_index("sha256", inplace=True)
            if "augmentation_index" in metadata.columns:
                augmentation_rows = metadata[
                    pd.to_numeric(metadata["augmentation_index"], errors="coerce")
                    .fillna(0)
                    .gt(0)
                ].copy()
            else:
                augmentation_rows = metadata.iloc[0:0].copy()
            state_columns = [
                column
                for column in metadata.columns
                if column
                in {"rendered", "voxelized", "num_voxels", "cond_rendered"}
                or column.startswith(("feature_", "latent_", "ss_latent_"))
            ]
            shared = fresh_metadata.index.intersection(metadata.index)
            for column in state_columns:
                if column not in fresh_metadata.columns:
                    fresh_metadata[column] = 0 if column == "num_voxels" else False
                fresh_metadata.loc[shared, column] = metadata.loc[shared, column]
            if not augmentation_rows.empty:
                fresh_metadata = pd.concat(
                    (fresh_metadata, augmentation_rows), axis=0, sort=False
                )
            metadata = fresh_metadata

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
                if metadata.index.name != "sha256":
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
                    if metadata.index.name != "sha256":
                        metadata.set_index("sha256", inplace=True)
                else:
                    raise e
        else:
            if metadata.index.name != "sha256":
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

    voxel_annotations = None
    if opt.from_file and sys.argv[1] == "allmats" and need_process("voxelized"):
        voxel_annotations = load_material_annotations(
            os.path.join(opt.output_dir, "material_annotations.json")
        )

    # build metadata from files
    if opt.from_file:
        with ThreadPoolExecutor(
            max_workers=min(32, os.cpu_count() or 1)
        ) as executor, tqdm(
            total=len(metadata), desc="Building metadata"
        ) as pbar:

            def worker(sha256):
                try:
                    if need_process("rendered"):
                        metadata.loc[sha256, "rendered"] = render_is_complete(
                            os.path.join(opt.output_dir, "renders", sha256),
                            strict=sys.argv[1] == "allmats",
                        )
                    if need_process("voxelized"):
                        valid, count = False, 0
                        if metadata.loc[sha256, "rendered"] == True:
                            if sys.argv[1] == "allmats":
                                valid, count = voxel_outputs_are_complete(
                                    opt.output_dir,
                                    str(sha256),
                                    str(metadata.loc[sha256, "local_path"]),
                                    str(metadata.loc[sha256, "dataset"]),
                                    voxel_annotations,
                                )
                            else:
                                try:
                                    pts = utils3d.io.read_ply(
                                        os.path.join(
                                            opt.output_dir,
                                            "voxels",
                                            f"{sha256}.ply",
                                        )
                                    )[0]
                                    valid, count = True, len(pts)
                                except (OSError, ValueError):
                                    pass
                        metadata.loc[sha256, "voxelized"] = valid
                        metadata.loc[sha256, "num_voxels"] = count
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
                        if need_process(f"feature_{model}"):
                            if sys.argv[1] == "allmats":
                                feature_complete = feature_file_is_complete(
                                    opt.output_dir, model, sha256
                                )
                            else:
                                feature_complete = os.path.exists(
                                    os.path.join(
                                        opt.output_dir,
                                        "features",
                                        model,
                                        f"{sha256}.npz",
                                    )
                                )
                            metadata.loc[sha256, f"feature_{model}"] = bool(
                                metadata.loc[sha256, "rendered"] == True
                                and metadata.loc[sha256, "voxelized"] == True
                                and feature_complete
                            )
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
                if sys.argv[1] == "allmats":
                    feature_complete = feature_file_is_complete(
                        opt.output_dir, model, sha256
                    )
                else:
                    feature_complete = os.path.exists(
                        os.path.join(
                            opt.output_dir, "features", model, f"{sha256}.npz"
                        )
                    )
                metadata.loc[sha256, f"feature_{model}"] = bool(
                    metadata.loc[sha256, "rendered"] == True
                    and metadata.loc[sha256, "voxelized"] == True
                    and feature_complete
                )

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
