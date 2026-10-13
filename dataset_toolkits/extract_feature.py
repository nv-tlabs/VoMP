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

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import utils3d

from dataset_toolkits.dino import (
    DINO_FEATURE_DIM,
    DINO_MODEL,
    DINO_REPOSITORY,
    render_hash,
)
from dataset_toolkits.voxel_sampling import MAX_DINO_VOXELS, sample_voxel_rows


IMAGE_SIZE = 518
GRID_RESOLUTION = 64

def load_views_from_directory(render_dir: str | Path, frames: list[dict]) -> list[dict]:
    """Resize RGBA renders and load their cameras for DINO feature extraction."""
    render_dir = Path(render_dir)
    views = []
    for frame in frames:
        image_path = render_dir / frame["file_path"]
        with Image.open(image_path) as source_image:
            if source_image.mode != "RGBA":
                raise ValueError(f"render is not RGBA: {image_path}")
            image_array = np.asarray(
                source_image.resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0

        alpha = image_array[:, :, 3]
        image = image_array[:, :, :3] * alpha[:, :, None]
        camera_to_world = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        if camera_to_world.shape != (4, 4) or not torch.isfinite(camera_to_world).all():
            raise ValueError(f"invalid camera transform in {image_path}")
        camera_to_world[:3, 1:3] *= -1
        field_of_view = torch.tensor(frame["camera_angle_x"], dtype=torch.float32)
        if not torch.isfinite(field_of_view) or not 0 < field_of_view < np.pi:
            raise ValueError(f"invalid camera field of view in {image_path}")

        views.append(
            {
                "image": torch.from_numpy(image).permute(2, 0, 1),
                "alpha": torch.from_numpy(alpha.copy()).unsqueeze(0),
                "extrinsics": torch.inverse(camera_to_world),
                "intrinsics": utils3d.torch.intrinsics_from_fov_xy(
                    field_of_view, field_of_view
                ),
            }
        )
    return views


def load_views(output_dir: str, frames: list[dict], sha256: str) -> list[dict]:
    return load_views_from_directory(Path(output_dir) / "renders" / sha256, frames)


def load_sampled_voxels(
    output_dir: str,
    sha256: str,
    segment_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voxel_dir = Path(output_dir) / "voxels"
    positions = np.asarray(
        utils3d.io.read_ply(str(voxel_dir / f"{sha256}.ply"))[0],
        dtype=np.float32,
    )
    material_path = voxel_dir / f"{sha256}_with_materials.npz"
    if not material_path.is_file():
        grid_coords = np.floor(
            (positions.astype(np.float64) + 0.5) * GRID_RESOLUTION
        ).astype(np.int64)
        grid_coords = np.clip(grid_coords, 0, GRID_RESOLUTION - 1)
        rows = np.arange(len(positions), dtype=np.int64)
        return positions, grid_coords, rows

    with np.load(material_path, allow_pickle=False) as archive:
        voxel_data = archive["voxel_data"]
        surface_mask = archive["is_surface"]

    if positions.shape != (len(positions), 3) or len(voxel_data) != len(positions):
        raise ValueError("voxel PLY and material rows do not align")
    material_positions = np.column_stack(
        (voxel_data["x"], voxel_data["y"], voxel_data["z"])
    )
    if not np.array_equal(positions, material_positions):
        raise ValueError("voxel PLY and material coordinates do not match")

    grid_coords = np.floor((positions.astype(np.float64) + 0.5) * GRID_RESOLUTION).astype(
        np.int64
    )
    expected_positions = (
        (grid_coords.astype(np.float64) + 0.5) / GRID_RESOLUTION - 0.5
    ).astype(np.float32)
    if (
        np.any(grid_coords < 0)
        or np.any(grid_coords >= GRID_RESOLUTION)
        or not np.array_equal(positions, expected_positions)
    ):
        raise ValueError("voxel coordinates are not exact 64^3 cell centers")

    selected_rows = sample_voxel_rows(
        grid_coords,
        voxel_data["segment_id"],
        surface_mask,
        segment_order,
        MAX_DINO_VOXELS,
    )
    # Keep the row IDs to align sampled features with the full material grid.
    return positions[selected_rows], grid_coords[selected_rows], selected_rows


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    views: list[dict],
    positions: np.ndarray,
    transform: transforms.Compose,
    batch_size: int,
    voxel_chunk_size: int,
) -> np.ndarray:
    positions_gpu = torch.from_numpy(positions).float().cuda()
    feature_sum = None
    feature_weight = torch.zeros(len(positions), dtype=torch.float32)

    for view_start in range(0, len(views), batch_size):
        batch = views[view_start : view_start + batch_size]
        images = torch.stack([transform(view["image"]) for view in batch]).cuda()
        alpha = torch.stack([view["alpha"] for view in batch]).cuda()
        extrinsics = torch.stack([view["extrinsics"] for view in batch]).cuda()
        intrinsics = torch.stack([view["intrinsics"] for view in batch]).cuda()

        model_output = model(images, is_training=True)
        # Skip class and register tokens; only patch tokens have image locations.
        patchtokens = model_output["x_prenorm"][:, model.num_register_tokens + 1 :]
        patch_count = patchtokens.shape[1]
        patch_side = int(round(patch_count**0.5))
        if patch_side * patch_side != patch_count:
            raise ValueError("DINO patch token grid is not square")
        patchtokens = patchtokens.permute(0, 2, 1).reshape(
            len(batch), patchtokens.shape[2], patch_side, patch_side
        )
        if feature_sum is None:
            feature_sum = torch.zeros(
                (len(positions), patchtokens.shape[1]), dtype=torch.float32
            )

        # Chunk the projections to limit GPU memory use.
        for voxel_start in range(0, len(positions), voxel_chunk_size):
            voxel_end = min(voxel_start + voxel_chunk_size, len(positions))
            projection = utils3d.torch.project_cv(
                positions_gpu[voxel_start:voxel_end], extrinsics, intrinsics
            )
            uv = projection[0] * 2 - 1
            depth = projection[1]
            in_frame = (
                (depth > 0)
                & (uv[:, :, 0] >= -1)
                & (uv[:, :, 0] <= 1)
                & (uv[:, :, 1] >= -1)
                & (uv[:, :, 1] <= 1)
            )
            alpha_weight = F.grid_sample(
                alpha,
                uv.unsqueeze(2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(3).squeeze(1)
            # Ignore background pixels when averaging features across views.
            weight = alpha_weight * in_frame
            sampled = F.grid_sample(
                patchtokens,
                uv.unsqueeze(2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(3).permute(0, 2, 1)
            feature_sum[voxel_start:voxel_end] += (
                sampled * weight.unsqueeze(-1)
            ).sum(dim=0).cpu()
            feature_weight[voxel_start:voxel_end] += weight.sum(dim=0).cpu()

        del model_output, patchtokens, images, alpha, extrinsics, intrinsics

    if feature_sum is None or torch.any(feature_weight <= 0):
        missing = int(torch.count_nonzero(feature_weight <= 0))
        raise ValueError(f"{missing} sampled voxels are not visible in any render")
    return (feature_sum / feature_weight.unsqueeze(1)).numpy().astype(np.float16)


def save_feature_file(path: Path, pack: dict[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **pack)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_instances(value: str) -> set[str]:
    if "," in value:
        values = value.split(",")
    else:
        path = Path(value)
        values = (
            path.read_text(encoding="utf-8").splitlines()
            if path.is_file()
            else [value]
        )
    instances = {item.strip() for item in values if item.strip()}
    if not instances:
        raise ValueError("--instances did not identify any assets")
    return instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_file")
    parser.add_argument("--filter_low_aesthetic_score", type=float)
    parser.add_argument("--model", default=DINO_MODEL, choices=[DINO_MODEL])
    parser.add_argument("--instances")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--voxel_chunk_size", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    opt = parser.parse_args()
    if (
        opt.batch_size <= 0
        or opt.voxel_chunk_size <= 0
        or opt.world_size <= 0
        or not 0 <= opt.rank < opt.world_size
    ):
        raise ValueError("invalid batch size, rank, or world size")

    feature_name = DINO_MODEL
    feature_dir = Path(opt.output_dir) / "features" / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    model = torch.hub.load(
        DINO_REPOSITORY,
        DINO_MODEL,
        trust_repo=True,
        skip_validation=True,
    )
    model.eval().cuda()
    image_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )

    metadata_path = (
        Path(opt.metadata_file).expanduser().resolve()
        if opt.metadata_file
        else Path(opt.output_dir) / "metadata.csv"
    )
    if not metadata_path.is_file():
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(metadata_path)
    if opt.instances:
        instances = parse_instances(opt.instances)
        metadata = metadata[metadata["sha256"].astype(str).isin(instances)]
        found = set(metadata["sha256"].astype(str))
        if found != instances:
            raise ValueError(f"unknown instance hashes: {sorted(instances - found)}")
    elif not opt.metadata_file:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[
                metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score
            ]
        metadata = metadata[metadata["voxelized"] == True]
        metadata = metadata[metadata["rendered"] == True]

    metadata = metadata.sort_values("sha256", kind="stable").reset_index(drop=True)
    metadata = metadata.iloc[opt.rank :: opt.world_size]
    records = []
    failures = []

    for sha256 in tqdm(metadata["sha256"].astype(str), desc="Extracting features"):
        try:
            feature_path = feature_dir / f"{sha256}.npz"
            if not opt.force and feature_path.is_file():
                records.append({"sha256": sha256, f"feature_{feature_name}": True})
                continue

            render_dir = Path(opt.output_dir) / "renders" / sha256
            transforms_path = render_dir / "transforms.json"
            with transforms_path.open("r", encoding="utf-8") as stream:
                render_metadata = json.load(stream)
            frames = render_metadata["frames"]
            expected_views = render_metadata.get("render_settings", {}).get(
                "num_views", len(frames)
            )
            segment_order = render_metadata.get("mesh", {}).get("segment_keys", [])
            if len(frames) != expected_views or not frames:
                raise ValueError("render manifest does not contain every expected view")
            render_sha256 = render_hash(render_dir, frames)

            positions, grid_coords, selected_rows = load_sampled_voxels(
                opt.output_dir,
                sha256,
                segment_order,
            )
            views = load_views(opt.output_dir, frames, sha256)
            patchtokens = extract_features(
                model,
                views,
                positions,
                image_transform,
                opt.batch_size,
                opt.voxel_chunk_size,
            )
            if patchtokens.shape != (len(selected_rows), DINO_FEATURE_DIM):
                raise ValueError(
                    f"unexpected DINO feature shape: {patchtokens.shape}"
                )
            pack = {
                "indices": grid_coords.astype(np.uint8),
                "patchtokens": patchtokens,
                "voxel_row_indices": selected_rows.astype(np.int32),
                "model_name": np.asarray(DINO_MODEL),
                "model_repository": np.asarray(DINO_REPOSITORY),
                "render_sha256": np.asarray(render_sha256),
            }
            save_feature_file(feature_path, pack)
            records.append({"sha256": sha256, f"feature_{feature_name}": True})
        except Exception as exc:
            tqdm.write(f"FAILED {sha256}: {type(exc).__name__}: {exc}")
            records.append({"sha256": sha256, f"feature_{feature_name}": False})
            failures.append(sha256)

    record_tag = f"{metadata_path.stem}_" if opt.metadata_file else ""
    record_path = (
        Path(opt.output_dir) / f"feature_{feature_name}_{record_tag}{opt.rank}.csv"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{record_path.name}.", suffix=".part", dir=record_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pd.DataFrame.from_records(records).to_csv(temporary_path, index=False)
        os.replace(temporary_path, record_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if failures:
        raise RuntimeError(f"feature extraction failed for {len(failures)} assets")


if __name__ == "__main__":
    main()
