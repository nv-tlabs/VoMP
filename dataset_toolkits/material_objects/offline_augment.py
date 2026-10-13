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

"""Create two rotated copies of each asset.

Rotate the mesh and cameras together to reuse the rendered images, then
voxelize the rotated mesh.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset_toolkits.build_metadata import feature_file_is_complete, render_is_complete
from dataset_toolkits.material_objects.voxelize import (
    load_material_annotations,
    voxel_outputs_are_complete,
)


AUGMENTATION_COUNT = 2
QUATERNION_COLUMNS = (
    "rotation_w",
    "rotation_x",
    "rotation_y",
    "rotation_z",
)


class AugmentationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant_id(base_sha256: str, index: int) -> str:
    return hashlib.sha256(
        f"{base_sha256}\0offline_rotation\0{index}".encode("ascii")
    ).hexdigest()


def _quaternion(base_sha256: str, index: int) -> np.ndarray:
    digest = hashlib.sha256(
        f"{base_sha256}\0haar_rotation\0{index}".encode("ascii")
    ).digest()
    unit = float(1 << 64)
    u1, u2, u3 = (
        (int.from_bytes(digest[offset : offset + 8], "big") + 0.5) / unit
        for offset in (0, 8, 16)
    )
    root1 = math.sqrt(1.0 - u1)
    root2 = math.sqrt(u1)
    x = root1 * math.sin(2.0 * math.pi * u2)
    y = root1 * math.cos(2.0 * math.pi * u2)
    z = root2 * math.sin(2.0 * math.pi * u3)
    w = root2 * math.cos(2.0 * math.pi * u3)
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    matrix = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-12) or not math.isclose(
        float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-12
    ):
        raise AugmentationError("generated rotation is not a proper 3D rotation")
    return matrix


def _prepare_metadata(output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise AugmentationError(f"canonical metadata is missing: {metadata_path}")
    metadata = pd.read_csv(metadata_path, dtype={"sha256": str})
    if "augmentation_index" in metadata.columns:
        metadata = metadata[
            pd.to_numeric(metadata["augmentation_index"], errors="coerce")
            .fillna(0)
            .eq(0)
        ].copy()
    required = {"sha256", "local_path", "dataset", "split"}
    if not required <= set(metadata.columns):
        raise AugmentationError(
            f"metadata is missing columns: {sorted(required - set(metadata.columns))}"
        )
    if metadata["sha256"].duplicated().any():
        raise AugmentationError("canonical metadata contains duplicate asset hashes")

    canonical_hashes = set(metadata["sha256"])
    records: list[dict[str, Any]] = []
    for row in metadata.to_dict("records"):
        base_sha256 = str(row["sha256"])
        if len(base_sha256) != 64:
            raise AugmentationError(f"invalid canonical asset hash: {base_sha256!r}")
        for index in range(1, AUGMENTATION_COUNT + 1):
            record = dict(row)
            variant_sha256 = _variant_id(base_sha256, index)
            if variant_sha256 in canonical_hashes:
                raise AugmentationError("augmentation hash collides with a canonical asset")
            quaternion = _quaternion(base_sha256, index)
            record.update(
                {
                    "sha256": variant_sha256,
                    "base_sha256": base_sha256,
                    "augmentation_index": index,
                    **dict(zip(QUATERNION_COLUMNS, quaternion)),
                }
            )
            if "original_name" in record:
                record["original_name"] = f"{record['original_name']}__rotation_{index}"
            if record["split"] in {"val", "test"}:
                record["split"] = f"rotated_{record['split']}"
            for column in record:
                if column in {"rendered", "voxelized", "cond_rendered"} or column.startswith(
                    ("feature_", "latent_", "ss_latent_")
                ):
                    record[column] = False
            if "num_voxels" in record:
                record["num_voxels"] = 0
            records.append(record)

    augmented = pd.DataFrame.from_records(records)
    if len(augmented) != len(metadata) * AUGMENTATION_COUNT:
        raise AugmentationError("augmentation metadata does not contain exactly two variants per asset")
    if augmented["sha256"].duplicated().any():
        raise AugmentationError("augmentation hashes are not unique")
    if not (augmented.groupby("base_sha256").size() == AUGMENTATION_COUNT).all():
        raise AugmentationError("a canonical asset does not have exactly two variants")

    destination = output_dir / "augmentation_metadata.csv"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".augmentation_metadata.", suffix=".csv.part", dir=output_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        augmented.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Prepared {len(augmented)} variants for {len(metadata)} canonical assets: "
        f"{destination}"
    )


def _face_dtype() -> np.dtype:
    return np.dtype(
        [("count", "u1"), ("indices", "<i4", (3,)), ("segment_index", "<i4")]
    )


def _mesh_header(mesh: Mapping[str, Any]) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment exact composed render geometry in normalized object coordinates\n"
        f"comment segment_keys_sha256 {mesh['segment_keys_sha256']}\n"
        f"element vertex {mesh['vertex_count']}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {mesh['triangle_count']}\n"
        "property list uchar int vertex_indices\n"
        "property int segment_index\n"
        "end_header\n"
    ).encode("ascii")


def _read_mesh(path: Path, mesh: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    header = _mesh_header(mesh)
    faces_offset = len(header) + int(mesh["vertex_count"]) * 12
    expected_size = faces_offset + int(mesh["triangle_count"]) * _face_dtype().itemsize
    if path.stat().st_size != expected_size or _sha256_file(path) != mesh["sha256"]:
        raise AugmentationError(f"canonical mesh does not match its manifest: {path}")
    with path.open("rb") as stream:
        if stream.read(len(header)) != header:
            raise AugmentationError(f"canonical mesh header is invalid: {path}")
    vertices = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=len(header),
        shape=(int(mesh["vertex_count"]), 3),
    )
    faces = np.memmap(
        path,
        mode="r",
        dtype=_face_dtype(),
        offset=faces_offset,
        shape=(int(mesh["triangle_count"]),),
    )
    vertex_array = np.asarray(vertices, dtype=np.float64).copy()
    face_array = np.asarray(faces).copy()
    del vertices
    del faces
    return vertex_array, face_array


def _write_mesh(
    path: Path, mesh: Mapping[str, Any], vertices: np.ndarray, faces: np.ndarray
) -> str:
    header = _mesh_header(mesh)
    with path.open("wb") as stream:
        stream.write(header)
        np.asarray(vertices, dtype="<f4").tofile(stream)
        np.asarray(faces, dtype=_face_dtype()).tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_file(path)


def _mesh_sha256(
    mesh: Mapping[str, Any], vertices: np.ndarray, faces: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(_mesh_header(mesh))
    digest.update(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=_face_dtype()).tobytes())
    return digest.hexdigest()


def _variant_complete(
    destination: Path,
    base_sha256: str,
    index: int,
    quaternion: np.ndarray,
    canonical_render_dir: Path,
    canonical_manifest: Mapping[str, Any],
    expected_rotation_center: np.ndarray,
    expected_rotation_scale: float,
    expected_mesh_sha256: str,
) -> bool:
    try:
        with (destination / "transforms.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        augmentation = manifest["augmentation"]
        normalization = manifest["normalization"]
        mesh = manifest["mesh"]
        frames = manifest["frames"]
        canonical_frames = canonical_manifest["frames"]
        rotation = np.asarray(augmentation["rotation_matrix"], dtype=np.float64)
        stored_rotation_center = np.asarray(
            normalization["rotation_center"], dtype=np.float64
        )
        stored_rotation_scale = float(normalization["rotation_scale"])
        cameras_match = len(frames) == len(canonical_frames) == 150
        if cameras_match:
            for frame, canonical_frame in zip(frames, canonical_frames):
                preserved_frame = {
                    key: value
                    for key, value in frame.items()
                    if key not in {"transform_matrix", "yaw", "pitch", "radius"}
                }
                preserved_canonical_frame = {
                    key: value
                    for key, value in canonical_frame.items()
                    if key not in {"transform_matrix", "yaw", "pitch", "radius"}
                }
                if (
                    preserved_frame != preserved_canonical_frame
                    or any(key in frame for key in ("yaw", "pitch", "radius"))
                ):
                    cameras_match = False
                    break
                matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
                canonical_matrix = np.asarray(
                    canonical_frame["transform_matrix"], dtype=np.float64
                )
                expected = np.eye(4, dtype=np.float64)
                expected[:3, :3] = rotation @ canonical_matrix[:3, :3]
                expected[:3, 3] = (
                    rotation @ canonical_matrix[:3, 3] - stored_rotation_center
                ) / stored_rotation_scale
                if not np.allclose(matrix, expected, rtol=0.0, atol=1.0e-12):
                    cameras_match = False
                    break
        return bool(
            augmentation["base_sha256"] == base_sha256
            and augmentation["index"] == index
            and augmentation["canonical_mesh_sha256"]
            == canonical_manifest["mesh"]["sha256"]
            and augmentation["canonical_input_fingerprint"]
            == canonical_manifest["input_fingerprint"]
            and np.allclose(
                np.asarray(augmentation["quaternion_wxyz"], dtype=np.float64),
                quaternion,
                rtol=0.0,
                atol=1.0e-15,
            )
            and np.allclose(
                rotation,
                _rotation_matrix(quaternion),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                stored_rotation_center,
                expected_rotation_center,
                rtol=0.0,
                atol=1.0e-12,
            )
            and math.isclose(
                stored_rotation_scale,
                expected_rotation_scale,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and mesh["sha256"] == expected_mesh_sha256
            and _sha256_file(destination / "mesh.ply") == expected_mesh_sha256
            and cameras_match
            and all(
                os.path.samefile(
                    destination / frame["file_path"],
                    canonical_render_dir / frame["file_path"],
                )
                for frame in frames
            )
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _create_variant(output_dir: Path, row: Mapping[str, Any]) -> None:
    variant_sha256 = str(row["sha256"])
    base_sha256 = str(row["base_sha256"])
    index = int(row["augmentation_index"])
    expected_variant = _variant_id(base_sha256, index)
    if variant_sha256 != expected_variant or index not in range(1, AUGMENTATION_COUNT + 1):
        raise AugmentationError(f"invalid augmentation identity: {variant_sha256}")

    quaternion = np.asarray([float(row[column]) for column in QUATERNION_COLUMNS])
    expected_quaternion = _quaternion(base_sha256, index)
    if not np.allclose(quaternion, expected_quaternion, rtol=0.0, atol=1.0e-15):
        raise AugmentationError(f"augmentation rotation changed for {variant_sha256}")
    rotation = _rotation_matrix(quaternion)

    canonical_render_dir = output_dir / "renders" / base_sha256
    with (canonical_render_dir / "transforms.json").open(
        "r", encoding="utf-8"
    ) as stream:
        canonical_manifest = json.load(stream)
    canonical_mesh = canonical_manifest["mesh"]
    frames = canonical_manifest["frames"]
    if len(frames) != 150:
        raise AugmentationError(f"canonical asset {base_sha256} does not have 150 views")

    vertices, faces = _read_mesh(canonical_render_dir / "mesh.ply", canonical_mesh)
    rotated = vertices @ rotation.T
    minimum = rotated.min(axis=0)
    maximum = rotated.max(axis=0)
    rotation_center = (minimum + maximum) * 0.5
    rotation_scale = float((maximum - minimum).max())
    if not math.isfinite(rotation_scale) or rotation_scale <= 1.0e-12:
        raise AugmentationError(f"rotated mesh is degenerate: {variant_sha256}")
    normalized = (rotated - rotation_center) / rotation_scale
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > 0.500001):
        raise AugmentationError(f"rotated mesh exceeds the unit cube: {variant_sha256}")
    augmented_mesh = dict(canonical_mesh)
    expected_mesh_sha256 = _mesh_sha256(augmented_mesh, normalized, faces)

    destination = output_dir / "renders" / variant_sha256
    if _variant_complete(
        destination,
        base_sha256,
        index,
        quaternion,
        canonical_render_dir,
        canonical_manifest,
        rotation_center,
        rotation_scale,
        expected_mesh_sha256,
    ):
        return

    staging = Path(
        tempfile.mkdtemp(prefix=f".{variant_sha256}.augment.", dir=output_dir / "renders")
    )
    try:
        manifest = deepcopy(canonical_manifest)
        mesh = augmented_mesh
        mesh["sha256"] = _write_mesh(staging / "mesh.ply", mesh, normalized, faces)
        if mesh["sha256"] != expected_mesh_sha256:
            raise AugmentationError("written augmented mesh hash is inconsistent")
        manifest["mesh"] = mesh
        manifest.pop("scale", None)
        manifest.pop("offset", None)
        normalization = dict(canonical_manifest["normalization"])
        normalization.update(
            {
                "rotation_matrix": rotation.tolist(),
                "rotation_center": rotation_center.tolist(),
                "rotation_scale": rotation_scale,
            }
        )
        manifest["normalization"] = normalization
        manifest["augmentation"] = {
            "base_sha256": base_sha256,
            "index": index,
            "quaternion_wxyz": quaternion.tolist(),
            "rotation_matrix": rotation.tolist(),
            "canonical_mesh_sha256": canonical_mesh["sha256"],
            "canonical_input_fingerprint": canonical_manifest["input_fingerprint"],
        }
        manifest["input_fingerprint"] = hashlib.sha256(
            json.dumps(
                {
                    "base_input_fingerprint": canonical_manifest["input_fingerprint"],
                    "base_mesh_sha256": canonical_mesh["sha256"],
                    "rotation_matrix": rotation.tolist(),
                    "rotation_center": rotation_center.tolist(),
                    "rotation_scale": rotation_scale,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        # Move cameras with the mesh so we can reuse the same rendered images.
        transformed_frames = []
        for frame in frames:
            transformed = dict(frame)
            matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
            augmented_matrix = np.eye(4, dtype=np.float64)
            augmented_matrix[:3, :3] = rotation @ matrix[:3, :3]
            augmented_matrix[:3, 3] = (
                rotation @ matrix[:3, 3] - rotation_center
            ) / rotation_scale
            transformed["transform_matrix"] = augmented_matrix.tolist()
            transformed.pop("yaw", None)
            transformed.pop("pitch", None)
            transformed.pop("radius", None)
            transformed_frames.append(transformed)
            os.link(
                canonical_render_dir / frame["file_path"],
                staging / frame["file_path"],
            )
        manifest["frames"] = transformed_frames

        with (staging / "transforms.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            if _variant_complete(
                destination,
                base_sha256,
                index,
                quaternion,
                canonical_render_dir,
                canonical_manifest,
                rotation_center,
                rotation_scale,
                expected_mesh_sha256,
            ):
                return
            shutil.rmtree(destination)
        try:
            os.rename(staging, destination)
        except OSError:
            if not _variant_complete(
                destination,
                base_sha256,
                index,
                quaternion,
                canonical_render_dir,
                canonical_manifest,
                rotation_center,
                rotation_scale,
                expected_mesh_sha256,
            ):
                raise
        if not _variant_complete(
            destination,
            base_sha256,
            index,
            quaternion,
            canonical_render_dir,
            canonical_manifest,
            rotation_center,
            rotation_scale,
            expected_mesh_sha256,
        ):
            raise AugmentationError(f"published augmentation is incomplete: {variant_sha256}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _create_geometry(output_dir: Path, rank: int, world_size: int) -> None:
    metadata_path = output_dir / "augmentation_metadata.csv"
    if not metadata_path.is_file():
        raise AugmentationError(
            f"run the prepare stage before geometry: {metadata_path}"
        )
    metadata = pd.read_csv(
        metadata_path,
        dtype={"sha256": str, "base_sha256": str},
    ).sort_values("sha256", kind="stable")
    rows = metadata.iloc[rank::world_size].to_dict("records")
    failures = []
    for row in tqdm(rows, desc=f"Augmenting geometry shard {rank}"):
        try:
            _create_variant(output_dir, row)
        except Exception as exc:
            failures.append(str(row["sha256"]))
            tqdm.write(
                f"FAILED {row['sha256']}: {type(exc).__name__}: {exc}"
            )
    print(
        f"Geometry shard {rank}: {len(rows) - len(failures)} complete, "
        f"{len(failures)} failed"
    )
    if failures:
        raise AugmentationError(f"geometry augmentation failed for {len(failures)} assets")


def _variant_artifacts_match(output_dir: Path, row: Mapping[str, Any]) -> bool:
    try:
        variant_sha256 = str(row["sha256"])
        base_sha256 = str(row["base_sha256"])
        index = int(row["augmentation_index"])
        if variant_sha256 != _variant_id(base_sha256, index):
            return False
        quaternion = np.asarray(
            [float(row[column]) for column in QUATERNION_COLUMNS], dtype=np.float64
        )
        if not np.allclose(
            quaternion,
            _quaternion(base_sha256, index),
            rtol=0.0,
            atol=1.0e-15,
        ):
            return False
        rotation = _rotation_matrix(quaternion)
        canonical_render_dir = output_dir / "renders" / base_sha256
        with (canonical_render_dir / "transforms.json").open(
            "r", encoding="utf-8"
        ) as stream:
            canonical_manifest = json.load(stream)
        vertices, faces = _read_mesh(
            canonical_render_dir / "mesh.ply", canonical_manifest["mesh"]
        )
        rotated = vertices @ rotation.T
        rotation_center = (rotated.min(axis=0) + rotated.max(axis=0)) * 0.5
        rotation_scale = float(np.ptp(rotated, axis=0).max())
        normalized = (rotated - rotation_center) / rotation_scale
        augmented_mesh = dict(canonical_manifest["mesh"])
        expected_mesh_sha256 = _mesh_sha256(augmented_mesh, normalized, faces)
        return _variant_complete(
            output_dir / "renders" / variant_sha256,
            base_sha256,
            index,
            quaternion,
            canonical_render_dir,
            canonical_manifest,
            rotation_center,
            rotation_scale,
            expected_mesh_sha256,
        )
    except (KeyError, OSError, TypeError, ValueError, AugmentationError):
        return False


def _write_csv_atomic(dataframe: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        dataframe.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_metadata(output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.csv"
    augmentation_path = output_dir / "augmentation_metadata.csv"
    canonical = pd.read_csv(metadata_path, dtype={"sha256": str})
    if "augmentation_index" in canonical.columns:
        canonical = canonical[
            pd.to_numeric(canonical["augmentation_index"], errors="coerce")
            .fillna(0)
            .eq(0)
        ].copy()
    augmented = pd.read_csv(
        augmentation_path, dtype={"sha256": str, "base_sha256": str}
    )
    combined = pd.concat((canonical, augmented), ignore_index=True, sort=False)
    if len(combined) != len(canonical) * (AUGMENTATION_COUNT + 1):
        raise AugmentationError("combined metadata is not exactly 3x canonical size")
    if combined["sha256"].duplicated().any():
        raise AugmentationError("combined metadata contains duplicate asset hashes")

    feature_model = "dinov2_vitl14_reg"
    feature_column = f"feature_{feature_model}"
    annotations = load_material_annotations(output_dir / "material_annotations.json")

    def validate(row: Mapping[str, Any]) -> tuple[str, bool, bool, int, bool]:
        sha256 = str(row["sha256"])
        raw_index = row.get("augmentation_index", 0)
        augmentation_index = 0 if pd.isna(raw_index) else int(raw_index)
        exact_variant = augmentation_index == 0 or _variant_artifacts_match(
            output_dir, row
        )
        rendered = exact_variant and render_is_complete(
            output_dir / "renders" / sha256, strict=True
        )
        voxelized, num_voxels = (False, 0)
        if rendered:
            voxelized, num_voxels = voxel_outputs_are_complete(
                output_dir,
                sha256,
                str(row["local_path"]),
                str(row["dataset"]),
                annotations,
            )
        featured = voxelized and feature_file_is_complete(
            output_dir, feature_model, sha256
        )
        return sha256, rendered, voxelized, num_voxels, featured

    results: dict[str, tuple[bool, bool, int, bool]] = {}
    rows = combined.to_dict("records")
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        futures = {executor.submit(validate, row): str(row["sha256"]) for row in rows}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Validating 3x artifacts"
        ):
            sha256 = futures[future]
            try:
                result_sha256, rendered, voxelized, count, featured = future.result()
            except Exception as exc:
                raise AugmentationError(
                    f"artifact validation crashed for {sha256}: {exc}"
                ) from exc
            results[result_sha256] = (rendered, voxelized, count, featured)

    incomplete = [
        sha256
        for sha256, (rendered, voxelized, _, featured) in results.items()
        if not (rendered and voxelized and featured)
    ]
    if incomplete:
        raise AugmentationError(
            f"refusing to publish: {len(incomplete)} assets are incomplete; "
            f"first failures: {sorted(incomplete)[:10]}"
        )

    combined["rendered"] = False
    combined["voxelized"] = False
    combined["num_voxels"] = 0
    combined[feature_column] = False
    combined = combined.set_index("sha256", drop=False)
    for sha256, (rendered, voxelized, count, featured) in results.items():
        combined.at[sha256, "rendered"] = rendered
        combined.at[sha256, "voxelized"] = voxelized
        combined.at[sha256, "num_voxels"] = count
        combined.at[sha256, feature_column] = featured
    combined = combined.reset_index(drop=True)

    splits_dir = output_dir / "splits"
    for split in ("train", "val", "test", "rotated_val", "rotated_test"):
        split_rows = combined[combined["split"] == split]
        if not split_rows.empty:
            _write_csv_atomic(split_rows, splits_dir / f"{split}.csv")
    _write_csv_atomic(combined, metadata_path)
    print(f"Published 3x metadata with {len(combined)} rows: {metadata_path}")


def _verify_metadata(output_dir: Path) -> None:
    metadata = pd.read_csv(output_dir / "metadata.csv", dtype={"sha256": str})
    if "augmentation_index" not in metadata.columns:
        raise AugmentationError("combined metadata has no augmentation rows")
    augmentation_index = pd.to_numeric(
        metadata["augmentation_index"], errors="coerce"
    ).fillna(0)
    variants = metadata[augmentation_index.gt(0)]
    canonical = metadata[augmentation_index.eq(0)]
    feature_column = "feature_dinov2_vitl14_reg"
    required = {"rendered", "voxelized", feature_column}
    if len(variants) * 3 != len(metadata) * 2 or not required <= set(metadata.columns):
        raise AugmentationError("combined metadata does not describe a complete 3x corpus")
    complete = (
        metadata["rendered"].eq(True)
        & metadata["voxelized"].eq(True)
        & metadata[feature_column].eq(True)
    )
    if not complete.all():
        failed = metadata.loc[~complete, "sha256"].tolist()
        raise AugmentationError(
            f"{len(failed)} assets are incomplete; first failures: {failed[:10]}"
        )
    if len(variants) != 2 * len(canonical):
        raise AugmentationError("combined metadata is not exactly 2 variants per canonical asset")
    print(f"Verified all {len(metadata)} assets in the 3x metadata")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a 3x offline orientation-augmented processed corpus."
    )
    parser.add_argument("stage", choices=("prepare", "geometry", "finalize", "verify"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.world_size <= 0 or not 0 <= arguments.rank < arguments.world_size:
        raise AugmentationError("rank must be in [0, world_size)")
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if arguments.stage == "prepare":
        _prepare_metadata(output_dir)
    elif arguments.stage == "geometry":
        _create_geometry(output_dir, arguments.rank, arguments.world_size)
    elif arguments.stage == "finalize":
        _finalize_metadata(output_dir)
    else:
        _verify_metadata(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
