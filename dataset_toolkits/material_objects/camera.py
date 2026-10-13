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

import hashlib
import math

import numpy as np

from dataset_toolkits.utils import sphere_hammersley_sequence

DEFAULT_CAMERA_SEED = 0
CAMERA_SEED = 163056382637340268291568450002442245000811058277574926236476791
RESOLUTION = 512
DEFAULT_NUM_VIEWS = 150
CAMERA_FOV = math.radians(40.0)
CAMERA_SENSOR_SIZE_MM = 32.0
CAMERA_NOMINAL_RADIUS = 1.5
CAMERA_FRAME_FRACTION = 0.97


def resolve_camera_id(
    asset_id: str | None = None, seed: int | None = None
) -> str:
    """Prefer an explicit seed, then an explicit identity, otherwise seed zero."""
    if seed is not None:
        return str(seed)
    if asset_id is not None:
        return str(asset_id)
    return str(DEFAULT_CAMERA_SEED)


def _camera_samples(asset_id: str, num_views: int, up_axis: str):
    seed_bytes = CAMERA_SEED.to_bytes((CAMERA_SEED.bit_length() + 7) // 8, "big")
    digest = hashlib.sha256(seed_bytes + b"\0" + asset_id.encode("utf-8")).digest()
    denominator = float(1 << 64)
    offset = (
        (int.from_bytes(digest[:8], "big") + 0.5) / denominator,
        (int.from_bytes(digest[8:16], "big") + 0.5) / denominator,
    )
    for index in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(index, num_views, offset)
        if up_axis == "Z":
            direction = np.asarray(
                (
                    math.cos(yaw) * math.cos(pitch),
                    math.sin(yaw) * math.cos(pitch),
                    math.sin(pitch),
                ),
                dtype=np.float64,
            )
        elif up_axis == "Y":
            direction = np.asarray(
                (
                    math.cos(yaw) * math.cos(pitch),
                    math.sin(pitch),
                    math.sin(yaw) * math.cos(pitch),
                ),
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unsupported stage up axis {up_axis!r}")
        direction /= np.linalg.norm(direction)
        yield index, float(yaw), float(pitch), direction


def _camera_radius(
    normalized_points: np.ndarray,
    direction: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> float:
    # Back the camera up enough to keep the whole mesh in frame.
    projected_depth = normalized_points @ direction
    projected_radius = np.maximum(
        np.abs(normalized_points @ horizontal),
        np.abs(normalized_points @ vertical),
    )
    required = np.max(
        projected_depth
        + projected_radius
        / (CAMERA_FRAME_FRACTION * math.tan(CAMERA_FOV * 0.5))
    )
    radius = max(CAMERA_NOMINAL_RADIUS, float(required) + 1.0e-6)
    if not math.isfinite(radius):
        raise ValueError("computed camera radius is not finite")
    return radius


def _camera_frame(direction: np.ndarray, up_axis: str) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    stage_up = np.asarray(
        (0.0, 0.0, 1.0)
        if str(up_axis).upper() == "Z"
        else (0.0, 1.0, 0.0)
        if str(up_axis).upper() == "Y"
        else (),
        dtype=np.float64,
    )
    if stage_up.shape != (3,):
        raise ValueError(f"unsupported stage up axis {up_axis!r}")
    vertical = stage_up - float(np.dot(stage_up, direction)) * direction
    if float(np.linalg.norm(vertical)) <= 1e-10:
        helper = min(
            (
                np.asarray((1.0, 0.0, 0.0)),
                np.asarray((0.0, 1.0, 0.0)),
                np.asarray((0.0, 0.0, 1.0)),
            ),
            key=lambda axis: abs(float(np.dot(axis, direction))),
        )
        horizontal = np.cross(helper, direction)
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(direction, horizontal)
    else:
        vertical /= np.linalg.norm(vertical)
        horizontal = np.cross(vertical, direction)
        horizontal /= np.linalg.norm(horizontal)
    return horizontal, vertical
