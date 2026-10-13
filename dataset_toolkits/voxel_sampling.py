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

"""Sample voxels across material segments, surfaces, and interiors."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from fpsample import bucket_fps_kdline_sampling


MAX_DINO_VOXELS = 72_000
MIN_VOXELS_PER_SEGMENT = 64
GRID_RESOLUTION = 64


def _segment_quotas(counts: np.ndarray, budget: int) -> np.ndarray:
    segment_count = len(counts)
    if segment_count > budget:
        raise ValueError(
            f"cannot represent {segment_count} segments with only {budget} voxels"
        )

    # Give small segments a minimum share before splitting the remaining budget.
    base = max(1, min(MIN_VOXELS_PER_SEGMENT, budget // segment_count))
    quotas = np.minimum(counts, base).astype(np.int64)
    remaining = budget - int(quotas.sum())

    while remaining:
        capacity = counts - quotas
        active = capacity > 0
        # Square-root weights keep large segments from dominating the sample.
        weights = np.sqrt(counts.astype(np.float64)) * active
        ideal = remaining * weights / weights.sum()
        additions = np.minimum(capacity, np.floor(ideal).astype(np.int64))
        added = int(additions.sum())
        if added:
            quotas += additions
            remaining -= added
            continue

        fractions = ideal - np.floor(ideal)
        order = sorted(
            np.flatnonzero(active),
            key=lambda index: (-fractions[index], -counts[index], index),
        )
        for index in order[:remaining]:
            quotas[index] += 1
        remaining = 0

    return quotas


def _quick_farthest_points(coords: np.ndarray, count: int) -> np.ndarray:
    if count == len(coords):
        return np.arange(len(coords), dtype=np.int64)
    if count == 1:
        distance = np.sum(
            (coords.astype(np.float64) - coords.mean(axis=0)) ** 2, axis=1
        )
        return np.asarray([int(np.argmin(distance))], dtype=np.int64)

    height = max(1, min(9, int(math.log2(len(coords)))))
    selected = np.asarray(
        bucket_fps_kdline_sampling(
            coords.astype(np.float32, copy=False),
            count,
            h=height,
            start_idx=0,
        ),
        dtype=np.int64,
    )
    if (
        selected.shape != (count,)
        or len(np.unique(selected)) != count
        or np.any(selected < 0)
        or np.any(selected >= len(coords))
    ):
        raise RuntimeError("fpsample returned invalid voxel rows")
    return selected


def _stratum_counts(surface_count: int, interior_count: int, quota: int) -> tuple[int, int]:
    surface_quota = min(surface_count, (quota + 1) // 2)
    interior_quota = min(interior_count, quota // 2)
    remaining = quota - surface_quota - interior_quota

    addition = min(remaining, surface_count - surface_quota)
    surface_quota += addition
    remaining -= addition
    interior_quota += remaining
    return surface_quota, interior_quota


def sample_voxel_rows(
    grid_coords: np.ndarray,
    segment_ids: np.ndarray,
    surface_mask: np.ndarray,
    segment_order: Sequence[str],
    budget: int = MAX_DINO_VOXELS,
) -> np.ndarray:
    """Return sorted voxel rows sampled from each segment's surface and interior."""

    coords = np.asarray(grid_coords)
    segments = np.asarray(segment_ids).astype(str)
    surface = np.asarray(surface_mask)
    order = tuple(str(segment) for segment in segment_order)
    voxel_count = len(coords)

    if (
        coords.shape != (voxel_count, 3)
        or not np.issubdtype(coords.dtype, np.integer)
        or np.any(coords < 0)
        or np.any(coords >= GRID_RESOLUTION)
    ):
        raise ValueError("voxel grid coordinates must be integer 64^3 indices")
    if segments.shape != (voxel_count,) or surface.shape != (voxel_count,):
        raise ValueError("voxel segment and surface arrays must align with coordinates")
    if surface.dtype != np.dtype(bool):
        raise ValueError("voxel surface mask must be boolean")
    if budget <= 0:
        raise ValueError("voxel sampling budget must be positive")
    if len(set(order)) != len(order) or set(order) != set(np.unique(segments)):
        raise ValueError("voxel segments do not match the render segment order")

    flat = np.ravel_multi_index(coords.T, (GRID_RESOLUTION,) * 3)
    if len(np.unique(flat)) != voxel_count:
        raise ValueError("voxel grid coordinates are duplicated")
    if voxel_count <= budget:
        return np.arange(voxel_count, dtype=np.int64)

    segment_index = {segment: index for index, segment in enumerate(order)}
    labels = np.fromiter(
        (segment_index[segment] for segment in segments),
        dtype=np.int32,
        count=voxel_count,
    )
    counts = np.bincount(labels, minlength=len(order)).astype(np.int64)
    quotas = _segment_quotas(counts, budget)
    selected_groups: list[np.ndarray] = []

    for label, quota in enumerate(quotas):
        rows = np.flatnonzero(labels == label)
        surface_rows = rows[surface[rows]]
        interior_rows = rows[~surface[rows]]
        surface_quota, interior_quota = _stratum_counts(
            len(surface_rows), len(interior_rows), int(quota)
        )
        for candidates, count in (
            (surface_rows, surface_quota),
            (interior_rows, interior_quota),
        ):
            if not count:
                continue
            # Stable spatial order keeps farthest-point sampling reproducible.
            candidates = candidates[np.argsort(flat[candidates], kind="stable")]
            local_rows = _quick_farthest_points(coords[candidates], count)
            selected_groups.append(candidates[local_rows])

    selected = np.sort(np.concatenate(selected_groups)).astype(np.int64, copy=False)
    if len(selected) != budget or len(np.unique(selected)) != budget:
        raise RuntimeError("voxel sampler did not produce the requested unique rows")
    return selected
