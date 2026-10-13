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

"""Find USD assets for material annotation across the four source datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


DATASET_NAMES = ("commercial", "residential", "vegetation", "simready")
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOTS = {
    name: _REPOSITORY_ROOT / "datasets" / "raw" / name for name in DATASET_NAMES
}


class AssetDiscoveryError(RuntimeError):
    """Raised when a dataset cannot be mapped to unambiguous entry points."""


@dataclass(frozen=True)
class VLMAsset:
    """An object USD path and its preview image."""

    dataset: str
    dataset_root: Path
    path: Path
    relative_path: str
    object_name: str
    identity_hash: str
    thumbnail_path: Path | None

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.relative_path}"


def is_usd_file(path: str | os.PathLike[str]) -> bool:
    """Return whether *path* has a supported USD-family suffix."""

    return Path(path).suffix.casefold() in USD_SUFFIXES


def make_identity_hash(dataset: str, relative_path: str) -> str:
    """Create a mount-independent identity for one logical source asset."""

    canonical = f"vlm-asset-v1\0{dataset}\0{Path(relative_path).as_posix()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_thumbnail(asset_path: str | os.PathLike[str], dataset: str) -> Path | None:
    """Find the exact preview image associated with an entry USD.

    Most datasets use ``.thumbs``.  Vegetation Leaves and Rocks use ``thumbs``
    instead, so both spellings are part of the source layout.  Fuzzy fallback
    is intentionally avoided: a preview for another file must never be sent to
    the VLM.
    """

    path = Path(asset_path)
    if dataset == "simready":
        directories = (path.parent / ".thumbs" / "256x256",)
        filenames = (f"{path.parent.name}.usd.png",)
    else:
        directories = (
            path.parent / ".thumbs" / "256x256",
            path.parent / "thumbs" / "256x256",
        )
        filenames = (f"{path.name}.png", f"{path.name}.auto.png")

    for directory in directories:
        for filename in filenames:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve(strict=True)
    return None


def select_simready_entrypoint(
    prop_dir: str | os.PathLike[str],
) -> Path | None:
    """Select the canonical SimReady layer for material annotation.

    A normal prop has ``<prop>.usd`` plus ``_base``, ``_inst``, and
    ``_inst_base`` layers.  Annotation uses ``_inst_base`` because it contains
    the prop geometry/materials without the thumbnail rig composed by the
    unsuffixed presentation layer.  Directories without any USD file are
    support directories and return ``None``.
    """

    directory = Path(prop_dir)
    if not directory.is_dir():
        raise AssetDiscoveryError(
            f"SimReady prop directory does not exist: {directory}"
        )

    usd_files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and is_usd_file(path)
        ),
        key=lambda path: path.name.casefold(),
    )
    if not usd_files:
        return None

    for layer_suffix in ("_inst_base", "_base"):
        candidates = [
            path
            for path in usd_files
            if path.stem.casefold().endswith(layer_suffix)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AssetDiscoveryError(
                f"Ambiguous SimReady {layer_suffix} entry points in {directory}: "
                + ", ".join(path.name for path in candidates)
            )

    if len(usd_files) == 1:
        return usd_files[0]

    raise AssetDiscoveryError(
        f"Could not select one SimReady annotation entry point in {directory}: "
        + ", ".join(path.name for path in usd_files)
    )


def _make_asset(dataset: str, dataset_root: Path, path: Path) -> VLMAsset:
    relative_path = path.relative_to(dataset_root).as_posix()
    return VLMAsset(
        dataset=dataset,
        dataset_root=dataset_root,
        path=path,
        relative_path=relative_path,
        object_name=path.parent.name if dataset == "simready" else path.stem,
        identity_hash=make_identity_hash(dataset, relative_path),
        thumbnail_path=find_thumbnail(path, dataset),
    )


def _discover_tree_assets(dataset: str, dataset_root: Path) -> list[VLMAsset]:
    paths: list[Path] = []
    for current_root, directories, filenames in os.walk(dataset_root):
        directories[:] = sorted(
            directory for directory in directories if not directory.startswith(".")
        )
        root = Path(current_root)
        if any(part.startswith(".") for part in root.relative_to(dataset_root).parts):
            continue
        paths.extend(
            root / filename
            for filename in filenames
            if not filename.startswith(".") and is_usd_file(filename)
        )

    return [
        _make_asset(dataset, dataset_root, path)
        for path in sorted(
            paths, key=lambda path: path.relative_to(dataset_root).as_posix()
        )
    ]


def _discover_simready_assets(dataset_root: Path) -> list[VLMAsset]:
    props_root = dataset_root / "common_assets" / "props"
    if not props_root.is_dir():
        raise AssetDiscoveryError(
            f"SimReady props directory does not exist: {props_root}"
        )

    assets = []
    for prop_dir in sorted(
        (path for path in props_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        entrypoint = select_simready_entrypoint(prop_dir)
        if entrypoint is not None:
            assets.append(_make_asset("simready", dataset_root, entrypoint))
    return assets


def discover_assets(
    dataset: str, dataset_root: str | os.PathLike[str] | None = None
) -> list[VLMAsset]:
    """List the assets in a dataset, sorted by path."""

    dataset = dataset.casefold()
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASET_NAMES}")

    root = (
        Path(dataset_root)
        if dataset_root is not None
        else DEFAULT_DATASET_ROOTS[dataset]
    )
    if not root.is_dir():
        raise AssetDiscoveryError(f"Dataset directory does not exist: {root}")

    assets = (
        _discover_simready_assets(root)
        if dataset == "simready"
        else _discover_tree_assets(dataset, root)
    )
    return assets
