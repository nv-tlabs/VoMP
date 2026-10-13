#!/usr/bin/env python3
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

"""Fetch and verify the NVIDIA vMaterials used by raw dataset assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_toolkits.material_objects.vlm_annotations.external_materials import (
    NVIDIA_VMATERIALS_ORIGIN,
    iter_material_files,
    material_file_url,
    material_manifest_path,
    material_root_from_environment,
    verify_material_file,
    verify_material_root,
)


_ALLOWED_HOST = urlparse(NVIDIA_VMATERIALS_ORIGIN).hostname


def _download_one(
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int,
    *,
    material_root: Path,
    relative_path: str,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise RuntimeError(f"refusing unapproved material URL: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if material_manifest_path(material_root, relative_path) != destination:
        raise RuntimeError(
            f"refusing unsafe or symlinked material destination: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        request = Request(url, headers={"User-Agent": "TRELLIS-VLM-material-fetch/1"})
        with os.fdopen(descriptor, "wb") as output, urlopen(request, timeout=120) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
                raise RuntimeError(
                    f"material URL redirected outside the approved origin: {response.geturl()}"
                )
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                size += len(block)
                if size > expected_size:
                    raise RuntimeError(
                        "approved material response exceeded its pinned size: "
                        f"{url}; size>{expected_size}"
                    )
            output.flush()
            os.fsync(output.fileno())
        observed_sha256 = digest.hexdigest()
        if size != expected_size or observed_sha256 != expected_sha256:
            raise RuntimeError(
                "downloaded material content does not match the pinned manifest: "
                f"{url}; size={size}/{expected_size}; "
                f"sha256={observed_sha256}/{expected_sha256}"
            )
        if material_manifest_path(material_root, relative_path) != destination:
            raise RuntimeError(
                f"material destination changed during download: {destination}"
            )
        os.chmod(temporary_path, 0o444)
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material-root",
        type=Path,
        default=material_root_from_environment(),
        help="local root corresponding to NVIDIA's Materials directory",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not download; fail if any pinned file is absent or changed",
    )
    args = parser.parse_args()

    root = args.material_root.resolve(strict=False)
    for relative_path, manifest in iter_material_files():
        destination = material_manifest_path(root, relative_path)
        if destination is None:
            raise RuntimeError(
                f"unsafe or symlinked material manifest path: {root / relative_path}"
            )
        if verify_material_file(destination, manifest):
            print(f"VERIFIED {relative_path}", flush=True)
            continue
        if args.verify_only:
            print(f"MISSING_OR_INVALID {relative_path}", file=sys.stderr, flush=True)
            continue
        url = material_file_url(relative_path)
        print(f"FETCH {url}", flush=True)
        _download_one(
            url,
            destination,
            str(manifest["sha256"]),
            int(manifest["size_bytes"]),
            material_root=root,
            relative_path=relative_path,
        )
        if not verify_material_file(destination, manifest):
            raise RuntimeError(f"post-download verification failed: {destination}")
        print(f"VERIFIED {relative_path}", flush=True)

    statuses = verify_material_root(root)
    summary = {
        "material_root": str(root),
        "source_origin": NVIDIA_VMATERIALS_ORIGIN,
        "file_count": len(statuses),
        "verified_count": sum(bool(item["verified"]) for item in statuses),
        "all_verified": all(bool(item["verified"]) for item in statuses),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
