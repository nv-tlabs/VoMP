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

import argparse
import fcntl
import stat
import sys
import logging
import os
import json
import hashlib
import math
import numbers
from pathlib import Path
from collections import deque
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import traceback


_VISUAL_IMPLEMENTATION_SOURCE_PATHS = tuple(
    Path(__file__).resolve().with_name(name)
    for name in (
        "segment_textured_renderer.py",
        "segment_visual_inputs.py",
    )
)
_LOADED_VISUAL_IMPLEMENTATION_SNAPSHOT = tuple(
    {
        "name": source_path.name,
        "path": source_path,
        "bytes": source_path.read_bytes(),
    }
    for source_path in _VISUAL_IMPLEMENTATION_SOURCE_PATHS
)


from dataset_toolkits.material_objects.vlm_annotations.data_subsets.simready import (
    extract_materials_from_usd as simready_extract_materials,
    PROMPTS as SIMREADY_PROMPTS,
    PROPS_DIR as SIMREADY_PROPS_DIR,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.residential import (
    PROMPTS as RESIDENTIAL_PROMPTS,
)
from dataset_toolkits.material_objects.vlm_annotations.utils.vlm import (
    ResponsesAPIConfig,
    STRICT_RETRY_QUERY_PROMPT_SUFFIX,
    STRICT_RETRY_USER_PROMPT_SUFFIX,
    load_vlm_model,
    analyze_material_with_vlm,
    current_vlm_text_context_hash,
    vlm_model_identity,
)
from dataset_toolkits.material_objects.vlm_annotations.data_subsets.common import (
    UsdJsonEncoder,
    extract_materials_from_usd as common_extract_materials,
)
from dataset_toolkits.material_objects.vlm_annotations.asset_finder import (
    DEFAULT_DATASET_ROOTS,
    discover_assets,
    find_thumbnail,
)
from dataset_toolkits.material_objects.vlm_annotations.vlm_input_builder import (
    VLM_IMAGE_ROLES,
    build_vlm_input_spec,
    compute_vlm_retry_attempt_hash,
    valid_vlm_property_value,
    validate_vlm_input_spec,
)
from dataset_toolkits.material_objects.vlm_annotations.vlm_output import (
    parse_vlm_properties,
    raw_vlm_output_sha256,
)
from dataset_toolkits.material_objects.vlm_annotations.segment_visual_inputs import (
    build_asset_geometry_payloads,
)
from dataset_toolkits.material_objects.vlm_annotations import (
    segment_textured_renderer as _textured_renderer,
)
from dataset_toolkits.material_objects.vlm_annotations.segment_textured_renderer import (
    render_textured_vlm_inputs,
)


from dataset_toolkits.material_objects.vlm_annotations.annotation_checkpoint import (
    build_asset_dependency_fingerprint,
    dependency_fingerprints_match,
    extraction_gate,
    is_annotation_complete,
    is_segment_annotation_complete,
    separate_simready_source_physics,
    update_annotation_status,
)
from dataset_toolkits.material_objects.vlm_annotations.annotation_input_policy import (
    asset_annotation_exclusion,
    asset_annotation_input_policy,
)
import copy
from tqdm import tqdm
import shutil


class VisualImplementationChangedError(RuntimeError):
    """The visual modules on disk no longer match the code loaded by this process."""


def _assert_visual_implementation_sources_unchanged():
    changed = []
    for source in _LOADED_VISUAL_IMPLEMENTATION_SNAPSHOT:
        try:
            current_bytes = source["path"].read_bytes()
        except OSError as exc:
            raise VisualImplementationChangedError(
                f"loaded visual source became unreadable: {source['path']}: {exc}"
            ) from exc
        if current_bytes != source["bytes"]:
            changed.append(source["name"])
    if changed:
        raise VisualImplementationChangedError(
            "loaded visual implementation changed on disk: "
            + ", ".join(sorted(changed))
        )


_assert_visual_implementation_sources_unchanged()


class CheckpointWriteError(RuntimeError):
    """Stop inference when an incremental output checkpoint cannot be saved."""


class VisualCacheBusyError(RuntimeError):
    """Another annotation writer currently owns this asset's exact PNG cache."""


class UnsafeVisualCachePathError(ValueError):
    """A cache path escapes its root or contains an unsafe filesystem object."""


_ACTIVE_VISUAL_CACHE_LOCKS = {}


def _visual_cache_root():
    configured = os.environ.get("VLM_VISUAL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[3] / "datasets" / ".vlm_visual_cache"


def _resolved_visual_cache_root(cache_root=None):
    root = Path(cache_root or _visual_cache_root()).expanduser().resolve(strict=False)
    if root == Path(root.anchor):
        raise UnsafeVisualCachePathError(f"refusing unsafe visual cache root: {root}")
    if os.path.lexists(root):
        mode = os.lstat(root).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise UnsafeVisualCachePathError(
                f"visual cache root is not a real directory: {root}"
            )
    return root


def _assert_visual_cache_containment(
    path,
    cache_root=None,
    *,
    final_must_be_directory=False,
):
    """Check that the cache path contains only directories beneath its root."""

    root = _resolved_visual_cache_root(cache_root)
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise UnsafeVisualCachePathError(
            f"visual cache path contains parent traversal: {candidate}"
        )
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.abspath(candidate))

    configured_root = Path(cache_root or _visual_cache_root()).expanduser()
    if not configured_root.is_absolute():
        configured_root = Path.cwd() / configured_root
    configured_root = Path(os.path.abspath(configured_root))

    relative = None
    for root_spelling in dict.fromkeys((root, configured_root)):
        try:
            relative = candidate.relative_to(root_spelling)
            break
        except ValueError:
            continue
    if relative is None:
        for ancestor in (candidate, *candidate.parents):
            if not os.path.lexists(ancestor):
                continue
            try:
                same_root = os.path.samefile(ancestor, root)
            except OSError:
                continue
            if same_root:
                relative = candidate.relative_to(ancestor)
                break
    if relative is None:
        raise UnsafeVisualCachePathError(
            f"visual cache path escapes its root: {candidate}"
        )

    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise UnsafeVisualCachePathError(
            f"visual cache path escapes its root: {candidate}"
        ) from exc

    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        if not os.path.lexists(current):
            break
        mode = os.lstat(current).st_mode
        if stat.S_ISLNK(mode):
            raise UnsafeVisualCachePathError(
                f"visual cache path contains a symlink: {current}"
            )
        is_final = index == len(relative.parts) - 1
        if not is_final and not stat.S_ISDIR(mode):
            raise ValueError(
                f"visual cache path has a non-directory ancestor: {current}"
            )
        if is_final and final_must_be_directory and not stat.S_ISDIR(mode):
            raise ValueError(
                f"visual cache directory is not a directory: {current}"
            )
    return candidate


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(label, value):
    encoded = json.dumps(
        value,
        cls=UsdJsonEncoder,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(encoded)
    return digest.hexdigest()


def _visual_implementation_sha256():
    records = [
        {
            "name": source["name"],
            "byte_count": len(source["bytes"]),
            "sha256": hashlib.sha256(source["bytes"]).hexdigest(),
        }
        for source in _LOADED_VISUAL_IMPLEMENTATION_SNAPSHOT
    ]
    return _canonical_sha256("visual-renderer-implementation", records)


def _visual_cache_binding(result, asset, geometry_payloads, *, size):
    dependency_fingerprint = result.get("asset_dependency_fingerprint")
    if not isinstance(dependency_fingerprint, Mapping):
        raise ValueError("asset dependency fingerprint is missing")
    dependency_sha256 = dependency_fingerprint.get("fingerprint_sha256")
    if (
        dependency_fingerprint.get("status") != "complete"
        or not _is_sha256(dependency_sha256)
    ):
        raise ValueError("asset dependency fingerprint is incomplete")
    if not _is_sha256(asset.identity_hash):
        raise ValueError(f"asset identity is malformed for {asset.key}")

    geometry_payload_hashes = {
        key: payload.manifest()["payload_sha256"]
        for key, payload in sorted(geometry_payloads.items())
    }
    segments = result.get("segments")
    if isinstance(segments, Mapping):
        segments = {
            key: (
                {
                    field: value
                    for field, value in segment.items()
                    if field != "diagnostic_runtime_source_prim_path"
                }
                if isinstance(segment, Mapping)
                else segment
            )
            for key, segment in segments.items()
        }
    render_input_sha256 = _canonical_sha256(
        "visual-renderer-input",
        {
            "segments": segments,
            "materials": result.get("materials"),
            "authored_perspective_camera_hint": result.get(
                "authored_perspective_camera_hint"
            ),
            "geometry_payload_hashes": geometry_payload_hashes,
        },
    )
    binding = {
        "status": "complete",
        "asset_identity_hash": asset.identity_hash,
        "asset_dependency_fingerprint_sha256": dependency_sha256,
        "renderer_implementation_sha256": _visual_implementation_sha256(),
        "render_input_sha256": render_input_sha256,
        "render_size": size,
    }
    binding["binding_sha256"] = _canonical_sha256(
        "durable-visual-cache-binding", binding
    )
    return binding


def _asset_visual_cache_parent(asset, cache_root=None):
    if not _is_sha256(asset.identity_hash):
        raise ValueError(f"asset identity is malformed for {asset.key}")
    root = _resolved_visual_cache_root(cache_root)
    return root / asset.identity_hash


def _asset_visual_cache_directory(asset, binding, cache_root=None):
    binding_sha256 = binding.get("binding_sha256")
    if not _is_sha256(binding_sha256):
        raise ValueError(f"visual cache binding is malformed for {asset.key}")
    return _asset_visual_cache_parent(asset, cache_root) / binding_sha256


def _visual_cache_lock_key(asset, cache_root=None):
    parent = _asset_visual_cache_parent(asset, cache_root)
    return str(parent.parent), asset.identity_hash


def _acquire_asset_visual_cache_lock(asset, cache_root=None, *, required=True):
    key = _visual_cache_lock_key(asset, cache_root)
    if key in _ACTIVE_VISUAL_CACHE_LOCKS:
        return True

    root = Path(key[0])
    lock_directory = root / ".locks"
    _assert_visual_cache_containment(lock_directory, root)
    lock_directory.mkdir(parents=True, exist_ok=True)
    _assert_visual_cache_containment(
        lock_directory,
        root,
        final_must_be_directory=True,
    )
    lock_path = lock_directory / f"{asset.identity_hash}.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(lock_directory, directory_flags)
    try:
        lock_fd = os.open(
            lock_path.name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
        os.close(lock_fd)
        raise ValueError(f"visual cache lock is not a regular file: {lock_path}")
    lock_file = os.fdopen(lock_fd, "r+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        if required:
            raise VisualCacheBusyError(
                f"another annotation process is using the visual cache for {asset.key}"
            ) from exc
        logging.warning(
            "Leaving visual cache for %s because another annotation process is using it",
            asset.key,
        )
        return False
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.uname().nodename}:{os.getpid()}\n")
        lock_file.flush()
        os.fsync(lock_file.fileno())
    except Exception:
        lock_file.close()
        raise
    _ACTIVE_VISUAL_CACHE_LOCKS[key] = lock_file
    return True


def _release_asset_visual_cache_lock(asset, cache_root=None):
    try:
        key = _visual_cache_lock_key(asset, cache_root)
        keys = [key] if key in _ACTIVE_VISUAL_CACHE_LOCKS else []
    except Exception as exc:
        logging.warning("Could not resolve visual cache lock during release: %s", exc)
        keys = []
    if not keys:
        keys = [
            key
            for key in _ACTIVE_VISUAL_CACHE_LOCKS
            if key[1] == asset.identity_hash
        ]
    for key in keys:
        lock_file = _ACTIVE_VISUAL_CACHE_LOCKS.pop(key)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            logging.warning("Could not explicitly unlock visual cache: %s", exc)
        try:
            lock_file.close()
        except OSError as exc:
            logging.warning("Could not close visual cache lock: %s", exc)


def _remove_asset_visual_cache(
    asset,
    cache_root=None,
    *,
    required=False,
):
    try:
        key = _visual_cache_lock_key(asset, cache_root)
        already_owned = key in _ACTIVE_VISUAL_CACHE_LOCKS
        asset_cache = _asset_visual_cache_parent(asset, cache_root)
        _assert_visual_cache_containment(asset_cache, cache_root)
    except UnsafeVisualCachePathError:
        raise
    except OSError as exc:
        if required:
            raise
        logging.warning("Could not inspect visual cache for %s: %s", asset.key, exc)
        return False
    if not already_owned and not required and not os.path.lexists(asset_cache):
        return True
    try:
        if not _acquire_asset_visual_cache_lock(
            asset, cache_root, required=required
        ):
            return False
    except UnsafeVisualCachePathError:
        raise
    except OSError as exc:
        if required:
            raise
        logging.warning(
            "Could not acquire the cleanup lock for %s: %s",
            asset.key,
            exc,
        )
        return False
    try:
        _assert_visual_cache_containment(asset_cache, cache_root)
        if asset_cache.is_dir():
            shutil.rmtree(asset_cache)
        elif asset_cache.is_file():
            asset_cache.unlink()
        elif os.path.lexists(asset_cache):
            raise UnsafeVisualCachePathError(
                f"visual cache has an unsafe filesystem type: {asset_cache}"
            )
        return True
    except OSError as exc:
        if required:
            raise
        logging.warning(
            "Could not clean completed visual cache for %s: %s",
            asset.key,
            exc,
        )
        return False
    finally:
        _release_asset_visual_cache_lock(asset, cache_root)


def _remove_stale_visual_cache_bindings(asset, binding, cache_root=None):
    asset_cache = _asset_visual_cache_parent(asset, cache_root)
    _assert_visual_cache_containment(asset_cache, cache_root)
    if not os.path.lexists(asset_cache):
        return
    _assert_visual_cache_containment(asset_cache, cache_root)
    if asset_cache.is_file():
        try:
            asset_cache.unlink()
        except OSError as exc:
            logging.warning("Could not remove stale visual cache %s: %s", asset_cache, exc)
        return
    if not asset_cache.is_dir():
        raise UnsafeVisualCachePathError(
            f"visual cache has an unsafe filesystem type: {asset_cache}"
        )
    current_binding = binding["binding_sha256"]
    try:
        candidates = list(asset_cache.iterdir())
    except OSError as exc:
        logging.warning("Could not inspect stale visual caches for %s: %s", asset.key, exc)
        return
    for candidate in candidates:
        try:
            _assert_visual_cache_containment(candidate, cache_root)
            if candidate.name == current_binding:
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
            elif candidate.is_file():
                candidate.unlink()
            else:
                raise UnsafeVisualCachePathError(
                    f"visual cache contains an unsafe filesystem object: {candidate}"
                )
        except UnsafeVisualCachePathError:
            raise
        except OSError as exc:
            logging.warning("Could not remove stale visual cache %s: %s", candidate, exc)


def _remove_bound_visual_cache(asset_cache, asset, cache_root=None):
    expected_parent = _asset_visual_cache_parent(asset, cache_root)
    cache_path = _assert_visual_cache_containment(asset_cache, cache_root)
    resolved_cache = cache_path.resolve(strict=False)
    if (
        resolved_cache.parent != expected_parent
        or not _is_sha256(resolved_cache.name)
    ):
        raise UnsafeVisualCachePathError(
            f"refusing unsafe visual cache removal: {resolved_cache}"
        )
    if not os.path.lexists(cache_path):
        return
    if cache_path.is_dir():
        shutil.rmtree(cache_path)
    elif cache_path.is_file():
        cache_path.unlink()
    else:
        raise UnsafeVisualCachePathError(
            f"visual cache binding has an unsafe filesystem type: {cache_path}"
        )


def _assert_visual_implementation_current(asset, cache_root=None):
    """Fail closed and discard visuals if loaded source bytes changed on disk."""

    try:
        _assert_visual_implementation_sources_unchanged()
    except VisualImplementationChangedError:
        _remove_asset_visual_cache(asset, cache_root, required=True)
        raise


def _cached_render_result(
    prior_result,
    result,
    asset_cache,
    binding,
    geometry_payloads,
    thumbnail_sha256,
):
    """Recover exact previously submitted PNG bytes from an incomplete checkpoint."""

    if not isinstance(prior_result, Mapping):
        return None
    if prior_result.get("visual_cache_binding") != binding:
        return None
    if not dependency_fingerprints_match(
        result.get("asset_dependency_fingerprint"),
        prior_result.get("asset_dependency_fingerprint"),
    ):
        return None

    segments = result.get("segments")
    prior_segments = prior_result.get("segments")
    prior_context = prior_result.get("visual_input_asset_context")
    if (
        not isinstance(segments, Mapping)
        or not isinstance(prior_segments, Mapping)
        or set(segments) != set(prior_segments)
        or not isinstance(prior_context, Mapping)
        or prior_context.get("status") != "complete"
    ):
        return None

    try:
        prior_thumbnail = prior_context["isaac_thumbnail"]
        if (
            not isinstance(prior_thumbnail, Mapping)
            or prior_thumbnail.get("status") != "complete"
            or prior_thumbnail.get("role") != "isaac_thumbnail"
            or prior_thumbnail.get("file_sha256") != thumbnail_sha256
        ):
            return None

        whole_object = copy.deepcopy(prior_context["textured_whole_object"])
        whole_path = asset_cache / "textured_whole_object.png"
        whole_object["output_path"] = str(whole_path)

        rendered_segments = {}
        for segment_key, geometry_payload in geometry_payloads.items():
            segment_info = segments[segment_key]
            prior_segment = prior_segments[segment_key]
            if not isinstance(segment_info, Mapping) or not isinstance(
                prior_segment, Mapping
            ):
                return None
            segment_id = segment_info.get("segment_id")
            if not _is_sha256(segment_id) or prior_segment.get("segment_id") != segment_id:
                return None
            provenance = prior_segment.get("visual_input_provenance")
            prior_input = prior_segment.get("vlm_input")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("status") != "complete"
                or not isinstance(prior_input, Mapping)
                or tuple(prior_input.get("image_roles", ())) != VLM_IMAGE_ROLES
                or tuple(provenance.get("image_roles", ())) != VLM_IMAGE_ROLES
                or provenance.get("geometry_payload") != geometry_payload.manifest()
            ):
                return None

            segment_directory = asset_cache / "segments" / segment_id
            selected_part = copy.deepcopy(provenance["textured_selected_part"])
            material_sphere = copy.deepcopy(provenance["selected_material_sphere"])
            selected_path = segment_directory / "textured_selected_part.png"
            sphere_path = segment_directory / "selected_material_sphere.png"
            selected_part["output_path"] = str(selected_path)
            material_sphere["output_path"] = str(sphere_path)
            expected_image_hashes = (
                prior_thumbnail["file_sha256"],
                whole_object["output_file_sha256"],
                selected_part["output_file_sha256"],
                material_sphere["output_file_sha256"],
            )
            if (
                tuple(prior_input.get("image_hashes", ())) != expected_image_hashes
                or tuple(provenance.get("image_hashes", ()))
                != expected_image_hashes
            ):
                return None
            rendered_segments[segment_key] = {
                "textured_selected_part": selected_part,
                "selected_material_sphere": material_sphere,
            }
    except (KeyError, TypeError, ValueError, OSError):
        return None

    return {
        "status": "complete",
        "asset": {"textured_whole_object": whole_object},
        "segments": rendered_segments,
    }


def acquire_output_writer_lock(output_file):
    """Hold a non-blocking, process-wide lock for one checkpoint writer."""

    output_path = Path(output_file).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.writer.lock")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
        os.close(lock_fd)
        raise RuntimeError(f"output writer lock is not a regular file: {lock_path}")
    lock_file = os.fdopen(lock_fd, "r+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(
            f"another annotation process is already writing {output_file!r}"
        ) from exc
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.uname().nodename}:{os.getpid()}\n")
        lock_file.flush()
        os.fsync(lock_file.fileno())
    except Exception:
        lock_file.close()
        raise
    return lock_file


def _without_runtime_paths(value):
    """Remove temporary renderer paths from saved metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_paths(child)
            for key, child in value.items()
            if key != "output_path"
        }
    if isinstance(value, list):
        return [_without_runtime_paths(child) for child in value]
    if isinstance(value, tuple):
        return [_without_runtime_paths(child) for child in value]
    return value


def _validate_gpu_render_record(
    record,
    *,
    role,
    size,
    segment_id=None,
    expected_output_path=None,
    cache_root=None,
):
    """Reject mislabeled, CPU, stale, or malformed renders before VLM use."""

    if not isinstance(record, Mapping):
        raise ValueError(f"{role} render record is missing")
    expected = {
        "status": "complete",
        "role": role,
        "engine": "CYCLES",
        "width": size,
        "height": size,
    }
    mismatches = [
        key for key, value in expected.items() if record.get(key) != value
    ]
    if segment_id is not None and record.get("segment_id") != segment_id:
        mismatches.append("segment_id")
    backend = record.get("render_backend")
    if backend not in {"CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"}:
        mismatches.append("render_backend")
    devices = record.get("render_device")
    devices = [devices] if isinstance(devices, str) else devices
    if (
        not isinstance(devices, (list, tuple))
        or not devices
        or any(
            not isinstance(device, str)
            or not device.strip()
            or "cpu" in device.casefold()
            or device.partition(":")[0].strip().upper() != backend
            for device in devices
        )
    ):
        mismatches.append("render_device")
    output_path = record.get("output_path")
    output_hash = record.get("output_file_sha256")
    if not isinstance(output_path, str):
        mismatches.append("output_path")
    else:
        checked_path = _assert_visual_cache_containment(output_path, cache_root)
        if expected_output_path is not None:
            expected_path = _assert_visual_cache_containment(
                expected_output_path, cache_root
            )
            if checked_path.resolve(strict=False) != expected_path.resolve(
                strict=False
            ):
                mismatches.append("output_path")
        if not os.path.lexists(checked_path) or not stat.S_ISREG(
            os.lstat(checked_path).st_mode
        ):
            mismatches.append("output_path")
        elif _sha256_file(checked_path) != output_hash:
            mismatches.append("output_file_sha256")
    if mismatches:
        raise ValueError(
            f"{role} render failed validation: {', '.join(sorted(set(mismatches)))}"
        )


def _metadata_matches(observed, expected):
    """Compare saved render settings, allowing for floating-point rounding."""

    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and set(observed) == set(expected)
            and all(
                _metadata_matches(observed[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(observed, (list, tuple))
            and len(observed) == len(expected)
            and all(
                _metadata_matches(left, right)
                for left, right in zip(observed, expected)
            )
        )
    if isinstance(expected, float):
        if not isinstance(observed, numbers.Real) or isinstance(observed, bool):
            return False
        try:
            observed_float = float(observed)
        except (TypeError, ValueError, OverflowError):
            return False
        return math.isfinite(observed_float) and math.isclose(
            observed_float, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        )
    return type(observed) is type(expected) and observed == expected


def _expected_whole_object_view(result, segments, geometry_payloads):
    """Build whole-object render metadata from the extracted geometry."""

    payloads = [
        geometry_payloads[key]
        for key, segment in segments.items()
        if _textured_renderer._render_content(segment)
    ]
    if not payloads:
        raise ValueError("asset has no visible render/default-purpose geometry")
    up_axes = {
        str(payload.provenance.get("stage_up_axis")).upper()
        for payload in payloads
    }
    if len(up_axes) != 1:
        raise ValueError(f"asset geometry has inconsistent up axes: {up_axes!r}")
    up_axis = next(iter(up_axes))
    perspective_hint = _textured_renderer._perspective_camera_horizontal_hint(
        result, up_axis
    )
    thickness_ratio = _textured_renderer._shape_thickness_ratio(payloads)
    uses_context_view = (
        thickness_ratio >= _textured_renderer._THREE_QUARTER_THICKNESS_RATIO
    )
    _, view = _textured_renderer._choose_view_direction(
        payloads,
        up_axis,
        three_quarter_context=uses_context_view,
        horizontal_hint=(
            perspective_hint[0] if perspective_hint is not None else None
        ),
        camera_hint_sha256=(
            perspective_hint[1] if perspective_hint is not None else None
        ),
    )
    view.update(
        {
            "shape_thickness_ratio": thickness_ratio,
            "three_quarter_threshold": (
                _textured_renderer._THREE_QUARTER_THICKNESS_RATIO
            ),
            "shape_view_class": (
                "volumetric_three_quarter"
                if uses_context_view
                else "thin_maximum_area"
            ),
        }
    )
    return view


def _validate_canonical_record_hash(record, field, domain, description):
    if not isinstance(record, Mapping):
        raise ValueError(f"{description} is missing")
    expected = record.get(field)
    if not _is_sha256(expected):
        raise ValueError(f"{description} has no valid {field}")
    payload = dict(record)
    payload.pop(field, None)
    if _canonical_sha256(domain, payload) != expected:
        raise ValueError(f"{description} {field} is stale or corrupt")
    return expected


def _validate_selected_geometry_isolation(selected_part, *, segment_key):
    isolation = selected_part.get("selected_geometry_isolation")
    _validate_canonical_record_hash(
        isolation,
        "physical_isolation_sha256",
        "selected-render-physical-isolation",
        f"selected-geometry isolation for {segment_key!r}",
    )

    source_count = isolation.get("source_occurrence_count")
    rendered_count = isolation.get("rendered_occurrence_count")
    proxies = isolation.get("proxies")
    if (
        type(source_count) is not int
        or source_count < 1
        or type(rendered_count) is not int
        or rendered_count != source_count
        or type(isolation.get("nonselected_polygon_count")) is not int
        or isolation["nonselected_polygon_count"] != 0
        or isolation.get("original_asset_objects_hidden_from_render") is not True
        or not _is_sha256(isolation.get("all_occurrence_matrices_sha256"))
        or not isinstance(proxies, list)
        or len(proxies) != rendered_count
    ):
        raise ValueError(
            f"selected geometry includes unexpected parts for {segment_key!r}"
        )

    occurrence_indices = []
    required_proxy_hashes = (
        "selected_source_polygon_indices_sha256",
        "selected_source_loop_indices_sha256",
        "vertices_sha256",
        "loop_vertices_sha256",
        "corner_normals_sha256",
        "renderer_custom_corner_normals_sha256",
        "matrix_world_sha256",
    )
    for proxy in proxies:
        if not isinstance(proxy, Mapping):
            raise ValueError(
                f"selected-geometry proxy is malformed for {segment_key!r}"
            )
        source_index = proxy.get("source_occurrence_index")
        source_uv_hashes = proxy.get("source_uv_sha256")
        display_hashes = (
            proxy.get("display_color_sha256"),
            proxy.get("display_opacity_sha256"),
        )
        if (
            type(source_index) is not int
            or source_index < 0
            or source_index >= source_count
            or any(not _is_sha256(proxy.get(field)) for field in required_proxy_hashes)
            or not isinstance(source_uv_hashes, Mapping)
            or any(
                not isinstance(name, str)
                or not name
                or not _is_sha256(value)
                for name, value in source_uv_hashes.items()
            )
            or any(value is not None and not _is_sha256(value) for value in display_hashes)
            or proxy.get("exact_corner_payload_and_display_pre_post_equal") is not True
            or any(
                type(proxy.get(field)) is not int or proxy[field] < 1
                for field in ("polygon_count", "loop_count", "vertex_count")
            )
        ):
            raise ValueError(
                f"selected-geometry proxy evidence is wrong for {segment_key!r}"
            )
        occurrence_indices.append(source_index)

    if occurrence_indices != list(range(source_count)):
        raise ValueError(
            f"selected-geometry occurrences are incomplete for {segment_key!r}"
        )


def _validate_material_sphere_sampling_provenance(
    material_sphere,
    *,
    segment_key,
    material_record_id,
    material_plan_sha256,
    geometry_payload_sha256,
    size,
):
    """Check the material sphere and texture swatches for a segment."""

    if (
        material_sphere.get("material_record_id") != material_record_id
        or material_sphere.get("material_plan_sha256") != material_plan_sha256
        or material_sphere.get("geometry_payload_sha256")
        != geometry_payload_sha256
    ):
        raise ValueError(f"material-sphere binding is wrong for {segment_key!r}")

    sampling = material_sphere.get("selected_segment_swatch_sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(
            f"selected-segment sphere sampling is missing for {segment_key!r}"
        )
    plan_sha256 = _validate_canonical_record_hash(
        sampling,
        "segment_sampling_plan_sha256",
        "selected-segment-swatch-plan",
        f"selected-segment sphere sampling for {segment_key!r}",
    )
    if (
        material_sphere.get("segment_sampling_plan_sha256") != plan_sha256
        or sampling.get("geometry_payload_sha256") != geometry_payload_sha256
    ):
        raise ValueError(
            f"selected-segment sphere sampling binding is wrong for {segment_key!r}"
        )

    recipe = sampling.get("swatch_recipe")
    expected_recipe = {
        "target_uv": {
            "angle_limit_degrees": 66.0,
            "margin_method": "FRACTION",
            "rotate_method": "AXIS_ALIGNED_Y",
            "island_margin_pixels": 1.0,
            "zero_mask_retry_island_margin": 0.0,
            "area_weight": 0.0,
            "correct_aspect": True,
            "scale_to_bounds": True,
        },
        "bake": {
            "samples": 1,
            "color_and_scalar_type": "EMIT",
            "tangent_normal_type": "NORMAL",
            "margin_pixels": 0,
        },
    }
    if (
        recipe != expected_recipe
        or sampling.get("swatch_recipe_sha256")
        != _canonical_sha256("selected-segment-swatch-recipe", expected_recipe)
    ):
        raise ValueError(
            f"selected-segment sphere swatch recipe is wrong for {segment_key!r}"
        )

    swatches = sampling.get("swatches")
    node_attestation = sampling.get("sphere_node_tree_attestation")
    if not isinstance(swatches, Mapping) or not isinstance(
        node_attestation, Mapping
    ):
        raise ValueError(f"sphere swatch evidence is missing for {segment_key!r}")
    _validate_canonical_record_hash(
        node_attestation,
        "node_tree_attestation_sha256",
        "segment-swatch-sphere-node-tree",
        f"sphere node-tree attestation for {segment_key!r}",
    )
    role_to_node = node_attestation.get("role_to_node")
    role_to_pixels = node_attestation.get("role_to_pixels_sha256")
    if (
        not isinstance(role_to_node, Mapping)
        or not isinstance(role_to_pixels, Mapping)
        or set(role_to_node) != set(swatches)
        or set(role_to_pixels) != set(swatches)
        or type(node_attestation.get("reachable_node_count")) is not int
        or type(node_attestation.get("reachable_image_node_count")) is not int
        or node_attestation["reachable_image_node_count"] != len(swatches)
        or node_attestation["reachable_node_count"]
        < node_attestation["reachable_image_node_count"] + 1
    ):
        raise ValueError(
            f"sphere node-tree swatch coverage is wrong for {segment_key!r}"
        )

    for role, swatch in swatches.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(swatch, Mapping)
            or swatch.get("role") != role
            or swatch.get("kind") not in {"color", "scalar", "tangent_normal"}
            or swatch.get("width") != size
            or swatch.get("height") != size
            or not _is_sha256(swatch.get("source_evidence_sha256"))
            or not _is_sha256(swatch.get("pixels_sha256"))
            or role_to_pixels.get(role) != swatch.get("pixels_sha256")
            or not isinstance(role_to_node.get(role), str)
            or not role_to_node.get(role)
            or swatch.get("bake_type")
            != ("NORMAL" if swatch.get("kind") == "tangent_normal" else "EMIT")
        ):
            raise ValueError(f"sphere swatch {role!r} is invalid for {segment_key!r}")

    if not swatches:
        return
    if (
        sampling.get("linear_float_working_images") is not True
        or sampling.get("cycles_bake_samples") != 1
        or sampling.get("png_render_settings_restored_after_each_bake") is not True
        or sampling.get("target_uv") != "SegmentSwatchUV"
    ):
        raise ValueError(f"sphere swatch bake settings are wrong for {segment_key!r}")

    geometry = sampling.get("geometry")
    _validate_canonical_record_hash(
        geometry,
        "evaluated_sampling_sha256",
        "selected-evaluated-segment-swatch",
        f"evaluated swatch geometry for {segment_key!r}",
    )
    mask = sampling.get("mask")
    if (
        not isinstance(mask, Mapping)
        or type(mask.get("valid_pixel_count")) is not int
        or mask["valid_pixel_count"] < 1
        or mask.get("total_pixel_count") != size * size
        or not isinstance(mask.get("valid_pixel_fraction"), (int, float))
        or abs(
            mask["valid_pixel_fraction"]
            - mask["valid_pixel_count"] / float(size * size)
        )
        > 1e-12
        or not _is_sha256(mask.get("valid_mask_sha256"))
        or not _is_sha256(mask.get("nearest_selected_pixel_indices_sha256"))
    ):
        raise ValueError(f"sphere swatch mask evidence is wrong for {segment_key!r}")


def _validate_rendered_visuals(
    rendered,
    result,
    asset,
    geometry_payloads,
    asset_cache,
    *,
    size,
    cache_root=None,
):
    """Check rendered images and their metadata."""

    if not isinstance(rendered, Mapping) or rendered.get("status") != "complete":
        raise ValueError(
            f"textured GPU rendering is incomplete for {asset.key}: "
            f"{rendered.get('status') if isinstance(rendered, Mapping) else None!r}"
        )
    segments = result.get("segments")
    rendered_asset = rendered.get("asset")
    rendered_segments = rendered.get("segments")
    if (
        not isinstance(segments, Mapping)
        or not isinstance(rendered_asset, Mapping)
        or not isinstance(rendered_segments, Mapping)
    ):
        raise ValueError(f"textured renderer returned a malformed result for {asset.key}")
    if set(rendered_segments) != set(segments):
        raise ValueError(f"textured renderer did not cover every segment for {asset.key}")

    whole_object = rendered_asset.get("textured_whole_object")
    _validate_gpu_render_record(
        whole_object,
        role="textured_whole_object",
        size=size,
        expected_output_path=asset_cache / "textured_whole_object.png",
        cache_root=cache_root,
    )
    expected_payload_hashes = {
        key: payload.manifest()["payload_sha256"]
        for key, payload in geometry_payloads.items()
    }
    if whole_object.get("geometry_payload_hashes") != expected_payload_hashes:
        raise ValueError(f"whole-object render geometry binding is wrong for {asset.key}")
    whole_material_plan_hashes = whole_object.get("material_plan_hashes")
    if not isinstance(whole_material_plan_hashes, Mapping) or set(
        whole_material_plan_hashes
    ) != set(segments):
        raise ValueError(
            f"whole-object render material coverage is wrong for {asset.key}"
        )
    expected_whole_view = _expected_whole_object_view(
        result, segments, geometry_payloads
    )
    if not _metadata_matches(whole_object.get("view"), expected_whole_view):
        raise ValueError(
            f"whole-object camera/view provenance is wrong for {asset.key}"
        )

    for segment_key, geometry_payload in geometry_payloads.items():
        segment_info = segments.get(segment_key)
        segment_renders = rendered_segments.get(segment_key)
        if not isinstance(segment_info, Mapping) or not isinstance(
            segment_renders, Mapping
        ):
            raise ValueError(f"textured renders are missing for {segment_key!r}")
        segment_id = segment_info.get("segment_id")
        if not _is_sha256(segment_id):
            raise ValueError(f"segment ID is malformed for {segment_key!r}")
        selected_part = segment_renders.get("textured_selected_part")
        material_sphere = segment_renders.get("selected_material_sphere")
        segment_directory = asset_cache / "segments" / segment_id
        _validate_gpu_render_record(
            selected_part,
            role="textured_selected_part",
            size=size,
            segment_id=segment_id,
            expected_output_path=segment_directory / "textured_selected_part.png",
            cache_root=cache_root,
        )
        _validate_gpu_render_record(
            material_sphere,
            role="selected_material_sphere",
            size=size,
            segment_id=segment_id,
            expected_output_path=segment_directory / "selected_material_sphere.png",
            cache_root=cache_root,
        )
        payload_sha256 = geometry_payload.manifest()["payload_sha256"]
        material_record_id = segment_info.get("material_record_id")
        if selected_part.get("geometry_payload_sha256") != payload_sha256:
            raise ValueError(
                f"selected-part render geometry binding is wrong for {segment_key!r}"
            )
        if selected_part.get("material_record_id") != material_record_id:
            raise ValueError(
                f"selected-part material binding is wrong for {segment_key!r}"
            )
        _validate_selected_geometry_isolation(
            selected_part,
            segment_key=segment_key,
        )
        material_plan_sha256 = selected_part.get("material_plan_sha256")
        if (
            not _is_sha256(material_plan_sha256)
            or whole_material_plan_hashes.get(segment_key) != material_plan_sha256
        ):
            raise ValueError(f"rendered material plans disagree for {segment_key!r}")
        _validate_material_sphere_sampling_provenance(
            material_sphere,
            segment_key=segment_key,
            material_record_id=material_record_id,
            material_plan_sha256=material_plan_sha256,
            geometry_payload_sha256=payload_sha256,
            size=size,
        )

    return whole_object, rendered_segments


def _prepare_asset_visual_inputs_loaded(
    result,
    asset,
    *,
    dataset_name,
    object_name,
    thumbnail_path,
    text_context_hash,
    prior_result=None,
    cache_root=None,
):
    """Prepare and check the four VLM input images for each segment."""

    annotation_input_gate = extraction_gate(result)
    if annotation_input_gate.get("status") != "complete":
        raise ValueError(
            "asset extraction is not safe for VLM visual construction: "
            f"{annotation_input_gate.get('annotation_input_blockers')!r}"
        )

    if not thumbnail_path or not os.path.isfile(thumbnail_path):
        raise FileNotFoundError(
            f"exact source/Isaac thumbnail is unavailable for {asset.key}"
        )
    thumbnail_hash = _sha256_file(thumbnail_path)
    thumbnail_record = {
        "status": "complete",
        "role": "isaac_thumbnail",
        "file_sha256": thumbnail_hash,
    }
    geometry_payloads = build_asset_geometry_payloads(asset.path, result)
    segments = result.get("segments")
    if not isinstance(segments, Mapping):
        raise ValueError("extraction has no segment mapping")
    if set(geometry_payloads) != set(segments):
        raise ValueError("geometry payload coverage does not match extracted segments")
    cache_binding = _visual_cache_binding(
        result,
        asset,
        geometry_payloads,
        size=512,
    )
    result["visual_cache_binding"] = cache_binding
    asset_cache = _asset_visual_cache_directory(asset, cache_binding, cache_root)
    _acquire_asset_visual_cache_lock(asset, cache_root)
    _assert_visual_cache_containment(asset_cache, cache_root)
    _remove_stale_visual_cache_bindings(asset, cache_binding, cache_root)
    rendered = _cached_render_result(
        prior_result,
        result,
        asset_cache,
        cache_binding,
        geometry_payloads,
        thumbnail_hash,
    )
    validated_render = None
    if rendered is not None:
        try:
            validated_render = _validate_rendered_visuals(
                rendered,
                result,
                asset,
                geometry_payloads,
                asset_cache,
                size=512,
                cache_root=cache_root,
            )
        except UnsafeVisualCachePathError:
            raise
        except (KeyError, OSError, TypeError, ValueError) as exc:
            logging.warning(
                "Discarding invalid cached VLM visuals for %s: %s",
                asset.key,
                exc,
            )
            rendered = None
    if validated_render is None:
        _remove_bound_visual_cache(asset_cache, asset, cache_root)
        asset_cache.mkdir(parents=True, exist_ok=True)
        _assert_visual_cache_containment(
            asset_cache,
            cache_root,
            final_must_be_directory=True,
        )
        rendered = render_textured_vlm_inputs(
            asset.path,
            result,
            asset_cache,
            geometry_payloads=geometry_payloads,
            size=512,
        )
        validated_render = _validate_rendered_visuals(
            rendered,
            result,
            asset,
            geometry_payloads,
            asset_cache,
            size=512,
            cache_root=cache_root,
        )
    else:
        logging.info("Reusing exact cached VLM visual bytes for %s", asset.key)
    whole_object, rendered_segments = validated_render
    if _sha256_file(thumbnail_path) != thumbnail_hash:
        _remove_bound_visual_cache(asset_cache, asset, cache_root)
        raise ValueError(
            f"Isaac thumbnail changed during visual preparation for {asset.key}"
        )
    result["visual_input_asset_context"] = {
        "status": "complete",
        "isaac_thumbnail": thumbnail_record,
        "textured_whole_object": _without_runtime_paths(whole_object),
    }

    runtime_paths = {}
    for segment_key, geometry_payload in geometry_payloads.items():
        segment_info = segments.get(segment_key)
        if not isinstance(segment_info, dict):
            raise ValueError(f"segment metadata is missing for {segment_key!r}")
        segment_renders = rendered_segments.get(segment_key)
        selected_part = segment_renders.get("textured_selected_part")
        material_sphere = segment_renders.get("selected_material_sphere")

        spec = build_vlm_input_spec(
            asset_key=asset.key,
            segment_info=segment_info,
            dataset=dataset_name,
            object_name=object_name,
            isaac_thumbnail_hash=thumbnail_hash,
            textured_whole_object_hash=whole_object["output_file_sha256"],
            textured_selected_part_hash=selected_part["output_file_sha256"],
            selected_material_sphere_hash=material_sphere["output_file_sha256"],
            text_context_hash=text_context_hash,
            object_category=result.get("category"),
        )
        input_errors = validate_vlm_input_spec(
            spec, object_name
        )
        if input_errors:
            raise ValueError(
                f"Invalid VLM inputs for {segment_key!r}: "
                + "; ".join(input_errors)
            )
        segment_info["vlm_input"] = spec.to_dict()
        segment_info["visual_input_provenance"] = {
            "status": "complete",
            "image_roles": list(spec.image_roles),
            "image_hashes": list(spec.image_hashes),
            "textured_selected_part": _without_runtime_paths(selected_part),
            "selected_material_sphere": _without_runtime_paths(material_sphere),
            "geometry_payload": geometry_payload.manifest(),
        }
        # The request builder is the sole source of the production prompt.
        segment_info["user_prompt"] = spec.prompt
        runtime_paths[segment_key] = (
            str(thumbnail_path),
            str(whole_object["output_path"]),
            str(selected_part["output_path"]),
            str(material_sphere["output_path"]),
        )

    if set(runtime_paths) != set(segments):
        raise ValueError(
            "not every extracted segment received the exact four visual inputs"
        )
    if _sha256_file(thumbnail_path) != thumbnail_hash:
        _remove_bound_visual_cache(asset_cache, asset, cache_root)
        raise ValueError(
            f"Isaac thumbnail changed during visual preparation for {asset.key}"
        )
    return runtime_paths


def _prepare_asset_visual_inputs(
    result,
    asset,
    *,
    dataset_name,
    object_name,
    thumbnail_path,
    text_context_hash,
    prior_result=None,
    cache_root=None,
):
    """Run visual preparation only while its loaded implementation stays current."""

    _assert_visual_implementation_current(asset, cache_root)
    try:
        return _prepare_asset_visual_inputs_loaded(
            result,
            asset,
            dataset_name=dataset_name,
            object_name=object_name,
            thumbnail_path=thumbnail_path,
            text_context_hash=text_context_hash,
            prior_result=prior_result,
            cache_root=cache_root,
        )
    finally:
        _assert_visual_implementation_current(asset, cache_root)


def _sha256_file(path, chunk_size=1024 * 1024):
    """Hash the selected entry layer's bytes without loading it into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_asset_identity(result, asset):
    """Store identity plus a complete, portable source-input fingerprint."""

    result["asset_key"] = asset.key
    result["asset_identity_hash"] = asset.identity_hash
    result["entrypoint_content_sha256"] = _sha256_file(asset.path)
    additional_dependencies = []
    if asset.thumbnail_path is not None:
        additional_dependencies.append(("object_thumbnail", asset.thumbnail_path))
    dependency_fingerprint = build_asset_dependency_fingerprint(
        asset.path,
        asset.dataset_root,
        extraction_result=result,
        additional_dependencies=additional_dependencies,
    )
    result["asset_dependency_fingerprint"] = dependency_fingerprint
    if dependency_fingerprint["status"] != "complete":
        inventory = result.setdefault("segment_inventory", {})
        inventory_errors = inventory.setdefault("errors", [])
        for error in dependency_fingerprint["errors"]:
            message = f"asset dependency fingerprint incomplete: {error}"
            if message not in inventory_errors:
                inventory_errors.append(message)
    result["asset_hash"] = asset.identity_hash


def _verify_asset_inputs_unchanged(result, asset):
    """Reject annotations if any source byte changed during processing."""

    dependencies = []
    if asset.thumbnail_path is not None:
        dependencies.append(("object_thumbnail", asset.thumbnail_path))
    final_fingerprint = build_asset_dependency_fingerprint(
        asset.path,
        asset.dataset_root,
        extraction_result=result,
        additional_dependencies=dependencies,
    )
    initial_fingerprint = result.get("asset_dependency_fingerprint")
    matched = dependency_fingerprints_match(
        initial_fingerprint, final_fingerprint
    )
    result["asset_dependency_final_verification"] = {
        "status": "unchanged" if matched else "changed_or_incomplete",
        "fingerprint_sha256": final_fingerprint.get("fingerprint_sha256"),
        "errors": final_fingerprint.get("errors", []),
        "warnings": final_fingerprint.get("warnings", []),
    }
    if matched:
        return True

    error = (
        "asset USD/material/texture/thumbnail dependencies changed or became "
        "unreadable while visual inputs/VLM annotations were being produced"
    )
    result["input_integrity_error"] = error
    segments = result.get("segments")
    if isinstance(segments, Mapping):
        for segment_info in segments.values():
            if not isinstance(segment_info, dict):
                continue
            annotation = segment_info.get("vlm_annotation")
            if isinstance(annotation, dict):
                annotation["status"] = "stale_input"
                annotation["error"] = error
    return False


def _restore_compatible_segment_annotations(
    result,
    prior_result,
    *,
    dataset_name,
    expected_model_identity,
    expected_text_context_hash,
):
    """Reuse saved segment responses when the prompt and images match."""

    if not isinstance(prior_result, Mapping):
        return 0
    if not dependency_fingerprints_match(
        result.get("asset_dependency_fingerprint"),
        prior_result.get("asset_dependency_fingerprint"),
    ):
        return 0
    current_segments = result.get("segments")
    prior_segments = prior_result.get("segments")
    if not isinstance(current_segments, Mapping) or not isinstance(
        prior_segments, Mapping
    ):
        return 0

    prior_by_segment_id = {}
    ambiguous_segment_ids = set()
    for prior_segment in prior_segments.values():
        segment_id = (
            prior_segment.get("segment_id")
            if isinstance(prior_segment, Mapping)
            else None
        )
        if not isinstance(segment_id, str) or not segment_id:
            continue
        if segment_id in prior_by_segment_id:
            ambiguous_segment_ids.add(segment_id)
            continue
        prior_by_segment_id[segment_id] = prior_segment
    for segment_id in ambiguous_segment_ids:
        prior_by_segment_id.pop(segment_id, None)
    restored = 0
    for segment_key, segment_info in current_segments.items():
        if not isinstance(segment_info, dict):
            continue
        prior_segment = prior_by_segment_id.get(segment_info.get("segment_id"))
        if not isinstance(prior_segment, Mapping):
            continue
        current_input = segment_info.get("vlm_input")
        prior_input = prior_segment.get("vlm_input")
        if (
            not isinstance(current_input, Mapping)
            or not isinstance(prior_input, Mapping)
            or current_input.get("request_hash") != prior_input.get("request_hash")
        ):
            continue
        if not is_segment_annotation_complete(
            prior_result,
            prior_segment,
            expected_model_identity=expected_model_identity,
            expected_text_context_hash=expected_text_context_hash,
        ):
            continue

        annotation = copy.deepcopy(prior_segment["vlm_annotation"])
        properties = annotation["properties"]
        segment_info["vlm_annotation"] = annotation
        segment_info["youngs_modulus"] = properties["youngs_modulus"]
        segment_info["poissons_ratio"] = properties["poissons_ratio"]
        if dataset_name == "simready":
            segment_info["vlm_density"] = properties["density"]
        else:
            segment_info["density"] = properties["density"]
        for field in ("vlm_analysis", "prompt"):
            value = prior_segment.get(field)
            if isinstance(value, str) and value:
                segment_info[field] = value
        restored += 1
        logging.info(
            "Resumed complete VLM annotation for %s from its exact request hash",
            segment_key,
        )
    return restored


def _replace_asset_result(all_results, result, asset):
    """Replace an old-schema/exact-path record instead of duplicating it."""

    target_path = os.path.realpath(asset.path)
    target_identity = asset.identity_hash
    all_results[:] = [
        item
        for item in all_results
        if (item.get("asset_identity_hash") or item.get("asset_hash"))
        != target_identity
        and item.get("asset_key") != asset.key
        and (
            not item.get("file_path")
            or os.path.realpath(item["file_path"]) != target_path
        )
    ]
    all_results.append(result)


def _checkpoint_pre_vlm_failure(
    result,
    prior_result,
    asset,
    all_results,
    output_file,
    *,
    error,
    vlm_model=None,
    vlm_processor=None,
    block_visual_inputs=False,
):
    """Keep paid partial responses unless no prior checkpoint exists to preserve."""

    if isinstance(prior_result, Mapping):
        logging.error(
            "Preserving the prior checkpoint for %s after pre-VLM failure: %s",
            asset.key,
            error,
        )
        return False

    if block_visual_inputs:
        result["visual_input_error"] = error
        model_id = vlm_model_identity(vlm_model, vlm_processor)
        segments = result.get("segments")
        if isinstance(segments, Mapping):
            for segment_info in segments.values():
                if isinstance(segment_info, dict):
                    segment_info["vlm_annotation"] = {
                        "status": "blocked",
                        "properties": {},
                        "model_identity": model_id,
                        "error": error,
                    }
    update_annotation_status(result)
    _replace_asset_result(all_results, result, asset)
    if output_file:
        append_result_to_file(result, all_results, output_file)
    return True


def load_existing_results(output_file):
    """
    Load existing results from a JSON file.

    Args:
        output_file (str): Path to the JSON file

    Returns:
        tuple: (results list, set of already processed object names, set of already processed file paths)
    """
    if not os.path.exists(output_file):
        return [], set(), set()
    try:
        with open(output_file, "r") as f:
            existing_results = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Refusing to overwrite unreadable checkpoint {output_file!r}"
        ) from exc
    if not isinstance(existing_results, list) or not all(
        isinstance(result, Mapping) for result in existing_results
    ):
        raise RuntimeError(
            f"Refusing to overwrite malformed checkpoint {output_file!r}: "
            "expected a JSON list of objects"
        )

    processed_objects = set()
    processed_file_paths = set()
    for result in existing_results:
        obj_name = result.get("object_name", "")
        file_path = result.get("file_path", "")
        if obj_name:
            processed_objects.add(obj_name)
        if file_path:
            processed_file_paths.add(file_path)

    logging.info(
        f"Loaded {len(existing_results)} existing results from {output_file}"
    )
    logging.info(
        f"Found {len(processed_objects)} already processed objects and "
        f"{len(processed_file_paths)} file paths"
    )
    if processed_objects:
        logging.info(f"Processed objects: {', '.join(processed_objects)}")
    return existing_results, processed_objects, processed_file_paths


def _attempt_prompt_for_output(segment_info, dataset_name, attempt_label):
    """Recover and verify the exact user text submitted for one attempt."""

    vlm_input = segment_info.get("vlm_input")
    if not isinstance(vlm_input, Mapping):
        raise ValueError("full annotation segment has no VLM input record")
    base_prompt = vlm_input.get("prompt")
    if not isinstance(base_prompt, str) or not base_prompt:
        raise ValueError("full annotation segment has no segment prompt")
    prompts = SIMREADY_PROMPTS if dataset_name == "simready" else RESIDENTIAL_PROMPTS
    expected_text_hash = current_vlm_text_context_hash(prompts)
    if vlm_input.get("text_context_hash") != expected_text_hash:
        raise ValueError("cannot export a prompt from a stale text context")
    query_prompt = prompts["query_prompt"]
    if attempt_label == "retry":
        base_prompt += STRICT_RETRY_USER_PROMPT_SUFFIX
        query_prompt += STRICT_RETRY_QUERY_PROMPT_SUFFIX
    elif attempt_label != "initial":
        raise ValueError(f"unknown VLM attempt label: {attempt_label!r}")
    return "\n\n".join((base_prompt, prompts["few_shot_examples"], query_prompt))


def _selected_attempt_for_output(segment_info, dataset_name):
    """Return one raw response, exact prompt, and its parsed numeric triple."""

    annotation = segment_info.get("vlm_annotation")
    if not isinstance(annotation, Mapping) or annotation.get("status") != "complete":
        raise ValueError("cannot compact an incomplete VLM annotation")
    attempts = annotation.get("attempts")
    properties = annotation.get("properties")
    provenance = annotation.get("property_provenance")
    if (
        not isinstance(attempts, list)
        or not isinstance(properties, Mapping)
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("complete VLM annotation is missing response provenance")

    selected_label = annotation.get("selected_attempt")
    if selected_label is None:
        sources = {
            provenance.get(name)
            for name in ("youngs_modulus", "poissons_ratio", "density")
        }
        if len(sources) != 1:
            raise ValueError("numeric properties came from different VLM responses")
        source = sources.pop()
        if not isinstance(source, str) or not source.startswith("vlm_"):
            raise ValueError("numeric properties have no VLM response provenance")
        selected_label = source.removeprefix("vlm_")

    selected = next(
        (
            attempt
            for attempt in attempts
            if isinstance(attempt, Mapping)
            and attempt.get("attempt") == selected_label
        ),
        None,
    )
    if not isinstance(selected, Mapping) or selected.get("status") != "parsed":
        raise ValueError("selected VLM response is missing or unparsed")
    selected_properties = selected.get("properties")
    if not isinstance(selected_properties, Mapping):
        raise ValueError("selected VLM response has no parsed properties")
    raw_output = selected.get("raw_vlm_output")
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("selected VLM response has no literal output text")
    if selected.get("raw_vlm_output_sha256") != raw_vlm_output_sha256(raw_output):
        raise ValueError("selected VLM response text failed its hash check")

    reparsed = parse_vlm_properties(raw_output)
    numeric_properties = {}
    for name in ("youngs_modulus", "poissons_ratio", "density"):
        selected_value = selected_properties.get(name)
        final_value = properties.get(name)
        parsed_value = reparsed.get(name)
        if not all(
            valid_vlm_property_value(name, value)
            for value in (selected_value, final_value, parsed_value)
        ) or not (
            float(selected_value) == float(final_value) == float(parsed_value)
        ):
            raise ValueError(
                f"saved {name} does not come from the selected raw VLM response"
            )
        numeric_properties[name] = float(parsed_value)

    exact_prompt = _attempt_prompt_for_output(
        segment_info, dataset_name, selected_label
    )
    stored_prompt = selected.get("request_prompt")
    if stored_prompt is not None and stored_prompt != exact_prompt:
        raise ValueError("stored VLM request prompt differs from the submitted input")
    return numeric_properties, raw_output, exact_prompt


def _compact_segment_for_output(segment_info, dataset_name):
    """Project one runtime/checkpoint segment onto the training-data schema."""

    if not isinstance(segment_info, Mapping):
        raise ValueError("segment output must be a mapping")
    if "vlm_annotation" in segment_info:
        properties, raw_output, prompt = _selected_attempt_for_output(
            segment_info, dataset_name
        )
    else:
        properties = {
            name: segment_info.get(name)
            for name in ("youngs_modulus", "poissons_ratio", "density")
        }
        raw_output = segment_info.get("vlm_analysis")
        prompt = segment_info.get("prompt")

    compact = {"name": segment_info.get("name")}
    if dataset_name == "simready":
        compact["opacity"] = segment_info.get("opacity")
    compact["material_type"] = segment_info.get("material_type")
    compact["semantic_usage"] = segment_info.get("semantic_usage")
    if dataset_name == "simready":
        source_physics = segment_info.get("source_physics")
        source_physics = (
            source_physics if isinstance(source_physics, Mapping) else segment_info
        )
        compact["density"] = properties["density"]
        compact["dynamic_friction"] = source_physics.get("dynamic_friction")
        compact["static_friction"] = source_physics.get("static_friction")
        compact["restitution"] = source_physics.get("restitution")
    compact["prompt"] = prompt
    compact["vlm_analysis"] = raw_output
    compact["youngs_modulus"] = properties["youngs_modulus"]
    compact["poissons_ratio"] = properties["poissons_ratio"]
    if dataset_name != "simready":
        compact["density"] = properties["density"]
    return compact


def _compact_result_for_output(result):
    """Project one asset onto the original compact dataset schema."""

    if not isinstance(result, Mapping):
        raise ValueError("asset output must be a mapping")
    dataset_name = result.get("dataset_type")
    if dataset_name not in {"simready", "residential", "commercial", "vegetation"}:
        raise ValueError(f"unknown output dataset: {dataset_name!r}")
    segments = result.get("segments")
    if not isinstance(segments, Mapping):
        raise ValueError("asset output has no segment mapping")
    compact_segments = {
        segment_key: _compact_segment_for_output(segment_info, dataset_name)
        for segment_key, segment_info in segments.items()
    }
    if dataset_name == "simready":
        return {
            "object_name": result.get("object_name"),
            "category": result.get("category"),
            "dataset_type": dataset_name,
            "segments": compact_segments,
            "file_path": result.get("file_path"),
        }
    return {
        "object_name": result.get("object_name"),
        "dataset_type": dataset_name,
        "file_path": result.get("file_path"),
        "date_processed": result.get("date_processed"),
        "segments": compact_segments,
    }


def _atomic_write_results(results, output_file):
    """Durably replace one JSON result list without exposing a partial file."""

    temporary_path = None
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f".{output_path.name}.tmp.{os.uname().nodename}.{os.getpid()}"
        )
        with temporary_path.open("w") as output_stream:
            json.dump(results, output_stream, indent=4, cls=UsdJsonEncoder)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary_path, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise OSError("replacement JSON is empty or missing")
    except Exception as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise CheckpointWriteError(
            f"could not atomically write results to {output_file!r}"
        ) from exc


def write_compact_results_to_file(all_results, output_file):
    """Replace a successful full checkpoint with the compact public dataset."""

    try:
        compact_results = [
            _compact_result_for_output(result) for result in all_results
        ]
    except Exception as exc:
        raise CheckpointWriteError(
            "could not project the full checkpoint onto the compact schema"
        ) from exc
    _atomic_write_results(compact_results, output_file)
    logging.info(
        "Wrote compact annotation dataset: %d assets, %d bytes",
        len(compact_results),
        os.path.getsize(output_file),
    )


def append_result_to_file(result, all_results, output_file):
    """
    Append a newly processed result to the output file.

    Args:
        result: The new result to append
        all_results: The full list of results (includes the new result)
        output_file: Path to the output file

    Returns:
        bool: True after an atomic, durable replacement.

    Raises:
        CheckpointWriteError: If the checkpoint cannot be written or verified.
    """
    _atomic_write_results(all_results, output_file)
    logging.info(
        f"Updated file with new object: {result.get('object_name', 'unknown')}"
    )
    logging.info(
        f"File size: {os.path.getsize(output_file)} bytes, contains {len(all_results)} objects"
    )
    return True


def analyze_material_with_retry(
    segment_key,
    segment_info,
    vlm_model,
    vlm_processor,
    dataset_name,
    prompts,
    user_prompt,
    visual_image_paths,
    input_request_hash,
    input_text_context_hash,
    model_identity,
):
    """Analyze one region and record VLM values separately from source physics."""

    required_props = ("youngs_modulus", "poissons_ratio", "density")
    properties = {name: None for name in required_props}
    provenance = {}
    attempts = []

    # These compatibility fields are regenerated from the structured result.
    segment_info.pop("youngs_modulus", None)
    segment_info.pop("poissons_ratio", None)
    segment_info.pop("vlm_density", None)
    segment_info.pop("vlm_analysis", None)
    segment_info.pop("prompt", None)
    if dataset_name != "simready":
        segment_info.pop("density", None)

    def valid_value(name, value):
        return valid_vlm_property_value(name, value)

    def require_current_text_context():
        current_hash = current_vlm_text_context_hash(prompts)
        if current_hash != input_text_context_hash:
            raise ValueError(
                "VLM text/reference context changed before request submission"
            )

    def record_attempt(
        label,
        response,
        *,
        attempt_request_hash,
        missing_properties,
    ):
        response = response if isinstance(response, Mapping) else {}
        raw_output = response.get("raw_vlm_output")
        response_error = str(response["error"]) if response.get("error") else None
        if not isinstance(raw_output, str):
            raw_output = ""
            response_error = response_error or (
                "VLM response omitted its exact raw output text"
            )
        computed_raw_hash = raw_vlm_output_sha256(raw_output)
        reported_raw_hash = response.get("raw_vlm_output_sha256")
        if reported_raw_hash not in (None, computed_raw_hash):
            response_error = response_error or (
                "VLM response raw-output SHA-256 did not match its exact text"
            )
        request_prompt = response.get("request_prompt")
        if not isinstance(request_prompt, str) or not request_prompt:
            request_prompt = ""
            response_error = response_error or (
                "VLM response omitted the exact submitted text prompt"
            )
        attempt_properties = {
            name: (None if response_error else response.get(name))
            for name in required_props
        }
        attempt_analysis = None if response_error else response.get("vlm_analysis")
        attempt = {
            "attempt": label,
            "attempt_request_hash": attempt_request_hash,
            "missing_properties": list(missing_properties),
            "status": "error" if response_error else "parsed",
            "properties": attempt_properties,
            "analysis": attempt_analysis,
            "raw_vlm_output": raw_output,
            "raw_vlm_output_sha256": computed_raw_hash,
            "request_prompt": request_prompt,
            "web_search_used": bool(response.get("web_search_used", False)),
            "response_model": response.get("response_model"),
            "response_system_fingerprint": response.get(
                "response_system_fingerprint"
            ),
        }
        if response_error:
            attempt["error"] = response_error
        attempts.append(attempt)
        return attempt

    require_current_text_context()
    first_response = analyze_material_with_vlm(
        user_prompt,
        vlm_model,
        vlm_processor,
        prompts,
        visual_image_paths,
    )
    first_attempt = record_attempt(
        "initial",
        first_response,
        attempt_request_hash=input_request_hash,
        missing_properties=(),
    )

    missing = [
        name
        for name, value in first_attempt["properties"].items()
        if not valid_value(name, value)
    ]
    if first_attempt["status"] == "parsed" and missing:
        logging.warning(
            "%s is missing valid VLM values for %s; retrying",
            segment_key,
            ", ".join(missing),
        )
        strict_prompt = user_prompt + STRICT_RETRY_USER_PROMPT_SUFFIX
        strict_prompts = {
            **prompts,
            "query_prompt": prompts["query_prompt"]
            + STRICT_RETRY_QUERY_PROMPT_SUFFIX,
        }
        retry_request_hash = compute_vlm_retry_attempt_hash(
            input_request_hash, missing
        )
        require_current_text_context()
        retry_response = analyze_material_with_vlm(
            strict_prompt,
            vlm_model,
            vlm_processor,
            strict_prompts,
            visual_image_paths,
        )
        record_attempt(
            "retry",
            retry_response,
            attempt_request_hash=retry_request_hash,
            missing_properties=missing,
        )

    selected_attempt = next(
        (
            attempt
            for attempt in attempts
            if attempt["status"] == "parsed"
            and all(
                valid_value(name, attempt["properties"].get(name))
                for name in required_props
            )
        ),
        None,
    )
    if selected_attempt is not None:
        properties = {
            name: float(selected_attempt["properties"][name])
            for name in required_props
        }
        selected_label = selected_attempt["attempt"]
        provenance = {
            name: f"vlm_{selected_label}" for name in required_props
        }
        status = "complete"
    else:
        selected_label = None
        status = "failed"

    annotation = {
        "status": status,
        "properties": properties,
        "property_provenance": provenance,
        "attempts": attempts,
        "input_request_hash": input_request_hash,
        "input_text_context_hash": input_text_context_hash,
        "model_identity": model_identity,
    }
    if selected_label is not None:
        annotation["selected_attempt"] = selected_label
    segment_info["vlm_annotation"] = annotation

    if status == "complete":
        segment_info["youngs_modulus"] = properties["youngs_modulus"]
        segment_info["poissons_ratio"] = properties["poissons_ratio"]
        if dataset_name == "simready":
            segment_info["vlm_density"] = properties["density"]
        else:
            segment_info["density"] = properties["density"]
        segment_info["vlm_analysis"] = selected_attempt["raw_vlm_output"]
        segment_info["prompt"] = selected_attempt["request_prompt"]

    logging.info("VLM annotation status for %s: %s", segment_key, status)
    return segment_info, status == "complete"


def _validate_asset_key_selection(requested_asset_keys, dataset_names):
    keys = tuple(requested_asset_keys)
    if not keys:
        raise ValueError("--asset-key requires at least one exact full asset key")

    seen = set()
    duplicates = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate --asset-key values: {sorted(duplicates)!r}")

    datasets = tuple(dict.fromkeys(dataset_names))
    unknown_datasets = sorted(
        dataset for dataset in datasets if dataset not in DEFAULT_DATASET_ROOTS
    )
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets!r}")

    selected_datasets = set(datasets)
    omitted = sorted(
        key
        for key in keys
        if key.partition("/")[0] in DEFAULT_DATASET_ROOTS
        and key.partition("/")[0] not in selected_datasets
    )
    if omitted:
        raise ValueError(
            "--dataset omits datasets required by --asset-key: " + repr(omitted)
        )

    known_keys = {
        asset.key
        for dataset in datasets
        for asset in discover_assets(
            dataset, os.path.join("datasets", "raw", dataset)
        )
    }
    unknown_keys = sorted(set(keys) - known_keys)
    if unknown_keys:
        raise ValueError(f"unknown --asset-key values: {unknown_keys!r}")
    return frozenset(keys)


def _balanced_shard_asset_keys(
    dataset_names,
    skip_existing_file_paths,
    num_shards,
    shard_index,
):
    """Select one balanced slice after removing merged completed assets."""

    remaining_assets = sorted(
        (
            asset
            for dataset in dict.fromkeys(dataset_names)
            for asset in discover_assets(dataset)
            if os.path.realpath(asset.path) not in skip_existing_file_paths
        ),
        key=lambda asset: asset.key,
    )
    remaining_keys = [asset.key for asset in remaining_assets]
    if len(remaining_keys) != len(set(remaining_keys)):
        raise ValueError("canonical remaining asset inventory contains duplicates")
    return frozenset(remaining_keys[shard_index::num_shards])


def _build_asset_annotation_state(
    result,
    runtime_visual_paths,
    *,
    dataset_name,
    vlm_model,
    vlm_processor,
    checkpoint_callback=None,
):
    """Validate immutable inputs and build isolated per-segment API jobs."""

    annotation_input_gate = extraction_gate(result)
    if annotation_input_gate.get("status") != "complete":
        raise ValueError(
            "asset extraction became unsafe before VLM submission: "
            f"{annotation_input_gate.get('annotation_input_blockers')!r}"
        )

    prompts = SIMREADY_PROMPTS if dataset_name == "simready" else RESIDENTIAL_PROMPTS

    runtime_text_context_hash = current_vlm_text_context_hash(prompts)
    model_id = vlm_model_identity(vlm_model, vlm_processor)
    accounted = 0
    segments = result.get("segments")
    if not isinstance(segments, Mapping):
        raise ValueError("extraction has no segment mapping")

    jobs = []
    for segment_key, segment_info in segments.items():
        if segment_key not in runtime_visual_paths:
            raise ValueError(
                f"strict visual inputs are missing for segment {segment_key!r}"
            )
        vlm_input = segment_info.get("vlm_input")
        if not isinstance(vlm_input, Mapping):
            raise ValueError(f"VLM input spec is missing for {segment_key!r}")
        if vlm_input.get("text_context_hash") != runtime_text_context_hash:
            raise ValueError(
                f"VLM text/reference context is stale for {segment_key!r}"
            )
        visual_paths = tuple(runtime_visual_paths[segment_key])
        expected_hashes = vlm_input.get("image_hashes")
        if not isinstance(expected_hashes, list) or len(expected_hashes) != len(
            visual_paths
        ):
            raise ValueError(
                f"VLM input hashes are missing for {segment_key!r}"
            )
        actual_hashes = tuple(_sha256_file(path) for path in visual_paths)
        if actual_hashes != tuple(expected_hashes):
            raise ValueError(
                f"generated visual bytes changed before VLM submission for "
                f"{segment_key!r}"
            )

        if is_segment_annotation_complete(
            result,
            segment_info,
            expected_model_identity=model_id,
            expected_text_context_hash=runtime_text_context_hash,
        ):
            accounted += 1
            continue
        jobs.append((segment_key, copy.deepcopy(segment_info), visual_paths))

    def run_job(job):
        segment_key, segment_info, visual_paths = job
        vlm_input = segment_info["vlm_input"]
        try:
            updated_segment, annotation_success = analyze_material_with_retry(
                segment_key,
                segment_info,
                vlm_model,
                vlm_processor,
                dataset_name,
                prompts,
                vlm_input["prompt"],
                visual_paths,
                vlm_input["request_hash"],
                vlm_input["text_context_hash"],
                model_id,
            )
            return segment_key, updated_segment, annotation_success
        except Exception as exc:
            segment_info["vlm_annotation"] = {
                "status": "failed",
                "properties": {},
                "input_request_hash": vlm_input.get("request_hash"),
                "input_text_context_hash": vlm_input.get("text_context_hash"),
                "model_identity": model_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            logging.exception("VLM analysis failed for %s", segment_key)
            return segment_key, segment_info, False

    return {
        "result": result,
        "segments": segments,
        "jobs": deque(jobs),
        "run_job": run_job,
        "futures": [],
        "executor": None,
        "successful": 0,
        "accounted": accounted,
        "checkpoint_callback": checkpoint_callback,
    }


def _merge_asset_annotation_results(state, completed_results):
    for segment_key, updated_segment, annotation_success in completed_results:
        state["segments"][segment_key] = updated_segment
        state["successful"] += int(annotation_success)
        state["accounted"] += 1


def _submit_asset_segments(
    result,
    runtime_visual_paths,
    *,
    dataset_name,
    vlm_model,
    vlm_processor,
    executor,
    checkpoint_callback,
):
    """Queue one rendered asset for bounded API submission."""

    state = _build_asset_annotation_state(
        result,
        runtime_visual_paths,
        dataset_name=dataset_name,
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
        checkpoint_callback=checkpoint_callback,
    )
    state["executor"] = executor
    return state


def _collect_submitted_asset_segments(state, *, wait_for_all):
    """Merge finished worker results on the main thread only."""

    ready = []
    remaining = []
    for future in state["futures"]:
        if wait_for_all or future.done():
            ready.append(future.result())
        else:
            remaining.append(future)
    state["futures"] = remaining
    _merge_asset_annotation_results(state, ready)

    complete = not remaining and not state["jobs"]
    if complete:
        state["result"].pop("checkpoint_state", None)
    elif ready and state["checkpoint_callback"] is not None:
        state["result"]["checkpoint_state"] = {
            "status": "in_progress",
            "accounted_segment_count": state["accounted"],
            "segment_count": len(state["segments"]),
        }
        update_annotation_status(state["result"])
        state["checkpoint_callback"]()
    outcome = state["successful"], state["accounted"]
    return complete, outcome


def _submitted_future_count(pending_assets):
    return sum(
        len(pending["annotation_state"]["futures"])
        for pending in pending_assets
    )


def _has_queued_api_jobs(pending_assets):
    return any(pending["annotation_state"]["jobs"] for pending in pending_assets)


def _schedule_queued_api_jobs(pending_assets, max_outstanding):
    available = max_outstanding - _submitted_future_count(pending_assets)
    for pending in pending_assets:
        if available <= 0:
            break
        state = pending["annotation_state"]
        while available > 0 and state["jobs"]:
            job = state["jobs"].popleft()
            state["futures"].append(
                state["executor"].submit(state["run_job"], job)
            )
            available -= 1


def _wait_for_submitted_api_progress(pending_assets):
    futures = tuple(
        future
        for pending in pending_assets
        for future in pending["annotation_state"]["futures"]
    )
    if not futures:
        raise RuntimeError("queued API work has no submitted request to await")
    wait(futures, return_when=FIRST_COMPLETED)


def _drain_submitted_assets(
    pending_assets,
    *,
    wait_for_all,
    max_outstanding,
):
    """Collect API work while keeping the executor queue strictly bounded."""

    if max_outstanding < 1:
        raise ValueError("maximum outstanding API requests must be positive")
    while True:
        remaining = []
        for pending in pending_assets:
            complete, outcome = _collect_submitted_asset_segments(
                pending["annotation_state"],
                wait_for_all=False,
            )
            if complete:
                pending["finalize"](outcome)
            else:
                remaining.append(pending)
        pending_assets[:] = remaining
        _schedule_queued_api_jobs(pending_assets, max_outstanding)
        if not wait_for_all or not pending_assets:
            return
        _wait_for_submitted_api_progress(pending_assets)


def _annotate_asset_segments(
    result,
    runtime_visual_paths,
    *,
    dataset_name,
    vlm_model,
    vlm_processor,
):
    """Keep the local Qwen backend sequential on its rendering GPU."""

    state = _build_asset_annotation_state(
        result,
        runtime_visual_paths,
        dataset_name=dataset_name,
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
    )
    for job in state.pop("jobs"):
        _merge_asset_annotation_results(state, [state["run_job"](job)])
    state["result"].pop("checkpoint_state", None)
    return state["successful"], state["accounted"]


def process_dataset(
    dataset_type,
    vlm_model,
    vlm_processor,
    api_executor=None,
    pending_api_assets=None,
    limit=None,
    output_file=None,
    dry_run=False,
    force_reprocess=False,
    existing_results=None,
    processed_objects=None,
    processed_file_paths=None,
    asset_keys=None,
    keep_visual_cache=False,
):
    """
    Process all objects in a dataset.

    Args:
        dataset_type: Type of dataset (simready, residential, commercial, vegetation)
        vlm_model: VLM model for analysis
        vlm_processor: VLM processor for analysis
        api_executor: Long-lived executor shared by every API dataset
        pending_api_assets: Main-thread merge queue shared across API datasets
        limit: Maximum number of objects to process
        output_file: Path to output file for incremental updates
        dry_run: If True, only process a few objects
        force_reprocess: If True, reprocess objects even if they exist in output_file
        existing_results: List of existing results to start with
        processed_objects: Set of already processed object names
        processed_file_paths: Set of already processed file paths
        asset_keys: Exact canonical keys to process from this dataset, or None
        keep_visual_cache: Retain successful assets' rendered visual cache

    Returns:
        Tuple of results and statistics
    """
    # Initialize tracking sets and results
    processed_objects = set() if processed_objects is None else processed_objects
    processed_file_paths = (
        set() if processed_file_paths is None else processed_file_paths
    )
    existing_results = [] if existing_results is None else existing_results
    shared_api_pipeline = pending_api_assets is not None
    pending_api_assets = [] if pending_api_assets is None else pending_api_assets
    dataset_prompts = (
        SIMREADY_PROMPTS if dataset_type == "simready" else RESIDENTIAL_PROMPTS
    )
    expected_text_context_hash = current_vlm_text_context_hash(dataset_prompts)
    expected_model_identity = (
        vlm_model_identity(vlm_model, vlm_processor)
        if vlm_model is not None
        else None
    )

    completed_results = [
        result
        for result in existing_results
        if is_annotation_complete(
            result,
            expected_model_identity=expected_model_identity,
            expected_text_context_hash=expected_text_context_hash,
        )
    ]

    # Build a set of already processed object names from current-schema results
    existing_object_names = {
        result.get("object_name")
        for result in completed_results
        if "object_name" in result
    }
    existing_file_paths = {
        result.get("file_path")
        for result in completed_results
        if "file_path" in result
    }
    completed_by_identity = {
        result.get("asset_identity_hash") or result.get("asset_hash"): result
        for result in completed_results
        if result.get("asset_identity_hash") or result.get("asset_hash")
    }
    completed_by_real_path = {
        os.path.realpath(result["file_path"]): result
        for result in completed_results
        if result.get("file_path")
    }
    prior_by_identity = {
        result.get("asset_identity_hash") or result.get("asset_hash"): result
        for result in existing_results
        if result.get("asset_identity_hash") or result.get("asset_hash")
    }
    prior_by_real_path = {
        os.path.realpath(result["file_path"]): result
        for result in existing_results
        if result.get("file_path")
    }

    logging.info(
        f"Found {len(existing_object_names)} already processed objects in existing results"
    )
    logging.info(
        f"Found {len(existing_file_paths)} already processed file paths in existing results"
    )

    # If not forcing reprocess, add names from existing_results to processed_objects
    if not force_reprocess:
        processed_objects.clear()
        processed_file_paths.clear()
        processed_objects.update(existing_object_names)
        processed_file_paths.update(existing_file_paths)

    all_results = (
        existing_results
        if shared_api_pipeline
        else copy.deepcopy(existing_results)
    )

    objects_to_process = []

    if dataset_type in DEFAULT_DATASET_ROOTS:
        try:
            assets = discover_assets(
                dataset_type, os.path.join("datasets", "raw", dataset_type)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not discover {dataset_type} assets"
            ) from exc
        if not assets:
            raise RuntimeError(
                f"Asset discovery returned no objects for {dataset_type}"
            )

        if asset_keys is None:
            objects_to_process = assets
        else:
            if limit is not None or dry_run:
                raise ValueError(
                    "exact asset-key selection cannot be combined with a limit or dry run"
                )
            requested_keys = tuple(asset_keys)
            if len(requested_keys) != len(set(requested_keys)):
                raise ValueError("exact asset-key selection contains duplicate keys")
            inventory_keys = {asset.key for asset in assets}
            unknown_keys = sorted(set(requested_keys) - inventory_keys)
            if unknown_keys:
                raise ValueError(
                    f"unknown {dataset_type} asset keys: {unknown_keys!r}"
                )
            requested_key_set = set(requested_keys)
            objects_to_process = [
                asset for asset in assets if asset.key in requested_key_set
            ]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Limit the number of objects to process if specified
    if limit and limit > 0:
        logging.info(f"Limiting to {limit} objects out of {len(objects_to_process)}")
        objects_to_process = objects_to_process[:limit]

    # Filter out already processed objects before we even start
    objects_to_process_filtered = []
    for asset in objects_to_process:
        if asset_annotation_exclusion(asset.key) is not None:
            objects_to_process_filtered.append(asset)
            continue

        prior_result = completed_by_identity.get(asset.identity_hash)
        if prior_result is None:
            prior_result = completed_by_real_path.get(os.path.realpath(asset.path))
        if not force_reprocess and prior_result is not None:
            additional_dependencies = []
            if asset.thumbnail_path is not None:
                additional_dependencies.append(
                    ("object_thumbnail", asset.thumbnail_path)
                )
            current_fingerprint = build_asset_dependency_fingerprint(
                asset.path,
                asset.dataset_root,
                extraction_result=prior_result,
                additional_dependencies=additional_dependencies,
                prior_fingerprint=prior_result.get(
                    "asset_dependency_fingerprint"
                ),
            )
            if is_annotation_complete(
                prior_result,
                expected_dependency_fingerprint=current_fingerprint,
                expected_model_identity=expected_model_identity,
                expected_text_context_hash=expected_text_context_hash,
            ):
                logging.info(f"Filtering out already processed asset: {asset.key}")
                if not keep_visual_cache:
                    _remove_asset_visual_cache(asset)
                continue
            logging.info(
                "Reprocessing %s because its complete USD/material/texture "
                "dependency fingerprint is missing, incomplete, or changed",
                asset.key,
            )

        objects_to_process_filtered.append(asset)

    # Update the objects to process list
    logging.info(
        f"Filtered {len(objects_to_process) - len(objects_to_process_filtered)} already processed objects"
    )
    logging.info(f"Processing {len(objects_to_process_filtered)} remaining objects")
    objects_to_process = objects_to_process_filtered

    # Initialize statistics
    progress = {
        "success_count": 0,
        "total_vlm_segments": 0,
    }
    failed_objects = []
    total_segments = 0
    total_rendered_segments = 0
    unique_materials = set()
    materials_per_object = {}

    # Count total segments from existing results
    for result in existing_results:
        segments = result.get("segments", {})
        total_segments += len(segments)

        # Count unique materials
        for segment_key, segment_info in segments.items():
            material_name = segment_info.get("name", "")
            if material_name:
                unique_materials.add(material_name)

    if isinstance(vlm_processor, ResponsesAPIConfig) and api_executor is None:
        raise ValueError("API annotation requires the shared request executor")

    def checkpoint_asset(context):
        result = context["result"]
        asset = context["asset"]
        _replace_asset_result(all_results, result, asset)
        if output_file and not append_result_to_file(
            result, all_results, output_file
        ):
            raise CheckpointWriteError(f"could not checkpoint {asset.key}")

    def finalize_annotated_asset(context, annotation_outcome=None):
        result = context["result"]
        asset = context["asset"]
        segments = context["segments"]
        object_name = context["object_name"]
        usd_file = context["usd_file"]

        if annotation_outcome is not None:
            vlm_successes, accounted_segments = annotation_outcome
            progress["total_vlm_segments"] += vlm_successes
            if accounted_segments != len(segments):
                raise AssertionError(
                    "not every segment was accounted for by VLM or resume"
                )

        _verify_asset_inputs_unchanged(result, asset)
        annotation_status = update_annotation_status(result)
        checkpoint_asset(context)
        if annotation_status["status"] == "complete":
            if keep_visual_cache:
                _release_asset_visual_cache_lock(asset)
            else:
                _remove_asset_visual_cache(asset)
            processed_objects.add(object_name)
            processed_file_paths.add(usd_file)
            progress["success_count"] += 1
            return

        _release_asset_visual_cache_lock(asset)
        failed_objects.append(asset.key)
        logging.error(
            "Asset annotation is incomplete for %s: VLM status %s",
            asset.key,
            annotation_status["vlm"]["status"],
        )

    # Process each object
    for obj_idx, asset in enumerate(
        tqdm(objects_to_process, desc=f"Processing {dataset_type} dataset")
    ):
        if pending_api_assets:
            _drain_submitted_assets(
                pending_api_assets,
                wait_for_all=False,
                max_outstanding=vlm_processor.concurrency,
            )
            while _has_queued_api_jobs(pending_api_assets):
                _wait_for_submitted_api_progress(pending_api_assets)
                _drain_submitted_assets(
                    pending_api_assets,
                    wait_for_all=False,
                    max_outstanding=vlm_processor.concurrency,
                )
        if force_reprocess:
            _remove_asset_visual_cache(asset, required=True)
        exclusion = asset_annotation_exclusion(asset.key)
        if exclusion is not None:
            _remove_asset_visual_cache(asset)
            logging.info(
                "Skipping scene-level asset before USD extraction or VLM input "
                "construction: %s",
                asset.key,
            )
            continue
        prior_asset_result = None
        candidate_prior_result = prior_by_identity.get(asset.identity_hash)
        if candidate_prior_result is None:
            candidate_prior_result = prior_by_real_path.get(
                os.path.realpath(asset.path)
            )
        if not force_reprocess:
            prior_asset_result = candidate_prior_result
        if dataset_type == "simready":
            prop_name = asset.object_name
            object_name = prop_name

            try:
                # Get the full prop directory path
                full_prop_dir = os.path.join(SIMREADY_PROPS_DIR, prop_name)

                if not os.path.isdir(full_prop_dir):
                    logging.error(f"Prop directory not found at {full_prop_dir}")
                    failed_objects.append(prop_name)
                    continue

                usd_file = str(asset.path)
                logging.info(
                    f"Found USD file for {prop_name}: {os.path.basename(usd_file)}"
                )

                # Extract material information
                materials_dict = simready_extract_materials(
                    usd_file, prop_name, full_prop_dir
                )

                # Ensure file_path is properly stored
                if "file_path" not in materials_dict or not materials_dict["file_path"]:
                    materials_dict["file_path"] = usd_file
                _set_asset_identity(materials_dict, asset)

                # Track statistics
                segments = materials_dict.get("segments", {})
                total_segments += len(segments)

                for segment_info in segments.values():
                    separate_simready_source_physics(segment_info)

                extraction_status = update_annotation_status(materials_dict)[
                    "extraction"
                ]
                if extraction_status["status"] != "complete":
                    logging.error(
                        "Blocking annotation for %s: %s inventory errors, %s "
                        "unresolved active textures, %s texture-role conflicts, "
                        "%s ambiguous texture roles, %s unknown texture roles, "
                        "%s heuristic texture resolutions, %s segments",
                        asset.key,
                        len(extraction_status["inventory_errors"]),
                        extraction_status["unresolved_active_texture_count"],
                        extraction_status["texture_role_conflict_count"],
                        extraction_status["ambiguous_texture_role_count"],
                        extraction_status["unknown_texture_role_count"],
                        extraction_status["heuristic_texture_resolution_count"],
                        extraction_status["segment_count"],
                    )
                    logging.error(
                        "Annotation-input blockers for %s (%s): %r",
                        asset.key,
                        extraction_status.get("annotation_input_blocker_count"),
                        extraction_status.get("annotation_input_blockers"),
                    )
                    _replace_asset_result(all_results, materials_dict, asset)
                    if output_file:
                        append_result_to_file(materials_dict, all_results, output_file)
                    failed_objects.append(asset.key)
                    continue

                # Count unique materials for this prop
                prop_materials = set()
                for segment_key, segment_info in segments.items():
                    unique_materials.add(segment_info["name"])
                    prop_materials.add(segment_info["name"])

                # Record materials per prop
                if len(segments) > 0:
                    materials_per_object[prop_name] = len(prop_materials)

                # Determine thumbnail path from SimReady structure
                thumb = find_thumbnail(usd_file, "simready")
                thumb_path = str(thumb) if thumb is not None else None
                has_thumbnail = thumb_path is not None
                if not has_thumbnail:
                    logging.warning(f"No exact thumbnail found for {prop_name}")

                if len(segments) == 0:
                    raise AssertionError("unreachable zero-segment annotation path")

                try:
                    runtime_visual_paths = _prepare_asset_visual_inputs(
                        materials_dict,
                        asset,
                        dataset_name="simready",
                        object_name=prop_name,
                        thumbnail_path=thumb_path,
                        text_context_hash=expected_text_context_hash,
                        prior_result=prior_asset_result,
                    )
                    total_rendered_segments += len(runtime_visual_paths)
                    _restore_compatible_segment_annotations(
                        materials_dict,
                        prior_asset_result,
                        dataset_name="simready",
                        expected_model_identity=expected_model_identity,
                        expected_text_context_hash=expected_text_context_hash,
                    )
                except VisualCacheBusyError as exc:
                    failed_objects.append(asset.key)
                    logging.error("Deferring %s: %s", asset.key, exc)
                    continue
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    _checkpoint_pre_vlm_failure(
                        materials_dict,
                        prior_asset_result,
                        asset,
                        all_results,
                        output_file,
                        error=error,
                        vlm_model=vlm_model,
                        vlm_processor=vlm_processor,
                        block_visual_inputs=True,
                    )
                    failed_objects.append(asset.key)
                    logging.exception(
                        "Strict visual-input construction failed for %s", asset.key
                    )
                    _release_asset_visual_cache_lock(asset)
                    continue

                if not _verify_asset_inputs_unchanged(materials_dict, asset):
                    _checkpoint_pre_vlm_failure(
                        materials_dict,
                        prior_asset_result,
                        asset,
                        all_results,
                        output_file,
                        error=materials_dict["input_integrity_error"],
                    )
                    failed_objects.append(asset.key)
                    logging.error(
                        "Blocking VLM submission because inputs changed for %s",
                        asset.key,
                    )
                    _release_asset_visual_cache_lock(asset)
                    continue

                annotation_context = {
                    "result": materials_dict,
                    "asset": asset,
                    "segments": segments,
                    "object_name": object_name,
                    "usd_file": usd_file,
                }
                if isinstance(vlm_processor, ResponsesAPIConfig):

                    def checkpoint_callback(context=annotation_context):
                        checkpoint_asset(context)

                    annotation_state = _submit_asset_segments(
                        materials_dict,
                        runtime_visual_paths,
                        dataset_name="simready",
                        vlm_model=vlm_model,
                        vlm_processor=vlm_processor,
                        executor=api_executor,
                        checkpoint_callback=checkpoint_callback,
                    )

                    def finalize_callback(
                        outcome,
                        context=annotation_context,
                    ):
                        finalize_annotated_asset(context, outcome)

                    pending_api_assets.append(
                        {
                            "annotation_state": annotation_state,
                            "finalize": finalize_callback,
                        }
                    )
                    _drain_submitted_assets(
                        pending_api_assets,
                        wait_for_all=False,
                        max_outstanding=vlm_processor.concurrency,
                    )
                else:
                    annotation_outcome = _annotate_asset_segments(
                        materials_dict,
                        runtime_visual_paths,
                        dataset_name="simready",
                        vlm_model=vlm_model,
                        vlm_processor=vlm_processor,
                    )
                    finalize_annotated_asset(
                        annotation_context, annotation_outcome
                    )
            except CheckpointWriteError:
                _release_asset_visual_cache_lock(asset)
                raise
            except Exception as e:
                _release_asset_visual_cache_lock(asset)
                logging.error(f"Error processing {prop_name}: {str(e)}")
                logging.error(traceback.format_exc())
                failed_objects.append(prop_name)
        else:
            usd_file = str(asset.path)
            object_name = asset.object_name

            try:
                # Extract material information
                result = common_extract_materials(usd_file, dataset_type)

                if result:
                    # Ensure file_path is stored in the result to improve future matching
                    if "file_path" not in result or not result["file_path"]:
                        result["file_path"] = usd_file
                    _set_asset_identity(result, asset)

                    # Track statistics
                    segments = result.get("segments", {})
                    total_segments += len(segments)

                    extraction_status = update_annotation_status(result)["extraction"]
                    if extraction_status["status"] != "complete":
                        logging.error(
                            "Blocking annotation for %s: %s inventory errors, %s "
                            "unresolved active textures, %s texture-role "
                            "conflicts, %s ambiguous texture roles, %s unknown "
                            "texture roles, %s heuristic texture resolutions, "
                            "%s segments",
                            asset.key,
                            len(extraction_status["inventory_errors"]),
                            extraction_status["unresolved_active_texture_count"],
                            extraction_status["texture_role_conflict_count"],
                            extraction_status["ambiguous_texture_role_count"],
                            extraction_status["unknown_texture_role_count"],
                            extraction_status[
                                "heuristic_texture_resolution_count"
                            ],
                            extraction_status["segment_count"],
                        )
                        logging.error(
                            "Annotation-input blockers for %s (%s): %r",
                            asset.key,
                            extraction_status.get(
                                "annotation_input_blocker_count"
                            ),
                            extraction_status.get("annotation_input_blockers"),
                        )
                        _replace_asset_result(all_results, result, asset)
                        if output_file:
                            append_result_to_file(result, all_results, output_file)
                        failed_objects.append(asset.key)
                        continue

                    # Remove object_name and note fields from segments
                    for segment_key, segment_info in segments.items():
                        if "object_name" in segment_info:
                            del segment_info["object_name"]
                        if "note" in segment_info:
                            del segment_info["note"]

                    # Count unique materials for this object
                    object_materials = set()
                    for segment_name, segment_info in segments.items():
                        material_name = segment_info.get("material_type", "unknown")
                        unique_materials.add(material_name)
                        object_materials.add(material_name)

                    # Record materials per object
                    if len(segments) > 0:
                        materials_per_object[object_name] = len(object_materials)

                    # Get thumbnail path if available
                    thumb = find_thumbnail(usd_file, dataset_type)
                    thumb_path = str(thumb) if thumb is not None else None
                    has_thumbnail = thumb_path is not None
                    if has_thumbnail:
                        logging.info(f"Found thumbnail: {thumb_path}")

                    if len(segments) == 0:
                        raise AssertionError(
                            "unreachable zero-segment annotation path"
                        )

                    try:
                        runtime_visual_paths = _prepare_asset_visual_inputs(
                            result,
                            asset,
                            dataset_name=dataset_type,
                            object_name=object_name,
                            thumbnail_path=thumb_path,
                            text_context_hash=expected_text_context_hash,
                            prior_result=prior_asset_result,
                        )
                        total_rendered_segments += len(runtime_visual_paths)
                        _restore_compatible_segment_annotations(
                            result,
                            prior_asset_result,
                            dataset_name=dataset_type,
                            expected_model_identity=expected_model_identity,
                            expected_text_context_hash=expected_text_context_hash,
                        )
                    except VisualCacheBusyError as exc:
                        failed_objects.append(asset.key)
                        logging.error("Deferring %s: %s", asset.key, exc)
                        continue
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        _checkpoint_pre_vlm_failure(
                            result,
                            prior_asset_result,
                            asset,
                            all_results,
                            output_file,
                            error=error,
                            vlm_model=vlm_model,
                            vlm_processor=vlm_processor,
                            block_visual_inputs=True,
                        )
                        failed_objects.append(asset.key)
                        logging.exception(
                            "Strict visual-input construction failed for %s",
                            asset.key,
                        )
                        _release_asset_visual_cache_lock(asset)
                        continue

                    if not _verify_asset_inputs_unchanged(result, asset):
                        _checkpoint_pre_vlm_failure(
                            result,
                            prior_asset_result,
                            asset,
                            all_results,
                            output_file,
                            error=result["input_integrity_error"],
                        )
                        failed_objects.append(asset.key)
                        logging.error(
                            "Blocking VLM submission because inputs changed for %s",
                            asset.key,
                        )
                        _release_asset_visual_cache_lock(asset)
                        continue

                    annotation_context = {
                        "result": result,
                        "asset": asset,
                        "segments": segments,
                        "object_name": object_name,
                        "usd_file": usd_file,
                    }
                    if isinstance(vlm_processor, ResponsesAPIConfig):

                        def checkpoint_callback(context=annotation_context):
                            checkpoint_asset(context)

                        annotation_state = _submit_asset_segments(
                            result,
                            runtime_visual_paths,
                            dataset_name=dataset_type,
                            vlm_model=vlm_model,
                            vlm_processor=vlm_processor,
                            executor=api_executor,
                            checkpoint_callback=checkpoint_callback,
                        )

                        def finalize_callback(
                            outcome,
                            context=annotation_context,
                        ):
                            finalize_annotated_asset(context, outcome)

                        pending_api_assets.append(
                            {
                                "annotation_state": annotation_state,
                                "finalize": finalize_callback,
                            }
                        )
                        _drain_submitted_assets(
                            pending_api_assets,
                            wait_for_all=False,
                            max_outstanding=vlm_processor.concurrency,
                        )
                    else:
                        annotation_outcome = _annotate_asset_segments(
                            result,
                            runtime_visual_paths,
                            dataset_name=dataset_type,
                            vlm_model=vlm_model,
                            vlm_processor=vlm_processor,
                        )
                        finalize_annotated_asset(
                            annotation_context, annotation_outcome
                        )

                else:
                    logging.warning(f"No material information extracted for {usd_file}")
                    failed_objects.append(object_name)
            except CheckpointWriteError:
                _release_asset_visual_cache_lock(asset)
                raise
            except Exception as e:
                _release_asset_visual_cache_lock(asset)
                logging.error(f"Error processing {usd_file}: {str(e)}")
                logging.error(traceback.format_exc())
                failed_objects.append(object_name)

    if pending_api_assets and not shared_api_pipeline:
        _drain_submitted_assets(
            pending_api_assets,
            wait_for_all=True,
            max_outstanding=vlm_processor.concurrency,
        )

    logging.info(f"  Total segments processed: {total_segments}")
    logging.info(
        "  Total VLM analyses completed: %s",
        progress["total_vlm_segments"],
    )

    return (
        all_results,
        len(objects_to_process),
        progress,
        failed_objects,
        total_segments,
        total_rendered_segments,
        list(unique_materials),
        materials_per_object,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract material information from datasets"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["simready", "residential", "commercial", "vegetation"],
        help="Name of the dataset to process",
    )
    parser.add_argument("--list", action="store_true", help="List all available props")
    parser.add_argument("--dry-run", action="store_true", help="Dry run the script")
    parser.add_argument(
        "--asset-key",
        action="append",
        help=(
            "Process one exact full canonical asset key; repeat for multiple assets"
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the remaining canonical inventory into balanced shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based asset shard to process",
    )
    parser.add_argument(
        "--skip-existing",
        help="Skip canonical asset paths already present in this merged JSON file",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging to console"
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON file to save combined material information"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of objects even if they exist in the output file",
    )
    parser.add_argument(
        "--keep-visual-cache",
        action="store_true",
        help="Retain successful assets' rendered visual cache",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLM_MODEL", "qwen"),
        help="VLM backend: 'qwen' (default) or 'api' (Responses API)",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "API key override. Prefer VLM_API_KEY/NVIDIA_API_KEY so the "
            "credential is not exposed in the process list."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("VLM_API_BASE_URL"),
        help="Base URL for --model api (required unless VLM_API_BASE_URL is set)",
    )
    parser.add_argument(
        "--api-model",
        default=os.getenv("VLM_API_MODEL"),
        help="Responses API model for --model api (required unless VLM_API_MODEL is set)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("VLM_REASONING_EFFORT", "max"),
        help="Responses API reasoning effort (default: max)",
    )
    parser.add_argument(
        "--api-image-detail",
        default=os.getenv("VLM_API_IMAGE_DETAIL", "original"),
        help="Responses API image detail: low, high, original, or auto",
    )
    parser.add_argument(
        "--api-max-output-tokens",
        type=int,
        default=int(os.getenv("VLM_API_MAX_OUTPUT_TOKENS", "16384")),
        help="Maximum Responses API output tokens per request",
    )
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=int(os.getenv("VLM_API_CONCURRENCY", "4")),
        help="Maximum number of simultaneous Responses API calls",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=int(os.getenv("VLM_API_MAX_RETRIES", "0")),
        help="One-minute retries for each Responses API request failure",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=float(os.getenv("VLM_API_TIMEOUT", "600")),
        help="Responses API request timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of objects to process per dataset. Useful for testing.",
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        parser.error("--num-shards must be greater than zero")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error(
            f"--shard-index must be in [0, {args.num_shards - 1}]"
        )
    if args.asset_key is not None and args.num_shards != 1:
        parser.error("--asset-key cannot be combined with sharding")
    if args.num_shards != 1 and (args.limit is not None or args.dry_run):
        parser.error("sharding cannot be combined with --limit or --dry-run")

    skip_existing_file_paths = set()
    if args.skip_existing:
        skip_existing_path = Path(args.skip_existing).expanduser()
        if not skip_existing_path.is_file():
            parser.error(f"--skip-existing file does not exist: {skip_existing_path}")
        try:
            skip_existing_results, _, _ = load_existing_results(
                os.fspath(skip_existing_path)
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        for result in skip_existing_results:
            file_path = result.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                parser.error("--skip-existing contains a record without file_path")
            skip_existing_file_paths.add(os.path.realpath(file_path))

    selected_asset_keys = None
    if args.asset_key is not None:
        if args.limit is not None or args.dry_run:
            parser.error("--asset-key cannot be combined with --limit or --dry-run")
        try:
            selected_asset_keys = _validate_asset_key_selection(
                args.asset_key, args.dataset
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.num_shards != 1 or skip_existing_file_paths:
        selected_asset_keys = _balanced_shard_asset_keys(
            args.dataset,
            skip_existing_file_paths,
            args.num_shards,
            args.shard_index,
        )

    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if args.model in {"api", "responses-api"} and not args.output:
        logging.error(
            "--model api requires --output so completed segments can be resumed"
        )
        sys.exit(1)

    if args.list:
        for dataset in args.dataset:
            print(f"Listing {dataset} objects")
            for asset in discover_assets(dataset):
                if (
                    selected_asset_keys is not None
                    and asset.key not in selected_asset_keys
                ):
                    continue
                print(f"  - {asset.relative_path} [{asset.identity_hash}]")
        sys.exit()

    output_writer_lock = None
    if args.output:
        args.output = os.fspath(
            Path(args.output).expanduser().resolve(strict=False)
        )
        try:
            output_writer_lock = acquire_output_writer_lock(args.output)
        except Exception as exc:
            logging.error("Could not acquire output writer lock: %s", exc)
            sys.exit(1)

    # Load existing results if output file is specified
    all_results = []
    processed_objects = set()
    processed_file_paths = set()
    if args.output:
        all_results, processed_objects, processed_file_paths = load_existing_results(
            args.output
        )
        if skip_existing_file_paths:
            all_results = [
                result
                for result in all_results
                if os.path.realpath(result.get("file_path", ""))
                not in skip_existing_file_paths
            ]
            processed_objects = {
                result.get("object_name")
                for result in all_results
                if result.get("object_name")
            }
            processed_file_paths = {
                result.get("file_path")
                for result in all_results
                if result.get("file_path")
            }

    print("Loading VLM model...")
    vlm_model, vlm_processor = load_vlm_model(
        args.model,
        args.api_key,
        api_base_url=args.api_base_url,
        api_model=args.api_model,
        api_reasoning_effort=args.reasoning_effort,
        api_image_detail=args.api_image_detail,
        api_max_output_tokens=args.api_max_output_tokens,
        api_concurrency=args.api_concurrency,
        api_max_retries=args.api_max_retries,
        api_timeout=args.api_timeout,
    )
    if vlm_model is None or vlm_processor is None:
        logging.error("VLM backend could not be loaded")
        sys.exit(1)
    api_executor = (
        ThreadPoolExecutor(
            max_workers=vlm_processor.concurrency,
            thread_name_prefix="vlm-api",
        )
        if isinstance(vlm_processor, ResponsesAPIConfig)
        else None
    )

    # Initialize statistics tracking dictionaries
    stats_dict = {
        "num_objects": {},
        "progress": {},
        "success_count": {},
        "failed_props": {},
        "total_segments": {},
        "total_rendered_segments": {},
        "total_vlm_segments": {},
        "unique_materials": set(),
        "materials_per_prop": {},
    }
    pending_api_assets = []

    # Process each dataset with the shared all_results and processed_objects
    for dataset in args.dataset:
        print(f"Processing {dataset} objects...")
        dataset_asset_keys = (
            frozenset(
                key
                for key in selected_asset_keys
                if key.partition("/")[0] == dataset
            )
            if selected_asset_keys is not None
            else None
        )

        (
            all_results,
            dataset_num_objects,
            dataset_progress,
            dataset_failed_props,
            dataset_total_segments,
            dataset_total_rendered_segments,
            dataset_unique_materials,
            dataset_materials_per_prop,
        ) = process_dataset(
            dataset,
            vlm_model,
            vlm_processor,
            api_executor=api_executor,
            pending_api_assets=(
                pending_api_assets if api_executor is not None else None
            ),
            limit=args.limit if args.limit else (2 if args.dry_run else None),
            asset_keys=dataset_asset_keys,
            output_file=args.output,
            dry_run=args.dry_run,
            force_reprocess=args.force_reprocess,
            keep_visual_cache=args.keep_visual_cache,
            existing_results=all_results,
            processed_objects=processed_objects,
            processed_file_paths=processed_file_paths,
        )

        # Update stats
        stats_dict["num_objects"][dataset] = dataset_num_objects
        stats_dict["progress"][dataset] = dataset_progress
        stats_dict["failed_props"][dataset] = dataset_failed_props
        stats_dict["total_segments"][dataset] = dataset_total_segments
        stats_dict["total_rendered_segments"][dataset] = dataset_total_rendered_segments
        stats_dict["unique_materials"].update(dataset_unique_materials)
        stats_dict["materials_per_prop"][dataset] = dataset_materials_per_prop

    if pending_api_assets:
        _drain_submitted_assets(
            pending_api_assets,
            wait_for_all=True,
            max_outstanding=vlm_processor.concurrency,
        )
    if api_executor is not None:
        api_executor.shutdown(wait=True)
    for dataset, progress in stats_dict["progress"].items():
        stats_dict["success_count"][dataset] = progress["success_count"]
        stats_dict["total_vlm_segments"][dataset] = progress[
            "total_vlm_segments"
        ]

    # Final logging
    if args.verbose:
        # Calculate total objects processed across all datasets
        total_objects = sum(stats_dict["num_objects"].values())
        print(f"Total objects processed: {total_objects}")

        # Print dataset-specific stats
        processed_datasets = []
        for dataset in args.dataset:
            if dataset in stats_dict["num_objects"]:
                dataset_upper = dataset.upper()
                processed_datasets.append(dataset_upper)
                print(f"    {dataset_upper}: {stats_dict['num_objects'][dataset]}")

        # Calculate success percentages
        total_success = sum(stats_dict["success_count"].values())
        if total_objects > 0:
            total_success_pct = total_success / total_objects * 100
        else:
            total_success_pct = 0
        print(f"Successfully processed: {total_success} ({total_success_pct:.1f}%)")

        # Print success percentage per dataset
        for dataset in processed_datasets:
            dataset_lower = dataset.lower()
            if (
                dataset_lower in stats_dict["success_count"]
                and dataset_lower in stats_dict["num_objects"]
                and stats_dict["num_objects"][dataset_lower] > 0
            ):
                success_pct = (
                    stats_dict["success_count"][dataset_lower]
                    / stats_dict["num_objects"][dataset_lower]
                    * 100
                )
                print(f"    {dataset}: {success_pct:.1f}%")
            elif dataset_lower in processed_datasets:
                print(f"    {dataset}: 0.0%")

        # Print segment stats
        print(f"Total segments: {sum(stats_dict['total_segments'].values())}")
        print(
            f"Total rendered segments: {sum(stats_dict['total_rendered_segments'].values())}"
        )
        print(f"Total VLM segments: {sum(stats_dict['total_vlm_segments'].values())}")
        print(f"Unique materials: {len(stats_dict['unique_materials'])}")

    failed_assets = sum(
        len(failures)
        for failures in stats_dict["failed_props"].values()
        if failures
    )
    if failed_assets:
        logging.error(
            "Annotation finished with %s incomplete asset(s); completed results "
            "were checkpointed and the same command can be rerun",
            failed_assets,
        )
        sys.exit(1)

    if args.output:
        try:
            write_compact_results_to_file(all_results, args.output)
        except CheckpointWriteError as exc:
            logging.error("Could not write compact annotation dataset: %s", exc)
            sys.exit(1)
