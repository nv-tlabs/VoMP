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

"""Check which saved material annotations can be reused."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping

from dataset_toolkits.material_objects.vlm_annotations.external_materials import (
    MATERIAL_FILES,
    material_manifest_path,
    material_root_from_environment,
    verify_material_file,
)

from dataset_toolkits.material_objects.vlm_annotations.vlm_input_builder import (
    VLM_IMAGE_ROLES,
    build_vlm_input_spec,
    compute_vlm_request_hash,
    compute_vlm_retry_attempt_hash,
    valid_vlm_properties,
    valid_vlm_property_value,
)
from dataset_toolkits.material_objects.vlm_annotations.vlm_output import (
    parse_vlm_properties,
    raw_vlm_output_sha256,
)
from dataset_toolkits.material_objects.vlm_annotations.annotation_input_policy import (
    asset_annotation_input_policy,
    segment_annotation_input_policy,
)
from pxr import Usd, UsdUtils


REQUIRED_VLM_PROPERTIES = (
    "youngs_modulus",
    "poissons_ratio",
    "density",
)

# These MDL modules come with the renderer
RENDERER_MDL_MODULES = frozenset(
    {
        "OmniGlass.mdl",
        "OmniPBR.mdl",
        "OmniPBR_Opacity.mdl",
        "OmniSurface.mdl",
        "OmniSurfaceLite.mdl",
        "OmniSurfacePresets.mdl",
    }
)
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_outer_path(path: str) -> str:
    """Return the filesystem part of a package-relative USD identifier."""

    return path.split("[", 1)[0] if "[" in path and path.endswith("]") else path


URI_PREFIX = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):/+(.*)$", re.DOTALL)


def canonical_uri_identifier(path: str) -> str | None:
    """Normalize URI schemes and separators."""

    match = URI_PREFIX.match(str(path))
    if not match or len(match.group(1)) == 1:
        return None
    return f"{match.group(1).casefold()}://{match.group(2)}"


def is_uri(path: str) -> bool:
    return canonical_uri_identifier(str(path)) is not None


def is_usd_dependency_identifier(path: str) -> bool:
    """Check whether a dependency refers to a USD layer, including packaged paths."""

    identifier = str(path)
    for separator in ("?", "#", ":SDF_FORMAT_ARGS:"):
        identifier = identifier.split(separator, 1)[0]
    filesystem_path = package_outer_path(identifier)
    return Path(filesystem_path).suffix.casefold() in USD_SUFFIXES


def dependency_locator(
    path: Path,
    dataset_root: Path,
    content_sha256: str | None,
) -> tuple[str, dict[str, str]]:
    """Store dataset-relative paths where possible."""

    try:
        relative = path.relative_to(dataset_root).as_posix()
    except ValueError:
        material_root = material_root_from_environment().resolve(strict=False)
        try:
            material_relative = path.relative_to(material_root).as_posix()
        except ValueError:
            material_relative = None
        if (
            material_relative in MATERIAL_FILES
            and material_manifest_path(material_root, material_relative) == path
        ):
            return (
                f"verified-vmaterials://{material_relative}",
                {
                    "scope": "verified_vmaterials",
                    "path": material_relative,
                },
            )
        # The USD resolver locates external files again on the next run.
        identity = content_sha256 or "missing"
        return (
            f"external-content://{identity}/{path.name}",
            {"scope": "external", "name": path.name},
        )
    return f"dataset://{relative}", {"scope": "dataset", "path": relative}


def portable_unresolved_identifier(value: str, dataset_root: Path) -> str:
    """Record unresolved paths without machine-specific directories."""

    value = str(value)
    canonical_uri = canonical_uri_identifier(value)
    if canonical_uri is not None:
        return canonical_uri
    path = Path(package_outer_path(value))
    if not path.is_absolute():
        normalized = Path(os.path.normpath(str(path))).as_posix()
        return f"authored://{normalized}"
    canonical = path.resolve(strict=False)
    try:
        relative = canonical.relative_to(dataset_root).as_posix()
    except ValueError:
        return f"external-authored://{canonical.name}"
    return f"dataset://{relative}"


def dependency_path_from_asset(value: Any) -> str | None:
    """Extract a resolved filesystem path from an OpenUSD dependency value."""

    resolved = str(getattr(value, "resolvedPath", "") or "")
    authored = str(getattr(value, "path", "") or "")
    text = resolved or authored or str(value or "")
    return text or None


def layer_real_path(layer: Any) -> str | None:
    return str(
        getattr(layer, "realPath", "")
        or getattr(layer, "resolvedPath", "")
        or ""
    ) or None


def openusd_dependency_paths(
    entrypoint_path: str | os.PathLike[str],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Collect loaded USD layers and their file dependencies."""

    dependencies: list[tuple[str, str]] = []
    unresolved: list[str] = []
    errors: list[str] = []
    entrypoint = str(entrypoint_path)

    try:
        stage = Usd.Stage.Open(entrypoint, load=Usd.Stage.LoadAll)
    except Exception as exc:
        return [], [], [
            f"USD stage open failed during dependency enumeration: "
            f"{type(exc).__name__}"
        ]
    if not stage:
        return [], [], ["USD stage open returned no stage during dependency enumeration"]

    try:
        if list(stage.GetCompositionErrors()):
            errors.append("USD stage reports composition errors")
    except AttributeError:
        # Older OpenUSD builds are still checked by ComputeAllDependencies.
        pass
    except Exception as exc:
        errors.append(
            "USD composition-error query failed during dependency enumeration: "
            f"{type(exc).__name__}"
        )

    try:
        used_layers = list(stage.GetUsedLayers())
    except Exception as exc:
        used_layers = []
        errors.append(
            "USD used-layer enumeration failed: "
            f"{type(exc).__name__}"
        )
    for layer in used_layers:
        real_path = layer_real_path(layer)
        if real_path:
            dependencies.append(("composed_usd_layer", real_path))
            continue
        try:
            is_empty = bool(layer.empty)
        except Exception:
            is_empty = False
        if not is_empty:
            # Anonymous layers cannot be reopened from a saved checkpoint.
            errors.append("composed USD stage contains a non-file-backed layer")

    try:
        dependency_layers, dependency_assets, unresolved_paths = (
            UsdUtils.ComputeAllDependencies(entrypoint)
        )
    except Exception as exc:
        errors.append(
            "USD recursive dependency enumeration failed: "
            f"{type(exc).__name__}"
        )
        dependency_layers, dependency_assets, unresolved_paths = (), (), ()

    for layer in dependency_layers:
        real_path = layer_real_path(layer)
        if real_path:
            dependencies.append(("usd_dependency_layer", real_path))
    for asset in dependency_assets:
        path = dependency_path_from_asset(asset)
        if path and not is_uri(path):
            dependencies.append(("usd_resolved_asset", package_outer_path(path)))
        elif path:
            unresolved.append(path)

    for item in unresolved_paths:
        path = dependency_path_from_asset(item)
        if not path:
            continue
        unresolved.append(path)
        if is_usd_dependency_identifier(path):
            errors.append("USD composition dependency is unresolved")

    return dependencies, sorted(set(unresolved)), errors


def walk_extraction_dependency_paths(result: Mapping[str, Any]):
    """Yield resolved files that the extracted material/render inputs use."""

    materials = result.get("materials")
    materials = materials if isinstance(materials, Mapping) else {}

    def walk(value: Any, key: str | None = None):
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                child_key = str(child_key)
                if child_key == "resolved_paths" and isinstance(
                    child, (list, tuple)
                ):
                    for path in child:
                        if isinstance(path, (str, os.PathLike)) and str(path):
                            yield "active_material_asset", str(path)
                    continue
                if child_key == "mdl_source_asset_path" and isinstance(
                    child, (str, os.PathLike)
                ):
                    if str(child):
                        yield "active_mdl_source", str(child)
                    continue
                if child_key == "matching_source_paths" and isinstance(
                    child, (list, tuple)
                ):
                    for path in child:
                        if isinstance(path, (str, os.PathLike)) and str(path):
                            yield "repair_dependency", str(path)
                    continue
                if child_key == "repair_dependency_paths" and isinstance(
                    child, (list, tuple)
                ):
                    for dependency in child:
                        path = (
                            dependency.get("path")
                            if isinstance(dependency, Mapping)
                            else dependency
                        )
                        if isinstance(path, (str, os.PathLike)) and str(path):
                            yield "repair_dependency", str(path)
                    continue
                yield from walk(child, child_key)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child, key)

    yield from walk(materials)

    for role, path, _ in selected_mdl_attested_dependencies(result):
        yield role, path

    color_configuration = result.get("stage_color_configuration")
    if isinstance(color_configuration, Mapping):
        configuration_path = next(
            (
                color_configuration.get(field)
                for field in (
                    "dependency_resolved_path",
                    "effective_resolved_path",
                    "authored_resolved_path",
                    "effective_asset_path",
                    "authored_asset_path",
                )
                if isinstance(color_configuration.get(field), (str, os.PathLike))
                and str(color_configuration.get(field))
            ),
            None,
        )
        if configuration_path is not None:
            yield "stage_color_configuration", str(configuration_path)

    # Include textures selected for rendering each segment.
    segments = result.get("segments")
    if isinstance(segments, Mapping):
        for segment in segments.values():
            if not isinstance(segment, Mapping):
                continue
            textures = segment.get("textures")
            if not isinstance(textures, Mapping):
                continue
            for path in textures.values():
                if isinstance(path, (str, os.PathLike)) and str(path):
                    yield "selected_texture", str(path)


def selected_mdl_attested_dependencies(result: Mapping[str, Any]):
    """Yield the selected MDL modules and their saved hashes."""

    materials = result.get("materials")
    if not isinstance(materials, Mapping):
        return
    for material in materials.values():
        if not isinstance(material, Mapping):
            continue
        records = material.get("selected_mdl_material_semantics")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            root_path = record.get("resolved_source_path")
            root_sha256 = record.get("source_sha256")
            if (
                isinstance(root_path, (str, os.PathLike))
                and str(root_path)
                and isinstance(root_sha256, str)
                and root_sha256
            ):
                yield "selected_mdl_source", str(root_path), root_sha256
            attestation = record.get("source_dependency_attestation")
            modules = (
                attestation.get("modules")
                if isinstance(attestation, Mapping)
                else None
            )
            if not isinstance(modules, list):
                continue
            for module in modules:
                if not isinstance(module, Mapping):
                    continue
                module_path = module.get("resolved_source_path")
                module_sha256 = module.get("source_sha256")
                if (
                    isinstance(module_path, (str, os.PathLike))
                    and str(module_path)
                    and isinstance(module_sha256, str)
                    and module_sha256
                ):
                    yield (
                        "selected_mdl_dependency",
                        str(module_path),
                        module_sha256,
                    )


def repair_attested_dependencies(result: Mapping[str, Any]):
    """Yield file states recorded during material repairs."""

    materials = result.get("materials")
    if not isinstance(materials, Mapping):
        return

    def walk(value: Any):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "repair_dependency_paths" and isinstance(child, list):
                    for dependency in child:
                        if not isinstance(dependency, Mapping):
                            yield None, None, None
                            continue
                        path = dependency.get("path")
                        sha256 = dependency.get("sha256")
                        expected_status = dependency.get("expected_status")
                        if expected_status is None:
                            expected_status = (
                                "present" if sha256 is not None else None
                            )
                        yield path, sha256, expected_status
                    continue
                yield from walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child)

    yield from walk(materials)


def paths_from_prior_fingerprint(
    fingerprint: Mapping[str, Any] | None,
    dataset_root: Path,
):
    """Resolve saved dependency paths for a resume check."""

    if not isinstance(fingerprint, Mapping):
        return
    dependencies = fingerprint.get("dependencies")
    if not isinstance(dependencies, list):
        return
    for record in dependencies:
        if not isinstance(record, Mapping):
            continue
        locator = record.get("locator")
        roles = record.get("roles")
        if not isinstance(locator, Mapping) or not isinstance(roles, list):
            continue
        scope = locator.get("scope")
        if scope == "verified_vmaterials":
            relative = locator.get("path")
            if (
                not isinstance(relative, str)
                or not relative
                or relative not in MATERIAL_FILES
            ):
                continue
            path = material_manifest_path(
                material_root_from_environment(), relative
            )
            # Reject unsafe paths and force re-extraction.
            if path is None or not verify_material_file(
                path, MATERIAL_FILES[relative]
            ):
                continue
            for role in roles:
                yield str(role), str(path)
            continue
        if scope != "dataset":
            # External files are rediscovered through the current USD resolver.
            continue
        relative = locator.get("path")
        if not isinstance(relative, str) or not relative:
            continue
        for role in roles:
            yield str(role), str(dataset_root / Path(relative))


def build_asset_dependency_fingerprint(
    entrypoint_path: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    *,
    extraction_result: Mapping[str, Any] | None = None,
    additional_dependencies: Any = (),
    prior_fingerprint: Mapping[str, Any] | None = None,
    unresolved_dependencies: Any = (),
) -> dict[str, Any]:
    """Hash input files using paths relative to the dataset."""

    entrypoint = Path(package_outer_path(os.fspath(entrypoint_path))).resolve(
        strict=False
    )
    root = Path(dataset_root).resolve(strict=False)
    errors: list[str] = []
    warnings: list[str] = []
    unresolved = [str(item) for item in unresolved_dependencies if str(item)]
    dependencies: list[tuple[str, str]] = [("entrypoint", str(entrypoint))]

    usd_paths, usd_unresolved, usd_errors = openusd_dependency_paths(entrypoint)
    dependencies.extend(usd_paths)
    unresolved.extend(usd_unresolved)
    errors.extend(usd_errors)

    if isinstance(extraction_result, Mapping):
        dependencies.extend(walk_extraction_dependency_paths(extraction_result))
    dependencies.extend(
        (str(role), os.fspath(path)) for role, path in additional_dependencies
    )
    dependencies.extend(paths_from_prior_fingerprint(prior_fingerprint, root))

    attested_mdl_hashes: dict[str, set[str]] = {}
    if isinstance(extraction_result, Mapping):
        for _, raw_path, expected_sha256 in (
            selected_mdl_attested_dependencies(extraction_result)
        ):
            if not is_sha256(expected_sha256):
                errors.append("selected MDL dependency has an invalid attested hash")
                continue
            if canonical_uri_identifier(str(raw_path)) is not None:
                errors.append("selected MDL dependency path is not a local file")
                continue
            path = Path(package_outer_path(raw_path))
            if not path.is_absolute():
                path = entrypoint.parent / path
            canonical = str(path.resolve(strict=False))
            attested_mdl_hashes.setdefault(canonical, set()).add(
                expected_sha256
            )

    repair_attestations: dict[str, set[tuple[str, str | None]]] = {}
    if isinstance(extraction_result, Mapping):
        for raw_path, expected_sha256, expected_status in (
            repair_attested_dependencies(extraction_result)
        ):
            valid_expectation = (
                expected_status == "present" and is_sha256(expected_sha256)
            ) or (
                expected_status == "missing" and expected_sha256 is None
            )
            if (
                not isinstance(raw_path, (str, os.PathLike))
                or not str(raw_path)
                or not valid_expectation
            ):
                errors.append("repair dependency attestation is malformed")
                continue
            if canonical_uri_identifier(str(raw_path)) is not None:
                errors.append("repair dependency path is not a local file")
                continue
            path = Path(package_outer_path(os.fspath(raw_path)))
            if not path.is_absolute():
                path = entrypoint.parent / path
            canonical = str(path.resolve(strict=False))
            repair_attestations.setdefault(canonical, set()).add(
                (str(expected_status), expected_sha256)
            )

    by_path: dict[str, set[str]] = {}
    for role, raw_path in dependencies:
        if not raw_path:
            continue
        canonical_uri = canonical_uri_identifier(str(raw_path))
        if canonical_uri is not None:
            unresolved.append(canonical_uri)
            continue
        path = Path(package_outer_path(raw_path))
        if not path.is_absolute():
            path = entrypoint.parent / path
        canonical = path.resolve(strict=False)
        by_path.setdefault(str(canonical), set()).add(role)

    records = []
    observed_hashes: dict[str, str | None] = {}
    observed_statuses: dict[str, str] = {}
    for canonical_path in sorted(by_path):
        path = Path(canonical_path)
        content_sha256 = None
        size_bytes = None
        file_status = "present"
        try:
            if not path.is_file():
                file_status = "missing"
            else:
                content_sha256 = sha256_file(path)
                size_bytes = path.stat().st_size
        except OSError:
            file_status = "unreadable"
        observed_hashes[canonical_path] = content_sha256
        observed_statuses[canonical_path] = file_status
        logical_path, locator = dependency_locator(path, root, content_sha256)
        if file_status == "unreadable":
            # Unreadable files cannot be checked against the saved hash.
            errors.append(f"unreadable dependency: {logical_path}")
        elif file_status == "missing":
            roles = by_path[canonical_path]
            is_composition_dependency = bool(
                roles
                & {
                    "entrypoint",
                    "composed_usd_layer",
                    "usd_dependency_layer",
                    "stage_color_configuration",
                    "selected_mdl_source",
                    "selected_mdl_dependency",
                }
            ) or is_usd_dependency_identifier(str(path))
            message = f"missing dependency: {logical_path}"
            if is_composition_dependency:
                errors.append(message)
            else:
                # Record missing textures so a later download invalidates the cache.
                warnings.append(message)
        records.append(
            {
                "logical_path": logical_path,
                "locator": locator,
                "roles": sorted(by_path[canonical_path]),
                "status": file_status,
                "size_bytes": size_bytes,
                "content_sha256": content_sha256,
            }
        )

    for canonical_path, expected_hashes in sorted(attested_mdl_hashes.items()):
        if len(expected_hashes) != 1:
            errors.append(
                "selected MDL dependency has conflicting attested hashes"
            )
            continue
        expected_sha256 = next(iter(expected_hashes))
        observed_sha256 = observed_hashes.get(canonical_path)
        if observed_sha256 is not None and observed_sha256 != expected_sha256:
            errors.append(
                "selected MDL dependency content differs from extraction"
            )

    for canonical_path, expectations in sorted(repair_attestations.items()):
        if len(expectations) != 1:
            errors.append("repair dependency has conflicting attestations")
            continue
        expected_status, expected_sha256 = next(iter(expectations))
        if observed_statuses.get(canonical_path) != expected_status:
            errors.append("repair dependency status differs from extraction")
        elif (
            expected_status == "present"
            and observed_hashes.get(canonical_path) != expected_sha256
        ):
            errors.append("repair dependency content differs from extraction")

    if isinstance(extraction_result, Mapping):
        configuration = extraction_result.get("stage_color_configuration")
        if (
            isinstance(configuration, Mapping)
            and configuration.get("resolution_status") == "resolved_file"
        ):
            configuration_path = configuration.get("dependency_resolved_path")
            if not isinstance(configuration_path, str) or not configuration_path:
                errors.append(
                    "resolved stage color configuration has no dependency path"
                )
            else:
                path = Path(configuration_path)
                try:
                    actual_hash = sha256_file(path) if path.is_file() else None
                    actual_size = path.stat().st_size if path.is_file() else None
                except OSError:
                    actual_hash = None
                    actual_size = None
                expected_hash = configuration.get("resolved_file_sha256")
                expected_size = configuration.get("resolved_file_size_bytes")
                if actual_hash is None:
                    errors.append(
                        "resolved stage color configuration is no longer readable"
                    )
                elif actual_hash != expected_hash:
                    errors.append(
                        "stage color configuration content differs from extraction"
                    )
                if (
                    isinstance(expected_size, int)
                    and not isinstance(expected_size, bool)
                    and actual_size != expected_size
                ):
                    errors.append(
                        "stage color configuration size differs from extraction"
                    )

    # Merge paths to identical external files, keeping all dependency roles.
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record["logical_path"],
            record["status"],
            record["size_bytes"],
            record["content_sha256"],
        )
        if key not in merged:
            merged[key] = dict(record)
        else:
            merged[key]["roles"] = sorted(
                set(merged[key]["roles"]) | set(record["roles"])
            )
    records = sorted(
        merged.values(),
        key=lambda item: (
            item["logical_path"],
            item["content_sha256"] or "",
            item["roles"],
        ),
    )

    runtime_dependencies = sorted(
        {
            item
            for item in unresolved
            if item in RENDERER_MDL_MODULES
        }
    )
    unresolved_source = [
        item for item in unresolved if item not in RENDERER_MDL_MODULES
    ]
    unresolved_composition = sorted(
        {
            portable_unresolved_identifier(item, root)
            for item in unresolved_source
            if is_usd_dependency_identifier(item)
        }
    )
    unresolved_noncomposition = sorted(
        {
            portable_unresolved_identifier(item, root)
            for item in unresolved_source
            if not is_usd_dependency_identifier(item)
        }
    )
    unresolved = sorted(
        {
            portable_unresolved_identifier(item, root)
            for item in unresolved_source
        }
    )
    if unresolved_composition:
        errors.append(
            f"{len(unresolved_composition)} USD composition dependency path(s) "
            "could not be resolved to files"
        )
    if unresolved_noncomposition:
        # Keep missing file references so they are checked again on resume.
        warnings.append(
            f"{len(unresolved_noncomposition)} non-composition dependency "
            "path(s) could not be resolved to files"
        )

    entrypoint_logical = next(
        (
            record["logical_path"]
            for record in records
            if "entrypoint" in record["roles"]
        ),
        None,
    )
    canonical_payload = {
        "entrypoint": entrypoint_logical,
        "dependencies": records,
        "renderer_runtime_dependencies": runtime_dependencies,
        "unresolved_dependencies": unresolved,
    }
    fingerprint_sha256 = canonical_json_sha256(canonical_payload)
    return {
        **canonical_payload,
        "status": "complete" if not errors else "incomplete",
        "dependency_count": len(records),
        "fingerprint_sha256": fingerprint_sha256,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def dependency_fingerprints_match(
    stored: Any,
    current: Any,
) -> bool:
    """Check whether the saved input files still match."""

    return (
        isinstance(stored, Mapping)
        and isinstance(current, Mapping)
        and stored.get("status") == "complete"
        and current.get("status") == "complete"
        and isinstance(stored.get("fingerprint_sha256"), str)
        and stored.get("fingerprint_sha256") == current.get("fingerprint_sha256")
    )


def separate_simready_source_physics(
    segment_info: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Move authored physics out of fields reserved for generated labels."""

    existing = segment_info.get("source_physics")
    source_physics = dict(existing) if isinstance(existing, Mapping) else {}
    for field in (
        "density",
        "dynamic_friction",
        "static_friction",
        "restitution",
    ):
        if field in segment_info:
            source_physics[field] = segment_info.pop(field)
    segment_info["source_physics"] = source_physics
    return source_physics


def valid_property_values(properties: Any) -> bool:
    return valid_vlm_properties(properties)


def is_sha256(value: Any) -> bool:
    """Check whether a value is a SHA-256 hex string."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def valid_thumbnail_record(record: Any, expected_hash: str) -> bool:
    return bool(
        isinstance(record, Mapping)
        and record.get("status") == "complete"
        and record.get("role") == "isaac_thumbnail"
        and is_sha256(record.get("file_sha256"))
        and record.get("file_sha256") == expected_hash
    )


def non_cpu_text(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and value.strip()
        and "cpu" not in value.casefold()
    )


def valid_render_devices(value: Any, backend: str) -> bool:
    if isinstance(value, str):
        value = [value]
    return bool(
        isinstance(value, (list, tuple))
        and value
        and all(
            non_cpu_text(device)
            and device.partition(":")[0].strip().upper() == backend
            for device in value
        )
    )


def valid_gpu_render_record(
    record: Any,
    *,
    role: str,
    expected_hash: str,
    segment_id: str | None = None,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    if (
        record.get("status") != "complete"
        or record.get("role") != role
        or not is_sha256(record.get("output_file_sha256"))
        or record.get("output_file_sha256") != expected_hash
        or record.get("engine") != "CYCLES"
        or record.get("render_backend")
        not in {"CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"}
        or not valid_render_devices(
            record.get("render_device"), record.get("render_backend")
        )
        or type(record.get("width")) is not int
        or record.get("width") != 512
        or type(record.get("height")) is not int
        or record.get("height") != 512
    ):
        return False
    return segment_id is None or record.get("segment_id") == segment_id


def valid_asset_visual_context(
    result: Mapping[str, Any],
    *,
    isaac_thumbnail_hash: str,
    textured_whole_object_hash: str,
) -> bool:
    asset_context = result.get("visual_input_asset_context")
    if not isinstance(asset_context, Mapping) or not valid_thumbnail_record(
        asset_context.get("isaac_thumbnail"), isaac_thumbnail_hash
    ):
        return False
    whole_object = asset_context.get("textured_whole_object")
    if not valid_gpu_render_record(
        whole_object,
        role="textured_whole_object",
        expected_hash=textured_whole_object_hash,
    ):
        return False
    segments = result.get("segments")
    geometry_hashes = whole_object.get("geometry_payload_hashes")
    material_plan_hashes = whole_object.get("material_plan_hashes")
    if (
        not isinstance(segments, Mapping)
        or not isinstance(geometry_hashes, Mapping)
        or not isinstance(material_plan_hashes, Mapping)
        or set(geometry_hashes) != set(segments)
        or set(material_plan_hashes) != set(segments)
    ):
        return False
    for segment_key, segment_info in segments.items():
        if not isinstance(segment_info, Mapping):
            return False
        provenance = segment_info.get("visual_input_provenance")
        if not isinstance(provenance, Mapping):
            return False
        geometry_payload = provenance.get("geometry_payload")
        selected_part = provenance.get("textured_selected_part")
        if (
            not isinstance(geometry_payload, Mapping)
            or not isinstance(selected_part, Mapping)
            or geometry_hashes.get(segment_key) != geometry_payload.get("payload_sha256")
            or material_plan_hashes.get(segment_key)
            != selected_part.get("material_plan_sha256")
        ):
            return False
    return True


def same_parsed_property_value(name: str, stored: Any, reparsed: Any) -> bool:
    if reparsed is None:
        return stored is None
    if not valid_vlm_property_value(name, reparsed):
        return False
    if not valid_vlm_property_value(name, stored):
        return False
    return float(stored) == float(reparsed)


def reparse_attested_attempt(attempt: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Check the saved response against the parser output."""

    raw_output = attempt.get("raw_vlm_output")
    stored_hash = attempt.get("raw_vlm_output_sha256")
    if (
        not isinstance(raw_output, str)
        or not is_sha256(stored_hash)
        or stored_hash != raw_vlm_output_sha256(raw_output)
    ):
        return None
    try:
        reparsed = parse_vlm_properties(raw_output)
    except Exception:
        return None
    if reparsed.get("raw_vlm_output_sha256") != stored_hash:
        return None

    properties = attempt.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != set(
        REQUIRED_VLM_PROPERTIES
    ):
        return None
    if any(
        not same_parsed_property_value(
            name, properties.get(name), reparsed.get(name)
        )
        for name in REQUIRED_VLM_PROPERTIES
    ):
        return None

    status = attempt.get("status")
    if status == "parsed":
        if attempt.get("error") not in (None, ""):
            return None
        if attempt.get("analysis") != reparsed.get("vlm_analysis"):
            return None
    elif status == "error":
        if not isinstance(attempt.get("error"), str) or not attempt["error"].strip():
            return None
        # Failed requests must not contain parsed material values.
        if raw_output or attempt.get("analysis") not in (None, ""):
            return None
        if any(properties.get(name) is not None for name in REQUIRED_VLM_PROPERTIES):
            return None
    else:
        return None
    return reparsed


def valid_attempt_and_property_provenance(
    annotation: Mapping[str, Any], base_request_hash: str
) -> bool:
    attempts = annotation.get("attempts")
    if not isinstance(attempts, list) or not attempts or len(attempts) > 2:
        return False
    initial = attempts[0]
    if not isinstance(initial, Mapping):
        return False
    if (
        initial.get("attempt") != "initial"
        or initial.get("attempt_request_hash") != base_request_hash
        or initial.get("missing_properties") != []
    ):
        return False
    initial_reparsed = reparse_attested_attempt(initial)
    if initial_reparsed is None:
        return False
    if initial.get("status") == "parsed":
        missing_after_initial = [
            name
            for name in REQUIRED_VLM_PROPERTIES
            if not valid_vlm_property_value(name, initial_reparsed.get(name))
        ]
    else:
        missing_after_initial = list(REQUIRED_VLM_PROPERTIES)

    attempts_by_name = {"initial": initial}
    if missing_after_initial:
        if len(attempts) != 2:
            return False
        retry = attempts[1]
        if not isinstance(retry, Mapping):
            return False
        if (
            retry.get("attempt") != "retry"
            or retry.get("missing_properties") != missing_after_initial
            or retry.get("attempt_request_hash")
            != compute_vlm_retry_attempt_hash(
                base_request_hash, missing_after_initial
            )
        ):
            return False
        retry_reparsed = reparse_attested_attempt(retry)
        if retry_reparsed is None:
            return False
        attempts_by_name["retry"] = retry
    elif len(attempts) != 1:
        return False

    properties = annotation.get("properties")
    property_provenance = annotation.get("property_provenance")
    if (
        not isinstance(properties, Mapping)
        or set(properties) != set(REQUIRED_VLM_PROPERTIES)
        or not isinstance(property_provenance, Mapping)
        or set(property_provenance) != set(REQUIRED_VLM_PROPERTIES)
    ):
        return False

    selected_name = annotation.get("selected_attempt")
    selected_attempt = attempts_by_name.get(selected_name)
    selected_properties = (
        selected_attempt.get("properties")
        if isinstance(selected_attempt, Mapping)
        else None
    )
    if (
        selected_name not in {"initial", "retry"}
        or not isinstance(selected_properties, Mapping)
        or selected_attempt.get("status") != "parsed"
        or not isinstance(selected_attempt.get("request_prompt"), str)
        or not selected_attempt["request_prompt"]
    ):
        return False
    for name in REQUIRED_VLM_PROPERTIES:
        expected_value = selected_properties.get(name)
        if not valid_vlm_property_value(name, expected_value):
            return False
        if property_provenance.get(name) != f"vlm_{selected_name}":
            return False
        final_value = properties.get(name)
        if not valid_vlm_property_value(name, final_value):
            return False
        if float(final_value) != float(expected_value):
            return False
    return True


def valid_completed_segment(result: Mapping[str, Any], segment_info: Any) -> bool:
    """Check saved prompts, images, and responses before reusing an annotation."""

    if not isinstance(segment_info, Mapping):
        return False
    annotation_input_policy = segment_annotation_input_policy(
        result, segment_info
    )
    if annotation_input_policy.get("vlm_input_usable") is not True:
        return False

    vlm_input = segment_info.get("vlm_input")
    if not isinstance(vlm_input, Mapping):
        return False
    image_roles = vlm_input.get("image_roles")
    image_hashes = vlm_input.get("image_hashes")
    if (
        not isinstance(image_roles, (list, tuple))
        or tuple(image_roles) != VLM_IMAGE_ROLES
    ):
        return False
    if not isinstance(image_hashes, (list, tuple)) or len(image_hashes) != len(
        VLM_IMAGE_ROLES
    ):
        return False
    if not all(is_sha256(image_hash) for image_hash in image_hashes):
        return False
    request_hash = vlm_input.get("request_hash")
    if not is_sha256(request_hash):
        return False
    asset_key = vlm_input.get("asset_key")
    segment_id = vlm_input.get("segment_id")
    prompt = vlm_input.get("prompt")
    text_context_hash = vlm_input.get("text_context_hash")
    if not all(
        isinstance(value, str) and bool(value)
        for value in (asset_key, segment_id, prompt)
    ) or not is_sha256(text_context_hash):
        return False
    if result.get("asset_key") != asset_key:
        return False
    if segment_info.get("segment_id") != segment_id:
        return False
    if request_hash != compute_vlm_request_hash(
        asset_key,
        segment_id,
        prompt,
        text_context_hash,
        list(image_roles),
        list(image_hashes),
    ):
        return False
    image_hash_by_role = dict(zip(VLM_IMAGE_ROLES, image_hashes))
    try:
        current_input = build_vlm_input_spec(
            asset_key=asset_key,
            segment_info=segment_info,
            dataset=str(result.get("dataset_type") or ""),
            object_name=str(result.get("object_name") or ""),
            isaac_thumbnail_hash=image_hash_by_role["isaac_thumbnail"],
            textured_whole_object_hash=image_hash_by_role[
                "textured_whole_object"
            ],
            textured_selected_part_hash=image_hash_by_role[
                "textured_selected_part"
            ],
            selected_material_sphere_hash=image_hash_by_role[
                "selected_material_sphere"
            ],
            text_context_hash=text_context_hash,
            object_category=result.get("category"),
        )
    except (KeyError, TypeError, ValueError):
        return False
    if dict(vlm_input) != current_input.to_dict():
        return False
    if not valid_asset_visual_context(
        result,
        isaac_thumbnail_hash=image_hash_by_role["isaac_thumbnail"],
        textured_whole_object_hash=image_hash_by_role["textured_whole_object"],
    ):
        return False

    visual_provenance = segment_info.get("visual_input_provenance")
    if not isinstance(visual_provenance, Mapping):
        return False
    provenance_roles = visual_provenance.get("image_roles")
    provenance_hashes = visual_provenance.get("image_hashes")
    if not isinstance(provenance_roles, (list, tuple)) or tuple(
        provenance_roles
    ) != tuple(image_roles):
        return False
    if not isinstance(provenance_hashes, (list, tuple)) or tuple(
        provenance_hashes
    ) != tuple(image_hashes):
        return False

    geometry_payload = visual_provenance.get("geometry_payload")
    if not isinstance(geometry_payload, Mapping):
        return False
    payload_hash = geometry_payload.get("payload_sha256")
    payload_provenance = geometry_payload.get("provenance")
    if (
        not is_sha256(payload_hash)
        or not isinstance(payload_provenance, Mapping)
        or payload_provenance.get("segment_id") != segment_id
    ):
        return False
    if not valid_gpu_render_record(
        visual_provenance.get("textured_selected_part"),
        role="textured_selected_part",
        expected_hash=image_hash_by_role["textured_selected_part"],
        segment_id=segment_id,
    ):
        return False
    if not valid_gpu_render_record(
        visual_provenance.get("selected_material_sphere"),
        role="selected_material_sphere",
        expected_hash=image_hash_by_role["selected_material_sphere"],
        segment_id=segment_id,
    ):
        return False
    selected_part = visual_provenance["textured_selected_part"]
    material_sphere = visual_provenance["selected_material_sphere"]
    material_record_id = segment_info.get("material_record_id")
    material_plan_hash = selected_part.get("material_plan_sha256")
    if (
        selected_part.get("geometry_payload_sha256") != payload_hash
        or selected_part.get("material_record_id") != material_record_id
        or material_sphere.get("material_record_id") != material_record_id
        or not is_sha256(material_plan_hash)
        or material_sphere.get("material_plan_sha256") != material_plan_hash
    ):
        return False

    annotation = segment_info.get("vlm_annotation")
    if not isinstance(annotation, Mapping):
        return False
    if annotation.get("status") != "complete" or not valid_property_values(
        annotation.get("properties")
    ):
        return False
    if annotation.get("input_request_hash") != request_hash:
        return False
    if annotation.get("input_text_context_hash") != text_context_hash:
        return False
    if not valid_attempt_and_property_provenance(annotation, request_hash):
        return False
    model_identity = annotation.get("model_identity")
    return isinstance(model_identity, str) and bool(model_identity.strip())


def is_segment_annotation_complete(
    result: Mapping[str, Any],
    segment_info: Any,
    *,
    expected_model_identity: str | None = None,
    expected_text_context_hash: str | None = None,
) -> bool:
    """Check whether a saved segment annotation can be reused."""

    if not valid_completed_segment(result, segment_info):
        return False
    annotation = segment_info.get("vlm_annotation")
    vlm_input = segment_info.get("vlm_input")
    if expected_model_identity is not None and (
        annotation.get("model_identity") != expected_model_identity
    ):
        return False
    if expected_text_context_hash is not None and (
        vlm_input.get("text_context_hash") != expected_text_context_hash
    ):
        return False
    return True


def extraction_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize extraction failures for an asset."""

    segments = result.get("segments")
    segment_items = list(segments.items()) if isinstance(segments, Mapping) else []
    inventory = result.get("segment_inventory")
    inventory = inventory if isinstance(inventory, Mapping) else {}
    inventory_errors = [str(error) for error in inventory.get("errors", [])]
    annotation_input_policy = asset_annotation_input_policy(result)

    unresolved = []
    unresolved_count = 0
    role_conflicts = []
    role_conflict_count = 0
    ambiguities = []
    ambiguity_count = 0
    unknown_roles = []
    unknown_role_count = 0
    texture_fallbacks = []
    texture_fallback_count = 0

    def combined_entries(segment_info: Mapping[str, Any], *fields: str) -> list[Any]:
        combined: list[Any] = []
        for field in fields:
            value = segment_info.get(field, [])
            values = (
                list(value)
                if isinstance(value, (list, tuple))
                else [value] if value else []
            )
            for entry in values:
                if entry not in combined:
                    combined.append(entry)
        return combined

    for segment_key, segment_info in segment_items:
        if not isinstance(segment_info, Mapping):
            inventory_errors.append(
                f"segment {segment_key!r} is not a material-info mapping"
            )
            continue
        entries = combined_entries(
            segment_info, "unresolved_texture_evidence", "unresolved_textures"
        )
        if entries:
            serialized = (
                list(entries) if isinstance(entries, (list, tuple)) else [entries]
            )
            unresolved.append(
                {"segment_key": str(segment_key), "entries": serialized}
            )
            unresolved_count += len(serialized)
        conflicts = combined_entries(
            segment_info,
            "texture_role_conflict_evidence",
            "texture_role_conflicts",
        )
        if conflicts:
            serialized = (
                list(conflicts)
                if isinstance(conflicts, (list, tuple))
                else [conflicts]
            )
            role_conflicts.append(
                {"segment_key": str(segment_key), "entries": serialized}
            )
            role_conflict_count += len(serialized)
        ambiguous_entries = combined_entries(
            segment_info, "texture_ambiguity_evidence", "texture_ambiguities"
        )
        if ambiguous_entries:
            serialized = (
                list(ambiguous_entries)
                if isinstance(ambiguous_entries, (list, tuple))
                else [ambiguous_entries]
            )
            ambiguities.append(
                {"segment_key": str(segment_key), "entries": serialized}
            )
            ambiguity_count += len(serialized)
        unknown_entries = combined_entries(
            segment_info,
            "unknown_texture_role_evidence",
            "unknown_texture_roles",
        )
        if unknown_entries:
            serialized = (
                list(unknown_entries)
                if isinstance(unknown_entries, (list, tuple))
                else [unknown_entries]
            )
            unknown_roles.append(
                {"segment_key": str(segment_key), "entries": serialized}
            )
            unknown_role_count += len(serialized)
        fallback_entries = combined_entries(
            segment_info,
            "texture_fallbacks",
        )
        if fallback_entries:
            serialized = (
                list(fallback_entries)
                if isinstance(fallback_entries, (list, tuple))
                else [fallback_entries]
            )
            texture_fallbacks.append(
                {"segment_key": str(segment_key), "entries": serialized}
            )
            texture_fallback_count += len(serialized)

    if not segment_items:
        inventory_errors.append("no material regions were extracted")

    status = (
        "complete"
        if not inventory_errors
        and annotation_input_policy.get("vlm_input_usable") is True
        else "blocked"
    )
    return {
        "status": status,
        "segment_count": len(segment_items),
        "inventory_errors": inventory_errors,
        "unresolved_active_texture_count": unresolved_count,
        "unresolved_active_textures": unresolved,
        "texture_role_conflict_count": role_conflict_count,
        "texture_role_conflicts": role_conflicts,
        "ambiguous_texture_role_count": ambiguity_count,
        "ambiguous_texture_roles": ambiguities,
        "unknown_texture_role_count": unknown_role_count,
        "unknown_texture_roles": unknown_roles,
        "texture_fallback_count": texture_fallback_count,
        "texture_fallbacks": texture_fallbacks,
        "annotation_input_blocker_count": len(
            annotation_input_policy.get("vlm_input_blockers", [])
        ),
        "annotation_input_blockers": annotation_input_policy.get(
            "vlm_input_blockers", []
        ),
    }


def build_annotation_status(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build asset-level extraction and VLM status from segment records."""

    extraction = extraction_gate(result)
    segments = result.get("segments")
    segment_items = list(segments.items()) if isinstance(segments, Mapping) else []

    complete = []
    fallback = []
    failed = []
    not_run = []
    invalid_segments = []
    for segment_key, segment_info in segment_items:
        annotation = (
            segment_info.get("vlm_annotation", {})
            if isinstance(segment_info, Mapping)
            else {}
        )
        status = annotation.get("status") if isinstance(annotation, Mapping) else None
        properties = (
            annotation.get("properties")
            if isinstance(annotation, Mapping)
            else None
        )
        if status == "complete":
            if valid_completed_segment(result, segment_info):
                complete.append(str(segment_key))
            else:
                invalid_segments.append(str(segment_key))
        elif status == "complete_with_fallback" and valid_property_values(properties):
            fallback.append(str(segment_key))
        elif status in {"failed", "partial"}:
            failed.append(str(segment_key))
        else:
            not_run.append(str(segment_key))

    if extraction["status"] != "complete":
        vlm_status = "blocked"
    elif segment_items and len(complete) == len(segment_items):
        vlm_status = "complete"
    elif not_run and len(not_run) == len(segment_items):
        vlm_status = "not_run"
    elif failed or fallback or complete or invalid_segments:
        vlm_status = "partial"
    else:
        vlm_status = "not_run"

    overall_status = (
        "complete"
        if extraction["status"] == "complete" and vlm_status == "complete"
        else "incomplete"
    )
    return {
        "status": overall_status,
        "extraction": extraction,
        "vlm": {
            "status": vlm_status,
            "segment_count": len(segment_items),
            "complete_segment_count": len(complete),
            "complete_segments": complete,
            "fallback_segment_count": len(fallback),
            "fallback_segments": fallback,
            "failed_segment_count": len(failed),
            "failed_segments": failed,
            "invalid_contract_segment_count": len(invalid_segments),
            "invalid_contract_segments": invalid_segments,
            "not_run_segment_count": len(not_run),
            "not_run_segments": not_run,
        },
    }


def update_annotation_status(result: MutableMapping[str, Any]) -> dict[str, Any]:
    """Recompute and store an asset's annotation status."""

    status = build_annotation_status(result)
    result["annotation_status"] = status
    return status


def is_annotation_complete(
    result: Mapping[str, Any],
    expected_dependency_fingerprint: Mapping[str, Any] | None = None,
    expected_model_identity: str | None = None,
    expected_text_context_hash: str | None = None,
) -> bool:
    """Check whether an asset annotation can be reused."""

    checkpoint_state = result.get("checkpoint_state")
    if isinstance(checkpoint_state, Mapping) and checkpoint_state.get("status") in {
        "in_progress",
        "paused_rate_limit",
    }:
        return False
    stored_fingerprint = result.get("asset_dependency_fingerprint")
    if not isinstance(stored_fingerprint, Mapping):
        return False
    if stored_fingerprint.get("status") != "complete":
        return False
    if not isinstance(stored_fingerprint.get("fingerprint_sha256"), str):
        return False
    if expected_dependency_fingerprint is not None and not dependency_fingerprints_match(
        stored_fingerprint, expected_dependency_fingerprint
    ):
        return False
    stored_status = result.get("annotation_status")
    if not isinstance(stored_status, Mapping) or stored_status.get("status") != "complete":
        return False
    if build_annotation_status(result)["status"] != "complete":
        return False
    if expected_model_identity is not None or expected_text_context_hash is not None:
        segments = result.get("segments")
        if not isinstance(segments, Mapping):
            return False
        for segment_info in segments.values():
            if not isinstance(segment_info, Mapping):
                return False
            annotation = segment_info.get("vlm_annotation")
            if not isinstance(annotation, Mapping):
                return False
            if (
                expected_model_identity is not None
                and annotation.get("model_identity") != expected_model_identity
            ):
                return False
            vlm_input = segment_info.get("vlm_input")
            if not isinstance(vlm_input, Mapping):
                return False
            if (
                expected_text_context_hash is not None
                and vlm_input.get("text_context_hash")
                != expected_text_context_hash
            ):
                return False
    return True
