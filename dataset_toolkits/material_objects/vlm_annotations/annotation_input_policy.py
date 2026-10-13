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

"""Check material and texture information before sending it to the VLM."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


BASE_COLOR_INPUTS = frozenset(
    {
        "albedo",
        "albedocolor",
        "basecolor",
        "basecolorconstant",
        "diffuse",
        "diffusecolor",
        "diffusecolorconstant",
        "diffusereflectioncolor",
        "glasscolor",
    }
)
# Skip whole scenes when annotating individual objects.
EXCLUDED_SCENES = {
    (
        "residential/Furniture/FurnitureSets/Appleseed/"
        "Appleseed_LivingRoom_Var01.usd"
    ): (
        "This entrypoint is a composed living-room scene containing multiple "
        "independent objects (furniture, electronics, decor, plants, and a "
        "floor plane), not one object-level annotation asset."
    ),
}


def asset_annotation_exclusion(asset_key: Any) -> dict[str, Any] | None:
    """Return the reason an asset is excluded from VLM annotation."""

    if not isinstance(asset_key, str):
        return None
    normalized = asset_key.replace("\\", "/")
    reason = EXCLUDED_SCENES.get(normalized)
    if reason is None:
        return None
    return input_error(
        "scene_level_asset_excluded",
        reason,
        asset_key=normalized,
        asset_scope="scene_level_multi_object_assembly",
        disposition="skip_all_vlm_annotation_requests",
    )


def extraction_asset_exclusion(
    extraction: Mapping[str, Any],
) -> dict[str, Any] | None:
    direct = asset_annotation_exclusion(extraction.get("asset_key"))
    if direct is not None:
        return direct
    file_path = extraction.get("file_path")
    if not isinstance(file_path, str):
        return None
    normalized_path = file_path.replace("\\", "/")
    for asset_key in EXCLUDED_SCENES:
        if normalized_path == asset_key or normalized_path.endswith(
            "/" + asset_key
        ):
            return asset_annotation_exclusion(asset_key)
    return None


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def input_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def as_record_list(value: Any) -> list[Any] | None:
    if not isinstance(value, list):
        return None
    return value


def is_nonnegative_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def canonical_json_sha256(label: str, value: Any) -> str | None:
    """Hash JSON using the same encoding as the USD reader."""

    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    digest = hashlib.sha256()
    for item in (label, serialized):
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def record_sha256_without(
    label: str,
    record: Mapping[str, Any],
    excluded_fields: frozenset[str],
) -> str | None:
    return canonical_json_sha256(
        label,
        {
            key: value
            for key, value in record.items()
            if key not in excluded_fields
        },
    )


def compute_surface_base_color_value_resolution_sha256(
    resolution: Any,
) -> str | None:
    """Hash the saved base-color record."""

    if not isinstance(resolution, Mapping):
        return None
    return record_sha256_without(
        "surface-base-color-value-resolution-v1",
        resolution,
        frozenset({"resolution_sha256"}),
    )


def normalize_shader_input_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(
        character for character in value.casefold() if character.isalnum()
    )


def surface_output_render_context(property_path: Any) -> str | None:
    if not isinstance(property_path, str) or ".outputs:" not in property_path:
        return None
    full_name = property_path.rsplit(".outputs:", 1)[1]
    tokens = full_name.split(":")
    if not tokens or tokens[-1].casefold() != "surface":
        return None
    return ":".join(tokens[:-1]) or "universal"


def finite_numeric_list(value: Any, lengths: set[int]) -> bool:
    return (
        isinstance(value, list)
        and len(value) in lengths
        and all(
            not isinstance(component, bool)
            and isinstance(component, (int, float))
            and math.isfinite(float(component))
            for component in value
        )
    )


def canonical_json(value: Any) -> str | None:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def connection_chain_errors(
    chain: Any,
    recorded_hash: Any,
    *,
    consumer_property_path: Any,
    effective_source_property_path: Any,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(chain, list) or not all(
        isinstance(edge, Mapping) for edge in chain
    ):
        return ["connection_endpoint_chain must be a list of mappings"]
    computed_hash = canonical_json_sha256(
        "surface-base-color-connection-chain-v1", chain
    )
    if not is_sha256(recorded_hash) or recorded_hash != computed_hash:
        errors.append("connection_chain_sha256 does not match the chain")

    previous_source: str | None = None
    for index, edge in enumerate(chain):
        edge_consumer = edge.get("consumer_property_path")
        edge_source = edge.get("source_property_path")
        if not isinstance(edge_consumer, str) or not edge_consumer:
            errors.append(f"chain edge {index} has no consumer property path")
        if not isinstance(edge_source, str) or not edge_source:
            errors.append(f"chain edge {index} has no source property path")
        if index == 0 and edge_consumer != consumer_property_path:
            errors.append("the first chain edge does not begin at the consumer")
        if index and edge_consumer != previous_source:
            errors.append("connection-chain endpoints are not contiguous")
        if not is_nonnegative_int(edge.get("source_connection_index")):
            errors.append(
                f"chain edge {index} has an invalid source connection index"
            )
        previous_source = edge_source if isinstance(edge_source, str) else None
    if chain:
        if previous_source != effective_source_property_path:
            errors.append("the chain does not end at the effective source")
    elif consumer_property_path != effective_source_property_path:
        errors.append(
            "an empty chain is valid only when the consumer is the source"
        )
    return errors


def consumer_resolution_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    consumer_shader_path = record.get("consumer_shader_path")
    consumer_property_path = record.get("consumer_property_path")
    render_context = record.get("render_context")
    for field, value in (
        ("consumer_shader_path", consumer_shader_path),
        ("consumer_property_path", consumer_property_path),
        ("render_context", render_context),
        ("recognized_input_name", record.get("recognized_input_name")),
        (
            "recognized_input_full_name",
            record.get("recognized_input_full_name"),
        ),
    ):
        if not isinstance(value, str) or not value:
            errors.append(f"{field} must be a nonempty string")

    normalized = normalize_shader_input_name(
        record.get("recognized_input_name")
    )
    if (
        normalized not in BASE_COLOR_INPUTS
        or record.get("recognized_input_normalized_name") != normalized
    ):
        errors.append("the consumer input is not in the exact recognized allowlist")
    surface_outputs = record.get("surface_terminal_output_paths")
    if (
        not isinstance(surface_outputs, list)
        or not surface_outputs
        or not all(
            isinstance(path, str)
            and path.startswith(f"{consumer_shader_path}.outputs:")
            for path in surface_outputs
        )
        or len(surface_outputs) != len(set(surface_outputs))
    ):
        errors.append(
            "surface_terminal_output_paths must be unique Outputs on the consumer Shader"
        )

    oracle_paths = record.get("oracle_producer_paths")
    provenance_paths = record.get("provenance_producer_paths")
    if (
        not isinstance(oracle_paths, list)
        or not all(isinstance(path, str) and path for path in oracle_paths)
    ):
        errors.append("oracle_producer_paths is not a list of property paths")
        oracle_paths = []
    if provenance_paths != oracle_paths or record.get(
        "ordered_provenance_agrees"
    ) is not True:
        errors.append("the ordered provenance paths do not match the oracle")

    for field in (
        "resolved_producers",
        "excluded_producers",
        "resolution_trace",
        "blockers",
    ):
        value = record.get(field)
        if not isinstance(value, list) or not all(
            isinstance(item, Mapping) for item in value
        ):
            errors.append(f"{field} must be a list of mappings")
    if record.get("blockers") != []:
        errors.append("a usable consumer must have no resolution blockers")

    resolved = record.get("resolved_producers")
    if isinstance(resolved, list):
        for producer_index, producer in enumerate(resolved):
            if not isinstance(producer, Mapping):
                continue
            ordinal = producer.get("oracle_producer_ordinal")
            property_path = producer.get("property_path")
            if (
                not is_nonnegative_int(ordinal)
                or ordinal >= len(oracle_paths)
                or oracle_paths[ordinal] != property_path
            ):
                errors.append(
                    f"resolved producer {producer_index} disagrees with the oracle"
                )
            errors.extend(
                f"resolved producer {producer_index}: {message}"
                for message in connection_chain_errors(
                    producer.get("connection_endpoint_chain"),
                    producer.get("connection_chain_sha256"),
                    consumer_property_path=consumer_property_path,
                    effective_source_property_path=property_path,
                )
            )
            if producer.get("classification") not in {
                "authored_constant_input",
                "excluded_nonconstant_producer",
            }:
                errors.append(
                    f"resolved producer {producer_index} has an invalid classification"
                )

    recorded_hash = record.get("consumer_resolution_sha256")
    computed_hash = record_sha256_without(
        "surface-base-color-consumer-resolution-v1",
        record,
        frozenset({"consumer_resolution_sha256"}),
    )
    if not is_sha256(recorded_hash) or recorded_hash != computed_hash:
        errors.append("consumer_resolution_sha256 does not match the record")
    return errors


def constant_source_errors(
    source: Mapping[str, Any],
    consumers_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    source_path = source.get("source_property_path")
    prim_path = source.get("source_prim_path")
    full_name = source.get("source_property_full_name")
    input_name = source.get("source_input_name")
    for field, value in (
        ("source_property_path", source_path),
        ("source_prim_path", prim_path),
        ("source_property_full_name", full_name),
        ("source_input_name", input_name),
    ):
        if not isinstance(value, str) or not value:
            errors.append(f"{field} must be a nonempty string")
    if source.get("source_property_kind") != "input":
        errors.append("source_property_kind must be 'input'")
    if isinstance(full_name, str) and isinstance(input_name, str):
        if full_name != f"inputs:{input_name}":
            errors.append("source input name and full name disagree")
    if (
        isinstance(source_path, str)
        and isinstance(prim_path, str)
        and isinstance(full_name, str)
        and source_path != f"{prim_path}.{full_name}"
    ):
        errors.append("source property path does not match its prim/full name")

    role = source.get("source_role")
    node_kind = source.get("source_node_kind")
    is_container = source.get("source_is_container")
    if role == "recognized_surface_shader_seed_input":
        if node_kind != "Shader" or is_container is not False:
            errors.append(
                "a recognized seed must be a non-container Shader Input"
            )
    elif role == "connected_container_interface_input":
        if node_kind not in {"Material", "NodeGraph"} or is_container is not True:
            errors.append(
                "a connected interface source must be a Material/NodeGraph container Input"
            )
    else:
        errors.append("source_role is not an admitted constant-source role")

    authored_value = source.get("authored_value")
    rgba = source.get("rgba")
    component_count = source.get("authored_component_count")
    if not finite_numeric_list(authored_value, {3, 4}):
        errors.append("authored_value must contain three or four finite numbers")
    if not finite_numeric_list(rgba, {4}):
        errors.append("rgba must contain four finite numbers")
    if (
        not is_nonnegative_int(component_count)
        or component_count not in {3, 4}
        or not isinstance(authored_value, list)
        or component_count != len(authored_value)
    ):
        errors.append("authored_component_count does not match authored_value")
    if finite_numeric_list(authored_value, {3, 4}) and finite_numeric_list(
        rgba, {4}
    ):
        expected_rgba = [float(value) for value in authored_value]
        if len(expected_rgba) == 3:
            expected_rgba.append(1.0)
        if [float(value) for value in rgba] != expected_rgba:
            errors.append("rgba is not the exact RGB(A) expansion of authored_value")
    if source.get("has_authored_value") is not True:
        errors.append("the source is not attested as an authored value")
    if source.get("producer_resolution") != "authored_value_fallback":
        errors.append("an admitted Input must be an authored-value producer")
    if source.get("invalid_connection_paths") != []:
        errors.append("an admitted source retains unresolved invalid connections")
    for field in (
        "connections",
        "ignored_invalid_connection_fallbacks",
    ):
        if not isinstance(source.get(field), list):
            errors.append(f"{field} must be a list")

    recorded_core_hash = source.get("source_provenance_sha256")
    computed_core_hash = record_sha256_without(
        "surface-base-color-constant-source-v1",
        source,
        frozenset(
            {
                "source_provenance_sha256",
                "consumer_uses",
                "source_with_consumer_uses_sha256",
            }
        ),
    )
    if (
        not is_sha256(recorded_core_hash)
        or recorded_core_hash != computed_core_hash
    ):
        errors.append("source_provenance_sha256 does not match the source core")

    uses = source.get("consumer_uses")
    if not isinstance(uses, list) or not uses or not all(
        isinstance(use, Mapping) for use in uses
    ):
        errors.append("consumer_uses must be a nonempty list of mappings")
        uses = []
    canonical_uses = [canonical_json(use) for use in uses]
    if None in canonical_uses or len(set(canonical_uses)) != len(canonical_uses):
        errors.append("consumer_uses must be canonical JSON and duplicate-free")

    for use_index, use in enumerate(uses):
        consumer_shader_path = use.get("consumer_shader_path")
        consumer_property_path = use.get("consumer_property_path")
        effective_path = use.get("effective_source_property_path")
        render_context = use.get("render_context")
        for field, value in (
            ("consumer_shader_path", consumer_shader_path),
            ("consumer_property_path", consumer_property_path),
            ("effective_source_property_path", effective_path),
            ("oracle_producer_path", use.get("oracle_producer_path")),
            ("render_context", render_context),
        ):
            if not isinstance(value, str) or not value:
                errors.append(f"consumer use {use_index}: {field} is invalid")
        if effective_path != source_path or use.get("oracle_producer_path") != source_path:
            errors.append(
                f"consumer use {use_index}: effective/oracle source does not match the source"
            )
        normalized = normalize_shader_input_name(
            use.get("recognized_input_name")
        )
        if normalized not in BASE_COLOR_INPUTS:
            errors.append(
                f"consumer use {use_index}: recognized input is outside the allowlist"
            )
        ordinal = use.get("oracle_producer_ordinal")
        if not is_nonnegative_int(ordinal):
            errors.append(f"consumer use {use_index}: oracle ordinal is invalid")
        errors.extend(
            f"consumer use {use_index}: {message}"
            for message in connection_chain_errors(
                use.get("connection_endpoint_chain"),
                use.get("connection_chain_sha256"),
                consumer_property_path=consumer_property_path,
                effective_source_property_path=effective_path,
            )
        )

        consumer = consumers_by_key.get(
            (
                str(consumer_shader_path),
                str(consumer_property_path),
                str(render_context),
            )
        )
        if consumer is None:
            errors.append(
                f"consumer use {use_index}: no matching consumer resolution exists"
            )
            continue
        oracle_paths = consumer.get("oracle_producer_paths")
        if (
            not isinstance(oracle_paths, list)
            or not is_nonnegative_int(ordinal)
            or ordinal >= len(oracle_paths)
            or oracle_paths[ordinal] != source_path
        ):
            errors.append(
                f"consumer use {use_index}: ordinal/path disagrees with its consumer oracle"
            )
        resolved = consumer.get("resolved_producers")
        matching_producers = [
            producer
            for producer in (resolved if isinstance(resolved, list) else [])
            if isinstance(producer, Mapping)
            and producer.get("oracle_producer_ordinal") == ordinal
            and producer.get("property_path") == source_path
            and producer.get("classification") == "authored_constant_input"
        ]
        if len(matching_producers) != 1:
            errors.append(
                f"consumer use {use_index}: no unique matching constant producer exists"
            )
        else:
            producer = matching_producers[0]
            if (
                producer.get("source_provenance_sha256")
                != recorded_core_hash
                or producer.get("connection_endpoint_chain")
                != use.get("connection_endpoint_chain")
                or producer.get("connection_chain_sha256")
                != use.get("connection_chain_sha256")
            ):
                errors.append(
                    f"consumer use {use_index}: producer provenance differs from the source use"
                )

    if uses:
        expected_order = sorted(
            uses,
            key=lambda item: (
                str(item.get("consumer_shader_path")),
                str(item.get("consumer_property_path")),
                str(item.get("render_context")),
                (
                    int(item.get("oracle_producer_ordinal"))
                    if is_nonnegative_int(item.get("oracle_producer_ordinal"))
                    else -1
                ),
                str(item.get("connection_chain_sha256")),
            ),
        )
        if uses != expected_order:
            errors.append("consumer_uses is not in deterministic reader order")

    recorded_with_uses_hash = source.get(
        "source_with_consumer_uses_sha256"
    )
    computed_with_uses_hash = record_sha256_without(
        "surface-base-color-constant-source-with-uses-v1",
        source,
        frozenset({"source_with_consumer_uses_sha256"}),
    )
    if (
        not is_sha256(recorded_with_uses_hash)
        or recorded_with_uses_hash != computed_with_uses_hash
    ):
        errors.append(
            "source_with_consumer_uses_sha256 does not match the source record"
        )
    return errors


def surface_base_color_resolution_blockers(
    material_record_id: Any,
    material: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Check the base-color values resolved by the USD reader."""

    blockers: list[dict[str, Any]] = []
    resolution = graph.get("surface_base_color_value_resolution")
    evidence = material.get("surface_base_color_constant_evidence")
    attestation = material.get(
        "surface_base_color_value_resolution_attestation"
    )

    if not isinstance(resolution, Mapping):
        blockers.append(
            input_error(
                "surface_base_color_value_resolution_invalid",
                (
                    "the active material graph has no structured surface "
                    "base-color value resolution"
                ),
                material_record_id=material_record_id,
            )
        )
        resolution = {}
    if not isinstance(evidence, list):
        blockers.append(
            input_error(
                "surface_base_color_constant_evidence_invalid",
                (
                    "the material's resolved surface base-color constant "
                    "evidence is not a list"
                ),
                material_record_id=material_record_id,
            )
        )
        evidence = None
    if not isinstance(attestation, Mapping):
        blockers.append(
            input_error(
                "surface_base_color_value_resolution_attestation_invalid",
                (
                    "the material has no structured surface base-color "
                    "value-resolution attestation"
                ),
                material_record_id=material_record_id,
            )
        )
        attestation = {}

    resolution_lists: dict[str, list[Any]] = {}
    for field in (
        "consumers",
        "constant_sources",
        "excluded_producers",
        "terminal_discovery_blockers",
        "blockers",
    ):
        value = resolution.get(field)
        if not isinstance(value, list) or not all(
            isinstance(record, Mapping) for record in value
        ):
            blockers.append(
                input_error(
                    "surface_base_color_value_resolution_field_invalid",
                    (
                        f"base-color value-resolution field {field!r} is "
                        "missing or is not a list of mappings"
                    ),
                    material_record_id=material_record_id,
                    field=field,
                )
            )
        else:
            resolution_lists[field] = value

    consumer_count = resolution.get("consumer_count")
    constant_source_count = resolution.get("constant_source_count")
    consumers = resolution_lists.get("consumers")
    constant_sources = resolution_lists.get("constant_sources")
    resolution_blocker_records = resolution_lists.get("blockers")
    if (
        not is_nonnegative_int(consumer_count)
        or consumers is None
        or consumer_count != len(consumers)
    ):
        blockers.append(
            input_error(
                "surface_base_color_value_resolution_count_mismatch",
                (
                    "the recorded base-color consumer count is malformed "
                    "or differs from the consumer table"
                ),
                material_record_id=material_record_id,
                field="consumer_count",
                observed=consumer_count,
                computed=(len(consumers) if consumers is not None else None),
            )
        )
    if (
        not is_nonnegative_int(constant_source_count)
        or constant_sources is None
        or constant_source_count != len(constant_sources)
    ):
        blockers.append(
            input_error(
                "surface_base_color_value_resolution_count_mismatch",
                (
                    "the recorded constant-source count is malformed or "
                    "differs from the source table"
                ),
                material_record_id=material_record_id,
                field="constant_source_count",
                observed=constant_source_count,
                computed=(
                    len(constant_sources)
                    if constant_sources is not None
                    else None
                ),
            )
        )

    recognized_names = resolution.get("recognized_shader_input_names")
    if recognized_names != sorted(
        BASE_COLOR_INPUTS
    ):
        blockers.append(
            input_error(
                "surface_base_color_recognized_input_policy_invalid",
                "the serialized recognized Shader-input allowlist is stale or malformed",
                material_record_id=material_record_id,
                observed=recognized_names,
                expected=sorted(
                    BASE_COLOR_INPUTS
                ),
            )
        )

    terminal_roots = resolution.get("surface_terminal_roots")
    terminal_root_keys: list[tuple[str, str]] = []
    if not isinstance(terminal_roots, list):
        blockers.append(
            input_error(
                "surface_base_color_terminal_roots_invalid",
                "surface-terminal root provenance is missing or malformed",
                material_record_id=material_record_id,
            )
        )
        terminal_roots = []
    else:
        for root_index, root in enumerate(terminal_roots):
            if not isinstance(root, Mapping):
                blockers.append(
                    input_error(
                        "surface_base_color_terminal_root_invalid",
                        "a surface-terminal root record is not a mapping",
                        material_record_id=material_record_id,
                        root_index=root_index,
                    )
                )
                continue
            material_output_path = root.get("material_output_path")
            render_context = root.get("render_context")
            derived_context = surface_output_render_context(
                material_output_path
            )
            if (
                not isinstance(material_output_path, str)
                or not material_output_path
                or not isinstance(render_context, str)
                or not render_context
                or derived_context != render_context
            ):
                blockers.append(
                    input_error(
                        "surface_base_color_terminal_root_invalid",
                        "a root must be an exact Material outputs:*:surface path with its derived render context",
                        material_record_id=material_record_id,
                        root_index=root_index,
                        root=dict(root),
                        derived_render_context=derived_context,
                    )
                )
                continue
            terminal_root_keys.append(
                (material_output_path, render_context)
            )
        if (
            terminal_root_keys
            != sorted(set(terminal_root_keys), key=lambda item: (item[0], item[1]))
            or len(terminal_root_keys) != len(terminal_roots)
        ):
            blockers.append(
                input_error(
                    "surface_base_color_terminal_root_order_invalid",
                    "surface-terminal roots must be unique and in deterministic reader order",
                    material_record_id=material_record_id,
                    observed=terminal_root_keys,
                )
            )

    terminal_shader_outputs = resolution.get(
        "surface_terminal_shader_outputs"
    )
    terminal_shader_by_key: dict[
        tuple[str, str], Mapping[str, Any]
    ] = {}
    covered_root_keys: set[tuple[str, str]] = set()
    if not isinstance(terminal_shader_outputs, list):
        blockers.append(
            input_error(
                "surface_base_color_terminal_shader_outputs_invalid",
                "surface-terminal Shader-output provenance is missing or malformed",
                material_record_id=material_record_id,
            )
        )
        terminal_shader_outputs = []
    else:
        for output_index, output_record in enumerate(
            terminal_shader_outputs
        ):
            output_errors: list[str] = []
            if not isinstance(output_record, Mapping):
                output_errors.append("record is not a mapping")
                output_record = {}
            shader_path = output_record.get("shader_path")
            render_context = output_record.get("render_context")
            terminal_paths = output_record.get("terminal_output_paths")
            material_paths = output_record.get("material_output_paths")
            if not isinstance(shader_path, str) or not shader_path:
                output_errors.append("shader_path is invalid")
            if not isinstance(render_context, str) or not render_context:
                output_errors.append("render_context is invalid")
            if (
                not isinstance(terminal_paths, list)
                or not terminal_paths
                or not all(
                    isinstance(path, str)
                    and path.startswith(f"{shader_path}.outputs:")
                    for path in terminal_paths
                )
                or len(terminal_paths) != len(set(terminal_paths))
            ):
                output_errors.append(
                    "terminal_output_paths must be unique Outputs on shader_path"
                )
            if (
                not isinstance(material_paths, list)
                or not material_paths
                or not all(
                    (path, render_context) in terminal_root_keys
                    for path in material_paths
                )
                or material_paths != sorted(set(material_paths))
            ):
                output_errors.append(
                    "material_output_paths must be sorted unique roots in the same render context"
                )
            key = (str(shader_path), str(render_context))
            if key in terminal_shader_by_key:
                output_errors.append("Shader/render-context identity is duplicated")
            terminal_shader_by_key[key] = output_record
            if isinstance(material_paths, list):
                covered_root_keys.update(
                    (str(path), str(render_context))
                    for path in material_paths
                    if (str(path), str(render_context))
                    in terminal_root_keys
                )
            if output_errors:
                blockers.append(
                    input_error(
                        "surface_base_color_terminal_shader_output_invalid",
                        "a resolved surface-terminal Shader-output record is malformed or untrusted",
                        material_record_id=material_record_id,
                        output_index=output_index,
                        errors=output_errors,
                    )
                )
        expected_terminal_output_order = sorted(
            terminal_shader_outputs,
            key=lambda item: (
                str(item.get("shader_path"))
                if isinstance(item, Mapping)
                else "",
                str(item.get("render_context"))
                if isinstance(item, Mapping)
                else "",
            ),
        )
        if terminal_shader_outputs != expected_terminal_output_order:
            blockers.append(
                input_error(
                    "surface_base_color_terminal_shader_output_order_invalid",
                    "surface-terminal Shader outputs are not in deterministic reader order",
                    material_record_id=material_record_id,
                )
            )
    if set(terminal_root_keys) != covered_root_keys:
        blockers.append(
            input_error(
                "surface_base_color_terminal_shader_coverage_incomplete",
                "every connected Material surface root must resolve to at least one non-container Shader Output",
                material_record_id=material_record_id,
                expected_roots=terminal_root_keys,
                covered_roots=sorted(covered_root_keys),
            )
        )

    terminal_discovery_blockers = resolution_lists.get(
        "terminal_discovery_blockers"
    )
    if terminal_discovery_blockers is not None:
        allowed_terminal_blocker_codes = {
            "surface_terminal_shader_oracle_failed",
            "surface_terminal_shader_attribute_invalid",
            "surface_terminal_oracle_nonshader_output",
            "surface_terminal_shader_oracle_empty",
        }
        for blocker_index, blocker_record in enumerate(
            terminal_discovery_blockers
        ):
            material_output_path = blocker_record.get(
                "material_output_path"
            )
            render_context = blocker_record.get("render_context")
            if (
                blocker_record.get("code")
                not in allowed_terminal_blocker_codes
                or not isinstance(blocker_record.get("message"), str)
                or not blocker_record.get("message")
                or (material_output_path, render_context)
                not in terminal_root_keys
            ):
                blockers.append(
                    input_error(
                        "surface_base_color_terminal_discovery_blocker_invalid",
                        "a terminal-discovery blocker is malformed or does not belong to an exact surface root",
                        material_record_id=material_record_id,
                        blocker_index=blocker_index,
                        blocker_record=blocker_record,
                    )
                )
        canonical_terminal_blockers = [
            canonical_json(item) for item in terminal_discovery_blockers
        ]
        if canonical_terminal_blockers != sorted(
            canonical_terminal_blockers, key=str
        ):
            blockers.append(
                input_error(
                    "surface_base_color_terminal_discovery_blocker_order_invalid",
                    "terminal-discovery blockers are not in deterministic reader order",
                    material_record_id=material_record_id,
                )
            )

    consumers_by_key: dict[
        tuple[str, str, str], Mapping[str, Any]
    ] = {}
    if consumers is not None:
        for consumer_index, consumer in enumerate(consumers):
            consumer_errors = consumer_resolution_errors(consumer)
            key = (
                str(consumer.get("consumer_shader_path")),
                str(consumer.get("consumer_property_path")),
                str(consumer.get("render_context")),
            )
            if key in consumers_by_key:
                consumer_errors.append("consumer identity is duplicated")
            terminal_shader_record = terminal_shader_by_key.get(
                (key[0], key[2])
            )
            if terminal_shader_record is None:
                consumer_errors.append(
                    "consumer Shader/render context is not an active surface terminal"
                )
            elif consumer.get("surface_terminal_output_paths") != (
                terminal_shader_record.get("terminal_output_paths")
            ):
                consumer_errors.append(
                    "consumer surface-terminal outputs disagree with terminal discovery"
                )
            consumers_by_key[key] = consumer
            if consumer_errors:
                blockers.append(
                    input_error(
                        "surface_base_color_consumer_resolution_invalid",
                        "an active recognized base-color consumer resolution is malformed or untrusted",
                        material_record_id=material_record_id,
                        consumer_index=consumer_index,
                        consumer_property_path=consumer.get(
                            "consumer_property_path"
                        ),
                        errors=consumer_errors,
                    )
                )
        expected_consumer_order = sorted(
            consumers,
            key=lambda item: (
                str(item.get("consumer_shader_path")),
                str(item.get("render_context")),
                str(item.get("consumer_property_path")),
            ),
        )
        if consumers != expected_consumer_order:
            blockers.append(
                input_error(
                    "surface_base_color_consumer_order_invalid",
                    "consumer resolutions are not in deterministic reader order",
                    material_record_id=material_record_id,
                )
            )

    if constant_sources is not None:
        source_paths: list[str] = []
        for source_index, source in enumerate(constant_sources):
            source_path = source.get("source_property_path")
            if isinstance(source_path, str):
                source_paths.append(source_path)
            source_errors = constant_source_errors(
                source, consumers_by_key
            )
            if source_errors:
                blockers.append(
                    input_error(
                        "surface_base_color_constant_source_invalid",
                        "a resolved base-color constant source is malformed or fails its integrity attestation",
                        material_record_id=material_record_id,
                        source_index=source_index,
                        source_property_path=source_path,
                        errors=source_errors,
                    )
                )
        if (
            source_paths != sorted(source_paths)
            or len(source_paths) != len(set(source_paths))
            or len(source_paths) != len(constant_sources)
        ):
            blockers.append(
                input_error(
                    "surface_base_color_constant_source_identity_invalid",
                    "constant sources must have unique paths in deterministic order",
                    material_record_id=material_record_id,
                    source_property_paths=source_paths,
                )
            )

    if consumers is not None:
        flattened_consumer_blockers = [
            item
            for consumer in consumers
            for item in (
                consumer.get("blockers")
                if isinstance(consumer.get("blockers"), list)
                else []
            )
        ]
        flattened_exclusions = [
            item
            for consumer in consumers
            for item in (
                consumer.get("excluded_producers")
                if isinstance(consumer.get("excluded_producers"), list)
                else []
            )
        ]
        global_exclusions = resolution_lists.get("excluded_producers")

        def canonical_sorted(values: list[Any]) -> list[str | None]:
            return sorted((canonical_json(value) for value in values), key=str)

        if (
            resolution_blocker_records is not None
            and canonical_sorted(resolution_blocker_records)
            != canonical_sorted(
                [
                    *(
                        terminal_discovery_blockers
                        if terminal_discovery_blockers is not None
                        else []
                    ),
                    *flattened_consumer_blockers,
                ]
            )
        ):
            blockers.append(
                input_error(
                    "surface_base_color_resolution_blocker_table_mismatch",
                    "global resolution blockers differ from terminal-discovery plus consumer blockers",
                    material_record_id=material_record_id,
                )
            )
        if (
            global_exclusions is not None
            and canonical_sorted(global_exclusions)
            != canonical_sorted(flattened_exclusions)
        ):
            blockers.append(
                input_error(
                    "surface_base_color_excluded_producer_table_mismatch",
                    "global excluded producers differ from consumer exclusions",
                    material_record_id=material_record_id,
                )
            )

    resolution_hash = resolution.get("resolution_sha256")
    computed_resolution_hash = (
        compute_surface_base_color_value_resolution_sha256(resolution)
    )
    if (
        not is_sha256(resolution_hash)
        or resolution_hash != computed_resolution_hash
    ):
        blockers.append(
            input_error(
                "surface_base_color_value_resolution_hash_mismatch",
                "the active-graph base-color resolution differs from its canonical digest",
                material_record_id=material_record_id,
                observed=resolution_hash,
                computed=computed_resolution_hash,
            )
        )
    if resolution_blocker_records:
        blockers.append(
            input_error(
                "surface_base_color_value_resolution_blocked",
                (
                    "the reader could not prove one or more active surface "
                    "base-color value producers"
                ),
                material_record_id=material_record_id,
                entry_count=len(resolution_blocker_records),
                entries=resolution_blocker_records,
            )
        )

    if (
        evidence is not None
        and constant_sources is not None
        and evidence != constant_sources
    ):
        blockers.append(
            input_error(
                "surface_base_color_constant_evidence_mismatch",
                (
                    "the material constant-evidence copy differs from the "
                    "attested active-graph source table"
                ),
                material_record_id=material_record_id,
            )
        )

    expected_attestation = {
        "active_graph_resolution_sha256": resolution_hash,
        "consumer_count": consumer_count,
        "constant_source_count": constant_source_count,
        "blocker_count": (
            len(resolution_blocker_records)
            if resolution_blocker_records is not None
            else None
        ),
    }
    attestation_scalar_types_valid = (
        is_nonnegative_int(attestation.get("consumer_count"))
        and is_nonnegative_int(attestation.get("constant_source_count"))
        and is_nonnegative_int(attestation.get("blocker_count"))
    )
    if (
        not attestation_scalar_types_valid
        or any(
            attestation.get(key) != value
            for key, value in expected_attestation.items()
        )
    ):
        blockers.append(
            input_error(
                "surface_base_color_value_resolution_attestation_mismatch",
                (
                    "the material attestation does not exactly match the "
                    "active-graph base-color resolution"
                ),
                material_record_id=material_record_id,
                observed=dict(attestation),
                expected=expected_attestation,
            )
        )
    return blockers


def compute_stage_color_management_sha256(
    configuration: Any,
    color_management_system: Any,
) -> str | None:
    """Hash the stage color metadata."""

    payload = {
        "stage_color_configuration": configuration,
        "stage_color_management_system": color_management_system,
    }
    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    digest = hashlib.sha256()
    for value in ("stage-color-management-v1", serialized):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stage_color_management_blockers(
    extraction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Check the stage color settings and their saved hash."""

    blockers: list[dict[str, Any]] = []
    configuration = extraction.get("stage_color_configuration")
    system = extraction.get("stage_color_management_system")
    recorded_hash = extraction.get("stage_color_management_sha256")

    if not isinstance(configuration, Mapping):
        blockers.append(
            input_error(
                "stage_color_configuration_invalid",
                "stage_color_configuration is missing or is not a mapping",
            )
        )
    if not isinstance(system, Mapping):
        blockers.append(
            input_error(
                "stage_color_management_system_invalid",
                "stage_color_management_system is missing or is not a mapping",
            )
        )
    computed_hash = compute_stage_color_management_sha256(
        configuration, system
    )
    if not is_sha256(recorded_hash) or recorded_hash != computed_hash:
        blockers.append(
            input_error(
                "stage_color_management_hash_mismatch",
                "stage color metadata is malformed, stale, or differs from its digest",
                observed=recorded_hash,
                computed=computed_hash,
            )
        )

    if isinstance(configuration, Mapping):
        api_status = configuration.get("api_status")
        metadata_authored = configuration.get("metadata_authored")
        resolution_status = configuration.get("resolution_status")
        if api_status != "complete" or configuration.get("error") is not None:
            blockers.append(
                input_error(
                    "stage_color_configuration_api_incomplete",
                    "the effective stage color configuration could not be read exactly",
                    api_status=api_status,
                    error=configuration.get("error"),
                )
            )
        if not isinstance(metadata_authored, bool):
            blockers.append(
                input_error(
                    "stage_color_configuration_authorship_invalid",
                    "stage color-configuration authorship is not a boolean",
                )
            )
        if resolution_status not in {"not_configured", "resolved_file"}:
            blockers.append(
                input_error(
                    "stage_color_configuration_unresolved",
                    "an effective stage color configuration is not a resolved file",
                    resolution_status=resolution_status,
                )
            )
        elif resolution_status == "resolved_file":
            resolved_path = configuration.get("dependency_resolved_path")
            resolved_size = configuration.get("resolved_file_size_bytes")
            if not isinstance(resolved_path, str) or not resolved_path:
                blockers.append(
                    input_error(
                        "stage_color_configuration_path_invalid",
                        "resolved stage color configuration has no dependency path",
                    )
                )
            if not is_sha256(configuration.get("resolved_file_sha256")):
                blockers.append(
                    input_error(
                        "stage_color_configuration_content_hash_invalid",
                        "resolved stage color configuration has no exact content hash",
                    )
                )
            if (
                not isinstance(resolved_size, int)
                or isinstance(resolved_size, bool)
                or resolved_size < 0
            ):
                blockers.append(
                    input_error(
                        "stage_color_configuration_size_invalid",
                        "resolved stage color configuration has no valid byte size",
                    )
                )
            if not any(
                isinstance(configuration.get(field), str)
                and bool(configuration.get(field))
                for field in ("authored_asset_path", "effective_asset_path")
            ):
                blockers.append(
                    input_error(
                        "stage_color_configuration_asset_identity_missing",
                        "resolved stage color configuration has no authored/effective asset identity",
                    )
                )
        else:
            absent_fields = (
                "authored_asset_path",
                "authored_resolved_path",
                "effective_asset_path",
                "effective_resolved_path",
                "dependency_resolved_path",
                "resolved_file_sha256",
                "resolved_file_size_bytes",
            )
            if metadata_authored is not False or any(
                configuration.get(field) is not None for field in absent_fields
            ):
                blockers.append(
                    input_error(
                        "stage_color_configuration_absence_inconsistent",
                        "not_configured stage color metadata contains authored or resolved data",
                    )
                )

    if isinstance(system, Mapping):
        api_status = system.get("api_status")
        metadata_authored = system.get("metadata_authored")
        authored_value = system.get("authored_value")
        effective_value = system.get("effective_value")
        if api_status != "complete" or system.get("error") is not None:
            blockers.append(
                input_error(
                    "stage_color_management_system_api_incomplete",
                    "the effective stage color-management system could not be read exactly",
                    api_status=api_status,
                    error=system.get("error"),
                )
            )
        if not isinstance(metadata_authored, bool):
            blockers.append(
                input_error(
                    "stage_color_management_system_authorship_invalid",
                    "stage color-management-system authorship is not a boolean",
                )
            )
        elif metadata_authored and (
            not isinstance(authored_value, str) or not authored_value
        ):
            blockers.append(
                input_error(
                    "stage_color_management_system_authored_value_invalid",
                    "an authored color-management system has no non-empty value",
                )
            )
        elif not metadata_authored and authored_value is not None:
            blockers.append(
                input_error(
                    "stage_color_management_system_absence_inconsistent",
                    "unauthored color-management-system metadata contains an authored value",
                )
            )
        if effective_value is not None and (
            not isinstance(effective_value, str) or not effective_value
        ):
            blockers.append(
                input_error(
                    "stage_color_management_system_effective_value_invalid",
                    "effective color-management-system value is malformed",
                )
            )
    return blockers


def untrusted_mdl_fully_superseded(
    material: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    """Ignore an MDL warning only when a matching source replaces it."""

    role = candidate.get("candidate_role")
    paths = candidate.get("resolved_paths")
    if not isinstance(role, str) or not role or not isinstance(paths, list) or not paths:
        return False
    candidate_paths = {str(path) for path in paths if str(path)}
    if len(candidate_paths) != len(paths):
        return False
    candidates = material.get("surface_texture_candidates")
    role_records = candidates.get(role) if isinstance(candidates, Mapping) else None
    if not isinstance(role_records, list):
        return False
    trusted_paths: set[str] = set()
    for record in role_records:
        if not isinstance(record, Mapping):
            continue
        if record.get("used_path_fallback") is True or record.get(
            "trust_status"
        ):
            continue
        resolved = record.get("resolved_paths")
        if isinstance(resolved, list):
            trusted_paths.update(str(path) for path in resolved if str(path))
    return bool(candidate_paths) and candidate_paths.issubset(trusted_paths)


def role_conflict_is_semantically_exact(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    semantic_role = record.get("semantic_consumer_inferred_role") or record.get(
        "input_inferred_role"
    )
    return (
        isinstance(semantic_role, str)
        and semantic_role not in {"", "unknown"}
        and record.get("selected_role") == semantic_role
        and record.get("role_selection_source")
        in {"downstream_consumer", "connected_input"}
        and isinstance(record.get("resolved_paths"), list)
        and bool(record.get("resolved_paths"))
    )


UDIM_MARKERS = ("<UDIM>", "<udim>", "%04d")


def serialized_udim_tile_id(
    authored_asset: Any, resolved_path: Any
) -> int | None:
    if not isinstance(authored_asset, str) or not isinstance(resolved_path, str):
        return None
    matches = [marker for marker in UDIM_MARKERS if marker in authored_asset]
    if len(matches) != 1 or authored_asset.count(matches[0]) != 1:
        return None
    marker = matches[0]
    authored_name = authored_asset.replace("\\", "/").rsplit("/", 1)[-1]
    resolved_name = resolved_path.replace("\\", "/").rsplit("/", 1)[-1]
    if marker not in authored_name:
        return None
    pattern = re.compile(
        "^"
        + re.escape(authored_name).replace(
            re.escape(marker), r"(?P<tile>[0-9]{4})"
        )
        + "$"
    )
    match = pattern.fullmatch(resolved_name)
    if match is None:
        return None
    tile_id = int(match.group("tile"))
    return tile_id if 1001 <= tile_id <= 1999 else None


def selected_udim_texture_set_errors(
    material: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_sets = material.get("selected_surface_texture_sets")
    selected_files = material.get("selected_surface_textures")
    selected_evidence = material.get("selected_surface_texture_evidence")
    candidates = material.get("surface_texture_candidates")
    unresolved = material.get("unresolved_surface_textures")
    ambiguities = material.get("surface_texture_ambiguities")
    if not all(
        isinstance(value, Mapping)
        for value in (
            selected_sets,
            selected_files,
            selected_evidence,
            candidates,
        )
    ):
        return []

    unresolved_roles = {
        str(record.get("role"))
        for record in (unresolved if isinstance(unresolved, list) else [])
        if isinstance(record, Mapping) and record.get("role")
    }
    ambiguous_roles = {
        str(record.get("role"))
        for record in (ambiguities if isinstance(ambiguities, list) else [])
        if isinstance(record, Mapping) and record.get("role")
    }
    errors: list[dict[str, Any]] = []
    for role, texture_set in selected_sets.items():
        reasons: list[str] = []
        if not isinstance(role, str) or not role:
            reasons.append("role")
        if role in selected_files:
            reasons.append("also_selected_as_single_file")
        if not isinstance(texture_set, Mapping):
            errors.append({"role": role, "reasons": ["set_record"]})
            continue
        if set(texture_set) != {
            "texture_type",
            "authored_asset",
            "tiles",
            "evidence_sha256",
        }:
            reasons.append("set_fields")
        if texture_set.get("texture_type") != "udim":
            reasons.append("texture_type")

        role_candidates = candidates.get(role)
        role_evidence = selected_evidence.get(role)
        candidate = (
            role_candidates[0]
            if isinstance(role_candidates, list)
            and len(role_candidates) == 1
            and isinstance(role_candidates[0], Mapping)
            else None
        )
        if candidate is None:
            reasons.append("single_candidate")
        elif role_evidence != role_candidates:
            reasons.append("selected_evidence")

        authored_asset = texture_set.get("authored_asset")
        tiles = texture_set.get("tiles")
        paths: list[str] = []
        tile_ids: list[int] = []
        parent_paths: set[str] = set()
        if not isinstance(authored_asset, str) or not authored_asset:
            reasons.append("authored_asset")
        if not isinstance(tiles, list) or not tiles:
            reasons.append("tiles")
        else:
            for tile in tiles:
                if not isinstance(tile, Mapping) or set(tile) != {
                    "tile_id",
                    "resolved_path",
                    "file_sha256",
                }:
                    reasons.append("tile_record")
                    continue
                tile_id = tile.get("tile_id")
                path = tile.get("resolved_path")
                valid_tile_id = (
                    type(tile_id) is int and 1001 <= tile_id <= 1999
                )
                if not valid_tile_id:
                    reasons.append("tile_id")
                if not isinstance(path, str) or not path:
                    reasons.append("tile_path")
                elif (
                    not valid_tile_id
                    or serialized_udim_tile_id(authored_asset, path) != tile_id
                ):
                    reasons.append("tile_substitution")
                else:
                    paths.append(path)
                    tile_ids.append(tile_id)
                    parent_paths.add(
                        path.replace("\\", "/").rsplit("/", 1)[0]
                    )
                if not is_sha256(tile.get("file_sha256")):
                    reasons.append("tile_sha256")
        if tile_ids != sorted(set(tile_ids)):
            reasons.append("tile_order")
        tile_coordinates = {
            ((tile_id - 1001) % 10, (tile_id - 1001) // 10)
            for tile_id in tile_ids
        }
        expected_coordinates = (
            {
                (u, v)
                for u in range(
                    min(point[0] for point in tile_coordinates),
                    max(point[0] for point in tile_coordinates) + 1,
                )
                for v in range(
                    min(point[1] for point in tile_coordinates),
                    max(point[1] for point in tile_coordinates) + 1,
                )
            }
            if tile_coordinates
            else set()
        )
        if tile_coordinates != expected_coordinates:
            reasons.append("non_rectangular_tile_grid")
        if paths and len(parent_paths) != 1:
            reasons.append("tile_parent")

        if candidate is not None:
            if candidate.get("role") != role:
                reasons.append("candidate_role")
            if candidate.get("authored_asset") != authored_asset:
                reasons.append("candidate_authored_asset")
            if candidate.get("resolved_paths") != paths:
                reasons.append("candidate_paths")
            if candidate.get("role_distinct_resolved_files") != paths:
                reasons.append("role_paths")
            if candidate.get("role_distinct_resolved_file_count") != len(paths):
                reasons.append("role_file_count")
            if candidate.get("resolved_file_count") != len(paths):
                reasons.append("candidate_file_count")
            if candidate.get("has_multiple_resolved_files") is not (
                len(paths) > 1
            ):
                reasons.append("candidate_multiple_file_status")
            if candidate.get("role_selection_status") != "selected_exact_udim_set":
                reasons.append("selection_status")
            if candidate.get("role_is_ambiguous") is not False:
                reasons.append("ambiguity_status")
            if candidate.get("role_has_udim") is not True:
                reasons.append("role_has_udim")
            if candidate.get("is_udim") is not True:
                reasons.append("candidate_is_udim")
            if candidate.get("role_has_multiple_sources") is not False:
                reasons.append("multiple_sources")
            if type(candidate.get("role_source_count")) is not int or candidate.get(
                "role_source_count"
            ) != 1:
                reasons.append("source_count")
            if candidate.get("role_competing_unresolved_sources") != []:
                reasons.append("unresolved_competitors")
            if candidate.get("used_path_fallback") is not False:
                reasons.append("path_fallback")
            expected_evidence_hash = canonical_json_sha256(
                "textured-render-provenance", candidate
            )
            if (
                not is_sha256(texture_set.get("evidence_sha256"))
                or texture_set.get("evidence_sha256") != expected_evidence_hash
            ):
                reasons.append("evidence_sha256")
        if role in unresolved_roles:
            reasons.append("unresolved_role")
        if role in ambiguous_roles:
            reasons.append("still_ambiguous")
        if reasons:
            errors.append({"role": role, "reasons": sorted(set(reasons))})

    selected_set_roles = set(selected_sets)
    for role, records in candidates.items():
        if not isinstance(records, list):
            continue
        if any(
            isinstance(record, Mapping)
            and record.get("role_selection_status") == "selected_exact_udim_set"
            for record in records
        ) and role not in selected_set_roles:
            errors.append(
                {"role": role, "reasons": ["selected_candidate_has_no_set"]}
            )
    return errors


def canonical_selected_mdl_semantics(record: Mapping[str, Any]) -> dict[str, Any] | None:
    terminal_module = record.get("terminal_material_module")
    if terminal_module is None:
        canonical_terminal = None
    elif isinstance(terminal_module, Mapping):
        authored_module = terminal_module.get("authored_module")
        source_sha256 = terminal_module.get("source_sha256")
        if not isinstance(authored_module, str) or not is_sha256(source_sha256):
            return None
        canonical_terminal = {
            "authored_module": authored_module,
            "source_sha256": source_sha256,
        }
    else:
        return None
    canonical = {
        "material_family": record.get("material_family"),
        "definition_expression_kind": record.get(
            "definition_expression_kind"
        ),
        "selected_material_definition": record.get(
            "selected_material_definition"
        ),
        "direct_callee": record.get("direct_callee"),
        "terminal_material_callee": record.get(
            "terminal_material_callee"
        ),
        "terminal_material_module": canonical_terminal,
        "parameter_defaults": record.get("parameter_defaults"),
    }
    if record.get("material_family") in {
        "OmniGlass",
        "OmniSurface",
        "OmniSurfacePreset",
        "NvidiaGlassWithVolume",
        "SimPBRTranslucent",
    } or (
        record.get("material_family") == "OmniPBR"
        and record.get("definition_expression_kind")
        in {
            "hash_attested_installed_material_implementation",
            "hash_attested_local_material_implementation",
        }
    ):
        dependency = record.get("source_dependency_attestation")
        canonical["source_dependency_attestation_sha256"] = (
            dependency.get("attestation_sha256")
            if isinstance(dependency, Mapping)
            else None
        )
    return canonical


def mdl_dependency_attestation_is_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    classification = value.get("classification")
    modules = value.get("modules")
    if (
        classification != "content_addressed_mdl_import_closure"
        or not isinstance(modules, list)
        or not modules
    ):
        return False
    canonical_modules: list[dict[str, str]] = []
    serialized_order: list[tuple[str, str, str]] = []
    for module in modules:
        if not isinstance(module, Mapping):
            return False
        authored_module = module.get("authored_module")
        relative_path = module.get("relative_path")
        resolved_source_path = module.get("resolved_source_path")
        source_sha256 = module.get("source_sha256")
        if (
            not isinstance(authored_module, str)
            or not authored_module
            or not isinstance(relative_path, str)
            or not relative_path
            or not relative_path.casefold().endswith(".mdl")
            or not isinstance(resolved_source_path, str)
            or not resolved_source_path
            or not resolved_source_path.startswith("/")
            or not is_sha256(source_sha256)
        ):
            return False
        canonical_modules.append(
            {
                "authored_module": authored_module,
                "relative_path": relative_path,
                "source_sha256": source_sha256,
            }
        )
        serialized_order.append(
            (authored_module, relative_path, source_sha256)
        )
    if serialized_order != sorted(serialized_order) or len(
        set(serialized_order)
    ) != len(serialized_order):
        return False
    canonical = {
        "classification": classification,
        "modules": canonical_modules,
    }
    return value.get("attestation_sha256") == canonical_json_sha256(
        "selected-mdl-source-dependency-attestation", canonical
    )


def selected_mdl_material_semantics_blockers(
    record_id: Any,
    material: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = material.get("selected_mdl_material_semantics")
    if not isinstance(records, list):
        return [
            input_error(
                "selected_mdl_material_semantics_invalid",
                "selected MDL material semantics are missing or malformed",
                material_record_id=record_id,
            )
        ]

    invalid: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    dependency_required_families = {
        "SimPBR",
        "SimPBRTranslucent",
        "OmniGlass",
        "OmniSurface",
        "OmniSurfacePreset",
        "NvidiaGlassWithVolume",
    }
    allowed_value_types = {
        "bool",
        "color",
        "float",
        "float2",
        "float3",
        "int",
        "int2",
        "qualified_enum",
        "state_normal",
        "texture_2d",
    }
    for ordinal, record in enumerate(records):
        reasons: list[str] = []
        if not isinstance(record, Mapping):
            invalid.append({"ordinal": ordinal, "reasons": ["not_mapping"]})
            continue
        shader_path = record.get("shader_path")
        contexts = record.get("render_contexts")
        source_sha256 = record.get("source_sha256")
        source_sub_identifier = record.get("source_asset_sub_identifier")
        selected_definition = record.get("selected_material_definition")
        family = record.get("material_family")
        expression_kind = record.get("definition_expression_kind")
        parameter_defaults = record.get("parameter_defaults")
        if not isinstance(shader_path, str) or not shader_path:
            reasons.append("shader_path")
        if (
            not isinstance(contexts, list)
            or not contexts
            or any(not isinstance(item, str) or not item for item in contexts)
            or contexts != sorted(set(contexts))
        ):
            reasons.append("render_contexts")
        for field in (
            "authored_source_asset",
            "resolved_source_path",
            "source_resolution_method",
        ):
            if not isinstance(record.get(field), str) or not record.get(field):
                reasons.append(field)
        if not is_sha256(source_sha256):
            reasons.append("source_sha256")
        if source_sub_identifier is not None and (
            not isinstance(source_sub_identifier, str)
            or not source_sub_identifier
        ):
            reasons.append("source_asset_sub_identifier")
        if not isinstance(selected_definition, str) or not selected_definition:
            reasons.append("selected_material_definition")
        if not isinstance(family, str) or not family:
            reasons.append("material_family")
        if not isinstance(expression_kind, str) or not expression_kind:
            reasons.append("definition_expression_kind")
        installed_identities = {
            "OmniPBR": {
                (
                    "OmniPBR.mdl",
                    "OmniPBR",
                    "685a82c18b381f1cd29831490e4c356e61dbeac2d86803babc115cd6a4a845db",
                ),
                (
                    "OmniPBR_Opacity.mdl",
                    "OmniPBR_Opacity",
                    "b9378c8cb61a8732228409ba350994421754d15533ab6c6d7e618cdfe4191470",
                ),
            },
            "OmniGlass": {
                (
                    "OmniGlass.mdl",
                    "OmniGlass",
                    "c3f9555fbf8db2c4d47ac25501afaf23720260255381bde3df003fecf01869af",
                )
            },
            "OmniSurface": {
                (
                    "OmniSurface.mdl",
                    "OmniSurface",
                    "f16f788a3b8c19524992c6ef01c45d552cf57735c6fddade9da3316bc897b222",
                ),
                (
                    "OmniSurfaceLite.mdl",
                    "OmniSurfaceLite",
                    "0175002c1c361e8c016e1131196ed0c76fef911b87b4dd016bb39fc0e77ee23b",
                ),
            },
        }
        if expression_kind == "hash_attested_installed_material_implementation":
            expected_identities = installed_identities.get(family, set())
            if (
                record.get("authored_source_asset"),
                source_sub_identifier,
                source_sha256,
            ) not in expected_identities or record.get(
                "source_resolution_method"
            ) != "pinned_installed_isaac_omni_mdl_module":
                reasons.append("installed_material_identity")
        elif expression_kind == "hash_attested_local_material_implementation":
            local_identities = {
                "OmniPBR": {
                    (
                        "OmniPBR_Opacity.mdl",
                        "OmniPBR_Opacity",
                        "fbd3447652dbbaa2169c6e2f3887c3f3a7133800b41a40d7748d454ad357a3ce",
                    ),
                },
                "OmniSurface": {
                    (
                        "OmniSurface.mdl",
                        "OmniSurface",
                        "2fe52255e29cae1cb65a1ff1299773380ee6703e14f7654692bf8b195567f446",
                    ),
                    (
                        "OmniSurfaceLite.mdl",
                        "OmniSurfaceLite",
                        "e20b8cb76b42afcbe2ea425636707b062cad71fa27ad8e1dcb89cb52cbaa7315",
                    ),
                },
            }
            local_resolution_methods = {
                "OmniPBR": "hash_attested_asset_local_omni_mdl_module",
                "OmniSurface": (
                    "hash_attested_asset_local_omnisurface_module"
                ),
            }
            authored_basename = str(
                record.get("authored_source_asset") or ""
            ).rsplit("/", 1)[-1]
            if (
                authored_basename,
                source_sub_identifier,
                source_sha256,
            ) not in local_identities.get(family, set()) or record.get(
                "source_resolution_method"
            ) != local_resolution_methods.get(family):
                reasons.append("asset_local_material_identity")
            if (
                family == "OmniPBR"
                and selected_definition != "OmniPBR_Opacity"
            ):
                reasons.append("asset_local_material_identity")
        elif expression_kind == (
            "hash_attested_simpbr_translucent_implementation"
        ):
            authored_basename = str(
                record.get("authored_source_asset") or ""
            ).rsplit("/", 1)[-1]
            if (
                family != "SimPBRTranslucent"
                or authored_basename != "SimPBR_Translucent.mdl"
                or source_sub_identifier != "SimPBR_Translucent"
                or selected_definition != "SimPBR_Translucent"
                or source_sha256
                != "6829af54259b898b4237c02d9375067b63f802cbbbde6f9935f2e69a1b11e4bf"
            ):
                reasons.append("simpbr_translucent_material_identity")
        elif expression_kind == "hash_attested_direct_call_with_composed_defaults":
            allowed_callees = {
                "OmniGlass": {"::OmniGlass::OmniGlass"},
                "OmniSurface": {
                    "::OmniSurface::OmniSurface",
                    "::OmniSurfaceLite::OmniSurfaceLite",
                },
                "OmniSurfacePreset": {
                    "::OmniSurface::OmniSurface",
                    "::OmniSurfaceLite::OmniSurfaceLite",
                },
                "NvidiaGlassWithVolume": {
                    "..::Templates::GlassWithVolume::GlassWithVolume"
                },
            }
            if record.get("direct_callee") not in allowed_callees.get(
                family, set()
            ):
                reasons.append("composed_material_callee")
            if family == "OmniSurfacePreset":
                authored_source = str(
                    record.get("authored_source_asset") or ""
                )
                authored_basename = authored_source.rsplit("/", 1)[-1]
                installed_preset = (
                    authored_source == "OmniSurfacePresets.mdl"
                    and source_sub_identifier
                    in {
                        "OmniSurface_ClearWater",
                        "OmniSurface_FrostedGlass",
                        "OmniSurface_Velvet",
                        "OmniSurface_Wax",
                    }
                    and source_sha256
                    == "5c86c8545a1e215ec4b99e60eb66f9112ca5952cc66ca13ec0c26687dcfcb930"
                    and record.get("source_resolution_method")
                    == "pinned_installed_isaac_omni_mdl_module"
                )
                local_preset = (
                    authored_basename == "OmniSurfacePresets.mdl"
                    and source_sub_identifier
                    in {
                        "OmniSurface_ClearWater",
                        "OmniSurface_FrostedGlass",
                        "OmniSurface_Velvet",
                        "OmniSurface_Wax",
                    }
                    and source_sha256
                    == "3d964ae0df3e2283b8b4236916fa17f9146ba740d5eda079551a74fe98ed6f0b"
                    and record.get("source_resolution_method")
                    == "hash_attested_asset_local_omnisurface_module"
                )
                if not installed_preset and not local_preset:
                    reasons.append("omnisurface_preset_identity")
        elif family in dependency_required_families - {"SimPBR"}:
            reasons.append("unsupported_material_expression_kind")
        if not isinstance(record.get("selected_definition_source_line"), int) or isinstance(
            record.get("selected_definition_source_line"), bool
        ) or record.get("selected_definition_source_line", 0) < 1:
            reasons.append("selected_definition_source_line")
        for field in (
            "authored_direct_callee",
            "direct_callee",
            "terminal_material_callee",
        ):
            value = record.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                reasons.append(field)
        if not isinstance(parameter_defaults, Mapping):
            reasons.append("parameter_defaults")
        else:
            for name, default in parameter_defaults.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(default, Mapping)
                    or default.get("value_type") not in allowed_value_types
                    or "value" not in default
                ):
                    reasons.append("parameter_default_record")
                    break

        canonical = canonical_selected_mdl_semantics(record)
        if canonical is None or record.get(
            "semantics_sha256"
        ) != canonical_json_sha256(
            "selected-mdl-material-semantics", canonical
        ):
            reasons.append("semantics_sha256")
        dependency_attestation = record.get(
            "source_dependency_attestation"
        )
        dependency_required = family in dependency_required_families or (
            family == "OmniPBR"
            and expression_kind
            in {
                "hash_attested_installed_material_implementation",
                "hash_attested_local_material_implementation",
            }
        )
        if dependency_required:
            if not mdl_dependency_attestation_is_valid(
                dependency_attestation
            ):
                reasons.append("source_dependency_attestation")
        elif dependency_attestation is not None and not (
            mdl_dependency_attestation_is_valid(dependency_attestation)
        ):
            reasons.append("source_dependency_attestation")

        identity = (
            shader_path,
            record.get("authored_source_asset"),
            source_sub_identifier,
            source_sha256,
            record.get("semantics_sha256"),
        )
        if identity in identities:
            reasons.append("duplicate_semantics_record")
        identities.add(identity)
        if reasons:
            invalid.append(
                {
                    "ordinal": ordinal,
                    "shader_path": shader_path,
                    "reasons": sorted(set(reasons)),
                }
            )

    if invalid:
        return [
            input_error(
                "selected_mdl_material_semantics_invalid",
                "selected MDL material semantics or their content-addressed dependency proof are invalid",
                material_record_id=record_id,
                entries=invalid,
            )
        ]

    untrusted_roots = {
        (
            candidate.get("shader_path"),
            candidate.get("authored_asset"),
            candidate.get("source_asset_sub_identifier"),
        )
        for candidate in material.get("untrusted_mdl_source_candidates", [])
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("shader_path"), str)
    }
    selected_roots = {
        (
            record.get("shader_path"),
            record.get("authored_source_asset"),
            record.get("source_asset_sub_identifier"),
        )
        for record in records
        if isinstance(record, Mapping)
    }
    missing_roots: list[dict[str, Any]] = []
    active_roots: set[tuple[Any, Any, Any]] = set()
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return []
    for node in nodes:
        if not isinstance(node, Mapping) or node.get(
            "implementation_source"
        ) != "sourceAsset":
            continue
        shader_path = node.get("prim_path")
        info = node.get("info_attributes")
        source_attribute = (
            info.get("info:mdl:sourceAsset")
            if isinstance(info, Mapping)
            else None
        )
        source_value = (
            source_attribute.get("value")
            if isinstance(source_attribute, Mapping)
            else None
        )
        authored_source = (
            source_value.get("authored_path")
            if isinstance(source_value, Mapping)
            else None
        )
        sub_identifier_attribute = (
            info.get("info:mdl:sourceAsset:subIdentifier")
            if isinstance(info, Mapping)
            else None
        )
        source_sub_identifier = (
            sub_identifier_attribute.get("value")
            if isinstance(sub_identifier_attribute, Mapping)
            else None
        )
        if source_sub_identifier == "":
            source_sub_identifier = None
        root = (shader_path, authored_source, source_sub_identifier)
        active_roots.add(root)
        if root not in selected_roots and root not in untrusted_roots:
            missing_roots.append(
                {
                    "shader_path": shader_path,
                    "authored_source_asset": authored_source,
                    "source_asset_sub_identifier": source_sub_identifier,
                }
            )
    if missing_roots:
        return [
            input_error(
                "active_mdl_source_semantics_missing",
                "an active sourceAsset shader root has neither selected semantics nor a fail-closed parser diagnostic",
                material_record_id=record_id,
                entries=missing_roots,
            )
        ]
    extra_roots = sorted(
        selected_roots - active_roots,
        key=lambda item: tuple(str(value) for value in item),
    )
    if extra_roots:
        return [
            input_error(
                "selected_mdl_material_semantics_unbound",
                "selected MDL semantics do not cross-bind to an active sourceAsset shader root",
                material_record_id=record_id,
                entries=[
                    {
                        "shader_path": item[0],
                        "authored_source_asset": item[1],
                        "source_asset_sub_identifier": item[2],
                    }
                    for item in extra_roots
                ],
            )
        ]
    return []


def ignored_invalid_connection_fallback_blockers(
    material: Any,
) -> list[dict[str, Any]]:
    """Allow known OmniPBR fallback connections; reject other broken inputs."""

    if not isinstance(material, Mapping):
        return [
            input_error(
                "ignored_invalid_connection_fallback_invalid",
                "the material record is not a mapping",
            )
        ]
    material_record_id = material.get("material_record_id")
    graph = material.get("active_material_graph")
    if not isinstance(graph, Mapping):
        return []
    graph_fallbacks = graph.get("ignored_invalid_connection_fallbacks")
    nodes = graph.get("nodes")
    if not isinstance(graph_fallbacks, list) or not isinstance(nodes, list):
        return [
            input_error(
                "ignored_invalid_connection_fallback_invalid",
                "ignored invalid-connection evidence is missing or malformed",
                material_record_id=material_record_id,
            )
        ]

    node_fallbacks: list[Any] = []
    input_records_by_path: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    malformed_nodes = False
    for node in nodes:
        if not isinstance(node, Mapping):
            malformed_nodes = True
            continue
        fallbacks = node.get("ignored_invalid_connection_fallbacks")
        inputs = node.get("inputs")
        if not isinstance(fallbacks, list) or not isinstance(inputs, list):
            malformed_nodes = True
            continue
        node_fallbacks.extend(fallbacks)
        for shade_input in inputs:
            if not isinstance(shade_input, Mapping):
                malformed_nodes = True
                continue
            property_path = shade_input.get("property_path")
            if isinstance(property_path, str) and property_path:
                input_records_by_path.setdefault(property_path, []).append(
                    (node, shade_input)
                )

    errors: list[dict[str, Any]] = []
    if malformed_nodes or graph_fallbacks != node_fallbacks:
        errors.append(
            {
                "reason": "graph/node ignored-fallback tables do not agree",
                "graph_fallback_count": len(graph_fallbacks),
                "node_fallback_count": len(node_fallbacks),
            }
        )

    expected_record_fields = {
        "authored_invalid_source_property_path",
        "invalid_source_reason",
        "source_prim_path",
        "source_prim_exists",
        "source_prim_type",
        "consumer_property_path",
        "consumer_has_authored_value",
        "consumer_authored_value",
        "consumer_value_type",
        "effective_value_source",
        "openusd_resolution",
        "repair_required",
    }
    expected_value_types = {
        "reflection_roughness_constant": "float",
        "diffuse_color_constant": "color3f",
    }
    for fallback_index, fallback in enumerate(graph_fallbacks):
        fallback_errors: list[str] = []
        if not isinstance(fallback, Mapping):
            errors.append(
                {
                    "fallback_index": fallback_index,
                    "reason": "fallback record is not a mapping",
                }
            )
            continue
        consumer_path = fallback.get("consumer_property_path")
        source_path = fallback.get("authored_invalid_source_property_path")
        source_prim_path = fallback.get("source_prim_path")
        if set(fallback) != expected_record_fields:
            fallback_errors.append(
                "record fields do not exactly match the audited fallback schema"
            )
        if (
            fallback.get("invalid_source_reason")
            != "source_property_does_not_exist"
            or fallback.get("source_prim_exists") is not True
            or fallback.get("source_prim_type") != "Material"
            or fallback.get("consumer_has_authored_value") is not True
            or fallback.get("effective_value_source")
            != "consumer_authored_value"
            or fallback.get("openusd_resolution")
            != "invalid_source_connection_ignored_use_authored_input_value"
            or fallback.get("repair_required") is not True
        ):
            fallback_errors.append(
                "record is not the audited nonexistent Material-interface fallback"
            )

        consumer_input_name = (
            consumer_path.rsplit(".inputs:", 1)[1]
            if isinstance(consumer_path, str) and ".inputs:" in consumer_path
            else None
        )
        source_input_name = (
            source_path.rsplit(".inputs:", 1)[1]
            if isinstance(source_path, str) and ".inputs:" in source_path
            else None
        )
        expected_value_type = expected_value_types.get(consumer_input_name)
        if (
            not isinstance(consumer_path, str)
            or ".inputs:" not in consumer_path
            or not isinstance(source_path, str)
            or ".inputs:" not in source_path
            or not isinstance(source_prim_path, str)
            or source_path.rsplit(".inputs:", 1)[0] != source_prim_path
            or consumer_input_name != source_input_name
            or expected_value_type is None
            or consumer_path
            != f"{source_prim_path}/Shader.inputs:{consumer_input_name}"
            or material.get("material_path") != source_prim_path
            or material.get("source_material_path") != source_prim_path
            or graph.get("material_path") != source_prim_path
            or graph.get("source_material_path") != source_prim_path
        ):
            fallback_errors.append(
                "source and consumer are not an exact audited co-named Material/Shader input pair"
            )
        consumer_value = fallback.get("consumer_authored_value")
        value_is_valid = (
            not isinstance(consumer_value, bool)
            and isinstance(consumer_value, (int, float))
            and math.isfinite(float(consumer_value))
            and 0.0 <= float(consumer_value) <= 1.0
            if consumer_input_name == "reflection_roughness_constant"
            else finite_numeric_list(consumer_value, {3})
            and all(0.0 <= float(component) <= 1.0 for component in consumer_value)
            if consumer_input_name == "diffuse_color_constant"
            else False
        )
        if (
            fallback.get("consumer_value_type") != expected_value_type
            or not value_is_valid
        ):
            fallback_errors.append(
                "fallback value is not the exact finite typed value expected for the audited input"
            )

        matches = (
            input_records_by_path.get(consumer_path, [])
            if isinstance(consumer_path, str)
            else []
        )
        if len(matches) != 1:
            fallback_errors.append(
                "consumer does not resolve to exactly one serialized shader input"
            )
        else:
            node, shade_input = matches[0]
            active_input_paths = node.get("active_input_paths")
            if (
                node.get("prim_path")
                != f"{source_prim_path}/Shader"
                or node.get("source_prim_path")
                != f"{source_prim_path}/Shader"
                or node.get("prim_type") != "Shader"
                or node.get("node_kind") != "Shader"
                or node.get("implementation_source") != "sourceAsset"
                or not isinstance(active_input_paths, list)
                or consumer_path not in active_input_paths
                or shade_input.get("name") != consumer_input_name
                or shade_input.get("full_name")
                != f"inputs:{consumer_input_name}"
                or shade_input.get("attribute_name")
                != f"inputs:{consumer_input_name}"
                or shade_input.get("is_active_property") is not True
                or shade_input.get("authored_value_is_active") is not True
                or shade_input.get("authored_value_suppressed_by_connection")
                is not False
                or shade_input.get("has_authored_value") is not True
                or shade_input.get("connections") != []
                or shade_input.get("invalid_connection_paths") != []
                or shade_input.get("ignored_invalid_connection_fallbacks")
                != [fallback]
                or shade_input.get("value")
                != fallback.get("consumer_authored_value")
                or shade_input.get("value_type")
                != fallback.get("consumer_value_type")
                or shade_input.get("value_might_be_time_varying") is not False
                or shade_input.get("authored_time_sample_count") != 0
                or shade_input.get("authored_time_samples") != []
            ):
                fallback_errors.append(
                    "active shader input does not attest the exact authored fallback value"
                )
        if fallback_errors:
            errors.append(
                {
                    "fallback_index": fallback_index,
                    "consumer_property_path": consumer_path,
                    "authored_invalid_source_property_path": source_path,
                    "errors": fallback_errors,
                }
            )

    if not errors:
        return []
    return [
        input_error(
            "ignored_invalid_connection_fallback_invalid",
            "a dangling material connection is not the exact audited OpenUSD authored-value fallback",
            material_record_id=material_record_id,
            entries=errors,
        )
    ]


def material_record_blockers(
    record_key: Any, material: Any
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not isinstance(material, Mapping):
        return [
            input_error(
                "material_record_invalid",
                "material table entry is not a mapping",
                material_record_id=record_key,
            )
        ]
    record_id = material.get("material_record_id")
    if record_key != record_id or not is_sha256(record_id):
        blockers.append(
            input_error(
                "material_record_identity_invalid",
                "material table key and canonical record ID are malformed or inconsistent",
                table_key=record_key,
                material_record_id=record_id,
            )
        )

    graph = material.get("active_material_graph")
    if not isinstance(graph, Mapping):
        blockers.append(
            input_error(
                "active_material_graph_invalid",
                "active material graph is missing or malformed",
                material_record_id=record_id,
            )
        )
        graph = {}

    blockers.extend(
        surface_base_color_resolution_blockers(record_id, material, graph)
    )
    blockers.extend(ignored_invalid_connection_fallback_blockers(material))

    required_list_fields = (
        "surface_texture_evidence",
        "texture_fallbacks",
        "source_repairs",
        "untrusted_mdl_source_candidates",
        "selected_mdl_material_semantics",
        "surface_texture_role_conflicts",
        "surface_texture_ambiguities",
        "surface_unknown_texture_roles",
        "unresolved_surface_textures",
        "annotation_warnings",
    )
    for field in required_list_fields:
        if as_record_list(material.get(field)) is None:
            blockers.append(
                input_error(
                    "material_evidence_field_invalid",
                    f"current material record field {field!r} is missing or is not a list",
                    material_record_id=record_id,
                    field=field,
                )
            )
    for field in (
        "selected_surface_textures",
        "selected_surface_texture_sets",
        "selected_surface_texture_evidence",
        "surface_texture_candidates",
    ):
        if not isinstance(material.get(field), Mapping):
            blockers.append(
                input_error(
                    "material_evidence_field_invalid",
                    f"current material record field {field!r} is missing or is not a mapping",
                    material_record_id=record_id,
                    field=field,
                )
            )

    udim_set_errors = selected_udim_texture_set_errors(material)
    if udim_set_errors:
        blockers.append(
            input_error(
                "selected_udim_texture_set_invalid",
                "a selected tiled texture is not an exact single-source, "
                "ordered, content-attested UDIM set",
                material_record_id=record_id,
                entry_count=len(udim_set_errors),
                entries=udim_set_errors,
            )
        )

    blockers.extend(
        selected_mdl_material_semantics_blockers(
            record_id, material, graph
        )
    )

    warning_field_by_fatal_code = {
        "unresolved_active_texture_evidence": "unresolved_surface_textures",
        "unknown_active_texture_role": "surface_unknown_texture_roles",
        "texture_path_fallback": "texture_fallbacks",
        "untrusted_mdl_source": "untrusted_mdl_source_candidates",
        "missing_connected_material_terminal": None,
        "unsupported_material_surface_terminal": None,
    }
    annotation_warnings = material.get("annotation_warnings")
    if isinstance(annotation_warnings, list):
        for warning in annotation_warnings:
            code = warning.get("code") if isinstance(warning, Mapping) else None
            if code not in warning_field_by_fatal_code:
                continue
            evidence_field = warning_field_by_fatal_code[code]
            if evidence_field is not None and material.get(evidence_field):
                # Report this below using the specific material field.
                continue
            blockers.append(
                input_error(
                    "fatal_material_warning_evidence",
                    "a reader warning identifies appearance evidence that cannot safely reach the VLM",
                    material_record_id=record_id,
                    warning_code=code,
                    warning=warning,
                )
            )

    fatal_fields = (
        (
            "unresolved_surface_textures",
            "unresolved_active_texture_evidence",
            "an active authored texture has no exact resolved source bytes",
        ),
        (
            "surface_unknown_texture_roles",
            "unknown_active_texture_role",
            "an active texture has no semantically proven role",
        ),
        (
            "texture_fallbacks",
            "texture_path_fallback",
            "an active texture path was recovered heuristically",
        ),
    )
    for field, code, message in fatal_fields:
        entries = material.get(field)
        if isinstance(entries, list) and entries:
            blockers.append(
                input_error(
                    code,
                    message,
                    material_record_id=record_id,
                    evidence_field=field,
                    entry_count=len(entries),
                    entries=entries,
                )
            )

    untrusted = material.get("untrusted_mdl_source_candidates")
    if isinstance(untrusted, list):
        unsafe = [
            candidate
            for candidate in untrusted
            if not isinstance(candidate, Mapping)
            or not untrusted_mdl_fully_superseded(material, candidate)
        ]
        if unsafe:
            blockers.append(
                input_error(
                    "untrusted_mdl_source_not_superseded",
                    "an active MDL source is unproven and has no exact same-role/path trusted replacement",
                    material_record_id=record_id,
                    entry_count=len(unsafe),
                    entries=unsafe,
                )
            )

    conflicts = material.get("surface_texture_role_conflicts")
    if isinstance(conflicts, list):
        unsafe_conflicts = [
            record
            for record in conflicts
            if not role_conflict_is_semantically_exact(record)
        ]
        if unsafe_conflicts:
            blockers.append(
                input_error(
                    "texture_role_conflict_prevents_exact_choice",
                    "a texture role conflict is not resolved by exact downstream consumer semantics",
                    material_record_id=record_id,
                    entry_count=len(unsafe_conflicts),
                    entries=unsafe_conflicts,
                )
            )

    ambiguities = material.get("surface_texture_ambiguities")
    if isinstance(ambiguities, list) and ambiguities:
        blockers.append(
            input_error(
                "active_texture_ambiguity_not_fully_represented",
                "an active texture role still has competing files or sources, "
                "or a tiled image set without an exact selected tile manifest",
                material_record_id=record_id,
                entry_count=len(ambiguities),
                entries=ambiguities,
            )
        )

    terminals = graph.get("terminals") if isinstance(graph, Mapping) else None
    custom = (
        graph.get("connected_nonstandard_outputs")
        if isinstance(graph, Mapping)
        else None
    )
    if not isinstance(terminals, list) or not isinstance(custom, list):
        blockers.append(
            input_error(
                "material_terminal_table_invalid",
                "active material terminal records are missing or malformed",
                material_record_id=record_id,
            )
        )
    else:
        has_surface = any(
            isinstance(terminal, Mapping)
            and str(terminal.get("terminal", "")).casefold() == "surface"
            for terminal in terminals
        )
        if not terminals and not custom:
            blockers.append(
                input_error(
                    "missing_connected_material_terminal",
                    "the bound material has no connected appearance terminal",
                    material_record_id=record_id,
                )
            )
        elif not has_surface:
            blockers.append(
                input_error(
                    "unsupported_material_surface_terminal",
                    "the bound material has no supported connected surface terminal",
                    material_record_id=record_id,
                )
            )
    return blockers


def segment_annotation_input_policy(
    extraction: Mapping[str, Any],
    segment_info: Any,
    *,
    include_stage: bool = True,
    include_material: bool = True,
) -> dict[str, Any]:
    """Check whether a segment can be sent to the VLM."""

    blockers = stage_color_management_blockers(extraction) if include_stage else []
    if not isinstance(segment_info, Mapping):
        blockers.append(
            input_error(
                "segment_record_invalid",
                "segment extraction record is not a mapping",
            )
        )
        return {
            "vlm_input_usable": False,
            "vlm_input_blockers": blockers,
            "material_record_id": None,
        }

    materials = extraction.get("materials")
    if not isinstance(materials, Mapping):
        blockers.append(
            input_error(
                "material_table_invalid",
                "extraction material table is missing or malformed",
            )
        )
        materials = {}
    material_id = segment_info.get("material_record_id")
    material_id_is_valid = material_id is None or is_sha256(material_id)
    if "material_record_id" not in segment_info or not material_id_is_valid:
        blockers.append(
            input_error(
                "segment_material_identity_invalid",
                "segment material_record_id is absent or malformed",
                material_record_id=material_id,
            )
        )
    material = (
        materials.get(material_id)
        if material_id is not None and material_id_is_valid
        else None
    )
    if material_id is not None and material_id_is_valid:
        if material is None:
            blockers.append(
                input_error(
                    "segment_material_record_missing",
                    "segment references a material record absent from the material table",
                    material_record_id=material_id,
                )
            )
        elif include_material:
            blockers.extend(material_record_blockers(material_id, material))

    return {
        "vlm_input_usable": not blockers,
        "vlm_input_blockers": blockers,
        "material_record_id": material_id,
    }


def asset_annotation_input_policy(
    extraction: Mapping[str, Any],
) -> dict[str, Any]:
    """Check whether an extracted asset has enough information for annotation."""

    exclusion = extraction_asset_exclusion(extraction)
    if exclusion is not None:
        return {
            "vlm_input_usable": False,
            "vlm_input_blockers": [exclusion],
            "segment_policies": {},
        }

    blockers = list(stage_color_management_blockers(extraction))
    materials = extraction.get("materials")
    if not isinstance(materials, Mapping):
        blockers.append(
            input_error(
                "material_table_invalid",
                "extraction material table is missing or malformed",
            )
        )
        materials = {}

    segments = extraction.get("segments")
    segment_policies: dict[str, dict[str, Any]] = {}
    if not isinstance(segments, Mapping) or not segments:
        blockers.append(
            input_error(
                "segment_table_invalid",
                "extraction has no non-empty segment mapping",
            )
        )
    else:
        # Check only materials used by the object's segments.
        referenced_material_ids: list[str] = []
        for segment_info in segments.values():
            material_id = (
                segment_info.get("material_record_id")
                if isinstance(segment_info, Mapping)
                else None
            )
            if is_sha256(material_id) and material_id not in referenced_material_ids:
                referenced_material_ids.append(material_id)
        for material_id in sorted(referenced_material_ids, key=str):
            material = materials.get(material_id)
            if material is not None:
                blockers.extend(material_record_blockers(material_id, material))
        for segment_key, segment_info in segments.items():
            policy = segment_annotation_input_policy(
                extraction,
                segment_info,
                include_stage=False,
                include_material=False,
            )
            segment_policies[str(segment_key)] = policy
            for blocker in policy["vlm_input_blockers"]:
                blockers.append({**blocker, "segment_key": str(segment_key)})

    return {
        "vlm_input_usable": not blockers,
        "vlm_input_blockers": blockers,
        "segment_policies": segment_policies,
    }


__all__ = (
    "BASE_COLOR_INPUTS",
    "EXCLUDED_SCENES",
    "asset_annotation_exclusion",
    "asset_annotation_input_policy",
    "compute_surface_base_color_value_resolution_sha256",
    "compute_stage_color_management_sha256",
    "ignored_invalid_connection_fallback_blockers",
    "segment_annotation_input_policy",
    "stage_color_management_blockers",
)
