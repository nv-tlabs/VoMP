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

"""Render textured objects and material segments with Blender Cycles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import stat
import subprocess
import sys
import traceback
from typing import Any, Mapping
import uuid

import numpy as np
from PIL import Image

from dataset_toolkits.material_objects.camera import _camera_frame

from dataset_toolkits.material_objects.vlm_annotations.segment_visual_inputs import (
    SegmentGeometryPayload,
    build_asset_geometry_payloads,
)
from dataset_toolkits.material_objects.vlm_annotations.annotation_input_policy import (
    ignored_invalid_connection_fallback_blockers,
)
from scipy.ndimage import distance_transform_edt
import bpy
from mathutils import Matrix, Vector


_BLENDER_VERSION = (4, 2)
_IMAGE_ROLES = (
    "textured_whole_object",
    "textured_selected_part",
    "selected_material_sphere",
)
_SUPPORTED_TEXTURE_ROLES = frozenset(
    {
        "albedo",
        "detail_normal",
        "displacement",
        "emissive",
        "emissive_mask",
        "coat_color",
        "coat_affect_color",
        "coat_affect_roughness",
        "coat_anisotropy",
        "coat_anisotropy_rotation",
        "coat_normal",
        "coat_roughness",
        "coat_weight",
        "diffuse_roughness",
        "diffuse_weight",
        "metallic",
        "normal",
        "normal_roughness",
        "occlusion",
        "opacity",
        "orm",
        "retroreflection_color",
        "retroreflection_roughness",
        "retroreflection_weight",
        "roughness",
        "specular",
        "specular_anisotropy",
        "specular_anisotropy_rotation",
        "specular_roughness",
        "specular_weight",
        "subsurface_scattering_color",
        "subsurface_anisotropy",
        "subsurface_scale",
        "subsurface_transmission_color",
        "subsurface_weight",
        "thin_film_thickness",
        "transmission_color",
        "transmission_depth",
        "transmission_scatter_anisotropy",
        "transmission_scattering_color",
        "transmission_weight",
        "displacement_scale",
        "emission_intensity",
        "emission_temperature",
        "emission_weight",
    }
)
_CHANNELS = frozenset({"r", "g", "b", "a", "rgb", "rgba"})
_GPU_BACKENDS = ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL")
_THREE_QUARTER_THICKNESS_RATIO = 0.08
_ORTHOGRAPHIC_FRAME_MARGIN = 1.4
_PERSPECTIVE_HINT_MINIMUM_HORIZONTAL_ALIGNMENT = 0.5


class TexturedRenderError(RuntimeError):
    """The exact textured VLM render cannot be produced safely."""


class _RenderVisibilityError(TexturedRenderError):
    def __init__(self, visible_pixels: int) -> None:
        self.visible_pixels = visible_pixels
        super().__init__(
            f"render visibility check found only {visible_pixels} non-background pixels"
        )


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(label: str, value: Any) -> str:
    encoded = json.dumps(
        value,
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


def _reader_contract_hash(label: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(encoded)
    digest.update(b"\0")
    return digest.hexdigest()


def _array_hash(label: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _normalized_name(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:96] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _finite_color(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) not in {3, 4}:
        return None
    numbers = [_finite_scalar(item) for item in value]
    if any(item is None for item in numbers):
        return None
    return (
        float(numbers[0]),
        float(numbers[1]),
        float(numbers[2]),
        float(numbers[3]) if len(numbers) == 4 else 1.0,
    )


def _json_value_hash(value: Any) -> str:
    return _canonical_hash("textured-render-provenance", value)


@dataclass(frozen=True)
class UVBinding:
    name: str
    scale: tuple[float, float] = (1.0, 1.0)
    rotation_degrees: float = 0.0
    translation: tuple[float, float] = (0.0, 0.0)
    transform_semantics: str = "identity"
    resolution: str = "exact_authored_primvar"
    coordinate_source: str = "uv"
    projection: str = "FLAT"

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scale": list(self.scale),
            "rotation_degrees": self.rotation_degrees,
            "translation": list(self.translation),
            "transform_semantics": self.transform_semantics,
            "resolution": self.resolution,
            "coordinate_source": self.coordinate_source,
            "projection": self.projection,
        }


def _uv_affine(
    binding: UVBinding,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    scale_u, scale_v = binding.scale
    translate_u, translate_v = binding.translation
    radians = math.radians(binding.rotation_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    if binding.transform_semantics == "identity":
        if (
            binding.scale != (1.0, 1.0)
            or binding.rotation_degrees != 0.0
            or binding.translation != (0.0, 0.0)
        ):
            raise TexturedRenderError(
                "non-identity texture controls lack an exact transform semantic"
            )
        return (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)
    if binding.transform_semantics == "usd_transform2d":
        return (
            (cosine * scale_u, -sine * scale_v),
            (sine * scale_u, cosine * scale_v),
            (translate_u, translate_v),
        )
    if binding.transform_semantics == "mdl_local_scale_rotate":
        return (
            (scale_u * cosine, scale_u * sine),
            (-scale_v * sine, scale_v * cosine),
            (translate_u, translate_v),
        )
    if binding.transform_semantics == "mdl_base_rotation_translation_scale":
        return (
            (cosine * scale_u, sine * scale_v),
            (-sine * scale_u, cosine * scale_v),
            (
                cosine * (translate_u - 0.5)
                + sine * (translate_v - 0.5)
                + 0.5,
                -sine * (translate_u - 0.5)
                + cosine * (translate_v - 0.5)
                + 0.5,
            ),
        )
    if binding.transform_semantics == "omnisurface_uv":
        if binding.rotation_degrees != 0.0:
            raise TexturedRenderError("OmniSurface UV controls cannot rotate")
        return (
            (scale_u, 0.0),
            (0.0, scale_v),
            (scale_u * translate_u, scale_v * translate_v),
        )
    raise TexturedRenderError(
        f"unsupported exact texture transform semantic "
        f"{binding.transform_semantics!r}"
    )


@dataclass(frozen=True)
class TextureTile:
    tile_id: int
    path: str
    file_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "resolved_path": self.path,
            "file_sha256": self.file_sha256,
        }


def _udim_grid_bounds(
    role: str, tiles: tuple[TextureTile, ...]
) -> tuple[int, int, int, int]:
    coordinates = {
        ((tile.tile_id - 1001) % 10, (tile.tile_id - 1001) // 10)
        for tile in tiles
    }
    minimum_u = min(item[0] for item in coordinates)
    maximum_u = max(item[0] for item in coordinates)
    minimum_v = min(item[1] for item in coordinates)
    maximum_v = max(item[1] for item in coordinates)
    expected = {
        (u, v)
        for v in range(minimum_v, maximum_v + 1)
        for u in range(minimum_u, maximum_u + 1)
    }
    if coordinates != expected:
        raise TexturedRenderError(
            f"UDIM role {role!r} does not form a complete rectangular tile grid"
        )
    return minimum_u, maximum_u, minimum_v, maximum_v


def _udim_geometry_sampling(
    role: str,
    tiles: tuple[TextureTile, ...],
    binding: UVBinding,
    payload: SegmentGeometryPayload,
) -> dict[str, Any]:
    if binding.coordinate_source != "uv" or binding.projection != "FLAT":
        raise TexturedRenderError(
            f"UDIM role {role!r} does not use an exact mesh UV coordinate set"
        )
    attribute = payload.uv_sets.get(binding.name)
    if attribute is None:
        raise TexturedRenderError(
            f"UDIM role {role!r} references unavailable UV set {binding.name!r}"
        )
    values = np.asarray(attribute.corner_values, dtype=np.float64)
    if (
        values.shape != (len(payload.face_vertex_indices), 2)
        or not np.isfinite(values).all()
    ):
        raise TexturedRenderError(f"UDIM UV values are invalid for role {role!r}")
    row_u, row_v, translation = _uv_affine(binding)
    transformed = np.empty_like(values)
    transformed[:, 0] = (
        row_u[0] * values[:, 0]
        + row_u[1] * values[:, 1]
        + translation[0]
    )
    transformed[:, 1] = (
        row_v[0] * values[:, 0]
        + row_v[1] * values[:, 1]
        + translation[1]
    )
    if not np.isfinite(transformed).all():
        raise TexturedRenderError(
            f"UDIM transformed UV values are invalid for role {role!r}"
        )
    minimum_u, maximum_u, minimum_v, maximum_v = _udim_grid_bounds(role, tiles)
    lower = np.asarray((minimum_u, minimum_v), dtype=np.float64)
    upper = np.asarray((maximum_u + 1, maximum_v + 1), dtype=np.float64)
    observed_minimum = transformed.min(axis=0)
    observed_maximum = transformed.max(axis=0)
    if np.any(observed_minimum < lower) or np.any(observed_maximum > upper):
        raise TexturedRenderError(
            f"UDIM role {role!r} samples outside its resolved tile grid: "
            f"uv_min={observed_minimum.tolist()}, uv_max={observed_maximum.tolist()}, "
            f"grid_min={lower.tolist()}, grid_max={upper.tolist()}"
        )
    referenced: set[int] = set()
    offset = 0
    for count_value in payload.face_vertex_counts:
        count = int(count_value)
        face_values = transformed[offset : offset + count]
        offset += count
        face_minimum = face_values.min(axis=0)
        face_maximum = face_values.max(axis=0)
        ranges: list[range] = []
        for axis, grid_minimum, grid_maximum in (
            (0, minimum_u, maximum_u),
            (1, minimum_v, maximum_v),
        ):
            low = float(face_minimum[axis])
            high = float(face_maximum[axis])
            start = math.floor(low)
            end = (
                math.floor(high)
                if high == low
                else math.floor(math.nextafter(high, -math.inf))
            )
            start = min(max(start, grid_minimum), grid_maximum)
            end = min(max(end, grid_minimum), grid_maximum)
            ranges.append(range(min(start, end), max(start, end) + 1))
        for v in ranges[1]:
            for u in ranges[0]:
                referenced.add(1001 + u + 10 * v)
    if offset != len(transformed):
        raise TexturedRenderError(f"UDIM topology offsets disagree for role {role!r}")
    available = {tile.tile_id for tile in tiles}
    if not referenced.issubset(available):
        raise TexturedRenderError(
            f"UDIM role {role!r} references missing tiles: "
            f"{sorted(referenced - available)!r}"
        )
    return {
        "uv_set": binding.name,
        "authored_transform": {
            "scale": list(binding.scale),
            "rotation_degrees": binding.rotation_degrees,
            "translation": list(binding.translation),
        },
        "transformed_uv_minimum": observed_minimum.tolist(),
        "transformed_uv_maximum": observed_maximum.tolist(),
        "referenced_tile_ids": sorted(referenced),
        "available_tile_ids": sorted(available),
    }


@dataclass(frozen=True)
class _TextureSource:
    path: str
    file_sha256: str
    tiles: tuple[TextureTile, ...]


@dataclass(frozen=True)
class TexturePlan:
    role: str
    path: str
    file_sha256: str
    tiles: tuple[TextureTile, ...]
    channel: str
    color_space: str
    uv: UVBinding
    wrap_mode: str
    scale: tuple[float, float, float, float]
    bias: tuple[float, float, float, float]
    mono_mode: str | None
    flip_red: bool
    flip_green: bool
    flip_blue: bool
    normal_decode_semantics: str | None
    evidence_sha256: str
    geometry_udim_sampling: Mapping[str, Any] | None

    def record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "texture_type": "udim" if self.tiles else "single_file",
            "resolved_path": self.path,
            "file_sha256": self.file_sha256,
            "tiles": [tile.record() for tile in self.tiles],
            "source_channel": self.channel,
            "color_space": self.color_space,
            "uv": self.uv.record(),
            "wrap_mode": self.wrap_mode,
            "scale": list(self.scale),
            "bias": list(self.bias),
            "mono_mode": self.mono_mode,
            "flip_red": self.flip_red,
            "flip_green": self.flip_green,
            "flip_blue": self.flip_blue,
            "normal_decode_semantics": self.normal_decode_semantics,
            "evidence_sha256": self.evidence_sha256,
            "geometry_udim_sampling": self.geometry_udim_sampling,
        }


@dataclass(frozen=True)
class _BakedSwatch:
    role: str
    kind: str
    image: Any
    width: int
    height: int
    source_evidence_sha256: str
    pixels_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "source_evidence_sha256": self.source_evidence_sha256,
            "pixels_sha256": self.pixels_sha256,
        }


@dataclass(frozen=True)
class MaterialPlan:
    material_record_id: str | None
    material_name: str
    material_model: str
    model_parameters: Mapping[str, Any]
    textures: tuple[TexturePlan, ...]
    base_color: tuple[float, float, float, float]
    roughness: float
    metallic: float
    opacity: float
    opacity_threshold: float
    emission_color: tuple[float, float, float, float]
    emission_strength: float
    double_sided: bool
    display_color_attribute: str | None
    display_opacity_attribute: str | None
    detail_normal_strength: float
    bump_strength: float
    bump_distance: float
    ior: float
    specular_ior_level: float
    transmission_weight: float
    coat_weight: float
    coat_roughness: float
    coat_ior: float
    coat_tint: tuple[float, float, float, float]
    sheen_weight: float
    sheen_roughness: float
    sheen_tint: tuple[float, float, float, float]
    anisotropic: float
    anisotropic_rotation: float
    subsurface_weight: float
    subsurface_radius: tuple[float, float, float]
    subsurface_scale: float
    subsurface_ior: float
    subsurface_anisotropy: float
    thin_film_thickness: float
    thin_film_ior: float
    thin_walled: bool
    albedo_desaturation: float
    albedo_add: float
    albedo_brightness: float
    diffuse_tint: tuple[float, float, float, float]
    ao_to_diffuse: float
    roughness_texture_influence: float
    metallic_texture_influence: float
    normal_strength: float
    omnipbr_controls: bool
    simpbr_controls: bool
    simpbr_orm_gate_repair: dict[str, Any] | None
    provenance_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "material_record_id": self.material_record_id,
            "material_name": self.material_name,
            "material_model": self.material_model,
            "model_parameters": dict(self.model_parameters),
            "textures": [item.record() for item in self.textures],
            "base_color": list(self.base_color),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "opacity": self.opacity,
            "opacity_threshold": self.opacity_threshold,
            "emission_color": list(self.emission_color),
            "emission_strength": self.emission_strength,
            "double_sided": self.double_sided,
            "display_color_attribute": self.display_color_attribute,
            "display_opacity_attribute": self.display_opacity_attribute,
            "detail_normal_strength": self.detail_normal_strength,
            "bump_strength": self.bump_strength,
            "bump_distance": self.bump_distance,
            "ior": self.ior,
            "specular_ior_level": self.specular_ior_level,
            "transmission_weight": self.transmission_weight,
            "coat_weight": self.coat_weight,
            "coat_roughness": self.coat_roughness,
            "coat_ior": self.coat_ior,
            "coat_tint": list(self.coat_tint),
            "sheen_weight": self.sheen_weight,
            "sheen_roughness": self.sheen_roughness,
            "sheen_tint": list(self.sheen_tint),
            "anisotropic": self.anisotropic,
            "anisotropic_rotation": self.anisotropic_rotation,
            "subsurface_weight": self.subsurface_weight,
            "subsurface_radius": list(self.subsurface_radius),
            "subsurface_scale": self.subsurface_scale,
            "subsurface_ior": self.subsurface_ior,
            "subsurface_anisotropy": self.subsurface_anisotropy,
            "thin_film_thickness": self.thin_film_thickness,
            "thin_film_ior": self.thin_film_ior,
            "thin_walled": self.thin_walled,
            "albedo_desaturation": self.albedo_desaturation,
            "albedo_add": self.albedo_add,
            "albedo_brightness": self.albedo_brightness,
            "diffuse_tint": list(self.diffuse_tint),
            "ao_to_diffuse": self.ao_to_diffuse,
            "roughness_texture_influence": self.roughness_texture_influence,
            "metallic_texture_influence": self.metallic_texture_influence,
            "normal_strength": self.normal_strength,
            "omnipbr_controls": self.omnipbr_controls,
            "simpbr_controls": self.simpbr_controls,
            "simpbr_orm_gate_repair": self.simpbr_orm_gate_repair,
            "provenance_sha256": self.provenance_sha256,
        }


def _uv_aliases(plan: MaterialPlan) -> dict[str, str]:
    names = sorted(
        {
            texture.uv.name
            for texture in plan.textures
            if texture.uv.coordinate_source == "uv"
        }
    )
    if len(names) > 8:
        raise TexturedRenderError(
            f"material {plan.material_name!r} requires {len(names)} UV sets; "
            "Blender 4.2 can expose at most eight"
        )
    return {name: f"vlm_uv_{index}" for index, name in enumerate(names)}


def _graph_nodes(material_record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    graph = material_record.get("active_material_graph")
    if not isinstance(graph, Mapping):
        raise TexturedRenderError("material record has no active material graph")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise TexturedRenderError("active material graph has no node list")
    result: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, Mapping) or not node.get("prim_path"):
            raise TexturedRenderError("active material graph contains an invalid node record")
        result[str(node["prim_path"])] = node
    return result


def _node_inputs(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    inputs = node.get("inputs")
    return [item for item in inputs if isinstance(item, Mapping)] if isinstance(inputs, list) else []


def _selected_surface_root_uses(
    material_record: Mapping[str, Any],
) -> dict[str, set[str]]:
    graph = material_record.get("active_material_graph")
    if not isinstance(graph, Mapping):
        return {}
    terminals = [
        item
        for item in graph.get("terminals") or []
        if isinstance(item, Mapping)
        and str(item.get("terminal") or "").casefold() == "surface"
    ]
    mdl_terminals = [
        item
        for item in terminals
        if "mdl" in _normalized_name(item.get("render_context"))
    ]
    selected_terminals = mdl_terminals or terminals
    nodes = _graph_nodes(material_record)
    root_uses: dict[str, set[str]] = {}
    for terminal in selected_terminals:
        render_context = str(terminal.get("render_context") or "")
        output = terminal.get("material_output")
        for connection in output.get("connections") or [] if isinstance(output, Mapping) else []:
            if not isinstance(connection, Mapping):
                continue
            prim_path = str(connection.get("source_prim_path") or "")
            if not prim_path:
                prim_path = str(connection.get("source_property_path") or "").rsplit(
                    ".", 1
                )[0]
            if prim_path in nodes:
                root_uses.setdefault(prim_path, set()).add(render_context)
    return root_uses


def _reader_canonical_hash(label: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(encoded)
    digest.update(b"\0")
    return digest.hexdigest()


def _validated_dependency_attestation(record: Mapping[str, Any]) -> dict[str, Any] | None:
    attestation = record.get("source_dependency_attestation")
    if attestation is None:
        return None
    if not isinstance(attestation, Mapping):
        raise TexturedRenderError("selected MDL dependency attestation is malformed")
    classification = attestation.get("classification")
    modules = attestation.get("modules")
    if classification != "content_addressed_mdl_import_closure" or not isinstance(
        modules, list
    ) or not modules:
        raise TexturedRenderError("selected MDL dependency closure is incomplete")

    validated: list[dict[str, str]] = []
    for module in modules:
        if not isinstance(module, Mapping):
            raise TexturedRenderError("selected MDL dependency module is malformed")
        authored_module = module.get("authored_module")
        relative_path = module.get("relative_path")
        resolved_source_path = module.get("resolved_source_path")
        source_sha256 = module.get("source_sha256")
        relative = Path(str(relative_path or ""))
        resolved = Path(str(resolved_source_path or ""))
        if (
            not isinstance(authored_module, str)
            or not authored_module
            or not isinstance(relative_path, str)
            or not relative_path
            or relative.suffix.casefold() != ".mdl"
            or relative.is_absolute()
            or ".." in relative.parts
            or not resolved.is_absolute()
            or not resolved.is_file()
            or not isinstance(source_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
            or _sha256_file(resolved) != source_sha256
        ):
            raise TexturedRenderError(
                f"selected MDL dependency attestation no longer matches {resolved}"
            )
        validated.append(
            {
                "authored_module": authored_module,
                "relative_path": relative_path,
                "resolved_source_path": str(resolved),
                "source_sha256": source_sha256,
            }
        )

    sort_key = lambda item: (
        item["authored_module"],
        item["relative_path"],
        item["source_sha256"],
    )
    if validated != sorted(validated, key=sort_key) or len(
        {(item["relative_path"], item["source_sha256"]) for item in validated}
    ) != len(validated):
        raise TexturedRenderError(
            "selected MDL dependency closure is not sorted and unique"
        )
    canonical = {
        "classification": classification,
        "modules": [
            {
                "authored_module": item["authored_module"],
                "relative_path": item["relative_path"],
                "source_sha256": item["source_sha256"],
            }
            for item in validated
        ],
    }
    if _reader_canonical_hash(
        "selected-mdl-source-dependency-attestation", canonical
    ) != attestation.get("attestation_sha256"):
        raise TexturedRenderError("selected MDL dependency attestation hash mismatch")
    return {
        "classification": classification,
        "modules": validated,
        "attestation_sha256": attestation.get("attestation_sha256"),
    }


def _attested_mdl_material_defaults(
    material_record: Mapping[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, str], list[dict[str, Any]]]:
    records = material_record.get("selected_mdl_material_semantics")
    if records is None:
        return {}, {}, []
    if not isinstance(records, list):
        raise TexturedRenderError("selected MDL material semantics are malformed")

    root_uses = _selected_surface_root_uses(material_record)
    defaults: dict[str, list[Any]] = {}
    root_families: dict[str, str] = {}
    provenance: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TexturedRenderError("selected MDL material semantics contain a malformed record")
        shader_path = str(record.get("shader_path") or "")
        if shader_path not in root_uses:
            raise TexturedRenderError(
                f"selected MDL attestation targets non-root shader {shader_path!r}"
            )
        contexts = record.get("render_contexts")
        if not isinstance(contexts, list) or not contexts or not all(
            isinstance(item, str) for item in contexts
        ):
            raise TexturedRenderError(
                f"selected MDL semantics for {shader_path!r} have invalid render contexts"
            )
        selected_contexts = sorted(root_uses[shader_path].intersection(contexts))
        if not selected_contexts:
            continue
        if shader_path in seen_roots:
            raise TexturedRenderError(
                f"selected surface root {shader_path!r} has multiple MDL semantic attestations"
        )
        seen_roots.add(shader_path)
        family = record.get("material_family")
        expression_kind = record.get("definition_expression_kind")
        direct_callee = record.get("direct_callee")
        selected_definition = record.get("selected_material_definition")
        is_installed = (
            expression_kind == "hash_attested_installed_material_implementation"
            and direct_callee is None
            and record.get("terminal_material_callee") is None
            and record.get("terminal_material_module") is None
        )
        is_direct_omnipbr = (
            family == "OmniPBR"
            and expression_kind == "sole_direct_call"
            and direct_callee == "::OmniPBR::OmniPBR"
            and record.get("terminal_material_callee") is None
            and record.get("terminal_material_module") is None
        )
        is_installed_omnipbr = (
            family == "OmniPBR"
            and is_installed
            and (
                (
                    selected_definition == "OmniPBR"
                    and record.get("authored_source_asset") == "OmniPBR.mdl"
                    and record.get("source_asset_sub_identifier") == "OmniPBR"
                )
                or (
                    selected_definition == "OmniPBR_Opacity"
                    and record.get("authored_source_asset")
                    == "OmniPBR_Opacity.mdl"
                    and record.get("source_asset_sub_identifier")
                    == "OmniPBR_Opacity"
                )
            )
            and record.get("source_resolution_method")
            == "pinned_installed_isaac_omni_mdl_module"
        )
        is_omnipbr = is_direct_omnipbr or is_installed_omnipbr
        is_simpbr = (
            family == "SimPBR"
            and expression_kind == "hash_attested_let_terminal_call"
            and selected_definition == "SimPBR"
            and direct_callee is None
            and record.get("terminal_material_callee")
            == ".::SimPBR_Model::SimPBR_Model"
            and isinstance(record.get("terminal_material_module"), Mapping)
            and record["terminal_material_module"].get("authored_module")
            == ".::SimPBR_Model"
        )
        is_simpbr_translucent = (
            family == "SimPBRTranslucent"
            and expression_kind
            == "hash_attested_simpbr_translucent_implementation"
            and selected_definition == "SimPBR_Translucent"
            and Path(str(record.get("authored_source_asset") or "")).name
            == "SimPBR_Translucent.mdl"
            and record.get("source_asset_sub_identifier")
            == "SimPBR_Translucent"
            and record.get("source_sha256")
            == "6829af54259b898b4237c02d9375067b63f802cbbbde6f9935f2e69a1b11e4bf"
            and direct_callee is None
            and record.get("terminal_material_callee") is None
            and record.get("terminal_material_module") is None
        )
        is_local_implementation = (
            expression_kind == "hash_attested_local_material_implementation"
            and direct_callee is None
            and record.get("terminal_material_callee") is None
            and record.get("terminal_material_module") is None
        )
        is_composed = (
            expression_kind == "hash_attested_direct_call_with_composed_defaults"
            and record.get("terminal_material_callee") is None
            and record.get("terminal_material_module") is None
        )
        is_asset_local_omnipbr = (
            family == "OmniPBR"
            and is_local_implementation
            and selected_definition == "OmniPBR_Opacity"
            and Path(str(record.get("authored_source_asset") or "")).name
            == "OmniPBR_Opacity.mdl"
            and record.get("source_asset_sub_identifier") == "OmniPBR_Opacity"
            and record.get("source_resolution_method")
            == "hash_attested_asset_local_omni_mdl_module"
            and record.get("source_sha256")
            == "fbd3447652dbbaa2169c6e2f3887c3f3a7133800b41a40d7748d454ad357a3ce"
        )
        is_omnipbr = is_omnipbr or is_asset_local_omnipbr
        is_omniglass = family == "OmniGlass" and (
            (
                is_installed
                and selected_definition == "OmniGlass"
                and record.get("authored_source_asset") == "OmniGlass.mdl"
                and record.get("source_asset_sub_identifier") == "OmniGlass"
                and record.get("source_resolution_method")
                == "pinned_installed_isaac_omni_mdl_module"
            )
            or (is_composed and direct_callee == "::OmniGlass::OmniGlass")
        )
        is_omnisurface = family == "OmniSurface" and (
            (
                is_installed
                and selected_definition == "OmniSurface"
                and record.get("authored_source_asset") == "OmniSurface.mdl"
                and record.get("source_asset_sub_identifier") == "OmniSurface"
                and record.get("source_resolution_method")
                == "pinned_installed_isaac_omni_mdl_module"
            )
            or (
                is_installed
                and selected_definition == "OmniSurfaceLite"
                and record.get("authored_source_asset") == "OmniSurfaceLite.mdl"
                and record.get("source_asset_sub_identifier") == "OmniSurfaceLite"
                and record.get("source_resolution_method")
                == "pinned_installed_isaac_omni_mdl_module"
            )
            or (
                is_local_implementation
                and selected_definition == "OmniSurface"
                and Path(str(record.get("authored_source_asset") or "")).name
                == "OmniSurface.mdl"
                and record.get("source_asset_sub_identifier") == "OmniSurface"
                and record.get("source_resolution_method")
                == "hash_attested_asset_local_omnisurface_module"
            )
            or (
                is_local_implementation
                and selected_definition == "OmniSurfaceLite"
                and Path(str(record.get("authored_source_asset") or "")).name
                == "OmniSurfaceLite.mdl"
                and record.get("source_asset_sub_identifier") == "OmniSurfaceLite"
                and record.get("source_resolution_method")
                == "hash_attested_asset_local_omnisurface_module"
            )
            or (
                is_composed
                and direct_callee
                in {
                    "::OmniSurface::OmniSurface",
                    "::OmniSurfaceLite::OmniSurfaceLite",
                }
            )
        )
        is_omnisurface_preset = (
            family == "OmniSurfacePreset"
            and is_composed
            and direct_callee
            in {
                "::OmniSurface::OmniSurface",
                "::OmniSurfaceLite::OmniSurfaceLite",
            }
            and record.get("source_asset_sub_identifier")
            in {
                "OmniSurface_FrostedGlass",
                "OmniSurface_Velvet",
                "OmniSurface_Wax",
                "OmniSurface_ClearWater",
            }
            and (
                (
                    record.get("authored_source_asset")
                    == "OmniSurfacePresets.mdl"
                    and record.get("source_resolution_method")
                    == "pinned_installed_isaac_omni_mdl_module"
                )
                or (
                    Path(str(record.get("authored_source_asset") or "")).name
                    == "OmniSurfacePresets.mdl"
                    and record.get("source_resolution_method")
                    == "hash_attested_asset_local_omnisurface_module"
                )
            )
        )
        is_nvidia_glass = (
            family == "NvidiaGlassWithVolume"
            and is_composed
            and direct_callee
            == "..::Templates::GlassWithVolume::GlassWithVolume"
        )
        exact_extended_family = any(
            (
                is_omniglass,
                is_omnisurface,
                is_omnisurface_preset,
                is_nvidia_glass,
            )
        )
        if not (
            is_omnipbr
            or is_simpbr
            or is_simpbr_translucent
            or exact_extended_family
        ):
            raise TexturedRenderError(
                f"selected MDL surface root {shader_path!r} has no supported exact family"
            )
        root_families[shader_path] = (
            "omnipbr"
            if is_omnipbr
            else "simpbr"
            if is_simpbr
            else "simpbrtranslucent"
            if is_simpbr_translucent
            else "omniglass"
            if is_omniglass
            else "nvidiaglasswithvolume"
            if is_nvidia_glass
            else "omnisurfacepreset"
            if is_omnisurface_preset
            else "omnisurface"
        )
        info = _graph_nodes(material_record)[shader_path].get("info_attributes")
        if not isinstance(info, Mapping):
            raise TexturedRenderError(
                f"selected MDL surface root {shader_path!r} has no exact info attributes"
            )
        source_records = [
            item
            for key, item in info.items()
            if _normalized_name(key) == "infomdlsourceasset"
            and isinstance(item, Mapping)
        ]
        source_matches = [
            asset
            for source_record in source_records
            for asset in source_record.get("asset_values") or []
            if isinstance(asset, Mapping)
            and asset.get("authored_path") == record.get("authored_source_asset")
            and (
                exact_extended_family
                or is_installed_omnipbr
                or is_asset_local_omnipbr
                or (
                    record.get("resolved_source_path")
                    in (asset.get("resolved_paths") or [])
                    and asset.get("resolution_method")
                    == record.get("source_resolution_method")
                )
            )
        ]
        sub_identifier_records = [
            item
            for key, item in info.items()
            if _normalized_name(key) == "infomdlsourceassetsubidentifier"
            and isinstance(item, Mapping)
        ]
        sub_identifier = (
            sub_identifier_records[0].get("value")
            if len(sub_identifier_records) == 1
            else None
        )
        if (
            len(source_records) != 1
            or len(source_matches) != 1
            or len(sub_identifier_records) > 1
            or sub_identifier != record.get("source_asset_sub_identifier")
        ):
            raise TexturedRenderError(
                f"selected MDL attestation does not bind to active root {shader_path!r}"
            )
        dependency_attestation = _validated_dependency_attestation(record)
        if (
            is_simpbr
            or is_simpbr_translucent
            or is_installed_omnipbr
            or is_asset_local_omnipbr
            or exact_extended_family
        ) and dependency_attestation is None:
            raise TexturedRenderError(
                f"selected MDL surface root {shader_path!r} has no dependency closure"
            )
        parameter_defaults = record.get("parameter_defaults")
        if not isinstance(parameter_defaults, Mapping):
            raise TexturedRenderError(
                f"selected MDL semantics for {shader_path!r} have no parameter defaults"
            )
        terminal_material_module = record.get("terminal_material_module")
        canonical_terminal_module = (
            {
                "authored_module": terminal_material_module.get(
                    "authored_module"
                ),
                "source_sha256": terminal_material_module.get("source_sha256"),
            }
            if isinstance(terminal_material_module, Mapping)
            else None
        )
        canonical_semantics = {
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
            "terminal_material_module": canonical_terminal_module,
            "parameter_defaults": parameter_defaults,
        }
        if (
            is_installed_omnipbr
            or is_asset_local_omnipbr
            or is_simpbr_translucent
            or exact_extended_family
        ):
            canonical_semantics["source_dependency_attestation_sha256"] = (
                dependency_attestation["attestation_sha256"]
            )
        encoded = json.dumps(
            canonical_semantics,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(b"selected-mdl-material-semantics\0")
        digest.update(encoded)
        digest.update(b"\0")
        if digest.hexdigest() != record.get("semantics_sha256"):
            raise TexturedRenderError(
                f"selected MDL semantics hash mismatch for {shader_path!r}"
            )
        source_path = Path(str(record.get("resolved_source_path") or ""))
        source_sha256 = str(record.get("source_sha256") or "")
        if (
            not source_path.is_absolute()
            or not source_path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
            or _sha256_file(source_path) != source_sha256
        ):
            raise TexturedRenderError(
                f"selected MDL source attestation no longer matches {source_path}"
            )
        terminal_module = record.get("terminal_material_module")
        if isinstance(terminal_module, Mapping):
            terminal_path = Path(
                str(terminal_module.get("resolved_source_path") or "")
            )
            terminal_sha256 = str(terminal_module.get("source_sha256") or "")
            if (
                not terminal_path.is_absolute()
                or not terminal_path.is_file()
                or not re.fullmatch(r"[0-9a-f]{64}", terminal_sha256)
                or _sha256_file(terminal_path) != terminal_sha256
            ):
                raise TexturedRenderError(
                    f"selected MDL terminal-module attestation no longer matches {terminal_path}"
                )

        for input_name, value_record in parameter_defaults.items():
            if not isinstance(input_name, str) or not isinstance(value_record, Mapping):
                raise TexturedRenderError(
                    f"selected MDL parameter defaults for {shader_path!r} are malformed"
                )
            value_type = value_record.get("value_type")
            value = value_record.get("value")
            if value_type == "texture_2d":
                continue
            if value_type == "bool":
                valid = isinstance(value, bool)
            elif value_type == "float":
                valid = _finite_scalar(value) is not None
            elif value_type == "int":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif value_type == "color":
                valid = _finite_color(value) is not None and len(value) == 3
            elif value_type == "float2":
                valid = (
                    isinstance(value, list)
                    and len(value) == 2
                    and all(_finite_scalar(item) is not None for item in value)
                )
            elif value_type == "float3":
                valid = (
                    isinstance(value, list)
                    and len(value) == 3
                    and all(_finite_scalar(item) is not None for item in value)
                )
            elif value_type == "int2":
                valid = (
                    isinstance(value, list)
                    and len(value) == 2
                    and all(
                        isinstance(item, int) and not isinstance(item, bool)
                        for item in value
                    )
                )
            elif value_type == "state_normal":
                valid = value == "::state::normal()"
            elif value_type == "qualified_enum":
                valid = isinstance(value, str) and bool(
                    re.fullmatch(r"::[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+", value)
                )
                if valid:
                    value = value.rsplit("::", 1)[-1]
            else:
                valid = False
            if not valid:
                raise TexturedRenderError(
                    f"selected MDL parameter {input_name!r} for {shader_path!r} "
                    f"has invalid {value_type!r} data"
                )
            if value_type != "state_normal":
                defaults.setdefault(_normalized_name(input_name), []).append(value)

        provenance.append(
            {
                "shader_path": shader_path,
                "render_contexts": selected_contexts,
                "resolved_source_path": str(source_path),
                "source_sha256": source_sha256,
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
                "terminal_material_module": terminal_module,
                "source_dependency_attestation": dependency_attestation,
                "semantics_sha256": record.get("semantics_sha256"),
            }
        )
    return (
        defaults,
        root_families,
        sorted(provenance, key=lambda item: item["shader_path"]),
    )


def _surface_root_material_family(
    material_record: Mapping[str, Any],
    attested_mdl_root_families: Mapping[str, str] | None = None,
) -> str:
    root_uses = _selected_surface_root_uses(material_record)
    if not root_uses:
        return "unknown"

    families: set[str] = set()
    nodes = _graph_nodes(material_record)
    for prim_path in root_uses:
        node = nodes[prim_path]
        shader_id = _normalized_name(node.get("shader_id"))
        source_asset = None
        sub_identifier = None
        info = node.get("info_attributes")
        if isinstance(info, Mapping):
            for key, record in info.items():
                if not isinstance(record, Mapping):
                    continue
                normalized_key = _normalized_name(key)
                value = record.get("value")
                if normalized_key.endswith("sourceasset") and isinstance(
                    value, Mapping
                ):
                    source_asset = value.get("authored_path") or value.get(
                        "usd_resolved_path"
                    )
                elif normalized_key.endswith("sourceassetsubidentifier"):
                    sub_identifier = value
        source_name = _normalized_name(Path(str(source_asset or "")).name)
        sub_name = _normalized_name(sub_identifier)
        if shader_id == "usdpreviewsurface":
            families.add("usd_preview_surface")
        elif shader_id in {"omnipbr", "omnipbrclearcoat"}:
            families.add(shader_id)
        elif source_name == "omnipbrmdl" and sub_name in {
            "omnipbr",
            "omnipbrclearcoat",
        }:
            # Runtime OmniPBR's exact module/subIdentifier identity predates
            # the local-wrapper attestation below and remains a separate path.
            families.add(sub_name)
        elif prim_path in (attested_mdl_root_families or {}):
            families.add(str(attested_mdl_root_families[prim_path]))
        else:
            families.add(f"other:{prim_path}:{shader_id}")
    if len(families) != 1:
        raise TexturedRenderError(
            f"selected surface terminals have mixed root material families: {sorted(families)!r}"
        )
    return next(iter(families))


def _input_by_name(node: Mapping[str, Any], names: set[str]) -> Mapping[str, Any] | None:
    matches = [
        item
        for item in _node_inputs(node)
        if _normalized_name(item.get("name") or item.get("full_name")) in names
    ]
    if len(matches) > 1:
        raise TexturedRenderError(
            f"material node {node.get('prim_path')} has multiple inputs for {sorted(names)}"
        )
    return matches[0] if matches else None


def _connection_source_node(
    input_record: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    connections = input_record.get("connections")
    if not isinstance(connections, list) or not connections:
        return None
    if len(connections) != 1 or not isinstance(connections[0], Mapping):
        raise TexturedRenderError(
            f"UV input {input_record.get('property_path')} has multiple/invalid connections"
        )
    connection = connections[0]
    source_path = str(connection.get("source_property_path") or "")
    if not source_path:
        raise TexturedRenderError(
            f"UV input {input_record.get('property_path')} has no source property path"
        )
    source_prim_path = str(connection.get("source_prim_path") or "")
    if not source_prim_path:
        source_prim_path = source_path.rsplit(".", 1)[0]
    return nodes.get(source_prim_path)


def _two_floats(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TexturedRenderError(f"{label} is not an exact two-component value")
    values = (_finite_scalar(value[0]), _finite_scalar(value[1]))
    if values[0] is None or values[1] is None:
        raise TexturedRenderError(f"{label} contains a non-finite value")
    return float(values[0]), float(values[1])


def _uv_from_node(
    node: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    *,
    depth: int = 0,
) -> UVBinding:
    if depth > 4:
        raise TexturedRenderError("UV connection graph is deeper than the supported exact mapping")
    shader_id = _normalized_name(node.get("shader_id"))
    if "primvarreader" in shader_id:
        varname = _input_by_name(node, {"varname", "primvar", "primvarname"})
        value = varname.get("value") if varname else None
        if not isinstance(value, str) or not value:
            raise TexturedRenderError(
                f"primvar reader {node.get('prim_path')} has no exact authored UV name"
            )
        return UVBinding(name=value)
    if "transform2d" not in shader_id:
        raise TexturedRenderError(
            "unsupported authored UV producer "
            f"{node.get('shader_id')!r} at {node.get('prim_path')}"
        )
    source_input = _input_by_name(node, {"in", "input", "st", "uv"})
    if source_input is None:
        raise TexturedRenderError(f"UV transform {node.get('prim_path')} has no input")
    source_node = _connection_source_node(source_input, nodes)
    if source_node is None:
        raise TexturedRenderError(f"UV transform {node.get('prim_path')} has no connected source")
    upstream = _uv_from_node(source_node, nodes, depth=depth + 1)
    scale_input = _input_by_name(node, {"scale"})
    rotation_input = _input_by_name(node, {"rotation"})
    translation_input = _input_by_name(node, {"translation"})
    scale = _two_floats(scale_input.get("value"), "UV scale") if scale_input else (1.0, 1.0)
    translation = (
        _two_floats(translation_input.get("value"), "UV translation")
        if translation_input
        else (0.0, 0.0)
    )
    rotation = _finite_scalar(rotation_input.get("value")) if rotation_input else 0.0
    if rotation is None:
        raise TexturedRenderError("UV rotation is non-finite")
    if upstream.scale != (1.0, 1.0) or upstream.rotation_degrees or upstream.translation != (0.0, 0.0):
        raise TexturedRenderError("stacked authored UV transforms are not supported exactly")
    return UVBinding(
        name=upstream.name,
        scale=scale,
        rotation_degrees=float(rotation),
        translation=translation,
        transform_semantics="usd_transform2d",
        resolution="exact_authored_primvar_and_transform2d",
    )


def _texture_uv_binding(
    evidence: Mapping[str, Any],
    material_record: Mapping[str, Any],
    available_uvs: set[str],
) -> UVBinding:
    nodes = _graph_nodes(material_record)
    shader_path = str(evidence.get("shader_path") or "")
    node = nodes.get(shader_path)
    if node is not None:
        st_input = _input_by_name(node, {"st", "uv", "texcoord", "texturecoordinate"})
        if st_input is not None:
            source = _connection_source_node(st_input, nodes)
            if source is not None:
                binding = _uv_from_node(source, nodes)
                if binding.name not in available_uvs:
                    raise TexturedRenderError(
                        f"authored UV set {binding.name!r} is absent from exact geometry"
                    )
                return binding
    if "st" in available_uvs:
        return UVBinding(name="st", resolution="exact_standard_st_primvar")
    if len(available_uvs) == 1:
        return UVBinding(
            name=next(iter(available_uvs)),
            resolution="only_exact_authored_uv_primvar",
        )
    raise TexturedRenderError(
        f"texture {evidence.get('authored_asset')!r} has no provable UV mapping; "
        f"available exact UV sets are {sorted(available_uvs)!r}"
    )


def _texture_has_authored_uv_input(
    evidence: Mapping[str, Any], material_record: Mapping[str, Any]
) -> bool:
    node = _graph_nodes(material_record).get(str(evidence.get("shader_path") or ""))
    return node is not None and _input_by_name(
        node, {"st", "uv", "texcoord", "texturecoordinate"}
    ) is not None


def _indexed_uv_name(index: int, available_uvs: set[str], family: str) -> str:
    candidates = ("st", "st0") if index == 0 else (f"st{index}",)
    matches = [name for name in candidates if name in available_uvs]
    if not matches:
        raise TexturedRenderError(
            f"{family} requests UV space {index}, but exact geometry has "
            f"{sorted(available_uvs)!r}"
        )
    return matches[0]


def _texture_color_space(
    role: str,
    evidence: Mapping[str, Any],
    *,
    specular_is_color: bool = False,
) -> str:
    authored = str(evidence.get("color_space") or "").casefold().replace("_", "")
    mdl_gamma = str(evidence.get("mdl_gamma_mode") or "").casefold().replace("_", "")
    value = authored or mdl_gamma
    if value in {"srgb", "gammasrgb"}:
        return "sRGB"
    if value in {"raw", "linear", "gammalinear"}:
        return "Non-Color"
    if value not in {"", "auto", "default", "gammadefault"}:
        raise TexturedRenderError(
            f"unsupported authored texture color space {value!r} for role {role!r}"
        )
    return (
        "sRGB"
        if role
        in {
            "albedo",
            "coat_color",
            "emissive",
            "retroreflection_color",
            "subsurface_scattering_color",
            "subsurface_transmission_color",
            "transmission_color",
            "transmission_scattering_color",
        }
        or (role == "specular" and specular_is_color)
        else "Non-Color"
    )


def _rgba_value(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise TexturedRenderError(f"{label} is not an exact four-component value")
    parsed = tuple(_finite_scalar(component) for component in value)
    if any(component is None for component in parsed):
        raise TexturedRenderError(f"{label} contains a non-finite component")
    return tuple(float(component) for component in parsed)  # type: ignore[arg-type]


def _texture_sampling(
    evidence: Mapping[str, Any], material_record: Mapping[str, Any]
) -> tuple[str, tuple[float, float, float, float], tuple[float, float, float, float]]:
    node = _graph_nodes(material_record).get(str(evidence.get("shader_path") or ""))
    if node is None or "usduvtexture" not in _normalized_name(node.get("shader_id")):
        return "REPEAT", (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)
    wrap_s_input = _input_by_name(node, {"wraps"})
    wrap_t_input = _input_by_name(node, {"wrapt"})
    wrap_s = str(wrap_s_input.get("value") if wrap_s_input else "repeat")
    wrap_t = str(wrap_t_input.get("value") if wrap_t_input else "repeat")
    if wrap_s != wrap_t:
        raise TexturedRenderError(
            f"mixed-axis authored texture wrap is not representable exactly: "
            f"wrapS={wrap_s!r}, wrapT={wrap_t!r}"
        )
    wrap_mapping = {
        "repeat": "REPEAT",
        "mirror": "MIRROR",
        "clamp": "EXTEND",
        "black": "CLIP",
    }
    if wrap_s not in wrap_mapping:
        raise TexturedRenderError(
            f"authored texture wrap {wrap_s!r} requires unresolved image metadata"
        )
    scale_input = _input_by_name(node, {"scale"})
    bias_input = _input_by_name(node, {"bias"})
    scale = (
        _rgba_value(scale_input.get("value"), "UsdUVTexture scale")
        if scale_input and scale_input.get("value") is not None
        else (1.0, 1.0, 1.0, 1.0)
    )
    bias = (
        _rgba_value(bias_input.get("value"), "UsdUVTexture bias")
        if bias_input and bias_input.get("value") is not None
        else (0.0, 0.0, 0.0, 0.0)
    )
    return wrap_mapping[wrap_s], scale, bias


def _normal_channel_flips(
    role: str,
    source_path: Path,
    material_record: Mapping[str, Any],
    material_family: str,
    active_values: Mapping[str, list[Any]],
) -> tuple[bool, bool, bool]:
    if role not in {"normal", "detail_normal"}:
        return False, False, False
    shader_ids = {
        _normalized_name(node.get("shader_id"))
        for node in _graph_nodes(material_record).values()
    }
    if material_family in {"omnipbr", "simpbr", "omniglass"}:
        # OmniPBR's declared defaults are U=false and V=true. Blender's
        # tangent-space Normal Map input is +X/+Y, so apply the template's
        # authored/default tangent flips directly to red and green.
        return (
            _unique_bool(
                active_values,
                {"fliptangentu", "normalmapfliptangentu"},
                False,
            ),
            _unique_bool(
                active_values,
                {"fliptangentv", "normalmapfliptangentv"},
                True,
            ),
            False,
        )
    if material_family == "nvidiaglasswithvolume":
        return False, False, False
    if material_family in {"omnisurface", "omnisurfacepreset"}:
        tangent_mapping = _unique_token(
            active_values,
            {"geometrytangentbitangentmapping"},
            "red_green",
        )
        if tangent_mapping != "redgreen":
            raise TexturedRenderError(
                "OmniSurface geometry normal encoding is unsupported"
            )
        return (
            _unique_bool(
                active_values,
                {"geometrynormalimagefliprchannel"},
                False,
            ),
            not _unique_bool(
                active_values,
                {"geometrynormalimageflipgchannel"},
                False,
            ),
            _unique_bool(
                active_values,
                {"geometrynormalimageflipbchannel"},
                False,
            ),
        )
    if material_family == "usd_preview_surface" or any(
        "usdpreviewsurface" in shader_id for shader_id in shader_ids
    ):
        return (
            _unique_bool(
                active_values,
                {"fliptangentu", "normalmapfliptangentu"},
                False,
            ),
            _unique_bool(
                active_values,
                {"fliptangentv", "normalmapfliptangentv"},
                False,
            ),
            False,
        )
    raise TexturedRenderError(
        f"normal convention is not attested by the material graph for {source_path}"
    )


def _resolved_udim_pattern(
    authored_asset: str, tiles: tuple[TextureTile, ...]
) -> str:
    authored_name = Path(authored_asset.replace("\\", "/")).name
    present_tokens = [
        token for token in ("<UDIM>", "<udim>", "%04d") if token in authored_name
    ]
    if len(present_tokens) != 1 or authored_name.count(present_tokens[0]) != 1:
        raise TexturedRenderError(
            f"UDIM source {authored_asset!r} does not contain one exact tile token"
        )
    token = present_tokens[0]
    parents = {str(Path(tile.path).parent) for tile in tiles}
    if len(parents) != 1:
        raise TexturedRenderError("UDIM tiles do not resolve into one directory")
    for tile in tiles:
        expected_name = authored_name.replace(token, f"{tile.tile_id:04d}")
        if Path(tile.path).name != expected_name:
            raise TexturedRenderError(
                f"UDIM tile {tile.tile_id} path does not match {authored_asset!r}"
            )
    return str(Path(next(iter(parents))) / authored_name.replace(token, "<UDIM>"))


def _selected_evidence_by_role(
    material_record: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, _TextureSource]]:
    selected = material_record.get("selected_surface_textures")
    selected_sets = material_record.get("selected_surface_texture_sets", {})
    evidence = material_record.get("selected_surface_texture_evidence")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(selected_sets, Mapping)
        or not isinstance(evidence, Mapping)
    ):
        raise TexturedRenderError("material record lacks selected surface texture evidence")
    blockers = {
        "unresolved_surface_textures": material_record.get("unresolved_surface_textures"),
        "surface_texture_ambiguities": material_record.get(
            "surface_texture_ambiguities"
        ),
        "surface_unknown_texture_roles": material_record.get("surface_unknown_texture_roles"),
        "texture_fallbacks": material_record.get(
            "texture_fallbacks"
        ),
    }
    active = [name for name, value in blockers.items() if value]
    if active:
        raise TexturedRenderError(
            "material appearance has unresolved/ambiguous authored evidence: " + ", ".join(active)
        )
    selected_roles = {str(role) for role in selected}
    set_roles = {str(role) for role in selected_sets}
    overlap = selected_roles & set_roles
    if overlap:
        raise TexturedRenderError(
            f"texture roles appear in both single-file and UDIM selections: {sorted(overlap)!r}"
        )
    result: dict[str, Mapping[str, Any]] = {}
    sources: dict[str, _TextureSource] = {}
    for role_value, selected_path_value in sorted(selected.items()):
        role = str(role_value)
        if role not in _SUPPORTED_TEXTURE_ROLES:
            raise TexturedRenderError(f"active texture role {role!r} has no exact Principled mapping")
        selected_path = str(selected_path_value)
        candidates = evidence.get(role)
        if not isinstance(candidates, list) or not candidates:
            raise TexturedRenderError(f"selected texture {role!r} has no evidence records")
        usable = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TexturedRenderError(f"selected texture evidence for {role!r} is malformed")
            paths = candidate.get("resolved_paths")
            if paths == [selected_path]:
                usable.append(candidate)
        if not usable:
            raise TexturedRenderError(f"selected texture path for {role!r} is not attested")
        signatures = {
            (
                str(item.get("semantic_source_output_channel") or ""),
                str(item.get("color_space") or ""),
                str(item.get("mdl_gamma_mode") or ""),
            )
            for item in usable
        }
        if len(signatures) > 1:
            raise TexturedRenderError(
                f"selected texture {role!r} has multiple incompatible active mappings"
            )
        result[role] = usable[0]
        source_path = Path(selected_path)
        if not source_path.is_absolute():
            source_path = source_path.resolve(strict=False)
        if not source_path.is_file():
            raise TexturedRenderError(
                f"resolved authored texture is unreadable for {role!r}: {source_path}"
            )
        sources[role] = _TextureSource(
            path=str(source_path),
            file_sha256=_sha256_file(source_path),
            tiles=(),
        )
    for role_value, set_value in sorted(selected_sets.items()):
        role = str(role_value)
        if role not in _SUPPORTED_TEXTURE_ROLES:
            raise TexturedRenderError(
                f"active tiled texture role {role!r} has no exact Principled mapping"
            )
        if (
            not isinstance(set_value, Mapping)
            or set(set_value)
            != {"texture_type", "authored_asset", "tiles", "evidence_sha256"}
            or set_value.get("texture_type") != "udim"
        ):
            raise TexturedRenderError(f"selected texture set for {role!r} is malformed")
        authored_asset = set_value.get("authored_asset")
        tile_values = set_value.get("tiles")
        if not isinstance(authored_asset, str) or not authored_asset:
            raise TexturedRenderError(f"selected UDIM source for {role!r} is missing")
        if not isinstance(tile_values, list) or not tile_values:
            raise TexturedRenderError(
                f"selected UDIM source for {role!r} has no resolved tiles"
            )
        tiles: list[TextureTile] = []
        for tile_value in tile_values:
            if not isinstance(tile_value, Mapping) or set(tile_value) != {
                "tile_id",
                "resolved_path",
                "file_sha256",
            }:
                raise TexturedRenderError(f"selected UDIM tile for {role!r} is malformed")
            tile_id = tile_value.get("tile_id")
            resolved_path = tile_value.get("resolved_path")
            declared_sha256 = tile_value.get("file_sha256")
            if (
                isinstance(tile_id, bool)
                or not isinstance(tile_id, int)
                or tile_id < 1001
                or tile_id > 1999
                or not isinstance(resolved_path, str)
                or not resolved_path
                or not isinstance(declared_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", declared_sha256) is None
            ):
                raise TexturedRenderError(f"selected UDIM tile for {role!r} is malformed")
            source_path = Path(resolved_path)
            if not source_path.is_absolute() or not source_path.is_file():
                raise TexturedRenderError(
                    f"resolved UDIM tile is unreadable for {role!r}: {source_path}"
                )
            actual_sha256 = _sha256_file(source_path)
            if actual_sha256 != declared_sha256:
                raise TexturedRenderError(
                    f"UDIM tile {tile_id} content changed for texture role {role!r}"
                )
            tiles.append(
                TextureTile(
                    tile_id=tile_id,
                    path=str(source_path),
                    file_sha256=actual_sha256,
                )
            )
        tile_tuple = tuple(tiles)
        tile_ids = [tile.tile_id for tile in tile_tuple]
        if tile_ids != sorted(set(tile_ids)):
            raise TexturedRenderError(
                f"UDIM tiles for texture role {role!r} are not uniquely sorted"
            )
        resolved_pattern = _resolved_udim_pattern(authored_asset, tile_tuple)
        candidates = evidence.get(role)
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise TexturedRenderError(
                f"selected UDIM texture {role!r} does not have one exact evidence record"
            )
        candidate = candidates[0]
        tile_paths = [tile.path for tile in tile_tuple]
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("role") != role
            or candidate.get("authored_asset") != authored_asset
            or candidate.get("resolved_paths") != tile_paths
            or candidate.get("role_distinct_resolved_files") != tile_paths
            or candidate.get("role_distinct_resolved_file_count") != len(tile_paths)
            or candidate.get("resolved_file_count") != len(tile_paths)
            or candidate.get("has_multiple_resolved_files") is not (len(tile_paths) > 1)
            or candidate.get("role_selection_status") != "selected_exact_udim_set"
            or candidate.get("role_is_ambiguous") is not False
            or candidate.get("role_has_udim") is not True
            or candidate.get("is_udim") is not True
            or candidate.get("role_has_multiple_sources") is not False
            or type(candidate.get("role_source_count")) is not int
            or candidate.get("role_source_count") != 1
            or candidate.get("role_competing_unresolved_sources") != []
            or candidate.get("used_path_fallback") is not False
        ):
            raise TexturedRenderError(
                f"selected UDIM texture {role!r} is not attested by exact candidate evidence"
            )
        evidence_sha256 = set_value.get("evidence_sha256")
        if evidence_sha256 != _reader_contract_hash(
            "textured-render-provenance", candidate
        ):
            raise TexturedRenderError(
                f"selected UDIM texture {role!r} evidence digest is invalid"
            )
        aggregate_sha256 = _canonical_hash(
            "udim-texture-set", [tile.record() for tile in tile_tuple]
        )
        source = _TextureSource(
            path=resolved_pattern,
            file_sha256=aggregate_sha256,
            tiles=tile_tuple,
        )
        _udim_grid_bounds(role, source.tiles)
        result[role] = candidate
        sources[role] = source
    if set(result) != selected_roles | set_roles or set(sources) != set(result):
        raise TexturedRenderError("selected texture source and evidence roles disagree")
    return result, sources


def _active_authored_values(material_record: Mapping[str, Any]) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {}
    for node in _graph_nodes(material_record).values():
        for item in _node_inputs(node):
            if not item.get("is_active_property") or not item.get(
                "authored_value_is_active"
            ):
                continue
            if item.get("value_might_be_time_varying") or item.get("authored_time_sample_count"):
                raise TexturedRenderError(
                    f"time-varying active material input is unsupported: {item.get('property_path')}"
                )
            name = _normalized_name(item.get("name") or item.get("full_name"))
            values.setdefault(name, []).append(item.get("value"))
    return values


def _active_authored_input_records(
    material_record: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    return [
        item
        for node in _graph_nodes(material_record).values()
        for item in _node_inputs(node)
        if item.get("is_active_property") and item.get("authored_value_is_active")
    ]


def _unique_scalar(
    values: Mapping[str, list[Any]], aliases: set[str], default: float
) -> float:
    found = {
        scalar
        for name in aliases
        for value in values.get(name, [])
        for scalar in [_finite_scalar(value)]
        if scalar is not None
    }
    if len(found) > 1:
        raise TexturedRenderError(f"active authored scalar inputs disagree for {sorted(aliases)}")
    return float(next(iter(found))) if found else float(default)


def _unique_color(
    values: Mapping[str, list[Any]], aliases: set[str]
) -> tuple[float, float, float, float] | None:
    found = {
        color
        for name in aliases
        for value in values.get(name, [])
        for color in [_finite_color(value)]
        if color is not None
    }
    if len(found) > 1:
        raise TexturedRenderError(f"active authored color inputs disagree for {sorted(aliases)}")
    return next(iter(found)) if found else None


def _unique_bool(
    values: Mapping[str, list[Any]], aliases: set[str], default: bool
) -> bool:
    found: set[bool] = set()
    for name in aliases:
        for value in values.get(name, []):
            if isinstance(value, bool):
                found.add(value)
            else:
                scalar = _finite_scalar(value)
                if scalar in {0.0, 1.0}:
                    found.add(bool(scalar))
    if len(found) > 1:
        raise TexturedRenderError(f"active authored boolean inputs disagree for {sorted(aliases)}")
    return next(iter(found)) if found else default


def _unique_mono_mode(
    values: Mapping[str, list[Any]], aliases: set[str], default: str
) -> str:
    ordered = ("alpha", "average", "luminance", "maximum")
    found: set[str] = set()
    for name in aliases:
        for value in values.get(name, []):
            if isinstance(value, int) and not isinstance(value, bool):
                if 0 <= value < len(ordered):
                    found.add(ordered[value])
                continue
            normalized = _normalized_name(value)
            for mode in ordered:
                if normalized in {mode, f"mono{mode}"}:
                    found.add(mode)
                    break
    if len(found) > 1:
        raise TexturedRenderError(
            f"active authored mono-mode inputs disagree for {sorted(aliases)}"
        )
    return next(iter(found)) if found else default


def _unique_vector3(
    values: Mapping[str, list[Any]], aliases: set[str], default: tuple[float, float, float]
) -> tuple[float, float, float]:
    found: set[tuple[float, float, float]] = set()
    for name in aliases:
        for value in values.get(name, []):
            if not isinstance(value, list) or len(value) != 3:
                continue
            parsed = tuple(_finite_scalar(component) for component in value)
            if all(component is not None for component in parsed):
                found.add(tuple(float(component) for component in parsed))  # type: ignore[arg-type]
    if len(found) > 1:
        raise TexturedRenderError(f"active authored vector inputs disagree for {sorted(aliases)}")
    return next(iter(found)) if found else default


def _unique_vector2(
    values: Mapping[str, list[Any]], aliases: set[str], default: tuple[float, float]
) -> tuple[float, float]:
    found: set[tuple[float, float]] = set()
    for name in aliases:
        for value in values.get(name, []):
            if not isinstance(value, list) or len(value) != 2:
                continue
            parsed = tuple(_finite_scalar(component) for component in value)
            if all(component is not None for component in parsed):
                found.add(tuple(float(component) for component in parsed))  # type: ignore[arg-type]
    if len(found) > 1:
        raise TexturedRenderError(f"active authored vector inputs disagree for {sorted(aliases)}")
    return next(iter(found)) if found else default


def _unique_token(
    values: Mapping[str, list[Any]], aliases: set[str], default: str
) -> str:
    found = {
        _normalized_name(value)
        for name in aliases
        for value in values.get(name, [])
        if isinstance(value, str) and _normalized_name(value)
    }
    if len(found) > 1:
        raise TexturedRenderError(
            f"active authored enum/string inputs disagree for {sorted(aliases)}"
        )
    return next(iter(found)) if found else _normalized_name(default)


def _require_unit_interval(label: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise TexturedRenderError(f"{label}={value} is outside [0, 1]")
    return value


def _omni_glass_model_parameters(
    active_values: Mapping[str, list[Any]],
    *,
    nvidia_glass_with_volume: bool = False,
) -> dict[str, Any]:
    color_alias = {"transmissioncolor"} if nvidia_glass_with_volume else {"glasscolor"}
    roughness_alias = {"roughness"} if nvidia_glass_with_volume else {"frostingroughness"}
    ior_alias = {"ior"} if nvidia_glass_with_volume else {"glassior"}
    parameters = {
        "glass_color": list(
            _unique_color(active_values, color_alias)
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "roughness": _unique_scalar(
            active_values, roughness_alias, 0.0
        ),
        "roughness_texture_influence": _unique_scalar(
            active_values, {"roughnesstextureinfluence"}, 1.0
        ),
        "ior": _unique_scalar(
            active_values, ior_alias, 1.341 if nvidia_glass_with_volume else 1.491
        ),
        "thin_walled": _unique_bool(active_values, {"thinwalled"}, False),
        "absorption_scale": _unique_scalar(active_values, {"depth"}, 0.001),
        "reflection_color": list(
            _unique_color(active_values, {"reflectioncolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "normal_strength": _unique_scalar(
            active_values,
            set() if nvidia_glass_with_volume else {"bumpfactor", "normalmapstrength"},
            1.0,
        ),
        "normal_roughness_strength": _unique_scalar(
            active_values,
            {
                "normalmaproughnessstrength",
                "normalroughnessstrength",
                "normalinducedroughnessstrength",
            },
            1.0,
        ),
        "transmission_texture_role": (
            "transmission_color" if nvidia_glass_with_volume else "albedo"
        ),
        "non_thin_surface_untinted": nvidia_glass_with_volume,
    }
    _require_unit_interval("OmniGlass roughness", parameters["roughness"])
    _require_unit_interval(
        "OmniGlass roughness texture influence",
        parameters["roughness_texture_influence"],
    )
    if parameters["ior"] <= 0.0:
        raise TexturedRenderError("OmniGlass IOR must be positive")
    if parameters["absorption_scale"] < 0.0:
        raise TexturedRenderError("OmniGlass absorption scale must be non-negative")
    if parameters["normal_strength"] < 0.0:
        raise TexturedRenderError("OmniGlass normal strength must be non-negative")
    if parameters["normal_roughness_strength"] != 1.0:
        raise TexturedRenderError(
            "OmniGlass normal-induced roughness is not representable exactly in Cycles"
        )
    return parameters


def _omni_surface_model_parameters(
    active_values: Mapping[str, list[Any]],
) -> dict[str, Any]:
    transmission_enabled = _unique_bool(
        active_values, {"enablespeculartransmission"}, False
    )
    transmission_weight = _unique_scalar(
        active_values, {"speculartransmissionweight"}, 0.0
    )
    if not transmission_enabled:
        transmission_weight = 0.0
    subsurface_enabled = _unique_bool(
        active_values, {"enablediffusetransmission"}, False
    )
    subsurface_weight = _unique_scalar(
        active_values, {"subsurfaceweight"}, 0.0
    )
    if not subsurface_enabled:
        subsurface_weight = 0.0
    thin_film_enabled = _unique_bool(active_values, {"enablethinfilm"}, False)
    emission_weight = _unique_scalar(active_values, {"emissionweight"}, 0.0)
    parameters = {
        "diffuse_weight": _unique_scalar(
            active_values, {"diffusereflectionweight"}, 0.8
        ),
        "diffuse_color": list(
            _unique_color(active_values, {"diffusereflectioncolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "diffuse_roughness": _unique_scalar(
            active_values, {"diffusereflectionroughness"}, 0.0
        ),
        "metalness": _unique_scalar(active_values, {"metalness"}, 0.0),
        "specular_weight": _unique_scalar(
            active_values, {"specularreflectionweight"}, 1.0
        ),
        "specular_color": list(
            _unique_color(active_values, {"specularreflectioncolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "specular_roughness": _unique_scalar(
            active_values, {"specularreflectionroughness"}, 0.2
        ),
        "specular_ior": _unique_scalar(
            active_values, {"specularreflectionior"}, 1.5
        ),
        "specular_anisotropy": _unique_scalar(
            active_values, {"specularreflectionanisotropy"}, 0.0
        ),
        "specular_anisotropy_rotation": _unique_scalar(
            active_values, {"specularreflectionanisotropyrotation"}, 0.0
        ),
        "transmission_enabled": transmission_enabled,
        "transmission_weight": transmission_weight,
        "transmission_color": list(
            _unique_color(active_values, {"speculartransmissioncolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "transmission_depth": _unique_scalar(
            active_values, {"speculartransmissionscatteringdepth"}, 0.0
        ),
        "transmission_scattering_color": list(
            _unique_color(
                active_values, {"speculartransmissionscatteringcolor"}
            )
            or (0.0, 0.0, 0.0, 1.0)
        ),
        "transmission_scatter_anisotropy": _unique_scalar(
            active_values, {"speculartransmissionscatteranisotropy"}, 0.0
        ),
        "transmission_dispersion_abbe": _unique_scalar(
            active_values, {"speculartransmissiondispersionabbe"}, 0.0
        ),
        "subsurface_enabled": subsurface_enabled,
        "subsurface_weight": subsurface_weight,
        "subsurface_transmission_color": list(
            _unique_color(active_values, {"subsurfacetransmissioncolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "subsurface_scattering_color": list(
            _unique_color(active_values, {"subsurfacescatteringcolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "subsurface_scale": _unique_scalar(
            active_values, {"subsurfacescale"}, 1.0
        ),
        "subsurface_anisotropy": _unique_scalar(
            active_values, {"subsurfaceanisotropy"}, 0.0
        ),
        "coat_weight": _unique_scalar(active_values, {"coatweight"}, 0.0),
        "coat_color": list(
            _unique_color(active_values, {"coatcolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "coat_roughness": _unique_scalar(
            active_values, {"coatroughness"}, 0.1
        ),
        "coat_ior": _unique_scalar(active_values, {"coatior"}, 1.5),
        "coat_anisotropy": _unique_scalar(
            active_values, {"coatanisotropy"}, 0.0
        ),
        "coat_anisotropy_rotation": _unique_scalar(
            active_values, {"coatanisotropyrotation"}, 0.0
        ),
        "coat_affect_color": _unique_scalar(
            active_values, {"coataffectcolor"}, 0.0
        ),
        "coat_affect_roughness": _unique_scalar(
            active_values, {"coataffectroughness"}, 0.0
        ),
        "retro_weight": _unique_scalar(
            active_values, {"specularretroreflectionweight"}, 0.0
        ),
        "retro_color": list(
            _unique_color(active_values, {"specularretroreflectioncolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "retro_roughness": _unique_scalar(
            active_values, {"specularretroreflectionroughness"}, 0.3
        ),
        "emission_weight": emission_weight,
        "emission_color": list(
            _unique_color(active_values, {"emissioncolor"})
            or (1.0, 1.0, 1.0, 1.0)
        ),
        "emission_intensity": _unique_scalar(
            active_values, {"emissionintensity"}, 1.0
        ),
        "emission_mode": _unique_token(
            active_values, {"emissionmode"}, "emission_lx"
        ),
        "emission_use_temperature": _unique_bool(
            active_values, {"emissionusetemperature"}, False
        ),
        "thin_film_enabled": thin_film_enabled,
        "thin_film_thickness": _unique_scalar(
            active_values, {"thinfilmthickness"}, 400.0
        ),
        "thin_film_ior": _unique_scalar(active_values, {"thinfilmior"}, 1.52),
        "thin_walled": _unique_bool(active_values, {"thinwalled"}, False),
        "opacity": _unique_scalar(active_values, {"geometryopacity"}, 1.0),
        "opacity_threshold": _unique_scalar(
            active_values, {"geometryopacitythreshold"}, 0.0
        ),
        "normal_strength": _unique_scalar(
            active_values, {"geometrynormalstrength"}, 1.0
        ),
        "normal_roughness_strength": _unique_scalar(
            active_values, {"geometrynormalroughnessstrength"}, 1.0
        ),
        "displacement_scale": _unique_scalar(
            active_values, {"geometrydisplacementscale"}, 1.0
        ),
        "displacement_zero": _unique_scalar(
            active_values, {"geometrydisplacementscalarzerovalue"}, 0.0
        ),
        "displacement_mode": _unique_token(
            active_values, {"geometrydisplacementmode"}, "displacement_height"
        ),
        "round_edges_radius": _unique_scalar(
            active_values, {"roundedgesradius"}, 0.0
        ),
    }
    for label in (
        "diffuse_weight",
        "diffuse_roughness",
        "metalness",
        "specular_weight",
        "specular_roughness",
        "specular_anisotropy",
        "transmission_weight",
        "subsurface_weight",
        "coat_weight",
        "coat_roughness",
        "coat_anisotropy",
        "coat_affect_color",
        "coat_affect_roughness",
        "retro_weight",
        "retro_roughness",
        "emission_weight",
    ):
        _require_unit_interval(f"OmniSurface {label}", parameters[label])
    for label in ("specular_ior", "coat_ior", "thin_film_ior"):
        if parameters[label] <= 0.0:
            raise TexturedRenderError(f"OmniSurface {label} must be positive")
    if parameters["transmission_depth"] < 0.0 or parameters["subsurface_scale"] < 0.0:
        raise TexturedRenderError("OmniSurface depth/scale must be non-negative")
    if not -1.0 <= parameters["transmission_scatter_anisotropy"] <= 1.0:
        raise TexturedRenderError("OmniSurface volume anisotropy is outside [-1, 1]")
    if not -1.0 <= parameters["subsurface_anisotropy"] <= 1.0:
        raise TexturedRenderError("OmniSurface subsurface anisotropy is outside [-1, 1]")
    if parameters["transmission_dispersion_abbe"] != 0.0:
        raise TexturedRenderError(
            "OmniSurface spectral dispersion is unsupported by the Cycles material graph"
        )
    if thin_film_enabled:
        raise TexturedRenderError(
            "OmniSurface thin-film interference is unsupported by the Cycles material graph"
        )
    if parameters["coat_weight"] > 0.0 and (
        parameters["coat_affect_color"] != 0.0
        or parameters["coat_affect_roughness"] != 0.0
    ):
        raise TexturedRenderError(
            "OmniSurface coat-affect controls require an unsupported layered closure"
        )
    if emission_weight > 0.0 and parameters["emission_use_temperature"]:
        raise TexturedRenderError(
            "OmniSurface blackbody emission is unsupported by the Cycles material graph"
        )
    if parameters["normal_strength"] < 0.0:
        raise TexturedRenderError("OmniSurface normal strength must be non-negative")
    if parameters["normal_roughness_strength"] != 1.0:
        raise TexturedRenderError(
            "OmniSurface normal-induced roughness is unsupported by the Cycles graph"
        )
    if parameters["round_edges_radius"] != 0.0:
        raise TexturedRenderError(
            "OmniSurface rounded edges are unsupported by the Cycles graph"
        )
    _require_unit_interval("OmniSurface opacity", parameters["opacity"])
    _require_unit_interval(
        "OmniSurface opacity threshold", parameters["opacity_threshold"]
    )
    if parameters["displacement_scale"] < 0.0:
        raise TexturedRenderError("OmniSurface displacement scale must be non-negative")
    if parameters["displacement_mode"] != "displacementheight":
        raise TexturedRenderError("OmniSurface vector displacement is unsupported")
    return parameters


def _omni_surface_uv_binding(
    active_values: Mapping[str, list[Any]], available_uvs: set[str]
) -> UVBinding:
    coordinate_system = _unique_token(
        active_values,
        {"uvwtexturecoordinatesystem"},
        "texture_coordinate_uvw",
    )
    if coordinate_system != "texturecoordinateuvw":
        raise TexturedRenderError(
            "OmniSurface uses a non-mesh texture-coordinate system"
        )
    if _unique_bool(active_values, {"uvwuseuvcoords"}, False):
        raise TexturedRenderError(
            "OmniSurface constant UV-coordinate override is unsupported"
        )
    uv_set = int(_unique_scalar(active_values, {"uvwuvset"}, 0.0))
    if uv_set < 0:
        raise TexturedRenderError("OmniSurface UV set is negative")
    uv_name = _indexed_uv_name(uv_set, available_uvs, "OmniSurface")
    wrap_s = _unique_token(
        active_values, {"uvwswrap"}, "wrap_periodic"
    )
    wrap_t = _unique_token(
        active_values, {"uvwtwrap"}, "wrap_periodic"
    )
    if wrap_s != "wrapperiodic" or wrap_t != "wrapperiodic":
        raise TexturedRenderError(
            "OmniSurface non-periodic per-axis texture wrapping is unsupported"
        )
    if (
        _unique_bool(active_values, {"uvwsflip"}, False)
        or _unique_bool(active_values, {"uvwtflip"}, False)
        or _unique_bool(active_values, {"uvwswapst"}, False)
    ):
        raise TexturedRenderError(
            "OmniSurface UV flip/swap controls are unsupported"
        )
    return UVBinding(
        name=uv_name,
        scale=(
            _unique_scalar(active_values, {"uvwsscale"}, 1.0),
            _unique_scalar(active_values, {"uvwtscale"}, 1.0),
        ),
        translation=(
            _unique_scalar(active_values, {"uvwsoffset"}, 0.0),
            _unique_scalar(active_values, {"uvwtoffset"}, 0.0),
        ),
        transform_semantics="omnisurface_uv",
        resolution="exact_omnisurface_uv_controls",
    )


def _constant_base_color(material_record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    sources = material_record.get("surface_base_color_constant_evidence")
    if not isinstance(sources, list):
        return None
    colors = {
        color
        for source in sources
        if isinstance(source, Mapping)
        for color in [_finite_color(source.get("rgba"))]
        if color is not None
    }
    if len(colors) > 1:
        raise TexturedRenderError("active base-color constants disagree")
    return next(iter(colors)) if colors else None


def _display_primvar_record(
    segment_info: Mapping[str, Any], name: str
) -> Mapping[str, Any] | None:
    primvars = segment_info.get("display_primvars")
    record = primvars.get(name) if isinstance(primvars, Mapping) else None
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise TexturedRenderError(f"unbound segment has invalid {name}")
    cardinality_valid = record.get("cardinality_valid")
    if cardinality_valid is False:
        return None
    if cardinality_valid is not True:
        raise TexturedRenderError(f"unbound segment has invalid {name}")
    interpolation = str(record.get("interpolation") or "constant")
    if interpolation not in {"constant", "uniform", "vertex", "varying", "faceVarying"}:
        raise TexturedRenderError(
            f"unbound segment {name} has unsupported interpolation {interpolation!r}"
        )
    values = record.get("computed_flattened_values")
    if not isinstance(values, list) or not values:
        raise TexturedRenderError(f"unbound segment {name} has no exact flattened values")
    return record


def _display_scalar(value: Any) -> float | None:
    scalar = _finite_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, list) and len(value) == 1:
        return _finite_scalar(value[0])
    return None


def _display_color_plan(
    segment_info: Mapping[str, Any],
) -> tuple[
    tuple[float, float, float, float],
    float,
    str | None,
    str | None,
    str,
]:
    color_record = _display_primvar_record(segment_info, "displayColor")
    if color_record is None:
        color = (0.5, 0.5, 0.5, 1.0)
        color_attribute = None
        appearance_source = "usd_imaging_gprim_fallback_display_color"
    else:
        color_values = _display_values(color_record, color=True)
        unique_colors = {tuple(float(component) for component in value) for value in color_values}
        if len(unique_colors) == 1:
            color = next(iter(unique_colors))
            color_attribute = None
        else:
            color = (1.0, 1.0, 1.0, 1.0)
            color_attribute = "vlm_displayColor"
        appearance_source = "exact_uniform_display_primvars"

    opacity_record = _display_primvar_record(segment_info, "displayOpacity")
    opacity = 1.0
    opacity_attribute = None
    if opacity_record is not None:
        opacity_values = _display_values(opacity_record, color=False)
        unique_opacities = {float(value) for value in opacity_values}
        if len(unique_opacities) == 1:
            opacity = next(iter(unique_opacities))
        else:
            opacity_attribute = "vlm_displayOpacity"
        if color_record is None:
            appearance_source = "exact_uniform_display_primvars"
    if color_attribute or opacity_attribute:
        appearance_source = "exact_nonuniform_display_primvars"
    return (
        color,
        opacity,
        color_attribute,
        opacity_attribute,
        appearance_source,
    )


def _build_material_plan(
    segment_info: Mapping[str, Any],
    payload: SegmentGeometryPayload,
    materials: Mapping[str, Any],
) -> MaterialPlan:
    material_record_id_value = segment_info.get("material_record_id")
    material_record_id = (
        str(material_record_id_value) if material_record_id_value is not None else None
    )
    payload_record_id = payload.provenance.get("material_record_id")
    if payload_record_id != material_record_id_value:
        raise TexturedRenderError(
            f"segment {segment_info.get('segment_id')} material binding changed after extraction"
        )
    double_sided = bool(payload.provenance.get("double_sided"))
    if material_record_id is None:
        (
            base_color,
            opacity,
            display_color_attribute,
            display_opacity_attribute,
            appearance_source,
        ) = _display_color_plan(segment_info)
        core = {
            "material_record_id": None,
            "material_name": "unbound_display_color",
            "material_model": "principled",
            "model_parameters": {},
            "textures": [],
            "base_color": list(base_color),
            "roughness": 0.5,
            "metallic": 0.0,
            "opacity": opacity,
            "emission_color": [0.0, 0.0, 0.0, 1.0],
            "emission_strength": 0.0,
            "double_sided": double_sided,
            "display_color_attribute": display_color_attribute,
            "display_opacity_attribute": display_opacity_attribute,
            "simpbr_controls": False,
            "simpbr_orm_gate_repair": None,
            "source": appearance_source,
        }
        provenance = _canonical_hash("principled-material-plan", core)
        return MaterialPlan(
            material_record_id=None,
            material_name="unbound_display_color",
            material_model="principled",
            model_parameters={},
            textures=(),
            base_color=base_color,
            roughness=0.5,
            metallic=0.0,
            opacity=opacity,
            opacity_threshold=0.0,
            emission_color=(0.0, 0.0, 0.0, 1.0),
            emission_strength=0.0,
            double_sided=double_sided,
            display_color_attribute=display_color_attribute,
            display_opacity_attribute=display_opacity_attribute,
            detail_normal_strength=1.0,
            bump_strength=1.0,
            bump_distance=0.1,
            ior=1.5,
            specular_ior_level=0.5,
            transmission_weight=0.0,
            coat_weight=0.0,
            coat_roughness=0.03,
            coat_ior=1.5,
            coat_tint=(1.0, 1.0, 1.0, 1.0),
            sheen_weight=0.0,
            sheen_roughness=0.5,
            sheen_tint=(1.0, 1.0, 1.0, 1.0),
            anisotropic=0.0,
            anisotropic_rotation=0.0,
            subsurface_weight=0.0,
            subsurface_radius=(1.0, 0.2, 0.1),
            subsurface_scale=0.05,
            subsurface_ior=1.4,
            subsurface_anisotropy=0.0,
            thin_film_thickness=0.0,
            thin_film_ior=1.33,
            thin_walled=False,
            albedo_desaturation=0.0,
            albedo_add=0.0,
            albedo_brightness=1.0,
            diffuse_tint=(1.0, 1.0, 1.0, 1.0),
            ao_to_diffuse=0.0,
            roughness_texture_influence=1.0,
            metallic_texture_influence=1.0,
            normal_strength=1.0,
            omnipbr_controls=False,
            simpbr_controls=False,
            simpbr_orm_gate_repair=None,
            provenance_sha256=provenance,
        )

    material_record = materials.get(material_record_id)
    if not isinstance(material_record, Mapping):
        raise TexturedRenderError(
            f"segment {segment_info.get('segment_id')} references missing material record "
            f"{material_record_id!r}"
        )
    if str(material_record.get("material_record_id") or "") != material_record_id:
        raise TexturedRenderError(f"material record key/identity mismatch for {material_record_id!r}")
    graph = material_record.get("active_material_graph")
    if not isinstance(graph, Mapping):
        raise TexturedRenderError(f"material {material_record_id!r} has no active graph")
    invalid_fallback_blockers = ignored_invalid_connection_fallback_blockers(
        material_record
    )
    if invalid_fallback_blockers:
        raise TexturedRenderError(
            f"material {material_record_id!r} has an unsupported invalid "
            f"connection fallback: {invalid_fallback_blockers!r}"
        )
    graph_blockers = {
        "time_varying": graph.get("has_time_varying_values"),
        "lossy_value_fallback": graph.get("has_lossy_value_fallback"),
        "invalid_material_outputs": graph.get("invalid_material_output_connections"),
        "untraversable_connections": graph.get("untraversable_connection_endpoints"),
    }
    active_blockers = [name for name, value in graph_blockers.items() if value]
    if active_blockers:
        raise TexturedRenderError(
            f"material {material_record_id!r} has unsupported active graph state: "
            + ", ".join(active_blockers)
        )
    terminals = graph.get("terminals")
    if not isinstance(terminals, list) or not terminals:
        raise TexturedRenderError(f"material {material_record_id!r} has no connected terminal")
    terminal_names = {
        str(item.get("terminal") or "").casefold()
        for item in terminals
        if isinstance(item, Mapping)
    }
    def terminal_source_paths(name: str) -> set[str]:
        return {
            str(connection.get("source_property_path"))
            for item in terminals
            if isinstance(item, Mapping)
            and str(item.get("terminal") or "").casefold() == name
            for output in [item.get("material_output")]
            if isinstance(output, Mapping)
            for connection in output.get("connections") or []
            if isinstance(connection, Mapping) and connection.get("source_property_path")
        }

    def nonzero(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if (
            isinstance(value, Mapping)
            and "authored_path" in value
            and "usd_resolved_path" in value
        ):
            return (
                value.get("authored_path") not in {None, ""}
                or value.get("usd_resolved_path") is not None
            )
        scalar = _finite_scalar(value)
        if scalar is not None:
            return abs(scalar) > 1e-12
        if isinstance(value, list):
            parsed = [_finite_scalar(component) for component in value]
            return all(component is not None for component in parsed) and any(
                abs(float(component)) > 1e-12 for component in parsed
            )
        return value is not None and value != "" and value != "none"

    volume_sources = terminal_source_paths("volume")
    surface_sources = terminal_source_paths("surface")
    active_volume_parameters = []
    for node in _graph_nodes(material_record).values():
        for item in _node_inputs(node):
            name = _normalized_name(item.get("name") or item.get("full_name"))
            if not any(
                token in name
                for token in (
                    "volumedensity",
                    "densityscale",
                    "absorptiondistance",
                    "scatteringweight",
                    "volumetransmission",
                )
            ):
                continue
            if (
                item.get("is_active_property")
                and item.get("has_authored_value")
                and nonzero(item.get("value"))
            ):
                active_volume_parameters.append(str(item.get("property_path")))
    boilerplate_volume = bool(volume_sources) and volume_sources.issubset(
        surface_sources
    ) and not active_volume_parameters
    if "volume" in terminal_names and not boilerplate_volume:
        raise TexturedRenderError(
            f"material {material_record_id!r} has a participating volume terminal "
            f"that cannot be represented by the surface renderer; sources="
            f"{sorted(volume_sources)!r}, active_parameters="
            f"{active_volume_parameters!r}"
        )
    displacement_sources = terminal_source_paths("displacement")
    active_displacement_maps = []
    for node in _graph_nodes(material_record).values():
        for item in _node_inputs(node):
            name = _normalized_name(item.get("name") or item.get("full_name"))
            if not any(token in name for token in ("displacement", "height", "bump")):
                continue
            if not any(token in name for token in ("texture", "map", "file")):
                continue
            if (
                item.get("is_active_property")
                and item.get("has_authored_value")
                and nonzero(item.get("value"))
            ):
                active_displacement_maps.append(str(item.get("property_path")))
    boilerplate_displacement = bool(displacement_sources) and displacement_sources.issubset(
        surface_sources
    ) and not active_displacement_maps
    if "displacement" in terminal_names and not boilerplate_displacement and "displacement" not in (
        material_record.get("selected_surface_textures") or {}
    ) and "displacement" not in (
        material_record.get("selected_surface_texture_sets") or {}
    ):
        raise TexturedRenderError(
            f"material {material_record_id!r} has a participating displacement "
            f"terminal without an exact selected map; sources="
            f"{sorted(displacement_sources)!r}, active_maps="
            f"{active_displacement_maps!r}"
        )

    mdl_defaults, mdl_root_families, mdl_semantics = (
        _attested_mdl_material_defaults(material_record)
    )
    usd_active_values = _active_authored_values(material_record)
    active_values = {name: list(values) for name, values in mdl_defaults.items()}
    active_values.update(usd_active_values)
    material_family = _surface_root_material_family(
        material_record,
        mdl_root_families,
    )
    simpbr_controls = material_family == "simpbr"
    simpbr_translucent_controls = material_family == "simpbrtranslucent"
    omnipbr_controls = material_family in {"omnipbr", "simpbr"}
    nvidia_glass_controls = material_family == "nvidiaglasswithvolume"
    omniglass_controls = material_family in {
        "omniglass",
        "nvidiaglasswithvolume",
    }
    omnisurface_controls = material_family in {
        "omnisurface",
        "omnisurfacepreset",
    }
    material_model = (
        "omniglass"
        if omniglass_controls
        else "omnisurface"
        if omnisurface_controls
        else "principled"
    )
    model_parameters = (
        _omni_glass_model_parameters(
            active_values,
            nvidia_glass_with_volume=nvidia_glass_controls,
        )
        if omniglass_controls
        else _omni_surface_model_parameters(active_values)
        if omnisurface_controls
        else {}
    )
    selected_evidence, selected_sources = _selected_evidence_by_role(material_record)
    expected_mdl_family = {
        "omniglass": "OmniGlass",
        "nvidiaglasswithvolume": "NvidiaGlassWithVolume",
        "omnisurface": "OmniSurface",
        "omnisurfacepreset": "OmniSurfacePreset",
        "simpbrtranslucent": "SimPBRTranslucent",
    }.get(material_family)
    if expected_mdl_family is not None:
        mismatched_texture_families = {
            role: evidence.get("mdl_material_family")
            for role, evidence in selected_evidence.items()
            if evidence.get("mdl_material_family") != expected_mdl_family
        }
        if mismatched_texture_families:
            raise TexturedRenderError(
                "exact MDL texture evidence is not bound to the selected material family: "
                f"{mismatched_texture_families!r}"
            )
    omni_enable_orm = _unique_bool(active_values, {"enableormtexture"}, False)
    roughness_texture_influence = _unique_scalar(
        active_values,
        {"reflectionroughnesstextureinfluence", "roughnesstextureinfluence"},
        (
            float(model_parameters["roughness_texture_influence"])
            if omniglass_controls
            else 0.0
            if omnipbr_controls
            else 1.0
        ),
    )
    metallic_texture_influence = _unique_scalar(
        active_values,
        {"metallictextureinfluence", "metalnesstextureinfluence"},
        0.0 if omnipbr_controls else 1.0,
    )
    ao_to_diffuse = _unique_scalar(active_values, {"aotodiffuse"}, 0.0)
    omni_enable_opacity = _unique_bool(active_values, {"enableopacity"}, False)
    omni_enable_opacity_texture = _unique_bool(
        active_values, {"enableopacitytexture", "enableopacitymap"}, False
    )
    omni_opacity_mode = _unique_mono_mode(
        active_values, {"opacitymode"}, "average"
    )
    if simpbr_controls:
        omni_enable_opacity_texture = True
        omni_opacity_mode = "alpha"
    omni_enable_emission = _unique_bool(
        active_values, {"enableemission", "enableemissive"}, False
    )
    project_uvw = (
        not nvidia_glass_controls
        and (
            omnipbr_controls
            or omniglass_controls
            or simpbr_translucent_controls
        )
        and _unique_bool(active_values, {"projectuvw"}, False)
    )
    project_in_world_space = project_uvw and _unique_bool(
        active_values, {"worldorobject"}, False
    )
    if simpbr_controls:
        unsupported_simpbr_features = {
            name
            for name in (
                "enabletransmission",
                "enableclearcoat",
                "enableretroreflection",
                "enableemissiveflipbook",
            )
            if _unique_bool(active_values, {name}, False)
        }
        anisotropy = _unique_scalar(active_values, {"anisotropyconstant"}, 0.0)
        anisotropy_influence = _unique_scalar(
            active_values, {"anisotropytextureinfluence"}, 0.0
        )
        texture_speed = _unique_vector2(
            active_values, {"texturespeed"}, (0.0, 0.0)
        )
        if unsupported_simpbr_features or anisotropy or anisotropy_influence:
            raise TexturedRenderError(
                f"material {material_record_id!r} enables unsupported SimPBR features: "
                f"{sorted(unsupported_simpbr_features)!r}, "
                f"anisotropy={anisotropy}, influence={anisotropy_influence}"
            )
        if texture_speed != (0.0, 0.0):
            raise TexturedRenderError(
                f"material {material_record_id!r} has non-static SimPBR texture speed"
            )

    simpbr_orm_gate_repair = None
    if simpbr_controls and omni_enable_orm and "orm" not in selected_evidence:
        standalone_triggers = [
            {"role": role, "influence": influence}
            for role, influence in (
                ("roughness", roughness_texture_influence),
                ("metallic", metallic_texture_influence),
                ("occlusion", ao_to_diffuse),
            )
            if influence > 0.0
            and role in selected_evidence
            and Path(
                selected_sources[role].tiles[0].path
                if selected_sources[role].tiles
                else selected_sources[role].path
            ).is_file()
        ]
        if standalone_triggers:
            if "enableormtexture" in usd_active_values:
                raise TexturedRenderError(
                    f"material {material_record_id!r} explicitly enables SimPBR ORM "
                    "without an ORM map while authoring standalone inputs"
                )
            sim_records = [
                item
                for item in mdl_semantics
                if item.get("material_family") == "SimPBR"
            ]
            source_hashes = sorted(
                {str(item.get("source_sha256")) for item in sim_records}
            )
            terminal_hashes = sorted(
                {
                    str(item["terminal_material_module"].get("source_sha256"))
                    for item in sim_records
                    if isinstance(item.get("terminal_material_module"), Mapping)
                }
            )
            if len(source_hashes) != 1 or len(terminal_hashes) != 1:
                raise TexturedRenderError(
                    "SimPBR ORM-gate repair lacks one exact source/module attestation"
                )
            omni_enable_orm = False
            simpbr_orm_gate_repair = {
                "reason_code": "simpbr_missing_orm_gate_exporter_repair",
                "parameter": "enable_ORM_texture",
                "authored_value": None,
                "declared_default": True,
                "effective_value": False,
                "triggering_standalone_inputs": standalone_triggers,
                "simpbr_source_sha256": source_hashes[0],
                "simpbr_model_source_sha256": terminal_hashes[0],
            }
    uv_space_index = int(
        _unique_scalar(active_values, {"uvspaceindex"}, 0.0)
    )
    if uv_space_index < 0:
        raise TexturedRenderError("OmniPBR uv_space_index is negative")
    main_uv_scale = _unique_vector2(active_values, {"texturescale"}, (1.0, 1.0))
    main_uv_translate = _unique_vector2(
        active_values, {"texturetranslate"}, (0.0, 0.0)
    )
    main_uv_rotate = _unique_scalar(active_values, {"texturerotate"}, 0.0)
    detail_uv_scale = _unique_vector2(
        active_values, {"detailtexturescale"}, main_uv_scale
    )
    detail_uv_translate = _unique_vector2(
        active_values, {"detailtexturetranslate"}, main_uv_translate
    )
    detail_uv_rotate = _unique_scalar(
        active_values, {"detailtexturerotate"}, main_uv_rotate
    )
    available_uvs = set(payload.uv_sets)
    omnisurface_coat_active = omnisurface_controls and (
        float(model_parameters["coat_weight"]) > 0.0
        or "coat_weight" in selected_evidence
    )
    omnisurface_retro_active = omnisurface_controls and (
        float(model_parameters["retro_weight"]) > 0.0
        or "retroreflection_weight" in selected_evidence
    )
    omnisurface_emission_active = omnisurface_controls and (
        float(model_parameters["emission_weight"]) > 0.0
        or "emission_weight" in selected_evidence
    )
    if omnisurface_coat_active and (
        float(model_parameters["coat_affect_color"]) != 0.0
        or float(model_parameters["coat_affect_roughness"]) != 0.0
    ):
        raise TexturedRenderError(
            "OmniSurface coat-affect controls require an unsupported layered closure"
        )
    texture_plans: list[TexturePlan] = []
    for role, evidence in sorted(selected_evidence.items()):
        if omnipbr_controls:
            if role == "displacement":
                raise TexturedRenderError(
                    f"{material_family} root cannot consume a displacement texture"
                )
            if role == "orm" and not omni_enable_orm:
                continue
            if omni_enable_orm and role in {"occlusion", "roughness", "metallic"}:
                continue
            if role == "roughness" and roughness_texture_influence == 0.0:
                continue
            if role == "metallic" and metallic_texture_influence == 0.0:
                continue
            if role == "occlusion" and ao_to_diffuse == 0.0:
                continue
            if role == "opacity" and not (
                omni_enable_opacity and omni_enable_opacity_texture
            ):
                continue
            if role in {"emissive", "emissive_mask"} and not omni_enable_emission:
                continue
        if simpbr_translucent_controls:
            if role == "roughness" and roughness_texture_influence == 0.0:
                continue
            if role == "emissive_mask" and not omni_enable_emission:
                continue
            if role not in {"roughness", "normal", "emissive_mask"}:
                raise TexturedRenderError(
                    f"SimPBR translucent root cannot consume texture role {role!r}"
                )
        if omniglass_controls:
            if role == "roughness" and roughness_texture_influence == 0.0:
                continue
            if role == "normal" and model_parameters["normal_strength"] == 0.0:
                continue
            if role == "opacity" and not omni_enable_opacity:
                continue
            if role == "normal_roughness":
                raise TexturedRenderError(
                    "normal-induced roughness textures are unsupported by the Cycles glass graph"
                )
        if omnisurface_controls:
            if role in {
                "coat_affect_color",
                "coat_affect_roughness",
                "transmission_depth",
            }:
                raise TexturedRenderError(
                    f"OmniSurface texture role {role!r} cannot be represented exactly in Cycles"
                )
            if role.startswith("transmission_") and not model_parameters[
                "transmission_enabled"
            ]:
                continue
            if role.startswith("subsurface_") and not model_parameters[
                "subsurface_enabled"
            ]:
                continue
            if role.startswith("coat_") and not omnisurface_coat_active:
                continue
            if role.startswith("retroreflection_") and not omnisurface_retro_active:
                continue
            if role.startswith("emission_") and not omnisurface_emission_active:
                continue
            if role == "emissive" and not omnisurface_emission_active:
                continue
            if role == "thin_film_thickness" and not model_parameters[
                "thin_film_enabled"
            ]:
                continue
        source = selected_sources[role]
        source_path = Path(source.tiles[0].path if source.tiles else source.path)
        channel = str(evidence.get("semantic_source_output_channel") or "").casefold()
        if channel in {"x", "y", "z", "w"}:
            channel = {"x": "r", "y": "g", "z": "b", "w": "a"}[channel]
        if simpbr_controls and role == "opacity":
            if channel not in {"", "a", "rgba"}:
                raise TexturedRenderError(
                    f"SimPBR opacity requires texture alpha, got {channel!r}"
                )
            channel = "a"
        elif not channel:
            color_roles = {
                "albedo",
                "coat_color",
                "detail_normal",
                "emissive",
                "emissive_mask",
                "normal",
                "orm",
                "retroreflection_color",
                "subsurface_scattering_color",
                "subsurface_transmission_color",
                "transmission_color",
                "transmission_scattering_color",
            }
            if omniglass_controls or omnisurface_controls:
                color_roles.add("specular")
            channel = (
                "rgb"
                if role in color_roles
                or (material_family == "omnipbr" and role == "occlusion")
                else "r"
            )
        if channel not in _CHANNELS:
            raise TexturedRenderError(
                f"unsupported authored source channel {channel!r} for texture role {role!r}"
            )
        rgb_roles = {
            "albedo",
            "coat_color",
            "detail_normal",
            "normal",
            "emissive",
            "emissive_mask",
            "orm",
            "retroreflection_color",
            "subsurface_scattering_color",
            "subsurface_transmission_color",
            "transmission_color",
            "transmission_scattering_color",
        }
        if omniglass_controls or omnisurface_controls:
            rgb_roles.add("specular")
        if role in rgb_roles and channel not in {"rgb", "rgba"}:
            raise TexturedRenderError(
                f"texture role {role!r} requires an authored RGB output, got {channel!r}"
            )
        missing_mdl_texture_space = (
            (
                omnipbr_controls
                or omniglass_controls
                or simpbr_translucent_controls
            )
            and not project_uvw
            and not available_uvs
            and not _texture_has_authored_uv_input(evidence, material_record)
        )
        binding = (
            UVBinding(
                name=(
                    "world_position"
                    if project_in_world_space
                    else "object_position"
                ),
                coordinate_source=(
                    "world" if project_in_world_space else "object"
                ),
                projection="BOX",
                resolution=(
                    "exact_omnipbr_world_projected_uvw"
                    if project_in_world_space
                    else "exact_omnipbr_object_projected_uvw"
                ),
            )
            if project_uvw
            else UVBinding(
                name=f"absent_texture_space_{uv_space_index}",
                coordinate_source="constant_zero",
                resolution="exact_mdl_out_of_range_texture_space_zero_vector",
            )
            if missing_mdl_texture_space
            else _texture_uv_binding(evidence, material_record, available_uvs)
        )
        if (
            omnipbr_controls
            or omniglass_controls
            or simpbr_translucent_controls
        ):
            expected_uv_name = (
                binding.name
                if project_uvw or missing_mdl_texture_space
                else _indexed_uv_name(
                    uv_space_index, available_uvs, material_family
                )
            )
            if binding.scale != (1.0, 1.0) or binding.rotation_degrees or binding.translation != (0.0, 0.0):
                raise TexturedRenderError(
                    f"{material_family} and UsdTransform2d UV transforms are stacked"
                )
            is_detail = role == "detail_normal"
            selected_scale = detail_uv_scale if is_detail else main_uv_scale
            selected_rotation = (
                0.0
                if simpbr_controls or simpbr_translucent_controls
                else detail_uv_rotate if is_detail else main_uv_rotate
            )
            selected_translation = (
                detail_uv_translate if is_detail else main_uv_translate
            )
            binding = UVBinding(
                name=expected_uv_name,
                scale=selected_scale,
                rotation_degrees=selected_rotation,
                translation=selected_translation,
                transform_semantics=(
                    "mdl_base_rotation_translation_scale"
                    if simpbr_controls or simpbr_translucent_controls
                    else "mdl_local_scale_rotate"
                ),
                resolution=(
                    binding.resolution
                    if project_uvw or missing_mdl_texture_space
                    else
                    "exact_simpbr_uv_controls"
                    if simpbr_controls
                    else "exact_simpbr_translucent_uv_controls"
                    if simpbr_translucent_controls
                    else "exact_omniglass_uv_controls"
                    if omniglass_controls
                    else "exact_omnipbr_uv_controls"
                ),
                coordinate_source=binding.coordinate_source,
                projection=binding.projection,
            )
        elif omnisurface_controls:
            if (
                binding.scale != (1.0, 1.0)
                or binding.rotation_degrees
                or binding.translation != (0.0, 0.0)
            ):
                raise TexturedRenderError(
                    "OmniSurface and UsdTransform2d UV transforms are stacked"
                )
            binding = _omni_surface_uv_binding(active_values, available_uvs)
        wrap_mode, texture_scale, texture_bias = _texture_sampling(
            evidence, material_record
        )
        if simpbr_controls and role == "specular":
            texture_scale = tuple(value * 0.5 for value in texture_scale)
            texture_bias = tuple(value * 0.5 for value in texture_bias)
        planned_role = role
        flip_red, flip_green, flip_blue = _normal_channel_flips(
            role,
            source_path,
            material_record,
            material_family,
            active_values,
        )
        evidence_hash = _json_value_hash(evidence)
        geometry_udim_sampling = (
            _udim_geometry_sampling(role, source.tiles, binding, payload)
            if source.tiles
            else None
        )
        expanded_roles = (
            tuple(
                item
                for item, enabled in (
                    (("occlusion", "r"), ao_to_diffuse > 0.0),
                    (("roughness", "g"), roughness_texture_influence > 0.0),
                    (("metallic", "b"), metallic_texture_influence > 0.0),
                )
                if enabled
            )
            if role == "orm"
            else ((planned_role, channel),)
        )
        for expanded_role, expanded_channel in expanded_roles:
            texture_plans.append(
                TexturePlan(
                    role=expanded_role,
                    path=source.path,
                    file_sha256=source.file_sha256,
                    tiles=source.tiles,
                    channel=expanded_channel,
                    color_space=_texture_color_space(
                        planned_role,
                        evidence,
                        specular_is_color=(
                            omniglass_controls or omnisurface_controls
                        ),
                    ),
                    uv=binding,
                    wrap_mode=wrap_mode,
                    scale=texture_scale,
                    bias=texture_bias,
                    mono_mode=(
                        omni_opacity_mode
                        if omnipbr_controls and expanded_role == "opacity"
                        else None
                    ),
                    flip_red=flip_red,
                    flip_green=flip_green,
                    flip_blue=flip_blue,
                    normal_decode_semantics=(
                        "simpbr_all_abs_z"
                        if simpbr_controls or simpbr_translucent_controls
                        else "omnisurface_all"
                        if omnisurface_controls
                        else "omnipbr_xy"
                    )
                    if expanded_role in _SWATCH_TANGENT_NORMAL_ROLES
                    else None,
                    evidence_sha256=evidence_hash,
                    geometry_udim_sampling=geometry_udim_sampling,
                )
            )
    texture_by_role: dict[str, TexturePlan] = {}
    for texture in texture_plans:
        previous = texture_by_role.get(texture.role)
        if previous is not None and previous != texture:
            raise TexturedRenderError(
                f"active material has competing exact mappings for {texture.role!r}"
            )
        texture_by_role[texture.role] = texture
    texture_plans = [texture_by_role[role] for role in sorted(texture_by_role)]
    base_color = (
        tuple(model_parameters["glass_color"])
        if omniglass_controls
        else tuple(model_parameters["diffuse_color"])
        if omnisurface_controls
        else _unique_color(active_values, {"transmittancecolor"})
        if simpbr_translucent_controls
        else _constant_base_color(material_record)
        or _unique_color(
            active_values,
            {
                "diffusecolor",
                "diffusecolorconstant",
                "basecolor",
                "basecolorconstant",
                "albedo",
                "albedoconstant",
                "glasscolor",
            },
        )
    )
    has_albedo = any(item.role == "albedo" for item in texture_plans)
    if base_color is None:
        if has_albedo:
            base_color = (1.0, 1.0, 1.0, 1.0)
        else:
            raise TexturedRenderError(
                f"material {material_record_id!r} has neither an exact albedo texture nor "
                "an exact finite base-color constant"
            )
    roughness = (
        float(model_parameters["roughness"])
        if omniglass_controls
        else float(model_parameters["specular_roughness"])
        if omnisurface_controls
        else _unique_scalar(
            active_values,
            {
                "roughness",
                "roughnessconstant",
                "reflectionroughness",
                "reflectionroughnessconstant",
                "specularroughness",
            },
            0.5,
        )
    )
    metallic = (
        float(model_parameters["metalness"])
        if omnisurface_controls
        else _unique_scalar(
            active_values,
            {"metallic", "metallicconstant", "metalness", "metalnessconstant"},
            0.0,
        )
    )
    opacity = (
        float(model_parameters["opacity"])
        if omnisurface_controls
        else _unique_scalar(
            active_values,
            {
                "opacity",
                "opacityconstant",
                "alpha",
                "alphaconstant",
                "cutoutopacity",
                "geometryopacity",
            },
            1.0 if omnipbr_controls or omniglass_controls else base_color[3],
        )
    )
    opacity_threshold = (
        _unique_scalar(active_values, {"alphacutoutcutoff"}, 1.0)
        if simpbr_controls
        and _unique_bool(active_values, {"enableopacitycutout"}, False)
        else float(model_parameters["opacity_threshold"])
        if omnisurface_controls
        else _unique_scalar(
            active_values,
            {"opacitythreshold", "geometryopacitythreshold"},
            0.0,
        )
    )
    emission_color = _unique_color(
        active_values,
        {"emissivecolor", "emissioncolor", "emissivecolorconstant"},
    ) or (
        (1.0, 1.0, 0.1, 1.0)
        if omnipbr_controls
        else (0.0, 0.0, 0.0, 1.0)
    )
    emission_strength = _unique_scalar(
        active_values,
        {"emissiveintensity", "emissionstrength", "emissionintensity"},
        (
            40.0
            if omnipbr_controls
            else 1.0
            if any(item.role == "emissive" for item in texture_plans)
            else 0.0
        ),
    )
    if simpbr_controls and not any(
        item.role == "emissive_mask" for item in texture_plans
    ):
        mask_input = _unique_color(active_values, {"emissivemaskinput"}) or (
            1.0,
            1.0,
            1.0,
            1.0,
        )
        emission_color = tuple(
            emission_color[index] * mask_input[index] for index in range(4)
        )
    detail_normal_strength = _unique_scalar(
        active_values,
        {
            "detailnormalstrength",
            "detailnormalmapstrength",
            "detailnormalweight",
            "detailbumpfactor",
        },
        0.3 if omnipbr_controls else 1.0,
    )
    bump_strength = _unique_scalar(
        active_values,
        {"bumpstrength", "displacementstrength", "heightstrength"},
        1.0,
    )
    bump_distance = _unique_scalar(
        active_values,
        {"bumpdistance", "displacementscale", "heightscale"},
        1.0,
    )
    ior_aliases = {
        "ior",
        "indexofrefraction",
        "glassior",
        "specularior",
        "reflectionior",
    }
    specular_aliases = {
        "speculariorlevel",
        "specularlevel",
        "specularconstant",
        "specularweight",
        "reflectionweight",
    }
    transmission_aliases = {
        "transmissionweight",
        "speculartransmission",
        "speculartransmissionweight",
        "refractionweight",
    }
    coat_weight_aliases = {"coatweight", "clearcoat", "clearcoatweight"}
    coat_roughness_aliases = {"coatroughness", "clearcoatroughness"}
    coat_ior_aliases = {"coatior", "clearcoatior"}
    coat_tint_aliases = {"coattint", "clearcoattint", "clearcoatcolor"}
    sheen_weight_aliases = {"sheen", "sheenweight"}
    sheen_roughness_aliases = {"sheenroughness"}
    sheen_tint_aliases = {"sheentint", "sheencolor"}
    anisotropic_aliases = {"anisotropic", "anisotropy", "anisotropicweight"}
    anisotropic_rotation_aliases = {"anisotropicrotation", "anisotropyrotation"}
    subsurface_weight_aliases = {
        "subsurface",
        "subsurfaceweight",
        "subsurfaceamount",
    }
    subsurface_radius_aliases = {"subsurfaceradius", "scatteringradius"}
    subsurface_scale_aliases = {"subsurfacescale", "scatteringdistance"}
    subsurface_ior_aliases = {"subsurfaceior"}
    subsurface_anisotropy_aliases = {"subsurfaceanisotropy"}
    thin_film_thickness_aliases = {"thinfilmthickness"}
    thin_film_ior_aliases = {"thinfilmior"}
    thin_walled_aliases = {"thinwalled", "thinwall", "enablethinwalled"}
    transmission_enable_aliases = {
        "enablespeculartransmission",
        "enabletransmission",
        "enablerefraction",
    }
    coat_enable_aliases = {"enableclearcoat", "enablecoat"}
    sheen_enable_aliases = {"enablesheen"}
    subsurface_enable_aliases = {"enablesubsurface", "enablescattering"}
    subsurface_enable_aliases.add("enablediffusetransmission")
    emission_enable_aliases = {"enableemission", "enableemissive"}
    opacity_enable_aliases = {"enableopacity"}

    if material_family == "omnipbr":
        unsupported_plain_omnipbr_aliases = {
            *ior_aliases,
            *transmission_aliases,
            *coat_weight_aliases,
            *coat_roughness_aliases,
            *coat_ior_aliases,
            *coat_tint_aliases,
            *sheen_weight_aliases,
            *sheen_roughness_aliases,
            *sheen_tint_aliases,
            *anisotropic_aliases,
            *anisotropic_rotation_aliases,
            *subsurface_weight_aliases,
            *subsurface_radius_aliases,
            *subsurface_scale_aliases,
            *subsurface_ior_aliases,
            *subsurface_anisotropy_aliases,
            *thin_film_thickness_aliases,
            *thin_film_ior_aliases,
            *thin_walled_aliases,
            *transmission_enable_aliases,
            *coat_enable_aliases,
            *sheen_enable_aliases,
            *subsurface_enable_aliases,
            "bumpstrength",
            "bumpdistance",
            "displacementstrength",
            "displacementscale",
            "heightstrength",
            "heightscale",
        }
        unsupported_authored = {
            name: values
            for name in sorted(unsupported_plain_omnipbr_aliases)
            for values in [active_values.get(name, [])]
            if any(nonzero(value) for value in values)
        }
        if unsupported_authored:
            raise TexturedRenderError(
                "plain OmniPBR root has active inputs from another material model: "
                f"{unsupported_authored!r}"
            )

    if omniglass_controls:
        ior = float(model_parameters["ior"])
        specular_ior_level = 0.5
    elif omnisurface_controls:
        ior = float(model_parameters["specular_ior"])
        specular_ior_level = 0.5
    elif simpbr_translucent_controls:
        ior = _unique_scalar(active_values, {"iorconstant"}, 1.5)
        specular_ior_level = 0.5
    elif simpbr_controls:
        sqrt_f0 = math.sqrt(0.03)
        ior = (1.0 + sqrt_f0) / (1.0 - sqrt_f0)
        # The MDL declaration calls this specular_constant while SimReady
        # instances author the corresponding interface as specular_level.
        # Authored interface values override declaration defaults.
        authored_specular_values = {
            name: values
            for name, values in usd_active_values.items()
            if name in specular_aliases
        }
        specular_ior_level = 0.5 * _unique_scalar(
            authored_specular_values or active_values,
            specular_aliases,
            1.0,
        )
    else:
        ior = _unique_scalar(active_values, ior_aliases, 1.5)
        specular_ior_level = _unique_scalar(
            active_values, specular_aliases, 0.5
        )
    transmission_weight = (
        1.0
        if omniglass_controls or simpbr_translucent_controls
        else float(model_parameters["transmission_weight"])
        if omnisurface_controls
        else _unique_scalar(active_values, transmission_aliases, 0.0)
    )
    if not (
        omniglass_controls
        or omnisurface_controls
        or simpbr_translucent_controls
    ) and not _unique_bool(
        active_values,
        transmission_enable_aliases,
        transmission_weight > 0.0,
    ):
        transmission_weight = 0.0
    coat_weight = _unique_scalar(active_values, coat_weight_aliases, 0.0)
    if not _unique_bool(active_values, coat_enable_aliases, coat_weight > 0.0):
        coat_weight = 0.0
    coat_roughness = _unique_scalar(active_values, coat_roughness_aliases, 0.03)
    coat_ior = _unique_scalar(active_values, coat_ior_aliases, 1.5)
    coat_tint = _unique_color(active_values, coat_tint_aliases) or (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    sheen_weight = _unique_scalar(active_values, sheen_weight_aliases, 0.0)
    if not _unique_bool(active_values, sheen_enable_aliases, sheen_weight > 0.0):
        sheen_weight = 0.0
    sheen_roughness = _unique_scalar(active_values, sheen_roughness_aliases, 0.5)
    sheen_tint = _unique_color(active_values, sheen_tint_aliases) or (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    anisotropic = _unique_scalar(active_values, anisotropic_aliases, 0.0)
    anisotropic_rotation = _unique_scalar(
        active_values, anisotropic_rotation_aliases, 0.0
    )
    subsurface_weight = (
        float(model_parameters["subsurface_weight"])
        if omnisurface_controls
        else _unique_scalar(active_values, subsurface_weight_aliases, 0.0)
    )
    if not omnisurface_controls and not _unique_bool(
        active_values, subsurface_enable_aliases, subsurface_weight > 0.0
    ):
        subsurface_weight = 0.0
    subsurface_radius = _unique_vector3(
        active_values, subsurface_radius_aliases, (1.0, 0.2, 0.1)
    )
    subsurface_scale = _unique_scalar(active_values, subsurface_scale_aliases, 0.05)
    subsurface_ior = _unique_scalar(active_values, subsurface_ior_aliases, 1.4)
    subsurface_anisotropy = _unique_scalar(
        active_values, subsurface_anisotropy_aliases, 0.0
    )
    thin_film_thickness = _unique_scalar(
        active_values, thin_film_thickness_aliases, 0.0
    )
    thin_film_ior = _unique_scalar(active_values, thin_film_ior_aliases, 1.33)
    thin_walled = (
        bool(model_parameters["thin_walled"])
        if omniglass_controls or omnisurface_controls
        else _unique_bool(active_values, thin_walled_aliases, False)
    )
    albedo_desaturation = _unique_scalar(
        active_values, {"albedodesaturation"}, 0.0
    )
    albedo_add = _unique_scalar(active_values, {"albedoadd"}, 0.0)
    albedo_brightness = _unique_scalar(
        active_values, {"albedobrightness"}, 1.0
    )
    diffuse_tint = _unique_color(active_values, {"diffusetint"}) or (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    normal_strength = (
        float(model_parameters["normal_strength"])
        if omniglass_controls or omnisurface_controls
        else _unique_scalar(
            active_values,
            {"normalmapstrength", "normalstrength", "bumpfactor"},
            1.0,
        )
    )
    if not (
        omni_enable_emission
        if omnipbr_controls or simpbr_translucent_controls
        else _unique_bool(active_values, emission_enable_aliases, emission_strength > 0.0)
    ):
        emission_strength = 0.0
    if not (
        True
        if omnisurface_controls
        else omni_enable_opacity
        if omnipbr_controls or omniglass_controls
        else _unique_bool(active_values, opacity_enable_aliases, opacity < 1.0)
    ):
        opacity = 1.0

    parameter_aliases = {
        *mdl_defaults,
        "diffusecolor",
        "diffusecolorconstant",
        "basecolor",
        "basecolorconstant",
        "albedo",
        "albedoconstant",
        "glasscolor",
        "roughness",
        "roughnessconstant",
        "reflectionroughness",
        "reflectionroughnessconstant",
        "specularroughness",
        "metallic",
        "metallicconstant",
        "metalness",
        "metalnessconstant",
        "opacity",
        "opacityconstant",
        "alpha",
        "alphaconstant",
        "alphacutoutcutoff",
        "cutoutopacity",
        "emissivecolor",
        "emissioncolor",
        "emissivecolorconstant",
        "emissiveintensity",
        "emissionstrength",
        "emissionintensity",
        "detailnormalstrength",
        "detailnormalmapstrength",
        "detailnormalweight",
        "detailbumpfactor",
        "bumpstrength",
        "displacementstrength",
        "heightstrength",
        "bumpdistance",
        "displacementscale",
        "heightscale",
        *ior_aliases,
        *specular_aliases,
        *transmission_aliases,
        *coat_weight_aliases,
        *coat_roughness_aliases,
        *coat_ior_aliases,
        *coat_tint_aliases,
        *sheen_weight_aliases,
        *sheen_roughness_aliases,
        *sheen_tint_aliases,
        *anisotropic_aliases,
        *anisotropic_rotation_aliases,
        *subsurface_weight_aliases,
        *subsurface_radius_aliases,
        *subsurface_scale_aliases,
        *subsurface_ior_aliases,
        *subsurface_anisotropy_aliases,
        *thin_film_thickness_aliases,
        *thin_film_ior_aliases,
        *thin_walled_aliases,
        *transmission_enable_aliases,
        *coat_enable_aliases,
        *sheen_enable_aliases,
        *subsurface_enable_aliases,
        *emission_enable_aliases,
        *opacity_enable_aliases,
        "enableopacitytexture",
        "enableopacitymap",
        "opacitymode",
        "opacitythreshold",
        "albedodesaturation",
        "albedoadd",
        "albedobrightness",
        "diffusetint",
        "aotodiffuse",
        "reflectionroughnesstextureinfluence",
        "roughnesstextureinfluence",
        "metallictextureinfluence",
        "metalnesstextureinfluence",
        "normalmapstrength",
        "normalstrength",
        "bumpfactor",
        "fliptangentu",
        "fliptangentv",
        "normalmapfliptangentu",
        "normalmapfliptangentv",
        "projectuvw",
        "worldorobject",
        "uvspaceindex",
        "texturescale",
        "texturetranslate",
        "texturerotate",
        "detailtexturescale",
        "detailtexturetranslate",
        "detailtexturerotate",
        "enableormtexture",
    }
    evidence_property_paths = {
        str(item.get("semantic_consumer_property_path") or item.get("property_path") or "")
        for item in material_record.get("surface_texture_evidence") or []
        if isinstance(item, Mapping)
    } | {
        str(item.get("shader_path") or "") + ".inputs:" + str(item.get("input_name") or "")
        for item in material_record.get("surface_texture_evidence") or []
        if isinstance(item, Mapping)
    }
    technical_names = {
        "file",
        "st",
        "uv",
        "texcoord",
        "varname",
        "wraps",
        "wrapt",
        "scale",
        "bias",
        "rotation",
        "translation",
        "sourcecolorspace",
        "fallback",
    }
    appearance_tokens = (
        "color",
        "rough",
        "metal",
        "specular",
        "transmission",
        "refraction",
        "coat",
        "sheen",
        "anisotrop",
        "subsurface",
        "scatter",
        "emiss",
        "opacity",
        "alpha",
        "normal",
        "bump",
        "height",
        "displacement",
        "ior",
        "thin",
        "texture",
    )
    unconsumed = []
    for item in _active_authored_input_records(material_record):
        name = _normalized_name(item.get("name") or item.get("full_name"))
        path = str(item.get("property_path") or "")
        if name in parameter_aliases or name in technical_names or path in evidence_property_paths:
            continue
        if not nonzero(item.get("value")):
            continue
        if name.startswith("enable") and not _unique_bool({name: [item.get("value")]}, {name}, False):
            continue
        if any(token in name for token in appearance_tokens):
            unconsumed.append(
                {"property_path": path, "name": name, "value": item.get("value")}
            )
    if unconsumed:
        raise TexturedRenderError(
            f"material {material_record_id!r} has active authored appearance inputs "
            f"without an exact Principled mapping: {unconsumed!r}"
        )
    for label, value in {
        "roughness": roughness,
        "metallic": metallic,
        "opacity": opacity,
        "opacity_threshold": opacity_threshold,
        "specular_ior_level": specular_ior_level,
        "transmission_weight": transmission_weight,
        "coat_weight": coat_weight,
        "coat_roughness": coat_roughness,
        "sheen_weight": sheen_weight,
        "sheen_roughness": sheen_roughness,
        "anisotropic": anisotropic,
        "subsurface_weight": subsurface_weight,
        "subsurface_anisotropy": subsurface_anisotropy,
        "albedo_desaturation": albedo_desaturation,
        "ao_to_diffuse": ao_to_diffuse,
        "roughness_texture_influence": roughness_texture_influence,
        "metallic_texture_influence": metallic_texture_influence,
        "emission_strength": emission_strength,
    }.items():
        unit_interval = label != "emission_strength"
        if value < 0.0 or (unit_interval and value > 1.0):
            raise TexturedRenderError(
                f"material {material_record_id!r} has out-of-range {label}={value}"
            )
    for label, value in {
        "ior": ior,
        "coat_ior": coat_ior,
        "subsurface_ior": subsurface_ior,
        "thin_film_ior": thin_film_ior,
        "albedo_brightness": albedo_brightness,
        "bump_strength": bump_strength,
        "bump_distance": bump_distance,
    }.items():
        if value < 0.0 or not math.isfinite(value):
            raise TexturedRenderError(
                f"material {material_record_id!r} has invalid {label}={value}"
            )
    for label, value in {
        "normal_strength": normal_strength,
        "detail_normal_strength": detail_normal_strength,
    }.items():
        if not math.isfinite(value) or (
            value < 0.0 and material_family != "omnipbr"
        ):
            raise TexturedRenderError(
                f"material {material_record_id!r} has invalid {label}={value}"
            )
    material_name = str(material_record.get("material_name") or material_record_id)
    core = {
        "material_record_id": material_record_id,
        "material_name": material_name,
        "material_family": material_family,
        "material_model": material_model,
        "model_parameters": model_parameters,
        "textures": [item.record() for item in texture_plans],
        "base_color": list(base_color),
        "roughness": roughness,
        "metallic": metallic,
        "opacity": opacity,
        "opacity_threshold": opacity_threshold,
        "emission_color": list(emission_color),
        "emission_strength": emission_strength,
        "double_sided": double_sided,
        "display_color_attribute": None,
        "display_opacity_attribute": None,
        "detail_normal_strength": detail_normal_strength,
        "bump_strength": bump_strength,
        "bump_distance": bump_distance,
        "ior": ior,
        "specular_ior_level": specular_ior_level,
        "transmission_weight": transmission_weight,
        "coat_weight": coat_weight,
        "coat_roughness": coat_roughness,
        "coat_ior": coat_ior,
        "coat_tint": list(coat_tint),
        "sheen_weight": sheen_weight,
        "sheen_roughness": sheen_roughness,
        "sheen_tint": list(sheen_tint),
        "anisotropic": anisotropic,
        "anisotropic_rotation": anisotropic_rotation,
        "subsurface_weight": subsurface_weight,
        "subsurface_radius": list(subsurface_radius),
        "subsurface_scale": subsurface_scale,
        "subsurface_ior": subsurface_ior,
        "subsurface_anisotropy": subsurface_anisotropy,
        "thin_film_thickness": thin_film_thickness,
        "thin_film_ior": thin_film_ior,
        "thin_walled": thin_walled,
        "albedo_desaturation": albedo_desaturation,
        "albedo_add": albedo_add,
        "albedo_brightness": albedo_brightness,
        "diffuse_tint": list(diffuse_tint),
        "ao_to_diffuse": ao_to_diffuse,
        "roughness_texture_influence": roughness_texture_influence,
        "metallic_texture_influence": metallic_texture_influence,
        "normal_strength": normal_strength,
        "omnipbr_controls": omnipbr_controls,
        "simpbr_controls": simpbr_controls,
        "simpbr_orm_gate_repair": simpbr_orm_gate_repair,
        "active_graph_sha256": _json_value_hash(graph),
        "selected_mdl_material_semantics": mdl_semantics,
    }
    provenance = _canonical_hash("principled-material-plan", core)
    return MaterialPlan(
        material_record_id=material_record_id,
        material_name=material_name,
        material_model=material_model,
        model_parameters=model_parameters,
        textures=tuple(texture_plans),
        base_color=base_color,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
        opacity_threshold=opacity_threshold,
        emission_color=emission_color,
        emission_strength=emission_strength,
        double_sided=double_sided,
        display_color_attribute=None,
        display_opacity_attribute=None,
        detail_normal_strength=detail_normal_strength,
        bump_strength=bump_strength,
        bump_distance=bump_distance,
        ior=ior,
        specular_ior_level=specular_ior_level,
        transmission_weight=transmission_weight,
        coat_weight=coat_weight,
        coat_roughness=coat_roughness,
        coat_ior=coat_ior,
        coat_tint=coat_tint,
        sheen_weight=sheen_weight,
        sheen_roughness=sheen_roughness,
        sheen_tint=sheen_tint,
        anisotropic=anisotropic,
        anisotropic_rotation=anisotropic_rotation,
        subsurface_weight=subsurface_weight,
        subsurface_radius=subsurface_radius,
        subsurface_scale=subsurface_scale,
        subsurface_ior=subsurface_ior,
        subsurface_anisotropy=subsurface_anisotropy,
        thin_film_thickness=thin_film_thickness,
        thin_film_ior=thin_film_ior,
        thin_walled=thin_walled,
        albedo_desaturation=albedo_desaturation,
        albedo_add=albedo_add,
        albedo_brightness=albedo_brightness,
        diffuse_tint=diffuse_tint,
        ao_to_diffuse=ao_to_diffuse,
        roughness_texture_influence=roughness_texture_influence,
        metallic_texture_influence=metallic_texture_influence,
        normal_strength=normal_strength,
        omnipbr_controls=omnipbr_controls,
        simpbr_controls=simpbr_controls,
        simpbr_orm_gate_repair=simpbr_orm_gate_repair,
        provenance_sha256=provenance,
    )


def _validate_payloads(
    extraction: Mapping[str, Any],
    geometry_payloads: Mapping[str, SegmentGeometryPayload],
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str]]:
    segments = extraction.get("segments")
    materials = extraction.get("materials")
    if not isinstance(segments, Mapping) or not segments:
        raise TexturedRenderError("extraction has no segment mapping")
    if not isinstance(materials, Mapping):
        raise TexturedRenderError("extraction has no material-record mapping")
    if set(geometry_payloads) != set(segments):
        missing = sorted(set(segments) - set(geometry_payloads))
        extra = sorted(set(geometry_payloads) - set(segments))
        raise TexturedRenderError(
            f"exact geometry payload coverage mismatch; missing={missing!r}, extra={extra!r}"
        )
    payload_hashes: dict[str, str] = {}
    segment_ids: set[str] = set()
    for key, payload in geometry_payloads.items():
        if not isinstance(payload, SegmentGeometryPayload):
            raise TexturedRenderError(f"geometry payload for {key!r} has the wrong type")
        segment = segments[key]
        if not isinstance(segment, Mapping):
            raise TexturedRenderError(f"segment record for {key!r} is malformed")
        segment_id = str(segment.get("segment_id") or "")
        if not segment_id or segment_id in segment_ids:
            raise TexturedRenderError(f"segment ID is missing or duplicated for {key!r}")
        segment_ids.add(segment_id)
        if str(payload.provenance.get("segment_id") or "") != segment_id:
            raise TexturedRenderError(f"geometry payload identity mismatch for {key!r}")
        manifest = payload.manifest()
        payload_hash = str(manifest.get("payload_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", payload_hash):
            raise TexturedRenderError(f"geometry payload hash is invalid for {key!r}")
        payload_hashes[str(key)] = payload_hash
    return segments, materials, payload_hashes


def _image_for_texture(
    bpy: Any,
    texture: TexturePlan,
    cache: dict[tuple[str, str, str], Any],
) -> Any:
    key = (texture.path, texture.file_sha256, texture.color_space)
    if key in cache:
        return cache[key]
    if texture.tiles:
        for tile in texture.tiles:
            probe = bpy.data.images.load(tile.path, check_existing=False)
            if probe is None or int(probe.size[0]) < 1 or int(probe.size[1]) < 1:
                raise TexturedRenderError(
                    f"Blender could not decode exact UDIM tile {tile.path}"
                )
            bpy.data.images.remove(probe)
        image = bpy.data.images.load(texture.tiles[0].path, check_existing=False)
        image.source = "TILED"
        image.filepath = texture.path
        expected_tiles = {tile.tile_id for tile in texture.tiles}
        for tile_id in sorted(expected_tiles):
            if tile_id not in {tile.number for tile in image.tiles}:
                image.tiles.new(tile_id)
        image.reload()
        if image.source != "TILED" or {
            tile.number for tile in image.tiles
        } != expected_tiles:
            raise TexturedRenderError(
                f"Blender did not retain exact UDIM tiles for {texture.path}"
            )
    else:
        image = bpy.data.images.load(texture.path, check_existing=False)
    if image is None or int(image.size[0]) < 1 or int(image.size[1]) < 1:
        raise TexturedRenderError(f"Blender could not decode exact texture {texture.path}")
    image.name = f"vlm_{texture.file_sha256[:16]}_{texture.color_space}"
    image.colorspace_settings.name = texture.color_space
    if image.colorspace_settings.name != texture.color_space:
        raise TexturedRenderError(
            f"Blender did not retain {texture.color_space!r} for {texture.path}"
        )
    image.alpha_mode = "STRAIGHT"
    cache[key] = image
    return image


def _math_node(nodes: Any, operation: str, left: Any, right: float | Any) -> Any:
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    if hasattr(left, "bl_idname"):
        node.id_data.links.new(left, node.inputs[0])
    else:
        node.inputs[0].default_value = float(left)
    if hasattr(right, "bl_idname"):
        node.id_data.links.new(right, node.inputs[1])
    else:
        node.inputs[1].default_value = float(right)
    return node.outputs[0]


def _coordinate_space_normal(nodes: Any, links: Any, socket: Any, space: str) -> Any:
    if space == "world":
        return socket
    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "NORMAL"
    transform.convert_from = "WORLD"
    transform.convert_to = "OBJECT"
    links.new(socket, transform.inputs["Vector"])
    return transform.outputs["Vector"]


def _coordinate_vector_to_world(
    nodes: Any, links: Any, socket: Any, space: str
) -> Any:
    if space == "world":
        return socket
    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "VECTOR"
    transform.convert_from = "OBJECT"
    transform.convert_to = "WORLD"
    links.new(socket, transform.inputs["Vector"])
    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    links.new(transform.outputs["Vector"], normalize.inputs[0])
    return normalize.outputs["Vector"]


def _vector_binary(
    nodes: Any, links: Any, operation: str, left: Any, right: Any
) -> Any:
    node = nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    links.new(left, node.inputs[0])
    links.new(right, node.inputs[1])
    return node.outputs[0]


def _vector_scale(nodes: Any, links: Any, vector: Any, scale: Any) -> Any:
    node = nodes.new("ShaderNodeVectorMath")
    node.operation = "SCALE"
    links.new(vector, node.inputs[0])
    if hasattr(scale, "bl_idname"):
        links.new(scale, node.inputs[3])
    else:
        node.inputs[3].default_value = float(scale)
    return node.outputs[0]


def _vector_normalize(nodes: Any, links: Any, vector: Any) -> Any:
    node = nodes.new("ShaderNodeVectorMath")
    node.operation = "NORMALIZE"
    links.new(vector, node.inputs[0])
    return node.outputs[0]


def _vector_mix(
    nodes: Any, links: Any, first: Any, second: Any, use_second: Any
) -> Any:
    return _vector_binary(
        nodes,
        links,
        "ADD",
        _vector_scale(
            nodes,
            links,
            first,
            _math_node(nodes, "SUBTRACT", 1.0, use_second),
        ),
        _vector_scale(nodes, links, second, use_second),
    )


def _mdl_transform_tangent_coefficients(
    binding: UVBinding,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if binding.transform_semantics not in {
        "mdl_local_scale_rotate",
        "mdl_base_rotation_translation_scale",
    }:
        raise TexturedRenderError(
            "texture transform does not carry an MDL tangent frame"
        )
    row_u, row_v, _ = _uv_affine(binding)
    tangent_u = np.asarray((row_v[1], -row_v[0]), dtype=np.float64)
    length = float(np.linalg.norm(tangent_u))
    if not math.isfinite(length) or length <= 1.0e-15:
        raise TexturedRenderError("texture transform has no usable tangent frame")
    tangent_u /= length
    tangent_v = np.asarray((-tangent_u[1], tangent_u[0]), dtype=np.float64)
    if float(np.dot(np.asarray(row_u), tangent_u)) < 0.0:
        tangent_u *= -1.0
    return (
        (float(tangent_u[0]), float(tangent_u[1])),
        (float(tangent_v[0]), float(tangent_v[1])),
    )


def _mdl_cubic_projection_sockets(
    nodes: Any, links: Any, binding: UVBinding
) -> tuple[Any, Any, Any]:
    geometry = nodes.new("ShaderNodeNewGeometry")
    if binding.coordinate_source == "world":
        position = geometry.outputs["Position"]
    elif binding.coordinate_source == "object":
        coordinates = nodes.new("ShaderNodeTexCoord")
        position = coordinates.outputs["Object"]
    else:
        raise TexturedRenderError(
            "MDL cubic projection requires object or world coordinates"
        )
    normal = _coordinate_space_normal(
        nodes,
        links,
        geometry.outputs["Normal"],
        binding.coordinate_source,
    )
    geometric_normal = _coordinate_space_normal(
        nodes,
        links,
        geometry.outputs["True Normal"],
        binding.coordinate_source,
    )

    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    links.new(position, dot.inputs[0])
    links.new(geometric_normal, dot.inputs[1])
    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    links.new(position, length.inputs[0])
    threshold = _math_node(
        nodes,
        "MINIMUM",
        _math_node(nodes, "MULTIPLY", length.outputs["Value"], 1.0e-9),
        1.0e-5,
    )
    flip = _math_node(nodes, "LESS_THAN", dot.outputs["Value"], threshold)
    normal_scale = _math_node(
        nodes,
        "SUBTRACT",
        1.0,
        _math_node(nodes, "MULTIPLY", flip, 2.0),
    )
    adjusted_normal = nodes.new("ShaderNodeVectorMath")
    adjusted_normal.operation = "SCALE"
    links.new(normal, adjusted_normal.inputs[0])
    links.new(normal_scale, adjusted_normal.inputs[3])

    normal_components = nodes.new("ShaderNodeSeparateXYZ")
    links.new(adjusted_normal.outputs["Vector"], normal_components.inputs["Vector"])
    position_components = nodes.new("ShaderNodeSeparateXYZ")
    links.new(position, position_components.inputs["Vector"])
    nx = normal_components.outputs["X"]
    ny = normal_components.outputs["Y"]
    nz = normal_components.outputs["Z"]
    px = position_components.outputs["X"]
    py = position_components.outputs["Y"]
    pz = position_components.outputs["Z"]
    ax = _math_node(nodes, "ABSOLUTE", nx, 0.0)
    ay = _math_node(nodes, "ABSOLUTE", ny, 0.0)
    az = _math_node(nodes, "ABSOLUTE", nz, 0.0)

    def greater_or_equal(left: Any, right: Any) -> Any:
        return _math_node(
            nodes,
            "SUBTRACT",
            1.0,
            _math_node(nodes, "LESS_THAN", left, right),
        )

    x_axis = _math_node(
        nodes,
        "MULTIPLY",
        greater_or_equal(ax, ay),
        greater_or_equal(ax, az),
    )
    y_major = _math_node(
        nodes,
        "MULTIPLY",
        greater_or_equal(ay, ax),
        greater_or_equal(ay, az),
    )
    y_axis = _math_node(
        nodes,
        "MULTIPLY",
        _math_node(nodes, "SUBTRACT", 1.0, x_axis),
        y_major,
    )
    z_axis = _math_node(
        nodes,
        "SUBTRACT",
        _math_node(nodes, "SUBTRACT", 1.0, x_axis),
        y_axis,
    )

    x_sign = _math_node(
        nodes,
        "SUBTRACT",
        _math_node(
            nodes,
            "MULTIPLY",
            _math_node(nodes, "GREATER_THAN", nx, 0.0),
            2.0,
        ),
        1.0,
    )
    y_sign = _math_node(
        nodes,
        "SUBTRACT",
        1.0,
        _math_node(
            nodes,
            "MULTIPLY",
            _math_node(nodes, "GREATER_THAN", ny, 0.0),
            2.0,
        ),
    )
    z_sign = _math_node(
        nodes,
        "SUBTRACT",
        1.0,
        _math_node(
            nodes,
            "MULTIPLY",
            _math_node(nodes, "LESS_THAN", nz, 0.0),
            2.0,
        ),
    )
    x_u = _math_node(nodes, "MULTIPLY", py, x_sign)
    y_u = _math_node(nodes, "MULTIPLY", px, y_sign)
    z_u = _math_node(nodes, "MULTIPLY", px, z_sign)
    u = _math_node(
        nodes,
        "ADD",
        _math_node(nodes, "MULTIPLY", x_axis, x_u),
        _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", y_axis, y_u),
            _math_node(nodes, "MULTIPLY", z_axis, z_u),
        ),
    )
    v = _math_node(
        nodes,
        "ADD",
        _math_node(
            nodes,
            "MULTIPLY",
            _math_node(nodes, "ADD", x_axis, y_axis),
            pz,
        ),
        _math_node(nodes, "MULTIPLY", z_axis, py),
    )

    row_u, row_v, translation = _uv_affine(binding)
    transformed_u = _math_node(
        nodes,
        "ADD",
        _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", u, row_u[0]),
            _math_node(nodes, "MULTIPLY", v, row_u[1]),
        ),
        translation[0],
    )
    transformed_v = _math_node(
        nodes,
        "ADD",
        _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", u, row_v[0]),
            _math_node(nodes, "MULTIPLY", v, row_v[1]),
        ),
        translation[1],
    )
    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(transformed_u, combine.inputs["X"])
    links.new(transformed_v, combine.inputs["Y"])

    projected_u = nodes.new("ShaderNodeCombineXYZ")
    links.new(
        _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", y_axis, y_sign),
            _math_node(nodes, "MULTIPLY", z_axis, z_sign),
        ),
        projected_u.inputs["X"],
    )
    links.new(
        _math_node(nodes, "MULTIPLY", x_axis, x_sign),
        projected_u.inputs["Y"],
    )
    projected_v = nodes.new("ShaderNodeCombineXYZ")
    links.new(z_axis, projected_v.inputs["Y"])
    links.new(
        _math_node(nodes, "ADD", x_axis, y_axis),
        projected_v.inputs["Z"],
    )
    u_direction = _coordinate_vector_to_world(
        nodes,
        links,
        projected_u.outputs["Vector"],
        binding.coordinate_source,
    )
    v_direction = _coordinate_vector_to_world(
        nodes,
        links,
        projected_v.outputs["Vector"],
        binding.coordinate_source,
    )
    shading_normal = geometry.outputs["Normal"]
    tangent_u = _vector_normalize(
        nodes,
        links,
        _vector_binary(
            nodes, links, "CROSS_PRODUCT", v_direction, shading_normal
        ),
    )
    tangent_v = _vector_normalize(
        nodes,
        links,
        _vector_binary(
            nodes, links, "CROSS_PRODUCT", shading_normal, tangent_u
        ),
    )
    def orient_tangent(tangent: Any, direction: Any) -> Any:
        alignment = nodes.new("ShaderNodeVectorMath")
        alignment.operation = "DOT_PRODUCT"
        links.new(tangent, alignment.inputs[0])
        links.new(direction, alignment.inputs[1])
        sign = _math_node(
            nodes,
            "SUBTRACT",
            1.0,
            _math_node(
                nodes,
                "MULTIPLY",
                _math_node(nodes, "LESS_THAN", alignment.outputs["Value"], 0.0),
                2.0,
            ),
        )
        return _vector_scale(nodes, links, tangent, sign)

    tangent_u = orient_tangent(tangent_u, u_direction)
    tangent_v = orient_tangent(tangent_v, v_direction)

    source_normal_components = nodes.new("ShaderNodeSeparateXYZ")
    links.new(normal, source_normal_components.inputs["Vector"])
    snx = source_normal_components.outputs["X"]
    sny = source_normal_components.outputs["Y"]
    snz = source_normal_components.outputs["Z"]
    pole = _math_node(
        nodes,
        "GREATER_THAN",
        _math_node(nodes, "ABSOLUTE", snz, 0.0),
        0.99999,
    )

    def select_scalar(first: Any, second: Any, use_first: Any) -> Any:
        return _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", first, use_first),
            _math_node(
                nodes,
                "MULTIPLY",
                second,
                _math_node(nodes, "SUBTRACT", 1.0, use_first),
            ),
        )

    fallback_v_source = nodes.new("ShaderNodeCombineXYZ")
    links.new(
        select_scalar(
            _math_node(
                nodes,
                "MULTIPLY",
                _math_node(nodes, "MULTIPLY", snx, sny),
                -1.0,
            ),
            _math_node(
                nodes,
                "MULTIPLY",
                _math_node(nodes, "MULTIPLY", snx, snz),
                -1.0,
            ),
            pole,
        ),
        fallback_v_source.inputs["X"],
    )
    links.new(
        select_scalar(
            _math_node(
                nodes,
                "SUBTRACT",
                1.0,
                _math_node(nodes, "MULTIPLY", sny, sny),
            ),
            _math_node(
                nodes,
                "MULTIPLY",
                _math_node(nodes, "MULTIPLY", sny, snz),
                -1.0,
            ),
            pole,
        ),
        fallback_v_source.inputs["Y"],
    )
    links.new(
        select_scalar(
            _math_node(
                nodes,
                "MULTIPLY",
                _math_node(nodes, "MULTIPLY", sny, snz),
                -1.0,
            ),
            _math_node(
                nodes,
                "SUBTRACT",
                1.0,
                _math_node(nodes, "MULTIPLY", snz, snz),
            ),
            pole,
        ),
        fallback_v_source.inputs["Z"],
    )
    fallback_v_source_normalized = _vector_normalize(
        nodes, links, fallback_v_source.outputs["Vector"]
    )
    fallback_u_source = _vector_normalize(
        nodes,
        links,
        _vector_binary(
            nodes,
            links,
            "CROSS_PRODUCT",
            fallback_v_source_normalized,
            normal,
        ),
    )
    fallback_u = _coordinate_vector_to_world(
        nodes, links, fallback_u_source, binding.coordinate_source
    )
    fallback_v = _coordinate_vector_to_world(
        nodes, links, fallback_v_source_normalized, binding.coordinate_source
    )
    u_parallel = nodes.new("ShaderNodeVectorMath")
    u_parallel.operation = "DOT_PRODUCT"
    links.new(shading_normal, u_parallel.inputs[0])
    links.new(u_direction, u_parallel.inputs[1])
    v_parallel = nodes.new("ShaderNodeVectorMath")
    v_parallel.operation = "DOT_PRODUCT"
    links.new(shading_normal, v_parallel.inputs[0])
    links.new(v_direction, v_parallel.inputs[1])
    use_fallback = _math_node(
        nodes,
        "MAXIMUM",
        _math_node(
            nodes,
            "GREATER_THAN",
            _math_node(nodes, "ABSOLUTE", u_parallel.outputs["Value"], 0.0),
            0.999,
        ),
        _math_node(
            nodes,
            "GREATER_THAN",
            _math_node(nodes, "ABSOLUTE", v_parallel.outputs["Value"], 0.0),
            0.999,
        ),
    )
    tangent_u = _vector_mix(
        nodes, links, tangent_u, fallback_u, use_fallback
    )
    tangent_v = _vector_mix(
        nodes, links, tangent_v, fallback_v, use_fallback
    )
    transformed_tangent_u, transformed_tangent_v = (
        _mdl_transform_tangent_coefficients(binding)
    )
    source_tangent_u = tangent_u
    source_tangent_v = tangent_v
    tangent_u = _vector_normalize(
        nodes,
        links,
        _vector_binary(
            nodes,
            links,
            "ADD",
            _vector_scale(
                nodes, links, source_tangent_u, transformed_tangent_u[0]
            ),
            _vector_scale(
                nodes, links, source_tangent_v, transformed_tangent_u[1]
            ),
        ),
    )
    tangent_v = _vector_normalize(
        nodes,
        links,
        _vector_binary(
            nodes,
            links,
            "ADD",
            _vector_scale(
                nodes, links, source_tangent_u, transformed_tangent_v[0]
            ),
            _vector_scale(
                nodes, links, source_tangent_v, transformed_tangent_v[1]
            ),
        ),
    )
    return combine.outputs["Vector"], tangent_u, tangent_v


def _mdl_cubic_uv_socket(nodes: Any, links: Any, binding: UVBinding) -> Any:
    return _mdl_cubic_projection_sockets(nodes, links, binding)[0]


def _uv_socket(nodes: Any, links: Any, binding: UVBinding, uv_override: str | None) -> Any:
    if binding.projection == "BOX":
        return _mdl_cubic_uv_socket(nodes, links, binding)
    if binding.coordinate_source == "uv":
        uv = nodes.new("ShaderNodeUVMap")
        uv.uv_map = uv_override or binding.name
        vector = uv.outputs["UV"]
    elif binding.coordinate_source == "object":
        coordinates = nodes.new("ShaderNodeTexCoord")
        vector = coordinates.outputs["Object"]
    elif binding.coordinate_source == "world":
        geometry = nodes.new("ShaderNodeNewGeometry")
        vector = geometry.outputs["Position"]
    elif binding.coordinate_source == "constant_zero":
        coordinates = nodes.new("ShaderNodeCombineXYZ")
        vector = coordinates.outputs["Vector"]
    else:
        raise TexturedRenderError(
            f"unsupported exact texture coordinate source {binding.coordinate_source!r}"
        )
    row_u, row_v, translation = _uv_affine(binding)
    if (
        row_u == (1.0, 0.0)
        and row_v == (0.0, 1.0)
        and translation == (0.0, 0.0)
    ):
        return vector
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(vector, separate.inputs["Vector"])
    transformed_u = _math_node(
        nodes,
        "ADD",
        _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", separate.outputs["X"], row_u[0]),
            _math_node(nodes, "MULTIPLY", separate.outputs["Y"], row_u[1]),
        ),
        translation[0],
    )
    transformed_v = _math_node(
        nodes,
        "ADD",
        _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", separate.outputs["X"], row_v[0]),
            _math_node(nodes, "MULTIPLY", separate.outputs["Y"], row_v[1]),
        ),
        translation[1],
    )
    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(transformed_u, combine.inputs["X"])
    links.new(transformed_v, combine.inputs["Y"])
    return combine.outputs["Vector"]


def _baked_swatch_sockets(
    nodes: Any,
    links: Any,
    swatch: _BakedSwatch,
) -> tuple[Any, Any]:
    node = nodes.new("ShaderNodeTexImage")
    node.name = f"segment_swatch_{swatch.role}_{swatch.pixels_sha256[:12]}"
    node.image = swatch.image
    node.extension = "EXTEND"
    node.interpolation = "Linear"
    node.projection = "FLAT"
    vector = _uv_socket(
        nodes,
        links,
        UVBinding(name="SphereConformalUV"),
        "SphereConformalUV",
    )
    links.new(vector, node.inputs["Vector"])
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    links.new(node.outputs["Color"], separate.inputs["Color"])
    return node.outputs["Color"], separate.outputs["Red"]


def _texture_sockets(
    bpy: Any,
    nodes: Any,
    links: Any,
    texture: TexturePlan,
    image_cache: dict[tuple[str, str, str], Any],
    uv_override: str | None,
    standardized_sphere: bool,
    swatch: _BakedSwatch | None = None,
) -> tuple[Any, Any]:
    if swatch is not None:
        if not standardized_sphere or uv_override != "SphereConformalUV":
            raise TexturedRenderError(
                "a baked segment swatch may only drive its standardized sphere"
            )
        if swatch.role != texture.role or swatch.kind not in {
            "color",
            "scalar",
            "tangent_normal",
        }:
            raise TexturedRenderError(
                f"invalid baked segment swatch for texture role {texture.role!r}"
            )
        return _baked_swatch_sockets(nodes, links, swatch)
    if standardized_sphere:
        raise TexturedRenderError(
            f"standardized sphere texture {texture.role!r} lacks a selected-segment swatch"
        )
    node = nodes.new("ShaderNodeTexImage")
    node.name = f"{texture.role}_{texture.file_sha256[:12]}"
    node.image = _image_for_texture(bpy, texture, image_cache)
    node.extension = texture.wrap_mode
    node.interpolation = "Linear"
    node.projection = "FLAT" if texture.uv.projection == "BOX" else texture.uv.projection
    vector = _uv_socket(nodes, links, texture.uv, uv_override)
    links.new(vector, node.inputs["Vector"])
    color = node.outputs["Color"]
    alpha = node.outputs["Alpha"]
    if texture.scale[:3] != (1.0, 1.0, 1.0):
        color = _multiply_color(nodes, links, color, texture.scale)
    if texture.bias[:3] != (0.0, 0.0, 0.0):
        add = nodes.new("ShaderNodeMixRGB")
        add.blend_type = "ADD"
        add.inputs[0].default_value = 1.0
        links.new(color, add.inputs[1])
        add.inputs[2].default_value = texture.bias[:3] + (1.0,)
        color = add.outputs["Color"]
    if texture.scale[3] != 1.0:
        alpha = _math_node(nodes, "MULTIPLY", alpha, texture.scale[3])
    if texture.bias[3] != 0.0:
        alpha = _math_node(nodes, "ADD", alpha, texture.bias[3])
    if texture.flip_red or texture.flip_green or texture.flip_blue:
        separate_normal = nodes.new("ShaderNodeSeparateColor")
        separate_normal.mode = "RGB"
        links.new(color, separate_normal.inputs["Color"])
        red = (
            _math_node(nodes, "SUBTRACT", 1.0, separate_normal.outputs["Red"])
            if texture.flip_red
            else separate_normal.outputs["Red"]
        )
        green = (
            _math_node(nodes, "SUBTRACT", 1.0, separate_normal.outputs["Green"])
            if texture.flip_green
            else separate_normal.outputs["Green"]
        )
        blue = (
            _math_node(nodes, "SUBTRACT", 1.0, separate_normal.outputs["Blue"])
            if texture.flip_blue
            else separate_normal.outputs["Blue"]
        )
        combine_normal = nodes.new("ShaderNodeCombineColor")
        combine_normal.mode = "RGB"
        links.new(red, combine_normal.inputs["Red"])
        links.new(green, combine_normal.inputs["Green"])
        links.new(blue, combine_normal.inputs["Blue"])
        color = combine_normal.outputs["Color"]
    if texture.mono_mode is not None:
        if texture.mono_mode == "alpha":
            return color, alpha
        separate_mono = nodes.new("ShaderNodeSeparateColor")
        separate_mono.mode = "RGB"
        links.new(color, separate_mono.inputs["Color"])
        red = separate_mono.outputs["Red"]
        green = separate_mono.outputs["Green"]
        blue = separate_mono.outputs["Blue"]
        if texture.mono_mode == "average":
            scalar = _math_node(
                nodes,
                "MULTIPLY",
                _math_node(nodes, "ADD", _math_node(nodes, "ADD", red, green), blue),
                1.0 / 3.0,
            )
        elif texture.mono_mode == "luminance":
            scalar = _math_node(
                nodes,
                "ADD",
                _math_node(
                    nodes,
                    "ADD",
                    _math_node(nodes, "MULTIPLY", red, 0.212671),
                    _math_node(nodes, "MULTIPLY", green, 0.715160),
                ),
                _math_node(nodes, "MULTIPLY", blue, 0.072169),
            )
        elif texture.mono_mode == "maximum":
            scalar = _math_node(
                nodes, "MAXIMUM", _math_node(nodes, "MAXIMUM", red, green), blue
            )
        else:
            raise TexturedRenderError(
                f"unsupported exact mono mode {texture.mono_mode!r}"
            )
        return color, scalar
    if texture.channel in {"rgb", "rgba"}:
        return color, alpha
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    links.new(color, separate.inputs["Color"])
    output_name = {"r": "Red", "g": "Green", "b": "Blue"}.get(texture.channel)
    return color, alpha if texture.channel == "a" else separate.outputs[output_name]


def _multiply_color(nodes: Any, links: Any, socket: Any, color: tuple[float, ...]) -> Any:
    node = nodes.new("ShaderNodeMixRGB")
    node.blend_type = "MULTIPLY"
    node.inputs[0].default_value = 1.0
    links.new(socket, node.inputs[1])
    node.inputs[2].default_value = tuple(color[:3]) + (1.0,)
    return node.outputs["Color"]


def _multiply_color_sockets(nodes: Any, links: Any, left: Any, right: Any) -> Any:
    node = nodes.new("ShaderNodeMixRGB")
    node.blend_type = "MULTIPLY"
    node.inputs[0].default_value = 1.0
    links.new(left, node.inputs[1])
    links.new(right, node.inputs[2])
    return node.outputs["Color"]


def _add_scalar_to_color(nodes: Any, links: Any, socket: Any, value: float) -> Any:
    node = nodes.new("ShaderNodeMixRGB")
    node.blend_type = "ADD"
    node.inputs[0].default_value = 1.0
    links.new(socket, node.inputs[1])
    node.inputs[2].default_value = (value, value, value, 1.0)
    return node.outputs["Color"]


def _linear_desaturate(
    nodes: Any, links: Any, socket: Any, amount: float
) -> Any:
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    links.new(socket, separate.inputs["Color"])
    red = _math_node(nodes, "MULTIPLY", separate.outputs["Red"], 0.212671)
    green = _math_node(nodes, "MULTIPLY", separate.outputs["Green"], 0.715160)
    blue = _math_node(nodes, "MULTIPLY", separate.outputs["Blue"], 0.072169)
    luminance = _math_node(nodes, "ADD", _math_node(nodes, "ADD", red, green), blue)
    grayscale = nodes.new("ShaderNodeCombineColor")
    grayscale.mode = "RGB"
    links.new(luminance, grayscale.inputs["Red"])
    links.new(luminance, grayscale.inputs["Green"])
    links.new(luminance, grayscale.inputs["Blue"])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[0].default_value = amount
    links.new(socket, mix.inputs[1])
    links.new(grayscale.outputs["Color"], mix.inputs[2])
    return mix.outputs["Color"]


def _decoded_tangent_normal(
    nodes: Any,
    links: Any,
    color_socket: Any,
    strength: float,
    *,
    strength_semantics: str,
) -> Any:
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    links.new(color_socket, separate.inputs["Color"])

    def component(name: str, scale: float) -> Any:
        decoded = _math_node(
            nodes,
            "SUBTRACT",
            _math_node(nodes, "MULTIPLY", separate.outputs[name], 2.0),
            1.0,
        )
        return _math_node(nodes, "MULTIPLY", decoded, scale)

    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(component("Red", strength), combine.inputs["X"])
    links.new(component("Green", strength), combine.inputs["Y"])
    if strength_semantics == "omnipbr_xy":
        blue = component("Blue", 1.0)
    elif strength_semantics == "simpbr_all_abs_z":
        blue = _math_node(
            nodes,
            "ADD",
            _math_node(
                nodes,
                "ABSOLUTE",
                component("Blue", strength),
                0.0,
            ),
            1.0 - strength,
        )
    elif strength_semantics == "omnisurface_all":
        blue = _math_node(
            nodes,
            "ADD",
            component("Blue", strength),
            1.0 - strength,
        )
    else:
        raise TexturedRenderError(
            f"unsupported normal strength semantic {strength_semantics!r}"
        )
    links.new(blue, combine.inputs["Z"])
    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    links.new(combine.outputs["Vector"], normalize.inputs[0])
    return normalize.outputs["Vector"]


def _encoded_tangent_normal(nodes: Any, links: Any, normal_socket: Any) -> Any:
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(normal_socket, separate.inputs["Vector"])
    combine = nodes.new("ShaderNodeCombineColor")
    combine.mode = "RGB"
    for source, target in (("X", "Red"), ("Y", "Green"), ("Z", "Blue")):
        encoded = _math_node(
            nodes,
            "MULTIPLY",
            _math_node(nodes, "ADD", separate.outputs[source], 1.0),
            0.5,
        )
        links.new(encoded, combine.inputs[target])
    return combine.outputs["Color"]


def _normal_world_from_texture(
    nodes: Any,
    links: Any,
    color_socket: Any,
    texture: TexturePlan,
    strength: float,
    uv_override: str | None,
    uv_aliases: Mapping[str, str],
    *,
    standardized_sphere: bool,
) -> Any:
    strength_semantics = texture.normal_decode_semantics
    if strength_semantics is None:
        raise TexturedRenderError(
            f"texture role {texture.role!r} has no normal decode semantic"
        )
    tangent = _decoded_tangent_normal(
        nodes,
        links,
        color_socket,
        strength,
        strength_semantics=strength_semantics,
    )
    if standardized_sphere:
        uv_name = uv_override
    elif texture.uv.coordinate_source == "constant_zero":
        # With no texture spaces, MDL state texture coordinates and tangents
        # are zero.  A tangent-space perturbation therefore cannot change the
        # renderer-active surface normal.
        geometry = nodes.new("ShaderNodeNewGeometry")
        return geometry.outputs["Normal"]
    elif texture.uv.coordinate_source == "uv" and texture.uv.projection == "FLAT":
        uv_name = uv_aliases[texture.uv.name]
        if texture.uv.transform_semantics in {
            "mdl_local_scale_rotate",
            "mdl_base_rotation_translation_scale",
        }:
            transformed_u, transformed_v = _mdl_transform_tangent_coefficients(
                texture.uv
            )
            components = nodes.new("ShaderNodeSeparateXYZ")
            links.new(tangent, components.inputs["Vector"])
            transformed = nodes.new("ShaderNodeCombineXYZ")
            links.new(
                _math_node(
                    nodes,
                    "ADD",
                    _math_node(
                        nodes,
                        "MULTIPLY",
                        components.outputs["X"],
                        transformed_u[0],
                    ),
                    _math_node(
                        nodes,
                        "MULTIPLY",
                        components.outputs["Y"],
                        transformed_v[0],
                    ),
                ),
                transformed.inputs["X"],
            )
            links.new(
                _math_node(
                    nodes,
                    "ADD",
                    _math_node(
                        nodes,
                        "MULTIPLY",
                        components.outputs["X"],
                        transformed_u[1],
                    ),
                    _math_node(
                        nodes,
                        "MULTIPLY",
                        components.outputs["Y"],
                        transformed_v[1],
                    ),
                ),
                transformed.inputs["Y"],
            )
            links.new(components.outputs["Z"], transformed.inputs["Z"])
            tangent = _vector_normalize(
                nodes, links, transformed.outputs["Vector"]
            )
    elif texture.uv.projection == "BOX" and texture.uv.coordinate_source in {
        "object",
        "world",
    }:
        _, tangent_u, tangent_v = _mdl_cubic_projection_sockets(
            nodes, links, texture.uv
        )
        components = nodes.new("ShaderNodeSeparateXYZ")
        links.new(tangent, components.inputs["Vector"])
        geometry = nodes.new("ShaderNodeNewGeometry")
        return _vector_normalize(
            nodes,
            links,
            _vector_binary(
                nodes,
                links,
                "ADD",
                _vector_binary(
                    nodes,
                    links,
                    "ADD",
                    _vector_scale(
                        nodes, links, tangent_u, components.outputs["X"]
                    ),
                    _vector_scale(
                        nodes, links, tangent_v, components.outputs["Y"]
                    ),
                ),
                _vector_scale(
                    nodes,
                    links,
                    geometry.outputs["Normal"],
                    components.outputs["Z"],
                ),
            ),
        )
    else:
        raise TexturedRenderError(
            f"normal texture {texture.role!r} has an unsupported coordinate frame"
        )
    if not uv_name:
        raise TexturedRenderError(
            f"normal texture {texture.role!r} has no tangent UV frame"
        )
    normal = nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    normal.uv_map = uv_name
    normal.inputs["Strength"].default_value = 1.0
    links.new(_encoded_tangent_normal(nodes, links, tangent), normal.inputs["Color"])
    return normal.outputs["Normal"]


def _reoriented_world_normal(
    nodes: Any, links: Any, reference: Any, base: Any, detail: Any
) -> Any:
    lifted_base = _vector_binary(nodes, links, "ADD", base, reference)
    detail_alignment = nodes.new("ShaderNodeVectorMath")
    detail_alignment.operation = "DOT_PRODUCT"
    links.new(detail, detail_alignment.inputs[0])
    links.new(reference, detail_alignment.inputs[1])
    reflected_detail = _vector_binary(
        nodes,
        links,
        "SUBTRACT",
        _vector_scale(
            nodes,
            links,
            reference,
            _math_node(
                nodes,
                "MULTIPLY",
                detail_alignment.outputs["Value"],
                2.0,
            ),
        ),
        detail,
    )
    numerator = nodes.new("ShaderNodeVectorMath")
    numerator.operation = "DOT_PRODUCT"
    links.new(lifted_base, numerator.inputs[0])
    links.new(reflected_detail, numerator.inputs[1])
    denominator = nodes.new("ShaderNodeVectorMath")
    denominator.operation = "DOT_PRODUCT"
    links.new(lifted_base, denominator.inputs[0])
    links.new(reference, denominator.inputs[1])
    return _vector_normalize(
        nodes,
        links,
        _vector_binary(
            nodes,
            links,
            "SUBTRACT",
            _vector_scale(
                nodes,
                links,
                lifted_base,
                _math_node(
                    nodes,
                    "DIVIDE",
                    numerator.outputs["Value"],
                    denominator.outputs["Value"],
                ),
            ),
            reflected_detail,
        ),
    )


def _texture_socket_map(
    bpy: Any,
    nodes: Any,
    links: Any,
    plan: MaterialPlan,
    image_cache: dict[tuple[str, str, str], Any],
    uv_override: str | None,
    uv_aliases: Mapping[str, str],
    standardized_sphere: bool,
    swatches: Mapping[str, _BakedSwatch] | None = None,
) -> dict[str, tuple[Any, Any, TexturePlan]]:
    texture_roles = {texture.role for texture in plan.textures}
    selected_swatches = {
        role: swatch
        for role, swatch in (swatches or {}).items()
        if role in texture_roles
    }
    if standardized_sphere and set(selected_swatches) != texture_roles:
        raise TexturedRenderError(
            "standardized sphere swatches do not cover every active texture role"
        )
    if not standardized_sphere and selected_swatches:
        raise TexturedRenderError("segment swatches cannot alter whole/part materials")
    sockets: dict[str, tuple[Any, Any, TexturePlan]] = {}
    for texture in plan.textures:
        if texture.role in sockets:
            raise TexturedRenderError(
                f"material plan duplicates texture role {texture.role!r}"
            )
        texture_uv_override = uv_override
        if texture_uv_override is None and texture.uv.coordinate_source == "uv":
            texture_uv_override = uv_aliases.get(texture.uv.name)
            if texture_uv_override is None:
                raise TexturedRenderError(
                    f"material UV {texture.uv.name!r} has no Blender alias"
                )
        color, scalar = _texture_sockets(
            bpy,
            nodes,
            links,
            texture,
            image_cache,
            texture_uv_override,
            standardized_sphere,
            selected_swatches.get(texture.role),
        )
        sockets[texture.role] = (color, scalar, texture)
    return sockets


def _color_value_socket(nodes: Any, value: Any) -> Any:
    node = nodes.new("ShaderNodeRGB")
    node.outputs[0].default_value = tuple(value)
    return node.outputs[0]


def _custom_normal_socket(
    nodes: Any,
    links: Any,
    sockets: Mapping[str, tuple[Any, Any, TexturePlan]],
    strength: float,
    uv_override: str | None,
    uv_aliases: Mapping[str, str],
    role: str = "normal",
) -> Any | None:
    normal_record = sockets.get(role)
    if normal_record is None:
        return None
    return _normal_world_from_texture(
        nodes,
        links,
        normal_record[0],
        normal_record[2],
        abs(strength),
        uv_override,
        uv_aliases,
        standardized_sphere=uv_override == "SphereConformalUV",
    )


def _link_or_set(links: Any, socket: Any, value: Any) -> None:
    if hasattr(value, "bl_idname"):
        links.new(value, socket)
    else:
        socket.default_value = value


def _add_shader(nodes: Any, links: Any, left: Any | None, right: Any) -> Any:
    if left is None:
        return right
    add = nodes.new("ShaderNodeAddShader")
    links.new(left, add.inputs[0])
    links.new(right, add.inputs[1])
    return add.outputs[0]


def _weighted_shader(nodes: Any, links: Any, shader: Any, weight: Any) -> Any:
    if not hasattr(weight, "bl_idname") and float(weight) == 1.0:
        return shader
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    _link_or_set(links, mix.inputs[0], weight)
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(shader, mix.inputs[2])
    return mix.outputs[0]


def _fresnel_weights(
    nodes: Any,
    links: Any,
    ior: float,
    reflection_weight: Any,
    transmission_weight: Any,
    normal_socket: Any | None,
) -> tuple[Any, Any]:
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = ior
    if normal_socket is not None:
        links.new(normal_socket, fresnel.inputs["Normal"])
    reflected = _math_node(
        nodes, "MULTIPLY", fresnel.outputs["Fac"], reflection_weight
    )
    transmitted = _math_node(
        nodes,
        "MULTIPLY",
        _math_node(nodes, "SUBTRACT", 1.0, fresnel.outputs["Fac"]),
        transmission_weight,
    )
    return reflected, transmitted


def _opacity_shader(
    nodes: Any,
    links: Any,
    surface: Any,
    opacity: Any,
    threshold: float,
) -> Any:
    if not hasattr(opacity, "bl_idname") and float(opacity) == 1.0 and threshold == 0.0:
        return surface
    if threshold > 0.0:
        opacity = _math_node(
            nodes,
            "SUBTRACT",
            1.0,
            _math_node(nodes, "LESS_THAN", opacity, threshold),
        )
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    _link_or_set(links, mix.inputs[0], opacity)
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(surface, mix.inputs[2])
    return mix.outputs[0]


def _one_sided_shader(
    nodes: Any, links: Any, surface: Any, double_sided: bool
) -> Any:
    if double_sided:
        return surface
    geometry = nodes.new("ShaderNodeNewGeometry")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(geometry.outputs["Backfacing"], mix.inputs[0])
    links.new(surface, mix.inputs[1])
    links.new(transparent.outputs[0], mix.inputs[2])
    return mix.outputs[0]


def _build_omniglass_graph(
    nodes: Any,
    links: Any,
    output: Any,
    plan: MaterialPlan,
    sockets: Mapping[str, tuple[Any, Any, TexturePlan]],
    uv_override: str | None,
    uv_aliases: Mapping[str, str],
) -> None:
    parameters = plan.model_parameters
    transmission_role = str(parameters["transmission_texture_role"])
    glass_color = (
        sockets[transmission_role][0]
        if transmission_role in sockets
        else _color_value_socket(nodes, parameters["glass_color"])
    )
    reflection_color = (
        sockets["specular"][0]
        if "specular" in sockets
        else _color_value_socket(nodes, parameters["reflection_color"])
    )
    roughness: Any = float(parameters["roughness"])
    if "roughness" in sockets:
        influence = float(parameters["roughness_texture_influence"])
        roughness = _math_node(
            nodes,
            "ADD",
            _math_node(nodes, "MULTIPLY", sockets["roughness"][1], influence),
            float(parameters["roughness"]) * (1.0 - influence),
        )
    normal = _custom_normal_socket(
        nodes,
        links,
        sockets,
        float(parameters["normal_strength"]),
        uv_override,
        uv_aliases,
    )
    reflected_weight, transmitted_weight = _fresnel_weights(
        nodes,
        links,
        float(parameters["ior"]),
        1.0,
        1.0,
        normal,
    )
    reflection = nodes.new("ShaderNodeBsdfAnisotropic")
    links.new(reflection_color, reflection.inputs["Color"])
    _link_or_set(links, reflection.inputs["Roughness"], roughness)
    if normal is not None:
        links.new(normal, reflection.inputs["Normal"])

    if bool(parameters["thin_walled"]):
        transmission = nodes.new("ShaderNodeBsdfTransparent")
        links.new(glass_color, transmission.inputs["Color"])
    else:
        transmission = nodes.new("ShaderNodeBsdfRefraction")
        links.new(
            _color_value_socket(nodes, (1.0, 1.0, 1.0, 1.0))
            if parameters["non_thin_surface_untinted"]
            else glass_color,
            transmission.inputs["Color"],
        )
        _link_or_set(links, transmission.inputs["Roughness"], roughness)
        transmission.inputs["IOR"].default_value = float(parameters["ior"])
        if normal is not None:
            links.new(normal, transmission.inputs["Normal"])
    reflection_shader = _weighted_shader(
        nodes, links, reflection.outputs[0], reflected_weight
    )
    transmission_shader = _weighted_shader(
        nodes, links, transmission.outputs[0], transmitted_weight
    )
    surface = _add_shader(
        nodes, links, reflection_shader, transmission_shader
    )
    opacity: Any = plan.opacity
    if "opacity" in sockets:
        opacity = sockets["opacity"][1]
    surface = _opacity_shader(
        nodes, links, surface, opacity, plan.opacity_threshold
    )
    surface = _one_sided_shader(nodes, links, surface, plan.double_sided)
    links.new(surface, output.inputs["Surface"])

    absorption_scale = float(parameters["absorption_scale"])
    if not parameters["thin_walled"] and absorption_scale > 0.0:
        absorption = nodes.new("ShaderNodeVolumeAbsorption")
        links.new(glass_color, absorption.inputs["Color"])
        absorption.inputs["Density"].default_value = absorption_scale
        links.new(absorption.outputs[0], output.inputs["Volume"])


def _build_omnisurface_graph(
    nodes: Any,
    links: Any,
    output: Any,
    plan: MaterialPlan,
    sockets: Mapping[str, tuple[Any, Any, TexturePlan]],
    uv_override: str | None,
    uv_aliases: Mapping[str, str],
) -> None:
    parameters = plan.model_parameters

    def color(role: str, parameter: str) -> Any:
        return (
            sockets[role][0]
            if role in sockets
            else _color_value_socket(nodes, parameters[parameter])
        )

    def scalar(role: str, parameter: str) -> Any:
        return sockets[role][1] if role in sockets else float(parameters[parameter])

    def active(role: str, parameter: str) -> bool:
        return role in sockets or float(parameters[parameter]) > 0.0

    normal = _custom_normal_socket(
        nodes,
        links,
        sockets,
        float(parameters["normal_strength"]),
        uv_override,
        uv_aliases,
    )
    if "displacement" in sockets:
        displacement_height: Any = sockets["displacement"][1]
        zero = float(parameters["displacement_zero"])
        if zero != 0.0:
            displacement_height = _math_node(
                nodes, "SUBTRACT", displacement_height, zero
            )
        displacement_height = _math_node(
            nodes,
            "MULTIPLY",
            displacement_height,
            scalar("displacement_scale", "displacement_scale"),
        )
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = 1.0
        links.new(displacement_height, bump.inputs["Height"])
        if normal is not None:
            links.new(normal, bump.inputs["Normal"])
        normal = bump.outputs["Normal"]
    surface = None
    diffuse_color = color("albedo", "diffuse_color")
    metalness = scalar("metallic", "metalness")
    diffuse_weight = scalar("diffuse_weight", "diffuse_weight")
    diffuse_factor = _math_node(
        nodes,
        "MULTIPLY",
        diffuse_weight,
        _math_node(nodes, "SUBTRACT", 1.0, metalness),
    )
    metal_factor = _math_node(nodes, "MULTIPLY", diffuse_weight, metalness)
    diffuse_active = active("diffuse_weight", "diffuse_weight")
    metal_active = diffuse_active and active("metallic", "metalness")
    if diffuse_active:
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        links.new(diffuse_color, diffuse.inputs["Color"])
        _link_or_set(
            links,
            diffuse.inputs["Roughness"],
            scalar("diffuse_roughness", "diffuse_roughness"),
        )
        if normal is not None:
            links.new(normal, diffuse.inputs["Normal"])
        surface = _add_shader(
            nodes,
            links,
            surface,
            _weighted_shader(
                nodes,
                links,
                diffuse.outputs[0],
                diffuse_factor,
            ),
        )
    if metal_active:
        metal = nodes.new("ShaderNodeBsdfAnisotropic")
        links.new(diffuse_color, metal.inputs["Color"])
        _link_or_set(
            links,
            metal.inputs["Roughness"],
            scalar("specular_roughness", "specular_roughness"),
        )
        _link_or_set(
            links,
            metal.inputs["Anisotropy"],
            scalar("specular_anisotropy", "specular_anisotropy"),
        )
        _link_or_set(
            links,
            metal.inputs["Rotation"],
            scalar(
                "specular_anisotropy_rotation",
                "specular_anisotropy_rotation",
            ),
        )
        if normal is not None:
            links.new(normal, metal.inputs["Normal"])
        surface = _add_shader(
            nodes,
            links,
            surface,
            _weighted_shader(
                nodes, links, metal.outputs[0], metal_factor
            ),
        )

    specular_weight = scalar("specular_weight", "specular_weight")
    transmission_weight = scalar("transmission_weight", "transmission_weight")
    reflected_weight, transmitted_weight = _fresnel_weights(
        nodes,
        links,
        float(parameters["specular_ior"]),
        specular_weight,
        transmission_weight,
        normal,
    )
    if active("specular_weight", "specular_weight"):
        reflection = nodes.new("ShaderNodeBsdfAnisotropic")
        links.new(
            color("specular", "specular_color"),
            reflection.inputs["Color"],
        )
        _link_or_set(
            links,
            reflection.inputs["Roughness"],
            scalar("specular_roughness", "specular_roughness"),
        )
        _link_or_set(
            links,
            reflection.inputs["Anisotropy"],
            scalar("specular_anisotropy", "specular_anisotropy"),
        )
        _link_or_set(
            links,
            reflection.inputs["Rotation"],
            scalar(
                "specular_anisotropy_rotation",
                "specular_anisotropy_rotation",
            ),
        )
        if normal is not None:
            links.new(normal, reflection.inputs["Normal"])
        surface = _add_shader(
            nodes,
            links,
            surface,
            _weighted_shader(
                nodes, links, reflection.outputs[0], reflected_weight
            ),
        )

    if parameters["transmission_enabled"] and active(
        "transmission_weight", "transmission_weight"
    ):
        transmission_color = color(
            "transmission_color", "transmission_color"
        )
        if parameters["thin_walled"]:
            transmission = nodes.new("ShaderNodeBsdfTransparent")
            links.new(transmission_color, transmission.inputs["Color"])
        else:
            transmission = nodes.new("ShaderNodeBsdfRefraction")
            links.new(transmission_color, transmission.inputs["Color"])
            _link_or_set(
                links,
                transmission.inputs["Roughness"],
                scalar("specular_roughness", "specular_roughness"),
            )
            transmission.inputs["IOR"].default_value = float(
                parameters["specular_ior"]
            )
            if normal is not None:
                links.new(normal, transmission.inputs["Normal"])
        surface = _add_shader(
            nodes,
            links,
            surface,
            _weighted_shader(
                nodes, links, transmission.outputs[0], transmitted_weight
            ),
        )

    if parameters["subsurface_enabled"] and active(
        "subsurface_weight", "subsurface_weight"
    ):
        subsurface = nodes.new("ShaderNodeSubsurfaceScattering")
        links.new(
            color("subsurface_transmission_color", "subsurface_transmission_color"),
            subsurface.inputs["Color"],
        )
        scattering = color(
            "subsurface_scattering_color", "subsurface_scattering_color"
        )
        links.new(scattering, subsurface.inputs["Radius"])
        _link_or_set(
            links,
            subsurface.inputs["Scale"],
            scalar("subsurface_scale", "subsurface_scale"),
        )
        subsurface.inputs["IOR"].default_value = float(parameters["specular_ior"])
        _link_or_set(
            links,
            subsurface.inputs["Anisotropy"],
            scalar("subsurface_anisotropy", "subsurface_anisotropy"),
        )
        if normal is not None:
            links.new(normal, subsurface.inputs["Normal"])
        surface = _add_shader(
            nodes,
            links,
            surface,
            _weighted_shader(
                nodes,
                links,
                subsurface.outputs[0],
                scalar("subsurface_weight", "subsurface_weight"),
            ),
        )

    if active("retroreflection_weight", "retro_weight"):
        retro = nodes.new("ShaderNodeBsdfSheen")
        links.new(
            color("retroreflection_color", "retro_color"),
            retro.inputs["Color"],
        )
        _link_or_set(
            links,
            retro.inputs["Roughness"],
            scalar("retroreflection_roughness", "retro_roughness"),
        )
        if normal is not None:
            links.new(normal, retro.inputs["Normal"])
        surface = _add_shader(
            nodes,
            links,
            surface,
            _weighted_shader(
                nodes,
                links,
                retro.outputs[0],
                scalar("retroreflection_weight", "retro_weight"),
            ),
        )

    if active("coat_weight", "coat_weight"):
        coat_normal = _custom_normal_socket(
            nodes,
            links,
            sockets,
            1.0,
            uv_override,
            uv_aliases,
            role="coat_normal",
        )
        effective_coat_normal = (
            coat_normal if coat_normal is not None else normal
        )
        coat_fresnel, _ = _fresnel_weights(
            nodes,
            links,
            float(parameters["coat_ior"]),
            scalar("coat_weight", "coat_weight"),
            0.0,
            effective_coat_normal,
        )
        coat = nodes.new("ShaderNodeBsdfAnisotropic")
        links.new(color("coat_color", "coat_color"), coat.inputs["Color"])
        _link_or_set(
            links,
            coat.inputs["Roughness"],
            scalar("coat_roughness", "coat_roughness"),
        )
        _link_or_set(
            links,
            coat.inputs["Anisotropy"],
            scalar("coat_anisotropy", "coat_anisotropy"),
        )
        _link_or_set(
            links,
            coat.inputs["Rotation"],
            scalar("coat_anisotropy_rotation", "coat_anisotropy_rotation"),
        )
        if effective_coat_normal is not None:
            links.new(effective_coat_normal, coat.inputs["Normal"])
        surface = _add_shader(
            nodes,
            links,
            surface,
            _weighted_shader(nodes, links, coat.outputs[0], coat_fresnel),
        )

    if active("emission_weight", "emission_weight"):
        emission = nodes.new("ShaderNodeEmission")
        links.new(color("emissive", "emission_color"), emission.inputs["Color"])
        _link_or_set(
            links,
            emission.inputs["Strength"],
            _math_node(
                nodes,
                "MULTIPLY",
                scalar("emission_intensity", "emission_intensity"),
                scalar("emission_weight", "emission_weight"),
            ),
        )
        surface = _add_shader(nodes, links, surface, emission.outputs[0])

    if surface is None:
        surface = nodes.new("ShaderNodeBsdfTransparent").outputs[0]
    opacity: Any = plan.opacity
    if "opacity" in sockets:
        opacity = sockets["opacity"][1]
    surface = _opacity_shader(
        nodes, links, surface, opacity, plan.opacity_threshold
    )
    surface = _one_sided_shader(nodes, links, surface, plan.double_sided)
    links.new(surface, output.inputs["Surface"])

    depth = float(parameters["transmission_depth"])
    if (
        parameters["transmission_enabled"]
        and active("transmission_weight", "transmission_weight")
        and not parameters["thin_walled"]
        and depth > 0.0
    ):
        transmission_color = color(
            "transmission_color", "transmission_color"
        )
        absorption = nodes.new("ShaderNodeVolumeAbsorption")
        links.new(transmission_color, absorption.inputs["Color"])
        volume_density = _math_node(
            nodes,
            "DIVIDE",
            scalar("transmission_weight", "transmission_weight"),
            depth,
        )
        links.new(volume_density, absorption.inputs["Density"])
        volume = absorption.outputs[0]
        scattering_color_value = tuple(
            parameters["transmission_scattering_color"][:3]
        )
        if "transmission_scattering_color" in sockets or any(
            component > 0.0 for component in scattering_color_value
        ):
            scatter = nodes.new("ShaderNodeVolumeScatter")
            links.new(
                color(
                    "transmission_scattering_color",
                    "transmission_scattering_color",
                ),
                scatter.inputs["Color"],
            )
            links.new(volume_density, scatter.inputs["Density"])
            _link_or_set(
                links,
                scatter.inputs["Anisotropy"],
                scalar(
                    "transmission_scatter_anisotropy",
                    "transmission_scatter_anisotropy",
                ),
            )
            volume = _add_shader(nodes, links, volume, scatter.outputs[0])
        links.new(volume, output.inputs["Volume"])


def _build_blender_material(
    bpy: Any,
    plan: MaterialPlan,
    image_cache: dict[tuple[str, str, str], Any],
    *,
    uv_override: str | None = None,
    uv_aliases: Mapping[str, str] | None = None,
    standardized_sphere: bool = False,
    swatches: Mapping[str, _BakedSwatch] | None = None,
) -> Any:
    if standardized_sphere and uv_override != "SphereConformalUV":
        raise TexturedRenderError(
            "the standardized material sphere requires SphereConformalUV"
        )
    expected_swatches = _expected_swatch_roles(plan)
    if standardized_sphere and set(swatches or {}) != expected_swatches:
        raise TexturedRenderError(
            "standardized sphere swatches do not cover every active appearance input"
        )
    if not standardized_sphere and swatches:
        raise TexturedRenderError("segment swatches cannot alter whole/part materials")
    resolved_uv_aliases = dict(uv_aliases or _uv_aliases(plan))
    suffix = "sphere" if standardized_sphere else plan.provenance_sha256[:12]
    material = bpy.data.materials.new(
        f"vlm_{_safe_component(plan.material_name)}_{suffix}"
    )
    material.use_nodes = True
    material.diffuse_color = plan.base_color
    material.use_backface_culling = not plan.double_sided
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    sockets = _texture_socket_map(
        bpy,
        nodes,
        links,
        plan,
        image_cache,
        uv_override,
        resolved_uv_aliases,
        standardized_sphere,
        swatches,
    )
    display_color_socket = None
    display_opacity_socket = None
    if standardized_sphere and _DISPLAY_COLOR_SWATCH in expected_swatches:
        display_color_socket = _baked_swatch_sockets(
            nodes, links, swatches[_DISPLAY_COLOR_SWATCH]
        )[0]
    if standardized_sphere and _DISPLAY_OPACITY_SWATCH in expected_swatches:
        display_opacity_socket = _baked_swatch_sockets(
            nodes, links, swatches[_DISPLAY_OPACITY_SWATCH]
        )[1]
    if plan.material_model == "omniglass":
        _build_omniglass_graph(
            nodes,
            links,
            output,
            plan,
            sockets,
            uv_override,
            resolved_uv_aliases,
        )
        return material
    if plan.material_model == "omnisurface":
        _build_omnisurface_graph(
            nodes,
            links,
            output,
            plan,
            sockets,
            uv_override,
            resolved_uv_aliases,
        )
        return material
    if plan.material_model != "principled":
        raise TexturedRenderError(
            f"unsupported exact material model {plan.material_model!r}"
        )
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    if plan.double_sided:
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    else:
        geometry = nodes.new("ShaderNodeNewGeometry")
        transparent_back = nodes.new("ShaderNodeBsdfTransparent")
        one_sided = nodes.new("ShaderNodeMixShader")
        links.new(geometry.outputs["Backfacing"], one_sided.inputs[0])
        links.new(principled.outputs["BSDF"], one_sided.inputs[1])
        links.new(transparent_back.outputs["BSDF"], one_sided.inputs[2])
        links.new(one_sided.outputs["Shader"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = plan.base_color
    principled.inputs["Roughness"].default_value = plan.roughness
    principled.inputs["Metallic"].default_value = plan.metallic
    principled.inputs["Alpha"].default_value = plan.opacity
    principled.inputs["Emission Color"].default_value = plan.emission_color
    principled.inputs["Emission Strength"].default_value = plan.emission_strength
    principled.inputs["IOR"].default_value = plan.ior
    principled.inputs["Specular IOR Level"].default_value = plan.specular_ior_level
    principled.inputs["Transmission Weight"].default_value = plan.transmission_weight
    principled.inputs["Coat Weight"].default_value = plan.coat_weight
    principled.inputs["Coat Roughness"].default_value = plan.coat_roughness
    principled.inputs["Coat IOR"].default_value = plan.coat_ior
    principled.inputs["Coat Tint"].default_value = plan.coat_tint
    principled.inputs["Sheen Weight"].default_value = plan.sheen_weight
    principled.inputs["Sheen Roughness"].default_value = plan.sheen_roughness
    principled.inputs["Sheen Tint"].default_value = plan.sheen_tint
    principled.inputs["Anisotropic"].default_value = plan.anisotropic
    principled.inputs["Anisotropic Rotation"].default_value = plan.anisotropic_rotation
    principled.inputs["Subsurface Weight"].default_value = plan.subsurface_weight
    principled.inputs["Subsurface Radius"].default_value = plan.subsurface_radius
    principled.inputs["Subsurface Scale"].default_value = plan.subsurface_scale
    subsurface_ior_socket = next(
        socket for socket in principled.inputs if socket.name == "Subsurface IOR"
    )
    subsurface_ior_socket.default_value = plan.subsurface_ior
    principled.inputs["Subsurface Anisotropy"].default_value = plan.subsurface_anisotropy
    principled.inputs["Thin Film Thickness"].default_value = plan.thin_film_thickness
    principled.inputs["Thin Film IOR"].default_value = plan.thin_film_ior

    base_socket = None
    if plan.display_color_attribute:
        if display_color_socket is not None:
            base_socket = display_color_socket
        else:
            attribute = nodes.new("ShaderNodeVertexColor")
            attribute.layer_name = plan.display_color_attribute
            base_socket = attribute.outputs["Color"]
    if "albedo" in sockets:
        base_socket = sockets["albedo"][0]
        if plan.omnipbr_controls:
            if plan.albedo_brightness != 1.0:
                base_socket = _multiply_color(
                    nodes,
                    links,
                    base_socket,
                    (plan.albedo_brightness,) * 3,
                )
            if plan.albedo_add != 0.0:
                base_socket = _add_scalar_to_color(
                    nodes, links, base_socket, plan.albedo_add
                )
            if plan.albedo_desaturation != 0.0:
                base_socket = _linear_desaturate(
                    nodes, links, base_socket, plan.albedo_desaturation
                )
            if plan.diffuse_tint != (1.0, 1.0, 1.0, 1.0):
                base_socket = _multiply_color(nodes, links, base_socket, plan.diffuse_tint)
        elif plan.base_color != (1.0, 1.0, 1.0, 1.0):
            base_socket = _multiply_color(nodes, links, base_socket, plan.base_color)
    elif plan.omnipbr_controls and plan.diffuse_tint != (1.0, 1.0, 1.0, 1.0):
        constant = nodes.new("ShaderNodeRGB")
        constant.outputs[0].default_value = plan.base_color
        base_socket = _multiply_color(
            nodes, links, constant.outputs[0], plan.diffuse_tint
        )
    if "occlusion" in sockets and (
        not plan.omnipbr_controls or plan.ao_to_diffuse > 0.0
    ):
        occlusion_color, occlusion_scalar, occlusion_plan = sockets["occlusion"]
        if occlusion_plan.channel in {"rgb", "rgba"}:
            occlusion = occlusion_color
        else:
            combine = nodes.new("ShaderNodeCombineColor")
            combine.mode = "RGB"
            links.new(occlusion_scalar, combine.inputs["Red"])
            links.new(occlusion_scalar, combine.inputs["Green"])
            links.new(occlusion_scalar, combine.inputs["Blue"])
            occlusion = combine.outputs["Color"]
        if plan.omnipbr_controls and plan.ao_to_diffuse < 1.0:
            influence = nodes.new("ShaderNodeMixRGB")
            influence.blend_type = "MIX"
            influence.inputs[0].default_value = plan.ao_to_diffuse
            influence.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)
            links.new(occlusion, influence.inputs[2])
            occlusion = influence.outputs["Color"]
        if base_socket is None:
            rgb = nodes.new("ShaderNodeRGB")
            rgb.outputs[0].default_value = plan.base_color
            base_socket = rgb.outputs[0]
        multiply = nodes.new("ShaderNodeMixRGB")
        multiply.blend_type = "MULTIPLY"
        multiply.inputs[0].default_value = 1.0
        links.new(base_socket, multiply.inputs[1])
        links.new(occlusion, multiply.inputs[2])
        base_socket = multiply.outputs["Color"]
    if base_socket is not None:
        links.new(base_socket, principled.inputs["Base Color"])

    if "roughness" in sockets:
        roughness_socket = sockets["roughness"][1]
        if plan.roughness_texture_influence != 1.0:
            textured = _math_node(
                nodes, "MULTIPLY", roughness_socket, plan.roughness_texture_influence
            )
            roughness_socket = _math_node(
                nodes,
                "ADD",
                textured,
                plan.roughness * (1.0 - plan.roughness_texture_influence),
            )
        links.new(roughness_socket, principled.inputs["Roughness"])
    if "metallic" in sockets:
        metallic_socket = sockets["metallic"][1]
        if plan.metallic_texture_influence != 1.0:
            textured = _math_node(
                nodes, "MULTIPLY", metallic_socket, plan.metallic_texture_influence
            )
            metallic_socket = _math_node(
                nodes,
                "ADD",
                textured,
                plan.metallic * (1.0 - plan.metallic_texture_influence),
            )
        links.new(metallic_socket, principled.inputs["Metallic"])
    if "specular" in sockets:
        links.new(sockets["specular"][1], principled.inputs["Specular IOR Level"])
    normal_socket = None
    standardized_normal_frame = standardized_sphere
    if "normal" in sockets:
        normal_socket = _normal_world_from_texture(
            nodes,
            links,
            sockets["normal"][0],
            sockets["normal"][2],
            plan.normal_strength,
            uv_override,
            resolved_uv_aliases,
            standardized_sphere=standardized_normal_frame,
        )
    if "detail_normal" in sockets:
        detail_normal = _normal_world_from_texture(
            nodes,
            links,
            sockets["detail_normal"][0],
            sockets["detail_normal"][2],
            plan.detail_normal_strength,
            uv_override,
            resolved_uv_aliases,
            standardized_sphere=standardized_normal_frame,
        )
        if normal_socket is None:
            normal_socket = detail_normal
        else:
            geometry = nodes.new("ShaderNodeNewGeometry")
            normal_socket = _reoriented_world_normal(
                nodes,
                links,
                geometry.outputs["Normal"],
                normal_socket,
                detail_normal,
            )
    if "displacement" in sockets:
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = plan.bump_strength
        bump.inputs["Distance"].default_value = plan.bump_distance
        links.new(sockets["displacement"][1], bump.inputs["Height"])
        if normal_socket is not None:
            links.new(normal_socket, bump.inputs["Normal"])
        normal_socket = bump.outputs["Normal"]
    if normal_socket is not None:
        links.new(normal_socket, principled.inputs["Normal"])
    if "opacity" in sockets:
        opacity_socket = sockets["opacity"][1]
        if not plan.omnipbr_controls and plan.opacity != 1.0:
            opacity_socket = _math_node(nodes, "MULTIPLY", opacity_socket, plan.opacity)
        if plan.opacity_threshold > 0.0:
            opacity_socket = _math_node(
                nodes,
                "SUBTRACT",
                1.0,
                _math_node(nodes, "LESS_THAN", opacity_socket, plan.opacity_threshold),
            )
        links.new(opacity_socket, principled.inputs["Alpha"])
    elif plan.display_opacity_attribute:
        if display_opacity_socket is not None:
            links.new(display_opacity_socket, principled.inputs["Alpha"])
        else:
            attribute = nodes.new("ShaderNodeAttribute")
            attribute.attribute_name = plan.display_opacity_attribute
            links.new(attribute.outputs["Fac"], principled.inputs["Alpha"])
    elif plan.opacity_threshold > 0.0:
        principled.inputs["Alpha"].default_value = (
            1.0 if plan.opacity >= plan.opacity_threshold else 0.0
        )
    emission_socket = sockets["emissive"][0] if "emissive" in sockets else None
    if emission_socket is not None:
        if (
            not plan.omnipbr_controls
            and plan.emission_color != (1.0, 1.0, 1.0, 1.0)
        ):
            emission_socket = _multiply_color(
                nodes, links, emission_socket, plan.emission_color
            )
    if "emissive_mask" in sockets:
        if emission_socket is None:
            constant_emission = nodes.new("ShaderNodeRGB")
            constant_emission.outputs[0].default_value = plan.emission_color
            emission_socket = constant_emission.outputs[0]
        emission_socket = _multiply_color_sockets(
            nodes, links, emission_socket, sockets["emissive_mask"][0]
        )
    if emission_socket is not None:
        links.new(emission_socket, principled.inputs["Emission Color"])
        principled.inputs["Emission Strength"].default_value = plan.emission_strength
    return material


def _transparent_material(bpy: Any) -> Any:
    material = bpy.data.materials.new("vlm_nonselected_transparent")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
    return material


def _polygon_arrays(
    counts: np.ndarray,
    indices: np.ndarray,
    orientation: str,
) -> tuple[list[list[int]], np.ndarray]:
    faces: list[list[int]] = []
    loop_order: list[int] = []
    offset = 0
    left_handed = str(orientation).casefold() == "lefthanded"
    for count_value in counts:
        count = int(count_value)
        face_indices = [int(item) for item in indices[offset : offset + count]]
        order = list(range(offset, offset + count))
        if left_handed:
            face_indices.reverse()
            order.reverse()
        faces.append(face_indices)
        loop_order.extend(order)
        offset += count
    if offset != len(indices):
        raise TexturedRenderError("face counts do not consume exact topology indices")
    return faces, np.asarray(loop_order, dtype=np.int64)


def _assign_uv_layers(
    mesh: Any,
    uv_values: Mapping[str, np.ndarray],
    loop_order: np.ndarray,
) -> None:
    for name, values in sorted(uv_values.items()):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 2 or len(array) != len(loop_order):
            raise TexturedRenderError(
                f"UV set {name!r} does not align with exact polygon corners"
            )
        ordered = array[loop_order, :2]
        if not np.isfinite(ordered).all():
            raise TexturedRenderError(f"UV set {name!r} contains non-finite values")
        layer = mesh.uv_layers.new(name=name)
        if layer is None or len(layer.data) != len(ordered):
            raise TexturedRenderError(f"Blender did not create exact UV layer {name!r}")
        for index, value in enumerate(ordered):
            layer.data[index].uv = (float(value[0]), float(value[1]))


def _matrix_world(matrix_type: Any, value: np.ndarray) -> Any:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise TexturedRenderError("occurrence transform is not a finite 4x4 matrix")
    result = matrix_type(matrix.tolist()).transposed()
    if abs(result.determinant()) < 1e-15:
        raise TexturedRenderError("occurrence transform is singular")
    return result


@dataclass
class _Renderable:
    objects: list[Any]
    mesh: Any
    face_segments: tuple[str, ...]
    segment_material_indices: dict[str, int]
    whole_material_indices: tuple[int, ...]
    transparent_index: int
    segment_keys: frozenset[str]

    def whole_state(self) -> None:
        for polygon, material_index in zip(self.mesh.polygons, self.whole_material_indices):
            polygon.material_index = material_index
        for obj in self.objects:
            obj.hide_render = False

    def isolated_state(self, segment_key: str) -> None:
        contains = segment_key in self.segment_keys
        for obj in self.objects:
            obj.hide_render = not contains
        if not contains:
            return
        selected_index = self.segment_material_indices[segment_key]
        for polygon, owner in zip(self.mesh.polygons, self.face_segments):
            polygon.material_index = selected_index if owner == segment_key else self.transparent_index


def _linear_color_component(value: float, color_space: str) -> float:
    normalized = color_space.casefold().replace("_", "").replace("-", "")
    if normalized in {"", "linear", "raw", "linrec709", "scenelinear"}:
        return value
    if normalized in {"srgb", "gammasrgb"}:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
    raise TexturedRenderError(f"unsupported displayColor color space {color_space!r}")


def _display_values(record: Mapping[str, Any], *, color: bool) -> np.ndarray:
    raw = record.get("computed_flattened_values")
    if not isinstance(raw, list) or not raw:
        raise TexturedRenderError("display primvar has no exact flattened values")
    if color:
        parsed = [_finite_color(value) for value in raw]
        if any(value is None for value in parsed):
            raise TexturedRenderError("displayColor contains invalid values")
        color_space = str(record.get("effective_color_space") or "")
        return np.asarray(
            [
                tuple(_linear_color_component(component, color_space) for component in value[:3])
                + (value[3],)
                for value in parsed
                if value is not None
            ],
            dtype=np.float64,
        )
    parsed_scalars = [_display_scalar(value) for value in raw]
    if any(value is None for value in parsed_scalars):
        raise TexturedRenderError("displayOpacity contains invalid values")
    values = np.asarray(parsed_scalars, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise TexturedRenderError("displayOpacity values are outside [0, 1]")
    return values


def _new_mesh_attribute(
    mesh: Any,
    name: str,
    domain: str,
    values: np.ndarray,
    *,
    color: bool,
) -> None:
    if color:
        attribute = mesh.color_attributes.new(name=name, type="FLOAT_COLOR", domain=domain)
        if attribute is None or len(attribute.data) != len(values):
            raise TexturedRenderError(f"Blender could not create exact {name} color attribute")
        for destination, value in zip(attribute.data, values):
            destination.color = tuple(float(component) for component in value)
    else:
        attribute = mesh.attributes.new(name=name, type="FLOAT", domain=domain)
        if attribute is None or len(attribute.data) != len(values):
            raise TexturedRenderError(f"Blender could not create exact {name} scalar attribute")
        for destination, value in zip(attribute.data, values):
            destination.value = float(value)


def _assign_selected_display_attribute(
    mesh: Any,
    segment: Mapping[str, Any],
    payload: SegmentGeometryPayload,
    loop_order: np.ndarray,
    primvar_name: str,
    attribute_name: str,
    *,
    color: bool,
) -> None:
    record = _display_primvar_record(segment, primvar_name)
    if record is None:
        raise TexturedRenderError(f"required {primvar_name} record is missing")
    values = _display_values(record, color=color)
    interpolation = str(record.get("interpolation") or "constant")
    if interpolation == "constant":
        selected = np.repeat(values[:1], len(mesh.loops), axis=0)
        _new_mesh_attribute(mesh, attribute_name, "CORNER", selected, color=color)
    elif interpolation == "uniform":
        indices = np.asarray(payload.source_face_indices, dtype=np.int64)
        if indices.size and int(indices.max()) >= len(values):
            raise TexturedRenderError(f"{primvar_name} uniform index exceeds exact values")
        selected = np.repeat(
            values[indices],
            np.asarray(payload.face_vertex_counts, dtype=np.int64),
            axis=0,
        )
        _new_mesh_attribute(
            mesh,
            attribute_name,
            "CORNER",
            selected[loop_order],
            color=color,
        )
    elif interpolation in {"vertex", "varying"}:
        indices = np.asarray(payload.source_vertex_indices, dtype=np.int64)
        if indices.size and int(indices.max()) >= len(values):
            raise TexturedRenderError(f"{primvar_name} vertex index exceeds exact values")
        _new_mesh_attribute(mesh, attribute_name, "POINT", values[indices], color=color)
    elif interpolation == "faceVarying":
        indices = np.asarray(payload.source_corner_indices, dtype=np.int64)
        if indices.size and int(indices.max()) >= len(values):
            raise TexturedRenderError(f"{primvar_name} corner index exceeds exact values")
        _new_mesh_attribute(
            mesh,
            attribute_name,
            "CORNER",
            values[indices][loop_order],
            color=color,
        )
    else:
        raise TexturedRenderError(f"unsupported {primvar_name} interpolation {interpolation!r}")


def _assign_selected_display_attributes(
    mesh: Any,
    segment: Mapping[str, Any],
    payload: SegmentGeometryPayload,
    loop_order: np.ndarray,
    plan: MaterialPlan,
) -> None:
    if plan.display_color_attribute is not None:
        _assign_selected_display_attribute(
            mesh,
            segment,
            payload,
            loop_order,
            "displayColor",
            "vlm_displayColor",
            color=True,
        )
    if plan.display_opacity_attribute is not None:
        _assign_selected_display_attribute(
            mesh,
            segment,
            payload,
            loop_order,
            "displayOpacity",
            "vlm_displayOpacity",
            color=False,
        )


def _add_subdivision_modifier(
    obj: Any,
    subdivision: Mapping[str, Any],
) -> Any:
    modifier = obj.modifiers.new("OpenUSD Catmull-Clark", "SUBSURF")
    modifier.subdivision_type = "CATMULL_CLARK"
    modifier.levels = int(subdivision["level"])
    modifier.render_levels = int(subdivision["level"])
    modifier.quality = 6
    modifier.boundary_smooth = str(subdivision["boundary_smooth"])
    modifier.uv_smooth = str(subdivision["uv_smooth"])
    modifier.use_limit_surface = True
    modifier.use_creases = bool(subdivision.get("use_creases"))
    modifier.show_only_control_edges = False
    return modifier


def _link_occurrences(
    bpy: Any,
    matrix_type: Any,
    collection: Any,
    mesh: Any,
    occurrence_transforms: np.ndarray,
    name: str,
    subdivision: Mapping[str, Any] | None,
) -> list[Any]:
    objects = []
    for occurrence_index, matrix in enumerate(occurrence_transforms):
        obj = bpy.data.objects.new(f"{name}_{occurrence_index:05d}", mesh)
        obj.matrix_world = _matrix_world(matrix_type, matrix)
        collection.objects.link(obj)
        if subdivision is not None:
            _add_subdivision_modifier(obj, subdivision)
        objects.append(obj)
    return objects


def _render_content(segment: Mapping[str, Any]) -> bool:
    return bool(segment.get("is_render_content")) and str(
        segment.get("visibility") or ""
    ).casefold() != "invisible"


def _build_polygon_renderable(
    bpy: Any,
    matrix_type: Any,
    collection: Any,
    segment_key: str,
    segment: Mapping[str, Any],
    payload: SegmentGeometryPayload,
    plan: MaterialPlan,
    blender_material: Any,
    transparent: Any,
    uv_aliases: Mapping[str, str],
) -> _Renderable:
    orientation = str(payload.provenance.get("orientation") or "rightHanded")
    if orientation not in {"rightHanded", "leftHanded"}:
        raise TexturedRenderError(f"unsupported mesh orientation {orientation!r}")
    faces, loop_order = _polygon_arrays(
        payload.face_vertex_counts, payload.face_vertex_indices, orientation
    )
    name = f"vlm_poly_{_safe_component(str(segment.get('segment_id')))}"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(payload.points.tolist(), [], faces)
    mesh.update(calc_edges=True)
    if len(mesh.polygons) != len(payload.face_vertex_counts):
        raise TexturedRenderError(f"Blender changed polygon count for {segment_key!r}")
    _assign_uv_layers(
        mesh,
        {
            alias: payload.uv_sets[source].corner_values
            for source, alias in uv_aliases.items()
        },
        loop_order,
    )
    _assign_selected_display_attributes(mesh, segment, payload, loop_order, plan)
    mesh.materials.append(blender_material)
    mesh.materials.append(transparent)
    actual_index = 0
    transparent_index = 1
    whole_index = actual_index if _render_content(segment) else transparent_index
    for polygon in mesh.polygons:
        polygon.material_index = whole_index
        polygon.use_smooth = payload.normals is not None
    if payload.normals is not None:
        normals = np.asarray(payload.normals.corner_values, dtype=np.float64)
        if normals.shape != (len(loop_order), 3) or not np.isfinite(normals).all():
            raise TexturedRenderError(f"authored normals are invalid for {segment_key!r}")
        ordered_normals = normals[loop_order]
        setter = getattr(mesh, "normals_split_custom_set", None)
        if setter is None:
            raise TexturedRenderError(
                "Blender 4.2 cannot accept exact authored corner normals in this runtime"
            )
        setter([tuple(float(value) for value in normal) for normal in ordered_normals])
    objects = _link_occurrences(
        bpy,
        matrix_type,
        collection,
        mesh,
        payload.occurrence_transforms,
        name,
        None,
    )
    return _Renderable(
        objects=objects,
        mesh=mesh,
        face_segments=tuple(segment_key for _ in mesh.polygons),
        segment_material_indices={segment_key: actual_index},
        whole_material_indices=tuple(whole_index for _ in mesh.polygons),
        transparent_index=transparent_index,
        segment_keys=frozenset({segment_key}),
    )


def _subdivision_settings(
    payload: SegmentGeometryPayload,
    subdivision_level: int,
) -> dict[str, Any]:
    if type(subdivision_level) is not int or not 1 <= subdivision_level <= 3:
        raise TexturedRenderError(
            f"invalid asset-wide Catmull-Clark level {subdivision_level!r}"
        )
    provenance = payload.provenance
    boundary = str(provenance.get("interpolate_boundary") or "edgeAndCorner")
    boundary_mapping = {"edgeAndCorner": "PRESERVE_CORNERS", "edgeOnly": "ALL"}
    if boundary not in boundary_mapping:
        raise TexturedRenderError(f"unsupported interpolateBoundary={boundary!r}")
    face_varying = str(
        provenance.get("face_varying_linear_interpolation") or "cornersPlus1"
    )
    uv_mapping = {
        "none": "NONE",
        "cornersOnly": "PRESERVE_CORNERS",
        "cornersPlus1": "PRESERVE_CORNERS_AND_JUNCTIONS",
        "cornersPlus2": "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE",
        "boundaries": "PRESERVE_BOUNDARIES",
        "all": "SMOOTH_ALL",
    }
    if face_varying not in uv_mapping:
        raise TexturedRenderError(
            f"unsupported faceVaryingLinearInterpolation={face_varying!r}"
        )
    triangle_rule = str(provenance.get("triangle_subdivision_rule") or "catmullClark")
    if triangle_rule != "catmullClark":
        raise TexturedRenderError(
            f"unsupported Catmull-Clark triangleSubdivisionRule={triangle_rule!r}"
        )
    control_indices = payload.subdivision_control_face_vertex_indices
    control_corners = len(control_indices) if control_indices is not None else 0
    if control_corners < 1:
        raise TexturedRenderError("Catmull-Clark payload has an empty control surface")
    return {
        "level": subdivision_level,
        "boundary_smooth": boundary_mapping[boundary],
        "uv_smooth": uv_mapping[face_varying],
        "usd_interpolate_boundary": boundary,
        "usd_face_varying_linear_interpolation": face_varying,
        "usd_triangle_subdivision_rule": triangle_rule,
        "use_creases": bool(
            provenance.get("crease_lengths") or provenance.get("corner_indices")
        ),
    }


def _assign_subdivision_creases(
    mesh: Any,
    group: list[tuple[str, Mapping[str, Any], SegmentGeometryPayload]],
    payload: SegmentGeometryPayload,
) -> None:
    provenance = payload.provenance
    crease_lengths = [int(value) for value in provenance.get("crease_lengths") or []]
    crease_indices = [int(value) for value in provenance.get("crease_indices") or []]
    crease_sharpnesses = [
        float(value) for value in provenance.get("crease_sharpnesses") or []
    ]
    corner_indices = [int(value) for value in provenance.get("corner_indices") or []]
    corner_sharpnesses = [
        float(value) for value in provenance.get("corner_sharpnesses") or []
    ]
    if not crease_lengths and not corner_indices:
        return
    source_vertices = np.unique(
        np.concatenate(
            [np.asarray(item.source_vertex_indices, dtype=np.int64) for _, _, item in group]
        )
    )
    if len(source_vertices) != len(mesh.vertices):
        raise TexturedRenderError("cannot recover source vertices for subdivision creases")
    source_to_local = {int(source): index for index, source in enumerate(source_vertices)}

    if crease_lengths:
        if any(length < 2 for length in crease_lengths) or sum(crease_lengths) != len(
            crease_indices
        ):
            raise TexturedRenderError("USD creaseLengths/creaseIndices cardinality is invalid")
        edge_count = sum(length - 1 for length in crease_lengths)
        if len(crease_sharpnesses) not in {len(crease_lengths), edge_count}:
            raise TexturedRenderError("USD creaseSharpnesses cardinality is invalid")
        if any(not math.isfinite(value) or value < 0.0 for value in crease_sharpnesses):
            raise TexturedRenderError("USD crease sharpness is negative or non-finite")
        edge_lookup = {
            tuple(sorted(int(value) for value in edge.vertices)): index
            for index, edge in enumerate(mesh.edges)
        }
        sharpness_by_edge: dict[int, float] = {}
        point_offset = 0
        edge_offset = 0
        per_edge = len(crease_sharpnesses) == edge_count
        for crease_index, length in enumerate(crease_lengths):
            chain = crease_indices[point_offset : point_offset + length]
            point_offset += length
            for chain_edge in range(length - 1):
                try:
                    pair = tuple(
                        sorted(
                            (
                                source_to_local[chain[chain_edge]],
                                source_to_local[chain[chain_edge + 1]],
                            )
                        )
                    )
                except KeyError as exc:
                    raise TexturedRenderError(
                        f"USD crease references source point absent from control mesh: {exc}"
                    ) from exc
                edge_index = edge_lookup.get(pair)
                if edge_index is None:
                    raise TexturedRenderError(f"USD crease pair {pair!r} is not a mesh edge")
                sharpness = (
                    crease_sharpnesses[edge_offset]
                    if per_edge
                    else crease_sharpnesses[crease_index]
                )
                previous = sharpness_by_edge.get(edge_index)
                if previous is not None and previous != sharpness:
                    raise TexturedRenderError("overlapping USD creases disagree in sharpness")
                sharpness_by_edge[edge_index] = sharpness
                edge_offset += 1
        attribute = mesh.attributes.new("crease_edge", "FLOAT", "EDGE")
        if attribute is None:
            raise TexturedRenderError("Blender could not create crease_edge")
        for edge_index, sharpness in sharpness_by_edge.items():
            attribute.data[edge_index].value = sharpness

    if corner_indices:
        if len(corner_indices) != len(corner_sharpnesses):
            raise TexturedRenderError("USD corner index/sharpness cardinality is invalid")
        if any(not math.isfinite(value) or value < 0.0 for value in corner_sharpnesses):
            raise TexturedRenderError("USD corner sharpness is negative or non-finite")
        sharpness_by_vertex: dict[int, float] = {}
        for source_index, sharpness in zip(corner_indices, corner_sharpnesses):
            if source_index not in source_to_local:
                raise TexturedRenderError(
                    f"USD corner references source point absent from control mesh: {source_index}"
                )
            local_index = source_to_local[source_index]
            previous = sharpness_by_vertex.get(local_index)
            if previous is not None and previous != sharpness:
                raise TexturedRenderError("duplicate USD corners disagree in sharpness")
            sharpness_by_vertex[local_index] = sharpness
        attribute = mesh.attributes.new("crease_vert", "FLOAT", "POINT")
        if attribute is None:
            raise TexturedRenderError("Blender could not create crease_vert")
        for vertex_index, sharpness in sharpness_by_vertex.items():
            attribute.data[vertex_index].value = sharpness


def _catmull_group_key(payload: SegmentGeometryPayload) -> str:
    manifest = payload.manifest()
    control = manifest.get("subdivision_control_surface")
    if not isinstance(control, Mapping):
        raise TexturedRenderError("Catmull-Clark payload lacks control-surface provenance")
    return _canonical_hash(
        "catmull-render-group",
        {
            "prim_path": payload.provenance.get("prim_path"),
            "control": control,
            "occurrence_transforms_sha256": manifest.get(
                "occurrence_transforms_sha256"
            ),
        },
    )


def _catmull_crease_provenance(payload: SegmentGeometryPayload) -> dict[str, Any]:
    fields = (
        "crease_lengths",
        "crease_lengths_sha256",
        "crease_indices",
        "crease_indices_sha256",
        "crease_sharpnesses",
        "crease_sharpnesses_sha256",
        "corner_indices",
        "corner_indices_sha256",
        "corner_sharpnesses",
        "corner_sharpnesses_sha256",
    )
    return {field: payload.provenance.get(field) for field in fields}


def _asset_catmull_subdivision_level(
    payloads: Mapping[str, SegmentGeometryPayload],
) -> int:
    control_corners_by_group: dict[str, int] = {}
    payloads_by_group: dict[str, list[SegmentGeometryPayload]] = {}
    for payload in payloads.values():
        scheme = str(
            payload.provenance.get("subdivision_scheme") or "catmullClark"
        )
        if scheme == "none":
            continue
        if scheme != "catmullClark":
            raise TexturedRenderError(
                f"unsupported subdivisionScheme={scheme!r} at "
                f"{payload.provenance.get('prim_path')}"
            )
        control_indices = payload.subdivision_control_face_vertex_indices
        control_corners = len(control_indices) if control_indices is not None else 0
        if control_corners < 1:
            raise TexturedRenderError(
                "Catmull-Clark payload has an empty control surface"
            )
        group_key = _catmull_group_key(payload)
        payloads_by_group.setdefault(group_key, []).append(payload)
        previous = control_corners_by_group.setdefault(group_key, control_corners)
        if previous != control_corners:
            raise TexturedRenderError(
                "Catmull-Clark group disagrees about its control-corner count"
            )
    total_control_corners = sum(control_corners_by_group.values())
    # Use fewer subdivision steps for big assets to keep mesh size manageable.
    subdivision_level = 3
    while (
        subdivision_level > 1
        and total_control_corners * (4**subdivision_level) > 2_000_000
    ):
        subdivision_level -= 1
    for group in payloads_by_group.values():
        canonical_settings = _subdivision_settings(group[0], subdivision_level)
        canonical_creases = _catmull_crease_provenance(group[0])
        if any(
            _subdivision_settings(payload, subdivision_level)
            != canonical_settings
            or _catmull_crease_provenance(payload) != canonical_creases
            for payload in group[1:]
        ):
            raise TexturedRenderError(
                "Catmull-Clark group disagrees about subdivision or crease provenance"
            )
    return subdivision_level


def _payload_face_uvs(
    payload: SegmentGeometryPayload,
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    offset = 0
    for source_face, count_value in zip(
        payload.source_face_indices, payload.face_vertex_counts
    ):
        count = int(count_value)
        face = int(source_face)
        if face in result:
            raise TexturedRenderError("payload repeats a source face")
        result[face] = {
            name: np.asarray(attribute.corner_values[offset : offset + count], dtype=np.float64)
            for name, attribute in payload.uv_sets.items()
        }
        offset += count
    if offset != len(payload.face_vertex_indices):
        raise TexturedRenderError("payload UV face offsets disagree with topology")
    return result


def _group_display_record(
    group: list[tuple[str, Mapping[str, Any], SegmentGeometryPayload]],
    plans: Mapping[str, MaterialPlan],
    primvar_name: str,
) -> Mapping[str, Any] | None:
    plan_field = (
        "display_color_attribute"
        if primvar_name == "displayColor"
        else "display_opacity_attribute"
    )
    records = [
        _display_primvar_record(segment, primvar_name)
        for key, segment, _ in group
        if getattr(plans[key], plan_field) is not None
    ]
    present = [record for record in records if record is not None]
    if not present:
        return None
    signatures = {
        _canonical_hash(
            f"{primvar_name}-control-mesh-record",
            {
                "primvar_path": record.get("primvar_path"),
                "interpolation": record.get("interpolation"),
                "element_size": record.get("element_size"),
                "effective_color_space": record.get("effective_color_space"),
                "computed_flattened_values": record.get("computed_flattened_values"),
            },
        )
        for record in present
    }
    if len(signatures) != 1:
        raise TexturedRenderError(
            f"Catmull-Clark regions disagree about exact {primvar_name}"
        )
    return present[0]


def _control_source_corners(
    group: list[tuple[str, Mapping[str, Any], SegmentGeometryPayload]],
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for _, _, payload in group:
        offset = 0
        for source_face, count_value in zip(
            payload.source_face_indices, payload.face_vertex_counts
        ):
            count = int(count_value)
            face = int(source_face)
            corners = np.asarray(
                payload.source_corner_indices[offset : offset + count], dtype=np.int64
            )
            if face in result and not np.array_equal(result[face], corners):
                raise TexturedRenderError("source-corner provenance disagrees across regions")
            result[face] = corners
            offset += count
    return result


def _assign_catmull_display_attribute(
    mesh: Any,
    group: list[tuple[str, Mapping[str, Any], SegmentGeometryPayload]],
    plans: Mapping[str, MaterialPlan],
    control_source_faces: np.ndarray,
    control_point_count: int,
    loop_order: np.ndarray,
    primvar_name: str,
    attribute_name: str,
    *,
    color: bool,
) -> None:
    record = _group_display_record(group, plans, primvar_name)
    if record is None:
        return
    values = _display_values(record, color=color)
    unique = {tuple(value) if color else float(value) for value in values}
    if len(unique) <= 1:
        return
    interpolation = str(record.get("interpolation") or "constant")
    if interpolation == "uniform":
        indices = np.asarray(control_source_faces, dtype=np.int64)
        if indices.size and int(indices.max()) >= len(values):
            raise TexturedRenderError(f"{primvar_name} control-face index exceeds values")
        selected = np.repeat(
            values[indices],
            np.asarray([len(polygon.loop_indices) for polygon in mesh.polygons]),
            axis=0,
        )
        _new_mesh_attribute(
            mesh,
            attribute_name,
            "CORNER",
            selected[loop_order],
            color=color,
        )
        return
    if interpolation in {"vertex", "varying"}:
        source_vertices = np.unique(
            np.concatenate(
                [np.asarray(payload.source_vertex_indices, dtype=np.int64) for _, _, payload in group]
            )
        )
        if len(source_vertices) != control_point_count:
            raise TexturedRenderError(
                f"cannot recover exact control-vertex provenance for {primvar_name}"
            )
        if source_vertices.size and int(source_vertices.max()) >= len(values):
            raise TexturedRenderError(f"{primvar_name} control-vertex index exceeds values")
        _new_mesh_attribute(
            mesh, attribute_name, "POINT", values[source_vertices], color=color
        )
        return
    if interpolation == "faceVarying":
        source_corners = _control_source_corners(group)
        ordered_indices = np.concatenate(
            [source_corners[int(face)] for face in control_source_faces]
        )
        if ordered_indices.size and int(ordered_indices.max()) >= len(values):
            raise TexturedRenderError(f"{primvar_name} control-corner index exceeds values")
        _new_mesh_attribute(
            mesh,
            attribute_name,
            "CORNER",
            values[ordered_indices][loop_order],
            color=color,
        )
        return
    if interpolation == "constant":
        repeated = np.repeat(values[:1], len(mesh.loops), axis=0)
        _new_mesh_attribute(mesh, attribute_name, "CORNER", repeated, color=color)
        return
    raise TexturedRenderError(f"unsupported {primvar_name} interpolation {interpolation!r}")


def _attribute_domain_size(mesh: Any, domain: str) -> int:
    sizes = {
        "POINT": len(mesh.vertices),
        "EDGE": len(mesh.edges),
        "FACE": len(mesh.polygons),
        "CORNER": len(mesh.loops),
    }
    if domain not in sizes:
        raise TexturedRenderError(
            f"evaluated mesh attribute has unsupported domain {domain!r}"
        )
    return sizes[domain]


def _validate_evaluated_display_attribute(
    control_mesh: Any,
    evaluated_mesh: Any,
    name: str,
    *,
    color: bool,
) -> None:
    control_collection = (
        control_mesh.color_attributes if color else control_mesh.attributes
    )
    evaluated_collection = (
        evaluated_mesh.color_attributes if color else evaluated_mesh.attributes
    )
    control = control_collection.get(name)
    evaluated = evaluated_collection.get(name)
    if (control is None) != (evaluated is None):
        raise TexturedRenderError(
            f"Catmull-Clark evaluation changed {name} attribute presence"
        )
    if control is None:
        return
    control_domain = str(control.domain)
    evaluated_domain = str(evaluated.domain)
    control_type = str(control.data_type)
    evaluated_type = str(evaluated.data_type)
    if control_domain != evaluated_domain or control_type != evaluated_type:
        raise TexturedRenderError(
            f"Catmull-Clark evaluation changed {name} attribute layout"
        )
    if len(evaluated.data) != _attribute_domain_size(
        evaluated_mesh, evaluated_domain
    ):
        raise TexturedRenderError(
            f"evaluated {name} attribute has invalid domain cardinality"
        )
    component_count = 4 if color else 1
    values = np.empty(len(evaluated.data) * component_count, dtype=np.float32)
    evaluated.data.foreach_get("color" if color else "value", values)
    if not np.isfinite(values).all():
        raise TexturedRenderError(
            f"evaluated {name} attribute contains non-finite values"
        )


def _evaluate_static_catmull_mesh(
    bpy: Any,
    matrix_type: Any,
    collection: Any,
    control_mesh: Any,
    group_id: str,
    subdivision: Mapping[str, Any],
    material_indices: Mapping[str, int],
    expected_uv_names: list[str],
) -> tuple[Any, tuple[str, ...]]:
    material_to_segment = {
        int(material_index): segment_key
        for segment_key, material_index in material_indices.items()
    }
    if len(material_to_segment) != len(material_indices):
        raise TexturedRenderError(
            "Catmull-Clark segment material slots are not unique"
        )
    expected_materials = tuple(control_mesh.materials)
    temporary = bpy.data.objects.new(
        f"vlm_catmull_evaluate_{group_id[:16]}", control_mesh
    )
    temporary.matrix_world = matrix_type.Identity(4)
    collection.objects.link(temporary)
    evaluated_mesh = None
    try:
        _add_subdivision_modifier(temporary, subdivision)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_mesh = bpy.data.meshes.new_from_object(
            temporary.evaluated_get(depsgraph),
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        if (
            evaluated_mesh is None
            or not evaluated_mesh.vertices
            or not evaluated_mesh.loops
            or not evaluated_mesh.polygons
        ):
            raise TexturedRenderError(
                "one-time Catmull-Clark evaluation produced no renderable surface"
            )
        evaluated_mesh.name = f"vlm_catmull_static_{group_id[:16]}"
        evaluated_materials = tuple(evaluated_mesh.materials)
        if len(evaluated_materials) != len(expected_materials) or any(
            actual is not expected
            for actual, expected in zip(evaluated_materials, expected_materials)
        ):
            raise TexturedRenderError(
                "Catmull-Clark evaluation changed exact material slots"
            )

        polygon_material_indices = np.empty(
            len(evaluated_mesh.polygons), dtype=np.int32
        )
        evaluated_mesh.polygons.foreach_get(
            "material_index", polygon_material_indices
        )
        unknown_material_indices = sorted(
            set(int(value) for value in polygon_material_indices)
            - set(material_to_segment)
        )
        if unknown_material_indices:
            raise TexturedRenderError(
                "Catmull-Clark evaluation lost exact face ownership; "
                f"unexpected material slots={unknown_material_indices!r}"
            )
        face_segments = tuple(
            material_to_segment[int(value)] for value in polygon_material_indices
        )
        if set(face_segments) != set(material_indices):
            raise TexturedRenderError(
                "Catmull-Clark evaluation did not preserve every exact segment"
            )

        control_uv_names = sorted(layer.name for layer in control_mesh.uv_layers)
        evaluated_uv_names = sorted(layer.name for layer in evaluated_mesh.uv_layers)
        if (
            control_uv_names != sorted(expected_uv_names)
            or evaluated_uv_names != control_uv_names
        ):
            raise TexturedRenderError(
                "Catmull-Clark evaluation changed exact material UV layers"
            )
        for name in evaluated_uv_names:
            layer = evaluated_mesh.uv_layers.get(name)
            if layer is None or len(layer.data) != len(evaluated_mesh.loops):
                raise TexturedRenderError(
                    f"evaluated material UV alias {name!r} has invalid cardinality"
                )
            values = np.empty(len(layer.data) * 2, dtype=np.float32)
            layer.data.foreach_get("uv", values)
            if not np.isfinite(values).all():
                raise TexturedRenderError(
                    f"evaluated material UV alias {name!r} is non-finite"
                )
        _validate_evaluated_display_attribute(
            control_mesh,
            evaluated_mesh,
            "vlm_displayColor",
            color=True,
        )
        _validate_evaluated_display_attribute(
            control_mesh,
            evaluated_mesh,
            "vlm_displayOpacity",
            color=False,
        )
        _corner_normals(evaluated_mesh)
        return evaluated_mesh, face_segments
    except Exception:
        if (
            evaluated_mesh is not None
            and bpy.data.meshes.get(evaluated_mesh.name) is evaluated_mesh
        ):
            bpy.data.meshes.remove(evaluated_mesh)
        raise
    finally:
        if bpy.data.objects.get(temporary.name) is temporary:
            bpy.data.objects.remove(temporary, do_unlink=True)


def _build_catmull_renderable(
    bpy: Any,
    matrix_type: Any,
    collection: Any,
    group: list[tuple[str, Mapping[str, Any], SegmentGeometryPayload]],
    plans: Mapping[str, MaterialPlan],
    blender_materials: Mapping[str, Any],
    transparent: Any,
    uv_aliases: Mapping[str, Mapping[str, str]],
    subdivision_level: int,
) -> _Renderable:
    canonical = group[0][2]
    control_arrays = (
        canonical.subdivision_control_points,
        canonical.subdivision_control_face_vertex_counts,
        canonical.subdivision_control_face_vertex_indices,
        canonical.subdivision_control_source_face_indices,
    )
    if not all(value is not None for value in control_arrays):
        raise TexturedRenderError("Catmull-Clark group has incomplete control arrays")
    control_points = canonical.subdivision_control_points
    control_counts = canonical.subdivision_control_face_vertex_counts
    control_indices = canonical.subdivision_control_face_vertex_indices
    control_source_faces = canonical.subdivision_control_source_face_indices
    assert control_points is not None
    assert control_counts is not None
    assert control_indices is not None
    assert control_source_faces is not None
    subdivision = _subdivision_settings(canonical, subdivision_level)
    crease_provenance = _catmull_crease_provenance(canonical)
    for key, _, payload in group[1:]:
        comparisons = (
            np.array_equal(control_points, payload.subdivision_control_points),
            np.array_equal(control_counts, payload.subdivision_control_face_vertex_counts),
            np.array_equal(control_indices, payload.subdivision_control_face_vertex_indices),
            np.array_equal(
                control_source_faces, payload.subdivision_control_source_face_indices
            ),
            np.array_equal(canonical.occurrence_transforms, payload.occurrence_transforms),
            _subdivision_settings(payload, subdivision_level) == subdivision,
            _catmull_crease_provenance(payload) == crease_provenance,
        )
        if not all(comparisons):
            raise TexturedRenderError(
                f"Catmull-Clark region {key!r} disagrees with its whole control mesh"
            )

    owner_by_face: dict[int, str] = {}
    segment_by_key = {key: segment for key, segment, _ in group}
    uvs_by_face: dict[int, dict[str, np.ndarray]] = {}
    for key, _, payload in group:
        payload_uvs = _payload_face_uvs(payload)
        for source_face in payload.source_face_indices:
            face = int(source_face)
            if face in owner_by_face:
                raise TexturedRenderError(
                    f"Catmull-Clark source face {face} belongs to multiple exact segments"
                )
            owner_by_face[face] = key
            uvs_by_face[face] = payload_uvs[face]
    control_face_set = {int(value) for value in control_source_faces}
    if set(owner_by_face) != control_face_set:
        raise TexturedRenderError(
            "Catmull-Clark exact segments do not partition every non-hole control face"
        )
    alias_names = sorted(
        {
            alias
            for key, _, _ in group
            for alias in uv_aliases[key].values()
        }
    )
    combined_uvs: dict[str, np.ndarray] = {}
    for alias in alias_names:
        pieces = []
        for source_face, count_value in zip(control_source_faces, control_counts):
            face = int(source_face)
            owner = owner_by_face[face]
            source_names = [
                source
                for source, owner_alias in uv_aliases[owner].items()
                if owner_alias == alias
            ]
            if len(source_names) > 1:
                raise TexturedRenderError(
                    f"material region {owner!r} maps multiple UV sets to {alias!r}"
                )
            values = (
                uvs_by_face[face].get(source_names[0])
                if source_names
                else np.zeros((int(count_value), 2), dtype=np.float64)
            )
            if values is None or len(values) != int(count_value):
                raise TexturedRenderError(
                    f"Catmull-Clark UV alias {alias!r} lacks exact values on "
                    f"material region {owner!r}"
                )
            pieces.append(values)
        combined_uvs[alias] = np.concatenate(pieces, axis=0)

    orientation = str(canonical.provenance.get("orientation") or "rightHanded")
    if orientation not in {"rightHanded", "leftHanded"}:
        raise TexturedRenderError(f"unsupported mesh orientation {orientation!r}")
    faces, loop_order = _polygon_arrays(control_counts, control_indices, orientation)
    group_id = _catmull_group_key(canonical)
    mesh = bpy.data.meshes.new(f"vlm_catmull_{group_id[:16]}")
    mesh.from_pydata(control_points.tolist(), [], faces)
    mesh.update(calc_edges=True)
    _assign_subdivision_creases(mesh, group, canonical)
    _assign_uv_layers(mesh, combined_uvs, loop_order)
    _assign_catmull_display_attribute(
        mesh,
        group,
        plans,
        control_source_faces,
        len(control_points),
        loop_order,
        "displayColor",
        "vlm_displayColor",
        color=True,
    )
    _assign_catmull_display_attribute(
        mesh,
        group,
        plans,
        control_source_faces,
        len(control_points),
        loop_order,
        "displayOpacity",
        "vlm_displayOpacity",
        color=False,
    )

    segment_keys = sorted(segment_by_key)
    material_indices: dict[str, int] = {}
    for key in segment_keys:
        material_indices[key] = len(mesh.materials)
        mesh.materials.append(blender_materials[key])
    transparent_index = len(mesh.materials)
    mesh.materials.append(transparent)
    control_face_segments = tuple(
        owner_by_face[int(face)] for face in control_source_faces
    )
    if len(control_face_segments) != len(mesh.polygons):
        raise TexturedRenderError(
            "Catmull-Clark control face ownership has invalid cardinality"
        )
    for polygon, owner in zip(mesh.polygons, control_face_segments):
        polygon.material_index = material_indices[owner]
        polygon.use_smooth = True

    control_mesh = mesh
    mesh, face_segments = _evaluate_static_catmull_mesh(
        bpy,
        matrix_type,
        collection,
        control_mesh,
        group_id,
        subdivision,
        material_indices,
        alias_names,
    )
    if bpy.data.meshes.get(control_mesh.name) is control_mesh:
        bpy.data.meshes.remove(control_mesh)
    whole_indices = tuple(
        material_indices[owner]
        if _render_content(segment_by_key[owner])
        else transparent_index
        for owner in face_segments
    )
    if len(whole_indices) != len(mesh.polygons):
        raise TexturedRenderError(
            "evaluated Catmull-Clark whole-material state has invalid cardinality"
        )
    for polygon, index in zip(mesh.polygons, whole_indices):
        polygon.material_index = index
    objects = _link_occurrences(
        bpy,
        matrix_type,
        collection,
        mesh,
        canonical.occurrence_transforms,
        f"vlm_catmull_{group_id[:16]}",
        None,
    )
    if len(objects) != len(canonical.occurrence_transforms):
        raise TexturedRenderError(
            "static Catmull-Clark occurrence coverage changed"
        )
    for obj, transform in zip(objects, canonical.occurrence_transforms):
        expected_matrix = _matrix_world(matrix_type, transform)
        actual = np.asarray(
            [
                [float(component) for component in row]
                for row in obj.matrix_world
            ],
            dtype=np.float64,
        )
        expected = np.asarray(
            [
                [float(component) for component in row]
                for row in expected_matrix
            ],
            dtype=np.float64,
        )
        if (
            obj.data is not mesh
            or len(obj.modifiers) != 0
            or not np.array_equal(actual, expected)
        ):
            raise TexturedRenderError(
                "static Catmull-Clark occurrence is not an exact shared-mesh instance"
            )
    return _Renderable(
        objects=objects,
        mesh=mesh,
        face_segments=face_segments,
        segment_material_indices=material_indices,
        whole_material_indices=whole_indices,
        transparent_index=transparent_index,
        segment_keys=frozenset(segment_keys),
    )


def _build_asset_renderables(
    bpy: Any,
    matrix_type: Any,
    collection: Any,
    segments: Mapping[str, Any],
    payloads: Mapping[str, SegmentGeometryPayload],
    plans: Mapping[str, MaterialPlan],
    blender_materials: Mapping[str, Any],
    transparent: Any,
    uv_aliases: Mapping[str, Mapping[str, str]],
    subdivision_level: int,
) -> list[_Renderable]:
    renderables: list[_Renderable] = []
    catmull_groups: dict[
        str, list[tuple[str, Mapping[str, Any], SegmentGeometryPayload]]
    ] = {}
    for key, payload in payloads.items():
        scheme = str(payload.provenance.get("subdivision_scheme") or "catmullClark")
        segment = segments[key]
        if scheme == "none":
            renderables.append(
                _build_polygon_renderable(
                    bpy,
                    matrix_type,
                    collection,
                    str(key),
                    segment,
                    payload,
                    plans[str(key)],
                    blender_materials[key],
                    transparent,
                    uv_aliases[str(key)],
                )
            )
        elif scheme == "catmullClark":
            catmull_groups.setdefault(_catmull_group_key(payload), []).append(
                (str(key), segment, payload)
            )
        else:
            raise TexturedRenderError(
                f"unsupported subdivisionScheme={scheme!r} at "
                f"{payload.provenance.get('prim_path')}"
            )
    for key in sorted(catmull_groups):
        renderables.append(
            _build_catmull_renderable(
                bpy,
                matrix_type,
                collection,
                catmull_groups[key],
                plans,
                blender_materials,
                transparent,
                uv_aliases,
                subdivision_level,
            )
        )
    covered = set().union(*(item.segment_keys for item in renderables))
    if covered != set(segments):
        raise TexturedRenderError("Blender scene does not cover every exact segment")
    return renderables


_SEGMENT_SWATCH_UV = "SegmentSwatchUV"
# Blender 4.2 stores custom corner normals as two signed 16-bit angular values
# and snaps nearly collinear directions at its 1e-4 trigonometric threshold.
# The resulting worst-case direction delta is about sqrt(2e-4) radians.
_CUSTOM_NORMAL_DIRECTION_ERROR_LIMIT_RADIANS = 0.015
_DISPLAY_COLOR_SWATCH = "__display_color"
_DISPLAY_OPACITY_SWATCH = "__display_opacity"
_SWATCH_TANGENT_NORMAL_ROLES = frozenset(
    {"normal", "detail_normal", "coat_normal"}
)

_SWATCH_RECIPE = {
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
_SWATCH_RECIPE_SHA256 = _canonical_hash(
    "selected-segment-swatch-recipe", _SWATCH_RECIPE
)


@dataclass
class _BakeProxy:
    object: Any
    source_uvs: dict[str, np.ndarray]
    source_normals: np.ndarray
    record: dict[str, Any]


def _attest_proxy_normal_state(
    proxies: list[_BakeProxy], operation: str
) -> None:
    for proxy in proxies:
        current = _array_hash(
            "swatch-renderer-custom-corner-normals",
            _corner_normals(proxy.object.data),
        )
        if current != proxy.record.get(
            "renderer_custom_corner_normals_sha256"
        ):
            raise TexturedRenderError(
                f"renderer normals changed during {operation}"
            )


_SWATCH_COLOR_ROLES = frozenset(
    {
        "albedo",
        "coat_color",
        "emissive",
        "emissive_mask",
        "retroreflection_color",
        "subsurface_scattering_color",
        "subsurface_transmission_color",
        "transmission_color",
        "transmission_scattering_color",
    }
)


def _expected_swatch_roles(plan: MaterialPlan) -> set[str]:
    roles = {texture.role for texture in plan.textures}
    if (
        plan.material_model == "principled"
        and plan.display_color_attribute
        and "albedo" not in roles
    ):
        roles.add(_DISPLAY_COLOR_SWATCH)
    if (
        plan.material_model == "principled"
        and plan.display_opacity_attribute
        and "opacity" not in roles
    ):
        roles.add(_DISPLAY_OPACITY_SWATCH)
    return roles


def _swatch_kind(plan: MaterialPlan, texture: TexturePlan) -> str:
    if texture.role in _SWATCH_TANGENT_NORMAL_ROLES:
        return "tangent_normal"
    if texture.role in _SWATCH_COLOR_ROLES:
        return "color"
    if texture.role == "specular" and plan.material_model in {
        "omniglass",
        "omnisurface",
    }:
        return "color"
    if texture.role == "occlusion" and texture.channel in {"rgb", "rgba"}:
        return "color"
    return "scalar"


def _image_pixels(image: Any) -> np.ndarray:
    width = int(image.size[0])
    height = int(image.size[1])
    if width < 1 or height < 1:
        raise TexturedRenderError("Cycles bake target has invalid dimensions")
    values = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(values)
    values = values.reshape((height, width, 4))
    if not np.isfinite(values).all():
        raise TexturedRenderError("Cycles bake produced non-finite pixels")
    return values


def _new_bake_image(bpy: Any, name: str, size: int) -> Any:
    image = bpy.data.images.new(
        name=name,
        width=size,
        height=size,
        alpha=True,
        float_buffer=True,
        is_data=True,
    )
    if image is None or tuple(int(value) for value in image.size) != (size, size):
        raise TexturedRenderError("Blender could not create the segment swatch image")
    image.colorspace_settings.name = "Non-Color"
    if image.colorspace_settings.name != "Non-Color":
        raise TexturedRenderError("Blender could not retain linear swatch pixels")
    image.alpha_mode = "STRAIGHT"
    image.generated_color = (0.0, 0.0, 0.0, 0.0)
    return image


def _corner_normals(mesh: Any) -> np.ndarray:
    values = getattr(mesh, "corner_normals", None)
    if values is None or len(values) != len(mesh.loops):
        raise TexturedRenderError("evaluated mesh has no exact split corner normals")
    result = np.asarray([tuple(item.vector) for item in values], dtype=np.float32)
    if result.shape != (len(mesh.loops), 3) or not np.isfinite(result).all():
        raise TexturedRenderError("evaluated split corner normals are invalid")
    return result


def _custom_normal_equivalence(
    source_normals: np.ndarray, restored_normals: np.ndarray
) -> dict[str, float]:
    if source_normals.shape != restored_normals.shape:
        raise TexturedRenderError("restored custom corner-normal shape changed")
    source = source_normals.astype(np.float64)
    restored = restored_normals.astype(np.float64)
    source_lengths = np.linalg.norm(source, axis=1)
    restored_lengths = np.linalg.norm(restored, axis=1)
    if (
        not np.isfinite(source_lengths).all()
        or not np.isfinite(restored_lengths).all()
        or np.any(source_lengths <= 1.0e-12)
        or np.any(restored_lengths <= 1.0e-12)
    ):
        raise TexturedRenderError("custom corner-normal roundtrip contains zero vectors")
    dots = np.einsum(
        "ij,ij->i",
        source / source_lengths[:, None],
        restored / restored_lengths[:, None],
    )
    maximum_error = float(np.arccos(np.clip(dots, -1.0, 1.0)).max())
    if maximum_error > _CUSTOM_NORMAL_DIRECTION_ERROR_LIMIT_RADIANS:
        raise TexturedRenderError(
            "renderer-active custom corner normals exceed Blender's encoding bound: "
            f"{maximum_error} radians"
        )
    return {
        "maximum_direction_error_radians": maximum_error,
        "minimum_direction_dot": float(dots.min()),
        "maximum_component_error": float(np.max(np.abs(restored - source))),
        "direction_error_limit_radians": (
            _CUSTOM_NORMAL_DIRECTION_ERROR_LIMIT_RADIANS
        ),
    }


def _selected_corner_attribute(
    mesh: Any,
    name: str,
    polygon_indices: list[int],
    source_loop_indices: np.ndarray,
    *,
    color: bool,
) -> np.ndarray | None:
    attribute = (
        mesh.color_attributes.get(name) if color else mesh.attributes.get(name)
    )
    if attribute is None:
        return None
    domain = str(attribute.domain)
    if domain not in {"CORNER", "POINT", "FACE"}:
        raise TexturedRenderError(
            f"evaluated {name} has unsupported domain {domain!r}"
        )
    values: list[Any] = []
    loop_offset = 0
    for polygon_index in polygon_indices:
        polygon = mesh.polygons[polygon_index]
        for source_loop_index in polygon.loop_indices:
            if int(source_loop_indices[loop_offset]) != int(source_loop_index):
                raise TexturedRenderError(f"evaluated {name} loop mapping changed")
            loop_offset += 1
            index = (
                int(source_loop_index)
                if domain == "CORNER"
                else int(mesh.loops[source_loop_index].vertex_index)
                if domain == "POINT"
                else int(polygon_index)
            )
            item = attribute.data[index]
            values.append(tuple(item.color) if color else float(item.value))
    result = np.asarray(values, dtype=np.float32)
    expected_shape = (len(source_loop_indices), 4) if color else (
        len(source_loop_indices),
    )
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise TexturedRenderError(f"evaluated {name} corner values are invalid")
    return result


def _isolated_evaluated_mesh(
    bpy: Any,
    evaluated_mesh: Any,
    selected_material_index: int,
    aliases: Mapping[str, str],
) -> tuple[Any, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    polygon_indices = [
        int(polygon.index)
        for polygon in evaluated_mesh.polygons
        if int(polygon.material_index) == selected_material_index
    ]
    if not polygon_indices:
        raise TexturedRenderError("evaluated mesh has no polygons for selected segment")
    faces = [
        [
            int(evaluated_mesh.loops[index].vertex_index)
            for index in evaluated_mesh.polygons[polygon_index].loop_indices
        ]
        for polygon_index in polygon_indices
    ]
    source_loop_indices = np.asarray(
        [
            int(loop_index)
            for polygon_index in polygon_indices
            for loop_index in evaluated_mesh.polygons[polygon_index].loop_indices
        ],
        dtype=np.int64,
    )
    vertices = np.asarray(
        [tuple(vertex.co) for vertex in evaluated_mesh.vertices], dtype=np.float64
    )
    source_normals = _corner_normals(evaluated_mesh)[source_loop_indices]
    source_uvs: dict[str, np.ndarray] = {}
    for source_name, alias in aliases.items():
        layer = evaluated_mesh.uv_layers.get(alias)
        if layer is None or len(layer.data) != len(evaluated_mesh.loops):
            raise TexturedRenderError(
                f"evaluated segment lacks material UV {source_name!r}"
            )
        values = np.asarray(
            [tuple(layer.data[index].uv) for index in source_loop_indices],
            dtype=np.float64,
        )
        if values.shape != (len(source_loop_indices), 2) or not np.isfinite(
            values
        ).all():
            raise TexturedRenderError(
                f"evaluated segment material UV {source_name!r} is invalid"
            )
        source_uvs[alias] = values
    display_color = _selected_corner_attribute(
        evaluated_mesh,
        "vlm_displayColor",
        polygon_indices,
        source_loop_indices,
        color=True,
    )
    display_opacity = _selected_corner_attribute(
        evaluated_mesh,
        "vlm_displayOpacity",
        polygon_indices,
        source_loop_indices,
        color=False,
    )

    mesh = bpy.data.meshes.new("vlm_selected_evaluated_swatch_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update(calc_edges=True)
    actual_loop_vertices = np.asarray(
        [int(loop.vertex_index) for loop in mesh.loops], dtype=np.int64
    )
    expected_loop_vertices = np.asarray(
        [value for face in faces for value in face], dtype=np.int64
    )
    if not np.array_equal(actual_loop_vertices, expected_loop_vertices):
        raise TexturedRenderError("Blender changed selected evaluated loop topology")
    for polygon, source_polygon_index in zip(mesh.polygons, polygon_indices):
        polygon.material_index = selected_material_index
        polygon.use_smooth = bool(
            evaluated_mesh.polygons[source_polygon_index].use_smooth
        )
    for alias, values in sorted(source_uvs.items()):
        layer = mesh.uv_layers.new(name=alias)
        if layer is None or len(layer.data) != len(values):
            raise TexturedRenderError(
                f"Blender could not preserve material UV alias {alias!r}"
            )
        for destination, value in zip(layer.data, values):
            destination.uv = (float(value[0]), float(value[1]))
    if display_color is not None:
        _new_mesh_attribute(
            mesh,
            "vlm_displayColor",
            "CORNER",
            display_color,
            color=True,
        )
    if display_opacity is not None:
        _new_mesh_attribute(
            mesh,
            "vlm_displayOpacity",
            "CORNER",
            display_opacity,
            color=False,
        )
    setter = getattr(mesh, "normals_split_custom_set", None)
    if setter is None:
        raise TexturedRenderError("Blender cannot restore selected corner normals")
    setter([tuple(float(value) for value in normal) for normal in source_normals])
    mesh.update()
    restored_normals = _corner_normals(mesh)
    normal_equivalence = _custom_normal_equivalence(source_normals, restored_normals)
    restored_color = _selected_corner_attribute(
        mesh,
        "vlm_displayColor",
        list(range(len(mesh.polygons))),
        np.arange(len(mesh.loops), dtype=np.int64),
        color=True,
    )
    restored_opacity = _selected_corner_attribute(
        mesh,
        "vlm_displayOpacity",
        list(range(len(mesh.polygons))),
        np.arange(len(mesh.loops), dtype=np.int64),
        color=False,
    )
    if (
        (display_color is None) != (restored_color is None)
        or display_color is not None
        and not np.array_equal(display_color, restored_color)
        or (display_opacity is None) != (restored_opacity is None)
        or display_opacity is not None
        and not np.array_equal(display_opacity, restored_opacity)
    ):
        raise TexturedRenderError(
            "selected evaluated display attributes changed during isolation"
        )
    record = {
        "selected_source_polygon_indices_sha256": _array_hash(
            "swatch-selected-source-polygons",
            np.asarray(polygon_indices, dtype=np.int64),
        ),
        "selected_source_loop_indices_sha256": _array_hash(
            "swatch-selected-source-loops", source_loop_indices
        ),
        "vertices_sha256": _array_hash("swatch-evaluated-vertices", vertices),
        "loop_vertices_sha256": _array_hash(
            "swatch-evaluated-loop-vertices", actual_loop_vertices
        ),
        "corner_normals_sha256": _array_hash(
            "swatch-evaluated-corner-normals", source_normals
        ),
        "renderer_custom_corner_normals_sha256": _array_hash(
            "swatch-renderer-custom-corner-normals", restored_normals
        ),
        "renderer_custom_normal_equivalence": normal_equivalence,
        "source_uv_sha256": {
            alias: _array_hash("swatch-source-uv", values)
            for alias, values in sorted(source_uvs.items())
        },
        "display_color_sha256": (
            _array_hash("swatch-display-color", display_color)
            if display_color is not None
            else None
        ),
        "display_opacity_sha256": (
            _array_hash("swatch-display-opacity", display_opacity)
            if display_opacity is not None
            else None
        ),
        "polygon_count": len(mesh.polygons),
        "loop_count": len(mesh.loops),
        "vertex_count": len(mesh.vertices),
        "exact_corner_payload_and_display_pre_post_equal": True,
    }
    return mesh, source_uvs, source_normals, record


def _smart_project_proxy(
    bpy: Any,
    proxy: _BakeProxy,
    *,
    column: int,
    row: int,
    columns: int,
    rows: int,
    size: int,
    local_island_margin: float,
) -> None:
    mesh = proxy.object.data
    projection_mesh = mesh.copy()
    projection_object = proxy.object.copy()
    projection_object.data = projection_mesh
    source_collections = list(proxy.object.users_collection)
    if not source_collections:
        bpy.data.objects.remove(projection_object, do_unlink=True)
        bpy.data.meshes.remove(projection_mesh)
        raise TexturedRenderError("segment swatch proxy has no Blender collection")
    source_collections[0].objects.link(projection_object)
    view_layer = bpy.context.view_layer
    result = None
    try:
        for layer in list(projection_mesh.uv_layers):
            projection_mesh.uv_layers.remove(layer)
        projection_target = projection_mesh.uv_layers.new(
            name=_SEGMENT_SWATCH_UV
        )
        if projection_target is None:
            raise TexturedRenderError(
                "Blender could not create the projection-only segment swatch UV"
            )
        projection_mesh.uv_layers.active_index = projection_mesh.uv_layers.find(
            _SEGMENT_SWATCH_UV
        )
        for item in view_layer.objects:
            item.select_set(False)
        projection_object.hide_set(False)
        projection_object.hide_viewport = False
        projection_object.select_set(True)
        view_layer.objects.active = projection_object
        with bpy.context.temp_override(
            active_object=projection_object,
            object=projection_object,
            selected_objects=[projection_object],
            selected_editable_objects=[projection_object],
        ):
            if bpy.ops.object.mode_set(mode="EDIT") != {"FINISHED"}:
                raise TexturedRenderError(
                    "could not enter edit mode for segment swatch UV"
                )
            try:
                bpy.ops.mesh.select_all(action="SELECT")
                result = bpy.ops.uv.smart_project(
                    angle_limit=math.radians(66.0),
                    margin_method="FRACTION",
                    rotate_method="AXIS_ALIGNED_Y",
                    island_margin=local_island_margin,
                    area_weight=0.0,
                    correct_aspect=True,
                    scale_to_bounds=True,
                )
            finally:
                bpy.ops.object.mode_set(mode="OBJECT")
        projection_target = projection_mesh.uv_layers.get(_SEGMENT_SWATCH_UV)
        values = np.asarray(
            [tuple(item.uv) for item in projection_target.data],
            dtype=np.float64,
        ).copy()
    finally:
        if bpy.data.objects.get(projection_object.name) is projection_object:
            bpy.data.objects.remove(projection_object, do_unlink=True)
        if bpy.data.meshes.get(projection_mesh.name) is projection_mesh:
            bpy.data.meshes.remove(projection_mesh)
    if result != {"FINISHED"}:
        raise TexturedRenderError("Blender smart-project swatch UV did not finish")
    if values.shape != (len(mesh.loops), 2) or not np.isfinite(values).all():
        raise TexturedRenderError("segment swatch UV is invalid")
    values[:, 0] = (values[:, 0] + column) / columns
    values[:, 1] = (values[:, 1] + row) / rows
    if (
        np.any(values < -1.0e-6)
        or np.any(values > 1.0 + 1.0e-6)
        or np.ptp(values[:, 0]) <= 1.0e-6
        or np.ptp(values[:, 1]) <= 1.0e-6
    ):
        raise TexturedRenderError("segment swatch UV is empty or outside [0, 1]")
    if mesh.uv_layers.get(_SEGMENT_SWATCH_UV) is not None:
        raise TexturedRenderError("segment swatch target UV already exists")
    target = mesh.uv_layers.new(name=_SEGMENT_SWATCH_UV)
    if target is None or len(target.data) != len(values):
        raise TexturedRenderError("Blender could not copy the segment swatch UV")
    for destination, value in zip(target.data, values):
        destination.uv = (float(value[0]), float(value[1]))
    mesh.uv_layers.active_index = mesh.uv_layers.find(_SEGMENT_SWATCH_UV)
    exact_normals = proxy.source_normals
    if _array_hash(
        "swatch-evaluated-corner-normals", exact_normals
    ) != proxy.record.get("corner_normals_sha256"):
        raise TexturedRenderError(
            "exact corner-normal payload changed during swatch UV generation"
        )
    renderer_normals_before_uv = proxy.record.get(
        "renderer_custom_corner_normals_sha256"
    )
    restored_normals = _corner_normals(mesh)
    restored_normals_sha256 = _array_hash(
        "swatch-renderer-custom-corner-normals", restored_normals
    )
    if restored_normals_sha256 == renderer_normals_before_uv:
        normal_equivalence = proxy.record.get(
            "renderer_custom_normal_equivalence"
        )
    else:
        zero_indices = np.flatnonzero(
            np.linalg.norm(exact_normals.astype(np.float64), axis=1) <= 1.0e-12
        )
        if len(zero_indices):
            raise TexturedRenderError(
                "zero renderer-active corner normals changed during swatch UV generation"
            )
        normal_equivalence = _custom_normal_equivalence(
            exact_normals, restored_normals
        )
    proxy.record.update(
        {
            "target_uv_cell": {
                "column": column,
                "row": row,
                "columns": columns,
                "rows": rows,
            },
            "local_island_margin": local_island_margin,
            "final_atlas_island_margin_pixels": (
                local_island_margin * size / columns
            ),
            "target_uv_sha256": _array_hash("swatch-target-uv", values),
            "corner_normals_survived_target_uv_generation": True,
            "renderer_custom_corner_normals_before_target_uv_sha256": (
                renderer_normals_before_uv
            ),
            "renderer_custom_corner_normals_sha256": restored_normals_sha256,
            "renderer_custom_normal_equivalence": normal_equivalence,
            "smart_project_used_projection_only_mesh": True,
        }
    )


def _retry_zero_margin_smart_projection(
    bpy: Any,
    proxies: list[_BakeProxy],
    size: int,
) -> dict[str, Any]:
    grid_dimension = int(math.ceil(math.sqrt(len(proxies))))
    retry_margin = float(
        _SWATCH_RECIPE["target_uv"]["zero_mask_retry_island_margin"]
    )
    attempts: list[dict[str, Any]] = []
    for index, proxy in enumerate(proxies):
        mesh = proxy.object.data
        target = mesh.uv_layers.get(_SEGMENT_SWATCH_UV)
        if target is None or len(target.data) != len(mesh.loops):
            raise TexturedRenderError(
                "zero-mask smart-project retry lacks its initial target UV"
            )
        initial_hash = proxy.record.get("target_uv_sha256")
        initial_margin = proxy.record.get("local_island_margin")
        if not isinstance(initial_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", initial_hash
        ):
            raise TexturedRenderError(
                "zero-mask smart-project retry lacks initial UV provenance"
            )
        mesh.uv_layers.remove(target)
        _smart_project_proxy(
            bpy,
            proxy,
            column=index % grid_dimension,
            row=index // grid_dimension,
            columns=grid_dimension,
            rows=grid_dimension,
            size=size,
            local_island_margin=retry_margin,
        )
        attempts.append(
            {
                "source_occurrence_index": proxy.record.get(
                    "source_occurrence_index"
                ),
                "target_uv_cell": dict(proxy.record["target_uv_cell"]),
                "initial_local_island_margin": initial_margin,
                "retry_local_island_margin": retry_margin,
                "initial_target_uv_sha256": initial_hash,
                "retry_target_uv_sha256": proxy.record["target_uv_sha256"],
            }
        )
    return {
        "trigger": "initial smart-project mask contained zero selected-surface pixels",
        "grid_dimension": grid_dimension,
        "retry_local_island_margin": retry_margin,
        "retry_final_atlas_island_margin_pixels": 0.0,
        "proxies": attempts,
    }


def _reuse_shared_source_uv_as_swatch_target(
    proxies: list[_BakeProxy], plan: MaterialPlan, size: int
) -> dict[str, Any] | None:
    """Use an exact shared material UV when a fragmented smart atlas is subpixel."""

    if (
        len(proxies) != 1
        or not plan.textures
        or plan.display_color_attribute
        or plan.display_opacity_attribute
        or any(
            texture.uv.coordinate_source != "uv"
            or texture.uv.projection != "FLAT"
            for texture in plan.textures
        )
    ):
        return None
    uv_names = {texture.uv.name for texture in plan.textures}
    if len(uv_names) != 1:
        return None
    uv_name = next(iter(uv_names))
    alias = _uv_aliases(plan).get(uv_name)
    proxy = proxies[0]
    values = proxy.source_uvs.get(alias or "")
    if (
        values is None
        or values.shape != (len(proxy.object.data.loops), 2)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or np.ptp(values[:, 0]) <= 1.0e-9
        or np.ptp(values[:, 1]) <= 1.0e-9
    ):
        return None

    # A positive affine inset preserves the source UV tangent orientation and
    # keeps authored boundary coordinates inside the bake raster.
    inset = 1.0 / float(size)
    target_values = inset + values * (1.0 - 2.0 * inset)
    target = proxy.object.data.uv_layers.get(_SEGMENT_SWATCH_UV)
    if target is None or len(target.data) != len(target_values):
        raise TexturedRenderError("segment swatch fallback lost its target UV")
    for destination, value in zip(target.data, target_values):
        destination.uv = (float(value[0]), float(value[1]))
    if _array_hash(
        "swatch-evaluated-corner-normals",
        proxy.source_normals,
    ) != proxy.record.get("corner_normals_sha256"):
        raise TexturedRenderError(
            "exact corner normals changed during shared-UV swatch fallback"
        )
    if _array_hash(
        "swatch-renderer-custom-corner-normals",
        _corner_normals(proxy.object.data),
    ) != proxy.record.get("renderer_custom_corner_normals_sha256"):
        raise TexturedRenderError(
            "renderer normals changed during shared-UV swatch fallback"
        )
    proxy.record["target_uv_sha256"] = _array_hash(
        "swatch-target-uv", target_values
    )
    fallback = {
        "reason": "smart_project_atlas_had_no_rasterized_surface_pixels",
        "source_uv_name": uv_name,
        "source_uv_alias": alias,
        "source_uv_sha256": _array_hash("swatch-source-uv", values),
        "target_uv_sha256": proxy.record["target_uv_sha256"],
        "inset_pixels": 1.0,
    }
    proxy.record["target_uv_fallback"] = fallback
    return fallback


def _segment_renderable(
    renderables: list[_Renderable], segment_key: str
) -> _Renderable:
    matches = [item for item in renderables if segment_key in item.segment_keys]
    if len(matches) != 1:
        raise TexturedRenderError(
            f"segment {segment_key!r} belongs to {len(matches)} Blender renderables"
        )
    if not matches[0].objects:
        raise TexturedRenderError(
            f"segment {segment_key!r} has no render occurrence"
        )
    return matches[0]


def _remove_selected_proxies(bpy: Any, proxies: list[_BakeProxy]) -> None:
    for proxy in proxies:
        mesh = proxy.object.data
        if bpy.data.objects.get(proxy.object.name) is proxy.object:
            bpy.data.objects.remove(proxy.object, do_unlink=True)
        if bpy.data.meshes.get(mesh.name) is mesh:
            bpy.data.meshes.remove(mesh)


def _copy_exact_polygon_mesh(
    bpy: Any,
    source_mesh: Any,
    selected_material_index: int,
    aliases: Mapping[str, str],
) -> tuple[Any, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    polygon_indices = list(range(len(source_mesh.polygons)))
    loop_indices = np.arange(len(source_mesh.loops), dtype=np.int64)
    if not polygon_indices or any(
        int(polygon.material_index) != selected_material_index
        for polygon in source_mesh.polygons
    ):
        raise TexturedRenderError(
            "exact polygon copy does not contain only the selected segment"
        )

    vertices = np.asarray(
        [tuple(vertex.co) for vertex in source_mesh.vertices], dtype=np.float64
    )
    loop_vertices = np.asarray(
        [int(loop.vertex_index) for loop in source_mesh.loops], dtype=np.int64
    )
    source_normals = _corner_normals(source_mesh)
    source_uvs: dict[str, np.ndarray] = {}
    for source_name, alias in aliases.items():
        layer = source_mesh.uv_layers.get(alias)
        if layer is None or len(layer.data) != len(source_mesh.loops):
            raise TexturedRenderError(
                f"exact polygon segment lacks material UV {source_name!r}"
            )
        values = np.asarray(
            [tuple(item.uv) for item in layer.data], dtype=np.float64
        )
        if values.shape != (len(source_mesh.loops), 2) or not np.isfinite(
            values
        ).all():
            raise TexturedRenderError(
                f"exact polygon material UV {source_name!r} is invalid"
            )
        source_uvs[alias] = values
    display_color = _selected_corner_attribute(
        source_mesh,
        "vlm_displayColor",
        polygon_indices,
        loop_indices,
        color=True,
    )
    display_opacity = _selected_corner_attribute(
        source_mesh,
        "vlm_displayOpacity",
        polygon_indices,
        loop_indices,
        color=False,
    )

    mesh = source_mesh.copy()
    copied_vertices = np.asarray(
        [tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64
    )
    copied_loop_vertices = np.asarray(
        [int(loop.vertex_index) for loop in mesh.loops], dtype=np.int64
    )
    copied_normals = _corner_normals(mesh)
    if (
        not np.array_equal(copied_vertices, vertices)
        or not np.array_equal(copied_loop_vertices, loop_vertices)
        or not np.array_equal(copied_normals, source_normals)
        or len(mesh.polygons) != len(source_mesh.polygons)
    ):
        bpy.data.meshes.remove(mesh)
        raise TexturedRenderError(
            "Blender mesh datablock copy changed exact polygon geometry"
        )
    for alias, values in source_uvs.items():
        copied_layer = mesh.uv_layers.get(alias)
        copied_values = (
            np.asarray([tuple(item.uv) for item in copied_layer.data], dtype=np.float64)
            if copied_layer is not None
            else None
        )
        if copied_values is None or not np.array_equal(copied_values, values):
            bpy.data.meshes.remove(mesh)
            raise TexturedRenderError(
                f"Blender mesh datablock copy changed material UV alias {alias!r}"
            )
    copied_color = _selected_corner_attribute(
        mesh, "vlm_displayColor", polygon_indices, loop_indices, color=True
    )
    copied_opacity = _selected_corner_attribute(
        mesh, "vlm_displayOpacity", polygon_indices, loop_indices, color=False
    )
    if (
        (display_color is None) != (copied_color is None)
        or display_color is not None
        and not np.array_equal(display_color, copied_color)
        or (display_opacity is None) != (copied_opacity is None)
        or display_opacity is not None
        and not np.array_equal(display_opacity, copied_opacity)
    ):
        bpy.data.meshes.remove(mesh)
        raise TexturedRenderError(
            "Blender mesh datablock copy changed exact display attributes"
        )

    zero_indices = np.flatnonzero(
        np.linalg.norm(source_normals.astype(np.float64), axis=1) <= 1.0e-12
    ).astype(np.int64)
    # The datablock copy above already proved exact equality.  Do not mutate
    # or update it: either operation can invalidate Blender's evaluated
    # split-normal state even though no surface geometry changed.
    restored_normals = copied_normals
    normal_equivalence = {
        "source_and_copy_equal": True,
        "zero_corner_count": int(len(zero_indices)),
        "zero_corner_indices_sha256": (
            _array_hash("swatch-renderer-zero-normal-corners", zero_indices)
            if len(zero_indices)
            else None
        ),
    }
    record = {
        "selected_source_polygon_indices_sha256": _array_hash(
            "swatch-selected-source-polygons",
            np.asarray(polygon_indices, dtype=np.int64),
        ),
        "selected_source_loop_indices_sha256": _array_hash(
            "swatch-selected-source-loops", loop_indices
        ),
        "vertices_sha256": _array_hash("swatch-evaluated-vertices", vertices),
        "loop_vertices_sha256": _array_hash(
            "swatch-evaluated-loop-vertices", loop_vertices
        ),
        "corner_normals_sha256": _array_hash(
            "swatch-evaluated-corner-normals", source_normals
        ),
        "renderer_custom_corner_normals_sha256": _array_hash(
            "swatch-renderer-custom-corner-normals", restored_normals
        ),
        "renderer_custom_normal_equivalence": normal_equivalence,
        "source_uv_sha256": {
            alias: _array_hash("swatch-source-uv", values)
            for alias, values in sorted(source_uvs.items())
        },
        "display_color_sha256": (
            _array_hash("swatch-display-color", display_color)
            if display_color is not None
            else None
        ),
        "display_opacity_sha256": (
            _array_hash("swatch-display-opacity", display_opacity)
            if display_opacity is not None
            else None
        ),
        "polygon_count": len(mesh.polygons),
        "loop_count": len(mesh.loops),
        "vertex_count": len(mesh.vertices),
        "exact_corner_payload_and_display_pre_post_equal": True,
    }
    return mesh, source_uvs, source_normals, record


def _selected_evaluated_proxies(
    bpy: Any,
    collection: Any,
    renderables: list[_Renderable],
    segment_key: str,
    plan: MaterialPlan,
    occurrence_indices: list[int] | None,
    name_prefix: str,
) -> tuple[_Renderable, list[_BakeProxy], list[np.ndarray]]:
    renderable = _segment_renderable(renderables, segment_key)
    if occurrence_indices is None:
        occurrence_indices = list(range(len(renderable.objects)))
    invalid_indices = [
        index
        for index in occurrence_indices
        if index < 0 or index >= len(renderable.objects)
    ]
    if not occurrence_indices or invalid_indices:
        raise TexturedRenderError(
            f"segment {segment_key!r} has invalid occurrence selection "
            f"{invalid_indices!r}"
        )
    renderable.isolated_state(segment_key)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    aliases = _uv_aliases(plan)
    occurrence_matrices = [
        np.asarray(
            [
                [float(component) for component in row]
                for row in object_value.matrix_world
            ],
            dtype=np.float64,
        )
        for object_value in renderable.objects
    ]
    proxies: list[_BakeProxy] = []
    for occurrence_index in occurrence_indices:
        source_object = renderable.objects[occurrence_index]
        selected_material_index = renderable.segment_material_indices[segment_key]
        already_static = (
            source_object.data is renderable.mesh
            and not source_object.modifiers
        )
        exact_polygon_copy = (
            already_static
            and renderable.segment_keys == frozenset({segment_key})
            and all(owner == segment_key for owner in renderable.face_segments)
        )
        if exact_polygon_copy:
            mesh, source_uvs, source_normals, mesh_record = _copy_exact_polygon_mesh(
                bpy,
                renderable.mesh,
                selected_material_index,
                aliases,
            )
        elif already_static:
            mesh, source_uvs, source_normals, mesh_record = _isolated_evaluated_mesh(
                bpy,
                renderable.mesh,
                selected_material_index,
                aliases,
            )
        else:
            evaluated = source_object.evaluated_get(depsgraph)
            evaluated_mesh = bpy.data.meshes.new_from_object(
                evaluated,
                preserve_all_data_layers=True,
                depsgraph=depsgraph,
            )
            if evaluated_mesh is None or not evaluated_mesh.polygons:
                raise TexturedRenderError(
                    f"evaluated segment {segment_key!r} has no renderable polygons"
                )
            try:
                exact_evaluated_copy = (
                    renderable.segment_keys == frozenset({segment_key})
                    and all(
                        owner == segment_key for owner in renderable.face_segments
                    )
                    and source_object.data is renderable.mesh
                    and all(
                        int(polygon.material_index) == selected_material_index
                        for polygon in evaluated_mesh.polygons
                    )
                    and bool(
                        np.any(
                            np.linalg.norm(
                                _corner_normals(evaluated_mesh).astype(np.float64),
                                axis=1,
                            )
                            <= 1.0e-12
                        )
                    )
                )
                if exact_evaluated_copy:
                    mesh, source_uvs, source_normals, mesh_record = _copy_exact_polygon_mesh(
                        bpy,
                        evaluated_mesh,
                        selected_material_index,
                        aliases,
                    )
                else:
                    mesh, source_uvs, source_normals, mesh_record = _isolated_evaluated_mesh(
                        bpy,
                        evaluated_mesh,
                        selected_material_index,
                        aliases,
                    )
            finally:
                bpy.data.meshes.remove(evaluated_mesh)
        proxy_object = bpy.data.objects.new(
            f"{name_prefix}_{_safe_component(segment_key)}_{occurrence_index}",
            mesh,
        )
        proxy_object.matrix_world = source_object.matrix_world.copy()
        collection.objects.link(proxy_object)
        proxies.append(
            _BakeProxy(
                object=proxy_object,
                source_uvs=source_uvs,
                source_normals=source_normals,
                record={
                    **mesh_record,
                    "source_occurrence_index": occurrence_index,
                    "matrix_world_sha256": _array_hash(
                        "selected-occurrence-matrix",
                        occurrence_matrices[occurrence_index],
                    ),
                },
            )
        )
    return renderable, proxies, occurrence_matrices


def _selected_render_proxies(
    bpy: Any,
    collection: Any,
    renderables: list[_Renderable],
    segment_key: str,
    plan: MaterialPlan,
    material: Any,
) -> tuple[list[_BakeProxy], dict[str, Any]]:
    renderable, proxies, occurrence_matrices = _selected_evaluated_proxies(
        bpy,
        collection,
        renderables,
        segment_key,
        plan,
        None,
        "vlm_physically_isolated_selected_render",
    )
    for proxy in proxies:
        mesh = proxy.object.data
        mesh.materials.clear()
        mesh.materials.append(material)
        for polygon in mesh.polygons:
            polygon.material_index = 0
        if len(mesh.materials) != 1 or any(
            int(polygon.material_index) != 0 for polygon in mesh.polygons
        ):
            raise TexturedRenderError(
                "physically isolated selected mesh has an invalid material binding"
            )
        proxy.object.hide_render = False
    for item in renderables:
        for object_value in item.objects:
            object_value.hide_render = True
    bpy.context.view_layer.update()
    expected = {int(proxy.object.as_pointer()) for proxy in proxies}
    visible_meshes = {
        int(object_value.as_pointer())
        for object_value in bpy.context.view_layer.objects
        if object_value.type == "MESH" and not object_value.hide_render
    }
    if visible_meshes != expected:
        raise TexturedRenderError(
            "selected render contains mesh geometry outside the selected segment"
        )
    record = {
        "source_occurrence_count": len(renderable.objects),
        "rendered_occurrence_count": len(proxies),
        "nonselected_polygon_count": 0,
        "original_asset_objects_hidden_from_render": True,
        "all_occurrence_matrices_sha256": _array_hash(
            "selected-render-all-occurrence-matrices",
            np.asarray(occurrence_matrices, dtype=np.float64),
        ),
        "proxies": [proxy.record for proxy in proxies],
    }
    record["physical_isolation_sha256"] = _canonical_hash(
        "selected-render-physical-isolation", record
    )
    return proxies, record


def _selected_bake_proxies(
    bpy: Any,
    collection: Any,
    renderables: list[_Renderable],
    segment_key: str,
    plan: MaterialPlan,
    size: int,
) -> tuple[list[_BakeProxy], dict[str, Any]]:
    renderable = _segment_renderable(renderables, segment_key)
    world_roles = sorted(
        texture.role
        for texture in plan.textures
        if texture.uv.coordinate_source == "world"
    )
    occurrence_indices = (
        list(range(len(renderable.objects))) if world_roles else [0]
    )
    _, proxies, occurrence_matrices = _selected_evaluated_proxies(
        bpy,
        collection,
        renderables,
        segment_key,
        plan,
        occurrence_indices,
        "vlm_segment_swatch_source",
    )
    for item in renderables:
        for object_value in item.objects:
            object_value.hide_render = True
    for proxy in proxies:
        proxy.object.hide_render = False
    grid_dimension = int(math.ceil(math.sqrt(len(proxies))))
    island_margin_pixels = float(
        _SWATCH_RECIPE["target_uv"]["island_margin_pixels"]
    )
    local_island_margin = island_margin_pixels * grid_dimension / size
    for index, proxy in enumerate(proxies):
        _smart_project_proxy(
            bpy,
            proxy,
            column=index % grid_dimension,
            row=index // grid_dimension,
            columns=grid_dimension,
            rows=grid_dimension,
            size=size,
            local_island_margin=local_island_margin,
        )
    record = {
        "source_occurrence_count": len(renderable.objects),
        "baked_occurrence_count": len(proxies),
        "world_coordinate_roles": world_roles,
        "occurrence_selection": (
            "all occurrences tiled because world-coordinate appearance varies"
            if world_roles
            else "occurrence 0; UV/object/display inputs are invariant to occurrence matrix"
        ),
        "occurrence_invariance_attestation": {
            "world_coordinates_present": bool(world_roles),
        },
        "all_occurrence_matrices_sha256": _array_hash(
            "swatch-all-occurrence-matrices",
            np.asarray(occurrence_matrices, dtype=np.float64),
        ),
        "target_uv_generation": {
            **_SWATCH_RECIPE["target_uv"],
            "grid_dimension": grid_dimension,
            "local_island_margin": local_island_margin,
        },
        "proxies": [proxy.record for proxy in proxies],
    }
    record["evaluated_sampling_sha256"] = _canonical_hash(
        "selected-evaluated-segment-swatch", record
    )
    return proxies, record


def _set_bake_source_uv(
    mesh: Any,
    alias: str | None,
    values: np.ndarray | None,
    expected_normals_sha256: str,
) -> None:
    for layer in list(mesh.uv_layers):
        if layer.name != _SEGMENT_SWATCH_UV:
            mesh.uv_layers.remove(layer)
    if alias is not None:
        if values is None or values.shape != (len(mesh.loops), 2):
            raise TexturedRenderError(f"segment swatch source UV {alias!r} is invalid")
        layer = mesh.uv_layers.new(name=alias)
        if layer is None or len(layer.data) != len(values):
            raise TexturedRenderError(
                f"Blender could not restore segment swatch source UV {alias!r}"
            )
        for destination, value in zip(layer.data, values):
            destination.uv = (float(value[0]), float(value[1]))
    target_index = mesh.uv_layers.find(_SEGMENT_SWATCH_UV)
    if target_index < 0:
        raise TexturedRenderError("segment swatch target UV disappeared")
    mesh.uv_layers.active_index = target_index
    if _array_hash(
        "swatch-renderer-custom-corner-normals", _corner_normals(mesh)
    ) != expected_normals_sha256:
        raise TexturedRenderError(
            "renderer normals changed while selecting a swatch source UV"
        )


def _bake_material(
    bpy: Any,
    texture: TexturePlan | None,
    kind: str,
    image: Any,
    image_cache: dict[tuple[str, str, str], Any],
    uv_alias: str | None,
    attribute_name: str | None = None,
) -> Any:
    material = bpy.data.materials.new(
        f"vlm_segment_swatch_bake_{kind}_{uuid.uuid4().hex[:12]}"
    )
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    target = nodes.new("ShaderNodeTexImage")
    target.name = "SegmentSwatchBakeTarget"
    target.image = image
    target.select = True
    nodes.active = target
    if texture is None and attribute_name is None:
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 1.0
        links.new(emission.outputs[0], output.inputs["Surface"])
        nodes.active = target
        return material

    if texture is None:
        emission = nodes.new("ShaderNodeEmission")
        if kind == "color":
            attribute = nodes.new("ShaderNodeVertexColor")
            attribute.layer_name = attribute_name
            links.new(attribute.outputs["Color"], emission.inputs["Color"])
        elif kind == "scalar":
            attribute = nodes.new("ShaderNodeAttribute")
            attribute.attribute_name = attribute_name
            combine = nodes.new("ShaderNodeCombineColor")
            combine.mode = "RGB"
            for name in ("Red", "Green", "Blue"):
                links.new(attribute.outputs["Fac"], combine.inputs[name])
            links.new(combine.outputs["Color"], emission.inputs["Color"])
        else:
            raise TexturedRenderError(
                f"unsupported display swatch kind {kind!r}"
            )
        emission.inputs["Strength"].default_value = 1.0
        links.new(emission.outputs[0], output.inputs["Surface"])
        nodes.active = target
        return material

    color, scalar = _texture_sockets(
        bpy,
        nodes,
        links,
        texture,
        image_cache,
        uv_alias,
        False,
    )
    if kind == "tangent_normal":
        normal = _normal_world_from_texture(
            nodes,
            links,
            color,
            texture,
            1.0,
            None,
            {texture.uv.name: uv_alias} if uv_alias is not None else {},
            standardized_sphere=False,
        )
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        links.new(normal, diffuse.inputs["Normal"])
        links.new(diffuse.outputs[0], output.inputs["Surface"])
        nodes.active = target
        return material

    emission = nodes.new("ShaderNodeEmission")
    if kind == "color":
        links.new(color, emission.inputs["Color"])
    elif kind == "scalar":
        combine = nodes.new("ShaderNodeCombineColor")
        combine.mode = "RGB"
        for name in ("Red", "Green", "Blue"):
            links.new(scalar, combine.inputs[name])
        links.new(combine.outputs["Color"], emission.inputs["Color"])
    else:
        raise TexturedRenderError(f"unsupported segment swatch kind {kind!r}")
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs[0], output.inputs["Surface"])
    nodes.active = target
    return material


def _cycles_bake(
    bpy: Any,
    scene: Any,
    bake_objects: list[Any],
    material: Any,
    kind: str,
) -> None:
    if not bake_objects:
        raise TexturedRenderError("segment swatch bake has no selected objects")
    for bake_object in bake_objects:
        bake_object.data.materials.clear()
        bake_object.data.materials.append(material)
        for polygon in bake_object.data.polygons:
            polygon.material_index = 0
        target_index = bake_object.data.uv_layers.find(_SEGMENT_SWATCH_UV)
        if target_index < 0:
            raise TexturedRenderError("segment swatch bake lost its target UV")
        bake_object.data.uv_layers.active_index = target_index
    for item in bpy.context.view_layer.objects:
        item.select_set(False)
    for bake_object in bake_objects:
        bake_object.hide_set(False)
        bake_object.hide_viewport = False
        bake_object.hide_render = False
        bake_object.select_set(True)
    active_object = bake_objects[0]
    bpy.context.view_layer.objects.active = active_object
    bake_type = "NORMAL" if kind == "tangent_normal" else "EMIT"
    arguments: dict[str, Any] = {
        "type": bake_type,
        "margin": 0,
        "use_clear": True,
        "target": "IMAGE_TEXTURES",
        "save_mode": "INTERNAL",
        "uv_layer": _SEGMENT_SWATCH_UV,
    }
    if bake_type == "NORMAL":
        arguments.update(
            {
                "normal_space": "TANGENT",
                "normal_r": "POS_X",
                "normal_g": "POS_Y",
                "normal_b": "POS_Z",
            }
        )
    previous_samples = int(scene.cycles.samples)
    previous_adaptive = bool(scene.cycles.use_adaptive_sampling)
    previous_denoising = bool(scene.cycles.use_denoising)
    image_settings = scene.render.image_settings
    previous_png_settings = (
        str(image_settings.file_format),
        str(image_settings.color_mode),
        str(image_settings.color_depth),
        int(image_settings.compression),
    )
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    try:
        with bpy.context.temp_override(
            scene=scene,
            active_object=active_object,
            object=active_object,
            selected_objects=bake_objects,
            selected_editable_objects=bake_objects,
        ):
            result = bpy.ops.object.bake(**arguments)
    finally:
        scene.cycles.samples = previous_samples
        scene.cycles.use_adaptive_sampling = previous_adaptive
        scene.cycles.use_denoising = previous_denoising
        (
            image_settings.file_format,
            image_settings.color_mode,
            image_settings.color_depth,
            image_settings.compression,
        ) = previous_png_settings
    if (
        int(scene.cycles.samples) != previous_samples
        or bool(scene.cycles.use_adaptive_sampling) is not previous_adaptive
        or bool(scene.cycles.use_denoising) is not previous_denoising
        or (
            str(image_settings.file_format),
            str(image_settings.color_mode),
            str(image_settings.color_depth),
            int(image_settings.compression),
        )
        != previous_png_settings
    ):
        raise TexturedRenderError("Cycles PNG settings changed after swatch bake")
    if result != {"FINISHED"}:
        raise TexturedRenderError(
            f"Cycles {bake_type} segment swatch bake did not finish: {result!r}"
        )


def _fill_segment_swatch(
    image: Any,
    valid: np.ndarray,
    nearest: np.ndarray,
    *,
    kind: str,
) -> str:
    pixels = _image_pixels(image)
    if valid.shape != pixels.shape[:2] or nearest.shape != (2,) + valid.shape:
        raise TexturedRenderError("segment swatch fill plan has invalid dimensions")
    rgb = np.asarray(pixels[:, :, :3], dtype=np.float32)
    filled = rgb.copy()
    invalid = ~valid
    filled[invalid] = rgb[nearest[0][invalid], nearest[1][invalid]]
    if not np.isfinite(filled).all():
        raise TexturedRenderError("nearest-filled segment swatch is non-finite")
    if kind == "tangent_normal" and (
        np.any(filled < -1.0e-5) or np.any(filled > 1.0 + 1.0e-5)
    ):
        raise TexturedRenderError("tangent-normal segment swatch is outside [0, 1]")
    rgba = np.empty(pixels.shape, dtype=np.float32)
    rgba[:, :, :3] = filled
    rgba[:, :, 3] = 1.0
    image.pixels.foreach_set(rgba.ravel())
    image.update()
    return _array_hash("nearest-filled-segment-swatch", rgba)


def _bake_segment_swatches(
    bpy: Any,
    scene: Any,
    collection: Any,
    renderables: list[_Renderable],
    segment_key: str,
    plan: MaterialPlan,
    image_cache: dict[tuple[str, str, str], Any],
    size: int,
    geometry_payload_sha256: str,
) -> tuple[dict[str, _BakedSwatch], dict[str, Any]]:
    expected_inputs: list[
        tuple[str, str, TexturePlan | None, str | None]
    ] = [
        (texture.role, _swatch_kind(plan, texture), texture, None)
        for texture in plan.textures
    ]
    expected_roles = _expected_swatch_roles(plan)
    if _DISPLAY_COLOR_SWATCH in expected_roles:
        expected_inputs.append(
            (
                _DISPLAY_COLOR_SWATCH,
                "color",
                None,
                plan.display_color_attribute,
            )
        )
    if _DISPLAY_OPACITY_SWATCH in expected_roles:
        expected_inputs.append(
            (
                _DISPLAY_OPACITY_SWATCH,
                "scalar",
                None,
                plan.display_opacity_attribute,
            )
        )
    if not expected_inputs:
        record = {
            "geometry_payload_sha256": geometry_payload_sha256,
            "swatch_recipe": dict(_SWATCH_RECIPE),
            "swatch_recipe_sha256": _SWATCH_RECIPE_SHA256,
            "swatches": {},
        }
        record["segment_sampling_plan_sha256"] = _canonical_hash(
            "selected-segment-swatch-plan", record
        )
        return {}, record
    proxies, geometry_record = _selected_bake_proxies(
        bpy, collection, renderables, segment_key, plan, size
    )
    bake_objects = [proxy.object for proxy in proxies]
    aliases = _uv_aliases(plan)
    bake_materials: list[Any] = []
    transient_images: list[Any] = []
    swatch_images: list[Any] = []
    swatches: dict[str, _BakedSwatch] = {}
    succeeded = False
    try:
        for proxy in proxies:
            _set_bake_source_uv(
                proxy.object.data,
                None,
                None,
                proxy.record["renderer_custom_corner_normals_sha256"],
            )
        mask_image = _new_bake_image(
            bpy, f"vlm_segment_swatch_mask_{uuid.uuid4().hex[:12]}", size
        )
        transient_images.append(mask_image)
        mask_material = _bake_material(
            bpy, None, "mask", mask_image, image_cache, None
        )
        bake_materials.append(mask_material)
        _cycles_bake(bpy, scene, bake_objects, mask_material, "mask")
        mask_pixels = _image_pixels(mask_image)
        valid = mask_pixels[:, :, 0] > 0.5
        valid_count = int(np.count_nonzero(valid))
        if valid_count < 1:
            initial_target_uv_generation = dict(
                geometry_record["target_uv_generation"]
            )
            zero_margin_retry = _retry_zero_margin_smart_projection(
                bpy, proxies, size
            )
            geometry_record["target_uv_generation"] = {
                "initial_attempt": initial_target_uv_generation,
                "zero_mask_retry": zero_margin_retry,
            }
            geometry_record["proxies"] = [proxy.record for proxy in proxies]
            geometry_record.pop("evaluated_sampling_sha256", None)
            geometry_record["evaluated_sampling_sha256"] = _canonical_hash(
                "selected-evaluated-segment-swatch", geometry_record
            )
            _cycles_bake(bpy, scene, bake_objects, mask_material, "mask")
            mask_pixels = _image_pixels(mask_image)
            valid = mask_pixels[:, :, 0] > 0.5
            valid_count = int(np.count_nonzero(valid))
            if valid_count < 1:
                preceding_smart_projection = dict(
                    geometry_record["target_uv_generation"]
                )
                fallback = _reuse_shared_source_uv_as_swatch_target(
                    proxies, plan, size
                )
                if fallback is None:
                    raise TexturedRenderError(
                        "zero-margin segment swatch target contains no "
                        "selected-surface pixels"
                    )
                geometry_record["target_uv_generation"] = {
                    "preceding_smart_projection": preceding_smart_projection,
                    **fallback,
                }
                geometry_record["proxies"] = [
                    proxy.record for proxy in proxies
                ]
                geometry_record.pop("evaluated_sampling_sha256", None)
                geometry_record["evaluated_sampling_sha256"] = _canonical_hash(
                    "selected-evaluated-segment-swatch", geometry_record
                )
                _cycles_bake(
                    bpy, scene, bake_objects, mask_material, "mask"
                )
                mask_pixels = _image_pixels(mask_image)
                valid = mask_pixels[:, :, 0] > 0.5
                valid_count = int(np.count_nonzero(valid))
                if valid_count < 1:
                    raise TexturedRenderError(
                        "exact shared-UV swatch fallback contains no "
                        "selected-surface pixels"
                    )
        valid_fraction = valid_count / float(size * size)
        nearest = distance_transform_edt(
            ~valid,
            return_distances=False,
            return_indices=True,
        ).astype(np.int32, copy=False)
        if nearest.shape != (2, size, size):
            raise TexturedRenderError(
                "nearest segment-swatch fill returned invalid indices"
            )

        role_records: dict[str, Any] = {}
        for role, kind, texture, attribute_name in expected_inputs:
            source_alias = (
                aliases[texture.uv.name]
                if texture is not None
                and texture.uv.coordinate_source == "uv"
                else None
            )
            for proxy in proxies:
                source_values = (
                    proxy.source_uvs.get(source_alias) if source_alias else None
                )
                _set_bake_source_uv(
                    proxy.object.data,
                    source_alias,
                    source_values,
                    proxy.record["renderer_custom_corner_normals_sha256"],
                )
            image = _new_bake_image(
                bpy,
                f"vlm_segment_swatch_{role}_{uuid.uuid4().hex[:12]}",
                size,
            )
            swatch_images.append(image)
            material = _bake_material(
                bpy,
                texture,
                kind,
                image,
                image_cache,
                source_alias,
                attribute_name,
            )
            bake_materials.append(material)
            _cycles_bake(bpy, scene, bake_objects, material, kind)
            pixels_sha256 = _fill_segment_swatch(
                image, valid, nearest, kind=kind
            )
            if texture is not None:
                source_evidence_sha256 = _canonical_hash(
                    "segment-swatch-source-texture-plan", texture.record()
                )
            else:
                display_hashes = [
                    proxy.record[
                        "display_color_sha256"
                        if role == _DISPLAY_COLOR_SWATCH
                        else "display_opacity_sha256"
                    ]
                    for proxy in proxies
                ]
                if any(value is None for value in display_hashes):
                    raise TexturedRenderError(
                        f"selected geometry lacks required {attribute_name} values"
                    )
                source_evidence_sha256 = _canonical_hash(
                    "segment-swatch-source-display-attribute",
                    display_hashes,
                )
            swatch = _BakedSwatch(
                role=role,
                kind=kind,
                image=image,
                width=size,
                height=size,
                source_evidence_sha256=source_evidence_sha256,
                pixels_sha256=pixels_sha256,
            )
            swatches[role] = swatch
            role_records[role] = {
                **swatch.record(),
                "bake_type": (
                    "NORMAL" if kind == "tangent_normal" else "EMIT"
                ),
                "source_uv_alias": source_alias,
                "source_attribute": attribute_name,
            }
        if set(swatches) != {item[0] for item in expected_inputs}:
            raise TexturedRenderError(
                "segment swatch roles do not cover every appearance input"
            )
        _attest_proxy_normal_state(proxies, "segment-swatch baking")
        record = {
            "geometry_payload_sha256": geometry_payload_sha256,
            "swatch_recipe": dict(_SWATCH_RECIPE),
            "swatch_recipe_sha256": _SWATCH_RECIPE_SHA256,
            "linear_float_working_images": True,
            "cycles_bake_samples": 1,
            "png_render_settings_restored_after_each_bake": True,
            "target_uv": _SEGMENT_SWATCH_UV,
            "geometry": geometry_record,
            "mask": {
                "valid_pixel_count": valid_count,
                "valid_pixel_fraction": valid_fraction,
                "total_pixel_count": int(size * size),
                "valid_mask_sha256": _array_hash(
                    "segment-swatch-valid-mask", valid.astype(np.uint8)
                ),
                "nearest_selected_pixel_indices_sha256": _array_hash(
                    "segment-swatch-nearest-indices", nearest
                ),
            },
            "swatches": dict(sorted(role_records.items())),
        }
        record["segment_sampling_plan_sha256"] = _canonical_hash(
            "selected-segment-swatch-plan", record
        )
        succeeded = True
        return swatches, record
    finally:
        _remove_selected_proxies(bpy, proxies)
        for material in bake_materials:
            if material.name in bpy.data.materials:
                bpy.data.materials.remove(material)
        for image in transient_images:
            if image.name in bpy.data.images:
                bpy.data.images.remove(image)
        if not succeeded:
            for image in swatch_images:
                if image.name in bpy.data.images:
                    bpy.data.images.remove(image)


def _sphere_swatch_node_attestation(
    material: Any,
    swatches: Mapping[str, _BakedSwatch],
) -> dict[str, Any]:
    outputs = [
        node
        for node in material.node_tree.nodes
        if node.type == "OUTPUT_MATERIAL" and bool(node.is_active_output)
    ]
    if len(outputs) != 1:
        raise TexturedRenderError(
            "standardized sphere material has no unique active output"
        )
    reachable: dict[int, Any] = {}
    pending = list(outputs)
    while pending:
        node = pending.pop()
        pointer = int(node.as_pointer())
        if pointer in reachable:
            continue
        reachable[pointer] = node
        for input_socket in node.inputs:
            pending.extend(link.from_node for link in input_socket.links)
    image_nodes = [node for node in reachable.values() if node.type == "TEX_IMAGE"]
    expected_images = {role: swatch.image for role, swatch in swatches.items()}
    roles_by_identity = {
        int(image.as_pointer()): role for role, image in expected_images.items()
    }
    observed: dict[str, str] = {}
    for node in image_nodes:
        if (
            node.image is None
            or int(node.image.as_pointer()) not in roles_by_identity
        ):
            raise TexturedRenderError(
                "standardized sphere node tree references a non-swatch image"
            )
        role = roles_by_identity[int(node.image.as_pointer())]
        if role in observed:
            raise TexturedRenderError(
                f"standardized sphere references swatch {role!r} more than once"
            )
        if str(node.image.filepath or ""):
            raise TexturedRenderError(
                "derived segment swatch unexpectedly references a source image path"
            )
        observed[role] = node.name
    if set(observed) != set(expected_images):
        raise TexturedRenderError(
            "standardized sphere node tree does not use every derived swatch exactly once"
        )
    record = {
        "reachable_node_count": len(reachable),
        "reachable_image_node_count": len(image_nodes),
        "role_to_node": dict(sorted(observed.items())),
        "role_to_pixels_sha256": {
            role: swatches[role].pixels_sha256 for role in sorted(swatches)
        },
    }
    record["node_tree_attestation_sha256"] = _canonical_hash(
        "segment-swatch-sphere-node-tree", record
    )
    return record


def _configure_cycles_gpu(bpy: Any, scene: Any) -> dict[str, Any]:
    if tuple(bpy.app.version[:2]) != _BLENDER_VERSION:
        raise TexturedRenderError(
            "textured rendering requires the pinned Blender Python 4.2 runtime; "
            f"found {tuple(bpy.app.version)}"
        )
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise TexturedRenderError("Blender Cycles add-on is unavailable")
    preferences = addon.preferences
    selected_backend = None
    selected_device = None
    backend_errors: list[str] = []
    for backend in _GPU_BACKENDS:
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
        except Exception as exc:
            backend_errors.append(f"{backend}: {type(exc).__name__}: {exc}")
            continue
        devices = list(preferences.devices)
        candidates = sorted(
            (
                device
                for device in devices
                if str(getattr(device, "type", "")).upper() == backend
                and "cpu" not in str(getattr(device, "name", "")).casefold()
            ),
            key=lambda device: (
                str(getattr(device, "name", "")),
                str(getattr(device, "id", "")),
            ),
        )
        if not candidates:
            continue
        selected_backend = backend
        selected_device = candidates[0]
        for device in devices:
            device.use = device is selected_device
        break
    if selected_backend is None or selected_device is None:
        detail = "; ".join(backend_errors) if backend_errors else "no GPU devices enumerated"
        raise TexturedRenderError(f"Cycles has no usable non-CPU device: {detail}")
    enabled = [device for device in preferences.devices if bool(device.use)]
    if enabled != [selected_device] or any(
        str(getattr(device, "type", "")).upper() == "CPU" for device in enabled
    ):
        raise TexturedRenderError("Cycles did not retain the GPU-only device selection")

    scene.render.engine = "CYCLES"
    if scene.render.engine != "CYCLES":
        raise TexturedRenderError("Blender did not retain the CYCLES engine")
    scene.cycles.device = "GPU"
    scene.cycles.samples = 256
    scene.cycles.preview_samples = 32
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.denoising_prefilter = "ACCURATE"
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.render.use_persistent_data = False
    device_name = str(getattr(selected_device, "name", "")).strip()
    device_type = str(getattr(selected_device, "type", "")).strip().upper()
    if not device_name or not device_type or "cpu" in (device_name + device_type).casefold():
        raise TexturedRenderError("Cycles selected device has no valid non-CPU identity")
    return {
        "engine": "CYCLES",
        "render_backend": selected_backend,
        "render_device": [f"{device_type}: {device_name}"],
        "blender_version": list(bpy.app.version),
        "cycles_samples": 256,
        "adaptive_sampling": False,
        "denoising": True,
        "denoiser": "OPENIMAGEDENOISE",
        "denoising_input_passes": "RGB_ALBEDO_NORMAL",
        "denoising_prefilter": "ACCURATE",
        "seed": 0,
        "animated_seed": False,
    }


def _new_scene(bpy: Any, size: int) -> tuple[Any, Any]:
    if type(size) is not int or size != 512:
        raise TexturedRenderError("production textured VLM renders must be exactly 512x512")
    scene = bpy.data.scenes.new(f"vlm_textured_{uuid.uuid4().hex}")
    collection = bpy.data.collections.new(f"vlm_textured_objects_{uuid.uuid4().hex}")
    scene.collection.children.link(collection)
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = bpy.data.worlds.new(f"vlm_textured_world_{uuid.uuid4().hex}")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
    background.inputs["Strength"].default_value = 0.35
    scene.world = world
    return scene, collection


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (np.asarray(points, dtype=np.float64), np.ones((len(points), 1))), axis=1
    )
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64)
    weights = transformed[:, 3:4]
    if np.any(np.abs(weights) < 1e-14):
        raise TexturedRenderError("occurrence transform maps geometry to w=0")
    result = transformed[:, :3] / weights
    if not np.isfinite(result).all():
        raise TexturedRenderError("occurrence transform produces non-finite geometry")
    return result


def _visibility_exemplar(
    payload: SegmentGeometryPayload,
) -> tuple[SegmentGeometryPayload, dict[str, Any]] | None:
    counts = np.asarray(payload.face_vertex_counts, dtype=np.int64)
    indices = np.asarray(payload.face_vertex_indices, dtype=np.int64)
    if len(counts) == 0 or len(payload.points) == 0:
        return None
    face_corners: list[np.ndarray] = []
    offset = 0
    parent = np.arange(len(payload.points), dtype=np.int64)

    def root(value: int) -> int:
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for count_value in counts:
        count = int(count_value)
        corners = indices[offset : offset + count]
        offset += count
        if count < 3:
            return None
        face_corners.append(corners)
        first = int(corners[0])
        for vertex in corners[1:]:
            union(first, int(vertex))
    if offset != len(indices):
        return None

    components: dict[int, list[int]] = {}
    for face_index, corners in enumerate(face_corners):
        components.setdefault(root(int(corners[0])), []).append(face_index)
    occurrence_count = len(payload.occurrence_transforms)
    if len(components) == 1 and occurrence_count == 1:
        return None

    def surface_area(points: np.ndarray, face_indices: list[int]) -> float:
        total = 0.0
        for face_index in face_indices:
            polygon = points[face_corners[face_index]]
            origin = polygon[0]
            triangles = np.cross(
                polygon[1:-1] - origin,
                polygon[2:] - origin,
            )
            total += 0.5 * float(np.linalg.norm(triangles, axis=1).sum())
        return total

    candidates = []
    total_area = 0.0
    ordered_components = sorted(components.values(), key=lambda value: value[0])
    for occurrence_index, matrix in enumerate(payload.occurrence_transforms):
        points = _transform_points(payload.points, matrix)
        for component_index, face_indices in enumerate(ordered_components):
            area = surface_area(points, face_indices)
            total_area += area
            candidates.append(
                (area, -component_index, -occurrence_index, face_indices)
            )
    area, negative_component, negative_occurrence, selected_faces = max(candidates)
    if not math.isfinite(area) or area <= 1e-18:
        return None
    component_index = -negative_component
    occurrence_index = -negative_occurrence
    used_vertices = np.unique(
        np.concatenate([face_corners[index] for index in selected_faces])
    )
    local_index = np.full(len(payload.points), -1, dtype=np.int64)
    local_index[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    selected_counts = counts[selected_faces]
    selected_indices = np.concatenate(
        [local_index[face_corners[index]] for index in selected_faces]
    )
    corner_offsets = np.concatenate(([0], np.cumsum(counts)))
    selected_source_corners = np.concatenate(
        [
            payload.source_corner_indices[
                corner_offsets[index] : corner_offsets[index + 1]
            ]
            for index in selected_faces
        ]
    )
    exemplar = SegmentGeometryPayload(
        points=np.asarray(payload.points[used_vertices], dtype=np.float64),
        source_vertex_indices=np.asarray(
            payload.source_vertex_indices[used_vertices], dtype=np.int64
        ),
        face_vertex_counts=np.asarray(selected_counts, dtype=np.int64),
        face_vertex_indices=np.asarray(selected_indices, dtype=np.int64),
        source_face_indices=np.asarray(
            payload.source_face_indices[selected_faces], dtype=np.int64
        ),
        source_corner_indices=np.asarray(selected_source_corners, dtype=np.int64),
        occurrence_transforms=np.asarray(
            payload.occurrence_transforms[occurrence_index : occurrence_index + 1],
            dtype=np.float64,
        ),
        provenance=dict(payload.provenance),
    )
    return exemplar, {
        "visibility_aid": "largest_exact_connected_component_exemplar",
        "full_segment_remains_unmodified": True,
        "component_index": component_index,
        "component_count": len(ordered_components),
        "component_face_count": len(selected_faces),
        "segment_face_count": len(counts),
        "occurrence_index": occurrence_index,
        "occurrence_count": occurrence_count,
        "component_surface_area_fraction": area / total_area,
    }


def _shape_thickness_ratio(payloads: list[SegmentGeometryPayload]) -> float:
    point_count = 0
    coordinate_sum = np.zeros(3, dtype=np.float64)
    outer_sum = np.zeros((3, 3), dtype=np.float64)
    for payload in payloads:
        for matrix in payload.occurrence_transforms:
            points = _transform_points(payload.points, matrix)
            point_count += len(points)
            coordinate_sum += points.sum(axis=0)
            outer_sum += points.T @ points
    if point_count < 3:
        return 0.0
    center = coordinate_sum / point_count
    covariance = outer_sum / point_count - np.outer(center, center)
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(eigenvalues[-1])
    if not math.isfinite(largest) or largest <= 1e-20:
        return 0.0
    smallest = max(float(eigenvalues[0]), 0.0)
    ratio = math.sqrt(smallest / largest)
    if not math.isfinite(ratio):
        raise TexturedRenderError("geometry thickness ratio is non-finite")
    return ratio


def _create_camera_and_lights(
    bpy: Any, collection: Any
) -> tuple[Any, list[Any]]:
    camera_data = bpy.data.cameras.new("vlm_textured_camera")
    camera_data.type = "ORTHO"
    camera_data.lens = 50.0
    camera = bpy.data.objects.new("vlm_textured_camera", camera_data)
    collection.objects.link(camera)
    lights = []
    for index in range(3):
        data = bpy.data.lights.new(f"vlm_area_{index}", "AREA")
        data.shape = "DISK"
        obj = bpy.data.objects.new(f"vlm_area_{index}", data)
        collection.objects.link(obj)
        lights.append(obj)
    return camera, lights


def _view_directions(up_axis: str) -> list[np.ndarray]:
    if str(up_axis).upper() == "Z":
        values = (
            (1.55, -2.15, 1.15),
            (-1.55, 2.15, 1.15),
            (1.0, 0.08, 0.18),
            (-1.0, -0.08, 0.18),
            (0.08, 1.0, 0.18),
            (-0.08, -1.0, 0.18),
            (0.08, -0.12, 1.0),
            (-0.08, 0.12, -1.0),
        )
        return [np.asarray(value, dtype=np.float64) for value in values]
    if str(up_axis).upper() == "Y":
        values = (
            (1.55, 1.15, -2.15),
            (-1.55, 1.15, 2.15),
            (1.0, 0.18, 0.08),
            (-1.0, 0.18, -0.08),
            (0.08, 0.18, 1.0),
            (-0.08, 0.18, -1.0),
            (0.08, 1.0, -0.12),
            (-0.08, -1.0, 0.12),
        )
        return [np.asarray(value, dtype=np.float64) for value in values]
    raise TexturedRenderError(f"unsupported stage up axis {up_axis!r}")


def _three_quarter_view_directions(up_axis: str) -> list[np.ndarray]:
    if str(up_axis).upper() == "Z":
        values = (
            (1.55, -2.15, 1.15),
            (-1.55, 2.15, 1.15),
            (1.55, 2.15, 1.15),
            (-1.55, -2.15, 1.15),
            (1.55, -2.15, -1.15),
            (-1.55, 2.15, -1.15),
            (1.55, 2.15, -1.15),
            (-1.55, -2.15, -1.15),
        )
    elif str(up_axis).upper() == "Y":
        values = (
            (1.55, 1.15, -2.15),
            (-1.55, 1.15, 2.15),
            (1.55, 1.15, 2.15),
            (-1.55, 1.15, -2.15),
            (1.55, -1.15, -2.15),
            (-1.55, -1.15, 2.15),
            (1.55, -1.15, 2.15),
            (-1.55, -1.15, -2.15),
        )
    else:
        raise TexturedRenderError(f"unsupported stage up axis {up_axis!r}")
    return [np.asarray(value, dtype=np.float64) for value in values]


def _perspective_camera_horizontal_hint(
    extraction: Mapping[str, Any], up_axis: str
) -> tuple[np.ndarray, str] | None:
    record = extraction.get("authored_perspective_camera_hint")
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise TexturedRenderError("authored Perspective camera hint is malformed")
    hint_sha256 = record.get("hint_sha256")
    payload = dict(record)
    payload.pop("hint_sha256", None)
    if (
        not isinstance(hint_sha256, str)
        or _reader_contract_hash(
            "authored-perspective-camera-hint-v1", payload
        )
        != hint_sha256
        or record.get("source")
        != "root_layer.customLayerData.cameraSettings.Perspective"
        or record.get("stage_up_axis") != up_axis
    ):
        raise TexturedRenderError("authored Perspective camera hint is stale")

    position = np.asarray(record.get("position"), dtype=np.float64)
    target = np.asarray(record.get("target"), dtype=np.float64)
    direction = np.asarray(record.get("direction"), dtype=np.float64)
    horizontal = np.asarray(
        record.get("horizontal_direction"), dtype=np.float64
    )
    if any(
        value.shape != (3,) or not np.isfinite(value).all()
        for value in (position, target, direction, horizontal)
    ):
        raise TexturedRenderError("authored Perspective camera vectors are invalid")
    expected_direction = position - target
    expected_length = float(np.linalg.norm(expected_direction))
    if expected_length <= 1.0e-10:
        raise TexturedRenderError("authored Perspective camera direction is zero")
    expected_direction /= expected_length
    stage_up = np.asarray(
        (0.0, 0.0, 1.0) if up_axis == "Z" else (0.0, 1.0, 0.0),
        dtype=np.float64,
    )
    expected_horizontal = expected_direction - float(
        np.dot(expected_direction, stage_up)
    ) * stage_up
    horizontal_fraction = float(np.linalg.norm(expected_horizontal))
    if horizontal_fraction <= 1.0e-10:
        raise TexturedRenderError(
            "authored Perspective camera has no horizontal direction"
        )
    expected_horizontal /= horizontal_fraction
    if (
        not np.allclose(direction, expected_direction, rtol=1.0e-12, atol=1.0e-12)
        or not np.allclose(
            horizontal, expected_horizontal, rtol=1.0e-12, atol=1.0e-12
        )
        or not math.isclose(
            float(record.get("horizontal_fraction")),
            horizontal_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise TexturedRenderError(
            "authored Perspective camera derived direction is inconsistent"
        )
    return horizontal, hint_sha256


def _choose_view_direction(
    payloads: list[SegmentGeometryPayload],
    up_axis: str,
    *,
    three_quarter_context: bool = False,
    horizontal_hint: np.ndarray | None = None,
    camera_hint_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (horizontal_hint is None) != (camera_hint_sha256 is None):
        raise TexturedRenderError("Perspective camera hint provenance is incomplete")
    hint = None
    stage_up = None
    if horizontal_hint is not None:
        hint = np.asarray(horizontal_hint, dtype=np.float64)
        hint_length = float(np.linalg.norm(hint))
        if (
            hint.shape != (3,)
            or not np.isfinite(hint).all()
            or hint_length <= 1.0e-10
        ):
            raise TexturedRenderError("Perspective horizontal camera hint is invalid")
        hint /= hint_length
        stage_up = np.asarray(
            (0.0, 0.0, 1.0)
            if up_axis == "Z"
            else (0.0, 1.0, 0.0)
            if up_axis == "Y"
            else (),
            dtype=np.float64,
        )
        if stage_up.shape != (3,):
            raise TexturedRenderError(f"unsupported stage up axis {up_axis!r}")
    prepared_occurrences: list[tuple[np.ndarray, np.ndarray, float, bool]] = []
    for payload in payloads:
        counts = np.asarray(payload.face_vertex_counts, dtype=np.int64)
        indices = np.asarray(payload.face_vertex_indices, dtype=np.int64)
        if (
            counts.ndim != 1
            or indices.ndim != 1
            or not len(counts)
            or np.any(counts < 1)
            or int(counts.sum()) != len(indices)
        ):
            raise TexturedRenderError(
                "exact face counts do not consume the view-selection topology"
            )
        face_starts = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1]))
        )
        face_ends = np.cumsum(counts) - 1
        next_corners = np.arange(len(indices), dtype=np.int64) + 1
        next_corners[face_ends] = face_starts
        orientation_sign = (
            -1.0
            if str(payload.provenance.get("orientation") or "rightHanded")
            == "leftHanded"
            else 1.0
        )
        double_sided = bool(payload.provenance.get("double_sided"))
        for matrix in payload.occurrence_transforms:
            points = _transform_points(payload.points, matrix)
            corners = points[indices]
            area_vectors = np.add.reduceat(
                np.cross(corners, corners[next_corners]),
                face_starts,
                axis=0,
            )
            if area_vectors.shape != (len(counts), 3) or not np.isfinite(
                area_vectors
            ).all():
                raise TexturedRenderError(
                    "vectorized view-selection face areas are invalid"
                )
            prepared_occurrences.append(
                (points, area_vectors, orientation_sign, double_sided)
            )
    candidates = (
        _three_quarter_view_directions(up_axis)
        if three_quarter_context
        else _view_directions(up_axis)
    )
    if hint is not None and not three_quarter_context:
        candidates.append(stage_up + math.hypot(0.08, 0.12) * hint)
    scored: list[tuple[float, float, int, np.ndarray, dict[str, Any]]] = []
    for candidate_index, candidate in enumerate(candidates):
        direction = candidate / np.linalg.norm(candidate)
        horizontal, vertical = _camera_frame(direction, up_axis)
        min_h = min_v = min_d = np.inf
        max_h = max_v = max_d = -np.inf
        for points, _, _, _ in prepared_occurrences:
            h = points @ horizontal
            v = points @ vertical
            d = points @ direction
            min_h, max_h = min(min_h, float(h.min())), max(max_h, float(h.max()))
            min_v, max_v = min(min_v, float(v.min())), max(max_v, float(v.max()))
            min_d, max_d = min(min_d, float(d.min())), max(max_d, float(d.max()))
        width = max_h - min_h
        height = max_v - min_v
        depth = max_d - min_d
        area = width * height
        if not math.isfinite(area) or area <= 1e-16:
            continue
        visible_surface_area = 0.0
        for _, area_vectors, orientation_sign, double_sided in prepared_occurrences:
            projected = orientation_sign * 0.5 * (area_vectors @ direction)
            contribution = (
                np.abs(projected)
                if double_sided
                else np.maximum(projected, 0.0)
            )
            visible_surface_area += float(
                np.sum(contribution, dtype=np.float64)
            )
        if not math.isfinite(visible_surface_area) or visible_surface_area <= 1e-18:
            continue
        scored.append(
            (
                visible_surface_area,
                area,
                -candidate_index,
                direction,
                {
                    "projected_width": width,
                    "projected_height": height,
                    "projected_depth": depth,
                    "projected_center": [
                        (min_h + max_h) * 0.5,
                        (min_v + max_v) * 0.5,
                        (min_d + max_d) * 0.5,
                    ],
                },
            )
        )
    if not scored:
        raise TexturedRenderError("every canonical view is degenerate for this geometry")
    selection_pool = scored
    below_up_fallback = False
    if three_quarter_context:
        upper_hemisphere = [item for item in scored if -item[2] < 4]
        if upper_hemisphere:
            selection_pool = upper_hemisphere
        else:
            below_up_fallback = True
    eligible_candidate_indices = None
    chosen_alignment = None
    if hint is not None:
        aligned = []
        for item in selection_pool:
            candidate_horizontal = item[3] - float(
                np.dot(item[3], stage_up)
            ) * stage_up
            candidate_horizontal /= np.linalg.norm(candidate_horizontal)
            alignment = float(np.dot(candidate_horizontal, hint))
            if alignment >= _PERSPECTIVE_HINT_MINIMUM_HORIZONTAL_ALIGNMENT:
                aligned.append((*item, alignment))
        if not aligned:
            raise TexturedRenderError(
                "no canonical whole-object view agrees with the authored Perspective camera"
            )
        eligible_candidate_indices = sorted(-item[2] for item in aligned)
        selection_pool = [item[:5] for item in aligned]
    surface_area, area, negative_index, direction, projection = max(
        selection_pool, key=lambda item: (item[0], item[1], item[2])
    )
    view = {
        "canonical_view_index": -negative_index,
        "camera_direction": [float(value) for value in direction],
        **projection,
        "projected_bbox_area": area,
        "projected_visible_surface_area": surface_area,
        "below_up_fallback": below_up_fallback,
        "view_selection": (
            "maximum_exact_surface_area_with_authored_perspective_azimuth"
            if hint is not None
            else "maximum_exact_front_facing_surface_area_among_three_quarter_views"
            if three_quarter_context
            else "maximum_exact_front_facing_projected_surface_area"
        ),
    }
    if hint is not None:
        chosen_horizontal = direction - float(
            np.dot(direction, stage_up)
        ) * stage_up
        chosen_horizontal /= np.linalg.norm(chosen_horizontal)
        chosen_alignment = float(np.dot(chosen_horizontal, hint))
        view.update(
            {
                "perspective_hint_sha256": camera_hint_sha256,
                "perspective_hint_minimum_horizontal_alignment": (
                    _PERSPECTIVE_HINT_MINIMUM_HORIZONTAL_ALIGNMENT
                ),
                "perspective_hint_eligible_candidate_indices": (
                    eligible_candidate_indices
                ),
                "chosen_perspective_horizontal_alignment": chosen_alignment,
            }
        )
    return direction, view


def _fit_camera_and_lights(
    matrix_type: Any,
    vector_type: Any,
    camera: Any,
    lights: list[Any],
    up_axis: str,
    direction: np.ndarray,
    projection: Mapping[str, Any],
) -> None:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    horizontal, vertical = _camera_frame(direction, up_axis)
    projected_center = np.asarray(projection["projected_center"], dtype=np.float64)
    projected_size = np.asarray(
        (
            projection["projected_width"],
            projection["projected_height"],
            projection["projected_depth"],
        ),
        dtype=np.float64,
    )
    if (
        projected_center.shape != (3,)
        or projected_size.shape != (3,)
        or not np.isfinite(projected_center).all()
        or not np.isfinite(projected_size).all()
        or np.any(projected_size < 0.0)
        or max(float(projected_size[0]), float(projected_size[1])) <= 1e-12
    ):
        raise TexturedRenderError("camera projection bounds are invalid")
    center = (
        horizontal * projected_center[0]
        + vertical * projected_center[1]
        + direction * projected_center[2]
    )
    radius = max(float(np.linalg.norm(projected_size) * 0.5), 1e-5)
    location = center + direction * radius * 3.2
    camera.location = tuple(float(value) for value in location)
    camera.rotation_euler = matrix_type(
        (
            (horizontal[0], vertical[0], direction[0]),
            (horizontal[1], vertical[1], direction[1]),
            (horizontal[2], vertical[2], direction[2]),
        )
    ).to_euler()
    camera.data.ortho_scale = max(
        max(float(projected_size[0]), float(projected_size[1]))
        * _ORTHOGRAPHIC_FRAME_MARGIN,
        1e-4,
    )
    camera.data.clip_start = max(radius * 1e-4, 1e-6)
    camera.data.clip_end = max(radius * 8.0, camera.data.clip_start * 1000.0)

    optical_axis = np.asarray(
        camera.rotation_euler.to_matrix()
        @ vector_type((0.0, 0.0, -1.0)),
        dtype=np.float64,
    )
    to_center = center - location
    optical_error = float(
        np.linalg.norm(
            to_center - float(np.dot(to_center, optical_axis)) * optical_axis
        )
    )
    if optical_error > max(radius * 1e-6, 1e-8):
        raise TexturedRenderError("camera optical axis does not pass through frame center")

    if str(up_axis).upper() == "Z":
        offsets = ((-1.6, -1.4, 2.2), (2.0, 0.4, 1.0), (-0.4, 2.0, 0.6))
    else:
        offsets = ((-1.6, 2.2, -1.4), (2.0, 1.0, 0.4), (-0.4, 0.6, 2.0))
    strengths = (68.0, 40.0, 28.0)
    for light, offset, strength in zip(lights, offsets, strengths):
        light_location = center + np.asarray(offset) * radius
        light.location = tuple(float(value) for value in light_location)
        light.rotation_euler = (
            vector_type(tuple(float(value) for value in center - light_location))
            .to_track_quat("-Z", "Y")
            .to_euler()
        )
        light.data.energy = strength * max(radius * radius, 0.01)
        light.data.size = max(radius * 1.4, 0.01)


def _sphere_mesh_data(segments: int = 64, rings: int = 32) -> tuple[list[Any], list[Any], set[Any]]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    for ring in range(1, rings):
        latitude = math.pi * ring / rings
        z = math.cos(latitude)
        radius = math.sin(latitude)
        for longitude in range(segments):
            angle = 2.0 * math.pi * longitude / segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    south = len(vertices)
    vertices.append((0.0, 0.0, -1.0))

    def vertex(ring_index: int, longitude: int) -> int:
        return 1 + (ring_index - 1) * segments + (longitude % segments)

    faces: list[tuple[int, ...]] = []
    for longitude in range(segments):
        faces.append((0, vertex(1, longitude + 1), vertex(1, longitude)))
    for ring in range(1, rings - 1):
        for longitude in range(segments):
            faces.append(
                (
                    vertex(ring, longitude),
                    vertex(ring, longitude + 1),
                    vertex(ring + 1, longitude + 1),
                    vertex(ring + 1, longitude),
                )
            )
    for longitude in range(segments):
        faces.append(
            (vertex(rings - 1, longitude), vertex(rings - 1, longitude + 1), south)
        )
    faces = [tuple(reversed(face)) for face in faces]
    vertex_array = np.asarray(vertices, dtype=np.float64)
    for face in faces:
        polygon = vertex_array[list(face)]
        area_vector = np.sum(
            np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0
        )
        if float(np.dot(area_vector, polygon.mean(axis=0))) <= 0.0:
            raise TexturedRenderError("material sphere contains an inward-facing polygon")
    seam_pairs = {tuple(sorted((0, vertex(1, 0)))), tuple(sorted((south, vertex(rings - 1, 0))))}
    for ring in range(1, rings - 1):
        seam_pairs.add(tuple(sorted((vertex(ring, 0), vertex(ring + 1, 0)))))
    return vertices, faces, seam_pairs


def _create_conformal_sphere(
    bpy: Any,
    scene: Any,
    collection: Any,
) -> Any:
    vertices, faces, seam_pairs = _sphere_mesh_data()
    mesh = bpy.data.meshes.new("vlm_conformal_material_sphere")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    for edge in mesh.edges:
        edge.use_seam = tuple(sorted(int(value) for value in edge.vertices)) in seam_pairs
    uv_layer = mesh.uv_layers.new(name="SphereConformalUV")
    if uv_layer is None:
        raise TexturedRenderError("Blender could not create the sphere UV layer")
    sphere = bpy.data.objects.new("vlm_conformal_material_sphere", mesh)
    collection.objects.link(sphere)
    view_layer = scene.view_layers[0]
    for obj in view_layer.objects:
        obj.select_set(False)
    with bpy.context.temp_override(
        scene=scene,
        view_layer=view_layer,
        active_object=sphere,
        object=sphere,
        selected_objects=[sphere],
        selected_editable_objects=[sphere],
    ):
        sphere.select_set(True)
        view_layer.objects.active = sphere
        if bpy.ops.object.mode_set(mode="EDIT") != {"FINISHED"}:
            raise TexturedRenderError("could not enter edit mode for conformal sphere unwrap")
        bpy.ops.mesh.select_all(action="SELECT")
        unwrap_result = bpy.ops.uv.unwrap(method="CONFORMAL", margin=0.002)
        bpy.ops.object.mode_set(mode="OBJECT")
    if unwrap_result != {"FINISHED"}:
        raise TexturedRenderError("Blender conformal sphere unwrap did not finish")
    mesh.update()
    uv_layer = mesh.uv_layers.get("SphereConformalUV")
    if uv_layer is None:
        raise TexturedRenderError("conformal sphere UV layer disappeared after unwrap")
    values = np.asarray([tuple(item.uv) for item in uv_layer.data], dtype=np.float64)
    if (
        values.shape != (len(mesh.loops), 2)
        or not np.isfinite(values).all()
        or np.ptp(values[:, 0]) <= 1e-6
        or np.ptp(values[:, 1]) <= 1e-6
    ):
        raise TexturedRenderError("conformal sphere UVs are empty or invalid")
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    sphere.hide_render = True
    return sphere


def check_render_visibility(path: Path, size: int) -> None:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate(
        (rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]), axis=0
    )
    background = np.median(border, axis=0)
    difference = np.max(np.abs(rgb - background[None, None, :]), axis=2)
    visible_pixels = int(np.count_nonzero(difference >= 3))
    if visible_pixels < max(64, size):
        raise _RenderVisibilityError(visible_pixels)


def _render_png(
    bpy: Any,
    scene: Any,
    destination: Path,
    role: str,
    runtime: Mapping[str, Any],
    view_record: Mapping[str, Any],
    extra: Mapping[str, Any],
    *,
    size: int,
) -> dict[str, Any]:
    if role not in _IMAGE_ROLES:
        raise TexturedRenderError(f"unknown production image role {role!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.part.png"
    )
    scene.render.filepath = str(temporary)
    try:
        result = bpy.ops.render.render(scene=scene.name, write_still=True)
        if result != {"FINISHED"}:
            raise TexturedRenderError(f"Cycles render did not finish for {role!r}: {result!r}")
        check_render_visibility(temporary, size)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    record = {
        "status": "complete",
        "role": role,
        "output_path": str(destination.resolve()),
        "output_file_sha256": _sha256_file(destination),
        "engine": runtime["engine"],
        "render_backend": runtime["render_backend"],
        "render_device": list(runtime["render_device"]),
        "width": size,
        "height": size,
        "view": dict(view_record),
        "render_settings": {
            "blender_version": list(runtime["blender_version"]),
            "cycles_samples": runtime["cycles_samples"],
            "adaptive_sampling": runtime["adaptive_sampling"],
            "denoising": runtime["denoising"],
            "denoiser": runtime["denoiser"],
            "denoising_input_passes": runtime["denoising_input_passes"],
            "denoising_prefilter": runtime["denoising_prefilter"],
            "seed": runtime["seed"],
            "animated_seed": runtime["animated_seed"],
            "studio_background": "opaque_neutral_gray",
        },
        **dict(extra),
    }
    return record


def _snapshot_data_names(bpy: Any) -> dict[str, set[str]]:
    return {
        name: {item.name for item in getattr(bpy.data, name)}
        for name in (
            "objects",
            "meshes",
            "materials",
            "images",
            "cameras",
            "lights",
            "worlds",
            "collections",
            "scenes",
        )
    }


def _cleanup_new_data(bpy: Any, before: Mapping[str, set[str]]) -> None:
    for category in (
        "objects",
        "meshes",
        "materials",
        "images",
        "cameras",
        "lights",
        "worlds",
        "collections",
        "scenes",
    ):
        datablocks = getattr(bpy.data, category)
        for block in list(datablocks):
            if block.name in before[category]:
                continue
            if category in {"objects", "collections"}:
                datablocks.remove(block, do_unlink=True)
            else:
                datablocks.remove(block)


def _render_whole_component_record(
    bpy: Any,
    matrix_type: Any,
    vector_type: Any,
    scene: Any,
    camera: Any,
    lights: list[Any],
    renderables: list[_Renderable],
    segments: Mapping[str, Any],
    geometry_payloads: Mapping[str, SegmentGeometryPayload],
    plans: Mapping[str, MaterialPlan],
    payload_hashes: Mapping[str, str],
    runtime: Mapping[str, Any],
    destination_root: Path,
    up_axis: str,
    perspective_hint: tuple[np.ndarray, str] | None,
    entrypoint_sha256: str,
    size: int,
) -> dict[str, Any]:
    whole_payloads = [
        geometry_payloads[key]
        for key, segment in segments.items()
        if _render_content(segment)
    ]
    if not whole_payloads:
        raise TexturedRenderError(
            "asset has no visible render/default-purpose geometry"
        )
    for renderable in renderables:
        renderable.whole_state()
    whole_thickness_ratio = _shape_thickness_ratio(whole_payloads)
    whole_uses_context_view = (
        whole_thickness_ratio >= _THREE_QUARTER_THICKNESS_RATIO
    )
    whole_direction, whole_view = _choose_view_direction(
        whole_payloads,
        up_axis,
        three_quarter_context=whole_uses_context_view,
        horizontal_hint=(
            perspective_hint[0] if perspective_hint is not None else None
        ),
        camera_hint_sha256=(
            perspective_hint[1] if perspective_hint is not None else None
        ),
    )
    whole_view.update(
        {
            "shape_thickness_ratio": whole_thickness_ratio,
            "three_quarter_threshold": _THREE_QUARTER_THICKNESS_RATIO,
            "shape_view_class": (
                "volumetric_three_quarter"
                if whole_uses_context_view
                else "thin_maximum_area"
            ),
        }
    )
    _fit_camera_and_lights(
        matrix_type,
        vector_type,
        camera,
        lights,
        up_axis,
        whole_direction,
        whole_view,
    )
    return _render_png(
        bpy,
        scene,
        destination_root / "textured_whole_object.png",
        "textured_whole_object",
        runtime,
        whole_view,
        {
            "geometry_payload_hashes": dict(sorted(payload_hashes.items())),
            "usd_entrypoint_sha256": entrypoint_sha256,
            "material_plan_hashes": {
                key: plans[key].provenance_sha256 for key in sorted(plans)
            },
        },
        size=size,
    )


def _render_segment_component_records(
    bpy: Any,
    matrix_type: Any,
    vector_type: Any,
    scene: Any,
    collection: Any,
    camera: Any,
    lights: list[Any],
    sphere: Any,
    renderables: list[_Renderable],
    image_cache: dict[tuple[str, str, str], Any],
    blender_materials: Mapping[str, Any],
    segments: Mapping[str, Any],
    geometry_payloads: Mapping[str, SegmentGeometryPayload],
    plans: Mapping[str, MaterialPlan],
    payload_hashes: Mapping[str, str],
    runtime: Mapping[str, Any],
    destination_root: Path,
    up_axis: str,
    segment_key: str,
    size: int,
) -> dict[str, Any]:
    segment = segments[segment_key]
    payload = geometry_payloads[segment_key]
    segment_id = str(segment["segment_id"])
    sphere.hide_render = True
    selected_thickness_ratio = _shape_thickness_ratio([payload])
    selected_uses_context_view = (
        selected_thickness_ratio >= _THREE_QUARTER_THICKNESS_RATIO
    )
    selected_direction, selected_view = _choose_view_direction(
        [payload],
        up_axis,
        three_quarter_context=selected_uses_context_view,
    )
    selected_view.update(
        {
            "shape_thickness_ratio": selected_thickness_ratio,
            "three_quarter_threshold": _THREE_QUARTER_THICKNESS_RATIO,
            "shape_view_class": (
                "volumetric_three_quarter"
                if selected_uses_context_view
                else "thin_maximum_area"
            ),
        }
    )
    _fit_camera_and_lights(
        matrix_type,
        vector_type,
        camera,
        lights,
        up_axis,
        selected_direction,
        selected_view,
    )
    segment_directory = (
        destination_root / "segments" / _safe_component(segment_id)
    )
    selected_provenance = {
        "segment_id": segment_id,
        "geometry_payload_sha256": payload_hashes[segment_key],
        "material_record_id": segment.get("material_record_id"),
        "material_plan_sha256": plans[segment_key].provenance_sha256,
    }
    selected_proxies, isolation_record = _selected_render_proxies(
        bpy,
        collection,
        renderables,
        segment_key,
        plans[segment_key],
        blender_materials[segment_key],
    )
    selected_provenance["selected_geometry_isolation"] = isolation_record
    try:
        try:
            selected_record = _render_png(
                bpy,
                scene,
                segment_directory / "textured_selected_part.png",
                "textured_selected_part",
                runtime,
                selected_view,
                selected_provenance,
                size=size,
            )
        except _RenderVisibilityError as full_view_error:
            aid = _visibility_exemplar(payload)
            if aid is None:
                raise
            exemplar, aid_record = aid
            exemplar_thickness = _shape_thickness_ratio([exemplar])
            exemplar_uses_context_view = (
                exemplar_thickness >= _THREE_QUARTER_THICKNESS_RATIO
            )
            selected_direction, selected_view = _choose_view_direction(
                [exemplar],
                up_axis,
                three_quarter_context=exemplar_uses_context_view,
            )
            selected_view.update(
                {
                    "shape_thickness_ratio": exemplar_thickness,
                    "three_quarter_threshold": _THREE_QUARTER_THICKNESS_RATIO,
                    "shape_view_class": (
                        "volumetric_three_quarter"
                        if exemplar_uses_context_view
                        else "thin_maximum_area"
                    ),
                    "full_segment_visible_pixels_before_aid": (
                        full_view_error.visible_pixels
                    ),
                    **aid_record,
                }
            )
            _fit_camera_and_lights(
                matrix_type,
                vector_type,
                camera,
                lights,
                up_axis,
                selected_direction,
                selected_view,
            )
            selected_record = _render_png(
                bpy,
                scene,
                segment_directory / "textured_selected_part.png",
                "textured_selected_part",
                runtime,
                selected_view,
                selected_provenance,
                size=size,
            )
        _attest_proxy_normal_state(selected_proxies, "selected-part rendering")
    finally:
        _remove_selected_proxies(bpy, selected_proxies)

    for renderable in renderables:
        for obj in renderable.objects:
            obj.hide_render = True
    swatches, sampling_record = _bake_segment_swatches(
        bpy,
        scene,
        collection,
        renderables,
        segment_key,
        plans[segment_key],
        image_cache,
        size,
        payload_hashes[segment_key],
    )
    for renderable in renderables:
        for obj in renderable.objects:
            obj.hide_render = True
    sphere_material = None
    try:
        sphere_material = _build_blender_material(
            bpy,
            plans[segment_key],
            image_cache,
            uv_override="SphereConformalUV",
            standardized_sphere=True,
            swatches=swatches,
        )
        sampling_record["sphere_node_tree_attestation"] = (
            _sphere_swatch_node_attestation(sphere_material, swatches)
        )
        sampling_record.pop("segment_sampling_plan_sha256", None)
        sampling_record["segment_sampling_plan_sha256"] = _canonical_hash(
            "selected-segment-swatch-plan", sampling_record
        )
        sphere.hide_render = False
        sphere.data.materials.clear()
        sphere.data.materials.append(sphere_material)
        sphere_direction = _view_directions(up_axis)[0]
        sphere_direction /= np.linalg.norm(sphere_direction)
        sphere_view = {
            "canonical_view_index": 0,
            "camera_direction": [float(value) for value in sphere_direction],
            "projected_width": 2.0,
            "projected_height": 2.0,
            "projected_depth": 2.0,
            "projected_center": [0.0, 0.0, 0.0],
        }
        _fit_camera_and_lights(
            matrix_type,
            vector_type,
            camera,
            lights,
            up_axis,
            sphere_direction,
            sphere_view,
        )
        sphere_record = _render_png(
            bpy,
            scene,
            segment_directory / "selected_material_sphere.png",
            "selected_material_sphere",
            runtime,
            sphere_view,
            {
                "segment_id": segment_id,
                "geometry_payload_sha256": payload_hashes[segment_key],
                "material_record_id": segment.get("material_record_id"),
                "material_plan_sha256": plans[segment_key].provenance_sha256,
                "conformal_uv_layer": "SphereConformalUV",
                "segment_sampling_plan_sha256": sampling_record[
                    "segment_sampling_plan_sha256"
                ],
                "selected_segment_swatch_sampling": sampling_record,
            },
            size=size,
        )
    finally:
        sphere.hide_render = True
        sphere.data.materials.clear()
        if (
            sphere_material is not None
            and bpy.data.materials.get(sphere_material.name) is sphere_material
        ):
            bpy.data.materials.remove(sphere_material)
        for swatch in swatches.values():
            if bpy.data.images.get(swatch.image.name) is swatch.image:
                bpy.data.images.remove(swatch.image)
    return {
        "textured_selected_part": selected_record,
        "selected_material_sphere": sphere_record,
    }


def _render_textured_vlm_component_in_process(
    usd_path: str | os.PathLike[str],
    extraction: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    geometry_payloads: Mapping[str, SegmentGeometryPayload],
    component: str,
    target_segment_key: str | None,
    subdivision_level: int,
    size: int = 512,
) -> dict[str, Any]:
    """Render one exact component inside a short-lived Blender process."""

    source_path = Path(usd_path)
    if not source_path.is_file():
        raise TexturedRenderError(f"USD entrypoint is unreadable: {source_path}")
    segments, materials, payload_hashes = _validate_payloads(
        extraction, geometry_payloads
    )
    if component not in {"whole", "segment"}:
        raise TexturedRenderError(f"unknown render component {component!r}")
    if component == "whole":
        if target_segment_key is not None:
            raise TexturedRenderError("whole component unexpectedly has a target")
    elif (
        not isinstance(target_segment_key, str)
        or target_segment_key not in segments
    ):
        raise TexturedRenderError("segment component has no exact target")
    if type(subdivision_level) is not int or not 1 <= subdivision_level <= 3:
        raise TexturedRenderError("component has an invalid asset-wide subdivision level")
    entrypoint_sha256 = _sha256_file(source_path)
    up_axes = {
        str(payload.provenance.get("stage_up_axis") or "")
        for payload in geometry_payloads.values()
    }
    if len(up_axes) != 1 or next(iter(up_axes)).upper() not in {"Y", "Z"}:
        raise TexturedRenderError(f"exact geometry has inconsistent up axes: {sorted(up_axes)!r}")
    up_axis = next(iter(up_axes)).upper()
    perspective_hint = _perspective_camera_horizontal_hint(extraction, up_axis)
    for key, payload in geometry_payloads.items():
        if payload.provenance.get("usd_entrypoint_sha256") != entrypoint_sha256:
            raise TexturedRenderError(f"USD entrypoint changed after payload extraction for {key!r}")

    plans = {
        str(key): _build_material_plan(segments[key], payload, materials)
        for key, payload in geometry_payloads.items()
    }
    uv_aliases = {key: _uv_aliases(plan) for key, plan in plans.items()}
    if tuple(bpy.app.version[:2]) != _BLENDER_VERSION:
        raise TexturedRenderError(
            f"expected pinned bpy 4.2.x, found {tuple(bpy.app.version)}"
        )

    before = _snapshot_data_names(bpy)
    previous_window_scene = bpy.context.window.scene if bpy.context.window else None
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    try:
        scene, collection = _new_scene(bpy, size)
        if bpy.context.window:
            bpy.context.window.scene = scene
        runtime = _configure_cycles_gpu(bpy, scene)
        sphere = (
            _create_conformal_sphere(bpy, scene, collection)
            if component == "segment"
            else None
        )
        image_cache: dict[tuple[str, str, str], Any] = {}
        blender_materials = {
            key: _build_blender_material(
                bpy,
                plan,
                image_cache,
                uv_aliases=uv_aliases[key],
            )
            for key, plan in plans.items()
        }
        transparent = _transparent_material(bpy)
        renderables = _build_asset_renderables(
            bpy,
            Matrix,
            collection,
            segments,
            geometry_payloads,
            plans,
            blender_materials,
            transparent,
            uv_aliases,
            subdivision_level,
        )
        camera, lights = _create_camera_and_lights(bpy, collection)
        scene.camera = camera

        if component == "whole":
            return {
                "runtime": {
                    "engine": runtime["engine"],
                    "render_backend": runtime["render_backend"],
                    "render_device": list(runtime["render_device"]),
                },
                "records": {
                    "textured_whole_object": _render_whole_component_record(
                        bpy,
                        Matrix,
                        Vector,
                        scene,
                        camera,
                        lights,
                        renderables,
                        segments,
                        geometry_payloads,
                        plans,
                        payload_hashes,
                        runtime,
                        destination_root,
                        up_axis,
                        perspective_hint,
                        entrypoint_sha256,
                        size,
                    )
                },
            }

        assert target_segment_key is not None
        assert sphere is not None
        return {
            "runtime": {
                "engine": runtime["engine"],
                "render_backend": runtime["render_backend"],
                "render_device": list(runtime["render_device"]),
            },
            "records": _render_segment_component_records(
                bpy,
                Matrix,
                Vector,
                scene,
                collection,
                camera,
                lights,
                sphere,
                renderables,
                image_cache,
                blender_materials,
                segments,
                geometry_payloads,
                plans,
                payload_hashes,
                runtime,
                destination_root,
                up_axis,
                target_segment_key,
                size,
            ),
        }
    except TexturedRenderError:
        raise
    except Exception as exc:
        raise TexturedRenderError(
            f"Blender GPU textured render failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if bpy.context.window and previous_window_scene is not None:
            bpy.context.window.scene = previous_window_scene
        _cleanup_new_data(bpy, before)


_COMPONENT_TIMEOUT_SECONDS = 40 * 60
_COMPONENT_TERMINATION_GRACE_SECONDS = 10
_RETRIABLE_NATIVE_SIGNALS = frozenset(
    {
        signal.SIGSEGV,
        signal.SIGBUS,
        signal.SIGABRT,
        signal.SIGILL,
        signal.SIGFPE,
    }
)


def _renderer_source_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    directories = [root]
    for candidate in root.rglob("*"):
        metadata = os.lstat(candidate)
        if stat.S_ISLNK(metadata.st_mode):
            raise TexturedRenderError(
                f"render staging tree contains a symlink: {candidate}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(candidate)
    for directory in sorted(
        directories,
        key=lambda value: len(value.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _require_private_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise TexturedRenderError(
            f"component attempt directory is not private: {path}"
        )


def _new_private_directory(parent: Path, prefix: str) -> Path:
    for _ in range(8):
        path = parent / f"{prefix}{uuid.uuid4().hex}"
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        try:
            _require_private_directory(path)
        except Exception:
            if os.path.lexists(path):
                metadata = os.lstat(path)
                if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
                    metadata.st_mode
                ):
                    path.rmdir()
            raise
        return path
    raise TexturedRenderError("could not allocate a private component directory")


def _remove_generated_directory(path: Path, parent: Path, prefix: str) -> None:
    if path.parent != parent or not path.name.startswith(prefix):
        raise TexturedRenderError(f"refusing unsafe component cleanup: {path}")
    if not os.path.lexists(path):
        return
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise TexturedRenderError(f"component cleanup target changed type: {path}")
    shutil.rmtree(path)


def _atomic_write_private_pickle(path: Path, value: Any) -> None:
    _require_private_directory(path.parent)
    if os.path.lexists(path):
        raise TexturedRenderError(f"private pickle already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _read_private_pickle(path: Path) -> Any:
    _require_private_directory(path.parent)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
        ):
            raise TexturedRenderError(f"component pickle is not private: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return pickle.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _component_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "engine": runtime.get("engine"),
        "render_backend": runtime.get("render_backend"),
        "render_device": list(runtime.get("render_device") or []),
    }


def _component_scope_keys(
    target_segment_key: str,
    geometry_payloads: Mapping[str, SegmentGeometryPayload],
) -> tuple[str, ...]:
    payload = geometry_payloads[target_segment_key]
    scheme = str(payload.provenance.get("subdivision_scheme") or "catmullClark")
    if scheme == "none":
        return (target_segment_key,)
    if scheme != "catmullClark":
        raise TexturedRenderError(
            f"unsupported subdivisionScheme={scheme!r} at "
            f"{payload.provenance.get('prim_path')}"
        )
    group_key = _catmull_group_key(payload)
    siblings = tuple(
        sorted(
            key
            for key, candidate in geometry_payloads.items()
            if str(
                candidate.provenance.get("subdivision_scheme")
                or "catmullClark"
            )
            == "catmullClark"
            and _catmull_group_key(candidate) == group_key
        )
    )
    if target_segment_key not in siblings:
        raise TexturedRenderError("Catmull-Clark component lost its target")
    return siblings


def _component_expected_paths(
    component: str,
    target_segment_key: str | None,
    segments: Mapping[str, Any],
) -> tuple[str, ...]:
    if component == "whole":
        return ("textured_whole_object.png",)
    if target_segment_key is None:
        raise TexturedRenderError("segment component is missing its target")
    segment_id = str(segments[target_segment_key].get("segment_id") or "")
    if not segment_id:
        raise TexturedRenderError("segment component has no segment ID")
    directory = f"segments/{_safe_component(segment_id)}"
    return (
        f"{directory}/textured_selected_part.png",
        f"{directory}/selected_material_sphere.png",
    )


def _component_context(
    *,
    component: str,
    target_segment_key: str | None,
    nonce: str,
    source_path: Path,
    renderer_source_sha256: str,
    entrypoint_sha256: str,
    extraction: Mapping[str, Any],
    geometry_payloads: Mapping[str, SegmentGeometryPayload],
    plans: Mapping[str, MaterialPlan],
    subdivision_level: int,
    up_axis: str,
    size: int,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise TexturedRenderError("component nonce is malformed")
    segments = extraction["segments"]
    materials = extraction["materials"]
    payload_hashes = {
        key: geometry_payloads[key].manifest()["payload_sha256"]
        for key in sorted(geometry_payloads)
    }
    material_plan_hashes = {
        key: plans[key].provenance_sha256 for key in sorted(plans)
    }
    return {
        "component": component,
        "target_segment_key": target_segment_key,
        "component_nonce": nonce,
        "source_path": str(source_path),
        "renderer_source_sha256": renderer_source_sha256,
        "entrypoint_sha256": entrypoint_sha256,
        "scope_segment_keys": sorted(geometry_payloads),
        "segments_sha256": _canonical_hash(
            "render-component-segments", segments
        ),
        "materials_sha256": _canonical_hash(
            "render-component-materials", materials
        ),
        "perspective_hint_sha256": _canonical_hash(
            "render-component-perspective-hint",
            extraction.get("authored_perspective_camera_hint"),
        ),
        "geometry_payload_hashes": payload_hashes,
        "material_plan_hashes": material_plan_hashes,
        "asset_subdivision_level": subdivision_level,
        "stage_up_axis": up_axis,
        "render_size": size,
        "expected_png_paths": list(
            _component_expected_paths(component, target_segment_key, segments)
        ),
    }


def _component_request_hash(context: Mapping[str, Any]) -> str:
    return _canonical_hash("fresh-process-render-component-request", context)


def _validate_worker_request(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(request, dict):
        raise TexturedRenderError("component request is not a mapping")
    context = request.get("context")
    if not isinstance(context, dict):
        raise TexturedRenderError("component request has no context")
    if not re.fullmatch(
        r"[0-9a-f]{32}", str(context.get("component_nonce") or "")
    ):
        raise TexturedRenderError("component request nonce is malformed")
    request_sha256 = request.get("request_sha256")
    if request_sha256 != _component_request_hash(context):
        raise TexturedRenderError("component request hash is stale or corrupt")
    if context.get("renderer_source_sha256") != _renderer_source_sha256():
        raise TexturedRenderError("component renderer source hash changed")

    source_path = Path(request.get("usd_path", ""))
    if str(source_path) != context.get("source_path") or not source_path.is_file():
        raise TexturedRenderError("component source path is stale or unreadable")
    entrypoint_sha256 = _sha256_file(source_path)
    if entrypoint_sha256 != context.get("entrypoint_sha256"):
        raise TexturedRenderError("component USD entrypoint bytes changed")

    extraction = request.get("extraction")
    geometry_payloads = request.get("geometry_payloads")
    if not isinstance(extraction, Mapping) or not isinstance(
        geometry_payloads, Mapping
    ):
        raise TexturedRenderError("component inputs are malformed")
    segments, materials, payload_hashes = _validate_payloads(
        extraction, geometry_payloads
    )
    scope_keys = tuple(sorted(segments))
    if (
        list(scope_keys) != context.get("scope_segment_keys")
        or set(geometry_payloads) != set(scope_keys)
        or payload_hashes != context.get("geometry_payload_hashes")
    ):
        raise TexturedRenderError("component geometry scope or hashes changed")
    if _canonical_hash("render-component-segments", segments) != context.get(
        "segments_sha256"
    ):
        raise TexturedRenderError("component segment records changed")
    if _canonical_hash("render-component-materials", materials) != context.get(
        "materials_sha256"
    ):
        raise TexturedRenderError("component material records changed")
    if _canonical_hash(
        "render-component-perspective-hint",
        extraction.get("authored_perspective_camera_hint"),
    ) != context.get("perspective_hint_sha256"):
        raise TexturedRenderError("component camera hint changed")

    up_axes = {
        str(payload.provenance.get("stage_up_axis") or "").upper()
        for payload in geometry_payloads.values()
    }
    if up_axes != {context.get("stage_up_axis")} or not up_axes <= {"Y", "Z"}:
        raise TexturedRenderError("component stage up axis changed")
    for key, payload in geometry_payloads.items():
        if payload.provenance.get("usd_entrypoint_sha256") != entrypoint_sha256:
            raise TexturedRenderError(
                f"component payload entrypoint binding changed for {key!r}"
            )
    plans = {
        str(key): _build_material_plan(segments[key], payload, materials)
        for key, payload in geometry_payloads.items()
    }
    plan_hashes = {
        key: plans[key].provenance_sha256 for key in sorted(plans)
    }
    if plan_hashes != context.get("material_plan_hashes"):
        raise TexturedRenderError("component material plans changed")

    component = context.get("component")
    target_segment_key = context.get("target_segment_key")
    if component == "whole":
        if target_segment_key is not None:
            raise TexturedRenderError("whole component has a target")
    elif component == "segment":
        if (
            not isinstance(target_segment_key, str)
            or target_segment_key not in segments
        ):
            raise TexturedRenderError("segment component target is absent")
    else:
        raise TexturedRenderError("component request has an unknown kind")
    if context.get("expected_png_paths") != list(
        _component_expected_paths(component, target_segment_key, segments)
    ):
        raise TexturedRenderError("component output path contract changed")

    size = context.get("render_size")
    subdivision_level = context.get("asset_subdivision_level")
    if type(size) is not int or size != 512:
        raise TexturedRenderError("component render size is not production size")
    if type(subdivision_level) is not int or not 1 <= subdivision_level <= 3:
        raise TexturedRenderError("component subdivision level is invalid")
    output_dir = Path(request.get("output_dir", ""))
    if not output_dir.is_absolute():
        raise TexturedRenderError("component output directory is not absolute")
    _require_private_directory(output_dir)
    if any(output_dir.iterdir()):
        raise TexturedRenderError("component output directory is not empty")

    rendered = _render_textured_vlm_component_in_process(
        source_path,
        extraction,
        output_dir,
        geometry_payloads=geometry_payloads,
        component=component,
        target_segment_key=target_segment_key,
        subdivision_level=subdivision_level,
        size=size,
    )
    result = {
        "status": "complete",
        "component": component,
        "target_segment_key": target_segment_key,
        "component_nonce": context.get("component_nonce"),
        "request_sha256": request_sha256,
        "renderer_source_sha256": context.get("renderer_source_sha256"),
        "context": context,
        "runtime": rendered["runtime"],
        "records": rendered["records"],
    }
    return result, context


def _component_worker_entry(request_path: Path, result_path: Path) -> int:
    request: Any = None
    try:
        if (
            request_path.parent != result_path.parent
            or request_path.name != "request.pkl"
        ):
            raise TexturedRenderError(
                "component request and result are not in one attempt directory"
            )
        _require_private_directory(request_path.parent)
        if result_path.name != "result.pkl" or os.path.lexists(result_path):
            raise TexturedRenderError("component result path is unsafe")
        request = _read_private_pickle(request_path)
        if (
            not isinstance(request, Mapping)
            or Path(request.get("output_dir", "")).resolve()
            != request_path.parent / "output"
        ):
            raise TexturedRenderError("component output path escaped its attempt")
        result, _ = _validate_worker_request(request)
        _atomic_write_private_pickle(result_path, result)
    except Exception as exc:
        traceback.print_exc()
        context = request.get("context") if isinstance(request, Mapping) else None
        error_result = {
            "status": "error",
            "component": (
                context.get("component") if isinstance(context, Mapping) else None
            ),
            "target_segment_key": (
                context.get("target_segment_key")
                if isinstance(context, Mapping)
                else None
            ),
            "component_nonce": (
                context.get("component_nonce")
                if isinstance(context, Mapping)
                else None
            ),
            "request_sha256": (
                request.get("request_sha256")
                if isinstance(request, Mapping)
                else None
            ),
            "renderer_source_sha256": _renderer_source_sha256(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        error_written = False
        try:
            if os.path.lexists(result_path):
                os.unlink(result_path)
            _atomic_write_private_pickle(result_path, error_result)
            error_written = True
        except Exception:
            traceback.print_exc()
        if not error_written:
            return 1
    return 0


def _validate_component_runtime(runtime: Any) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise TexturedRenderError("component runtime is missing")
    if set(runtime) != {
        "engine",
        "render_backend",
        "render_device",
    }:
        raise TexturedRenderError("component runtime has unexpected fields")
    normalized = _component_runtime(runtime)
    backend = normalized["render_backend"]
    devices = normalized["render_device"]
    if (
        normalized["engine"] != "CYCLES"
        or backend not in _GPU_BACKENDS
        or not devices
        or any(
            not isinstance(device, str)
            or not device.strip()
            or "cpu" in device.casefold()
            or device.partition(":")[0].strip().upper() != backend
            for device in devices
        )
    ):
        raise TexturedRenderError("component did not attest one GPU-only runtime")
    return normalized


def _component_output_files(output_root: Path) -> set[str]:
    _require_private_directory(output_root)
    files: set[str] = set()
    for candidate in output_root.rglob("*"):
        metadata = os.lstat(candidate)
        if stat.S_ISLNK(metadata.st_mode):
            raise TexturedRenderError(
                f"component output contains a symlink: {candidate}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise TexturedRenderError(
                f"component output contains a non-file: {candidate}"
            )
        relative = candidate.relative_to(output_root).as_posix()
        if relative in files:
            raise TexturedRenderError("component output contains a duplicate path")
        files.add(relative)
    return files


def _validate_canonical_record(
    record: Any,
    field: str,
    label: str,
    description: str,
) -> None:
    if not isinstance(record, Mapping):
        raise TexturedRenderError(f"{description} is missing")
    observed = record.get(field)
    if not isinstance(observed, str) or not re.fullmatch(r"[0-9a-f]{64}", observed):
        raise TexturedRenderError(f"{description} has no valid {field}")
    unhashed = dict(record)
    unhashed.pop(field, None)
    if _canonical_hash(label, unhashed) != observed:
        raise TexturedRenderError(f"{description} is stale or corrupt")


def _validate_component_render_record(
    record: Any,
    *,
    role: str,
    expected_path: Path,
    runtime: Mapping[str, Any],
    size: int,
) -> None:
    if not isinstance(record, Mapping):
        raise TexturedRenderError(f"{role} record is missing")
    if (
        record.get("status") != "complete"
        or record.get("role") != role
        or record.get("engine") != runtime["engine"]
        or record.get("render_backend") != runtime["render_backend"]
        or record.get("render_device") != runtime["render_device"]
        or record.get("width") != size
        or record.get("height") != size
        or not isinstance(record.get("view"), Mapping)
        or not isinstance(record.get("render_settings"), Mapping)
    ):
        raise TexturedRenderError(f"{role} record metadata is malformed")
    if record.get("output_path") != str(expected_path.resolve()):
        raise TexturedRenderError(f"{role} record points at the wrong file")
    metadata = os.lstat(expected_path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TexturedRenderError(f"{role} output is not a regular file")
    output_hash = _sha256_file(expected_path)
    if record.get("output_file_sha256") != output_hash:
        raise TexturedRenderError(f"{role} output hash changed")


def _validate_component_result(
    result: Any,
    request: Mapping[str, Any],
    output_root: Path,
    expected_runtime: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = request["context"]
    if not isinstance(result, Mapping):
        raise TexturedRenderError("component result is not a mapping")
    if result.get("status") == "error":
        error_echo = {
            "component": context["component"],
            "target_segment_key": context["target_segment_key"],
            "component_nonce": context["component_nonce"],
            "request_sha256": request["request_sha256"],
            "renderer_source_sha256": context["renderer_source_sha256"],
        }
        if any(result.get(key) != value for key, value in error_echo.items()):
            raise TexturedRenderError(
                "component error result is stale or mismatched"
            )
        raise TexturedRenderError(
            "component worker failed with "
            f"{result.get('error_type')}: {result.get('error')}"
        )
    if set(result) != {
        "status",
        "component",
        "target_segment_key",
        "component_nonce",
        "request_sha256",
        "renderer_source_sha256",
        "context",
        "runtime",
        "records",
    }:
        raise TexturedRenderError("component result has unexpected fields")
    exact_echo = {
        "status": "complete",
        "component": context["component"],
        "target_segment_key": context["target_segment_key"],
        "component_nonce": context["component_nonce"],
        "request_sha256": request["request_sha256"],
        "renderer_source_sha256": context["renderer_source_sha256"],
        "context": context,
    }
    if any(result.get(key) != value for key, value in exact_echo.items()):
        raise TexturedRenderError("component result context is stale or mismatched")
    runtime = _validate_component_runtime(result.get("runtime"))
    if expected_runtime is not None and runtime != dict(expected_runtime):
        raise TexturedRenderError("component GPU runtime differs from whole render")

    expected_paths = set(context["expected_png_paths"])
    if _component_output_files(output_root) != expected_paths:
        raise TexturedRenderError("component produced an unexpected PNG file set")
    records = result.get("records")
    if not isinstance(records, dict):
        raise TexturedRenderError("component render records are missing")
    size = context["render_size"]
    component = context["component"]
    target_segment_key = context["target_segment_key"]
    if component == "whole":
        if set(records) != {"textured_whole_object"}:
            raise TexturedRenderError("whole component returned the wrong records")
        record = records["textured_whole_object"]
        _validate_component_render_record(
            record,
            role="textured_whole_object",
            expected_path=output_root / "textured_whole_object.png",
            runtime=runtime,
            size=size,
        )
        if (
            record.get("geometry_payload_hashes")
            != context["geometry_payload_hashes"]
            or record.get("material_plan_hashes")
            != context["material_plan_hashes"]
            or record.get("usd_entrypoint_sha256")
            != context["entrypoint_sha256"]
        ):
            raise TexturedRenderError("whole component provenance binding changed")
        return runtime, records

    if set(records) != {
        "textured_selected_part",
        "selected_material_sphere",
    }:
        raise TexturedRenderError("segment component returned the wrong records")
    if not isinstance(target_segment_key, str):
        raise TexturedRenderError("segment component result has no target")
    extraction = request["extraction"]
    segment = extraction["segments"][target_segment_key]
    segment_id = str(segment["segment_id"])
    directory = output_root / "segments" / _safe_component(segment_id)
    selected = records["textured_selected_part"]
    sphere = records["selected_material_sphere"]
    _validate_component_render_record(
        selected,
        role="textured_selected_part",
        expected_path=directory / "textured_selected_part.png",
        runtime=runtime,
        size=size,
    )
    _validate_component_render_record(
        sphere,
        role="selected_material_sphere",
        expected_path=directory / "selected_material_sphere.png",
        runtime=runtime,
        size=size,
    )
    payload_hash = context["geometry_payload_hashes"][target_segment_key]
    plan_hash = context["material_plan_hashes"][target_segment_key]
    material_record_id = segment.get("material_record_id")
    for role, record in (("selected", selected), ("sphere", sphere)):
        if (
            record.get("segment_id") != segment_id
            or record.get("geometry_payload_sha256") != payload_hash
            or record.get("material_record_id") != material_record_id
            or record.get("material_plan_sha256") != plan_hash
        ):
            raise TexturedRenderError(f"{role} component provenance binding changed")
    isolation = selected.get("selected_geometry_isolation")
    _validate_canonical_record(
        isolation,
        "physical_isolation_sha256",
        "selected-render-physical-isolation",
        "selected geometry isolation",
    )
    sampling = sphere.get("selected_segment_swatch_sampling")
    _validate_canonical_record(
        sampling,
        "segment_sampling_plan_sha256",
        "selected-segment-swatch-plan",
        "selected material sphere sampling",
    )
    if (
        sphere.get("conformal_uv_layer") != "SphereConformalUV"
        or sphere.get("segment_sampling_plan_sha256")
        != sampling.get("segment_sampling_plan_sha256")
        or sampling.get("geometry_payload_sha256") != payload_hash
    ):
        raise TexturedRenderError("selected material sphere provenance changed")
    return runtime, records


def _promote_component_files(
    records: dict[str, Any],
    output_root: Path,
    staging_root: Path,
    expected_paths: tuple[str, ...],
) -> dict[str, Any]:
    for relative in expected_paths:
        source = output_root / relative
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination):
            raise TexturedRenderError(
                f"component output collides in staging: {relative}"
            )
        os.replace(source, destination)
        descriptor = os.open(destination, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for record in records.values():
        old_path = Path(record["output_path"])
        try:
            relative = old_path.relative_to(output_root)
        except ValueError as exc:
            raise TexturedRenderError(
                "component record escaped its output directory"
            ) from exc
        record["output_path"] = str((staging_root / relative).resolve())
    return records


def _run_component_process(
    *,
    component: str,
    target_segment_key: str | None,
    scope_keys: tuple[str, ...],
    source_path: Path,
    renderer_source_sha256: str,
    entrypoint_sha256: str,
    extraction: Mapping[str, Any],
    geometry_payloads: Mapping[str, SegmentGeometryPayload],
    plans: Mapping[str, MaterialPlan],
    subdivision_level: int,
    up_axis: str,
    size: int,
    staging_root: Path,
    expected_runtime: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    segments = extraction["segments"]
    scoped_segments = {key: segments[key] for key in scope_keys}
    scoped_extraction = dict(extraction)
    scoped_extraction["segments"] = scoped_segments
    scoped_extraction["materials"] = extraction["materials"]
    scoped_payloads = {key: geometry_payloads[key] for key in scope_keys}
    scoped_plans = {key: plans[key] for key in scope_keys}
    attempt_parent = staging_root.parent
    attempt_prefix = f".{staging_root.name}.{component}.attempt."

    for attempt_number in range(1, 4):
        attempt_directory = _new_private_directory(
            attempt_parent, attempt_prefix
        )
        try:
            output_root = attempt_directory / "output"
            output_root.mkdir(mode=0o700)
            _require_private_directory(output_root)
            request_path = attempt_directory / "request.pkl"
            result_path = attempt_directory / "result.pkl"
            if _renderer_source_sha256() != renderer_source_sha256:
                raise TexturedRenderError(
                    "renderer source changed before component launch"
                )
            nonce = uuid.uuid4().hex
            context = _component_context(
                component=component,
                target_segment_key=target_segment_key,
                nonce=nonce,
                source_path=source_path,
                renderer_source_sha256=renderer_source_sha256,
                entrypoint_sha256=entrypoint_sha256,
                extraction=scoped_extraction,
                geometry_payloads=scoped_payloads,
                plans=scoped_plans,
                subdivision_level=subdivision_level,
                up_axis=up_axis,
                size=size,
            )
            request = {
                "request_sha256": _component_request_hash(context),
                "context": context,
                "usd_path": str(source_path),
                "extraction": scoped_extraction,
                "geometry_payloads": scoped_payloads,
                "output_dir": str(output_root.resolve()),
            }
            _atomic_write_private_pickle(request_path, request)
            child_environment = {
                key: value
                for key, value in os.environ.items()
                if "API_KEY" not in key.upper()
            }
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    (
                        "dataset_toolkits.material_objects.vlm_annotations."
                        "segment_textured_renderer"
                    ),
                    "--component-worker",
                    str(request_path),
                    str(result_path),
                ],
                close_fds=True,
                env=child_environment,
                shell=False,
            )
            try:
                return_code = process.wait(timeout=_COMPONENT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=_COMPONENT_TERMINATION_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise TexturedRenderError(
                    f"{component} component exceeded the 40-minute timeout"
                ) from exc

            if return_code < 0:
                native_signal = -return_code
                if (
                    native_signal in _RETRIABLE_NATIVE_SIGNALS
                    and attempt_number < 3
                ):
                    continue
                signal_name = next(
                    (
                        member.name
                        for member in signal.Signals
                        if member.value == native_signal
                    ),
                    f"signal {native_signal}",
                )
                raise TexturedRenderError(
                    f"{component} component died from {signal_name}; "
                    f"attempt {attempt_number} of 3"
                )
            if return_code != 0:
                if return_code > 0 and attempt_number < 3:
                    continue
                raise TexturedRenderError(
                    f"{component} component exited with code {return_code}; "
                    f"attempt {attempt_number} of 3"
                )
            if not os.path.lexists(result_path):
                raise TexturedRenderError(
                    f"{component} component exited cleanly without a result"
                )
            result = _read_private_pickle(result_path)
            runtime, records = _validate_component_result(
                result,
                request,
                output_root,
                expected_runtime,
            )
            promoted = _promote_component_files(
                records,
                output_root,
                staging_root,
                tuple(context["expected_png_paths"]),
            )
            return runtime, promoted
        finally:
            _remove_generated_directory(
                attempt_directory,
                attempt_parent,
                attempt_prefix,
            )
    raise AssertionError("unreachable component-attempt state")


def _rewrite_result_output_paths(
    rendered: dict[str, Any],
    old_root: Path,
    new_root: Path,
) -> None:
    records = [rendered["asset"]["textured_whole_object"]]
    for segment_records in rendered["segments"].values():
        records.extend(
            (
                segment_records["textured_selected_part"],
                segment_records["selected_material_sphere"],
            )
        )
    for record in records:
        old_path = Path(record["output_path"])
        try:
            relative = old_path.relative_to(old_root)
        except ValueError as exc:
            raise TexturedRenderError(
                "assembled component output escaped the staging tree"
            ) from exc
        record["output_path"] = str((new_root / relative).resolve())


def _validate_final_render_tree(
    rendered: Mapping[str, Any],
    destination_root: Path,
    segments: Mapping[str, Any],
    runtime: Mapping[str, Any],
    size: int,
) -> None:
    expected_paths = {"textured_whole_object.png"}
    for segment in segments.values():
        segment_id = str(segment["segment_id"])
        directory = f"segments/{_safe_component(segment_id)}"
        expected_paths.update(
            {
                f"{directory}/textured_selected_part.png",
                f"{directory}/selected_material_sphere.png",
            }
        )
    if _component_output_files(destination_root) != expected_paths:
        raise TexturedRenderError("published render tree has the wrong file set")
    whole = rendered["asset"]["textured_whole_object"]
    _validate_component_render_record(
        whole,
        role="textured_whole_object",
        expected_path=destination_root / "textured_whole_object.png",
        runtime=runtime,
        size=size,
    )
    for key, segment in segments.items():
        segment_id = str(segment["segment_id"])
        directory = destination_root / "segments" / _safe_component(segment_id)
        segment_records = rendered["segments"][key]
        _validate_component_render_record(
            segment_records["textured_selected_part"],
            role="textured_selected_part",
            expected_path=directory / "textured_selected_part.png",
            runtime=runtime,
            size=size,
        )
        _validate_component_render_record(
            segment_records["selected_material_sphere"],
            role="selected_material_sphere",
            expected_path=directory / "selected_material_sphere.png",
            runtime=runtime,
            size=size,
        )


def render_textured_vlm_inputs(
    usd_path: str | os.PathLike[str],
    extraction: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    geometry_payloads: Mapping[str, SegmentGeometryPayload] | None = None,
    size: int = 512,
) -> dict[str, Any]:
    """Render each GPU component in a fresh process and publish one exact tree."""

    if type(size) is not int or size != 512:
        raise TexturedRenderError(
            "production textured VLM renders must be exactly 512x512"
        )
    source_path = Path(usd_path).expanduser().resolve()
    if not source_path.is_file():
        raise TexturedRenderError(f"USD entrypoint is unreadable: {source_path}")
    if geometry_payloads is None:
        geometry_payloads = build_asset_geometry_payloads(source_path, extraction)
    segments, materials, _ = _validate_payloads(extraction, geometry_payloads)
    if any(not isinstance(key, str) for key in segments):
        raise TexturedRenderError("segment keys must be exact strings")
    subdivision_level = _asset_catmull_subdivision_level(geometry_payloads)
    entrypoint_sha256 = _sha256_file(source_path)
    up_axes = {
        str(payload.provenance.get("stage_up_axis") or "").upper()
        for payload in geometry_payloads.values()
    }
    if len(up_axes) != 1 or not up_axes <= {"Y", "Z"}:
        raise TexturedRenderError(
            f"exact geometry has inconsistent up axes: {sorted(up_axes)!r}"
    )
    up_axis = next(iter(up_axes))
    _perspective_camera_horizontal_hint(extraction, up_axis)
    for key, payload in geometry_payloads.items():
        if payload.provenance.get("usd_entrypoint_sha256") != entrypoint_sha256:
            raise TexturedRenderError(
                f"USD entrypoint changed after payload extraction for {key!r}"
            )
    plans = {
        str(key): _build_material_plan(segments[key], payload, materials)
        for key, payload in geometry_payloads.items()
    }

    destination_root = Path(output_dir).expanduser()
    if not destination_root.is_absolute():
        destination_root = Path.cwd() / destination_root
    destination_root = Path(os.path.abspath(destination_root))
    metadata = os.lstat(destination_root)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise TexturedRenderError("main-created render destination is not a directory")
    if any(destination_root.iterdir()):
        raise TexturedRenderError("main-created render destination is not empty")
    destination_root = destination_root.resolve()
    destination_parent = destination_root.parent
    renderer_source_sha256 = _renderer_source_sha256()
    staging_prefix = f".{destination_root.name}.staging."
    staging_root = _new_private_directory(destination_parent, staging_prefix)
    published = False
    try:
        all_keys = tuple(sorted(segments))
        whole_runtime, whole_records = _run_component_process(
            component="whole",
            target_segment_key=None,
            scope_keys=all_keys,
            source_path=source_path,
            renderer_source_sha256=renderer_source_sha256,
            entrypoint_sha256=entrypoint_sha256,
            extraction=extraction,
            geometry_payloads=geometry_payloads,
            plans=plans,
            subdivision_level=subdivision_level,
            up_axis=up_axis,
            size=size,
            staging_root=staging_root,
            expected_runtime=None,
        )
        segment_records: dict[str, Any] = {}
        for key in all_keys:
            scope_keys = _component_scope_keys(key, geometry_payloads)
            component_runtime, records = _run_component_process(
                component="segment",
                target_segment_key=key,
                scope_keys=scope_keys,
                source_path=source_path,
                renderer_source_sha256=renderer_source_sha256,
                entrypoint_sha256=entrypoint_sha256,
                extraction=extraction,
                geometry_payloads=geometry_payloads,
                plans=plans,
                subdivision_level=subdivision_level,
                up_axis=up_axis,
                size=size,
                staging_root=staging_root,
                expected_runtime=whole_runtime,
            )
            if component_runtime != whole_runtime:
                raise TexturedRenderError(
                    f"segment component runtime changed for {key!r}"
                )
            segment_records[key] = records
        if set(segment_records) != set(segments):
            raise TexturedRenderError("fresh workers did not cover every segment")
        rendered = {
            "status": "complete",
            "engine": whole_runtime["engine"],
            "render_backend": whole_runtime["render_backend"],
            "render_device": list(whole_runtime["render_device"]),
            "asset": {
                "textured_whole_object": whole_records[
                    "textured_whole_object"
                ]
            },
            "segments": segment_records,
        }
        if _renderer_source_sha256() != renderer_source_sha256:
            raise TexturedRenderError("renderer source changed before publication")
        expected_paths = {"textured_whole_object.png"}
        for segment in segments.values():
            segment_id = str(segment["segment_id"])
            directory = f"segments/{_safe_component(segment_id)}"
            expected_paths.update(
                {
                    f"{directory}/textured_selected_part.png",
                    f"{directory}/selected_material_sphere.png",
                }
            )
        if _component_output_files(staging_root) != expected_paths:
            raise TexturedRenderError("assembled staging tree is incomplete")
        _fsync_tree_directories(staging_root)
        current_destination = os.lstat(destination_root)
        if (
            stat.S_ISLNK(current_destination.st_mode)
            or not stat.S_ISDIR(current_destination.st_mode)
            or any(destination_root.iterdir())
        ):
            raise TexturedRenderError(
                "main-created render destination changed before publication"
            )
        os.replace(staging_root, destination_root)
        published = True
        _fsync_directory(destination_parent)
        _rewrite_result_output_paths(rendered, staging_root, destination_root)
        _validate_final_render_tree(
            rendered,
            destination_root,
            segments,
            whole_runtime,
            size,
        )
        return rendered
    except Exception:
        if published and os.path.lexists(destination_root):
            final_metadata = os.lstat(destination_root)
            if stat.S_ISLNK(final_metadata.st_mode) or not stat.S_ISDIR(
                final_metadata.st_mode
            ):
                raise TexturedRenderError(
                    "published render destination changed type during cleanup"
                )
            shutil.rmtree(destination_root)
        raise
    finally:
        if os.path.lexists(staging_root):
            _remove_generated_directory(
                staging_root,
                destination_parent,
                staging_prefix,
            )


def _component_cli() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--component-worker",
        nargs=2,
        metavar=("REQUEST", "RESULT"),
        required=True,
    )
    arguments = parser.parse_args()
    request_path, result_path = (
        Path(value).expanduser().resolve()
        for value in arguments.component_worker
    )
    return _component_worker_entry(request_path, result_path)


__all__ = ["TexturedRenderError", "render_textured_vlm_inputs"]


if __name__ == "__main__":
    raise SystemExit(_component_cli())
