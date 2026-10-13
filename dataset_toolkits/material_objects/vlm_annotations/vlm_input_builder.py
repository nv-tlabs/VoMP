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

"""Pure construction and validation of per-segment VLM request metadata.

Every production request has exactly four visual roles.  Keeping the order,
meaning, and hashes in one small module prevents a caller from silently
downgrading a difficult segment to a thumbnail-only or text-only request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping

from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    format_reference_material_hint,
)

VLM_MAX_YOUNGS_MODULUS_PA = 2.0e12
VLM_MAX_DENSITY_KG_M3 = 3.0e4
VLM_MIN_POISSONS_RATIO = 0.0
VLM_MAX_POISSONS_RATIO = 0.5
VLM_IMAGE_ROLES = (
    "isaac_thumbnail",
    "textured_whole_object",
    "textured_selected_part",
    "selected_material_sphere",
)


@dataclass(frozen=True)
class VLMInputSpec:
    asset_key: str
    segment_id: str
    prompt: str
    text_context_hash: str
    image_roles: tuple[str, ...]
    image_hashes: tuple[str, ...]
    request_hash: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["image_roles"] = list(self.image_roles)
        result["image_hashes"] = list(self.image_hashes)
        return result


def compute_vlm_request_hash(
    asset_key: str,
    segment_id: str,
    prompt: str,
    text_context_hash: str,
    image_roles: tuple[str, ...] | list[str],
    image_hashes: tuple[str, ...] | list[str],
) -> str:
    """Hash the prompt and images to identify a VLM request."""

    digest = hashlib.sha256()
    for value in (
        "vlm-input",
        asset_key,
        segment_id,
        prompt,
        text_context_hash,
        *image_roles,
        *image_hashes,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def compute_vlm_retry_attempt_hash(
    base_request_hash: str,
    missing_properties: tuple[str, ...] | list[str],
) -> str:
    """Identify a retry for the missing material properties."""

    canonical_missing = tuple(sorted(set(missing_properties)))
    digest = hashlib.sha256()
    for value in (
        "vlm-retry-attempt",
        base_request_hash,
        *canonical_missing,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def valid_vlm_property_value(name: str, value: Any) -> bool:
    """Apply conservative finite bounds for persisted bulk-material labels."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    numeric = float(value)
    if not math.isfinite(numeric):
        return False
    if name == "youngs_modulus":
        return 0.0 < numeric <= VLM_MAX_YOUNGS_MODULUS_PA
    if name == "poissons_ratio":
        return VLM_MIN_POISSONS_RATIO <= numeric <= VLM_MAX_POISSONS_RATIO
    if name == "density":
        return 0.0 < numeric <= VLM_MAX_DENSITY_KG_M3
    return False


def valid_vlm_properties(properties: Any) -> bool:
    if not isinstance(properties, Mapping):
        return False
    return all(
        valid_vlm_property_value(name, properties.get(name))
        for name in ("youngs_modulus", "poissons_ratio", "density")
    )


def _quoted_prompt_value(value: Any, *, default: str = "unknown") -> str:
    """Quote a hint without letting its contents alter prompt layout."""

    text = str(value).strip() if value is not None else ""
    return json.dumps(text or default, ensure_ascii=True)


def _geometry_labels(segment_info: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return the extracted part and optional subpart names."""

    mesh_label = str(segment_info.get("geometry_mesh_label") or "").strip()
    subset_value = segment_info.get("geometry_subset_label")
    subset_label = str(subset_value).strip() if subset_value is not None else ""
    if not mesh_label:
        raise ValueError("segment is missing its geometry part name")
    return mesh_label, subset_label or None


def _active_base_color_filenames(
    segment_info: Mapping[str, Any],
) -> tuple[str, ...]:
    """List only filenames for the active albedo sources of the selected material."""

    candidates = segment_info.get("texture_candidates")
    role_records = candidates.get("albedo", []) if isinstance(candidates, Mapping) else []
    filenames = set()
    for record in role_records if isinstance(role_records, list) else []:
        if not isinstance(record, Mapping):
            continue
        authored = str(record.get("authored_asset") or "").strip().replace(
            "\\", "/"
        )
        markers = [
            marker
            for marker in ("<UDIM>", "<udim>", "%04d")
            if marker in authored
        ]
        if record.get("is_udim") is True and len(markers) == 1:
            filename = authored.rsplit("/", 1)[-1]
            stem = filename.rsplit(".", 1)[0] if "." in filename else filename
            semantic_name = stem.replace(markers[0], "").strip(" ._-")
            for separator in (".", "_", "-"):
                while separator * 2 in semantic_name:
                    semantic_name = semantic_name.replace(
                        separator * 2, separator
                    )
            if semantic_name:
                filenames.add(semantic_name)
            continue
        resolved_paths = record.get("resolved_paths")
        path_values = (
            resolved_paths
            if isinstance(resolved_paths, (list, tuple)) and resolved_paths
            else [record.get("authored_asset")]
        )
        for path_value in path_values:
            text = str(path_value or "").strip().replace("\\", "/").rstrip("/")
            if text:
                filenames.add(text.rsplit("/", 1)[-1])
    return tuple(sorted(filenames, key=lambda value: (value.casefold(), value)))


def _text_hint_lines(
    *,
    asset_key: str | None,
    object_name: str | None,
    object_category: str | None,
    material_label: str | None,
    geometry_mesh_label: str | None,
    geometry_subset_label: str | None,
    base_color_filenames: tuple[str, ...] | list[str] = (),
) -> list[str]:
    lines = [f"- Object: {_quoted_prompt_value(object_name)}"]
    category = str(object_category or "").strip()
    if not category or category.casefold() == "unknown":
        parts = str(asset_key or "").replace("\\", "/").split("/")
        ignored = {
            "simready_content",
            "common_assets",
            "props",
            str(object_name or "").casefold(),
        }
        category_parts = [
            part.replace("_", " ")
            for part in parts[1:-1]
            if part and part.casefold() not in ignored
        ]
        category = " > ".join(category_parts)
    if category:
        lines.append(f"- Object type: {_quoted_prompt_value(category)}")
    lines.extend(
        (
            f"- Selected part: {_quoted_prompt_value(geometry_mesh_label)}",
            f"- Material name hint: {_quoted_prompt_value(material_label)}",
        )
    )
    if geometry_subset_label:
        lines.append(
            f"- Subpart: {_quoted_prompt_value(geometry_subset_label)}"
        )
    if base_color_filenames:
        lines.append(
            "- Texture filename hint(s): "
            + json.dumps(list(base_color_filenames), ensure_ascii=True)
        )
    return lines


def _material_prompt(hint_lines, material_label):
    reference_hint = format_reference_material_hint(material_label)
    return f"""You are a materials science expert analyzing exactly four separate images, in this order:
1. ORIGINAL THUMBNAIL: the original preview of the complete object, providing overall identity, context, and visible appearance.
2. TEXTURED WHOLE OBJECT: a fully textured render of the complete object, showing how its parts and materials relate to one another.
3. TEXTURED SELECTED PART: a fully textured render containing only the selected part. This image preserves the selected part's original shape and its original material mapping.
4. MATERIAL SPHERE: a standardized material swatch on a conformally mapped sphere, showing the selected material independently of the part's shape.

Use image 3 to understand both the selected part's real shape and how the material is mapped across it. Image 4 replaces that shape with a standard sphere, so use it to compare material appearance under standardized geometry and never as evidence about the selected part's shape.

The names below are optional clues, not verified facts. A material name may describe a color, finish, coating, or upholstery instead of the part's underlying material, and a part name may be generic or misleading. Use the selected-part shape and whole-object context to decide whether the selected part is a bulk component, a thin surface layer, a cover, or a coating. Do not assign the properties of a visible surface finish to an internal core unless the selected part actually represents that finish.

For all three properties, estimate the effective bulk material behavior of the entire selected segment as it would respond physically, rather than only the behavior of its visible surface finish.

Using the four images and the context below, identify the most likely real-world material and estimate its mechanical properties.

Context hints:
{chr(10).join(hint_lines)}
{reference_hint}
Your task is to provide three specific properties:
1. Young's modulus in Pa using scientific notation
2. Poisson's ratio between 0.0 and 0.5
3. Density in kg/m^3 using scientific notation
"""


def make_common_user_prompt(
    material_type,
    semantic_usage,
    object_name,
    *,
    asset_key=None,
    material_label=None,
    geometry_mesh_label=None,
    geometry_subset_label=None,
    base_color_filenames=(),
):
    hint_lines = _text_hint_lines(
        asset_key=asset_key,
        object_name=object_name,
        object_category=None,
        material_label=material_label or material_type,
        geometry_mesh_label=geometry_mesh_label or semantic_usage,
        geometry_subset_label=geometry_subset_label,
        base_color_filenames=base_color_filenames,
    )
    return _material_prompt(hint_lines, material_label or material_type)


def make_simready_user_prompt(
    material_type,
    opacity,
    semantic_usage,
    object_name=None,
    *,
    asset_key=None,
    object_category=None,
    material_label=None,
    material_usage=None,
    geometry_mesh_label=None,
    geometry_subset_label=None,
    base_color_filenames=(),
):
    hint_lines = _text_hint_lines(
        asset_key=asset_key,
        object_name=object_name,
        object_category=object_category,
        material_label=material_label or material_type,
        geometry_mesh_label=geometry_mesh_label or semantic_usage,
        geometry_subset_label=geometry_subset_label,
        base_color_filenames=base_color_filenames,
    )
    if material_type and str(material_type).casefold() != "unknown":
        hint_lines.append(
            f"- Material type hint: {_quoted_prompt_value(material_type)}"
        )
    if material_usage:
        hint_lines.append(f"- Part/use hint: {_quoted_prompt_value(material_usage)}")
    if opacity and str(opacity).casefold() != "unknown":
        hint_lines.append(f"- Appearance hint: {_quoted_prompt_value(opacity)}")
    return _material_prompt(hint_lines, material_label or material_type)


def build_vlm_input_spec(
    *,
    asset_key: str,
    segment_info: dict[str, Any],
    dataset: str,
    object_name: str,
    isaac_thumbnail_hash: str,
    textured_whole_object_hash: str,
    textured_selected_part_hash: str,
    selected_material_sphere_hash: str,
    text_context_hash: str,
    object_category: str | None = None,
) -> VLMInputSpec:
    """Prepare the prompt and ordered images for a VLM request."""

    geometry_mesh_label, geometry_subset_label = _geometry_labels(segment_info)
    base_color_filenames = _active_base_color_filenames(segment_info)
    material_label = segment_info.get("name") or segment_info.get(
        "material_type", "unknown"
    )
    if dataset == "simready":
        prompt = make_simready_user_prompt(
            segment_info.get("material_type", "unknown"),
            segment_info.get("opacity", "unknown"),
            segment_info.get("semantic_usage", ""),
            object_name=object_name,
            asset_key=asset_key,
            object_category=object_category,
            material_label=material_label,
            material_usage=segment_info.get("material_usage"),
            geometry_mesh_label=geometry_mesh_label,
            geometry_subset_label=geometry_subset_label,
            base_color_filenames=base_color_filenames,
        )
    else:
        prompt = make_common_user_prompt(
            segment_info.get("material_type", "unknown"),
            segment_info.get("semantic_usage", ""),
            object_name,
            asset_key=asset_key,
            material_label=material_label,
            geometry_mesh_label=geometry_mesh_label,
            geometry_subset_label=geometry_subset_label,
            base_color_filenames=base_color_filenames,
        )

    image_roles = list(VLM_IMAGE_ROLES)
    image_hashes = [
        isaac_thumbnail_hash,
        textured_whole_object_hash,
        textured_selected_part_hash,
        selected_material_sphere_hash,
    ]

    segment_id = str(segment_info.get("segment_id", ""))
    request_hash = compute_vlm_request_hash(
        asset_key,
        segment_id,
        prompt,
        text_context_hash,
        image_roles,
        image_hashes,
    )
    return VLMInputSpec(
        asset_key=asset_key,
        segment_id=segment_id,
        prompt=prompt,
        text_context_hash=text_context_hash,
        image_roles=tuple(image_roles),
        image_hashes=tuple(image_hashes),
        request_hash=request_hash,
    )


def validate_vlm_input_spec(spec: VLMInputSpec, object_name: str) -> list[str]:
    """Check the prompt and image order."""

    errors = []
    if not isinstance(spec.asset_key, str) or not spec.asset_key:
        errors.append("asset key is missing")
    if not spec.segment_id:
        errors.append("segment id is missing")
    if (
        not isinstance(spec.text_context_hash, str)
        or len(spec.text_context_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in spec.text_context_hash
        )
    ):
        errors.append("text context does not have a canonical SHA-256 hash")
    if object_name not in spec.prompt:
        errors.append("object identity is missing from prompt")
    for property_name in ("Young's modulus", "Poisson's ratio", "Density"):
        if property_name not in spec.prompt:
            errors.append(f"requested property is missing: {property_name}")
    if tuple(spec.image_roles) != VLM_IMAGE_ROLES:
        errors.append(
            "image roles must be exactly original thumbnail, textured whole object, "
            "textured selected part, then selected material sphere"
        )
    if len(spec.image_roles) != len(spec.image_hashes):
        errors.append("image roles and hashes have different lengths")
    if len(spec.image_hashes) != 4:
        errors.append("every request must contain exactly four image hashes")
    for index, image_hash in enumerate(spec.image_hashes):
        if not isinstance(image_hash, str) or len(image_hash) != 64:
            errors.append(f"image {index + 1} does not have a SHA-256 hash")
            continue
        if any(character not in "0123456789abcdef" for character in image_hash):
            errors.append(f"image {index + 1} hash is not hexadecimal")
    required_prompt_phrases = (
        "exactly four separate images",
        "ORIGINAL THUMBNAIL",
        "TEXTURED WHOLE OBJECT",
        "TEXTURED SELECTED PART",
        "MATERIAL SPHERE",
        "preserves the selected part's original shape",
        "standardized material swatch",
        "never as evidence about the selected part's shape",
        "Material name hint",
    )
    for phrase in required_prompt_phrases:
        if phrase not in spec.prompt:
            errors.append(f"prompt does not explain required visual role: {phrase}")
    request_fields_are_strings = all(
        isinstance(value, str)
        for value in (
            spec.asset_key,
            spec.segment_id,
            spec.prompt,
            spec.text_context_hash,
            *spec.image_roles,
            *spec.image_hashes,
        )
    )
    if request_fields_are_strings:
        expected_request_hash = compute_vlm_request_hash(
            spec.asset_key,
            spec.segment_id,
            spec.prompt,
            spec.text_context_hash,
            spec.image_roles,
            spec.image_hashes,
        )
        if spec.request_hash != expected_request_hash:
            errors.append("request hash does not match the exact VLM input")
    else:
        errors.append("request hash inputs must all be strings")
    return errors
