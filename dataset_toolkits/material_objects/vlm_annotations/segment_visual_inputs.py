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

"""Load USD geometry and material segments for the VLM renderer.

Use OpenUSD to resolve transforms, instances, and material assignments before
passing the geometry to Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from dataset_toolkits.material_objects.vlm_annotations.usd_segment_reader import (
    material_record_id,
    decode_face_indices,
    enumerate_material_regions,
    stage_composition_errors,
)
from pxr import Sdf, Usd, UsdGeom


class VisualInputError(RuntimeError):
    """Raised when a VLM visual input cannot be represented faithfully."""


def _sha256_file(path: str | os.PathLike[str], chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_values(label: str, *values: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8") + b"\0")
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_array(array: Any, dtype: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.dtype(dtype))
    if not result.flags.c_contiguous:
        result = np.ascontiguousarray(result)
    return result


def _array_sha256(label: str, array: np.ndarray) -> str:
    """Hash an array using little-endian values and its shape."""

    canonical = np.ascontiguousarray(array)
    if canonical.dtype.kind == "f":
        canonical = canonical.astype("<f8", copy=False)
    elif canonical.dtype.kind in {"i", "u"}:
        canonical = canonical.astype("<i8", copy=False)
    elif canonical.dtype.kind == "b":
        canonical = canonical.astype("u1", copy=False)
    else:
        raise TypeError(f"Unsupported array dtype for hashing: {canonical.dtype}")
    digest = hashlib.sha256(label.encode("utf-8") + b"\0")
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _matrix_array(matrix: Any) -> np.ndarray:
    values = np.asarray(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype="<f8",
    )
    if values.shape != (4, 4) or not np.isfinite(values).all():
        raise VisualInputError("USD transform is not a finite 4x4 matrix")
    return values


def _reader_transform_hash(transforms: Sequence[Any], indices: Iterable[int]) -> str:
    """Reproduce the segment reader's PointInstancer transform hash."""

    digest = hashlib.sha256(b"vlm-point-instancer-transforms-v1\0")
    for index in indices:
        matrix = transforms[int(index)]
        for row in range(4):
            for column in range(4):
                digest.update(struct.pack("<d", float(matrix[row][column])))
    return digest.hexdigest()


def _json_canonical_hash(label: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _sha256_values(label, encoded)


@dataclass(frozen=True)
class CornerAttribute:
    """One authored USD attribute expanded to selected polygon corners."""

    name: str
    property_path: str
    source_prim_path: str
    value_type: str
    interpolation: str
    element_size: int
    corner_values: np.ndarray
    source_values_sha256: str
    source_indices_sha256: str | None
    corner_values_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "property_path": self.property_path,
            "source_prim_path": self.source_prim_path,
            "value_type": self.value_type,
            "interpolation": self.interpolation,
            "element_size": self.element_size,
            "corner_value_count": int(len(self.corner_values)),
            "corner_value_shape": list(self.corner_values.shape),
            "source_values_sha256": self.source_values_sha256,
            "source_indices_sha256": self.source_indices_sha256,
            "corner_values_sha256": self.corner_values_sha256,
        }


@dataclass
class SegmentGeometryPayload:
    """Exact selected polygons plus transforms, ready for the GPU renderer."""

    points: np.ndarray
    source_vertex_indices: np.ndarray
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray
    source_face_indices: np.ndarray
    source_corner_indices: np.ndarray
    occurrence_transforms: np.ndarray
    subdivision_control_points: np.ndarray | None = None
    subdivision_control_face_vertex_counts: np.ndarray | None = None
    subdivision_control_face_vertex_indices: np.ndarray | None = None
    subdivision_control_source_face_indices: np.ndarray | None = None
    uv_sets: dict[str, CornerAttribute] = field(default_factory=dict)
    normals: CornerAttribute | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        arrays = {
            "points_sha256": _array_sha256("points-v1", self.points),
            "source_vertex_indices_sha256": _array_sha256(
                "source-vertex-indices-v1", self.source_vertex_indices
            ),
            "face_vertex_counts_sha256": _array_sha256(
                "face-vertex-counts-v1", self.face_vertex_counts
            ),
            "face_vertex_indices_sha256": _array_sha256(
                "face-vertex-indices-v1", self.face_vertex_indices
            ),
            "source_face_indices_sha256": _array_sha256(
                "source-face-indices-v1", self.source_face_indices
            ),
            "source_corner_indices_sha256": _array_sha256(
                "source-corner-indices-v1", self.source_corner_indices
            ),
            "occurrence_transforms_sha256": _array_sha256(
                "occurrence-transforms-v1", self.occurrence_transforms
            ),
        }
        subdivision_control = None
        control_arrays = (
            self.subdivision_control_points,
            self.subdivision_control_face_vertex_counts,
            self.subdivision_control_face_vertex_indices,
            self.subdivision_control_source_face_indices,
        )
        if any(value is not None for value in control_arrays):
            if not all(value is not None for value in control_arrays):
                raise VisualInputError(
                    "Subdivision control surface arrays are only partially populated"
                )
            control_points = self.subdivision_control_points
            control_counts = self.subdivision_control_face_vertex_counts
            control_indices = self.subdivision_control_face_vertex_indices
            control_faces = self.subdivision_control_source_face_indices
            assert control_points is not None
            assert control_counts is not None
            assert control_indices is not None
            assert control_faces is not None
            subdivision_control = {
                "point_count": int(len(control_points)),
                "polygon_count": int(len(control_counts)),
                "corner_count": int(len(control_indices)),
                "points_sha256": _array_sha256(
                    "subdivision-control-points-v1", control_points
                ),
                "face_vertex_counts_sha256": _array_sha256(
                    "subdivision-control-face-counts-v1", control_counts
                ),
                "face_vertex_indices_sha256": _array_sha256(
                    "subdivision-control-face-indices-v1", control_indices
                ),
                "source_face_indices_sha256": _array_sha256(
                    "subdivision-control-source-faces-v1", control_faces
                ),
            }
        manifest = {
            "point_count": int(len(self.points)),
            "polygon_count": int(len(self.face_vertex_counts)),
            "corner_count": int(len(self.face_vertex_indices)),
            "occurrence_count": int(len(self.occurrence_transforms)),
            **arrays,
            "subdivision_control_surface": subdivision_control,
            "uv_sets": {
                name: value.manifest() for name, value in sorted(self.uv_sets.items())
            },
            "normals": self.normals.manifest() if self.normals else None,
            "provenance": self.provenance,
        }
        manifest["payload_sha256"] = _json_canonical_hash(
            "segment-geometry-payload-v2", manifest
        )
        return manifest

def _fresh_material_binding_attestation(region: Any) -> dict[str, Any]:
    """Check the material binding for a region against the loaded USD stage."""

    bindings = region.material_bindings
    selected_purpose = region.selected_binding_purpose
    selected_record = (
        bindings.get(selected_purpose)
        if isinstance(bindings, Mapping) and isinstance(selected_purpose, str)
        else None
    )
    record = {
        "segment_id": region.segment_id,
        "material_record_id": material_record_id(region.material),
        "material_path": region.material_path,
        "source_material_path": region.source_material_path,
        "binding_relationship": region.binding_relationship,
        "selected_binding_purpose": selected_purpose,
        "selected_binding_record": selected_record,
        "material_bindings_sha256": _json_canonical_hash(
            "fresh-region-material-bindings-v1", bindings
        ),
        "is_bound": region.material is not None,
    }
    record["attestation_sha256"] = _json_canonical_hash(
        "fresh-segment-material-binding-attestation-v1", record
    )
    return record


def _match_fresh_region(
    stage: Any,
    segment_info: Mapping[str, Any],
    *,
    inventory: Any | None = None,
    composition_validated: bool = False,
) -> Any:
    """Prove the serialized segment still names one exact composed region."""

    if not composition_validated:
        composition_errors = stage_composition_errors(stage)
        if composition_errors:
            raise VisualInputError(
                "USD composition is incomplete: " + "; ".join(composition_errors)
            )
    segment_id = segment_info.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id:
        raise VisualInputError("Segment metadata has no segment_id")
    encoded_faces = segment_info.get("face_indices")
    try:
        serialized_faces = decode_face_indices(encoded_faces)
    except Exception as exc:
        raise VisualInputError(
            "Serialized face indices are invalid: "
            f"{type(exc).__name__}: {exc}; value={encoded_faces!r}"
        ) from exc

    inventory = inventory or enumerate_material_regions(stage)
    inventory_errors = getattr(inventory, "errors", [])
    if inventory_errors:
        raise VisualInputError(
            "Fresh composed segment inventory contains hard errors: "
            + "; ".join(str(value) for value in inventory_errors)
        )
    matches = [region for region in inventory.regions if region.segment_id == segment_id]
    if len(matches) != 1:
        raise VisualInputError(
            f"Segment ID resolves to {len(matches)} composed regions, expected one: "
            f"{segment_id}"
        )
    region = matches[0]
    identity_checks = {
        "prim_path": region.mesh_path,
        "source_prim_path": region.source_mesh_path,
        "source_prim_path_kind": region.source_mesh_path_kind,
        "source_prim_prototype_relative_path": (
            region.source_mesh_prototype_relative_path
        ),
        "subset_path": region.subset_path,
        "region_kind": region.region_kind,
        "face_index_hash": region.face_index_hash,
        "material_record_id": material_record_id(region.material),
        "material_path": region.material_path,
        "source_material_path": region.source_material_path,
        "binding_relationship": region.binding_relationship,
        "selected_binding_purpose": region.selected_binding_purpose,
        "material_bindings": region.material_bindings,
        "display_primvars": region.display_primvars,
    }
    mismatches = {
        key: {"serialized": segment_info.get(key), "composed": value}
        for key, value in identity_checks.items()
        if segment_info.get(key) != value
    }
    if tuple(serialized_faces) != tuple(region.face_indices):
        mismatches["face_indices"] = {
            "serialized_count": len(serialized_faces),
            "composed_count": len(region.face_indices),
        }
    if mismatches:
        raise VisualInputError(
            "Serialized segment identity differs from composed USD: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return region


def _flatten_values(value: Any) -> np.ndarray:
    """Convert numeric USD scalar/vector arrays to a finite 2-D array."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VisualInputError(
            f"USD attribute has unsupported numeric values: {exc}"
        ) from exc
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise VisualInputError(
            f"USD attribute has unsupported value shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise VisualInputError("USD attribute contains non-finite values")
    return array.astype("<f8", copy=False)


def _corner_lookup(
    interpolation: str,
    *,
    source_face_indices: np.ndarray,
    selected_face_counts: np.ndarray,
    source_vertex_indices_by_corner: np.ndarray,
    source_corner_indices: np.ndarray,
) -> np.ndarray:
    token = str(interpolation or "constant").casefold()
    if token == "constant":
        return np.zeros(len(source_corner_indices), dtype="<i8")
    if token == "uniform":
        return np.repeat(source_face_indices, selected_face_counts).astype(
            "<i8", copy=False
        )
    if token in {"vertex", "varying"}:
        return source_vertex_indices_by_corner.astype("<i8", copy=False)
    if token == "facevarying":
        return source_corner_indices.astype("<i8", copy=False)
    raise VisualInputError(f"Unsupported USD interpolation: {interpolation!r}")


def _partition_renderable_faces(
    source_face_counts: np.ndarray,
    region_face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a region into polygonal and non-renderable source faces.

    OpenUSD can faithfully carry zero-, one-, and two-corner face records, but
    surface renderers cannot rasterize them as polygons.  They remain part of
    the exact serialized segment identity and are attested separately; only
    faces with at least three corners enter the polygon render arrays.
    """

    counts = np.asarray(source_face_counts, dtype="<i8")
    faces = np.asarray(region_face_indices, dtype="<i8")
    if counts.ndim != 1 or faces.ndim != 1:
        raise VisualInputError("Face partition inputs must be one-dimensional")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(counts)):
        raise VisualInputError("Region references an out-of-range source face")
    renderable_mask = counts[faces] >= 3
    return faces[renderable_mask], faces[~renderable_mask]


def _make_corner_attribute(
    *,
    name: str,
    property_path: str,
    source_prim_path: str,
    value_type: str,
    interpolation: str,
    element_size: int,
    flattened_values: Any,
    authored_values: Any,
    authored_indices: Any,
    source_face_indices: np.ndarray,
    selected_face_counts: np.ndarray,
    source_vertex_indices_by_corner: np.ndarray,
    source_corner_indices: np.ndarray,
) -> CornerAttribute:
    values = _flatten_values(flattened_values)
    lookup = _corner_lookup(
        interpolation,
        source_face_indices=source_face_indices,
        selected_face_counts=selected_face_counts,
        source_vertex_indices_by_corner=source_vertex_indices_by_corner,
        source_corner_indices=source_corner_indices,
    )
    if lookup.size and (int(lookup.min()) < 0 or int(lookup.max()) >= len(values)):
        raise VisualInputError(
            f"Attribute {property_path} has {len(values)} flattened values but "
            f"selected corner lookup reaches {int(lookup.max())}"
        )
    corner_values = values[lookup] if lookup.size else values[:0]
    authored_array = _flatten_values(authored_values)
    indices_array = (
        _canonical_array(authored_indices, "<i8")
        if authored_indices is not None
        else None
    )
    return CornerAttribute(
        name=name,
        property_path=property_path,
        source_prim_path=source_prim_path,
        value_type=value_type,
        interpolation=str(interpolation),
        element_size=int(element_size),
        corner_values=corner_values,
        source_values_sha256=_array_sha256(
            f"{property_path}-authored-values-v1", authored_array
        ),
        source_indices_sha256=(
            _array_sha256(f"{property_path}-authored-indices-v1", indices_array)
            if indices_array is not None
            else None
        ),
        corner_values_sha256=_array_sha256(
            f"{property_path}-selected-corners-v1", corner_values
        ),
    )


def _usd_primvar_payload(
    primvar: Any,
    *,
    source_face_indices: np.ndarray,
    selected_face_counts: np.ndarray,
    source_vertex_indices_by_corner: np.ndarray,
    source_corner_indices: np.ndarray,
) -> CornerAttribute:
    attribute = primvar.GetAttr()
    try:
        flattened = primvar.ComputeFlattened(Usd.TimeCode.Default())
    except TypeError:
        flattened = primvar.ComputeFlattened()
    try:
        authored_indices = primvar.GetIndices(Usd.TimeCode.Default())
    except TypeError:
        authored_indices = primvar.GetIndices()
    return _make_corner_attribute(
        name=str(primvar.GetPrimvarName()),
        property_path=str(attribute.GetPath()),
        source_prim_path=str(attribute.GetPrim().GetPath()),
        value_type=str(attribute.GetTypeName()),
        interpolation=str(primvar.GetInterpolation() or "constant"),
        element_size=int(primvar.GetElementSize() or 1),
        flattened_values=flattened,
        authored_values=attribute.Get(Usd.TimeCode.Default()),
        authored_indices=authored_indices,
        source_face_indices=source_face_indices,
        selected_face_counts=selected_face_counts,
        source_vertex_indices_by_corner=source_vertex_indices_by_corner,
        source_corner_indices=source_corner_indices,
    )


def _is_texcoord_primvar(primvar: Any) -> bool:
    """Recognize authored 2-D texture-coordinate primvars by type role/shape."""

    try:
        type_name = primvar.GetTypeName()
        role = str(getattr(type_name, "role", "") or "").casefold()
        type_text = str(type_name).casefold()
    except Exception:
        return False
    if "texcoord2" in type_text or role == "texturecoordinate":
        return True
    name = str(primvar.GetPrimvarName()).casefold()
    return name in {"st", "uv", "uvmap", "texcoord", "texcoords"} and (
        "2f" in type_text or "float2" in type_text or "half2" in type_text
    )


def _capture_uv_sets(
    mesh_prim: Any,
    *,
    source_face_indices: np.ndarray,
    selected_face_counts: np.ndarray,
    source_vertex_indices_by_corner: np.ndarray,
    source_corner_indices: np.ndarray,
) -> tuple[dict[str, CornerAttribute], list[dict[str, Any]]]:
    try:
        primvars = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvarsWithValues()
    except Exception:
        primvars = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvars()
    result: dict[str, CornerAttribute] = {}
    rejected: list[dict[str, Any]] = []
    for primvar in sorted(primvars, key=lambda value: str(value.GetPrimvarName())):
        if not _is_texcoord_primvar(primvar):
            continue
        try:
            payload = _usd_primvar_payload(
                primvar,
                source_face_indices=source_face_indices,
                selected_face_counts=selected_face_counts,
                source_vertex_indices_by_corner=source_vertex_indices_by_corner,
                source_corner_indices=source_corner_indices,
            )
        except VisualInputError as exc:
            attribute = primvar.GetAttr()
            authored_values = attribute.Get()
            authored_indices = primvar.GetIndices()
            values_hash = None
            indices_hash = None
            try:
                values_array = np.asarray(authored_values)
                values_hash = _array_sha256(
                    f"{attribute.GetPath()}-rejected-values-v1", values_array
                )
            except Exception:
                pass
            try:
                if authored_indices is not None:
                    indices_hash = _array_sha256(
                        f"{attribute.GetPath()}-rejected-indices-v1",
                        np.asarray(authored_indices),
                    )
            except Exception:
                pass
            rejected.append(
                {
                    "attribute_kind": "texture_coordinate_primvar",
                    "name": str(primvar.GetPrimvarName()),
                    "property_path": str(attribute.GetPath()),
                    "source_prim_path": str(attribute.GetPrim().GetPath()),
                    "value_type": str(attribute.GetTypeName()),
                    "interpolation": str(primvar.GetInterpolation() or "constant"),
                    "element_size": int(primvar.GetElementSize() or 1),
                    "authored_value_count": (
                        len(authored_values)
                        if hasattr(authored_values, "__len__")
                        else None
                    ),
                    "authored_index_count": (
                        len(authored_indices)
                        if authored_indices is not None
                        and hasattr(authored_indices, "__len__")
                        else None
                    ),
                    "authored_values_sha256": values_hash,
                    "authored_indices_sha256": indices_hash,
                    "rejection_reason": str(exc),
                }
            )
            continue
        if payload.corner_values.shape[1] != 2:
            rejected.append(
                {
                    "attribute_kind": "texture_coordinate_primvar",
                    "name": payload.name,
                    "property_path": payload.property_path,
                    "source_prim_path": payload.source_prim_path,
                    "value_type": payload.value_type,
                    "interpolation": payload.interpolation,
                    "element_size": payload.element_size,
                    "authored_values_sha256": payload.source_values_sha256,
                    "authored_indices_sha256": payload.source_indices_sha256,
                    "rejection_reason": (
                        "selected flattened texture coordinate is not 2-D: "
                        f"{payload.corner_values.shape}"
                    ),
                }
            )
            continue
        if payload.name in result:
            raise VisualInputError(
                f"Multiple inherited texture coordinate primvars named {payload.name!r}"
            )
        result[payload.name] = payload
    return result, rejected


def _capture_normals(
    mesh: Any,
    *,
    source_face_indices: np.ndarray,
    selected_face_counts: np.ndarray,
    source_vertex_indices_by_corner: np.ndarray,
    source_corner_indices: np.ndarray,
) -> CornerAttribute | None:
    attribute = mesh.GetNormalsAttr()
    values = attribute.Get(Usd.TimeCode.Default()) if attribute else None
    if values is None or len(values) == 0:
        return None
    interpolation = str(mesh.GetNormalsInterpolation() or "vertex")
    payload = _make_corner_attribute(
        name="normals",
        property_path=str(attribute.GetPath()),
        source_prim_path=str(attribute.GetPrim().GetPath()),
        value_type=str(attribute.GetTypeName()),
        interpolation=interpolation,
        element_size=1,
        flattened_values=values,
        authored_values=values,
        authored_indices=None,
        source_face_indices=source_face_indices,
        selected_face_counts=selected_face_counts,
        source_vertex_indices_by_corner=source_vertex_indices_by_corner,
        source_corner_indices=source_corner_indices,
    )
    if payload.corner_values.shape[1] != 3:
        raise VisualInputError(
            f"Normals {payload.property_path} are not 3-D: {payload.corner_values.shape}"
        )
    return payload


def _point_instancer_transforms(
    stage: Any,
    mesh_prim: Any,
    occurrences: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute and validate every selected PointInstancer occurrence matrix.

    OpenUSD documents these matrices as PointInstancer-parent-relative.  The
    prototype-root transform is included by the API, while the selected mesh's
    transform below that root is composed explicitly.  The instancer parent's
    local-to-world matrix supplies the final component.
    """

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    final_matrices: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    mesh_path = mesh_prim.GetPath()
    for serialized in occurrences:
        instancer_path = str(serialized.get("instancer_path") or "")
        target_path_text = str(serialized.get("prototype_target") or "")
        occurrence_indices = serialized.get("occurrence_indices")
        if not instancer_path or not target_path_text:
            raise VisualInputError(
                "PointInstancer occurrence lacks instancer/prototype identity"
            )
        if not isinstance(occurrence_indices, list) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in occurrence_indices
        ):
            raise VisualInputError(
                f"PointInstancer occurrence indices are invalid at {instancer_path}"
            )
        instancer_prim = stage.GetPrimAtPath(instancer_path)
        target_path = Sdf.Path(target_path_text)
        target_prim = stage.GetPrimAtPath(target_path)
        if not instancer_prim or not instancer_prim.IsA(UsdGeom.PointInstancer):
            raise VisualInputError(
                f"PointInstancer prim is missing or wrong type: {instancer_path}"
            )
        if not target_prim or not mesh_path.HasPrefix(target_path):
            raise VisualInputError(
                f"Mesh {mesh_path} is not below prototype {target_path}"
            )
        instancer = UsdGeom.PointInstancer(instancer_prim)
        targets = list(instancer.GetPrototypesRel().GetTargets())
        prototype_index = int(serialized.get("prototype_index", -1))
        if (
            prototype_index < 0
            or prototype_index >= len(targets)
            or targets[prototype_index] != target_path
        ):
            raise VisualInputError(
                f"Prototype slot changed at {instancer_path}: {prototype_index}"
            )
        proto_indices_value = instancer.GetProtoIndicesAttr().Get(
            Usd.TimeCode.Default()
        )
        proto_indices = tuple(int(value) for value in (proto_indices_value or ()))
        try:
            mask_value = tuple(
                bool(value)
                for value in instancer.ComputeMaskAtTime(Usd.TimeCode.Default())
            )
        except Exception as exc:
            raise VisualInputError(
                f"PointInstancer mask query failed at {instancer_path}: {exc}"
            ) from exc
        active_mask = (
            mask_value
            if len(mask_value) == len(proto_indices)
            else (True,) * len(proto_indices)
        )
        expected_indices = [
            index
            for index, value in enumerate(proto_indices)
            if value == prototype_index and active_mask[index]
        ]
        if occurrence_indices != expected_indices:
            raise VisualInputError(
                f"PointInstancer occurrence set changed at {instancer_path}"
            )
        transforms = tuple(
            instancer.ComputeInstanceTransformsAtTime(
                Usd.TimeCode.Default(),
                Usd.TimeCode.Default(),
                UsdGeom.PointInstancer.IncludeProtoXform,
                UsdGeom.PointInstancer.IgnoreMask,
            )
        )
        if len(transforms) != len(proto_indices):
            raise VisualInputError(
                f"PointInstancer transform count mismatch at {instancer_path}"
            )
        raw_hash = _reader_transform_hash(transforms, occurrence_indices)
        if serialized.get("transform_hash") != raw_hash:
            raise VisualInputError(
                f"PointInstancer transform hash changed at {instancer_path}"
            )

        try:
            mesh_to_target, resets_stack = xform_cache.ComputeRelativeTransform(
                mesh_prim, target_prim
            )
        except Exception as exc:
            raise VisualInputError(
                f"Could not compose mesh/prototype transform at {mesh_path}: {exc}"
            ) from exc
        if resets_stack:
            raise VisualInputError(
                f"Mesh {mesh_path} resets its transform below PointInstancer "
                f"prototype {target_path}"
            )
        parent_prim = instancer_prim.GetParent()
        parent_to_world = xform_cache.GetLocalToWorldTransform(parent_prim)
        record_start = len(final_matrices)
        for occurrence_index in occurrence_indices:
            # Gf matrices use row-vector composition: the left-most transform
            # is applied first by GfMatrix4d.Transform().
            final = mesh_to_target * transforms[occurrence_index] * parent_to_world
            final_matrices.append(_matrix_array(final))
        provenance.append(
            {
                "instancer_path": instancer_path,
                "prototype_target": target_path_text,
                "prototype_index": prototype_index,
                "occurrence_indices": occurrence_indices,
                "occurrence_index_hash": serialized.get("occurrence_index_hash"),
                "instance_ids": serialized.get("instance_ids"),
                "instance_id_hash": serialized.get("instance_id_hash"),
                "reader_transform_hash": raw_hash,
                "flattened_transform_begin": record_start,
                "flattened_transform_count": len(occurrence_indices),
            }
        )
    if not final_matrices:
        raise VisualInputError(
            f"PointInstancer-backed segment has no active occurrences: {mesh_path}"
        )
    return np.stack(final_matrices).astype("<f8", copy=False), provenance


def _used_layer_records(stage: Any) -> list[dict[str, Any]]:
    persistent_layer_ids = {
        str(layer.identifier) for layer in stage.GetLayerStack(False)
    }
    session_layer_ids = {
        str(layer.identifier) for layer in stage.GetLayerStack(True)
    } - persistent_layer_ids
    records = []
    layers = (
        layer
        for layer in stage.GetUsedLayers()
        if str(layer.identifier) not in session_layer_ids
    )
    for layer in sorted(layers, key=lambda value: str(value.identifier)):
        real_path = str(getattr(layer, "realPath", "") or "")
        records.append(
            {
                "identifier": str(layer.identifier),
                "real_path": real_path or None,
                "content_sha256": (
                    _sha256_file(real_path)
                    if real_path and Path(real_path).is_file()
                    else None
                ),
            }
        )
    return records


def build_segment_geometry_payload(
    usd_path: str | os.PathLike[str],
    segment_info: Mapping[str, Any],
    *,
    _stage: Any | None = None,
    _inventory: Any | None = None,
    _used_layers: list[dict[str, Any]] | None = None,
    _composition_validated: bool = False,
) -> SegmentGeometryPayload:
    """Build exact composed polygon data for one serialized material region."""

    source_path = Path(usd_path)
    if not source_path.is_file():
        raise VisualInputError(f"USD entry point is missing: {source_path}")
    stage = _stage
    if stage is None:
        try:
            stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
        except Exception as exc:
            raise VisualInputError(
                f"Could not open USD stage: {type(exc).__name__}: {exc}"
            ) from exc
        if not stage:
            raise VisualInputError("Usd.Stage.Open returned no stage")
    region = _match_fresh_region(
        stage,
        segment_info,
        inventory=_inventory,
        composition_validated=_composition_validated,
    )
    fresh_material_binding = _fresh_material_binding_attestation(region)
    mesh_prim = stage.GetPrimAtPath(region.mesh_path)
    if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
        raise VisualInputError(
            f"Composed segment prim is not a Mesh: {region.mesh_path}"
        )
    mesh = UsdGeom.Mesh(mesh_prim)
    points_value = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    counts_value = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices_value = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if points_value is None or counts_value is None or indices_value is None:
        raise VisualInputError(f"Mesh topology is unreadable: {region.mesh_path}")
    source_points = np.asarray(points_value, dtype="<f8")
    source_counts = np.asarray(counts_value, dtype="<i8")
    source_indices = np.asarray(indices_value, dtype="<i8")
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise VisualInputError(
            f"Mesh points have invalid shape at {region.mesh_path}: {source_points.shape}"
        )
    if not np.isfinite(source_points).all():
        raise VisualInputError(f"Mesh points are non-finite: {region.mesh_path}")
    if source_counts.ndim != 1 or source_indices.ndim != 1:
        raise VisualInputError(f"Mesh topology arrays are not 1-D: {region.mesh_path}")
    if int(source_counts.sum()) != len(source_indices):
        raise VisualInputError(
            f"Mesh face counts/indices disagree: {region.mesh_path}"
        )
    region_face_indices = np.asarray(region.face_indices, dtype="<i8")
    if region_face_indices.size == 0:
        raise VisualInputError(f"Segment contains no renderable faces: {region.key}")
    source_face_indices, omitted_degenerate_faces = _partition_renderable_faces(
        source_counts, region_face_indices
    )
    if source_face_indices.size == 0:
        raise VisualInputError(
            "Every source face in the exact segment has fewer than three corners; "
            f"cannot produce a polygon render without dropping the segment: {region.key}"
        )
    offsets = np.concatenate(
        (np.asarray([0], dtype="<i8"), np.cumsum(source_counts, dtype="<i8"))
    )
    selected_counts = source_counts[source_face_indices]
    source_corner_parts = [
        np.arange(offsets[index], offsets[index + 1], dtype="<i8")
        for index in source_face_indices
    ]
    source_corner_indices = np.concatenate(source_corner_parts)
    source_vertex_indices_by_corner = source_indices[source_corner_indices]
    if source_vertex_indices_by_corner.size and (
        int(source_vertex_indices_by_corner.min()) < 0
        or int(source_vertex_indices_by_corner.max()) >= len(source_points)
    ):
        raise VisualInputError(
            f"Segment topology references an invalid point: {region.mesh_path}"
        )
    used_vertices = np.unique(source_vertex_indices_by_corner)
    remap = np.full(len(source_points), -1, dtype="<i8")
    remap[used_vertices] = np.arange(len(used_vertices), dtype="<i8")
    selected_indices = remap[source_vertex_indices_by_corner]
    selected_points = source_points[used_vertices]

    uv_sets, rejected_auxiliary_attributes = _capture_uv_sets(
        mesh_prim,
        source_face_indices=source_face_indices,
        selected_face_counts=selected_counts,
        source_vertex_indices_by_corner=source_vertex_indices_by_corner,
        source_corner_indices=source_corner_indices,
    )
    try:
        normals = _capture_normals(
            mesh,
            source_face_indices=source_face_indices,
            selected_face_counts=selected_counts,
            source_vertex_indices_by_corner=source_vertex_indices_by_corner,
            source_corner_indices=source_corner_indices,
        )
    except VisualInputError as exc:
        normals = None
        normal_attribute = mesh.GetNormalsAttr()
        normal_values = normal_attribute.Get(Usd.TimeCode.Default())
        normal_values_hash = None
        try:
            normal_values_hash = _array_sha256(
                f"{normal_attribute.GetPath()}-rejected-values-v1",
                np.asarray(normal_values),
            )
        except Exception:
            pass
        rejected_auxiliary_attributes.append(
            {
                "attribute_kind": "authored_normals",
                "name": "normals",
                "property_path": str(normal_attribute.GetPath()),
                "source_prim_path": str(normal_attribute.GetPrim().GetPath()),
                "value_type": str(normal_attribute.GetTypeName()),
                "interpolation": str(mesh.GetNormalsInterpolation() or "vertex"),
                "authored_value_count": (
                    len(normal_values)
                    if hasattr(normal_values, "__len__")
                    else None
                ),
                "authored_values_sha256": normal_values_hash,
                "rejection_reason": str(exc),
            }
        )

    occurrence_records = segment_info.get("point_instancer_occurrences")
    if not isinstance(occurrence_records, list):
        raise VisualInputError("point_instancer_occurrences is not a list")
    if occurrence_records:
        occurrence_transforms, point_instancer_provenance = (
            _point_instancer_transforms(
                stage, mesh_prim, occurrence_records
            )
        )
        transform_kind = "point_instancer_occurrences"
    else:
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_transform = xform_cache.GetLocalToWorldTransform(mesh_prim)
        occurrence_transforms = _matrix_array(world_transform)[None, ...]
        point_instancer_provenance = []
        transform_kind = (
            "native_instance_proxy_world"
            if bool(mesh_prim.IsInstanceProxy())
            else "composed_mesh_world"
        )

    used_layers = _used_layers if _used_layers is not None else _used_layer_records(stage)
    hole_values = mesh.GetHoleIndicesAttr().Get(Usd.TimeCode.Default())
    hole_indices = np.asarray(hole_values or (), dtype="<i8")
    crease_lengths = np.asarray(
        mesh.GetCreaseLengthsAttr().Get(Usd.TimeCode.Default()) or (), dtype="<i8"
    )
    crease_indices = np.asarray(
        mesh.GetCreaseIndicesAttr().Get(Usd.TimeCode.Default()) or (), dtype="<i8"
    )
    crease_sharpnesses = np.asarray(
        mesh.GetCreaseSharpnessesAttr().Get(Usd.TimeCode.Default()) or (),
        dtype="<f8",
    )
    corner_indices = np.asarray(
        mesh.GetCornerIndicesAttr().Get(Usd.TimeCode.Default()) or (), dtype="<i8"
    )
    corner_sharpnesses = np.asarray(
        mesh.GetCornerSharpnessesAttr().Get(Usd.TimeCode.Default()) or (),
        dtype="<f8",
    )
    selected_hole_overlap = np.intersect1d(
        source_face_indices, hole_indices, assume_unique=False
    )
    if len(selected_hole_overlap):
        raise VisualInputError(
            "Selected renderable segment unexpectedly contains authored hole faces: "
            f"{selected_hole_overlap[:10].tolist()}"
        )
    subdivision_scheme = str(
        mesh.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.catmullClark
    )
    subdivision_control_points = None
    subdivision_control_counts = None
    subdivision_control_indices = None
    subdivision_control_source_faces = None
    if subdivision_scheme == str(UsdGeom.Tokens.catmullClark):
        all_source_faces = np.arange(len(source_counts), dtype="<i8")
        non_hole_mask = ~np.isin(all_source_faces, hole_indices)
        subdivision_control_source_faces = all_source_faces[non_hole_mask]
        degenerate_control_faces = subdivision_control_source_faces[
            source_counts[subdivision_control_source_faces] < 3
        ]
        if len(degenerate_control_faces):
            raise VisualInputError(
                "Catmull-Clark control mesh contains non-hole faces with fewer "
                "than three corners and cannot be materialized faithfully: "
                f"{degenerate_control_faces[:10].tolist()}"
            )
        subdivision_control_counts = source_counts[
            subdivision_control_source_faces
        ]
        control_corner_parts = [
            np.arange(offsets[index], offsets[index + 1], dtype="<i8")
            for index in subdivision_control_source_faces
        ]
        control_corner_indices = (
            np.concatenate(control_corner_parts)
            if control_corner_parts
            else np.asarray([], dtype="<i8")
        )
        control_source_vertex_indices = source_indices[control_corner_indices]
        if control_source_vertex_indices.size and (
            int(control_source_vertex_indices.min()) < 0
            or int(control_source_vertex_indices.max()) >= len(source_points)
        ):
            raise VisualInputError(
                "Catmull-Clark control topology references an invalid point: "
                f"{region.mesh_path}"
            )
        control_used_vertices = np.unique(control_source_vertex_indices)
        control_remap = np.full(len(source_points), -1, dtype="<i8")
        control_remap[control_used_vertices] = np.arange(
            len(control_used_vertices), dtype="<i8"
        )
        subdivision_control_points = source_points[control_used_vertices]
        subdivision_control_indices = control_remap[control_source_vertex_indices]
        if not len(subdivision_control_counts):
            raise VisualInputError(
                f"Catmull-Clark mesh has no non-hole control faces: {region.mesh_path}"
            )
    provenance = {
        "usd_entrypoint": str(source_path),
        "usd_entrypoint_real_path": str(source_path.resolve()),
        "usd_entrypoint_sha256": _sha256_file(source_path),
        "used_layers": used_layers,
        "used_layer_set_sha256": _json_canonical_hash(
            "used-layer-set-v1", used_layers
        ),
        "segment_id": region.segment_id,
        "segment_key": region.key,
        "prim_path": region.mesh_path,
        "source_prim_path": region.source_mesh_path,
        "source_prim_path_kind": region.source_mesh_path_kind,
        "source_prim_prototype_relative_path": (
            region.source_mesh_prototype_relative_path
        ),
        "subset_path": region.subset_path,
        "region_kind": region.region_kind,
        "face_index_hash": region.face_index_hash,
        "exact_segment_source_face_count": int(len(region_face_indices)),
        "renderable_source_face_count": int(len(source_face_indices)),
        "renderable_source_faces_sha256": _array_sha256(
            "renderable-source-faces-v1", source_face_indices
        ),
        "omitted_degenerate_source_face_count": int(len(omitted_degenerate_faces)),
        "omitted_degenerate_source_faces": omitted_degenerate_faces.tolist(),
        "omitted_degenerate_source_faces_sha256": _array_sha256(
            "omitted-degenerate-source-faces-v1", omitted_degenerate_faces
        ),
        "material_path": region.material_path,
        "source_material_path": region.source_material_path,
        "material_record_id": fresh_material_binding["material_record_id"],
        "binding_relationship": region.binding_relationship,
        "selected_binding_purpose": region.selected_binding_purpose,
        "fresh_material_binding_attestation": fresh_material_binding,
        "is_instance_proxy": bool(region.is_instance_proxy),
        "transform_kind": transform_kind,
        "point_instancers": point_instancer_provenance,
        "orientation": str(mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded),
        "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
        "authored_hole_face_count": int(len(hole_indices)),
        "authored_hole_faces_sha256": _array_sha256(
            "authored-hole-faces-v1", hole_indices
        ),
        "source_mesh_face_count": int(len(source_counts)),
        "source_mesh_corner_count": int(len(source_indices)),
        "rejected_auxiliary_attributes": rejected_auxiliary_attributes,
        "rejected_auxiliary_attribute_count": len(rejected_auxiliary_attributes),
        "crease_lengths": crease_lengths.tolist(),
        "crease_lengths_sha256": _array_sha256(
            "subdivision-crease-lengths-v1", crease_lengths
        ),
        "crease_indices": crease_indices.tolist(),
        "crease_indices_sha256": _array_sha256(
            "subdivision-crease-indices-v1", crease_indices
        ),
        "crease_sharpnesses": crease_sharpnesses.tolist(),
        "crease_sharpnesses_sha256": _array_sha256(
            "subdivision-crease-sharpnesses-v1", crease_sharpnesses
        ),
        "corner_indices": corner_indices.tolist(),
        "corner_indices_sha256": _array_sha256(
            "subdivision-corner-indices-v1", corner_indices
        ),
        "corner_sharpnesses": corner_sharpnesses.tolist(),
        "corner_sharpnesses_sha256": _array_sha256(
            "subdivision-corner-sharpnesses-v1", corner_sharpnesses
        ),
        "subdivision_scheme": subdivision_scheme,
        "interpolate_boundary": str(mesh.GetInterpolateBoundaryAttr().Get() or ""),
        "face_varying_linear_interpolation": str(
            mesh.GetFaceVaryingLinearInterpolationAttr().Get() or ""
        ),
        "triangle_subdivision_rule": str(
            mesh.GetTriangleSubdivisionRuleAttr().Get() or ""
        ),
        "purpose": region.purpose,
        "visibility": region.visibility,
        "time_code": "default",
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
    }
    return SegmentGeometryPayload(
        points=selected_points,
        source_vertex_indices=used_vertices,
        face_vertex_counts=selected_counts,
        face_vertex_indices=selected_indices,
        source_face_indices=source_face_indices,
        source_corner_indices=source_corner_indices,
        occurrence_transforms=occurrence_transforms,
        subdivision_control_points=subdivision_control_points,
        subdivision_control_face_vertex_counts=subdivision_control_counts,
        subdivision_control_face_vertex_indices=subdivision_control_indices,
        subdivision_control_source_face_indices=(
            subdivision_control_source_faces
        ),
        uv_sets=uv_sets,
        normals=normals,
        provenance=provenance,
    )


def build_asset_geometry_payloads(
    usd_path: str | os.PathLike[str],
    extraction: Mapping[str, Any],
) -> dict[str, SegmentGeometryPayload]:
    """Build every segment from one shared composed stage/inventory pass."""

    source_path = Path(usd_path)
    if not source_path.is_file():
        raise VisualInputError(f"USD entry point is missing: {source_path}")
    extraction_path = extraction.get("file_path")
    if not isinstance(extraction_path, str) or not extraction_path:
        raise VisualInputError("Extraction has no USD file_path provenance")
    if Path(extraction_path).resolve() != source_path.resolve():
        raise VisualInputError(
            "Extraction provenance belongs to a different USD entry point: "
            f"{extraction_path!r} != {str(source_path)!r}"
        )
    serialized_inventory = extraction.get("segment_inventory")
    if not isinstance(serialized_inventory, Mapping):
        raise VisualInputError(
            "Extraction has no serialized segment_inventory gate"
        )
    serialized_inventory_errors = serialized_inventory.get("errors")
    if not isinstance(serialized_inventory_errors, list):
        raise VisualInputError(
            "Extraction segment_inventory.errors is not a list"
        )
    if serialized_inventory_errors:
        raise VisualInputError(
            "Extraction segment inventory contains hard errors: "
            + "; ".join(str(value) for value in serialized_inventory_errors)
        )
    try:
        stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    except Exception as exc:
        raise VisualInputError(
            f"Could not open USD stage: {type(exc).__name__}: {exc}"
        ) from exc
    if not stage:
        raise VisualInputError("Usd.Stage.Open returned no stage")
    composition_errors = stage_composition_errors(stage)
    if composition_errors:
        raise VisualInputError(
            "USD composition is incomplete: " + "; ".join(composition_errors)
        )
    inventory = enumerate_material_regions(stage)
    if inventory.errors:
        raise VisualInputError(
            "Fresh composed segment inventory contains hard errors: "
            + "; ".join(str(value) for value in inventory.errors)
        )
    segments = extraction.get("segments")
    if not isinstance(segments, Mapping) or not segments:
        raise VisualInputError("Extraction contains no segment mapping")
    serialized_ids = [
        segment.get("segment_id") if isinstance(segment, Mapping) else None
        for segment in segments.values()
    ]
    composed_ids = [region.segment_id for region in inventory.regions]
    if (
        any(not isinstance(value, str) or not value for value in serialized_ids)
        or len(serialized_ids) != len(set(serialized_ids))
        or set(serialized_ids) != set(composed_ids)
    ):
        raise VisualInputError(
            "Extraction segment set differs from the freshly composed USD inventory"
        )
    used_layers = _used_layer_records(stage)
    return {
        str(segment_key): build_segment_geometry_payload(
            source_path,
            segment_info,
            _stage=stage,
            _inventory=inventory,
            _used_layers=used_layers,
            _composition_validated=True,
        )
        for segment_key, segment_info in segments.items()
        if isinstance(segment_info, Mapping)
    }
