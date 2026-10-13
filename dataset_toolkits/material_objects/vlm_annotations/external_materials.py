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

"""External NVIDIA materials referenced by the USD datasets.

Lists the required files and checks their paths, sizes, and hashes.
Download them with ``dataset_toolkits/material_objects/fetch_external_materials.py``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable


NVIDIA_VMATERIALS_ORIGIN = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MATERIAL_ROOT = _REPOSITORY_ROOT / "Materials"


MATERIAL_FILES: dict[str, dict[str, Any]] = {
    "Base/Carpet/Carpet_Berber_Multi.mdl": {
        "sha256": "c2deb0df12cb10f4cdda593652d2732f10f0a9ce4c40b45a2cb181b540fa911f",
        "size_bytes": 1505,
    },
    "Base/Carpet/Carpet_Berber_Multi/Carpet_Berber_Multi_BaseColor.png": {
        "sha256": "d036db8c3fbd514d1758451868cc30f2255be34130cdd15a0d8a018fca1765bd",
        "size_bytes": 3989551,
    },
    "Base/Carpet/Carpet_Berber_Multi/Carpet_Berber_Multi_Normal.png": {
        "sha256": "6e030222655ff7c2369cdd0e21f52a2a050887304df2bc7caba548e4466111ae",
        "size_bytes": 8295994,
    },
    "Base/Carpet/Carpet_Berber_Multi/Carpet_Berber_Multi_ORM.png": {
        "sha256": "5fcad54032aee01a91e3785b6484b96864cb2a26509cdab27919ee3c73860388",
        "size_bytes": 1310373,
    },
    "Base/Glass/Clear_Glass.mdl": {
        "sha256": "5454a44463d100c648eeec2ae1c2e12ef9d16f0d4caebf675ff65e8427a6b40a",
        "size_bytes": 1177,
    },
    "Base/Glass/Frosted_Glass.mdl": {
        "sha256": "aa9220780bdab3d018cf22475ce117e9f96920557e7516f955b642ce82312c08",
        "size_bytes": 569,
    },
    "Base/Metals/Bronze/Bronze_BaseColor.png": {
        "sha256": "257de5ee4878a8cc1cf6cc975573cc0c1607d1fdd5bb2633ef27fdf12f4eb53c",
        "size_bytes": 6908853,
    },
    "Base/Metals/Bronze/Bronze_N.png": {
        "sha256": "6fb4f880a84aebb710eba3028b6ccf59391be328825e5e058aca537ec73334bb",
        "size_bytes": 3192099,
    },
    "Base/Metals/Bronze/Bronze_ORM.png": {
        "sha256": "89794fb03268b50f9a78f87a0694d7fbd83d3282d79adca4c0b9be2c16849825",
        "size_bytes": 4901007,
    },
    "Base/Templates/GlassUtils.mdl": {
        "sha256": "18baaa8cd508d85075148b80097c56c4924212928bb196f2d9ee2352d5c618a4",
        "size_bytes": 2417,
    },
    "Base/Templates/GlassWithVolume.mdl": {
        "sha256": "54f4f99e1bd8c6cde022dd246a954b9efe9b251a169854f49f582501a9bdf089",
        "size_bytes": 4906,
    },
    "Base/Wood/Ash.mdl": {
        "sha256": "08092f21c5cba9672483c5ad39fd02c0932e38390fe159a0ca25608a4406d0c8",
        "size_bytes": 1488,
    },
    "Base/Wood/Ash/Ash_BaseColor.png": {
        "sha256": "e12d4e9b5e39987673d26f84726c796fba7bddea8a14a4a8514381878e6aeddd",
        "size_bytes": 6691777,
    },
    "Base/Wood/Ash/Ash_ORM.png": {
        "sha256": "549dadc96c0e9cdfa05ab487552acd600d03a6ba0688864171eeba27e6a1183d",
        "size_bytes": 1129426,
    },
    "Base/Wood/Ash/Ash_Normal.png": {
        "sha256": "44b5b518bdb52957ecd8159ee7c2f59ad0ceca5fd4799bab8122c97335f0547c",
        "size_bytes": 5932653,
    },
    "Base/Textiles/Cloth_Gray.mdl": {
        "sha256": "675f410b15d0f02ef74aa06a2f323dd1a719a09f58208846cd5bbf4eb768d79c",
        "size_bytes": 1715,
    },
    "Base/Textiles/Cloth_Gray/Cloth_Gray_BaseColor.png": {
        "sha256": "e52ef797076dcca67ca5e9e71b2549719fae58066c9160315f15ac23b8215ac7",
        "size_bytes": 1077916,
    },
    "Base/Textiles/Cloth_Gray/Cloth_Gray_ORM.png": {
        "sha256": "f44248c5375625d1f23b2b3e7ec25b14d6c9fdf00044df8eeead3075c3e7b857",
        "size_bytes": 3828433,
    },
    "Base/Textiles/Cloth_Gray/Cloth_Gray_N.png": {
        "sha256": "5c4ff4b779b96b8acf31f7934ee9988eb56f0a07665812a9d0e6928b5f8249e7",
        "size_bytes": 1601051,
    },
    "Base/Textiles/Cloth_Gray/Cloth_Gray_DN.png": {
        "sha256": "01539c4dc33c314a174e508ed54ad4c6e0cec679c94083e77430d23fa18a89db",
        "size_bytes": 426540,
    },
    "Base/Textiles/Cloth_Black.mdl": {
        "sha256": "e22eff838b3a3e709d92f366a5b365f4043a0dec6efe3ea5b8bc7609c99f0828",
        "size_bytes": 1724,
    },
    "Base/Textiles/Cloth_Black/Cloth_Black_BaseColor.png": {
        "sha256": "b6ece9dd7cf6d30e2be8dfb2689f587e75cc0ace3997a4b1492c06cfc74f32ed",
        "size_bytes": 11803,
    },
    "Base/Textiles/Cloth_Black/Cloth_Black_ORM.png": {
        "sha256": "caa7aa74758e3deb270280937616ad4694913510e44326f0c4e59013607bb837",
        "size_bytes": 2683462,
    },
    "Base/Textiles/Cloth_Black/Cloth_Black_N.png": {
        "sha256": "8015158d7306597f132a80982d6da5fd21a6b7c50eeffd833d83f3df3a3c2d8f",
        "size_bytes": 1605889,
    },
    "Base/Textiles/Cloth_Black/Cloth_Black_DN.png": {
        "sha256": "01539c4dc33c314a174e508ed54ad4c6e0cec679c94083e77430d23fa18a89db",
        "size_bytes": 426540,
    },
}


MATERIAL_BUNDLES: dict[str, dict[str, Any]] = {
    "nvidia_vmaterials_carpet_berber_multi": {
        "module": "Base/Carpet/Carpet_Berber_Multi.mdl",
        "files": (
            "Base/Carpet/Carpet_Berber_Multi.mdl",
            "Base/Carpet/Carpet_Berber_Multi/Carpet_Berber_Multi_BaseColor.png",
            "Base/Carpet/Carpet_Berber_Multi/Carpet_Berber_Multi_Normal.png",
            "Base/Carpet/Carpet_Berber_Multi/Carpet_Berber_Multi_ORM.png",
        ),
    },
    "nvidia_vmaterials_clear_glass": {
        "module": "Base/Glass/Clear_Glass.mdl",
        "files": ("Base/Glass/Clear_Glass.mdl",),
    },
    "nvidia_vmaterials_frosted_glass": {
        "module": "Base/Glass/Frosted_Glass.mdl",
        "files": (
            "Base/Glass/Frosted_Glass.mdl",
            "Base/Templates/GlassWithVolume.mdl",
            "Base/Templates/GlassUtils.mdl",
        ),
    },
    "nvidia_vmaterials_ash": {
        "module": "Base/Wood/Ash.mdl",
        "files": (
            "Base/Wood/Ash.mdl",
            "Base/Wood/Ash/Ash_BaseColor.png",
            "Base/Wood/Ash/Ash_ORM.png",
            "Base/Wood/Ash/Ash_Normal.png",
        ),
    },
    "nvidia_vmaterials_cloth_gray": {
        "module": "Base/Textiles/Cloth_Gray.mdl",
        "files": (
            "Base/Textiles/Cloth_Gray.mdl",
            "Base/Textiles/Cloth_Gray/Cloth_Gray_BaseColor.png",
            "Base/Textiles/Cloth_Gray/Cloth_Gray_ORM.png",
            "Base/Textiles/Cloth_Gray/Cloth_Gray_N.png",
            "Base/Textiles/Cloth_Gray/Cloth_Gray_DN.png",
        ),
    },
    "nvidia_vmaterials_cloth_black": {
        "module": "Base/Textiles/Cloth_Black.mdl",
        "files": (
            "Base/Textiles/Cloth_Black.mdl",
            "Base/Textiles/Cloth_Black/Cloth_Black_BaseColor.png",
            "Base/Textiles/Cloth_Black/Cloth_Black_ORM.png",
            "Base/Textiles/Cloth_Black/Cloth_Black_N.png",
            "Base/Textiles/Cloth_Black/Cloth_Black_DN.png",
        ),
    },
}


_CLOTH_BLACK_AUTHORED = (
    "../../../../../Materials/Base/Textiles/Cloth_Black.mdl"
)
_CLOTH_GRAY_AUTHORED = (
    "../../../../../Materials/Base/Textiles/Cloth_Gray.mdl"
)
_ASH_AUTHORED = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Materials/Base/Wood/Ash.mdl"
)
_FROSTED_GLASS_AUTHORED = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Materials/Base/Glass/Frosted_Glass.mdl"
)
_CLEAR_GLASS_AUTHORED = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Materials/Base/Glass/Clear_Glass.mdl"
)
_CARPET_BERBER_MULTI_AUTHORED = (
    "/Library/Materials/Base/Carpet/Carpet_Berber_Multi.mdl"
)


MDL_REPAIRS: tuple[dict[str, str], ...] = (
    {
        "asset": "residential/Lighting/Floor Lamps/BrassFloorLamp.usd",
        "entrypoint_sha256": "e45ec893b4f0bcfa6e505011347ab9ba55ca418f1b74711de42803fc24893945",
        "authored_source_asset": _FROSTED_GLASS_AUTHORED,
        "bundle": "nvidia_vmaterials_frosted_glass",
    },
    {
        "asset": "residential/Lighting/Table Lamps/Colbert.usd",
        "entrypoint_sha256": "1746902e46a2bbd648ff7fc27a27249bc96bcf0390b2d4720e317a082c406bc1",
        "authored_source_asset": _FROSTED_GLASS_AUTHORED,
        "bundle": "nvidia_vmaterials_frosted_glass",
    },
    {
        "asset": "residential/Outdoors/Lighting/CircularWallSconce.usd",
        "entrypoint_sha256": "7eff960badfce23d6f58a314331a5cebc4b6dfeb27aec50b7b3c7c855c04bffe",
        "authored_source_asset": _FROSTED_GLASS_AUTHORED,
        "bundle": "nvidia_vmaterials_frosted_glass",
    },
    {
        "asset": "residential/Outdoors/Lighting/WallSconce.usd",
        "entrypoint_sha256": "8a53e4885e8f8279ce07915456b6efb405fe314d07add1f43af4add133a44334",
        "authored_source_asset": _FROSTED_GLASS_AUTHORED,
        "bundle": "nvidia_vmaterials_frosted_glass",
    },
    {
        "asset": "residential/Outdoors/Lighting/GardenSpotlight.usd",
        "entrypoint_sha256": "a9f1a28c782b3616af00876236c03033d7f7fb046a73ff226619aa2753a2977f",
        "authored_source_asset": _CLEAR_GLASS_AUTHORED,
        "bundle": "nvidia_vmaterials_clear_glass",
    },
    {
        "asset": "residential/Decor/Rugs/Rug_01.usd",
        "entrypoint_sha256": "afac886ee1d028d7138c6035c12389221aa5b4f952e5bb78dd67dab56f4556fc",
        "authored_source_asset": _CARPET_BERBER_MULTI_AUTHORED,
        "bundle": "nvidia_vmaterials_carpet_berber_multi",
    },
    {
        "asset": "commercial/Reception/L_Desk.usd",
        "entrypoint_sha256": "69c958404d98bfca6db5e65b5509f1d31bfd0b1dc4e3a41f18f1d063d558cf09",
        "authored_source_asset": _ASH_AUTHORED,
        "bundle": "nvidia_vmaterials_ash",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_A.usd",
        "entrypoint_sha256": "c577a0f36c570b3f82a8ccea897e2affeb4f06f9be518b357295cec5f499e915",
        "authored_source_asset": _CLOTH_BLACK_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_black",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_A.usd",
        "entrypoint_sha256": "c577a0f36c570b3f82a8ccea897e2affeb4f06f9be518b357295cec5f499e915",
        "authored_source_asset": _CLOTH_GRAY_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_gray",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_B.usd",
        "entrypoint_sha256": "5f9b54f3b72eda3e3920fcd1ac654273fd00a505db656bcaf0444518b8fbd4c5",
        "authored_source_asset": _CLOTH_BLACK_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_black",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_B.usd",
        "entrypoint_sha256": "5f9b54f3b72eda3e3920fcd1ac654273fd00a505db656bcaf0444518b8fbd4c5",
        "authored_source_asset": _CLOTH_GRAY_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_gray",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_C.usd",
        "entrypoint_sha256": "44344722837d54dae878b9eac9af13d12d445dbdd4caa7410d8a78d6daf4989d",
        "authored_source_asset": _CLOTH_BLACK_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_black",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_C.usd",
        "entrypoint_sha256": "44344722837d54dae878b9eac9af13d12d445dbdd4caa7410d8a78d6daf4989d",
        "authored_source_asset": _CLOTH_GRAY_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_gray",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_Yin.usd",
        "entrypoint_sha256": "7b4ec26ac21125db41a282b5bc1c9418baf7aa1d2cce638e228d4d461ead391f",
        "authored_source_asset": _CLOTH_BLACK_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_black",
    },
    {
        "asset": "commercial/Seating/Caprice/Caprice_Yin.usd",
        "entrypoint_sha256": "7b4ec26ac21125db41a282b5bc1c9418baf7aa1d2cce638e228d4d461ead391f",
        "authored_source_asset": _CLOTH_GRAY_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_gray",
    },
    {
        "asset": "commercial/Seating/Caprice/CapriceBackless_A.usd",
        "entrypoint_sha256": "cfc6155daeae8e7f41033d2f7e3e723973664a35bcce08a85718b868da9b413c",
        "authored_source_asset": _CLOTH_GRAY_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_gray",
    },
    {
        "asset": "commercial/Seating/Caprice/CapriceBackless_B.usd",
        "entrypoint_sha256": "e711234bfecd39e9dd0f80c73385a412202fa8e0c6e081fd231d88fa23956fd2",
        "authored_source_asset": _CLOTH_BLACK_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_black",
    },
    {
        "asset": "commercial/Seating/Caprice/CapriceBackless_C.usd",
        "entrypoint_sha256": "98b4e402d48d1430b471c5a8200bc4467bd5a3991af431e0b7131ac4489695e7",
        "authored_source_asset": _CLOTH_BLACK_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_black",
    },
    {
        "asset": "commercial/Seating/Caprice/CapriceBackless_Yin.usd",
        "entrypoint_sha256": "7f2e262557cd9b4241653011f2d42bce126f0abf58edcdcbc019d05d2dbf1142",
        "authored_source_asset": _CLOTH_GRAY_AUTHORED,
        "bundle": "nvidia_vmaterials_cloth_gray",
    },
)


def material_root_from_environment() -> Path:
    """Return the configured vMaterials root without creating it."""

    override = os.environ.get("TRELLIS_VLM_MATERIAL_ROOT")
    return Path(override) if override else MATERIAL_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def material_file_url(relative_path: str) -> str:
    if relative_path not in MATERIAL_FILES:
        raise KeyError(f"unrecognized vMaterials file: {relative_path}")
    return f"{NVIDIA_VMATERIALS_ORIGIN}/{relative_path}"


def material_manifest_path(
    material_root: str | os.PathLike[str], relative_path: str
) -> Path | None:
    """Return a canonical manifest path with no symlinked descendants.

    MDL texture literals resolve relative to the canonical module location.
    The configured root itself may be a symlink to a shared cache and is
    canonicalized once.  Allowing a directory *below* that root, or a manifest
    file itself, to be a symlink could make literals consume files outside the
    bundle whose hashes were attested at the lexical path.
    """

    if relative_path not in MATERIAL_FILES:
        return None
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    root = Path(material_root)
    if root.is_symlink():
        try:
            canonical_root = root.resolve(strict=True)
        except OSError:
            return None
        if not canonical_root.is_dir():
            return None
    else:
        if root.exists() and not root.is_dir():
            return None
        canonical_root = root.resolve(strict=False)
    candidate = canonical_root.joinpath(*relative.parts)
    current = canonical_root
    if current.is_symlink():
        return None
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    if candidate.resolve(strict=False) != candidate:
        return None
    return candidate


def iter_material_files() -> Iterable[tuple[str, dict[str, Any]]]:
    for relative_path in sorted(MATERIAL_FILES):
        yield relative_path, dict(MATERIAL_FILES[relative_path])


def verify_material_file(path: Path, record: dict[str, Any]) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size == int(record["size_bytes"])
            and sha256_file(path) == str(record["sha256"])
        )
    except OSError:
        return False


def verify_material_root(
    material_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the status of each required material file."""

    root = Path(material_root) if material_root is not None else material_root_from_environment()
    records: list[dict[str, Any]] = []
    for relative_path, manifest in iter_material_files():
        path = material_manifest_path(root, relative_path)
        display_path = path or root.resolve(strict=False) / relative_path
        records.append(
            {
                "relative_path": relative_path,
                "path": str(display_path),
                "url": material_file_url(relative_path),
                "expected_sha256": str(manifest["sha256"]),
                "expected_size_bytes": int(manifest["size_bytes"]),
                "path_safe": path is not None,
                "verified": path is not None and verify_material_file(path, manifest),
            }
        )
    return records


def external_mdl_repair_is_applicable(
    authored_source_asset: str,
    root_layer_identifier: str | os.PathLike[str] | None,
) -> bool:
    """Return whether an exact repair key applies, independent of file state.

    The distinction between "not allowlisted" and "allowlisted but corrupt"
    is security-critical.  A corrupt relative bundle must not fall through to
    ordinary filesystem resolution merely because its module still exists.
    """

    if not root_layer_identifier:
        return False
    root_text = Path(root_layer_identifier).as_posix()
    return any(
        authored_source_asset == repair["authored_source_asset"]
        and (
            root_text == str(repair["asset"])
            or root_text.endswith(f"/{repair['asset']}")
        )
        for repair in MDL_REPAIRS
    )


def resolve_external_mdl_source(
    authored_source_asset: str,
    root_layer_identifier: str | os.PathLike[str] | None,
    *,
    material_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Locate and verify the files required by an external MDL material."""

    if not root_layer_identifier:
        return None
    root_text = Path(root_layer_identifier).as_posix()
    matching_repair = None
    for repair in MDL_REPAIRS:
        asset = str(repair["asset"])
        if (
            authored_source_asset == repair["authored_source_asset"]
            and (root_text == asset or root_text.endswith(f"/{asset}"))
        ):
            matching_repair = repair
            break
    if matching_repair is None:
        return None

    entrypoint = Path(root_layer_identifier)
    try:
        if (
            not entrypoint.is_file()
            or sha256_file(entrypoint)
            != matching_repair["entrypoint_sha256"]
        ):
            return None
    except OSError:
        return None

    root = Path(material_root) if material_root is not None else material_root_from_environment()
    bundle_name = str(matching_repair["bundle"])
    bundle = MATERIAL_BUNDLES[bundle_name]
    dependency_records: list[dict[str, Any]] = [
        {
            "path": str(entrypoint.resolve()),
            "sha256": str(matching_repair["entrypoint_sha256"]),
            "role": "target_entrypoint",
        }
    ]
    bundle_records: list[dict[str, Any]] = []
    for relative_path in bundle["files"]:
        manifest = MATERIAL_FILES[str(relative_path)]
        path = material_manifest_path(root, str(relative_path))
        if path is None or not verify_material_file(path, manifest):
            return None
        resolved_path = path.resolve()
        file_record = {
            "relative_path": str(relative_path),
            "path": str(resolved_path),
            "url": material_file_url(str(relative_path)),
            "sha256": str(manifest["sha256"]),
            "size_bytes": int(manifest["size_bytes"]),
        }
        bundle_records.append(file_record)
        dependency_records.append(
            {
                "path": str(resolved_path),
                "sha256": str(manifest["sha256"]),
                "role": (
                    "pinned_external_mdl_module"
                    if str(relative_path) == bundle["module"]
                    else "pinned_external_mdl_dependency"
                    if str(relative_path).casefold().endswith(".mdl")
                    else "pinned_external_mdl_texture"
                ),
            }
        )

    module_path = material_manifest_path(root, str(bundle["module"]))
    if module_path is None:
        return None
    module_path = module_path.resolve()
    provenance = {
        "target_asset": str(matching_repair["asset"]),
        "target_entrypoint_path": str(entrypoint.resolve()),
        "target_entrypoint_sha256": str(matching_repair["entrypoint_sha256"]),
        "authored_source_asset": authored_source_asset,
        "bundle": bundle_name,
        "resolved_source_path": str(module_path),
        "resolved_source_sha256": MATERIAL_FILES[str(bundle["module"])][
            "sha256"
        ],
        "bundle_files": bundle_records,
        "repair_dependency_paths": dependency_records,
        "reason": (
            "the asset and authored sourceAsset match a known repair; "
            "the USD file, MDL module, and its textures match the recorded "
            "SHA-256 hashes"
        ),
        "requires_upstream_source_repair": True,
    }
    return module_path, provenance


__all__ = [
    "MATERIAL_ROOT",
    "NVIDIA_VMATERIALS_ORIGIN",
    "MDL_REPAIRS",
    "MATERIAL_BUNDLES",
    "MATERIAL_FILES",
    "iter_material_files",
    "material_file_url",
    "material_manifest_path",
    "material_root_from_environment",
    "resolve_external_mdl_source",
    "sha256_file",
    "verify_material_file",
    "verify_material_root",
    "external_mdl_repair_is_applicable",
]
