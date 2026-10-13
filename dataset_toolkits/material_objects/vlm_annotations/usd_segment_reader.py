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

"""Read USD material segments and their textures for VLM annotation.

Preserve face assignments, instances, and USD paths when grouping geometry.
Find textures through each material's surface shader connections.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sysconfig
from typing import Any, Iterable, Mapping
import zlib

from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

from dataset_toolkits.material_objects.vlm_annotations.external_materials import (
    MATERIAL_FILES,
    material_manifest_path,
    material_root_from_environment,
    resolve_external_mdl_source,
    verify_material_file,
    external_mdl_repair_is_applicable,
)


_SURFACE_BASE_COLOR_SHADER_INPUT_NAMES = frozenset(
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
MATERIAL_BINDING_PURPOSE_PRIORITY = ("full", "allPurpose", "preview")
IMAGE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".dds",
        ".exr",
        ".hdr",
        ".jpeg",
        ".jpg",
        ".png",
        ".tga",
        ".tif",
        ".tiff",
        ".webp",
    }
)
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})

SOURCE_REPAIR_METHOD = "verified_exact_sibling_content_repair"
CASE_REPAIR_METHOD = (
    "verified_unique_same_directory_casefold_spelling_repair"
)
PATH_CASE_REPAIR_METHOD = (
    "verified_unique_full_path_casefold_spelling_repair"
)
TEXTURE_REPAIR_METHOD = (
    "verified_exact_property_semantic_texture_repair"
)
SOURCE_REPAIRS: dict[tuple[str, str], dict[str, Any]] = {
    (
        "residential/Decor/Pictures/Painting_01.usd",
        "./textures/Oak_BaseColor.png",
    ): {
        "source_asset": "residential/Decor/Shelves/Textures/Oak_BaseColor.png",
        "source_sha256": (
            "58648d35e3b8a9d953948ce0394b56aa95184e1f88b5abab139e7a32a2b44043"
        ),
        "matching_source_assets": [
            "residential/Decor/Shelves/Textures/Oak_BaseColor.png",
            "residential/Furniture/Bookshelves/Textures/Oak_BaseColor.png",
            "residential/Misc/Ladders/Textures/Oak_BaseColor.png",
        ],
    },
    (
        "residential/Decor/Pictures/Painting_01.usd",
        "./textures/Oak_N.png",
    ): {
        "source_asset": "residential/Decor/Shelves/Textures/Oak_N.png",
        "source_sha256": (
            "4538148c60da7c94d88d656bbc62106a16bd652052bc196cab56ee6d9e01c249"
        ),
        "matching_source_assets": [
            "residential/Decor/Shelves/Textures/Oak_N.png",
            "residential/Furniture/Bookshelves/Textures/Oak_N.png",
            "residential/Misc/Ladders/Textures/Oak_N.png",
        ],
    },
    (
        "residential/Decor/Pictures/Painting_01.usd",
        "./textures/Oak_ORM.png",
    ): {
        "source_asset": "residential/Decor/Shelves/Textures/Oak_ORM.png",
        "source_sha256": (
            "754ccca3176ccb3bbeb648af574c57eb5be0b876c579faaf45f9e8a09f71ec39"
        ),
        "matching_source_assets": [
            "residential/Decor/Shelves/Textures/Oak_ORM.png",
            "residential/Furniture/Bookshelves/Textures/Oak_ORM.png",
            "residential/Misc/Ladders/Textures/Oak_ORM.png",
        ],
    },
    (
        "residential/Decor/Sculptures/Cube_Wireframe.usd",
        "./Textures/Iron_BaseColor.png",
    ): {
        "source_asset": "residential/Decor/Shelves/Textures/Iron_BaseColor.png",
        "source_sha256": (
            "a6dfd99dda89054212b3610ec32efc35ac09607a58f560fee42cc7aef721ec07"
        ),
        "matching_source_assets": [
            "residential/Decor/Shelves/Textures/Iron_BaseColor.png",
            "residential/Furniture/Beds/Textures/Iron_BaseColor.png",
            "residential/Furniture/Sofas/textures/Iron_BaseColor.png",
        ],
    },
    (
        "residential/Decor/Sculptures/Cube_Wireframe.usd",
        "./Textures/Iron_N.png",
    ): {
        "source_asset": "residential/Decor/Shelves/Textures/Iron_N.png",
        "source_sha256": (
            "9e1127afe0a33ac5d7c9ab888228080c6d46b34b0d5b98f6484d6f7d0f40915b"
        ),
        "matching_source_assets": [
            "residential/Decor/Shelves/Textures/Iron_N.png",
            "residential/Furniture/Beds/Textures/Iron_N.png",
            "residential/Furniture/Sofas/textures/Iron_N.png",
        ],
    },
    (
        "residential/Decor/Sculptures/Cube_Wireframe.usd",
        "./Textures/Iron_ORM.png",
    ): {
        "source_asset": "residential/Decor/Shelves/Textures/Iron_ORM.png",
        "source_sha256": (
            "f1b4f0f3db1db87e07e2ba8eaf96af1b8a68425ebd4b180d22d74979d4b4a0f6"
        ),
        "matching_source_assets": [
            "residential/Decor/Shelves/Textures/Iron_ORM.png",
            "residential/Furniture/Beds/Textures/Iron_ORM.png",
            "residential/Furniture/Sofas/textures/Iron_ORM.png",
        ],
    },
}

TEXTURE_REPAIRS: dict[
    tuple[str, str, str], dict[str, Any]
] = {
    (
        "commercial/Seating/Petite_Loveseat.usd",
        "/World/Looks/Metal/Shader.inputs:diffuse_texture",
        "./Textures/PetiteLoveseat_Metal_M.png",
    ): {
        "entrypoint_sha256": (
            "57326b8fbef3e9623dd0e353592d6db8802a333946558bb71274600d6b09f2c6"
        ),
        "original_file_sha256": (
            "46ed219a57dc78081c83b825ce8f6f09a4acd94c7a313770493d3e26ffe76ec4"
        ),
        "replacement_authored_asset": (
            "./Textures/PetiteLoveseat_Metal_A.png"
        ),
        "replacement_asset": (
            "commercial/Seating/Textures/PetiteLoveseat_Metal_A.png"
        ),
        "replacement_sha256": (
            "b1d4ce02cbb1105f7e3f0886a111c3bc54d650b597023120eb2767fa808a840f"
        ),
        "reference_assets": [
            {
                "asset": "commercial/Seating/Petite_Chair.usd",
                "sha256": (
                    "15799922cfceea0ffbd083c11c49b72af52e55c3fcad4272eeae7b0a54d75dc3"
                ),
                "property_path": (
                    "/World/Looks/Metal/Shader.inputs:diffuse_texture"
                ),
                "authored_asset": "./Textures/PetiteChair_Metal_A.png",
                "texture_asset": (
                    "commercial/Seating/Textures/PetiteChair_Metal_A.png"
                ),
                "texture_sha256": (
                    "b1d4ce02cbb1105f7e3f0886a111c3bc54d650b597023120eb2767fa808a840f"
                ),
            }
        ],
        "reference_files": [
            {
                "asset": (
                    "commercial/Seating/.thumbs/256x256/"
                    "Petite_Loveseat.usd.png"
                ),
                "sha256": (
                    "0dbf4b6cf3ceb44311193a91fc6b56be3a4a765228045fee5f0003a6acdedf2f"
                ),
            },
            {
                "asset": (
                    "commercial/Seating/.thumbs/256x256/"
                    "Petite_Chair.usd.png"
                ),
                "sha256": (
                    "13d21cf143609d3bf31cc9dfcb3a2436b93df44a1a2261f5ac5bb5635fd38f7f"
                ),
            },
        ],
        "reason": (
            "the diffuse input points to a white metallic mask (_M); "
            "Petite Chair uses the matching albedo map (_A), and both "
            "thumbnails show silver-gray legs"
        ),
    },
    (
        "vegetation/Leaves/cluster_2.usd",
        "/World/leaves/material/Shader.inputs:normalmap_texture",
        "./roughness.jpg",
    ): {
        "entrypoint_sha256": (
            "6b0301a817eb6f18877da232f8e1ba3992174fcfc399efac83c14789d6c1bf54"
        ),
        "original_file_sha256": (
            "c46327a75455cadf57d59159e94e72829f10ad41851252ea47ae0d9f8fe48855"
        ),
        "replacement_authored_asset": "./normal.jpg",
        "replacement_asset": "vegetation/Leaves/normal.jpg",
        "replacement_sha256": (
            "8703d4efc396072a1be811320b10a259d725b1b3d9b21305ddda0bc22d003525"
        ),
        "reference_assets": [],
        "reference_files": [],
        "reason": (
            "the normal input points to grayscale roughness.jpg; "
            "normal.jpg is the normal map used by cluster_1 for this input"
        ),
    },
    (
        "vegetation/Leaves/cluster_2.usd",
        "/World/leaves/material/Shader.inputs:reflectionroughness_texture",
        "./basecolor.jpg",
    ): {
        "entrypoint_sha256": (
            "6b0301a817eb6f18877da232f8e1ba3992174fcfc399efac83c14789d6c1bf54"
        ),
        "original_file_sha256": None,
        "replacement_authored_asset": "./roughness.jpg",
        "replacement_asset": "vegetation/Leaves/roughness.jpg",
        "replacement_sha256": (
            "c46327a75455cadf57d59159e94e72829f10ad41851252ea47ae0d9f8fe48855"
        ),
        "reference_assets": [],
        "reference_files": [],
        "reason": (
            "the roughness input points to missing basecolor.jpg; "
            "cluster_1 uses the existing roughness.jpg for this input"
        ),
    },
}

_BRONZE_TEXTURE_REPAIR_TARGETS = {
    "residential/Decor/Sculptures/FlattenedSphere.usd": {
        "entrypoint_sha256": (
            "115d93be68c617c047448780e5720b61163de0b0baec778badf9c1816709bc24"
        ),
        "authoring_layer_asset": (
            "residential/Decor/Sculptures/FlattenedSphere.usd"
        ),
        "authoring_layer_sha256": (
            "115d93be68c617c047448780e5720b61163de0b0baec778badf9c1816709bc24"
        ),
        "property_prefix": "/RootNode/Looks/Bronze/Shader.inputs:",
        "thumbnail": (
            "residential/Decor/Sculptures/.thumbs/256x256/"
            "FlattenedSphere.usd.png"
        ),
        "thumbnail_sha256": (
            "d44c8f0b0b5584ada9cb1f6e9d57712a8d9e6974bed93c65627c9776239b8b2c"
        ),
    },
    "residential/Furniture/Bookshelves/Ferguson_Var01.usd": {
        "entrypoint_sha256": (
            "072e9b951b625af5d8aca4837847228d3225d08372cb1f9323738f777578201c"
        ),
        "authoring_layer_asset": (
            "residential/Decor/Sculptures/FlattenedSphere.usd"
        ),
        "authoring_layer_sha256": (
            "115d93be68c617c047448780e5720b61163de0b0baec778badf9c1816709bc24"
        ),
        "property_prefix": (
            "/World/FlattenedSphere/Looks/Bronze/Shader.inputs:"
        ),
        "thumbnail": (
            "residential/Furniture/Bookshelves/.thumbs/256x256/"
            "Ferguson_Var01.usd.png"
        ),
        "thumbnail_sha256": (
            "ab88a87ba9cabe9da2a4ae88925420e49d08c4cc895680669b1bd0c25ad78a2d"
        ),
    },
}
_BRONZE_TEXTURE_REPAIR_INPUTS = {
    "diffuse_texture": ("Bronze_BaseColor.png", "Bronze_BaseColor.png"),
    "normalmap_texture": ("Bronze_N.png", "Bronze_N.png"),
    "ORM_texture": ("Bronze_ORM.png", "Bronze_ORM.png"),
}
for _asset_suffix, _target in _BRONZE_TEXTURE_REPAIR_TARGETS.items():
    for _input_name, (_authored_filename, _verified_filename) in (
        _BRONZE_TEXTURE_REPAIR_INPUTS.items()
    ):
        _verified_relative_path = (
            f"Base/Metals/Bronze/{_verified_filename}"
        )
        TEXTURE_REPAIRS[
            (
                _asset_suffix,
                f"{_target['property_prefix']}{_input_name}",
                f"./Textures/{_authored_filename}",
            )
        ] = {
            "entrypoint_sha256": _target["entrypoint_sha256"],
            "authoring_layer_asset": _target["authoring_layer_asset"],
            "authoring_layer_sha256": _target[
                "authoring_layer_sha256"
            ],
            "original_file_sha256": None,
            "replacement_authored_asset": (
                f"/Materials/{_verified_relative_path}"
            ),
            "replacement_verified_material_asset": _verified_relative_path,
            "replacement_sha256": MATERIAL_FILES[
                _verified_relative_path
            ]["sha256"],
            "reference_assets": [],
            "reference_files": [
                {
                    "asset": _target["thumbnail"],
                    "sha256": _target["thumbnail_sha256"],
                }
            ],
            "reason": (
                "the Bronze material is missing a BaseColor, N, or ORM map; "
                "the NVIDIA material library supplies the matching map, "
                "and the asset thumbnail shows the same Bronze appearance"
            ),
        }

_LEAVES_SEMANTIC_REPAIR_REFERENCE_ASSETS = (
    (
        "vegetation/Leaves/cluster_1.usd",
        "6ca4713542b7f4bbce6951bc295ab19c0489e4dcad22f542c57e21eac53ca96e",
    ),
)
for _repair_key, _repair in TEXTURE_REPAIRS.items():
    if _repair_key[0] != "vegetation/Leaves/cluster_2.usd":
        continue
    _role_filename = (
        "normal.jpg" if _repair_key[1].endswith("normalmap_texture")
        else "roughness.jpg"
    )
    _repair["reference_assets"] = [
        {
            "asset": asset,
            "sha256": sha256,
            "property_path": _repair_key[1],
            "authored_asset": f"./{_role_filename}",
            "texture_asset": f"vegetation/Leaves/{_role_filename}",
            "texture_sha256": _repair["replacement_sha256"],
        }
        for asset, sha256 in _LEAVES_SEMANTIC_REPAIR_REFERENCE_ASSETS
    ]
    _repair["reference_files"] = [
        {
            "asset": "vegetation/Leaves/material.mdl",
            "sha256": (
                "2ad1275a1bd1d88faae7463442c5b40cee65373da70fbb6bd0bc48b0f9876b29"
            ),
        },
        {
            "asset": "vegetation/Leaves/thumbs/256x256/cluster_2.usd.png",
            "sha256": (
                "6189c05e4291df105fb814bc2fa0603e4e8588822562333a172fdeaca2326f18"
            ),
        },
    ]


_CHESS_KNIGHT_REPAIR_TARGETS = {
    "residential/Entertainment/Games/ChessSet/ChessSet_FullSet.usd": {
        "entrypoint_sha256": (
            "558064f1ec9a8eb9dab06781d8b1f7095c6cc8e3e669e073ad4fa510e28338dc"
        ),
        "property_paths": (
            "/World/Black_Pieces/Chess_Knight/Looks/White/Shader.inputs:normalmap_texture",
            "/World/Black_Pieces/Chess_Knight_01/Looks/White/Shader.inputs:normalmap_texture",
            "/World/White_Pieces/Chess_Knight/Looks/White/Shader.inputs:normalmap_texture",
        ),
    },
    "residential/Entertainment/Games/ChessSet/Chess_Knight.usd": {
        "entrypoint_sha256": (
            "31aacc405692be2305507380760c6fb67342acb04cf96eeb8030ef0cfd01f80f"
        ),
        "property_paths": (
            "/World/Looks/White/Shader.inputs:normalmap_texture",
        ),
    },
    "residential/Furniture/Bookshelves/TetriminoShelf_Var01.usd": {
        "entrypoint_sha256": (
            "c5b8bfc8c7a90b7274a000e962d2cfff824aff5305b38551208db0e5de5cdfbb"
        ),
        "property_paths": (
            "/World/Chess_Knight/Looks/White/Shader.inputs:normalmap_texture",
        ),
    },
}
_CHESS_KNIGHT_REFERENCE_ASSET = (
    "residential/Entertainment/Games/ChessSet/Chess_Knight.usd"
)
_CHESS_KNIGHT_REFERENCE_ENTRYPOINT_SHA256 = (
    "31aacc405692be2305507380760c6fb67342acb04cf96eeb8030ef0cfd01f80f"
)
_CHESS_KNIGHT_REPLACEMENT_ASSET = (
    "residential/Entertainment/Games/ChessSet/Textures/"
    "ChessSet_WoodLightKnight_N.png"
)
_CHESS_KNIGHT_REPLACEMENT_SHA256 = (
    "258caa646278882bb093c8cad7de7604deeaed980fe05204f2b069825daba383"
)
for _asset_suffix, _target in _CHESS_KNIGHT_REPAIR_TARGETS.items():
    for _property_path in _target["property_paths"]:
        TEXTURE_REPAIRS[
            (
                _asset_suffix,
                _property_path,
                "./Textures/ChessSet_WoodLightHorse_N.png",
            )
        ] = {
            "entrypoint_sha256": _target["entrypoint_sha256"],
            "original_file_sha256": None,
            "replacement_authored_asset": (
                "./Textures/ChessSet_WoodLightKnight_N.png"
            ),
            "replacement_asset": _CHESS_KNIGHT_REPLACEMENT_ASSET,
            "replacement_sha256": _CHESS_KNIGHT_REPLACEMENT_SHA256,
            "reference_assets": [
                {
                    "asset": _CHESS_KNIGHT_REFERENCE_ASSET,
                    "sha256": _CHESS_KNIGHT_REFERENCE_ENTRYPOINT_SHA256,
                    "property_path": (
                        "/World/Looks/White/Shader.inputs:diffuse_texture"
                    ),
                    "authored_asset": (
                        "./Textures/ChessSet_WoodLightKnight_A.png"
                    ),
                    "texture_asset": (
                        "residential/Entertainment/Games/ChessSet/Textures/"
                        "ChessSet_WoodLightKnight_A.png"
                    ),
                    "texture_sha256": (
                        "2a10fb49035b27a463c9d3afbdf231cf90af975d465b25e204edc0570b8bc1ba"
                    ),
                },
                {
                    "asset": _CHESS_KNIGHT_REFERENCE_ASSET,
                    "sha256": _CHESS_KNIGHT_REFERENCE_ENTRYPOINT_SHA256,
                    "property_path": (
                        "/World/Looks/White/Shader.inputs:reflectionroughness_texture"
                    ),
                    "authored_asset": (
                        "./Textures/ChessSet_WoodLightKnight_SG.png"
                    ),
                    "texture_asset": (
                        "residential/Entertainment/Games/ChessSet/Textures/"
                        "ChessSet_WoodLightKnight_SG.png"
                    ),
                    "texture_sha256": (
                        "8e5381a9a156ecd932d270c25a34d186bfaf436d295de68d77f7fe1bda982489"
                    ),
                },
            ],
            "reference_files": [
                {
                    "asset": (
                        "residential/Entertainment/Games/ChessSet/"
                        ".thumbs/256x256/Chess_Knight.usd.png"
                    ),
                    "sha256": (
                        "071513112bc81171c292e8c14dec456d4fb840cf9b0c11b3b01fd2d9340b6f4f"
                    ),
                }
            ],
            "reason": (
                "White Chess_Knight uses LightKnight albedo and roughness "
                "maps, but its normal input points to missing LightHorse; "
                "the existing LightKnight normal map matches DarkHorse_N"
            ),
        }


_AFTON_ENTRYPOINT = "residential/Furniture/SofaTables/AftonConsole.usd"
_AFTON_ENTRYPOINT_SHA256 = (
    "4f7ef3c38d99acf9279834b191fd6e61f160d5e14818915f2c75b5a889d505be"
)
TEXTURE_REPAIRS[
    (
        _AFTON_ENTRYPOINT,
        "/World/Looks/MetalFrame/Shader.inputs:normalmap_texture",
        "Textures/AbbyConsole_BlackMetal_Normal.png",
    )
] = {
    "entrypoint_sha256": _AFTON_ENTRYPOINT_SHA256,
    "original_file_sha256": None,
    "replacement_authored_asset": (
        "Textures/AftonConsole_BlackMetal_Normal.png"
    ),
    "replacement_asset": (
        "residential/Furniture/SofaTables/Textures/"
        "AftonConsole_BlackMetal_Normal.png"
    ),
    "replacement_sha256": (
        "fdced241c9850f0c17a4b0b0bd5fbab4bcc1cdedf6a280b3dd12ee60bfa5fe1f"
    ),
    "reference_assets": [
        {
            "asset": _AFTON_ENTRYPOINT,
            "sha256": _AFTON_ENTRYPOINT_SHA256,
            "property_path": (
                "/World/Looks/MetalFrame/Shader.inputs:diffuse_texture"
            ),
            "authored_asset": (
                "Textures/AftonConsole_BlackMetal_BaseColor.png"
            ),
            "texture_asset": (
                "residential/Furniture/SofaTables/Textures/"
                "AftonConsole_BlackMetal_BaseColor.png"
            ),
            "texture_sha256": (
                "9b23ca01a117bc32bfdc244ee48099f74ce52c9628aaf9d9eeb03a588aa54b89"
            ),
        },
        {
            "asset": _AFTON_ENTRYPOINT,
            "sha256": _AFTON_ENTRYPOINT_SHA256,
            "property_path": (
                "/World/Looks/MetalFrame/Shader.inputs:metallic_texture"
            ),
            "authored_asset": (
                "Textures/AftonConsole_BlackMetal_Metallic.png"
            ),
            "texture_asset": (
                "residential/Furniture/SofaTables/Textures/"
                "AftonConsole_BlackMetal_Metallic.png"
            ),
            "texture_sha256": (
                "f790625c24510389aba626a3b4ea80806ad5012b1f838bfbee3b69dd87e3a70e"
            ),
        },
        {
            "asset": _AFTON_ENTRYPOINT,
            "sha256": _AFTON_ENTRYPOINT_SHA256,
            "property_path": (
                "/World/Looks/MetalFrame/Shader.inputs:reflectionroughness_texture"
            ),
            "authored_asset": (
                "Textures/AftonConsole_BlackMetal_Roughness.png"
            ),
            "texture_asset": (
                "residential/Furniture/SofaTables/Textures/"
                "AftonConsole_BlackMetal_Roughness.png"
            ),
            "texture_sha256": (
                "c9ff522099715c814948fb0896f79e8b857e80a04c99bef3f2fcd611dfb0bb30"
            ),
        },
    ],
    "reference_files": [
        {
            "asset": (
                "residential/Furniture/SofaTables/.thumbs/256x256/"
                "AftonConsole.usd.png"
            ),
            "sha256": (
                "9cca058515d48b760418148a25be479e290d933a8d50ca6de1de90668e0d5c49"
            ),
        }
    ],
    "reason": (
        "MetalFrame uses AftonConsole_BlackMetal for its other inputs, "
        "but the normal input points to missing AbbyConsole; "
        "the AftonConsole normal map exists"
    ),
}


_ROXANA_ROCKER_ENTRYPOINT = (
    "residential/Furniture/Outdoors/Roxana/Roxana_Rocker.usd"
)
_ROXANA_ROCKER_ENTRYPOINT_SHA256 = (
    "220712f24c5201efdf1d65577aa89bebaa195d43ac5debe6a92a8a58c4149559"
)
_ROXANA_REFERENCE_ENTRYPOINTS = {
    "Roxana_CoffeeTable.usd": (
        "198796aba3e8a70153125df9d3c879c805bc7e6b4a06a623fc823a3ccd6e88f9"
    ),
    "Roxana_DiningBench.usd": (
        "20b6e7724b8e571ce6ae11350b871113068c83a20f68263fa2c5be7876394ec7"
    ),
    "Roxana_DiningTable.usd": (
        "d5c8000c2ab61f12535c88fb87f842748538be93627888eec8af4eda16a4d3cd"
    ),
    "Roxana_FoldingChair.usd": (
        "36fe691d9a804f2a4cb4738312509214f4d9afd76400277e0e6d8f2dfefdfea6"
    ),
    "Roxana_LoungeChair.usd": (
        "840febe7f49e97adcf8f210c9e151757679f02e92e12131c73b32dce259017f7"
    ),
    "Roxana_Ottoman.usd": (
        "c3c8de23486b33429268f465b9ea8df8c8190b08fcc3add64bc017c278f8ea85"
    ),
    "Roxana_RoundCoffeeTable.usd": (
        "35be5c7584fcd029803d7b472c6089a354e09f161fd59a4952015f10ce2d4233"
    ),
    "Roxana_RoundEndTable.usd": (
        "c91b400b146f0e17fdbfa74677aff6813dbb6199f355e8b93ebe3444c8c05a07"
    ),
    "Roxana_Sofa.usd": (
        "2148a9c8c6af4061270cd94f8e9afbd536a819d71c166de9650ff34eed0dc2e5"
    ),
    "Roxana_TeteATeteChair.usd": (
        "a8815e9bf6fc8ac07d59a11bc2a423658afc577f984ec239f2306bac1d4a0bdd"
    ),
    "Rozana_EndTable.usd": (
        "41eae2c1667a7bdfa92bd232e81362d57f329ea169820a9edd676498058e205e"
    ),
}
_ROXANA_TEXTURE_INPUTS = (
    (
        "diffuse_texture",
        "BaseColor",
        "89655f0fb344d463c001bb88a990ea0440c7be2987cd56f6cda4d1f04f324569",
    ),
    (
        "metallic_texture",
        "Metalness",
        "0ed7123d5178464f94bcd6f53ec88f498eaf09b4167003c058cae89684e5c3cf",
    ),
    (
        "normalmap_texture",
        "Normal",
        "2cbc0ee77ec4e254e8c7192e3fdb47d0922ed73fadfe1ac4ea3a4b75f4aaf6ee",
    ),
    (
        "reflectionroughness_texture",
        "Roughness",
        "b8869b7f9d3029a2e4a54c775558a6ffd5e0079e72873ff2b3837a8e167e9ef0",
    ),
)
for _input_name, _filename_role, _texture_sha256 in _ROXANA_TEXTURE_INPUTS:
    _replacement_authored_asset = (
        f"./Textures/RoxanaMetal_{_filename_role}.png"
    )
    _replacement_asset = (
        "residential/Furniture/Outdoors/Roxana/Textures/"
        f"RoxanaMetal_{_filename_role}.png"
    )
    TEXTURE_REPAIRS[
        (
            _ROXANA_ROCKER_ENTRYPOINT,
            f"/World/Looks/Screws/Shader.inputs:{_input_name}",
            f"./Textures/RoxanaRocker_Metal_{_filename_role}.png",
        )
    ] = {
        "entrypoint_sha256": _ROXANA_ROCKER_ENTRYPOINT_SHA256,
        "original_file_sha256": None,
        "replacement_authored_asset": _replacement_authored_asset,
        "replacement_asset": _replacement_asset,
        "replacement_sha256": _texture_sha256,
        "reference_assets": [
            {
                "asset": (
                    "residential/Furniture/Outdoors/Roxana/"
                    f"{_reference_entrypoint}"
                ),
                "sha256": _reference_sha256,
                "property_path": (
                    f"/World/Looks/Metal/Shader.inputs:{_input_name}"
                ),
                "authored_asset": _replacement_authored_asset,
                "texture_asset": _replacement_asset,
                "texture_sha256": _texture_sha256,
            }
            for _reference_entrypoint, _reference_sha256 in sorted(
                _ROXANA_REFERENCE_ENTRYPOINTS.items()
            )
        ],
        "reference_files": [
            {
                "asset": (
                    "residential/Furniture/Outdoors/Roxana/"
                    ".thumbs/256x256/Roxana_Rocker.usd.png"
                ),
                "sha256": (
                    "be38e4eb9c22f1e6424a735431aabc167c4046ab0bddc364d2ed43667fb51fc4"
                ),
            }
        ],
        "reason": (
            "Roxana_Rocker Screws references missing RoxanaRocker_Metal; "
            "eleven related Roxana assets use the existing RoxanaMetal "
            "file for this input"
        ),
    }


class SegmentReadError(RuntimeError):
    """Raised when a USD stage cannot be inventoried safely."""


@dataclass
class MaterialRegion:
    """One non-overlapping region of faces with one effective material."""

    key: str
    segment_id: str
    mesh_path: str
    source_mesh_path: str
    source_mesh_path_kind: str
    source_mesh_prototype_relative_path: str | None
    diagnostic_runtime_source_mesh_path: str | None
    geometry_mesh_label: str
    geometry_subset_label: str | None
    subset_path: str | None
    region_kind: str
    face_indices: tuple[int, ...]
    face_index_hash: str
    material_path: str | None
    source_material_path: str | None
    binding_relationship: str | None
    semantic_usage: str
    visibility: str
    purpose: str
    default_prim_path: str | None
    default_prim_membership: str
    selected_binding_purpose: str | None
    is_instance_proxy: bool
    material_bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    display_primvars: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    point_instancer_occurrences: list[dict[str, Any]] = field(default_factory=list)
    materials_by_binding_purpose: dict[str, Any] = field(
        repr=False, compare=False, default_factory=dict
    )
    material: Any = field(repr=False, compare=False, default=None)

    @property
    def face_count(self) -> int:
        return len(self.face_indices)


@dataclass
class SegmentInventory:
    """Regions plus diagnostics that a strict preflight can gate on."""

    regions: list[MaterialRegion] = field(default_factory=list)
    mesh_paths: list[str] = field(default_factory=list)
    empty_mesh_paths: list[str] = field(default_factory=list)
    invalid_mesh_paths: list[str] = field(default_factory=list)
    hole_faces: list[dict[str, Any]] = field(default_factory=list)
    point_instancers: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    warning_records: list[dict[str, Any]] = field(default_factory=list)


def _add_inventory_warning(
    inventory: SegmentInventory,
    code: str,
    message: str,
    **details: Any,
) -> None:
    if message not in inventory.warnings:
        inventory.warnings.append(message)
    record = {"code": code, "message": message, **_json_safe_value(details)}
    if record not in inventory.warning_records:
        inventory.warning_records.append(record)


def stage_composition_errors(stage) -> list[str]:
    """Return missing USD-layer dependencies for an opened stage.

    ``Usd.Stage.Open`` can return some meshes even when a reference or payload
    is missing.  Stop annotation if any USD dependency is unresolved, since
    segments may be missing.  Check non-USD assets through the active material
    graph so unused asset paths do not block annotation.
    """

    if not stage:
        return ["cannot validate dependencies for an invalid USD stage"]
    errors: list[str] = []
    try:
        composition_errors = list(stage.GetCompositionErrors())
    except AttributeError:
        composition_errors = []
    except Exception as exc:
        errors.append(
            "USD composition-error query failed: "
            f"{type(exc).__name__}: {exc}"
        )
        composition_errors = []
    errors.extend(
        f"USD composition error: {composition_error}"
        for composition_error in composition_errors
    )
    root_layer = stage.GetRootLayer()
    root_identifier = (
        getattr(root_layer, "realPath", "")
        or getattr(root_layer, "identifier", "")
    )
    if not root_identifier:
        errors.append("USD root layer has no resolvable identifier")
    else:
        try:
            _, _, unresolved = UsdUtils.ComputeAllDependencies(
                str(root_identifier)
            )
            for unresolved_path in sorted(
                {str(item) for item in unresolved if str(item)}
            ):
                suffix = Path(
                    unresolved_path.split("?", 1)[0]
                ).suffix.casefold()
                if suffix in USD_SUFFIXES:
                    errors.append(
                        "unresolved USD composition dependency: "
                        f"{unresolved_path}"
                    )
        except Exception as exc:
            errors.append(
                "USD dependency traversal failed: "
                f"{type(exc).__name__}: {exc}"
            )

    try:
        used_layers = list(stage.GetUsedLayers())
    except Exception as exc:
        errors.append(
            f"could not enumerate used USD layers: {type(exc).__name__}: {exc}"
        )
        used_layers = []
    for layer in used_layers:
        real_path = str(getattr(layer, "realPath", "") or "")
        if real_path and not Path(real_path).is_file():
            errors.append(f"used USD layer is unreadable: {real_path}")
    return sorted(set(errors))


@dataclass(frozen=True)
class TextureEvidence:
    """An authored image input reachable from a material surface output."""

    role: str
    authored_asset: str
    resolved_paths: tuple[str, ...]
    authoring_layer: str | None
    shader_path: str
    input_name: str
    render_contexts: tuple[str, ...]
    resolution_method: str | None
    color_space: str | None = None
    semantic_consumer_property_path: str | None = None
    semantic_consumer_property_full_name: str | None = None
    semantic_source_output_path: str | None = None
    semantic_source_output_name: str | None = None
    semantic_source_output_channel: str | None = None
    connection_endpoint_chain: tuple[dict[str, Any], ...] = ()
    evidence_source: str = "usd_shader_input"
    mdl_source_asset_path: str | None = None
    mdl_source_authored_asset: str | None = None
    mdl_source_resolution_method: str | None = None
    mdl_source_resolution_provenance: dict[str, Any] | None = None
    mdl_source_asset_sub_identifier: str | None = None
    mdl_selected_material_definition: str | None = None
    mdl_material_family: str | None = None
    mdl_gamma_expression: str | None = None
    mdl_gamma_mode: str | None = None
    mdl_gamma_is_implicit: bool | None = None
    color_space_provenance: str | None = None
    verified_semantic_repair: dict[str, Any] | None = None

    @property
    def resolved_path(self) -> str | None:
        return self.resolved_paths[0] if self.resolved_paths else None

    @property
    def is_udim(self) -> bool:
        return any(
            marker in self.authored_asset
            for marker in ("<UDIM>", "<udim>", "%04d")
        )

    @property
    def has_multiple_resolved_files(self) -> bool:
        return len(self.resolved_paths) > 1

    @property
    def used_path_fallback(self) -> bool:
        return is_path_fallback(self.resolution_method)

    def to_dict(self) -> dict[str, Any]:
        role_provenance = _texture_role_provenance(
            self.input_name,
            self.authored_asset,
            semantic_input_name=self.semantic_consumer_property_full_name,
            source_output_channel=self.semantic_source_output_channel,
            selected_role=self.role,
            material_family=self.mdl_material_family,
        )
        return {
            "role": self.role,
            **role_provenance,
            "authored_asset": self.authored_asset,
            "resolved_paths": list(self.resolved_paths),
            "authoring_layer": self.authoring_layer,
            "shader_path": self.shader_path,
            "input_name": self.input_name,
            "semantic_consumer_property_path": (
                self.semantic_consumer_property_path
            ),
            "semantic_consumer_property_full_name": (
                self.semantic_consumer_property_full_name
            ),
            "semantic_source_output_path": self.semantic_source_output_path,
            "semantic_source_output_name": self.semantic_source_output_name,
            "semantic_source_output_channel": self.semantic_source_output_channel,
            "connection_endpoint_chain": [
                _json_safe_value(edge) for edge in self.connection_endpoint_chain
            ],
            "evidence_source": self.evidence_source,
            "mdl_source_asset_path": self.mdl_source_asset_path,
            "mdl_source_authored_asset": self.mdl_source_authored_asset,
            "mdl_source_resolution_method": self.mdl_source_resolution_method,
            "mdl_source_resolution_provenance": (
                _json_safe_value(self.mdl_source_resolution_provenance)
                if self.mdl_source_resolution_provenance
                else None
            ),
            "mdl_source_asset_sub_identifier": (
                self.mdl_source_asset_sub_identifier
            ),
            "mdl_selected_material_definition": (
                self.mdl_selected_material_definition
            ),
            "mdl_material_family": self.mdl_material_family,
            "mdl_gamma_expression": self.mdl_gamma_expression,
            "mdl_gamma_mode": self.mdl_gamma_mode,
            "mdl_gamma_is_implicit": self.mdl_gamma_is_implicit,
            "color_space_provenance": self.color_space_provenance,
            "render_contexts": list(self.render_contexts),
            "resolution_method": self.resolution_method,
            "used_path_fallback": self.used_path_fallback,
            "verified_semantic_repair": (
                _json_safe_value(self.verified_semantic_repair)
                if self.verified_semantic_repair
                else None
            ),
            "resolution_provenance": (
                _json_safe_value(self.verified_semantic_repair)
                if self.verified_semantic_repair
                else _asset_resolution_provenance(
                    self.resolution_method,
                    self.resolved_paths,
                    authored_asset=self.authored_asset,
                    authoring_layer=self.authoring_layer,
                )
            ),
            "color_space": self.color_space,
            "is_udim": self.is_udim,
            "resolved_file_count": len(self.resolved_paths),
            "has_multiple_resolved_files": self.has_multiple_resolved_files,
        }


def _sha256_text(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_canonical_hash(label: str, value: Any) -> str:
    """Hash one JSON-safe value with a named, versioned domain."""

    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _sha256_text(label, serialized)


def _face_hash(face_indices: Iterable[int]) -> str:
    digest = hashlib.sha256(b"vlm-face-region-v1\0")
    for index in face_indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _integer_hash(label: bytes, values: Iterable[int]) -> str:
    digest = hashlib.sha256(label + b"\0")
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _transform_hash(transforms: Iterable[Any], indices: Iterable[int]) -> str:
    digest = hashlib.sha256(b"vlm-point-instancer-transforms-v1\0")
    transforms = tuple(transforms)
    for index in indices:
        matrix = transforms[index]
        for row in range(4):
            for column in range(4):
                digest.update(struct.pack("<d", float(matrix[row][column])))
    return digest.hexdigest()


def _json_safe_value(value: Any) -> Any:
    """Convert USD values to JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Sdf.AssetPath):
        return {
            "authored_path": value.path,
            "usd_resolved_path": value.resolvedPath or None,
        }
    if isinstance(value, Sdf.Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    try:
        return [_json_safe_value(item) for item in value]
    except TypeError:
        return {
            "conversion": "lossy_string_fallback",
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "string_value": str(value),
        }


def _contains_lossy_value_fallback(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("conversion") == "lossy_string_fallback":
            return True
        return any(_contains_lossy_value_fallback(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_lossy_value_fallback(item) for item in value)
    return False


def _encode_unsigned_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint values must be non-negative")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def encode_face_indices(face_indices: Iterable[int]) -> dict[str, Any]:
    """Compress sorted face indices."""

    values = tuple(int(value) for value in face_indices)
    if any(value < 0 for value in values):
        raise ValueError("face indices must be non-negative")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError("face indices must be strictly increasing and unique")

    previous = 0
    raw = bytearray()
    for position, value in enumerate(values):
        delta = value if position == 0 else value - previous
        raw.extend(_encode_unsigned_varint(delta))
        previous = value
    compressed = zlib.compress(bytes(raw), level=9)
    record = {
        "count": len(values),
        "uncompressed_byte_count": len(raw),
        "data": base64.b64encode(compressed).decode("ascii"),
        "sha256": _face_hash(values),
    }
    return record


def decode_face_indices(encoded: dict[str, Any]) -> tuple[int, ...]:
    """Restore compressed face indices."""

    if not isinstance(encoded, dict):
        raise TypeError("encoded face indices must be a dictionary")
    expected_count = int(encoded.get("count", -1))
    if expected_count < 0:
        raise ValueError("encoded face index count is invalid")
    try:
        compressed = base64.b64decode(encoded["data"], validate=True)
        raw = zlib.decompress(compressed)
    except (KeyError, TypeError, ValueError, binascii.Error, zlib.error) as exc:
        raise ValueError(f"invalid encoded face-index payload: {exc}") from exc
    expected_bytes = encoded.get("uncompressed_byte_count")
    if expected_bytes is not None and len(raw) != int(expected_bytes):
        raise ValueError("face-index payload byte count does not match metadata")

    values: list[int] = []
    offset = 0
    previous = 0
    while offset < len(raw):
        delta = 0
        shift = 0
        while True:
            if offset >= len(raw):
                raise ValueError("truncated face-index varint")
            byte = raw[offset]
            offset += 1
            delta |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
            if shift > 63:
                raise ValueError("face-index varint exceeds 64 bits")
        if values and delta == 0:
            raise ValueError("face-index deltas must be positive after the first value")
        value = delta if not values else previous + delta
        values.append(value)
        previous = value

    result = tuple(values)
    if len(result) != expected_count:
        raise ValueError("decoded face-index count does not match metadata")
    expected_hash = encoded.get("sha256")
    if expected_hash is not None and _face_hash(result) != expected_hash:
        raise ValueError("decoded face-index hash does not match metadata")
    return result


def _stage_prims_with_point_prototypes(stage) -> list[Any]:
    """Traverse the stage and every PointInstancer target explicitly."""

    prims_by_path = {
        str(prim.GetPath()): prim
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    }
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    visited_instancers: set[str] = set()
    while True:
        point_instancers = sorted(
            (
                prim
                for path, prim in prims_by_path.items()
                if path not in visited_instancers
                and prim.IsA(UsdGeom.PointInstancer)
            ),
            key=lambda prim: str(prim.GetPath()),
        )
        if not point_instancers:
            break
        for prim in point_instancers:
            visited_instancers.add(str(prim.GetPath()))
            instancer = UsdGeom.PointInstancer(prim)
            for target in instancer.GetPrototypesRel().GetTargets():
                target_prim = stage.GetPrimAtPath(target)
                if not target_prim:
                    continue
                for target_descendant in Usd.PrimRange(target_prim, predicate):
                    prims_by_path.setdefault(
                        str(target_descendant.GetPath()), target_descendant
                    )
    return [prims_by_path[path] for path in sorted(prims_by_path)]


def _safe_key(mesh_path: str, region_label: str, segment_id: str) -> str:
    readable = f"{mesh_path.strip('/')}__{region_label}"
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", readable).strip("_.")
    if not readable:
        readable = "segment"
    return f"{readable[:120]}__{segment_id[:12]}"


def _binding_purpose_token(purpose: str):
    if purpose == "allPurpose":
        return UsdShade.Tokens.allPurpose
    return getattr(UsdShade.Tokens, purpose, purpose)


def _computed_material(prim, purpose: str = "allPurpose"):
    purpose_token = _binding_purpose_token(purpose)
    try:
        material, relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial(purpose_token)
    except TypeError:
        if purpose != "allPurpose":
            return None, None, False
        material, relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
    return (
        material if material else None,
        relationship if relationship else None,
        True,
    )


def _canonical_material_layer_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    repository_root = Path(__file__).resolve().parents[3]
    for parent in path.parents:
        try:
            if parent.samefile(repository_root):
                return str(repository_root / path.relative_to(parent))
        except OSError:
            continue
    return value


def _material_identity(material) -> dict[str, Any] | None:
    if not material:
        return None
    prim = material.GetPrim()
    authored_specs = []
    try:
        for prim_spec in prim.GetPrimStack():
            layer = prim_spec.layer
            try:
                layer_name = str(layer.GetDisplayName())
            except Exception:
                layer_name = Path(
                    getattr(layer, "realPath", "")
                    or getattr(layer, "identifier", "")
                ).name
            layer_identifier = str(getattr(layer, "identifier", "") or "")
            layer_real_path = str(getattr(layer, "realPath", "") or "")
            if layer_identifier:
                layer_identifier = _canonical_material_layer_path(
                    layer_identifier
                )
            if layer_real_path:
                layer_real_path = _canonical_material_layer_path(
                    layer_real_path
                )
            authored_specs.append(
                {
                    "layer_display_name": layer_name,
                    "layer_identifier": layer_identifier or None,
                    "layer_real_path": layer_real_path or None,
                    "spec_path": str(prim_spec.path),
                }
            )
    except Exception:
        authored_specs = []
    return {
        "visible_material_path": str(material.GetPath()),
        "authored_specs": authored_specs,
    }


def material_record_id_from_identity(identity: Any) -> str | None:
    """Identify a material by its source layers and prim paths."""

    if not isinstance(identity, Mapping):
        return None
    authored_specs = identity.get("authored_specs")
    if not isinstance(authored_specs, list):
        return None
    authored_values = [
        value
        for spec in authored_specs
        if isinstance(spec, Mapping)
        for value in (
            spec.get("layer_real_path")
            or spec.get("layer_identifier")
            or spec.get("layer_display_name"),
            spec.get("spec_path"),
        )
    ]
    if authored_values and all(
        isinstance(value, str) and bool(value) for value in authored_values
    ) and len(authored_values) == 2 * len(authored_specs):
        return _sha256_text("vlm-material-record-v3", *authored_values)
    if authored_specs:
        return None
    visible_material_path = identity.get("visible_material_path")
    if not isinstance(visible_material_path, str) or not visible_material_path:
        return None
    return _sha256_text("vlm-material-record-v3", visible_material_path)


def material_record_id(material) -> str | None:
    """Return a stable identifier for a composed material prim."""

    return material_record_id_from_identity(_material_identity(material))


def _relationship_binding_purpose(relationship) -> str | None:
    if not relationship:
        return None
    name = str(relationship.GetName())
    tokens = name.split(":")
    if tokens[:2] != ["material", "binding"]:
        return None
    if len(tokens) == 2:
        return "allPurpose"
    if tokens[2] == "collection":
        if len(tokens) >= 5 and tokens[3] in {"full", "preview"}:
            return tokens[3]
        return "allPurpose"
    if len(tokens) == 3 and tokens[2] in {"full", "preview"}:
        return tokens[2]
    return None


def _material_binding_record(
    purpose: str, material, relationship, query_supported: bool
) -> dict[str, Any]:
    material_path = str(material.GetPath()) if material else None
    source_material_path = (
        _source_mesh_path(material.GetPrim()) if material else None
    )
    relationship_path = str(relationship.GetPath()) if relationship else None
    try:
        targets = [str(path) for path in relationship.GetTargets()] if relationship else []
    except Exception:
        targets = []
    try:
        binding_strength = str(
            UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relationship)
        ) if relationship else None
    except Exception:
        binding_strength = None
    try:
        resolved_target_path = (
            str(
                UsdShade.MaterialBindingAPI.GetResolvedTargetPathFromBindingRel(
                    relationship
                )
            )
            if relationship
            else None
        )
    except Exception:
        resolved_target_path = None
    authored_binding_purpose = _relationship_binding_purpose(relationship)
    malformed = bool(query_supported and relationship and not material)
    return {
        "query_purpose": purpose,
        "query_token": str(_binding_purpose_token(purpose)),
        "material_path": material_path,
        "source_material_path": source_material_path,
        "material_record_id": material_record_id(material),
        "binding_relationship": relationship_path,
        "binding_relationship_targets": targets,
        "resolved_target_path": resolved_target_path,
        "authored_binding_purpose": authored_binding_purpose,
        "used_all_purpose_fallback": bool(
            material
            and purpose != "allPurpose"
            and authored_binding_purpose == "allPurpose"
        ),
        "binding_strength": binding_strength,
        "query_supported": query_supported,
        "malformed_winning_binding": malformed,
        "is_bound": bool(material),
        "effective_binding_status": (
            "bound"
            if material
            else "unbound_invalid_target_fallback"
            if malformed
            else "unbound"
        ),
        "renderer_fallback": (
            {
                "classification": "invalid_binding_target_is_unbound",
                "effective_material_path": None,
                "material_name_inference_allowed": False,
                "repair_required": True,
            }
            if malformed
            else None
        ),
    }


def _computed_material_bindings(prim):
    bindings: dict[str, dict[str, Any]] = {}
    materials: dict[str, Any] = {}
    relationships: dict[str, Any] = {}
    for purpose in MATERIAL_BINDING_PURPOSE_PRIORITY:
        material, relationship, query_supported = _computed_material(prim, purpose)
        materials[purpose] = material
        relationships[purpose] = relationship
        bindings[purpose] = _material_binding_record(
            purpose, material, relationship, query_supported
        )

    selected_purpose = None
    for purpose in MATERIAL_BINDING_PURPOSE_PRIORITY:
        record = bindings[purpose]
        if not materials[purpose]:
            continue
        if purpose in {"full", "preview"} and record[
            "used_all_purpose_fallback"
        ]:
            continue
        selected_purpose = purpose
        break
    selected_material = materials[selected_purpose] if selected_purpose else None
    selected_relationship = (
        relationships[selected_purpose] if selected_purpose else None
    )
    return (
        selected_material,
        selected_relationship,
        selected_purpose,
        bindings,
        materials,
    )


def _imageable_metadata(prim) -> tuple[str, str]:
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return "inherited", "default"
    visibility = imageable.ComputeVisibility()
    purpose = imageable.ComputePurpose()
    return str(visibility or "inherited"), str(purpose or "default")


def _render_state_time_errors(prim) -> list[str]:
    """Reject animated ancestral state when extraction evaluates Default time."""

    errors: list[str] = []
    prim_path = str(prim.GetPath())
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        for label, attribute in (
            ("visibility", imageable.GetVisibilityAttr()),
            ("purpose", imageable.GetPurposeAttr()),
        ):
            metadata = _attribute_time_metadata(attribute)
            if (
                metadata["value_might_be_time_varying"]
                or metadata["authored_time_sample_count"]
            ):
                errors.append(
                    f"time-varying {label} is unsupported at {prim_path}: "
                    f"samples={metadata['authored_time_samples']}"
                )

    xformable = UsdGeom.Xformable(prim)
    if xformable:
        attributes = [("xformOpOrder", xformable.GetXformOpOrderAttr())]
        try:
            for op in xformable.GetOrderedXformOps():
                attribute = op.GetAttr()
                attributes.append((str(attribute.GetName()), attribute))
        except Exception:
            pass
        for label, attribute in attributes:
            metadata = _attribute_time_metadata(attribute)
            if (
                metadata["value_might_be_time_varying"]
                or metadata["authored_time_sample_count"]
            ):
                errors.append(
                    f"time-varying transform {label} is unsupported at "
                    f"{prim_path}: samples={metadata['authored_time_samples']}"
                )
    return errors


def _default_prim_metadata(prim) -> tuple[str | None, str]:
    stage = prim.GetStage()
    default_prim = stage.GetDefaultPrim() if stage else None
    if not default_prim:
        return None, "stage_has_no_default_prim"
    default_path = default_prim.GetPath()
    prim_path = prim.GetPath()
    if prim_path == default_path:
        membership = "is_default_prim"
    elif prim_path.HasPrefix(default_path):
        membership = "descendant_of_default_prim"
    else:
        membership = "outside_default_prim"
    return str(default_path), membership


def _attribute_time_metadata(attribute) -> dict[str, Any]:
    try:
        time_samples = [float(value) for value in attribute.GetTimeSamples()]
    except Exception:
        time_samples = []
    try:
        might_vary = bool(attribute.ValueMightBeTimeVarying())
    except Exception:
        might_vary = bool(time_samples)
    return {
        "value_might_be_time_varying": might_vary,
        "authored_time_sample_count": len(time_samples),
        "authored_time_samples": time_samples,
    }


def _primvar_domain_count(prim, interpolation: str) -> int | None:
    """Return the expected flattened element count for supported Gprim data."""

    interpolation = interpolation.casefold()
    if interpolation == "constant":
        return 1
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        if interpolation == "uniform":
            return len(counts) if counts is not None else None
        if interpolation in {"vertex", "varying"}:
            return len(points) if points is not None else None
        if interpolation == "facevarying":
            return sum(int(count) for count in counts) if counts is not None else None
    if prim.IsA(UsdGeom.PointInstancer):
        proto_indices = UsdGeom.PointInstancer(prim).GetProtoIndicesAttr().Get(
            Usd.TimeCode.Default()
        )
        if interpolation in {"uniform", "vertex", "varying"}:
            return len(proto_indices) if proto_indices is not None else None
    return None


def authored_perspective_camera_hint(stage) -> dict[str, Any] | None:
    """Return the entrypoint's saved object-view direction when authored."""

    custom_data = stage.GetRootLayer().customLayerData
    if not isinstance(custom_data, Mapping):
        return None
    camera_settings = custom_data.get("cameraSettings")
    if camera_settings is None:
        return None
    if not isinstance(camera_settings, Mapping):
        raise ValueError("root-layer cameraSettings is not a mapping")
    perspective = camera_settings.get("Perspective")
    if perspective is None:
        return None
    if not isinstance(perspective, Mapping):
        raise ValueError("root-layer Perspective camera setting is not a mapping")

    def vector(name: str) -> tuple[float, float, float]:
        value = perspective.get(name)
        try:
            components = tuple(float(component) for component in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"root-layer Perspective camera {name} is not a finite 3-vector"
            ) from exc
        if len(components) != 3 or not all(math.isfinite(item) for item in components):
            raise ValueError(
                f"root-layer Perspective camera {name} is not a finite 3-vector"
            )
        return components

    position = vector("position")
    target = vector("target")
    direction = tuple(left - right for left, right in zip(position, target))
    direction_length = math.sqrt(sum(value * value for value in direction))
    if direction_length <= 1.0e-10:
        raise ValueError("root-layer Perspective camera direction is zero")
    direction = tuple(value / direction_length for value in direction)

    up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    if up_axis not in {"Y", "Z"}:
        raise ValueError(f"unsupported stage up axis for camera hint: {up_axis!r}")
    up_index = 1 if up_axis == "Y" else 2
    horizontal = list(direction)
    horizontal[up_index] = 0.0
    horizontal_length = math.sqrt(sum(value * value for value in horizontal))
    if horizontal_length <= 1.0e-10:
        raise ValueError("root-layer Perspective camera has no horizontal direction")
    horizontal_direction = tuple(value / horizontal_length for value in horizontal)

    record = {
        "source": "root_layer.customLayerData.cameraSettings.Perspective",
        "position": list(position),
        "target": list(target),
        "direction": list(direction),
        "horizontal_direction": list(horizontal_direction),
        "stage_up_axis": up_axis,
        "horizontal_fraction": horizontal_length,
    }
    record["hint_sha256"] = _json_canonical_hash(
        "authored-perspective-camera-hint-v1", record
    )
    return record


def stage_color_management_metadata(stage) -> dict[str, Any]:
    """Read the stage color settings without applying a transform."""

    def authored_metadata(name: str) -> tuple[bool | None, Any]:
        try:
            authored = bool(stage.HasAuthoredMetadata(name))
        except Exception:
            return None, None
        if not authored:
            return False, None
        try:
            return True, stage.GetMetadata(name)
        except Exception:
            return True, None

    configuration_authored, authored_configuration = authored_metadata(
        "colorConfiguration"
    )
    try:
        effective_configuration = stage.GetColorConfiguration()
        configuration_api_status = "complete"
        configuration_error = None
    except Exception as exc:
        effective_configuration = None
        configuration_api_status = "unavailable"
        configuration_error = f"{type(exc).__name__}: {exc}"

    def asset_path_fields(value: Any) -> tuple[str | None, str | None]:
        if value is None:
            return None, None
        authored_path = str(getattr(value, "path", "") or "") or None
        resolved_path = str(getattr(value, "resolvedPath", "") or "") or None
        return authored_path, resolved_path

    authored_path, authored_resolved_path = asset_path_fields(
        authored_configuration
    )
    effective_path, effective_resolved_path = asset_path_fields(
        effective_configuration
    )
    resolved_path = effective_resolved_path or authored_resolved_path
    if not resolved_path:
        absolute_candidate = effective_path or authored_path
        if absolute_candidate and Path(absolute_candidate).is_absolute():
            resolved_path = str(Path(absolute_candidate).resolve())
    resolved_file = Path(resolved_path) if resolved_path else None
    if resolved_file is not None and resolved_file.is_file():
        resolution_status = "resolved_file"
        content_sha256 = _file_sha256(resolved_file)
        size_bytes = resolved_file.stat().st_size
    elif effective_path or authored_path:
        resolution_status = "unresolved"
        content_sha256 = None
        size_bytes = None
    else:
        resolution_status = "not_configured"
        content_sha256 = None
        size_bytes = None
    configuration = {
        "api_status": configuration_api_status,
        "metadata_authored": configuration_authored,
        "authored_value_type": (
            type(authored_configuration).__name__
            if authored_configuration is not None
            else None
        ),
        "authored_asset_path": authored_path,
        "authored_resolved_path": authored_resolved_path,
        "effective_value_type": (
            type(effective_configuration).__name__
            if effective_configuration is not None
            else None
        ),
        "effective_asset_path": effective_path,
        "effective_resolved_path": effective_resolved_path,
        "dependency_resolved_path": resolved_path,
        "resolution_status": resolution_status,
        "resolved_file_sha256": content_sha256,
        "resolved_file_size_bytes": size_bytes,
        "error": configuration_error,
    }

    system_authored, authored_system = authored_metadata("colorManagementSystem")
    try:
        effective_system_value = stage.GetColorManagementSystem()
        effective_system = str(effective_system_value or "") or None
        system_api_status = "complete"
        system_error = None
    except Exception as exc:
        effective_system_value = None
        effective_system = None
        system_api_status = "unavailable"
        system_error = f"{type(exc).__name__}: {exc}"
    color_management_system = {
        "api_status": system_api_status,
        "metadata_authored": system_authored,
        "authored_value_type": (
            type(authored_system).__name__ if authored_system is not None else None
        ),
        "authored_value": (
            str(authored_system or "") or None
            if authored_system is not None
            else None
        ),
        "effective_value_type": (
            type(effective_system_value).__name__
            if effective_system_value is not None
            else None
        ),
        "effective_value": effective_system,
        "error": system_error,
    }
    result = {
        "stage_color_configuration": configuration,
        "stage_color_management_system": color_management_system,
    }
    result["stage_color_management_sha256"] = _sha256_text(
        "stage-color-management-v1",
        json.dumps(
            _json_safe_value(result),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
    )
    return result


def _display_primvar_record(mesh_prim, name: str) -> dict[str, Any] | None:
    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
    try:
        primvar = primvars_api.FindPrimvarWithInheritance(name)
    except Exception:
        primvar = primvars_api.GetPrimvar(name)
    if not primvar:
        return None
    attribute = primvar.GetAttr()
    try:
        has_value = bool(attribute and attribute.HasValue())
    except Exception:
        has_value = attribute.Get() is not None if attribute else False
    if not attribute or not has_value:
        return None
    try:
        indices = primvar.GetIndices(Usd.TimeCode.Default())
    except TypeError:
        indices = primvar.GetIndices()
    except Exception:
        indices = None
    try:
        flattened = primvar.ComputeFlattened(Usd.TimeCode.Default())
    except TypeError:
        flattened = primvar.ComputeFlattened()
    except Exception:
        flattened = None
    try:
        authored_color_space = (
            str(attribute.GetMetadata("colorSpace"))
            if attribute.HasAuthoredMetadata("colorSpace")
            else None
        )
    except Exception:
        authored_color_space = None
    try:
        effective_color_space = str(attribute.GetColorSpace() or "") or None
    except Exception:
        effective_color_space = authored_color_space
    source_prim_path = str(attribute.GetPrim().GetPath())
    interpolation = str(primvar.GetInterpolation() or "constant")
    expected_element_count = _primvar_domain_count(mesh_prim, interpolation)
    try:
        flattened_element_count = len(flattened) if flattened is not None else None
    except TypeError:
        flattened_element_count = 1 if flattened is not None else None
    is_indexed = bool(primvar.IsIndexed())
    try:
        unauthored_values_index = int(primvar.GetUnauthoredValuesIndex())
    except Exception:
        unauthored_values_index = -1
    indices_list = [int(value) for value in indices] if indices is not None else None
    cardinality_errors = []
    if (
        expected_element_count is not None
        and flattened_element_count != expected_element_count
    ):
        cardinality_errors.append(
            f"flattened element count {flattened_element_count} != "
            f"expected {expected_element_count} for {interpolation}"
        )
    if (
        is_indexed
        and expected_element_count is not None
        and (indices_list is None or len(indices_list) != expected_element_count)
    ):
        cardinality_errors.append(
            f"index count {len(indices_list) if indices_list is not None else None} "
            f"!= expected {expected_element_count} for {interpolation}"
        )
    record = {
        "primvar_path": str(attribute.GetPath()),
        "source_prim_path": source_prim_path,
        "is_inherited": source_prim_path != str(mesh_prim.GetPath()),
        "interpolation": interpolation,
        "element_size": int(primvar.GetElementSize() or 1),
        "is_indexed": is_indexed,
        "unauthored_values_index": unauthored_values_index,
        "expected_element_count": expected_element_count,
        "flattened_element_count": flattened_element_count,
        "cardinality_valid": not cardinality_errors,
        "cardinality_errors": cardinality_errors,
        "authored_color_space": authored_color_space,
        "effective_color_space": effective_color_space,
        "authored_values": _json_safe_value(attribute.Get(Usd.TimeCode.Default())),
        "authored_indices": indices_list,
        "computed_flattened_values": _json_safe_value(flattened),
    }
    record.update(_attribute_time_metadata(attribute))
    try:
        indices_attribute = primvar.GetIndicesAttr()
    except Exception:
        indices_attribute = None
    if indices_attribute:
        record["indices_time_metadata"] = _attribute_time_metadata(
            indices_attribute
        )
    return record


def _source_prim_provenance(prim) -> dict[str, Any]:
    """Return durable instance-source identity plus volatile diagnostics.

    OpenUSD assigns hidden prototype roots names such as ``__Prototype_12``
    at runtime.  The number is explicitly not durable across fresh stage
    opens, so it must never participate in persisted identity or matching.
    For an instance proxy, retain only the path relative to that prototype as
    stable provenance and expose the complete runtime path in a field whose
    name marks it as diagnostic.  The visible composed proxy path is the
    fail-closed stable fallback if an older runtime cannot identify the
    prototype root.
    """

    visible_path = str(prim.GetPath())
    try:
        is_instance_proxy = bool(prim.IsInstanceProxy())
    except Exception:
        is_instance_proxy = False
    if not is_instance_proxy:
        return {
            "source_path": visible_path,
            "source_path_kind": "composed_prim_path",
            "prototype_relative_path": None,
            "diagnostic_runtime_source_path": None,
        }

    try:
        source = prim.GetPrimInPrototype()
    except Exception:
        source = None
    if not source:
        return {
            "source_path": visible_path,
            "source_path_kind": "composed_instance_proxy_path_fallback",
            "prototype_relative_path": None,
            "diagnostic_runtime_source_path": None,
        }

    runtime_source_path = str(source.GetPath())
    prototype_root = None
    current = source
    while current:
        try:
            if current.IsPrototype():
                prototype_root = current
                break
        except Exception:
            break
        try:
            parent = current.GetParent()
        except Exception:
            break
        if not parent or parent == current:
            break
        current = parent

    relative_path = None
    if prototype_root:
        try:
            relative_path = str(
                source.GetPath().MakeRelativePath(prototype_root.GetPath())
            )
        except Exception:
            root_path = str(prototype_root.GetPath()).rstrip("/")
            prefix = root_path + "/"
            if runtime_source_path.startswith(prefix):
                relative_path = runtime_source_path[len(prefix) :]
    if not relative_path:
        components = runtime_source_path.split("/")
        if len(components) > 2 and components[1].startswith("__Prototype_"):
            relative_path = "/".join(components[2:])
    if relative_path:
        if relative_path.startswith("./"):
            relative_path = relative_path[2:]
        relative_path = "/" + relative_path.lstrip("/")
        return {
            "source_path": f"prototype-relative:{relative_path}",
            "source_path_kind": "prototype_relative",
            "prototype_relative_path": relative_path,
            "diagnostic_runtime_source_path": runtime_source_path,
        }

    return {
        "source_path": visible_path,
        "source_path_kind": "composed_instance_proxy_path_fallback",
        "prototype_relative_path": None,
        "diagnostic_runtime_source_path": runtime_source_path,
    }


def _source_mesh_path(prim) -> str:
    """Return the saved source path."""

    return str(_source_prim_provenance(prim)["source_path"])


def _make_region(
    *,
    mesh_prim,
    subset_prim,
    region_kind: str,
    face_indices: Iterable[int],
) -> MaterialRegion:
    indices = tuple(sorted(int(index) for index in face_indices))
    mesh_path = str(mesh_prim.GetPath())
    subset_path = str(subset_prim.GetPath()) if subset_prim else None
    binding_prim = subset_prim if subset_prim else mesh_prim
    (
        material,
        relationship,
        selected_binding_purpose,
        material_bindings,
        materials_by_binding_purpose,
    ) = _computed_material_bindings(binding_prim)
    material_path = str(material.GetPath()) if material else None
    source_material_path = None
    if material:
        material_prim = material.GetPrim()
        source_material_path = _source_mesh_path(material_prim)
    relationship_path = str(relationship.GetPath()) if relationship else None
    face_index_hash = _face_hash(indices)
    region_label = subset_prim.GetName() if subset_prim else region_kind
    segment_id = _sha256_text(
        "vlm-segment-v2",
        mesh_path,
        subset_path or "",
        region_kind,
        face_index_hash,
    )
    visibility, purpose = _imageable_metadata(mesh_prim)
    default_prim_path, default_prim_membership = _default_prim_metadata(mesh_prim)
    source_mesh_provenance = _source_prim_provenance(mesh_prim)
    return MaterialRegion(
        key=_safe_key(mesh_path, str(region_label), segment_id),
        segment_id=segment_id,
        mesh_path=mesh_path,
        source_mesh_path=str(source_mesh_provenance["source_path"]),
        source_mesh_path_kind=str(
            source_mesh_provenance["source_path_kind"]
        ),
        source_mesh_prototype_relative_path=(
            source_mesh_provenance["prototype_relative_path"]
        ),
        diagnostic_runtime_source_mesh_path=(
            source_mesh_provenance["diagnostic_runtime_source_path"]
        ),
        geometry_mesh_label=str(mesh_prim.GetName()),
        geometry_subset_label=(
            str(subset_prim.GetName()) if subset_prim else None
        ),
        subset_path=subset_path,
        region_kind=region_kind,
        face_indices=indices,
        face_index_hash=face_index_hash,
        material_path=material_path,
        source_material_path=source_material_path,
        binding_relationship=relationship_path,
        semantic_usage=str(region_label if subset_prim else mesh_prim.GetName()),
        visibility=visibility,
        purpose=purpose,
        default_prim_path=default_prim_path,
        default_prim_membership=default_prim_membership,
        selected_binding_purpose=selected_binding_purpose,
        is_instance_proxy=bool(mesh_prim.IsInstanceProxy()),
        material_bindings=material_bindings,
        display_primvars={
            "displayColor": _display_primvar_record(mesh_prim, "displayColor"),
            "displayOpacity": _display_primvar_record(mesh_prim, "displayOpacity"),
        },
        materials_by_binding_purpose=materials_by_binding_purpose,
        material=material,
    )


def enumerate_material_regions(stage) -> SegmentInventory:
    """Enumerate every mesh face exactly once, including instance proxies.

    Material-binding subsets are treated as the specific regions they author.
    Any unassigned faces form a separate remainder region with the mesh's
    effective binding.  Consequently, a directly bound mesh with no subsets
    produces one region, not overlapping ``mesh`` and ``mesh_whole`` records.
    """

    if not stage:
        raise SegmentReadError("Cannot enumerate segments on an invalid USD stage")

    inventory = SegmentInventory()
    keys: set[str] = set()
    segment_ids: set[str] = set()
    checked_render_state_paths: set[str] = set()
    for prim in _stage_prims_with_point_prototypes(stage):
        if prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Mesh):
            inventory.errors.append(
                f"unsupported non-Mesh Gprim at {prim.GetPath()}: {prim.GetTypeName()}"
            )
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue

        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            ancestor_path = str(ancestor.GetPath())
            if ancestor_path not in checked_render_state_paths:
                checked_render_state_paths.add(ancestor_path)
                inventory.errors.extend(_render_state_time_errors(ancestor))
            ancestor = ancestor.GetParent()

        mesh_path = str(prim.GetPath())
        inventory.mesh_paths.append(mesh_path)
        mesh = UsdGeom.Mesh(prim)
        for attribute_label, attribute in (
            ("points", mesh.GetPointsAttr()),
            ("faceVertexCounts", mesh.GetFaceVertexCountsAttr()),
            ("faceVertexIndices", mesh.GetFaceVertexIndicesAttr()),
        ):
            time_metadata = _attribute_time_metadata(attribute)
            if (
                time_metadata["value_might_be_time_varying"]
                or time_metadata["authored_time_sample_count"]
            ):
                inventory.errors.append(
                    f"time-varying mesh {attribute_label} is unsupported at "
                    f"{mesh_path}: samples={time_metadata['authored_time_samples']}"
                )
        counts = mesh.GetFaceVertexCountsAttr().Get()
        face_count = len(counts) if counts is not None else 0
        if face_count == 0:
            inventory.empty_mesh_paths.append(mesh_path)
            inventory.warnings.append(f"mesh has no faces: {mesh_path}")
            continue

        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or indices is None:
            inventory.invalid_mesh_paths.append(mesh_path)
            inventory.errors.append(f"mesh topology is unreadable: {mesh_path}")
            continue
        topology_ok, topology_reason = UsdGeom.Mesh.ValidateTopology(
            indices, counts, len(points)
        )
        if not topology_ok:
            inventory.invalid_mesh_paths.append(mesh_path)
            inventory.errors.append(
                f"invalid mesh topology at {mesh_path}: {topology_reason}"
            )
            continue
        if any(
            not math.isfinite(component)
            for point in points
            for component in point
        ):
            inventory.invalid_mesh_paths.append(mesh_path)
            inventory.errors.append(f"mesh has non-finite points: {mesh_path}")
            continue

        hole_indices_attribute = mesh.GetHoleIndicesAttr()
        hole_time = _attribute_time_metadata(hole_indices_attribute)
        raw_hole_indices = hole_indices_attribute.Get(Usd.TimeCode.Default())
        authored_hole_indices = tuple(
            int(index) for index in (raw_hole_indices or ())
        )
        if (
            hole_time["value_might_be_time_varying"]
            or hole_time["authored_time_sample_count"]
        ):
            inventory.errors.append(
                f"time-varying mesh holeIndices are unsupported at {mesh_path}: "
                f"samples={hole_time['authored_time_samples']}"
            )
        if len(authored_hole_indices) != len(set(authored_hole_indices)):
            inventory.errors.append(f"duplicate hole face indices at {mesh_path}")
        invalid_holes = sorted(
            {
                index
                for index in authored_hole_indices
                if index < 0 or index >= face_count
            }
        )
        if invalid_holes:
            inventory.errors.append(
                f"out-of-range hole face indices at {mesh_path}: "
                f"{invalid_holes[:10]}"
            )
        hole_indices = tuple(
            sorted(
                {
                    index
                    for index in authored_hole_indices
                    if 0 <= index < face_count
                }
            )
        )
        hole_set = set(hole_indices)
        if hole_indices or hole_time["authored_time_sample_count"]:
            inventory.hole_faces.append(
                {
                    "mesh_path": mesh_path,
                    "face_count": len(hole_indices),
                    "face_index_hash": _face_hash(hole_indices),
                    "face_indices": encode_face_indices(hole_indices),
                    **hole_time,
                }
            )

        try:
            subsets = list(
                UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
            )
        except Exception as exc:
            inventory.errors.append(
                f"could not read materialBind subsets at {mesh_path}: {exc}"
            )
            subsets = []

        assigned: set[int] = set()
        authored_assigned: set[int] = set()
        for subset in sorted(subsets, key=lambda item: str(item.GetPath())):
            subset_prim = subset.GetPrim()
            element_type_attribute = subset.GetElementTypeAttr()
            element_type = element_type_attribute.Get()
            if str(element_type) != str(UsdGeom.Tokens.face):
                inventory.errors.append(
                    f"materialBind subset is not a face subset at "
                    f"{subset_prim.GetPath()}: elementType={element_type!s}"
                )
                continue
            indices_attribute = subset.GetIndicesAttr()
            indices_time = _attribute_time_metadata(indices_attribute)
            if (
                indices_time["value_might_be_time_varying"]
                or indices_time["authored_time_sample_count"]
            ):
                inventory.errors.append(
                    f"time-varying materialBind subset is unsupported at "
                    f"{subset_prim.GetPath()}: "
                    f"samples={indices_time['authored_time_samples']}"
                )
            raw_indices = indices_attribute.Get(Usd.TimeCode.Default())
            indices = tuple(int(index) for index in (raw_indices or ()))
            if len(indices) != len(set(indices)):
                inventory.errors.append(
                    f"duplicate face indices on {subset_prim.GetPath()}"
                )
            invalid = sorted({index for index in indices if index < 0 or index >= face_count})
            if invalid:
                inventory.errors.append(
                    f"out-of-range face indices on {subset_prim.GetPath()}: {invalid[:10]}"
                )
            valid_authored_indices = tuple(
                sorted({index for index in indices if 0 <= index < face_count})
            )
            duplicates = sorted(
                authored_assigned.intersection(valid_authored_indices)
            )
            if duplicates:
                inventory.errors.append(
                    f"overlapping materialBind subsets at {mesh_path}; faces {duplicates[:10]}"
                )
            authored_assigned.update(valid_authored_indices)
            valid_indices = tuple(
                index for index in valid_authored_indices if index not in hole_set
            )
            assigned.update(valid_indices)
            subset_holes = sorted(set(valid_authored_indices).intersection(hole_set))
            if subset_holes:
                inventory.warnings.append(
                    f"materialBind subset includes non-rendered hole faces at "
                    f"{subset_prim.GetPath()}: {subset_holes[:10]}"
                )
            if not valid_indices:
                inventory.warnings.append(
                    f"empty materialBind subset: {subset_prim.GetPath()}"
                )
                continue
            inventory.regions.append(
                _make_region(
                    mesh_prim=prim,
                    subset_prim=subset_prim,
                    region_kind="subset",
                    face_indices=valid_indices,
                )
            )

        remainder = tuple(
            index
            for index in range(face_count)
            if index not in assigned and index not in hole_set
        )
        if remainder:
            inventory.regions.append(
                _make_region(
                    mesh_prim=prim,
                    subset_prim=None,
                    region_kind="remainder" if subsets else "whole",
                    face_indices=remainder,
                )
            )

        if subsets:
            try:
                family_type = str(
                    UsdShade.MaterialBindingAPI(
                        prim
                    ).GetMaterialBindSubsetsFamilyType()
                )
            except Exception:
                family_type = ""
            authored_remainder = tuple(
                index
                for index in range(face_count)
                if index not in authored_assigned
            )
            if family_type == str(UsdGeom.Tokens.partition) and authored_remainder:
                _add_inventory_warning(
                    inventory,
                    "incomplete_material_partition_mesh_binding_fallback",
                    "partition materialBind subsets do not cover every face at "
                    f"{mesh_path}; uncovered renderable faces use the exact "
                    "composed mesh binding",
                    mesh_path=mesh_path,
                    authored_uncovered_face_count=len(authored_remainder),
                    authored_uncovered_face_index_hash=_face_hash(
                        authored_remainder
                    ),
                    fallback_region_kind="remainder",
                    repair_required=True,
                )

        if hole_indices and len(hole_indices) == face_count:
            inventory.warnings.append(
                f"mesh has no renderable faces because every face is a hole: {mesh_path}"
            )

    binding_errors: set[str] = set()
    for region in inventory.regions:
        for binding_purpose, binding in region.material_bindings.items():
            if not binding.get("query_supported", True):
                binding_errors.add(
                    "material binding-purpose query is unsupported by this "
                    f"OpenUSD runtime at {region.subset_path or region.mesh_path}: "
                    f"purpose={binding_purpose}"
                )
            if binding.get("malformed_winning_binding"):
                _add_inventory_warning(
                    inventory,
                    "malformed_material_binding_unbound_fallback",
                    "winning material binding does not resolve to a Material at "
                    f"{region.subset_path or region.mesh_path}; OpenUSD leaves "
                    "the region unbound and no material name is inferred",
                    segment_id=region.segment_id,
                    region_path=region.subset_path or region.mesh_path,
                    binding_purpose=binding_purpose,
                    binding_relationship=binding.get(
                        "binding_relationship"
                    ),
                    authored_targets=binding.get(
                        "binding_relationship_targets"
                    ),
                    resolved_target_path=binding.get(
                        "resolved_target_path"
                    ),
                    renderer_fallback=binding.get("renderer_fallback"),
                )
        if region.material is None:
            for primvar_name, primvar_record in region.display_primvars.items():
                if primvar_record and not primvar_record.get(
                    "cardinality_valid", True
                ):
                    _add_inventory_warning(
                        inventory,
                        "invalid_unbound_display_primvar_ignored",
                        f"unbound segment has invalid {primvar_name} evidence at "
                        f"{region.mesh_path}: "
                        f"{primvar_record.get('cardinality_errors')}; neutral "
                        "isolated geometry remains available",
                        segment_id=region.segment_id,
                        mesh_path=region.mesh_path,
                        primvar_name=primvar_name,
                        cardinality_errors=primvar_record.get(
                            "cardinality_errors"
                        ),
                    )
                if primvar_record and (
                    primvar_record.get("value_might_be_time_varying")
                    or primvar_record.get("authored_time_sample_count")
                ):
                    _add_inventory_warning(
                        inventory,
                        "time_varying_unbound_display_primvar_snapshot",
                        f"unbound segment has time-varying {primvar_name} "
                        f"evidence at {region.mesh_path}; extraction records "
                        "the default-time snapshot",
                        segment_id=region.segment_id,
                        mesh_path=region.mesh_path,
                        primvar_name=primvar_name,
                    )
        if region.key in keys:
            inventory.errors.append(f"duplicate segment key: {region.key}")
        if region.segment_id in segment_ids:
            inventory.errors.append(f"duplicate segment id: {region.segment_id}")
        keys.add(region.key)
        segment_ids.add(region.segment_id)

    inventory.errors.extend(sorted(binding_errors))

    _map_point_instancer_occurrences(stage, inventory)

    return inventory


def _map_point_instancer_occurrences(stage, inventory: SegmentInventory) -> None:
    """Map PointInstancer instances to their prototype regions."""

    regions_by_path = [
        (Sdf.Path(region.mesh_path), region) for region in inventory.regions
    ]
    stage_prims = _stage_prims_with_point_prototypes(stage)
    point_instancer_prims = [
        prim for prim in stage_prims if prim.IsA(UsdGeom.PointInstancer)
    ]
    point_instancer_paths = [prim.GetPath() for prim in point_instancer_prims]
    nested_by_parent: dict[str, list[str]] = {}
    for prim in point_instancer_prims:
        parent_path = str(prim.GetPath())
        nested = set()
        for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
            for candidate_path in point_instancer_paths:
                if candidate_path != prim.GetPath() and candidate_path.HasPrefix(target):
                    nested.add(str(candidate_path))
        nested_by_parent[parent_path] = sorted(nested)
        if nested:
            inventory.errors.append(
                "nested PointInstancer occurrence flattening is unsupported: "
                f"{parent_path} contains {', '.join(sorted(nested))}"
            )

    for prim in point_instancer_prims:

        instancer = UsdGeom.PointInstancer(prim)
        instancer_path = str(prim.GetPath())
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            for error in _render_state_time_errors(ancestor):
                if error not in inventory.errors:
                    inventory.errors.append(error)
            ancestor = ancestor.GetParent()
        for attribute_label, getter_name in (
            ("protoIndices", "GetProtoIndicesAttr"),
            ("positions", "GetPositionsAttr"),
            ("orientations", "GetOrientationsAttr"),
            ("scales", "GetScalesAttr"),
            ("ids", "GetIdsAttr"),
            ("invisibleIds", "GetInvisibleIdsAttr"),
            ("velocities", "GetVelocitiesAttr"),
            ("angularVelocities", "GetAngularVelocitiesAttr"),
            ("accelerations", "GetAccelerationsAttr"),
        ):
            getter = getattr(instancer, getter_name, None)
            if getter is None:
                continue
            attribute = getter()
            time_metadata = _attribute_time_metadata(attribute)
            if (
                time_metadata["value_might_be_time_varying"]
                or time_metadata["authored_time_sample_count"]
            ):
                inventory.errors.append(
                    f"time-varying PointInstancer {attribute_label} is "
                    f"unsupported at {instancer_path}: "
                    f"samples={time_metadata['authored_time_samples']}"
                )
        instancer_primvars = {}
        try:
            authored_primvars = UsdGeom.PrimvarsAPI(prim).GetPrimvars()
        except Exception:
            authored_primvars = []
        for primvar in sorted(
            authored_primvars,
            key=lambda item: str(item.GetPrimvarName()),
        ):
            primvar_name = str(primvar.GetPrimvarName())
            record = _display_primvar_record(prim, primvar_name)
            if record is not None:
                instancer_primvars[primvar_name] = record
                if not record.get("cardinality_valid", True):
                    inventory.errors.append(
                        f"PointInstancer primvar has invalid cardinality at "
                        f"{instancer_path}.primvars:{primvar_name}: "
                        f"{record.get('cardinality_errors')}"
                    )
                if (
                    record.get("value_might_be_time_varying")
                    or record.get("authored_time_sample_count")
                ):
                    inventory.errors.append(
                        f"time-varying PointInstancer primvar is unsupported at "
                        f"{instancer_path}.primvars:{primvar_name}"
                    )
        targets = list(instancer.GetPrototypesRel().GetTargets())
        raw_proto_indices = instancer.GetProtoIndicesAttr().Get()
        proto_indices = tuple(int(value) for value in (raw_proto_indices or ()))
        positions = instancer.GetPositionsAttr().Get()
        ids_value = instancer.GetIdsAttr().Get()
        authored_ids = tuple(int(value) for value in (ids_value or ()))

        if not targets:
            inventory.errors.append(
                f"PointInstancer has no prototype targets: {instancer_path}"
            )
        if positions is None or len(positions) != len(proto_indices):
            inventory.errors.append(
                f"PointInstancer position/protoIndices length mismatch: {instancer_path}"
            )
        if authored_ids and len(authored_ids) != len(proto_indices):
            inventory.errors.append(
                f"PointInstancer ids/protoIndices length mismatch: {instancer_path}"
            )
        invalid_proto_indices = sorted(
            {
                value
                for value in proto_indices
                if value < 0 or value >= len(targets)
            }
        )
        if invalid_proto_indices:
            inventory.errors.append(
                f"PointInstancer has invalid prototype indices at {instancer_path}: "
                f"{invalid_proto_indices[:10]}"
            )

        try:
            mask_value = tuple(
                bool(value)
                for value in instancer.ComputeMaskAtTime(Usd.TimeCode.Default())
            )
        except Exception as exc:
            inventory.errors.append(
                f"PointInstancer mask query failed at {instancer_path}: {exc}"
            )
            mask_value = ()
        if mask_value and len(mask_value) != len(proto_indices):
            inventory.errors.append(
                f"PointInstancer mask/protoIndices length mismatch: {instancer_path}"
            )
        active_mask = (
            mask_value
            if len(mask_value) == len(proto_indices)
            else (True,) * len(proto_indices)
        )
        instance_ids = (
            authored_ids
            if len(authored_ids) == len(proto_indices)
            else tuple(range(len(proto_indices)))
        )

        try:
            transforms = tuple(
                instancer.ComputeInstanceTransformsAtTime(
                    Usd.TimeCode.Default(),
                    Usd.TimeCode.Default(),
                    UsdGeom.PointInstancer.IncludeProtoXform,
                    UsdGeom.PointInstancer.IgnoreMask,
                )
            )
        except Exception as exc:
            inventory.errors.append(
                f"PointInstancer transform query failed at {instancer_path}: {exc}"
            )
            transforms = ()
        if transforms and len(transforms) != len(proto_indices):
            inventory.errors.append(
                f"PointInstancer transform/protoIndices length mismatch: {instancer_path}"
            )

        slot_records = []
        for slot, target in enumerate(targets):
            target_regions = [
                region for path, region in regions_by_path if path.HasPrefix(target)
            ]
            if not target_regions:
                inventory.errors.append(
                    f"PointInstancer prototype has no supported mesh region: "
                    f"{instancer_path} -> {target}"
                )
            occurrence_indices = tuple(
                index
                for index, prototype_index in enumerate(proto_indices)
                if prototype_index == slot and active_mask[index]
            )
            occurrence_ids = tuple(instance_ids[index] for index in occurrence_indices)
            occurrence_record = {
                "instancer_path": instancer_path,
                "time_code": "default",
                "prototype_target": str(target),
                "prototype_index": slot,
                "occurrence_count": len(occurrence_indices),
                "occurrence_indices": list(occurrence_indices),
                "occurrence_index_hash": _integer_hash(
                    b"vlm-point-instancer-occurrences-v1", occurrence_indices
                ),
                "instance_ids_authored": bool(authored_ids),
                "instance_ids": list(occurrence_ids),
                "instance_id_hash": _integer_hash(
                    b"vlm-point-instancer-ids-v1", occurrence_ids
                ),
                "transform_hash": (
                    _transform_hash(transforms, occurrence_indices)
                    if len(transforms) == len(proto_indices)
                    else None
                ),
            }
            for region in target_regions:
                region.point_instancer_occurrences.append(dict(occurrence_record))
            slot_records.append(
                {
                    **occurrence_record,
                    "region_ids": [region.segment_id for region in target_regions],
                }
            )

        inventory.point_instancers.append(
            {
                "prim_path": instancer_path,
                "nested_point_instancers": nested_by_parent.get(
                    instancer_path, []
                ),
                "occurrence_representation": (
                    "canonical_prototype_occurrences_not_flattened"
                    if nested_by_parent.get(instancer_path)
                    else "canonical_prototype_occurrences"
                ),
                "primvars": instancer_primvars,
                "time_code": "default",
                "prototype_targets": [str(target) for target in targets],
                "proto_indices": list(proto_indices),
                "active_mask": list(active_mask),
                "instance_ids_authored": bool(authored_ids),
                "instance_ids": list(instance_ids),
                "all_transform_hash": (
                    _transform_hash(transforms, range(len(transforms)))
                    if len(transforms) == len(proto_indices)
                    else None
                ),
                "total_occurrence_count": len(proto_indices),
                "active_occurrence_count": sum(active_mask),
                "masked_occurrence_count": len(proto_indices) - sum(active_mask),
                "prototype_slots": slot_records,
            }
        )


def _asset_paths(value: Any) -> tuple[Sdf.AssetPath, ...]:
    if isinstance(value, Sdf.AssetPath):
        return (value,) if value.path else ()
    if isinstance(value, (str, bytes)) or value is None:
        return ()
    try:
        values = tuple(value)
    except TypeError:
        return ()
    return tuple(
        item for item in values if isinstance(item, Sdf.AssetPath) and item.path
    )


_EXACT_MDL_TEXTURE_INPUT_ROLES = {
    # OmniGlass and the NVIDIA GlassWithVolume template.
    "glass_color_texture": "albedo",
    "roughness_texture": "roughness",
    "reflection_color_texture": "specular",
    "transmission_color_texture": "transmission_color",
    "normal_map_texture": "normal",
    "cutout_opacity_texture": "opacity",
    "diffuse_reflection_weight_image": "diffuse_weight",
    "diffuse_reflection_color_image": "albedo",
    "diffuse_reflection_roughness_image": "diffuse_roughness",
    "metalness_image": "metallic",
    "specular_reflection_weight_image": "specular_weight",
    "specular_reflection_color_image": "specular",
    "specular_reflection_roughness_image": "specular_roughness",
    "specular_reflection_anisotropy_image": "specular_anisotropy",
    "specular_reflection_anisotropy_rotation_image": (
        "specular_anisotropy_rotation"
    ),
    "specular_transmission_weight_image": "transmission_weight",
    "specular_transmission_color_image": "transmission_color",
    "specular_transmission_scattering_depth_image": "transmission_depth",
    "specular_transmission_scattering_color_image": (
        "transmission_scattering_color"
    ),
    "specular_transmission_scatter_anisotropy_image": (
        "transmission_scatter_anisotropy"
    ),
    "subsurface_weight_image": "subsurface_weight",
    "subsurface_transmission_color_image": "subsurface_transmission_color",
    "subsurface_scattering_color_image": "subsurface_scattering_color",
    "subsurface_scale_image": "subsurface_scale",
    "subsurface_anisotropy_image": "subsurface_anisotropy",
    "coat_weight_image": "coat_weight",
    "coat_color_image": "coat_color",
    "coat_roughness_image": "coat_roughness",
    "coat_anisotropy_image": "coat_anisotropy",
    "coat_anisotropy_rotation_image": "coat_anisotropy_rotation",
    "coat_affect_color_image": "coat_affect_color",
    "coat_affect_roughness_image": "coat_affect_roughness",
    "coat_normal_image": "coat_normal",
    "specular_retro_reflection_weight_image": "retroreflection_weight",
    "specular_retro_reflection_color_image": "retroreflection_color",
    "specular_retro_reflection_roughness_image": "retroreflection_roughness",
    "emission_weight_image": "emission_weight",
    "emission_intensity_image": "emission_intensity",
    "emission_color_image": "emissive",
    "emission_temperature_image": "emission_temperature",
    "thin_film_thickness_image": "thin_film_thickness",
    "geometry_opacity_image": "opacity",
    "geometry_normal_image": "normal",
    "geometry_displacement_image": "displacement",
    "geometry_displacement_scale_image": "displacement_scale",
}
_OMNIGLASS_TEXTURE_INPUTS = frozenset(
    {
        "cutout_opacity_texture",
        "glass_color_texture",
        "normal_map_texture",
        "reflection_color_texture",
        "roughness_texture",
    }
)
_GLASS_WITH_VOLUME_TEXTURE_INPUTS = frozenset(
    {
        "normal_map_texture",
        "reflection_color_texture",
        "roughness_texture",
        "transmission_color_texture",
    }
)
_OMNISURFACE_TEXTURE_INPUTS = frozenset(
    set(_EXACT_MDL_TEXTURE_INPUT_ROLES)
    - _OMNIGLASS_TEXTURE_INPUTS
    - _GLASS_WITH_VOLUME_TEXTURE_INPUTS
)
_EXACT_MDL_TEXTURE_INPUTS_BY_FAMILY = {
    "OmniGlass": _OMNIGLASS_TEXTURE_INPUTS,
    "NvidiaGlassWithVolume": _GLASS_WITH_VOLUME_TEXTURE_INPUTS,
    "OmniSurface": _OMNISURFACE_TEXTURE_INPUTS,
    "OmniSurfacePreset": _OMNISURFACE_TEXTURE_INPUTS,
}
_STRICT_EXACT_TEXTURE_ROLE_MATERIAL_FAMILIES = frozenset(
    {
        "OmniGlass",
        "OmniSurface",
        "OmniSurfacePreset",
        "NvidiaGlassWithVolume",
    }
)


def _exact_mdl_texture_input_role(
    text: str | None, material_family: str | None = None
) -> str | None:
    if not text:
        return None
    base_name = str(text).rsplit(":", 1)[-1].casefold()
    allowed_inputs = _EXACT_MDL_TEXTURE_INPUTS_BY_FAMILY.get(
        material_family
    )
    if allowed_inputs is not None and base_name not in allowed_inputs:
        return None
    return _EXACT_MDL_TEXTURE_INPUT_ROLES.get(base_name)


def _role_from_semantic_text(text: str) -> str | None:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(text))
    normalized = separated.casefold()
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", normalized) if token
    }
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if (
        tokens.intersection({"orm", "arm", "mrao"})
        or "occlusionroughnessmetallic" in compact
        or {"occlusion", "roughness", "metallic"}.issubset(tokens)
    ):
        return "orm"
    normal_semantics = (
        tokens.intersection({"normal", "normalmap", "nrm", "nor"})
        or "normalmap" in compact
    )
    if normal_semantics and (
        "detail" in tokens or "detailnormal" in compact
    ):
        return "detail_normal"
    if normal_semantics:
        return "normal"
    if tokens.intersection({"opacity", "alpha", "transparency"}):
        return "opacity"
    if (
        tokens.intersection({"emissive", "emission"})
        and "mask" in tokens
    ) or "emissivemask" in compact or "emissionmask" in compact:
        return "emissive_mask"
    if tokens.intersection({"emissive", "emission"}):
        return "emissive"
    if (
        tokens.intersection({"displacement", "height", "bump"})
        or "displacement" in compact
    ):
        return "displacement"
    if tokens.intersection({"roughness", "rough"}) or "roughness" in compact:
        return "roughness"
    if (
        tokens.intersection({"metallic", "metalness"})
        or "metallic" in compact
        or "metalness" in compact
    ):
        return "metallic"
    if tokens.intersection(
        {"ambientocclusion", "occlusion", "ao"}
    ):
        return "occlusion"
    if tokens.intersection({"specular", "spec"}):
        return "specular"
    if (
        tokens.intersection({"basecolor", "albedo", "diffuse", "diff"})
        or "basecolor" in compact
        or "glasscolor" in compact
        or {"glass", "color"}.issubset(tokens)
        or {"base", "color"}.issubset(tokens)
    ):
        return "albedo"
    return None


def _filename_suffix_role(authored_asset: str) -> str | None:
    filename = Path(authored_asset.replace("<UDIM>", "1001")).name.casefold()
    stem = Path(filename).stem
    stem = re.sub(r"[._-]1\d{3}$", "", stem)
    while True:
        stripped = re.sub(
            r"[._-](?:[1248]k|dx|gl|opengl|directx|v?\d+)$", "", stem
        )
        if stripped == stem:
            break
        stem = stripped
    match = re.search(
        r"(?:^|[._-])"
        r"(ambientocclusion|ambient_occlusion|basecolor|base_color|"
        r"diffuse|albedo|normalmap|normal|roughness|rough|metallic|"
        r"metalness|opacity|alpha|emissive|emission|displacement|"
        r"height|emissivemask|emissive_mask|emissionmask|emission_mask|"
        r"diff|nor|nrm|ao|base|a|n|r|m|o|e|h)$",
        stem,
    )
    if not match:
        return None
    return {
        "ambientocclusion": "occlusion",
        "ambient_occlusion": "occlusion",
        "basecolor": "albedo",
        "base_color": "albedo",
        "diffuse": "albedo",
        "albedo": "albedo",
        "normalmap": "normal",
        "normal": "normal",
        "roughness": "roughness",
        "rough": "roughness",
        "metallic": "metallic",
        "metalness": "metallic",
        "opacity": "opacity",
        "alpha": "opacity",
        "emissive": "emissive",
        "emission": "emissive",
        "emissivemask": "emissive_mask",
        "emissive_mask": "emissive_mask",
        "emissionmask": "emissive_mask",
        "emission_mask": "emissive_mask",
        "displacement": "displacement",
        "height": "displacement",
        "diff": "albedo",
        "nor": "normal",
        "nrm": "normal",
        "ao": "occlusion",
        "base": "albedo",
        "a": "albedo",
        "n": "normal",
        "r": "roughness",
        "m": "metallic",
        "o": "opacity",
        "e": "emissive",
        "h": "displacement",
    }[match.group(1)]


def _texture_role_provenance(
    input_name: str,
    authored_asset: str,
    *,
    semantic_input_name: str | None = None,
    source_output_channel: str | None = None,
    selected_role: str | None = None,
    material_family: str | None = None,
) -> dict[str, Any]:
    """Explain semantic-vs-filename role selection without overriding USD."""

    exact_roles_required = (
        material_family in _STRICT_EXACT_TEXTURE_ROLE_MATERIAL_FAMILIES
    )
    direct_input_role = _exact_mdl_texture_input_role(
        input_name, material_family
    )
    if direct_input_role is None and not exact_roles_required:
        direct_input_role = _role_from_semantic_text(input_name)
    consumer_role = (
        _exact_mdl_texture_input_role(
            semantic_input_name, material_family
        )
        or (
            None
            if exact_roles_required
            else _role_from_semantic_text(semantic_input_name)
        )
        if semantic_input_name
        else None
    )
    input_role = consumer_role or direct_input_role
    filename_role = _filename_suffix_role(authored_asset)
    if filename_role is None:
        filename_role = _role_from_semantic_text(Path(authored_asset).name)
    inferred_role = input_role or (
        "unknown" if exact_roles_required else filename_role or "unknown"
    )
    selected_role = selected_role or inferred_role
    packed_map_role_use = bool(
        filename_role == "orm"
        and input_role in {"occlusion", "roughness", "metallic"}
    )
    alpha_channel_role_use = bool(
        source_output_channel == "a"
        and input_role == "opacity"
    )
    conflict = bool(
        input_role
        and filename_role
        and input_role != filename_role
        and input_role in {"albedo", "normal", "opacity"}
        and not packed_map_role_use
        and not alpha_channel_role_use
    )
    return {
        "input_inferred_role": input_role,
        "direct_input_inferred_role": direct_input_role,
        "semantic_consumer_inferred_role": consumer_role,
        "filename_inferred_role": filename_role,
        "packed_map_role_use": packed_map_role_use,
        "alpha_channel_role_use": alpha_channel_role_use,
        "role_selection_source": (
            "downstream_consumer"
            if consumer_role and semantic_input_name != input_name
            else "connected_input"
            if direct_input_role
            else "unrecognized_exact_family_input"
            if exact_roles_required
            else "authored_filename" if filename_role else "unknown"
        ),
        "authored_role_conflict": conflict,
        "selected_role": selected_role,
    }


def _classify_texture(
    input_name: str,
    authored_asset: str,
    *,
    material_family: str | None = None,
) -> str:
    return _texture_role_provenance(
        input_name,
        authored_asset,
        material_family=material_family,
    )["selected_role"]


def _authoring_layer(attribute) -> Any | None:
    try:
        for spec in attribute.GetPropertyStack(Usd.TimeCode.Default()):
            if getattr(spec, "default", None) is not None:
                return spec.layer
    except Exception:
        pass
    try:
        stack = attribute.GetPropertyStack()
        if stack:
            return stack[0].layer
    except Exception:
        pass
    return None


def _casefold_existing_path(path: Path) -> Path | None:
    """Recover an exact path with capitalization mistakes, if unambiguous."""

    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        return None
    current = Path(path.anchor)
    for part in path.parts[1:]:
        exact = current / part
        if exact.exists():
            current = exact
            continue
        try:
            matches = [
                child for child in current.iterdir() if child.name.casefold() == part.casefold()
            ]
        except OSError:
            return None
        if len(matches) != 1:
            return None
        current = matches[0]
    return current.resolve() if current.is_file() else None


def _unique_full_path_casefold_file(
    path: Path,
) -> tuple[Path, list[dict[str, str]]] | None:
    """Find a unique path that differs only in capitalization."""

    if not path.is_absolute():
        return None
    current = Path(path.anchor)
    mismatches: list[dict[str, str]] = []
    for ordinal, part in enumerate(path.parts[1:]):
        exact = current / part
        if exact.exists():
            current = exact
            continue
        if not current.is_dir():
            return None
        try:
            matches = [
                child
                for child in current.iterdir()
                if child.name.casefold() == part.casefold()
            ]
        except OSError:
            return None
        if len(matches) != 1:
            return None
        selected = matches[0]
        if ordinal < len(path.parts[1:]) - 1 and not selected.is_dir():
            return None
        mismatches.append(
            {
                "authored_component": part,
                "resolved_component": selected.name,
                "parent_path": str(current),
            }
        )
        current = selected
    if not mismatches or not current.is_file():
        return None
    return current.resolve(), mismatches


def _unique_same_directory_casefold_file(path: Path) -> Path | None:
    """Find a unique filename match that differs only in capitalization.

    Unlike :func:`_casefold_existing_path`, this does not search through a
    differently-capitalized directory hierarchy.  The authored parent
    directory must exist exactly and contain exactly one regular file whose
    filename is equal under Unicode case-folding.
    """

    if path.is_file() or not path.is_absolute() or not path.parent.is_dir():
        return None
    try:
        matches = [
            child
            for child in path.parent.iterdir()
            if child.is_file() and child.name.casefold() == path.name.casefold()
        ]
    except OSError:
        return None
    if len(matches) != 1:
        return None
    return matches[0].resolve()


_UDIM_MARKERS = ("<UDIM>", "<udim>", "%04d")


def _single_udim_marker(value: str) -> str | None:
    matches = [marker for marker in _UDIM_MARKERS if marker in value]
    if len(matches) != 1 or value.count(matches[0]) != 1:
        return None
    return matches[0]


def _udim_tile_id(authored_asset: str, resolved_path: Path) -> int | None:
    """Return the exact UDIM tile substituted into one resolved filename."""

    marker = _single_udim_marker(authored_asset)
    authored_name = Path(authored_asset).name
    if marker is None or marker not in authored_name:
        return None
    pattern = re.compile(
        "^"
        + re.escape(authored_name).replace(
            re.escape(marker), r"(?P<tile>[0-9]{4})"
        )
        + "$"
    )
    match = pattern.fullmatch(resolved_path.name)
    if match is None:
        return None
    tile_id = int(match.group("tile"))
    return tile_id if 1001 <= tile_id <= 1999 else None


def _expand_udim(path: Path) -> tuple[Path, ...]:
    text = str(path)
    contains_udim_marker = any(marker in text for marker in _UDIM_MARKERS)
    marker = _single_udim_marker(text)
    if not contains_udim_marker:
        return (path.resolve(),) if path.is_file() else ()
    if marker is None or marker not in path.name or not path.parent.is_dir():
        return ()
    try:
        tiles = [
            (tile_id, item.resolve())
            for item in path.parent.iterdir()
            if item.is_file()
            for tile_id in [_udim_tile_id(text, item)]
            if tile_id is not None
        ]
    except OSError:
        return ()
    return tuple(item for _, item in sorted(tiles))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_source_repair(
    authored: str, root_layer
) -> tuple[Path, dict[str, Any]] | None:
    root_identifier = _layer_identifier(root_layer)
    if not root_identifier:
        return None
    root_text = Path(root_identifier).as_posix()
    for (asset_suffix, exact_authored), repair in sorted(
        SOURCE_REPAIRS.items()
    ):
        if authored != exact_authored:
            continue
        suffix_marker = f"/{asset_suffix}"
        if root_text != asset_suffix and not root_text.endswith(suffix_marker):
            continue
        dataset_root_text = (
            ""
            if root_text == asset_suffix
            else root_text[: -len(asset_suffix)]
        )
        dataset_root = Path(dataset_root_text or ".")
        expected_sha256 = str(repair["source_sha256"])
        matching_paths = [
            dataset_root / str(source_asset)
            for source_asset in repair["matching_source_assets"]
        ]
        if not matching_paths or any(
            not path.is_file() or _file_sha256(path) != expected_sha256
            for path in matching_paths
        ):
            return None
        source_path = dataset_root / str(repair["source_asset"])
        if source_path not in matching_paths or not source_path.is_file():
            return None
        return source_path.resolve(), {
            "classification": "verified_exact_sibling_content_repair",
            "source_asset": str(repair["source_asset"]),
            "source_path": str(source_path.resolve()),
            "source_sha256": expected_sha256,
            "matching_source_paths": [
                str(path.resolve()) for path in matching_paths
            ],
            "reason": (
                "the asset and authored path match a known repair; "
                "all listed files for this material have matching SHA-256 hashes"
            ),
            "requires_upstream_source_repair": True,
        }
    return None


def _verified_semantic_texture_repair(
    authored: str,
    property_path: str,
    resolved_paths: Iterable[str],
    authoring_layer,
    root_layer,
) -> tuple[str, tuple[str, ...], dict[str, Any]] | None:
    """Apply a property correction after checking the manifest's file hashes.

    This is intentionally not a filename classifier.  A repair is admitted
    only while the target entrypoint, exact authored property/value, original
    file state, replacement file, audited sibling entrypoints/textures, and
    supporting visual/source files all retain their recorded hashes.
    """

    root_identifier = _layer_identifier(root_layer)
    if not root_identifier:
        return None
    root_text = Path(root_identifier).as_posix()
    original_paths = tuple(Path(path).resolve() for path in resolved_paths)
    for (asset_suffix, exact_property, exact_authored), repair in sorted(
        TEXTURE_REPAIRS.items()
    ):
        if property_path != exact_property or authored != exact_authored:
            continue
        suffix_marker = f"/{asset_suffix}"
        if root_text != asset_suffix and not root_text.endswith(suffix_marker):
            continue
        dataset_root_text = (
            "" if root_text == asset_suffix else root_text[: -len(asset_suffix)]
        )
        dataset_root = Path(dataset_root_text or ".")
        entrypoint = dataset_root / asset_suffix
        if (
            not entrypoint.is_file()
            or _file_sha256(entrypoint)
            != str(repair["entrypoint_sha256"])
        ):
            return None

        authoring_identifier = _layer_identifier(authoring_layer)
        if not authoring_identifier:
            return None
        expected_authoring_asset = repair.get("authoring_layer_asset")
        expected_authoring_sha256 = repair.get("authoring_layer_sha256")
        authoring_path = Path(authoring_identifier).resolve()
        if (
            expected_authoring_asset is not None
            or expected_authoring_sha256 is not None
        ):
            expected_authoring_path = (
                dataset_root / str(expected_authoring_asset)
            ).resolve()
            if (
                expected_authoring_asset is None
                or expected_authoring_sha256 is None
                or authoring_path != expected_authoring_path
                or not authoring_path.is_file()
                or _file_sha256(authoring_path)
                != str(expected_authoring_sha256)
            ):
                return None
        original_authored_path = Path(authored)
        original_expected_path = (
            original_authored_path
            if original_authored_path.is_absolute()
            else Path(authoring_identifier).resolve().parent
            / original_authored_path
        ).resolve()

        expected_original_sha256 = repair.get("original_file_sha256")
        if expected_original_sha256 is None:
            if original_paths or original_expected_path.is_file():
                return None
            original_sha256 = None
        else:
            if (
                len(original_paths) != 1
                or original_paths[0] != original_expected_path
                or not original_paths[0].is_file()
                or _file_sha256(original_paths[0])
                != str(expected_original_sha256)
            ):
                return None
            original_sha256 = str(expected_original_sha256)

        replacement_authored = str(repair["replacement_authored_asset"])
        verified_material_asset = repair.get(
            "replacement_verified_material_asset"
        )
        if verified_material_asset is not None:
            verified_material_asset = str(verified_material_asset)
            manifest = MATERIAL_FILES.get(verified_material_asset)
            replacement_path = material_manifest_path(
                material_root_from_environment(), verified_material_asset
            )
            if (
                manifest is None
                or replacement_path is None
                or not verify_material_file(replacement_path, manifest)
                or str(manifest["sha256"])
                != str(repair["replacement_sha256"])
            ):
                return None
            replacement_path = replacement_path.resolve()
        else:
            replacement_path = (
                dataset_root / str(repair["replacement_asset"])
            ).resolve()
            if (
                not replacement_path.is_file()
                or _file_sha256(replacement_path)
                != str(repair["replacement_sha256"])
            ):
                return None

            authored_replacement_path = (
                Path(authoring_identifier).resolve().parent
                / replacement_authored
            ).resolve()
            if authored_replacement_path != replacement_path:
                return None

        reference_assets = list(repair.get("reference_assets") or [])
        if not reference_assets and verified_material_asset is None:
            return None
        for reference_asset in reference_assets:
            reference_entrypoint = dataset_root / str(reference_asset["asset"])
            reference_texture = dataset_root / str(reference_asset["texture_asset"])
            if (
                not reference_entrypoint.is_file()
                or _file_sha256(reference_entrypoint)
                != str(reference_asset["sha256"])
                or not reference_texture.is_file()
                or _file_sha256(reference_texture)
                != str(reference_asset["texture_sha256"])
            ):
                return None
        reference_files = list(repair.get("reference_files") or [])
        if not reference_files:
            return None
        for reference_file in reference_files:
            path = dataset_root / str(reference_file["asset"])
            if (
                not path.is_file()
                or _file_sha256(path) != str(reference_file["sha256"])
            ):
                return None

        repair_dependency_paths = [
            {
                "path": str(entrypoint.resolve()),
                "sha256": str(repair["entrypoint_sha256"]),
                "role": "target_entrypoint",
            },
            {
                "path": str(original_expected_path),
                "sha256": original_sha256,
                "role": "original_authored_file",
                "expected_status": (
                    "present" if original_sha256 is not None else "missing"
                ),
            },
            {
                "path": str(replacement_path),
                "sha256": str(repair["replacement_sha256"]),
                "role": (
                    "pinned_external_material_texture"
                    if verified_material_asset is not None
                    else "semantic_replacement_file"
                ),
            },
        ]
        if expected_authoring_asset is not None:
            repair_dependency_paths.append(
                {
                    "path": str(authoring_path),
                    "sha256": str(expected_authoring_sha256),
                    "role": "target_property_authoring_layer",
                }
            )
        for reference_asset in reference_assets:
            repair_dependency_paths.extend(
                [
                    {
                        "path": str(
                            (dataset_root / str(reference_asset["asset"])).resolve()
                        ),
                        "sha256": str(reference_asset["sha256"]),
                        "role": "sibling_entrypoint",
                    },
                    {
                        "path": str(
                            (
                                dataset_root
                                / str(reference_asset["texture_asset"])
                            ).resolve()
                        ),
                        "sha256": str(reference_asset["texture_sha256"]),
                        "role": "sibling_texture",
                    },
                ]
            )
        for reference_file in reference_files:
            repair_dependency_paths.append(
                {
                    "path": str(
                        (dataset_root / str(reference_file["asset"])).resolve()
                    ),
                    "sha256": str(reference_file["sha256"]),
                    "role": "supporting_source_or_thumbnail",
                }
            )

        provenance = {
            "classification": TEXTURE_REPAIR_METHOD,
            "target_asset": asset_suffix,
            "target_entrypoint_path": str(entrypoint.resolve()),
            "target_entrypoint_sha256": str(repair["entrypoint_sha256"]),
            "target_property_path": property_path,
            "target_authoring_layer_path": str(authoring_path),
            "target_authoring_layer_sha256": (
                str(expected_authoring_sha256)
                if expected_authoring_sha256 is not None
                else None
            ),
            "original_authored_asset": authored,
            "original_resolved_paths": [str(path) for path in original_paths],
            "original_file_sha256": original_sha256,
            "replacement_authored_asset": replacement_authored,
            "replacement_path": str(replacement_path),
            "replacement_sha256": str(repair["replacement_sha256"]),
            "replacement_verified_material_asset": verified_material_asset,
            "reference_assets": _json_safe_value(reference_assets),
            "reference_files": _json_safe_value(reference_files),
            "repair_dependency_paths": _json_safe_value(
                repair_dependency_paths
            ),
            "reason": str(repair["reason"]),
            "requires_upstream_source_repair": True,
        }
        return replacement_authored, (str(replacement_path),), provenance
    return None


def _asset_resolution_provenance(
    method: str | None,
    resolved_paths: Iterable[str],
    *,
    authored_asset: str | None = None,
    authoring_layer: Any | None = None,
) -> dict[str, Any] | None:
    if method == PATH_CASE_REPAIR_METHOD:
        paths = [Path(path) for path in resolved_paths]
        if len(paths) != 1 or not paths[0].is_file() or not authored_asset:
            return None
        layer_identifier = (
            str(authoring_layer)
            if isinstance(authoring_layer, (str, Path))
            else _layer_identifier(authoring_layer)
        )
        authored_path = Path(authored_asset)
        if authored_path.is_absolute():
            expected_path = authored_path
            layer_path = None
        else:
            if not layer_identifier:
                return None
            layer_path = Path(layer_identifier).resolve()
            expected_path = layer_path.parent / authored_path
        match = _unique_full_path_casefold_file(expected_path)
        if match is None or match[0] != paths[0].resolve():
            return None
        source_path, mismatched_components = match
        source_sha256 = _file_sha256(source_path)
        return {
            "classification": PATH_CASE_REPAIR_METHOD,
            "authored_asset": authored_asset,
            "authoring_layer": str(layer_path) if layer_path else None,
            "expected_path": str(expected_path),
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "mismatched_components": mismatched_components,
            "repair_dependency_paths": [
                {
                    "path": str(expected_path),
                    "sha256": None,
                    "role": "original_case_mismatched_path",
                    "expected_status": "missing",
                },
                {
                    "path": str(source_path),
                    "sha256": source_sha256,
                    "role": "unique_componentwise_casefold_match",
                    "expected_status": "present",
                },
            ],
            "reason": (
                "each mismatched path component has one match ignoring case; "
                "the resolved file's SHA-256 hash is recorded"
            ),
            "requires_upstream_source_repair": True,
        }
    if method == CASE_REPAIR_METHOD:
        paths = [Path(path) for path in resolved_paths]
        if len(paths) != 1 or not paths[0].is_file() or not authored_asset:
            return None
        source_path = paths[0].resolve()
        authored_path = Path(authored_asset)
        if authored_path.is_absolute():
            expected_path = authored_path
            layer_path = None
        else:
            layer_identifier = (
                str(authoring_layer)
                if isinstance(authoring_layer, (str, Path))
                else _layer_identifier(authoring_layer)
            )
            if not layer_identifier:
                return None
            layer_path = Path(layer_identifier).resolve()
            expected_path = layer_path.parent / authored_path
        if (
            not expected_path.is_absolute()
            or expected_path.parent.resolve() != source_path.parent
            or expected_path.name == source_path.name
            or expected_path.name.casefold() != source_path.name.casefold()
            or _unique_same_directory_casefold_file(expected_path) != source_path
        ):
            return None
        return {
            "classification": CASE_REPAIR_METHOD,
            "authored_asset": authored_asset,
            "authoring_layer": str(layer_path) if layer_path else None,
            "expected_path": str(expected_path),
            "source_path": str(source_path),
            "source_sha256": _file_sha256(source_path),
            "repair_dependency_paths": [
                {
                    "path": str(expected_path),
                    "sha256": None,
                    "role": "original_case_mismatched_path",
                    "expected_status": "missing",
                },
                {
                    "path": str(source_path),
                    "sha256": _file_sha256(source_path),
                    "role": "unique_casefold_match",
                    "expected_status": "present",
                },
            ],
            "reason": (
                "the parent directory contains one filename match ignoring "
                "case; the resolved file's SHA-256 hash is recorded"
            ),
            "requires_upstream_source_repair": True,
        }
    if method != SOURCE_REPAIR_METHOD:
        return None
    paths = [Path(path) for path in resolved_paths]
    if len(paths) != 1 or not paths[0].is_file():
        return None
    source_path = paths[0].resolve()
    source_sha256 = _file_sha256(source_path)
    for repair in SOURCE_REPAIRS.values():
        source_asset = str(repair["source_asset"])
        if (
            source_sha256 != str(repair["source_sha256"])
            or not source_path.as_posix().endswith(f"/{source_asset}")
        ):
            continue
        dataset_root_text = source_path.as_posix()[: -len(source_asset)]
        dataset_root = Path(dataset_root_text)
        layer_identifier = (
            str(authoring_layer)
            if isinstance(authoring_layer, (str, Path))
            else _layer_identifier(authoring_layer)
        )
        if not authored_asset or not layer_identifier:
            return None
        authored_path = Path(authored_asset)
        expected_path = (
            authored_path
            if authored_path.is_absolute()
            else Path(layer_identifier).resolve().parent / authored_path
        ).resolve()
        if expected_path.is_file():
            return None
        matching_paths = [
            (dataset_root / path).resolve()
            for path in repair["matching_source_assets"]
        ]
        return {
            "classification": SOURCE_REPAIR_METHOD,
            "source_asset": source_asset,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "matching_source_paths": [str(path) for path in matching_paths],
            "repair_dependency_paths": [
                {
                    "path": str(expected_path),
                    "sha256": None,
                    "role": "original_missing_authored_file",
                    "expected_status": "missing",
                },
                *(
                    {
                        "path": str(path),
                        "sha256": source_sha256,
                        "role": "same_material_sibling_source",
                        "expected_status": "present",
                    }
                    for path in matching_paths
                ),
            ],
            "reason": (
                "the asset and authored path match a known repair; "
                "all listed files for this material have matching SHA-256 hashes"
            ),
            "requires_upstream_source_repair": True,
        }
    return None


def _resolve_asset_path(
    asset_path: Sdf.AssetPath,
    authoring_layer,
    root_layer,
) -> tuple[tuple[str, ...], str | None]:
    authored = asset_path.path
    if not authored:
        return (), None

    external_source = resolve_external_mdl_source(
        authored,
        _layer_identifier(root_layer),
    )
    if external_source:
        source_path, _ = external_source
        return (str(source_path),), None
    if external_mdl_repair_is_applicable(
        authored,
        _layer_identifier(root_layer),
    ):
        return (), None
    if "://" in authored:
        return (), None

    candidates: list[tuple[Path, str]] = []
    authoring_layer_relative_candidate: Path | None = None
    if asset_path.resolvedPath:
        candidates.append((Path(asset_path.resolvedPath), "usd_resolved_path"))
    authored_path = Path(authored)
    if authored_path.is_absolute():
        candidates.append((authored_path, "authored_absolute_path"))

    for layer, label in (
        (authoring_layer, "authoring_layer"),
        (root_layer, "root_layer"),
    ):
        if not layer:
            continue
        try:
            computed = Sdf.ComputeAssetPathRelativeToLayer(layer, authored)
            if computed:
                resolved = str(Ar.GetResolver().Resolve(computed))
                candidates.append(
                    (Path(resolved or computed), f"{label}_resolver")
                )
        except Exception:
            pass
        real_path = getattr(layer, "realPath", "")
        if real_path and not authored_path.is_absolute():
            relative_candidate = Path(real_path).resolve().parent / authored_path
            candidates.append((relative_candidate, f"{label}_relative"))
            if label == "authoring_layer":
                authoring_layer_relative_candidate = relative_candidate

    if authoring_layer_relative_candidate is not None:
        repaired_spelling = _unique_same_directory_casefold_file(
            authoring_layer_relative_candidate
        )
        if repaired_spelling:
            return (
                (str(repaired_spelling),),
                CASE_REPAIR_METHOD,
            )
        repaired_path = _unique_full_path_casefold_file(
            authoring_layer_relative_candidate
        )
        if repaired_path:
            return (
                (str(repaired_path[0]),),
                PATH_CASE_REPAIR_METHOD,
            )

    seen: set[str] = set()
    path_fallback: tuple[tuple[str, ...], str] | None = None
    for candidate, method in candidates:
        if method == "authored_absolute_path":
            repaired_spelling = _unique_same_directory_casefold_file(candidate)
            if repaired_spelling:
                return (
                    (str(repaired_spelling),),
                    CASE_REPAIR_METHOD,
                )
        key = os.path.normpath(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        expanded = _expand_udim(candidate)
        if expanded:
            return tuple(str(path) for path in expanded), method
        recovered = _casefold_existing_path(candidate)
        if recovered and path_fallback is None:
            path_fallback = (
                (str(recovered),),
                f"{method}_casefold_recovery",
            )
    if path_fallback:
        return path_fallback
    verified_repair = _verified_source_repair(authored, root_layer)
    if verified_repair:
        repaired_path, _ = verified_repair
        return (str(repaired_path),), SOURCE_REPAIR_METHOD
    return (), None


def is_path_fallback(method: str | None) -> bool:
    """Check whether a path was found outside its authored location."""

    if not method:
        return False
    normalized = str(method).casefold()
    if normalized in {
        CASE_REPAIR_METHOD,
        PATH_CASE_REPAIR_METHOD,
        TEXTURE_REPAIR_METHOD,
        _INSTALLED_OMNI_MDL_RESOLUTION_METHOD,
        _ASSET_LOCAL_OMNI_MDL_RESOLUTION_METHOD,
    }:
        return False
    if "casefold" in normalized or normalized.startswith("root_layer_"):
        return True
    if normalized in {"usd_resolved_path", "authored_absolute_path"}:
        return False
    if normalized.startswith("authoring_layer_"):
        return False
    if normalized in {"mdl_source_relative", "mdl_authored_absolute_path"}:
        return False
    if normalized == SOURCE_REPAIR_METHOD:
        return False
    return True


def _layer_identifier(layer) -> str | None:
    if not layer:
        return None
    return (
        getattr(layer, "realPath", "")
        or getattr(layer, "identifier", "")
        or None
    )


def _connection_source_parts(source_info):
    if hasattr(source_info, "source"):
        return (
            source_info.source,
            source_info.sourceName,
            source_info.sourceType,
        )
    try:
        return source_info[0], source_info[1], source_info[2]
    except (IndexError, TypeError):
        return None, None, None


def _source_property_object(source, source_name, source_type):
    """Return the exact UsdShade property named by a connection endpoint."""

    if not source:
        return None
    normalized_type = str(source_type).casefold()
    try:
        if "input" in normalized_type:
            return source.GetInput(source_name)
        return source.GetOutput(source_name)
    except Exception:
        return None


def _source_property_record(source, source_name, source_type) -> dict[str, Any] | None:
    if not source:
        return None
    try:
        prim = source.GetPrim()
    except Exception:
        return None
    source_name = str(source_name)
    source_type_text = str(source_type)
    normalized_type = source_type_text.casefold()
    property_kind = "input" if "input" in normalized_type else "output"
    property_object = _source_property_object(source, source_name, source_type)
    try:
        full_name = str(property_object.GetFullName()) if property_object else None
        property_path = (
            str(property_object.GetAttr().GetPath()) if property_object else None
        )
    except Exception:
        full_name = None
        property_path = None
    if not full_name:
        prefix = "inputs" if property_kind == "input" else "outputs"
        full_name = f"{prefix}:{source_name}"
    if not property_path:
        property_path = str(prim.GetPath().AppendProperty(full_name))
    return {
        "source_prim_path": str(prim.GetPath()),
        "source_prim_type": str(prim.GetTypeName()),
        "source_property_name": source_name,
        "source_property_type": property_kind,
        "source_property_full_name": full_name,
        "source_property_path": property_path,
        "usd_source_type": source_type_text,
    }


def _consumer_property_endpoint(shade_property) -> dict[str, str | None]:
    try:
        attribute = shade_property.GetAttr()
        property_path = str(attribute.GetPath())
    except Exception:
        property_path = None
    try:
        full_name = str(shade_property.GetFullName())
    except Exception:
        full_name = None
    try:
        prim_path = str(shade_property.GetPrim().GetPath())
    except Exception:
        try:
            prim_path = str(attribute.GetPrim().GetPath())
        except Exception:
            prim_path = None
    return {
        "consumer_prim_path": prim_path,
        "consumer_property_full_name": full_name,
        "consumer_property_path": property_path,
    }


def _property_connection_endpoints(
    shade_property,
) -> tuple[list[tuple[dict[str, Any], Any]], list[str]]:
    """Return exact, ordered source properties and both edge endpoints.

    The public JSON record includes the consumer as well as the source.  The
    private property object is paired with that record so traversal follows
    the named output/input, never every property on the source prim.
    """

    source_infos = []
    invalid_paths: list[str] = []
    try:
        connected = shade_property.GetConnectedSources()
        if isinstance(connected, tuple) and len(connected) == 2:
            source_infos = list(connected[0] or ())
            invalid_paths = [str(path) for path in (connected[1] or ())]
        else:
            source_infos = list(connected or ())
    except Exception:
        try:
            connected = shade_property.GetConnectedSource()
        except Exception:
            connected = None
        if connected and connected[0]:
            source_infos = [connected]

    consumer = _consumer_property_endpoint(shade_property)
    endpoints: list[tuple[dict[str, Any], Any]] = []
    for source_connection_index, source_info in enumerate(source_infos):
        source, source_name, source_type = _connection_source_parts(source_info)
        record = _source_property_record(source, source_name, source_type)
        if not record:
            continue
        endpoints.append(
            (
                {
                    **consumer,
                    **record,
                    "source_connection_index": source_connection_index,
                },
                _source_property_object(source, source_name, source_type),
            )
        )
    return endpoints, sorted(invalid_paths)


def _ignored_invalid_connection_fallbacks(
    shade_property,
    endpoints: Iterable[tuple[dict[str, Any], Any]],
    invalid_paths: Iterable[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Classify only OpenUSD's deterministic authored-input fallback.

    ``UsdShadeInput.GetConnectedSources`` reports authored source paths whose
    properties do not exist separately from valid sources.  OpenUSD ignores
    those invalid connections.  If there is no valid source and the consumer
    input has an authored value, that value is therefore the renderer's
    effective fallback.  Retain the bad authored path as repair provenance,
    but do not treat it as an untraversable active edge.

    Existing-but-invalid properties, outputs, consumers without an authored
    value, and cases with any valid source deliberately remain unhandled.  In
    particular this helper never follows a similarly named property or
    invents a value.
    """

    invalid = sorted({str(path) for path in invalid_paths if str(path)})
    if not invalid or list(endpoints):
        return [], invalid
    if _shade_property_kind(shade_property) != "input":
        return [], invalid

    try:
        attribute = shade_property.GetAttr()
        has_authored_value = bool(attribute.HasAuthoredValueOpinion())
    except Exception:
        return [], invalid
    if not has_authored_value:
        return [], invalid

    prim = _shade_property_prim(shade_property)
    try:
        stage = prim.GetStage() if prim else None
    except Exception:
        stage = None
    if not stage:
        return [], invalid

    try:
        consumer_value = attribute.Get(Usd.TimeCode.Default())
    except Exception:
        try:
            consumer_value = attribute.Get()
        except Exception:
            return [], invalid
    try:
        consumer_value_type = str(attribute.GetTypeName())
    except Exception:
        consumer_value_type = None

    consumer_path = _shade_property_path(shade_property)
    handled: list[dict[str, Any]] = []
    remaining: list[str] = []
    for invalid_path in invalid:
        try:
            source_property = stage.GetPropertyAtPath(Sdf.Path(invalid_path))
        except Exception:
            remaining.append(invalid_path)
            continue
        if source_property:
            remaining.append(invalid_path)
            continue

        source_prim_path = None
        source_prim_type = None
        source_prim_exists = False
        try:
            sdf_path = Sdf.Path(invalid_path)
            source_prim_path = str(sdf_path.GetPrimPath())
            source_prim = stage.GetPrimAtPath(sdf_path.GetPrimPath())
            source_prim_exists = bool(source_prim)
            if source_prim:
                source_prim_type = str(source_prim.GetTypeName())
        except Exception:
            pass
        handled.append(
            {
                "authored_invalid_source_property_path": invalid_path,
                "invalid_source_reason": "source_property_does_not_exist",
                "source_prim_path": source_prim_path,
                "source_prim_exists": source_prim_exists,
                "source_prim_type": source_prim_type,
                "consumer_property_path": consumer_path,
                "consumer_has_authored_value": True,
                "consumer_authored_value": _json_safe_value(consumer_value),
                "consumer_value_type": consumer_value_type,
                "effective_value_source": "consumer_authored_value",
                "openusd_resolution": (
                    "invalid_source_connection_ignored_use_authored_input_value"
                ),
                "repair_required": True,
            }
        )
    return handled, remaining


def _attribute_value_record(attribute, root_layer) -> dict[str, Any]:
    value = attribute.Get(Usd.TimeCode.Default())
    authoring_layer = _authoring_layer(attribute)
    asset_values = []
    for asset_value in _asset_paths(value):
        resolved_paths, method = _resolve_asset_path(
            asset_value, authoring_layer, root_layer
        )
        asset_values.append(
            {
                "authored_path": asset_value.path,
                "usd_resolved_path": asset_value.resolvedPath or None,
                "resolved_paths": list(resolved_paths),
                "resolution_method": method,
                "used_path_fallback": is_path_fallback(
                    method
                ),
                "resolution_provenance": _asset_resolution_provenance(
                    method,
                    resolved_paths,
                    authored_asset=asset_value.path,
                    authoring_layer=authoring_layer,
                ),
            }
        )
    try:
        color_space = str(attribute.GetColorSpace() or "") or None
    except Exception:
        color_space = None
    try:
        has_authored_value = bool(attribute.HasAuthoredValueOpinion())
    except Exception:
        has_authored_value = False
    record = {
        "attribute_name": str(attribute.GetName()),
        "property_path": str(attribute.GetPath()),
        "value": _json_safe_value(value),
        "value_type": str(attribute.GetTypeName()),
        "variability": str(attribute.GetVariability()),
        "has_authored_value": has_authored_value,
        "authoring_layer": _layer_identifier(authoring_layer),
        "color_space": color_space,
        "asset_values": asset_values,
    }
    record.update(_attribute_time_metadata(attribute))
    return record


def _shade_property_record(shade_property, root_layer) -> dict[str, Any]:
    attribute = shade_property.GetAttr()
    endpoints, invalid_paths = _property_connection_endpoints(shade_property)
    ignored_fallbacks, invalid_paths = _ignored_invalid_connection_fallbacks(
        shade_property, endpoints, invalid_paths
    )
    connections = [record for record, _ in endpoints]
    record = {
        "name": str(shade_property.GetBaseName()),
        "full_name": str(shade_property.GetFullName()),
        "property_path": str(attribute.GetPath()),
        **_attribute_value_record(attribute, root_layer),
        "connections": connections,
        "invalid_connection_paths": invalid_paths,
        "ignored_invalid_connection_fallbacks": ignored_fallbacks,
    }
    try:
        record["documentation"] = shade_property.GetDocumentation() or None
    except Exception:
        record["documentation"] = None
    return record


def _terminal_context_and_name(output) -> tuple[str, str]:
    full_name = str(output.GetFullName())
    tokens = full_name.split(":")
    terminal_name = tokens[-1]
    context_tokens = tokens[1:-1] if tokens and tokens[0] == "outputs" else []
    return (":".join(context_tokens) or "universal"), terminal_name


def _shade_property_path(shade_property) -> str:
    try:
        return str(shade_property.GetAttr().GetPath())
    except Exception:
        return ""


def _shade_property_prim(shade_property):
    try:
        return shade_property.GetPrim()
    except Exception:
        try:
            return shade_property.GetAttr().GetPrim()
        except Exception:
            return None


def _shade_property_kind(shade_property) -> str:
    try:
        full_name = str(shade_property.GetFullName())
    except Exception:
        full_name = ""
    return "input" if full_name.startswith("inputs:") else "output"


def _shade_node_kind(prim) -> str:
    """Classify UsdShade containers without mislabeling Materials.

    ``UsdShadeMaterial`` derives from ``UsdShadeNodeGraph`` in the schema
    hierarchy.  Testing NodeGraph first therefore incorrectly labels every
    Material as a NodeGraph in real OpenUSD stages.
    """

    try:
        if prim.IsA(UsdShade.Shader):
            return "Shader"
        if prim.IsA(UsdShade.Material):
            return "Material"
        if prim.IsA(UsdShade.NodeGraph):
            return "NodeGraph"
    except Exception:
        pass
    try:
        return str(prim.GetTypeName())
    except Exception:
        return "unknown"


def _connectable_is_container(prim) -> bool:
    """Use OpenUSD's connectable classification, with a schema fallback."""

    try:
        return bool(UsdShade.ConnectableAPI(prim).IsContainer())
    except Exception:
        try:
            return bool(
                prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.NodeGraph)
            )
        except Exception:
            return False


def _normalize_shader_input_name(value: Any) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isalnum()
    )


def _finite_rgba(value: Any) -> tuple[float, float, float, float] | None:
    serialized = _json_safe_value(value)
    if not isinstance(serialized, list) or len(serialized) not in {3, 4}:
        return None
    if any(
        isinstance(component, bool)
        or not isinstance(component, (int, float))
        for component in serialized
    ):
        return None
    numeric = tuple(float(component) for component in serialized)
    if not all(math.isfinite(component) for component in numeric):
        return None
    return (
        numeric[0],
        numeric[1],
        numeric[2],
        numeric[3] if len(numeric) == 4 else 1.0,
    )


def _attribute_has_authored_value(attribute) -> bool:
    """Match the oracle's HasAuthoredValue test exactly when available."""

    try:
        return bool(attribute.HasAuthoredValue())
    except Exception:
        try:
            return bool(attribute.HasAuthoredValueOpinion())
        except Exception:
            return False


def _attribute_has_authored_value_opinion(attribute) -> bool:
    try:
        return bool(attribute.HasAuthoredValueOpinion())
    except Exception:
        return _attribute_has_authored_value(attribute)


def _attribute_default_value(attribute) -> Any:
    try:
        return attribute.Get(Usd.TimeCode.Default())
    except Exception:
        try:
            return attribute.Get()
        except Exception:
            return None


def _shader_identifier(prim) -> Any:
    try:
        if not prim.IsA(UsdShade.Shader):
            return None
        return _json_safe_value(UsdShade.Shader(prim).GetIdAttr().Get())
    except Exception:
        return None


def _ordered_value_producer_trace(shade_input) -> dict[str, Any]:
    """Independently reproduce OpenUSD's ordered recursive resolution.

    The returned producers are private live UsdShade properties; every public
    trace and edge is JSON-safe.  OpenUSD's own
    ``GetValueProducingAttributes(False)`` is still queried separately and is
    treated as the oracle.  Constants are admitted only if both ordered path
    lists agree and this trace contains no provenance defect.
    """

    trace: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    def block(code: str, message: str, **details: Any) -> None:
        blockers.append({"code": code, "message": message, **details})

    def visit(
        shade_property,
        connection_chain: tuple[dict[str, Any], ...],
        visited_paths: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        property_path = _shade_property_path(shade_property)
        if not property_path:
            block(
                "value_producer_property_path_missing",
                "a value-producing connection property has no stable path",
            )
            return []
        if property_path in visited_paths:
            block(
                "value_producer_connection_cycle",
                "recursive UsdShade value resolution encountered a cycle",
                property_path=property_path,
                visited_property_paths=list(visited_paths),
            )
            trace.append(
                {
                    "property_path": property_path,
                    "decision": "cycle_rejected",
                    "connection_endpoint_chain": list(connection_chain),
                }
            )
            return []

        endpoints, invalid_paths = _property_connection_endpoints(
            shade_property
        )
        ignored_fallbacks, remaining_invalid_paths = (
            _ignored_invalid_connection_fallbacks(
                shade_property, endpoints, invalid_paths
            )
        )
        attribute = shade_property.GetAttr()
        has_authored_value = _attribute_has_authored_value(attribute)
        if ignored_fallbacks and not has_authored_value:
            remaining_invalid_paths = sorted(
                {
                    *remaining_invalid_paths,
                    *(
                        str(item.get("authored_invalid_source_property_path"))
                        for item in ignored_fallbacks
                        if item.get("authored_invalid_source_property_path")
                    ),
                }
            )
            ignored_fallbacks = []
        if remaining_invalid_paths:
            block(
                "value_producer_invalid_connection_path",
                "value resolution contains an invalid source path that cannot be represented as an exact authored fallback",
                property_path=property_path,
                invalid_connection_paths=remaining_invalid_paths,
            )

        trace_record = {
            "property_path": property_path,
            "property_kind": _shade_property_kind(shade_property),
            "prim_path": str(_shade_property_prim(shade_property).GetPath()),
            "node_kind": _shade_node_kind(_shade_property_prim(shade_property)),
            "is_container": _connectable_is_container(
                _shade_property_prim(shade_property)
            ),
            "has_authored_value": has_authored_value,
            "has_authored_value_opinion": (
                _attribute_has_authored_value_opinion(attribute)
            ),
            "connections": [edge for edge, _ in endpoints],
            "invalid_connection_paths": remaining_invalid_paths,
            "ignored_invalid_connection_fallbacks": ignored_fallbacks,
            "connection_endpoint_chain": list(connection_chain),
            "decision": None,
        }
        trace.append(trace_record)

        producers: list[dict[str, Any]] = []
        next_visited = (*visited_paths, property_path)
        for edge, source_property in endpoints:
            source_chain = (*connection_chain, edge)
            if not source_property:
                block(
                    "value_producer_source_property_unopenable",
                    "a valid UsdShade connection source could not be opened",
                    consumer_property_path=property_path,
                    source_property_path=edge.get("source_property_path"),
                    source_connection_index=edge.get(
                        "source_connection_index"
                    ),
                )
                continue
            source_prim = _shade_property_prim(source_property)
            if not source_prim:
                block(
                    "value_producer_source_prim_missing",
                    "a valid UsdShade connection source has no prim",
                    consumer_property_path=property_path,
                    source_property_path=edge.get("source_property_path"),
                )
                continue
            source_kind = _shade_property_kind(source_property)
            source_is_container = _connectable_is_container(source_prim)
            if source_kind == "output" and not source_is_container:
                producers.append(
                    {
                        "property": source_property,
                        "connection_endpoint_chain": source_chain,
                        "producer_resolution": (
                            "non_container_output_terminal"
                        ),
                    }
                )
            elif source_is_container:
                producers.extend(
                    visit(source_property, source_chain, next_visited)
                )
            else:
                block(
                    "value_producer_noncontainer_input_source",
                    "a Shader/non-container Input is not a legal recursive value source",
                    consumer_property_path=property_path,
                    source_property_path=_shade_property_path(source_property),
                    source_connection_index=edge.get(
                        "source_connection_index"
                    ),
                )

        if producers:
            trace_record["decision"] = "connections_produced_values"
            return producers
        if has_authored_value:
            trace_record["decision"] = "authored_value_fallback"
            return [
                {
                    "property": shade_property,
                    "connection_endpoint_chain": connection_chain,
                    "producer_resolution": "authored_value_fallback",
                }
            ]
        trace_record["decision"] = "no_value_producer"
        return []

    producers = visit(shade_input, (), ())
    return {
        "producers": producers,
        "producer_paths": [
            _shade_property_path(item["property"]) for item in producers
        ],
        "trace": trace,
        "blockers": blockers,
    }


def _oracle_value_producer_paths(shade_input) -> tuple[list[str], str | None]:
    try:
        attributes = list(
            shade_input.GetValueProducingAttributes(False) or ()
        )
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    paths: list[str] = []
    for attribute in attributes:
        try:
            paths.append(str(attribute.GetPath()))
        except Exception as exc:
            return [], (
                "oracle returned an attribute without a readable path: "
                f"{type(exc).__name__}: {exc}"
            )
    return paths, None


def _constant_source_core(
    shade_property,
    *,
    seed_property_path: str,
    producer_resolution: str,
    root_layer=None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read a shader input value and its source, rejecting nonfinite values."""

    property_path = _shade_property_path(shade_property)
    prim = _shade_property_prim(shade_property)
    property_kind = _shade_property_kind(shade_property)
    is_seed = property_path == seed_property_path
    is_container = _connectable_is_container(prim) if prim else False
    if property_kind != "input" or not (is_seed or is_container):
        return None, {
            "code": "nonconstant_value_producer",
            "reason": (
                "producer is not the recognized Shader seed Input or an "
                "Input on a UsdShade container"
            ),
            "property_path": property_path,
            "property_kind": property_kind,
            "prim_path": str(prim.GetPath()) if prim else None,
            "node_kind": _shade_node_kind(prim) if prim else None,
            "is_container": is_container,
        }

    attribute = shade_property.GetAttr()
    if not _attribute_has_authored_value(attribute):
        return None, {
            "code": "constant_input_has_no_authored_value",
            "reason": (
                "OpenUSD selected an Input that does not expose an authored "
                "value through HasAuthoredValue"
            ),
            "property_path": property_path,
        }
    authored_value = _attribute_default_value(attribute)
    rgba = _finite_rgba(authored_value)
    time_metadata = _attribute_time_metadata(attribute)
    if rgba is None:
        return None, {
            "code": (
                "constant_input_time_samples_without_default"
                if time_metadata["authored_time_sample_count"]
                and authored_value is None
                else "constant_input_not_finite_rgb_or_rgba"
            ),
            "reason": (
                "the exact authored base-color producer cannot be represented "
                "as a finite default-time RGB or RGBA value"
            ),
            "property_path": property_path,
            "authored_value": _json_safe_value(authored_value),
            **time_metadata,
        }

    endpoints, invalid_paths = _property_connection_endpoints(shade_property)
    ignored_fallbacks, remaining_invalid_paths = (
        _ignored_invalid_connection_fallbacks(
            shade_property, endpoints, invalid_paths
        )
    )
    try:
        color_space = str(attribute.GetColorSpace() or "") or None
    except Exception:
        color_space = None
    try:
        full_name = str(shade_property.GetFullName())
        input_name = str(shade_property.GetBaseName())
    except Exception:
        full_name = None
        input_name = None
    serialized_value = _json_safe_value(authored_value)
    effective_value_source = (
        "authored_seed_shader_input"
        if is_seed
        else "authored_container_interface_input"
    )
    if endpoints and producer_resolution == "authored_value_fallback":
        effective_value_source += "_after_uninitialized_connections"
    if ignored_fallbacks:
        effective_value_source += "_after_ignored_invalid_connections"
    core = {
        "source_property_path": property_path,
        "source_property_full_name": full_name,
        "source_input_name": input_name,
        "source_property_kind": property_kind,
        "source_prim_path": str(prim.GetPath()) if prim else None,
        "source_prim_type": str(prim.GetTypeName()) if prim else None,
        "source_node_kind": _shade_node_kind(prim) if prim else None,
        "source_is_container": is_container,
        "source_shader_id": _shader_identifier(prim) if prim else None,
        "source_role": (
            "recognized_surface_shader_seed_input"
            if is_seed
            else "connected_container_interface_input"
        ),
        "authored_value": serialized_value,
        "authored_component_count": len(serialized_value),
        "rgba": list(rgba),
        "value_type": str(attribute.GetTypeName()),
        "has_authored_value": True,
        "has_authored_value_opinion": (
            _attribute_has_authored_value_opinion(attribute)
        ),
        "authoring_layer": _layer_identifier(_authoring_layer(attribute)),
        "color_space": color_space,
        "effective_value_source": effective_value_source,
        "producer_resolution": producer_resolution,
        "connections": [edge for edge, _ in endpoints],
        "invalid_connection_paths": remaining_invalid_paths,
        "ignored_invalid_connection_fallbacks": ignored_fallbacks,
        **time_metadata,
    }
    core["source_provenance_sha256"] = _json_canonical_hash(
        "surface-base-color-constant-source-v1", core
    )
    return core, None


def _connection_output_channel(source_property_name: str | None) -> str | None:
    """Return a channel token only when the source names one explicitly."""

    if not source_property_name:
        return None
    token = source_property_name.rsplit(":", 1)[-1].casefold()
    channel_aliases = {
        "red": "r",
        "green": "g",
        "blue": "b",
        "alpha": "a",
        "x": "x",
        "y": "y",
        "z": "z",
        "w": "w",
        "r": "r",
        "g": "g",
        "b": "b",
        "a": "a",
        "rgb": "rgb",
        "rgba": "rgba",
    }
    return channel_aliases.get(token)


def _empty_texture_semantic() -> dict[str, Any]:
    return {
        "role": None,
        "consumer_property_path": None,
        "consumer_property_full_name": None,
        "source_output_path": None,
        "source_output_name": None,
        "source_output_channel": None,
    }


def _semantic_for_shader_input(
    semantic: dict[str, Any], shade_input
) -> dict[str, Any]:
    """Propagate the downstream role through generic upstream nodes.

    Once a material-facing input supplies a meaning (for example
    ``diffuseColor``), generic upstream inputs such as ``file`` or ``in`` must
    not erase it.  A semantic is only established on a deeper input when no
    downstream consumer has already established one.
    """

    updated = dict(semantic)
    if updated.get("role"):
        return updated
    try:
        full_name = str(shade_input.GetFullName())
    except Exception:
        full_name = ""
    try:
        base_name = str(shade_input.GetBaseName())
    except Exception:
        base_name = full_name
    role = (
        _exact_mdl_texture_input_role(full_name)
        or _exact_mdl_texture_input_role(base_name)
        or _role_from_semantic_text(full_name)
        or _role_from_semantic_text(base_name)
    )
    if role:
        updated["role"] = role
        updated["consumer_property_path"] = _shade_property_path(shade_input)
        updated["consumer_property_full_name"] = full_name or base_name
    return updated


def _semantic_through_connection(
    semantic: dict[str, Any], edge: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(semantic)
    if updated.get("role") and edge.get("source_property_type") == "output":
        candidate_channel = _connection_output_channel(
            edge.get("source_property_name")
        )
        if not updated.get("source_output_path") or (
            candidate_channel and not updated.get("source_output_channel")
        ):
            updated["source_output_path"] = edge.get("source_property_path")
            updated["source_output_name"] = edge.get(
                "source_property_full_name"
            )
            updated["source_output_channel"] = candidate_channel
    return updated


def _semantic_visit_key(semantic: dict[str, Any]) -> tuple[Any, ...]:
    return (
        semantic.get("role"),
        semantic.get("consumer_property_path"),
        semantic.get("source_output_path"),
    )


def _trace_active_network(
    output_roots: Iterable[tuple[Any, str]],
) -> dict[str, Any]:
    """Trace exact property endpoints reachable from material outputs.

    A reached Shader output activates that shader's inputs.  A reached
    NodeGraph output follows only that exact output connection.  For every
    active input, a valid connection wins over its authored fallback value.
    The routine is intentionally value-agnostic and follows only the supplied
    composed property endpoints.
    """

    nodes: dict[str, dict[str, Any]] = {}
    asset_inputs: list[dict[str, Any]] = []
    active_shader_uses: dict[tuple[str, str], Any] = {}
    untraversable_connections: list[dict[str, Any]] = []
    visited_follow: set[tuple[Any, ...]] = set()
    expanded_shaders: set[tuple[Any, ...]] = set()
    pending: list[tuple[Any, str, dict[str, Any], tuple[dict[str, Any], ...]]] = [
        (output, context, _empty_texture_semantic(), ())
        for output, context in output_roots
    ]

    def node_state_for(shade_property) -> dict[str, Any] | None:
        prim = _shade_property_prim(shade_property)
        if not prim:
            return None
        try:
            prim_path = str(prim.GetPath())
        except Exception:
            return None
        state = nodes.setdefault(
            prim_path,
            {
                "prim": prim,
                "active_property_paths": set(),
                "active_input_paths": set(),
                "active_output_paths": set(),
                "authored_fallback_active_paths": set(),
                "authored_fallback_suppressed_paths": set(),
                "activation_edges": [],
                "invalid_connection_paths": set(),
                "ignored_invalid_connection_fallbacks": {},
            },
        )
        property_path = _shade_property_path(shade_property)
        if property_path:
            state["active_property_paths"].add(property_path)
            state[f"active_{_shade_property_kind(shade_property)}_paths"].add(
                property_path
            )
        return state

    while pending:
        consumer_property, context, semantic, chain = pending.pop()
        consumer_path = _shade_property_path(consumer_property)
        visit_key = (
            consumer_path,
            context,
            *_semantic_visit_key(semantic),
        )
        if visit_key in visited_follow:
            continue
        visited_follow.add(visit_key)

        consumer_state = node_state_for(consumer_property)
        endpoints, invalid_paths = _property_connection_endpoints(
            consumer_property
        )
        ignored_fallbacks, invalid_paths = (
            _ignored_invalid_connection_fallbacks(
                consumer_property, endpoints, invalid_paths
            )
        )
        if consumer_state is not None:
            consumer_state["invalid_connection_paths"].update(invalid_paths)
            for fallback in ignored_fallbacks:
                key = (
                    fallback.get("consumer_property_path"),
                    fallback.get("authored_invalid_source_property_path"),
                )
                consumer_state["ignored_invalid_connection_fallbacks"][
                    key
                ] = fallback
        if _shade_property_kind(consumer_property) == "input":
            if endpoints:
                if consumer_state is not None and consumer_path:
                    consumer_state[
                        "authored_fallback_suppressed_paths"
                    ].add(consumer_path)
            else:
                if consumer_state is not None and consumer_path:
                    consumer_state["authored_fallback_active_paths"].add(
                        consumer_path
                    )
                asset_inputs.append(
                    {
                        "input": consumer_property,
                        "context": context,
                        "semantic": dict(semantic),
                        "connection_endpoint_chain": chain,
                    }
                )

        for edge, source_property in endpoints:
            edge_chain = (*chain, edge)
            source_semantic = _semantic_through_connection(semantic, edge)
            if not source_property:
                untraversable_connections.append(
                    {
                        **edge,
                        "render_context": context,
                        "reason": "connected source property could not be opened",
                    }
                )
                continue
            source_state = node_state_for(source_property)
            if source_state is not None:
                source_state["activation_edges"].append(edge)
            source_prim = _shade_property_prim(source_property)
            if not source_prim:
                continue

            is_shader = bool(source_prim.IsA(UsdShade.Shader))
            is_output = _shade_property_kind(source_property) == "output"
            if is_shader and is_output:
                prim_path = str(source_prim.GetPath())
                active_shader_uses[(prim_path, context)] = source_prim
                expansion_key = (
                    prim_path,
                    _shade_property_path(source_property),
                    context,
                    *_semantic_visit_key(source_semantic),
                )
                if expansion_key in expanded_shaders:
                    continue
                expanded_shaders.add(expansion_key)
                try:
                    shader_inputs = sorted(
                        UsdShade.ConnectableAPI(source_prim).GetInputs(),
                        key=lambda item: str(item.GetFullName()),
                    )
                except Exception:
                    shader_inputs = []
                for shade_input in shader_inputs:
                    input_semantic = _semantic_for_shader_input(
                        source_semantic, shade_input
                    )
                    node_state_for(shade_input)
                    pending.append(
                        (shade_input, context, input_semantic, edge_chain)
                    )
            else:
                pending.append(
                    (source_property, context, source_semantic, edge_chain)
                )

    for state in nodes.values():
        unique_edges: dict[tuple[Any, ...], dict[str, Any]] = {}
        for edge in state["activation_edges"]:
            key = (
                edge.get("consumer_property_path"),
                edge.get("source_property_path"),
                edge.get("source_connection_index"),
            )
            unique_edges.setdefault(key, edge)
        state["activation_edges"] = list(unique_edges.values())

    return {
        "nodes": nodes,
        "asset_inputs": asset_inputs,
        "active_shader_uses": [
            {
                "prim": active_shader_uses[key],
                "render_context": key[1],
            }
            for key in sorted(active_shader_uses)
        ],
        "untraversable_connections": sorted(
            untraversable_connections,
            key=lambda item: (
                str(item.get("consumer_property_path")),
                str(item.get("source_property_path")),
            ),
        ),
    }


def _surface_material_output_roots(material) -> list[tuple[Any, str]]:
    if not material:
        return []
    try:
        outputs = UsdShade.ConnectableAPI(material.GetPrim()).GetOutputs()
    except Exception:
        outputs = []
    roots: list[tuple[Any, str]] = []
    for output in sorted(outputs, key=lambda item: str(item.GetFullName())):
        render_context, terminal_name = _terminal_context_and_name(output)
        if terminal_name.casefold() != "surface":
            continue
        endpoints, _ = _property_connection_endpoints(output)
        if endpoints:
            roots.append((output, render_context))
    return roots


def _empty_surface_base_color_value_resolution() -> dict[str, Any]:
    record = {
        "recognized_shader_input_names": sorted(
            _SURFACE_BASE_COLOR_SHADER_INPUT_NAMES
        ),
        "surface_terminal_roots": [],
        "surface_terminal_shader_outputs": [],
        "consumer_count": 0,
        "constant_source_count": 0,
        "consumers": [],
        "constant_sources": [],
        "excluded_producers": [],
        "terminal_discovery_blockers": [],
        "blockers": [],
    }
    record["resolution_sha256"] = _json_canonical_hash(
        "surface-base-color-value-resolution-v1", record
    )
    return record


def collect_surface_base_color_value_resolution(
    material, root_layer=None
) -> dict[str, Any]:
    """Resolve every exact surface-reachable authored base-color constant.

    Candidate consumers are restricted to recognized Inputs on Shaders whose
    outputs are reachable from connected Material surface terminals.  Their
    logical producers are resolved twice: once with OpenUSD's public oracle
    and once with the provenance-preserving ordered trace above.  Any error or
    ordered path disagreement blocks that consumer and, through the shared
    annotation policy, the material from reaching the VLM.
    """

    if not material:
        return _empty_surface_base_color_value_resolution()

    output_roots = _surface_material_output_roots(material)
    terminal_shader_uses: dict[tuple[str, str], dict[str, Any]] = {}
    terminal_discovery_blockers: list[dict[str, Any]] = []
    for material_output, render_context in output_roots:
        material_output_path = _shade_property_path(material_output)
        try:
            terminal_attributes = list(
                material_output.GetValueProducingAttributes(True) or ()
            )
        except Exception as exc:
            terminal_discovery_blockers.append(
                {
                    "code": "surface_terminal_shader_oracle_failed",
                    "message": (
                        "OpenUSD could not resolve the active non-container "
                        "Shader output for a Material surface terminal"
                    ),
                    "material_output_path": material_output_path,
                    "render_context": render_context,
                    "oracle_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        valid_terminal_shader_count = 0
        for terminal_attribute_ordinal, attribute in enumerate(
            terminal_attributes
        ):
            try:
                shader_prim = attribute.GetPrim()
                shader_path = str(shader_prim.GetPath())
                shader_output_path = str(attribute.GetPath())
                shader_output_name = str(attribute.GetName())
            except Exception as exc:
                terminal_discovery_blockers.append(
                    {
                        "code": "surface_terminal_shader_attribute_invalid",
                        "message": (
                            "OpenUSD returned an unreadable surface-terminal "
                            "value-producing attribute"
                        ),
                        "material_output_path": material_output_path,
                        "render_context": render_context,
                        "terminal_attribute_ordinal": (
                            terminal_attribute_ordinal
                        ),
                        "oracle_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            try:
                is_shader = bool(shader_prim.IsA(UsdShade.Shader))
            except Exception:
                is_shader = False
            if (
                not is_shader
                or _connectable_is_container(shader_prim)
                or not shader_output_name.startswith("outputs:")
                or not shader_output_path.startswith(
                    f"{shader_path}.outputs:"
                )
            ):
                terminal_discovery_blockers.append(
                    {
                        "code": "surface_terminal_oracle_nonshader_output",
                        "message": (
                            "surface-terminal resolution did not end at a "
                            "non-container Shader output"
                        ),
                        "material_output_path": material_output_path,
                        "render_context": render_context,
                        "terminal_attribute_ordinal": (
                            terminal_attribute_ordinal
                        ),
                        "resolved_property_path": shader_output_path,
                        "resolved_prim_path": shader_path,
                        "resolved_node_kind": _shade_node_kind(shader_prim),
                        "resolved_attribute_name": shader_output_name,
                    }
                )
                continue
            valid_terminal_shader_count += 1
            use = terminal_shader_uses.setdefault(
                (shader_path, str(render_context)),
                {
                    "prim": shader_prim,
                    "shader_path": shader_path,
                    "render_context": str(render_context),
                    "terminal_output_paths": [],
                    "material_output_paths": [],
                },
            )
            if shader_output_path not in use["terminal_output_paths"]:
                use["terminal_output_paths"].append(shader_output_path)
            if material_output_path not in use["material_output_paths"]:
                use["material_output_paths"].append(material_output_path)
        if valid_terminal_shader_count == 0:
            terminal_discovery_blockers.append(
                {
                    "code": "surface_terminal_shader_oracle_empty",
                    "message": (
                        "a connected Material surface terminal has no active "
                        "non-container Shader output"
                    ),
                    "material_output_path": material_output_path,
                    "render_context": render_context,
                    "oracle_result_count": len(terminal_attributes),
                }
            )

    candidates: list[tuple[str, str, Any, Any, tuple[str, ...]]] = []
    for key in sorted(terminal_shader_uses):
        shader_use = terminal_shader_uses[key]
        shader_prim = shader_use["prim"]
        shader_path = shader_use["shader_path"]
        try:
            shader_inputs = sorted(
                UsdShade.ConnectableAPI(shader_prim).GetInputs(),
                key=lambda item: str(item.GetFullName()),
            )
        except Exception:
            shader_inputs = []
        for shade_input in shader_inputs:
            try:
                input_name = str(shade_input.GetBaseName())
            except Exception:
                input_name = str(shade_input.GetFullName())
            if (
                _normalize_shader_input_name(input_name)
                not in _SURFACE_BASE_COLOR_SHADER_INPUT_NAMES
            ):
                continue
            candidates.append(
                (
                    shader_path,
                    str(shader_use["render_context"]),
                    shader_prim,
                    shade_input,
                    tuple(shader_use["terminal_output_paths"]),
                )
            )
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            _shade_property_path(item[3]),
        )
    )

    consumers: list[dict[str, Any]] = []
    merged_sources: dict[str, dict[str, Any]] = {}
    excluded_producers: list[dict[str, Any]] = []
    resolution_blockers: list[dict[str, Any]] = list(
        terminal_discovery_blockers
    )

    for (
        shader_path,
        render_context,
        shader_prim,
        shade_input,
        surface_terminal_output_paths,
    ) in candidates:
        consumer_path = _shade_property_path(shade_input)
        custom = _ordered_value_producer_trace(shade_input)
        oracle_paths, oracle_error = _oracle_value_producer_paths(shade_input)
        custom_paths = custom["producer_paths"]
        consumer_blockers = [dict(item) for item in custom["blockers"]]
        if oracle_error:
            consumer_blockers.append(
                {
                    "code": "value_producer_oracle_failed",
                    "message": (
                        "OpenUSD could not resolve value-producing attributes"
                    ),
                    "oracle_error": oracle_error,
                }
            )
        ordered_provenance_agrees = (
            oracle_error is None and oracle_paths == custom_paths
        )
        if oracle_error is None and not ordered_provenance_agrees:
            consumer_blockers.append(
                {
                    "code": "value_producer_oracle_provenance_disagreement",
                    "message": (
                        "the provenance trace disagrees with OpenUSD's ordered "
                        "value-producing attribute paths"
                    ),
                    "oracle_producer_paths": oracle_paths,
                    "provenance_producer_paths": custom_paths,
                }
            )

        pending_sources: list[tuple[dict[str, Any], dict[str, Any]]] = []
        producer_records: list[dict[str, Any]] = []
        consumer_exclusions: list[dict[str, Any]] = []
        if not consumer_blockers:
            for ordinal, producer in enumerate(custom["producers"]):
                property_object = producer["property"]
                producer_path = _shade_property_path(property_object)
                chain = list(producer["connection_endpoint_chain"])
                chain_sha256 = _json_canonical_hash(
                    "surface-base-color-connection-chain-v1", chain
                )
                core, rejection = _constant_source_core(
                    property_object,
                    seed_property_path=consumer_path,
                    producer_resolution=producer["producer_resolution"],
                    root_layer=root_layer,
                )
                use = {
                    "consumer_shader_path": shader_path,
                    "consumer_shader_id": _shader_identifier(shader_prim),
                    "consumer_property_path": consumer_path,
                    "recognized_input_name": str(
                        shade_input.GetBaseName()
                    ),
                    "recognized_input_full_name": str(
                        shade_input.GetFullName()
                    ),
                    "recognized_input_normalized_name": (
                        _normalize_shader_input_name(
                            shade_input.GetBaseName()
                        )
                    ),
                    "render_context": render_context,
                    "oracle_producer_ordinal": ordinal,
                    "oracle_producer_path": oracle_paths[ordinal],
                    "effective_source_property_path": producer_path,
                    "connection_endpoint_chain": chain,
                    "connection_chain_sha256": chain_sha256,
                }
                if core is not None:
                    pending_sources.append((core, use))
                    producer_records.append(
                        {
                            "oracle_producer_ordinal": ordinal,
                            "property_path": producer_path,
                            "classification": "authored_constant_input",
                            "source_provenance_sha256": core[
                                "source_provenance_sha256"
                            ],
                            "connection_endpoint_chain": chain,
                            "connection_chain_sha256": chain_sha256,
                        }
                    )
                    continue

                rejection_record = {
                    **dict(rejection or {}),
                    **use,
                }
                if rejection_record.get("code") == "nonconstant_value_producer":
                    rejection_record["classification"] = (
                        "excluded_nonconstant_producer"
                    )
                    consumer_exclusions.append(rejection_record)
                    producer_records.append(
                        {
                            "oracle_producer_ordinal": ordinal,
                            "property_path": producer_path,
                            "classification": "excluded_nonconstant_producer",
                            "reason_code": rejection_record.get("code"),
                            "connection_endpoint_chain": chain,
                            "connection_chain_sha256": chain_sha256,
                        }
                    )
                else:
                    consumer_blockers.append(
                        {
                            **dict(rejection or {}),
                            "message": (rejection or {}).get(
                                "reason",
                                "an authored base-color producer could not be represented exactly",
                            ),
                            "oracle_producer_ordinal": ordinal,
                            "consumer_property_path": consumer_path,
                        }
                    )

        if not consumer_blockers:
            for core, use in pending_sources:
                source_path = str(core["source_property_path"])
                existing = merged_sources.get(source_path)
                if existing is not None and existing.get(
                    "source_provenance_sha256"
                ) != core.get("source_provenance_sha256"):
                    consumer_blockers.append(
                        {
                            "code": "constant_source_provenance_conflict",
                            "message": (
                                "one source property produced inconsistent "
                                "constant provenance across consumer uses"
                            ),
                            "source_property_path": source_path,
                            "existing_source_provenance_sha256": existing.get(
                                "source_provenance_sha256"
                            ),
                            "observed_source_provenance_sha256": core.get(
                                "source_provenance_sha256"
                            ),
                        }
                    )
            if not consumer_blockers:
                for core, use in pending_sources:
                    source_path = str(core["source_property_path"])
                    source = merged_sources.setdefault(
                        source_path, {**core, "consumer_uses": []}
                    )
                    serialized_use = json.dumps(
                        use,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    if all(
                        json.dumps(
                            item,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        != serialized_use
                        for item in source["consumer_uses"]
                    ):
                        source["consumer_uses"].append(use)
                excluded_producers.extend(consumer_exclusions)

        try:
            input_name = str(shade_input.GetBaseName())
            input_full_name = str(shade_input.GetFullName())
        except Exception:
            input_name = None
            input_full_name = None
        if consumer_blockers:
            status = "blocked"
        elif pending_sources and consumer_exclusions:
            status = "resolved_mixed_constant_and_nonconstant"
        elif pending_sources:
            status = "resolved_authored_constants"
        elif consumer_exclusions:
            status = "resolved_nonconstant_producers"
        else:
            status = "no_value_producer"
        contextual_blockers = [
            {
                **blocker,
                "consumer_property_path": blocker.get(
                    "consumer_property_path", consumer_path
                ),
                "consumer_shader_path": shader_path,
                "render_context": render_context,
            }
            for blocker in consumer_blockers
        ]
        consumer_record = {
            "consumer_shader_path": shader_path,
            "consumer_shader_id": _shader_identifier(shader_prim),
            "consumer_property_path": consumer_path,
            "recognized_input_name": input_name,
            "recognized_input_full_name": input_full_name,
            "recognized_input_normalized_name": (
                _normalize_shader_input_name(input_name)
            ),
            "render_context": render_context,
            "surface_terminal_output_paths": list(
                surface_terminal_output_paths
            ),
            "oracle_producer_paths": oracle_paths,
            "provenance_producer_paths": custom_paths,
            "ordered_provenance_agrees": ordered_provenance_agrees,
            "resolution_status": status,
            "resolved_producers": producer_records,
            "excluded_producers": consumer_exclusions,
            "resolution_trace": custom["trace"],
            "blockers": contextual_blockers,
        }
        consumer_record["consumer_resolution_sha256"] = (
            _json_canonical_hash(
                "surface-base-color-consumer-resolution-v1",
                consumer_record,
            )
        )
        consumers.append(consumer_record)
        resolution_blockers.extend(contextual_blockers)

    constant_sources = [merged_sources[path] for path in sorted(merged_sources)]
    for source in constant_sources:
        source["consumer_uses"] = sorted(
            source["consumer_uses"],
            key=lambda item: (
                str(item.get("consumer_shader_path")),
                str(item.get("consumer_property_path")),
                str(item.get("render_context")),
                int(item.get("oracle_producer_ordinal", 0)),
                str(item.get("connection_chain_sha256")),
            ),
        )
        source["source_with_consumer_uses_sha256"] = _json_canonical_hash(
            "surface-base-color-constant-source-with-uses-v1", source
        )

    def serialized_sort_key(value: dict[str, Any]) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    excluded_producers.sort(key=serialized_sort_key)
    resolution_blockers.sort(key=serialized_sort_key)
    terminal_root_records = [
        {
            "material_output_path": _shade_property_path(output),
            "render_context": render_context,
        }
        for output, render_context in output_roots
    ]
    record = {
        "recognized_shader_input_names": sorted(
            _SURFACE_BASE_COLOR_SHADER_INPUT_NAMES
        ),
        "surface_terminal_roots": terminal_root_records,
        "surface_terminal_shader_outputs": [
            {
                "shader_path": terminal_shader_uses[key]["shader_path"],
                "shader_id": _shader_identifier(
                    terminal_shader_uses[key]["prim"]
                ),
                "render_context": terminal_shader_uses[key][
                    "render_context"
                ],
                "terminal_output_paths": terminal_shader_uses[key][
                    "terminal_output_paths"
                ],
                "material_output_paths": terminal_shader_uses[key][
                    "material_output_paths"
                ],
            }
            for key in sorted(terminal_shader_uses)
        ],
        "consumer_count": len(consumers),
        "constant_source_count": len(constant_sources),
        "consumers": consumers,
        "constant_sources": constant_sources,
        "excluded_producers": excluded_producers,
        "terminal_discovery_blockers": sorted(
            terminal_discovery_blockers, key=serialized_sort_key
        ),
        "blockers": resolution_blockers,
    }
    record["resolution_sha256"] = _json_canonical_hash(
        "surface-base-color-value-resolution-v1", record
    )
    return record


def collect_active_material_graph(material, root_layer=None) -> dict[str, Any]:
    """Serialize the active Material/Shader/NodeGraph network.

    Surface, displacement, and volume terminals are retained independently by
    render context.  Every composed input value and exact connection endpoint
    is recorded.  Reached NodeGraph output connections are followed
    recursively, which prevents interface nodes from hiding the actual shader
    and texture inputs used by a terminal.
    """

    if not material:
        return {
            "material_path": None,
            "source_material_path": None,
            "terminals": [],
            "connected_nonstandard_outputs": [],
            "invalid_material_output_connections": [],
            "nodes": [],
            "ignored_invalid_connection_fallbacks": [],
            "untraversable_connection_endpoints": [],
            "has_time_varying_values": False,
            "has_lossy_value_fallback": False,
            "surface_base_color_value_resolution": (
                _empty_surface_base_color_value_resolution()
            ),
        }

    material_prim = material.GetPrim()
    material_connectable = UsdShade.ConnectableAPI(material_prim)
    terminals = []
    connected_nonstandard_outputs = []
    invalid_material_output_connections = []
    output_roots: list[tuple[Any, str]] = []
    try:
        material_outputs = material_connectable.GetOutputs()
    except Exception:
        material_outputs = []
    for output in sorted(material_outputs, key=lambda item: str(item.GetFullName())):
        render_context, terminal_name = _terminal_context_and_name(output)
        property_record = _shade_property_record(output, root_layer)
        if property_record["invalid_connection_paths"]:
            invalid_material_output_connections.append(
                {
                    "terminal": terminal_name,
                    "render_context": render_context,
                    "material_output": property_record,
                }
            )
        if not property_record["connections"]:
            continue
        terminal_record = {
            "terminal": terminal_name,
            "render_context": render_context,
            "material_output": property_record,
        }
        if terminal_name.casefold() in {"surface", "displacement", "volume"}:
            terminals.append(terminal_record)
            output_roots.append((output, render_context))
        else:
            connected_nonstandard_outputs.append(terminal_record)

    trace = _trace_active_network(output_roots)
    nodes: dict[str, dict[str, Any]] = {}
    for prim_path, state in trace["nodes"].items():
        prim = state["prim"]
        connectable = UsdShade.ConnectableAPI(prim)
        try:
            inputs = sorted(
                connectable.GetInputs(), key=lambda item: str(item.GetFullName())
            )
        except Exception:
            inputs = []
        try:
            outputs = sorted(
                connectable.GetOutputs(), key=lambda item: str(item.GetFullName())
            )
        except Exception:
            outputs = []

        def active_property_record(shade_property, property_kind: str):
            property_record = _shade_property_record(
                shade_property, root_layer
            )
            property_path = property_record["property_path"]
            property_record["is_active_property"] = property_path in state[
                "active_property_paths"
            ]
            if property_kind == "input":
                property_record["authored_value_is_active"] = (
                    property_path
                    in state["authored_fallback_active_paths"]
                )
                property_record[
                    "authored_value_suppressed_by_connection"
                ] = (
                    property_path
                    in state["authored_fallback_suppressed_paths"]
                )
            return property_record

        input_records = [
            active_property_record(shade_input, "input")
            for shade_input in inputs
        ]
        output_records = [
            active_property_record(output, "output") for output in outputs
        ]

        info_attributes = {}
        for attribute in sorted(
            prim.GetAttributes(), key=lambda item: str(item.GetName())
        ):
            name = str(attribute.GetName())
            if name.startswith("info:"):
                info_attributes[name] = _attribute_value_record(
                    attribute, root_layer
                )

        shader_id = None
        implementation_source = None
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            try:
                shader_id = _json_safe_value(shader.GetIdAttr().Get())
            except Exception:
                shader_id = None
            try:
                implementation_source = str(shader.GetImplementationSource() or "") or None
            except Exception:
                implementation_source = None
        node_kind = _shade_node_kind(prim)
        nodes[prim_path] = {
            "prim_path": prim_path,
            "source_prim_path": _source_mesh_path(prim),
            "prim_type": str(prim.GetTypeName()),
            "node_kind": node_kind,
            "shader_id": shader_id,
            "implementation_source": implementation_source,
            "active_property_paths": sorted(state["active_property_paths"]),
            "active_input_paths": sorted(state["active_input_paths"]),
            "active_output_paths": sorted(state["active_output_paths"]),
            "activation_edges": state["activation_edges"],
            "invalid_connection_paths": sorted(
                state["invalid_connection_paths"]
            ),
            "ignored_invalid_connection_fallbacks": [
                state["ignored_invalid_connection_fallbacks"][key]
                for key in sorted(
                    state["ignored_invalid_connection_fallbacks"],
                    key=lambda item: (str(item[0]), str(item[1])),
                )
            ],
            "info_attributes": info_attributes,
            "inputs": input_records,
            "outputs": output_records,
        }

    def property_varies(record: dict[str, Any]) -> bool:
        return bool(
            record.get("value_might_be_time_varying")
            or record.get("authored_time_sample_count")
        )

    has_time_varying_values = any(
        property_varies(terminal["material_output"])
        for terminal in terminals
    ) or any(
        property_varies(property_record)
        for node in nodes.values()
        for property_record in (
            list(node["inputs"])
            + list(node["outputs"])
            + list(node["info_attributes"].values())
        )
    )
    graph_record = {
        "material_path": str(material.GetPath()),
        "source_material_path": _source_mesh_path(material_prim),
        "terminals": terminals,
        "connected_nonstandard_outputs": connected_nonstandard_outputs,
        "invalid_material_output_connections": (
            invalid_material_output_connections
        ),
        "nodes": [nodes[path] for path in sorted(nodes)],
        "ignored_invalid_connection_fallbacks": [
            fallback
            for path in sorted(nodes)
            for fallback in nodes[path].get(
                "ignored_invalid_connection_fallbacks", []
            )
        ],
        "untraversable_connection_endpoints": trace[
            "untraversable_connections"
        ],
        "has_time_varying_values": has_time_varying_values,
        "surface_base_color_value_resolution": (
            collect_surface_base_color_value_resolution(
                material, root_layer=root_layer
            )
        ),
    }
    graph_record["has_lossy_value_fallback"] = _contains_lossy_value_fallback(
        graph_record
    )
    return graph_record


@dataclass(frozen=True)
class _MdlToken:
    """One comment-free MDL lexical token with source offsets."""

    kind: str
    value: str
    start: int
    end: int


class _MdlParseError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_MDL_IDENTIFIER_SOURCE = r"[A-Za-z_][A-Za-z0-9_]*"
_MDL_IDENTIFIER_PATTERN = re.compile(rf"{_MDL_IDENTIFIER_SOURCE}\Z")
_MDL_GAMMA_COLOR_SPACES = {
    "gamma_srgb": "sRGB",
    "gamma_linear": "linear",
    "gamma_default": "default",
}
_MDL_OMNIPBR_ARGUMENT_TYPES = {
    "diffuse_color_constant": "color",
    "diffuse_texture": "texture_2d",
    "albedo_desaturation": "float",
    "albedo_add": "float",
    "albedo_brightness": "float",
    "diffuse_tint": "color",
    "enable_transmission": "bool",
    "opacity_ratio": "float",
    "opacity_tint": "color",
    "opacity_multiplier": "float",
    "reflection_roughness_constant": "float",
    "reflection_roughness_texture_influence": "float",
    "reflectionroughness_texture": "texture_2d",
    "anisotropy_constant": "float",
    "anisotropy_texture_influence": "float",
    "anisotropy_texture": "texture_2d",
    "metallic_constant": "float",
    "metallic_texture_influence": "float",
    "metallic_texture": "texture_2d",
    "specular_level": "float",
    "specular_constant": "float",
    "specular_texture": "texture_2d",
    "enable_ORM_texture": "bool",
    "ORM_texture": "texture_2d",
    "enable_clearcoat": "bool",
    "clearcoat_tint": "color",
    "clearcoat_transparency": "float",
    "clearcoat_reflection_roughness": "float",
    "clearcoat_weight": "float",
    "clearcoat_flatten": "float",
    "clearcoat_ior": "float",
    "clearcoat_bump_factor": "float",
    "clearcoat_normal_input": "float3",
    "clearcoat_normalmap_texture": "texture_2d",
    "clearcoat_flip_tangent_u": "bool",
    "clearcoat_flip_tangent_v": "bool",
    "enable_retroreflection": "bool",
    "retroreflection_tint": "color",
    "normal_reflectivity": "float",
    "grazing_reflectivity": "float",
    "ao_to_diffuse": "float",
    "ao_input": "float",
    "ao_texture": "texture_2d",
    "enable_emission": "bool",
    "emissive_color": "color",
    "emissive_color_texture": "texture_2d",
    "emissive_mask_input": "color",
    "emissive_mask_texture": "texture_2d",
    "emissive_intensity": "float",
    "enable_emissive_flipbook": "bool",
    "flipbook_dim": "int2",
    "target_fps": "int",
    "enable_opacity": "bool",
    "enable_opacity_texture": "bool",
    "opacity_constant": "float",
    "alpha_constant": "float",
    "opacity_texture": "texture_2d",
    "opacity_mode": "qualified_enum",
    "opacity_threshold": "float",
    "enable_opacity_cutout": "bool",
    "alpha_cutout_cutoff": "float",
    "bump_factor": "float",
    "normal_input": "state_normal",
    "normalmap_texture": "texture_2d",
    "flip_tangent_u": "bool",
    "flip_tangent_v": "bool",
    "detail_bump_factor": "float",
    "detail_normalmap_texture": "texture_2d",
    "project_uvw": "bool",
    "world_or_object": "bool",
    "uv_space_index": "int",
    "texture_translate": "float2",
    "texture_rotate": "float",
    "texture_scale": "float2",
    "texture_speed": "float2",
    "detail_texture_translate": "float2",
    "detail_texture_rotate": "float",
    "detail_texture_scale": "float2",
}
_MDL_OMNIPBR_ENUM_VALUES = {
    "opacity_mode": frozenset(
        {
            "::base::mono_alpha",
            "::base::mono_average",
            "::base::mono_luminance",
            "::base::mono_maximum",
        }
    )
}
_MDL_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))"
    r"(?:[eE][+-]?[0-9]+)?(?:[fFdD])?\Z"
)
_MDL_QUALIFIED_NAME_PATTERN = re.compile(
    rf"(?:::)?{_MDL_IDENTIFIER_SOURCE}"
    rf"(?:::{_MDL_IDENTIFIER_SOURCE})+\Z"
)
_SIM_PBR_SOURCE_SHA256 = (
    "3e465cacc163b3aab07e7a788d21870222e6e1216b4ad3cb0b3191fcce0ec244"
)
_SIM_PBR_MODEL_SOURCE_SHA256 = (
    "78df79d1727d29348df1353272bf21aa139b53d7cc799b83549d36376ace2399"
)
_SIM_PBR_TRANSLUCENT_SOURCE_SHA256 = (
    "6829af54259b898b4237c02d9375067b63f802cbbbde6f9935f2e69a1b11e4bf"
)

_INSTALLED_OMNI_MDL_FILE_SHA256 = {
    "Base/OmniPBR.mdl": (
        "685a82c18b381f1cd29831490e4c356e61dbeac2d86803babc115cd6a4a845db"
    ),
    "Base/OmniPBR_Opacity.mdl": (
        "b9378c8cb61a8732228409ba350994421754d15533ab6c6d7e618cdfe4191470"
    ),
    "Base/OmniPBR_ClearCoat.mdl": (
        "53f5f5e5ca67477f9e105cdc17db13201a876ef54f005978286fef428221bfd8"
    ),
    "Base/OmniPBR_ClearCoat_Opacity.mdl": (
        "17317ed689dcd0dc1924c320a8912b53eeba8c39082638b8ed6d9c807083a5e0"
    ),
    "Base/OmniGlass.mdl": (
        "c3f9555fbf8db2c4d47ac25501afaf23720260255381bde3df003fecf01869af"
    ),
    "Base/OmniGlass_Opacity.mdl": (
        "87facd037b11888d3052b663bb20e41367b769095bde4c9cf686925fa17eb2a4"
    ),
    "Base/OmniPBRBase.mdl": (
        "edcc669a238798a4d90ce36bd535282d23f88cddc544d68a532a5b732432788e"
    ),
    "Base/OmniSurface.mdl": (
        "f16f788a3b8c19524992c6ef01c45d552cf57735c6fddade9da3316bc897b222"
    ),
    "Base/OmniSurfaceLite.mdl": (
        "0175002c1c361e8c016e1131196ed0c76fef911b87b4dd016bb39fc0e77ee23b"
    ),
    "Base/OmniSurfacePresets.mdl": (
        "5c86c8545a1e215ec4b99e60eb66f9112ca5952cc66ca13ec0c26687dcfcb930"
    ),
    "mdl/OmniSurface/OmniSurfaceBase.mdl": (
        "50336376f1298511f22cfe54847487dc96b17f203448cd8d63124a5985d3f895"
    ),
    "mdl/OmniSurface/OmniSurfaceLiteBase.mdl": (
        "832583420e96f4367f8b510c17aa5ae94198a60f38c8dedd59d70fd43d27649e"
    ),
    "mdl/OmniSurface/OmniShared.mdl": (
        "c52816bc2ecb4cebb36f9e28c14bf21f05cef962dd45e89278508a1e1b551275"
    ),
    "mdl/OmniSurface/OmniImage.mdl": (
        "25601c038c9cca778506c8434a4cb7e7cb48f2dbd9a4dff848738a702e2de64e"
    ),
    "mdl/nvidia/aux_definitions.mdl": (
        "168c28acdc30f3a3e0828d006b262644f555951a737b08663c55158868d6361a"
    ),
    "mdl/nvidia/core_definitions.mdl": (
        "e020d5c2865b0c24563b09620b739f8b147661f6e0f221030757a426cfe00e83"
    ),
    "mdl/nvidia/support_definitions.mdl": (
        "c04891737fff406e650e4522e44ec28e8b204c0016f2d4e2de9d7680ca216d73"
    ),
}
_INSTALLED_OMNI_MDL_MODULE_SPECS = {
    "OmniPBR.mdl": {
        "root": "Base/OmniPBR.mdl",
        "family": "OmniPBR",
        "sub_identifiers": frozenset({"OmniPBR"}),
        "files": (
            "Base/OmniPBR.mdl",
            "Base/OmniPBR_ClearCoat.mdl",
            "Base/OmniPBRBase.mdl",
        ),
    },
    "OmniPBR_Opacity.mdl": {
        "root": "Base/OmniPBR_Opacity.mdl",
        "family": "OmniPBR",
        "sub_identifiers": frozenset({"OmniPBR_Opacity"}),
        "files": (
            "Base/OmniPBR_Opacity.mdl",
            "Base/OmniPBR_ClearCoat_Opacity.mdl",
            "Base/OmniPBRBase.mdl",
        ),
    },
    "OmniGlass.mdl": {
        "root": "Base/OmniGlass.mdl",
        "family": "OmniGlass",
        "sub_identifiers": frozenset({"OmniGlass"}),
        "files": (
            "Base/OmniGlass.mdl",
            "Base/OmniGlass_Opacity.mdl",
            "Base/OmniPBRBase.mdl",
        ),
    },
    "OmniSurface.mdl": {
        "root": "Base/OmniSurface.mdl",
        "family": "OmniSurface",
        "sub_identifiers": frozenset({"OmniSurface"}),
        "files": (
            "Base/OmniSurface.mdl",
            "mdl/OmniSurface/OmniSurfaceBase.mdl",
            "mdl/OmniSurface/OmniShared.mdl",
            "mdl/OmniSurface/OmniImage.mdl",
        ),
    },
    "OmniSurfaceLite.mdl": {
        "root": "Base/OmniSurfaceLite.mdl",
        "family": "OmniSurface",
        "sub_identifiers": frozenset({"OmniSurfaceLite"}),
        "files": (
            "Base/OmniSurfaceLite.mdl",
            "mdl/OmniSurface/OmniSurfaceLiteBase.mdl",
            "mdl/OmniSurface/OmniShared.mdl",
            "mdl/OmniSurface/OmniImage.mdl",
        ),
    },
    "OmniSurfacePresets.mdl": {
        "root": "Base/OmniSurfacePresets.mdl",
        "family": "OmniSurfacePreset",
        "sub_identifiers": frozenset(
            {
                "OmniSurface_ClearWater",
                "OmniSurface_FrostedGlass",
                "OmniSurface_Velvet",
                "OmniSurface_Wax",
            }
        ),
        "files": (
            "Base/OmniSurfacePresets.mdl",
            "Base/OmniSurface.mdl",
            "Base/OmniSurfaceLite.mdl",
            "mdl/OmniSurface/OmniSurfaceBase.mdl",
            "mdl/OmniSurface/OmniSurfaceLiteBase.mdl",
            "mdl/OmniSurface/OmniShared.mdl",
            "mdl/OmniSurface/OmniImage.mdl",
        ),
    },
}
_INSTALLED_OMNI_MDL_RESOLUTION_METHOD = (
    "pinned_installed_isaac_omni_mdl_module"
)

_ASSET_LOCAL_OMNI_MDL_FILE_SHA256 = {
    "OmniSurface.mdl": (
        "2fe52255e29cae1cb65a1ff1299773380ee6703e14f7654692bf8b195567f446"
    ),
    "OmniSurfaceLite.mdl": (
        "e20b8cb76b42afcbe2ea425636707b062cad71fa27ad8e1dcb89cb52cbaa7315"
    ),
    "OmniSurfacePresets.mdl": (
        "3d964ae0df3e2283b8b4236916fa17f9146ba740d5eda079551a74fe98ed6f0b"
    ),
}
_ASSET_LOCAL_OMNI_MDL_ROOT_SPECS = {
    "OmniSurface.mdl": {
        "family": "OmniSurface",
        "sub_identifiers": frozenset({"OmniSurface"}),
        "local_files": ("OmniSurface.mdl",),
        "installed_files": (
            "mdl/OmniSurface/OmniSurfaceBase.mdl",
            "mdl/OmniSurface/OmniShared.mdl",
            "mdl/OmniSurface/OmniImage.mdl",
        ),
    },
    "OmniSurfaceLite.mdl": {
        "family": "OmniSurface",
        "sub_identifiers": frozenset({"OmniSurfaceLite"}),
        "local_files": ("OmniSurfaceLite.mdl",),
        "installed_files": (
            "mdl/OmniSurface/OmniSurfaceLiteBase.mdl",
            "mdl/OmniSurface/OmniShared.mdl",
            "mdl/OmniSurface/OmniImage.mdl",
        ),
    },
    "OmniSurfacePresets.mdl": {
        "family": "OmniSurfacePreset",
        "sub_identifiers": frozenset(
            {
                "OmniSurface_ClearWater",
                "OmniSurface_FrostedGlass",
                "OmniSurface_Velvet",
                "OmniSurface_Wax",
            }
        ),
        "local_files": (
            "OmniSurfacePresets.mdl",
            "OmniSurface.mdl",
            "OmniSurfaceLite.mdl",
        ),
        "installed_files": (
            "mdl/OmniSurface/OmniSurfaceBase.mdl",
            "mdl/OmniSurface/OmniSurfaceLiteBase.mdl",
            "mdl/OmniSurface/OmniShared.mdl",
            "mdl/OmniSurface/OmniImage.mdl",
        ),
    },
}
_ASSET_LOCAL_OMNI_MDL_RESOLUTION_METHOD = (
    "hash_attested_asset_local_omnisurface_module"
)

_ASSET_LOCAL_OMNIPBR_MDL_FILE_SHA256 = {
    "OmniPBR_Opacity.mdl": (
        "fbd3447652dbbaa2169c6e2f3887c3f3a7133800b41a40d7748d454ad357a3ce"
    ),
}
_ASSET_LOCAL_OMNIPBR_MDL_ROOT_SPECS = {
    "OmniPBR_Opacity.mdl": {
        "family": "OmniPBR",
        "sub_identifiers": frozenset({"OmniPBR_Opacity"}),
        "local_files": ("OmniPBR_Opacity.mdl",),
        "installed_files": (
            "Base/OmniPBR_ClearCoat_Opacity.mdl",
            "Base/OmniPBRBase.mdl",
        ),
    },
}
_ASSET_LOCAL_OMNIPBR_MDL_RESOLUTION_METHOD = (
    "hash_attested_asset_local_omni_mdl_module"
)


def _installed_omni_mdl_core_roots() -> list[Path]:
    purelib = sysconfig.get_path("purelib")
    if not purelib:
        return []
    candidate = Path(purelib) / "omni" / "mdl" / "core"
    try:
        return [candidate.resolve(strict=True)] if candidate.is_dir() else []
    except OSError:
        return []


def _installed_omni_mdl_path(
    core_root: Path, relative_path: str
) -> Path | None:
    try:
        root = core_root.resolve(strict=True)
        candidate = (root / relative_path).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _resolve_installed_omni_mdl_module(
    authored_source_asset: str,
    sub_identifier: str | None,
) -> tuple[Path, dict[str, Any]] | None:
    spec = _INSTALLED_OMNI_MDL_MODULE_SPECS.get(authored_source_asset)
    if spec is None:
        return None
    selected_identifier = (
        sub_identifier.rsplit("::", 1)[-1] if sub_identifier else None
    )
    if selected_identifier not in spec["sub_identifiers"]:
        return None

    verified: list[tuple[Path, list[dict[str, Any]]]] = []
    for root in _installed_omni_mdl_core_roots():
        module_records: list[dict[str, Any]] = []
        for relative_path in spec["files"]:
            path = _installed_omni_mdl_path(root, relative_path)
            if path is None:
                break
            try:
                source_sha256 = _file_sha256(path)
            except OSError:
                break
            if source_sha256 != _INSTALLED_OMNI_MDL_FILE_SHA256[relative_path]:
                break
            module_records.append(
                {
                    "authored_module": (
                        f"::{Path(relative_path).stem}"
                        if relative_path.startswith("Base/")
                        else "::OmniSurface::" + Path(relative_path).stem
                    ),
                    "relative_path": relative_path,
                    "resolved_source_path": str(path),
                    "source_sha256": source_sha256,
                }
            )
        else:
            root_path = _installed_omni_mdl_path(root, spec["root"])
            if root_path is not None:
                verified.append((root_path, module_records))

    unique = {
        (str(path), tuple(record["source_sha256"] for record in records)):
        (path, records)
        for path, records in verified
    }
    if len(unique) != 1:
        return None
    root_path, modules = next(iter(unique.values()))
    return root_path, {
        "classification": _INSTALLED_OMNI_MDL_RESOLUTION_METHOD,
        "authored_source_asset": authored_source_asset,
        "resolved_source_path": str(root_path),
        "resolved_source_sha256": _INSTALLED_OMNI_MDL_FILE_SHA256[
            spec["root"]
        ],
        "material_family": spec["family"],
        "module_files": modules,
    }


def _resolve_installed_omni_mdl_dependency_files(
    relative_paths: Iterable[str],
) -> list[dict[str, Any]] | None:
    expected_paths = tuple(relative_paths)
    verified: list[list[dict[str, Any]]] = []
    for root in _installed_omni_mdl_core_roots():
        module_records: list[dict[str, Any]] = []
        for relative_path in expected_paths:
            path = _installed_omni_mdl_path(root, relative_path)
            if path is None:
                break
            try:
                source_sha256 = _file_sha256(path)
            except OSError:
                break
            if source_sha256 != _INSTALLED_OMNI_MDL_FILE_SHA256.get(
                relative_path
            ):
                break
            module_records.append(
                {
                    "authored_module": "::" + "::".join(
                        Path(relative_path).with_suffix("").parts[1:]
                    ),
                    "relative_path": relative_path,
                    "resolved_source_path": str(path),
                    "source_sha256": source_sha256,
                }
            )
        else:
            verified.append(module_records)

    unique = {
        tuple(
            (record["resolved_source_path"], record["source_sha256"])
            for record in records
        ): records
        for records in verified
    }
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _resolve_hash_attested_asset_local_mdl_module(
    authored_source_asset: str,
    mdl_path: Path,
    source_sha256: str,
    original_resolution_method: str | None,
    *,
    root_specs: Mapping[str, Mapping[str, Any]],
    file_sha256s: Mapping[str, str],
    classification: str,
) -> dict[str, Any] | None:
    filename = mdl_path.name
    spec = root_specs.get(filename)
    if (
        spec is None
        or Path(authored_source_asset).name != filename
        or source_sha256 != file_sha256s[filename]
    ):
        return None

    try:
        bundle_root = mdl_path.resolve(strict=True).parent
    except OSError:
        return None
    local_modules: list[dict[str, Any]] = []
    for relative_path in spec["local_files"]:
        try:
            path = (bundle_root / str(relative_path)).resolve(strict=True)
        except OSError:
            return None
        if path.parent != bundle_root or not path.is_file():
            return None
        try:
            observed_sha256 = _file_sha256(path)
        except OSError:
            return None
        if observed_sha256 != file_sha256s[relative_path]:
            return None
        local_modules.append(
            {
                "authored_module": f"::{Path(relative_path).stem}",
                "relative_path": str(relative_path),
                "resolved_source_path": str(path),
                "source_sha256": observed_sha256,
            }
        )

    installed_modules = _resolve_installed_omni_mdl_dependency_files(
        spec["installed_files"]
    )
    if installed_modules is None:
        return None
    return {
        "classification": classification,
        "authored_source_asset": authored_source_asset,
        "resolved_source_path": str(mdl_path.resolve()),
        "resolved_source_sha256": source_sha256,
        "original_source_resolution_method": original_resolution_method,
        "material_family": spec["family"],
        "module_files": local_modules + installed_modules,
    }


def _resolve_asset_local_omni_mdl_module(
    authored_source_asset: str,
    mdl_path: Path,
    source_sha256: str,
    original_resolution_method: str | None,
) -> dict[str, Any] | None:
    return _resolve_hash_attested_asset_local_mdl_module(
        authored_source_asset,
        mdl_path,
        source_sha256,
        original_resolution_method,
        root_specs=_ASSET_LOCAL_OMNI_MDL_ROOT_SPECS,
        file_sha256s=_ASSET_LOCAL_OMNI_MDL_FILE_SHA256,
        classification=_ASSET_LOCAL_OMNI_MDL_RESOLUTION_METHOD,
    )


def _resolve_asset_local_omnipbr_mdl_module(
    authored_source_asset: str,
    mdl_path: Path,
    source_sha256: str,
    original_resolution_method: str | None,
) -> dict[str, Any] | None:
    return _resolve_hash_attested_asset_local_mdl_module(
        authored_source_asset,
        mdl_path,
        source_sha256,
        original_resolution_method,
        root_specs=_ASSET_LOCAL_OMNIPBR_MDL_ROOT_SPECS,
        file_sha256s=_ASSET_LOCAL_OMNIPBR_MDL_FILE_SHA256,
        classification=_ASSET_LOCAL_OMNIPBR_MDL_RESOLUTION_METHOD,
    )


def _canonical_dependency_modules(
    modules: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    canonical = [
        {
            "authored_module": str(module.get("authored_module") or ""),
            "relative_path": str(module.get("relative_path") or ""),
            "source_sha256": str(module.get("source_sha256") or ""),
        }
        for module in modules
    ]
    return sorted(
        canonical,
        key=lambda item: (
            item["authored_module"],
            item["relative_path"],
            item["source_sha256"],
        ),
    )


def _source_dependency_attestation(
    classification: str, modules: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    serialized_modules = sorted(
        (_json_safe_value(dict(module)) for module in modules),
        key=lambda item: (
            str(item.get("authored_module") or ""),
            str(item.get("relative_path") or ""),
            str(item.get("source_sha256") or ""),
        ),
    )
    canonical = {
        "classification": classification,
        "modules": _canonical_dependency_modules(serialized_modules),
    }
    return {
        "classification": classification,
        "modules": serialized_modules,
        "attestation_sha256": _json_canonical_hash(
            "selected-mdl-source-dependency-attestation", canonical
        ),
    }
_SIM_PBR_DEFAULT_SPECS = (
    ("diffuse_color_constant", "color", (0.2, 0.2, 0.2)),
    ("diffuse_texture", "texture_2d", None),
    ("diffuse_tint", "color", (1.0, 1.0, 1.0)),
    ("enable_transmission", "bool", False),
    ("opacity_ratio", "float", 1.0),
    ("opacity_tint", "color", (1.0, 1.0, 1.0)),
    ("opacity_multiplier", "float", 1.0),
    ("reflection_roughness_constant", "float", 0.5),
    ("reflection_roughness_texture_influence", "float", 0.0),
    ("reflectionroughness_texture", "texture_2d", None),
    ("anisotropy_constant", "float", 0.0),
    ("anisotropy_texture_influence", "float", 0.0),
    ("anisotropy_texture", "texture_2d", None),
    ("metallic_constant", "float", 0.0),
    ("metallic_texture_influence", "float", 0.0),
    ("metallic_texture", "texture_2d", None),
    ("specular_constant", "float", 1.0),
    ("specular_texture", "texture_2d", None),
    ("enable_ORM_texture", "bool", True),
    ("ORM_texture", "texture_2d", None),
    ("enable_clearcoat", "bool", False),
    ("clearcoat_tint", "color", (1.0, 1.0, 1.0)),
    ("clearcoat_transparency", "float", 1.0),
    ("clearcoat_reflection_roughness", "float", 0.0),
    ("clearcoat_weight", "float", 1.0),
    ("clearcoat_flatten", "float", 1.0),
    ("clearcoat_ior", "float", 1.56),
    ("clearcoat_bump_factor", "float", 1.0),
    ("clearcoat_normal_input", "float3", (0.0, 0.0, 0.0)),
    ("clearcoat_normalmap_texture", "texture_2d", None),
    ("clearcoat_flip_tangent_u", "bool", False),
    ("clearcoat_flip_tangent_v", "bool", True),
    ("enable_retroreflection", "bool", False),
    ("retroreflection_tint", "color", (0.182, 0.8, 0.0)),
    ("normal_reflectivity", "float", 0.5),
    ("grazing_reflectivity", "float", 1.0),
    ("ao_to_diffuse", "float", 0.0),
    ("ao_input", "float", 1.0),
    ("ao_texture", "texture_2d", None),
    ("enable_emission", "bool", False),
    ("emissive_color", "color", (1.0, 0.1, 0.1)),
    ("emissive_mask_input", "color", (1.0, 1.0, 1.0)),
    ("emissive_mask_texture", "texture_2d", None),
    ("emissive_intensity", "float", 40.0),
    ("enable_emissive_flipbook", "bool", False),
    ("flipbook_dim", "int2", (1, 1)),
    ("target_fps", "int", 1),
    ("enable_opacity", "bool", False),
    ("alpha_constant", "float", 1.0),
    ("opacity_texture", "texture_2d", None),
    ("enable_opacity_cutout", "bool", False),
    ("alpha_cutout_cutoff", "float", 1.0),
    ("bump_factor", "float", 1.0),
    ("normal_input", "state_normal", None),
    ("normalmap_texture", "texture_2d", None),
    ("flip_tangent_u", "bool", False),
    ("flip_tangent_v", "bool", True),
    ("project_uvw", "bool", False),
    ("world_or_object", "bool", False),
    ("texture_translate", "float2", (0.0, 0.0)),
    ("texture_scale", "float2", (1.0, 1.0)),
    ("texture_speed", "float2", (0.0, 0.0)),
)

def _lex_mdl_source(source_text: str) -> list[_MdlToken]:
    """Lex enough MDL to select one material without regex false positives.

    Comments are skipped only while outside strings.  String boundaries and
    all delimiters remain tokens, so a comment containing an apparent export
    or texture constructor can never become evidence.  This is deliberately
    a small recognizer rather than a permissive MDL parser: malformed lexical
    structure is a hard failure.
    """

    tokens: list[_MdlToken] = []
    index = 0
    length = len(source_text)
    while index < length:
        character = source_text[index]
        if character.isspace() or character == "\ufeff":
            index += 1
            continue
        if source_text.startswith("//", index):
            newline = source_text.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if source_text.startswith("/*", index):
            close = source_text.find("*/", index + 2)
            if close < 0:
                raise _MdlParseError(
                    "unterminated_block_comment",
                    "MDL source contains an unterminated block comment",
                )
            index = close + 2
            continue
        if character == '"':
            start = index
            index += 1
            content_start = index
            while index < length:
                if source_text[index] == "\\":
                    index += 2
                    continue
                if source_text[index] == '"':
                    tokens.append(
                        _MdlToken(
                            "string",
                            source_text[content_start:index],
                            start,
                            index + 1,
                        )
                    )
                    index += 1
                    break
                if source_text[index] in "\r\n":
                    raise _MdlParseError(
                        "newline_in_string",
                        "MDL source contains a newline in a string literal",
                    )
                index += 1
            else:
                raise _MdlParseError(
                    "unterminated_string",
                    "MDL source contains an unterminated string literal",
                )
            continue
        if character.isalpha() or character == "_":
            start = index
            index += 1
            while index < length and (
                source_text[index].isalnum() or source_text[index] == "_"
            ):
                index += 1
            tokens.append(
                _MdlToken("identifier", source_text[start:index], start, index)
            )
            continue
        tokens.append(_MdlToken("symbol", character, index, index + 1))
        index += 1
    return tokens


def _validate_mdl_delimiters(tokens: list[_MdlToken]) -> None:
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[_MdlToken] = []
    for token in tokens:
        if token.value in opening:
            stack.append(token)
        elif token.value in closing:
            if not stack or stack[-1].value != closing[token.value]:
                raise _MdlParseError(
                    "unbalanced_delimiter",
                    f"unexpected MDL delimiter {token.value!r}",
                )
            stack.pop()
    if stack:
        raise _MdlParseError(
            "unbalanced_delimiter",
            f"unterminated MDL delimiter {stack[-1].value!r}",
        )


def _mdl_declaration_end(
    tokens: list[_MdlToken], start_index: int
) -> int:
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    for index in range(start_index, len(tokens)):
        value = tokens[index].value
        if value in opening:
            stack.append(value)
        elif value in closing:
            if not stack or stack[-1] != closing[value]:
                raise _MdlParseError(
                    "malformed_material_definition",
                    "selected MDL material has unbalanced delimiters",
                )
            stack.pop()
        elif value == ";" and not stack:
            return index
    raise _MdlParseError(
        "missing_material_semicolon",
        "exported MDL material does not end at a top-level semicolon",
    )


def _mdl_export_material_declarations(
    tokens: list[_MdlToken], source_text: str
) -> list[dict[str, Any]]:
    """Locate syntactically bounded, top-level ``export material`` entries."""

    _validate_mdl_delimiters(tokens)
    declarations: list[dict[str, Any]] = []
    stack: list[str] = []
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            not stack
            and token.kind == "identifier"
            and token.value == "export"
            and index + 1 < len(tokens)
            and tokens[index + 1].kind == "identifier"
            and tokens[index + 1].value == "material"
        ):
            if index + 3 >= len(tokens):
                raise _MdlParseError(
                    "malformed_material_definition",
                    "export material declaration is incomplete",
                )
            name_token = tokens[index + 2]
            if (
                name_token.kind != "identifier"
                or not _MDL_IDENTIFIER_PATTERN.fullmatch(name_token.value)
                or tokens[index + 3].value != "("
            ):
                raise _MdlParseError(
                    "malformed_material_definition",
                    "export material must be followed by a name and signature",
                )
            end_index = _mdl_declaration_end(tokens, index)
            signature_close_index = _mdl_matching_close(
                tokens,
                index + 3,
                end_index,
                error_code="malformed_material_definition",
                description=f"material {name_token.value!r} signature",
            )
            relative_stack: list[str] = []
            equals_indices: list[int] = []
            for candidate_index in range(index + 3, end_index):
                value = tokens[candidate_index].value
                if value in opening:
                    relative_stack.append(value)
                elif value in closing:
                    if not relative_stack or relative_stack[-1] != closing[value]:
                        raise _MdlParseError(
                            "malformed_material_definition",
                            f"material {name_token.value!r} has unbalanced syntax",
                        )
                    relative_stack.pop()
                elif value == "=" and not relative_stack:
                    equals_indices.append(candidate_index)
            if len(equals_indices) != 1 or equals_indices[0] + 1 >= end_index:
                raise _MdlParseError(
                    "malformed_material_definition",
                    f"material {name_token.value!r} must have one top-level definition expression",
                )
            declarations.append(
                {
                    "name": name_token.value,
                    "start_token_index": index,
                    "signature_open_token_index": index + 3,
                    "signature_close_token_index": signature_close_index,
                    "equals_token_index": equals_indices[0],
                    "end_token_index": end_index,
                    "source_start": token.start,
                    "source_end": tokens[end_index].end,
                    "source_line": source_text.count("\n", 0, token.start) + 1,
                }
            )
            index = end_index + 1
            continue
        if token.value in opening:
            stack.append(token.value)
        elif token.value in closing:
            # Global balance was checked above, so this is necessarily valid.
            stack.pop()
        index += 1
    return declarations


def _mdl_matching_close(
    tokens: list[_MdlToken],
    open_index: int,
    limit: int,
    *,
    error_code: str = "malformed_expression",
    description: str = "MDL expression",
) -> int:
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    first = tokens[open_index].value
    if first not in opening:
        raise _MdlParseError(
            error_code,
            f"{description} has no opening delimiter",
        )
    stack = [first]
    for index in range(open_index + 1, limit):
        value = tokens[index].value
        if value in opening:
            stack.append(value)
        elif value in closing:
            if stack[-1] != closing[value]:
                raise _MdlParseError(
                    error_code,
                    f"{description} contains unbalanced delimiters",
                )
            stack.pop()
            if not stack:
                return index
    raise _MdlParseError(
        error_code,
        f"{description} has no balanced closing delimiter",
    )


def _mdl_split_arguments(
    tokens: list[_MdlToken],
    start_index: int,
    end_index: int,
    *,
    error_code: str = "malformed_expression",
    description: str = "MDL expression",
) -> list[list[_MdlToken]]:
    if start_index >= end_index:
        return []
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    arguments: list[list[_MdlToken]] = []
    argument_start = start_index
    for index in range(start_index, end_index):
        value = tokens[index].value
        if value in opening:
            stack.append(value)
        elif value in closing:
            if not stack or stack[-1] != closing[value]:
                raise _MdlParseError(
                    error_code,
                    f"{description} arguments are unbalanced",
                )
            stack.pop()
        elif value == "," and not stack:
            arguments.append(tokens[argument_start:index])
            argument_start = index + 1
    arguments.append(tokens[argument_start:end_index])
    if any(not argument for argument in arguments):
        raise _MdlParseError(
            error_code,
            f"{description} has an empty argument",
        )
    return arguments


def _decode_mdl_string_literal(raw_content: str) -> str:
    decoded: list[str] = []
    index = 0
    simple_escapes = {
        "\\": "\\",
        '"': '"',
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "b": "\b",
        "f": "\f",
    }
    while index < len(raw_content):
        character = raw_content[index]
        if character != "\\":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(raw_content):
            raise _MdlParseError(
                "invalid_string_escape",
                "MDL string literal ends with an escape character",
            )
        escape = raw_content[index + 1]
        if escape in simple_escapes:
            decoded.append(simple_escapes[escape])
            index += 2
            continue
        width = {"x": 2, "u": 4, "U": 8}.get(escape)
        if width is None:
            raise _MdlParseError(
                "invalid_string_escape",
                f"unsupported MDL string escape \\{escape}",
            )
        digits = raw_content[index + 2 : index + 2 + width]
        if len(digits) != width or not all(
            digit in "0123456789abcdefABCDEF" for digit in digits
        ):
            raise _MdlParseError(
                "invalid_string_escape",
                f"malformed MDL hexadecimal escape \\{escape}{digits}",
            )
        try:
            decoded.append(chr(int(digits, 16)))
        except (ValueError, OverflowError):
            raise _MdlParseError(
                "invalid_string_escape",
                "MDL string escape is outside the Unicode range",
            ) from None
        index += 2 + width
    value = "".join(decoded)
    if any(ord(character) < 32 for character in value):
        raise _MdlParseError(
            "invalid_texture_literal",
            "MDL texture path contains a control character",
        )
    return value


def _mdl_gamma_provenance(tokens: list[_MdlToken]) -> tuple[str, str, str]:
    expression = "".join(token.value for token in tokens)
    if not re.fullmatch(
        r"(?:::)?[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*",
        expression,
    ):
        raise _MdlParseError(
            "dynamic_texture_gamma",
            "texture_2d gamma must be a static qualified gamma-mode name",
        )
    gamma_mode = expression.rsplit("::", 1)[-1]
    color_space = _MDL_GAMMA_COLOR_SPACES.get(gamma_mode)
    if color_space is None:
        raise _MdlParseError(
            "unsupported_texture_gamma",
            f"texture_2d uses unsupported gamma mode {expression!r}",
        )
    return expression, gamma_mode, color_space


def _mdl_top_level_statements(
    tokens: list[_MdlToken],
) -> list[tuple[str, ...]]:
    """Return token values for complete top-level MDL statements."""

    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    statements: list[tuple[str, ...]] = []
    statement_start = 0
    for index, token in enumerate(tokens):
        value = token.value
        if value in opening:
            stack.append(value)
        elif value in closing:
            if not stack or stack[-1] != closing[value]:
                raise _MdlParseError(
                    "unbalanced_delimiter",
                    f"unexpected MDL delimiter {value!r}",
                )
            stack.pop()
        elif value == ";" and not stack:
            statements.append(
                tuple(item.value for item in tokens[statement_start:index])
            )
            statement_start = index + 1
    if statement_start != len(tokens):
        raise _MdlParseError(
            "incomplete_top_level_statement",
            "MDL source ends outside a complete top-level statement",
        )
    return statements


def _mdl_attest_omnipbr_callee(
    tokens: list[_MdlToken], callee_expression: str
) -> str:
    """Resolve one supported call spelling to the exact OmniPBR symbol."""

    statements = set(_mdl_top_level_statements(tokens))
    using_statement = (
        "using",
        ":",
        ":",
        "OmniPBR",
        "import",
        "OmniPBR",
    )
    absolute_import = (
        "import",
        ":",
        ":",
        "OmniPBR",
        ":",
        ":",
        "OmniPBR",
    )
    legacy_import = (
        "import",
        "OmniPBR",
        ":",
        ":",
        "OmniPBR",
    )
    if callee_expression == "OmniPBR" and using_statement in statements:
        return "::OmniPBR::OmniPBR"
    if callee_expression in {
        "OmniPBR::OmniPBR",
        "::OmniPBR::OmniPBR",
    } and (absolute_import in statements or legacy_import in statements):
        return "::OmniPBR::OmniPBR"
    raise _MdlParseError(
        "unsupported_or_unverified_material_callee",
        "selected material must directly call the exactly imported "
        "::OmniPBR::OmniPBR symbol",
    )


def _mdl_finite_float(tokens: list[_MdlToken]) -> float:
    expression = "".join(token.value for token in tokens)
    if not _MDL_NUMBER_PATTERN.fullmatch(expression):
        raise _MdlParseError(
            "dynamic_or_wrong_typed_argument",
            f"expected a static numeric literal, found {expression!r}",
        )
    if expression[-1:] in {"f", "F", "d", "D"}:
        expression = expression[:-1]
    try:
        value = float(expression)
    except ValueError:
        raise _MdlParseError(
            "invalid_numeric_literal",
            f"could not decode MDL numeric literal {expression!r}",
        ) from None
    if not math.isfinite(value):
        raise _MdlParseError(
            "nonfinite_numeric_literal",
            "MDL numeric defaults must be finite",
        )
    return 0.0 if value == 0.0 else value


def _mdl_static_constructor_components(
    tokens: list[_MdlToken], constructor: str, arities: frozenset[int]
) -> list[float]:
    if (
        len(tokens) < 3
        or tokens[0].kind != "identifier"
        or tokens[0].value != constructor
        or tokens[1].value != "("
    ):
        raise _MdlParseError(
            "dynamic_or_wrong_typed_argument",
            f"expected a direct {constructor} constructor",
        )
    close_index = _mdl_matching_close(
        tokens,
        1,
        len(tokens),
        error_code="malformed_static_constructor",
        description=f"{constructor} constructor",
    )
    if close_index != len(tokens) - 1:
        raise _MdlParseError(
            "dynamic_or_wrong_typed_argument",
            f"{constructor} constructor has a trailing expression",
        )
    arguments = _mdl_split_arguments(
        tokens,
        2,
        close_index,
        error_code="malformed_static_constructor",
        description=f"{constructor} constructor",
    )
    if len(arguments) not in arities:
        expected = " or ".join(str(value) for value in sorted(arities))
        raise _MdlParseError(
            "wrong_static_constructor_arity",
            f"{constructor} requires {expected} numeric component(s)",
        )
    return [_mdl_finite_float(argument) for argument in arguments]


def _mdl_static_texture_value(
    tokens: list[_MdlToken], input_name: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if (
        len(tokens) < 3
        or tokens[0].kind != "identifier"
        or tokens[0].value != "texture_2d"
        or tokens[1].value != "("
    ):
        raise _MdlParseError(
            "dynamic_texture_expression",
            f"texture input {input_name!r} is not a direct texture_2d constructor",
        )
    close_index = _mdl_matching_close(
        tokens,
        1,
        len(tokens),
        error_code="malformed_texture_constructor",
        description="texture_2d constructor",
    )
    if close_index != len(tokens) - 1:
        raise _MdlParseError(
            "dynamic_texture_expression",
            f"texture input {input_name!r} has a trailing expression",
        )
    arguments = _mdl_split_arguments(
        tokens,
        2,
        close_index,
        error_code="malformed_texture_constructor",
        description="texture_2d constructor",
    )
    if not arguments:
        authored_asset = None
        gamma_expression = None
        gamma_mode = "gamma_default"
        color_space = "default"
        gamma_is_implicit = True
    else:
        if len(arguments) not in {1, 2} or (
            len(arguments[0]) != 1 or arguments[0][0].kind != "string"
        ):
            raise _MdlParseError(
                "dynamic_texture_expression",
                f"texture input {input_name!r} requires one literal path "
                "and an optional static gamma mode",
            )
        decoded_asset = _decode_mdl_string_literal(arguments[0][0].value)
        authored_asset = decoded_asset or None
        gamma_is_implicit = len(arguments) == 1
        if gamma_is_implicit:
            gamma_expression = None
            gamma_mode = "gamma_default"
            color_space = "default"
        else:
            gamma_expression, gamma_mode, color_space = (
                _mdl_gamma_provenance(arguments[1])
            )
    if authored_asset is not None:
        suffix = Path(
            authored_asset.replace("<UDIM>", "1001").replace(
                "<udim>", "1001"
            )
        ).suffix.casefold()
        if suffix not in IMAGE_SUFFIXES:
            raise _MdlParseError(
                "unsupported_texture_literal",
                f"texture input {input_name!r} does not name a supported image",
            )
    value = {
        "authored_asset": authored_asset,
        "gamma_expression": gamma_expression,
        "gamma_mode": gamma_mode,
        "gamma_is_implicit": gamma_is_implicit,
        "color_space": color_space,
    }
    texture = None
    if authored_asset is not None:
        texture = {
            "input_name": input_name,
            **value,
            "constructor_source_start": tokens[0].start,
            "constructor_source_end": tokens[-1].end,
        }
    return {"value_type": "texture_2d", "value": value}, texture


def _mdl_static_argument_value(
    tokens: list[_MdlToken],
    input_name: str,
    expected_type: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not tokens:
        raise _MdlParseError(
            "missing_named_argument_value",
            f"material input {input_name!r} has no value",
        )
    if expected_type == "texture_2d":
        return _mdl_static_texture_value(tokens, input_name)
    expression = "".join(token.value for token in tokens)
    if expected_type == "bool":
        if len(tokens) != 1 or expression not in {"true", "false"}:
            raise _MdlParseError(
                "dynamic_or_wrong_typed_argument",
                f"material input {input_name!r} requires a bool literal",
            )
        return {
            "value_type": "bool",
            "value": expression == "true",
        }, None
    if expected_type == "float":
        if (
            len(tokens) >= 3
            and tokens[0].value == "float"
            and tokens[1].value == "("
        ):
            components = _mdl_static_constructor_components(
                tokens, "float", frozenset({1})
            )
            return {"value_type": "float", "value": components[0]}, None
        return {
            "value_type": "float",
            "value": _mdl_finite_float(tokens),
        }, None
    if expected_type == "int":
        if not re.fullmatch(r"[+-]?[0-9]+", expression):
            raise _MdlParseError(
                "dynamic_or_wrong_typed_argument",
                f"material input {input_name!r} requires an int literal",
            )
        value = int(expression)
        if not -(2**31) <= value < 2**31:
            raise _MdlParseError(
                "integer_literal_out_of_range",
                f"material input {input_name!r} is outside the MDL int range",
            )
        return {"value_type": "int", "value": value}, None
    if expected_type == "float3" and expression in {
        "state::normal()",
        "::state::normal()",
    }:
        return {
            "value_type": "state_normal",
            "value": "::state::normal()",
        }, None
    if expected_type in {"color", "float2", "float3"}:
        width = {"color": 3, "float2": 2, "float3": 3}[expected_type]
        arities = frozenset({0, 1, width})
        components = _mdl_static_constructor_components(
            tokens, expected_type, arities
        )
        value = (
            [0.0] * width
            if not components
            else components * width
            if len(components) == 1
            else components
        )
        return {"value_type": expected_type, "value": value}, None
    if expected_type == "int2":
        components = _mdl_static_constructor_components(
            tokens, "int2", frozenset({0, 1, 2})
        )
        if any(value != int(value) for value in components):
            raise _MdlParseError(
                "dynamic_or_wrong_typed_argument",
                f"material input {input_name!r} requires integer components",
            )
        values = [int(value) for value in components]
        if not values:
            values = [0, 0]
        elif len(values) == 1:
            values *= 2
        return {"value_type": "int2", "value": values}, None
    if expected_type == "state_normal":
        if expression not in {"state::normal()", "::state::normal()"}:
            raise _MdlParseError(
                "dynamic_or_wrong_typed_argument",
                f"material input {input_name!r} requires state::normal()",
            )
        return {
            "value_type": "state_normal",
            "value": "::state::normal()",
        }, None
    if expected_type == "qualified_enum":
        if not _MDL_QUALIFIED_NAME_PATTERN.fullmatch(expression):
            raise _MdlParseError(
                "dynamic_or_wrong_typed_argument",
                f"material input {input_name!r} requires a qualified enum literal",
            )
        canonical_value = f"::{expression.lstrip(':')}"
        if canonical_value not in _MDL_OMNIPBR_ENUM_VALUES.get(
            input_name, frozenset()
        ):
            raise _MdlParseError(
                "unsupported_enum_literal",
                f"material input {input_name!r} uses unsupported enum "
                f"{canonical_value!r}",
            )
        return {
            "value_type": "qualified_enum",
            "value": canonical_value,
        }, None
    if expected_type == "known_qualified_enum":
        if not _MDL_QUALIFIED_NAME_PATTERN.fullmatch(expression):
            raise _MdlParseError(
                "dynamic_or_wrong_typed_argument",
                f"material input {input_name!r} requires a qualified enum literal",
            )
        return {
            "value_type": "qualified_enum",
            "value": f"::{expression.lstrip(':')}",
        }, None
    raise _MdlParseError(
        "unsupported_argument_type",
        f"material input {input_name!r} has unsupported type {expected_type!r}",
    )


def _mdl_selected_material_declaration(
    declarations: list[dict[str, Any]], sub_identifier: str | None
) -> dict[str, Any]:
    if sub_identifier is not None:
        qualified_identifier_pattern = re.compile(
            rf"(?:\.?::)?{_MDL_IDENTIFIER_SOURCE}"
            rf"(?:::{_MDL_IDENTIFIER_SOURCE})*\Z"
        )
        if not qualified_identifier_pattern.fullmatch(sub_identifier):
            raise _MdlParseError(
                "invalid_sub_identifier",
                "MDL sourceAsset subIdentifier is not a material identifier",
            )
        selected_identifier = sub_identifier.rsplit("::", 1)[-1]
        selected = [
            declaration
            for declaration in declarations
            if declaration["name"] == selected_identifier
        ]
        if not selected:
            raise _MdlParseError(
                "selected_material_not_found",
                f"no export material matches subIdentifier {sub_identifier!r}",
            )
        if len(selected) != 1:
            raise _MdlParseError(
                "multiple_selected_material_definitions",
                f"multiple export material declarations match {sub_identifier!r}",
            )
        return selected[0]
    if not declarations:
        raise _MdlParseError(
            "no_exported_material",
            "MDL source contains no export material declaration",
        )
    if len(declarations) != 1:
        raise _MdlParseError(
            "ambiguous_material_without_sub_identifier",
            "MDL source has multiple exported materials but no subIdentifier",
        )
    return declarations[0]


def _mdl_signature_parameter_defaults(
    tokens: list[_MdlToken], declaration: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    arguments = _mdl_split_arguments(
        tokens,
        int(declaration["signature_open_token_index"]) + 1,
        int(declaration["signature_close_token_index"]),
        error_code="malformed_material_signature",
        description=f"material {declaration['name']!r} signature",
    )
    if len(arguments) == 1 and len(arguments[0]) == 1 and arguments[0][0].value == "*":
        return {}, []

    defaults: dict[str, dict[str, Any]] = {}
    textures: list[dict[str, Any]] = []
    simple_types = {
        "bool",
        "color",
        "float",
        "float2",
        "float3",
        "int",
        "int2",
        "texture_2d",
    }
    for argument in arguments:
        stack: list[str] = []
        equals_index = None
        opening = {"(": ")", "[": "]", "{": "}"}
        closing = {value: key for key, value in opening.items()}
        for index, token in enumerate(argument):
            if token.value in opening:
                stack.append(token.value)
            elif token.value in closing:
                if not stack or stack[-1] != closing[token.value]:
                    raise _MdlParseError(
                        "malformed_material_signature",
                        "material signature parameter has unbalanced syntax",
                    )
                stack.pop()
            elif token.value == "=" and not stack:
                if equals_index is not None:
                    raise _MdlParseError(
                        "malformed_material_signature",
                        "material signature parameter has multiple defaults",
                    )
                equals_index = index
        if equals_index is None or equals_index < 2:
            raise _MdlParseError(
                "material_parameter_without_static_default",
                f"material {declaration['name']!r} has a parameter without a default",
            )
        left = argument[:equals_index]
        while left and left[0].value in {"uniform", "varying"}:
            left = left[1:]
        if len(left) < 2 or left[-1].kind != "identifier":
            raise _MdlParseError(
                "malformed_material_signature",
                "material signature parameter has no exact type and name",
            )
        input_name = left[-1].value
        if input_name in defaults:
            raise _MdlParseError(
                "duplicate_material_parameter",
                f"material signature repeats parameter {input_name!r}",
            )
        authored_type = "".join(token.value for token in left[:-1])
        expected_type = (
            authored_type
            if authored_type in simple_types
            else "known_qualified_enum"
        )

        default_tokens = argument[equals_index + 1 :]
        annotation_start = None
        stack = []
        for index, token in enumerate(default_tokens):
            if token.value == "[" and not stack:
                annotation_start = index
                break
            if token.value in opening:
                stack.append(token.value)
            elif token.value in closing:
                if not stack or stack[-1] != closing[token.value]:
                    raise _MdlParseError(
                        "malformed_material_signature",
                        "material parameter default has unbalanced syntax",
                    )
                stack.pop()
        if annotation_start is not None:
            default_tokens = default_tokens[:annotation_start]
        value_record, texture = _mdl_static_argument_value(
            default_tokens, input_name, expected_type
        )
        defaults[input_name] = value_record
        if texture is not None:
            textures.append(texture)
    return defaults, textures


def _mdl_direct_call(
    tokens: list[_MdlToken], declaration: Mapping[str, Any]
) -> tuple[str, list[list[_MdlToken]]]:
    expression_tokens = tokens[
        int(declaration["equals_token_index"]) + 1 :
        int(declaration["end_token_index"])
    ]
    call_open_index = next(
        (
            index
            for index, token in enumerate(expression_tokens)
            if token.value == "("
        ),
        None,
    )
    if call_open_index is None:
        raise _MdlParseError(
            "malformed_material_definition",
            "selected material definition is not a direct call",
        )
    authored_callee = "".join(
        token.value for token in expression_tokens[:call_open_index]
    )
    if not (
        _MDL_IDENTIFIER_PATTERN.fullmatch(authored_callee)
        or _MDL_QUALIFIED_NAME_PATTERN.fullmatch(authored_callee)
        or authored_callee
        == "..::Templates::GlassWithVolume::GlassWithVolume"
    ):
        raise _MdlParseError(
            "malformed_material_callee",
            "selected material call has a dynamic or malformed callee",
        )
    call_close_index = _mdl_matching_close(
        expression_tokens,
        call_open_index,
        len(expression_tokens),
        error_code="malformed_material_definition",
        description="selected material call",
    )
    if call_close_index != len(expression_tokens) - 1:
        raise _MdlParseError(
            "nonsole_material_definition_expression",
            "selected material definition must contain only one direct call",
        )
    arguments = _mdl_split_arguments(
        expression_tokens,
        call_open_index + 1,
        call_close_index,
        error_code="malformed_material_arguments",
        description="selected material call",
    )
    return authored_callee, arguments


def _mdl_attest_supported_composed_callee(
    tokens: list[_MdlToken], authored_callee: str
) -> tuple[str, str, str, str]:
    statements = {"".join(statement) for statement in _mdl_top_level_statements(tokens)}
    candidates = {
        "OmniSurface": (
            "::OmniSurface::OmniSurface",
            "OmniSurface",
            "OmniSurface.mdl",
            "OmniSurface",
            {
                "import::OmniSurface::OmniSurface",
                "import::OmniSurface::*",
                "importOmniSurface::*",
                "using::OmniSurfaceimportOmniSurface",
            },
        ),
        "OmniSurfaceLite": (
            "::OmniSurfaceLite::OmniSurfaceLite",
            "OmniSurface",
            "OmniSurfaceLite.mdl",
            "OmniSurfaceLite",
            {
                "import::OmniSurfaceLite::OmniSurfaceLite",
                "import::OmniSurfaceLite::*",
                "importOmniSurfaceLite::*",
                "using::OmniSurfaceLiteimportOmniSurfaceLite",
            },
        ),
        "OmniGlass": (
            "::OmniGlass::OmniGlass",
            "OmniGlass",
            "OmniGlass.mdl",
            "OmniGlass",
            {
                "import::OmniGlass::OmniGlass",
                "using::OmniGlassimportOmniGlass",
            },
        ),
        "GlassWithVolume": (
            "..::Templates::GlassWithVolume::GlassWithVolume",
            "NvidiaGlassWithVolume",
            "Base/Templates/GlassWithVolume.mdl",
            "GlassWithVolume",
            {
                "using..::Templates::GlassWithVolumeimportGlassWithVolume",
            },
        ),
    }
    terminal = authored_callee.lstrip(":").rsplit("::", 1)[-1]
    candidate = candidates.get(terminal)
    if candidate is None:
        raise _MdlParseError(
            "unsupported_or_unverified_material_callee",
            f"unsupported composed material callee {authored_callee!r}",
        )
    canonical_callee, family, module, sub_identifier, expected_imports = candidate
    if not statements.intersection(expected_imports):
        raise _MdlParseError(
            "unsupported_or_unverified_material_callee",
            f"selected material does not exactly import {canonical_callee}",
        )
    allowed_spellings = {
        terminal,
        canonical_callee,
        canonical_callee.lstrip(":"),
    }
    if terminal == "GlassWithVolume":
        allowed_spellings.add("GlassWithVolume")
    if authored_callee not in allowed_spellings:
        raise _MdlParseError(
            "unsupported_or_unverified_material_callee",
            f"selected material uses unsupported spelling {authored_callee!r}",
        )
    return canonical_callee, family, module, sub_identifier


def _provenance_mdl_dependency_modules(
    source_resolution_provenance: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(source_resolution_provenance, Mapping):
        return []
    modules = [
        dict(record)
        for record in source_resolution_provenance.get("module_files", [])
        if isinstance(record, Mapping)
    ]
    for record in source_resolution_provenance.get("bundle_files", []):
        if not isinstance(record, Mapping):
            continue
        relative_path = str(record.get("relative_path") or "")
        if Path(relative_path).suffix.casefold() != ".mdl":
            continue
        module_name = "::" + "::".join(Path(relative_path).with_suffix("").parts)
        modules.append(
            {
                "authored_module": module_name,
                "relative_path": relative_path,
                "resolved_source_path": str(record.get("path") or ""),
                "source_sha256": str(record.get("sha256") or ""),
            }
        )
    return modules


def _read_exact_mdl_module(
    path: Path, expected_sha256: str
) -> tuple[str, str]:
    try:
        source_bytes = path.read_bytes()
        source_text = source_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise _MdlParseError(
            "mdl_dependency_read_failed",
            f"could not read exact MDL dependency: {type(exc).__name__}: {exc}",
        ) from None
    observed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise _MdlParseError(
            "mdl_dependency_hash_changed",
            "MDL dependency changed after content-addressed resolution",
        )
    return source_text, observed_sha256


def _parse_composed_mdl_material_source(
    source_text: str,
    source_sha256: str,
    mdl_path: Path,
    sub_identifier: str | None,
    source_resolution_provenance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Parse supported Omni material families after checking their file hashes.

    Return ``None`` for other families so the caller can try the OmniPBR
    wrapper parser.  If a supported family's imports cannot be resolved and
    checked, return an untrusted result.
    """

    known_installed_root = next(
        (
            (authored_name, spec)
            for authored_name, spec in _INSTALLED_OMNI_MDL_MODULE_SPECS.items()
            if source_sha256
            == _INSTALLED_OMNI_MDL_FILE_SHA256[str(spec["root"])]
        ),
        None,
    )
    known_local_root = next(
        (
            (authored_name, spec, classification)
            for root_specs, file_sha256s, classification in (
                (
                    _ASSET_LOCAL_OMNI_MDL_ROOT_SPECS,
                    _ASSET_LOCAL_OMNI_MDL_FILE_SHA256,
                    _ASSET_LOCAL_OMNI_MDL_RESOLUTION_METHOD,
                ),
                (
                    _ASSET_LOCAL_OMNIPBR_MDL_ROOT_SPECS,
                    _ASSET_LOCAL_OMNIPBR_MDL_FILE_SHA256,
                    _ASSET_LOCAL_OMNIPBR_MDL_RESOLUTION_METHOD,
                ),
            )
            for authored_name, spec in root_specs.items()
            if mdl_path.name == authored_name
            and source_sha256
            == file_sha256s[authored_name]
        ),
        None,
    )
    source_looks_supported = (
        known_installed_root is not None
        or known_local_root is not None
        or any(
            marker in source_text
            for marker in (
                "::OmniSurface::OmniSurface",
                "::OmniGlass::OmniGlass",
                "GlassWithVolume",
            )
        )
    )
    if not source_looks_supported:
        return None

    try:
        tokens = _lex_mdl_source(source_text)
        declarations = _mdl_export_material_declarations(tokens, source_text)
        declaration = _mdl_selected_material_declaration(
            declarations, sub_identifier
        )
        if known_local_root is not None and (
            not isinstance(source_resolution_provenance, Mapping)
            or source_resolution_provenance.get("classification")
            != known_local_root[2]
        ):
            raise _MdlParseError(
                "asset_local_mdl_dependency_closure_unavailable",
                "the exact asset-local OmniSurface module closure is unavailable",
            )
        dependency_modules = _provenance_mdl_dependency_modules(
            source_resolution_provenance
        )
        preset_root = next(
            (
                candidate
                for candidate in (known_installed_root, known_local_root)
                if candidate is not None
                and candidate[0] == "OmniSurfacePresets.mdl"
            ),
            None,
        )
        if preset_root is not None:
            selected_identifier = (
                sub_identifier.rsplit("::", 1)[-1]
                if sub_identifier
                else declaration["name"]
            )
            if selected_identifier not in preset_root[1]["sub_identifiers"]:
                raise _MdlParseError(
                    "material_sub_identifier_mismatch",
                    "preset subIdentifier is not audited for these exact bytes",
                )

        implementation_root = None
        implementation_kind = None
        if (
            known_installed_root is not None
            and known_installed_root[0] != "OmniSurfacePresets.mdl"
        ):
            implementation_root = known_installed_root
            implementation_kind = (
                "hash_attested_installed_material_implementation"
            )
        elif (
            known_local_root is not None
            and known_local_root[0] != "OmniSurfacePresets.mdl"
        ):
            implementation_root = known_local_root
            implementation_kind = "hash_attested_local_material_implementation"

        if implementation_root is not None:
            authored_name, spec = implementation_root[:2]
            selected_identifier = (
                sub_identifier.rsplit("::", 1)[-1]
                if sub_identifier
                else declaration["name"]
            )
            if selected_identifier not in spec["sub_identifiers"]:
                raise _MdlParseError(
                    "material_sub_identifier_mismatch",
                    "implementation subIdentifier is not audited for these exact bytes",
                )
            parameter_defaults, textures = _mdl_signature_parameter_defaults(
                tokens, declaration
            )
            material_family = str(spec["family"])
            direct_callee = None
            authored_direct_callee = None
            expression_kind = str(implementation_kind)
        else:
            authored_direct_callee, arguments = _mdl_direct_call(
                tokens, declaration
            )
            (
                direct_callee,
                material_family,
                dependency_module,
                dependency_sub_identifier,
            ) = _mdl_attest_supported_composed_callee(
                tokens, authored_direct_callee
            )

            dependency_stem = Path(dependency_module).stem
            matching_dependencies = {
                (
                    str(record.get("resolved_source_path") or ""),
                    str(record.get("source_sha256") or ""),
                ): record
                for record in dependency_modules
                if record.get("relative_path") == dependency_module
                or record.get("authored_module") == f"::{dependency_stem}"
            }
            if len(matching_dependencies) > 1:
                raise _MdlParseError(
                    "ambiguous_pinned_mdl_dependency",
                    f"multiple attested modules resolve {dependency_module}",
                )
            dependency_record = (
                next(iter(matching_dependencies.values()))
                if matching_dependencies
                else None
            )
            if dependency_record is not None:
                dependency_path = Path(
                    str(dependency_record["resolved_source_path"])
                )
                dependency_sha256 = str(dependency_record["source_sha256"])
            elif dependency_module in _INSTALLED_OMNI_MDL_MODULE_SPECS:
                installed = _resolve_installed_omni_mdl_module(
                    dependency_module, dependency_sub_identifier
                )
                if installed is None:
                    raise _MdlParseError(
                        "installed_mdl_dependency_unavailable",
                        f"no unique exact installation resolves {dependency_module}",
                    )
                dependency_path, dependency_provenance = installed
                dependency_modules.extend(dependency_provenance["module_files"])
                dependency_sha256 = str(
                    dependency_provenance["resolved_source_sha256"]
                )
            else:
                raise _MdlParseError(
                    "pinned_mdl_dependency_unavailable",
                    f"verified bundle does not contain {dependency_module}",
                )

            dependency_text, _ = _read_exact_mdl_module(
                dependency_path, dependency_sha256
            )
            dependency_tokens = _lex_mdl_source(dependency_text)
            dependency_declarations = _mdl_export_material_declarations(
                dependency_tokens, dependency_text
            )
            dependency_declaration = _mdl_selected_material_declaration(
                dependency_declarations, dependency_sub_identifier
            )
            parameter_defaults, _ = _mdl_signature_parameter_defaults(
                dependency_tokens, dependency_declaration
            )
            textures_by_name: dict[str, dict[str, Any]] = {}
            assigned_inputs: set[str] = set()
            for argument in arguments:
                if (
                    len(argument) < 3
                    or argument[0].kind != "identifier"
                    or argument[1].value != ":"
                    or (
                        argument[2].value == ":"
                        and (
                            len(argument) < 4
                            or argument[3].value != ":"
                        )
                    )
                ):
                    raise _MdlParseError(
                        "unnamed_or_dynamic_material_argument",
                        "every composed material argument must be a direct named value",
                    )
                input_name = argument[0].value
                if input_name in assigned_inputs:
                    raise _MdlParseError(
                        "duplicate_material_argument",
                        f"material input {input_name!r} is assigned more than once",
                    )
                assigned_inputs.add(input_name)
                current = parameter_defaults.get(input_name)
                if current is None:
                    raise _MdlParseError(
                        "unknown_composed_material_argument",
                        f"material input {input_name!r} is absent from the exact imported signature",
                    )
                expected_type = str(current["value_type"])
                if expected_type == "qualified_enum":
                    expected_type = "known_qualified_enum"
                value_record, texture = _mdl_static_argument_value(
                    argument[2:], input_name, expected_type
                )
                parameter_defaults[input_name] = value_record
                if texture is not None:
                    textures_by_name[input_name] = texture
            textures = list(textures_by_name.values())
            expression_kind = "hash_attested_direct_call_with_composed_defaults"
            if preset_root is not None:
                material_family = "OmniSurfacePreset"

        unique_modules = {
            (
                str(module.get("authored_module") or ""),
                str(module.get("relative_path") or ""),
                str(module.get("source_sha256") or ""),
            ): dict(module)
            for module in dependency_modules
        }
        if not unique_modules:
            raise _MdlParseError(
                "empty_mdl_dependency_attestation",
                "supported material has no content-addressed import closure",
            )
        dependency_attestation = _source_dependency_attestation(
            "content_addressed_mdl_import_closure",
            unique_modules.values(),
        )
        canonical_semantics = {
            "material_family": material_family,
            "definition_expression_kind": expression_kind,
            "selected_material_definition": declaration["name"],
            "direct_callee": direct_callee,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": parameter_defaults,
            "source_dependency_attestation_sha256": dependency_attestation[
                "attestation_sha256"
            ],
        }
        return {
            "trusted": True,
            "status": "trusted_hash_attested_composed_material",
            "error_code": None,
            "error": None,
            "sub_identifier": sub_identifier,
            "selected_material_definition": declaration["name"],
            "selected_definition_source_line": declaration["source_line"],
            "selected_definition_source_start": declaration["source_start"],
            "selected_definition_source_end": declaration["source_end"],
            "material_family": material_family,
            "definition_expression_kind": expression_kind,
            "authored_direct_callee": authored_direct_callee,
            "direct_callee": direct_callee,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": parameter_defaults,
            "semantics_sha256": _json_canonical_hash(
                "selected-mdl-material-semantics", canonical_semantics
            ),
            "source_dependency_attestation": dependency_attestation,
            "exported_material_names": [
                item["name"] for item in declarations
            ],
            "textures": textures,
        }
    except _MdlParseError as exc:
        return {
            "trusted": False,
            "status": "untrusted_mdl_source",
            "error_code": exc.code,
            "error": str(exc),
            "sub_identifier": sub_identifier,
            "selected_material_definition": None,
            "material_family": None,
            "definition_expression_kind": None,
            "authored_direct_callee": None,
            "direct_callee": None,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": {},
            "semantics_sha256": None,
            "source_dependency_attestation": None,
            "textures": [],
        }


def _parse_selected_mdl_material_source(
    source_text: str, sub_identifier: str | None
) -> dict[str, Any]:
    """Check a static MDL wrapper around OmniPBR."""

    try:
        tokens = _lex_mdl_source(source_text)
        declarations = _mdl_export_material_declarations(tokens, source_text)
        if sub_identifier is not None:
            qualified_identifier_pattern = re.compile(
                rf"(?:\.?::)?{_MDL_IDENTIFIER_SOURCE}"
                rf"(?:::{_MDL_IDENTIFIER_SOURCE})*\Z"
            )
            if not qualified_identifier_pattern.fullmatch(sub_identifier):
                raise _MdlParseError(
                    "invalid_sub_identifier",
                    "MDL sourceAsset subIdentifier is not a material identifier",
                )
            selected_identifier = sub_identifier.rsplit("::", 1)[-1]
            selected = [
                declaration
                for declaration in declarations
                if declaration["name"] == selected_identifier
            ]
            if not selected:
                raise _MdlParseError(
                    "selected_material_not_found",
                    f"no export material matches subIdentifier {sub_identifier!r}",
                )
            if len(selected) != 1:
                raise _MdlParseError(
                    "multiple_selected_material_definitions",
                    f"multiple export material declarations match {sub_identifier!r}",
                )
        else:
            if not declarations:
                raise _MdlParseError(
                    "no_exported_material",
                    "MDL source contains no export material declaration",
                )
            if len(declarations) != 1:
                raise _MdlParseError(
                    "ambiguous_material_without_sub_identifier",
                    "MDL source has multiple exported materials but no subIdentifier",
                )
            selected = declarations

        declaration = selected[0]
        expression_start = declaration["equals_token_index"] + 1
        expression_end = declaration["end_token_index"]
        expression_tokens = tokens[expression_start:expression_end]
        if len(expression_tokens) < 3:
            raise _MdlParseError(
                "malformed_material_definition",
                "selected material definition is not a direct call",
            )
        call_open_index = next(
            (
                index
                for index, token in enumerate(expression_tokens)
                if token.value == "("
            ),
            None,
        )
        if call_open_index is None:
            raise _MdlParseError(
                "malformed_material_definition",
                "selected material definition is not a direct call",
            )
        authored_callee = "".join(
            token.value for token in expression_tokens[:call_open_index]
        )
        if not (
            authored_callee == "OmniPBR"
            or _MDL_QUALIFIED_NAME_PATTERN.fullmatch(authored_callee)
        ):
            raise _MdlParseError(
                "malformed_material_callee",
                "selected material call has a dynamic or malformed callee",
            )
        call_close_index = _mdl_matching_close(
            expression_tokens,
            call_open_index,
            len(expression_tokens),
            error_code="malformed_material_definition",
            description="selected material call",
        )
        if call_close_index != len(expression_tokens) - 1:
            raise _MdlParseError(
                "nonsole_material_definition_expression",
                "selected material definition must contain only one direct call",
            )
        canonical_callee = _mdl_attest_omnipbr_callee(
            tokens, authored_callee
        )
        arguments = _mdl_split_arguments(
            expression_tokens,
            call_open_index + 1,
            call_close_index,
            error_code="malformed_material_arguments",
            description="selected material call",
        )
        parameter_defaults: dict[str, dict[str, Any]] = {}
        textures: list[dict[str, Any]] = []
        for argument in arguments:
            if (
                len(argument) < 3
                or argument[0].kind != "identifier"
                or argument[1].value != ":"
                or (
                    argument[2].value == ":"
                    and (
                        len(argument) < 4
                        or argument[3].value != ":"
                    )
                )
            ):
                raise _MdlParseError(
                    "unnamed_or_dynamic_material_argument",
                    "every OmniPBR argument must be a direct named value",
                )
            input_name = argument[0].value
            if input_name in parameter_defaults:
                raise _MdlParseError(
                    "duplicate_material_argument",
                    f"material input {input_name!r} is assigned more than once",
                )
            expected_type = _MDL_OMNIPBR_ARGUMENT_TYPES.get(input_name)
            if expected_type is None:
                raise _MdlParseError(
                    "unknown_omnipbr_argument",
                    f"material input {input_name!r} is not in the "
                    "supported OmniPBR signature",
                )
            value_record, texture = _mdl_static_argument_value(
                argument[2:], input_name, expected_type
            )
            parameter_defaults[input_name] = value_record
            if texture is not None:
                textures.append(texture)

        canonical_semantics = {
            "material_family": "OmniPBR",
            "definition_expression_kind": "sole_direct_call",
            "selected_material_definition": declaration["name"],
            "direct_callee": canonical_callee,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": parameter_defaults,
        }

        return {
            "trusted": True,
            "status": "trusted_selected_export_material",
            "error_code": None,
            "error": None,
            "sub_identifier": sub_identifier,
            "selected_material_definition": declaration["name"],
            "selected_definition_source_line": declaration["source_line"],
            "selected_definition_source_start": declaration["source_start"],
            "selected_definition_source_end": declaration["source_end"],
            "material_family": "OmniPBR",
            "definition_expression_kind": "sole_direct_call",
            "authored_direct_callee": authored_callee,
            "direct_callee": canonical_callee,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": parameter_defaults,
            "semantics_sha256": _json_canonical_hash(
                "selected-mdl-material-semantics", canonical_semantics
            ),
            "source_dependency_attestation": None,
            "exported_material_names": [
                item["name"] for item in declarations
            ],
            "textures": textures,
        }
    except _MdlParseError as exc:
        return {
            "trusted": False,
            "status": "untrusted_mdl_source",
            "error_code": exc.code,
            "error": str(exc),
            "sub_identifier": sub_identifier,
            "selected_material_definition": None,
            "material_family": None,
            "definition_expression_kind": None,
            "authored_direct_callee": None,
            "direct_callee": None,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": {},
            "semantics_sha256": None,
            "source_dependency_attestation": None,
            "textures": [],
        }


def _known_simpbr_parameter_defaults() -> dict[str, dict[str, Any]]:
    defaults: dict[str, dict[str, Any]] = {}
    for input_name, value_type, value in _SIM_PBR_DEFAULT_SPECS:
        if value_type == "texture_2d":
            serialized_value: Any = {
                "authored_asset": None,
                "gamma_expression": None,
                "gamma_mode": "gamma_default",
                "gamma_is_implicit": True,
                "color_space": "default",
            }
        elif value_type == "state_normal":
            serialized_value = "::state::normal()"
        elif isinstance(value, tuple):
            serialized_value = list(value)
        else:
            serialized_value = value
        defaults[input_name] = {
            "value_type": value_type,
            "value": serialized_value,
        }
    return defaults


def _attest_known_simpbr_translucent_material_source(
    source_text: str,
    source_sha256: str,
    mdl_path: Path,
    sub_identifier: str | None,
) -> dict[str, Any]:
    """Check the SimPBR translucent material implementation."""

    try:
        if sub_identifier != "SimPBR_Translucent":
            raise _MdlParseError(
                "simpbr_translucent_sub_identifier_mismatch",
                "hash-attested SimPBR translucent material requires "
                "subIdentifier 'SimPBR_Translucent'",
            )
        if mdl_path.name != "SimPBR_Translucent.mdl":
            raise _MdlParseError(
                "simpbr_translucent_source_module_mismatch",
                "hash-attested SimPBR translucent material must resolve "
                "SimPBR_Translucent.mdl",
            )
        if source_sha256 != _SIM_PBR_TRANSLUCENT_SOURCE_SHA256:
            raise _MdlParseError(
                "unsupported_simpbr_translucent_source_hash",
                "resolved SimPBR_Translucent.mdl does not match the audited "
                "source hash",
            )

        tokens = _lex_mdl_source(source_text)
        declarations = _mdl_export_material_declarations(tokens, source_text)
        selected = [
            declaration
            for declaration in declarations
            if declaration["name"] == "SimPBR_Translucent"
        ]
        if len(selected) != 1:
            raise _MdlParseError(
                "simpbr_translucent_selected_material_mismatch",
                "audited source must contain one SimPBR_Translucent export",
            )
        statements = set(_mdl_top_level_statements(tokens))
        required_imports = {
            (
                "import",
                ":",
                ":",
                "nvidia",
                ":",
                ":",
                "core_definitions",
                ":",
                ":",
                symbol,
            )
            for symbol in ("file_texture", "normalmap_texture")
        }
        if not required_imports.issubset(statements):
            raise _MdlParseError(
                "simpbr_translucent_import_mismatch",
                "audited SimPBR translucent source does not import its exact "
                "core-definition symbols",
            )
        dependency_modules = _resolve_installed_omni_mdl_dependency_files(
            (
                "mdl/nvidia/aux_definitions.mdl",
                "mdl/nvidia/core_definitions.mdl",
                "mdl/nvidia/support_definitions.mdl",
            )
        )
        if dependency_modules is None:
            raise _MdlParseError(
                "simpbr_translucent_dependency_closure_unavailable",
                "no unique installed module set matches the exact audited "
                "SimPBR translucent dependency closure",
            )
        dependency_attestation = _source_dependency_attestation(
            "content_addressed_mdl_import_closure", dependency_modules
        )

        declaration = selected[0]
        parameter_defaults, textures = _mdl_signature_parameter_defaults(
            tokens, declaration
        )
        canonical_semantics = {
            "material_family": "SimPBRTranslucent",
            "definition_expression_kind": (
                "hash_attested_simpbr_translucent_implementation"
            ),
            "selected_material_definition": "SimPBR_Translucent",
            "direct_callee": None,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": parameter_defaults,
            "source_dependency_attestation_sha256": dependency_attestation[
                "attestation_sha256"
            ],
        }
        return {
            "trusted": True,
            "status": "trusted_hash_attested_simpbr_translucent_material",
            "error_code": None,
            "error": None,
            "sub_identifier": sub_identifier,
            "selected_material_definition": "SimPBR_Translucent",
            "selected_definition_source_line": declaration["source_line"],
            "selected_definition_source_start": declaration["source_start"],
            "selected_definition_source_end": declaration["source_end"],
            "material_family": "SimPBRTranslucent",
            "definition_expression_kind": (
                "hash_attested_simpbr_translucent_implementation"
            ),
            "authored_direct_callee": None,
            "direct_callee": None,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": parameter_defaults,
            "semantics_sha256": _json_canonical_hash(
                "selected-mdl-material-semantics", canonical_semantics
            ),
            "source_dependency_attestation": dependency_attestation,
            "exported_material_names": [
                item["name"] for item in declarations
            ],
            "textures": textures,
        }
    except _MdlParseError as exc:
        return {
            "trusted": False,
            "status": "untrusted_mdl_source",
            "error_code": exc.code,
            "error": str(exc),
            "sub_identifier": sub_identifier,
            "selected_material_definition": None,
            "material_family": None,
            "definition_expression_kind": None,
            "authored_direct_callee": None,
            "direct_callee": None,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": {},
            "semantics_sha256": None,
            "source_dependency_attestation": None,
            "textures": [],
        }


def _attest_known_simpbr_material_source(
    source_text: str,
    source_sha256: str,
    mdl_path: Path,
    sub_identifier: str | None,
) -> dict[str, Any]:
    """Check the SimPBR and SimPBR_Model module contents."""

    try:
        if sub_identifier != "SimPBR":
            raise _MdlParseError(
                "simpbr_sub_identifier_mismatch",
                "hash-attested SimPBR requires subIdentifier 'SimPBR'",
            )
        if mdl_path.name != "SimPBR.mdl":
            raise _MdlParseError(
                "simpbr_source_module_mismatch",
                "hash-attested SimPBR must resolve the exact SimPBR.mdl module",
            )
        if source_sha256 != _SIM_PBR_SOURCE_SHA256:
            raise _MdlParseError(
                "unsupported_simpbr_source_hash",
                "resolved SimPBR.mdl does not match the audited source hash",
            )

        tokens = _lex_mdl_source(source_text)
        declarations = _mdl_export_material_declarations(tokens, source_text)
        selected = [
            declaration
            for declaration in declarations
            if declaration["name"] == "SimPBR"
        ]
        if len(selected) != 1:
            raise _MdlParseError(
                "simpbr_selected_material_mismatch",
                "audited SimPBR source must contain one selected SimPBR export",
            )
        expected_import = (
            "using",
            ".",
            ":",
            ":",
            "SimPBR_Model",
            "import",
            "*",
        )
        if expected_import not in set(_mdl_top_level_statements(tokens)):
            raise _MdlParseError(
                "simpbr_terminal_module_import_mismatch",
                "audited SimPBR source does not import its exact local terminal module",
            )

        terminal_path = mdl_path.with_name("SimPBR_Model.mdl")
        try:
            terminal_bytes = terminal_path.read_bytes()
        except OSError as exc:
            raise _MdlParseError(
                "simpbr_terminal_module_read_failed",
                "could not read the exact adjacent SimPBR_Model.mdl dependency: "
                f"{type(exc).__name__}: {exc}",
            ) from None
        terminal_sha256 = hashlib.sha256(terminal_bytes).hexdigest()
        if terminal_sha256 != _SIM_PBR_MODEL_SOURCE_SHA256:
            raise _MdlParseError(
                "unsupported_simpbr_terminal_module_hash",
                "resolved SimPBR_Model.mdl does not match the audited source hash",
            )

        declaration = selected[0]
        parameter_defaults = _known_simpbr_parameter_defaults()
        terminal_module = {
            "authored_module": ".::SimPBR_Model",
            "relative_path": "SimPBR_Model.mdl",
            "resolved_source_path": str(terminal_path),
            "source_sha256": terminal_sha256,
        }
        canonical_semantics = {
            "material_family": "SimPBR",
            "definition_expression_kind": (
                "hash_attested_let_terminal_call"
            ),
            "selected_material_definition": "SimPBR",
            "direct_callee": None,
            "terminal_material_callee": (
                ".::SimPBR_Model::SimPBR_Model"
            ),
            "terminal_material_module": {
                "authored_module": terminal_module["authored_module"],
                "source_sha256": terminal_module["source_sha256"],
            },
            "parameter_defaults": parameter_defaults,
        }
        return {
            "trusted": True,
            "status": "trusted_hash_attested_simpbr_material",
            "error_code": None,
            "error": None,
            "sub_identifier": sub_identifier,
            "selected_material_definition": "SimPBR",
            "selected_definition_source_line": declaration["source_line"],
            "selected_definition_source_start": declaration["source_start"],
            "selected_definition_source_end": declaration["source_end"],
            "material_family": "SimPBR",
            "definition_expression_kind": (
                "hash_attested_let_terminal_call"
            ),
            "authored_direct_callee": None,
            "direct_callee": None,
            "terminal_material_callee": (
                ".::SimPBR_Model::SimPBR_Model"
            ),
            "terminal_material_module": terminal_module,
            "parameter_defaults": parameter_defaults,
            "semantics_sha256": _json_canonical_hash(
                "selected-mdl-material-semantics", canonical_semantics
            ),
            "source_dependency_attestation": _source_dependency_attestation(
                "content_addressed_mdl_import_closure", [terminal_module]
            ),
            "exported_material_names": [
                item["name"] for item in declarations
            ],
            "textures": [],
        }
    except _MdlParseError as exc:
        return {
            "trusted": False,
            "status": "untrusted_mdl_source",
            "error_code": exc.code,
            "error": str(exc),
            "sub_identifier": sub_identifier,
            "selected_material_definition": None,
            "material_family": None,
            "definition_expression_kind": None,
            "authored_direct_callee": None,
            "direct_callee": None,
            "terminal_material_callee": None,
            "terminal_material_module": None,
            "parameter_defaults": {},
            "semantics_sha256": None,
            "source_dependency_attestation": None,
            "textures": [],
        }


def _resolve_mdl_texture_path(
    authored_asset: str, mdl_path: Path
) -> tuple[tuple[str, ...], str | None]:
    if not authored_asset or "://" in authored_asset:
        return (), None
    authored_path = Path(authored_asset)
    candidate = (
        authored_path
        if authored_path.is_absolute()
        else mdl_path.parent / authored_path
    )
    exact_method = (
        "mdl_authored_absolute_path"
        if authored_path.is_absolute()
        else "mdl_source_relative"
    )
    expanded = _expand_udim(candidate)
    if expanded:
        return tuple(str(path) for path in expanded), exact_method
    repaired_spelling = _unique_same_directory_casefold_file(candidate)
    if repaired_spelling:
        return (
            (str(repaired_spelling),),
            CASE_REPAIR_METHOD,
        )
    recovered = _casefold_existing_path(candidate)
    if recovered:
        return (str(recovered),), f"{exact_method}_casefold_recovery"
    return (), None


def _mdl_source_diagnostic(
    shader_prim,
    context: str,
    *,
    reason_code: str,
    reason: str,
    authored_source_asset: str | None,
    resolved_source_paths: tuple[str, ...] = (),
    source_resolution_method: str | None = None,
    sub_identifier: str | None = None,
) -> dict[str, Any]:
    return {
        "trust_status": "untrusted_mdl_source",
        "selection_eligible": False,
        "reason_code": reason_code,
        "reason": reason,
        "candidate_role": None,
        "authored_asset": authored_source_asset,
        "resolved_paths": list(resolved_source_paths),
        "authoring_layer": (
            resolved_source_paths[0] if len(resolved_source_paths) == 1 else None
        ),
        "shader_path": str(shader_prim.GetPath()),
        "input_name": None,
        "render_context": context,
        "resolution_method": source_resolution_method,
        "used_path_fallback": is_path_fallback(
            source_resolution_method
        ),
        "source_asset_sub_identifier": sub_identifier,
    }


def _read_active_mdl_source_material(
    shader_prim, context: str, root_layer
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Read and check an active local MDL material wrapper."""

    source_attribute = shader_prim.GetAttribute("info:mdl:sourceAsset")
    if not source_attribute:
        return None, []
    try:
        implementation_source = str(
            UsdShade.Shader(shader_prim).GetImplementationSource() or ""
        )
    except Exception as exc:
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="implementation_source_read_failed",
                reason=(
                    "could not read active MDL sourceAsset implementation: "
                    f"{type(exc).__name__}: {exc}"
                ),
                authored_source_asset=None,
            )
        ]
    if implementation_source != "sourceAsset":
        return None, []
    try:
        source_value = source_attribute.Get(Usd.TimeCode.Default())
    except Exception as exc:
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="source_asset_read_failed",
                reason=f"could not read MDL sourceAsset: {type(exc).__name__}: {exc}",
                authored_source_asset=None,
            )
        ]
    if not isinstance(source_value, Sdf.AssetPath) or not source_value.path:
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="invalid_source_asset",
                reason="active info:mdl:sourceAsset is not a non-empty asset path",
                authored_source_asset=None,
            )
        ]

    authored_source_asset = source_value.path
    sub_identifier_attribute = shader_prim.GetAttribute(
        "info:mdl:sourceAsset:subIdentifier"
    )
    try:
        sub_identifier_value = (
            sub_identifier_attribute.Get(Usd.TimeCode.Default())
            if sub_identifier_attribute
            else None
        )
        sub_identifier_text = (
            str(sub_identifier_value).strip()
            if sub_identifier_value is not None
            else ""
        )
        sub_identifier = sub_identifier_text or None
    except Exception as exc:
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="sub_identifier_read_failed",
                reason=f"could not read MDL subIdentifier: {type(exc).__name__}: {exc}",
                authored_source_asset=authored_source_asset,
            )
        ]

    source_layer = _authoring_layer(source_attribute)
    source_paths, source_method = _resolve_asset_path(
        source_value, source_layer, root_layer
    )
    source_resolution_provenance = None
    if (
        not source_paths
        and authored_source_asset in _INSTALLED_OMNI_MDL_MODULE_SPECS
    ):
        installed_source = _resolve_installed_omni_mdl_module(
            authored_source_asset, sub_identifier
        )
        if installed_source is None:
            return None, [
                _mdl_source_diagnostic(
                    shader_prim,
                    context,
                    reason_code="installed_mdl_source_unavailable",
                    reason=(
                        "no unique installed module matches the exact audited "
                        "MDL dependency closure and selected subIdentifier"
                    ),
                    authored_source_asset=authored_source_asset,
                    source_resolution_method=(
                        _INSTALLED_OMNI_MDL_RESOLUTION_METHOD
                    ),
                    sub_identifier=sub_identifier,
                )
            ]
        source_paths = (str(installed_source[0]),)
        source_method = _INSTALLED_OMNI_MDL_RESOLUTION_METHOD
        source_resolution_provenance = installed_source[1]
    if external_mdl_repair_is_applicable(
        authored_source_asset, _layer_identifier(root_layer)
    ):
        external_source = resolve_external_mdl_source(
            authored_source_asset,
            _layer_identifier(root_layer),
        )
        if (
            external_source is None
            or len(source_paths) != 1
            or str(external_source[0]) != source_paths[0]
        ):
            source_paths = ()
            source_method = None
        else:
            source_resolution_provenance = external_source[1]
    if len(source_paths) != 1:
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code=(
                    "unresolved_source_asset"
                    if not source_paths
                    else "ambiguous_source_asset"
                ),
                reason=(
                    "active MDL sourceAsset could not be resolved"
                    if not source_paths
                    else "active MDL sourceAsset resolved to multiple files"
                ),
                authored_source_asset=authored_source_asset,
                resolved_source_paths=source_paths,
                source_resolution_method=source_method,
            )
        ]
    if is_path_fallback(source_method):
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="source_path_fallback",
                reason="active MDL sourceAsset required heuristic path recovery",
                authored_source_asset=authored_source_asset,
                resolved_source_paths=source_paths,
                source_resolution_method=source_method,
            )
        ]

    mdl_path = Path(source_paths[0])
    if mdl_path.suffix.casefold() != ".mdl" or not mdl_path.is_file():
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="invalid_mdl_source_file",
                reason="active MDL sourceAsset is not a readable .mdl file",
                authored_source_asset=authored_source_asset,
                resolved_source_paths=source_paths,
                source_resolution_method=source_method,
            )
        ]
    try:
        source_bytes = mdl_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="mdl_source_read_failed",
                reason=f"could not read MDL source: {type(exc).__name__}: {exc}",
                authored_source_asset=authored_source_asset,
                resolved_source_paths=source_paths,
                source_resolution_method=source_method,
            )
        ]
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    is_known_asset_local_omni_root = (
        mdl_path.name in _ASSET_LOCAL_OMNI_MDL_ROOT_SPECS
        and source_sha256
        == _ASSET_LOCAL_OMNI_MDL_FILE_SHA256.get(mdl_path.name)
    )
    is_known_asset_local_omnipbr_root = (
        mdl_path.name in _ASSET_LOCAL_OMNIPBR_MDL_ROOT_SPECS
        and source_sha256
        == _ASSET_LOCAL_OMNIPBR_MDL_FILE_SHA256.get(mdl_path.name)
    )
    if is_known_asset_local_omni_root:
        local_provenance = _resolve_asset_local_omni_mdl_module(
            authored_source_asset,
            mdl_path,
            source_sha256,
            source_method,
        )
        if local_provenance is None:
            return None, [
                _mdl_source_diagnostic(
                    shader_prim,
                    context,
                    reason_code=(
                        "asset_local_mdl_dependency_closure_unavailable"
                    ),
                    reason=(
                        "the selected legacy OmniSurface source did not have "
                        "its exact adjacent and installed import closure"
                    ),
                    authored_source_asset=authored_source_asset,
                    resolved_source_paths=source_paths,
                    source_resolution_method=source_method,
                    sub_identifier=sub_identifier,
                )
            ]
        source_method = _ASSET_LOCAL_OMNI_MDL_RESOLUTION_METHOD
        source_resolution_provenance = local_provenance
    elif is_known_asset_local_omnipbr_root:
        local_provenance = _resolve_asset_local_omnipbr_mdl_module(
            authored_source_asset,
            mdl_path,
            source_sha256,
            source_method,
        )
        if local_provenance is None:
            return None, [
                _mdl_source_diagnostic(
                    shader_prim,
                    context,
                    reason_code=(
                        "asset_local_mdl_dependency_closure_unavailable"
                    ),
                    reason=(
                        "the selected asset-local OmniPBR source did not have "
                        "its exact installed import closure"
                    ),
                    authored_source_asset=authored_source_asset,
                    resolved_source_paths=source_paths,
                    source_resolution_method=source_method,
                    sub_identifier=sub_identifier,
                )
            ]
        source_method = _ASSET_LOCAL_OMNIPBR_MDL_RESOLUTION_METHOD
        source_resolution_provenance = local_provenance
    elif source_resolution_provenance is None and (
        authored_source_asset in _INSTALLED_OMNI_MDL_MODULE_SPECS
    ):
        installed_source = _resolve_installed_omni_mdl_module(
            authored_source_asset, sub_identifier
        )
        if (
            installed_source is not None
            and installed_source[0].resolve() == mdl_path.resolve()
            and installed_source[1].get("resolved_source_sha256")
            == source_sha256
        ):
            source_method = _INSTALLED_OMNI_MDL_RESOLUTION_METHOD
            source_resolution_provenance = installed_source[1]
    if source_resolution_provenance is not None and (
        source_sha256
        != source_resolution_provenance.get("resolved_source_sha256")
    ):
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code="pinned_mdl_source_changed_after_resolution",
                reason=(
                    "pinned MDL module content changed between bundle "
                    "verification and selected-definition parsing"
                ),
                authored_source_asset=authored_source_asset,
                resolved_source_paths=source_paths,
                source_resolution_method=source_method,
            )
        ]

    is_simpbr_translucent_candidate = (
        mdl_path.name == "SimPBR_Translucent.mdl"
        or sub_identifier == "SimPBR_Translucent"
        or source_sha256 == _SIM_PBR_TRANSLUCENT_SOURCE_SHA256
    )
    is_simpbr_candidate = (
        mdl_path.name == "SimPBR.mdl"
        or sub_identifier == "SimPBR"
        or source_sha256 == _SIM_PBR_SOURCE_SHA256
    )
    if is_simpbr_translucent_candidate:
        parsed = _attest_known_simpbr_translucent_material_source(
            source_text,
            source_sha256,
            mdl_path,
            sub_identifier,
        )
    elif is_simpbr_candidate:
        parsed = _attest_known_simpbr_material_source(
            source_text,
            source_sha256,
            mdl_path,
            sub_identifier,
        )
    else:
        parsed = _parse_composed_mdl_material_source(
            source_text,
            source_sha256,
            mdl_path,
            sub_identifier,
            source_resolution_provenance,
        )
        if parsed is None:
            parsed = _parse_selected_mdl_material_source(
                source_text, sub_identifier
            )
    if not parsed["trusted"]:
        return None, [
            _mdl_source_diagnostic(
                shader_prim,
                context,
                reason_code=str(parsed["error_code"]),
                reason=str(parsed["error"]),
                authored_source_asset=authored_source_asset,
                resolved_source_paths=source_paths,
                source_resolution_method=source_method,
                sub_identifier=sub_identifier,
            )
        ]

    return {
        "shader_prim": shader_prim,
        "shader_path": str(shader_prim.GetPath()),
        "render_context": context,
        "authored_source_asset": authored_source_asset,
        "source_resolution_method": source_method,
        "source_resolution_provenance": source_resolution_provenance,
        "mdl_path": mdl_path,
        "source_sha256": source_sha256,
        "sub_identifier": sub_identifier,
        "parsed": parsed,
    }, []


def _extract_active_mdl_source_textures(
    shader_prim, context: str, root_layer
) -> tuple[list[TextureEvidence], list[dict[str, Any]]]:
    """Return trusted selected-definition textures or one blocking diagnostic."""

    source_material, diagnostics = _read_active_mdl_source_material(
        shader_prim, context, root_layer
    )
    if source_material is None:
        return [], diagnostics

    evidence: list[TextureEvidence] = []
    parsed = source_material["parsed"]
    mdl_path = source_material["mdl_path"]
    authored_source_asset = source_material["authored_source_asset"]
    source_method = source_material["source_resolution_method"]
    source_resolution_provenance = source_material[
        "source_resolution_provenance"
    ]
    sub_identifier = source_material["sub_identifier"]
    shader_path = source_material["shader_path"]
    selected_definition = str(parsed["selected_material_definition"])
    material_family = str(parsed["material_family"])
    for texture in parsed["textures"]:
        input_name = str(texture["input_name"])
        authored_asset = str(texture["authored_asset"])
        resolved_paths, resolution_method = _resolve_mdl_texture_path(
            authored_asset, mdl_path
        )
        if (
            isinstance(source_resolution_provenance, Mapping)
            and "bundle_files" in source_resolution_provenance
        ):
            pinned_textures = {
                str(record.get("path")): str(record.get("sha256"))
                for record in source_resolution_provenance.get(
                    "bundle_files", []
                )
                if isinstance(record, Mapping)
                and Path(str(record.get("relative_path", ""))).suffix.casefold()
                in IMAGE_SUFFIXES
            }
            pinned_path = (
                str(Path(resolved_paths[0]).resolve())
                if len(resolved_paths) == 1
                else None
            )
            expected_texture_sha256 = pinned_textures.get(pinned_path)
            try:
                pinned_texture_matches = bool(
                    pinned_path is not None
                    and expected_texture_sha256 is not None
                    and Path(pinned_path).is_file()
                    and _file_sha256(Path(pinned_path))
                    == expected_texture_sha256
                )
            except OSError:
                pinned_texture_matches = False
            if not pinned_texture_matches:
                return [], [
                    _mdl_source_diagnostic(
                        shader_prim,
                        context,
                        reason_code="pinned_mdl_texture_resolution_mismatch",
                        reason=(
                            "a selected MDL texture literal did not resolve "
                            "to its exact hash-attested bundle file"
                        ),
                        authored_source_asset=authored_source_asset,
                        resolved_source_paths=(str(mdl_path),),
                        source_resolution_method=source_method,
                        sub_identifier=sub_identifier,
                    )
                ]
        evidence.append(
            TextureEvidence(
                role=_classify_texture(
                    input_name,
                    authored_asset,
                    material_family=material_family,
                ),
                authored_asset=authored_asset,
                resolved_paths=resolved_paths,
                authoring_layer=str(mdl_path),
                shader_path=shader_path,
                input_name=input_name,
                render_contexts=(context,),
                resolution_method=resolution_method,
                color_space=str(texture["color_space"]),
                evidence_source="mdl_selected_source_material",
                mdl_source_asset_path=str(mdl_path),
                mdl_source_authored_asset=authored_source_asset,
                mdl_source_resolution_method=source_method,
                mdl_source_resolution_provenance=source_resolution_provenance,
                mdl_source_asset_sub_identifier=sub_identifier,
                mdl_selected_material_definition=selected_definition,
                mdl_material_family=material_family,
                mdl_gamma_expression=(
                    str(texture["gamma_expression"])
                    if texture["gamma_expression"] is not None
                    else None
                ),
                mdl_gamma_mode=str(texture["gamma_mode"]),
                mdl_gamma_is_implicit=bool(texture["gamma_is_implicit"]),
                color_space_provenance=(
                    "mdl_texture_2d_implicit_default_gamma"
                    if texture["gamma_is_implicit"]
                    else "mdl_texture_2d_gamma_argument"
                ),
            )
        )
    return evidence, []


def _suppress_overridden_mdl_texture_defaults(
    evidence: Iterable[TextureEvidence],
) -> list[TextureEvidence]:
    """Drop MDL source defaults replaced by authored USD shader inputs.

    A texture constructor in an MDL export defines a parameter default.  If
    the active USD shader authors that same input in the same render context,
    the USD value wins; retaining both invents ambiguity and can produce
    contradictory color-space evidence for one effective texture.
    """

    items = list(evidence)
    authored_usd_inputs = {
        (item.shader_path, item.input_name, context)
        for item in items
        if item.evidence_source == "usd_shader_input"
        for context in item.render_contexts
    }
    authored_usd_consumer_inputs = {
        (item.semantic_consumer_property_path, context)
        for item in items
        if item.evidence_source == "usd_shader_input"
        and item.semantic_consumer_property_path
        for context in item.render_contexts
    }

    def mdl_default_is_overridden(item: TextureEvidence) -> bool:
        mdl_input_path = f"{item.shader_path}.inputs:{item.input_name}"
        return any(
            (item.shader_path, item.input_name, context)
            in authored_usd_inputs
            or (mdl_input_path, context) in authored_usd_consumer_inputs
            for context in item.render_contexts
        )

    return [
        item
        for item in items
        if item.evidence_source != "mdl_selected_source_material"
        or not mdl_default_is_overridden(item)
    ]


def _collect_untrusted_mdl_source_candidates(
    shader_prim, context: str, root_layer
) -> list[dict[str, Any]]:
    """Return diagnostics when the selected MDL definition cannot be checked."""

    _, diagnostics = _extract_active_mdl_source_textures(
        shader_prim, context, root_layer
    )
    return diagnostics


def _collect_material_untrusted_mdl_candidates(
    material, root_layer=None
) -> list[dict[str, Any]]:
    if not material:
        return []
    output_roots: list[tuple[Any, str]] = []
    try:
        outputs = UsdShade.ConnectableAPI(material.GetPrim()).GetOutputs()
    except Exception:
        outputs = []
    for output in outputs:
        context, terminal_name = _terminal_context_and_name(output)
        if terminal_name.casefold() != "surface":
            continue
        endpoints, _ = _property_connection_endpoints(output)
        if endpoints:
            output_roots.append((output, context))
    trace = _trace_active_network(output_roots)
    candidates: list[dict[str, Any]] = []
    for shader_use in trace["active_shader_uses"]:
        candidates.extend(
            _collect_untrusted_mdl_source_candidates(
                shader_use["prim"],
                shader_use["render_context"],
                root_layer,
            )
        )
    unique = {
        (
            str(item.get("shader_path") or ""),
            str(item.get("render_context") or ""),
            str(item.get("reason_code") or ""),
            str(item.get("input_name") or ""),
            str(item.get("authored_asset") or ""),
        ): item
        for item in candidates
    }
    return [unique[key] for key in sorted(unique)]


def _collect_selected_mdl_material_semantics(
    material, root_layer=None
) -> list[dict[str, Any]]:
    """Return one record per active local MDL surface module."""

    if not material:
        return []
    output_roots: list[tuple[Any, str]] = []
    try:
        outputs = UsdShade.ConnectableAPI(material.GetPrim()).GetOutputs()
    except Exception:
        outputs = []
    for output in outputs:
        context, terminal_name = _terminal_context_and_name(output)
        if terminal_name.casefold() != "surface":
            continue
        endpoints, _ = _property_connection_endpoints(output)
        if endpoints:
            output_roots.append((output, context))

    trace = _trace_active_network(output_roots)
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    contexts: dict[tuple[str, ...], set[str]] = {}
    for shader_use in trace["active_shader_uses"]:
        context = str(shader_use["render_context"])
        source_material, diagnostics = _read_active_mdl_source_material(
            shader_use["prim"], context, root_layer
        )
        if source_material is None or diagnostics:
            continue
        parsed = source_material["parsed"]
        record = {
            "shader_path": source_material["shader_path"],
            "render_contexts": [],
            "authored_source_asset": source_material[
                "authored_source_asset"
            ],
            "resolved_source_path": str(source_material["mdl_path"]),
            "source_resolution_method": source_material[
                "source_resolution_method"
            ],
            "source_sha256": source_material["source_sha256"],
            "source_asset_sub_identifier": source_material[
                "sub_identifier"
            ],
            "selected_material_definition": parsed[
                "selected_material_definition"
            ],
            "selected_definition_source_line": parsed[
                "selected_definition_source_line"
            ],
            "material_family": parsed["material_family"],
            "definition_expression_kind": parsed[
                "definition_expression_kind"
            ],
            "authored_direct_callee": parsed["authored_direct_callee"],
            "direct_callee": parsed["direct_callee"],
            "terminal_material_callee": parsed[
                "terminal_material_callee"
            ],
            "terminal_material_module": _json_safe_value(
                parsed["terminal_material_module"]
            ),
            "parameter_defaults": _json_safe_value(
                parsed["parameter_defaults"]
            ),
            "semantics_sha256": parsed["semantics_sha256"],
            "source_dependency_attestation": _json_safe_value(
                parsed.get("source_dependency_attestation")
            ),
        }
        key = (
            str(record["shader_path"]),
            str(record["authored_source_asset"]),
            str(record["resolved_source_path"]),
            str(record["source_resolution_method"]),
            str(record["source_sha256"]),
            str(record["source_asset_sub_identifier"]),
            str(record["selected_material_definition"]),
            str(record["semantics_sha256"]),
            str(
                (record.get("source_dependency_attestation") or {}).get(
                    "attestation_sha256"
                )
            ),
        )
        records[key] = record
        contexts.setdefault(key, set()).add(context)
    for key, record in records.items():
        record["render_contexts"] = sorted(contexts[key])
    return [records[key] for key in sorted(records)]


def _collect_material_trusted_mdl_source_resolutions(
    material, root_layer=None
) -> list[dict[str, Any]]:
    """Keep MDL source dependencies when USD overrides the texture defaults.

    The module is still needed to identify those defaults, so keep its file
    hashes in the checkpoint even if none of its textures are used.
    """

    if not material:
        return []
    output_roots: list[tuple[Any, str]] = []
    try:
        outputs = UsdShade.ConnectableAPI(material.GetPrim()).GetOutputs()
    except Exception:
        outputs = []
    for output in outputs:
        context, terminal_name = _terminal_context_and_name(output)
        if terminal_name.casefold() != "surface":
            continue
        endpoints, _ = _property_connection_endpoints(output)
        if endpoints:
            output_roots.append((output, context))

    trace = _trace_active_network(output_roots)
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for shader_use in trace["active_shader_uses"]:
        context = str(shader_use["render_context"])
        evidence, diagnostics = _extract_active_mdl_source_textures(
            shader_use["prim"],
            context,
            root_layer,
        )
        if diagnostics:
            continue
        for item in evidence:
            provenance = item.mdl_source_resolution_provenance
            if not provenance:
                continue
            record = {
                "classification": provenance.get("classification"),
                "shader_path": item.shader_path,
                "render_context": context,
                "authored_source_asset": item.mdl_source_authored_asset,
                "resolved_source_path": item.mdl_source_asset_path,
                "source_resolution_method": item.mdl_source_resolution_method,
                "source_asset_sub_identifier": (
                    item.mdl_source_asset_sub_identifier
                ),
                "selected_material_definition": (
                    item.mdl_selected_material_definition
                ),
                "resolution_provenance": _json_safe_value(provenance),
            }
            key = (
                str(record["shader_path"] or ""),
                str(record["render_context"] or ""),
                str(record["authored_source_asset"] or ""),
                str(record["resolved_source_path"] or ""),
                str(record["source_asset_sub_identifier"] or ""),
                str(record["selected_material_definition"] or ""),
                _json_canonical_hash(
                    "trusted-mdl-source-resolution-v1",
                    record["resolution_provenance"],
                ),
            )
            records[key] = record
    return [records[key] for key in sorted(records)]


def collect_material_texture_evidence(material, root_layer=None) -> list[TextureEvidence]:
    """Return image inputs on active surface graphs for *material*.

    All connected render contexts are inspected (for example ``mdl`` and the
    universal context).  Duplicate references are merged without inventing
    evidence for textureless materials.
    """

    if not material:
        return []

    material_prim = material.GetPrim()
    output_roots: list[tuple[Any, str]] = []
    try:
        outputs = UsdShade.ConnectableAPI(material_prim).GetOutputs()
    except Exception:
        outputs = []
    for output in outputs:
        context, terminal_name = _terminal_context_and_name(output)
        if terminal_name.casefold() != "surface":
            continue
        endpoints, _ = _property_connection_endpoints(output)
        if endpoints:
            output_roots.append((output, context))

    if not output_roots:
        return []

    trace = _trace_active_network(output_roots)
    mdl_family_by_shader_context: dict[tuple[str, str], str] = {}
    for shader_use in trace["active_shader_uses"]:
        source_material, diagnostics = _read_active_mdl_source_material(
            shader_use["prim"], shader_use["render_context"], root_layer
        )
        if source_material is None or diagnostics:
            continue
        mdl_family_by_shader_context[
            (
                source_material["shader_path"],
                str(shader_use["render_context"]),
            )
        ] = str(source_material["parsed"]["material_family"])
    raw: dict[tuple[Any, ...], dict[str, Any]] = {}
    for activation in trace["asset_inputs"]:
        shade_input = activation["input"]
        context = activation["context"]
        semantic = activation["semantic"]
        chain = activation["connection_endpoint_chain"]
        input_name = str(shade_input.GetBaseName())
        value = shade_input.Get()
        attribute = shade_input.GetAttr()
        try:
            color_space = str(attribute.GetColorSpace() or "") or None
        except Exception:
            color_space = None
        layer = _authoring_layer(attribute)
        layer_path = _layer_identifier(layer)
        prim = _shade_property_prim(shade_input)
        prim_path = str(prim.GetPath()) if prim else ""
        property_path = str(attribute.GetPath())
        consumer_property_path = semantic.get("consumer_property_path")
        consumer_shader_path = (
            str(consumer_property_path).split(".inputs:", 1)[0]
            if consumer_property_path
            else prim_path
        )
        mdl_material_family = mdl_family_by_shader_context.get(
            (consumer_shader_path, str(context))
        ) or mdl_family_by_shader_context.get((prim_path, str(context)))
        chain_key = tuple(
            (
                edge.get("consumer_property_path"),
                edge.get("source_property_path"),
            )
            for edge in chain
        )
        for authored_asset in _asset_paths(value):
            resolved_paths, method = _resolve_asset_path(
                authored_asset, layer, root_layer
            )
            semantic_repair = _verified_semantic_texture_repair(
                authored_asset.path,
                property_path,
                resolved_paths,
                layer,
                root_layer,
            )
            verified_semantic_repair = None
            effective_authored_asset = authored_asset.path
            if semantic_repair:
                (
                    effective_authored_asset,
                    resolved_paths,
                    verified_semantic_repair,
                ) = semantic_repair
                method = TEXTURE_REPAIR_METHOD
            suffix = Path(
                effective_authored_asset.replace("<UDIM>", "1001")
            ).suffix.casefold()
            if suffix not in IMAGE_SUFFIXES:
                continue
            if (
                mdl_material_family
                in _STRICT_EXACT_TEXTURE_ROLE_MATERIAL_FAMILIES
            ):
                role = (
                    _exact_mdl_texture_input_role(
                        semantic.get("consumer_property_full_name")
                    )
                    or _exact_mdl_texture_input_role(input_name)
                    or "unknown"
                )
            else:
                role = semantic.get("role") or _classify_texture(
                    input_name, effective_authored_asset
                )
            key = (
                role,
                effective_authored_asset,
                prim_path,
                input_name,
                layer_path,
                resolved_paths,
                semantic.get("consumer_property_path"),
                semantic.get("consumer_property_full_name"),
                semantic.get("source_output_path"),
                semantic.get("source_output_name"),
                semantic.get("source_output_channel"),
                color_space,
                mdl_material_family,
                chain_key,
            )
            record = raw.setdefault(
                key,
                {
                    "contexts": set(),
                    "methods": set(),
                    "connection_endpoint_chain": chain,
                    "verified_semantic_repair": verified_semantic_repair,
                },
            )
            record["contexts"].add(context)
            if method:
                record["methods"].add(method)

    evidence = [
        TextureEvidence(
            role=key[0],
            authored_asset=key[1],
            shader_path=key[2],
            input_name=key[3],
            authoring_layer=key[4],
            resolved_paths=key[5],
            render_contexts=tuple(sorted(record["contexts"])),
            resolution_method=(
                sorted(record["methods"])[0] if record["methods"] else None
            ),
            semantic_consumer_property_path=key[6],
            semantic_consumer_property_full_name=key[7],
            semantic_source_output_path=key[8],
            semantic_source_output_name=key[9],
            semantic_source_output_channel=key[10],
            color_space=key[11],
            mdl_material_family=key[12],
            connection_endpoint_chain=record["connection_endpoint_chain"],
            verified_semantic_repair=record["verified_semantic_repair"],
        )
        for key, record in raw.items()
    ]
    for shader_use in trace["active_shader_uses"]:
        mdl_evidence, _ = _extract_active_mdl_source_textures(
            shader_use["prim"],
            shader_use["render_context"],
            root_layer,
        )
        evidence.extend(mdl_evidence)
    evidence = _suppress_overridden_mdl_texture_defaults(evidence)
    unique_evidence = {
        (
            item.role,
            item.authored_asset,
            item.resolved_paths,
            item.authoring_layer,
            item.shader_path,
            item.input_name,
            item.render_contexts,
            item.semantic_consumer_property_path,
            item.semantic_source_output_path,
            item.evidence_source,
            item.mdl_source_asset_path,
            item.mdl_selected_material_definition,
            item.mdl_material_family,
            item.mdl_gamma_expression,
            item.mdl_gamma_is_implicit,
            tuple(
                (
                    edge.get("consumer_property_path"),
                    edge.get("source_property_path"),
                )
                for edge in item.connection_endpoint_chain
            ),
        ): item
        for item in evidence
    }
    return sorted(
        unique_evidence.values(),
        key=lambda item: (
            item.role,
            item.authored_asset,
            item.shader_path,
            item.input_name,
        ),
    )


def select_resolved_textures(
    evidence: Iterable[TextureEvidence],
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Select a file only when a role has one unambiguous image.

    Candidate ordering is deterministic for inspection, but ordering is never
    used to hide a material with multiple distinct sources or an image set
    that requires UDIM-aware rendering.  Existing callers keep the same
    three-return-value API; ambiguity provenance lives in each candidate and
    unresolved record.
    """

    evidence_by_role: dict[str, list[TextureEvidence]] = {}
    for item in evidence:
        evidence_by_role.setdefault(item.role, []).append(item)

    selected: dict[str, str] = {}
    serialized_candidates: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[dict[str, Any]] = []
    for role, all_items in sorted(evidence_by_role.items()):
        candidates = [item for item in all_items if item.resolved_paths]
        unresolved_items = [item for item in all_items if not item.resolved_paths]
        resolved_source_assets = {item.authored_asset for item in candidates}
        competing_unresolved_sources = {
            item.authored_asset
            for item in unresolved_items
            if item.authored_asset not in resolved_source_assets
        }
        distinct_sources = {
            (item.authored_asset, item.shader_path, item.input_name)
            for item in all_items
        }
        resolved_files = sorted(
            {path for item in candidates for path in item.resolved_paths}
        )
        role_has_udim = any(item.is_udim for item in all_items)
        role_has_multiple_sources = len(distinct_sources) > 1
        ambiguous = (
            role_has_udim
            or len(resolved_files) > 1
            or bool(competing_unresolved_sources)
        )
        if role_has_udim:
            selection_status = "ambiguous_udim_set"
        elif len(resolved_files) > 1:
            selection_status = "ambiguous_multiple_resolved_files"
        elif competing_unresolved_sources:
            selection_status = "ambiguous_unresolved_competing_source"
        elif not resolved_files:
            selection_status = "unresolved"
        elif role_has_multiple_sources:
            selection_status = "selected_shared_file_multiple_sources"
        else:
            selection_status = "selected_single_source"

        role_metadata = {
            "role_selection_status": selection_status,
            "role_is_ambiguous": ambiguous,
            "role_has_udim": role_has_udim,
            "role_has_multiple_sources": role_has_multiple_sources,
            "role_source_count": len(distinct_sources),
            "role_distinct_resolved_file_count": len(resolved_files),
            "role_distinct_resolved_files": resolved_files,
            "role_competing_unresolved_sources": sorted(
                competing_unresolved_sources
            ),
        }
        for item in unresolved_items:
            unresolved.append({**item.to_dict(), **role_metadata})
        if not candidates:
            continue

        items = candidates
        items = sorted(
            items,
            key=lambda item: (
                "mdl" not in item.render_contexts,
                "universal" not in item.render_contexts,
                item.resolution_method is None,
                item.authored_asset,
                item.shader_path,
            ),
        )
        if not ambiguous:
            selected[role] = resolved_files[0]
        serialized_candidates[role] = [
            {**item.to_dict(), **role_metadata} for item in items
        ]
    return selected, serialized_candidates, unresolved


def _select_exact_udim_texture_sets(
    texture_candidates: dict[str, list[dict[str, Any]]],
    unresolved_textures: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select only a single-source UDIM role the renderer can map exactly."""

    unresolved_roles = {
        str(item.get("role"))
        for item in unresolved_textures
        if isinstance(item, Mapping) and item.get("role")
    }
    selected: dict[str, dict[str, Any]] = {}
    for role, records in sorted(texture_candidates.items()):
        if len(records) != 1 or role in unresolved_roles:
            continue
        candidate = records[0]
        if not isinstance(candidate, dict):
            continue
        paths = candidate.get("resolved_paths")
        authored_asset = candidate.get("authored_asset")
        if (
            candidate.get("role") != role
            or candidate.get("role_selection_status") != "ambiguous_udim_set"
            or candidate.get("role_is_ambiguous") is not True
            or candidate.get("role_has_udim") is not True
            or candidate.get("role_has_multiple_sources") is not False
            or type(candidate.get("role_source_count")) is not int
            or candidate.get("role_source_count") != 1
            or candidate.get("role_competing_unresolved_sources") != []
            or candidate.get("is_udim") is not True
            or candidate.get("used_path_fallback") is not False
            or not isinstance(authored_asset, str)
            or not isinstance(paths, list)
            or not paths
            or not all(isinstance(path, str) and path for path in paths)
            or candidate.get("role_distinct_resolved_files") != paths
            or candidate.get("role_distinct_resolved_file_count") != len(paths)
            or candidate.get("resolved_file_count") != len(paths)
            or candidate.get("has_multiple_resolved_files") is not (len(paths) > 1)
        ):
            continue

        tile_records: list[dict[str, Any]] = []
        parent_paths: set[Path] = set()
        try:
            for serialized_path in paths:
                path = Path(serialized_path)
                if (
                    not path.is_absolute()
                    or not path.is_file()
                    or str(path.resolve()) != serialized_path
                ):
                    tile_records = []
                    break
                tile_id = _udim_tile_id(authored_asset, path)
                if tile_id is None:
                    tile_records = []
                    break
                parent_paths.add(path.parent)
                tile_records.append(
                    {
                        "tile_id": tile_id,
                        "resolved_path": serialized_path,
                        "file_sha256": _file_sha256(path),
                    }
                )
        except OSError:
            tile_records = []
        tile_ids = [record["tile_id"] for record in tile_records]
        tile_coordinates = {
            ((tile_id - 1001) % 10, (tile_id - 1001) // 10)
            for tile_id in tile_ids
        }
        complete_rectangle = (
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
        if (
            len(tile_records) != len(paths)
            or len(parent_paths) != 1
            or tile_ids != sorted(set(tile_ids))
            or tile_coordinates != complete_rectangle
        ):
            continue

        canonical_candidate = {
            **candidate,
            "role_selection_status": "selected_exact_udim_set",
            "role_is_ambiguous": False,
        }
        records[:] = [canonical_candidate]
        selected[role] = {
            "texture_type": "udim",
            "authored_asset": authored_asset,
            "tiles": tile_records,
            "evidence_sha256": _json_canonical_hash(
                "textured-render-provenance", canonical_candidate
            ),
        }
    return selected


def build_material_record(material, root_layer=None) -> dict[str, Any] | None:
    """Build a material record with its USD and texture sources."""

    if not material:
        return None
    evidence = collect_material_texture_evidence(material, root_layer=root_layer)
    textures, texture_candidates, unresolved_textures = select_resolved_textures(
        evidence
    )
    texture_sets = _select_exact_udim_texture_sets(
        texture_candidates, unresolved_textures
    )
    material_prim = material.GetPrim()
    serialized_evidence = [item.to_dict() for item in evidence]
    texture_fallbacks = [
        item for item in serialized_evidence if item["used_path_fallback"]
    ]
    source_repairs = [
        item
        for item in serialized_evidence
        if item.get("resolution_method")
        in {
            SOURCE_REPAIR_METHOD,
            CASE_REPAIR_METHOD,
            PATH_CASE_REPAIR_METHOD,
            TEXTURE_REPAIR_METHOD,
        }
    ]
    source_repairs.extend(
        _collect_material_trusted_mdl_source_resolutions(
            material,
            root_layer=root_layer,
        )
    )
    untrusted_mdl_candidates = _collect_material_untrusted_mdl_candidates(
        material, root_layer=root_layer
    )
    selected_mdl_material_semantics = (
        _collect_selected_mdl_material_semantics(
            material, root_layer=root_layer
        )
    )
    ambiguity_by_role: dict[str, dict[str, Any]] = {}
    selection_records = [
        item
        for role_records in texture_candidates.values()
        for item in role_records
    ] + list(unresolved_textures)
    for item in selection_records:
        if not item.get("role_is_ambiguous"):
            continue
        role = str(item.get("role", "unknown"))
        ambiguity_by_role.setdefault(
            role,
            {
                "role": role,
                "role_selection_status": item.get("role_selection_status"),
                "role_has_udim": bool(item.get("role_has_udim")),
                "role_source_count": item.get("role_source_count"),
                "role_distinct_resolved_file_count": item.get(
                    "role_distinct_resolved_file_count"
                ),
                "role_distinct_resolved_files": item.get(
                    "role_distinct_resolved_files", []
                ),
                "role_competing_unresolved_sources": item.get(
                    "role_competing_unresolved_sources", []
                ),
            },
        )
    selected_texture_evidence = {
        role: candidates
        for role, selected_path in textures.items()
        for candidates in [texture_candidates.get(role, [])]
        if candidates
        and all(
            candidate.get("resolved_paths") == [selected_path]
            for candidate in candidates
        )
    }
    selected_texture_evidence.update(
        {role: texture_candidates[role] for role in texture_sets}
    )
    active_material_graph = collect_active_material_graph(
        material, root_layer=root_layer
    )
    color_resolution = active_material_graph[
        "surface_base_color_value_resolution"
    ]
    color_resolution_attestation = {
        "active_graph_resolution_sha256": color_resolution[
            "resolution_sha256"
        ],
        "consumer_count": color_resolution["consumer_count"],
        "constant_source_count": color_resolution[
            "constant_source_count"
        ],
        "blocker_count": len(color_resolution["blockers"]),
    }
    record = {
        "material_record_id": material_record_id(material),
        "material_identity": _material_identity(material),
        "material_name": str(material_prim.GetName()),
        "material_path": str(material.GetPath()),
        "source_material_path": _source_mesh_path(material_prim),
        "active_material_graph": active_material_graph,
        "selected_mdl_material_semantics": (
            selected_mdl_material_semantics
        ),
        "surface_base_color_constant_evidence": color_resolution[
            "constant_sources"
        ],
        "surface_base_color_value_resolution_attestation": (
            color_resolution_attestation
        ),
        "surface_texture_evidence": serialized_evidence,
        "texture_fallbacks": texture_fallbacks,
        "source_repairs": source_repairs,
        "source_repair_warning": (
            "A known source error was corrected using a matching repair. "
            "File hashes and the repair reason are recorded; "
            "the source package still needs fixing."
            if source_repairs
            else None
        ),
        "untrusted_mdl_source_candidates": untrusted_mdl_candidates,
        "untrusted_mdl_source_candidate_warning": (
            "This MDL source is excluded from texture selection because its "
            "selected material definition or texture expressions could not "
            "be checked."
            if untrusted_mdl_candidates
            else None
        ),
        "surface_texture_role_conflicts": [
            item for item in serialized_evidence if item["authored_role_conflict"]
        ],
        "surface_texture_ambiguities": [
            ambiguity_by_role[role] for role in sorted(ambiguity_by_role)
        ],
        "surface_unknown_texture_roles": [
            item for item in serialized_evidence if item["role"] == "unknown"
        ],
        "selected_surface_textures": textures,
        "selected_surface_texture_sets": texture_sets,
        "selected_surface_texture_evidence": selected_texture_evidence,
        "surface_texture_candidates": texture_candidates,
        "unresolved_surface_textures": unresolved_textures,
    }
    record["annotation_warnings"] = material_record_annotation_warnings(record)
    return record


def material_record_annotation_warnings(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return coded appearance-evidence diagnostics for policy evaluation.

    These conditions do not corrupt USD composition or segment identity, but
    some still make annotation unsafe (for example unresolved or semantically
    unknown active textures).  ``annotation_input_policy`` is the authority
    that separates fail-closed blockers from fully represented warnings.  Each
    diagnostic points back to the complete canonical field; no texture,
    connection, or authored value is discarded.
    """

    material_path = str(record.get("material_path") or "unknown material")
    warnings: list[dict[str, Any]] = []

    def add(
        code: str,
        message: str,
        evidence_field: str,
        entries: Any,
        **details: Any,
    ) -> None:
        if not entries:
            return
        serialized = (
            list(entries)
            if isinstance(entries, (list, tuple))
            else entries
        )
        entry_count = (
            len(serialized) if isinstance(serialized, list) else 1
        )
        warnings.append(
            {
                "code": code,
                "message": message,
                "material_path": material_path,
                "evidence_field": evidence_field,
                "entry_count": entry_count,
                "entries": _json_safe_value(serialized),
                **_json_safe_value(details),
            }
        )

    graph = record.get("active_material_graph")
    graph = graph if isinstance(graph, dict) else {}
    add(
        "ignored_invalid_material_connection_fallback",
        "Invalid authored source properties are ignored by OpenUSD; the exact "
        "authored consumer-input values are effective and retained.",
        "active_material_graph.ignored_invalid_connection_fallbacks",
        graph.get("ignored_invalid_connection_fallbacks"),
        repair_required=True,
    )
    add(
        "unresolved_active_texture_evidence",
        "An active authored image is missing from the source package; it is "
        "shown as missing evidence and is never guessed.",
        "unresolved_surface_textures",
        record.get("unresolved_surface_textures"),
        repair_required=True,
    )
    add(
        "authored_texture_role_conflict",
        "The authored shader-input role conflicts with the filename; authored "
        "graph semantics are retained and the mismatch is labeled.",
        "surface_texture_role_conflicts",
        record.get("surface_texture_role_conflicts"),
        repair_required=True,
    )
    add(
        "ambiguous_active_texture_evidence",
        "Competing active files or sources, or a tiled source without an exact "
        "selected tile manifest, exists for one role; every candidate is "
        "retained and none is chosen by order.",
        "surface_texture_ambiguities",
        record.get("surface_texture_ambiguities"),
    )
    add(
        "unknown_active_texture_role",
        "An active image role is not classified; the authored input and image "
        "remain labeled as unknown.",
        "surface_unknown_texture_roles",
        record.get("surface_unknown_texture_roles"),
    )
    add(
        "texture_path_fallback",
        "A filesystem recovery resolved an authored image non-authoritatively; "
        "the method is retained and labeled.",
        "texture_fallbacks",
        record.get("texture_fallbacks"),
        repair_required=True,
    )
    add(
        "source_repair",
        "A known source error was corrected using a matching repair. "
        "File hashes and the repair reason are recorded; "
        "the source package still needs fixing.",
        "source_repairs",
        record.get("source_repairs"),
        repair_required=True,
    )
    add(
        "untrusted_mdl_source",
        "The selected MDL implementation or its texture expressions could "
        "not be checked, so they are excluded from texture selection.",
        "untrusted_mdl_source_candidates",
        record.get("untrusted_mdl_source_candidates"),
        repair_required=True,
    )
    if graph.get("has_time_varying_values"):
        add(
            "time_varying_material_default_snapshot",
            "The active material varies with time; the record explicitly "
            "represents the default-time snapshot.",
            "active_material_graph.has_time_varying_values",
            [True],
        )

    terminals = graph.get("terminals") or []
    custom_outputs = graph.get("connected_nonstandard_outputs") or []
    has_surface = any(
        isinstance(terminal, dict)
        and str(terminal.get("terminal", "")).casefold() == "surface"
        for terminal in terminals
    )
    if not terminals and not custom_outputs:
        add(
            "missing_connected_material_terminal",
            "The bound Material has no connected terminal; appearance is "
            "retained as unavailable and annotation is blocked before visual "
            "construction.",
            "active_material_graph.terminals",
            [{"terminals": [], "connected_nonstandard_outputs": []}],
            repair_required=True,
        )
    elif not has_surface:
        add(
            "unsupported_material_surface_terminal",
            "The Material has no supported surface terminal; its authored "
            "terminal metadata is retained without inferring appearance.",
            "active_material_graph.terminals",
            [
                {
                    "terminals": terminals,
                    "connected_nonstandard_outputs": custom_outputs,
                }
            ],
        )
    return warnings


def material_record_extraction_errors(record: dict[str, Any]) -> list[str]:
    """Return only graph corruption that makes extraction untrustworthy.

    The full graph is retained for diagnostics, but annotation must stop when
    an active endpoint cannot be followed.  Otherwise a dangling connection
    can silently turn an inactive fallback value into the selected texture.
    Missing or unsupported appearance is retained as a coded warning record;
    the shared annotation policy consumes that record and blocks production
    before any VLM input is constructed.
    """

    material_path = str(record.get("material_path") or "unknown material")
    graph = record.get("active_material_graph")
    if not isinstance(graph, dict):
        return [f"active material graph is missing at {material_path}"]

    errors: list[str] = []
    if graph.get("has_lossy_value_fallback"):
        errors.append(
            "lossy active material graph value serialization is unsupported at "
            f"{material_path}"
        )

    untraversable = graph.get("untraversable_connection_endpoints") or []
    if untraversable:
        errors.append(
            f"active material graph has {len(untraversable)} untraversable "
            f"connection endpoint(s) at {material_path}"
        )

    invalid_paths = sorted(
        {
            str(path)
            for node in (graph.get("nodes") or [])
            if isinstance(node, dict)
            for path in (node.get("invalid_connection_paths") or [])
            if str(path)
        }
    )
    if invalid_paths:
        errors.append(
            f"active material graph has invalid connection path(s) at "
            f"{material_path}: {invalid_paths[:10]}"
        )

    invalid_material_outputs = graph.get(
        "invalid_material_output_connections"
    ) or []
    if invalid_material_outputs:
        errors.append(
            "material has invalid terminal connection path(s) at "
            f"{material_path}"
        )

    resolution = graph.get("surface_base_color_value_resolution")
    if not isinstance(resolution, dict):
        errors.append(
            "surface base-color value resolution is missing at "
            f"{material_path}"
        )
    else:
        blockers = resolution.get("blockers")
        if not isinstance(blockers, list):
            errors.append(
                "surface base-color value-resolution blockers are malformed at "
                f"{material_path}"
            )
        elif blockers:
            errors.append(
                "surface base-color value resolution has "
                f"{len(blockers)} blocker(s) at {material_path}"
            )
        try:
            expected_resolution_sha256 = _json_canonical_hash(
                "surface-base-color-value-resolution-v1",
                {
                    key: value
                    for key, value in resolution.items()
                    if key != "resolution_sha256"
                },
            )
        except (TypeError, ValueError):
            expected_resolution_sha256 = None
        if resolution.get("resolution_sha256") != expected_resolution_sha256:
            errors.append(
                "surface base-color value-resolution attestation is invalid at "
                f"{material_path}"
            )
        if record.get("surface_base_color_constant_evidence") != resolution.get(
            "constant_sources"
        ):
            errors.append(
                "material-level base-color constant evidence disagrees with "
                f"the active graph at {material_path}"
            )
        attestation = record.get(
            "surface_base_color_value_resolution_attestation"
        )
        expected_attestation = {
            "active_graph_resolution_sha256": resolution.get(
                "resolution_sha256"
            ),
            "consumer_count": resolution.get("consumer_count"),
            "constant_source_count": resolution.get(
                "constant_source_count"
            ),
            "blocker_count": (
                len(blockers) if isinstance(blockers, list) else None
            ),
        }
        if not isinstance(attestation, Mapping) or any(
            attestation.get(key) != value
            for key, value in expected_attestation.items()
        ):
            errors.append(
                "material-level base-color value-resolution attestation "
                f"disagrees with the active graph at {material_path}"
            )
    return errors


def region_metadata(region: MaterialRegion) -> dict[str, Any]:
    """Return JSON-safe geometry/binding metadata for one region."""

    encoded_face_indices = encode_face_indices(region.face_indices)
    purpose_text = region.purpose.casefold()
    return {
        "segment_id": region.segment_id,
        "prim_path": region.mesh_path,
        "source_prim_path": region.source_mesh_path,
        "source_prim_path_kind": region.source_mesh_path_kind,
        "source_prim_prototype_relative_path": (
            region.source_mesh_prototype_relative_path
        ),
        "diagnostic_runtime_source_prim_path": (
            region.diagnostic_runtime_source_mesh_path
        ),
        "diagnostic_runtime_source_prim_path_is_stable": False,
        "geometry_mesh_label": region.geometry_mesh_label,
        "geometry_subset_label": region.geometry_subset_label,
        "subset_path": region.subset_path,
        "region_kind": region.region_kind,
        "face_count": region.face_count,
        "face_index_hash": region.face_index_hash,
        "face_indices": encoded_face_indices,
        "material_path": region.material_path,
        "source_material_path": region.source_material_path,
        "material_record_id": material_record_id(region.material),
        "binding_relationship": region.binding_relationship,
        "selected_binding_purpose": region.selected_binding_purpose,
        "material_binding_purpose_priority": list(
            MATERIAL_BINDING_PURPOSE_PRIORITY
        ),
        "material_bindings": region.material_bindings,
        "visibility": region.visibility,
        "visibility_classification": (
            "invisible" if region.visibility.casefold() == "invisible" else "visible"
        ),
        "purpose": region.purpose,
        "purpose_classification": (
            "render_content"
            if purpose_text in {"default", "render"}
            else "proxy_content" if purpose_text == "proxy"
            else "guide_content" if purpose_text == "guide" else "other"
        ),
        "is_render_content": purpose_text in {"default", "render"},
        "default_prim_path": region.default_prim_path,
        "default_prim_membership": region.default_prim_membership,
        "display_primvars": region.display_primvars,
        "is_instance_proxy": region.is_instance_proxy,
        "point_instancer_occurrences": region.point_instancer_occurrences,
    }
