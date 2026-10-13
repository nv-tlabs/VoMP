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

import os
from typing import *

BACKEND = "spconv"
DEBUG = False
ATTN = "flash_attn"

env_sparse_backend = os.environ.get("SPARSE_BACKEND")
env_sparse_debug = os.environ.get("SPARSE_DEBUG")
env_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
if env_sparse_attn is None:
    env_sparse_attn = os.environ.get("ATTN_BACKEND")
if env_sparse_backend is not None and env_sparse_backend in ["spconv", "torchsparse"]:
    BACKEND = env_sparse_backend
if env_sparse_debug is not None:
    DEBUG = env_sparse_debug == "1"
if env_sparse_attn is not None and env_sparse_attn in ["xformers", "flash_attn"]:
    ATTN = env_sparse_attn
print(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")

from .basic import (
    SparseTensor,
    sparse_batch_broadcast,
    sparse_batch_op,
    sparse_cat,
    sparse_unbind,
)
from .norm import (
    SparseGroupNorm,
    SparseLayerNorm,
    SparseGroupNorm32,
    SparseLayerNorm32,
)
from .nonlinearity import (
    SparseReLU,
    SparseSiLU,
    SparseGELU,
    SparseActivation,
)
from .linear import SparseLinear
from .attention import (
    sparse_scaled_dot_product_attention,
    SerializeMode,
    sparse_serialized_scaled_dot_product_self_attention,
    sparse_windowed_scaled_dot_product_self_attention,
    SparseMultiHeadAttention,
)
from .conv import (
    SparseConv3d,
    SparseInverseConv3d,
)
from .spatial import (
    SparseDownsample,
    SparseUpsample,
    SparseSubdivide,
)
from . import transformer

__all__ = [
    "SparseTensor",
    "sparse_batch_broadcast",
    "sparse_batch_op",
    "sparse_cat",
    "sparse_unbind",
    "SparseGroupNorm",
    "SparseLayerNorm",
    "SparseGroupNorm32",
    "SparseLayerNorm32",
    "SparseReLU",
    "SparseSiLU",
    "SparseGELU",
    "SparseActivation",
    "SparseLinear",
    "sparse_scaled_dot_product_attention",
    "SerializeMode",
    "sparse_serialized_scaled_dot_product_self_attention",
    "sparse_windowed_scaled_dot_product_self_attention",
    "SparseMultiHeadAttention",
    "SparseConv3d",
    "SparseInverseConv3d",
    "SparseDownsample",
    "SparseUpsample",
    "SparseSubdivide",
    "transformer",
]


def set_backend(backend: Literal["spconv", "torchsparse"]):
    global BACKEND
    BACKEND = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


def set_attn(attn: Literal["xformers", "flash_attn"]):
    global ATTN
    ATTN = attn
