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

from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as sp


@wp.kernel
def dense_chol_batched(
    size: int,
    regul: float,
    lhs: wp.array2d(dtype=float),
    L: wp.array2d(dtype=float),
):
    i = wp.tid()
    wp.dense_chol(size, lhs[i], regul, L[i])


@wp.kernel
def dense_chol_subs_batched(
    rhs: wp.array2d(dtype=float),
    res: wp.array2d(dtype=float),
    L: wp.array2d(dtype=float),
):
    i = wp.tid()
    wp.dense_subs(rhs.shape[1], L[i], rhs[i], res[i])


def create_batched_cholesky_kernel(num_dofs):
    @fem.cache.dynamic_kernel(
        suffix=num_dofs, kernel_options={"enable_backward": False}
    )
    def eval_tiled_dense_cholesky_batched(A: wp.array3d(dtype=float), reg: float):
        block, _ = wp.tid()

        a = wp.tile_load(
            A[block], shape=(num_dofs, num_dofs), offset=(0, 0), storage="shared"
        )
        r = wp.tile_ones(dtype=float, shape=(num_dofs)) * reg
        b = wp.tile_diag_add(a, r)
        l = wp.tile_cholesky(b)
        wp.tile_store(A[block], l)

    return eval_tiled_dense_cholesky_batched


def create_batched_cholesky_solve_kernel(num_dofs):
    @fem.cache.dynamic_kernel(suffix=num_dofs)
    def solve_tiled_dense_cholesky_batched(
        L: wp.array3d(dtype=float),
        X: wp.array1d(dtype=float),
        Y: wp.array1d(dtype=float),
    ):
        block, _ = wp.tid()

        a = wp.tile_load(L[block], shape=(num_dofs, num_dofs), storage="shared")
        x = wp.tile_load(X, offset=block * num_dofs, shape=num_dofs)
        y = wp.tile_cholesky_solve(a, x)
        wp.tile_store(Y, y, offset=block * num_dofs)

    return solve_tiled_dense_cholesky_batched


@wp.kernel
def _coarsen_structure(
    row_ratio: int,
    col_ratio: int,
    src_offsets: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    dst_offsets: wp.array(dtype=int),
    dst_columns: wp.array(dtype=int),
):
    row = wp.tid()
    src_beg = src_offsets[row * row_ratio]
    beg = src_beg // (row_ratio * col_ratio)
    end = src_offsets[(row + 1) * row_ratio] // (row_ratio * col_ratio)

    dst_offsets[row + 1] = end
    for block in range(beg, end):
        dst_columns[block] = (
            src_columns[src_beg + col_ratio * (block - beg)] // col_ratio
        )


def bsr_coarsen_aligned(src: sp.BsrMatrix, block_shape, coarse=None):
    coarse_type = wp.mat(shape=block_shape, dtype=src.scalar_type)
    block_ratios = (
        block_shape[0] // src.block_shape[0],
        block_shape[1] // src.block_shape[1],
    )

    if coarse is None:
        coarse = sp.bsr_zeros(
            rows_of_blocks=src.nrow // block_ratios[0],
            cols_of_blocks=src.ncol // block_ratios[1],
            block_type=coarse_type,
        )
    else:
        sp.bsr_set_zero(
            coarse,
            rows_of_blocks=src.nrow // block_ratios[0],
            cols_of_blocks=src.ncol // block_ratios[1],
        )

    # compute the structure
    nnz = src.nnz_sync() // (block_ratios[0] * block_ratios[1])
    if coarse.columns.shape[0] < nnz:
        coarse.columns = wp.empty(nnz, dtype=int)
    if coarse.values.shape[0] < nnz:
        coarse.values = wp.empty(nnz, dtype=coarse_type)
    wp.launch(
        _coarsen_structure,
        dim=coarse.nrow,
        inputs=[
            block_ratios[0],
            block_ratios[1],
            src.offsets,
            src.columns,
            coarse.offsets,
            coarse.columns,
        ],
    )
    coarse.nnz = nnz
    coarse.copy_nnz_async()
    sp.bsr_assign(src=src, dest=coarse, masked=True)

    return coarse

    # TODO for later
    # masked always keep structure
    # bsr_prune to remove actual zeros?


@wp.kernel
def bsr_mul_diag(
    Bt_values: wp.array3d(dtype=float),
    Bt_columns: wp.array(dtype=int),
    C_values: wp.array(dtype=Any),
):
    i, r = wp.tid()
    col = Bt_columns[i]

    C = C_values[col]

    Btr = Bt_values[i, r]
    BtC = wp.vec3(Btr[0], Btr[1], Btr[2]) @ C
    for k in range(3):
        Btr[k] = BtC[k]
