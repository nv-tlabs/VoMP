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

from typing import Any, Tuple

import numpy as np
import warp as wp
import warp.sparse as sp
from warp.fem.utils import inverse_qr
from warp.fem.utils import array_axpy

from warp.examples.fem.utils import bsr_cg

wp.set_module_options({"enable_backward": False})
wp.set_module_options({"fast_math": True})


def diff_bsr_mv(
    A: sp.BsrMatrix,
    x: wp.array,
    y: wp.array,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
    self_adjoint: bool = False,
):
    """Performs y = alpha*A*x + beta*y and records the adjoint on the tape"""

    from warp.context import runtime

    tape = runtime.tape
    if tape is not None and (x.requires_grad or y.requires_grad):

        def backward():
            # adj_x += adj_y * alpha
            # adj_y = adj_y * beta

            sp.bsr_mv(
                A=A,
                x=y.grad,
                y=x.grad,
                alpha=alpha,
                beta=1.0,
                transpose=(not transpose) and (not self_adjoint),
            )
            if beta != 1.0:
                array_axpy(x=y.grad, y=y.grad, alpha=0.0, beta=beta)

        runtime.tape.record_func(backward, arrays=[x, y])

    runtime.tape = None
    # in case array_axpy eventually records its own stuff
    sp.bsr_mv(A, x, y, alpha, beta, transpose)
    runtime.tape = tape


# MFEM SYSTEM
# for F = RS variant
#
# [ A          -B' ]
# [     H      Cd' ]
# [         W  Cr' ]
# [ -B  Cs  Cr  0  ]
#


class MFEMSystem:
    """Builds a linear operator corresponding to the saddle-point linear system [A B^T; B 0]"""

    def __init__(
        self,
        A: sp.BsrMatrix,
        H: sp.BsrMatrix,
        W: sp.BsrMatrix,
        B: sp.BsrMatrix,
        Cs: sp.BsrMatrix,
        Cr: sp.BsrMatrix,
        Bt: sp.BsrMatrix = None,
    ):
        self._A = A
        self._H = H
        self._W = W
        self._B = B
        self._Bt = Bt
        self._Cs = Cs
        self._Cr = Cr

    def cast(self, scalar_type):
        if wp.types.types_equal(scalar_type, self._A.scalar_type):
            return self

        return MFEMSystem(
            sp.bsr_copy(self._A, scalar_type=scalar_type),
            sp.bsr_copy(self._H, scalar_type=scalar_type),
            (
                sp.bsr_copy(self._W, scalar_type=scalar_type)
                if self._W is not None
                else None
            ),
            sp.bsr_copy(self._B, scalar_type=scalar_type),
            sp.bsr_copy(self._Cs, scalar_type=scalar_type),
            (
                sp.bsr_copy(self._Cr, scalar_type=scalar_type)
                if self._W is not None
                else None
            ),
        )

    def solve_schur(
        lhs,
        rhs: Tuple,
        tol=1.0e-8,
        max_iters=1000,
        work_arrays=None,
        reuse_topology=False,
    ):
        rhs_type = wp.types.type_scalar_type(rhs[0].dtype)
        lhs_type = lhs._A.scalar_type

        if not wp.types.types_equal(lhs_type, rhs_type):
            rhs_cast = tuple(
                wp.empty(
                    shape=v.shape,
                    dtype=wp.vec(length=wp.types.type_length(v.dtype), dtype=lhs_type),
                )
                for v in rhs
            )
            for v, v_cast in zip(rhs, rhs_cast):
                wp.utils.array_cast(in_array=v, out_array=v_cast)

            res = lhs.solve_schur(
                rhs_cast,
                tol=tol,
                max_iters=max_iters,
                work_arrays=work_arrays,
                reuse_topology=reuse_topology,
            )

            res_cast = tuple(
                wp.empty(
                    shape=v.shape,
                    dtype=wp.vec(length=wp.types.type_length(v.dtype), dtype=rhs_type),
                )
                for v in res
            )
            for v, v_cast in zip(res, res_cast):
                wp.utils.array_cast(in_array=v, out_array=v_cast)

            return res_cast

        if lhs._Cr is None:
            return lhs.solve_schur_no_R(
                rhs,
                tol=tol,
                max_iters=max_iters,
                work_arrays=work_arrays,
                reuse_topology=reuse_topology,
            )

        u_rhs, f, w_lambda, c_k = rhs

        A = lhs._A
        H = lhs._H
        W_skew = lhs._W
        B = lhs._B
        CSk = lhs._Cs
        CRk = lhs._Cr

        u_matrix = sp.bsr_copy(A)

        H_inv = wp.empty_like(H.values)
        W_skew_inv = wp.empty_like(W_skew.values)
        CHiCt_inv = wp.empty(
            shape=H.nrow, dtype=wp.mat(shape=(9, 9), dtype=H.scalar_type)
        )
        lambda_rhs = wp.clone(c_k, requires_grad=False)

        wp.launch(
            invert_blocks,
            dim=W_skew.nnz,
            inputs=[W_skew.values, W_skew_inv],
            device=W_skew.device,
        )
        wp.launch(
            invert_blocks,
            dim=H.nnz,
            inputs=[H.values, H_inv],
            device=H.device,
        )

        wp.launch(
            kernel=compute_first_schur,
            dim=CHiCt_inv.shape,
            inputs=[
                CHiCt_inv,
                CSk.values,
                CRk.values,
                H_inv,
                W_skew_inv,
                lambda_rhs,
                f,
                w_lambda,
            ],
        )

        BtCHiCt_inv = sp.bsr_transposed(B) if lhs._Bt is None else sp.bsr_copy(lhs._Bt)
        wp.launch(
            bsr_mul_diag,
            dim=BtCHiCt_inv.nnz,
            inputs=[BtCHiCt_inv.values, BtCHiCt_inv.columns, CHiCt_inv],
        )

        sp.bsr_mm(
            x=BtCHiCt_inv,
            y=B,
            z=u_matrix,
            alpha=1.0,
            beta=1.0,
            work_arrays=work_arrays,
            reuse_topology=reuse_topology,
        )
        sp.bsr_mv(A=BtCHiCt_inv, x=lambda_rhs, y=u_rhs, alpha=-1.0, beta=1.0)

        delta_du = wp.zeros_like(u_rhs)
        err, niter = bsr_cg(
            u_matrix, b=u_rhs, x=delta_du, tol=tol, max_iters=max_iters, quiet=True
        )

        if np.isnan(err):
            raise RuntimeError(f"Solver fail, rhs= {np.linalg.norm(u_rhs.numpy())}")

        # other variable updates

        # get back lambda
        # -B du  -ChiC lambda = lambda_k

        sp.bsr_mv(A=B, x=delta_du, y=lambda_rhs, alpha=1.0, beta=1.0)
        dLambda = wp.empty_like(lambda_rhs)
        dS = wp.empty_like(f)
        dR = wp.empty_like(w_lambda)

        wp.launch(
            kernel=compute_dLambdadRdS,
            dim=H_inv.shape[0],
            inputs=[
                CSk.values,
                CRk.values,
                CHiCt_inv,
                H_inv,
                W_skew_inv,
                lambda_rhs,
                f,
                w_lambda,
                dLambda,
                dS,
                dR,
            ],
        )

        return delta_du, dS, dR, dLambda

    def solve_schur_no_R(
        lhs,
        rhs: Tuple,
        tol=1.0e-8,
        max_iters=1000,
        work_arrays=None,
        reuse_topology=False,
    ):
        u_rhs, f, w_lambda, c_k = rhs

        A = lhs._A
        H = lhs._H
        B = lhs._B
        CSk = lhs._Cs

        u_matrix = sp.bsr_copy(A)

        Cs_inv = sp.bsr_copy(CSk)
        wp.launch(
            invert_blocks,
            dim=Cs_inv.nnz,
            inputs=[CSk.values, Cs_inv.values],
            device=H.device,
        )

        CHiCt_inv = wp.empty(
            shape=H.nrow, dtype=wp.mat(shape=(6, 6), dtype=H.scalar_type)
        )

        wp.launch(
            kernel=compute_first_schur_no_R,
            dim=CHiCt_inv.shape,
            inputs=[
                CHiCt_inv,
                Cs_inv.values,
                H.values,
            ],
        )

        ci_f = Cs_inv @ f
        Bt = sp.bsr_transposed(B) if lhs._Bt is None else sp.bsr_copy(lhs._Bt)
        sp.bsr_mv(A=Bt, x=ci_f, y=u_rhs, alpha=-1.0, beta=1.0)

        BtCHiCt_inv = Bt
        wp.launch(
            bsr_mul_diag,
            dim=BtCHiCt_inv.nnz,
            inputs=[BtCHiCt_inv.values, BtCHiCt_inv.columns, CHiCt_inv],
        )

        sp.bsr_mm(
            x=BtCHiCt_inv,
            y=B,
            z=u_matrix,
            alpha=1.0,
            beta=1.0,
            work_arrays=work_arrays,
            reuse_topology=reuse_topology,
        )
        sp.bsr_mv(A=BtCHiCt_inv, x=c_k, y=u_rhs, alpha=-1.0, beta=1.0)

        delta_du = wp.zeros_like(u_rhs)
        err, niter = bsr_cg(
            u_matrix, b=u_rhs, x=delta_du, tol=tol, max_iters=max_iters, quiet=True
        )

        if np.isnan(err):
            raise RuntimeError(f"Solver fail, rhs= {np.linalg.norm(u_rhs.numpy())}")

        # other variable updates

        # get back lambda
        # -B du  -ChiC lambda = lambda_k

        lambda_rhs = wp.clone(c_k, requires_grad=False)
        sp.bsr_mv(A=B, x=delta_du, y=lambda_rhs, alpha=1.0, beta=1.0)

        dLambda = wp.empty_like(lambda_rhs)
        dS = wp.empty_like(f)
        dR = wp.empty_like(w_lambda)

        wp.launch(
            kernel=compute_dLambdadS,
            dim=Cs_inv.values.shape[0],
            inputs=[
                Cs_inv.values,
                CHiCt_inv,
                lambda_rhs,
                ci_f,
                dLambda,
                dS,
            ],
        )

        return delta_du, dS, dR, dLambda


@wp.func
def invert_schur_block(M: Any):
    eps = type(M[0])(M.dtype(1.0e-16))
    return inverse_qr(M + wp.diag(eps))


@wp.kernel
def compute_first_schur(
    CHiC_inv: wp.array(dtype=Any),
    Cs: wp.array(dtype=Any),
    Cr: wp.array(dtype=Any),
    H_inv: wp.array(dtype=Any),
    W_inv: wp.array(dtype=Any),
    lambda_rhs: wp.array(dtype=Any),
    f: wp.array(dtype=Any),
    w_lambda: wp.array(dtype=Any),
):
    i = wp.tid()

    cr = Cr[i]
    cs = Cs[i]

    csHi = cs * H_inv[i]
    crWi = cr * W_inv[i]

    lambda_rhs[i] += csHi * f[i] + crWi * w_lambda[i]

    CHiC = csHi * wp.transpose(cs) + crWi * wp.transpose(cr)
    CHiC_inv[i] = invert_schur_block(CHiC)


@wp.kernel
def compute_dLambdadRdS(
    Cs: wp.array(dtype=Any),
    Cr: wp.array(dtype=Any),
    C_inv: wp.array(dtype=Any),
    H_inv: wp.array(dtype=Any),
    W_inv: wp.array(dtype=Any),
    lambda_rhs: wp.array(dtype=Any),
    f: wp.array(dtype=Any),
    w_lambda: wp.array(dtype=Any),
    dLambda: wp.array(dtype=Any),
    dS: wp.array(dtype=Any),
    dR: wp.array(dtype=Any),
):
    i = wp.tid()
    dL = -C_inv[i] * lambda_rhs[i]
    dLambda[i] = dL
    dS[i] = -H_inv[i] * (f[i] + wp.transpose(Cs[i]) * dL)
    dR[i] = -W_inv[i] * (w_lambda[i] + wp.transpose(Cr[i]) * dL)


@wp.kernel
def compute_first_schur_no_R(
    CHiC_inv: wp.array(dtype=Any),
    Csi: wp.array(dtype=Any),
    H: wp.array(dtype=Any),
):
    i = wp.tid()

    CHiC_inv[i] = Csi[i] * H[i] * Csi[i]


@wp.kernel
def compute_dLambdadS(
    Csi: wp.array(dtype=Any),
    CHiC_inv: wp.array(dtype=Any),
    lambda_rhs: wp.array(dtype=Any),
    ci_f: wp.array(dtype=Any),
    dLambda: wp.array(dtype=Any),
    dS: wp.array(dtype=Any),
):
    i = wp.tid()

    dLambda[i] = -CHiC_inv[i] * lambda_rhs[i] - ci_f[i]
    dS[i] = Csi[i] * lambda_rhs[i]


@wp.kernel
def invert_blocks(A: wp.array(dtype=Any), A_inv: wp.array(dtype=Any)):
    i = wp.tid()
    A_inv[i] = inverse_qr(A[i])


@wp.kernel
def invert_schur_blocks(values: wp.array(dtype=Any)):
    i = wp.tid()

    values[i] = invert_schur_block(values[i])


@wp.kernel
def bsr_mul_diag(
    Bt_values: wp.array(dtype=Any),
    Bt_columns: wp.array(dtype=int),
    C_values: wp.array(dtype=Any),
):
    i = wp.tid()
    col = Bt_columns[i]
    Bt_values[i] *= C_values[col]
