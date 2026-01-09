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

import warp as wp
import math

__all__ = [
    "hooke_energy",
    "hooke_stress",
    "hooke_hessian",
    "nh_energy",
    "nh_stress",
    "nh_hessian_proj",
    "nh_hessian_proj_analytic",
    "snh_energy",
    "snh_stress",
    "snh_hessian_proj",
    "snh_hessian_proj_analytic",
]

_SQRT_1_2 = wp.constant(math.sqrt(1.0 / 2.0))


@wp.func
def hooke_stress(S: wp.mat33, lame: wp.vec2):
    strain = S - wp.identity(n=3, dtype=float)
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(
        n=3, dtype=float
    )


@wp.func
def hooke_energy(S: wp.mat33, lame: wp.vec2):
    strain = S - wp.identity(n=3, dtype=float)
    return 0.5 * wp.ddot(strain, hooke_stress(S, lame))


@wp.func
def hooke_hessian(S: wp.mat33, tau: wp.mat33, sig: wp.mat33, lame: wp.vec2):
    return wp.ddot(hooke_stress(sig + wp.identity(n=3, dtype=float), lame), tau)


# Neo-Hookean -- Eq (13) from Smith et al paper
#


@wp.func
def nh_parameters_from_lame(lame: wp.vec2):
    """Parameters such that for small strains model behaves according to Hooke's law"""
    mu_nh = lame[1]
    lambda_nh = lame[0] + lame[1]

    return mu_nh, lambda_nh


@wp.func
def nh_energy(F: wp.mat33, lame: wp.vec2):
    J = wp.determinant(F)
    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    gamma = 1.0 + mu_nh / lambda_nh

    E0 = lambda_nh * (1.0 - gamma) * (1.0 - gamma) + mu_nh * 3.0

    return 0.5 * (lambda_nh * (J - gamma) * (J - gamma) + mu_nh * wp.ddot(F, F) - E0)


@wp.func
def nh_stress(F: wp.mat33, lame: wp.vec2):
    J = wp.determinant(F)
    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    gamma = 1.0 + mu_nh / lambda_nh

    return mu_nh * F + lambda_nh * (J - gamma) * _dJ_dF(F)


@wp.func
def nh_hessian_proj(F: wp.mat33, tau: wp.mat33, sig: wp.mat33, lame: wp.vec2):
    dJ_dF_s = _dJ_dF(F)

    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    dpsi_dpsi = mu_nh * wp.ddot(tau, sig) + lambda_nh * wp.ddot(dJ_dF_s, tau) * wp.ddot(
        dJ_dF_s, sig
    )

    # clamp the hessian of J so that it does not compritube eigenvalue smaller than - mu * F_scale
    J = wp.determinant(F)
    Ic = wp.ddot(F, F)
    gamma = 1.0 + mu_nh / lambda_nh
    muT = mu_nh
    d2J_scale = _d2J_dF2_scale(J, Ic, lambda_nh * (J - gamma), 0.99 * muT)
    # d2J_scale = lambda_nh * (J - gamma)
    # d2J_scale = 0.0

    return dpsi_dpsi + d2J_scale * _d2J_dF2(F, sig, tau)


@wp.func
def nh_hessian_proj_analytic(F: wp.mat33, tau: wp.mat33, sig: wp.mat33, lame: wp.vec2):
    mu_nh, lambda_nh = snh_parameters_from_lame(lame)

    J = wp.determinant(F)
    gamma = 1.0 + 0.75 * mu_nh / lambda_nh

    muT = mu_nh
    muL = 0.0
    lbdJ = lambda_nh * (J - gamma)

    return hessian_proj_analytic(F, muT, muL, lambda_nh, lbdJ, sig, tau)


# Stable Neo-Hookean -- complete model from Simat et al. with log(Ic + 1) term
#


@wp.func
def snh_parameters_from_lame(lame: wp.vec2):
    """Parameters such that for small strains model behaves according to Hooke's law"""
    mu_nh = 4.0 / 3.0 * lame[1]
    lambda_nh = lame[0] + 5.0 / 6.0 * lame[1]

    return mu_nh, lambda_nh


@wp.func
def snh_energy(F: wp.mat33, lame: wp.vec2):
    mu_nh, lambda_nh = snh_parameters_from_lame(lame)
    gamma = 1.0 + 0.75 * mu_nh / lambda_nh

    J = wp.determinant(F)
    Ic = wp.ddot(F, F)

    E0 = lambda_nh * (1.0 - gamma) * (1.0 - gamma) + mu_nh * (3.0 - wp.log(4.0))

    return 0.5 * (
        lambda_nh * (J - gamma) * (J - gamma) + mu_nh * (Ic - wp.log(Ic + 1.0)) - E0
    )


@wp.func
def snh_stress(F: wp.mat33, lame: wp.vec2):
    J = wp.determinant(F)
    mu_nh, lambda_nh = snh_parameters_from_lame(lame)
    gamma = 1.0 + 0.75 * mu_nh / lambda_nh

    Ic = wp.ddot(F, F)
    F_scale = 1.0 - 1.0 / (Ic + 1.0)
    return mu_nh * F * F_scale + lambda_nh * (J - gamma) * _dJ_dF(F)


@wp.func
def snh_hessian_proj(F: wp.mat33, tau: wp.mat33, sig: wp.mat33, lame: wp.vec2):
    dJ_dF_s = _dJ_dF(F)

    mu_nh, lambda_nh = snh_parameters_from_lame(lame)

    Ic = wp.ddot(F, F)

    F_scale = 1.0 - 1.0 / (Ic + 1.0)

    dpsi_dpsi = (
        mu_nh * F_scale * wp.ddot(tau, sig)
        + 2.0 * mu_nh / ((Ic + 1.0) * (Ic + 1.0)) * wp.ddot(F, tau) * wp.ddot(F, sig)
        + lambda_nh * wp.ddot(dJ_dF_s, tau) * wp.ddot(dJ_dF_s, sig)
    )

    # clamp the hessian of J so that it does not compritube eigenvalue smaller than - mu * F_scale
    J = wp.determinant(F)
    gamma = 1.0 + 0.75 * mu_nh / lambda_nh
    muT = mu_nh * F_scale
    d2J_scale = _d2J_dF2_scale(J, Ic, lambda_nh * (J - gamma), 0.99 * muT)
    # d2J_scale = lambda_nh * (J - gamma)
    # d2J_scale = 0.0

    return dpsi_dpsi + d2J_scale * _d2J_dF2(F, sig, tau)


@wp.func
def snh_hessian_proj_analytic(F: wp.mat33, tau: wp.mat33, sig: wp.mat33, lame: wp.vec2):
    mu_nh, lambda_nh = snh_parameters_from_lame(lame)

    Ic = wp.ddot(F, F)
    J = wp.determinant(F)
    gamma = 1.0 + 0.75 * mu_nh / lambda_nh

    muT = mu_nh * (1.0 - 1.0 / (Ic + 1.0))
    muL = 2.0 * mu_nh / ((Ic + 1.0) * (Ic + 1.0))
    lbdJ = lambda_nh * (J - gamma)

    return hessian_proj_analytic(F, muT, muL, lambda_nh, lbdJ, sig, tau)


# Utilities


@wp.func
def hessian_proj_analytic(
    F: wp.mat33,
    muT: float,
    muL: float,
    lbd: float,
    lbdJ: float,
    sig: wp.mat33,
    tau: wp.mat33,
):
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, S, V)

    # Solve eigensystem on principal stresses -- analytical is ugly
    # (and formula (44) for eigenvectors does not seem to yield the correct result)
    Scross = wp.vec3(S[1] * S[2], S[0] * S[2], S[0] * S[1])
    Soff = wp.mat33(0.0, S[2], S[1], S[2], 0.0, S[0], S[1], S[0], 0.0)
    A = (
        muT * wp.identity(n=3, dtype=float)
        + muL * wp.outer(S, S)
        + lbd * wp.outer(Scross, Scross)
        + lbdJ * Soff
    )

    Q = wp.mat33()
    d = wp.vec3()
    wp.eig3(A, Q, d)
    Qt = wp.transpose(Q)

    # d, Qt = fem.utils.symmetric_eigenvalues_qr(A, 1.0e-16)

    # accumulate eigenvectors corresponding to positive eigenvalues
    tauU = wp.transpose(U) * tau
    sigU = wp.transpose(U) * sig
    Vt = wp.transpose(V)

    res = float(0.0)
    clamp = 0.0

    for k in range(3):
        Pk = wp.diag(Qt[k]) * Vt
        res += wp.max(clamp, d[k]) * wp.ddot(tauU, Pk) * wp.ddot(sigU, Pk)

        Pk = _flip_rot_eivec(k, 1.0, Vt)
        res += wp.max(clamp, muT + lbdJ * S[k]) * wp.ddot(tauU, Pk) * wp.ddot(sigU, Pk)

        Pk = _flip_rot_eivec(k, -1.0, Vt)
        res += wp.max(clamp, muT - lbdJ * S[k]) * wp.ddot(tauU, Pk) * wp.ddot(sigU, Pk)

    return res


@wp.func
def _flip_rot_eivec(k: int, sign: float, mat: wp.mat33):
    E = wp.mat33(0.0)
    k2 = (k + 2) % 3
    k1 = (k + 1) % 3
    E[k1] = _SQRT_1_2 * mat[k2]
    E[k2] = -sign * _SQRT_1_2 * mat[k1]
    return E


@wp.func
def _dJ_dF(F: wp.mat33):
    Ft = wp.transpose(F)
    return wp.mat33(
        wp.cross(Ft[1], Ft[2]), wp.cross(Ft[2], Ft[0]), wp.cross(Ft[0], Ft[1])
    )


@wp.func
def _d2J_dF2(F: wp.mat33, sig: wp.mat33, tau: wp.mat33):
    Ft = wp.transpose(F)
    sigt = wp.transpose(sig)
    return wp.ddot(
        tau,
        wp.mat33(
            wp.cross(Ft[1], sigt[2]) + wp.cross(sigt[1], Ft[2]),
            wp.cross(Ft[2], sigt[0]) + wp.cross(sigt[2], Ft[0]),
            wp.cross(Ft[0], sigt[1]) + wp.cross(sigt[0], Ft[1]),
        ),
    )


@wp.func
def _d2J_dF2_scale(J: float, Ic: float, J_scale: float, Id_scale: float):
    # compute a scaling for d2J such that Id_scale * Id + J_scale * d2J
    # has no negative eigenvalues

    # Min/max eigenvalues for d2J are estimated according to
    # sec 4.5 of "Stable Neo-Hookean Flesh Simulation" (Smith et al. 2018)

    d2J_ev = _depressed_cubic_roots(-Ic, -2.0 * J)
    sig_max = wp.sqrt(Ic)

    ev_min = wp.min(wp.min(d2J_ev), -sig_max)
    ev_max = wp.max(wp.max(d2J_ev), sig_max)
    return wp.clamp(J_scale, -Id_scale / ev_max, -Id_scale / ev_min)


@wp.func
def _depressed_cubic_roots(p: float, q: float):
    alpha = wp.sqrt(-p / 3.0)
    beta = wp.acos(1.5 * q / (p * alpha)) / 3.0
    return (
        2.0
        * alpha
        * wp.vec3(
            wp.cos(beta),
            wp.cos(beta - 2.0 / 3.0 * wp.pi),
            wp.cos(beta - 4.0 / 3.0 * wp.pi),
        )
    )


@wp.func
def symmetric_strain(F: wp.mat33):
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)

    S = V * wp.diag(sig) * wp.transpose(V)

    return S


@wp.func
def symmetric_strain_delta(F: wp.mat33, dF: wp.mat33):
    # see supplementary of `WRAPD: Weighted Rotation-aware ADMM`, Brown and Narain 21

    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)

    Ut = wp.transpose(U)
    Vt = wp.transpose(V)

    dF_loc = Ut * dF * V
    SigdF_loc = wp.diag(sig) * dF_loc

    sig_op = wp.matrix_from_cols(wp.vec3(sig[0]), wp.vec3(sig[1]), wp.vec3(sig[2]))
    dSig = wp.cw_div(SigdF_loc + wp.transpose(SigdF_loc), sig_op + wp.transpose(sig_op))
    dS = V * dSig * Vt

    return dS
