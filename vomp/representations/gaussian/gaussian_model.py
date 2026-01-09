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

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
import utils3d
import kaolin


def transform_shs(shs_feat, rotation_matrix):
    try:
        # Solution from: https://github.com/graphdeco-inria/gaussian-splatting/issues/176
        from e3nn import o3

        ## rotate shs
        device = shs_feat.device
        P = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            dtype=rotation_matrix.dtype,
            device=rotation_matrix.device,
        )  # switch axes: yzx -> xyz
        permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
        rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())
        # Construction coefficient
        D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2]).to(device)
        D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2]).to(device)
        D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2]).to(device)

        # rotation of the shs features
        res = torch.zeros_like(shs_feat)
        res[:, :3] = D_1 @ shs_feat[:, :3]
        res[:, 3:8] = D_2 @ shs_feat[:, 3:8]
        res[:, 8:15] = D_3 @ shs_feat[:, 8:15]
        return res
    except Exception as e:
        logger.error(f"Failed to transform sh features with error: {e}")
        return shs_feat


def transform_xyz(xyz: torch.Tensor, se_transform: torch.Tensor):
    res = (
        se_transform[None, :3, :3] @ xyz[:, :, None] + se_transform[None, :3, 3:]
    ).squeeze(-1)
    return res


def transform_rot(rot: torch.Tensor, transform: torch.Tensor):
    rot_quat = kaolin.math.quat.quat_from_rot33(transform[:3, :3].unsqueeze(0))
    rot_unit = rot / torch.linalg.norm(rot, dim=-1).unsqueeze(-1)

    # Note: gsplats use Hamiltonion convention [real, imag], whereas Kaolin uses the other convention[imag, real]
    rot_unit = torch.cat([rot_unit[:, 1:], rot_unit[:, :1]], dim=-1)

    result = kaolin.math.quat.quat_mul(rot_quat, rot_unit)
    result = torch.cat([result[:, 3:], result[:, :3]], dim=-1)
    return result


def decompose_4x4_transform(transform):
    """Decompose 4x4 transform into translation, rotation, scale.
    Returns:
        translation, rotation, scale
    """
    translation = transform[:3, 3:]
    scale = torch.linalg.norm(transform[:3, :3], dim=0)
    rotation = transform[:3, :3] / scale.unsqueeze(0)
    return translation, rotation, scale


def transform_gaussians(
    xyz, rotations, raw_scales, transform, shs_feat=None, use_log_scales=True
):
    translation, rotation, scale = decompose_4x4_transform(transform)

    new_xyz = transform_xyz(xyz, transform)
    new_rotations = transform_rot(rotations, rotation)

    if not use_log_scales:
        new_scales = raw_scales * scale.unsqueeze(0)
    else:
        scaling_norm_factor = torch.log(scale).unsqueeze(0) / raw_scales + 1
        new_scales = raw_scales * scaling_norm_factor

    if shs_feat is None:
        return new_xyz, new_rotations, new_scales
    new_shs_feat = transform_shs(shs_feat, transform[:3, :3])
    return new_xyz, new_rotations, new_scales, new_shs_feat


class Gaussian:
    def __init__(
        self,
        aabb: list,
        sh_degree: int = 0,
        mininum_kernel_size: float = 0.0,
        scaling_bias: float = 0.01,
        opacity_bias: float = 0.1,
        scaling_activation: str = "exp",
        device="cuda",
    ):
        self.init_params = {
            "aabb": aabb,
            "sh_degree": sh_degree,
            "mininum_kernel_size": mininum_kernel_size,
            "scaling_bias": scaling_bias,
            "opacity_bias": opacity_bias,
            "scaling_activation": scaling_activation,
        }

        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.scale_bias = self.inverse_scaling_activation(
            torch.tensor(self.scaling_bias)
        ).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(
            torch.tensor(self.opacity_bias)
        ).cuda()

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size**2
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])

    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]

    @property
    def get_features(self):
        return (
            torch.cat((self._features_dc, self._features_rest), dim=1)
            if self._features_rest is not None
            else self._features_dc
        )

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :]
        )

    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias

    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]

    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]

    def from_features(self, features):
        self._features_dc = features

    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()

        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(transform, rotation)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1)
            )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if transform is not None:
            _transform = torch.eye(4, dtype=torch.float)
            _transform[:3, :3] = torch.tensor(transform, dtype=torch.float)
            xyz = torch.tensor(xyz)
            rots = torch.tensor(rots)
            scales = torch.tensor(scales)
            xyz, rots, scales = transform_gaussians(xyz, rots, scales, _transform)
            # transform = np.array(transform).astype(float)
            # xyz = np.matmul(xyz, transform)
            # rots = transform_rot(torch.tensor(rots), torch.tensor(transform)).numpy()
            # rots = utils3d.numpy.quaternion_to_matrix(rots)
            # rots = np.matmul(rots, transform)
            # rots = utils3d.numpy.matrix_to_quaternion(rots)

        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
        )
        if self.sh_degree > 0:
            features_extra = (
                torch.tensor(features_extra, dtype=torch.float, device=self.device)
                .transpose(1, 2)
                .contiguous()
            )
        opacities = torch.sigmoid(
            torch.tensor(opacities, dtype=torch.float, device=self.device)
        )
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)

        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = (
            self.inverse_scaling_activation(
                torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
            )
            - self.scale_bias
        )
        self._rotation = rots - self.rots_bias[None, :]
