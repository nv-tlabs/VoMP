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
import torch.nn as nn
from .. import SparseTensor


class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super(SparseConv3d, self).__init__()
        if "torchsparse" not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, 0, dilation, bias
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(
            out,
            shape=torch.Size(new_shape),
            layout=x.layout if all(s == 1 for s in self.conv.stride) else None,
        )
        out._spatial_cache = x._spatial_cache
        out._scale = tuple(
            [s * stride for s, stride in zip(x._scale, self.conv.stride)]
        )
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super(SparseInverseConv3d, self).__init__()
        if "torchsparse" not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            bias,
            transposed=True,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(
            out,
            shape=torch.Size(new_shape),
            layout=x.layout if all(s == 1 for s in self.conv.stride) else None,
        )
        out._spatial_cache = x._spatial_cache
        out._scale = tuple(
            [s // stride for s, stride in zip(x._scale, self.conv.stride)]
        )
        return out
