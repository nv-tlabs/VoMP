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

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class MaterialProperties:
    """
    Container for material properties predictions.

    Args:
        youngs_modulus: Young's modulus values per voxel
        poissons_ratio: Poisson's ratio values per voxel
        density: Density values per voxel
        coords: The coordinates of the voxels in the grid
    """

    def __init__(
        self,
        youngs_modulus: torch.Tensor,
        poissons_ratio: torch.Tensor,
        density: torch.Tensor,
        coords: torch.Tensor,
    ):
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio
        self.density = density
        self.coords = coords

    @property
    def device(self):
        return self.youngs_modulus.device


class MaterialClassification:
    """
    Container for material classification predictions.

    Args:
        material_logits: Logits for material class prediction per voxel
        coords: The coordinates of the voxels in the grid
    """

    def __init__(
        self,
        material_logits: torch.Tensor,
        coords: torch.Tensor,
    ):
        self.material_logits = material_logits
        self.coords = coords

    @property
    def device(self):
        return self.material_logits.device

    @property
    def material_classes(self):
        return torch.argmax(self.material_logits, dim=1)

    @property
    def material_probs(self):
        return F.softmax(self.material_logits, dim=1)


class SLatMaterialDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        num_classes: int = 18,  # Number of material classes for classification mode
        mode: Literal["regression", "classification"] = "regression",
    ):
        print("Creating SLatMaterialDecoder with:")
        print(f"  - Resolution: {resolution}")
        print(f"  - Model channels: {model_channels}")
        print(f"  - Latent channels: {latent_channels}")
        print(f"  - Num blocks: {num_blocks}")
        print(f"  - Attention mode: {attn_mode}")
        print(f"  - Number of material classes: {num_classes}")

        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.num_classes = num_classes
        self.mode = mode

        if self.mode == "regression":
            # Output layer for regression mode (material properties)
            self.out_channels = 3
            self.out_layer = sp.SparseLinear(model_channels, self.out_channels)
        else:
            # Output layer for classification mode (material class)
            self.classification_layer = sp.SparseLinear(model_channels, num_classes)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
            print("  - Using FP16 precision")
        print("Decoder created for material properties prediction.")

    def initialize_weights(self) -> None:
        super().initialize_weights()

        if self.mode == "regression":
            nn.init.xavier_uniform_(self.out_layer.weight)
            nn.init.constant_(self.out_layer.bias, 0)
        else:
            nn.init.xavier_uniform_(self.classification_layer.weight)
            nn.init.constant_(self.classification_layer.bias, 0)

    def to_material_properties(self, x: sp.SparseTensor) -> List[MaterialProperties]:
        """
        Convert a batch of network outputs to material properties.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of MaterialProperties objects for each item in the batch
        """
        ret = []
        for i in range(x.shape[0]):
            feats = x.feats[x.layout[i]]
            coords = x.coords[x.layout[i]][:, 1:]  # remove batch dimension

            # Raw outputs - no activation functions
            # The model learns to predict in the normalized transformed space
            youngs_modulus = feats[:, 0]
            poissons_ratio = feats[:, 1]
            density = feats[:, 2]

            representation = MaterialProperties(
                youngs_modulus=youngs_modulus,
                poissons_ratio=poissons_ratio,
                density=density,
                coords=coords,
            )
            ret.append(representation)

        return ret

    def to_material_classification(
        self, x: sp.SparseTensor
    ) -> List[MaterialClassification]:
        """
        Convert a batch of network outputs to material classification.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of MaterialClassification objects for each item in the batch
        """
        ret = []
        for i in range(x.shape[0]):
            feats = x.feats[x.layout[i]]
            coords = x.coords[x.layout[i]][:, 1:]  # remove batch dimension

            # Material classification logits
            material_logits = feats  # Already the logits, no activation

            representation = MaterialClassification(
                material_logits=material_logits,
                coords=coords,
            )
            ret.append(representation)

        return ret

    def forward(
        self, x: sp.SparseTensor
    ) -> Union[List[MaterialProperties], List[MaterialClassification]]:
        """
        Forward pass through the decoder.

        Args:
            x: The input sparse tensor.

        Returns:
            List of either MaterialProperties or MaterialClassification objects, depending on mode.
        """
        h = super().forward(x)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))

        if self.mode == "classification":
            h_class = self.classification_layer(h)
            return self.to_material_classification(h_class)
        else:  # default to regression mode
            h_reg = self.out_layer(h)
            return self.to_material_properties(h_reg)


class ElasticSLatMaterialDecoder(SparseTransformerElasticMixin, SLatMaterialDecoder):
    """
    SLat material decoder with elastic memory management.
    Used for training with low VRAM.
    """

    pass
