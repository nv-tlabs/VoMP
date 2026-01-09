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

from contextlib import contextmanager
from typing import *
import math
from ..modules import sparse as sp
from ..utils.elastic_utils import ElasticModuleMixin


class SparseTransformerElasticMixin(ElasticModuleMixin):
    def _get_input_size(self, x: sp.SparseTensor, *args, **kwargs):
        return x.feats.shape[0]

    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        if mem_ratio == 1.0:
            yield 1.0
            return
        num_blocks = len(self.blocks)
        num_checkpoint_blocks = min(
            math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks
        )
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = i < num_checkpoint_blocks
        yield exact_mem_ratio
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = False
