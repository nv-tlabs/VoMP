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

from .. import BACKEND

SPCONV_ALGO = "auto"  # 'auto', 'implicit_gemm', 'native'


def __from_env():
    import os

    global SPCONV_ALGO
    env_spconv_algo = os.environ.get("SPCONV_ALGO")
    if env_spconv_algo is not None and env_spconv_algo in [
        "auto",
        "implicit_gemm",
        "native",
    ]:
        SPCONV_ALGO = env_spconv_algo
    print(f"[SPARSE][CONV] spconv algo: {SPCONV_ALGO}")


__from_env()

if BACKEND == "torchsparse":
    from .conv_torchsparse import *
elif BACKEND == "spconv":
    from .conv_spconv import *
