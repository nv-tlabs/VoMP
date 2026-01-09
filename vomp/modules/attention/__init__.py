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

BACKEND = "flash_attn"
DEBUG = False


def __from_env():
    import os

    global BACKEND
    global DEBUG

    env_attn_backend = os.environ.get("ATTN_BACKEND")
    env_sttn_debug = os.environ.get("ATTN_DEBUG")

    if env_attn_backend is not None and env_attn_backend in [
        "xformers",
        "flash_attn",
        "sdpa",
        "naive",
    ]:
        BACKEND = env_attn_backend
    if env_sttn_debug is not None:
        DEBUG = env_sttn_debug == "1"

    print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()


def set_backend(backend: Literal["xformers", "flash_attn"]):
    global BACKEND
    BACKEND = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .full_attn import *
from .modules import *
