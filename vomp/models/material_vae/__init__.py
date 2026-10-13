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

from .beta_tc import TripletVAE
from .standard_vae import StandardVAE


def from_config(config):
    """Build the MatVAE class explicitly selected by its JSON name and args."""
    classes = {"StandardVAE": StandardVAE, "TripletVAE": TripletVAE,
               "matvae": TripletVAE}
    name = config.get("name")
    if name not in classes:
        raise ValueError(
            f"models.matvae.name must be StandardVAE or TripletVAE, got {name!r}"
        )
    return classes[name](**config["args"])
