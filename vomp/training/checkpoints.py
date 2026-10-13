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

"""Locate normalization artifacts paired with material VAE checkpoints."""

from pathlib import Path


def find_normalization_params(config):
    """Locate the normalization artifact paired with the frozen MatVAE."""
    explicit = config.get("normalization_params_path")
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        checkpoint = config.get("matvae_checkpoint")
        if not checkpoint:
            raise ValueError("matvae_checkpoint must be specified in config")
        checkpoint = Path(checkpoint)
        candidates = [checkpoint.parent / "normalization_params.json"]
        for parent in checkpoint.parents:
            if parent.name == "checkpoints":
                candidates.append(parent.parent / "normalization_params.json")
                break
        if checkpoint.parent.name == "final_model":
            candidates.append(checkpoint.parent.parent / "normalization_params.json")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Required MatVAE normalization parameters not found: {searched}")
