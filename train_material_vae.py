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

"""Train a material VAE using a JSON configuration."""

import argparse
import json
from pathlib import Path

from vomp.training.material_vae import train
from vomp.utils.material_vae_normalization import (
    physics_aware_inverse_transform,
    forward_transform,
    inverse_transform,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the material VAE JSON configuration",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate and disable its scheduler",
    )
    parser.add_argument("--project_dir", default=None)
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=None,
        help="Override KL weight and disable KL annealing",
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
        config["lr_scheduler"] = None
    if args.project_dir is not None:
        config["project_dir"] = args.project_dir
    if args.kl_weight is not None:
        config["kl_weight"] = args.kl_weight
        config["kl_annealing"] = False
        config["kl_schedule"] = "constant"
    train(config)


if __name__ == "__main__":
    main()
