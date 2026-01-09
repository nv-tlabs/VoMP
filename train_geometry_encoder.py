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

import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random
from safetensors.torch import load_file

from vomp.datasets.sparse_voxel_materials import SparseVoxelMaterials
from vomp.models.geometry_encoder import ElasticGeometryEncoder, ElasticSLatVoxelDecoder
from vomp.models.material_vae.standard_vae import StandardVAE
from vomp.models.material_vae.beta_tc import TripletVAE
from vomp.trainers.vae.slat_materials import SLatVaeMaterialsTrainer
from vomp.utils.dist_utils import setup_dist


def find_ckpt(cfg):
    # Load checkpoint
    cfg["load_ckpt"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":
            files = glob.glob(os.path.join(cfg.load_dir, "ckpts", "misc_*.pt"))
            if len(files) != 0:
                cfg.load_ckpt = max(
                    [
                        int(os.path.basename(f).split("step")[-1].split(".")[0])
                        for f in files
                    ]
                )
        elif cfg.ckpt == "none":
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = "Parameters:\n"
    model_summary += "=" * 128 + "\n"
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f"{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n"
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += "\n"
    model_summary += f"Number of parameters: {num_params}\n"
    model_summary += f"Number of trainable parameters: {num_trainable_params}\n"
    return model_summary


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Determine normalization parameters file path
    normalization_params_file = None
    if cfg.get("matvae_checkpoint") is not None:
        # Extract the directory containing the matvae checkpoint
        import os

        matvae_checkpoint_path = cfg.matvae_checkpoint

        # Navigate to the project directory (up from checkpoints/checkpoint_X/)
        if "checkpoints" in matvae_checkpoint_path:
            # For paths like: outputs/matvae/checkpoints/checkpoint_821/model.safetensors
            # Need to go up 3 levels: model.safetensors -> checkpoint_X -> checkpoints -> project_dir
            matvae_project_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(matvae_checkpoint_path))
            )
            normalization_params_file = os.path.join(
                matvae_project_dir, "normalization_params.json"
            )
        else:
            # For other checkpoint path formats, try to find the normalization params
            checkpoint_dir = os.path.dirname(matvae_checkpoint_path)
            normalization_params_file = os.path.join(
                checkpoint_dir, "normalization_params.json"
            )

        if os.path.exists(normalization_params_file):
            if rank == 0:
                print(
                    f"Found normalization parameters file: {normalization_params_file}"
                )
        else:
            if rank == 0:
                print(
                    f"ERROR: Could not find normalization parameters file at {normalization_params_file}"
                )
                print(
                    "This file is required for consistent normalization between matvae and geometry encoder training!"
                )
                print(
                    "Make sure the matvae was trained with the updated train_material_vae.py that saves normalization_params.json"
                )
            raise FileNotFoundError(
                f"Required normalization parameters file not found: {normalization_params_file}"
            )
    else:
        if rank == 0:
            print("ERROR: No matvae_checkpoint specified in config!")
            print(
                "The matvae_checkpoint is required to locate the normalization parameters file."
            )
        raise ValueError(
            "matvae_checkpoint must be specified in config to load normalization parameters"
        )

    # Load data
    train_dataset_args = cfg.dataset.args.copy()
    train_dataset_args["split"] = "train"

    # Add normalization type if specified in config
    if "normalization_type" in cfg.dataset:
        train_dataset_args["normalization_type"] = cfg.dataset.normalization_type
        if rank == 0:
            print(
                f"Using material normalization type: {cfg.dataset.normalization_type}"
            )

    # Add normalization parameters file (now mandatory)
    train_dataset_args["normalization_params_file"] = normalization_params_file

    dataset = SparseVoxelMaterials(**train_dataset_args)

    # Create validation dataset
    val_dataset_args = cfg.dataset.args.copy()
    val_dataset_args["split"] = "val"

    # Add normalization type if specified in config
    if "normalization_type" in cfg.dataset:
        val_dataset_args["normalization_type"] = cfg.dataset.normalization_type

    # Add normalization parameters file (now mandatory)
    val_dataset_args["normalization_params_file"] = normalization_params_file

    val_dataset = SparseVoxelMaterials(**val_dataset_args)

    # Build trainable models (only geometry encoder)
    model_dict = {
        "geometry_encoder": (
            ElasticGeometryEncoder(**cfg.models.geometry_encoder.args).cuda()
        ),
    }

    # Add decoder if specified in config
    if "decoder" in cfg.models:
        model_dict["decoder"] = ElasticSLatVoxelDecoder(
            **cfg.models.decoder.args
        ).cuda()

    # Load frozen matvae model separately
    from vomp.models.material_vae.beta_tc import TripletVAE

    matvae = TripletVAE(**cfg.models.matvae.args).cuda()

    # Load matvae checkpoint if provided
    if cfg.get("matvae_checkpoint") is not None:
        if rank == 0:
            print(f"Loading matvae checkpoint from: {cfg.matvae_checkpoint}")

        # Load safetensors checkpoint
        checkpoint = load_file(cfg.matvae_checkpoint)
        matvae.load_state_dict(checkpoint, strict=True)

        if rank == 0:
            print("Successfully loaded matvae checkpoint")

    matvae.eval()
    for param in matvae.parameters():
        param.requires_grad = False

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f"\n\nBackbone: {name}\n" + model_summary)
            with open(
                os.path.join(cfg.output_dir, f"{name}_model_summary.txt"), "w"
            ) as fp:
                print(model_summary, file=fp)

        # Also print matvae summary
        matvae_summary = get_model_summary(matvae)
        print(f"\n\nFrozen Model: matvae\n" + matvae_summary)
        with open(os.path.join(cfg.output_dir, "matvae_model_summary.txt"), "w") as fp:
            print(matvae_summary, file=fp)

    # Build trainer
    trainer = SLatVaeMaterialsTrainer(
        model_dict,
        dataset,
        matvae=matvae,  # Pass matvae as a separate argument
        val_dataset=val_dataset,  # Pass validation dataset
        trellis_weights_path=cfg.get(
            "trellis_weights_path"
        ),  # Pass TRELLIS weights path from config
        training_mode=cfg.get("training_mode", "encoder_only"),  # Add training mode
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt,
    )

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == "__main__":
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument(
        "--config", type=str, required=True, help="Experiment config file"
    )
    ## io and resume
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--load_dir", type=str, default="", help="Load directory, default to output_dir"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latest",
        help="Checkpoint step to resume training, default to latest",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/", help="Data directory"
    )
    parser.add_argument(
        "--auto_retry", type=int, default=3, help="Number of retries on error"
    )
    ## dubug
    parser.add_argument(
        "--tryrun", action="store_true", help="Try run without training"
    )
    parser.add_argument("--profile", action="store_true", help="Profile training")
    ## multi-node and multi-gpu
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="Number of GPUs per node, default to all",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master address for distributed training",
    )
    parser.add_argument(
        "--master_port", type=str, default="12345", help="Port for distributed training"
    )
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != "" else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    config = json.load(open(opt.config, "r"))
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print("\n\nConfig:")
    print("=" * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, "command.txt"), "w") as fp:
            print(" ".join(["python"] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying ({rty + 1}/{cfg.auto_retry})...")
