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
from pathlib import Path

import torch
import torch.multiprocessing as mp
import numpy as np
import random
from vomp.training.checkpoints import find_normalization_params
from vomp.utils.dist_utils import setup_dist
from safetensors.torch import load_file
from vomp.models.geometry_encoder import ElasticGeometryEncoder, ElasticSLatVoxelDecoder
from vomp.models import material_vae
from vomp.trainers.vae.slat_materials import SLatVaeMaterialsTrainer
from vomp.datasets import SparseVoxelMaterials
from easydict import EasyDict as edict


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


def dataset_arguments(config, normalization_params_file):
    """Apply configured archive handling and disable evaluation augmentation."""
    common = dict(config["dataset"]["args"])
    common["normalization_params_file"] = normalization_params_file
    if "normalization_type" in config["dataset"]:
        common["normalization_type"] = config["dataset"]["normalization_type"]
    common.setdefault("archive_format", "benchmark")
    train_args = dict(common)
    train_args.setdefault("split", "train")
    val_args = dict(common, split="val", random_grid_rotation=False, feature_noise_std=0.0)
    return train_args, val_args


def validate_model_config(config):
    """Validate dimensions along the configured encoder/decoder path."""
    models = config["models"]
    encoder = models["geometry_encoder"]["args"]
    if config["dataset"]["args"]["resolution"] != encoder["resolution"]:
        raise ValueError("dataset and geometry encoder resolutions must match")
    mode = config.get("training_mode", "encoder_only")
    if mode == "encoder_only":
        if encoder["latent_channels"] != models["matvae"]["args"]["z_dim"]:
            raise ValueError("geometry encoder latent channels must match MatVAE z_dim")
    elif mode in {"encoder_decoder_matvae", "encoder_decoder_direct"}:
        decoder = models["decoder"]["args"]
        if decoder["latent_channels"] != encoder["latent_channels"]:
            raise ValueError("geometry encoder and decoder latent channels must match")
        output_channels = 3 if mode == "encoder_decoder_direct" else models["matvae"]["args"]["z_dim"]
        if decoder["out_channels"] != output_channels:
            raise ValueError("decoder output channels must match its material prediction target")
    else:
        raise ValueError(f"Invalid training_mode: {mode}")


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    validate_model_config(cfg)

    normalization_params_file = find_normalization_params(cfg)
    train_dataset_args, val_dataset_args = dataset_arguments(cfg, normalization_params_file)
    dataset = SparseVoxelMaterials(**train_dataset_args)
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
    matvae = material_vae.from_config(cfg.models.matvae).cuda()

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


def cli(argv=None):
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument(
        "--config", type=str, required=True, help="Experiment configuration"
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
        "--data_dir", type=str, default=None, help="Override dataset roots (comma-separated)"
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
    opt = parser.parse_args(argv)
    opt.load_dir = opt.load_dir if opt.load_dir != "" else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    path = Path(opt.config)
    config = json.loads(path.read_text())
    if opt.data_dir is not None:
        config["dataset"]["args"]["roots"] = opt.data_dir
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
                if rty + 1 == cfg.auto_retry:
                    raise
                print(f"Retrying ({rty + 1}/{cfg.auto_retry})...")


if __name__ == "__main__":
    cli()
