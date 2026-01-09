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

"""
Main Vomp model class for material property inference.

This module contains the core Vomp model that combines geometry encoding,
feature extraction, and material property prediction for 3D objects.
"""

import os
import json
import glob
import math
from typing import Dict, Any, Optional, Union, Tuple, List, Protocol
from subprocess import DEVNULL, call

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from PIL import Image
from safetensors.torch import load_file
from easydict import EasyDict as edict
from scipy.spatial.distance import cdist
from vomp.inference.ply_utils import write_ply_vertices, read_ply_vertices
import utils3d
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import kaolin.ops.conversions.gaussians as gs_ops

from vomp.models.geometry_encoder import (
    ElasticGeometryEncoder,
    ElasticSLatVoxelDecoder,
)
from vomp.models.material_vae.beta_tc import TripletVAE
from vomp.utils.data_utils import recursive_to_device
from vomp.modules.sparse import SparseTensor
from vomp.utils.material_transforms import MaterialPropertyTransform
from vomp.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
from vomp.inference.utils import (
    LazyLoadDino,
    MaterialUpsampler,
    get_mesh_transform_params,
    denormalize_coords,
)
from vomp.representations.gaussian import Gaussian
from vomp.inference.replicator_renderer import render_with_replicator
from vomp.inference.usd_utils import convert_usd_to_obj
import tempfile
import shutil
import multiprocessing
from functools import partial
from subprocess import PIPE, Popen


class RenderFunction(Protocol):
    """Protocol for custom rendering functions."""

    def __call__(
        self,
        obj_3d: Any,
        output_dir: str,
        num_views: int,
        image_size: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Render object and return frame metadata."""
        ...


class VoxelizeFunction(Protocol):
    """Protocol for custom voxelization functions."""

    def __call__(self, obj_3d: Any, output_dir: str, **kwargs: Any) -> np.ndarray:
        """Voxelize object and return voxel centers as (N, 3) array."""
        ...


class Vomp(nn.Module):
    """
    End-to-end material property inference model for 3D objects.

    Vomp combines geometry encoding, visual feature extraction via DINO,
    and material VAE to predict mechanical properties (Young's modulus, Poisson's
    ratio, density) from 3D representations including Gaussian splats and meshes.

    The model processes 3D objects through the following pipeline:
    1. Multi-view rendering from sampled camera positions
    2. Voxelization of the 3D geometry
    3. DINO feature extraction from rendered views
    4. Geometry encoding with transformer architecture
    5. Material property prediction via VAE decoder

    Supports both high-level APIs (get_splat_materials, get_mesh_materials) and
    low-level APIs (get_features, predict_materials) for custom workflows.

    Examples:
        >>> # High-level API for Gaussian splats
        >>> model = Vomp.from_checkpoint(config_path, geo_dir, mat_dir, norm_path)
        >>> results = model.get_splat_materials("gaussian.ply")
        >>>
        >>> # High-level API for meshes
        >>> results = model.get_mesh_materials("mesh.obj")
        >>>
        >>> # Custom workflow
        >>> coords, features = model.get_features(obj, render_func, voxel_func)
        >>> results = model.predict_materials(coords, features)
    """

    def __init__(
        self,
        geometry_encoder: ElasticGeometryEncoder,
        matvae: TripletVAE,
        material_transform: MaterialPropertyTransform,
        decoder: Optional[ElasticSLatVoxelDecoder] = None,
        config: Optional[Dict[str, Any]] = None,
        use_trt: bool = False,
    ):
        """
        Initialize Vomp model.

        Sets up the complete material property inference pipeline by combining
        a geometry encoder, material VAE, and property transforms. All models
        are automatically set to evaluation mode and MatVAE weights are frozen
        to preserve pre-trained material knowledge.

        Args:
            geometry_encoder: Pre-trained geometry transformer for encoding 3D structure
                into latent representations from sparse voxel features
            matvae: Pre-trained material VAE for decoding material properties from
                latent codes (Young's modulus, Poisson's ratio, density)
            material_transform: Transform handler for normalizing/denormalizing
                material properties between physical units and model space
            decoder: Optional voxel decoder for encoder-decoder training modes.
                Required when config['training_mode'] is 'encoder_decoder_*'
            config: Model configuration dictionary containing training mode,
                normalization parameters, and other model settings
            use_trt: Whether to use TensorRT acceleration for DINO model inference (significantly faster).
                Requires torch-tensorrt package to be installed.

        Note:
            - All models are automatically moved to evaluation mode
            - MatVAE parameters are frozen to preserve pre-trained weights
            - The geometry encoder remains trainable for fine-tuning
        """
        super().__init__()

        self.geometry_encoder = geometry_encoder
        self.matvae = matvae
        self.decoder = decoder
        self.material_transform = material_transform
        self.config = config or {}
        self.use_trt = use_trt

        # Configure models for inference
        self.geometry_encoder.eval()
        self.matvae.eval()
        if self.decoder is not None:
            self.decoder.eval()

        # Freeze pre-trained MatVAE weights
        for param in self.matvae.parameters():
            param.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        config_path: Optional[str] = None,
        geometry_checkpoint_dir: Optional[str] = None,
        matvae_checkpoint_dir: Optional[str] = None,
        normalization_params_path: Optional[str] = None,
        geometry_ckpt: str = "latest",
        device: Union[str, torch.device, None] = None,
        use_trt: bool = False,
    ) -> "Vomp":
        """
        Load Vomp model from pretrained checkpoints.

        This method loads all required components including the geometry encoder,
        material VAE, and normalization parameters from their respective checkpoint
        directories or direct file paths and combines them into a ready-to-use inference model.

        Args:
            config_path: Path to configuration JSON file. Can be either:
                - Training config (contains models, dataset, trainer sections)
                - Inference config (contains checkpoint paths + models, dataset sections)
                If inference config is provided, checkpoint paths are automatically loaded.
            geometry_checkpoint_dir: Directory containing geometry encoder checkpoints
                (looks for files like "geometry_encoder_step*.pt") OR direct path to
                geometry encoder .pt file. If None and config_path is inference config,
                loaded from config.
            matvae_checkpoint_dir: Directory containing MatVAE checkpoints
                (looks for "checkpoints/checkpoint_*/model.safetensors") OR direct path to
                MatVAE .safetensors file. If None and config_path is inference config,
                loaded from config.
            normalization_params_path: Path to JSON file with material property
                normalization parameters. If None and config_path is inference config,
                loaded from config.
            geometry_ckpt: Geometry checkpoint to load. Only used when geometry_checkpoint_dir
                is a directory. Can be:
                - "latest": Load most recent checkpoint by step number
                - Step number (str): Load specific step checkpoint
            device: Device to load models on. If None, uses CUDA if available,
                otherwise CPU
            use_trt: Whether to use TensorRT acceleration for DINO model inference (significantly faster).
                Requires torch-tensorrt package to be installed.

        Returns:
            Fully loaded Vomp model ready for inference

        Raises:
            FileNotFoundError: If required checkpoint files are not found
            ValueError: If checkpoint format is invalid or required arguments not provided

        Examples:
            # Using inference config (recommended)
            >>> model = Vomp.from_checkpoint(
            ...     config_path="configs/materials/inference.json"
            ... )

            # Using inference config with overrides
            >>> model = Vomp.from_checkpoint(
            ...     config_path="configs/materials/inference.json",
            ...     geometry_checkpoint_dir="custom/path/to/geometry_transformer.pt"
            ... )

            # Using training config with direct file paths
            >>> model = Vomp.from_checkpoint(
            ...     config_path="configs/materials/geometry_encoder/train.json",
            ...     geometry_checkpoint_dir="weights/geometry_transformer.pt",
            ...     matvae_checkpoint_dir="weights/matvae.safetensors",
            ...     normalization_params_path="weights/normalization_params.json"
            ... )

            # Using training config with directories (advanced)
            >>> model = Vomp.from_checkpoint(
            ...     config_path="configs/materials/geometry_encoder/train.json",
            ...     geometry_checkpoint_dir="outputs/fixes/normal",
            ...     matvae_checkpoint_dir="outputs/matvae2",
            ...     normalization_params_path="outputs/matvae2/normalization_params.json",
            ...     geometry_ckpt="latest"
            ... )
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        print(f"Loading Vomp model on device: {device}")

        # Validate config_path is provided
        if config_path is None:
            raise ValueError("config_path must be provided")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check if this is an inference config (has checkpoint paths) or training config
        is_inference_config = any(
            key in config
            for key in [
                "geometry_checkpoint_dir",
                "matvae_checkpoint_dir",
                "normalization_params_path",
            ]
        )

        if is_inference_config:
            print(f"✓ Detected inference config: {config_path}")
            # Use inference config values as defaults if individual arguments not provided
            if geometry_checkpoint_dir is None:
                geometry_checkpoint_dir = config.get("geometry_checkpoint_dir")
            if matvae_checkpoint_dir is None:
                matvae_checkpoint_dir = config.get("matvae_checkpoint_dir")
            if normalization_params_path is None:
                normalization_params_path = config.get("normalization_params_path")
        else:
            print(f"✓ Detected training config: {config_path}")

        # Validate that we have all required parameters
        if geometry_checkpoint_dir is None:
            raise ValueError(
                "geometry_checkpoint_dir must be provided either directly or via inference config"
            )
        if matvae_checkpoint_dir is None:
            raise ValueError(
                "matvae_checkpoint_dir must be provided either directly or via inference config"
            )
        if normalization_params_path is None:
            raise ValueError(
                "normalization_params_path must be provided either directly or via inference config"
            )

        cfg = edict(config)
        cfg.load_dir = geometry_checkpoint_dir
        cfg.ckpt = geometry_ckpt
        cfg.matvae_dir = matvae_checkpoint_dir

        # Find geometry checkpoint
        cfg = cls._find_ckpt(cfg)

        # Find MatVAE checkpoint
        cfg = cls._find_matvae_ckpt(cfg)

        # Load material transform
        normalization_type = cfg.dataset.get("normalization_type", "log_minmax")
        material_transform = cls._load_material_transform(
            normalization_params_path, normalization_type
        )

        # Load models
        geometry_encoder, decoder = cls._load_geometry_models(cfg, device)
        matvae = cls._load_matvae(cfg, device)

        print("✓ All models loaded successfully")

        # Create instance
        instance = cls(
            geometry_encoder=geometry_encoder,
            matvae=matvae,
            material_transform=material_transform,
            decoder=decoder,
            config=cfg,
            use_trt=use_trt,
        )

        # Warmup DINO model with TensorRT
        if use_trt:
            print("Warming up DINO model with TensorRT...")
            dino = LazyLoadDino(
                model_name="dinov2_vitl14_reg", device=device, use_trt=True
            )
            # Trigger model loading and TensorRT compilation
            _ = dino.get_model()
            print("✓ DINO model warmed up with TensorRT")

        return instance

    @staticmethod
    def _find_ckpt(cfg: edict) -> edict:
        """
        Locate and validate geometry encoder checkpoint file.

        Determines the appropriate geometry encoder checkpoint to load based on
        configuration. Supports both direct file paths and directory-based
        checkpoint discovery with step-based selection.

        Args:
            cfg: Configuration object containing:
                - load_dir: Checkpoint directory or direct file path
                - ckpt: Checkpoint selection ('latest', 'none', or step number)

        Returns:
            Updated configuration with checkpoint information:
                - load_ckpt: Step number to load (None if no checkpoint)
                - geometry_checkpoint_file: Direct path if file provided

        Raises:
            FileNotFoundError: If required checkpoint files are not found

        Note:
            - If load_dir is a file path, uses it directly
            - If load_dir is a directory, searches for 'misc_step*.pt' files
            - 'latest' selects highest step number checkpoint
            - 'none' disables checkpoint loading
        """
        cfg["load_ckpt"] = None
        cfg["geometry_checkpoint_file"] = None
        searched_paths = []

        if cfg.load_dir != "":
            # Check if load_dir is a direct file path
            if os.path.isfile(cfg.load_dir):
                # Direct file path provided
                cfg.geometry_checkpoint_file = cfg.load_dir
                print(f"✓ Using direct geometry checkpoint: {cfg.load_dir}")
            else:
                # Directory path provided - use original logic
                if cfg.ckpt == "latest":
                    search_pattern = os.path.join(
                        cfg.load_dir, "ckpts", "misc_step*.pt"
                    )
                    searched_paths.append(search_pattern)
                    files = glob.glob(search_pattern)
                    if len(files) != 0:
                        cfg.load_ckpt = max(
                            [
                                int(os.path.basename(f).split("step")[-1].split(".")[0])
                                for f in files
                            ]
                        )
                    else:
                        raise FileNotFoundError(
                            f"No geometry encoder checkpoint files found.\n"
                            f"Searched for pattern: {search_pattern}\n"
                            f"Directory contents: {os.listdir(os.path.join(cfg.load_dir, 'ckpts')) if os.path.exists(os.path.join(cfg.load_dir, 'ckpts')) else 'Directory does not exist'}\n"
                            f"Expected filename pattern: misc_step*.pt"
                        )
                elif cfg.ckpt == "none":
                    cfg.load_ckpt = None
                else:
                    cfg.load_ckpt = int(cfg.ckpt)
                    # Validate that the specific checkpoint exists
                    expected_path = os.path.join(
                        cfg.load_dir, "ckpts", f"misc_step{cfg.load_ckpt:07d}.pt"
                    )
                    searched_paths.append(expected_path)
                    if not os.path.exists(expected_path):
                        raise FileNotFoundError(
                            f"Specified geometry encoder checkpoint not found.\n"
                            f"Searched for: {expected_path}\n"
                            f"Directory contents: {os.listdir(os.path.join(cfg.load_dir, 'ckpts')) if os.path.exists(os.path.join(cfg.load_dir, 'ckpts')) else 'Directory does not exist'}"
                        )
        else:
            raise ValueError("geometry_checkpoint_dir cannot be empty")

        return cfg

    @staticmethod
    def _find_matvae_ckpt(cfg: edict) -> edict:
        """
        Locate and validate MatVAE checkpoint file.

        Searches for MatVAE model checkpoints in either direct file paths or
        structured checkpoint directories. Automatically selects the latest
        checkpoint when multiple versions are available.

        Args:
            cfg: Configuration object containing:
                - matvae_dir: MatVAE checkpoint directory or direct .safetensors path

        Returns:
            Updated configuration with:
                - matvae_checkpoint: Path to selected checkpoint file

        Raises:
            FileNotFoundError: If no valid MatVAE checkpoint is found

        Note:
            Checkpoint search priority:
            1. Direct .safetensors file path
            2. {matvae_dir}/checkpoints/checkpoint_*/model.safetensors (latest by number)
            3. {matvae_dir}/model.safetensors (fallback)
        """
        cfg["matvae_checkpoint"] = None
        searched_paths = []

        if cfg.matvae_dir == "":
            raise ValueError("matvae_checkpoint_dir cannot be empty")

        # Check if matvae_dir is a direct file path
        if os.path.isfile(cfg.matvae_dir):
            # Direct file path provided
            cfg.matvae_checkpoint = cfg.matvae_dir
            print(f"✓ Using direct MatVAE checkpoint: {cfg.matvae_dir}")
        else:
            # Directory path provided - use original logic
            # Look for model.safetensors in checkpoints directory
            # Pattern: {matvae_dir}/checkpoints/checkpoint_*/model.safetensors
            checkpoint_pattern = os.path.join(
                cfg.matvae_dir, "checkpoints", "checkpoint_*", "model.safetensors"
            )
            searched_paths.append(checkpoint_pattern)
            files = glob.glob(checkpoint_pattern)

            if len(files) > 0:
                # Find the latest checkpoint by extracting checkpoint number
                def get_checkpoint_num(path):
                    # Extract number from checkpoint_XXX directory name
                    checkpoint_dir = os.path.basename(os.path.dirname(path))
                    if checkpoint_dir.startswith("checkpoint_"):
                        try:
                            return int(checkpoint_dir.split("_")[1])
                        except (IndexError, ValueError):
                            return 0
                    return 0

                # Use the checkpoint with the highest number
                latest_file = max(files, key=get_checkpoint_num)
                cfg.matvae_checkpoint = latest_file
                print(f"✓ Found MatVAE checkpoint: {latest_file}")
            else:
                # Also try looking for model.safetensors directly in the directory
                direct_path = os.path.join(cfg.matvae_dir, "model.safetensors")
                searched_paths.append(direct_path)
                if os.path.exists(direct_path):
                    cfg.matvae_checkpoint = direct_path
                    print(f"✓ Found MatVAE checkpoint: {direct_path}")
                else:
                    # Gather directory information for error message
                    dir_contents = "Directory does not exist"
                    checkpoints_dir = os.path.join(cfg.matvae_dir, "checkpoints")
                    if os.path.exists(cfg.matvae_dir):
                        dir_contents = (
                            f"Directory contents: {os.listdir(cfg.matvae_dir)}"
                        )
                        if os.path.exists(checkpoints_dir):
                            checkpoint_subdirs = [
                                d
                                for d in os.listdir(checkpoints_dir)
                                if os.path.isdir(os.path.join(checkpoints_dir, d))
                            ]
                            dir_contents += (
                                f"\nCheckpoints subdirectories: {checkpoint_subdirs}"
                            )

                    raise FileNotFoundError(
                        f"No MatVAE checkpoint found.\n"
                        f"Searched for patterns:\n"
                        f"  1. {checkpoint_pattern}\n"
                        f"  2. {direct_path}\n"
                        f"{dir_contents}\n"
                        f"Expected:\n"
                        f"  - model.safetensors in checkpoints/checkpoint_*/ subdirectories, or\n"
                        f"  - model.safetensors directly in {cfg.matvae_dir}"
                    )
        return cfg

    @staticmethod
    def _load_material_transform(
        normalization_params_file: str, normalization_type: str = "log_minmax"
    ) -> MaterialPropertyTransform:
        """
        Load and configure material property normalization transform.

        Creates a MaterialPropertyTransform instance with normalization parameters
        loaded from a JSON file. Supports log-min-max normalization for converting
        between physical units and normalized model space.

        Args:
            normalization_params_file: Path to JSON file containing normalization
                parameters with keys: E_min, E_max, nu_min, nu_max, rho_min, rho_max
            normalization_type: Type of normalization to apply. Currently supports:
                - 'log_minmax': Log transform for E and rho, linear for nu

        Returns:
            Configured MaterialPropertyTransform ready for use

        Note:
            - Young's modulus (E) and density (rho) use log10 transformation
            - Poisson's ratio (nu) uses linear normalization
            - All parameters are normalized to [0,1] range for model input

        Raises:
            FileNotFoundError: If normalization_params_file doesn't exist
            KeyError: If required parameters are missing from the file
        """
        with open(normalization_params_file, "r") as f:
            params = json.load(f)

        transform = MaterialPropertyTransform(normalization_type=normalization_type)

        if normalization_type == "log_minmax":
            transform.E_min = math.log10(params["E_min"])
            transform.E_max = math.log10(params["E_max"])
            transform.nu_min = params["nu_min"]
            transform.nu_max = params["nu_max"]
            transform.rho_min = math.log10(params["rho_min"])
            transform.rho_max = math.log10(params["rho_max"])
            transform._stats_computed = True

        return transform

    @staticmethod
    def _load_geometry_models(
        cfg: edict, device: torch.device
    ) -> Tuple[ElasticGeometryEncoder, Optional[ElasticSLatVoxelDecoder]]:
        """
        Initialize and load geometry encoder and optional decoder models.

        Creates model instances from configuration and loads pre-trained weights
        from checkpoint files. Supports both encoder-only and encoder-decoder
        architectures based on configuration.

        Args:
            cfg: Configuration object containing:
                - models.geometry_encoder.args: Encoder model arguments
                - models.decoder.args: Decoder model arguments (optional)
                - geometry_checkpoint_file: Direct path to checkpoint (if provided)
                - load_dir: Checkpoint directory (if using directory structure)
                - load_ckpt: Step number for checkpoint loading
            device: Target device for model placement

        Returns:
            Tuple of (geometry_encoder, decoder):
                - geometry_encoder: Loaded ElasticGeometryEncoder instance
                - decoder: ElasticSLatVoxelDecoder instance or None if not configured

        Raises:
            FileNotFoundError: If checkpoint files don't exist
            RuntimeError: If model state dict loading fails

        Note:
            - Models are automatically moved to specified device
            - Checkpoint loading is strict (all parameters must match)
            - Decoder is only loaded if specified in configuration
            - Supports both direct file paths and directory-based checkpoints
        """

        # Load geometry encoder
        try:
            geometry_encoder = ElasticGeometryEncoder(
                **cfg.models.geometry_encoder.args
            ).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize geometry encoder model: {e}")

        # Load decoder if specified
        decoder = None
        if "decoder" in cfg.models:
            try:
                decoder = ElasticSLatVoxelDecoder(**cfg.models.decoder.args).to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize decoder model: {e}")

        # Load checkpoints
        if cfg.get("geometry_checkpoint_file") is not None:
            # Direct file path provided
            geometry_encoder_path = cfg.geometry_checkpoint_file
            if os.path.exists(geometry_encoder_path):
                try:
                    checkpoint = torch.load(
                        geometry_encoder_path, map_location="cpu", weights_only=False
                    )
                    geometry_encoder.load_state_dict(checkpoint, strict=True)
                    print(f"✓ Loaded geometry encoder from: {geometry_encoder_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load geometry encoder checkpoint from {geometry_encoder_path}: {e}\n"
                        f"This could be due to:\n"
                        f"  - Checkpoint corruption\n"
                        f"  - Model architecture mismatch\n"
                        f"  - Incompatible checkpoint format"
                    )
            else:
                raise FileNotFoundError(
                    f"Geometry checkpoint file not found: {geometry_encoder_path}"
                )
        elif cfg.load_ckpt is not None:
            # Directory path provided - use original logic
            geometry_encoder_path = os.path.join(
                cfg.load_dir, "ckpts", f"geometry_encoder_step{cfg.load_ckpt:07d}.pt"
            )
            if os.path.exists(geometry_encoder_path):
                try:
                    checkpoint = torch.load(
                        geometry_encoder_path, map_location="cpu", weights_only=False
                    )
                    geometry_encoder.load_state_dict(checkpoint, strict=True)
                    print(f"✓ Loaded geometry encoder from: {geometry_encoder_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load geometry encoder checkpoint from {geometry_encoder_path}: {e}\n"
                        f"This could be due to:\n"
                        f"  - Checkpoint corruption\n"
                        f"  - Model architecture mismatch\n"
                        f"  - Incompatible checkpoint format"
                    )
            else:
                # This should not happen if _find_ckpt is working correctly, but adding for safety
                raise FileNotFoundError(
                    f"Geometry encoder checkpoint not found: {geometry_encoder_path}\n"
                    f"Directory contents: {os.listdir(os.path.join(cfg.load_dir, 'ckpts')) if os.path.exists(os.path.join(cfg.load_dir, 'ckpts')) else 'Directory does not exist'}"
                )

            if decoder is not None:
                decoder_path = os.path.join(
                    cfg.load_dir, "ckpts", f"decoder_step{cfg.load_ckpt:07d}.pt"
                )
                if os.path.exists(decoder_path):
                    try:
                        checkpoint = torch.load(
                            decoder_path, map_location="cpu", weights_only=False
                        )
                        decoder.load_state_dict(checkpoint, strict=True)
                        print(f"✓ Loaded decoder from: {decoder_path}")
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load decoder checkpoint from {decoder_path}: {e}\n"
                            f"This could be due to:\n"
                            f"  - Checkpoint corruption\n"
                            f"  - Model architecture mismatch\n"
                            f"  - Incompatible checkpoint format"
                        )
                else:
                    print(f"⚠ Decoder checkpoint not found (optional): {decoder_path}")
        else:
            print(
                "⚠ No geometry encoder checkpoint specified - using randomly initialized weights"
            )

        return geometry_encoder, decoder

    @staticmethod
    def _load_matvae(cfg: edict, device: torch.device) -> TripletVAE:
        """
        Initialize and load pre-trained MatVAE model.

        Creates a TripletVAE instance from configuration and loads pre-trained
        weights from a safetensors checkpoint file. The MatVAE handles encoding
        and decoding of material properties (Young's modulus, Poisson's ratio, density).

        Args:
            cfg: Configuration object containing:
                - models.matvae.args: MatVAE model architecture arguments
                - matvae_checkpoint: Path to .safetensors checkpoint file
            device: Target device for model placement

        Returns:
            Loaded TripletVAE model ready for inference

        Raises:
            RuntimeError: If model initialization or checkpoint loading fails
            FileNotFoundError: If checkpoint file doesn't exist

        Note:
            - Model is automatically moved to specified device
            - Uses safetensors format for secure checkpoint loading
            - Checkpoint loading is strict (all parameters must match)
            - MatVAE weights are typically frozen during geometry encoder training
        """
        try:
            matvae = TripletVAE(**cfg.models.matvae.args).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MatVAE model: {e}")

        if cfg.get("matvae_checkpoint") is not None:
            if not os.path.exists(cfg.matvae_checkpoint):
                raise FileNotFoundError(
                    f"MatVAE checkpoint file not found: {cfg.matvae_checkpoint}"
                )

            try:
                checkpoint = load_file(cfg.matvae_checkpoint)
                matvae.load_state_dict(checkpoint, strict=True)
                print(f"✓ Loaded MatVAE from: {cfg.matvae_checkpoint}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load MatVAE checkpoint from {cfg.matvae_checkpoint}: {e}\n"
                    f"This could be due to:\n"
                    f"  - Checkpoint corruption\n"
                    f"  - Model architecture mismatch\n"
                    f"  - Incompatible safetensors format\n"
                    f"  - Missing or extra parameters in checkpoint"
                )
        else:
            raise RuntimeError(
                "No MatVAE checkpoint specified. MatVAE requires pre-trained weights for proper inference."
            )

        return matvae

    def to(self, device: Union[str, torch.device]) -> "Vomp":
        """
        Move all model components to specified device.

        Transfers the geometry encoder, MatVAE, and optional decoder to the
        target device (CPU/GPU). Useful for switching between devices after
        model initialization.

        Args:
            device: Target device specification. Can be:
                - String: 'cpu', 'cuda', 'cuda:0', etc.
                - torch.device: Device object

        Returns:
            Self reference for method chaining

        Example:
            >>> model = Vomp.from_checkpoint(config_path)
            >>> model = model.to('cuda:1')  # Move to specific GPU
        """
        device = torch.device(device)

        self.geometry_encoder = self.geometry_encoder.to(device)
        self.matvae = self.matvae.to(device)
        if self.decoder is not None:
            self.decoder = self.decoder.to(device)

        return self

    @torch.inference_mode()
    def predict_materials(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        max_voxels: int = 32000,
        sample_posterior: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict material properties for given coordinates and features.

        Takes voxel coordinates and their corresponding DINO features and runs them
        through the geometry encoder and material VAE to predict mechanical properties.
        Handles memory management by downsampling if too many voxels are provided.

        Args:
            coords: Voxel coordinates tensor (N, 4) in format [batch_idx, x, y, z]
                where coordinates are normalized to [-0.5, 0.5] range
            features: DINO feature vectors (N, feature_dim) extracted from rendered views
            max_voxels: Maximum number of voxels to process at once. If input exceeds
                this, random downsampling is applied to manage GPU memory usage
            sample_posterior: Whether to sample from VAE posterior distribution instead
                of using mean predictions for more diverse outputs

        Returns:
            Dictionary containing:
            - 'youngs_modulus': Young's modulus values in Pascals (N,)
            - 'poisson_ratio': Poisson's ratio values (N,)
            - 'density': Density values in kg/m³ (N,)
            - 'voxel_coords_world': World space coordinates (N, 3)
            - 'num_voxels': Total number of processed voxels

        Note:
            All predictions are automatically denormalized using the loaded
            normalization parameters to return physically meaningful values.
        """
        device = next(self.parameters()).device
        coords = coords.to(device)
        features = features.to(device)

        # Downsample if too many voxels
        num_voxels = coords.shape[0]
        if num_voxels > max_voxels:
            print(f"Downsampling from {num_voxels} to {max_voxels} voxels")
            indices = torch.randperm(num_voxels)[:max_voxels]
            coords = coords[indices]
            features = features[indices]
            num_voxels = max_voxels

        # Create sparse tensor
        sparse_tensor = SparseTensor(
            feats=features, coords=coords, shape=(1, 64, 64, 64)
        )

        print(f"Running inference on {num_voxels} voxels...")

        # Get training mode from config
        training_mode = self.config.get("training_mode", "encoder_only")

        # Forward pass through geometry encoder
        z, mean, logvar = self.geometry_encoder(
            sparse_tensor, sample_posterior=sample_posterior, return_raw=True
        )

        # Decode materials based on training mode
        if training_mode == "encoder_only":
            latent_2d = z.feats
            (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                self.matvae.decode(latent_2d)
            )
            E_pred = E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
            nu_pred = nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
            rho_pred = rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu

        elif training_mode == "encoder_decoder_matvae":
            decoder_output = self.decoder(z)
            latent_2d = decoder_output.feats
            (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = (
                self.matvae.decode(latent_2d)
            )
            E_pred = E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
            nu_pred = nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
            rho_pred = rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu

        elif training_mode == "encoder_decoder_direct":
            decoder_output = self.decoder(z)
            material_predictions = decoder_output.feats
            E_pred = material_predictions[:, 0]
            nu_pred = material_predictions[:, 1]
            rho_pred = material_predictions[:, 2]

        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        # Denormalize predictions
        pred_materials_normalized = torch.stack([E_pred, nu_pred, rho_pred], dim=-1)
        pred_materials_raw = (
            self.material_transform.destandardize_and_inverse_transform_tensor(
                pred_materials_normalized
            )
        )

        # Convert to numpy
        E_values = pred_materials_raw[:, 0].cpu().numpy()
        nu_values = pred_materials_raw[:, 1].cpu().numpy()
        rho_values = pred_materials_raw[:, 2].cpu().numpy()

        # Calculate world coordinates
        voxel_indices = coords[:, 1:].cpu().numpy()
        voxel_coords_world = (voxel_indices + 0.5) / 64 - 0.5

        return {
            "youngs_modulus": E_values,
            "poisson_ratio": nu_values,
            "density": rho_values,
            "num_voxels": num_voxels,
            "voxel_coords": coords.cpu().numpy(),
            "voxel_coords_world": voxel_coords_world,
            "material_properties_per_voxel": pred_materials_raw.cpu().numpy(),
        }

    @property
    def device(self) -> torch.device:
        """
        Get the current device of the model.

        Returns the device where the model parameters are currently located.
        All components (geometry encoder, MatVAE, decoder) should be on the
        same device.

        Returns:
            torch.device: Current device of model parameters

        Example:
            >>> print(f"Model is on: {model.device}")
            Model is on: cuda:0
        """
        return next(self.parameters()).device

    @torch.inference_mode()
    def run_single_object_inference(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        max_voxels: int = 32000,
        sample_posterior: bool = False,
        return_latents: bool = False,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Low-level raw input and low-level raw output inference.

        Args:
            coords: Voxel coordinates (N, 4) - [batch_idx, x, y, z]
            features: Feature vectors (N, feature_dim)
            max_voxels: Maximum number of voxels to process (for memory)
            sample_posterior: Whether to sample from posterior
            return_latents: Whether to return raw latent representations

        Returns:
            Dictionary containing raw model outputs
        """
        device = self.device
        coords = coords.to(device)
        features = features.to(device)

        # Downsample if too many voxels
        num_voxels = coords.shape[0]
        if num_voxels > max_voxels:
            indices = torch.randperm(num_voxels)[:max_voxels]
            coords = coords[indices]
            features = features[indices]
            num_voxels = max_voxels

        # Create sparse tensor
        sparse_tensor = SparseTensor(
            feats=features, coords=coords, shape=(1, 64, 64, 64)
        )

        # Forward pass through geometry encoder
        z, mean, logvar = self.geometry_encoder(
            sparse_tensor, sample_posterior=sample_posterior, return_raw=True
        )

        result = {
            "geometry_latent": z.feats,
            "geometry_mean": mean,
            "geometry_logvar": logvar,
            "voxel_coords": coords,
            "num_voxels": num_voxels,
        }

        # Get training mode and decode accordingly
        training_mode = self.config.get("training_mode", "encoder_only")

        if training_mode == "encoder_decoder_matvae" and self.decoder is not None:
            decoder_output = self.decoder(z)
            result["decoder_output"] = decoder_output.feats
            material_latents = decoder_output.feats
        elif training_mode == "encoder_decoder_direct" and self.decoder is not None:
            decoder_output = self.decoder(z)
            result["material_predictions_raw"] = decoder_output.feats
            return result
        else:
            material_latents = z.feats

        # Decode through MatVAE
        (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = self.matvae.decode(
            material_latents
        )

        result.update(
            {
                "E_mu": E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu,
                "E_logvar": E_logvar.squeeze(-1) if E_logvar.dim() > 1 else E_logvar,
                "nu_mu": nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu,
                "nu_logvar": (
                    nu_logvar.squeeze(-1) if nu_logvar.dim() > 1 else nu_logvar
                ),
                "rho_mu": rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu,
                "rho_logvar": (
                    rho_logvar.squeeze(-1) if rho_logvar.dim() > 1 else rho_logvar
                ),
            }
        )

        if return_latents:
            result["material_latents"] = material_latents

        return result

    @torch.inference_mode()
    def decode_material(
        self, latent: torch.Tensor, denormalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Decode material properties from latent representation.

        Args:
            latent: Latent material representation (N, latent_dim)
            denormalize: Whether to denormalize to physical units

        Returns:
            Dictionary with material properties
        """
        (E_mu, E_logvar), (nu_mu, nu_logvar), (rho_mu, rho_logvar) = self.matvae.decode(
            latent.to(self.device)
        )

        E_pred = E_mu.squeeze(-1) if E_mu.dim() > 1 else E_mu
        nu_pred = nu_mu.squeeze(-1) if nu_mu.dim() > 1 else nu_mu
        rho_pred = rho_mu.squeeze(-1) if rho_mu.dim() > 1 else rho_mu

        result = {
            "youngs_modulus": E_pred,
            "poisson_ratio": nu_pred,
            "density": rho_pred,
            "E_logvar": E_logvar.squeeze(-1) if E_logvar.dim() > 1 else E_logvar,
            "nu_logvar": nu_logvar.squeeze(-1) if nu_logvar.dim() > 1 else nu_logvar,
            "rho_logvar": (
                rho_logvar.squeeze(-1) if rho_logvar.dim() > 1 else rho_logvar
            ),
        }

        if denormalize:
            # Denormalize to physical units
            pred_materials_normalized = torch.stack([E_pred, nu_pred, rho_pred], dim=-1)
            pred_materials_raw = (
                self.material_transform.destandardize_and_inverse_transform_tensor(
                    pred_materials_normalized
                )
            )

            result.update(
                {
                    "youngs_modulus_pa": pred_materials_raw[:, 0],
                    "poisson_ratio_raw": pred_materials_raw[:, 1],
                    "density_kg_m3": pred_materials_raw[:, 2],
                }
            )

        return result

    @torch.inference_mode()
    def encode_material(
        self,
        youngs_modulus: torch.Tensor,
        poisson_ratio: torch.Tensor,
        density: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode material properties to latent representation.

        Args:
            youngs_modulus: Young's modulus values (N,) in Pa
            poisson_ratio: Poisson's ratio values (N,)
            density: Density values (N,) in kg/m³
            normalize: Whether to normalize from physical units

        Returns:
            Latent material representation (N, latent_dim)
        """
        device = self.device
        E = youngs_modulus.to(device)
        nu = poisson_ratio.to(device)
        rho = density.to(device)

        if normalize:
            # Stack and normalize
            materials = torch.stack([E, nu, rho], dim=-1)
            materials_normalized = (
                self.material_transform.standardize_and_transform_tensor(materials)
            )
            E, nu, rho = (
                materials_normalized[:, 0],
                materials_normalized[:, 1],
                materials_normalized[:, 2],
            )

        # Encode through MatVAE
        material_tensor = torch.stack([E, nu, rho], dim=-1)
        latent_mean, latent_logvar = self.matvae.encode(material_tensor)

        return latent_mean

    @torch.inference_mode()
    def get_features(
        self,
        obj_3d: Any = None,
        renders_metadata: Optional[List[Dict[str, Any]]] = None,
        voxel_centers: Optional[np.ndarray] = None,
        render_func: Optional[RenderFunction] = None,
        voxelize_func: Optional[VoxelizeFunction] = None,
        num_views: int = 150,
        model_name: str = "dinov2_vitl14_reg",
        batch_size: int = 16,
        image_size: int = 518,
        output_dir: Optional[str] = None,
        save_features: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DINO features from 3D object through rendering and voxelization.

        Core feature extraction supporting custom rendering/voxelization workflows.
        For Gaussian splats, prefer get_splat_materials() for simplicity.

        Args:
            obj_3d: 3D object for feature extraction
            renders_metadata: Pre-computed render metadata list
            voxel_centers: Pre-computed voxel centers (N, 3)
            render_func: Custom rendering function (see RenderFunction protocol)
            voxelize_func: Custom voxelization function (see VoxelizeFunction protocol)
            num_views: Number of camera views for rendering
            model_name: DINO model identifier
            batch_size: Feature extraction batch size
            image_size: Render target image size
            output_dir: Directory for intermediate files
            save_features: Whether to cache features to disk
            **kwargs: Additional arguments for custom functions

        Returns:
            Tuple of (voxel_coordinates, dino_features):
            - voxel_coordinates: (N, 4) tensor [batch_idx, x, y, z]
            - dino_features: (N, feature_dim) tensor with extracted features

        Raises:
            ValueError: If required inputs are missing for chosen workflow
        """
        if output_dir is None:
            output_dir = "/tmp/Vomp_features"
        os.makedirs(output_dir, exist_ok=True)

        # Obtain or generate render metadata
        if renders_metadata is None:
            if render_func is not None:
                # Use custom rendering function
                if obj_3d is None:
                    raise ValueError(
                        "obj_3d must be provided when using custom render_func"
                    )
                print("Using custom rendering function...")
                renders_metadata = render_func(
                    obj_3d, output_dir, num_views, image_size, **kwargs
                )
            else:
                # Try to load existing renders metadata
                metadata_path = os.path.join(output_dir, "renders_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        renders_metadata = json.load(f)
                else:
                    raise ValueError(
                        "No renders_metadata provided and no existing metadata found. "
                        "Either provide renders_metadata, a custom render_func, or ensure renders exist."
                    )

        # Obtain or generate voxel centers
        if voxel_centers is None:
            if voxelize_func is not None:
                # Use custom voxelization function
                if obj_3d is None:
                    raise ValueError(
                        "obj_3d must be provided when using custom voxelize_func"
                    )
                print("Using custom voxelization function...")
                voxel_centers = voxelize_func(obj_3d, output_dir, **kwargs)
            else:
                # Try to load existing voxels
                voxel_path = os.path.join(output_dir, "voxels", "voxels.ply")
                if os.path.exists(voxel_path):
                    voxel_centers = self._load_voxel_centers(voxel_path)
                else:
                    raise ValueError(
                        "No voxel_centers provided and no existing voxels found. "
                        "Either provide voxel_centers, a custom voxelize_func, or ensure voxels exist."
                    )

        # Step 3: Extract features using DINO
        coords, features = self._extract_dino_features(
            output_dir,
            voxel_centers,
            renders_metadata,
            model_name,
            batch_size,
            image_size,
            save_features,
        )

        return coords, features

    def _sample_camera_views(
        self,
        num_views: int,
        radius: float = 2.0,
        fov: float = 45.0,
        seed: Optional[int] = None,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Generate camera viewpoints using quasi-random sampling.

        Uses Hammersley sequence for uniform distribution of camera positions
        on a sphere around the object. Provides better coverage than random
        sampling and ensures consistent results across runs.

        Args:
            num_views: Number of camera viewpoints to generate
            radius: Distance from object center to camera positions
            fov: Field of view in degrees for all cameras
            seed: Random seed for reproducible sampling. If None, uses random offset

        Returns:
            Tuple containing:
                - yaws: List of yaw angles in radians
                - pitchs: List of pitch angles in radians
                - radius_list: List of camera distances (all same value)
                - fov_list: List of FOV values in degrees (all same value)
        """
        if seed is not None:
            np.random.seed(seed)

        yaws = []
        pitchs = []
        offset = (np.random.rand(), np.random.rand())

        # Prime numbers for Halton sequence generation
        PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

        def radical_inverse(base, n):
            val = 0
            inv_base = 1.0 / base
            inv_base_n = inv_base
            while n > 0:
                digit = n % base
                val += digit * inv_base_n
                n //= base
                inv_base_n *= inv_base
            return val

        def halton_sequence(dim, n):
            return [radical_inverse(PRIMES[d], n) for d in range(dim)]

        def hammersley_sequence(dim, n, num_samples):
            return [n / num_samples] + halton_sequence(dim - 1, n)

        def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
            u, v = hammersley_sequence(2, n, num_samples)
            u += offset[0] / num_samples
            v += offset[1]
            u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
            theta = np.arccos(1 - 2 * u) - np.pi / 2
            phi = v * 2 * np.pi
            return [phi, theta]

        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset)
            yaws.append(y)
            pitchs.append(p)

        radius_list = [radius] * num_views
        fov_list = [fov] * num_views

        return yaws, pitchs, radius_list, fov_list

    @torch.inference_mode()
    def render_sampled_views(
        self,
        gaussian_model: "Gaussian",
        output_dir: str,
        num_views: int = 150,
        image_size: int = 518,
        radius: float = 2.0,
        fov: float = 40.0,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """
        Render multiple views of Gaussian splat from sampled camera positions.

        Generates camera viewpoints using quasi-random sampling and renders
        the Gaussian splat model from each position. Saves RGBA images and
        camera metadata in NeRF-compatible format.

        Args:
            gaussian_model: Gaussian splat model to render
            output_dir: Directory to save rendered images and metadata
            num_views: Number of camera views to render
            image_size: Resolution of rendered images (square)
            radius: Camera distance from object center
            fov: Field of view in degrees
            seed: Random seed for reproducible camera sampling

        Returns:
            List of frame metadata dictionaries containing:
                - file_path: Relative path to rendered image
                - transform_matrix: 4x4 camera-to-world matrix
                - camera_angle_x: FOV in radians
                - yaw, pitch, radius: Camera positioning parameters
        """
        renders_dir = os.path.join(output_dir, "renders")
        os.makedirs(renders_dir, exist_ok=True)

        print(f"Rendering {num_views} Gaussian splat views...")

        # Sample camera views using shared utility
        yaws, pitchs, radius_list, fov_list = self._sample_camera_views(
            num_views, radius, fov, seed
        )

        # Generate camera extrinsics and intrinsics matrices
        extrinsics_list, intrinsics_list = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitchs, radius_list, fov_list
        )

        frames_metadata = []
        for i, (extrinsics, intrinsics) in enumerate(
            zip(extrinsics_list, intrinsics_list)
        ):
            # Create camera configuration for rendering
            fovx = 2 * torch.atan(0.5 / intrinsics[0, 0])
            fovy = 2 * torch.atan(0.5 / intrinsics[1, 1])
            camera_center = torch.inverse(extrinsics)[:3, 3]

            camera = edict(
                {
                    "image_height": image_size,
                    "image_width": image_size,
                    "FoVx": fovx,
                    "FoVy": fovy,
                    "znear": 0.1,
                    "zfar": 100.0,
                    "world_view_transform": extrinsics.T.contiguous(),
                    "projection_matrix": torch.eye(4, device=self.device),
                    "full_proj_transform": extrinsics.T.contiguous(),
                    "camera_center": camera_center,
                }
            )

            # Render the Gaussian splat
            bg_color = torch.tensor(
                [1.0, 1.0, 1.0], dtype=torch.float32, device=self.device
            )
            with torch.inference_mode():
                render_result = self._simple_render(
                    gaussian_model, camera, bg_color, render_alpha=True
                )
                rendered_rgb = render_result["render"]
                rendered_alpha = render_result.get(
                    "alpha", torch.ones_like(rendered_rgb[:1])
                )

            # Combine RGB and alpha channels
            rendered_rgba = torch.cat([rendered_rgb, rendered_alpha], dim=0)
            rendered_rgba = rendered_rgba.cpu().numpy()
            rendered_rgba = np.transpose(rendered_rgba, (1, 2, 0))
            rendered_rgba = np.clip(rendered_rgba * 255, 0, 255).astype(np.uint8)

            # Save rendered image
            filename = f"{i:04d}.png"
            image_path = os.path.join(renders_dir, filename)
            Image.fromarray(rendered_rgba, "RGBA").save(image_path)

            # Create camera-to-world transform matrix for metadata
            c2w = torch.inverse(extrinsics).cpu().numpy()
            c2w[:3, 1:3] *= -1

            frame_metadata = {
                "file_path": filename,
                "transform_matrix": c2w.tolist(),
                "camera_angle_x": np.radians(fov),
                "yaw": yaws[i],
                "pitch": pitchs[i],
                "radius": radius,
            }
            frames_metadata.append(frame_metadata)

        # Save NeRF-compatible transforms.json
        transforms_data = {
            "camera_angle_x": np.radians(fov),
            "frames": frames_metadata,
        }

        transforms_path = os.path.join(renders_dir, "transforms.json")
        with open(transforms_path, "w") as f:
            json.dump(transforms_data, f, indent=2)

        # Save metadata for pipeline compatibility
        metadata_path = os.path.join(output_dir, "renders_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(frames_metadata, f, indent=2)

        print(f"✓ Rendered {num_views} views")
        return frames_metadata

    @torch.inference_mode()
    def get_splat_materials(
        self,
        gaussian_model: Union["Gaussian", str],
        output_dir: Optional[str] = None,
        num_views: int = 150,
        image_size: int = 518,
        render_image_size: int = 512,
        radius: float = 2.0,
        fov: float = 40.0,
        seed: Optional[int] = None,
        sh_degree: int = 3,
        aabb: Optional[List[float]] = None,
        device: Optional[Union[str, torch.device]] = None,
        voxel_method: str = "centers",
        voxel_level: int = 6,
        voxel_iso: float = 11.345,
        voxel_tol: float = 1.0 / 8.0,
        voxel_step: int = 10,
        voxel_opacity_threshold: float = 0.35,
        max_voxels: Optional[int] = 32768,
        query_points: Union[str, np.ndarray, None] = "splat_centers",
        dino_batch_size: int = 16,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        High-level API for Gaussian splat material property inference.

        Automatically handles PLY loading, rendering, voxelization, material prediction,
        and upsampling for Gaussian splat representations.

        Args:
            gaussian_model: Gaussian object or PLY file path
            output_dir: Directory for intermediate files
            num_views: Camera views for rendering (default: 150, matches training data)
            image_size: Target image size for DINO feature extraction. Rendered images are
                resized to this resolution before processing (default: 518)
            render_image_size: Resolution for rendering. Set to 512 to match training
                data pipeline where images were rendered at 512x512 then resized to 518x518
                for DINO feature extraction (default: 512)
            radius: Camera distance from object center (default: 2.0, matches training data)
            fov: Field of view in degrees (default: 40.0, matches training data)
            seed: Random seed for camera view sampling
            sh_degree: Spherical harmonics degree for PLY loading
            aabb: Bounding box for PLY loading (defaults to [-1,-1,-1,2,2,2])
            device: Compute device (defaults to CUDA if available)
            voxel_method: Voxelization method ('centers' or 'kaolin')
                - 'centers': Simple center-based voxelization (fast, less accurate)
                - 'kaolin': Full Gaussian voxelization using kaolin (slower, more accurate)
            voxel_level: Voxel grid resolution level for kaolin voxelization (2^level), default 6 for 64^3
            voxel_iso: Iso value for kaolin voxelization
            voxel_tol: Tolerance for kaolin voxelization
            voxel_step: Number of samples for opacity integration in kaolin method
            voxel_opacity_threshold: Minimum opacity threshold for voxels in kaolin method
            max_voxels: Maximum number of voxels to process (for memory management)
            query_points: Where to evaluate material properties. Can be:
                - "splat_centers" (default): Evaluate at Gaussian splat centers
                - "voxel_centers": Evaluate at voxel centers (original behavior)
                - numpy array (N, 3): Custom 3D coordinates for evaluation
                - None: Same as "voxel_centers"
            dino_batch_size: Number of images to process simultaneously during DINO feature extraction (higher values use more GPU memory but may be faster)
            **kwargs: Additional feature extraction arguments

        Returns:
            Dictionary containing material properties:
            When query_points="splat_centers" or custom array:
            - 'youngs_modulus': Young's modulus values at query points (Pa)
            - 'poisson_ratio': Poisson's ratio values at query points
            - 'density': Density values at query points (kg/m³)
            - 'query_coords_world': World coordinates of query points
            - 'voxel_coords_world': World coordinates of voxels (for reference)
            - 'query_distances': Distance from each query point to nearest voxel
            - 'num_voxels', 'num_query_points': Counts for reference

            When query_points="voxel_centers" or None:
            - 'youngs_modulus': Young's modulus values at voxel centers (Pa)
            - 'poisson_ratio': Poisson's ratio values at voxel centers
            - 'density': Density values at voxel centers (kg/m³)
            - 'voxel_coords_world': World coordinates of voxels
            - 'num_voxels': Number of voxels
        """
        # Handle automatic Gaussian loading from PLY path
        if isinstance(gaussian_model, str):
            # Set defaults
            if aabb is None:
                aabb = [-1, -1, -1, 2, 2, 2]
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Create and load Gaussian
            ply_path = gaussian_model
            gaussian_model = Gaussian(sh_degree=sh_degree, aabb=aabb, device=device)
            gaussian_model.load_ply(ply_path)

        if output_dir is None:
            output_dir = f"/tmp/Vomp_splat_{id(gaussian_model)}"

        print("=== Vomp: Splat Material Estimation ===")

        # Define built-in splat rendering function
        def splat_render_func(
            gaussian_model, output_dir, num_views, image_size, **kwargs
        ):
            # Filter kwargs to only pass render-related parameters
            render_kwargs = {
                k: v for k, v in kwargs.items() if k in ["radius", "fov", "seed"]
            }
            # Use render_image_size for actual rendering (default 512 to match training data)
            return self.render_sampled_views(
                gaussian_model,
                output_dir,
                num_views,
                render_image_size,
                **render_kwargs,
            )

        # Define built-in splat voxelization function
        def splat_voxelize_func(gaussian_model, output_dir, **kwargs):
            # Extract voxelization parameters from kwargs
            voxel_method = kwargs.get("voxel_method", "centers")
            voxel_level = kwargs.get("voxel_level", 6)
            voxel_iso = kwargs.get("voxel_iso", 11.345)
            voxel_tol = kwargs.get("voxel_tol", 1.0 / 8.0)
            voxel_step = kwargs.get("voxel_step", 10)
            voxel_opacity_threshold = kwargs.get("voxel_opacity_threshold", 0.35)

            return self._voxelize_gaussian(
                gaussian_model,
                output_dir,
                method=voxel_method,
                level=voxel_level,
                iso=voxel_iso,
                tol=voxel_tol,
                step=voxel_step,
                opacity_threshold=voxel_opacity_threshold,
            )

        # Step 1: Extract features using built-in splat functions
        print("Step 1: Extracting features...")

        # Normalize Gaussian to standard coordinate system
        gaussian_model = self._normalize_gaussian(gaussian_model)

        # Prepare voxelization parameters for kwargs
        voxel_kwargs = {
            "voxel_method": voxel_method,
            "voxel_level": voxel_level,
            "voxel_iso": voxel_iso,
            "voxel_tol": voxel_tol,
            "voxel_step": voxel_step,
            "voxel_opacity_threshold": voxel_opacity_threshold,
        }

        # Prepare rendering parameters for kwargs
        render_kwargs = {
            "radius": radius,
            "fov": fov,
            "seed": seed,
        }

        coords, features = self.get_features(
            obj_3d=gaussian_model,
            render_func=splat_render_func,
            voxelize_func=splat_voxelize_func,
            num_views=num_views,
            image_size=image_size,
            output_dir=output_dir,
            batch_size=dino_batch_size,
            **voxel_kwargs,
            **render_kwargs,
            **kwargs,
        )

        # Step 2: Run inference on the features
        print("Step 2: Running material inference...")
        predict_kwargs = {}
        if max_voxels is not None:
            predict_kwargs["max_voxels"] = max_voxels
        voxel_results = self.predict_materials(coords, features, **predict_kwargs)

        # Add splat count for reporting
        num_splats = int(gaussian_model.get_xyz.shape[0])
        voxel_results["num_splats"] = num_splats

        # Handle query_points parameter
        if query_points == "splat_centers":
            # Step 3: Evaluate materials at splat centers
            print("Step 3: Evaluating materials at splat centers...")

            # Create upsampler from voxel results
            upsampler = MaterialUpsampler(
                voxel_coords=voxel_results["voxel_coords_world"],
                voxel_materials=np.column_stack(
                    [
                        voxel_results["youngs_modulus"],
                        voxel_results["poisson_ratio"],
                        voxel_results["density"],
                    ]
                ),
            )

            # Interpolate to splat centers
            query_materials, query_distances = upsampler.interpolate_to_gaussians(
                gaussian_model
            )

            # Create final results with splat-level materials
            results = {
                "youngs_modulus": query_materials[:, 0],
                "poisson_ratio": query_materials[:, 1],
                "density": query_materials[:, 2],
                "query_coords_world": gaussian_model.get_xyz.detach().cpu().numpy(),
                "query_distances": query_distances,
                "voxel_coords_world": voxel_results[
                    "voxel_coords_world"
                ],  # Keep for reference
                "num_voxels": voxel_results["num_voxels"],
                "num_query_points": num_splats,
            }

            print(f"✓ Evaluated materials at {num_splats:,} splat centers")
        elif query_points == "voxel_centers" or query_points is None:
            # Return voxel-level results (original behavior)
            results = voxel_results
        elif isinstance(query_points, np.ndarray):
            # Step 3: Evaluate materials at custom query points
            print(
                f"Step 3: Evaluating materials at {len(query_points)} custom query points..."
            )

            # Create upsampler from voxel results
            upsampler = MaterialUpsampler(
                voxel_coords=voxel_results["voxel_coords_world"],
                voxel_materials=np.column_stack(
                    [
                        voxel_results["youngs_modulus"],
                        voxel_results["poisson_ratio"],
                        voxel_results["density"],
                    ]
                ),
            )

            # Interpolate to custom query points
            query_materials, query_distances = upsampler.interpolate(query_points)

            # Create final results with custom query point materials
            results = {
                "youngs_modulus": query_materials[:, 0],
                "poisson_ratio": query_materials[:, 1],
                "density": query_materials[:, 2],
                "query_coords_world": query_points,  # Primary coordinates for this case
                "voxel_coords_world": voxel_results[
                    "voxel_coords_world"
                ],  # Keep for reference
                "query_distances": query_distances,
                "num_voxels": voxel_results["num_voxels"],
                "num_query_points": len(query_points),
            }

            print(f"✓ Evaluated materials at {len(query_points):,} custom query points")
        else:
            raise ValueError(
                f"Invalid query_points value: {query_points}. Must be 'splat_centers', 'voxel_centers', None, or numpy array."
            )

        print("✓ Material estimation complete!")
        return results

    @torch.inference_mode()
    def get_custom_materials(
        self,
        obj_3d: Any,
        render_func: callable,
        voxelize_func: callable,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        High-level API for custom 3D representations with custom render/voxelize functions.

        This method demonstrates how to use your own 3D representation with custom
        rendering and voxelization functions.

        Args:
            obj_3d: Your custom 3D object/representation
            render_func: Custom rendering function with signature:
                render_func(obj_3d, output_dir, num_views, image_size, **kwargs) -> List[Dict]
            voxelize_func: Custom voxelization function with signature:
                voxelize_func(obj_3d, output_dir, **kwargs) -> np.ndarray
            output_dir: Output directory for intermediate files
            **kwargs: Additional arguments passed to render_func and voxelize_func

        Returns:
            Dictionary with material properties

        Example usage:
            def my_render_func(my_mesh, output_dir, num_views, image_size, **kwargs):
                # Your rendering logic here
                return frames_metadata  # List[Dict] with required keys

            def my_voxelize_func(my_mesh, output_dir, **kwargs):
                # Your voxelization logic here
                return voxel_centers  # np.ndarray shape (N, 3)

            results = model.get_custom_materials(
                obj_3d=my_mesh,
                render_func=my_render_func,
                voxelize_func=my_voxelize_func
            )
        """
        if output_dir is None:
            output_dir = f"/tmp/Vomp_custom_{id(obj_3d)}"

        print("=== Vomp: Custom Material Estimation ===")

        # Step 1: Extract features using custom functions
        print("Step 1: Extracting features with custom functions...")
        coords, features = self.get_features(
            obj_3d=obj_3d,
            render_func=render_func,
            voxelize_func=voxelize_func,
            output_dir=output_dir,
            **kwargs,
        )

        # Step 2: Run inference on the features
        print("Step 2: Running material inference...")
        predict_kwargs = {}
        if "max_voxels" in kwargs and kwargs["max_voxels"] is not None:
            predict_kwargs["max_voxels"] = kwargs["max_voxels"]
        results = self.predict_materials(coords, features, **predict_kwargs)

        print("✓ Material estimation complete!")
        return results

    def load_features(self, features_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load previously saved DINO features from disk.

        Args:
            features_path: Path to saved features NPZ file

        Returns:
            Tuple of (coordinates, features) tensors
        """
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        return self._load_saved_features(features_path)

    # Helper methods
    @torch.inference_mode()
    def _simple_render(
        self, gaussian_model: "Gaussian", camera, bg_color, render_alpha=True
    ) -> Dict[str, torch.Tensor]:
        """
        Render single view of Gaussian splat using differentiable rasterization.

        Core rendering function that rasterizes Gaussian splats to images using
        GPU-accelerated splatting. Supports both RGB and alpha channel rendering
        for proper transparency handling.

        Args:
            gaussian_model: Gaussian splat model containing positions, colors, etc.
            camera: Camera configuration object with intrinsics and extrinsics
            bg_color: Background color as 3D tensor [R, G, B]
            render_alpha: Whether to compute alpha channel for transparency

        Returns:
            Dictionary containing render outputs:
                - 'render': RGB image tensor (3, H, W)
                - 'alpha': Alpha channel tensor (1, H, W) if render_alpha=True
                - 'viewspace_points': Screen space points for gradients
                - 'visibility_filter': Mask of visible Gaussians
                - 'radii': Projected radii of Gaussians
        """

        # Initialize screen space points for gradients
        screenspace_points = (
            torch.zeros_like(
                gaussian_model.get_xyz,
                dtype=gaussian_model.get_xyz.dtype,
                requires_grad=True,
                device=self.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Calculate FOV
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)

        # Rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=gaussian_model.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gaussian_model.get_xyz
        means2D = screenspace_points
        opacity = gaussian_model.get_opacity
        scales = gaussian_model.get_scaling
        rotations = gaussian_model.get_rotation
        shs = gaussian_model.get_features

        # Render with background color
        rendered_rgb, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        result = {
            "render": rendered_rgb,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

        if render_alpha:
            # Render with black background to compute alpha channel
            black_bg = torch.zeros(3, dtype=torch.float32, device=self.device)
            raster_settings_black = GaussianRasterizationSettings(
                image_height=int(camera.image_height),
                image_width=int(camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=black_bg,
                scale_modifier=1.0,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform,
                sh_degree=gaussian_model.active_sh_degree,
                campos=camera.camera_center,
                prefiltered=False,
                debug=False,
            )

            rasterizer_black = GaussianRasterizer(raster_settings=raster_settings_black)
            rendered_black, _ = rasterizer_black(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=None,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )

            # Compute alpha from difference between white and black backgrounds
            alpha = 1.0 - (rendered_rgb - rendered_black).mean(dim=0, keepdim=True)
            alpha = torch.clamp(alpha, 0.0, 1.0)

            result["alpha"] = alpha

        return result

    def _normalize_gaussian(self, gaussian_model: "Gaussian") -> "Gaussian":
        """
        Normalize Gaussian splat to standard coordinate system.

        Transforms the Gaussian splat coordinates and scaling parameters to fit
        within the [-0.5, 0.5] coordinate system used by the model. This ensures
        consistent spatial encoding regardless of original object scale.

        Args:
            gaussian_model: Input Gaussian splat model to normalize

        Returns:
            Modified Gaussian model with normalized coordinates and scaling
        """
        xyz = gaussian_model.get_xyz
        min_vals = xyz.min(dim=0, keepdim=True)[0]
        max_vals = xyz.max(dim=0, keepdim=True)[0]
        center = (min_vals + max_vals) / 2
        extent = (max_vals - min_vals).max()
        scale_factor = 0.98 / extent

        normalized_xyz = (xyz - center) * scale_factor
        normalized_scaling = gaussian_model.get_scaling * scale_factor

        gaussian_model.from_xyz(normalized_xyz)
        gaussian_model.from_scaling(normalized_scaling)

        return gaussian_model

    def _voxelize_gaussian(
        self,
        gaussian_model: "Gaussian",
        output_dir: str,
        method: str = "centers",
        level: int = 6,
        iso: float = 11.345,
        tol: float = 1.0 / 8.0,
        step: int = 10,
        opacity_threshold: float = 0.35,
    ) -> np.ndarray:
        """
        Convert Gaussian splat to discrete voxel representation.

        Transforms continuous Gaussian splat representation into discrete voxels
        for material property prediction. Supports two methods with different
        accuracy/speed trade-offs.

        Args:
            gaussian_model: Gaussian splat model with normalized coordinates
            output_dir: Directory to save voxel data and metadata
            method: Voxelization algorithm:
                - 'centers': Fast, uses only Gaussian center positions
                - 'kaolin': Accurate, integrates full Gaussian parameters
            level: Voxel grid resolution as power of 2 (2^level grid size)
            iso: Iso-surface value for kaolin voxelization
            tol: Numerical tolerance for kaolin voxelization
            step: Number of integration samples per voxel (kaolin only)
            opacity_threshold: Minimum opacity to include voxel (kaolin only)

        Returns:
            Array of voxel center coordinates in world space (N, 3)
        """
        voxels_dir = os.path.join(output_dir, "voxels")
        os.makedirs(voxels_dir, exist_ok=True)

        if method == "centers":
            return self._voxelize_gaussian_centers(gaussian_model, voxels_dir)
        elif method == "kaolin":
            return self._voxelize_gaussian_kaolin(
                gaussian_model, voxels_dir, level, iso, tol, step, opacity_threshold
            )
        else:
            raise ValueError(
                f"Unknown voxelization method: {method}. Use 'centers' or 'kaolin'"
            )

    def _voxelize_gaussian_centers(
        self, gaussian_model: "Gaussian", voxels_dir: str
    ) -> np.ndarray:
        """
        Fast voxelization using only Gaussian center positions.

        Converts Gaussian splat centers to discrete voxel grid by quantizing
        positions to nearest voxel cells. Simple and fast but doesn't consider
        Gaussian shape or opacity information.

        Args:
            gaussian_model: Gaussian splat model with normalized coordinates
            voxels_dir: Directory to save voxel PLY file

        Returns:
            Array of unique voxel center coordinates (N, 3)
        """
        # Get Gaussian centers in CPU numpy
        centers = gaussian_model.get_xyz.detach().cpu().numpy()

        # Handle empty case
        if len(centers) == 0:
            print("No Gaussians to voxelize")
            return np.empty((0, 3), dtype=np.float32)

        # Clamp coordinates to valid range
        if centers.min() < -0.5 or centers.max() > 0.5:
            centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)

        # Convert world coordinates to 64x64x64 voxel indices
        voxel_indices = ((centers + 0.5) * 64).astype(np.int32)
        voxel_indices = np.clip(voxel_indices, 0, 63)

        # Unique voxel indices, then convert back to world-space voxel centers
        unique_indices = np.unique(voxel_indices, axis=0)
        voxel_centers = unique_indices.astype(np.float32) / 64.0 - 0.5

        # Save as PLY for compatibility
        voxel_path = os.path.join(voxels_dir, "voxels.ply")
        self._save_voxels_ply(voxel_centers, voxel_path)

        print(f"Voxelized to {len(voxel_centers)} voxels (centers method)")
        return voxel_centers

    def _voxelize_gaussian_kaolin(
        self,
        gaussian_model: "Gaussian",
        voxels_dir: str,
        level: int = 6,
        iso: float = 11.345,
        tol: float = 1.0 / 8.0,
        step: int = 10,
        opacity_threshold: float = 0.35,
    ) -> np.ndarray:
        """
        Accurate voxelization using full Gaussian parameters with Kaolin.

        Uses Kaolin's GPU-accelerated Gaussian-to-voxel conversion that considers
        complete Gaussian parameters (position, scale, rotation, opacity) to
        compute accurate voxel occupancy through numerical integration.

        Args:
            gaussian_model: Gaussian splat model with normalized coordinates
            voxels_dir: Directory to save voxel data and opacity information
            level: Voxel grid resolution level (2^level), e.g., 6 → 64³ grid
            iso: Iso-surface value for voxel generation
            tol: Numerical tolerance for integration accuracy
            step: Number of sample points per voxel for opacity integration
            opacity_threshold: Minimum opacity required to include voxel

        Returns:
            Array of voxel center coordinates for high-opacity regions (N, 3)

        Note: Slower but recommended for accurate material prediction
        """
        # Extract Gaussian parameters
        xyz = gaussian_model.get_xyz  # (N, 3)
        scales = gaussian_model.get_scaling  # (N, 3)
        rots = gaussian_model.get_rotation  # (N, 4) quaternions
        opacities = gaussian_model.get_opacity  # (N,)

        # Handle empty case
        if xyz.shape[0] == 0:
            print("No Gaussians to voxelize")
            return np.empty((0, 3), dtype=np.float32)

        print(f"Using Kaolin voxelization with {xyz.shape[0]} Gaussians...")
        print(f"Resolution: {2**level}^3, opacity_threshold: {opacity_threshold}")

        # Use Kaolin's Gaussian to voxel grid conversion
        voxel_coords, voxel_opacities = gs_ops.gs_to_voxelgrid(
            xyz, scales, rots, opacities, level=level, iso=iso, tol=tol, step=step
        )

        # Filter by opacity threshold
        mask = voxel_opacities >= opacity_threshold
        voxel_coords = voxel_coords[mask].contiguous()
        voxel_opacities = voxel_opacities[mask].contiguous()

        # Handle case where no voxels meet the threshold
        if voxel_coords.shape[0] == 0:
            print(
                f"Voxelized to 0 voxels (kaolin method) - no voxels above threshold {opacity_threshold}"
            )
            return np.empty((0, 3), dtype=np.float32)

        # Convert voxel coordinates to world space centers
        # Kaolin returns integer voxel coordinates, convert to world space [-0.5, 0.5]
        voxel_coords_cpu = voxel_coords.cpu().numpy()
        voxel_centers = voxel_coords_cpu.astype(np.float32) / (2**level) - 0.5

        # Save as PLY for compatibility
        voxel_path = os.path.join(voxels_dir, "voxels.ply")
        self._save_voxels_ply(voxel_centers, voxel_path)

        # Also save opacity information for analysis
        voxel_opacity_path = os.path.join(voxels_dir, "voxel_opacities.npz")
        np.savez_compressed(
            voxel_opacity_path,
            voxel_coords=voxel_coords_cpu,
            voxel_opacities=voxel_opacities.cpu().numpy(),
            voxel_centers=voxel_centers,
        )

        print(f"Voxelized to {len(voxel_centers)} voxels (kaolin method)")
        print(
            f"Opacity range: [{voxel_opacities.min():.3f}, {voxel_opacities.max():.3f}]"
        )
        return voxel_centers

    def _save_voxels_ply(self, voxel_centers: np.ndarray, output_path: str):
        """
        Save voxel centers to PLY format file.

        Exports voxel center coordinates to PLY format for visualization
        and pipeline compatibility. Creates directory structure as needed.

        Args:
            voxel_centers: Array of 3D voxel center positions (N, 3)
            output_path: Full path for output PLY file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_ply_vertices(voxel_centers, output_path, binary=True)

    def _load_voxel_centers(self, voxel_path: str) -> np.ndarray:
        """
        Load voxel center coordinates from PLY format file.

        Reads previously saved voxel positions for feature extraction
        or analysis. Compatible with files created by _save_voxels_ply.

        Args:
            voxel_path: Path to PLY file containing voxel data

        Returns:
            Array of 3D voxel center positions (N, 3)

        Raises:
            FileNotFoundError: If PLY file doesn't exist
            ValueError: If PLY file format is invalid
        """
        return read_ply_vertices(voxel_path)

    def _extract_dino_features(
        self,
        output_dir: str,
        voxel_centers: np.ndarray,
        frames_metadata: List[Dict],
        model_name: str = "dinov2_vitl14_reg",
        batch_size: int = 16,
        image_size: int = 518,
        save_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from rendered images using DINO vision transformer.

        Projects voxel positions to image coordinates across multiple views and
        samples DINO patch features at those locations. Features are aggregated
        across views to create robust visual representations for each voxel.

        Args:
            output_dir: Directory containing rendered images and to save features
            voxel_centers: 3D positions of voxels in world coordinates (N, 3)
            frames_metadata: List of frame info with camera transforms and paths
            model_name: DINO model variant to use for feature extraction
            batch_size: Number of images to process simultaneously
            image_size: Expected size of input images (assumed square)
            save_features: Whether to cache extracted features to disk

        Returns:
            Tuple of (coordinates, features):
                - coordinates: Voxel indices in sparse tensor format (N, 4)
                - features: Aggregated DINO features per voxel (N, feature_dim)
        """

        # Create features directory
        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_path = os.path.join(features_dir, f"{model_name}.npz")

        # Try to load existing features first
        if os.path.exists(features_path):
            print(f"Loading existing features from {features_path}")
            return self._load_saved_features(features_path)

        print(f"Extracting new DINO features...")

        # Initialize lazy DINO
        dino = LazyLoadDino(
            model_name=model_name, device=self.device, use_trt=self.use_trt
        )

        # Convert voxel centers to tensor coordinates
        positions = torch.from_numpy(voxel_centers).float().to(self.device)
        indices = ((positions + 0.5) * 64).long()
        indices = torch.clamp(indices, 0, 63)

        # Convert to int32 for sparse tensor coordinates
        indices = indices.int()

        # Create coordinates tensor (batch_idx, x, y, z)
        batch_dim = torch.zeros(
            (indices.shape[0], 1), dtype=torch.int32, device=self.device
        )
        coords = torch.cat([batch_dim, indices], dim=1)

        # Process rendered images
        data = []
        renders_dir = os.path.join(output_dir, "renders")

        for frame in frames_metadata:
            image_path = os.path.join(renders_dir, frame["file_path"])

            if not os.path.exists(image_path):
                continue

            # Load and preprocess image
            image = Image.open(image_path)
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255

            if image.shape[2] == 4:
                image = image[:, :, :3] * image[:, :, 3:]
            else:
                image = image[:, :, :3]

            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Camera matrices
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = frame["camera_angle_x"]
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(
                torch.tensor(fov, dtype=torch.float32),
                torch.tensor(fov, dtype=torch.float32),
            )

            data.append(
                {"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics}
            )

        if len(data) == 0:
            raise ValueError("No valid rendered images found")

        # Apply transforms and extract features
        transform = dino.get_transform()
        for datum in data:
            datum["image"] = transform(datum["image"])

        n_patch = dino.n_patch
        patchtokens_lst = []
        uv_lst = []

        # Process in batches
        for i in range(0, len(data), batch_size):
            batch_data = data[i : i + batch_size]
            bs = len(batch_data)

            batch_images = torch.stack([d["image"] for d in batch_data]).to(self.device)
            batch_extrinsics = torch.stack([d["extrinsics"] for d in batch_data]).to(
                self.device
            )
            batch_intrinsics = torch.stack([d["intrinsics"] for d in batch_data]).to(
                self.device
            )

            # Extract DINO features
            model = dino.get_model()
            with torch.inference_mode():
                features = model(batch_images, is_training=True)

            # Project voxels to image coordinates
            uv = (
                utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[
                    0
                ]
                * 2
                - 1
            )

            # Get patch tokens
            patchtokens = (
                features["x_prenorm"][:, model.num_register_tokens + 1 :]
                .permute(0, 2, 1)
                .reshape(bs, 1024, n_patch, n_patch)
            )

            patchtokens_lst.append(patchtokens)
            uv_lst.append(uv)

        # Concatenate all features
        patchtokens = torch.cat(patchtokens_lst, dim=0)
        uv = torch.cat(uv_lst, dim=0)

        # Sample features at voxel locations
        with torch.inference_mode():
            sampled_features = (
                F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(2)
                .permute(0, 2, 1)
            )

        # Aggregate features across views: mean in float32, then cast to float16 for storage
        arr = sampled_features.cpu().numpy()  # float32
        arr_mean_f16 = np.mean(arr, axis=0).astype(np.float16)

        # Save features to disk if requested
        if save_features:
            pack = {
                "indices": indices.cpu().numpy().astype(np.uint8),
                "patchtokens": arr_mean_f16,
            }
            np.savez_compressed(features_path, **pack)
            print(f"Saved features to {features_path}")
            print(f"Feature shape: {pack['patchtokens'].shape}")
            print(f"Voxel indices shape: {pack['indices'].shape}")

        # Convert to float32 tensor for inference
        features = torch.from_numpy(arr_mean_f16).to(self.device).float()

        return coords, features

    def _load_saved_features(
        self, features_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load cached DINO features and coordinates from disk.

        Reads previously extracted and saved DINO features from compressed
        NPZ format. Reconstructs sparse tensor coordinates for use in model.

        Args:
            features_path: Path to compressed NPZ features file

        Returns:
            Tuple of (coordinates, features):
                - coordinates: Sparse tensor coordinates (N, 4) [batch, x, y, z]
                - features: DINO feature vectors (N, feature_dim)
        """
        data = np.load(features_path)

        # Load indices and convert to coordinates
        indices = torch.tensor(data["indices"]).int()
        patchtokens = torch.tensor(data["patchtokens"]).float()

        # Create coordinates tensor (batch_idx, x, y, z)
        batch_dim = torch.zeros((indices.shape[0], 1), dtype=torch.int32)
        coords = torch.cat([batch_dim, indices], dim=1)

        # Move to device
        coords = coords.to(self.device)
        features = patchtokens.to(self.device)

        print(f"Loaded features:")
        print(f"  Voxels: {indices.shape[0]}")
        print(f"  Feature dimension: {patchtokens.shape[1]}")
        print(f"  Voxel index range: {indices.min().item()} to {indices.max().item()}")

        return coords, features

    def _map_voxels_to_splats(
        self,
        gaussian_model: "Gaussian",
        voxel_coords: np.ndarray,
        voxel_results: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Map voxel materials to closest splats"""
        # Get splat positions
        splat_positions = gaussian_model.get_xyz.detach().cpu().numpy()

        # Find closest voxel for each splat

        distances = cdist(splat_positions, voxel_coords)
        closest_voxel_indices = np.argmin(distances, axis=1)

        # Map materials to splats
        splat_materials = {
            "splat_positions": splat_positions,
            "youngs_modulus": voxel_results["youngs_modulus"][closest_voxel_indices],
            "poisson_ratio": voxel_results["poisson_ratio"][closest_voxel_indices],
            "density": voxel_results["density"][closest_voxel_indices],
            "closest_voxel_distance": distances[
                np.arange(len(splat_positions)), closest_voxel_indices
            ],
            "num_splats": len(splat_positions),
            "num_voxels": len(voxel_coords),
        }

        return splat_materials

    def _install_blender(self):
        """Install Blender 3.0.1 for mesh rendering if not already present."""
        BLENDER_LINK = "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz"
        BLENDER_INSTALLATION_PATH = "/tmp"
        BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"

        if not os.path.exists(BLENDER_PATH):
            print("Installing Blender...")
            os.system("sudo apt-get update")
            os.system(
                "sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6"
            )
            os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
            os.system(
                f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}"
            )
            print(f"✓ Blender installed at: {BLENDER_PATH}")

        return BLENDER_PATH

    def _render_views_sequential(
        self,
        views: List[Dict],
        mesh_path: str,
        renders_dir: str,
        image_size: int,
        blender_path: str,
        use_gpu: bool,
        gpu_device: str,
    ) -> List[Dict]:
        """Render views sequentially using single Blender process."""
        # Call Blender rendering script
        args = [
            blender_path,
            "-b",
            "-P",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "dataset_toolkits",
                "blender_script",
                "render.py",
            ),
            "--",
            "--views",
            json.dumps(views),
            "--object",
            os.path.abspath(os.path.expanduser(mesh_path)),
            "--resolution",
            str(image_size),
            "--output_folder",
            renders_dir,
            "--engine",
            "CYCLES",
            "--save_mesh",
        ]

        # Add GPU rendering arguments
        if use_gpu:
            args.extend(["--use_gpu", "--gpu_device", gpu_device])
            print(f"✓ GPU rendering enabled: {gpu_device}")

        if mesh_path.endswith(".blend"):
            args.insert(1, mesh_path)

        call(args, stdout=DEVNULL, stderr=DEVNULL)

        # Load metadata
        transforms_path = os.path.join(renders_dir, "transforms.json")
        if os.path.exists(transforms_path):
            with open(transforms_path, "r") as f:
                transforms_data = json.load(f)
            return transforms_data.get("frames", [])
        else:
            raise RuntimeError(
                f"Blender did not create transforms.json at {transforms_path}"
            )

    def _render_views_parallel(
        self,
        views: List[Dict],
        mesh_path: str,
        renders_dir: str,
        image_size: int,
        blender_path: str,
        use_gpu: bool,
        gpu_device: str,
        num_jobs: int,
    ) -> List[Dict]:
        """Render views in parallel using multiple Blender processes."""

        # Split views into chunks for parallel processing
        chunk_size = max(1, len(views) // num_jobs)
        view_chunks = [
            views[i : i + chunk_size] for i in range(0, len(views), chunk_size)
        ]

        print(
            f"Splitting {len(views)} views into {len(view_chunks)} chunks (max {chunk_size} views per job)"
        )

        # Create temporary directories for each job
        temp_dirs = []
        for i in range(len(view_chunks)):
            temp_dir = os.path.join(renders_dir, f"temp_job_{i}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_dirs.append(temp_dir)

        # Create worker arguments with proper view indexing
        worker_args = []
        view_index_offset = 0
        for i, (chunk, temp_dir) in enumerate(zip(view_chunks, temp_dirs)):
            worker_args.append(
                {
                    "job_id": i,
                    "views": chunk,
                    "view_index_offset": view_index_offset,
                    "mesh_path": mesh_path,
                    "temp_dir": temp_dir,
                    "image_size": image_size,
                    "blender_path": blender_path,
                    "use_gpu": use_gpu,
                    "gpu_device": gpu_device,
                    "render_script": os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "dataset_toolkits",
                        "blender_script",
                        "render.py",
                    ),
                }
            )
            view_index_offset += len(chunk)

        # Run parallel rendering
        with multiprocessing.Pool(processes=num_jobs) as pool:
            results = pool.map(self._render_chunk_worker, worker_args)

        # Merge results and rename files to avoid conflicts
        all_frames = []
        global_view_index = 0

        for i, frames in enumerate(results):
            if frames is None:
                raise RuntimeError(f"Job {i} failed to render")

            # Copy rendered images to main renders directory with correct global numbering
            temp_dir = temp_dirs[i]
            for j, frame in enumerate(frames):
                # Original filename from job (e.g., "000.png", "001.png")
                old_path = os.path.join(temp_dir, frame["file_path"])

                # New filename with global indexing (e.g., "000.png", "038.png", "075.png")
                new_filename = f"{global_view_index:03d}.png"
                new_path = os.path.join(renders_dir, new_filename)

                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    # Update frame metadata with new filename
                    frame["file_path"] = new_filename
                else:
                    print(f"Warning: Expected file not found: {old_path}")

                global_view_index += 1

            all_frames.extend(frames)

            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Save merged transforms.json
        transforms_data = {
            "camera_angle_x": views[0]["fov"] if views else 0.0,
            "frames": all_frames,
        }

        transforms_path = os.path.join(renders_dir, "transforms.json")
        with open(transforms_path, "w") as f:
            json.dump(transforms_data, f, indent=2)

        print(
            f"✓ Merged {len(all_frames)} rendered views from {num_jobs} parallel jobs"
        )
        return all_frames

    @staticmethod
    def _render_chunk_worker(args):
        """
        Worker function for parallel mesh rendering using Blender.

        Renders a subset of camera views in a separate Blender process as part
        of parallel rendering pipeline. Each worker handles its assigned views
        independently to improve throughput.

        Args:
            args: Dictionary containing worker configuration:
                - job_id: Unique worker identifier for logging
                - views: List of camera view parameters to render
                - mesh_path: Path to input mesh file
                - temp_dir: Temporary directory for this worker's output
                - image_size: Target image resolution
                - blender_path: Path to Blender executable
                - use_gpu: Whether to use GPU acceleration
                - gpu_device: GPU device type string
                - render_script: Path to Blender rendering script

        Returns:
            List of frame metadata dictionaries for rendered views, or None on failure
        """
        job_id = args["job_id"]
        views = args["views"]
        mesh_path = args["mesh_path"]
        temp_dir = args["temp_dir"]
        image_size = args["image_size"]
        blender_path = args["blender_path"]
        use_gpu = args["use_gpu"]
        gpu_device = args["gpu_device"]
        render_script = args["render_script"]

        print(f"  Job {job_id}: Rendering {len(views)} views...")

        # Prepare Blender arguments
        blender_args = [
            blender_path,
            "-b",
            "-P",
            render_script,
            "--",
            "--views",
            json.dumps(views),
            "--object",
            os.path.expanduser(mesh_path),
            "--resolution",
            str(image_size),
            "--output_folder",
            temp_dir,
            "--engine",
            "CYCLES",
        ]

        # Add GPU rendering arguments
        if use_gpu:
            blender_args.extend(["--use_gpu", "--gpu_device", gpu_device])

        if mesh_path.endswith(".blend"):
            blender_args.insert(1, mesh_path)

        try:
            # Run Blender with error output to help debug

            process = Popen(blender_args, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(
                    f"  Job {job_id}: Blender error - returncode {process.returncode}"
                )
                if stderr:
                    print(f"  Job {job_id}: stderr: {stderr.decode()[:200]}...")
                return None

            # Load and return metadata
            transforms_path = os.path.join(temp_dir, "transforms.json")
            if os.path.exists(transforms_path):
                with open(transforms_path, "r") as f:
                    transforms_data = json.load(f)
                frames = transforms_data.get("frames", [])
                print(f"  Job {job_id}: ✓ Completed {len(frames)} views")
                return frames
            else:
                print(f"  Job {job_id}: ✗ Failed - no transforms.json created")
                print(
                    f"  Job {job_id}: temp_dir contents: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'N/A'}"
                )
                return None

        except Exception as e:
            print(f"  Job {job_id}: ✗ Failed with error: {e}")
            return None

    def render_mesh_views(
        self,
        mesh_path: str,
        output_dir: str,
        num_views: int = 150,
        image_size: int = 518,
        radius: float = 2.0,
        fov: float = 40.0,
        seed: Optional[int] = None,
        blender_path: Optional[str] = None,
        use_gpu: bool = True,
        gpu_device: str = "OPTIX",
        num_render_jobs: int = 1,
    ) -> List[Dict]:
        """
        Render multiple views of mesh using Blender for photorealistic results.

        Uses Blender's Cycles render engine to generate high-quality mesh renders
        from sampled camera positions. Supports GPU acceleration and parallel
        rendering for improved performance.

        Args:
            mesh_path: Path to mesh file (OBJ, PLY, STL, blend, etc.)
            output_dir: Directory to save rendered images and metadata
            num_views: Number of camera viewpoints to render
            image_size: Square image resolution in pixels
            radius: Camera distance from object center
            fov: Field of view angle in degrees
            seed: Random seed for reproducible camera sampling
            blender_path: Path to Blender executable (uses BLENDER_BIN env var if None)
            use_gpu: Whether to enable GPU acceleration (CUDA/OptiX/OpenCL)
            gpu_device: GPU compute device type for Blender
            num_render_jobs: Number of parallel Blender processes

        Returns:
            List of frame metadata dictionaries with camera poses and file paths

        Raises:
            EnvironmentError: If Blender executable not found
            FileNotFoundError: If mesh file doesn't exist
            RuntimeError: If rendering process fails
        """
        renders_dir = os.path.join(output_dir, "renders")
        os.makedirs(renders_dir, exist_ok=True)

        # Determine Blender path
        if blender_path is not None:
            BLENDER_PATH = blender_path
            if not os.path.exists(BLENDER_PATH):
                raise FileNotFoundError(
                    f"Blender executable not found at: {BLENDER_PATH}"
                )
        else:
            # Try to get from environment variable
            BLENDER_PATH = os.environ.get("BLENDER_BIN")
            if BLENDER_PATH is None:
                raise EnvironmentError(
                    "No Blender path provided and BLENDER_BIN environment variable not set. "
                    "Either provide blender_path parameter or set BLENDER_BIN environment variable. "
                    "You can install Blender using the install_env.sh script."
                )
            if not os.path.exists(BLENDER_PATH):
                raise FileNotFoundError(
                    f"Blender executable not found at environment path: {BLENDER_PATH}. "
                    "Please check your BLENDER_BIN environment variable or reinstall Blender using install_env.sh"
                )

        print(f"Rendering {num_views} mesh views with Blender...")

        # Sample camera views using shared utility
        yaws, pitchs, radius_list, fov_list = self._sample_camera_views(
            num_views, radius, fov, seed
        )

        # Convert to Blender format
        views = [
            {"yaw": y, "pitch": p, "radius": r, "fov": f / 180 * np.pi}
            for y, p, r, f in zip(yaws, pitchs, radius_list, fov_list)
        ]

        if num_render_jobs == 1:
            # Sequential rendering (original behavior)
            frames_metadata = self._render_views_sequential(
                views,
                mesh_path,
                renders_dir,
                image_size,
                BLENDER_PATH,
                use_gpu,
                gpu_device,
            )
        else:
            # Parallel rendering
            print(f"Using {num_render_jobs} parallel Blender processes...")
            frames_metadata = self._render_views_parallel(
                views,
                mesh_path,
                renders_dir,
                image_size,
                BLENDER_PATH,
                use_gpu,
                gpu_device,
                num_render_jobs,
            )

        # Save metadata for pipeline compatibility
        metadata_path = os.path.join(output_dir, "renders_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(frames_metadata, f, indent=2)

        print(f"✓ Rendered {len(frames_metadata)} views")
        return frames_metadata

    def render_views_replicator(
        self,
        asset_path: str,
        output_dir: str,
        num_views: int = 150,
        image_size: int = 518,
        radius: float = 2.0,
        fov: float = 40.0,
        seed: Optional[int] = None,
        isaac_sim_path: Optional[str] = None,
        render_mode: str = "path_tracing",
        rtx_settings_override: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Render multiple views using Isaac Sim Replicator (alternative to Blender).

        Uses NVIDIA Isaac Sim's Replicator API for GPU-accelerated rendering with
        RTX ray tracing or path tracing. Provides fast rendering with physically
        accurate lighting and materials.

        Args:
            asset_path: Path to USD asset file (.usd, .usda, .usdc, .ply converted to USD)
            output_dir: Directory to save rendered images and metadata
            num_views: Number of camera viewpoints to render
            image_size: Square image resolution in pixels
            radius: Camera distance from object center
            fov: Field of view angle in degrees
            seed: Random seed for reproducible camera sampling
            isaac_sim_path: Path to Isaac Sim executable (isaac-sim.sh).
                If None, uses ISAAC_SIM_PATH environment variable.
            render_mode: Rendering quality preset:
                - "fast": Real-time ray tracing (faster, good quality)
                - "path_tracing": Path tracing (slower, highest quality)
            rtx_settings_override: Optional dict to override specific RTX settings.
                Example: {"/rtx/pathtracing/spp": 512} for higher quality

        Returns:
            List of frame metadata dictionaries with camera poses and file paths

        Raises:
            EnvironmentError: If Isaac Sim path not provided and not in environment
            FileNotFoundError: If Isaac Sim executable not found
            RuntimeError: If rendering process fails

        Example:
            >>> # Using path tracing with default settings
            >>> frames = model.render_views_replicator(
            ...     asset_path="model.usd",
            ...     output_dir="./renders",
            ...     num_views=150,
            ...     isaac_sim_path="~/isaac-sim/isaac-sim.sh",
            ...     render_mode="path_tracing"
            ... )
            >>>
            >>> # Using fast mode with custom RTX settings
            >>> frames = model.render_views_replicator(
            ...     asset_path="model.usd",
            ...     output_dir="./renders",
            ...     isaac_sim_path="~/isaac-sim/isaac-sim.sh",
            ...     render_mode="fast",
            ...     rtx_settings_override={
            ...         "/rtx/pathtracing/spp": 512,  # Override samples
            ...     }
            ... )
        """

        renders_dir = os.path.join(output_dir, "renders")
        os.makedirs(renders_dir, exist_ok=True)

        # Determine Isaac Sim path
        if isaac_sim_path is not None:
            ISAAC_SIM_PATH = isaac_sim_path
            if not os.path.exists(os.path.expanduser(ISAAC_SIM_PATH)):
                raise FileNotFoundError(
                    f"Isaac Sim executable not found at: {ISAAC_SIM_PATH}"
                )
        else:
            # Try to get from environment variable
            ISAAC_SIM_PATH = os.environ.get("ISAAC_SIM_PATH")
            if ISAAC_SIM_PATH is None:
                raise EnvironmentError(
                    "No Isaac Sim path provided and ISAAC_SIM_PATH environment variable not set. "
                    "Either provide isaac_sim_path parameter or set ISAAC_SIM_PATH environment variable. "
                    "Example: export ISAAC_SIM_PATH=~/isaac-sim/isaac-sim.sh"
                )
            if not os.path.exists(os.path.expanduser(ISAAC_SIM_PATH)):
                raise FileNotFoundError(
                    f"Isaac Sim executable not found at environment path: {ISAAC_SIM_PATH}. "
                    "Please check your ISAAC_SIM_PATH environment variable."
                )

        print(f"Rendering {num_views} views with Isaac Sim Replicator...")
        print(f"Render mode: {render_mode}")

        # Sample camera views using shared utility
        yaws, pitchs, radius_list, fov_list = self._sample_camera_views(
            num_views, radius, fov, seed
        )

        # Call Replicator rendering
        frames_metadata = render_with_replicator(
            asset_path=asset_path,
            output_dir=output_dir,
            num_views=num_views,
            yaws=yaws,
            pitchs=pitchs,
            radius_list=radius_list,
            fov_list=fov_list,
            isaac_sim_path=ISAAC_SIM_PATH,
            resolution=(image_size, image_size),
            render_mode=render_mode,
            rtx_settings_override=rtx_settings_override,
            light_intensity=1000.0,
            normalize_object=True,
        )

        # Save metadata for pipeline compatibility
        metadata_path = os.path.join(output_dir, "renders_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(frames_metadata, f, indent=2)

        print(f"✓ Rendered {len(frames_metadata)} views with Replicator")
        return frames_metadata

    def _voxelize_mesh(
        self,
        mesh_path: str,
        output_dir: str,
        voxel_size: float = 1 / 64,
        max_voxels: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert mesh geometry to discrete voxel representation.

        Loads mesh file and generates voxel occupancy grid using trimesh
        voxelization. Handles both single meshes and complex scenes with
        multiple geometries. Normalizes to standard coordinate system.

        Args:
            mesh_path: Path to mesh file (supports OBJ, PLY, STL, etc.)
            output_dir: Directory to save voxel PLY file
            voxel_size: Voxel spacing in normalized coordinates (1/64 = 64³ grid)
            max_voxels: Maximum voxels to generate; subsamples if exceeded

        Returns:
            Array of voxel center coordinates in world space (N, 3)
        """
        voxels_dir = os.path.join(output_dir, "voxels")
        os.makedirs(voxels_dir, exist_ok=True)

        print(f"Voxelizing mesh: {mesh_path}")

        # Load and combine mesh geometry
        mesh = trimesh.load(mesh_path)
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            vertices, faces = mesh.vertices, mesh.faces
        else:
            # Mesh scene - combine all geometries
            vertices_list, faces_list = [], []
            vertex_offset = 0
            for geometry in mesh.geometry.values():
                if hasattr(geometry, "vertices") and hasattr(geometry, "faces"):
                    vertices_list.append(geometry.vertices)
                    faces_list.append(geometry.faces + vertex_offset)
                    vertex_offset += len(geometry.vertices)
            if not vertices_list:
                raise ValueError("No valid geometry found in mesh")
            vertices, faces = np.vstack(vertices_list), np.vstack(faces_list)

        # Step 1: Compute scale from original bbox
        bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
        bbox_size = (bbox_max - bbox_min).max()
        blender_scale = 1 / bbox_size

        # Step 2: Apply scale to vertices
        vertices_scaled = vertices * blender_scale

        # Step 3: Compute offset from SCALED bbox (this is what Blender does!)
        bbox_min_scaled = vertices_scaled.min(axis=0)
        bbox_max_scaled = vertices_scaled.max(axis=0)
        blender_offset = -(bbox_min_scaled + bbox_max_scaled) / 2

        # Step 4: Apply offset
        vertices_normalized = vertices_scaled + blender_offset
        vertices_normalized = np.clip(vertices_normalized, -0.5 + 1e-6, 0.5 - 1e-6)

        # Now voxelize the normalized mesh
        normalized_mesh = trimesh.Trimesh(vertices=vertices_normalized, faces=faces)
        voxel_grid = normalized_mesh.voxelized(pitch=voxel_size).fill()
        voxel_centers = voxel_grid.points

        # Subsample if too many voxels
        if max_voxels is not None and len(voxel_centers) > max_voxels:
            print(f"Subsampling voxels: {len(voxel_centers):,} -> {max_voxels:,}")
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(voxel_centers), max_voxels, replace=False)
            voxel_centers = voxel_centers[indices]

        indices = ((voxel_centers + 0.5) * 64).astype(np.int64)
        indices = np.clip(indices, 0, 63)
        discretized_positions = indices.astype(np.float32) / 64.0 - 0.5

        voxel_centers = discretized_positions

        # Save as PLY for compatibility with existing pipeline
        voxel_path = os.path.join(voxels_dir, "voxels.ply")
        self._save_voxels_ply(voxel_centers, voxel_path)

        print(f"Generated {len(voxel_centers)} voxels")
        return voxel_centers

    @torch.inference_mode()
    def get_mesh_materials(
        self,
        mesh_path: str,
        output_dir: Optional[str] = None,
        num_views: int = 150,
        image_size: int = 518,
        render_image_size: int = 512,
        radius: float = 2.0,
        fov: float = 40.0,
        seed: Optional[int] = None,
        voxel_size: float = 1 / 64,
        max_voxels: Optional[int] = 32768,
        blender_path: Optional[str] = None,
        query_points: Union[str, np.ndarray, None] = "mesh_vertices",
        use_gpu: bool = True,
        gpu_device: str = "OPTIX",
        num_render_jobs: int = 1,
        dino_batch_size: int = 16,
        return_original_scale: bool = False,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        High-level API for mesh material property inference.

        Args:
            mesh_path: Path to mesh file
            output_dir: Directory for intermediate files
            num_views: Camera views for rendering (default: 150, matches training data)
            image_size: Target image size for DINO feature extraction. Rendered images are
                resized to this resolution before processing (default: 518)
            render_image_size: Resolution for Blender rendering. Set to 512 to match training
                data pipeline where images were rendered at 512x512 then resized to 518x518
                for DINO feature extraction (default: 512)
            radius: Camera distance from object center (default: 2.0, matches training data)
            fov: Field of view in degrees (default: 40.0, matches training data)
            seed: Random seed for camera view sampling
            voxel_size: Voxel size for discretization (default: 1/64)
            max_voxels: Maximum voxels to generate
            blender_path: Path to Blender executable. If None, uses fallback installation.
            query_points: Where to evaluate material properties. Can be:
                - "mesh_vertices" (default): Evaluate at mesh vertices
                - "voxel_centers": Evaluate at voxel centers
                - numpy array (N, 3): Custom 3D coordinates for evaluation
                - None: Same as "voxel_centers"
            use_gpu: Whether to use GPU acceleration for Blender rendering
            gpu_device: GPU device type for Blender - "OPTIX", "CUDA", or "OPENCL"
            num_render_jobs: Number of parallel Blender processes for rendering (default: 1)
            dino_batch_size: Number of images to process simultaneously during DINO feature extraction (higher values use more GPU memory but may be faster)
            return_original_scale: If True, return coordinates in original mesh scale instead of
                normalized [-0.5, 0.5] space. Transformation parameters (center, scale) are included
                in results for reference. (default: False)
            **kwargs: Additional feature extraction arguments

        Returns:
            Dictionary containing material properties:
            - 'youngs_modulus': Young's modulus values at query points (Pa)
            - 'poisson_ratio': Poisson's ratio values at query points
            - 'density': Density values at query points (kg/m³)
            - 'query_coords_world': World coordinates of query points (normalized or original scale)
            - 'voxel_coords_world': World coordinates of voxels (normalized or original scale)
            - 'num_voxels': Number of voxels
            - 'num_query_points': Number of query points (when applicable)
            - 'transform_center': Center point used for normalization (when return_original_scale=True)
            - 'transform_scale': Scale factor used for normalization (when return_original_scale=True)
        """
        if output_dir is None:
            mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            output_dir = f"/tmp/Vomp_mesh_{mesh_name}"

        print("=== Vomp: Mesh Material Estimation ===")

        # Define built-in mesh rendering and voxelization functions
        def mesh_render_func(mesh_path, output_dir, num_views, image_size, **kwargs):
            return self.render_mesh_views(
                mesh_path,
                output_dir,
                num_views,
                render_image_size,
                blender_path=blender_path,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                num_render_jobs=num_render_jobs,
                **kwargs,
            )

        def mesh_voxelize_func(mesh_path, output_dir, **kwargs):
            return self._voxelize_mesh(
                mesh_path, output_dir, voxel_size=voxel_size, max_voxels=max_voxels
            )

        # Step 1: Extract features using built-in mesh functions
        print("Step 1: Extracting features...")

        # Prepare rendering parameters for kwargs
        render_kwargs = {
            "radius": radius,
            "fov": fov,
            "seed": seed,
        }

        coords, features = self.get_features(
            obj_3d=mesh_path,
            render_func=mesh_render_func,
            voxelize_func=mesh_voxelize_func,
            num_views=num_views,
            image_size=image_size,
            output_dir=output_dir,
            batch_size=dino_batch_size,
            **render_kwargs,
            **kwargs,
        )

        # Step 2: Run inference on the features
        print("Step 2: Running material inference...")
        predict_kwargs = {}
        if max_voxels is not None:
            predict_kwargs["max_voxels"] = max_voxels
        voxel_results = self.predict_materials(coords, features, **predict_kwargs)

        # Handle query_points parameter
        if query_points == "mesh_vertices":
            # Step 3: Evaluate materials at mesh vertices
            print("Step 3: Evaluating materials at mesh vertices...")

            # Load mesh to get vertices
            mesh = trimesh.load(mesh_path)
            if hasattr(mesh, "vertices"):
                mesh_vertices = mesh.vertices
            else:
                # Mesh scene - combine all vertices
                vertices_list = []
                for geometry in mesh.geometry.values():
                    if hasattr(geometry, "vertices"):
                        vertices_list.append(geometry.vertices)
                if not vertices_list:
                    raise ValueError("No vertices found in mesh")
                mesh_vertices = np.vstack(vertices_list)

            # Get transformation parameters
            center, scale = get_mesh_transform_params(mesh_path)

            # Normalize vertices to match model's coordinate system (same as _voxelize_mesh)
            normalized_vertices = (mesh_vertices - center) / scale
            normalized_vertices = np.clip(normalized_vertices, -0.5 + 1e-6, 0.5 - 1e-6)

            # Create upsampler from voxel results
            upsampler = MaterialUpsampler(
                voxel_coords=voxel_results["voxel_coords_world"],
                voxel_materials=np.column_stack(
                    [
                        voxel_results["youngs_modulus"],
                        voxel_results["poisson_ratio"],
                        voxel_results["density"],
                    ]
                ),
            )

            # Interpolate to mesh vertices
            vertex_materials, vertex_distances = upsampler.interpolate(
                normalized_vertices
            )

            # Apply inverse transformation
            if return_original_scale:
                output_coords = denormalize_coords(normalized_vertices, center, scale)
                output_voxel_coords = denormalize_coords(
                    voxel_results["voxel_coords_world"], center, scale
                )
            else:
                output_coords = normalized_vertices
                output_voxel_coords = voxel_results["voxel_coords_world"]

            # Create final results
            results = {
                "youngs_modulus": vertex_materials[:, 0],
                "poisson_ratio": vertex_materials[:, 1],
                "density": vertex_materials[:, 2],
                "query_coords_world": output_coords,
                "query_distances": vertex_distances,
                "voxel_coords_world": output_voxel_coords,
                "num_voxels": voxel_results["num_voxels"],
                "num_query_points": len(normalized_vertices),
            }

            # Add transformation parameters if in original scale
            if return_original_scale:
                results["transform_center"] = center
                results["transform_scale"] = scale

            print(
                f"✓ Evaluated materials at {len(normalized_vertices):,} mesh vertices"
            )
        elif query_points == "voxel_centers" or query_points is None:
            # Return voxel-level results (original behavior)
            results = voxel_results

            # Apply inverse transformation
            if return_original_scale:
                # Get transformation parameters
                center, scale = get_mesh_transform_params(mesh_path)

                # Transform coordinates back to original scale
                results["voxel_coords_world"] = denormalize_coords(
                    results["voxel_coords_world"], center, scale
                )
                results["transform_center"] = center
                results["transform_scale"] = scale

        elif isinstance(query_points, np.ndarray):
            # Step 3: Evaluate materials at custom query points
            print(
                f"Step 3: Evaluating materials at {len(query_points)} custom query points..."
            )

            # Create upsampler from voxel results
            upsampler = MaterialUpsampler(
                voxel_coords=voxel_results["voxel_coords_world"],
                voxel_materials=np.column_stack(
                    [
                        voxel_results["youngs_modulus"],
                        voxel_results["poisson_ratio"],
                        voxel_results["density"],
                    ]
                ),
            )

            # Interpolate to custom query points
            query_materials, query_distances = upsampler.interpolate(query_points)

            # Apply inverse transformation
            if return_original_scale:
                # Get transformation parameters
                center, scale = get_mesh_transform_params(mesh_path)

                output_coords = denormalize_coords(query_points, center, scale)
                output_voxel_coords = denormalize_coords(
                    voxel_results["voxel_coords_world"], center, scale
                )
            else:
                output_coords = query_points
                output_voxel_coords = voxel_results["voxel_coords_world"]

            # Create final results
            results = {
                "youngs_modulus": query_materials[:, 0],
                "poisson_ratio": query_materials[:, 1],
                "density": query_materials[:, 2],
                "query_coords_world": output_coords,
                "query_distances": query_distances,
                "voxel_coords_world": output_voxel_coords,
                "num_voxels": voxel_results["num_voxels"],
                "num_query_points": len(query_points),
            }

            # Add transformation parameters if in original scale
            if return_original_scale:
                results["transform_center"] = center
                results["transform_scale"] = scale

            print(f"✓ Evaluated materials at {len(query_points):,} custom query points")
        else:
            raise ValueError(
                f"Invalid query_points value: {query_points}. Must be 'mesh_vertices', 'voxel_centers', None, or numpy array."
            )

        print("✓ Material estimation complete!")
        return results

    @torch.inference_mode()
    def get_usd_materials(
        self,
        usd_path: str,
        mesh_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_views: int = 150,
        image_size: int = 518,
        render_image_size: int = 512,
        radius: float = 2.0,
        fov: float = 40.0,
        seed: Optional[int] = None,
        voxel_size: float = 1 / 64,
        max_voxels: Optional[int] = 32768,
        isaac_sim_path: Optional[str] = None,
        render_mode: str = "path_tracing",
        rtx_settings_override: Optional[Dict[str, Any]] = None,
        query_points: Union[str, np.ndarray, None] = "voxel_centers",
        dino_batch_size: int = 16,
        use_simready_usd_format: bool = False,
        blender_path: Optional[str] = None,
        gpu_device: str = "OPTIX",
        num_render_jobs: int = 1,
        return_original_scale: bool = False,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        High-level API for USD asset material property inference.

        USD files can come in many different formats with varying internal structures,
        materials, and organization. In the general case, users should provide both:
        1. The USD file for rendering
        2. An extracted mesh file (OBJ/PLY/STL) for voxelization

        However, if your USD file is in the SimReady format (like the USD files in our
        dataset), you can set use_simready_usd_format=True.
        This will automatically:
        - Extract mesh geometry, materials, and UV coordinates from the USD
        - Convert to OBJ format matching the dataset preprocessing pipeline
        - Render with Blender using the exact same pipeline used to create training data

        Args:
            usd_path: Path to USD asset file (.usd, .usda, .usdc)
            mesh_path: Path to mesh file (OBJ, PLY, STL, etc.) for voxelization.
                If None and use_simready_usd_format=True, voxelizes the converted OBJ.
                Required if use_simready_usd_format=False (for Replicator mode).
            output_dir: Directory for intermediate files
            num_views: Camera views for rendering (default: 150, matches training data)
            image_size: Target image size for DINO feature extraction. Rendered images are
                resized to this resolution before processing (default: 518)
            render_image_size: Resolution for rendering. Set to 512 to match training
                data pipeline where images were rendered at 512x512 then resized to 518x518
                for DINO feature extraction (default: 512)
            radius: Camera distance from object center (default: 2.0, matches training data)
            fov: Field of view in degrees (default: 40.0, matches training data)
            seed: Random seed for camera view sampling
            voxel_size: Voxel size for discretization
            max_voxels: Maximum voxels to generate
            isaac_sim_path: Path to Isaac Sim executable (isaac-sim.sh).
                Only used if use_simready_usd_format=False.
                If None, uses ISAAC_SIM_PATH environment variable.
            render_mode: Rendering quality preset (Replicator only):
                - "fast": Real-time ray tracing (faster, good quality)
                - "path_tracing": Path tracing (slower, highest quality)
            rtx_settings_override: Optional dict to override specific RTX settings (Replicator only).
                Example: {"/rtx/pathtracing/spp": 512} for higher quality
            query_points: Where to evaluate material properties. Can be:
                - "voxel_centers" (default): Evaluate at voxel centers
                - numpy array (N, 3): Custom 3D coordinates for evaluation
                - None: Same as "voxel_centers"
            dino_batch_size: Number of images to process simultaneously during
                DINO feature extraction
            use_simready_usd_format: If True, treats the USD as SimReady format
                (like assets in datasets/raw/simready). Automatically converts USD to OBJ
                with materials/textures and renders with Blender, matching the dataset
                preprocessing pipeline. If False (default), uses Isaac Sim Replicator.
            blender_path: Path to Blender executable (only used if use_simready_usd_format=True).
                If None, attempts to auto-detect.
            gpu_device: GPU rendering device for Blender (only used if use_simready_usd_format=True).
                Options: "OPTIX" (default), "CUDA", "METAL", "NONE"
            num_render_jobs: Number of parallel Blender render jobs (only used if use_simready_usd_format=True).
            return_original_scale: If True, return coordinates in original mesh scale instead of
                normalized [-0.5, 0.5] space. Transformation parameters (center, scale) are included
                in results for reference. (default: False)
            **kwargs: Additional feature extraction arguments

        Returns:
            Dictionary containing material properties:
            - 'youngs_modulus': Young's modulus values (Pa)
            - 'poisson_ratio': Poisson's ratio values
            - 'density': Density values (kg/m³)
            - 'voxel_coords_world': World coordinates of voxels (normalized or original scale)
            - 'query_coords_world': World coordinates of query points (when using custom points)
            - 'num_voxels': Number of voxels
            - 'num_query_points': Number of query points (when applicable)
            - 'transform_center': Center point used for normalization (when return_original_scale=True)
            - 'transform_scale': Scale factor used for normalization (when return_original_scale=True)

        Example:
            >>> # Replicator mode (default) - requires separate mesh file
            >>> model = Vomp.from_checkpoint(config_path)
            >>> results = model.get_usd_materials(
            ...     usd_path="model.usd",
            ...     mesh_path="model.ply",
            ...     isaac_sim_path="~/isaac-sim/isaac-sim.sh",
            ...     render_mode="path_tracing"
            ... )
            >>>
            >>> # SimReady USD format
            >>> results = model.get_usd_materials(
            ...     usd_path="datasets/raw/simready/common_assets/props/dellwood_ottoman/dellwood_ottoman_inst_base.usd",
            ...     use_simready_usd_format=True,
            ...     blender_path="/usr/bin/blender",
            ...     seed=42
            ... )
        """
        if output_dir is None:
            asset_name = os.path.splitext(os.path.basename(usd_path))[0]
            output_dir = f"/tmp/Vomp_usd_{asset_name}"

        # Select rendering backend
        if use_simready_usd_format:

            # Convert USD to OBJ in a temporary directory
            temp_obj_path, temp_dir = convert_usd_to_obj(usd_path)

            try:
                print(f"✓ USD converted to OBJ: {temp_obj_path}")

                results = self.get_mesh_materials(
                    mesh_path=temp_obj_path,
                    output_dir=output_dir,
                    num_views=num_views,
                    image_size=image_size,
                    radius=radius,
                    fov=fov,
                    seed=seed,
                    voxel_size=voxel_size,
                    max_voxels=max_voxels,
                    blender_path=blender_path,
                    gpu_device=gpu_device,
                    num_render_jobs=num_render_jobs,
                    query_points=query_points,
                    dino_batch_size=dino_batch_size,
                    return_original_scale=return_original_scale,
                    **kwargs,
                )

                return results

            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    print(f"Cleaning up temporary directory: {temp_dir}")
                    shutil.rmtree(temp_dir)

        else:

            if mesh_path is None:
                raise ValueError(
                    "mesh_path is required when use_blender_render=False. "
                    "Provide a separate mesh file for voxelization, or set use_blender_render=True "
                    "to automatically convert USD to OBJ."
                )

            # Define rendering function using Replicator
            def usd_render_func(usd_path, output_dir, num_views, image_size, **kwargs):
                # Use render_image_size for actual rendering (default 512 to match training data)
                return self.render_views_replicator(
                    asset_path=usd_path,
                    output_dir=output_dir,
                    num_views=num_views,
                    image_size=render_image_size,
                    isaac_sim_path=isaac_sim_path,
                    render_mode=render_mode,
                    rtx_settings_override=rtx_settings_override,
                    **kwargs,
                )

            # Define voxelization function using the provided mesh
            def usd_voxelize_func(usd_path, output_dir, **kwargs):
                # Use the separate mesh file for voxelization
                # Note: The mesh may not be normalized, so _voxelize_mesh will handle that
                return self._voxelize_mesh(
                    mesh_path, output_dir, voxel_size=voxel_size, max_voxels=max_voxels
                )

            # Step 1: Extract features
            print("Step 1: Extracting features...")

            # Prepare rendering parameters for kwargs
            render_kwargs = {
                "radius": radius,
                "fov": fov,
                "seed": seed,
            }

            coords, features = self.get_features(
                obj_3d=usd_path,
                render_func=usd_render_func,
                voxelize_func=usd_voxelize_func,
                num_views=num_views,
                image_size=image_size,
                output_dir=output_dir,
                batch_size=dino_batch_size,
                **render_kwargs,
                **kwargs,
            )

            # Step 2: Run inference on the features
            print("Step 2: Running material inference...")
            predict_kwargs = {}
            if max_voxels is not None:
                predict_kwargs["max_voxels"] = max_voxels
            voxel_results = self.predict_materials(coords, features, **predict_kwargs)

            # Handle query_points parameter
            if query_points == "voxel_centers" or query_points is None:
                # Return voxel-level results
                results = voxel_results

                # Apply inverse transformation
                if return_original_scale:
                    # Get transformation parameters
                    center, scale = get_mesh_transform_params(mesh_path)

                    # Transform coordinates back to original scale
                    results["voxel_coords_world"] = denormalize_coords(
                        results["voxel_coords_world"], center, scale
                    )
                    results["transform_center"] = center
                    results["transform_scale"] = scale

            elif isinstance(query_points, np.ndarray):
                # Step 3: Evaluate materials at custom query points
                print(
                    f"Step 3: Evaluating materials at {len(query_points)} custom query points..."
                )

                # Create upsampler from voxel results
                upsampler = MaterialUpsampler(
                    voxel_coords=voxel_results["voxel_coords_world"],
                    voxel_materials=np.column_stack(
                        [
                            voxel_results["youngs_modulus"],
                            voxel_results["poisson_ratio"],
                            voxel_results["density"],
                        ]
                    ),
                )

                # Interpolate to custom query points
                query_materials, query_distances = upsampler.interpolate(query_points)

                # Apply inverse transformation
                if return_original_scale:
                    # Get transformation parameters
                    center, scale = get_mesh_transform_params(mesh_path)

                    output_coords = denormalize_coords(query_points, center, scale)
                    output_voxel_coords = denormalize_coords(
                        voxel_results["voxel_coords_world"], center, scale
                    )
                else:
                    output_coords = query_points
                    output_voxel_coords = voxel_results["voxel_coords_world"]

                # Create final results
                results = {
                    "youngs_modulus": query_materials[:, 0],
                    "poisson_ratio": query_materials[:, 1],
                    "density": query_materials[:, 2],
                    "query_coords_world": output_coords,
                    "query_distances": query_distances,
                    "voxel_coords_world": output_voxel_coords,
                    "num_voxels": voxel_results["num_voxels"],
                    "num_query_points": len(query_points),
                }

                # Add transformation parameters if in original scale
                if return_original_scale:
                    results["transform_center"] = center
                    results["transform_scale"] = scale

                print(
                    f"✓ Evaluated materials at {len(query_points):,} custom query points"
                )
            else:
                raise ValueError(
                    f"Invalid query_points value: {query_points}. Must be 'voxel_centers', None, or numpy array."
                )

            print("✓ Material estimation complete!")
            return results
