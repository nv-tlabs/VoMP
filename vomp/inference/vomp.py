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



from __future__ import annotations

import copy
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from dataset_toolkits.dino import DINO_MODEL, DINO_REPOSITORY, render_hash
from dataset_toolkits.material_objects import camera
from dataset_toolkits.material_objects.camera import DEFAULT_CAMERA_SEED, resolve_camera_id
from dataset_toolkits.voxel_sampling import MAX_DINO_VOXELS, sample_voxel_rows
from .ply_utils import read_ply_vertices, write_ply_vertices
from .utils import LazyLoadDino, MaterialUpsampler, denormalize_coords
from easydict import EasyDict as edict
from vomp.utils.material_transforms import MaterialPropertyTransform
from vomp.models.geometry_encoder import ElasticGeometryEncoder, ElasticSLatVoxelDecoder
from safetensors.torch import load_file
from vomp.models import material_vae
from vomp.modules.sparse import SparseTensor
from vomp.datasets.sparse_voxel_materials import _segment_balanced_random_rows
from PIL import Image
from gsplat import rasterization
from vomp.representations.gaussian import Gaussian
import kaolin.ops.conversions.gaussians as gs_ops
from dataset_toolkits.extract_feature import IMAGE_SIZE, extract_features, load_views_from_directory
from scipy.spatial.distance import cdist
from vomp.inference.replicator_renderer import render_with_replicator


class RenderFunction(Protocol):
    """Custom rendering function returning view metadata and RGBA images."""

    def __call__(
        self, obj_3d: Any, output_dir: str, num_views: int, image_size: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]: ...


class VoxelizeFunction(Protocol):
    """Custom voxelization function returning normalized occupied cell centers."""

    def __call__(self, obj_3d: Any, output_dir: str, **kwargs: Any) -> np.ndarray: ...


class Vomp(nn.Module):
    """Infer material properties from meshes, USD assets, splats or voxel features.

    The supplied config selects the geometry encoder, material VAE,
    normalization and checkpoints. All configurations use the shared rendering,
    voxelization and DINO feature pipeline.
    """

    _dino_loader = LazyLoadDino

    def __init__(
        self,
        geometry_encoder: ElasticGeometryEncoder,
        matvae: nn.Module,
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
            ...     config_path="configs/materials/v1.1/inference.json"
            ... )

            # Using inference config with overrides
            >>> model = Vomp.from_checkpoint(
            ...     config_path="configs/materials/v1.1/inference.json",
            ...     geometry_checkpoint_dir="custom/path/to/geometry_transformer.pt"
            ... )

            # Using training config with direct file paths
            >>> model = Vomp.from_checkpoint(
            ...     config_path="configs/materials/v1.1/geometry_encoder/train.json",
            ...     geometry_checkpoint_dir="weights/vomp_1_1/geometry_transformer.pt",
            ...     matvae_checkpoint_dir="weights/vomp_1_1/matvae.safetensors",
            ...     normalization_params_path="weights/vomp_1_1/normalization_params.json"
            ... )

            # Using training config with directories (advanced)
            >>> model = Vomp.from_checkpoint(
            ...     config_path="configs/materials/v1.1/geometry_encoder/train.json",
            ...     geometry_checkpoint_dir="outputs/vomp1.1/geometry_encoder",
            ...     matvae_checkpoint_dir="outputs/vomp1.1/matvae/final_model",
            ...     normalization_params_path="outputs/vomp1.1/matvae/normalization_params.json",
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
            dino = cls._dino_loader(
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

        return MaterialPropertyTransform.from_json(
            normalization_params_file, normalization_type
        )

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
    def _load_matvae(cfg: edict, device: torch.device) -> nn.Module:
        """
        Initialize and load pre-trained MatVAE model.

        Creates the MatVAE class selected by the JSON and loads pre-trained
        weights from a safetensors checkpoint file. The MatVAE handles encoding
        and decoding of material properties (Young's modulus, Poisson's ratio, density).

        Args:
            cfg: Configuration object containing:
                - models.matvae.name: StandardVAE or TripletVAE
                - models.matvae.args: MatVAE model architecture arguments
                - matvae_checkpoint: Path to .safetensors checkpoint file
            device: Target device for model placement

        Returns:
            Loaded MatVAE model of the configured class

        Raises:
            ValueError: If the JSON does not name a supported MatVAE class
            RuntimeError: If checkpoint loading fails
            FileNotFoundError: If checkpoint file doesn't exist

        Note:
            - Model is automatically moved to specified device
            - Uses safetensors format for secure checkpoint loading
            - Checkpoint loading is strict (all parameters must match)
            - MatVAE weights are typically frozen during geometry encoder training
        """

        matvae = material_vae.from_config(cfg.models.matvae).to(device)

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

    def _input_dimensions(self) -> Tuple[int, int]:
        """Read the encoder's configured voxel resolution and feature width."""
        args = self.config.get("models", {}).get("geometry_encoder", {}).get("args", {})
        resolution = int(args.get("resolution", getattr(self.geometry_encoder, "resolution", 64)))
        channels = int(args.get("in_channels", getattr(self.geometry_encoder, "in_channels", 1024)))
        if resolution <= 0 or channels <= 0:
            raise ValueError("Encoder resolution and in_channels must be positive")
        return resolution, channels

    def _require_preprocessing_dimensions(self) -> None:
        """Built-in DINO preprocessing produces the released 64³/1024 contract."""
        if self._input_dimensions() != (64, 1024):
            raise ValueError(
                "Built-in preprocessing requires resolution=64 and in_channels=1024; "
                "use predict_materials with matching precomputed features for other encoders"
            )

    @torch.inference_mode()
    def predict_materials(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        max_voxels: int = 32768,
        sample_posterior: bool = False,
        sampling_seed: int = DEFAULT_CAMERA_SEED,
        segment_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict material properties for given coordinates and features.

        Takes voxel coordinates and their corresponding DINO features and runs them
        through the geometry encoder and material VAE to predict mechanical properties.
        Handles memory management by downsampling if too many voxels are provided.

        Args:
            coords: Voxel coordinates tensor (N, 4) in format [batch_idx, x, y, z]
                with integer spatial indices in [0, resolution - 1]
            features: DINO feature vectors (N, feature_dim) extracted from rendered views
            max_voxels: Maximum number of voxels to process at once. If input exceeds
                this, the training loader's segment-balanced sampling is applied
            sample_posterior: Whether to sample from VAE posterior distribution instead
                of using mean predictions for more diverse outputs
            sampling_seed: CPU sampling seed; use the same seed across checkpoints
            segment_ids: Optional per-row material segment IDs from preprocessing

        Returns:
            Dictionary containing:
            - 'youngs_modulus': Young's modulus values in Pascals (N,)
            - 'poisson_ratio': Poisson's ratio values (N,)
            - 'density': Density values in kg/m³ (N,)
            - 'voxel_coords_world': Normalized model-frame cell centers (N, 3)
            - 'num_voxels': Total number of processed voxels

        Note:
            All predictions are automatically denormalized using the loaded
            normalization parameters to return physically meaningful values.
        """

        resolution, channels = self._input_dimensions()
        if coords.ndim != 2 or coords.shape[1] != 4 or not len(coords):
            raise ValueError("coords must be a nonempty N x 4 integer tensor")
        if coords.is_floating_point() or torch.any(coords[:, 1:] < 0) or torch.any(coords[:, 1:] >= resolution):
            raise ValueError(f"coords must contain integer {resolution}^3 grid indices")
        if features.shape != (len(coords), channels) or not torch.isfinite(features).all():
            raise ValueError(f"features must be finite N x {channels} encoder inputs")
        if max_voxels is None:
            max_voxels = 32768
        if not 0 < max_voxels <= 32768:
            raise ValueError("max_voxels must be between 1 and the training cap 32768")
        if segment_ids is None:
            segment_ids = np.full(len(coords), "object")
        segment_ids = np.asarray(segment_ids).astype(str)
        if segment_ids.shape != (len(coords),):
            raise ValueError("segment_ids must have one entry per feature row")
        device = next(self.parameters()).device
        coords = coords.to(device=device, dtype=torch.int32)
        features = features.to(device=device, dtype=torch.float32)

        # Downsample if too many voxels
        num_voxels = coords.shape[0]
        if num_voxels > max_voxels:
            print(f"Downsampling from {num_voxels} to {max_voxels} voxels")
            generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
            rows = _segment_balanced_random_rows(segment_ids, max_voxels, generator)
            indices = torch.as_tensor(rows, dtype=torch.long, device=device)
            coords = coords[indices]
            features = features[indices]
            num_voxels = max_voxels

        # Create sparse tensor
        sparse_tensor = SparseTensor(feats=features, coords=coords)

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
        voxel_coords_world = (voxel_indices + 0.5) / resolution - 0.5

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
            feats=features, coords=coords
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
                self.material_transform.forward_transform_and_standardize_tensor(materials)
            )
            E, nu, rho = (
                materials_normalized[:, 0],
                materials_normalized[:, 1],
                materials_normalized[:, 2],
            )

        # Encode through MatVAE
        material_tensor = torch.stack([E, nu, rho], dim=-1)
        _, latent_mean, _ = self.matvae.encode(material_tensor, sample=False)

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
        render_image_size: int = 512,
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
            image_size: DINO input size (518), after the training RGBA resize
            render_image_size: Render resolution (512), passed to render_func
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
        self._require_preprocessing_dimensions()

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
                    obj_3d, output_dir, num_views, render_image_size, **kwargs
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
            seed: Optional sampling seed; defaults to 0

        Returns:
            Tuple containing:
                - yaws: List of yaw angles in radians
                - pitchs: List of pitch angles in radians
                - radius_list: List of camera distances (all same value)
                - fov_list: List of FOV values in degrees (all same value)
        """
        rng = np.random.RandomState(DEFAULT_CAMERA_SEED if seed is None else seed)
        yaws = []
        pitchs = []
        offset = (rng.rand(), rng.rand())

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
        image_size: int = 512,
        radius: Optional[float] = None,
        fov: float = 40.0,
        seed: Optional[int] = None,
        asset_id: Optional[str] = None,
        up_axis: str = "Z",
    ) -> List[Dict]:
        """
        Render multiple views of Gaussian splat from sampled camera positions.

        Generates camera viewpoints using quasi-random sampling and renders
        the Gaussian splat model from each position. Saves RGBA images and
        camera metadata in NeRF-compatible format.

        Args:
            gaussian_model: Gaussian splat normalized with _normalize_gaussian
            output_dir: Directory to save rendered images and metadata
            num_views: Number of camera views to render
            image_size: Resolution of rendered images (square)
            radius: Optional minimum camera distance; defaults to training camera fitting
            fov: Field of view in degrees
            seed: Optional camera seed override; defaults to 0
            asset_id: Optional asset identity, used when seed is not supplied

        Returns:
            List of frame metadata dictionaries containing:
                - file_path: Relative path to rendered image
                - transform_matrix: 4x4 camera-to-world matrix
                - camera_angle_x: FOV in radians
                - yaw, pitch, radius: Camera positioning parameters
        """
        self._require_preprocessing_dimensions()

        if num_views != camera.DEFAULT_NUM_VIEWS:
            raise ValueError(
                f"splat inference requires {camera.DEFAULT_NUM_VIEWS} views"
            )
        if image_size != camera.RESOLUTION:
            raise ValueError(
                f"splat inference requires {camera.RESOLUTION}px renders"
            )
        if not math.isclose(
            math.radians(fov), camera.CAMERA_FOV, abs_tol=1.0e-12
        ):
            raise ValueError("splat inference requires the training 40-degree FOV")

        up_axis = up_axis.upper()
        if up_axis not in {"Y", "Z"}:
            raise ValueError("up_axis must be 'Y' or 'Z'")

        renders_dir = os.path.join(output_dir, "renders")
        os.makedirs(renders_dir, exist_ok=True)

        print(f"Rendering {num_views} Gaussian splat views...")
        gaussian_xyz = gaussian_model.get_xyz
        gaussian_scales = gaussian_model.get_scaling
        visible_support = gaussian_model.get_opacity.reshape(-1) >= 0.05
        if not torch.any(visible_support):
            raise ValueError("Gaussian splat has no visible support")
        support_radius = 3.0 * gaussian_scales[visible_support].max(dim=1).values
        support_xyz = gaussian_xyz[visible_support]
        support_minimum = (support_xyz - support_radius[:, None]).amin(dim=0)
        support_maximum = (support_xyz + support_radius[:, None]).amax(dim=0)
        support_corners = torch.stack(
            [
                torch.stack((x, y, z))
                for x in (support_minimum[0], support_maximum[0])
                for y in (support_minimum[1], support_maximum[1])
                for z in (support_minimum[2], support_maximum[2])
            ]
        )
        normalized_points = torch.cat((support_xyz, support_corners), dim=0)
        normalized_points = normalized_points.detach().cpu().numpy().astype(np.float64)
        camera_id = resolve_camera_id(
            asset_id=asset_id, seed=seed,
        )
        fov_radians = camera.CAMERA_FOV
        focal = 0.5 * image_size / math.tan(fov_radians * 0.5)
        intrinsics = torch.tensor(
            (
                (focal, 0.0, image_size * 0.5),
                (0.0, focal, image_size * 0.5),
                (0.0, 0.0, 1.0),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        means = gaussian_xyz
        quaternions = gaussian_model.get_rotation
        scales = gaussian_scales
        opacities = gaussian_model.get_opacity.reshape(-1)
        colors = gaussian_model.get_features
        frames_metadata = []
        for i, yaw, pitch, direction in camera._camera_samples(
            camera_id, num_views, up_axis
        ):
            horizontal, vertical = camera._camera_frame(direction, up_axis)
            fitted_radius = camera._camera_radius(
                normalized_points, direction, horizontal, vertical
            )
            view_radius = fitted_radius if radius is None else max(radius, fitted_radius)
            location = direction * view_radius
            camera_to_world = np.asarray(
                (
                    (horizontal[0], vertical[0], direction[0], location[0]),
                    (horizontal[1], vertical[1], direction[1], location[1]),
                    (horizontal[2], vertical[2], direction[2], location[2]),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dtype=np.float32,
            )
            camera_to_world_cv = camera_to_world.copy()
            camera_to_world_cv[:3, 1:3] *= -1
            extrinsics = torch.inverse(
                torch.from_numpy(camera_to_world_cv).to(self.device)
            )
            rendered, alpha, _ = rasterization(
                means=means,
                quats=quaternions,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=extrinsics.unsqueeze(0),
                Ks=intrinsics.unsqueeze(0),
                width=image_size,
                height=image_size,
                near_plane=1.0e-4,
                far_plane=8.0,
                sh_degree=gaussian_model.active_sh_degree,
                render_mode="RGB",
            )
            premultiplied_rgb = rendered[0].permute(2, 0, 1)
            rendered_alpha = alpha[0].permute(2, 0, 1).clamp(0.0, 1.0)
            rendered_rgb = torch.where(
                rendered_alpha > 1.0e-6,
                premultiplied_rgb / rendered_alpha.clamp_min(1.0e-6),
                torch.zeros_like(premultiplied_rgb),
            ).clamp(0.0, 1.0)

            # Combine RGB and alpha channels
            rendered_rgba = torch.cat([rendered_rgb, rendered_alpha], dim=0)
            rendered_rgba = rendered_rgba.cpu().numpy()
            rendered_rgba = np.transpose(rendered_rgba, (1, 2, 0))
            rendered_rgba = np.clip(rendered_rgba * 255, 0, 255).astype(np.uint8)

            # Save rendered image
            filename = f"{i:03d}.png"
            image_path = os.path.join(renders_dir, filename)
            Image.fromarray(rendered_rgba, "RGBA").save(image_path)

            frame_metadata = {
                "file_path": filename,
                "transform_matrix": camera_to_world.astype(float).tolist(),
                "camera_angle_x": fov_radians,
                "yaw": yaw,
                "pitch": pitch,
                "radius": view_radius,
            }
            frames_metadata.append(frame_metadata)

        # Save NeRF-compatible transforms.json
        transforms_data = {
            "camera_angle_x": fov_radians,
            "asset_id": camera_id,
            "render_settings": {
                "num_views": num_views,
                "resolution": image_size,
                "field_of_view_radians": fov_radians,
                "camera_nominal_radius": camera.CAMERA_NOMINAL_RADIUS,
                "camera_fit_fraction": camera.CAMERA_FRAME_FRACTION,
                "up_axis": up_axis,
                "representation": "3d_gaussian_splat_sh",
            },
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
        radius: Optional[float] = None,
        fov: float = 40.0,
        seed: Optional[int] = None,
        up_axis: str = "Z",
        sh_degree: int = 3,
        aabb: Optional[List[float]] = None,
        device: Optional[Union[str, torch.device]] = None,
        voxel_method: str = "kaolin",
        voxel_level: int = 6,
        voxel_iso: float = 11.345,
        voxel_tol: float = 1.0 / 8.0,
        voxel_step: int = 10,
        voxel_opacity_threshold: float = 0.35,
        max_voxels: Optional[int] = 32768,
        query_points: Union[str, np.ndarray, None] = "splat_centers",
        dino_batch_size: int = 16,
        save_features: bool = True,
        return_original_scale: bool = True,
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
            radius: Optional minimum camera distance (default: adaptive training camera fit)
            fov: Field of view in degrees (default: 40.0, matches training data)
            seed: Optional camera and voxel sampling seed override; defaults to 0
            sh_degree: Spherical harmonics degree for PLY loading
            up_axis: Authored splat up-axis ("Z" or "Y")
            return_original_scale: Return output coordinates in the authored splat frame.
                Custom query_points arrays are always supplied in normalized coordinates.
            aabb: Bounding box for PLY loading (defaults to [-1,-1,-1,2,2,2])
            device: Compute device for PLY loading (defaults to the model device)
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
                - numpy array (N, 3): Custom coordinates in the normalized [-0.5, 0.5] frame
                - None: Same as "voxel_centers"
            dino_batch_size: Number of images to process simultaneously during DINO feature extraction (higher values use more GPU memory but may be faster)
            save_features: Whether to cache extracted DINO features

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
        self._require_preprocessing_dimensions()


        # Handle automatic Gaussian loading from PLY path
        if isinstance(gaussian_model, str):
            # Set defaults
            if aabb is None:
                aabb = [-1, -1, -1, 2, 2, 2]
            if device is None:
                device = self.device

            # Create and load Gaussian
            ply_path = gaussian_model
            gaussian_model = Gaussian(sh_degree=sh_degree, aabb=aabb, device=device)
            # Standard 3DGS PLYs do not declare an up-axis.  Preserve their
            # authored coordinates; the default loader transform also leaves SH
            # coefficients unrotated and therefore corrupts view-dependent color.
            gaussian_model.load_ply(ply_path, transform=None)
        else:
            gaussian_model = copy.deepcopy(gaussian_model)
        sampling_seed = DEFAULT_CAMERA_SEED if seed is None else int(seed)

        if output_dir is None:
            output_dir = f"/tmp/Vomp_splat_{id(gaussian_model)}"

        print("=== Vomp: Splat Material Estimation ===")

        # Define built-in splat rendering function
        def splat_render_func(
            gaussian_model, output_dir, num_views, image_size, **kwargs
        ):
            # Filter kwargs to only pass render-related parameters
            render_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["radius", "fov", "seed", "asset_id", "up_axis"]
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
            voxel_method = kwargs.get("voxel_method", "kaolin")
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

        original_xyz = gaussian_model.get_xyz
        source_min = original_xyz.amin(dim=0)
        source_max = original_xyz.amax(dim=0)
        source_center = ((source_min + source_max) * 0.5).detach().cpu().numpy()
        source_scale = float((source_max - source_min).max())

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
            "up_axis": up_axis,
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
            save_features=save_features,
        )

        # Step 2: Run inference on the features
        print("Step 2: Running material inference...")
        predict_kwargs = {}
        if max_voxels is not None:
            predict_kwargs["max_voxels"] = max_voxels
        predict_kwargs["sampling_seed"] = sampling_seed
        predict_kwargs["sample_posterior"] = False
        voxel_results = self.predict_materials(coords, features, **predict_kwargs)

        # Add splat count for reporting
        num_splats = int(gaussian_model.get_xyz.shape[0])
        voxel_results["num_splats"] = num_splats

        # Handle query_points parameter
        if isinstance(query_points, str) and query_points == "splat_centers":
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
        elif query_points is None or (
            isinstance(query_points, str) and query_points == "voxel_centers"
        ):
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

        if return_original_scale:
            for key in ("query_coords_world", "voxel_coords_world"):
                if results.get(key) is not None:
                    results[key] = denormalize_coords(
                        np.asarray(results[key], dtype=np.float64), source_center, source_scale
                    )
            results["transform_center"] = source_center
            results["transform_scale"] = source_scale

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
        if not torch.isfinite(extent) or extent <= 0:
            raise ValueError("Gaussian splat bounds are degenerate")
        # Match material-object preprocessing exactly: center the axis-aligned
        # bounds and divide by their maximum extent.  Do not add an extra 0.98
        # margin; camera fitting is responsible for keeping the object in frame.
        scale_factor = 1.0 / extent

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
        voxel_centers = (unique_indices.astype(np.float64) + 0.5) / 64.0 - 0.5
        voxel_centers = voxel_centers.astype(np.float32)

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

        if level != 6:
            raise ValueError("material inference requires a 64^3 (level 6) grid")

        # Extract Gaussian parameters
        # Kaolin's grid covers [-1, 1], whereas the material model uses
        # [-0.5, 0.5].  Scale both centers and Gaussian sizes into Kaolin's
        # domain, then map its integer cells back below.
        xyz = (gaussian_model.get_xyz * 2.0).contiguous()  # (N, 3)
        scales = (gaussian_model.get_scaling * 2.0).contiguous()  # (N, 3)
        rots = gaussian_model.get_rotation.contiguous()  # (N, 4) quaternions
        opacities = gaussian_model.get_opacity.reshape(-1).contiguous()  # (N,)

        # Handle empty case
        if xyz.shape[0] == 0:
            print("No Gaussians to voxelize")
            return np.empty((0, 3), dtype=np.float32)

        print(f"Using Kaolin voxelization with {xyz.shape[0]} Gaussians...")
        print(f"Resolution: {2**level}^3, opacity_threshold: {opacity_threshold}")

        # Keep Kaolin's opacity integration (including step) intact.  The private
        # gs_to_spc_cuda operator only produces intersections, not voxel opacity.
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
        voxel_centers = (
            (voxel_coords_cpu.astype(np.float64) + 0.5) / (2**level) - 0.5
        ).astype(np.float32)

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
        """Extract voxel DINO features using the training preprocessing."""

        if model_name != DINO_MODEL:
            raise ValueError(f"The material model uses {DINO_MODEL}")
        if image_size != IMAGE_SIZE:
            raise ValueError(f"DINO inputs must use the training size {IMAGE_SIZE}")
        voxel_centers = np.asarray(voxel_centers, dtype=np.float32)
        if voxel_centers.ndim != 2 or voxel_centers.shape[1] != 3 or not len(
            voxel_centers
        ):
            raise ValueError("voxel centers must be a nonempty N x 3 array")
        indices = np.floor(
            (voxel_centers.astype(np.float64) + 0.5) * 64.0
        ).astype(np.int64)
        expected_centers = (
            (indices.astype(np.float64) + 0.5) / 64.0 - 0.5
        ).astype(np.float32)
        if (
            np.any(indices < 0)
            or np.any(indices >= 64)
            or not np.array_equal(voxel_centers, expected_centers)
            or len(np.unique(np.ravel_multi_index(indices.T, (64, 64, 64))))
            != len(indices)
        ):
            raise ValueError("voxels are not unique exact 64^3 cell centers")

        occupancy = np.zeros((64, 64, 64), dtype=bool)
        occupancy[tuple(indices.T)] = True
        interior = occupancy.copy()
        for axis in range(3):
            forward = np.zeros_like(occupancy)
            backward = np.zeros_like(occupancy)
            source = [slice(None)] * 3
            destination = [slice(None)] * 3
            source[axis] = slice(1, None)
            destination[axis] = slice(None, -1)
            forward[tuple(destination)] = occupancy[tuple(source)]
            source[axis] = slice(None, -1)
            destination[axis] = slice(1, None)
            backward[tuple(destination)] = occupancy[tuple(source)]
            interior &= forward & backward
        surface_mask = ~interior[tuple(indices.T)]
        segment_ids = np.full(len(indices), "splat", dtype="U5")
        segment_order = ("splat",)
        metadata_path = os.path.join(output_dir, "voxels", "voxel_metadata.npz")
        if os.path.isfile(metadata_path):
            with np.load(metadata_path, allow_pickle=False) as metadata:
                if not np.array_equal(metadata["indices"], indices):
                    raise ValueError("voxel metadata does not match these voxel centers; use a fresh output_dir")
                surface_mask = metadata["surface_mask"].astype(bool)
                segment_ids = metadata["segment_ids"].astype(str)
                segment_order = (
                    tuple(metadata["segment_order"].astype(str))
                    if "segment_order" in metadata else tuple(np.unique(segment_ids))
                )
            if surface_mask.shape != (len(indices),) or segment_ids.shape != (len(indices),):
                raise ValueError("voxel metadata must contain one surface/segment label per voxel")
        selected_rows = sample_voxel_rows(
            indices,
            segment_ids,
            surface_mask,
            segment_order,
            MAX_DINO_VOXELS,
        )
        indices = indices[selected_rows]
        positions = voxel_centers[selected_rows]
        segment_ids = segment_ids[selected_rows]

        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_path = os.path.join(features_dir, f"{model_name}.npz")
        render_sha256 = render_hash(os.path.join(output_dir, "renders"), frames_metadata)
        if os.path.exists(features_path):
            try:
                with np.load(features_path, allow_pickle=False) as archive:
                    aggregation = str(archive["feature_contract"].item())
                    cached_indices = archive["indices"]
                    patchtokens = archive["patchtokens"]
                    cached_repository = str(archive["model_repository"].item())
                    cached_render = str(archive["render_sha256"].item())
                    cached_segments = archive["segment_ids"]
                if (
                    aggregation == "training_alpha_in_frame_weighted_v1"
                    and cached_repository == DINO_REPOSITORY
                    and cached_render == render_sha256
                    and np.array_equal(cached_indices, indices.astype(np.uint8))
                    and np.array_equal(cached_segments, segment_ids)
                    and patchtokens.shape == (len(indices), 1024)
                    and patchtokens.dtype == np.float16
                    and np.isfinite(patchtokens).all()
                ):
                    print(f"Loading DINO features from {features_path}")
                    return self._load_saved_features(features_path)
            except (KeyError, OSError, ValueError):
                pass
            print(f"Ignoring stale feature cache: {features_path}")

        print("Extracting DINO features...")
        dino = self._dino_loader(
            model_name=model_name, device=self.device, use_trt=self.use_trt
        )
        views = load_views_from_directory(
            os.path.join(output_dir, "renders"), frames_metadata
        )
        # The shared preprocessing uses .cuda(); select this model's CUDA device.
        with torch.cuda.device(self.device):
            patchtokens = extract_features(
                dino.get_model(),
                views,
                positions,
                dino.get_transform(),
                batch_size,
                8192,
            )
        if save_features:
            pack = {
                "indices": indices.astype(np.uint8),
                "patchtokens": patchtokens,
                "voxel_row_indices": selected_rows.astype(np.int32),
                "segment_ids": segment_ids,
                "feature_contract": np.asarray("training_alpha_in_frame_weighted_v1"),
                "model_repository": np.asarray(DINO_REPOSITORY),
                "render_sha256": np.asarray(render_sha256),
            }
            np.savez_compressed(features_path, **pack)
            print(f"Saved features to {features_path}")
        indices_tensor = torch.from_numpy(indices.astype(np.int32)).to(self.device)
        batch_dim = torch.zeros(
            (len(indices), 1), dtype=torch.int32, device=self.device
        )
        coords = torch.cat((batch_dim, indices_tensor), dim=1)
        features = torch.from_numpy(patchtokens.astype(np.float32)).to(self.device)
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

    def render_mesh_views(
        self,
        mesh_path: str,
        output_dir: str,
        num_views: int = 150,
        image_size: int = 512,
        radius: Optional[float] = None,
        fov: float = 40.0,
        seed: Optional[int] = None,
        blender_path: Optional[str] = None,
        dataset: str = "simready",
    ) -> List[Dict]:
        """Render the training 150-view, 512px Cycles setup and exact mesh geometry.

        Uses adaptive camera distance, the production lights/world and Blender
        4.2. With no blender_path, the current Python must provide bpy 4.2.
        USD assets use the production USD material/geometry loader; other supported
        inputs are GLB/GLTF, OBJ, PLY and STL. Parallelize across assets/GPUs.
        Camera sampling defaults to seed 0; pass seed to override it.
        """
        self._require_preprocessing_dimensions()

        from dataset_toolkits.material_objects.render_mesh import render_mesh

        if num_views != 150 or image_size != 512 or fov != 40.0 or radius is not None:
            raise ValueError(
                "Training rendering requires num_views=150, image_size=512, "
                "fov=40 and radius=None (adaptive distance)"
            )
        return render_mesh(
            mesh_path, output_dir, blender_path=blender_path, seed=seed, dataset=dataset
        )

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
    ) -> np.ndarray:
        """Voxelize the geometry saved by render_mesh_views on the training grid.

        Returns all occupied cell centers; feature extraction applies the 72k
        surface/interior budget, then predict_materials applies the 32768 budget.
        """
        from dataset_toolkits.material_objects.render_mesh import voxelize_render_geometry

        if voxel_size != 1 / 64:
            raise ValueError("Mesh voxelization requires the training voxel_size=1/64")
        return voxelize_render_geometry(mesh_path, output_dir)

    @torch.inference_mode()
    def get_mesh_materials(
        self,
        mesh_path: str,
        output_dir: Optional[str] = None,
        num_views: int = 150,
        image_size: int = 518,
        render_image_size: int = 512,
        radius: Optional[float] = None,
        fov: float = 40.0,
        seed: Optional[int] = None,
        voxel_size: float = 1 / 64,
        max_voxels: Optional[int] = 32768,
        blender_path: Optional[str] = None,
        query_points: Union[str, np.ndarray, None] = "mesh_vertices",
        dino_batch_size: int = 16,
        return_original_scale: bool = False,
        usd_dataset: str = "simready",
        save_features: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Predict materials using the shared training preprocessing.

        Mesh/voxel coordinates share the rendered object's bounding-box frame.
        Use query_points="voxel_centers" for direct model predictions. The default
        interpolates onto rendered mesh vertices. Custom N x 3 query arrays are
        in normalized coordinates; return_original_scale converts output locations
        to the renderer's source-world frame. No inference noise is added.
        Camera and voxel sampling both default to seed 0.
        """
        self._require_preprocessing_dimensions()

        from dataset_toolkits.material_objects.render_mesh import feature_segment_ids, load_render_geometry, sampling_seed

        if output_dir is None:
            mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            output_dir = f"/tmp/Vomp_mesh_{mesh_name}"
        frames = self.render_mesh_views(
            mesh_path, output_dir, num_views=num_views, image_size=render_image_size,
            radius=radius, fov=fov, seed=seed, blender_path=blender_path,
            dataset=usd_dataset,
        )
        centers = self._voxelize_mesh(mesh_path, output_dir, voxel_size=voxel_size)
        coords, features = self.get_features(
            renders_metadata=frames, voxel_centers=centers, output_dir=output_dir,
            image_size=image_size, batch_size=dino_batch_size, save_features=save_features,
        )
        labels = feature_segment_ids(output_dir, coords[:, 1:].cpu().numpy())
        voxel_results = self.predict_materials(
            coords, features, max_voxels=32768 if max_voxels is None else max_voxels,
            sampling_seed=sampling_seed(seed), segment_ids=labels,
        )
        geometry, manifest = load_render_geometry(mesh_path, output_dir)
        normalization = manifest["normalization"]
        center = np.asarray(normalization["source_center"])
        scale = float(normalization["source_scale"])

        if isinstance(query_points, str) and query_points == "mesh_vertices":
            query_coords = geometry["vertices"]
        elif query_points is None or (
            isinstance(query_points, str) and query_points == "voxel_centers"
        ):
            results = voxel_results
            if return_original_scale:
                results["voxel_coords_world"] = denormalize_coords(
                    results["voxel_coords_world"], center, scale
                )
                results["transform_center"] = center
                results["transform_scale"] = scale
            return results
        elif isinstance(query_points, np.ndarray):
            query_coords = np.asarray(query_points, dtype=np.float32)
            if query_coords.ndim != 2 or query_coords.shape[1] != 3 or not np.isfinite(query_coords).all():
                raise ValueError("query_points must be a finite N x 3 array in normalized coordinates")
        else:
            raise ValueError("query_points must be mesh_vertices, voxel_centers, None or an N x 3 array")

        upsampler = MaterialUpsampler(
            voxel_coords=voxel_results["voxel_coords_world"],
            voxel_materials=np.column_stack([
                voxel_results["youngs_modulus"], voxel_results["poisson_ratio"],
                voxel_results["density"],
            ]),
        )
        materials, distances = upsampler.interpolate(query_coords)
        output_coords = query_coords
        output_voxels = voxel_results["voxel_coords_world"]
        if return_original_scale:
            output_coords = denormalize_coords(output_coords, center, scale)
            output_voxels = denormalize_coords(output_voxels, center, scale)
        results = {
            "youngs_modulus": materials[:, 0], "poisson_ratio": materials[:, 1],
            "density": materials[:, 2], "query_coords_world": output_coords,
            "query_distances": distances, "voxel_coords_world": output_voxels,
            "num_voxels": voxel_results["num_voxels"], "num_query_points": len(query_coords),
        }
        if return_original_scale:
            results["transform_center"] = center
            results["transform_scale"] = scale
        return results

    @torch.inference_mode()
    def get_usd_materials(
        self,
        usd_path: str,
        output_dir: Optional[str] = None,
        num_views: int = 150,
        image_size: int = 518,
        render_image_size: int = 512,
        radius: Optional[float] = None,
        fov: float = 40.0,
        seed: Optional[int] = None,
        voxel_size: float = 1 / 64,
        max_voxels: Optional[int] = 32768,
        query_points: Union[str, np.ndarray, None] = "voxel_centers",
        dino_batch_size: int = 16,
        use_simready_usd_format: bool = False,
        blender_path: Optional[str] = None,
        return_original_scale: bool = False,
        save_features: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Use the training USD renderer, composed mesh and material segments.

        use_simready_usd_format selects the SimReady material loader; otherwise
        the common USD loader is used. Both render with Cycles 4.2.
        """
        if output_dir is None:
            asset_name = os.path.splitext(os.path.basename(usd_path))[0]
            output_dir = f"/tmp/Vomp_usd_{asset_name}"
        return self.get_mesh_materials(
            usd_path, output_dir=output_dir, num_views=num_views,
            image_size=image_size, render_image_size=render_image_size,
            radius=radius, fov=fov, seed=seed, voxel_size=voxel_size,
            max_voxels=max_voxels, blender_path=blender_path,
            query_points=query_points, dino_batch_size=dino_batch_size,
            return_original_scale=return_original_scale,
            usd_dataset="simready" if use_simready_usd_format else "common",
            save_features=save_features,
        )
