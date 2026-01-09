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

import torch.nn as nn

# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()


# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0


import os
import torch
import json
from typing import Dict, Any, Optional


def get_optimizer(config, params):
    """
    Create an optimizer from a config dictionary.

    Args:
        config: A dictionary with "name" and "args" keys
        params: The parameters to optimize

    Returns:
        A PyTorch optimizer
    """
    name = config.get("name", "Adam")
    args = config.get("args", {})

    if name == "Adam":
        return torch.optim.Adam(params, **args)
    elif name == "AdamW":
        return torch.optim.AdamW(params, **args)
    elif name == "SGD":
        return torch.optim.SGD(params, **args)
    else:
        raise ValueError(f"Optimizer {name} not supported")


def save_checkpoint(
    step: int,
    models: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    save_name: str = "latest",
):
    """
    Save a checkpoint of the models and optimizer.

    Args:
        step: Current training step
        models: Dictionary of models to save
        optimizer: Optimizer to save
        save_dir: Directory to save checkpoint
        save_name: Name of the checkpoint
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ckpt_dir = os.path.join(save_dir, "ckpts")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    # Save models
    for name, model in models.items():
        state_dict = model.state_dict()
        torch.save(
            state_dict, os.path.join(ckpt_dir, f"{save_name}_{name}_step_{step}.pt")
        )

    # Save optimizer
    torch.save(
        optimizer.state_dict(),
        os.path.join(ckpt_dir, f"{save_name}_optimizer_step_{step}.pt"),
    )

    # Save metadata
    with open(os.path.join(ckpt_dir, f"{save_name}_meta.json"), "w") as f:
        json.dump({"step": step}, f)

    print(f"Saved checkpoint at step {step}")
