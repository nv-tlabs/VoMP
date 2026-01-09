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

"""This is the textgrad script for vlm annotation task. This requires installing textgrad."""

OPENAI_API_KEY = ""  # fill in your OpenAI API key
ANNOTATIONS_FILE = "/home/rdagli/code/TRELLIS/datasets/raw/material_annotations.json"  # fill in the path to the annotations file
OUTPUT_DIR = "./optimization_results"

import os
import sys
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import textgrad as tg
from textgrad.engine import EngineLM

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from dataset_toolkits.material_objects.vlm_annotations.utils.vlm import (
    parse_vlm_properties,
)
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

tg.set_backward_engine("gpt-4o-mini")

import logging

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"{OUTPUT_DIR}_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/optimization.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("textgrad").setLevel(logging.WARNING)


class QwenEngine(EngineLM):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-72B-Instruct"):
        self.model_string = model_id
        logger.info(f"Loading {model_id}...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
        logger.info("Model loaded successfully")

    def generate(self, prompt, system_prompt=None, **kwargs):
        """
        TextGrad EngineLM interface.
        prompt: The input prompt (textgrad Variable value or string)
        system_prompt: The system prompt (textgrad Variable value or string)
        """

        try:
            if isinstance(prompt, str) and prompt.startswith("{"):
                data = json.loads(prompt)
                material_type = data.get("material_type")
                semantic_usage = data.get("semantic_usage")
                images = data.get("images", [])
            else:
                material_type = prompt
                semantic_usage = None
                images = []
        except:
            material_type = prompt
            semantic_usage = None
            images = []

        intro_text = """
You are a materials science expert analyzing an image of the full object (showing how the material appears in context).

Using this image and the information below, identify the real-world material and estimate its mechanical properties.
"""

        user_prompt = f"""{intro_text}
Material context:
  * Material type: {material_type}
  * Usage: {semantic_usage or material_type}

Your task is to provide three specific properties:
1. Young's modulus (in Pa using scientific notation)
2. Poisson's ratio (a value between 0.0 and 0.5)
3. Density (in kg/m^3 using scientific notation)

Based on the provided images and context information, analyze the material properties.
Note: The material segment might be internal to the object and not visible from the outside.

Respond using EXACTLY the following format (do not deviate from this structure):

Analysis: 
Step 1: Identify the material class/type based on visual appearance
Step 2: Describe the surface characteristics (texture, reflectivity, color)
Step 3: Determine the specific material subtype considering its physical properties
Step 4: Reason through each property estimate based on visual and measured data

Young's modulus: <value in scientific notation> Pa
Poisson's ratio: <single decimal value between 0.0 and 0.5>
Density: <value in scientific notation> kg/m^3

Critical Instructions:
1. You MUST provide numerical estimates for ALL materials, including organic or unusual materials
2. For natural materials like leaves, wood, or paper, provide estimates based on similar materials with known properties
3. Never use "N/A", "unknown", or any non-numeric responses for the material properties
4. For Poisson's ratio, provide a simple decimal number (like 0.3 or 0.42)
5. Each property should be on its own line with exact formatting shown above
"""

        user_content = []
        if images:
            for img_path in images:
                if img_path and os.path.exists(img_path):
                    user_content.append(
                        {
                            "type": "image",
                            "image": f"file://{os.path.abspath(img_path)}",
                        }
                    )

        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt if system_prompt else "You are a helpful assistant."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return output

    def __call__(self, prompt, system_prompt=None):
        return self.generate(prompt, system_prompt)


def main():

    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)

    samples = []
    max_objects = 23
    # rest 7 will serve as validation set
    for obj in annotations[:max_objects]:
        for seg_name, seg_info in obj.get("segments", {}).items():
            if all(
                k in seg_info for k in ["youngs_modulus", "poissons_ratio", "density"]
            ):
                samples.append(
                    {
                        "name": f"{obj['object_name']}/{seg_name}",
                        "material_type": seg_info.get("material_type", "unknown"),
                        "semantic_usage": seg_info.get("semantic_usage", seg_name),
                        "ground_truth": {
                            "youngs_modulus": float(seg_info["youngs_modulus"]),
                            "poissons_ratio": float(seg_info["poissons_ratio"]),
                            "density": float(seg_info["density"]),
                        },
                    }
                )

    logger.info(f"Loaded {len(samples)} samples")
    random.shuffle(samples)

    qwen_engine = QwenEngine()
    initial_prompt = """You are a materials science expert. Help analyze material properties from images and context."""

    system_prompt_var = tg.Variable(
        initial_prompt,
        requires_grad=True,
        role_description="system prompt for material property prediction",
    )

    model = tg.BlackboxLLM(qwen_engine, system_prompt_var)

    optimizer = tg.TextualGradientDescent(
        engine=tg.get_engine("gpt-4o-mini"), parameters=[system_prompt_var]
    )

    eval_prompt = """Evaluate this material property prediction for:
1. Format correctness: Are Young's modulus (in Pa), Poisson's ratio (0.0-0.5), and density (in kg/m³) all present?
2. Value reasonableness: Are the magnitudes appropriate for the material type?
3. Scientific notation: Is Young's modulus in proper scientific notation (e.g., 2.0e11 Pa)?

Provide concise, specific feedback on what's missing or incorrect."""
    loss_fn = tg.TextLoss(eval_prompt)

    n_epochs = 25
    batch_size = 3
    metrics = {"loss": [], "prompts": []}

    for epoch in range(n_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{n_epochs}")

        optimizer.zero_grad()

        for i in tqdm(range(0, len(samples), batch_size), desc="Training"):
            batch = samples[i : i + batch_size]
            optimizer.zero_grad()

            batch_losses = []

            for sample in batch:
                input_data = json.dumps(
                    {
                        "material_type": sample["material_type"],
                        "semantic_usage": sample.get("semantic_usage"),
                    }
                )

                input_var = tg.Variable(
                    input_data,
                    requires_grad=False,
                    role_description="input material info",
                )

                prediction = model(input_var)

                loss = loss_fn(prediction)
                batch_losses.append(loss)

                parsed = parse_vlm_properties(prediction.value)
                gt = sample["ground_truth"]

                logger.info(f"  {sample['name']}:")
                logger.info(f"    Pred: {parsed}")
                logger.info(f"    GT:   {gt}")

            if batch_losses:
                total_loss = tg.sum(batch_losses)
                total_loss.backward()

                try:
                    optimizer.step()
                    logger.info("  Optimizer step successful")
                except Exception as e:
                    logger.warning(f"  Optimizer failed: {e}")

    with open(f"{OUTPUT_DIR}/final_prompt.txt", "w") as f:
        f.write(system_prompt_var.value)
    logger.info(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
