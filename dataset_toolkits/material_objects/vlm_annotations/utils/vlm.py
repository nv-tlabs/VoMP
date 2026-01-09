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

import logging
import os
import re
import torch
import base64

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from openai import OpenAI

from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    find_reference_materials,
    load_material_ranges,
    parse_numerical_range_str,
)

SYSTEM_PROMPT = """
You are a materials science expert specializing in analyzing material properties from visual appearances and physical data. Your task is to provide precise numerical estimates for Young's modulus, Poisson's ratio, and density based on the images and context provided.

Important Context: The material segment you are analyzing may be an internal component or structure that is not visible from the outside of the object. For example:
- Internal support structures, frames, or reinforcements
- Hidden layers or core materials
- Components enclosed within the outer shell
- Structural elements that are only visible when the object is disassembled

When analyzing:
- Consider that the material might be completely hidden from external view
- Use the semantic usage and material type hints to infer properties of internal components
- Internal structural components often have different properties than visible surfaces
- For example, a soft exterior might hide a rigid internal frame

Critical Instruction: You MUST provide numerical estimates for ALL materials, even organic, biological, or unusual materials like leaves, feathers, or paper. 
- For organic materials, estimate properties based on similar natural materials with known values
- For leaves, consider them as thin plant fiber composites with values similar to paper or dried plant fibers
- Never respond with "N/A" or any non-numeric value in your property estimates

When analyzing materials, use step-by-step reasoning:
1. First identify the likely material class and subtype based on visual appearance (if visible) or contextual clues (if internal)
2. Consider how texture, color, and reflectivity inform your understanding of the material (when visible)
3. Incorporate the provided physical properties and contextual usage information
4. For each mechanical property, reason through how the visual and physical attributes lead to your estimate
5. Consider how the material compares to reference materials with known properties
6. If the material appears to be internal/hidden, use the object type and usage context to make informed estimates

Important Formatting Requirements:
- Young's modulus must be provided in scientific notation followed by "Pa" (e.g., 2.0e11 Pa)
- Poisson's ratio must be a simple decimal between 0.0 and 0.5 with no units (e.g., 0.34)
- Density must be provided in kg/m^3 (e.g., 7800 kg/m^3)
- Each property must be on its own line with exactly the label shown in the examples
- Do not include explanatory text or parenthetical notes after the values
- ALWAYS provide numerical values, never text like "N/A" or "unknown"
"""

MATERIAL_RANGES = load_material_ranges()


def encode_image(image_path):
    """Encode image to base64 for Gemini API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_vlm_properties(output_text):
    """
    Parses VLM output text to extract analysis and mechanical properties.
    Uses multiple strategies to handle different formats and potential parsing issues.

    Args:
        output_text (str): Raw output text from the VLM

    Returns:
        dict: Dictionary with extracted analysis and properties
    """
    analysis = ""
    youngs_modulus = None
    poissons_ratio = None
    density = None

    analysis_lines = []
    capturing_analysis = False

    for line in output_text.split("\n"):
        line = line.strip()
        if line.lower().startswith("analysis:"):
            capturing_analysis = True
            analysis_lines.append(line.split(":", 1)[1].strip())
        elif capturing_analysis and (
            line.lower().startswith("young")
            or line.lower().startswith("poisson")
            or line.lower().startswith("density")
        ):
            capturing_analysis = False
        elif capturing_analysis and line:
            analysis_lines.append(line)

    analysis = "\n".join(analysis_lines)

    try:

        modulus_line = ""
        for line in output_text.split("\n"):
            if line.lower().startswith("young"):
                modulus_line = line
                break

        if modulus_line:

            numeric_matches = re.findall(r"([0-9][0-9.eE+\-]*)", modulus_line)
            if numeric_matches:
                value_str = numeric_matches[0].replace(",", "").strip()
                youngs_modulus = float(value_str)
                logging.info(
                    f"Successfully extracted Young's modulus: {youngs_modulus} from line '{modulus_line}'"
                )

                if youngs_modulus < 1000 and "g" in modulus_line.lower():
                    youngs_modulus *= 1.0e9
                    logging.info(
                        f"Converted Young's modulus from GPa to Pa: {youngs_modulus}"
                    )
    except Exception as e:
        logging.error(f"First Young's modulus extraction approach failed: {e}")

    if youngs_modulus is None:
        try:
            modulus_pattern = re.compile(
                r"young'?s\s+modulus\s*[:\-]?\s*([0-9][0-9.eE+\-]*)", re.IGNORECASE
            )
            modulus_match = modulus_pattern.search(output_text)
            if modulus_match:
                value_str = modulus_match.group(1).replace(",", "")
                youngs_modulus = float(value_str)

                if (
                    youngs_modulus < 1000
                    and "g" in output_text.lower().split("young")[1].split("\n")[0]
                ):
                    youngs_modulus *= 1.0e9

                logging.info(
                    f"Second Young's modulus extraction approach succeeded: {youngs_modulus}"
                )
        except Exception as e:
            logging.error(f"Second Young's modulus extraction approach failed: {e}")

    try:

        ratio_line = ""
        for line in output_text.split("\n"):
            if line.lower().startswith("poisson"):
                ratio_line = line
                break

        if ratio_line:

            numeric_matches = re.findall(r"([0-9][0-9.eE+\-]*)", ratio_line)
            if numeric_matches:
                value_str = numeric_matches[0].replace(",", "").strip()
                poissons_ratio = float(value_str)
                logging.info(
                    f"Successfully extracted Poisson's ratio: {poissons_ratio}"
                )

                if poissons_ratio < 0 or poissons_ratio > 0.5:
                    logging.warning(
                        f"Poisson's ratio out of normal range: {poissons_ratio}"
                    )
    except Exception as e:
        logging.error(f"First Poisson's ratio extraction approach failed: {e}")

    if poissons_ratio is None:
        try:
            ratio_pattern = re.compile(
                r"poisson'?s\s+ratio\s*[:\-]?\s*([0-9][0-9.eE+\-]*)", re.IGNORECASE
            )
            ratio_match = ratio_pattern.search(output_text)
            if ratio_match:
                value_str = ratio_match.group(1).replace(",", "")
                poissons_ratio = float(value_str)
                logging.info(
                    f"Second Poisson's ratio extraction approach succeeded: {poissons_ratio}"
                )
        except Exception as e:
            logging.error(f"Second Poisson's ratio extraction approach failed: {e}")

    try:

        density_line = ""
        for line in output_text.split("\n"):
            if line.lower().startswith("density"):
                density_line = line
                break

        if density_line:

            cleaned_line = density_line.replace("kg/m³", "kg/m3").replace(
                "kg/m^3", "kg/m3"
            )

            numeric_matches = re.findall(r"([0-9][0-9,.eE+\-]*)", cleaned_line)
            if numeric_matches:
                value_str = numeric_matches[0].replace(",", "").strip()
                density = float(value_str)
                logging.info(f"Successfully extracted density: {density}")
    except Exception as e:
        logging.error(f"First density extraction approach failed: {e}")

    if density is None:
        try:

            density_pattern = re.compile(
                r"density\s*[:\-]?\s*([0-9][0-9,.eE+\-]*)", re.IGNORECASE
            )
            density_match = density_pattern.search(output_text)
            if density_match:
                value_str = density_match.group(1).replace(",", "")
                density = float(value_str)
                logging.info(f"Second density extraction approach succeeded: {density}")
        except Exception as e:
            logging.error(f"Second density extraction approach failed: {e}")

    if density is None:
        try:
            simple_pattern = re.compile(
                r"density.*?(\d+(?:\.\d+)?)", re.IGNORECASE | re.DOTALL
            )
            simple_match = simple_pattern.search(output_text)
            if simple_match:
                density = float(simple_match.group(1))
                logging.info(f"Last resort density extraction succeeded: {density}")
        except Exception as e:
            logging.error(f"Last resort density extraction failed: {e}")

    result = {
        "vlm_analysis": analysis,
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "density": density,
        "raw_vlm_output": output_text,
    }

    missing_properties = []
    if youngs_modulus is None:
        missing_properties.append("Young's modulus")
    if poissons_ratio is None:
        missing_properties.append("Poisson's ratio")
    if density is None:
        missing_properties.append("density")

    if missing_properties:
        logging.warning(
            f"Missing properties in VLM output: {', '.join(missing_properties)}"
        )
    else:
        logging.info(
            f"All properties extracted: Young's modulus={youngs_modulus}, Poisson's ratio={poissons_ratio}, Density={density}"
        )

    return result


def load_vlm_model(model_type="qwen", api_key=None):
    logging.info(f"Loading VLM model: {model_type}")

    if model_type == "qwen":
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-72B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-72B-Instruct",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            logging.info("Qwen VLM model loaded successfully")
            return model, processor
        except Exception as e:
            logging.error(f"Failed to load Qwen VLM model: {str(e)}")
            return None, None

    elif model_type.startswith("gemini"):
        if not api_key:
            logging.error("API key is required for Gemini models")
            return None, None

        try:

            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

            logging.info(f"Gemini client created successfully for model: {model_type}")

            return client, model_type
        except Exception as e:
            logging.error(f"Failed to create Gemini client: {str(e)}")
            return None, None

    elif model_type == "llama-3.2-90b-vision" or model_type == "llama-vision":

        if not api_key:

            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                logging.error(
                    "NVIDIA API key is required for llama-3.2-90b-vision model"
                )
                return None, None

        try:

            client = OpenAI(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1",
            )

            logging.info("NVIDIA llama-3.2-90b-vision client created successfully")

            return client, "nvdev/meta/llama-3.2-90b-vision-instruct"
        except Exception as e:
            logging.error(f"Failed to create NVIDIA llama vision client: {str(e)}")
            return None, None

    else:
        logging.error(f"Unsupported model type: {model_type}")
        return None, None


def analyze_material_with_vlm(
    image_path,
    material_info,
    model,
    processor,
    thumbnail_path,
    dataset_name,
    PROMPTS,
    make_user_prompt,
    parse_vlm_output,
):
    if dataset_name == "simready":
        material_type = material_info.get("material_type", "unknown")
        opacity = material_info.get("opacity", "opaque")
        density = material_info.get("density", None)
        dynamic_friction = material_info.get("dynamic_friction", None)
        static_friction = material_info.get("static_friction", None)
        restitution = material_info.get("restitution", None)
        semantic_usage = material_info.get("semantic_usage", "")

        if "user_prompt" in material_info:
            part1 = material_info["user_prompt"]
        else:

            part1 = make_user_prompt(
                material_type,
                opacity,
                density,
                dynamic_friction,
                static_friction,
                restitution,
                semantic_usage,
                has_texture_sphere=image_path is not None,
            )
    elif (
        dataset_name == "residential"
        or dataset_name == "commercial"
        or dataset_name == "vegetation"
    ):
        material_type = material_info.get("material_type", "unknown")
        semantic_usage = material_info.get("semantic_usage", "")
        object_name = material_info.get("object_name", "")

        if "user_prompt" in material_info:
            part1 = material_info["user_prompt"]
        else:

            part1 = make_user_prompt(
                material_type,
                semantic_usage,
                object_name,
                has_texture_sphere=image_path is not None,
            )

    reference_info = ""
    ref_materials = find_reference_materials(MATERIAL_RANGES, material_type)
    if ref_materials:
        reference_info += "\nAdditional reference material property ranges to help you make accurate estimations:\n"
        for rm in ref_materials:
            try:
                y_min, y_max = parse_numerical_range_str(rm["youngs"])
                p_min, p_max = parse_numerical_range_str(rm["poisson"])
                d_min, d_max = parse_numerical_range_str(rm["density"])

                youngs_range_str = f"{y_min}" if y_min == y_max else f"{y_min}-{y_max}"
                poisson_range_str = f"{p_min}" if p_min == p_max else f"{p_min}-{p_max}"
                density_range_str = f"{d_min}" if d_min == d_max else f"{d_min}-{d_max}"

            except ValueError as e:
                logging.warning(
                    f"Could not parse range for material {rm.get('name', 'Unknown')}: {e}. Skipping this reference material."
                )
                continue

            reference_info += (
                f"  - {rm['name']}: Young's modulus range {youngs_range_str} GPa, "
                f"Poisson's ratio range {poisson_range_str}, Density range {density_range_str} kg/m^3\n"
            )

    prompt_text = f"""
{part1}

{reference_info}

{PROMPTS["few_shot_examples"]}

{PROMPTS["query_prompt"]}
"""

    if isinstance(processor, str) and processor.startswith("gemini"):

        try:

            message_content = []

            message_content.append(
                {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + prompt_text}
            )

            if thumbnail_path and os.path.exists(thumbnail_path):
                base64_thumbnail = encode_image(thumbnail_path)
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_thumbnail}"
                        },
                    }
                )
                logging.info(f"Added thumbnail image to Gemini request")

            if image_path and os.path.exists(image_path):
                base64_texture = encode_image(image_path)
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_texture}"},
                    }
                )
                logging.info(f"Added texture sphere image to Gemini request")

            response = model.chat.completions.create(
                model=processor,
                messages=[{"role": "user", "content": message_content}],
                max_tokens=4096,
                temperature=0.7,
            )

            output_text = response.choices[0].message.content

            return parse_vlm_properties(output_text)

        except Exception as e:
            logging.error(f"Error in Gemini VLM analysis: {str(e)}")
            import traceback

            logging.error(traceback.format_exc())
            return {"error": str(e), "raw_vlm_output": "Error generating response"}

    elif isinstance(processor, str) and (
        processor.startswith("nvdev") or "llama-3.2-90b-vision" in processor
    ):

        try:

            message_content = []

            message_content.append(
                {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + prompt_text}
            )

            if thumbnail_path and os.path.exists(thumbnail_path):
                base64_thumbnail = encode_image(thumbnail_path)
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_thumbnail}"
                        },
                    }
                )
                logging.info(f"Added thumbnail image to NVIDIA llama vision request")

            if image_path and os.path.exists(image_path):
                base64_texture = encode_image(image_path)
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_texture}"},
                    }
                )
                logging.info(
                    f"Added texture sphere image to NVIDIA llama vision request"
                )

            response = model.chat.completions.create(
                model=processor,
                messages=[{"role": "user", "content": message_content}],
                max_tokens=4096,
                temperature=0.5,
            )

            output_text = response.choices[0].message.content

            return parse_vlm_properties(output_text)

        except Exception as e:
            logging.error(f"Error in NVIDIA llama vision VLM analysis: {str(e)}")
            import traceback

            logging.error(traceback.format_exc())
            return {"error": str(e), "raw_vlm_output": "Error generating response"}

    else:

        user_content = []

        if thumbnail_path:
            thumb_uri = f"file://{os.path.abspath(thumbnail_path)}"
            logging.info(f"Using thumbnail image: {thumb_uri}")
            user_content.append({"type": "image", "image": thumb_uri})

        if image_path:

            absolute_image_path = os.path.abspath(image_path)
            file_uri = f"file://{absolute_image_path}"
            user_content.append({"type": "image", "image": file_uri})

        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )

            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return parse_vlm_properties(output_text)

        except Exception as e:
            logging.error(f"Error in VLM analysis: {str(e)}")
            import traceback

            logging.error(traceback.format_exc())
            return {"error": str(e), "raw_vlm_output": "Error generating response"}
