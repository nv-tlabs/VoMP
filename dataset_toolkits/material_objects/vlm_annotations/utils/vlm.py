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
import hashlib
import json
import os
import base64
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

from dataset_toolkits.material_objects.vlm_annotations.utils.utils import (
    MATERIAL_RANGES_CSV,
)
from dataset_toolkits.material_objects.vlm_annotations.vlm_input_builder import (
    VLM_IMAGE_ROLES,
)
from dataset_toolkits.material_objects.vlm_annotations.vlm_output import (
    parse_vlm_properties,
    raw_vlm_output_sha256,
)
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from openai import OpenAI, APIConnectionError, APIStatusError
from qwen_vl_utils import process_vision_info
import traceback


QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"


class APIRetryGate:
    """Pause every API worker together after a transient service failure."""

    def __init__(self):
        self.condition = threading.Condition()
        self.resume_at = 0.0

    def wait(self):
        with self.condition:
            while True:
                remaining = self.resume_at - time.monotonic()
                if remaining <= 0:
                    return
                self.condition.wait(timeout=remaining)

    def pause(self, seconds):
        seconds = max(60.0, float(seconds))
        with self.condition:
            now = time.monotonic()
            resume_at = now + seconds
            should_log = (
                self.resume_at <= now or resume_at - self.resume_at >= 1.0
            )
            self.resume_at = max(self.resume_at, resume_at)
            self.condition.notify_all()
        if should_log:
            logging.warning(
                "Responses API failure; pausing all API requests for "
                "%.0f seconds before retrying",
                seconds,
            )


api_retry_gate = APIRetryGate()


def retry_after_seconds(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return 60.0

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return max(60.0, float(retry_after_ms) / 1000.0)
        except (TypeError, ValueError):
            pass

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(60.0, float(retry_after))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                return max(60.0, retry_at.timestamp() - time.time())
            except (AttributeError, OSError, TypeError, ValueError, OverflowError):
                pass
    return 60.0


@dataclass(frozen=True)
class ResponsesAPIConfig:
    """Settings for the Responses API."""

    base_url: str
    model: str
    reasoning_effort: str = "max"
    image_detail: str = "original"
    max_output_tokens: int = 16384
    concurrency: int = 4
    max_retries: int = 0

    @property
    def identity(self):
        policy = {
            "backend": "responses",
            "base_url": self.base_url,
            "image_detail": self.image_detail,
            "max_output_tokens": self.max_output_tokens,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }
        return "responses-api:" + json.dumps(
            policy,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

SYSTEM_PROMPT = """
You are a materials science expert estimating Young's modulus, Poisson's ratio, and density from images and context clues.

The selected part may be visible from the outside or may be an internal support, frame, reinforcement, hidden layer, or core. A soft exterior can hide a rigid frame, and a visible finish can cover a different underlying material. Use the selected part's shape, material-name clue, part-name clue, and role in the whole object to decide what physical material the selected part represents. Treat every name as a clue rather than ground truth.

When web search is available, use it to check the likely material and representative property values before answering.

You must provide numerical estimates for every material, including organic, biological, or unusual materials such as leaves, feathers, and paper. For natural materials, use comparable materials with known properties. Never answer with N/A, unknown, or another non-numeric value.

Reason in this order:
1. Identify the likely material class and subtype from the images and context clues.
2. Consider texture, color, reflectivity, shape, and the part's role in the object.
3. Decide whether the selected part is a bulk component, internal structure, cover, coating, or surface layer.
4. Reason through each property estimate using the likely material and physical form.
5. Use an optional reference range only when it agrees with the images and the selected part.

Formatting requirements:
- Young's modulus must use scientific notation followed by Pa.
- Poisson's ratio must be a decimal between 0.0 and 0.5 with no units.
- Density must be in kg/m^3.
- Use the exact property labels shown in the examples, with one property per line.
- Do not put explanatory text after the three values.
"""

STRICT_RETRY_USER_PROMPT_SUFFIX = (
    "\n\nIMPORTANT: You MUST provide numerical values for Young's modulus, "
    "Poisson's ratio, and density."
)
STRICT_RETRY_QUERY_PROMPT_SUFFIX = (
    "\n\nCRITICAL: You MUST provide valid numerical estimates for ALL three "
    "properties."
)


def current_vlm_text_context_hash(
    prompts: Mapping[str, str],
    *,
    material_ranges_path: str | os.PathLike[str] | None = None,
) -> str:
    """Hash the prompts and material reference table."""

    if not isinstance(prompts, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in prompts.items()
    ):
        raise TypeError("VLM prompt templates must be a string-to-string mapping")
    missing_templates = {"few_shot_examples", "query_prompt"} - set(prompts)
    if missing_templates:
        raise ValueError(
            "VLM prompt templates are missing: "
            + ", ".join(sorted(missing_templates))
        )
    ranges_path = Path(material_ranges_path or MATERIAL_RANGES_CSV)
    ranges_bytes = ranges_path.read_bytes()
    prompt_bytes = json.dumps(
        dict(prompts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    components = (
        ("system_prompt", SYSTEM_PROMPT.encode("utf-8")),
        ("prompt_templates", prompt_bytes),
        ("material_ranges_csv", ranges_bytes),
        (
            "retry_user_prompt_suffix",
            STRICT_RETRY_USER_PROMPT_SUFFIX.encode("utf-8"),
        ),
        (
            "retry_query_prompt_suffix",
            STRICT_RETRY_QUERY_PROMPT_SUFFIX.encode("utf-8"),
        ),
    )
    digest = hashlib.sha256(b"vlm-text-context\0")
    for label, payload in components:
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def encode_image(image_path):
    """Encode one request image as Base64 for an API data URL."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def request_image_paths(visual_image_paths):
    """Check and return the ordered VLM input images."""

    if visual_image_paths is None:
        raise ValueError(
            "VLM annotation requires the current four-image input tuple: original "
            "thumbnail, textured whole object, textured selected part, and "
            "selected material sphere"
        )
    paths = tuple(os.fspath(path) for path in visual_image_paths)
    if len(paths) != len(VLM_IMAGE_ROLES):
        raise ValueError(
            f"production VLM request requires exactly {len(VLM_IMAGE_ROLES)} "
            f"images, received {len(paths)}"
        )
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "production VLM request image(s) are missing: " + ", ".join(missing)
        )
    return tuple(zip(VLM_IMAGE_ROLES, paths))


def vlm_error_result(
    error,
    *,
    raw_output="",
    request_prompt="",
    web_search_used=False,
):
    """Record a model attempt that returned no response."""

    if not isinstance(raw_output, str):
        raw_output = ""
    return {
        "error": str(error),
        "raw_vlm_output": raw_output,
        "raw_vlm_output_sha256": raw_vlm_output_sha256(raw_output),
        "request_prompt": request_prompt,
        "web_search_used": bool(web_search_used),
    }


def response_field(item, field):
    """Read one Responses output field from either an SDK object or mapping."""

    if isinstance(item, Mapping):
        return item.get(field)
    return getattr(item, field, None)


def completed_web_search_calls(response):
    """Return completed web-search output items from a Responses API result."""

    output_items = getattr(response, "output", ()) or ()
    return tuple(
        item
        for item in output_items
        if response_field(item, "type") == "web_search_call"
        and response_field(item, "status") == "completed"
    )


def vlm_model_identity(model, processor):
    """Return the model name or API settings saved with annotations."""

    if isinstance(processor, ResponsesAPIConfig):
        return processor.identity
    if isinstance(processor, str):
        return f"api-model:{processor}"
    return getattr(model, "name_or_path", None) or type(model).__name__


def load_vlm_model(
    model_type="qwen",
    api_key=None,
    *,
    api_base_url=None,
    api_model=None,
    api_reasoning_effort="max",
    api_image_detail="original",
    api_max_output_tokens=16384,
    api_concurrency=4,
    api_max_retries=0,
    api_timeout=600.0,
):
    logging.info(f"Loading VLM model: {model_type}")

    if model_type == "qwen":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_ID,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        return model, processor

    if model_type not in {"api", "responses-api"}:
        raise ValueError(f"Unsupported VLM backend: {model_type}")

    api_base_url = api_base_url or os.getenv("VLM_API_BASE_URL")
    api_model = api_model or os.getenv("VLM_API_MODEL")
    if not api_base_url:
        raise ValueError("Set --api-base-url or VLM_API_BASE_URL for the API backend")
    if not api_model:
        raise ValueError("Set --api-model or VLM_API_MODEL for the API backend")

    config = ResponsesAPIConfig(
        base_url=api_base_url,
        model=api_model,
        reasoning_effort=api_reasoning_effort,
        image_detail=api_image_detail,
        max_output_tokens=api_max_output_tokens,
        concurrency=api_concurrency,
        max_retries=api_max_retries,
    )
    resolved_key = api_key or os.getenv("VLM_API_KEY")
    if not resolved_key:
        hostname = (urlsplit(config.base_url).hostname or "").casefold()
        environment_key = {
            "api.openai.com": "OPENAI_API_KEY",
            "inference-api.nvidia.com": "NVIDIA_API_KEY",
        }.get(hostname)
        resolved_key = os.getenv(environment_key) if environment_key else None
    if not resolved_key:
        raise ValueError(
            "Responses API key is missing; set VLM_API_KEY, OPENAI_API_KEY, "
            "or NVIDIA_API_KEY for the selected host"
        )
    if isinstance(api_timeout, bool) or float(api_timeout) <= 0:
        raise ValueError("API timeout must be positive")
    client = OpenAI(
        api_key=resolved_key,
        base_url=config.base_url,
        max_retries=0,
        timeout=float(api_timeout),
    )
    logging.info(
        "Responses API client created for %s with reasoning effort %s and "
        "concurrency %d",
        config.model,
        config.reasoning_effort,
        config.concurrency,
    )
    return client, config


def analyze_material_with_vlm(
    user_prompt,
    model,
    processor,
    prompts,
    visual_image_paths,
):
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("VLM user prompt is missing")
    request_images = request_image_paths(visual_image_paths)
    prompt_text = "\n\n".join(
        (user_prompt, prompts["few_shot_examples"], prompts["query_prompt"])
    )

    if isinstance(processor, ResponsesAPIConfig):
        output_text = ""
        web_search_used = False
        try:
            message_content = []
            for role, request_image_path in request_images:
                encoded_image = encode_image(request_image_path)
                message_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded_image}",
                        "detail": processor.image_detail,
                    }
                )
                logging.info("Added %s image to Responses API request", role)
            message_content.append(
                {
                    "type": "input_text",
                    "text": prompt_text,
                }
            )

            retry_count = 0
            while True:
                api_retry_gate.wait()
                try:
                    response = model.responses.create(
                        model=processor.model,
                        reasoning={"effort": processor.reasoning_effort},
                        tools=[{"type": "web_search"}],
                        input=[
                            {
                                "role": "system",
                                "content": [
                                    {"type": "input_text", "text": SYSTEM_PROMPT}
                                ],
                            },
                            {"role": "user", "content": message_content},
                        ],
                        max_output_tokens=processor.max_output_tokens,
                    )
                    break
                except (APIConnectionError, APIStatusError) as exc:
                    status_code = getattr(exc, "status_code", None)
                    transient = isinstance(exc, APIConnectionError) or (
                        isinstance(status_code, int) and status_code >= 400
                    )
                    if not transient:
                        raise
                    if retry_count >= processor.max_retries:
                        raise
                    retry_count += 1
                    api_retry_gate.pause(retry_after_seconds(exc))
            search_calls = completed_web_search_calls(response)
            web_search_used = bool(search_calls)
            logging.info(
                "Responses API completed web search: %s (completed calls: %d)",
                web_search_used,
                len(search_calls),
            )
            response_status = getattr(response, "status", None)
            if response_status != "completed":
                raise ValueError(
                    "Responses API did not complete successfully "
                    f"(status={response_status!r})"
                )
            if getattr(response, "error", None) is not None:
                raise ValueError("Responses API returned a response-level error")
            if getattr(response, "incomplete_details", None) is not None:
                raise ValueError("Responses API returned incomplete output")
            response_model = str(getattr(response, "model", None) or processor.model)
            output_text = response.output_text
            if not isinstance(output_text, str) or not output_text.strip():
                raise ValueError("Responses API returned no output text")

            parsed = parse_vlm_properties(output_text)
            parsed["response_model"] = response_model
            parsed["request_prompt"] = prompt_text
            parsed["web_search_used"] = web_search_used
            fingerprint = getattr(response, "system_fingerprint", None)
            if fingerprint is not None:
                parsed["response_system_fingerprint"] = str(fingerprint)
            return parsed
        except Exception as e:
            logging.error(f"Error in Responses API VLM analysis: {str(e)}")
            logging.exception("Responses API request failed")
            return vlm_error_result(
                e,
                raw_output=output_text,
                request_prompt=prompt_text,
                web_search_used=web_search_used,
            )

    user_content = []

    for role, request_image_path in request_images:
        file_uri = f"file://{os.path.abspath(request_image_path)}"
        logging.info("Using %s image: %s", role, file_uri)
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

        generated_ids = model.generate(
            **inputs, max_new_tokens=4096, do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = parse_vlm_properties(output_text)
        parsed["response_model"] = vlm_model_identity(model, processor)
        parsed["request_prompt"] = prompt_text
        return parsed

    except Exception as e:
        logging.error(f"Error in VLM analysis: {str(e)}")

        logging.error(traceback.format_exc())
        return vlm_error_result(e, request_prompt=prompt_text)
