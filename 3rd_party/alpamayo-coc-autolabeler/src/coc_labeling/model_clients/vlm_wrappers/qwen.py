# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Qwen-based VLM wrapper implementations for local and remote inference."""

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from coc_labeling.model_clients.chat import run_one_round_conversation
from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.model_clients.vlm_wrappers.common import BaseWrapper, encode_image

try:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
except ImportError:
    logging.warning("VLLM not available. Please install the required packages.")


vllm = None
QWEN_MODEL_SPECS: Dict[str, tuple[str, str, Optional[str]]] = {
    "qwen3_vl_235b_awq": (
        "models--QuantTrio--Qwen3-VL-235B-A22B-Instruct-AWQ",
        "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ",
        "awq",
    ),
    "qwen3.5_35b": (
        "models--Qwen--Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-35B-A3B",
        None,
    ),
    "qwen3.5_397b_fp8": (
        "models--Qwen--Qwen3.5-397B-A17B-FP8",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "fp8",
    ),
    "qwen3.5_27b_fp8": (
        "models--Qwen--Qwen3.5-27B-FP8",
        "Qwen/Qwen3.5-27B-FP8",
        "fp8",
    ),
    # Small dense VL model for fast pipeline smoke tests (fits 1x A100 40GB,
    # ~5GB download). Uses the same Qwen3-VL code path as the 235B backend.
    "qwen3_vl_2b": (
        "models--Qwen--Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-2B-Instruct",
        None,
    ),
}


def _vllm() -> Any:
    """Import and cache the vLLM module lazily."""
    global vllm
    if vllm is None:
        import vllm
    return vllm


def _schema_dict(json_schema: Any) -> Dict[str, Any]:
    """Return a Pydantic v1/v2 compatible JSON schema dictionary."""
    if hasattr(json_schema, "model_json_schema"):
        return json_schema.model_json_schema()
    if hasattr(json_schema, "schema"):
        return json_schema.schema()
    raise TypeError("json_schema must provide model_json_schema() or schema()")


def _structured_decoding_kwargs(json_schema: Optional[Any]) -> Dict[str, Any]:
    """Build version-compatible structured decoding kwargs for SamplingParams."""
    if json_schema is None:
        return {}

    schema = _schema_dict(json_schema)
    sampling_params_mod = _vllm().sampling_params

    if hasattr(sampling_params_mod, "StructuredOutputsParams"):
        return {"structured_outputs": sampling_params_mod.StructuredOutputsParams(json=schema)}
    if hasattr(sampling_params_mod, "GuidedDecodingParams"):
        return {"guided_decoding": sampling_params_mod.GuidedDecodingParams(json=schema)}

    raise AttributeError(
        "Current vLLM version does not expose StructuredOutputsParams or "
        "GuidedDecodingParams in vllm.sampling_params."
    )


def _parse_structured_output(output_text: Any) -> Any:
    """Parse structured output robustly across model variants."""
    if isinstance(output_text, (dict, list)):
        return output_text

    if not isinstance(output_text, str):
        raise TypeError(f"Expected string/dict/list output, got {type(output_text)}")

    cleaned = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()

    # Fast path: output is already pure JSON.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, str):
            return _parse_structured_output(parsed)
        return parsed
    except json.JSONDecodeError:
        pass

    # Common case: JSON is wrapped in fenced code blocks.
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                return _parse_structured_output(parsed)
            return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: locate the first decodable JSON object/array span.
    for idx, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(parsed, str):
                return _parse_structured_output(parsed)
            return parsed
        except json.JSONDecodeError:
            continue

    # Fallback for model outputs that look like Python dict/list literals.
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, str):
            return _parse_structured_output(parsed)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (ValueError, SyntaxError):
        pass

    # Final heuristic: evaluate bracketed segment if wrapper text exists.
    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        candidate = cleaned[first_obj : last_obj + 1]
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, SyntaxError):
            pass

    raise json.JSONDecodeError("No JSON object found in model output", cleaned, 0)


class QwenWrapper(BaseWrapper):
    """Qwen VLM wrapper for local vLLM inference or remote vLLM serving."""

    def __init__(
        self,
        model_name: str,
        init_model: bool = True,
        ip_addr: Optional[str] = None,
        timeout_sec: int = TIMEOUT_MAX,
    ) -> None:
        if model_name not in QWEN_MODEL_SPECS:
            raise ValueError(f"Invalid model: {model_name}")

        max_model_len = 128000
        cache_folder, hf_model_id, quantization = QWEN_MODEL_SPECS[model_name]
        cache_dir_env = os.environ.get("MODEL_CACHE_DIR") or os.environ.get("HF_HOME") or "./models"
        model_cache_dir = Path(cache_dir_env).expanduser().resolve()
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        for env_name in (
            "MODEL_CACHE_DIR",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
        ):
            if not os.environ.get(env_name):
                os.environ[env_name] = str(model_cache_dir)

        def _resolve_local_model(
            cache_folder: str, hf_repo_id: str, quant: Optional[str] = None
        ) -> tuple[str, str, Optional[str], bool]:
            """Resolve local HF cache path, or fall back to remote repo id."""
            local_root = model_cache_dir / cache_folder
            if not local_root.exists():
                logging.warning(
                    "Local model cache not found: %s. Falling back to remote download for %s.",
                    local_root,
                    hf_repo_id,
                )
                return hf_repo_id, hf_repo_id, quant, False

            # If files are already materialized at root, use it directly.
            if (local_root / "config.json").exists():
                return str(local_root), hf_repo_id, quant, True

            # HF cache layout: models--*/snapshots/<revision>/...
            snapshots_dir = local_root / "snapshots"
            if snapshots_dir.exists():
                snapshots = sorted(
                    (p for p in snapshots_dir.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for snapshot in snapshots:
                    if (snapshot / "config.json").exists():
                        return str(snapshot), hf_repo_id, quant, True

            logging.warning(
                "No loadable local model files found under %s. Falling back to "
                "remote download for %s.",
                local_root,
                hf_repo_id,
            )
            return hf_repo_id, hf_repo_id, quant, False

        if init_model:
            (
                model_name,
                hf_model_id,
                quantization,
                local_files_only,
            ) = _resolve_local_model(cache_folder, hf_model_id, quantization)
        else:
            model_name = hf_model_id
            local_files_only = False
        self.model_name = model_name
        self.hf_model_id = hf_model_id
        self.remote_model_name = hf_model_id
        self.timeout_sec = timeout_sec

        self.fps = 2
        init_kwargs = {
            "model": model_name,
            "tensor_parallel_size": torch.cuda.device_count(),
            "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95")),
            "max_model_len": max_model_len,
            "max_num_seqs": 1,
            "limit_mm_per_prompt": {"image": 0, "video": 16},
            "download_dir": str(model_cache_dir),
            "seed": 42,
            "safetensors_load_strategy": "eager",
            "quantization": quantization,
        }
        if quantization == "awq":
            # vLLM AWQ only supports fp16 compute dtype.
            init_kwargs["dtype"] = "float16"

        if "Qwen3-VL" in hf_model_id:
            # Expert parallelism only applies to MoE variants (e.g. 235B-A22B);
            # dense Qwen3-VL models (2B/4B/8B/32B) have no experts and vLLM raises
            # "Number of experts must be > 0 when expert parallelism is enabled".
            if "A22B" in hf_model_id or "A3B" in hf_model_id:
                init_kwargs["enable_expert_parallel"] = True
            init_kwargs["limit_mm_per_prompt"]["image"] = 0
            init_kwargs["limit_mm_per_prompt"]["video"] = 1
            init_kwargs["mm_encoder_tp_mode"] = "data"
            init_kwargs["mm_processor_cache_gb"] = 0

        logging.info("-" * 100)
        logging.info("Offline VLLM Configuration:")
        logging.info(init_kwargs)

        if init_model:
            self.llm = _vllm().LLM(**init_kwargs)
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=str(model_cache_dir),
                local_files_only=local_files_only,
            )
            self.remote_inference = False
        else:
            self.remote_inference = True
            self.ip_addr = ip_addr
            if self.ip_addr is not None:
                logging.info("IP address set to %s for VLM.", self.ip_addr)

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        json_schema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run Qwen inference and return the normalized wrapper response payload."""
        disable_thinking = "Qwen3.5" in self.hf_model_id
        if self.remote_inference:
            if json_schema is not None:
                _, response_message = run_one_round_conversation(
                    full_messages=messages,
                    system_message=None,
                    user_message=None,
                    model_name=self.remote_model_name,
                    temperature=temperature,
                    json_schema=_schema_dict(json_schema),
                    video_kwargs={"fps": [self.fps]},
                    ip_addr=self.ip_addr,
                    timeout=self.timeout_sec,
                    chat_template_kwargs=({"enable_thinking": False} if disable_thinking else None),
                    backend="local",
                )
            else:
                _, response_message = run_one_round_conversation(
                    full_messages=messages,
                    system_message=None,
                    user_message=None,
                    model_name=self.remote_model_name,
                    temperature=temperature,
                    video_kwargs={"fps": [self.fps]},
                    ip_addr=self.ip_addr,
                    timeout=self.timeout_sec,
                    chat_template_kwargs=({"enable_thinking": False} if disable_thinking else None),
                    backend="local",
                )
            output_text = response_message["content"]
        else:
            if disable_thinking:
                try:
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            vision_info_kwargs = {"return_video_kwargs": True}
            is_qwen3_family = "Qwen3-VL" in self.hf_model_id or "Qwen3.5" in self.hf_model_id
            if "Qwen3-VL" in self.hf_model_id:
                vision_info_kwargs["image_patch_size"] = self.processor.video_processor.patch_size

            if is_qwen3_family:
                # Nightly vLLM paths for Qwen3-family may require video metadata.
                vision_info_kwargs["return_video_metadata"] = True
                try:
                    vision_outputs = process_vision_info(messages, **vision_info_kwargs)
                except TypeError:
                    vision_info_kwargs.pop("return_video_metadata", None)
                    vision_outputs = process_vision_info(messages, **vision_info_kwargs)
            else:
                vision_outputs = process_vision_info(messages, **vision_info_kwargs)

            video_metadata = None
            if len(vision_outputs) == 4:
                (
                    image_inputs,
                    video_inputs,
                    video_kwargs,
                    video_metadata,
                ) = vision_outputs
            elif len(vision_outputs) == 3:
                image_inputs, video_inputs, video_kwargs = vision_outputs
                if isinstance(video_kwargs, dict):
                    video_metadata = video_kwargs.get("video_metadata")
            else:
                raise ValueError(
                    f"Unexpected process_vision_info output length: {len(vision_outputs)}"
                )

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
                if is_qwen3_family and video_metadata is not None:
                    mm_data["video_metadata"] = video_metadata

            if not isinstance(video_kwargs, dict):
                video_kwargs = {}
            if is_qwen3_family and video_metadata is not None:
                video_kwargs.setdefault("video_metadata", video_metadata)

            input_ids = self.processor.tokenizer(prompt).input_ids
            llm_inputs = {
                "prompt": prompt,
                "prompt_token_ids": input_ids,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }

            sampling_params = _vllm().SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                **_structured_decoding_kwargs(json_schema),
            )
            outputs = self.llm.generate(
                [llm_inputs], sampling_params=sampling_params, use_tqdm=False
            )
            output_text = outputs[0].outputs[0].text

        if json_schema is not None:
            try:
                output_text = _parse_structured_output(output_text)
            except (json.JSONDecodeError, TypeError) as exc:
                logging.error(
                    "Error parsing inference output for model %s: %r",
                    self.model_name,
                    output_text,
                )
                logging.error("Structured parse exception: %s", exc)
                raise

        return {
            "finish_reason": "completed",
            "content": output_text,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "system_fingerprint": None,
        }

    def batch_infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """Reject batch inference until the backend-specific implementation is added."""
        del messages, max_tokens, seed, temperature, top_p, repetition_penalty
        raise NotImplementedError(
            "QwenWrapper.batch_infer is not implemented for current vLLM backend."
        )

    def add_message(self, role: str, m_type: str, content: Any) -> Dict[str, Any]:
        """Build a Qwen-compatible chat message entry."""
        if role not in ["user", "system", "assistant"]:
            raise ValueError(f"Invalid message role: {role}")
        if m_type not in ["text", "image", "video"]:
            raise ValueError(f"Invalid message type: {m_type}")

        if m_type == "text":
            return {"role": role, "content": [{"type": "text", "text": content}]}
        if m_type == "image":
            return {
                "role": role,
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{encode_image(content)}",
                    },
                ],
            }
        if m_type == "video":
            if not isinstance(content, list):
                raise TypeError("Video content must be a list of frames.")
            video = []
            for f_i in content:
                base64_str = encode_image(f_i)
                if self.remote_inference:
                    video.append(base64_str)
                else:
                    video.append(f"data:image/jpeg;base64,{base64_str}")

            if self.remote_inference:
                return {
                    "role": role,
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/jpeg;base64,{','.join(video)}"},
                        }
                    ],
                }
            return {
                "role": role,
                "content": [{"type": "video", "video": video, "fps": 2}],
            }
        raise ValueError(f"Invalid message type: {m_type}")
