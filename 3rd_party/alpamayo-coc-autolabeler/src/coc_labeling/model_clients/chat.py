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

"""Chat orchestration helpers for cloud/local model completion APIs."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from coc_labeling.model_clients.chat_utils_cloud import completion_with_backoff_cloud
from coc_labeling.model_clients.chat_utils_local_server import completion_with_backoff_local_server
from coc_labeling.model_clients.timeout import TIMEOUT_MAX

logging.getLogger("httpx").setLevel(logging.WARNING)

Message = Dict[str, Any]


def run_one_round_conversation(
    full_messages: List[Message],
    system_message: Optional[str] = None,
    user_message: Optional[Union[str, List[Message]]] = None,
    model_name: str = "",
    temperature: float = 0,
    timeout: int = TIMEOUT_MAX,
    json_schema: Optional[Dict[str, Any]] = None,
    max_completion_tokens: int = 8192,
    video_kwargs: Optional[Dict[str, Any]] = None,
    ip_addr: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
) -> Tuple[List[Message], Message]:
    """Run one chat-completion round and append assistant output in-place.

    Args:
        full_messages: Chat messages to send and mutate with the assistant response.
        system_message: Optional system message appended before the user message.
        user_message: Optional user message or message list appended before inference.
        model_name: Model identifier expected by the target backend.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds for local-server calls.
        json_schema: Optional JSON schema for structured decoding.
        max_completion_tokens: Maximum generated tokens.
        video_kwargs: Optional multimodal processor kwargs for vLLM.
        ip_addr: Local OpenAI-compatible server IP address.
        chat_template_kwargs: Optional chat-template kwargs for vLLM.
        backend: Explicit backend selection: ``"local"`` or ``"cloud"``.

    Returns:
        The mutated full message list and the assistant response message.
    """
    if system_message:
        full_messages.extend([{"role": "system", "content": system_message}])

    if isinstance(user_message, str):
        full_messages.extend([{"role": "user", "content": user_message}])
    elif isinstance(user_message, list):
        full_messages.extend(user_message)

    if json_schema is not None:
        extra_body = {
            "guided_json": json_schema,
            "guided_decoding_backend": "outlines",  # or xgrammar/lm-format-enforcer
        }
    else:
        extra_body = None

    if video_kwargs is not None:
        if extra_body is None:
            extra_body = {}
        extra_body["mm_processor_kwargs"] = video_kwargs

    if chat_template_kwargs:
        if extra_body is None:
            extra_body = {}
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    if not model_name:
        raise ValueError("model_name must be provided.")
    if backend not in {"local", "cloud"}:
        raise ValueError("backend must be explicitly set to 'local' or 'cloud'.")

    if backend == "local":
        response = completion_with_backoff_local_server(
            model=model_name,
            messages=full_messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
            stream=False,
            extra_body=extra_body,
            ip_addr=ip_addr,
        )
    else:
        try:
            response = completion_with_backoff_cloud(
                model=model_name,
                messages=full_messages,
                temperature=temperature,
            )
        except Exception:
            response = completion_with_backoff_cloud(
                model=model_name,
                messages=full_messages,
            )

    response_message: Message = {
        "content": response.choices[0].message.content,
        "role": "assistant",
    }
    full_messages.append(response_message)

    return full_messages, response_message
