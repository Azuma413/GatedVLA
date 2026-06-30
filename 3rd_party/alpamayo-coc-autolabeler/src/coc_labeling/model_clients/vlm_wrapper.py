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

"""High-level VLM wrapper facade used by agents."""

import logging
from typing import Any, Dict, List, Optional

from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.model_clients.vlm_wrappers import (
    MODEL_WRAPPER_REGISTRY,
    BaseWrapper,
    DummyWrapper,
    OpenAIWrapper,
    QwenWrapper,
    create_model_wrapper,
    encode_image,
)

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class VLMWrapper:
    """Facade around model-specific wrappers with a unified interface."""

    def __init__(
        self,
        model_name: str,
        init_model: bool = True,
        ip_addr: Optional[str] = None,
        timeout_sec: int = TIMEOUT_MAX,
    ) -> None:
        self.vlm = create_model_wrapper(
            model_name=model_name,
            init_model=init_model,
            ip_addr=ip_addr,
            timeout_sec=timeout_sec,
        )

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        seed: int = 42,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        json_schema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run inference with the selected concrete wrapper."""
        return self.vlm.infer(
            messages=messages,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            json_schema=json_schema,
        )

    def add_message(self, role: str, m_type: str, content: Any) -> Dict[str, Any]:
        """Delegate message conversion to the active wrapper."""
        return self.vlm.add_message(role, m_type, content)

    def add_message_seq(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Delegate message sequence conversion to the active wrapper."""
        return self.vlm.add_message_seq(messages)


__all__ = [
    "BaseWrapper",
    "DummyWrapper",
    "OpenAIWrapper",
    "QwenWrapper",
    "MODEL_WRAPPER_REGISTRY",
    "encode_image",
    "VLMWrapper",
]
