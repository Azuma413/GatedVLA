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

"""Dummy model wrapper for testing/integration plumbing."""

from typing import Any, Dict, List, Optional

from coc_labeling.model_clients.vlm_wrappers.common import BaseWrapper


class DummyWrapper(BaseWrapper):
    """Return deterministic placeholder responses without model calls."""

    def __init__(
        self,
        model_name: str,
        init_model: bool = True,
        ip_addr: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ) -> None:
        del init_model, ip_addr, timeout_sec
        self.model_name = model_name

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 0,
        seed: int = 0,
        temperature: float = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        json_schema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        del messages, max_tokens, seed, temperature, top_p, repetition_penalty, json_schema
        return {
            "finish_reason": "completed",
            "content": "Dummy response.",
            "prompt_tokens": 0,
            "response_tokens": 0,
            "system_fingerprint": None,
        }

    def add_message(self, role: str, m_type: str, content: Any) -> Dict[str, Any]:
        """Build a minimal message payload for interface compatibility."""
        return {"role": role, "m_type": m_type, "content": content}

    def add_message_seq(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return input message sequence unchanged."""
        return messages
