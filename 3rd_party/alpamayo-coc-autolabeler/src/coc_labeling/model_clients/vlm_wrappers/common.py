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

"""Shared utilities and abstract interface for VLM wrappers."""

import base64
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def encode_image(image: Any) -> Optional[str]:
    """Encode image bytes/path/ndarray to base64 JPEG string."""
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")
    if isinstance(image, str):
        if not os.path.exists(image):
            return None
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    if isinstance(image, np.ndarray):
        success, buffer = cv2.imencode(".jpg", image)
        if success:
            return base64.b64encode(buffer.tobytes()).decode("utf-8")
        raise ValueError("Failed to encode numpy array as image bytes.")
    raise ValueError("Image must be either a path, bytes, or numpy array.")


class BaseWrapper:
    """Base interface for model-specific wrapper implementations."""

    def __init__(self, model_config: Any) -> None:
        del model_config
        raise NotImplementedError

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        seed: int,
        temperature: float,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        json_schema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def add_message(self, role: str, m_type: str, content: Any) -> Dict[str, Any]:
        """Build a single model-formatted chat message entry."""
        raise NotImplementedError

    def add_message_seq(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert flat message entries into grouped model chat messages."""
        out_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            m_type = message["m_type"]
            content = message["content"]
            out_messages.append(self.add_message(role, m_type, content))

        combined_message = [out_messages[0]]
        role = out_messages[0]["role"]
        for next_message in out_messages[1:]:
            next_role = next_message["role"]
            if next_role == role:
                combined_message[-1]["content"].extend(next_message["content"])
            else:
                combined_message.append(next_message)
                role = next_role
        return combined_message
