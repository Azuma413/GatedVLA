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

"""OpenAI-compatible server chat completion utilities (e.g., vLLM)."""

import logging
import os
from typing import Any, Optional

import httpx
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from coc_labeling.model_clients.timeout import TIMEOUT_MAX

transport = httpx.HTTPTransport(retries=1, http1=True, http2=False)
http_client = httpx.Client(transport=transport, timeout=TIMEOUT_MAX)
logger = logging.getLogger(__name__)


def _resolve_ip_addr(ip_addr: Optional[str]) -> str:
    """Resolve reasoning endpoint host from argument/env."""
    resolved = ip_addr or os.environ.get("COC_LABELING_MODEL_SERVER_IP")
    if not resolved:
        raise ValueError(
            "Missing reasoning model server IP. "
            "Pass `ip_addr` explicitly or set COC_LABELING_MODEL_SERVER_IP."
        )
    return resolved


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def completion_with_backoff_local_server(**kwargs: Any) -> Any:
    """Call reasoning endpoint with retry on transient connection issues."""
    ip_addr = _resolve_ip_addr(kwargs.pop("ip_addr", None))
    client = openai.Client(
        base_url=f"http://{ip_addr}:50000/v1",
        api_key="EMPTY",
        timeout=TIMEOUT_MAX,
        max_retries=1,
        http_client=http_client,
    )
    try:
        return client.chat.completions.create(**kwargs)
    except (httpx.ConnectError, openai.APIConnectionError) as exc:
        logger.warning("Connection issue with reasoning endpoint. Will retry. Error: %s", exc)
        raise ValueError("Connection issue with OpenAI API, possibly the server is down.") from exc
