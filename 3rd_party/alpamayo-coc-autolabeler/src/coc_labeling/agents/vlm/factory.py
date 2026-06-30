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

from __future__ import annotations

from typing import Any, Callable

from coc_labeling.agents.vlm.coc_agent import VLMCoCAgent, VLMCoCRemoteAgent
from coc_labeling.model_clients.timeout import TIMEOUT_MAX


class VLMFactory:
    """Factory wrapper that dispatches to a concrete VLM agent."""

    def __init__(
        self,
        model_name: str,
        agent_config: Any,
        vector_config: Any,
        data_loader_config: Any | None = None,
        ip_addr: str | None = None,
        timeout_sec: int = TIMEOUT_MAX,
    ) -> None:
        """Instantiate the concrete VLM agent selected by ``agent_config.agent_name``.

        Args:
            model_name: Model identifier to pass to concrete agent.
            agent_config: Agent configuration containing ``agent_name``.
            vector_config: Vector configuration passed through to the agent.
            data_loader_config: Resolved data-loader config used for prompt timing.
            timeout_sec: Request timeout in seconds for remote model calls.

        Raises:
            ValueError: If ``agent_name`` is not in the local registry.
        """
        agent_registry: dict[str, Callable[..., Any]] = {
            "VLMCoCAgent": VLMCoCAgent,
            "VLMCoCRemoteAgent": VLMCoCRemoteAgent,
        }
        agent_name = agent_config.agent_name
        if agent_name not in agent_registry:
            supported = ", ".join(sorted(agent_registry))
            raise ValueError(f"Unsupported VLM agent '{agent_name}'. Supported: {supported}")

        self.agent_config = agent_config
        self.agent = agent_registry[agent_name](
            model_name,
            agent_config,
            vector_config,
            data_loader_config=data_loader_config,
            ip_addr=ip_addr,
            timeout_sec=timeout_sec,
        )

    def run(
        self, data: dict[str, Any], save_root: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run the selected concrete VLM agent."""
        use_metaaction_check = bool(
            getattr(self.agent_config, "use_metaaction_check_pipeline", False)
        )
        if use_metaaction_check:
            return self.agent.run_metaaction_check(data, save_root)
        return self.agent.run(data, save_root)
