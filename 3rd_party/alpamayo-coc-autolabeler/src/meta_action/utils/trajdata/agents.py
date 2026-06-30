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

import logging
from typing import Any, Dict, Optional

import torch
from trajdata.data_structures.batch import SceneBatch

logger = logging.getLogger(__name__)


def get_agent_data(
    scene_batch: Optional[SceneBatch], debug: bool = False
) -> Optional[Dict[str, Any]]:
    """Build agent kinematic tensors from a trajdata scene batch."""
    agent_data: Dict[str, Any] = {}

    if scene_batch is None:
        return None

    scene_ts = scene_batch.scene_ts
    agent_curr = scene_batch.agent_hist[0]  # num_agents x (num_his + 1) x 6
    agent_fut = scene_batch.agent_fut[0]  # num_agents x num_fut x 6

    if debug:
        logger.info(
            "scene batch current ts is %s, history is %s, future is %s, dt is %s",
            scene_ts,
            agent_curr.as_tensor().size(1) - 1,
            agent_fut.as_tensor().size(1),
            scene_batch.dt,
        )
        logger.info("agent_curr %s", agent_curr.size())
        logger.info("agent_fut %s", agent_fut.size())

    # obtain xyh and speed of all agents across all ts
    # the past and future do not matter since we use them altogether
    agent_xyh = torch.cat(
        [agent_curr.as_format("x,y,h"), agent_fut.as_format("x,y,h")], dim=1
    )  # (N, T, 3)
    agent_speed = torch.cat(
        [agent_curr.as_format("xd,yd"), agent_fut.as_format("xd,yd")], dim=1
    ).norm(dim=-1)  # (N, T)
    agent_vec_speed = torch.cat(
        [agent_curr.as_format("xd,yd"), agent_fut.as_format("xd,yd")], dim=1
    )  # (N, T, 2)
    agent_speed_along_heading = torch.einsum(
        "ijk,kij->ij",
        agent_vec_speed,
        torch.stack([agent_xyh[:, :, 2].cos(), agent_xyh[:, :, 2].sin()]),
    )

    # default ego idx is 0
    ego_idx = 0
    ego_xyzh = torch.cat([agent_curr.as_format("x,y,z,h"), agent_fut.as_format("x,y,z,h")], dim=1)[
        ego_idx
    ]  # (N, T, 3))

    agent_data["agent_names"] = scene_batch.agent_names[0]
    agent_data["agent_xyh"] = agent_xyh
    agent_data["agent_speed"] = agent_speed
    agent_data["agent_speed_along_heading"] = agent_speed_along_heading
    agent_data["agent_type"] = scene_batch.agent_type[0]
    agent_data["ego_xyzh"] = ego_xyzh

    return agent_data
