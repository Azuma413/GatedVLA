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

from typing import Any, Dict, List, Optional, Type

import numpy as np
from trajdata.data_structures.batch import SceneBatch

from meta_action.data_structures.motion import TemporalMotionChunk
from meta_action.utils.constant import START_TS, STEP
from meta_action.utils.trajdata.agents import get_agent_data
from meta_action.utils.trajdata.lanegraph import update_ego_lane_relation


class TemporalScenario:
    """Per-clip container with agent trajectories and lane/map context."""

    def __init__(
        self,
        clip_id: str,
        scene_batch_data: SceneBatch,
        cfg: Optional[dict] = None,
        use_lane: bool = True,
    ) -> None:
        self.clip_id = clip_id
        self.cfg = cfg

        # scene_batch_data can be None if pair_motion_cache is provided
        self.all_agents_names = scene_batch_data.agent_names[0]
        self.agent_trajdata = get_agent_data(scene_batch_data)
        # Use actual sequence length from loaded batch (supports variable clip lengths).
        self.scene_len = int(self.agent_trajdata["agent_xyh"].shape[1])
        self.all_ts = list(range(START_TS, self.scene_len, STEP))
        # Cache per-agent intermediate segment outputs used by ego_meta_action.
        self.segment_cache: Dict[str, Dict[str, Any]] = {}
        # retain scene batch for downstream map queries
        self.scene_batch = scene_batch_data

        # initiate ego lane relation
        if use_lane:
            ego_xyzh = self.agent_trajdata["ego_xyzh"].numpy()
            world_ego_xyzh = self.get_world_states(ego_xyzh, scene_batch_data)
            self.ego_lr = update_ego_lane_relation(scene_batch_data, world_ego_xyzh)

    def get_world_states(self, states: np.ndarray, scene_batch: SceneBatch) -> np.ndarray:
        """Map agent-frame states to world-frame `[x, y, z, heading]` states."""
        ego_xyzh = states
        tf = scene_batch.centered_world_from_agent_tf[0]
        if hasattr(tf, "detach"):
            tf_np = tf.detach().cpu().numpy()
        else:
            tf_np = np.asarray(tf)

        ego_xy = ego_xyzh[:, 0:2]
        num_pts = ego_xy.shape[0]

        # trajdata_old may provide 4x4 transforms (with z); trajdata now uses 3x3 SE(2).
        if tf_np.shape == (4, 4):
            hom_xy = np.concatenate([ego_xy, np.zeros((num_pts, 1)), np.ones((num_pts, 1))], axis=1)
        elif tf_np.shape == (3, 3):
            hom_xy = np.concatenate([ego_xy, np.ones((num_pts, 1))], axis=1)
        else:
            raise ValueError(f"Unexpected centered_world_from_agent_tf shape: {tf_np.shape}")

        world_ego_xy = (hom_xy @ tf_np.T)[:, :2]
        dh = float(np.arctan2(tf_np[1, 0], tf_np[0, 0]))
        world_ego_h = ego_xyzh[:, -1] + dh
        world_ego_z = ego_xyzh[:, 2] if ego_xyzh.shape[1] >= 4 else np.zeros(num_pts)

        return np.concatenate([world_ego_xy, world_ego_z[:, None], world_ego_h[:, None]], axis=1)

    def get_tag_motions(
        self, motion_class: Type["TemporalMotionChunk"], ego_only: bool = True
    ) -> List["TemporalMotionChunk"]:
        """Generate temporal motion tags for each selected agent."""
        motions = []

        # if issubclass(motion_class, V_ActionTemporalMotion):
        for agent in self.all_agents_names:
            if (ego_only and agent == "ego") or (not ego_only):
                motions.extend(motion_class.get_motion_for_scenario(agent, self))

        return motions
