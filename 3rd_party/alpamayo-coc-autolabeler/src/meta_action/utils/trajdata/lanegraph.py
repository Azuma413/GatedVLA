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

from typing import Any, Dict, List, Set

import numpy as np
from trajdata.data_structures.batch import SceneBatch
from trajdata.maps.vec_map_elements import MapElementType


# DPS lane relation utils
def _get_all_next_lanes(
    cur_lane_id: str, all_lanes: Any, all_next_lanes: Set[str], depth: int
) -> Set[str]:
    """Get all reachable next-lane IDs recursively up to a depth limit."""
    if cur_lane_id == "-1" or depth == 0:
        return all_next_lanes
    cur_next_lane_ids = all_lanes[cur_lane_id].next_lanes
    for cur_next_lane_id in cur_next_lane_ids:
        all_next_lanes.update(cur_next_lane_ids)
        _get_all_next_lanes(cur_next_lane_id, all_lanes, all_next_lanes, depth - 1)
    return all_next_lanes


def get_next_lanes(cur_lanes: Set[str], all_lanes: Any, depth: int = 5) -> Set[str]:
    """Get next-lane IDs from current lanes up to a specified depth."""
    all_next_lanes: Set[str] = set()
    for cur_lane in cur_lanes:
        _get_all_next_lanes(cur_lane, all_lanes, all_next_lanes, depth=depth)
    return all_next_lanes


def update_ego_lane_relation(
    scene_batch: SceneBatch, world_ego_xyzh: np.ndarray
) -> List[Dict[str, Set[str]]]:
    """Build per-timestep lane neighborhoods for the ego trajectory."""
    scene_batch_any: Any = scene_batch
    vector_map = scene_batch_any.vector_maps[0]
    all_lanes = vector_map.elements[MapElementType.ROAD_LANE]
    ego_lane_relations: List[Dict[str, Set[str]]] = []
    for ts in range(world_ego_xyzh.shape[0]):
        query_state = world_ego_xyzh[ts]
        if not np.all(np.isfinite(query_state)):
            # Skip invalid poses to avoid KDTree finite-value exceptions.
            current_lanes = set()
        else:
            try:
                lane_candidates = vector_map.get_current_lane(
                    query_state, max_dist=2, max_heading_error=np.pi / 3
                )
            except Exception:
                lane_candidates = []

            if len(lane_candidates) > 0:
                current_lanes = {cur_lane.id for cur_lane in lane_candidates}
            else:
                current_lanes = set()
        next_lanes = get_next_lanes(current_lanes, all_lanes, depth=5)
        left_lanes = set()
        right_lanes = set()
        for lane in current_lanes:
            lane_obj: Any = all_lanes[lane]
            left_lanes.update(lane_obj.adj_lanes_left)
            right_lanes.update(lane_obj.adj_lanes_right)
        for lane in next_lanes:
            lane_obj: Any = all_lanes[lane]
            left_lanes.update(lane_obj.adj_lanes_left)
            right_lanes.update(lane_obj.adj_lanes_right)
        cur_ego_lane_relation: Dict[str, Set[str]] = {
            "current_lanes": current_lanes,
            "next_lanes": next_lanes,
            "left_lanes": left_lanes,
            "right_lanes": right_lanes,
        }
        ego_lane_relations.append(cur_ego_lane_relation)

    return ego_lane_relations
