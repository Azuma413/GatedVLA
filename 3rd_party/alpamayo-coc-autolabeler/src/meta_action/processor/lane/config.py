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

from meta_action.processor.common import Threshold

LANE_SEGMENTATION_THRESHOLDS = [
    Threshold("heading_change_rate_rad_m", 0.01),  # Steer Left
    Threshold("heading_change_rate_rad_m", -0.01),  # Steer Right
]

LANE_SEGMENTATION_CATEGORICAL = [
    "lane_decision",
]


# Temporal smoothing knobs for per-timestep lane decision.
# - LOOKAHEAD uses future lane id relation checks (old-algorithm spirit).
# - WINDOW_STEP enables multi-window checks from WINDOW_STEP..LOOKAHEAD.
# - CHANGE_MIN_RUN removes very short left/right lane-change flickers.
LANE_DECISION_LOOKAHEAD_TICKS = 20
LANE_DECISION_WINDOW_STEP_TICKS = 5
LANE_CHANGE_MIN_RUN_TICKS = 4


# Caption-specific minimum segment lengths (in ticks) used for lane cleanup.
DEFAULT_LANE_MIN_LEN_TICKS = 6
LANE_MIN_LEN_TICKS_BY_CAPTION = {
    "LaneKeep": DEFAULT_LANE_MIN_LEN_TICKS,
    "LeftLaneChange": 5,
    "RightLaneChange": 5,
    "SlightlyShiftLeft": DEFAULT_LANE_MIN_LEN_TICKS,
    "SlightlyShiftRight": DEFAULT_LANE_MIN_LEN_TICKS,
    "TurnLeft": DEFAULT_LANE_MIN_LEN_TICKS,
    "TurnRight": DEFAULT_LANE_MIN_LEN_TICKS,
    # "FollowCurveLeft": DEFAULT_LANE_MIN_LEN_TICKS,
    # "FollowCurveRight": DEFAULT_LANE_MIN_LEN_TICKS,
}
