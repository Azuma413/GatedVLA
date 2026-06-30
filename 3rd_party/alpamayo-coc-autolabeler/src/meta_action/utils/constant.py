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


# assume that clips have at least 17.5s, even though the majority
# of the clips have 20s, this can help avoid some clips without
# the full 20s and without breaking the pipeline during data processing
# we use a minimum of 1s from the history as the backward-looking window
HISTORY_SEC = 1.0
# Keep a conservative minimum for short clips and allow longer context for 20s clips.
FUTURE_SEC_MIN = 16.5
FUTURE_SEC = 19.0

DELTA_TIMESTAMP = 0.1  # 10 Hz, 0.1s between every two frames
AGENT_NAME_LEN = 5
START_TS = 0
END_TS = HISTORY_SEC + FUTURE_SEC

# STEP * DT = 0.2s, i.e., 5 Hz for identifying meta actions
# the higher frequency will lead to more expensive computation but
# more sensitive meta actions
STEP = 2

# number of frames labelled for meta action
SCENE_LEN = int(END_TS / DELTA_TIMESTAMP)

### visualization
PANO_VIDEO = False  # whether to use pano video
VIS_FPS = 30.0
NUM_FRAMES_VIS = int(END_TS * VIS_FPS)

META_ACTION2TEXT = {
    # New longitudinal temporal classes
    "GentleAccelerationTemporal": "GentleAcceleration",
    "StrongAccelerationTemporal": "StrongAcceleration",
    "GentleDecelerationTemporal": "GentleDeceleration",
    "StrongDecelerationTemporal": "StrongDeceleration",
    "MaintainSpeedTemporal": "MaintainSpeed",
    "StopTemporal": "Stop",
    "ReverseTemporal": "Reverse",
    # New lateral temporal classes
    "GoStraightTemporal": "GoStraight",
    "SteerLeftTemporal": "SteerLeft",
    "SteerRightTemporal": "SteerRight",
    "SharpSteerLeftTemporal": "SharpSteerLeft",
    "SharpSteerRightTemporal": "SharpSteerRight",
    "ReverseLeftTemporal": "ReverseLeft",
    "ReverseRightTemporal": "ReverseRight",
    # New lane temporal classes
    "LaneKeepTemporal": "LaneKeep",
    "LeftLaneChangeTemporal": "LeftLaneChange",
    "RightLaneChangeTemporal": "RightLaneChange",
    "SlightlyShiftLeftTemporal": "SlightlyShiftLeft",
    "SlightlyShiftRightTemporal": "SlightlyShiftRight",
    "TurnLeftTemporal": "TurnLeft",
    "TurnRightTemporal": "TurnRight",
    "FollowCurveLeftTemporal": "FollowCurveLeft",
    "FollowCurveRightTemporal": "FollowCurveRight",
    # post-smoothing
    "Stop": "Stop",
}

META_ACTION_LONGITUDINAL = [
    "GentleAcceleration",
    "StrongAcceleration",
    "GentleDeceleration",
    "StrongDeceleration",
    "MaintainSpeed",
    "Stop",
    "Reverse",
]
META_ACTION_LATERAL = [
    "GoStraight",
    "SteerLeft",
    "SteerRight",
    "SharpSteerLeft",
    "SharpSteerRight",
    "ReverseLeft",
    "ReverseRight",
]
META_ACTION_LANE = [
    "LaneKeep",
    "LeftLaneChange",
    "RightLaneChange",
    "SlightlyShiftLeft",
    "SlightlyShiftRight",
    "TurnLeft",
    "TurnRight",
    "FollowCurveLeft",
    "FollowCurveRight",
]
