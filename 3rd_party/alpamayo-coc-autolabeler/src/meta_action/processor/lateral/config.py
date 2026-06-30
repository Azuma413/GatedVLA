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

LATERAL_SEGMENTATION_THRESHOLDS = [
    Threshold("agent_velocity_along_heading", -0.1),  # Reverse
    Threshold("agent_heading_change_rate", 0.001),  # Steer Left
    Threshold("agent_heading_change_rate", -0.001),  # Steer Right
    Threshold("agent_heading_change_rate", 0.05),  # Sharp Left
    Threshold("agent_heading_change_rate", -0.05),  # Sharp Right
]


# Caption-specific minimum segment lengths (in ticks) used for cleaning/merging.
# Values can be tuned; unspecified captions default to LATERAL_MIN_LEN_TICKS.
DEFAULT_LATERAL_MIN_LEN_TICKS = 10
LATERAL_MIN_LEN_TICKS_BY_CAPTION = {
    "GoStraight": DEFAULT_LATERAL_MIN_LEN_TICKS,
    "SteerLeft": DEFAULT_LATERAL_MIN_LEN_TICKS,
    "SteerRight": DEFAULT_LATERAL_MIN_LEN_TICKS,
    "SharpSteerLeft": DEFAULT_LATERAL_MIN_LEN_TICKS,
    "SharpSteerRight": DEFAULT_LATERAL_MIN_LEN_TICKS,
    "ReverseLeft": DEFAULT_LATERAL_MIN_LEN_TICKS,
    "ReverseRight": DEFAULT_LATERAL_MIN_LEN_TICKS,
}
