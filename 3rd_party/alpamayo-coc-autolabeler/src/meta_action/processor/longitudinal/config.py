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

LONGITUDINAL_SEGMENTATION_THRESHOLDS = [
    Threshold("agent_speed_along_heading", 0.3),
    Threshold("agent_speed_along_heading", -0.3),
    Threshold("agent_acceleration", 0.1),  # low acceleration
    Threshold("agent_acceleration", -0.1),  # low  deceleration
    Threshold("agent_acceleration", 1.25),  # high acceleration
    Threshold("agent_acceleration", -1.25),  # high deceleration
]


# Caption-specific minimum segment lengths (in ticks) used for cleaning/merging.
# Values can be tuned; unspecified captions default to LONGITUDINAL_MIN_LEN_TICKS.
DEFAULT_LONGITUDINAL_MIN_LEN_TICKS = 8
DEFAULT_MIN_LEN_TICKS_MAINTAIN_SPEED = 12
LONGITUDINAL_MIN_LEN_TICKS_BY_CAPTION = {
    # Longitudinal
    "GentleAcceleration": DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
    "StrongAcceleration": DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
    "GentleDeceleration": DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
    "StrongDeceleration": DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
    # MaintainSpeed typically needs longer to be meaningful; keep stricter default
    "MaintainSpeed": DEFAULT_MIN_LEN_TICKS_MAINTAIN_SPEED,
    "Stop": DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
    "Reverse": DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
}
