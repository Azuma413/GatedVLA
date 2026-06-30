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

from typing import Optional

import pandas as pd

from meta_action.processor.common import Action, MaxThreshold, MinThreshold


class ReverseRight(Action):
    """Detect reversing motion with rightward curvature."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_speed_along_heading", -0.1),
        MaxThreshold("average_turn_rate_rad_per_m", -0.002),
        MaxThreshold("total_heading_change_rad", -0.2),
    ]

    TAG = "ReverseRight"


class ReverseLeft(Action):
    """Detect reversing motion with leftward curvature."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_speed_along_heading", -0.1),
        MinThreshold("average_turn_rate_rad_per_m", 0.002),
        MinThreshold("total_heading_change_rad", 0.2),
    ]

    TAG = "ReverseLeft"


class SteerRight(Action):
    """Detect mild right steering while moving forward."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_turn_rate_rad_per_m", -0.0015),
        MaxThreshold("total_heading_change_rad", -0.015),
    ]

    TAG = "SteerRight"


class SteerLeft(Action):
    """Detect mild left steering while moving forward."""

    # config
    THRESHOLDS = [
        MinThreshold("average_turn_rate_rad_per_m", 0.0015),
        MinThreshold("total_heading_change_rad", 0.015),
    ]

    TAG = "SteerLeft"


class SharpSteerRight(Action):
    """Detect strong right steering maneuvers."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_turn_rate_rad_per_m", -0.05),
        MaxThreshold("total_heading_change_rad", -0.2),
    ]

    TAG = "SharpSteerRight"


class SharpSteerLeft(Action):
    """Detect strong left steering maneuvers."""

    # config
    THRESHOLDS = [
        MinThreshold("average_turn_rate_rad_per_m", 0.05),
        MinThreshold("total_heading_change_rad", 0.2),
    ]

    TAG = "SharpSteerLeft"


class GoStraight(Action):
    """Fallback label when no lateral turn/steer action matches."""

    # Applies to all remaining
    THRESHOLDS = []

    TAG = "GoStraight"


ORDERED_ACTIONS = [
    ReverseRight,
    ReverseLeft,
    SharpSteerRight,
    SharpSteerLeft,
    SteerRight,
    SteerLeft,
    GoStraight,
]


def classify_chunk(chunk: pd.Series) -> Optional[str]:
    """Map one lateral segment row to a caption tag."""
    for action in ORDERED_ACTIONS:
        if action.is_applicable(chunk):
            return action.TAG
    return None


def classify_lateral(chunks: pd.DataFrame) -> pd.DataFrame:
    """Assign lateral caption tags to all segment rows."""
    chunks["caption"] = chunks.apply(classify_chunk, axis=1)

    return chunks
