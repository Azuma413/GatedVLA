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


class GentleAcceleration(Action):
    """Detect forward acceleration in the gentle range."""

    # config
    THRESHOLDS = [
        MinThreshold("average_acceleration", 0.15),
        MaxThreshold("average_acceleration", 1.4),
        MinThreshold("delta_velocity", 0.8),
    ]

    TAG = "GentleAcceleration"


class StrongAcceleration(Action):
    """Detect strong forward acceleration."""

    # config
    THRESHOLDS = [
        MinThreshold("average_acceleration", 1.4),
        MinThreshold("delta_velocity", 0.8),
    ]

    TAG = "StrongAcceleration"


class GentleDeceleration(Action):
    """Detect forward deceleration in the gentle range."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_acceleration", -0.15),
        MinThreshold("average_acceleration", -1.4),
        MaxThreshold("delta_velocity", -0.8),
    ]

    TAG = "GentleDeceleration"


class StrongDeceleration(Action):
    """Detect strong deceleration."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_acceleration", -1.4),
        MaxThreshold("delta_velocity", -0.8),
    ]

    TAG = "StrongDeceleration"


class MaintainSpeed(Action):
    """Fallback label when speed change is not significant."""

    # Applies to all remaining
    THRESHOLDS = []

    TAG = "MaintainSpeed"


class Stop(Action):
    """Detect near-stationary segments."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_speed", 0.12),
    ]

    TAG = "Stop"


class Reverse(Action):
    """Detect backward-driving segments."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_speed_along_heading", -0.12),
    ]

    TAG = "Reverse"


ORDERED_ACTIONS = [
    Reverse,
    Stop,
    StrongAcceleration,
    GentleAcceleration,
    StrongDeceleration,
    GentleDeceleration,
    MaintainSpeed,
]


def classify_chunk(chunk: pd.Series) -> Optional[str]:
    """Map one longitudinal segment row to a caption tag."""
    for action in ORDERED_ACTIONS:
        if action.is_applicable(chunk):
            return action.TAG
    return None


def classify_longitudinal(chunks: pd.DataFrame) -> pd.DataFrame:
    """Assign longitudinal caption tags to all segment rows."""
    chunks["caption"] = chunks.apply(classify_chunk, axis=1)

    return chunks
