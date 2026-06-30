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

from typing import Any

import pandas as pd


class Threshold:
    """Base threshold predicate configured on one segment property."""

    def __init__(self, property_name: str, threshold: Any) -> None:
        self.property_name = property_name
        self.threshold = threshold


class MinThreshold(Threshold):
    """Numeric lower-bound threshold check."""

    def __init__(self, property_name: str, threshold: float) -> None:
        super().__init__(property_name, threshold)

    def passes(self, value: float) -> bool:
        """Return True when `value` is greater than or equal to the threshold."""
        return value >= self.threshold


class MaxThreshold(Threshold):
    """Numeric upper-bound threshold check."""

    def __init__(self, property_name: str, threshold: float) -> None:
        super().__init__(property_name, threshold)

    def passes(self, value: float) -> bool:
        """Return True when `value` is less than or equal to the threshold."""
        return value <= self.threshold


class CategoricalCheck(Threshold):
    """Exact-match threshold check for categorical values."""

    def __init__(self, property_name: str, threshold: Any) -> None:
        super().__init__(property_name, threshold)

    def passes(self, value: Any) -> bool:
        """Return True when `value` equals the configured categorical threshold."""
        return value == self.threshold


class Action:
    """Base action classifier using a list of threshold predicates."""

    THRESHOLDS = []

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return whether all configured thresholds pass for one segment row."""
        for threshold in cls.THRESHOLDS:
            if not threshold.passes(chunk[threshold.property_name]):
                return False
        return True
