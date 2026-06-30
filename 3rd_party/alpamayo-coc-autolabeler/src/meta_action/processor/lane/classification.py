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

from meta_action.processor.common import Action, CategoricalCheck, MaxThreshold, MinThreshold

# Keep strong nudge evidence as direct pass.
SHIFT_PRIMARY_RATE_M_PER_M = 0.095
SHIFT_PRIMARY_MIN_DISTANCE_M = 0.8
SHIFT_PRIMARY_RIGHT_LONG_DURATION_S = 5.0
SHIFT_PRIMARY_RIGHT_LONG_MAX_ABS_OFFSET_M = 0.40
SHIFT_PRIMARY_RIGHT_LONG_MIN_NEG_STEP_RATIO = 0.20
# Borderline band for slight shifts; requires extra straightness/displacement checks.
SHIFT_BORDERLINE_RATE_M_PER_M = 0.0895
SHIFT_SECONDARY_RATE_M_PER_M = 0.06
SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M = 0.02
SHIFT_STRAIGHT_HEADING_ABS_MAX_RAD = 0.10
SHIFT_MIN_NET_LATERAL_M = 0.20
SHIFT_SHORT_SEGMENT_MAX_DURATION_S = 1.2
SHIFT_SHORT_PEAK_MIN_AVG_RATE_M_PER_M = 0.03
SHIFT_SHORT_HEADING_ABS_MAX_RAD = 0.14
SHIFT_SHORT_MIN_OBSERVABLE_RATIO = 0.90
SHIFT_SHORT_MIN_DISTANCE_M = 0.8
SHIFT_WEAK_LC_SHIFT_MAX_DURATION_S = 0.7
SHIFT_WEAK_LC_SHIFT_MAX_ABS_OFFSET_M = 0.35
SHIFT_WEAK_LC_SHIFT_MIN_OBSERVABLE_RATIO = 0.90
SHIFT_PULSE_SHIFT_MIN_DURATION_S = 1.0
SHIFT_PULSE_SHIFT_MIN_OBSERVABLE_RATIO = 0.90
SHIFT_PULSE_SHIFT_HEADING_ABS_MAX_RAD = 0.34
SHIFT_PULSE_SHIFT_RIGHT_MIN_HEADING_ABS_RAD = 0.10
SHIFT_PULSE_SHIFT_MIN_GAIN_M = 0.50
SHIFT_PULSE_SHIFT_DOMINANCE_RATIO = 0.80
SHIFT_CURVED_LEFT_MIN_AVG_RATE_M_PER_M = 0.055
SHIFT_CURVED_LEFT_MAX_AVG_RATE_M_PER_M = 0.0895
SHIFT_CURVED_LEFT_MIN_DURATION_S = 1.0
SHIFT_CURVED_LEFT_MAX_DURATION_S = 3.0
SHIFT_CURVED_LEFT_MIN_DISTANCE_M = 2.0
SHIFT_CURVED_LEFT_MIN_HEADING_ABS_RAD = 0.12
SHIFT_CURVED_LEFT_MAX_HEADING_ABS_RAD = 0.35
SHIFT_CURVED_LEFT_MAX_HEADING_RATE_ABS_RAD_PER_M = 0.16
SHIFT_CURVED_LEFT_MIN_PULSE_GAIN_M = 0.30
SHIFT_CURVED_LEFT_PULSE_DOMINANCE_RATIO = 1.50
SHIFT_CURVED_LEFT_MAX_ABS_OFFSET_M = 1.35
SHIFT_MID_CURVE_LEFT_MIN_AVG_RATE_M_PER_M = 0.05
SHIFT_MID_CURVE_LEFT_MAX_AVG_RATE_M_PER_M = 0.075
SHIFT_MID_CURVE_LEFT_MIN_DURATION_S = 2.0
SHIFT_MID_CURVE_LEFT_MAX_DURATION_S = 3.2
SHIFT_MID_CURVE_LEFT_MIN_DISTANCE_M = 3.0
SHIFT_MID_CURVE_LEFT_MAX_DISTANCE_M = 6.0
SHIFT_MID_CURVE_LEFT_MIN_HEADING_ABS_RAD = 0.12
SHIFT_MID_CURVE_LEFT_MAX_HEADING_ABS_RAD = 0.22
SHIFT_MID_CURVE_LEFT_MAX_HEADING_RATE_ABS_RAD_PER_M = 0.07
SHIFT_MID_CURVE_LEFT_MIN_POS_STEP_RATIO = 0.08
SHIFT_MID_CURVE_LEFT_MIN_RISE_M = 0.30
SHIFT_MID_CURVE_LEFT_MAX_ABS_OFFSET_M = 0.45
LANE_CHANGE_MIN_ABS_LATERAL_RATE_M_PER_M = 0.02
LANE_CHANGE_MIN_NET_LATERAL_M = 0.50
LANE_CHANGE_MIN_MAX_ABS_OFFSET_M = 1.00

FOLLOW_CURVE_HEADING_RATE_MIN_RAD_PER_M = 0.02
FOLLOW_CURVE_HEADING_MIN_RAD = 0.10
FOLLOW_CURVE_MAX_ABS_LATERAL_RATE_M_PER_M = 0.06
FOLLOW_CURVE_MAX_ABS_LATERAL_OFFSET_M = 1.2
TURN_MAX_ABS_LATERAL_RATE_M_PER_M = 0.06
TURN_MAX_OPPOSING_LATERAL_RATE_M_PER_M = 0.02


def _safe_float(chunk: pd.Series, key: str, default: float = 0.0) -> float:
    """Safely read one float-like feature from a segment row."""
    value = chunk.get(key, default)
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_keep_lane(chunk: pd.Series) -> bool:
    """Follow-curve is only valid for explicit keep-lane decisions."""
    lane_decision = chunk.get("lane_decision", "keep_lane")
    if pd.isna(lane_decision):
        return True
    return str(lane_decision) == "keep_lane"


def _follow_curve_lateral_ok(chunk: pd.Series) -> bool:
    """Check lateral-excursion bounds required for follow-curve labels."""
    max_abs_offset_raw = chunk.get("max_abs_lane_lateral_offset")
    lat_rate_raw = chunk.get("average_lane_lateral_change_rate_m_per_m")
    if pd.isna(max_abs_offset_raw) or pd.isna(lat_rate_raw):
        # If lane geometry is unavailable across a segment, do not claim follow-curve.
        return False

    lat_rate = abs(_safe_float(chunk, "average_lane_lateral_change_rate_m_per_m"))
    if lat_rate > FOLLOW_CURVE_MAX_ABS_LATERAL_RATE_M_PER_M:
        return False

    # Reject lane-departure-like segments: avg rate can be small even with large excursion.
    max_abs_offset = abs(_safe_float(chunk, "max_abs_lane_lateral_offset"))
    return max_abs_offset <= FOLLOW_CURVE_MAX_ABS_LATERAL_OFFSET_M


def _lane_change_motion_supported(chunk: pd.Series) -> bool:
    """Reject map-relation-only lane-change artifacts with negligible lateral motion."""
    lat_rate = abs(_safe_float(chunk, "average_lane_lateral_change_rate_m_per_m"))
    dist = max(_safe_float(chunk, "delta_distance_driven_m"), 0.0)
    max_abs_offset = abs(_safe_float(chunk, "max_abs_lane_lateral_offset"))
    if max_abs_offset >= LANE_CHANGE_MIN_MAX_ABS_OFFSET_M:
        return True
    return (
        lat_rate >= LANE_CHANGE_MIN_ABS_LATERAL_RATE_M_PER_M
        and (lat_rate * dist) >= LANE_CHANGE_MIN_NET_LATERAL_M
    )


def _turn_intersection_like(chunk: pd.Series, turn_sign: int) -> bool:
    """Keep turn labels focused on intersection-like path turns instead of lane nudges.
    turn_sign: +1 for left, -1 for right.
    """
    lat_rate = _safe_float(chunk, "average_lane_lateral_change_rate_m_per_m")
    if abs(lat_rate) > TURN_MAX_ABS_LATERAL_RATE_M_PER_M:
        return False
    if turn_sign > 0 and lat_rate < -TURN_MAX_OPPOSING_LATERAL_RATE_M_PER_M:
        return False
    if turn_sign < 0 and lat_rate > TURN_MAX_OPPOSING_LATERAL_RATE_M_PER_M:
        return False
    return True


def _short_peak_shift_supported(chunk: pd.Series, direction: str) -> bool:
    """Conservative fallback for short keep-lane nudge bursts where avg lateral-rate
    is diluted but peak per-step lateral-rate is strong.
    """
    if not _is_keep_lane(chunk):
        return False
    if bool(chunk.get("lane_centerline_id_changed", False)):
        return False
    if _safe_float(chunk, "lane_observable_ratio") < SHIFT_SHORT_MIN_OBSERVABLE_RATIO:
        return False
    if _safe_float(chunk, "duration") > SHIFT_SHORT_SEGMENT_MAX_DURATION_S:
        return False
    if _safe_float(chunk, "delta_distance_driven_m") < SHIFT_SHORT_MIN_DISTANCE_M:
        return False

    avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
    heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
    if avg_heading_rate >= SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M:
        return False
    if heading_sum >= SHIFT_SHORT_HEADING_ABS_MAX_RAD:
        return False

    lat_rate = _safe_float(chunk, "average_lane_lateral_change_rate_m_per_m")
    if direction == "left":
        peak_rate = _safe_float(chunk, "max_lane_lateral_change_rate_m_per_m")
        return (
            peak_rate >= SHIFT_PRIMARY_RATE_M_PER_M
            and lat_rate >= SHIFT_SHORT_PEAK_MIN_AVG_RATE_M_PER_M
        )

    peak_rate = _safe_float(chunk, "min_lane_lateral_change_rate_m_per_m")
    return (
        peak_rate <= -SHIFT_PRIMARY_RATE_M_PER_M
        and lat_rate <= -SHIFT_SHORT_PEAK_MIN_AVG_RATE_M_PER_M
    )


def _weak_lane_change_shift_supported(chunk: pd.Series, direction: str) -> bool:
    """Recover slight-shift segments that are tagged as lane-id changes by map jitter
    but do not satisfy lane-change motion support.
    """
    if _lane_change_motion_supported(chunk):
        return False
    if _safe_float(chunk, "duration") > SHIFT_WEAK_LC_SHIFT_MAX_DURATION_S:
        return False
    if _safe_float(chunk, "lane_observable_ratio") < SHIFT_WEAK_LC_SHIFT_MIN_OBSERVABLE_RATIO:
        return False
    if (
        abs(_safe_float(chunk, "max_abs_lane_lateral_offset"))
        > SHIFT_WEAK_LC_SHIFT_MAX_ABS_OFFSET_M
    ):
        return False

    avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
    heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
    if avg_heading_rate >= SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M:
        return False
    if heading_sum >= SHIFT_STRAIGHT_HEADING_ABS_MAX_RAD:
        return False

    lane_decision = str(chunk.get("lane_decision", "keep_lane"))
    if direction == "left":
        return lane_decision in ("keep_lane", "left_lane_change")
    return lane_decision in ("keep_lane", "right_lane_change")


def _pulse_jitter_shift_supported(chunk: pd.Series, direction: str) -> bool:
    """Keep-lane fallback for map-jitter segments using robust offset pulse trend
    instead of average lateral-rate (which can cancel out).
    """
    if not _is_keep_lane(chunk):
        return False
    if not bool(chunk.get("lane_centerline_id_changed", False)):
        return False
    if _safe_float(chunk, "lane_observable_ratio") < SHIFT_PULSE_SHIFT_MIN_OBSERVABLE_RATIO:
        return False

    dur = _safe_float(chunk, "duration")
    if dur < SHIFT_PULSE_SHIFT_MIN_DURATION_S:
        return False

    avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
    heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
    if avg_heading_rate >= SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M:
        return False
    if heading_sum >= SHIFT_PULSE_SHIFT_HEADING_ABS_MAX_RAD:
        return False

    max_rise = _safe_float(chunk, "offset_pulse_max_rise_m")
    max_fall = _safe_float(chunk, "offset_pulse_max_fall_m")
    first_dir = str(chunk.get("offset_pulse_first_dir", "none"))
    if direction == "left":
        return (
            first_dir == "left"
            and max_rise >= SHIFT_PULSE_SHIFT_MIN_GAIN_M
            and max_rise >= SHIFT_PULSE_SHIFT_DOMINANCE_RATIO * max_fall
        )
    if heading_sum < SHIFT_PULSE_SHIFT_RIGHT_MIN_HEADING_ABS_RAD:
        return False
    return (
        first_dir == "right"
        and max_fall >= SHIFT_PULSE_SHIFT_MIN_GAIN_M
        and max_fall >= SHIFT_PULSE_SHIFT_DOMINANCE_RATIO * max_rise
    )


def _curved_left_shift_supported(chunk: pd.Series) -> bool:
    """Recover short left nudges on curved road where heading-based straightness
    guards are intentionally too strict for the main left-shift path.
    """
    if not _is_keep_lane(chunk):
        return False
    if not bool(chunk.get("lane_centerline_id_changed", False)):
        return False
    if _safe_float(chunk, "lane_observable_ratio") < SHIFT_SHORT_MIN_OBSERVABLE_RATIO:
        return False

    lat_rate = _safe_float(chunk, "average_lane_lateral_change_rate_m_per_m")
    if (
        lat_rate < SHIFT_CURVED_LEFT_MIN_AVG_RATE_M_PER_M
        or lat_rate >= SHIFT_CURVED_LEFT_MAX_AVG_RATE_M_PER_M
    ):
        return False

    duration = _safe_float(chunk, "duration")
    if duration < SHIFT_CURVED_LEFT_MIN_DURATION_S or duration > SHIFT_CURVED_LEFT_MAX_DURATION_S:
        return False
    if _safe_float(chunk, "delta_distance_driven_m") < SHIFT_CURVED_LEFT_MIN_DISTANCE_M:
        return False

    avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
    heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
    if avg_heading_rate > SHIFT_CURVED_LEFT_MAX_HEADING_RATE_ABS_RAD_PER_M:
        return False
    if (
        heading_sum < SHIFT_CURVED_LEFT_MIN_HEADING_ABS_RAD
        or heading_sum > SHIFT_CURVED_LEFT_MAX_HEADING_ABS_RAD
    ):
        return False

    if abs(_safe_float(chunk, "max_abs_lane_lateral_offset")) > SHIFT_CURVED_LEFT_MAX_ABS_OFFSET_M:
        return False

    max_rise = _safe_float(chunk, "offset_pulse_max_rise_m")
    max_fall = _safe_float(chunk, "offset_pulse_max_fall_m")
    first_dir = str(chunk.get("offset_pulse_first_dir", "none"))
    return (
        first_dir == "left"
        and max_rise >= SHIFT_CURVED_LEFT_MIN_PULSE_GAIN_M
        and max_rise >= SHIFT_CURVED_LEFT_PULSE_DOMINANCE_RATIO * max_fall
    )


def _mid_curve_left_shift_supported(chunk: pd.Series) -> bool:
    """Conservative fallback for moderate-curvature left nudges where pulse
    direction can be mixed, but the net evidence remains left-biased.
    """
    if not _is_keep_lane(chunk):
        return False
    if not bool(chunk.get("lane_centerline_id_changed", False)):
        return False
    if _safe_float(chunk, "lane_observable_ratio") < SHIFT_SHORT_MIN_OBSERVABLE_RATIO:
        return False

    lat_rate = _safe_float(chunk, "average_lane_lateral_change_rate_m_per_m")
    if (
        lat_rate < SHIFT_MID_CURVE_LEFT_MIN_AVG_RATE_M_PER_M
        or lat_rate >= SHIFT_MID_CURVE_LEFT_MAX_AVG_RATE_M_PER_M
    ):
        return False

    duration = _safe_float(chunk, "duration")
    if (
        duration < SHIFT_MID_CURVE_LEFT_MIN_DURATION_S
        or duration > SHIFT_MID_CURVE_LEFT_MAX_DURATION_S
    ):
        return False

    dist = _safe_float(chunk, "delta_distance_driven_m")
    if dist < SHIFT_MID_CURVE_LEFT_MIN_DISTANCE_M or dist > SHIFT_MID_CURVE_LEFT_MAX_DISTANCE_M:
        return False

    avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
    heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
    if avg_heading_rate > SHIFT_MID_CURVE_LEFT_MAX_HEADING_RATE_ABS_RAD_PER_M:
        return False
    if (
        heading_sum < SHIFT_MID_CURVE_LEFT_MIN_HEADING_ABS_RAD
        or heading_sum > SHIFT_MID_CURVE_LEFT_MAX_HEADING_ABS_RAD
    ):
        return False

    if _safe_float(chunk, "ego_positive_step_ratio") < SHIFT_MID_CURVE_LEFT_MIN_POS_STEP_RATIO:
        return False
    if _safe_float(chunk, "offset_pulse_max_rise_m") < SHIFT_MID_CURVE_LEFT_MIN_RISE_M:
        return False
    return (
        abs(_safe_float(chunk, "max_abs_lane_lateral_offset"))
        <= SHIFT_MID_CURVE_LEFT_MAX_ABS_OFFSET_M
    )


def _right_primary_shift_consistent(chunk: pd.Series) -> bool:
    """Suppress pathological long right-shift artifacts where average lateral-rate
    is high but directional ego lateral evidence is weak and lane excursion is tiny.
    """
    if not bool(chunk.get("lane_centerline_id_changed", False)):
        return True

    if _safe_float(chunk, "duration") < SHIFT_PRIMARY_RIGHT_LONG_DURATION_S:
        return True
    if (
        abs(_safe_float(chunk, "max_abs_lane_lateral_offset"))
        > SHIFT_PRIMARY_RIGHT_LONG_MAX_ABS_OFFSET_M
    ):
        return True
    return (
        _safe_float(chunk, "ego_negative_step_ratio") >= SHIFT_PRIMARY_RIGHT_LONG_MIN_NEG_STEP_RATIO
    )


class LeftLaneChange(Action):
    """Detect confident left lane changes from lane-decision evidence."""

    # config
    TAG = "LeftLaneChange"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when the segment is a supported left lane change."""
        if chunk.get("lane_decision") != "left_lane_change":
            return False
        return _lane_change_motion_supported(chunk)


class RightLaneChange(Action):
    """Detect confident right lane changes from lane-decision evidence."""

    # config
    TAG = "RightLaneChange"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when the segment is a supported right lane change."""
        if chunk.get("lane_decision") != "right_lane_change":
            return False
        return _lane_change_motion_supported(chunk)


class SlightlyShiftLeft(Action):
    """Detect subtle left shifts that do not satisfy full lane-change criteria."""

    THRESHOLDS = []

    TAG = "SlightlyShiftLeft"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when left-shift heuristics are satisfied."""
        lat_rate = _safe_float(chunk, "average_lane_lateral_change_rate_m_per_m")
        if lat_rate >= SHIFT_PRIMARY_RATE_M_PER_M:
            return _safe_float(chunk, "delta_distance_driven_m") >= SHIFT_PRIMARY_MIN_DISTANCE_M
        if lat_rate >= SHIFT_BORDERLINE_RATE_M_PER_M:
            avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
            heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
            if avg_heading_rate >= SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M:
                return False
            if heading_sum >= SHIFT_STRAIGHT_HEADING_ABS_MAX_RAD:
                return False

            dist = _safe_float(chunk, "delta_distance_driven_m")
            est_net_lateral = abs(lat_rate) * max(dist, 0.0)
            return est_net_lateral >= SHIFT_MIN_NET_LATERAL_M

        if lat_rate < SHIFT_SECONDARY_RATE_M_PER_M:
            if _short_peak_shift_supported(chunk, direction="left"):
                return True
            if _pulse_jitter_shift_supported(chunk, direction="left"):
                return True
            return _mid_curve_left_shift_supported(chunk)
        if bool(chunk.get("lane_centerline_id_changed", False)):
            if _weak_lane_change_shift_supported(chunk, direction="left"):
                return True
            if _curved_left_shift_supported(chunk):
                return True
            return _mid_curve_left_shift_supported(chunk)

        avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
        heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
        if avg_heading_rate >= SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M:
            return False
        if heading_sum >= SHIFT_STRAIGHT_HEADING_ABS_MAX_RAD:
            return False

        dist = _safe_float(chunk, "delta_distance_driven_m")
        est_net_lateral = abs(lat_rate) * max(dist, 0.0)
        return est_net_lateral >= SHIFT_MIN_NET_LATERAL_M


class SlightlyShiftRight(Action):
    """Detect subtle right shifts that do not satisfy full lane-change criteria."""

    THRESHOLDS = []

    TAG = "SlightlyShiftRight"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when right-shift heuristics are satisfied."""
        lat_rate = _safe_float(chunk, "average_lane_lateral_change_rate_m_per_m")
        if lat_rate <= -SHIFT_PRIMARY_RATE_M_PER_M:
            return _safe_float(
                chunk, "delta_distance_driven_m"
            ) >= SHIFT_PRIMARY_MIN_DISTANCE_M and _right_primary_shift_consistent(chunk)
        if lat_rate <= -SHIFT_BORDERLINE_RATE_M_PER_M:
            avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
            heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
            if avg_heading_rate >= SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M:
                return False
            if heading_sum >= SHIFT_STRAIGHT_HEADING_ABS_MAX_RAD:
                return False

            dist = _safe_float(chunk, "delta_distance_driven_m")
            est_net_lateral = abs(lat_rate) * max(dist, 0.0)
            return est_net_lateral >= SHIFT_MIN_NET_LATERAL_M

        if lat_rate > -SHIFT_SECONDARY_RATE_M_PER_M:
            if _short_peak_shift_supported(chunk, direction="right"):
                return True
            return _pulse_jitter_shift_supported(chunk, direction="right")
        if bool(chunk.get("lane_centerline_id_changed", False)):
            return _weak_lane_change_shift_supported(chunk, direction="right")

        avg_heading_rate = abs(_safe_float(chunk, "average_heading_change_rate_rad_m"))
        heading_sum = abs(_safe_float(chunk, "heading_change_rad"))
        if avg_heading_rate >= SHIFT_STRAIGHT_HEADING_RATE_ABS_MAX_RAD_PER_M:
            return False
        if heading_sum >= SHIFT_STRAIGHT_HEADING_ABS_MAX_RAD:
            return False

        dist = _safe_float(chunk, "delta_distance_driven_m")
        est_net_lateral = abs(lat_rate) * max(dist, 0.0)
        return est_net_lateral >= SHIFT_MIN_NET_LATERAL_M


class TurnLeft(Action):
    """Detect left turns using heading-change and lane-transition evidence."""

    # config
    THRESHOLDS = [
        MinThreshold("average_heading_change_rate_rad_m", 0.04),
        MinThreshold("heading_change_rad", 0.2),
        CategoricalCheck("lane_centerline_id_changed", True),
    ]

    TAG = "TurnLeft"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when left-turn thresholds and context checks pass."""
        for threshold in cls.THRESHOLDS:
            if not threshold.passes(chunk[threshold.property_name]):
                return False
        return _turn_intersection_like(chunk, turn_sign=1)


class TurnRight(Action):
    """Detect right turns using heading-change and lane-transition evidence."""

    # config
    THRESHOLDS = [
        MaxThreshold("average_heading_change_rate_rad_m", -0.04),
        MaxThreshold("heading_change_rad", -0.2),
        CategoricalCheck("lane_centerline_id_changed", True),
    ]

    TAG = "TurnRight"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when right-turn thresholds and context checks pass."""
        for threshold in cls.THRESHOLDS:
            if not threshold.passes(chunk[threshold.property_name]):
                return False
        return _turn_intersection_like(chunk, turn_sign=-1)


class FollowCurveLeft(Action):
    """Detect leftward road curvature while staying in lane."""

    THRESHOLDS = []

    TAG = "FollowCurveLeft"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when left follow-curve conditions are satisfied."""
        if not _is_keep_lane(chunk):
            return False
        if bool(chunk.get("lane_centerline_id_changed", False)):
            return False
        if (
            _safe_float(chunk, "average_heading_change_rate_rad_m")
            < FOLLOW_CURVE_HEADING_RATE_MIN_RAD_PER_M
        ):
            return False
        if _safe_float(chunk, "heading_change_rad") < FOLLOW_CURVE_HEADING_MIN_RAD:
            return False
        return _follow_curve_lateral_ok(chunk)


class FollowCurveRight(Action):
    """Detect rightward road curvature while staying in lane."""

    THRESHOLDS = []

    TAG = "FollowCurveRight"

    @classmethod
    def is_applicable(cls, chunk: pd.Series) -> bool:
        """Return True when right follow-curve conditions are satisfied."""
        if not _is_keep_lane(chunk):
            return False
        if bool(chunk.get("lane_centerline_id_changed", False)):
            return False
        if (
            _safe_float(chunk, "average_heading_change_rate_rad_m")
            > -FOLLOW_CURVE_HEADING_RATE_MIN_RAD_PER_M
        ):
            return False
        if _safe_float(chunk, "heading_change_rad") > -FOLLOW_CURVE_HEADING_MIN_RAD:
            return False
        return _follow_curve_lateral_ok(chunk)


class LaneKeep(Action):
    """Fallback lane label for segments without stronger lane maneuvers."""

    THRESHOLDS = []

    TAG = "LaneKeep"


ORDERED_ACTIONS = [
    LeftLaneChange,
    RightLaneChange,
    TurnLeft,
    TurnRight,
    SlightlyShiftLeft,
    SlightlyShiftRight,
    # Intentionally disabled: VLM captures curvy-road behavior better than this
    # heuristic, so we do not emit FollowCurve* meta actions in current outputs.
    # FollowCurveLeft,
    # FollowCurveRight,
    LaneKeep,
]


def classify_chunk(chunk: pd.Series) -> Optional[str]:
    """Map one lane segment row to a caption tag."""
    for action in ORDERED_ACTIONS:
        if action.is_applicable(chunk):
            return action.TAG
    return None


def classify_lane(chunks: pd.DataFrame) -> pd.DataFrame:
    """Assign lane caption tags to all segment rows."""
    chunks["caption"] = chunks.apply(classify_chunk, axis=1)
    return chunks
