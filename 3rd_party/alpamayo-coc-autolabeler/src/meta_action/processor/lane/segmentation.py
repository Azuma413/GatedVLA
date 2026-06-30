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

import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from meta_action.processor.common import Threshold
from meta_action.processor.lane.config import (
    LANE_CHANGE_MIN_RUN_TICKS,
    LANE_DECISION_LOOKAHEAD_TICKS,
    LANE_DECISION_WINDOW_STEP_TICKS,
    LANE_SEGMENTATION_CATEGORICAL,
    LANE_SEGMENTATION_THRESHOLDS,
)
from meta_action.utils.constant import DELTA_TIMESTAMP

AMBIGUOUS_DIRECTION_MIN_NET_LATERAL_OFFSET_M = 0.35
ZERO_EVIDENCE_SHORT_RUN_MAX_TICKS = 24
ZERO_EVIDENCE_LONG_RUN_SUPPRESS_TICKS = 60
ZERO_EVIDENCE_SHORT_RUN_MIN_NET_LATERAL_OFFSET_M = 0.15


def _safe_nanmean(values: np.ndarray) -> float:
    """Return nanmean without warnings on empty/all-NaN slices."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _safe_nanmax_abs(values: np.ndarray) -> float:
    """Return max(abs(values)) without warnings on empty/all-NaN slices."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.abs(valid).max())


def _safe_nanmax(values: np.ndarray, default: float = 0.0) -> float:
    """Return nanmax with default on empty/all-NaN slices."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float(default)
    return float(valid.max())


def _safe_nanmin(values: np.ndarray, default: float = 0.0) -> float:
    """Return nanmin with default on empty/all-NaN slices."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float(default)
    return float(valid.min())


def _offset_pulse_metrics(offsets: np.ndarray) -> Tuple[float, float, str]:
    """Compute robust directional pulse stats from segment offset trace."""
    valid = offsets[np.isfinite(offsets)]
    if valid.size < 2:
        return 0.0, 0.0, "none"

    diffs = np.diff(valid)
    max_rise = 0.0
    max_fall = 0.0
    rise = 0.0
    fall = 0.0
    rise_ticks = 0
    fall_ticks = 0
    first_left = None
    first_right = None
    for idx, d in enumerate(diffs, start=1):
        if d > 0:
            rise += float(d)
            rise_ticks += 1
            fall = 0.0
            fall_ticks = 0
        elif d < 0:
            fall += float(-d)
            fall_ticks += 1
            rise = 0.0
            rise_ticks = 0
        else:
            rise = 0.0
            fall = 0.0
            rise_ticks = 0
            fall_ticks = 0

        max_rise = max(max_rise, rise)
        max_fall = max(max_fall, fall)
        if first_left is None and rise_ticks >= 4 and rise >= 0.12:
            first_left = idx
        if first_right is None and fall_ticks >= 4 and fall >= 0.12:
            first_right = idx

    if first_left is None and first_right is None:
        first_dir = "none"
    elif first_right is None:
        first_dir = "left"
    elif first_left is None:
        first_dir = "right"
    else:
        first_dir = "left" if first_left <= first_right else "right"

    return float(max_rise), float(max_fall), first_dir


def _signed_lateral_offset_delta(
    offsets: List[Optional[float]], start_idx: int, end_idx: int
) -> Optional[float]:
    """Return signed lateral-offset delta using nearest finite endpoints."""
    if not offsets:
        return None
    n = len(offsets)
    s = max(0, min(int(start_idx), n - 1))
    e = max(0, min(int(end_idx), n - 1))
    if e < s:
        s, e = e, s

    start_v = None
    for i in range(s, e + 1):
        v = offsets[i]
        if v is not None and np.isfinite(v):
            start_v = float(v)
            break

    end_v = None
    for i in range(e, s - 1, -1):
        v = offsets[i]
        if v is not None and np.isfinite(v):
            end_v = float(v)
            break

    if start_v is None or end_v is None:
        return None
    return end_v - start_v


def _offset_delta_to_left_positive_signal(
    delta_off: Optional[float],
) -> Optional[float]:
    """Convert lane-offset delta to a left-positive lateral signal.

    `lane_lateral_offset_m` uses the local lane tangent for sign. Under this map
    convention, moving toward the left-adjacent lane tends to *decrease* the signed
    offset, so we negate `delta_off` to align with ego-frame displacement where
    positive means left.
    """
    if delta_off is None or not np.isfinite(delta_off):
        return None
    return float(-delta_off)


def _signed_ego_lateral_displacement(
    world_xyzh: np.ndarray, start_idx: int, end_idx: int
) -> Optional[float]:
    """Signed lateral displacement in ego frame at start_idx (left positive)."""
    if world_xyzh is None or len(world_xyzh) == 0:
        return None
    n = int(world_xyzh.shape[0])
    s = max(0, min(int(start_idx), n - 1))
    e = max(0, min(int(end_idx), n - 1))
    if e <= s:
        return 0.0

    p0 = world_xyzh[s, :2]
    p1 = world_xyzh[e, :2]
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        return None

    heading0 = float(world_xyzh[s, 3])
    if not np.isfinite(heading0):
        return None

    left_vec = np.array([-np.sin(heading0), np.cos(heading0)], dtype=float)
    delta_xy = p1 - p0
    return float(np.dot(delta_xy, left_vec))


def _old_style_relation_decision(start_relation: dict, end_lane_id: Optional[str]) -> str:
    """Transition-style lane decision inspired by the old implementation:
    compare a future lane id against lane sets from the current timestep.
    """
    if not start_relation or end_lane_id is None:
        return "keep_lane"

    cur = start_relation.get("current_lanes", set())
    nxt = start_relation.get("next_lanes", set())
    left = start_relation.get("left_lanes", set())
    right = start_relation.get("right_lanes", set())

    if end_lane_id in cur or end_lane_id in nxt:
        return "keep_lane"
    if end_lane_id in left:
        return "left_lane_change"
    if end_lane_id in right:
        return "right_lane_change"
    return "keep_lane"


def _decision_from_end_lane_set(
    start_relation: dict,
    end_lane_ids: set,
    instantaneous_decision: str = "keep_lane",
) -> str:
    """Old-style transition decision but using the full candidate end-lane set.
    This improves recall versus picking a single matched lane id.
    """
    if not start_relation or not end_lane_ids:
        return instantaneous_decision

    left_hits = 0
    right_hits = 0
    keep_hits = 0
    for end_lane_id in end_lane_ids:
        d = _old_style_relation_decision(start_relation, end_lane_id)
        if d == "left_lane_change":
            left_hits += 1
        elif d == "right_lane_change":
            right_hits += 1
        else:
            keep_hits += 1

    if left_hits > 0 and right_hits == 0:
        return "left_lane_change"
    if right_hits > 0 and left_hits == 0:
        return "right_lane_change"
    if left_hits > 0 and right_hits > 0:
        if instantaneous_decision in ("left_lane_change", "right_lane_change"):
            return instantaneous_decision
        return "left_lane_change" if left_hits >= right_hits else "right_lane_change"
    if keep_hits > 0:
        return "keep_lane"
    return instantaneous_decision


def _drop_short_lane_change_runs(
    decisions: List[str], min_run_ticks: int = LANE_CHANGE_MIN_RUN_TICKS
) -> List[str]:
    """Remove short left/right lane-change runs to reduce per-frame flicker."""
    if not decisions:
        return decisions

    out = list(decisions)
    n = len(out)
    i = 0
    while i < n:
        label = out[i]
        j = i + 1
        while j < n and out[j] == label:
            j += 1

        run_len = j - i
        if label in ("left_lane_change", "right_lane_change") and run_len < int(min_run_ticks):
            prev_label = out[i - 1] if i > 0 else None
            next_label = out[j] if j < n else None
            fill = (
                prev_label if prev_label is not None and prev_label == next_label else "keep_lane"
            )
            for k in range(i, j):
                out[k] = fill
        i = j

    return out


def _realign_zero_evidence_lane_change_runs(
    decisions: List[str],
    lane_ids: List[Optional[str]],
    lane_offsets: List[Optional[float]],
    left_hits: List[int],
    right_hits: List[int],
    world_xyzh: Optional[np.ndarray] = None,
) -> Tuple[List[str], List[str]]:
    """For contiguous lane-change runs with zero temporal evidence, force one direction
    from net lateral offset change across the run.
    """
    if not decisions:
        return decisions, []

    out = list(decisions)
    reasons = ["none"] * len(out)
    n = len(out)
    i = 0
    while i < n:
        label = out[i]
        if label not in ("left_lane_change", "right_lane_change"):
            reasons[i] = "not_lane_change"
            i += 1
            continue

        j = i + 1
        while j < n and out[j] in ("left_lane_change", "right_lane_change"):
            j += 1
        run_len = j - i

        if any((left_hits[k] > 0 or right_hits[k] > 0) for k in range(i, j)):
            for k in range(i, j):
                reasons[k] = "kept_has_temporal_hits"
            i = j
            continue

        start_lane = lane_ids[i]
        end_lane = lane_ids[j - 1]
        ego_lat_disp = None
        if world_xyzh is not None:
            ego_lat_disp = _signed_ego_lateral_displacement(world_xyzh, i, j - 1)

        delta_off = _signed_lateral_offset_delta(lane_offsets, i, j - 1)
        offset_signal = _offset_delta_to_left_positive_signal(delta_off)
        primary_delta = (
            ego_lat_disp
            if ego_lat_disp is not None and np.isfinite(ego_lat_disp)
            else offset_signal
        )

        # Use earliest detected lane-id transition as local cue; whole-run net delta can cancel out.
        local_delta = None
        first_change_idx = None
        if start_lane is not None:
            for t in range(i + 1, j):
                lid = lane_ids[t]
                if lid is not None and lid != start_lane:
                    first_change_idx = t
                    break
        if first_change_idx is not None:
            local_ego_lat_disp = None
            if world_xyzh is not None:
                local_ego_lat_disp = _signed_ego_lateral_displacement(
                    world_xyzh, i, first_change_idx
                )
            local_delta_off = _signed_lateral_offset_delta(lane_offsets, i, first_change_idx)
            local_offset_signal = _offset_delta_to_left_positive_signal(local_delta_off)
            local_delta = (
                local_ego_lat_disp
                if local_ego_lat_disp is not None and np.isfinite(local_ego_lat_disp)
                else local_offset_signal
            )
            if local_delta is not None and abs(local_delta) >= abs(primary_delta or 0.0):
                primary_delta = local_delta

        # Very long zero-evidence runs are usually lane-id jitter artifacts.
        if run_len >= int(ZERO_EVIDENCE_LONG_RUN_SUPPRESS_TICKS):
            for k in range(i, j):
                out[k] = "keep_lane"
                reasons[k] = "suppressed_long_zero_evidence_run"
            i = j
            continue

        is_short = run_len <= int(ZERO_EVIDENCE_SHORT_RUN_MAX_TICKS)
        has_lane_transition = (
            start_lane is not None and end_lane is not None and start_lane != end_lane
        )

        if is_short:
            # Preserve short ambiguous runs to avoid false negatives.
            # Direction uses strongest local/global displacement cue if available.
            direction_signal = None
            if local_delta is not None and np.isfinite(local_delta):
                direction_signal = float(local_delta)
            elif primary_delta is not None and np.isfinite(primary_delta):
                direction_signal = float(primary_delta)

            if direction_signal is None:
                # Keep original short-run labels if no reliable signal.
                for k in range(i, j):
                    reasons[k] = "kept_short_zero_evidence_no_signal"
                i = j
                continue

            target = "left_lane_change" if direction_signal > 0.0 else "right_lane_change"
            for k in range(i, j):
                out[k] = target
                reasons[k] = "short_zero_evidence_direction_from_displacement"
            i = j
            continue

        # For medium runs, require stronger displacement evidence.
        if primary_delta is None or abs(primary_delta) < float(
            AMBIGUOUS_DIRECTION_MIN_NET_LATERAL_OFFSET_M
        ):
            for k in range(i, j):
                out[k] = "keep_lane"
                reasons[k] = "suppressed_medium_weak_displacement"
            i = j
            continue

        # If there is no lane-id transition across the run, suppress medium/long runs.
        if not has_lane_transition:
            for k in range(i, j):
                out[k] = "keep_lane"
                reasons[k] = "suppressed_medium_no_lane_transition"
            i = j
            continue

        # Ego-frame convention: positive means lateral movement toward left.
        target = "left_lane_change" if primary_delta > 0.0 else "right_lane_change"
        for k in range(i, j):
            out[k] = target
            reasons[k] = "medium_zero_evidence_direction_from_displacement"
        i = j

    return out, reasons


def _confidence_label(score: float) -> str:
    """Map a numeric confidence score to a coarse confidence bucket."""
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def _compute_lane_confidence_tracks(
    lane_decisions: List[str],
    lane_ids: List[Optional[str]],
    lane_offsets: List[Optional[float]],
    lat_rates: np.ndarray,
    step_distances: np.ndarray,
    temporal_left_hits: List[int],
    temporal_right_hits: List[int],
    run_realign_reasons: Optional[List[str]],
) -> Tuple[List[float], List[str]]:
    """Compute diagnostic confidence for lane labels without affecting decisions."""
    n = len(lane_decisions)
    scores: List[float] = []
    labels: List[str] = []
    reasons = run_realign_reasons if run_realign_reasons is not None else ["none"] * n
    for i in range(n):
        decision = str(lane_decisions[i])
        map_ok = not (pd.isna(lane_ids[i]) or pd.isna(lane_offsets[i]))
        hits = int(temporal_left_hits[i]) + int(temporal_right_hits[i])
        lat_rate = float(abs(lat_rates[i])) if i < len(lat_rates) else 0.0
        step_d = float(max(step_distances[i], 0.0)) if i < len(step_distances) else 0.0
        disp = lat_rate * step_d
        reason = reasons[i] if i < len(reasons) else "none"

        if decision in ("left_lane_change", "right_lane_change"):
            if hits > 0:
                score = 0.92 if map_ok else 0.80
            elif "direction_from_displacement" in str(reason):
                score = 0.78 if map_ok else 0.70
                if disp >= 0.35:
                    score += 0.05
            else:
                score = 0.66 if map_ok else 0.56
                if disp < 0.10:
                    score -= 0.08
        else:
            score = 0.86 if map_ok else 0.62
            if "suppressed_" in str(reason):
                score = max(score, 0.74)

        score = float(max(0.05, min(0.99, score)))
        scores.append(score)
        labels.append(_confidence_label(score))

    return scores, labels


def _multi_window_lane_decision(
    start_relation: dict,
    end_lane_id_sets: List[set],
    instantaneous_decision: str,
) -> Tuple[str, int, int]:
    """Multi-window transition decision:
    - evaluate old-style relation checks over multiple lookahead horizons
    - prefer any consistent left/right evidence to boost lane-change recall
    """
    left_hits = 0
    right_hits = 0
    for end_lane_set in end_lane_id_sets:
        decision = _decision_from_end_lane_set(start_relation, end_lane_set, instantaneous_decision)
        if decision == "left_lane_change":
            left_hits += 1
        elif decision == "right_lane_change":
            right_hits += 1

    if left_hits > 0 and right_hits == 0:
        return ("left_lane_change", left_hits, right_hits)
    if right_hits > 0 and left_hits == 0:
        return ("right_lane_change", left_hits, right_hits)
    if left_hits > 0 and right_hits > 0:
        if instantaneous_decision in ("left_lane_change", "right_lane_change"):
            return (instantaneous_decision, left_hits, right_hits)
        return (
            "left_lane_change" if left_hits >= right_hits else "right_lane_change",
            left_hits,
            right_hits,
        )

    return (instantaneous_decision, left_hits, right_hits)


def _resolve_lane_debug_dir(scenario: Any) -> Optional[str]:
    """Resolve optional lane debug output dir."""
    cfg = getattr(scenario, "cfg", None)
    if isinstance(cfg, dict) and cfg.get("lane_debug", False):
        return str(cfg.get("lane_debug_dir", "tmp/lane_debug"))
    if os.environ.get("META_ACTION_LANE_DEBUG", "0") == "1":
        return os.environ.get("META_ACTION_LANE_DEBUG_DIR", "tmp/lane_debug")
    return None


def _dump_lane_prepare_debug(
    scenario: Any,
    agent_token: str,
    lane_center_ids: List[Optional[str]],
    inst_decisions: List[str],
    temporal_decisions: List[str],
    fused_pre_filter: List[str],
    final_decisions: List[str],
    left_hits: List[int],
    right_hits: List[int],
    horizons: List[int],
    lane_candidate_ids: List[str],
    lane_offsets: Optional[List[Optional[float]]] = None,
    lane_lateral_rates: Optional[List[float]] = None,
    heading_rates: Optional[List[float]] = None,
    step_distances: Optional[List[float]] = None,
    run_realign_reasons: Optional[List[str]] = None,
    lane_confidence_scores: Optional[List[float]] = None,
    lane_confidence_labels: Optional[List[str]] = None,
) -> None:
    """Dump per-timestep lane-decision diagnostics to CSV when debug is enabled."""
    debug_dir = _resolve_lane_debug_dir(scenario)
    if not debug_dir:
        return

    try:
        os.makedirs(debug_dir, exist_ok=True)
        clip_id = str(getattr(scenario, "clip_id", "unknown_clip"))
        safe_agent = str(agent_token).replace("/", "_")
        save_path = os.path.join(debug_dir, f"{clip_id}_{safe_agent}_lane_prepare_debug.csv")
        df = pd.DataFrame(
            {
                "ts": list(range(len(lane_center_ids))),
                "lane_centerline_id": lane_center_ids,
                "current_lane_candidates": lane_candidate_ids,
                "inst_lane_decision": inst_decisions,
                "temporal_lane_decision": temporal_decisions,
                "fused_before_flicker_filter": fused_pre_filter,
                "lane_decision_final": final_decisions,
                "temporal_left_hits": left_hits,
                "temporal_right_hits": right_hits,
                "lookahead_horizons_ticks": [",".join(map(str, horizons))] * len(lane_center_ids),
            }
        )
        if lane_offsets is not None:
            df["lane_lateral_offset_m"] = lane_offsets
        if lane_lateral_rates is not None:
            df["lane_lateral_change_rate_m_per_m"] = lane_lateral_rates
        if heading_rates is not None:
            df["heading_change_rate_rad_m"] = heading_rates
        if step_distances is not None:
            df["step_distance_m"] = step_distances
        if run_realign_reasons is not None:
            df["run_realign_reason"] = run_realign_reasons
        if lane_confidence_scores is not None:
            df["lane_confidence_score"] = lane_confidence_scores
        if lane_confidence_labels is not None:
            df["lane_confidence_label"] = lane_confidence_labels
        df.to_csv(save_path, index=False)
    except Exception:
        # Debug dump must not break production labeling flow.
        pass


def prepare_lane_data(scenario: Any, agent_token: str) -> pd.DataFrame:
    """Prepare per-timestep lane-related series for an agent.

    Columns returned (length T):
    - lane_centerline_id: str or None (current lane id at timestep, if any)
    - lane_lateral_offset_m: float or None (signed lateral offset to lane center, meters)
    - heading_change_rad: float (per-timestep heading delta, radians)
    - heading_change_rate_rad_m: float (per-timestep heading change rate, rad/m)
    - lane_lateral_change_rate_m_per_m: float (per-step change in lateral offset per meter driven)
    - step_distance_m: float (per-timestep distance from t to t+1, meters; 0 at last)

    Notes:
    - Requires vector map to be enabled (scenario.scene_batch.vector_maps).
    - Uses scenario.get_world_states to convert agent local states to world coords.
    - For now, the coordinate transform uses scenario's method which is correct for ego.
    """
    # Resolve agent index and per-timestep xyh
    agent_idx = list(scenario.all_agents_names).index(agent_token)
    agent_xyh = scenario.agent_trajdata["agent_xyh"][agent_idx].detach().cpu().numpy()  # (T, 3)

    # Build xyzh for transform
    agent_xyzh = np.concatenate(
        [agent_xyh[:, :2], np.zeros((agent_xyh.shape[0], 1)), agent_xyh[:, 2:3]], axis=1
    )  # (T,4)

    # World coords using scenario's transform
    world_xyzh = scenario.get_world_states(agent_xyzh, scenario.scene_batch)

    lane_center_ids: List[Optional[str]] = []
    lane_offsets: List[Optional[float]] = []
    lane_decisions: List[str] = []
    inst_lane_decisions: List[str] = []
    lane_candidate_sets: List[set] = []
    for ts in range(world_xyzh.shape[0]):
        lane_center_id = None
        offset_m = None
        try:
            current_lanes = scenario.scene_batch.vector_maps[0].get_current_lane(
                world_xyzh[ts], max_dist=2.0, max_heading_error=np.pi / 3
            )
            if current_lanes:
                current_lane_ids = {cur_lane.id for cur_lane in current_lanes}
                lane = current_lanes[0]
                lane_center_id = lane.id
                center = lane.center.xy  # (N,2)
                p = world_xyzh[ts, :2]
                segs = np.stack([center[:-1], center[1:]], axis=1)  # (N-1,2,2)
                v = segs[:, 1] - segs[:, 0]
                w = p - segs[:, 0]
                v_norm2 = (v**2).sum(axis=1)
                t = np.clip((w * v).sum(axis=1) / np.maximum(v_norm2, 1e-9), 0.0, 1.0)
                proj = segs[:, 0] + (v.T * t).T
                d = np.linalg.norm(proj - p, axis=1)
                idx = int(np.argmin(d))
                cross = np.cross(v[idx], p - segs[idx, 0])
                offset_m = float(np.sign(cross) * d[idx])
            else:
                current_lane_ids = set()
        except Exception:
            current_lane_ids = set()
            pass

        # Derive coarse lane decision for this timestep if available (keep/left/right)
        lane_decision = "keep_lane"
        try:
            if hasattr(scenario, "ego_lr"):
                lr = scenario.ego_lr[ts]
                # Use direct left/right set hits for instantaneous decision to preserve
                # historical lane-change recall.
                left_hits = len(current_lane_ids.intersection(lr.get("left_lanes", set())))
                right_hits = len(current_lane_ids.intersection(lr.get("right_lanes", set())))
                if left_hits > 0 and right_hits == 0:
                    lane_decision = "left_lane_change"
                elif right_hits > 0 and left_hits == 0:
                    lane_decision = "right_lane_change"
                elif left_hits > 0 and right_hits > 0:
                    # Ambiguous instantaneous hits: resolve with chosen center lane if possible.
                    if lane_center_id in lr.get("left_lanes", set()):
                        lane_decision = "left_lane_change"
                    elif lane_center_id in lr.get("right_lanes", set()):
                        lane_decision = "right_lane_change"
                    else:
                        lane_decision = "left_lane_change"
        except Exception:
            # Keep the default decision when lane-relation lookup is unavailable.
            pass

        lane_center_ids.append(lane_center_id)
        lane_offsets.append(offset_m)
        lane_decisions.append(lane_decision)
        inst_lane_decisions.append(lane_decision)
        lane_candidate_sets.append(current_lane_ids)

    # Temporal smoothing/fusion:
    # 1) old-style transition decision using lookahead lane id
    # 2) fuse with instantaneous decision
    # 3) suppress short left/right flickers
    temporal_lane_decisions = list(inst_lane_decisions)
    temporal_left_hits = [0] * len(lane_center_ids)
    temporal_right_hits = [0] * len(lane_center_ids)
    fused_pre_filter = list(lane_decisions)
    used_horizons: List[int] = []
    run_realign_reasons: Optional[List[str]] = None
    if hasattr(scenario, "ego_lr") and len(lane_center_ids) == len(lane_decisions):
        fused: List[str] = []
        max_horizon = int(max(1, LANE_DECISION_LOOKAHEAD_TICKS))
        step_horizon = int(max(1, LANE_DECISION_WINDOW_STEP_TICKS))
        horizons = list(range(step_horizon, max_horizon + 1, step_horizon))
        if len(horizons) == 0 or horizons[-1] != max_horizon:
            horizons.append(max_horizon)
        used_horizons = horizons

        for ts in range(len(lane_center_ids)):
            start_relation = scenario.ego_lr[ts] if ts < len(scenario.ego_lr) else {}
            inst_decision = lane_decisions[ts]
            end_lane_id_sets = [
                lane_candidate_sets[min(ts + h, len(lane_candidate_sets) - 1)] for h in horizons
            ]
            temporal_decision, left_hits, right_hits = _multi_window_lane_decision(
                start_relation, end_lane_id_sets, inst_decision
            )
            if (
                temporal_decision in ("left_lane_change", "right_lane_change")
                and left_hits == 0
                and right_hits == 0
            ):
                # No temporal evidence: first try a strong lateral-direction cue;
                # if unavailable, fall back to center-lane transition.
                future_idx = min(ts + max_horizon, len(lane_center_ids) - 1)
                future_center_lane = lane_center_ids[future_idx]
                center_temporal_decision = _old_style_relation_decision(
                    start_relation, future_center_lane
                )
                lateral_temporal_decision = None
                start_lane = lane_center_ids[ts]
                lane_change_idx = future_idx
                if start_lane is not None:
                    for j in range(ts + 1, future_idx + 1):
                        end_lane = lane_center_ids[j]
                        if end_lane is not None and end_lane != start_lane:
                            lane_change_idx = j
                            break

                if start_lane is not None and lane_center_ids[lane_change_idx] is not None:
                    # First preference: map-independent ego-frame lateral displacement.
                    ego_lat_disp = _signed_ego_lateral_displacement(world_xyzh, ts, lane_change_idx)
                    if ego_lat_disp is not None and abs(ego_lat_disp) >= float(
                        AMBIGUOUS_DIRECTION_MIN_NET_LATERAL_OFFSET_M
                    ):
                        lateral_temporal_decision = (
                            "left_lane_change" if ego_lat_disp > 0.0 else "right_lane_change"
                        )

                if (
                    lateral_temporal_decision is None
                    and start_lane is not None
                    and lane_center_ids[lane_change_idx] is not None
                ):
                    delta_off = _signed_lateral_offset_delta(lane_offsets, ts, lane_change_idx)
                    offset_signal = _offset_delta_to_left_positive_signal(delta_off)
                    if offset_signal is not None and abs(offset_signal) >= float(
                        AMBIGUOUS_DIRECTION_MIN_NET_LATERAL_OFFSET_M
                    ):
                        # `offset_signal` is normalized to ego-like convention:
                        # positive means shifting toward the left adjacent lane.
                        lateral_temporal_decision = (
                            "left_lane_change" if offset_signal > 0.0 else "right_lane_change"
                        )

                if lateral_temporal_decision in (
                    "left_lane_change",
                    "right_lane_change",
                ):
                    temporal_decision = lateral_temporal_decision
                elif center_temporal_decision in (
                    "left_lane_change",
                    "right_lane_change",
                ):
                    temporal_decision = center_temporal_decision
                else:
                    # Preserve historical behavior (avoid recall drop):
                    # if all disambiguation is inconclusive, keep instantaneous label.
                    temporal_decision = inst_decision
            temporal_lane_decisions[ts] = temporal_decision
            temporal_left_hits[ts] = left_hits
            temporal_right_hits[ts] = right_hits
            fused.append(temporal_decision)
        fused, run_realign_reasons = _realign_zero_evidence_lane_change_runs(
            fused,
            lane_center_ids,
            lane_offsets,
            temporal_left_hits,
            temporal_right_hits,
            world_xyzh=world_xyzh,
        )
        fused_pre_filter = list(fused)
        lane_decisions = _drop_short_lane_change_runs(fused)

    # Compute per-timestep heading change and rate using world states
    # Differences are length T-1; pad at the start to align to T
    world_xy = world_xyzh[:, :2]
    world_h = world_xyzh[:, 3]
    diff_xy = np.diff(world_xy, axis=0)
    dist = np.linalg.norm(diff_xy, axis=1)
    dh = np.diff(world_h)
    # normalize dh to [-pi, pi]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    rate = np.zeros_like(dh)
    nz = dist > 1e-6
    rate[nz] = dh[nz] / dist[nz]
    # pad to length T
    if dh.size == 0:
        dh_pad = np.zeros(world_h.shape[0])
        rate_pad = np.zeros(world_h.shape[0])
    else:
        dh_pad = np.concatenate([[dh[0]], dh])
        rate_pad = np.concatenate([[rate[0]], rate])
    # step distance aligned to length T by padding zero at last index
    if dist.size == 0:
        step_dist_pad = np.zeros(world_h.shape[0])
        ego_lat_step_pad = np.zeros(world_h.shape[0])
    else:
        step_dist_pad = np.concatenate([dist, [0.0]])
        # Ego-frame signed lateral motion per step (left positive) from world trajectory.
        left_vec_x = -np.sin(world_h[:-1])
        left_vec_y = np.cos(world_h[:-1])
        ego_lat_step = diff_xy[:, 0] * left_vec_x + diff_xy[:, 1] * left_vec_y
        ego_lat_step_pad = np.concatenate([ego_lat_step, [0.0]])

    # lateral change rate per meter (based on lane_lateral_offset_m)
    off_arr = np.array([np.nan if v is None else float(v) for v in lane_offsets], dtype=float)
    doff = np.diff(off_arr)
    lat_rate = np.zeros_like(step_dist_pad[:-1])
    nz_lat = dist > 1e-6
    lat_rate[nz_lat] = doff[nz_lat] / dist[nz_lat]
    if doff.size == 0:
        lat_rate_pad = np.zeros(off_arr.shape[0])
    else:
        lat_rate_pad = np.concatenate([[lat_rate[0]], lat_rate])

    lane_conf_scores, lane_conf_labels = _compute_lane_confidence_tracks(
        lane_decisions=lane_decisions,
        lane_ids=lane_center_ids,
        lane_offsets=lane_offsets,
        lat_rates=lat_rate_pad,
        step_distances=step_dist_pad,
        temporal_left_hits=temporal_left_hits,
        temporal_right_hits=temporal_right_hits,
        run_realign_reasons=run_realign_reasons,
    )

    # Optional detailed debug dump with final per-timestep kinematics and resolver reasons.
    _dump_lane_prepare_debug(
        scenario=scenario,
        agent_token=agent_token,
        lane_center_ids=lane_center_ids,
        inst_decisions=inst_lane_decisions,
        temporal_decisions=temporal_lane_decisions,
        fused_pre_filter=fused_pre_filter,
        final_decisions=lane_decisions,
        left_hits=temporal_left_hits,
        right_hits=temporal_right_hits,
        horizons=used_horizons,
        lane_candidate_ids=[",".join(sorted(list(s))) for s in lane_candidate_sets],
        lane_offsets=lane_offsets,
        lane_lateral_rates=lat_rate_pad.tolist(),
        heading_rates=rate_pad.tolist(),
        step_distances=step_dist_pad.tolist(),
        run_realign_reasons=run_realign_reasons,
        lane_confidence_scores=lane_conf_scores,
        lane_confidence_labels=lane_conf_labels,
    )
    return pd.DataFrame(
        {
            "lane_centerline_id": lane_center_ids,
            "lane_lateral_offset_m": lane_offsets,
            "lane_decision": lane_decisions,
            "heading_change_rad": dh_pad,
            "heading_change_rate_rad_m": rate_pad,
            "lane_lateral_change_rate_m_per_m": lat_rate_pad,
            "step_distance_m": step_dist_pad,
            "ego_lateral_step_m": ego_lat_step_pad,
            "lane_confidence_score": lane_conf_scores,
            "lane_confidence_label": lane_conf_labels,
        }
    )


def segment_lane(
    scenario: Any,
    agent_token: str,
    segmentation_thresholds: List[Threshold] = LANE_SEGMENTATION_THRESHOLDS,
    min_segment_len_ticks: int = 3,
) -> pd.DataFrame:
    """Segment one agent's lane timeline into stable decision intervals."""
    df_series = prepare_lane_data(scenario, agent_token)

    # Threshold states
    threshold_states = []
    for th in list(segmentation_thresholds or []):
        series_np = df_series[th.property_name].to_numpy(dtype=float)
        threshold_states.append((series_np >= float(th.threshold)).astype(np.int8))

    # Categorical states
    cat_series = [df_series[k].to_numpy() for k in LANE_SEGMENTATION_CATEGORICAL]

    T = len(df_series)
    if T < 2:
        return pd.DataFrame(
            {
                "segment_start_time": [],
                "duration": [],
                "average_lane_lateral_offset": [],
                "max_abs_lane_lateral_offset": [],
                "average_heading_change_rate_rad_m": [],
                "heading_change_rad": [],
                "max_lane_lateral_change_rate_m_per_m": [],
                "min_lane_lateral_change_rate_m_per_m": [],
                "ego_net_lateral_displacement_m": [],
                "ego_positive_step_ratio": [],
                "ego_negative_step_ratio": [],
                "offset_pulse_max_rise_m": [],
                "offset_pulse_max_fall_m": [],
                "offset_pulse_first_dir": [],
                "lane_decision": [],
                "lane_centerline_id_changed": [],
                "lane_observable_ratio": [],
                "lane_unavailable_ticks": [],
                "lane_total_ticks": [],
                "lane_map_unavailable": [],
                "lane_confidence_score": [],
                "lane_confidence_label": [],
            }
        )

    # Segment boundaries
    segments = []
    start = 0

    def tuple_state(i: int) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        """Build the threshold+categorical state tuple at index `i`."""
        num = tuple(int(s[i]) for s in threshold_states)
        cat = tuple(cs[i] for cs in cat_series)
        return (num, cat)

    cur = tuple_state(0)
    i = 1
    while i < T:
        nxt = tuple_state(i)
        if nxt != cur and (i - start) >= int(min_segment_len_ticks):
            # Avoid cutting within last 3 ticks
            if (T - i) <= 3:
                break
            segments.append((start, i))
            start = i
            cur = nxt
            # 3-tick cooldown after a cut
            i += 3
            continue
        i += 1
    segments.append((start, T - 1))

    # Metrics
    dt = float(DELTA_TIMESTAMP)
    off = df_series["lane_lateral_offset_m"].to_numpy()
    h_rate = df_series["heading_change_rate_rad_m"].to_numpy()
    dhead = df_series["heading_change_rad"].to_numpy()
    step_dist = df_series["step_distance_m"].to_numpy()
    ego_lat_step = df_series["ego_lateral_step_m"].to_numpy()
    lat_rate_mpm = df_series["lane_lateral_change_rate_m_per_m"].to_numpy()
    ldec = df_series["lane_decision"].to_numpy()
    lid = df_series["lane_centerline_id"].to_numpy()
    lconf = df_series["lane_confidence_score"].to_numpy(dtype=float)

    rows = []
    for s0, s1 in segments:
        s0 = max(0, min(int(s0), T - 2))
        s1 = min(int(s1), T - 1)
        if s1 <= s0:
            continue
        start_time = s0 * dt
        duration = (s1 - s0) * dt
        avg_off = _safe_nanmean(off[s0:s1])
        max_abs_off = _safe_nanmax_abs(off[s0:s1])
        avg_h_rate = _safe_nanmean(h_rate[s0:s1])
        avg_lat_rate = _safe_nanmean(lat_rate_mpm[s0:s1])
        max_lat_rate = _safe_nanmax(lat_rate_mpm[s0:s1], default=0.0)
        min_lat_rate = _safe_nanmin(lat_rate_mpm[s0:s1], default=0.0)
        pulse_rise, pulse_fall, pulse_dir = _offset_pulse_metrics(off[s0:s1])
        ego_net_lat = float(np.nansum(ego_lat_step[s0:s1]))
        ego_valid = ego_lat_step[s0:s1][np.isfinite(ego_lat_step[s0:s1])]
        if ego_valid.size > 0:
            eps = 1e-3
            ego_pos_ratio = float((ego_valid > eps).sum() / ego_valid.size)
            ego_neg_ratio = float((ego_valid < -eps).sum() / ego_valid.size)
        else:
            ego_pos_ratio = 0.0
            ego_neg_ratio = 0.0
        sum_dh = float(np.nansum(dhead[s0:s1]))
        sum_dist = float(np.nansum(step_dist[s0:s1]))
        seg_dec = ldec[s0]
        ids = lid[s0:s1]
        changed = bool(len(ids) > 1 and any(ids[j] != ids[0] for j in range(1, len(ids))))
        total_ticks = int(max(0, s1 - s0))
        unavailable_ticks = int(sum(pd.isna(v) for v in ids))
        observable_ratio = (
            float(total_ticks - unavailable_ticks) / float(total_ticks) if total_ticks > 0 else 0.0
        )
        conf_score = _safe_nanmean(lconf[s0:s1])
        rows.append(
            {
                "segment_start_time": start_time,
                "duration": duration,
                "average_lane_lateral_offset": avg_off,
                "max_abs_lane_lateral_offset": max_abs_off,
                "average_heading_change_rate_rad_m": avg_h_rate,
                "average_lane_lateral_change_rate_m_per_m": avg_lat_rate,
                "max_lane_lateral_change_rate_m_per_m": max_lat_rate,
                "min_lane_lateral_change_rate_m_per_m": min_lat_rate,
                "ego_net_lateral_displacement_m": ego_net_lat,
                "ego_positive_step_ratio": ego_pos_ratio,
                "ego_negative_step_ratio": ego_neg_ratio,
                "offset_pulse_max_rise_m": pulse_rise,
                "offset_pulse_max_fall_m": pulse_fall,
                "offset_pulse_first_dir": pulse_dir,
                "heading_change_rad": sum_dh,
                "delta_distance_driven_m": sum_dist,
                "lane_decision": seg_dec,
                "lane_centerline_id_changed": changed,
                "lane_observable_ratio": observable_ratio,
                "lane_unavailable_ticks": unavailable_ticks,
                "lane_total_ticks": total_ticks,
                "lane_map_unavailable": bool(observable_ratio < 0.5),
                "lane_confidence_score": conf_score,
                "lane_confidence_label": _confidence_label(
                    conf_score if np.isfinite(conf_score) else 0.5
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("segment_start_time").reset_index(drop=True)
