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

# ruff: noqa: E501

"""Prompt templates for the VLM CoC agent."""

from typing import Any, Dict

PROMPT_VERSION = "coc_autolabeling_release"


_TEMPORAL_FALLBACK: Dict[str, float] = {
    "hist_length_sec": 2,
    "fut_length_sec": 6,
    "time_interval": 0.5,
}


def _cfg_get(cfg: Any, key: str) -> Any:
    """Read a key from dict-like, OmegaConf, or namespace config objects."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(key)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key)
        except (KeyError, TypeError, AttributeError):
            return None
    return getattr(cfg, key, None)


def _resolve_temporal_config(data_loader_config: Any = None) -> Dict[str, float]:
    """Resolve prompt timing from the same data_loader config used by loaders."""
    video_cfg = _cfg_get(data_loader_config, "video")
    meta_action_cfg = _cfg_get(data_loader_config, "meta_action")
    vector_cfg = _cfg_get(data_loader_config, "vector")
    sections = (video_cfg, meta_action_cfg, vector_cfg, data_loader_config)

    temporal: Dict[str, float] = {}
    for key, fallback in _TEMPORAL_FALLBACK.items():
        value = next(
            (_cfg_get(section, key) for section in sections if _cfg_get(section, key) is not None),
            fallback,
        )
        temporal[key] = float(value)
    return temporal


def _format_number(value: float) -> str:
    """Render whole-number floats without a trailing decimal."""
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _temporal_values(data_loader_config: Any = None) -> Dict[str, str]:
    """Return formatted temporal values used by prompt text."""
    temporal = _resolve_temporal_config(data_loader_config)
    hist_sec = temporal["hist_length_sec"]
    fut_sec = temporal["fut_length_sec"]
    time_interval = temporal["time_interval"]
    sampled_fps = 1.0 / time_interval if time_interval else 0.0
    return {
        "hist_sec": _format_number(hist_sec),
        "fut_sec": _format_number(fut_sec),
        "total_sec": _format_number(hist_sec + fut_sec),
        "sampled_fps": _format_number(sampled_fps),
    }


def render_images_prompt(data_loader_config: Any = None) -> str:
    """Render the image prompt with resolved loader timing."""
    values = _temporal_values(data_loader_config)
    return f"""
These are front-view image frames at **{values["sampled_fps"]} Hz** captured by the ego vehicle.
The input video length is {values["total_sec"]} seconds, with {values["hist_sec"]} seconds of historical window and {values["fut_sec"]} seconds of future window.
"""


def _render_general_prompt(data_loader_config: Any = None) -> str:
    """Render timing-sensitive portions of the main task prompt."""
    default_values = _temporal_values()
    values = _temporal_values(data_loader_config)
    text = general_prompt
    text = text.replace(
        "The video length is "
        f"{default_values['total_sec']} seconds, with {default_values['hist_sec']} "
        f"seconds of historical window and {default_values['fut_sec']} seconds of future window.",
        "The video length is "
        f"{values['total_sec']} seconds, with {values['hist_sec']} seconds of historical "
        f"window and {values['fut_sec']} seconds of future window.",
    )
    return text.replace(
        f"at the {default_values['hist_sec']} to {DECISION_MAKING_TIME_SEC} seconds mark",
        f"at the {values['hist_sec']} to {DECISION_MAKING_TIME_SEC} seconds mark",
    )


def render_output_prompts(data_loader_config: Any = None) -> str:
    """Render the output prompt with resolved loader timing."""
    return (
        question_prompt + "\n" + _render_general_prompt(data_loader_config) + "\n" + general_example
    )


_temporal = _resolve_temporal_config()

HISTORY_LENGTH_SEC = (
    int(_temporal["hist_length_sec"])
    if _temporal["hist_length_sec"].is_integer()
    else _temporal["hist_length_sec"]
)
DECISION_MAKING_TIME_SEC = 4
FUTURE_LENGTH_SEC = (
    int(_temporal["fut_length_sec"])
    if _temporal["fut_length_sec"].is_integer()
    else _temporal["fut_length_sec"]
)
VIDEO_FPS = int(round(1.0 / _temporal["time_interval"])) if _temporal["time_interval"] else 2
TOTEL_LENGTH_SEC = HISTORY_LENGTH_SEC + FUTURE_LENGTH_SEC

system = """
You are a specialized agent for visual scene understanding in autonomous driving.
You are required to answer questions based on the provided inputs of a sequence of images.
You should outputs structured JSON following the schema strictly. Some hints are provided below to help you answer the questions.
"""

#### Input Prompts ####

images_prompt = render_images_prompt()

meta_action_prompt = """
These are the corresponding meta-actions paired with each frame above.
The "lane" field provides lane-level prior signals, including:
- Left Lane Change / Right Lane Change
- Slightly Shift Left / Slightly Shift Right
Use these as high-value hints for lateral behavior, but do not treat them as always correct.
Also inspect longitudinal meta-actions (Stop, deceleration, maintain-speed holding, acceleration)
together with lane priors to detect gap-search waiting/negotiation before lane change or nudge.
"""


ego_state_prompt = """
These are the ego vehicle's motion states at each frame. Positive lateral indicates rightward movement, negative lateral indicates leftward movement.
"""


#### Output Prompts ####

general_prompt = f"""
Now analyze the dash cam video inputs of ego vehicle to deduce following fields: effect_on_ego_behavior.

- effect_on_ego_behavior Analysis:
    Input:
        - Dash cam video inputs with the ego trajectory
        - The video length is {TOTEL_LENGTH_SEC} seconds, with {HISTORY_LENGTH_SEC} seconds of historical window and {FUTURE_LENGTH_SEC} seconds of future window.

    Instructions:
        Analyze the video and motion to deduce ego's driving decision at the {HISTORY_LENGTH_SEC} to {DECISION_MAKING_TIME_SEC} seconds mark according to the pre-defined decision list below:
        Here's a list of pre-defined LONGITUDINAL and LATERAL driving decisions:
        - LONGITUDINAL decisions
            -- Stop
                Decelerate to, hold at stop/yield lines or other control points (traffic light, stop sign, school bus/railroad rules, blocked path by lead vehicle).
            -- Yield
                Yield to the [pedestrian/cross-traffic/cyclist/emergency vehicle] / give way for / wait for priority traffic to clear.
            -- Pass/Overtake
                Accelerate to overtake / build speed to pass a moving vehicle.
            -- Gap-search
                Adjust speed to fit a planned merge/lane change/zipper, matching the target stream's speed or creating a usable gap. Only used when a lateral maneuver is planned; excludes generic car-follow.
                Especially use this when lane change is intended but the adjacent/target lane is temporarily blocked and ego modulates speed to wait for or create a safe opening.
                Also use this when a nudge is intended but ego must modulate speed to wait for a safe lateral clearance window around a blocker/hazard.
                Meta-action fusion trigger: when lane meta-action indicates lane change/nudge and longitudinal meta-actions show waiting/modulation (full stop, decel hold, crawl, or accel timing to enter), it is possible to choose "Gap-search" compared to generic keep-distance/resume-speed phrasing.
                Mandatory prep trigger: if lateral maneuver starts after a waiting phase (for example Stop/hold or slow-roll plateau) and acceleration resumes during/after lane entry, treat it as maneuver preparation and choose "Gap-search" rather than lane-change-only wording.
                Include the blocking reason when available (for example, parked vehicle, stopped vehicle, trailer truck, slow vehicle, oncoming vehicle, or temporary obstacle near the nudge side/target lane).
            -- Keep Distance
                Follow the lead vehicle/pedestrian/cyclist / keep a safe distance from [lead obstacle] / maintain a safe distance to [lead obstacle] / decelerate to maintain a safe distance to [lead obstacle].
            -- Adapt Speed
                Slow for road curvature / adjust speed for the [ramp/grade/curve] / modulate speed for the turn / pre-brake for the roundabout / slow down for the roundabout / slow down for speed bump.
            -- Resume Speed
                Track the set speed / hold the target speed / resume cruise / return to the speed profile / accelerate/decelerate to the speed limit. This is low priority and should not override any causal decision.
        - LATERAL decisions
            -- Lane Change
                Transition from current lane to adjacent target lane to follow route, overtake, or prepare for merge/diverge; includes lateral negotiation to secure the gap. Must specify direction, i.e. left or right, in the reasoning trace. A full lane transition, not a nudge.
            -- Nudge
                Temporary, small departure over a lane line to increase clearance around a blockage/hazard while staying on the same route. For example, nudge to the left to pass a stopped vehicle in the same lane.
            -- Turn
                Turn at an intersection, roundabout, or U-turn. Must specify direction, i.e. left or right, in the reasoning trace.
            -- Merge
                Merge into the mainline from an off-ramp or a weaving segment.
            -- Split
                Split from the mainline to an off-ramp or a weaving segment.
            -- Keep Lane
                Maintain lane position / stay centered / follow the lane / keep lane to proceed along the route. This is the strict lowest-priority fallback decision.

    Priority policy (must follow):
        1. First check if ego is conducting 'interesting' decisions:
           - LONGITUDINAL: 'Yield', 'Stop', 'Pass/Overtake', 'Gap-search', 'Adapt Speed'.
           - LATERAL: 'Lane Change', 'Nudge', 'Turn', 'Merge', 'Split'.
           - If any interesting decision is clearly present, output that decision and do not downgrade to generic 'Keep Distance', 'Keep Lane' or 'Resume Speed'.
        2. Lane change and nudge are top-priority lateral decisions:
           - If a lane change happens at any time in the entire video, lane change MUST be explicitly mentioned in effect_on_ego_behavior.
           - If a nudge happens at any time in the entire video, nudge should be explicitly mentioned unless full lane transition is clearer.
           - Treat transient nudge windows as valid: even 1 frame or 2 consecutive "Slightly Shift Left/Right" frames are still nudge candidates at any timestamp, including later future frames (for example around +2.0s to +5.0s).
           - Never omit lane change when present, and do not output only longitudinal behavior when lane transition exists.
           - Add one longitudinal decision only when it is interesting or clearly needed for safety context.
           - If lane change is present and ego speed is being adjusted to negotiate surrounding traffic in the target lane, pair lane change with longitudinal "Gap-search" (not generic "Resume Speed").
           - Lane-change priority over nudge: when explicit lane-change priors ("Left Lane Change"/"Right Lane Change") are sustained across consecutive frames around maneuver onset AND there is no meaningful slight-shift evidence near onset, prioritize lane change; do not downgrade to nudge unless strong evidence shows only a brief partial lateral departure without lane transition.
           - For lane change with delayed execution (ego waits before crossing due to target-lane occupancy), add "Gap-search" and mention the target-lane blocker.
           - For nudge with delayed execution (ego waits or modulates speed before moving laterally for clearance), pair "Nudge" with longitudinal "Gap-search" and mention the blocker/hazard side.
           - When lane prior indicates upcoming lane change/nudge and longitudinal meta-actions show pre-maneuver waiting or modulation (including short stop-hold-release or non-stop speed shaping), carefully consider to pair lateral maneuver with "Gap-search".
           - If longitudinal timeline shows Stop/hold -> Maintain/creep -> lane-change prior onset -> acceleration during/after lane entry, treat Gap-search as required unless strong contradiction exists.
           - Strong stop-prep trigger: if a sustained Stop/hold segment (for example about 2s or more at 2 Hz) occurs before first lane-change/nudge onset and acceleration follows onset, Gap-search is mandatory unless strong contradiction exists.
           - If lead vehicle is stopped/slow ahead and target lane is occupied (for example trailer vehicle), do not output lane-change-only route wording; output lane change + Gap-search with blocker cause.
           - If nudge is present while pedestrians/crossing users also constrain speed, keep the lateral "Nudge" and pair it with one longitudinal decision ("Yield" or "Adapt Speed") instead of outputting only longitudinal behavior.
           - Anti-collapse rule for pedestrian scenes: if any directional slight-shift evidence exists and blocker/clearance semantics are present, do not output "Yield" alone; output "Nudge left/right + Yield" or "Nudge left/right + Adapt Speed".
           - If slight-shift priors or multi-source cues suggest nudge, run a dedicated nudge verification before deciding between "Nudge" and generic keep-lane wording.
           - Exception for intersection turning conflict: if strong turn evidence exists at an intersection, resolve Turn vs Lane Change by earliest onset and stronger confidence, and output the dominant behavior.
           - Stop-sign/intersection anti-suppression rule: when lane prior or multi-source evidence indicates lane change/nudge, do not output longitudinal-only "Adapt Speed/Stop" wording for stop sign or intersection context; include the lateral decision and use stop-sign/intersection as contextual or paired longitudinal factor.
           - In stop-sign scenes with slight-shift evidence, prefer paired wording such as "Nudge left/right ... while adapting speed for the stop sign at the intersection" rather than longitudinal-only control-response wording.
        3. Lane-level meta-action prior usage (high priority, non-binding):
           - "Left Lane Change" / "Right Lane Change" in lane meta-action is strong evidence for "Lane Change" with direction.
           - "Slightly Shift Left" / "Slightly Shift Right" in lane meta-action is strong evidence for "Nudge" with direction.
           - Treat "Slightly Shift Left/Right" at any timestamp (history or future window) as a high-priority nudge-alert signal to inspect, not as ignorable noise.
           - If prior indicates lane maneuver, actively verify it using full-video visual evidence, ego motion trend.
           - If prior is absent, do NOT assume no lane change/nudge; still search for visual and motion evidence through the full video.
           - If prior conflicts with clear visual/motion evidence, trust the stronger multi-source evidence and keep reasoning factual.
           - If lane prior repeatedly shows "Slightly Shift Left/Right" across consecutive frames and there is no strong contradiction, explicitly output "Nudge" with that direction even if visual nudge cues are subtle.
           - Strong trigger: 2 or more consecutive "Slightly Shift Left/Right" frames at any timestamp should be treated as directional nudge-positive by default unless strong contradictory evidence proves no lateral clearance behavior.
           - Low-FPS interpretation rule: lane priors are sampled at about 2 Hz, so even a single "Slightly Shift Left/Right" frame corresponds to about 0.5s behavior and should be treated as meaningful evidence, not flicker, unless strong contradiction exists.
           - Denoised-prior trust rule: because lane priors are pre-processed before prompting, assume rare flicker; therefore single-frame slight-shift still deserves nudge verification with high priority.
           - Multi-window reinforcement: if slight-shift of the same direction appears in separated windows (for example early and later parts of the clip), treat this as persistent nudge intent, not noise.
           - Side-aware blocker mapping for nudge direction:
             - Slightly Shift Left usually implies creating clearance from blocker/hazard on ego's right (parked/stopped vehicle, cones, road-edge obstacle).
             - Slightly Shift Right usually implies creating clearance from blocker/hazard on ego's left (oncoming vehicle, centerline-side hazard, left-side obstacle).
           - When lateral meta-action says "Slightly Shift Left/Right", proactively validate directional nudge using visual + ego/map evidence before falling back to generic keep-lane wording.
        4. Cut-in causal-factor rule (must follow):
           - If a same-direction vehicle cuts into ego lane from another lane and affects ego behavior, explicitly mention the cut-in behavior in effect_on_ego_behavior.
           - This remains required even when other factors are also present (for example green light, intersection context, lead vehicle, construction).
           - When observable, include cut-in direction (from left/from right).
           - Do not use the generic fallback sentence when cut-in evidence exists, even if the cut-in appears only in a few frames.
        5. Construction-aware lateral policy (must follow):
           - If construction cones/zone/workers are visible anywhere in the video and ego laterally adjusts for clearance, explicitly output "Nudge" with direction (left or right) unless full lane transition is clearer.
           - Special case: if construction zone/cones/barriers/workers appear, include construction in causal factors even when impact on ego behavior is weak or indirect.
           - Be alert to nudge-left/right near cones, temporary lane boundaries, or road work equipment.
        5.1 Curvy-road mention rule (must follow):
           - If a curvy road/road curvature is visible, include curvy-road context in causal factors.
           - This can be combined with other valid factors (lead vehicle, construction, traffic light, etc.) in one sentence.
           - Do not fall back to generic "Keep lane and maintain speed when no immediate hazard or maneuver is required" when clear curvature is present.
        5.2 Speed-bump mention rule (must follow):
           - If speed bump is visible, include speed-bump context in causal factors.
           - This can be combined with other valid factors in one sentence.
           - Do not use the generic lowest-priority fallback sentence when clear speed-bump cues are present.
        6. Pass/Overtake usage constraint (must follow):
           - Use "Pass/Overtake" only when the target vehicle is moving in the traffic stream.
           - For parked/stationary vehicles or static blockers, use lateral "Nudge" (with direction) rather than "Pass/Overtake".
           - If speed modulation is also present around a stationary blocker, pair "Nudge" with "Adapt Speed" when relevant.
        7. If no interesting decision is present, choose from less specific decisions:
           - Keep Distance, Resume Speed.
        8. 'Keep Lane' is the lowest-priority fallback decision:
           - Use 'Keep Lane' only when there is no interesting behavior and no meaningful causal factor affecting ego behavior.
           - If any causal factor exists (lead vehicle, cut-in vehicle, construction, workers, traffic control, etc.), do not choose the generic 'Keep Lane' decision.
           - If curvy road is visible, include curvature context instead of this generic fallback.
           - If speed bump is visible, include speed-bump context instead of this generic fallback.
           - If really nothing interesting throughout the entire video, you can output 'Keep lane and maintain speed when no immediate hazard or maneuver is required'.
           - Hard ban for this fallback sentence: if lane meta-action contains any "Slightly Shift Left/Right" at any timestamp, do NOT output this generic fallback sentence.

    Critical definitions:
        1. Lead vehicle means a vehicle ahead in the same lane and moving in the same direction.
           - Do NOT call crossing traffic, adjacent-lane vehicles, or oncoming traffic a lead vehicle. Understand the lane present in the video first.
        2. Cut-in vehicle means a vehicle entering ego lane from adjacent lane and reducing headway.
           - If cut-in behavior happens and it affects ego behavior, explicitly mention the cut-in vehicle as a causal factor.
           - If observable, include cut-in direction (from left or from right).
           - Do not omit cut-in wording just because another factor is also true (traffic light/intersection/lead vehicle/construction).
           - Visual-temporal pattern to detect cut-in:
             1) A vehicle is initially in adjacent lane (left or right) and moving in same direction as ego.
             2) Across frames, that vehicle shifts laterally toward ego lane (crosses/straddles lane divider).
             3) The vehicle ends up ahead in ego lane (or partially occupies ego lane) with reduced headway.
             4) Ego shows reaction (deceleration, adapted speed, yielding, or increased following caution).
           - If this pattern is present, explicitly mention cut-in even if ego later stabilizes.
        3. Traffic light color must be identified explicitly as red, yellow, or green.
           - Do not collapse yellow into red or green.
           - If yellow and red both appear in sequence and affect ego stopping, use yellow/red wording.
           - Loose time-window rule for stop events: if yellow appears in the stop sequence, include yellow in final stop wording (e.g., "yellow/red traffic light"), even if red is also present later.
           - Use red-only stop wording only when yellow is never observed.
           - When a traffic light is visible, explicitly check whether an intersection/junction is present and mention intersection context in the final sentence when present.
        4. Construction/workers rule:
           - If construction zone, cones, lane closure, barriers, or workers appear, explicitly mention construction as a causal/context factor.
           - This is required even if construction does not clearly change the selected driving decision.
           - Prefer explicit wording such as "construction cones", "construction zone", or "workers" instead of generic "obstacle" when visible.
        4.1 Curvy-road rule:
           - If road curvature/curvy road appears, explicitly mention curvy-road context in the output.
           - This remains valid when combined with other factors such as construction, lead vehicle, or traffic controls.
        4.2 Speed-bump rule:
           - If speed bump appears, explicitly mention speed-bump context in the output.
           - This remains valid when combined with other factors such as lead vehicle, construction, or intersection context.
        5. Lane change vs nudge disambiguation:
           - Lane Change: ego fully transitions into adjacent lane (sustained lane occupancy change).
           - Nudge: temporary partial lane departure or slight lateral shift to create clearance, then remain/return on the original route lane.
           - If lane priors provide sustained explicit lane-change labels and slight-shift labels are absent, default to lane-change interpretation (not nudge).
           - If any meaningful slight-shift appears near maneuver onset, explicitly keep nudge as a competing candidate and resolve using full-transition evidence rather than lane-change-count alone.
           - Treat "Slightly Shift Left/Right" prior as nudge-biased unless full transition evidence supports lane change.
           - Around construction cones, default to nudge-left/right wording unless full adjacent-lane takeover is clearly sustained.
           - Repeated "Slightly Shift Left/Right" lane priors should be interpreted as nudge intent unless strong evidence proves no lateral clearance maneuver occurred.
           - A "Slightly Shift Left" followed later by "Slightly Shift Right" (or the reverse) is strong evidence of a pass-around nudge pattern (move out for clearance, then return).
           - In pass-around patterns with opposite-direction shifts, use the first slight-shift direction as the main nudge direction; treat the later opposite shift as return behavior after passing.
           - Time-order precedence over frequency: when both slight-shift directions appear in one pass-around episode, prioritize the first slight-shift direction as the nudge direction even if the later opposite direction appears more times.
           - Extended onset-anchor rule: apply the same time-order precedence when the first lateral evidence is lane-change prior (left/right lane change) and later opposite-direction shift/lane-change appears shortly after; treat later opposite behavior as likely return/recovery unless sustained-route-transition evidence is stronger.
           - Recovery-window rule: if a first slight-shift direction is followed by opposite-direction "Slightly Shift" or opposite-direction lane change within a short window (about 1-4s), treat the later opposite behavior as path recovery and keep the first slight-shift direction as the anchor lateral decision.
           - Apply the same rule symmetrically for both directions:
             - Slightly Shift Left -> (Slightly Shift Right or Lane Change Right) within ~1-4s: anchor to left nudge.
             - Slightly Shift Right -> (Slightly Shift Left or Lane Change Left) within ~1-4s: anchor to right nudge.
           - Priority override for pass-around pattern: when first slight-shift occurs near blocker/clearance interaction and opposite-direction shift or lane change appears shortly after, output nudge direction from the first shift and treat later opposite move as return-to-path behavior.
           - Do not flip nudge direction to match the later opposite lane change unless there is strong evidence of sustained route-following lane transition (not a brief recovery maneuver).
           - Do not use majority-count voting across timestamps to pick nudge direction when opposite-direction shifts exist; use onset order + blocker-side consistency instead.
           - In mixed opposite-direction lane-change priors (e.g., right lane change then left lane change), prefer earliest sustained lane-change direction as the primary maneuver when it aligns with blocker side and delayed-entry evidence; treat later opposite lane-change burst as possible recovery unless sustained route-transition evidence is stronger.
        6. Gap-search in lane-change or delayed-nudge scenarios:
            - If target-lane vehicles constrain immediate lane entry and ego accelerates/decelerates to find an opening, explicitly use "Gap-search" as the longitudinal decision.
            - Do not replace this with generic phrasing such as "accelerating to match traffic flow" when the purpose is to secure a lane-change gap.
            - If lane change direction is identified, explicitly inspect the corresponding target lane (left for left lane change, right for right lane change) for blockers and waiting behavior.
            - If blocker is observed OR waiting/negotiation behavior is observed, output lane change + gap-search together; mention blocker cause when visible.
            - If lane-change prior appears but lane crossing is delayed and speed is modulated before crossing, default to lane change + gap-search unless strong evidence shows free immediate entry.
            - If nudge direction is identified and nearby blocker/hazard constrains immediate lateral clearance, use nudge + gap-search when ego modulates speed to create a safe clearance window.
            - Explicitly use longitudinal meta-action timeline to help detect gap-search:
              - Pattern A (stop-based): pre-maneuver Stop/near-stop appears, then lane change/nudge starts, then gentle/strong acceleration while crossing or clearing.
              - Pattern B (non-stop modulation): no full stop, but clear decel/hold/accel shaping occurs before or during lane change/nudge to negotiate an opening.
            - In both patterns, if lateral maneuver exists and target side is constrained, do not output lane-change-only wording, also include the gap search.
            - Pattern C (prep with delayed onset): lane-change prior appears after an earlier waiting segment (Stop/decel/maintain-speed plateau), followed by acceleration while entering/clearing. Treat this as waiting-for-opening behavior and include Gap-search.
            - Pattern D (dual blocker): lead-vehicle constraint + target-lane blocker (for example trailer/parked vehicle in target lane) together imply delayed lane entry preparation; include Gap-search and mention both constraints when visible.
            - Pattern E (long-stop release): multi-frame Stop/hold, then first lateral onset, then acceleration ramp. Treat as explicit lane-change/nudge preparation and include Gap-search.
            - Pattern F (mild-modulation prep): long maintain-speed plateau, delayed lane-change onset near later frames, and only gentle acceleration around onset. If target lane is constrained (including rear-left/rear-right blocker in target lane), treat as gap-negotiation and include Gap-search (not lane-change-only overtake wording).
            - Pattern G (blocked-target-lane at intersection): if target-side blocker (e.g., scooter on right for right lane change) coincides with sustained right/left lane-change priors and longitudinal deceleration/yielding context, keep lane change + Gap-search; do not convert to opposite-direction nudge unless clear partial-lateral-only evidence dominates.
        7. Pass/Overtake vs stationary blocker:
           - Do not use "Pass/Overtake" for parked vehicles, stopped obstacle vehicles, cones, or other static blockers.
           - For these cases, prefer "Nudge left/right ..." with optional "Adapt Speed" if ego slows or modulates speed for safety.
        8. Lane Change vs Turn at intersections:
           - At or approaching an intersection, treat route-following turn intent as a first-class candidate that can override noisy lane-change priors.
           - If visual evidence clearly shows turning geometry (intersection entry + heading change into turn path), prefer "Turn left/right" over "Lane Change".
           - If both are plausible, choose whichever begins first and has stronger multi-source confidence (video geometry, ego heading/yaw trend, map/junction cues, meta-action consistency).

    Stop-cause priority (must follow whenever Stop is selected):
        1. Lead vehicle in ego lane, yellow or red traffic light (same level of priority)
        2. Crossing traffic.
        3. Other controls (stop sign, etc.).
        - If lead vehicle and red/yellow light both materially constrain ego stop, mention both in one sentence.
        - If lead vehicle stops for a red/yellow light ahead, include both lead vehicle and red/yellow light as causes.
        - If yellow appears anywhere in the stop event (with or without later red), prefer wording: "Stop for the yellow/red traffic light at the stop line."
        - Only use "Stop for the red traffic light at the stop line." when yellow light is never present in the video.

    Speed bump rule (must follow):
        - If a speed bump appears, explicitly mention speed bump as a causal/context factor.
        - Prefer "Adapt Speed" with a speed-bump cause unless another interesting decision has higher priority.
        - Reference cues for identifying speed bumps (supportive, not mandatory):
          1) Visual cues: raised hump profile, painted speed-bump stripes/chevrons, warning signs, or road text markings.
          2) Behavioral cues: brief deceleration before crossing, slower traversal over the bump, then acceleration after passing.
          3) Context cues: school/residential/traffic-calming areas where speed bumps are common.
          4) Caution at low FPS: do not rely only on camera up/down jitter; use it as weak supporting evidence.

    Best practices:
        0. Decision count rule:
           - Output at most one longitudinal decision and at most one lateral decision.
           - If both decisions are interesting, include one decision from each axis in one sentence and the associated causal factors.
           - If lane change is present, always include it even when combined with a longitudinal decision.
           - In lane-change negotiation scenarios, prefer the pair: lateral "Lane Change" + longitudinal "Gap-search".
           - In delayed-nudge negotiation scenarios, allow the pair: lateral "Nudge" + longitudinal "Gap-search".
           - Do not output lane-change-only wording when clear target-lane blocker OR waiting-for-opening behavior exists.
           - Do not output generic route-progress lane-change wording ("change lanes ... to proceed along the route") when prep patterns A/B/C/D indicate maneuver negotiation.
           - If only one axis has an interesting decision, output only that interesting decision and the associated causal factors (could be 1 or 2) tied to it.
           - If no interesting decision is present in either longitudinal or lateral direction, choose the most appropriate decision and causal factors to output.
           - Before finalizing a longitudinal-only sentence, run a lateral-presence check over full clip; if any clear lane-change or nudge evidence exists, revise to include one lateral decision.
        1. Use "crosswalk" and only mention it when a pedestrian is walking across the road.
        2. Avoid describing "potential", "possible", "likely" events; only describe factual causal factors and driving decision.
        3. Do not discuss road users that are not effectively affecting ego behavior.
        4. If a cut-in vehicle is the cause of braking or yielding, mention cut-in direction if observable.
        4.1 If cut-in behavior exists and affects ego, include it as a causal factor even when the chosen decision is "Adapt Speed", "Resume Speed", or includes intersection context.
        4.2 Cut-in anti-lazy guardrail:
           - Do not output "Keep lane and maintain speed when no immediate hazard or maneuver is required" when clear cut-in evidence exists.
           - Prefer wording like "Slow down/Adapt speed ... because [vehicle] is cutting in from the left/right."
        5. Do not output generic "continue driving" if there is clear lane change, merge, turn, nudge, cut-in reaction, yielding cue, construction impact, or speed-bump slowing.
        6. For lateral behavior, combine evidence from three sources before fallback:
           - Lane meta-action prior ("lane: ...")
           - Visual lane-marker and relative-position changes across frames
           - Ego/map signals (lateral movement sign trend, lane-divider distance changes)
           If this combined evidence supports lane change/nudge, output it.
           - If visual cues are weak but lane prior and ego/map signals support slight lateral shift, still output nudge direction rather than defaulting to Keep Lane.
        7. Construction mention rule:
           - If any construction cues (cones, barriers, workers, temporary channelization) are present, include construction in effect_on_ego_behavior.
           - Do not drop construction wording even when construction impact on behavior is subtle.
        8. Nudge fallback guardrail:
           - Do not output only "Keep Distance" or "Keep Lane" when repeated "Slightly Shift Left/Right" lane priors indicate a directional nudge around a blockage.
           - In such cases, include lateral nudge decision and direction; add one longitudinal decision only if it is clearly interesting.
           - Do not output generic "Resume Speed" when any "Slightly Shift Left/Right" prior appears and a plausible clearance/blockage interaction exists.
           - If short-burst slight-shift evidence appears, explicitly run a nudge-check and only use generic curvy-road keep-lane wording when that nudge-check is not supported by multi-source evidence.
           - If slight-shift evidence exists, do not output longitudinal-only wording ("Yield"/"Adapt Speed"/"Keep Distance") unless strong contradiction clearly rules out lateral clearance behavior.
           - If any "Slightly Shift Left/Right" appears, do not output the generic sentence "Keep lane and maintain speed when no immediate hazard or maneuver is required."
           - If any "Slightly Shift Left/Right" appears, avoid generic lead-vehicle-follow wording such as "Keep lane and maintain a safe speed while following the lead vehicle" unless strong contradiction explicitly rules out lateral clearance behavior.
           - If any "Slightly Shift Left/Right" appears after a stop-to-go transition, do not output longitudinal-only "Resume speed ... keep lane" wording; run a late-window nudge check and keep directional nudge when clearance/blocker evidence is present.
           - For 3 or more consecutive same-direction slight-shift frames (especially in early/history timestamps), treat nudge as mandatory unless strong contradictory evidence is explicitly present.
        9. Mixed nudge + pedestrian interactions:
           - If ego nudges around a stopped/parked blocker while also slowing for a crossing pedestrian, prefer "Nudge + Adapt Speed" or "Nudge + Yield" in one sentence.
           - Do not collapse to "Yield" alone when lateral nudge evidence is present.
           - If speed bump context is also present, prefer phrasing that keeps both lateral and longitudinal intent, e.g. "Nudge left ... and adapt speed for the speed bump while yielding to the pedestrian."
        10. Stationary blocker guardrail:
           - When the object being passed is parked/stationary, never output "Pass/Overtake"; use directional "Nudge" wording instead.
        11. Intersection mention rule:
           - If an intersection/junction is visible and relevant context exists, explicitly mention "intersection" in effect_on_ego_behavior.
           - This is especially required when a traffic light is mentioned and the scene is at an intersection.
           - Keep the main decision unchanged; add intersection as contextual causal wording when present.
        12. Curvy-road fallback guardrail:
           - Do not use the generic lowest-priority sentence when clear road curvature appears anywhere in the clip.
           - Mention curvy road explicitly, and combine with other valid factors when present.
        13. Speed-bump fallback guardrail:
           - Do not use the generic lowest-priority sentence when clear speed-bump cues appear anywhere in the clip.
           - Mention speed bump explicitly, and combine with other valid factors when present.

    Output:
        - effect_on_ego_behavior: A verb phrase with short reasoning. Refer to the examples below for format. Use clear and factual language.
        - Include one decision by default, or two decisions only when both are interesting (one longitudinal + one lateral), while always including lane change when present.
        - Special case: if construction cues appear, include construction wording together with other identified causal factors.
        - Special case: if curvy-road cues appear, include curvy-road wording together with other identified causal factors.
        - Special case: if speed-bump cues appear, include speed-bump wording together with other identified causal factors.
        - Respond in one sentence.
        - Do not use chained actions with "then".
        - Keep nouns functional (vehicle, pedestrian, cyclist, red light, yellow light, speed bump, cut-in vehicle, construction cones, workers) unless color is necessary for disambiguation.

    Strategy:
        1. Use the video from the first {DECISION_MAKING_TIME_SEC} seconds to identify the causal factors and reasoning chain.
        2. Use the entire video to better deduce the ego's driving decision at the {HISTORY_LENGTH_SEC} to {DECISION_MAKING_TIME_SEC} seconds mark.
        3. To identify decisions such as 'Lane Change' and 'Nudge', use the entire video to identify them, not restricted to the first {DECISION_MAKING_TIME_SEC} seconds. Mention them if present.
        4. Always double check again before finalizing:
           - whether any lane change exists,
           - whether any nudge/slight lateral shift exists,
           - whether construction zone or cut-in exists.
           If so, revise effect_on_ego_behavior and include them.
        4.1 If cut-in exists, do not finalize without explicit cut-in wording in the output sentence.
        4.2 Run dedicated cut-in sweep:
           - track adjacent-lane vehicle -> ego-lane entry -> reduced headway sequence.
        5. Never use "no lane change/nudge" reasoning solely because lane prior does not contain lane-change/shift labels.
        6. Perform a dedicated sweep:
           - check for construction cones/workers/lane closure signs that appear,
           - verify if ego nudges left/right to create clearance around them,
           - if construction is seen, include construction in final output regardless of whether nudge is selected.
        6.1 Check for curvy-road cues:
           - if road curvature appears, include curvy-road context in final output (do not ignore).
        6.2 Check for speed-bump cues:
           - if speed bump appears, include speed-bump context in final output (do not ignore).
           - use visual/behavioral/context cues as references, but do not require all cues to be present.
        7. For any detected lane change, run a gap-check before finalizing:
            - check whether a vehicle in the target lane blocks immediate entry near the time window of the lane change,
            - check whether ego modulates speed to wait for/create a usable opening,
            - if either check is yes, include longitudinal "Gap-search" together with lateral lane-change wording.
            - explicitly mention the blocking object in the target lane (e.g., trailer vehicle, parked vehicles, stopped vehicle, temporary obstacle).
        7.1 Step-by-step lane-change gap-search procedure:
            - Step A: determine lane-change direction (left/right).
            - Step B: inspect that target adjacent lane for occupancy/blockers across frames.
            - Step B.1: classify lateral type from priors:
              - if sustained explicit lane-change priors exist around onset and no meaningful slight-shift appears near onset, mark lateral type as lane-change-first;
              - only switch to nudge-first when slight-shift evidence dominates and full lane transition evidence is weak.
            - Step C: check for delayed crossing or speed modulation before lane entry.
            - Step C.1: explicitly cross-check lane meta-action timeline with longitudinal timeline:
              - if Stop/decel-hold appears before lane-change onset, treat as waiting-for-gap candidate;
              - if acceleration starts around lane crossing after waiting/modulation, strengthen gap-search confidence.
              - do not require a full stop; non-stop decel/hold/accel shaping is also valid for gap-search.
            - Step C.2: compute onset order explicitly:
              - mark the first lane-change timestamp;
              - check whether waiting segment starts before this onset and whether acceleration rises around/after onset;
              - if yes, classify as lane-change preparation (Gap-search-positive).
            - Step C.3: mild-modulation check:
              - if there is a long maintain-speed plateau before onset and only gentle acceleration after onset, do not treat this as evidence against Gap-search;
              - if target-lane blocker is observed (including rear target-lane vehicle reducing immediate entry feasibility), classify as Gap-search-positive.
            - Step D: if B or C is true, keep lane change and add longitudinal "Gap-search"; add blocker cause when visible.
            - Step E: if C.2 is true, Gap-search is mandatory unless strong evidence shows immediate free entry with no negotiation.
            - Step F: if C.3 is true and target-lane blocker exists, Gap-search is mandatory unless strong contradiction exists.
            - Step G: if B.1 is lane-change-first and blocker is on target side, do not output opposite-direction nudge as primary maneuver.
        7.2 For any detected nudge, run a delayed-clearance gap-check before finalizing:
            - Step A: determine nudge direction (left/right).
            - Step B: inspect the corresponding hazard side for blocker persistence and instantaneous clearance tightness.
            - Step C: check whether ego modulates speed before/during the nudge to wait for a safe clearance window.
            - Step D: if B and C are true (or strong waiting behavior is evident), keep nudge and add longitudinal "Gap-search"; mention blocker/hazard side when visible.
        8. For any repeated "Slightly Shift Left/Right" lane prior sequence, run a nudge-check before finalizing:
           - treat 3 or more consecutive frames with the same slight-shift direction as strong nudge evidence,
           - cross-check for nearby blockage/clearance reason (stopped lead vehicle, cones, parked vehicle, road edge hazard),
           - if no strong contradiction exists, output "Nudge left/right ..." instead of generic keep-distance/keep-lane wording.
           - If this repeated slight-shift sequence occurs in early/history timestamps (for example around -2.0s to -0.5s), preserve it as a valid nudge cue and do not discard it just because later frames return to "Keep Lane".
        9. For any single or short burst of "Slightly Shift Left/Right" at any timestamp, run an early/late-window nudge-check:
           - inspect nearby frames around that timestamp for blockers (reversing vehicle, parked/stopped vehicle, cones, temporary obstruction),
           - If blocker is reversing or backing into ego path, raise nudge confidence and avoid generic keep-lane fallback.
           - at 2 Hz, treat even 1 frame (about 0.5s) or 2 consecutive slight-shift frames as meaningful if blocker/clearance evidence exists near that time,
           - if evidence supports lateral clearance behavior, keep "Nudge" in final output even if later lane priors return to "Keep Lane".
           - if blocker evidence is weak but no contradiction exists, keep directional nudge even when the slight-shift burst appears in later future frames.
           - If the slight-shift burst appears after a stop-hold-release pattern, prioritize lateral interpretation over generic post-stop resume wording.
           - if slight-shift appears in both an early window and a later window with the same direction, treat nudge confidence as high and preserve directional nudge in final output.
        9.1 If no "Slightly Shift Left/Right" prior appears, still run a proactive nudge sweep:
           - inspect full clip for subtle lateral clearance behavior around blockers/hazards (parked vehicles, oncoming vehicles near centerline, cones, temporary obstacles),
           - use visual lane-marker relation changes + ego/map lateral signals to detect nudging intent,
           - only then decide between "Nudge" and generic keep-lane wording.
        10. Detect pass-around return pattern:
           - if slight-shift-left appears before/around the blocker and slight-shift-right appears after passing (or vice versa), treat this as one nudge maneuver with return-to-lane.
           - when opposite slight-shift directions appear close in time, anchor output direction to the first shift near the blocker (entry direction), not the later return shift.
           - if first slight-shift is followed by opposite-direction slight-shift or opposite-direction lane change within about 1-4s, treat the second behavior as recovery-to-path and keep the first slight-shift direction as the anchor for final lateral decision.
           - tolerate a delayed return: even when opposite-direction return appears after about 2s (for example around 2-4s later), still keep first slight-shift direction as nudge anchor when pass-around semantics are present.
           - If first slight-shift appears for only one frame but occurs earlier than a later opposite-direction multi-frame burst, still keep the first direction as the nudge anchor under the 2 Hz rule unless strong contradiction exists.
           - If first evidence is "Left Lane Change" (or "Right Lane Change") in early frames and opposite-direction lane change/shift follows within about 1-4s, treat this as move-out then return pattern by default; anchor final lateral direction to the first evidence unless sustained opposite-direction route transition is clearly stronger.
           - Tail-noise guardrail: an isolated opposite-direction lane prior appearing only in the final ~0.5-1.0s (especially after sustained Keep Lane) should not override earlier maneuver direction or suppress Gap-search if prep evidence already exists.
           - This anchor rule still applies when traffic control context (stop sign/intersection) is present; traffic control should be paired as longitudinal context, not used to replace the lateral nudge decision.
           - pair with "Adapt Speed" or "Yield" when pedestrians/crossing conflicts are also present.
        10.1 Side-specific nudge cause check:
           - If output is "Nudge left", preferentially look for right-side blocker causes (parked/stopped vehicle, right-edge construction cones, right-side temporary obstacle).
           - If output is "Nudge right", preferentially look for left-side blocker causes (oncoming vehicle near centerline, left-side obstacle, left-side temporary hazard).
           - Mention this blocker side explicitly when visible.
           - If right-side parked/stopped vehicle and "Slightly Shift Left" prior are both observed, default to "Nudge left" framing unless strong contradiction exists.
           - Direction-cause consistency check: reject outputs where nudge direction conflicts with visible blocker side (for example "Nudge right ... parked vehicle on the left" when evidence indicates blocker is on the front right with first slight-shift-left).
        11. Intersection context check before finalizing:
           - if traffic light is mentioned, explicitly verify whether ego is approaching/passing/inside an intersection.
           - when intersection is present, include intersection wording in the final sentence (for example, "through the intersection", "at the intersection").
           - intersection/stop-sign context should usually modify or pair with the main maneuver wording, not replace a detected lane change/nudge.
        12. Turn vs lane-change conflict check near intersections:
           - explicitly compare the onset time of turn cues vs lane-change cues.
           - if turn cues appear first and remain strong, output Turn and do not downgrade to Lane Change.
           - if lane-change cues appear first and turning cues are weak/late, keep Lane Change.
        13. Yellow-light stop wording check:
           - for any Stop at traffic light, scan for yellow-light presence.
           - if yellow appears anywhere, use yellow/red stop wording; if not, use red-only wording.


Organize the answers in the following format:
{{
    "ego_behavior_schema": {{
        "effect_on_ego_behavior": "<your answer based on the video>"
    }}
}}
"""

general_example = """
Some examples for effect_on_ego_behavior field:
Change lanes to the right while maintaining a safe distance from the lead vehicle and the vehicle in the right adjacent lane.
Lane change to the left while maintaining a safe distance from the yellow car in the left adjacent lane.
Lane change to the right to proceed along the route while maintaining a safe distance from the lead vehicle.
Change lane to the left to proceed along the route.
Turn left at the intersection to follow the route.
Decelerate to maintain a safe distance from the slower lead vehicle ahead.
Stop behind the lead vehicle in the same lane at the yellow/red traffic light.
Stop for the yellow/red traffic light at the stop line.
Stop for the red traffic light at the stop line.
Yield to crossing traffic before proceeding through the intersection.
Maintain lane and accelerate through the intersection because the light is green and the intersection is clear.
Keep lane and maintain a safe speed on the curvy road.
Keep lane and maintain a safe speed on the curvy road while following the lead vehicle and navigating through the construction zone.
Adapt speed for the upcoming speed bump.
Slow down to pass over the speed bump at a safe speed.
Decelerate to pass over the speed bump at a safe speed while following the lead vehicle.
Slow down to maintain a safe distance from the black SUV cutting into the lane from the right.
Decelerate because of the gray sedan cutting into the lane from the left.
Adapt speed through the green-light intersection because a gray sedan is cutting into the lane from the left.
Slow down to maintain a safe distance from the gray sedan cutting into the lane from the left.
Yield to the black car cutting in from the right lane.
Nudge left to pass the stopped vehicle blocking the lane near construction cones.
Nudge right to create clearance from road work equipment while keeping the same lane path.
Nudge left out of the lane to navigate around the construction cone.
Nudge left out of the lane to safely pass the stationary SUV in front.
Nudge right to create clearance from the stopped truck partially blocking the lane.
Nudge left out of the lane and adapt speed to pass the red car parking in front while maintaining a safe distance from the pedestrian crossing the street.
Nudge left around the parked vehicle while yielding to the pedestrian crossing ahead.
Nudge left out of the lane to safely pass the parked vehicle on the right.
Nudge left out of the lane to safely pass the parked vehicle on the front right.
Nudge left out of the lane to pass the black car reversing in front.
Nudge right out of the lane to increase clearance from the oncoming sedan on the left.
Adjust speed to find a gap for a right lane change because the vehicle on the right is blocking the adjacent lane.
Gap-search for a left lane change while maintaining clearance from the vehicle occupying the target lane.
Nudge right and adjust speed to search for a safe clearance gap from the oncoming vehicle on the left.
Adjust speed to find a gap because the vehicle with a trailer on the right is blocking the adjacent lane and the vehicle ahead is stopped, preventing an immediate right lane change.
Adjust speed to find a gap because parked vehicles on the left are blocking the left lane for a left lane change.
Follow the temporary lane delineated by traffic cones through the construction zone.
Make a slight leftward adjustment to navigate around construction cones while continuing to follow the lead vehicle.
Maintain a safe distance from the lead vehicle in the same lane.
Resume speed after the intersection clears.
"""

question_prompt = """
Answer the following questions based on the provided inputs and put the answers in the corresponding fields of the output schema.
"""


output_prompts = render_output_prompts()
