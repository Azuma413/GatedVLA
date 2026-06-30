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

"""Focused second-pass repair prompt for lane-change/nudge contradiction fixes."""

PROMPT_VERSION = "coc_repair_v1"

system = """
You are a repair assistant for autonomous-driving behavior summaries.
Revise the previous output only when it contradicts strong lane-level meta-action evidence.
Return valid JSON that strictly matches the provided schema.
"""

images_prompt = """
These are front-view image frames for the same clip.
Use them to verify whether lane change or nudge behavior is present.
"""

meta_action_prompt = """
These are sampled meta-actions for the same clip.
The lane field is high-priority for this repair step:
- Left Lane Change / Right Lane Change
- Slightly Shift Left / Slightly Shift Right
"""

repair_context_template = """
Repair task context:
- Base prompt version: {base_prompt_version}
- Trigger reason: {trigger_reason}
- Onset-based lane-prior summary: {lateral_prior_summary}

Initial model output (JSON):
{initial_json}

Repair policy:
1. Keep factual content from the initial output when it does not conflict.
2. If lane prior indicates lane change or nudge and initial output collapses to keep-lane or wrong lateral type, revise to include the expected lateral maneuver with direction.
3. Use onset timing preference:
   - prioritize maneuver near decision window (around t=0) and early-onset maneuver;
   - later opposite-direction behavior is often recovery and should not override earlier primary maneuver.
4. Nudge trigger can be a single frame of Slightly Shift Left/Right.
5. Lane-change trigger requires at least two frames of same-direction lane-change labels.
6. Gap-search repair rule:
   - if lateral maneuver (lane change or nudge) is present and longitudinal timeline shows waiting/modulation before onset (stop-hold, decel-hold, creep, or delayed acceleration), prefer Gap-search over generic keep-distance/resume-speed wording.
   - if Gap-search is used, explicitly include the blocker cause in the same sentence (target-lane vehicle, parked/stopped vehicle, cones/roadwork, oncoming vehicle, etc.).
   - do not output blocker-free Gap-search phrasing.
7. Keep output as one concise sentence in effect_on_ego_behavior.
"""

output_prompts = """
Return only schema-compliant JSON with field:
- ego_behavior_schema.effect_on_ego_behavior

Do not add extra fields.
"""

repair_examples = """
Repair examples:
1. Keep-lane collapse -> Nudge
- Prior summary: slight_shift_left appears at t=-1.0s; no strong lane-change run.
- Initial: "Keep lane and maintain a safe speed while following the lead vehicle."
- Repaired: "Nudge left out of the lane to pass the vehicle in front."

2. Keep-lane collapse -> Lane Change
- Prior summary: left_lane_change has 2+ consecutive frames near t=0.
- Initial: "Maintain a safe distance from the lead vehicle while keeping lane."
- Repaired: "Lane change to the left while maintaining clearance from the vehicle in the target lane."

3. Opposite-direction recovery
- Prior summary: slight_shift_left starts earlier; slight_shift_right appears later as recovery.
- Initial: "Nudge right to create clearance from the roadside obstacle."
- Repaired: "Nudge left out of the lane to create clearance from the roadside obstacle."

4. Onset order Lane Change vs Nudge
- Prior summary: nudge starts at t=-1.5s; lane-change onset starts at t=+2.0s.
- Initial: "Lane change to the left to proceed along the route."
- Repaired: "Nudge left out of the lane to pass the stopped vehicle blocking the lane."

5. Lane Change with Gap-search
- Prior summary: right_lane_change 2+ consecutive; pre-maneuver speed modulation before onset.
- Initial: "Lane change to the right to proceed along the route."
- Repaired: "Adjust speed to find a gap for a right lane change because the adjacent lane is temporarily occupied."
"""

output_prompts = output_prompts + "\n" + repair_examples
