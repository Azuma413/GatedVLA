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

from typing import TYPE_CHECKING, Any, List, Sequence

if TYPE_CHECKING:
    from meta_action.data_structures.scenario import TemporalScenario


class TemporalMotionChunk:
    """Base temporal span for one motion label on one agent."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        self.agent_token = agent_token
        self.start_ts = start_ts
        self.end_ts = end_ts

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: "TemporalScenario"
    ) -> Sequence["TemporalMotionChunk"]:
        """Return temporal motion chunks for one agent in one scenario.

        Concrete subclasses should override this method.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Render a compact debug string for the temporal chunk."""
        return f"{self.__class__.__name__}({self.agent_token} at {self.start_ts}-{self.end_ts})"


class MotionTags:
    """Container wrapper used by downstream dataset pipelines."""

    def __init__(self, motion_tags: List[Any]) -> None:
        # List[Dict]-like snapshots
        self.motion_tags = motion_tags

    def __to__(self, device: Any, non_blocking: bool = False) -> "MotionTags":
        """Keep API compatibility with tensor-like `.to(...)` calls."""
        return self

    def __collate__(self, batch: List["MotionTags"]) -> "MotionTags":
        """Collate a batch of `MotionTags` objects into one object."""
        result: List[Any] = []
        for item in batch:
            result += item.motion_tags

        return MotionTags(result)

    def __getitem__(self, idx: int) -> Any:
        """Return one stored motion-tag entry."""
        return self.motion_tags[idx]

    def __len__(self) -> int:
        """Return number of tags in the first snapshot entry."""
        return len(self.motion_tags[0]) if self.motion_tags else 0

    def __str__(self) -> str:
        """Render a human-readable summary of current motion tags."""
        try:
            # Expect a single snapshot list under self.motion_tags[0]
            tags = self.motion_tags[0]
            parts = []
            for tag in tags:
                name = tag.get("tag", "?")
                agents = tag.get("agents", [])
                if isinstance(agents, list):
                    agent_str = ",".join(str(a) for a in agents)
                else:
                    agent_str = str(agents)
                interval = tag.get("interval", ("?", "?"))
                start_ts, end_ts = interval
                parts.append(f"{name}({agent_str} at {start_ts}-{end_ts})")
            return "[" + "; ".join(parts) + "]"
        except Exception:
            return f"MotionTags(len={len(self)})"

    def __repr__(self) -> str:
        """Return the canonical debug representation."""
        return self.__str__()
