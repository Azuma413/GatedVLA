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

"""General-purpose helpers for model/config/path data handling."""

import os
from collections import namedtuple
from typing import Any, Dict

import yaml

import coc_labeling.utils.io as io_utils


def save_yaml(file_path: str, data: Dict[str, Any]) -> None:
    """Save mapping to YAML file, creating parent folder if needed."""
    io_utils.mkdir_if_missing(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, indent=4, sort_keys=False)


def dict_to_namedtuple(name: str, data_dict: Dict[str, Any]) -> Any:
    """Recursively convert nested dictionaries to namedtuples."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = dict_to_namedtuple(key, value)

    return namedtuple(name, data_dict.keys())(*data_dict.values())


def namedtuple_to_dict(obj: Any) -> Any:
    """Recursively convert namedtuple/list/dict containers to dictionaries."""
    as_dict = getattr(obj, "_asdict", None)
    if isinstance(obj, tuple) and callable(as_dict):
        as_dict_result = as_dict()
        if isinstance(as_dict_result, dict):
            return {k: namedtuple_to_dict(v) for k, v in as_dict_result.items()}
    if isinstance(obj, list):
        return [namedtuple_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: namedtuple_to_dict(v) for k, v in obj.items()}
    return obj


def get_vlm_yaml_path(save_root: str, data: Dict[str, Any]) -> str:
    """Construct the output YAML file path."""
    clip_dir = os.path.join(save_root, data["clip_id"])
    save_path = os.path.join(clip_dir, f"cot_{data['event_start_timestamp']}.yaml")
    return save_path
