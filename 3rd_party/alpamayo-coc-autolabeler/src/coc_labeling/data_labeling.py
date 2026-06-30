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

"""Hydra entrypoint for dataset labeling jobs.

This module configures runtime/server settings, resolves output directories,
and executes the labeling agent for either a single dataset or a split group
batch run.
"""

# pylint: disable=import-error

import logging
import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast

import hydra
from omegaconf import DictConfig

from coc_labeling.agents.labeling_agent import LabelingAgent
from coc_labeling.model_clients.runtime_config import ModelRuntimeConfig
from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.utils import io as my_io_utils


def config_ip(cfg: DictConfig) -> Optional[str]:
    """Resolve server IP from config.

    Args:
        cfg: Runtime configuration.

    Returns:
        The selected IP address when provided, otherwise ``None``.
    """
    # Override IP extracted during autoresume jobs.
    server_ip = cfg.server_ip
    if server_ip is not None:
        if "," in server_ip:
            ip_addr = None
            logging.info(
                "Multiple server IPs were provided; model wrappers will receive the "
                "IP assignment from their model arguments."
            )
        else:
            ip_addr = server_ip
            logging.info("Job server IP provided as %s.", ip_addr)
    else:
        ip_addr = None

    return ip_addr


def _build_save_root(
    resume_exp_dir: Optional[str],
    save_name: str,
    model_name: str,
    has_group_split: bool,
) -> Path:
    """Build the output root path based on resume and split settings."""
    if not has_group_split:
        if resume_exp_dir is None:
            return Path("experiments") / save_name
        if os.path.isabs(resume_exp_dir):
            return Path(resume_exp_dir)
        return Path("experiments") / resume_exp_dir

    base_root = Path("experiments") / f"cot_data_{model_name}"
    if resume_exp_dir is None:
        return base_root / save_name
    if os.path.isabs(resume_exp_dir):
        return Path(resume_exp_dir)
    return base_root / resume_exp_dir


def _get_optional_cfg_value(cfg: DictConfig, key: str) -> Optional[str]:
    """Return a config value only when the key exists and is not null."""
    value = getattr(cfg, key, None)
    if value is None:
        return None
    return str(value)


def _get_segment_generator_type(cfg: DictConfig) -> str:
    """Resolve the configured segment generator type."""
    keyframe_cfg = getattr(cfg.data_loader, "keyframe", None)
    if keyframe_cfg is not None:
        keyframe_segment_generator_type = getattr(keyframe_cfg, "segment_generator_type", None)
        if keyframe_segment_generator_type is not None:
            return str(keyframe_segment_generator_type)

    segment_generator_type = getattr(cfg.data_loader, "segment_generator_type", None)
    if segment_generator_type is None:
        raise ValueError(
            "Missing data_loader.keyframe.segment_generator_type or "
            "data_loader.segment_generator_type."
        )
    return str(segment_generator_type)


def _join_group_root(root_dir: str, group_id: str) -> str:
    """Build a per-group path from a validated non-null root directory."""
    return os.path.join(root_dir, group_id)


def _get_timeout_sec(cfg: DictConfig) -> int:
    """Resolve model request timeout from config, falling back to the runtime default."""
    timeout_sec = getattr(cfg, "timeout_sec", TIMEOUT_MAX)
    if timeout_sec is None:
        return TIMEOUT_MAX
    return int(timeout_sec)


def _apply_group_paths(cfg: DictConfig, group_id: str, segment_generator_type: str) -> None:
    """Apply group-specific data paths when group root configs are provided."""
    data_dir_root = _get_optional_cfg_value(cfg.data, "data_dir_root")
    cache_dir_root = _get_optional_cfg_value(cfg.data, "cache_dir_root")
    if data_dir_root is not None:
        if cache_dir_root is None:
            raise ValueError("Group-split data_dir_root requires data.cache_dir_root.")
        cfg.data.data_dir = _join_group_root(data_dir_root, group_id)
        cfg.data.cache_dir = _join_group_root(cache_dir_root, group_id)

    segment_config_dir = _get_optional_cfg_value(cfg.data, "segment_config_dir")
    if segment_generator_type == "json" and segment_config_dir is not None:
        cfg.data.segment_config_path = os.path.join(segment_config_dir, group_id + ".json")

    video_dir_root = _get_optional_cfg_value(cfg.data, "video_dir_root")
    if video_dir_root is not None:
        cfg.data.video_dir = _join_group_root(video_dir_root, group_id)

    meta_action_dir_root = _get_optional_cfg_value(cfg.data, "meta_action_dir_root")
    if meta_action_dir_root is not None:
        cfg.data.meta_action_dir = _join_group_root(meta_action_dir_root, group_id)


def config_job(cfg: DictConfig) -> Tuple[str, str, Optional[List[str]]]:
    """Resolve verified model name, output root, and optional group split list.

    Args:
        cfg: Runtime configuration.

    Returns:
        A tuple of:
        1. Verified model name string.
        2. Output directory path as string.
        3. Optional list of group IDs for split processing.
    """
    # Override the group_list during auto-resume
    group_list: Optional[List[str]] = None
    if cfg.job_split_id is not None:
        group_id = str(cfg.job_split_id)

        # single group
        if "-" not in group_id:
            group_list = [group_id]
        else:
            group_list = group_id.split("-")

    # set up exp_name
    current_time = time.strftime("%Y%m%d_%H%M%S")
    if cfg.exp_name is not None and len(cfg.exp_name) > 0:
        save_name = current_time + "_" + cfg.exp_name
    else:
        save_name = current_time

    # set up save folder
    model_name = str(cfg.model_name)
    logging.info("Using model: %s", model_name)
    save_root = _build_save_root(
        resume_exp_dir=cfg.resume_exp_dir,
        save_name=save_name,
        model_name=model_name,
        has_group_split=group_list is not None,
    )
    if group_list is None:
        save_root.mkdir(exist_ok=True, parents=True)
    else:
        my_io_utils.mkdir_if_missing(str(save_root))
    logging.info("CoT results saved to %s", save_root)

    return model_name, str(save_root), group_list


@hydra.main(config_path="config", config_name="base_config_vlm", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run labeling job for one dataset or for multiple split groups.

    Args:
        cfg: Hydra-composed runtime configuration.
    """
    # check if we need to split the job id and assign corresponding ip address
    ip_addr = config_ip(cfg)
    runtime_config = ModelRuntimeConfig(ip_addr=ip_addr, timeout_sec=_get_timeout_sec(cfg))
    verified_model_name, save_root, group_list = config_job(cfg)
    cfg.model_name = verified_model_name
    segment_generator_type = _get_segment_generator_type(cfg)

    # single job
    if group_list is None:
        labeling_agent = LabelingAgent(
            cfg=cfg,
            save_root=save_root,
            verbose=cfg.verbose,
            mode=cfg.mode,
            ip_addr=ip_addr,
            runtime_config=runtime_config,
        )
        labeling_agent.parse_dataset(cfg, save_root)
        labeling_agent.run()

    # go through a list of data dirs
    else:
        # only init the model once
        labeling_agent = LabelingAgent(
            cfg=cfg,
            save_root=save_root,
            verbose=cfg.verbose,
            mode=cfg.mode,
            ip_addr=ip_addr,
            runtime_config=runtime_config,
        )

        for group_id_str in group_list:
            # Null roots mean "leave the single-node path as configured" for small OSS runs.
            _apply_group_paths(cfg, group_id_str, segment_generator_type)

            # update the save folder to use group_id_str
            save_root_tmp = os.path.join(save_root, group_id_str)
            my_io_utils.mkdir_if_missing(save_root_tmp)

            logging.info("Running group job %s.", group_id_str)
            logging.info("Data directory: %s", cfg.data.data_dir)
            logging.info("Cache directory: %s", cfg.data.cache_dir)
            if "video_dir" in cfg.data:
                logging.info("Video directory: %s", cfg.data.video_dir)
            if "meta_action_dir" in cfg.data:
                logging.info("Meta-action directory: %s", cfg.data.meta_action_dir)
            logging.info("Segment config path: %s", cfg.data.segment_config_path)
            logging.info("Group save root: %s", save_root_tmp)

            # inference according to the meta action loop
            try:
                labeling_agent.parse_dataset(cfg, save_root_tmp)
                labeling_agent.run()
            # Keep batch execution alive for other groups even if one fails.
            except Exception:  # pylint: disable=broad-except
                logging.exception("Error parsing or running dataset for %s", group_id_str)


if __name__ == "__main__":
    hydra_entrypoint: Callable[[], None] = cast(Callable[[], None], main)
    hydra_entrypoint()  # pylint: disable=no-value-for-parameter
