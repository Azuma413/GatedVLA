# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
Qwen-Drive-1.0 VLM backbone (https://huggingface.co/Qwen/Qwen-Drive-1.0-4B).

The release wraps an *unchanged* Qwen3.5-4B VLM inside ``QwenDriveForPlanning``:
the top-level ``config.json`` has ``model_type: qwen_drive`` with the VLM config
nested under ``vlm_config``, and the root ``model.safetensors`` stores the VLM
weights under a ``vlm.`` prefix (``vlm.model.language_model.*``,
``vlm.model.visual.*``).  The BEV perception head and the planning experts live
in sub-folders and are not loaded here.

This interface rebuilds a plain ``Qwen3_5ForConditionalGeneration`` from
``vlm_config`` and strips the ``vlm.`` prefix while loading, so it behaves
exactly like ``_QWen3_5_VL_Interface`` afterwards.  Plain Qwen3.5 checkpoints
(no ``vlm_config``) are also accepted.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, PretrainedConfig

from starVLA.model.modules.vlm.QWen3_5 import Qwen3_5ForConditionalGeneration, _QWen3_5_VL_Interface
from starVLA.model.tools import has_flash_attn
from starVLA.training.trainer_utils import initialize_overwatch

try:
    from transformers import Qwen3_5Config
except ImportError as import_error:
    raise ImportError("Qwen3.5 config class is unavailable. Please install transformers >= 5.2.0.") from import_error

logger = initialize_overwatch(__name__)


def load_qwen_drive_vlm(model_id: str, attn_implementation: str = "sdpa", dtype=torch.bfloat16, load_weights: bool = True):
    """Load the Qwen3.5 VLM contained in a Qwen-Drive checkpoint (local dir or HF hub id).

    ``load_weights=False`` only builds the architecture from ``config.json`` (randomly initialised), for
    inference from a full starVLA checkpoint that overwrites every VLM weight anyway; ``model_id`` then only
    needs the config and processor files, not the 9 GB ``model.safetensors``.
    """
    config_dict, _ = PretrainedConfig.get_config_dict(model_id)
    if "vlm_config" in config_dict:
        vlm_config = Qwen3_5Config(**config_dict["vlm_config"])
        key_mapping = {r"^vlm\.": ""}
    else:
        vlm_config = Qwen3_5Config(**config_dict)
        key_mapping = None
    if not load_weights:
        return Qwen3_5ForConditionalGeneration._from_config(
            vlm_config, attn_implementation=attn_implementation, dtype=dtype
        )
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        config=vlm_config,
        key_mapping=key_mapping,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )


def load_qwen_drive_processor(model_id: str):
    """AutoProcessor cannot resolve ``model_type: qwen_drive`` (no video processor config), so assemble
    the Qwen3-VL processor (used by Qwen3.5) from its parts when needed."""
    try:
        return AutoProcessor.from_pretrained(model_id)
    except ValueError:
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen3VLProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return Qwen3VLProcessor(
            image_processor=AutoImageProcessor.from_pretrained(model_id),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(),
            chat_template=tokenizer.chat_template,
        )


class _QwenDrive_VL_Interface(_QWen3_5_VL_Interface):
    """Qwen-Drive VLM wrapper; forward / generate / build_qwenvl_inputs are inherited from Qwen3.5."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        nn.Module.__init__(self)

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen-Drive-1.0-4B")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        if attn_implementation == "flash_attention_2" and not has_flash_attn():
            logger.warning("flash_attn not installed, falling back to sdpa")
            attn_implementation = "sdpa"

        self.model = load_qwen_drive_vlm(
            model_id,
            attn_implementation=attn_implementation,
            load_weights=bool(qwenvl_config.get("load_pretrained_weights", True)),
        )
        self.processor = load_qwen_drive_processor(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.config = config

        # align qwen3.5 with qwen2.5 (frameworks read config.hidden_size)
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
