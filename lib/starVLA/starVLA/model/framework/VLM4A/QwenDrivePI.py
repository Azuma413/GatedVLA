# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenDrivePI Framework
A pi-style driving policy: Qwen-Drive-1.0 VLM (Qwen3.5-4B) + layer-wise flow-matching action expert.

Architecture
────────────
- VLM: ``Qwen/Qwen-Drive-1.0-4B`` (driving-pretrained Qwen3.5-4B, 32 layers, hidden 2560).
  Input is the front camera only: ``[front_{t-k}, ..., front_{t-1}, front_t]`` plus a fixed
  prompt.  Following Qwen-Drive, the current frame gets a large pixel budget and history
  frames a small one.
- Action expert: layer-wise DiT (as in QwenPI_v3).  Each VLM layer is projected
  (LayerNorm + Linear) to the expert width (1024) and the DiT interleaves cross-attention
  to the VLM layers with self-attention over ``[proprio tokens | future tokens | action tokens]``,
  so actions attend to both the scene and the ego state (pi0 / DriveMoE style).

Proprio / output design (follows DriveMoE DrivePi0, Thinklab-SJTU/DriveMoE)
───────────────────────────────────────────────────────────────────────────
- state  : (state_horizon=5, state_dim=10) ego history at 10 Hz, frames ``[t-4 .. t]``, per step
           ``[speed(1), acceleration(3), angular_velocity(3), heading_rel(1), target_x(1), target_y(1)]``.
           heading is relative to the current yaw; the target point is in the current ego frame.
           Each step is one token (MLP + learned step position embedding) fed to the action expert,
           not to the VLM.
- action : (action_horizon=10, action_dim=2) ego-frame (x forward, y lateral) waypoints at 10 Hz,
           ``t+0 .. t+9``, absolute positions in the current ego frame (not deltas).
- Both are min-max normalised to [-1, 1] with 1st/99th percentiles (starVLA ``q01/q99`` stats);
  sampled actions are clamped to [-1, 1].
- Flow matching: Beta(1.5, 1) time sampling with s=0.999, MSE on velocity, 10 Euler steps at inference.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenDrivePIDefaultConfig:
    """QwenDrivePI framework default parameters (YAML ``framework:`` keys override these)."""

    name: str = "QwenDrivePI"

    # === VLM backbone (Qwen-Drive-1.0 / Qwen3.5) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # HF hub id or local snapshot of Qwen-Drive-1.0 (must contain "Qwen-Drive" to be routed here)
            "base_vlm": "Qwen/Qwen-Drive-1.0-4B",
            "attn_implementation": "sdpa",
            "vl_hidden_dim": 2560,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 32,  # auto-overridden at runtime from the loaded VLM
        }
    )

    # === Driving-specific inputs ===
    drive: dict = field(
        default_factory=lambda: {
            # Fixed instruction (DriveMoE uses "predict trajectory"); a per-sample "lang" overrides it.
            "prompt": "predict trajectory",
            # Pixel budgets from Qwen-Drive config.json (current: 1280x720, history: 544x320).
            "current_image_pixels": 921600,
            "history_image_pixels": 174080,
            # Qwen3.5 vision: patch 16 x spatial merge 2 -> one token per 32x32 pixels.
            "image_factor": 32,
            # Sampled (normalised) actions are clamped to [-action_clip, action_clip].
            "action_clip": 1.0,
        }
    )

    # === Action expert (layer-wise flow-matching DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 2,
            "action_horizon": 10,
            "state_dim": 10,
            "state_horizon": 5,
            "repeated_diffusion_steps": 4,
            "num_inference_timesteps": 10,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                # Odd layers self-attend over [state | future | action] tokens so that actions see the
                # ego state; even layers cross-attend to the corresponding VLM layer.
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


class StateTokenEncoder(nn.Module):
    """One token per proprio history step: shared MLP + learned step embedding."""

    def __init__(self, mlp: nn.Module, state_horizon: int, hidden_dim: int):
        super().__init__()
        self.mlp = mlp
        self.step_embedding = nn.Parameter(torch.zeros(1, state_horizon, hidden_dim))
        nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.mlp(state)  # (B, T_state, D)
        return x + self.step_embedding[:, -x.shape[1] :].to(x.dtype)


def resize_to_pixel_budget(image: Image.Image, max_pixels: int, factor: int) -> Image.Image:
    """Aspect-preserving downscale so that both sides are multiples of ``factor`` and H*W <= max_pixels."""
    w, h = image.size
    scale = min(1.0, math.sqrt(max_pixels / float(w * h)))
    new_w = max(factor, int(w * scale // factor) * factor)
    new_h = max(factor, int(h * scale // factor) * factor)
    if (new_w, new_h) == (w, h):
        return image
    return image.resize((new_w, new_h), Image.BICUBIC)


@FRAMEWORK_REGISTRY.register("QwenDrivePI")
class Qwen_Drive_PI(baseframework):
    """
    Front-camera driving VLA: Qwen-Drive VLM + DriveMoE-style proprio/trajectory flow-matching expert.

    Sample schema (``examples``: List[dict]):
        - image : List[PIL.Image | np.ndarray]  front camera frames, oldest first, last = current frame
        - lang  : str (optional)                 defaults to ``framework.drive.prompt``
        - state : np.ndarray (state_horizon, state_dim)   normalised ego history
        - action: np.ndarray (T >= action_horizon, action_dim)  normalised future waypoints (training only)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenDrivePIDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None) or llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        # The layer-wise head decodes the DiT pre-output (return_pre_output=True); the DiT's own output
        # projection is never used, so keep it out of the optimizer.
        for name in ("proj_out_1", "proj_out_2"):
            module = getattr(self.action_model.model, name, None)
            if module is not None:
                module.requires_grad_(False)

        action_cfg = self.config.framework.action_model
        self.action_horizon = int(action_cfg.action_horizon)
        self.state_horizon = int(action_cfg.get("state_horizon", 1))
        if self.action_model.state_encoder is not None:
            self.action_model.state_encoder = StateTokenEncoder(
                self.action_model.state_encoder, self.state_horizon, self.action_dit_hidden_dim
            )

        # With interleaved self-attention the odd DiT blocks never read the VLM, so only the
        # cross-attention blocks get a projector (avoids unused parameters under DDP).
        self.interleave_self_attention = bool(diffusion_model_cfg.get("interleave_self_attention", False))
        self.cross_layer_ids = [
            i for i in range(self.num_action_dit_layers) if not (self.interleave_self_attention and i % 2 == 1)
        ]
        # VLM layer read by each cross-attention block: the deeper layer of each (cross, self) pair,
        # so that the final VLM layer is used.
        self.vlm_layer_ids = [
            min(i + 1, num_vl_layers - 1) if self.interleave_self_attention else i for i in self.cross_layer_ids
        ]
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in self.cross_layer_ids
            ]
        )

        drive_cfg = self.config.framework.drive
        self.prompt = str(drive_cfg.get("prompt", "predict trajectory"))
        self.current_image_pixels = int(drive_cfg.get("current_image_pixels", 921600))
        self.history_image_pixels = int(drive_cfg.get("history_image_pixels", 174080))
        self.image_factor = int(drive_cfg.get("image_factor", 32))
        self.action_clip = drive_cfg.get("action_clip", 1.0)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _prepare_images(self, frames: List) -> List[Image.Image]:
        frames = to_pil_preserve(list(frames))
        out = []
        for i, img in enumerate(frames):
            budget = self.current_image_pixels if i == len(frames) - 1 else self.history_image_pixels
            out.append(resize_to_pixel_budget(img.convert("RGB"), budget, self.image_factor))
        return out

    def _prepare_inputs(self, examples: List[dict]):
        batch_images = [self._prepare_images(example["image"]) for example in examples]
        instructions = [example.get("lang") or self.prompt for example in examples]
        state = None
        if self.action_model.state_encoder is not None and "state" in examples[0]:
            state = np.stack([np.asarray(example["state"], dtype=np.float32) for example in examples])
            state = state.reshape(len(examples), -1, state.shape[-1])  # (B, T_state, state_dim)
        return batch_images, instructions, state

    def _head_dtype(self) -> torch.dtype:
        return next(self.action_model.parameters()).dtype

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> tuple:
        """Run the VLM, keep the last ``num_action_dit_layers`` hidden states and project them to the expert width.

        The returned list has one entry per DiT block; entries for self-attention blocks are placeholders
        (the DiT ignores them) that alias the preceding cross-attention entry.
        """
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        attention_mask = qwen_inputs.get("attention_mask", None)
        # Call the inner Qwen3.5 model (no LM head): only hidden states are needed.
        qwenvl_outputs = self.qwen_vl_interface.model.model(
            **qwen_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = qwenvl_outputs.hidden_states[-self.num_action_dit_layers :]
        proj_dtype = next(self.project_layers.parameters(), torch.empty(0, dtype=self._head_dtype())).dtype
        projected = {
            i: proj(hidden_states[j].to(proj_dtype))
            for i, j, proj in zip(self.cross_layer_ids, self.vlm_layer_ids, self.project_layers)
        }
        vl_embs_list = []
        for i in range(self.num_action_dit_layers):
            vl_embs_list.append(projected[i] if i in projected else vl_embs_list[-1])
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        return vl_embs_list, attention_mask

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------
    def forward(self, examples: List[dict] = None, predict: bool = False, **kwargs) -> dict:
        """Training loss, or (``predict=True``) sampled actions.

        Prediction is routed through ``forward`` so that parameter-sharding wrappers (FSDP2) that hook the
        root ``forward`` gather the parameters for inference as well.
        """
        if predict:
            with torch.no_grad():
                return self._predict(examples)
        batch_images, instructions, state = self._prepare_inputs(examples)
        vl_embs_list, attention_mask = self._encode_vl_hidden_states(batch_images, instructions)

        device, dtype = vl_embs_list[-1].device, self._head_dtype()
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        actions = torch.as_tensor(np.array([example["action"] for example in examples]), device=device, dtype=dtype)
        actions_target = actions[:, -self.action_horizon :, :]

        repeat = self.config.trainer.get("repeated_diffusion_steps", None) if self.config.get("trainer") else None
        repeat = int(repeat or self.config.framework.action_model.get("repeated_diffusion_steps", 1))

        actions_target = actions_target.repeat(repeat, 1, 1)
        vl_embs_list = [h.repeat(repeat, 1, 1) for h in vl_embs_list]
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(repeat, 1)
        state_t = None
        if state is not None:
            state_t = torch.as_tensor(state, device=device, dtype=dtype).repeat(repeat, 1, 1)

        action_loss = self.action_model(
            vl_embs_list,
            actions_target,
            state_t,
            encoder_attention_mask=attention_mask,
        )
        return {"action_loss": action_loss}

    def _predict(self, examples: List[dict]) -> dict:
        batch_images, instructions, state = self._prepare_inputs(examples)
        vl_embs_list, attention_mask = self._encode_vl_hidden_states(batch_images, instructions)

        device, dtype = vl_embs_list[-1].device, self._head_dtype()
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        state_t = torch.as_tensor(state, device=device, dtype=dtype) if state is not None else None

        pred_actions = self.action_model.predict_action(vl_embs_list, state_t, encoder_attention_mask=attention_mask)
        if self.action_clip is not None:
            pred_actions = pred_actions.clamp(-float(self.action_clip), float(self.action_clip))
        return {"normalized_actions": pred_actions.float().cpu().numpy()}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        """Returns ``normalized_actions`` (B, action_horizon, action_dim) in [-action_clip, action_clip]."""
        if type(examples) is not list:
            examples = [examples]
        return self._predict(examples)
