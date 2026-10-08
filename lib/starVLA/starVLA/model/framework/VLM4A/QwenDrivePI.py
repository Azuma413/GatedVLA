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

Optional visual evidence gate (``framework.eye_gate``, EyePolicy / See Less style)
─────────────────────────────────────────────────────────────────────────────────
Per frame, a gate MLP reads the VLM final-layer hidden of each visual token ``h_p`` and the mean of the
text-token hidden states ``q`` and predicts ``m_p = σ(MLP([h_p, q, h_p ⊙ q]) / τ)``. ``log m_p`` is added to the
action expert's cross-attention logits on that visual token in every cross-attention block (the VLM itself is
unchanged; text tokens stay visible). Loss: ``L_gated + λ_full·L_full + λ_budget·Σ_v (mean_p m_p − ρ)²`` with
``L_full`` the same flow-matching sample without the gate. Inference uses the gated path.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # === Visual evidence gate on the action expert's view of the VLM visual tokens (EyePolicy) ===
    eye_gate: dict = field(
        default_factory=lambda: {
            "enabled": False,
            # "learned": gate MLP. Controls with the same keep ratio `budget`:
            # "uniform": m = budget for every token; "random": m ~ Bernoulli(budget) per token.
            "mode": "learned",
            "hidden_dim": 256,
            "temperature": 1.0,
            "init_keep": 0.99,  # initial σ(ℓ): training starts close to the ungated model
            "budget": 0.2,  # ρ, target mean of m per frame
            "budget_loss_weight": 0.1,
            "full_loss_weight": 1.0,  # weight of the ungated loss; 0 skips it
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


class ViewGate(nn.Module):
    """Per-frame gate head: ``ℓ_p = W2·swish(W1·[h_p, q, h_p⊙q])`` over RMS-normalised hidden states
    (same as EyePolicy's ``ViewGate``)."""

    def __init__(self, width: int, hidden_dim: int, init_keep: float):
        super().__init__()
        self.norm_h = nn.RMSNorm(width)
        self.norm_q = nn.RMSNorm(width)
        self.fc1 = nn.Linear(3 * width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, math.log(init_keep / (1 - init_keep)))

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """h: (N, W) visual-token hidden of one frame, q: (W,) pooled text hidden -> logits (N,)."""
        dtype = self.fc1.weight.dtype
        h = self.norm_h(h.to(dtype))
        q = self.norm_q(q.to(dtype))[None].expand_as(h)
        return self.fc2(F.silu(self.fc1(torch.cat([h, q, h * q], dim=-1)))).squeeze(-1).float()


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

        gate_cfg = self.config.framework.eye_gate
        self.gate_enabled = bool(gate_cfg.get("enabled", False))
        self.gate_mode = str(gate_cfg.get("mode", "learned"))
        if self.gate_mode not in ("learned", "uniform", "random"):
            raise ValueError(f"eye_gate.mode must be learned / uniform / random, got {self.gate_mode}")
        self.num_views = int(self.config.framework.drive.get("num_views", 2))
        if self.gate_enabled and self.gate_mode == "learned":
            self.eye_gates = nn.ModuleList(
                ViewGate(llm_hidden_size, int(gate_cfg.hidden_dim), float(gate_cfg.init_keep))
                for _ in range(self.num_views)
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
        all_hidden = qwenvl_outputs.hidden_states
        hidden_states = all_hidden[-self.num_action_dit_layers :]
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
        else:
            attention_mask = torch.ones(qwen_inputs["input_ids"].shape, dtype=torch.bool, device=all_hidden[-1].device)
        gate_inputs = {
            "final_hidden": all_hidden[-1],
            "view_ids": self._visual_view_ids(qwen_inputs),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
        }
        return vl_embs_list, attention_mask, gate_inputs

    # ------------------------------------------------------------------
    # Visual evidence gate
    # ------------------------------------------------------------------
    def _visual_view_ids(self, qwen_inputs) -> torch.Tensor:
        """(B, L) long: index of the frame each visual token belongs to, -1 for non-visual tokens."""
        input_ids = qwen_inputs["input_ids"]
        is_visual = input_ids == self.qwen_vl_interface.model.config.image_token_id
        merge = int(self.qwen_vl_interface.processor.image_processor.merge_size)
        per_image = (qwen_inputs["image_grid_thw"].prod(-1) // merge**2).view(input_ids.shape[0], -1)  # (B, V)
        order = torch.cumsum(is_visual.long(), dim=1) - 1  # rank of each visual token within its sample
        bounds = torch.cumsum(per_image, dim=1)  # (B, V) exclusive upper bounds per frame
        view = torch.searchsorted(bounds, order.clamp(min=0), right=True)
        return torch.where(is_visual, view, torch.full_like(view, -1))

    def _gate_keep(self, gate_inputs: dict, attention_mask: torch.Tensor) -> torch.Tensor:
        """Keep values m (B, L) in [0, 1] for visual tokens, 1 elsewhere."""
        view_ids = gate_inputs["view_ids"]
        keep = torch.ones(view_ids.shape, dtype=torch.float32, device=view_ids.device)
        visual = view_ids >= 0
        budget = float(self.config.framework.eye_gate.budget)
        if self.gate_mode == "uniform":
            return torch.where(visual, torch.full_like(keep, budget), keep)
        if self.gate_mode == "random":
            return torch.where(visual, torch.bernoulli(torch.full_like(keep, budget)), keep)

        hidden = gate_inputs["final_hidden"]
        text = (~visual) & attention_mask
        tau = float(self.config.framework.eye_gate.get("temperature", 1.0))
        rows = []
        for b in range(hidden.shape[0]):
            q = hidden[b][text[b]].float().mean(0)
            row = keep[b].clone()
            for v, gate in enumerate(self.eye_gates):
                idx = view_ids[b] == v
                row[idx] = torch.sigmoid(gate(hidden[b][idx], q) / tau)
            rows.append(row)
        return torch.stack(rows)

    @staticmethod
    def _cross_attention_bias(attention_mask: torch.Tensor, keep: Optional[torch.Tensor], dtype) -> torch.Tensor:
        """Additive cross-attention bias: log m on visual tokens (0 when keep is None), -inf on padding."""
        bias = torch.zeros(attention_mask.shape, dtype=torch.float32, device=attention_mask.device)
        if keep is not None:
            bias = torch.log(keep.clamp(min=0.0))  # log 0 = -inf for fully dropped tokens (random mode)
        bias = bias.masked_fill(~attention_mask, float("-inf"))
        return bias.to(dtype)

    def _keep_stats(self, keep: torch.Tensor, view_ids: torch.Tensor) -> torch.Tensor:
        """(B, V) mean keep per frame."""
        means = []
        for v in range(self.num_views):
            idx = (view_ids == v).float()
            means.append((keep * idx).sum(1) / idx.sum(1).clamp(min=1.0))
        return torch.stack(means, dim=1)

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
        vl_embs_list, attention_mask, gate_inputs = self._encode_vl_hidden_states(batch_images, instructions)

        device, dtype = vl_embs_list[-1].device, self._head_dtype()
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        actions = torch.as_tensor(np.array([example["action"] for example in examples]), device=device, dtype=dtype)
        actions_target = actions[:, -self.action_horizon :, :]

        repeat = self.config.trainer.get("repeated_diffusion_steps", None) if self.config.get("trainer") else None
        repeat = int(repeat or self.config.framework.action_model.get("repeated_diffusion_steps", 1))

        actions_target = actions_target.repeat(repeat, 1, 1)
        vl_embs_list = [h.repeat(repeat, 1, 1) for h in vl_embs_list]
        state_t = None
        if state is not None:
            state_t = torch.as_tensor(state, device=device, dtype=dtype).repeat(repeat, 1, 1)

        if not self.gate_enabled:
            action_loss = self.action_model(
                vl_embs_list, actions_target, state_t, encoder_attention_mask=attention_mask.repeat(repeat, 1)
            )
            return {"action_loss": action_loss}

        # Gated and ungated paths share the VLM forward and the flow-matching sample (noise, t).
        gate_cfg = self.config.framework.eye_gate
        keep = self._gate_keep(gate_inputs, attention_mask)
        noise = torch.randn(actions_target.shape, device=device, dtype=dtype)
        t = self.action_model.sample_time(actions_target.shape[0], device=device, dtype=dtype)

        def flow_loss(keep_or_none):
            bias = self._cross_attention_bias(attention_mask, keep_or_none, dtype).repeat(repeat, 1)
            return self.action_model(
                vl_embs_list, actions_target, state_t, encoder_attention_mask=bias, noise=noise, t=t
            )

        loss_gated = flow_loss(keep)
        out = {"loss_gated": loss_gated.detach()}
        total = loss_gated
        keep_mean = self._keep_stats(keep, gate_inputs["view_ids"])  # (B, V)
        for v in range(self.num_views):
            out[f"keep_view{v}"] = keep_mean[:, v].mean().detach()
        if self.gate_mode == "learned":
            if float(gate_cfg.full_loss_weight) > 0:
                loss_full = flow_loss(None)
                total = total + float(gate_cfg.full_loss_weight) * loss_full
                out["loss_full"] = loss_full.detach()
            loss_budget = ((keep_mean - float(gate_cfg.budget)) ** 2).sum(1).mean()
            total = total + float(gate_cfg.budget_loss_weight) * loss_budget
            out["loss_budget"] = loss_budget.detach()
        out["action_loss"] = total
        return out

    def _predict(self, examples: List[dict]) -> dict:
        batch_images, instructions, state = self._prepare_inputs(examples)
        vl_embs_list, attention_mask, gate_inputs = self._encode_vl_hidden_states(batch_images, instructions)

        device, dtype = vl_embs_list[-1].device, self._head_dtype()
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        state_t = torch.as_tensor(state, device=device, dtype=dtype) if state is not None else None

        out = {}
        encoder_mask = attention_mask
        if self.gate_enabled:
            keep = self._gate_keep(gate_inputs, attention_mask)
            encoder_mask = self._cross_attention_bias(attention_mask, keep, dtype)
            out["keep_maps"] = self._keep_maps(keep, gate_inputs)
        pred_actions = self.action_model.predict_action(vl_embs_list, state_t, encoder_attention_mask=encoder_mask)
        if self.action_clip is not None:
            pred_actions = pred_actions.clamp(-float(self.action_clip), float(self.action_clip))
        out["normalized_actions"] = pred_actions.float().cpu().numpy()
        return out

    def _keep_maps(self, keep: torch.Tensor, gate_inputs: dict) -> List[List[np.ndarray]]:
        """Per sample, per frame keep map of shape (h, w) in merged-token units (for visualisation)."""
        merge = int(self.qwen_vl_interface.processor.image_processor.merge_size)
        grids = gate_inputs["image_grid_thw"].view(keep.shape[0], -1, 3)
        maps = []
        for b in range(keep.shape[0]):
            frames = []
            for v in range(grids.shape[1]):
                _, h, w = (int(x) for x in grids[b, v])
                values = keep[b][gate_inputs["view_ids"][b] == v]
                frames.append(values.float().cpu().numpy().reshape(h // merge, w // merge))
            maps.append(frames)
        return maps

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        """Returns ``normalized_actions`` (B, action_horizon, action_dim) in [-action_clip, action_clip]."""
        if type(examples) is not list:
            examples = [examples]
        return self._predict(examples)
