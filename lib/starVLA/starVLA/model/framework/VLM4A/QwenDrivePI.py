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
  sampled actions are not clamped (turns and fast driving exceed the 1st/99th percentiles).
- Flow matching: Beta(1.5, 1) time sampling with s=0.999, MSE on velocity, 10 Euler steps at inference.

Optional visual evidence gate (``framework.eye_gate``, EyePolicy / See Less style)
─────────────────────────────────────────────────────────────────────────────────
A gate MLP (shared by the frames, which all come from the front camera) reads the VLM final-layer hidden
of each visual token ``h_p`` and the mean of the
text-token hidden states ``q`` and predicts ``m_p = σ(MLP([h_p, q, h_p ⊙ q]) / τ)``. ``log m_p`` is added to the
action expert's cross-attention logits on that visual token in every cross-attention block (the VLM itself is
unchanged; text tokens stay visible). Loss: ``L_gated + λ_full·L_full + λ_budget·Σ_v (mean_p m_p − ρ)² − λ_MI·Σ_v I_v
+ β·mean_p ℓ_p²`` with ``L_full`` the same flow-matching sample without the gate (each term optional).
Inference uses the gated path.

``I_v = mean_p [H(E_x m_p) − E_x H(m_p)]`` (binary entropy H, expectation over the global batch) is the mutual
information between the observation and a keep decision b_p ~ Bernoulli(m_p). It is 0 for any mask that does not
depend on the input (uniform, or a fixed spatial pattern), which ``log m`` biases leave free under the softmax.
The gate output layer is randomly initialised: at an input-independent mask the gradient of I is exactly 0.

Optional decision anchors (``framework.anchor``, AnchorVLA DAAR + DARF, modified)
──────────────────────────────────────────────────────────────────────────────────
A k-means codebook of K trajectory-pattern anchors (``prepare_paiav.py --num_anchors``). A learned query token is
appended to the prompt; an MLP on its VLM output predicts K anchor logits, trained with AnchorVLA's soft target
over the nearest anchors (Eq. 3-6). The reference trajectory is the softmax-weighted mean of the top-N anchors
(``reference: soft``; differentiable, so the flow loss also reaches the logits) or the top-1 anchor
(``reference: hard``). The DiT generates the residual ``(τ − reference) / σ_r`` by flow matching (DARF); the
reference enters every DiT block as a per-step query bias. Stage 1 trains only the classifier path
(prediction = reference); stage 2 adds the residual flow loss. No top-M candidates / confidence branch.
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
from starVLA.model.modules.action_space.unicycle import unicycle_rollout
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.action_model.attention_probe import CrossAttentionProbe
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
            # Optional clamp of sampled (normalised) actions to [-action_clip, action_clip]. Off: q01/q99 are
            # pooled over all waypoints, so a clamp at 1 cuts lateral offsets beyond ~3 m and forward beyond ~69 m.
            "action_clip": None,
            # VLM input order [text, images] instead of [images, text]: every visual token is computed with the
            # instruction (speed / target point) in context, and the text tokens do not see the images.
            "text_first": False,
            # Cross-attention of the action expert reads the visual tokens only (text / template tokens masked), so
            # the expert gets the scene and the instruction only through the (gated) visual tokens.
            "dit_visual_only": False,
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
            # Initial σ(ℓ) (bias of the gate output). Under log m attention biases a uniform m is, up to the
            # text tokens, the ungated model for any value, so start at the budget where σ' is larger.
            "init_keep": 0.2,
            "budget": 0.2,  # ρ, target mean of m per frame
            "budget_loss_weight": 0.1,
            "full_loss_weight": 1.0,  # weight of the ungated loss; 0 skips it
            "mi_loss_weight": 0.0,  # λ_MI, weight of −I(m; x)
            "logit_l2_weight": 0.0,  # β, weight of mean ℓ² over the visual tokens' gate logits
            # Per-channel standardisation of the gate inputs (removes the VLM's massive-activation channels).
            "standardize_inputs": True,
            # "sigmoid": m = σ(ℓ/τ) (EyePolicy). "budgeted": m = σ(b + contrast·z), z = ℓ standardised per frame and b
            # set so that mean m = budget (see budgeted_keep); use with budget_loss_weight 0.
            "parameterization": "sigmoid",
            "contrast": 2.0,
            # Loss of the gated path. "flow": flow matching against the ground truth. "distill": ‖v_gated − sg(v_full)‖²
            # with the same noise and t, so the part both paths solve from the ego state cancels.
            "gated_loss": "flow",
            # What the gated-path loss updates. "all": VLM, DiT and gate. "gate_dit": DiT and gate (VLM features and
            # the gate input detached). "gate": the gate only (DiT gradients of this loss are dropped by the trainer,
            # see ``gate_loss`` in the output).
            "gated_grad": "all",
            # Centre the gate's visual input per token position (h_p − μ_p, EMA over the data): a mask that depends
            # only on the position (e.g. row stripes) is then impossible. Needs a fixed token grid.
            "position_center": False,
            # Gate query q: "text" (mean of the text-token hidden) or "target" (MLP embedding of the navigation target
            # point (x, y) / target_scale, read from the sample's meta["target_xy"]).
            "query": "text",
            "target_scale": 25.0,
            # "standard": losses above. "textdrop": per step (all from one VLM forward)
            #   ② ungated, all tokens (text visible)                -> teacher v_ref (no gradient)
            #   ③ gated (m), text tokens hidden from the DiT        -> ‖v③ − v_ref‖², updates the gate only
            #   ④ gated (m fixed), all tokens                       -> flow matching, updates VLM + DiT
            # Inference uses ④ (gated, all tokens).
            # "vlm_keep": the mask acts inside the VLM (see modules/vlm/visual_keep.py) from the image embeddings
            #   (gate input: pre-LLM image-token embeddings + query) and on the DiT cross-attention. Per step, from one
            #   vision-encoder pass:
            #   A-full   VLM + DiT ungated                                   -> L_full     (VLM, DiT)
            #   A-gated  keep = m in the VLM and log m on the DiT            -> L_gated    (gate, VLM, DiT)
            #   B        tokens with m < tau removed (keep 0), others 1      -> L_removed  (VLM, DiT); inference path
            #   tau ramps linearly from 0 to removal_threshold over removal_warmup_steps.
            "scheme": "standard",
            "removal_threshold": 0.1,
            "removal_warmup_steps": 2000,
            "speed_scale": 20.0,  # query "target_speed": [x / target_scale, y / target_scale, v0 / speed_scale]
            "position_slots": 2048,  # buffer size for μ_p (>= visual tokens per sample)
        }
    )

    # === Decision anchors (AnchorVLA-style, see module docstring) ===
    anchor: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "path": None,  # anchors.npz from prepare_paiav.py --num_anchors
            "stage": 2,  # 1: anchor classifier only; 2: + residual flow
            "query_token": "<|quad_start|>",  # unused special token whose embedding is replaced by a learned query
            "classifier_hidden": 1024,
            "target_top_n": 8,  # soft target over the nearest anchors (AnchorVLA Eq. 4)
            "target_gamma": 1.0,  # metres
            "reference": "soft",  # soft: softmax-weighted mean of top-N anchors; hard: top-1 anchor
            "reference_top_n": 6,
            "reference_temperature": 1.0,
            "cls_loss_weight": 1.0,
            "flow_loss_weight": 1.0,
            "flow_loss": "l1",  # AnchorVLA Eq. 11
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


class ChannelStandardizer(nn.Module):
    """Per-channel (x − μ_c) / σ_c with running statistics (no affine, no gradient through the statistics).

    The VLM's final hidden has a few near-constant "massive activation" channels (e.g. channel 0 ≈ 45× the RMS,
    shared by visual and text tokens). Under RMSNorm alone they dominate the gate input (and h⊙q squares them),
    so the gate output is input-independent. Standardising each channel removes their constant part and puts the
    token-varying part of every channel on the same scale. The statistics are an EMA updated in training and used in
    both training and eval; before the first update the current input's statistics are used.
    """

    def __init__(self, width: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, W) tokens."""
        x = x.float()
        with torch.no_grad():
            mean, var = x.mean(0), x.var(0, unbiased=False)
            if self.training:
                if bool(self.initialized):
                    self.mean.lerp_(mean, self.momentum)
                    self.var.lerp_(var, self.momentum)
                else:
                    self.mean.copy_(mean)
                    self.var.copy_(var)
                    self.initialized.fill_(True)
            if bool(self.initialized):
                mean, var = self.mean, self.var
        return (x - mean) / torch.sqrt(var + self.eps)


class ViewGate(nn.Module):
    """Gate head ``ℓ_p = W2·swish(W1·[h_p, q, h_p⊙q])`` over RMS-normalised hidden states (EyePolicy's
    ``ViewGate``). EyePolicy uses one per camera; here all frames come from the same camera and share one.

    With ``standardize`` the visual-token hidden h and each text-token hidden are first standardised per channel
    (``ChannelStandardizer``, separate statistics for visual and text tokens); q is the mean of the standardised
    text tokens."""

    def __init__(self, width: int, hidden_dim: int, init_keep: float, standardize: bool = True,
                 position_slots: int = 0, momentum: float = 0.01, target_query: int = 0):
        super().__init__()
        # target_query: q is an embedding of the (scaled) navigation target point (x, y) instead of the mean of the
        # text-token hidden states.
        self.target_embed = (
            nn.Sequential(nn.Linear(int(target_query), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, width))
            if target_query else None
        )
        # Per-position centring (position_slots > 0): h_p − μ_p with μ_p an EMA of the hidden at visual-token position
        # p over the data. The token grid is fixed, so this removes everything shared by all samples at a position
        # (row / sequence-position components); the gate only sees how the content deviates from the average.
        self.position_slots, self.momentum = int(position_slots), momentum
        if self.position_slots > 0:
            self.register_buffer("pos_mean", torch.zeros(self.position_slots, width))
            self.register_buffer("pos_count", torch.zeros((), dtype=torch.long))  # visual tokens per sample (0: unset)
            self.register_buffer("pos_initialized", torch.zeros((), dtype=torch.bool))
        self.std_h = ChannelStandardizer(width) if standardize else None
        self.std_t = ChannelStandardizer(width) if standardize else None
        self.norm_h = nn.RMSNorm(width)
        self.norm_q = nn.RMSNorm(width)
        self.fc1 = nn.Linear(3 * width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # fc2.weight keeps its default random init so that the initial mask depends on the input.
        nn.init.constant_(self.fc2.bias, math.log(init_keep / (1 - init_keep)))

    def center_positions(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, W) visual-token hidden of B samples in sequence order -> h − μ (same shape, float32)."""
        h = h.float()
        n = h.shape[1]
        if n > self.position_slots:
            raise ValueError(f"{n} visual tokens per sample > eye_gate.position_slots {self.position_slots}")
        with torch.no_grad():
            if int(self.pos_count) == 0:
                self.pos_count.fill_(n)
            elif int(self.pos_count) != n:
                raise ValueError(f"per-position centring needs a fixed token grid: {n} != {int(self.pos_count)} tokens")
            if self.training:
                batch_mean = h.mean(0)
                if bool(self.pos_initialized):
                    self.pos_mean[:n].lerp_(batch_mean, self.momentum)
                else:
                    self.pos_mean[:n].copy_(batch_mean)
                    self.pos_initialized.fill_(True)
            mean = self.pos_mean[:n] if bool(self.pos_initialized) else h.mean(0)
        return h - mean[None]

    def forward(self, h: torch.Tensor, text: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h: (N, W) visual-token hidden of one sample, text: (T, W) text-token hidden, target: (2,) scaled target
        point (with target_query) -> logits (N,)."""
        dtype = self.fc1.weight.dtype
        if self.target_embed is not None:
            h = self.std_h(h) if self.std_h is not None else h
            q = self.target_embed(target.to(self.target_embed[0].weight.dtype)).float()
        elif self.std_h is not None:
            h, q = self.std_h(h), self.std_t(text).mean(0)
        else:
            q = text.float().mean(0)
        h = self.norm_h(h.to(dtype))
        q = self.norm_q(q.to(dtype))[None].expand_as(h)
        return self.fc2(F.silu(self.fc1(torch.cat([h, q, h * q], dim=-1)))).squeeze(-1).float()


def budgeted_keep(scores: torch.Tensor, budget: float, contrast: float, iters: int = 40) -> torch.Tensor:
    """Soft keep values of one frame with the budget and the contrast fixed by construction.

    ``m_p = σ(b + α·z_p)`` with ``z`` the scores standardised over the frame's tokens, ``α = contrast`` and ``b``
    found by bisection so that ``mean_p m_p = budget``. The gate only decides which tokens are kept relatively more:
    a uniform mask is impossible (z is standardised) and the spread of m is set by α, so the mask cannot binarise
    and no budget loss is needed.

    Bisection itself is not differentiated. ``b(z)`` is defined implicitly by ``Σ_p σ(b + α z_p) = N·budget``, so
    ``∂b/∂z_q = −α w_q`` with ``w = σ' / Σσ'``; adding ``−α Σ_q w_q (z_q − sg[z_q])`` (value 0) to the bisection
    result gives exactly this gradient.
    """
    z = (scores - scores.mean()) / (scores.std() + 1e-6)
    with torch.no_grad():
        lo = torch.full((), -30.0, device=z.device, dtype=z.dtype)
        hi = torch.full((), 30.0, device=z.device, dtype=z.dtype)
        for _ in range(iters):
            mid = (lo + hi) / 2
            above = torch.sigmoid(mid + contrast * z).mean() > budget
            lo, hi = torch.where(above, lo, mid), torch.where(above, mid, hi)
        bias0 = (lo + hi) / 2
        slope = torch.sigmoid(bias0 + contrast * z)
        slope = slope * (1 - slope)
        w = slope / slope.sum().clamp(min=1e-12)
    bias = bias0 - contrast * (w * (z - z.detach())).sum()
    return torch.sigmoid(bias + contrast * z)


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
            self.eye_gate = ViewGate(
                llm_hidden_size, int(gate_cfg.hidden_dim), float(gate_cfg.init_keep),
                standardize=bool(gate_cfg.get("standardize_inputs", True)),
                position_slots=int(gate_cfg.get("position_slots", 2048)) if bool(gate_cfg.get("position_center", False)) else 0,
                target_query={"text": 0, "target": 2, "target_speed": 3}[str(gate_cfg.get("query", "text"))],
            )

        anchor_cfg = self.config.framework.anchor
        self.anchor_enabled = bool(anchor_cfg.get("enabled", False))
        if self.anchor_enabled:
            self._init_anchor(anchor_cfg, llm_hidden_size)

        drive_cfg = self.config.framework.drive
        self.prompt = str(drive_cfg.get("prompt", "predict trajectory"))
        self.current_image_pixels = int(drive_cfg.get("current_image_pixels", 921600))
        self.history_image_pixels = int(drive_cfg.get("history_image_pixels", 174080))
        self.image_factor = int(drive_cfg.get("image_factor", 32))
        self.action_clip = drive_cfg.get("action_clip", None)
        self.vlm_keep = self.gate_enabled and str(gate_cfg.get("scheme", "standard")) == "vlm_keep"
        if self.vlm_keep:
            from starVLA.model.modules.vlm.visual_keep import attach

            attach(self.qwen_vl_interface.model)
            self.register_buffer("keep_step", torch.zeros((), dtype=torch.long))  # training forwards (tau ramp)
        self.text_first = bool(drive_cfg.get("text_first", False))
        self.dit_visual_only = bool(drive_cfg.get("dit_visual_only", False))
        if self.gate_enabled:
            if str(gate_cfg.get("gated_loss", "flow")) not in ("flow", "distill"):
                raise ValueError(f"eye_gate.gated_loss must be flow / distill, got {gate_cfg.gated_loss}")
            if str(gate_cfg.get("gated_grad", "all")) not in ("all", "gate_dit", "gate"):
                raise ValueError(f"eye_gate.gated_grad must be all / gate_dit / gate, got {gate_cfg.gated_grad}")

    # ------------------------------------------------------------------
    # Decision anchors
    # ------------------------------------------------------------------
    def _init_anchor(self, anchor_cfg, llm_hidden_size: int) -> None:
        data = np.load(anchor_cfg.path)
        # "waypoints": anchors are trajectories (K, H, 2) in metres (q01/q99 action normalisation).
        # "control": anchors are [accel, curvature] sequences (K, H, 2) (mean/std action normalisation); distances
        # to the ground truth are measured in metres after integrating from each sample's current speed v0.
        self.anchor_space = str(data["space"]) if "space" in data else "waypoints"
        anchors = torch.as_tensor(data["anchors"], dtype=torch.float32)
        if anchors.shape[1] != self.action_horizon:
            raise ValueError(f"anchors have horizon {anchors.shape[1]}, action_horizon is {self.action_horizon}")
        self.register_buffer("anchors", anchors, persistent=False)
        self.register_buffer("residual_std", torch.as_tensor(data["residual_std"], dtype=torch.float32), persistent=False)
        if self.anchor_space == "control":
            self.anchor_dt = float(data["dt"])
            self.register_buffer("action_mean", torch.as_tensor(data["action_mean"], dtype=torch.float32), persistent=False)
            self.register_buffer("action_std", torch.as_tensor(data["action_std"], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("action_q01", torch.as_tensor(data["action_q01"], dtype=torch.float32), persistent=False)
            self.register_buffer("action_q99", torch.as_tensor(data["action_q99"], dtype=torch.float32), persistent=False)
        self.anchor_stage = int(anchor_cfg.stage)
        if anchor_cfg.reference not in ("soft", "hard"):
            raise ValueError(f"anchor.reference must be soft / hard, got {anchor_cfg.reference}")

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self.anchor_query_text = str(anchor_cfg.query_token)
        self.anchor_query_id = int(tokenizer.convert_tokens_to_ids(self.anchor_query_text))
        embed = self.qwen_vl_interface.model.model.language_model.embed_tokens
        self.anchor_query = nn.Parameter(embed.weight[self.anchor_query_id].detach().clone().float())
        embed.register_forward_hook(self._replace_query_embedding)

        hidden = int(anchor_cfg.classifier_hidden)
        self.anchor_classifier = nn.Sequential(
            nn.LayerNorm(llm_hidden_size), nn.Linear(llm_hidden_size, hidden), nn.GELU(), nn.Linear(hidden, len(anchors))
        )
        # per-step embedding of the (normalised) reference trajectory -> query bias of the action tokens
        self.anchor_embed = nn.Sequential(
            nn.Linear(2, self.action_dit_hidden_dim), nn.SiLU(), nn.Linear(self.action_dit_hidden_dim, self.action_dit_hidden_dim)
        )

    def _replace_query_embedding(self, module, inputs, output):
        """Forward hook on the token embedding: the query token gets the learned query vector."""
        is_query = inputs[0] == self.anchor_query_id
        if not is_query.any():
            return output
        output = output.clone()
        output[is_query] = self.anchor_query.to(output.dtype)
        return output

    def _normalize_actions(self, a: torch.Tensor) -> torch.Tensor:
        if self.anchor_space == "control":
            return (a - self.action_mean) / self.action_std.clamp(min=1e-6)
        return 2.0 * (a - self.action_q01) / (self.action_q99 - self.action_q01).clamp(min=1e-6) - 1.0

    def _unnormalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        if self.anchor_space == "control":
            return x * self.action_std + self.action_mean
        return (x + 1.0) / 2.0 * (self.action_q99 - self.action_q01) + self.action_q01

    def _to_metres(self, a: torch.Tensor, v0: Optional[torch.Tensor]) -> torch.Tensor:
        """Raw actions (B, [K,] H, 2) -> positions in metres (controls are integrated from v0 (B,))."""
        if self.anchor_space != "control":
            return a
        v0 = v0.to(a.device, a.dtype)
        if a.dim() == 4:
            v0 = v0[:, None].expand(-1, a.shape[1])
        return unicycle_rollout(a, v0, self.anchor_dt)

    def _gate_target(self, examples: List[dict], device) -> Optional[torch.Tensor]:
        """(B, 2) navigation target point / target_scale for the gate query (eye_gate.query = target)."""
        gate_cfg = self.config.framework.eye_gate
        if not (self.gate_enabled and str(gate_cfg.get("query", "text")) in ("target", "target_speed")):
            return None
        if "target_xy" in examples[0]:
            xy = [ex["target_xy"] for ex in examples]
        else:
            xy = [ex["meta"]["target_xy"] for ex in examples]
        xy = torch.as_tensor(np.asarray(xy, dtype=np.float32), device=device) / float(gate_cfg.get("target_scale", 25.0))
        if str(gate_cfg.get("query", "text")) == "target_speed":
            v0 = self._current_speed(examples, device)[:, None] / float(gate_cfg.get("speed_scale", 20.0))
            xy = torch.cat([xy, v0], dim=1)
        return xy

    @staticmethod
    def _current_speed(examples: List[dict], device) -> Optional[torch.Tensor]:
        """Current speed v0 (B,) in m/s (``v0``, or ``meta["v0"]``), needed by control-space anchors."""
        if "v0" in examples[0]:
            speeds = [ex["v0"] for ex in examples]
        elif "v0" in examples[0].get("meta", {}):
            speeds = [ex["meta"]["v0"] for ex in examples]
        else:
            return None
        return torch.as_tensor(np.asarray(speeds, dtype=np.float32), device=device)

    def _anchor_reference(self, logits: torch.Tensor):
        """Reference trajectory (B, H, 2) in metres, top-N anchor ids and weights."""
        cfg = self.config.framework.anchor
        if cfg.reference == "hard":
            ids = logits.argmax(-1, keepdim=True)
            weights = torch.ones_like(ids, dtype=torch.float32)
            return self.anchors[ids[:, 0]], ids, weights
        top_logits, ids = logits.topk(int(cfg.reference_top_n), dim=-1)
        weights = torch.softmax(top_logits.float() / float(cfg.reference_temperature), dim=-1)
        return (weights[..., None, None] * self.anchors[ids]).sum(1), ids, weights

    def _anchor_distances(self, gt: torch.Tensor, v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """AnchorVLA Eq. 3: mean point-wise L2 distance in metres (B, K) between the ground truth and every anchor
        (control anchors are integrated from each sample's v0, like the ground-truth controls)."""
        anchors = self.anchors[None].expand(gt.shape[0], -1, -1, -1)
        return torch.linalg.norm(self._to_metres(gt, v0)[:, None] - self._to_metres(anchors, v0), dim=-1).mean(-1)

    def _anchor_losses(self, logits: torch.Tensor, gt_m: torch.Tensor, v0: Optional[torch.Tensor] = None) -> dict:
        cfg = self.config.framework.anchor
        dist = self._anchor_distances(gt_m, v0)
        near_d, near_ids = (-dist).topk(int(cfg.target_top_n), dim=-1)
        target = torch.zeros_like(dist).scatter_(1, near_ids, torch.softmax(near_d / float(cfg.target_gamma), dim=-1))
        log_p = torch.log_softmax(logits.float(), dim=-1)
        nearest = dist.argmin(-1)
        top_ref = logits.topk(int(cfg.reference_top_n), dim=-1).indices
        return {
            "loss_anchor_cls": -(target * log_p).sum(-1).mean(),
            "anchor_top1_acc": (logits.argmax(-1) == nearest).float().mean().detach(),
            "anchor_topn_recall": (top_ref == nearest[:, None]).any(-1).float().mean().detach(),
        }

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
        if self.anchor_enabled:
            instructions = [f"{text}{self.anchor_query_text}" for text in instructions]
        state = None
        if self.action_model.state_encoder is not None and "state" in examples[0]:
            state = np.stack([np.asarray(example["state"], dtype=np.float32) for example in examples])
            state = state.reshape(len(examples), -1, state.shape[-1])  # (B, T_state, state_dim)
        return batch_images, instructions, state

    def _head_dtype(self) -> torch.dtype:
        return next(self.action_model.parameters()).dtype

    def _project_hidden(self, all_hidden) -> list:
        """VLM hidden states (all layers) -> one projected entry per DiT block (self-attention blocks alias the
        preceding cross-attention entry)."""
        hidden_states = all_hidden[-self.num_action_dit_layers :]
        proj_dtype = next(self.project_layers.parameters(), torch.empty(0, dtype=self._head_dtype())).dtype
        projected = {
            i: proj(hidden_states[j].to(proj_dtype))
            for i, j, proj in zip(self.cross_layer_ids, self.vlm_layer_ids, self.project_layers)
        }
        vl_embs_list = []
        for i in range(self.num_action_dit_layers):
            vl_embs_list.append(projected[i] if i in projected else vl_embs_list[-1])
        return vl_embs_list

    def _vlm_keep_prepare(self, batch_images: List, instructions: List[str], examples: List[dict]) -> dict:
        """scheme vlm_keep: one vision-encoder pass, the gate on the pre-LLM image embeddings, and a function that runs
        the language model with given per-token keep weights."""
        qi = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        vm = self.qwen_vl_interface.model.model
        input_ids = qi["input_ids"]
        emb = vm.get_input_embeddings()(input_ids)
        image = vm.get_image_features(qi["pixel_values"], qi["image_grid_thw"], return_dict=True).pooler_output
        image = torch.cat(image, dim=0).to(emb.device, emb.dtype)
        image_mask, _ = vm.get_placeholder_mask(input_ids, inputs_embeds=emb, image_features=image)
        emb = emb.masked_scatter(image_mask, image)
        attn = qi.get("attention_mask")
        attn = torch.ones(input_ids.shape, dtype=torch.long, device=emb.device) if attn is None else attn
        position_ids = vm.compute_3d_position_ids(
            input_ids=input_ids, inputs_embeds=emb, image_grid_thw=qi["image_grid_thw"], attention_mask=attn,
            mm_token_type_ids=qi.get("mm_token_type_ids"),
        )
        view_ids = self._visual_view_ids(qi)
        visual = view_ids >= 0
        gate_inputs = {
            "final_hidden": emb,  # the gate reads the image-token embeddings before the language model
            "view_ids": view_ids,
            "image_grid_thw": qi["image_grid_thw"],
            "text_mask": (~visual) & attn.bool(),
        }
        target = self._gate_target(examples, emb.device)
        if target is not None:
            gate_inputs["target"] = target

        def run(keep):
            out = vm.language_model(
                inputs_embeds=emb, position_ids=position_ids, attention_mask=attn, use_cache=False,
                output_hidden_states=True, return_dict=True, visual_keep=keep,
            )
            return self._project_hidden(out.hidden_states)

        return {"run": run, "gate_inputs": gate_inputs, "attention_mask": attn.bool(), "visual": visual}

    def _removal_tau(self) -> float:
        gate_cfg = self.config.framework.eye_gate
        tau = float(gate_cfg.get("removal_threshold", 0.1))
        warm = int(gate_cfg.get("removal_warmup_steps", 0))
        if self.training and warm > 0:
            return tau * min(1.0, float(self.keep_step) / warm)
        return tau

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> tuple:
        """Run the VLM, keep the last ``num_action_dit_layers`` hidden states and project them to the expert width.

        The returned list has one entry per DiT block; entries for self-attention blocks are placeholders
        (the DiT ignores them) that alias the preceding cross-attention entry.
        """
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, text_first=self.text_first
        )
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
        vl_embs_list = self._project_hidden(all_hidden)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        else:
            attention_mask = torch.ones(qwen_inputs["input_ids"].shape, dtype=torch.bool, device=all_hidden[-1].device)
        view_ids = self._visual_view_ids(qwen_inputs)
        visual = view_ids >= 0
        text_mask = (~visual) & attention_mask
        if self.text_first:
            # q: the prompt only. Template tokens after the images have seen the images.
            pos = torch.arange(visual.shape[1], device=visual.device)[None]
            first_visual = torch.where(visual, pos, visual.shape[1]).min(1, keepdim=True).values
            text_mask = text_mask & (pos < first_visual)
        gate_inputs = {
            "final_hidden": all_hidden[-1],
            "view_ids": view_ids,
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "text_mask": text_mask,
        }
        if self.dit_visual_only:
            attention_mask = attention_mask & visual  # the expert's cross-attention sees the visual tokens only
        if self.anchor_enabled:
            is_query = qwen_inputs["input_ids"] == self.anchor_query_id
            if not bool((is_query.sum(1) == 1).all()):
                raise RuntimeError("expected exactly one anchor query token per sample")
            gate_inputs["query_hidden"] = all_hidden[-1][is_query]  # (B, W)
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

    def _gate_keep(self, gate_inputs: dict, attention_mask: torch.Tensor, return_logits: bool = False):
        """Keep values m (B, L) in [0, 1] for visual tokens, 1 elsewhere.

        With ``return_logits`` (learned mode) also returns the pre-sigmoid gate logits ℓ of the visual tokens
        (concatenated over the batch)."""
        view_ids = gate_inputs["view_ids"]
        keep = torch.ones(view_ids.shape, dtype=torch.float32, device=view_ids.device)
        visual = view_ids >= 0
        budget = float(self.config.framework.eye_gate.budget)
        if self.gate_mode == "uniform":
            keep = torch.where(visual, torch.full_like(keep, budget), keep)
            return (keep, None) if return_logits else keep
        if self.gate_mode == "random":
            keep = torch.where(visual, torch.bernoulli(torch.full_like(keep, budget)), keep)
            return (keep, None) if return_logits else keep

        hidden = gate_inputs["final_hidden"]
        text = gate_inputs["text_mask"] if "text_mask" in gate_inputs else (~visual) & attention_mask
        gate_cfg = self.config.framework.eye_gate
        tau = float(gate_cfg.get("temperature", 1.0))
        budgeted = str(gate_cfg.get("parameterization", "sigmoid")) == "budgeted"
        rows, logits = [], []
        vis_hidden = [hidden[b][visual[b]] for b in range(hidden.shape[0])]
        if self.eye_gate.position_slots > 0:
            if len({v.shape[0] for v in vis_hidden}) != 1:
                raise ValueError("per-position centring needs the same number of visual tokens in every sample")
            vis_hidden = list(self.eye_gate.center_positions(torch.stack(vis_hidden)))
        for b in range(hidden.shape[0]):
            row = keep[b].clone()
            target = gate_inputs["target"][b] if "target" in gate_inputs else None
            logit = self.eye_gate(vis_hidden[b], hidden[b][text[b]], target)
            if budgeted:
                frames = view_ids[b][visual[b]]
                m = torch.empty_like(logit)
                for v in frames.unique():
                    sel = frames == v
                    m[sel] = budgeted_keep(logit[sel], budget, float(gate_cfg.get("contrast", 2.0)))
                row[visual[b]] = m
            else:
                row[visual[b]] = torch.sigmoid(logit / tau)
            rows.append(row)
            logits.append(logit)
        keep = torch.stack(rows)
        return (keep, torch.cat(logits)) if return_logits else keep

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

    def _mask_mutual_information(self, keep: torch.Tensor, view_ids: torch.Tensor) -> tuple:
        """Σ_v I_v with I_v = mean_p [H(m̄_p) − mean_x H(m_p(x))] over the global batch.

        Every frame of a view has the same token grid (fixed camera resolution), so token p is the same image
        location in every sample. Returns (I, marginal entropy, conditional entropy), each summed over views.
        """
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world = torch.distributed.get_world_size() if distributed else 1

        def entropy(p):
            p = p.clamp(1e-6, 1 - 1e-6)
            return -(p * p.log() + (1 - p) * (1 - p).log())

        mi, h_marg, h_cond = 0.0, 0.0, 0.0
        for v in range(self.num_views):
            m = torch.stack([keep[b][view_ids[b] == v] for b in range(keep.shape[0])])  # (B, P)
            local_sum = m.sum(0)
            total = local_sum.detach().clone()
            n = torch.tensor([float(m.shape[0])], device=m.device)
            if distributed:
                sizes = torch.tensor([m.shape[1], -m.shape[1]], device=m.device)
                torch.distributed.all_reduce(sizes, op=torch.distributed.ReduceOp.MAX)
                if int(sizes[0]) != -int(sizes[1]):
                    raise ValueError(f"view {v}: token grids differ across ranks; the mask MI needs a fixed grid")
                torch.distributed.all_reduce(total)
                torch.distributed.all_reduce(n)
            # Value: the global mean. Gradient: scaled by the world size, because the gradient reduction averages
            # over ranks while the samples of every rank enter the marginal.
            marginal = (total + world * (local_sum - local_sum.detach())) / n
            hm = entropy(marginal).mean()
            hc = entropy(m).mean()
            mi, h_marg, h_cond = mi + hm - hc, h_marg + hm.detach(), h_cond + hc.detach()
        return mi, h_marg, h_cond

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
        if self.vlm_keep:
            return self._vlm_keep_losses(examples, batch_images, instructions, state)
        vl_embs_list, attention_mask, gate_inputs = self._encode_vl_hidden_states(batch_images, instructions)

        device, dtype = vl_embs_list[-1].device, self._head_dtype()
        gate_target = self._gate_target(examples, device)
        if gate_target is not None:
            gate_inputs["target"] = gate_target
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        actions = torch.as_tensor(np.array([example["action"] for example in examples]), device=device, dtype=dtype)
        actions_target = actions[:, -self.action_horizon :, :]

        out, total = {}, 0.0
        flow_target, query_bias, flow_weight, loss_type = actions_target, None, 1.0, "mse"
        if self.anchor_enabled:
            anchor_cfg = self.config.framework.anchor
            logits = self.anchor_classifier(gate_inputs["query_hidden"].to(self.anchor_classifier[0].weight.dtype))
            gt_m = self._unnormalize_actions(actions_target.float())
            v0 = self._current_speed(examples, device)
            anchor_out = self._anchor_losses(logits, gt_m, v0)
            total = total + float(anchor_cfg.cls_loss_weight) * anchor_out["loss_anchor_cls"]
            out.update({k: v.detach() for k, v in anchor_out.items()})
            if self.anchor_stage == 1:
                out["action_loss"] = total
                return out
            reference, _, weights = self._anchor_reference(logits)
            out["anchor_weight_top1"] = weights[:, 0].mean().detach()
            out["anchor_weight_entropy"] = -(weights * weights.clamp(min=1e-12).log()).sum(-1).mean().detach()
            flow_target = ((gt_m - reference) / self.residual_std).to(dtype)
            query_bias = self.anchor_embed(self._normalize_actions(reference).to(self.anchor_embed[0].weight.dtype))
            flow_weight, loss_type = float(anchor_cfg.flow_loss_weight), str(anchor_cfg.flow_loss)

        repeat = self.config.trainer.get("repeated_diffusion_steps", None) if self.config.get("trainer") else None
        repeat = int(repeat or self.config.framework.action_model.get("repeated_diffusion_steps", 1))

        flow_target = flow_target.repeat(repeat, 1, 1)
        query_bias = query_bias.repeat(repeat, 1, 1) if query_bias is not None else None
        vl_embs_list = [h.repeat(repeat, 1, 1) for h in vl_embs_list]
        state_t = None
        if state is not None:
            state_t = torch.as_tensor(state, device=device, dtype=dtype).repeat(repeat, 1, 1)

        if not self.gate_enabled:
            flow = self.action_model(
                vl_embs_list, flow_target, state_t, encoder_attention_mask=attention_mask.repeat(repeat, 1),
                query_bias=query_bias, loss_type=loss_type,
            )
            if self.anchor_enabled:
                out["loss_flow"] = flow.detach()
            out["action_loss"] = total + flow_weight * flow
            return out

        # Gated and ungated paths share the VLM forward and the flow-matching sample (noise, t).
        gate_cfg = self.config.framework.eye_gate
        if self.gate_mode == "learned" and str(gate_cfg.get("scheme", "standard")) == "textdrop":
            return self._textdrop_losses(out, total, gate_inputs, attention_mask, vl_embs_list, flow_target, state_t,
                                         query_bias, flow_weight, loss_type, repeat, device, dtype)
        distill = self.gate_mode == "learned" and str(gate_cfg.get("gated_loss", "flow")) == "distill"
        gated_grad = str(gate_cfg.get("gated_grad", "all")) if self.gate_mode == "learned" else "all"
        # Separated gradients: the gated path sees the VLM features and the gate input detached.
        gated_embs, gate_in = vl_embs_list, gate_inputs
        if gated_grad != "all":
            gated_embs = [h.detach() for h in vl_embs_list]
            gate_in = dict(gate_inputs, final_hidden=gate_inputs["final_hidden"].detach())
        keep, gate_logits = self._gate_keep(gate_in, attention_mask, return_logits=True)
        noise = torch.randn(flow_target.shape, device=device, dtype=dtype)
        t = self.action_model.sample_time(flow_target.shape[0], device=device, dtype=dtype)

        def flow_loss(keep_or_none, embs):
            bias = self._cross_attention_bias(attention_mask, keep_or_none, dtype).repeat(repeat, 1)
            return self.action_model(
                embs, flow_target, state_t, encoder_attention_mask=bias, noise=noise, t=t,
                query_bias=query_bias, loss_type=loss_type, return_velocity=True,
            )

        full_w = float(gate_cfg.full_loss_weight) if self.gate_mode == "learned" else 0.0
        if full_w > 0 or distill:
            loss_full, v_full = flow_loss(None, vl_embs_list)
            out["loss_full"] = loss_full.detach()
            if full_w > 0:
                total = total + flow_weight * full_w * loss_full
        loss_gated, v_gated = flow_loss(keep, gated_embs)
        out["loss_gated"] = loss_gated.detach()
        if distill:
            loss_gate = ((v_gated.float() - v_full.detach().float()) ** 2).mean()
            out["loss_distill"] = loss_gate.detach()
        else:
            loss_gate = loss_gated
        if gated_grad == "gate":
            out["gate_loss"] = flow_weight * loss_gate  # backpropagated separately; the trainer drops its DiT grads
        else:
            total = total + flow_weight * loss_gate
        keep_mean = self._keep_stats(keep, gate_inputs["view_ids"])  # (B, V)
        for v in range(self.num_views):
            out[f"keep_view{v}"] = keep_mean[:, v].mean().detach()
        cur = gate_inputs["view_ids"] == self.num_views - 1
        if keep.shape[0] > 1 and bool((cur.sum(1) == cur.sum(1)[0]).all()):
            # input dependence of the current-frame mask: std over the batch per token, mean over tokens
            m_cur = torch.stack([keep[b][cur[b]] for b in range(keep.shape[0])])
            out["keep_xsample_std"] = m_cur.std(0).mean().detach()
        if self.gate_mode == "learned":
            if float(gate_cfg.budget_loss_weight) > 0:
                loss_budget = ((keep_mean - float(gate_cfg.budget)) ** 2).sum(1).mean()
                total = total + float(gate_cfg.budget_loss_weight) * loss_budget
                out["loss_budget"] = loss_budget.detach()
            # Scale of the gate scores: the budgeted mask is invariant to it, so it is not trained and only drifts;
            # it sets the effective learning rate of the gate (larger scores -> smaller change of z per update).
            out["gate_score_std"] = gate_logits.detach().std()
            if float(gate_cfg.get("logit_l2_weight", 0.0)) > 0:
                # β·mean ℓ²: a finite optimum for the MI term (which grows as m -> 0/1) and a gradient on ℓ that
                # does not pass through the sigmoid, so saturated tokens are pulled back.
                logit_sq = (gate_logits**2).mean()
                total = total + float(gate_cfg.logit_l2_weight) * logit_sq
                out["gate_logit_rms"] = logit_sq.detach().sqrt()
            if float(gate_cfg.get("mi_loss_weight", 0.0)) > 0:
                mi, h_marg, h_cond = self._mask_mutual_information(keep, gate_inputs["view_ids"])
                total = total - float(gate_cfg.mi_loss_weight) * mi
                out["mask_mi"] = mi.detach()
                out["mask_marginal_entropy"] = h_marg
                out["mask_cond_entropy"] = h_cond
        out["action_loss"] = total
        return out

    def _vlm_keep_losses(self, examples, batch_images, instructions, state) -> dict:
        """eye_gate.scheme = vlm_keep (see QwenDrivePIDefaultConfig.eye_gate)."""
        if self.anchor_enabled:
            raise NotImplementedError("eye_gate.scheme vlm_keep is not combined with anchors")
        gate_cfg = self.config.framework.eye_gate
        prep = self._vlm_keep_prepare(batch_images, instructions, examples)
        gi, attn, visual = prep["gate_inputs"], prep["attention_mask"], prep["visual"]
        keep, gate_logits = self._gate_keep(gi, attn, return_logits=True)  # (B, L), 1 on non-visual tokens
        tau = self._removal_tau()
        hard = torch.where(visual, (keep.detach() >= tau).to(keep.dtype), torch.ones_like(keep))
        if self.training:
            self.keep_step += 1
        device, dtype = keep.device, self._head_dtype()
        embs = {
            "full": prep["run"](None),
            "gated": prep["run"](keep),
            "removed": prep["run"](hard),
        }
        actions = torch.as_tensor(np.array([ex["action"] for ex in examples]), device=device, dtype=dtype)
        flow_target = actions[:, -self.action_horizon :, :]
        repeat = self.config.trainer.get("repeated_diffusion_steps", None) if self.config.get("trainer") else None
        repeat = int(repeat or self.config.framework.action_model.get("repeated_diffusion_steps", 1))
        flow_target = flow_target.repeat(repeat, 1, 1)
        state_t = None
        if state is not None:
            state_t = torch.as_tensor(state, device=device, dtype=dtype).repeat(repeat, 1, 1)
        noise = torch.randn(flow_target.shape, device=device, dtype=dtype)
        t = self.action_model.sample_time(flow_target.shape[0], device=device, dtype=dtype)

        def flow(name, keep_or_none):
            bias = self._cross_attention_bias(attn, keep_or_none, dtype).repeat(repeat, 1)
            vl = [h.to(dtype).repeat(repeat, 1, 1) for h in embs[name]]
            return self.action_model(vl, flow_target, state_t, encoder_attention_mask=bias, noise=noise, t=t)

        out = {
            "loss_full": flow("full", None),
            "loss_gated": flow("gated", keep),
            "loss_removed": flow("removed", hard),  # removed keys -inf, kept tokens unmasked
        }
        weights = {"loss_full": float(gate_cfg.get("full_loss_weight", 1.0)), "loss_gated": 1.0,
                   "loss_removed": float(gate_cfg.get("removed_loss_weight", 1.0))}
        total = sum(weights[k] * v for k, v in out.items())
        out = {k: v.detach() for k, v in out.items()}
        out["removal_tau"] = torch.tensor(tau)
        out["removed_frac"] = (1 - hard[visual]).mean().detach()
        out["gate_score_std"] = gate_logits.detach().std()
        keep_mean = self._keep_stats(keep, gi["view_ids"])
        for v in range(self.num_views):
            out[f"keep_view{v}"] = keep_mean[:, v].mean().detach()
        cur = gi["view_ids"] == self.num_views - 1
        if keep.shape[0] > 1 and bool((cur.sum(1) == cur.sum(1)[0]).all()):
            m_cur = torch.stack([keep[b][cur[b]] for b in range(keep.shape[0])]).detach().float()
            out["keep_xsample_std"] = m_cur.std(0).mean()
            ranks = m_cur.argsort(1).argsort(1).float()
            ranks = ranks - ranks.mean(1, keepdim=True)
            ranks = ranks / ranks.norm(dim=1, keepdim=True).clamp(min=1e-12)
            corr = ranks @ ranks.T
            n = corr.shape[0]
            out["keep_pairwise_rankcorr"] = (corr.sum() - corr.diagonal().sum()) / (n * (n - 1))
        out["action_loss"] = total
        return out

    def _textdrop_losses(self, out, total, gate_inputs, attention_mask, vl_embs_list, flow_target, state_t,
                         query_bias, flow_weight, loss_type, repeat, device, dtype) -> dict:
        """eye_gate.scheme = textdrop (see QwenDrivePIDefaultConfig.eye_gate)."""
        visual = gate_inputs["view_ids"] >= 0
        visual_mask = attention_mask & visual
        gate_in = dict(gate_inputs, final_hidden=gate_inputs["final_hidden"].detach())
        keep, gate_logits = self._gate_keep(gate_in, attention_mask, return_logits=True)
        out["gate_score_std"] = gate_logits.detach().std()
        cur = gate_inputs["view_ids"] == self.num_views - 1
        if keep.shape[0] > 1 and bool((cur.sum(1) == cur.sum(1)[0]).all()):
            m_cur = torch.stack([keep[b][cur[b]] for b in range(keep.shape[0])]).detach().float()
            out["keep_xsample_std"] = m_cur.std(0).mean()
            ranks = m_cur.argsort(1).argsort(1).float()
            ranks = ranks - ranks.mean(1, keepdim=True)
            ranks = ranks / ranks.norm(dim=1, keepdim=True).clamp(min=1e-12)
            corr = ranks @ ranks.T
            n = corr.shape[0]
            out["keep_pairwise_rankcorr"] = (corr.sum() - corr.diagonal().sum()) / (n * (n - 1))
        noise = torch.randn(flow_target.shape, device=device, dtype=dtype)
        t = self.action_model.sample_time(flow_target.shape[0], device=device, dtype=dtype)

        def run(keep_or_none, embs, mask):
            bias = self._cross_attention_bias(mask, keep_or_none, dtype).repeat(repeat, 1)
            return self.action_model(
                embs, flow_target, state_t, encoder_attention_mask=bias, noise=noise, t=t,
                query_bias=query_bias, loss_type=loss_type, return_velocity=True,
            )

        detached = [h.detach() for h in vl_embs_list]
        with torch.no_grad():
            loss_ref, v_ref = run(None, detached, attention_mask)  # ② teacher: ungated, all tokens
        _, v_drop = run(keep, detached, visual_mask)  # ③ gated, no text tokens
        loss_distill = ((v_drop.float() - v_ref.float()) ** 2).mean()
        out["gate_loss"] = flow_weight * loss_distill  # gate only (the trainer drops its DiT gradients)
        loss_gated, _ = run(keep.detach(), vl_embs_list, attention_mask)  # ④ gated, all tokens: VLM + DiT
        out.update(loss_distill=loss_distill.detach(), loss_teacher=loss_ref.detach(), loss_gated=loss_gated.detach())
        keep_mean = self._keep_stats(keep, gate_inputs["view_ids"])
        for v in range(self.num_views):
            out[f"keep_view{v}"] = keep_mean[:, v].mean().detach()
        out["action_loss"] = total + flow_weight * loss_gated
        return out

    def _predict(self, examples: List[dict]) -> dict:
        batch_images, instructions, state = self._prepare_inputs(examples)
        if self.vlm_keep:
            return self._vlm_keep_predict(examples, batch_images, instructions, state)
        vl_embs_list, attention_mask, gate_inputs = self._encode_vl_hidden_states(batch_images, instructions)

        device, dtype = vl_embs_list[-1].device, self._head_dtype()
        gate_target = self._gate_target(examples, device)
        if gate_target is not None:
            gate_inputs["target"] = gate_target
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        state_t = torch.as_tensor(state, device=device, dtype=dtype) if state is not None else None

        out = {}
        encoder_mask = attention_mask
        probe = None
        if self.gate_enabled:
            keep, logits = self._gate_keep(gate_inputs, attention_mask, return_logits=True)
            encoder_mask = self._cross_attention_bias(attention_mask, keep, dtype)
            out["keep_maps"] = self._keep_maps(keep, gate_inputs)
            if logits is not None:
                out["score_maps"] = self._keep_maps(self._frame_standardized(logits, gate_inputs), gate_inputs)
            probe = CrossAttentionProbe(self.action_model.model, self.cross_layer_ids)
            probe.__enter__()
        if self.anchor_enabled:
            logits = self.anchor_classifier(gate_inputs["query_hidden"].to(self.anchor_classifier[0].weight.dtype))
            reference, ids, weights = self._anchor_reference(logits)
            out.update(
                anchor_reference=reference.float().cpu().numpy(),
                anchor_ids=ids.cpu().numpy(),
                anchor_weights=weights.float().cpu().numpy(),
                anchor_top1=logits.argmax(-1).cpu().numpy(),
            )
            if self.anchor_space == "control":  # positions of the reference / every anchor for eval and plots
                v0 = self._current_speed(examples, device)
                zero = torch.zeros(reference.shape[0], 1, 2, device=device)
                out["anchor_reference"] = torch.cat([zero, self._to_metres(reference, v0)], 1).float().cpu().numpy()
                anchors_m = self._to_metres(self.anchors[None].expand(reference.shape[0], -1, -1, -1), v0)
                zero_k = torch.zeros(*anchors_m.shape[:2], 1, 2, device=device)
                out["anchor_xy"] = torch.cat([zero_k, anchors_m], 2).float().cpu().numpy()
            if self.anchor_stage == 1:
                pred_m = reference
            else:
                query_bias = self.anchor_embed(self._normalize_actions(reference).to(self.anchor_embed[0].weight.dtype))
                residual = self.action_model.predict_action(
                    vl_embs_list, state_t, encoder_attention_mask=encoder_mask, query_bias=query_bias
                )
                pred_m = reference + residual.float() * self.residual_std
            pred_actions = self._normalize_actions(pred_m)
        else:
            pred_actions = self.action_model.predict_action(vl_embs_list, state_t, encoder_attention_mask=encoder_mask)
        if probe is not None:
            probe.__exit__()
            self._add_attention_outputs(out, probe.mean(), gate_inputs, attention_mask, "")
            gate_cfg = self.config.framework.eye_gate
            if self.gate_mode == "learned" and str(gate_cfg.get("scheme", "standard")) == "textdrop":
                # the gate's training path ③: gated, text tokens hidden from the DiT
                visual_mask = attention_mask & (gate_inputs["view_ids"] >= 0)
                with CrossAttentionProbe(self.action_model.model, self.cross_layer_ids) as probe_td:
                    drop = self.action_model.predict_action(
                        vl_embs_list, state_t, encoder_attention_mask=self._cross_attention_bias(visual_mask, keep, dtype)
                    )
                self._add_attention_outputs(out, probe_td.mean(), gate_inputs, attention_mask, "_textdrop")
                out["normalized_actions_textdrop"] = drop.float().cpu().numpy()
        if self.action_clip is not None:
            pred_actions = pred_actions.clamp(-float(self.action_clip), float(self.action_clip))
        out["normalized_actions"] = pred_actions.float().cpu().numpy()
        return out

    def _vlm_keep_predict(self, examples, batch_images, instructions, state) -> dict:
        """Inference path B: tokens with m < removal_threshold removed in the VLM and the DiT, others unmasked."""
        prep = self._vlm_keep_prepare(batch_images, instructions, examples)
        gi, attn, visual = prep["gate_inputs"], prep["attention_mask"], prep["visual"]
        keep, logits = self._gate_keep(gi, attn, return_logits=True)
        hard = torch.where(visual, (keep >= self._removal_tau()).to(keep.dtype), torch.ones_like(keep))
        dtype = self._head_dtype()
        vl = [h.to(dtype) for h in prep["run"](hard)]
        state_t = torch.as_tensor(state, device=keep.device, dtype=dtype) if state is not None else None
        out = {"keep_maps": self._keep_maps(keep, gi), "removed_maps": self._keep_maps(1 - hard, gi)}
        out["score_maps"] = self._keep_maps(self._frame_standardized(logits, gi), gi)
        with CrossAttentionProbe(self.action_model.model, self.cross_layer_ids) as probe:
            pred = self.action_model.predict_action(vl, state_t, encoder_attention_mask=self._cross_attention_bias(attn, hard, dtype))
        self._add_attention_outputs(out, probe.mean(), gi, attn, "")
        if self.action_clip is not None:
            pred = pred.clamp(-float(self.action_clip), float(self.action_clip))
        out["normalized_actions"] = pred.float().cpu().numpy()
        return out

    def _frame_standardized(self, logits: torch.Tensor, gate_inputs: dict) -> torch.Tensor:
        """Gate scores (concatenated visual tokens of the batch) -> (B, L) with each frame's scores standardised
        (the z of budgeted_keep), 0 on non-visual tokens."""
        view_ids = gate_inputs["view_ids"]
        visual = view_ids >= 0
        z = torch.zeros(view_ids.shape, dtype=torch.float32, device=view_ids.device)
        z[visual] = logits.float()
        for b in range(z.shape[0]):
            for v in view_ids[b][visual[b]].unique():
                sel = view_ids[b] == v
                z[b, sel] = (z[b, sel] - z[b, sel].mean()) / (z[b, sel].std() + 1e-6)
        return z

    def _add_attention_outputs(self, out: dict, attn: torch.Tensor, gate_inputs: dict, attention_mask, suffix: str):
        """Per-frame maps of the DiT cross-attention mass and its share on the text tokens."""
        visual = gate_inputs["view_ids"] >= 0
        out[f"attn_maps{suffix}"] = self._keep_maps(attn, gate_inputs)
        out[f"attn_text_mass{suffix}"] = (attn * ((~visual) & attention_mask)).sum(1).float().cpu().numpy()

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
        """Returns ``normalized_actions`` (B, action_horizon, action_dim), clamped only if ``action_clip`` is set."""
        if type(examples) is not list:
            examples = [examples]
        return self._predict(examples)
