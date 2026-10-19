"""Per-token keep weights inside the Qwen3.5 language model (``visual_keep`` kwarg of the language model forward).

``visual_keep`` (B, L) in [0, 1] (1 for every non-visual token) scales how much the other tokens can read token p,
at the operations that move information between tokens (after each layer's RMSNorm and projections, so the
normalisation does not undo it):

- softmax attention layers (``Qwen3_5Attention``): ``log keep_p`` is added to every query's logit on key p;
- Gated DeltaNet layers (``Qwen3_5GatedDeltaNet``): token p's write strength β_p and its state decay
  (log α_p = g_p) are scaled by keep_p, and its q / k / v enter the short causal convolution scaled by keep_p.

keep_p = 0: nothing of token p reaches any other token (no key, no write, no decay, no conv input); the other
tokens' positions are unchanged. (It differs from deleting p from the sequence only in the DeltaNet short
convolution, whose window then still spans p's slot.) The kwarg travels through the decoder layers' ``**kwargs``, so gradient
checkpointing recomputes with the same weights. Call ``install()`` once and ``attach(model)`` on the VLM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

HARD_OFF = -1e4  # attention logit offset of a fully removed key (finite: no NaN on fully masked rows)
EPS = 1e-6


def log_keep(keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep > 0, torch.log(keep.clamp(min=EPS)), torch.full_like(keep, HARD_OFF))


def _attention_bias(mask, keep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    b, length = hidden_states.shape[:2]
    if mask is None:  # sdpa's is_causal path: build the causal mask explicitly
        mask = torch.full((length, length), HARD_OFF, device=hidden_states.device).triu(1)[None, None]
    elif mask.dtype == torch.bool:
        mask = torch.zeros(mask.shape, device=mask.device).masked_fill(~mask, HARD_OFF)
    bias = mask.float() + log_keep(keep.float())[:, None, None, :]
    return bias.to(hidden_states.dtype)


def install() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    if getattr(M, "_visual_keep_installed", False):
        return
    attn_forward, dn_forward = M.Qwen3_5Attention.forward, M.Qwen3_5GatedDeltaNet.forward

    def attention(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                  visual_keep=None, **kwargs):
        if visual_keep is not None:
            attention_mask = _attention_bias(attention_mask, visual_keep, hidden_states)
        return attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

    def delta_net(self, hidden_states, cache_params=None, attention_mask=None, visual_keep=None, **kwargs):
        self._visual_keep = visual_keep
        try:
            return dn_forward(self, hidden_states, cache_params, attention_mask, **kwargs)
        finally:
            self._visual_keep = None

    M.Qwen3_5Attention.forward = attention
    M.Qwen3_5GatedDeltaNet.forward = delta_net
    M._visual_keep_installed = True


def attach(vlm: torch.nn.Module) -> None:
    """Register the DeltaNet projection hooks on every Gated DeltaNet layer of ``vlm`` (idempotent)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    install()
    for module in vlm.modules():
        if not isinstance(module, M.Qwen3_5GatedDeltaNet) or getattr(module, "_visual_keep_hooks", False):
            continue
        parent = module
        parent._visual_keep = None

        def qkv_hook(mod, args, out, parent=parent):  # conv input of token p scaled by keep_p
            keep = parent._visual_keep
            return out if keep is None else out * keep[..., None].to(out.dtype)

        def beta_hook(mod, args, out, parent=parent):  # sigmoid(b') = keep_p * sigmoid(b)
            keep = parent._visual_keep
            if keep is None:
                return out
            log_beta = F.logsigmoid(out.float()) + torch.log(keep.float().clamp(min=EPS))[..., None]
            return (log_beta - torch.log1p(-log_beta.exp().clamp(max=1 - EPS))).to(out.dtype)

        def decay_hook(mod, args, out, parent=parent):  # softplus(a' + dt) = keep_p * softplus(a + dt)
            keep = parent._visual_keep
            if keep is None:
                return out
            dt = parent.dt_bias.float()
            y = (keep.float()[..., None] * F.softplus(out.float() + dt)).clamp(min=EPS)
            inv = torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(max=20))))  # softplus^-1
            return (inv - dt).to(out.dtype)

        module.in_proj_qkv.register_forward_hook(qkv_hook)
        module.in_proj_b.register_forward_hook(beta_hook)
        module.in_proj_a.register_forward_hook(decay_hook)
        module._visual_keep_hooks = True
