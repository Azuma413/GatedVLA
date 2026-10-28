"""Record the action expert's cross-attention weights over the VLM tokens.

``CrossAttentionProbe`` hooks the cross-attention (``attn1`` of the DiT blocks that read the VLM) and recomputes the
attention probabilities exactly as diffusers' ``AttnProcessor2_0`` does (same projections, q/k norms, 1/sqrt(d)
scale, additive (B, L) bias or boolean mask). The probabilities are averaged over heads and query tokens and
accumulated over blocks and calls (denoising steps), giving per sample the mean attention mass on every key token.
Use only under ``torch.no_grad`` (evaluation); it costs one extra q·k product per block.
"""

from __future__ import annotations

import math
from typing import List

import torch


class CrossAttentionProbe:
    def __init__(self, dit, block_ids: List[int], stats: bool = False, key_mask: "torch.Tensor" = None):
        """stats: also accumulate per-sample diagnostics over ``key_mask`` (B, L) bool keys (e.g. image tokens):
        normalised attention entropy per query, std of the logits per query, and the share of the keys' energy
        that varies over those tokens (0: every key identical)."""
        self.attns = [dit.transformer_blocks[i].attn1 for i in block_ids]
        self.handles, self.sum, self.calls = [], None, 0
        self.stats, self.key_mask, self.stat_sums = stats, key_mask, {}

    def _hook(self, module, args, kwargs, output):
        x = args[0] if args else kwargs["hidden_states"]
        enc = kwargs.get("encoder_hidden_states")
        if enc is None:
            return
        mask = kwargs.get("attention_mask")
        if module.norm_cross:
            enc = module.norm_encoder_hidden_states(enc)
        q, k = module.to_q(x), module.to_k(enc)
        b, heads = q.shape[0], module.heads
        d = k.shape[-1] // heads
        q = q.view(b, -1, heads, d).transpose(1, 2)
        k = k.view(b, -1, heads, d).transpose(1, 2)
        if module.norm_q is not None:
            q = module.norm_q(q)
        if module.norm_k is not None:
            k = module.norm_k(k)
        logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d)  # (B, H, N, L)
        if mask is not None:
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
            else:
                logits = logits + mask.float()[:, None, None, :]
        full = torch.softmax(logits, dim=-1)
        probs = full.mean((1, 2))  # (B, L)
        self.sum = probs if self.sum is None else self.sum + probs
        self.calls += 1
        if self.stats and self.key_mask is not None:
            km = self.key_mask[: logits.shape[0]]
            n = km.sum(-1).clamp(min=2).float()  # (B,)
            lg = logits.masked_fill(~km[:, None, None, :], float("-inf"))
            p = torch.softmax(lg, dim=-1)
            ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean((1, 2)) / n.log()
            lz = torch.where(km[:, None, None, :], logits, torch.zeros_like(logits))
            mu = lz.sum(-1, keepdim=True) / n[:, None, None, None]
            var = (((lz - mu) * km[:, None, None, :]) ** 2).sum(-1) / n[:, None, None]
            spread = var.sqrt().mean((1, 2))
            kf = k.float() * km[:, None, :, None]
            kmean = kf.sum(2, keepdim=True) / n[:, None, None, None]
            share = (((kf - kmean) * km[:, None, :, None]) ** 2).sum((1, 2, 3)) / (kf**2).sum((1, 2, 3)).clamp_min(1e-12)
            for name, v in (("entropy", ent), ("logit_spread", spread), ("key_varying_share", share)):
                self.stat_sums[name] = self.stat_sums.get(name, 0) + v.detach()

    def __enter__(self):
        self.handles = [a.register_forward_hook(self._hook, with_kwargs=True) for a in self.attns]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stat_means(self) -> dict:
        """Per-sample (B,) means over blocks and calls of the diagnostics collected with ``stats``."""
        return {k: v / max(self.calls, 1) for k, v in self.stat_sums.items()}

    def mean(self) -> torch.Tensor:
        """(B, L) attention mass per key token, averaged over heads, queries, blocks and calls (sums to 1)."""
        return self.sum / max(self.calls, 1)
