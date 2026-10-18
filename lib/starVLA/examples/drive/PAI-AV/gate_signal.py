"""Does the eye gate's training loss prefer any mask over a random one? (eval only, one checkpoint)

For each eval sample the gate loss of the checkpoint's scheme is evaluated under budgeted soft masks of the same
family (mean m = budget, contrast fixed), with the same flow-matching samples (noise, t) for all masks:

    learned  the gate's mask
    random   the learned mask's values permuted within each frame (same value distribution, other placement)
    uniform  m = budget everywhere
    oracle   per-sample scores optimised directly on the loss (Adam through budgeted_keep) on one set of (noise, t)
             draws; every loss is reported on a held-out set of draws

Loss (scheme textdrop): ‖v(gated, text tokens hidden) − sg(v(ungated, all tokens))‖²; (scheme standard):
flow matching of the gated path against the ground truth.

Outputs summary.json (per-sample losses, advantages over random) and figs/ (image, learned mask, oracle mask).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from starVLA.model.framework.VLM4A.QwenDrivePI import budgeted_keep  # noqa: E402


def frame_keep(scores: torch.Tensor, frames: torch.Tensor, budget: float, contrast: float) -> torch.Tensor:
    """Budgeted soft mask per frame for the visual tokens' scores (N,) with frame ids (N,)."""
    m = torch.empty_like(scores)
    for v in frames.unique():
        sel = frames == v
        m[sel] = budgeted_keep(scores[sel], budget, contrast)
    return m


class GateLoss:
    """Gate loss of one sample as a function of the visual tokens' keep values, averaged over K (noise, t) draws."""

    def __init__(self, model, example: dict, draws: int, generator: torch.Generator):
        self.model = m = model
        images, instructions, state = m._prepare_inputs([example])
        with torch.no_grad():
            vl, attn, gi = m._encode_vl_hidden_states(images, instructions)
        device, dtype = vl[-1].device, m._head_dtype()
        target = m._gate_target([example], device)
        if target is not None:
            gi["target"] = target
        self.vl = [h.to(dtype) for h in vl]
        self.attn, self.gi, self.dtype = attn, gi, dtype
        self.visual = gi["view_ids"][0] >= 0
        self.frames = gi["view_ids"][0][self.visual]
        gate_cfg = m.config.framework.eye_gate
        self.budget, self.contrast = float(gate_cfg.budget), float(gate_cfg.get("contrast", 2.0))
        self.textdrop = str(gate_cfg.get("scheme", "standard")) == "textdrop"
        action = torch.as_tensor(np.asarray(example["action"]), device=device, dtype=dtype)[None, -m.action_horizon :]
        self.state = torch.as_tensor(state, device=device, dtype=dtype) if state is not None else None
        self.draws = []
        for _ in range(2 * draws):  # first half: oracle optimisation, second half: evaluation
            noise = torch.randn(action.shape, generator=generator).to(device, dtype)
            t = m.action_model.sample_time(1, device=device, dtype=dtype)
            self.draws.append((noise, t))
        self.action, self.K = action, draws
        self.teacher = None
        if self.textdrop:
            with torch.no_grad():
                self.teacher = [self._velocity(None, all_tokens=True, draw=d) for d in self.draws]

    def _velocity(self, keep_visual, all_tokens: bool, draw):
        noise, t = draw
        mask = self.attn if all_tokens else self.attn & self.gi["view_ids"].ge(0)
        bias = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        if keep_visual is not None:
            bias = bias.clone()
            bias[0, self.visual] = torch.log(keep_visual.clamp(min=1e-12))
        bias = bias.masked_fill(~mask, float("-inf")).to(self.dtype)
        loss, v = self.model.action_model(
            self.vl, self.action, self.state, encoder_attention_mask=bias, noise=noise, t=t, return_velocity=True
        )
        return v if self.textdrop else loss

    def loss(self, keep_visual, which: str) -> torch.Tensor:
        draws = range(self.K) if which == "opt" else range(self.K, 2 * self.K)
        total = 0.0
        for d in draws:
            if self.textdrop:
                v = self._velocity(keep_visual, all_tokens=False, draw=self.draws[d])
                total = total + ((v.float() - self.teacher[d].float()) ** 2).mean()
            else:
                total = total + self._velocity(keep_visual, all_tokens=True, draw=self.draws[d]).float()
        return total / len(draws)

    def learned_keep(self) -> torch.Tensor:
        with torch.no_grad():
            keep = self.model._gate_keep(self.gi, self.attn)
        return keep[0][self.visual].float()

    def permuted(self, keep: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        out = keep.clone()
        for v in self.frames.unique():
            idx = torch.nonzero(self.frames == v).squeeze(-1)
            out[idx] = keep[idx][torch.randperm(len(idx), generator=generator).to(idx.device)]
        return out

    def oracle(self, init_scores: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
        scores = init_scores.detach().clone().float().requires_grad_(True)
        opt = torch.optim.Adam([scores], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            self.loss(frame_keep(scores, self.frames, self.budget, self.contrast), "opt").backward()
            opt.step()
        with torch.no_grad():
            return frame_keep(scores, self.frames, self.budget, self.contrast)


def measure(model, example: dict, draws: int, permutations: int, oracle_steps: int, oracle_lr: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    gl = GateLoss(model, example, draws, g)
    learned = gl.learned_keep()
    uniform = torch.full_like(learned, gl.budget)
    with torch.no_grad():
        res = {
            "learned": float(gl.loss(learned, "eval")),
            "uniform": float(gl.loss(uniform, "eval")),
            "random": float(np.mean([float(gl.loss(gl.permuted(learned, g), "eval")) for _ in range(permutations)])),
        }
    oracle = gl.oracle(torch.randn(learned.shape, generator=g).to(learned.device), oracle_steps, oracle_lr)
    with torch.no_grad():
        res["oracle"] = float(gl.loss(oracle, "eval"))
    for k in ("learned", "oracle"):
        res[f"adv_{k}"] = (res["random"] - res[k]) / max(res["random"], 1e-12)  # > 0: better than random
    return res, learned, oracle, gl


def main():
    from omegaconf import OmegaConf

    from diagnose_mask import build_dataset
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.train_drive import load_weights

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--eval_index", required=True)
    p.add_argument("--per_class", type=int, default=16)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--permutations", type=int, default=4)
    p.add_argument("--oracle_steps", type=int, default=40)
    p.add_argument("--oracle_lr", type=float, default=0.3)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

    cfg = OmegaConf.load(str(Path(args.ckpt) / "config.yaml"))
    torch.manual_seed(0)
    model = build_framework(cfg).float()
    load_weights(model, args.ckpt, print)
    model = model.to(device).eval().requires_grad_(False)  # only the oracle scores are optimised
    index = json.load(open(args.eval_index))
    samples = [s for m in ("left", "right", "straight") for s in [x for x in index if x["maneuver"] == m][: args.per_class]]
    ds = build_dataset(cfg, args.ckpt, samples, args.data_root)
    print(f"{len(samples)} samples on {device}, scheme {cfg.framework.eye_gate.get('scheme', 'standard')}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for i, s in enumerate(samples):
        ex = ds[i]
        with torch.autocast(device.type, dtype=torch.bfloat16):
            res, learned, oracle, gl = measure(model, ex, args.draws, args.permutations, args.oracle_steps, args.oracle_lr, i)
        res.update(clip=s["clip_id"], t0=s["t0"], maneuver=s["maneuver"])
        rows.append(res)
        print(i, s["maneuver"], {k: round(v, 5) for k, v in res.items() if isinstance(v, float)}, flush=True)
        cur = gl.frames == gl.frames.max()
        _, h, w = (int(x) for x in gl.gi["image_grid_thw"].view(-1, 3)[-1])
        merge = int(model.qwen_vl_interface.processor.image_processor.merge_size)
        maps = [x[cur].float().cpu().numpy().reshape(h // merge, w // merge) for x in (learned, oracle)]
        img = ex["image"][-1]
        fig, axes = plt.subplots(1, 3, figsize=(18, 3.6))
        axes[0].imshow(img)
        for ax, mp, name in zip(axes[1:], maps, ("learned", "oracle (optimised on the loss)")):
            ax.imshow(np.asarray(img.convert("L")), cmap="gray")
            im = ax.imshow(mp, cmap="viridis", vmin=0, vmax=1, alpha=0.6, interpolation="nearest",
                           extent=(-0.5, img.width - 0.5, img.height - 0.5, -0.5))
            ax.set_title(f"{name}: loss {res[name.split()[0]]:.4f}", fontsize=9)
        fig.colorbar(im, ax=axes[2], fraction=0.03)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"{s['maneuver']} {s['clip_id'][:8]} | random {res['random']:.4f} uniform {res['uniform']:.4f}", fontsize=9)
        fig.savefig(out / "figs" / f"{i:02d}_{s['maneuver']}_{s['clip_id'][:8]}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        np.savez(out / "figs" / f"{i:02d}.npz", learned=maps[0], oracle=maps[1])

    keys = ("learned", "random", "uniform", "oracle", "adv_learned", "adv_oracle")
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update({f"{k}_median": float(np.median([r[k] for r in rows])) for k in ("adv_learned", "adv_oracle")})
    # how sample-specific are the oracle masks (1 = one fixed pattern)
    ors = []
    for i in range(len(rows)):
        ors.append(np.load(out / "figs" / f"{i:02d}.npz")["oracle"].ravel())
    if len({o.shape for o in ors}) == 1 and len(ors) > 1:
        r = np.argsort(np.argsort(np.stack(ors), 1), 1).astype(float)
        r -= r.mean(1, keepdims=True)
        r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
        c = r @ r.T
        summary["oracle_pairwise_rankcorr"] = float((c.sum() - np.trace(c)) / (len(ors) * (len(ors) - 1)))
    json.dump({"summary": summary, "samples": rows, "args": vars(args)}, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
