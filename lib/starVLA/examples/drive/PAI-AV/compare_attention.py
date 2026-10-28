"""DiT cross-attention maps of a gated (eye) and an ungated checkpoint on the same samples and noise.

Per sample (the eval figure samples: per maneuver the brightest-half clips, plus the intersection scenes): current-frame
maps of the DiT cross-attention mass (averaged over cross blocks, heads, action queries and denoising steps) of
  ungated model (target given) | gated model: keep mask | gated model with its mask (inference) | gated model, no mask,
and per-sample statistics: normalised entropy over the image tokens, mass on the top 5 % image tokens, attention mass on
the text tokens, Spearman correlation between the two models' maps and between the gated model's attention and mask.

    python examples/drive/PAI-AV/compare_attention.py --ungated <ckpt dir> --gated <ckpt dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.pai_av_drive import DriveSampler, Normalizer, PAIAVDriveDataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.modules.action_model.attention_probe import CrossAttentionProbe
from starVLA.training.drive_openloop import fig_to_image
from starVLA.training.train_drive import load_weights


def load(ckpt: str):
    cfg = OmegaConf.load(f"{ckpt}/config.yaml")
    model = build_framework(cfg).float().eval()
    load_weights(model, ckpt)
    return cfg, model


def samples_for(cfg, per_class: int):
    d = cfg.datasets.vla_data
    stats = json.load(open(d.stats_path))
    sampler = DriveSampler(**{k: v for k, v in stats["sampler"].items() if k in DriveSampler.__init__.__code__.co_varnames})
    kw = dict(sampler=sampler, current_image_pixels=int(cfg.framework.drive.current_image_pixels),
              history_image_pixels=int(cfg.framework.drive.history_image_pixels))
    ev = json.load(open(d.eval_index_path))[: int(d.get("num_eval_samples", 240))]
    picks = []
    for m in ("left", "right", "straight"):
        seen, cands = set(), []
        for s in ev:
            if s["maneuver"] == m and s["clip_id"] not in seen:
                seen.add(s["clip_id"])
                cands.append(s)
        idx = np.linspace(0, len(cands) - 1, per_class).round().astype(int)
        picks += [dict(cands[i], label=f"{m}") for i in idx]
    if d.get("scene_index_path"):
        picks += [dict(s, label=f"{s['maneuver']} intersection") for s in json.load(open(d.scene_index_path))]
    return PAIAVDriveDataset(d.data_root, Normalizer(stats), samples=picks, **kw), picks


@torch.no_grad()
def attention(model, item, noise, gated: bool):
    imgs, ins, state = model._prepare_inputs([item])
    vl, attn_mask, gi = model._encode_vl_hidden_states(imgs, ins)
    dtype = model._head_dtype()
    target = model._gate_target([item], vl[-1].device)
    if target is not None:
        gi["target"] = target
    vl = [h.to(dtype) for h in vl]
    st = torch.as_tensor(state, dtype=dtype) if state is not None else None
    if st is not None and model.target_flag:
        st = model._flag_state(st, torch.ones(st.shape[0]))

    def run(bias):
        with CrossAttentionProbe(model.action_model.model, model.cross_layer_ids) as p:
            model.action_model.predict_action(vl, st, encoder_attention_mask=bias, noise=noise)
        a = p.mean()
        return model._keep_maps(a, gi)[0][-1], float((a * ((gi["view_ids"] < 0) & attn_mask)).sum())

    out = {}
    out["attn_nomask"], out["text_nomask"] = run(attn_mask)
    if gated:
        keep = model._gate_keep(gi, attn_mask)
        out["keep"] = model._keep_maps(keep, gi)[0][-1]
        out["attn_mask"], out["text_mask"] = run(model._cross_attention_bias(attn_mask, keep, dtype))
    return out


def stats(a: np.ndarray) -> dict:
    p = a.reshape(-1) / max(a.sum(), 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    top = float(np.sort(p)[::-1][: max(1, len(p) // 20)].sum())
    return {"entropy": ent, "top5pct_mass": top}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a.reshape(-1))), np.argsort(np.argsort(b.reshape(-1)))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ungated", required=True)
    p.add_argument("--gated", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per_class", type=int, default=4)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_u, model_u = load(args.ungated)
    ds, picks = samples_for(cfg_u, args.per_class)
    items = [ds[i] for i in range(len(ds))]
    torch.manual_seed(0)
    noises = [torch.randn(1, model_u.action_horizon, model_u.action_model.action_dim) for _ in items]
    res_u = [attention(model_u, it, n, gated=False) for it, n in zip(items, noises)]
    del model_u
    _, model_g = load(args.gated)
    res_g = [attention(model_g, it, n, gated=True) for it, n in zip(items, noises)]

    import matplotlib.pyplot as plt

    rows = []
    for k, (it, pk, ru, rg) in enumerate(zip(items, picks, res_u, res_g)):
        row = {"label": pk["label"], "clip": pk["clip_id"][:8], "t0_s": pk["t0"] * 1e-6,
               **{f"ungated_{n}": v for n, v in stats(ru["attn_nomask"]).items()},
               **{f"gated_{n}": v for n, v in stats(rg["attn_mask"]).items()},
               **{f"gated_nomask_{n}": v for n, v in stats(rg["attn_nomask"]).items()},
               "ungated_text_mass": ru["text_nomask"], "gated_text_mass": rg["text_mask"],
               "rank_corr_ungated_vs_gated": spearman(ru["attn_nomask"], rg["attn_mask"]),
               "rank_corr_gated_attn_vs_mask": spearman(rg["attn_mask"], rg["keep"]),
               "rank_corr_ungated_attn_vs_mask": spearman(ru["attn_nomask"], rg["keep"])}
        rows.append(row)
        img = it["image"][-1]
        panels = [("front camera (t0)", None), ("ungated: DiT attention", ru["attn_nomask"]),
                  ("gated: keep mask", rg["keep"]), ("gated: DiT attention (mask, inference)", rg["attn_mask"]),
                  ("gated model without mask: DiT attention", rg["attn_nomask"])]
        fig, axes = plt.subplots(2, 3, figsize=(16, 6.6))
        axes = axes.reshape(-1)
        axes[-1].axis("off")
        for ax, (title, m) in zip(axes, panels):
            ax.imshow(img)
            if m is not None:  # attention: colour scale clipped at the 99th percentile so that a few peaks do not hide the rest
                vmax = float(np.percentile(m, 99)) if "attention" in title else 1.0
                ax.imshow(m, cmap="magma" if "attention" in title else "viridis", alpha=0.6, vmin=0, vmax=vmax,
                          extent=(0, img.width, img.height, 0), interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{pk['label']} | {pk['clip_id'][:8]} t0={pk['t0'] * 1e-6:.1f}s | rank corr ungated vs gated "
                     f"{row['rank_corr_ungated_vs_gated']:.2f}", fontsize=10)
        fig_to_image(fig, canvas=(1700, 760)).save(out_dir / f"{k:02d}_{pk['label'].replace(' ', '_')}.png")
        plt.close(fig)
    json.dump(rows, open(out_dir / "summary.json", "w"), indent=1)
    keys = [k for k in rows[0] if isinstance(rows[0][k], float) and k != "t0_s"]
    print("mean over samples:")
    for k in keys:
        print(f"  {k}: {np.mean([r[k] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
