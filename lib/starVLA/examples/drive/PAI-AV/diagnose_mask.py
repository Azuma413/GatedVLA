"""Why does the v2 eye gate learn an uninformative mask? Evaluation-only diagnostics on the same images.

Models: v1 baseline, v2 baseline, v2 eye (gate). All three see the same samples (clip, t0) taken from the v2 eval
index (+ any extra clips given), each with its own input layout / normaliser (from the checkpoint).

  (1) dependence map   current frame split into BxB-token blocks; each block is removed from the DiT
                       (cross-attention bias -inf) and the change of the predicted trajectory is measured
                       (mean L2 over the horizon vs. the unmodified prediction, same noise; K noises averaged).
  (2) pixel occlusion  3x3 image regions of the current frame are painted grey and the whole model is rerun
                       (VLM included); compared with removing the same region only from the DiT. A large pixel
                       effect with a small DiT effect means the information reaches the DiT through other tokens.
  (3) mask vs. dependence (eye): Spearman correlation between the learned keep values (block means) and the
                       dependence map of the eye model itself and of the v2 baseline.
  (4) mask strength (eye): prediction change and ADE for no gate / learned (contrast 2) / contrast 4, 8 / hard
                       top-20 % / hard random 20 % / all visual tokens removed. Baselines: random 20 % / all removed.
  (5) gate-input features (v1 vs v2 VLM): explained variance of the top principal components of the standardised
                       current-frame visual hidden states, adjacent-token cosine similarity vs random pairs.
  (6) q contribution (eye): rank correlation of the gate logits of image b with its own text vs. other samples'
                       text (1 = the mask ignores the text conditioning).
Outputs: <out>/summary.json, <out>/figs/<idx>_<clip>.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

STARVLA = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, STARVLA)
from starVLA.dataloader.pai_av_drive import DriveSampler, Normalizer, PAIAVDriveDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.model.framework.VLM4A.QwenDrivePI import budgeted_keep  # noqa: E402
from starVLA.training.train_drive import load_weights  # noqa: E402


def spearman(a, b):
    a, b = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def build_dataset(cfg, ckpt, samples, data_root):
    fw, dcfg = cfg.framework, cfg.datasets.vla_data
    sampler = DriveSampler(
        state_horizon=int(fw.action_model.get("state_horizon", 5)), action_horizon=int(fw.action_model.action_horizon),
        target_distance=float(dcfg.get("target_distance", 25.0)), state_mode=str(dcfg.get("state_mode", "drivemoe")),
        action_mode=str(dcfg.get("action_mode", "waypoints")), state_target_point=bool(dcfg.get("state_target_point", True)),
    )
    return PAIAVDriveDataset(
        data_root, Normalizer.from_file(str(Path(ckpt) / "norm_stats.json")), samples=samples, sampler=sampler,
        current_image_pixels=int(fw.drive.current_image_pixels), history_image_pixels=int(fw.drive.history_image_pixels),
        prompt=str(fw.drive.get("prompt", "predict trajectory")),
        target_point_in_prompt=bool(dcfg.get("target_point_in_prompt", False)),
        speed_in_prompt=bool(dcfg.get("speed_in_prompt", False)),
    )


class Runner:
    def __init__(self, model, ds, device, noises):
        self.model, self.ds, self.device, self.noises = model, ds, device, noises  # noises (K, H, 2)

    def encode(self, examples):
        m = self.model
        batch_images, instructions, state = m._prepare_inputs(examples)
        vl, attn, gi = m._encode_vl_hidden_states(batch_images, instructions)
        dtype = m._head_dtype()
        vl = [h.to(dtype) for h in vl]
        state_t = torch.as_tensor(state, device=self.device, dtype=dtype) if state is not None else None
        return vl, attn, gi, state_t

    def predict(self, vl, state_t, biases, meta, chunk=8):
        """vl from one sample (1, L, D); biases (V, L) -> positions (V, K, H+, 2). Runs `chunk` bias rows at a time."""
        K = self.noises.shape[0]
        outs = []
        for b in biases.split(chunk):
            V = b.shape[0]
            rep = lambda x: x.expand(V * K, *x.shape[1:]).contiguous()  # noqa: E731
            copies = {}  # the per-block list aliases one tensor for each (cross, self) pair: copy each tensor once
            vl_rep = [copies.setdefault(id(h), rep(h)) for h in vl]
            outs.append(self.model.action_model.predict_action(
                vl_rep, rep(state_t) if state_t is not None else None,
                encoder_attention_mask=b.repeat_interleave(K, 0),
                noise=self.noises.repeat(V, 1, 1),
            ).float().cpu().numpy())
            del vl_rep, copies
        out = np.concatenate(outs)
        xy = np.stack([self.ds.action_to_xy(o, meta) for o in out])
        return xy.reshape(biases.shape[0], K, *xy.shape[1:])

    def predict_batch(self, examples, metas, extra_bias=None):
        """Full model (VLM included) on several examples; one bias row per example (or the plain mask)."""
        vl, attn, gi, state_t = self.encode(examples)
        bias = self.model._cross_attention_bias(attn, extra_bias(gi, attn) if extra_bias else None, vl[0].dtype)
        res = []
        for b in range(len(examples)):
            res.append(self.predict([h[b : b + 1] for h in vl], state_t[b : b + 1] if state_t is not None else None,
                                    bias[b : b + 1], metas[b])[0])
        return np.stack(res)  # (B, K, H+, 2)


def change(xy, ref):
    """mean L2 over the horizon, averaged over noises: (V, K, T, 2) vs (K, T, 2) -> (V,)"""
    return np.linalg.norm(xy - ref[None], axis=-1).mean(-1).mean(-1)


def frame_layout(model, gi):
    view = gi["view_ids"][0]
    cur_v = int(view.max())
    merge = int(model.qwen_vl_interface.processor.image_processor.merge_size)
    _, h, w = (int(x) for x in gi["image_grid_thw"].view(-1, 3)[cur_v])
    pos = torch.nonzero(view == cur_v).squeeze(-1)  # sequence positions of the current-frame tokens (raster order)
    return pos, h // merge, w // merge


def blocks(gh, gw, b):
    ids = np.arange(gh * gw).reshape(gh, gw)
    return [ids[i : i + b, j : j + b].ravel() for i in range(0, gh, b) for j in range(0, gw, b)], \
        (math.ceil(gh / b), math.ceil(gw / b))


def regions3(gh, gw):
    ys, xs = np.linspace(0, gh, 4).round().astype(int), np.linspace(0, gw, 4).round().astype(int)
    ids = np.arange(gh * gw).reshape(gh, gw)
    return [ids[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].ravel() for i in range(3) for j in range(3)]


def occlude_image(img, r):
    im = img.copy()
    W, H = im.size
    i, j = divmod(r, 3)
    box = (round(j * W / 3), round(i * H / 3), round((j + 1) * W / 3), round((i + 1) * H / 3))
    im.paste((128, 128, 128), box)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--v1", required=True)
    p.add_argument("--v2", required=True)
    p.add_argument("--eye", required=True)
    p.add_argument("--eval_index", required=True)
    p.add_argument("--extra_index", default="", help="another eval index; samples of --extra_clips are taken from it")
    p.add_argument("--extra_clips", default="", help="comma-separated clip id prefixes to add")
    p.add_argument("--per_class", type=int, default=16)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--noises", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16)  # noqa: E731

    index = json.load(open(args.eval_index))
    samples = [s for m in ("left", "right", "straight") for s in [x for x in index if x["maneuver"] == m][: args.per_class]]
    if args.extra_clips:
        extra = json.load(open(args.extra_index or args.eval_index))
        for pre in args.extra_clips.split(","):
            samples += [s for s in extra if s["clip_id"].startswith(pre) and s not in samples]
    print(f"{len(samples)} samples on {device}", flush=True)

    results = {s_i: {"clip": s["clip_id"], "t0": s["t0"], "maneuver": s["maneuver"]} for s_i, s in enumerate(samples)}
    feats = {}
    for name, ckpt in (("v1", args.v1), ("v2", args.v2), ("eye", args.eye)):
        cfg = OmegaConf.load(str(Path(ckpt) / "config.yaml"))
        torch.manual_seed(0)
        model = build_framework(cfg).float()
        load_weights(model, ckpt, print)
        model = model.to(device).eval()
        ds = build_dataset(cfg, ckpt, samples, args.data_root)
        g = torch.Generator().manual_seed(1)
        H, A = int(cfg.framework.action_model.action_horizon), int(cfg.framework.action_model.action_dim)
        run = Runner(model, ds, device, torch.randn(args.noises, H, A, generator=g).to(device))
        gate = getattr(model, "gate_enabled", False)
        print(f"== {name}: gate {gate}", flush=True)
        feat_list, gate_cache = [], []
        with torch.no_grad(), autocast():
            for s_i in range(len(samples)):
                ex = ds[s_i]
                meta = ex["meta"]
                vl, attn, gi, state_t = run.encode([ex])
                pos, gh, gw = frame_layout(model, gi)
                keep = model._gate_keep(gi, attn) if gate else None
                base = model._cross_attention_bias(attn, keep, vl[0].dtype)  # (1, L): the policy as deployed
                blk, bshape = blocks(gh, gw, args.block)
                reg = regions3(gh, gw)
                rows = [base[0]]
                for ids in blk + reg:
                    r = base[0].clone()
                    r[pos[torch.as_tensor(ids, device=pos.device)]] = float("-inf")
                    rows.append(r)
                xy = run.predict(vl, state_t, torch.stack(rows), meta)
                ref = xy[0]
                ch = change(xy[1:], ref)
                gt = meta["action_xy"]
                rec = results[s_i].setdefault(name, {})
                rec["ade"] = float(np.linalg.norm(ref - gt[None], axis=-1).mean())
                rec["dep_map"] = ch[: len(blk)].reshape(bshape).tolist()
                rec["dit_region"] = ch[len(blk):].tolist()
                # (2) pixel occlusion through the whole model
                variants = [dict(ex, image=list(ex["image"][:-1]) + [occlude_image(ex["image"][-1], r)]) for r in range(9)]
                pix = run.predict_batch(variants, [meta] * 9,
                                        extra_bias=(lambda gi_, a_: model._gate_keep(gi_, a_)) if gate else None)
                rec["pixel_region"] = change(pix, ref).tolist()
                # (4) mask strength / removal controls
                vis = torch.nonzero(gi["view_ids"][0] >= 0).squeeze(-1)
                plain = model._cross_attention_bias(attn, None, vl[0].dtype)[0]
                ctrl = {"none": plain.clone()}
                all_off = plain.clone(); all_off[vis] = float("-inf"); ctrl["all_visual_removed"] = all_off
                rnd = plain.clone(); gen = torch.Generator().manual_seed(s_i)
                drop = vis[torch.rand(len(vis), generator=gen) >= 0.2]; rnd[drop] = float("-inf"); ctrl["random_hard_20"] = rnd
                if gate:
                    _, logits = model._gate_keep(gi, attn, return_logits=True)
                    frames = gi["view_ids"][0][vis]
                    for c in (2.0, 4.0, 8.0):
                        m_ = torch.empty_like(logits)
                        for v in frames.unique():
                            sel = frames == v
                            m_[sel] = budgeted_keep(logits[sel], float(cfg.framework.eye_gate.budget), c)
                        r = plain.clone(); r[vis] = torch.log(m_.clamp(min=1e-30)).to(r.dtype); ctrl[f"contrast_{c:g}"] = r
                    hard = plain.clone()
                    for v in frames.unique():
                        sel = torch.nonzero(frames == v).squeeze(-1)
                        k = max(1, int(round(0.2 * len(sel))))
                        off = sel[logits[sel].argsort(descending=True)[k:]]
                        hard[vis[off]] = float("-inf")
                    ctrl["hard_top_20"] = hard
                    m_cur = keep[0][pos].float().cpu().numpy()
                    rec["keep_map"] = m_cur.reshape(gh, gw).tolist()
                    bm = np.array([m_cur[ids].mean() for ids in blk])
                    rec["mask_vs_own_dep"] = spearman(bm, ch[: len(blk)])
                    if "v2" in results[s_i]:
                        rec["mask_vs_v2_dep"] = spearman(bm, np.array(results[s_i]["v2"]["dep_map"]).ravel())
                    gate_cache.append((gi["final_hidden"][0][vis].float().cpu(), gi["final_hidden"][0][(~(gi["view_ids"][0] >= 0)) & attn[0]].float().cpu(), pos.cpu(), vis.cpu()))
                cx = run.predict(vl, state_t, torch.stack(list(ctrl.values())), meta)
                rec["controls"] = {k: {"change_vs_none": float(change(cx[i : i + 1], cx[0])[0]),
                                       "ade": float(np.linalg.norm(cx[i] - gt[None], axis=-1).mean())}
                                   for i, k in enumerate(ctrl)}
                if name in ("v1", "v2"):
                    h = gi["final_hidden"][0][pos].float()
                    feat_list.append((h.cpu(), gh, gw))
                if s_i == 0 or (s_i + 1) % 10 == 0:
                    print(f"  {name} sample {s_i + 1}/{len(samples)}", flush=True)
            if feat_list:  # (5) gate-input features
                X = torch.cat([f for f, _, _ in feat_list])
                Z = (X - X.mean(0)) / (X.std(0) + 1e-5)
                sv = torch.linalg.svdvals(Z.double())
                var = (sv**2 / (sv**2).sum()).numpy()
                adj, rnd_ = [], []
                off = 0
                for f, gh_, gw_ in feat_list:
                    z = Z[off : off + len(f)].reshape(gh_, gw_, -1)
                    off += len(f)
                    zn = torch.nn.functional.normalize(z, dim=-1)
                    adj.append((zn[:, 1:] * zn[:, :-1]).sum(-1).mean().item())
                    flat = zn.reshape(-1, zn.shape[-1])
                    perm = torch.randperm(len(flat))
                    rnd_.append((flat * flat[perm]).sum(-1).mean().item())
                feats[name] = {"pc_var_top1": float(var[:1].sum()), "pc_var_top5": float(var[:5].sum()),
                               "pc_var_top20": float(var[:20].sum()), "adjacent_cos": float(np.mean(adj)),
                               "random_pair_cos": float(np.mean(rnd_))}
            if gate and gate_cache:  # (6) q contribution
                n = min(16, len(gate_cache))
                corr = []
                for b in range(n):
                    hb, tb, posb, visb = gate_cache[b]
                    own = model.eye_gate(hb.to(device), tb.to(device)).float().cpu().numpy()
                    cur = np.isin(visb.numpy(), posb.numpy())
                    for c in range(n):
                        if c != b:
                            other = model.eye_gate(hb.to(device), gate_cache[c][1].to(device)).float().cpu().numpy()
                            corr.append(spearman(own[cur], other[cur]))
                feats["eye_q_swap_rank_corr"] = float(np.nanmean(corr))
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()
        json.dump({"samples": results, "features": feats}, open(out / "summary.json", "w"))

    # figures + summary
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_v2 = build_dataset(OmegaConf.load(str(Path(args.v2) / "config.yaml")), args.v2, samples, args.data_root)
    for s_i, r in results.items():
        img = ds_v2[s_i]["image"][-1]
        fig, ax = plt.subplots(1, 5, figsize=(25, 3.6))
        ax[0].imshow(img); ax[0].set_title(f"{r['maneuver']} {r['clip'][:8]} t0={r['t0'] * 1e-6:.1f}s")
        for k, (key, title) in enumerate([("v1", "v1 baseline dependence"), ("v2", "v2 baseline dependence"),
                                          ("eye", "eye dependence (gate on)")]):
            im = ax[k + 1 if k < 2 else 4].imshow(np.array(r[key]["dep_map"]), cmap="inferno")
            ax[k + 1 if k < 2 else 4].set_title(f"{title} (m), max {np.max(r[key]['dep_map']):.2f}")
            plt.colorbar(im, ax=ax[k + 1 if k < 2 else 4], fraction=0.03)
        im = ax[3].imshow(np.array(r["eye"]["keep_map"]), cmap="viridis", vmin=0, vmax=1)
        ax[3].set_title(f"eye keep m (abs), rho vs own dep {r['eye'].get('mask_vs_own_dep', float('nan')):.2f}")
        plt.colorbar(im, ax=ax[3], fraction=0.03)
        for a in ax:
            a.axis("off")
        fig.savefig(out / "figs" / f"{s_i:02d}_{r['maneuver']}_{r['clip'][:8]}.png", dpi=70, bbox_inches="tight")
        plt.close(fig)

    def agg(model_key, f):
        vals = [f(r[model_key]) for r in results.values() if model_key in r]
        return float(np.nanmean(vals))

    summary = {"features": feats}
    for k in ("v1", "v2", "eye"):
        summary[k] = {
            "ade": agg(k, lambda r: r["ade"]),
            "dep_map_max": agg(k, lambda r: np.max(r["dep_map"])),
            "dep_map_mean": agg(k, lambda r: np.mean(r["dep_map"])),
            "dit_region_mean": agg(k, lambda r: np.mean(r["dit_region"])),
            "pixel_region_mean": agg(k, lambda r: np.mean(r["pixel_region"])),
            "controls": {c: {m: agg(k, lambda r, c=c, m=m: r["controls"][c][m]) for m in ("change_vs_none", "ade")}
                         for c in results[0][k]["controls"]},
        }
    summary["eye"]["mask_vs_own_dep"] = agg("eye", lambda r: r.get("mask_vs_own_dep", np.nan))
    summary["eye"]["mask_vs_v2_dep"] = agg("eye", lambda r: r.get("mask_vs_v2_dep", np.nan))
    json.dump({"summary": summary, "samples": results}, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
