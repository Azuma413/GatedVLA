# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Open-loop evaluation utilities for driving policies: displacement metrics and BEV trajectory plots."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

MANEUVERS = ("left", "right", "straight")


def displacement_metrics(
    pred_xy: np.ndarray, gt_xy: np.ndarray, maneuvers: Sequence[str], dt: float = 0.1
) -> Dict[str, float]:
    """ADE / FDE (m) over the horizon, L2 at intermediate horizons, lateral / longitudinal final errors,
    overall and per maneuver.  pred_xy, gt_xy: (N, H, 2) in the ego frame (x forward, y left)."""
    err = np.linalg.norm(pred_xy - gt_xy, axis=-1)  # (N, H)
    horizon = pred_xy.shape[1]
    maneuvers = np.asarray(maneuvers)

    def _block(mask: np.ndarray, prefix: str) -> Dict[str, float]:
        if not mask.any():
            return {}
        e = err[mask]
        out = {
            f"{prefix}ADE": float(e.mean()),
            f"{prefix}FDE": float(e[:, -1].mean()),
            f"{prefix}lat_err_final": float(np.abs(pred_xy[mask, -1, 1] - gt_xy[mask, -1, 1]).mean()),
            f"{prefix}lon_err_final": float(np.abs(pred_xy[mask, -1, 0] - gt_xy[mask, -1, 0]).mean()),
            f"{prefix}miss_rate_2m": float((e.max(axis=1) > 2.0).mean()),
        }
        for k in range(4, horizon, 5):  # L2 @ 0.5 s, 1.0 s, ...
            out[f"{prefix}L2@{(k + 1) * dt:.1f}s"] = float(e[:, k].mean())
        return out

    metrics = {"num_samples": float(len(err)), **_block(np.ones(len(err), bool), "")}
    for m in MANEUVERS:
        metrics.update(_block(maneuvers == m, f"{m}/"))
    return metrics


def anchor_headings(anchors: np.ndarray) -> np.ndarray:
    """Final heading (deg, left +) of each anchor (K, H, 2), from its last three steps."""
    d = anchors[:, -1] - anchors[:, -4]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def anchor_classes(anchors: np.ndarray, turn_deg: float = 15.0, stop_m: float = 1.0) -> np.ndarray:
    """left / right / straight / stop label of each anchor (same ±15° rule as the eval maneuvers)."""
    heading = anchor_headings(anchors)
    dist = np.linalg.norm(anchors[:, -1], axis=-1)
    return np.where(dist < stop_m, "stop", np.where(heading > turn_deg, "left",
                                                    np.where(heading < -turn_deg, "right", "straight")))


def anchor_metrics(anchors: np.ndarray, gt_xy: np.ndarray, ids: np.ndarray, weights: np.ndarray,
                   top1: np.ndarray, reference: np.ndarray, maneuvers: Sequence[str]) -> Dict[str, float]:
    """Decision-anchor diagnostics: classification vs. the GT-nearest anchor, how mixed the top-N anchors are,
    and the error of the reference trajectory alone (before the residual flow).

    ``anchors``: (K, H, 2) trajectories, or (N, K, H, 2) per-sample trajectories (control-space anchors integrated
    from each sample's current speed)."""
    per_sample = anchors.ndim == 4
    anchors_n = anchors if per_sample else np.broadcast_to(anchors, (len(gt_xy), *anchors.shape))
    nearest = np.linalg.norm(gt_xy[:, None] - anchors_n, axis=-1).mean(-1).argmin(1)
    if per_sample:
        top_classes = np.stack([anchor_classes(a)[i] for a, i in zip(anchors, ids)])
        top_heading = np.stack([anchor_headings(a)[i] for a, i in zip(anchors, ids)])
    else:
        top_classes, top_heading = anchor_classes(anchors)[ids], anchor_headings(anchors)[ids]  # (N, top_n)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(-1)
    ref_err = np.linalg.norm(reference - gt_xy, axis=-1)
    out = {
        "anchor/top1_acc": float((top1 == nearest).mean()),
        "anchor/topn_recall": float((ids == nearest[:, None]).any(-1).mean()),
        "anchor/topn_mixed_class_frac": float(np.mean([len(set(c)) > 1 for c in top_classes])),
        "anchor/topn_left_and_right_frac": float(np.mean([("left" in c) and ("right" in c) for c in top_classes])),
        "anchor/topn_heading_spread_deg": float((top_heading.max(-1) - top_heading.min(-1)).mean()),
        "anchor/weight_top1": float(weights[:, 0].mean()),
        "anchor/weight_entropy": float(entropy.mean()),
        "anchor/reference_ADE": float(ref_err.mean()),
        "anchor/reference_FDE": float(ref_err[:, -1].mean()),
    }
    maneuvers = np.asarray(maneuvers)
    for m in MANEUVERS:
        mask = maneuvers == m
        if mask.any():
            out[f"anchor/{m}/top1_acc"] = float((top1[mask] == nearest[mask]).mean())
            out[f"anchor/{m}/reference_ADE"] = float(ref_err[mask].mean())
    return out


def fig_to_image(fig, dpi: int = 90, canvas: tuple = (1700, 520)) -> Image.Image:
    """Render a matplotlib figure to PIL with a tight bounding box (multi-line titles are not clipped), then
    fit it onto a fixed white canvas so that every logged image has the same size."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image.thumbnail(canvas)
    out = Image.new("RGB", canvas, "white")
    out.paste(image, ((canvas[0] - image.width) // 2, (canvas[1] - image.height) // 2))
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(np.float64)


def _rank_corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation between matching rows of a and b (N, P)."""
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra**2).sum(1) * (rb**2).sum(1) + 1e-12)


def turn_metrics(pred: np.ndarray, gt: np.ndarray, maneuvers: List[str]) -> dict:
    """Whether the predicted path follows the ground truth laterally: correlation of the final lateral offset (all
    samples), per turn class the share with the right turn direction, the median pred / GT final lateral ratio, and
    the spread of the predicted final lateral offset."""
    yp, yg = pred[:, -1, 1], gt[:, -1, 1]
    out = {"turn/corr_final_y": float(np.corrcoef(yp, yg)[0, 1]) if yp.std() > 0 else 0.0,
           "turn/pred_final_y_std": float(yp.std())}
    man = np.asarray(maneuvers)
    for m in ("left", "right"):
        sel = (man == m) & (np.abs(yg) > 0.5)
        if sel.any():
            out[f"turn/{m}_direction_acc"] = float((np.sign(yp[sel]) == np.sign(yg[sel])).mean())
            out[f"turn/{m}_ratio_median"] = float(np.median(yp[sel] / yg[sel]))
    return out


def longitudinal_metrics(pred: np.ndarray, gt: np.ndarray, dt: float = 0.1) -> dict:
    """Along-track behaviour on the samples whose ground truth decelerates (end speed < start speed - 2 m/s) or comes
    to a stop (start > 2 m/s, end < 0.5 m/s): mean signed final forward error (pred - GT; > 0 = overshoots, i.e. stops
    too late) and mean absolute error. Speeds from the first / last ground-truth steps."""
    v_start = np.linalg.norm(gt[:, 1] - gt[:, 0], axis=-1) / dt
    v_end = np.linalg.norm(gt[:, -1] - gt[:, -2], axis=-1) / dt
    err = pred[:, -1, 0] - gt[:, -1, 0]
    out = {}
    for name, sel in (("decel", v_end < v_start - 2.0), ("stop", (v_start > 2.0) & (v_end < 0.5))):
        out[f"long/{name}_n"] = int(sel.sum())
        if sel.any():
            out[f"long/{name}_fwd_err_mean"] = float(err[sel].mean())
            out[f"long/{name}_fwd_err_abs"] = float(np.abs(err[sel]).mean())
    return out


def gate_diagnostics(records: List[dict]) -> dict:
    """Collapse / attention metrics of the visual gate over the eval set (current frame).

    mask_pairwise_rankcorr: mean Spearman correlation of the masks of two different samples (1 = one fixed pattern).
    mask_row_r2 / mask_col_r2: share of the mask variance explained by the per-row / per-column mean pattern.
    mask_lr_diff/<maneuver>: mean keep on the image's left half minus the right half.
    attn_text_mass[_textdrop]: DiT cross-attention mass on the text tokens.
    rankcorr_mask_attn[_textdrop]: Spearman correlation between a sample's mask and its DiT attention map."""
    out = {}
    keeps = [r["keep_cur"] for r in records if "keep_cur" in r]
    if len(keeps) < 2 or len({k.shape for k in keeps}) != 1:
        return out
    k = np.stack(keeps)  # (N, h, w)
    flat = k.reshape(len(k), -1)
    r = _ranks(flat)
    r -= r.mean(1, keepdims=True)
    r /= np.sqrt((r**2).sum(1, keepdims=True)) + 1e-12
    c = r @ r.T
    n = len(k)
    out["mask_pairwise_rankcorr"] = float((c.sum() - np.trace(c)) / (n * (n - 1)))
    row_pattern = k.mean((0, 2))  # (h,)
    resid = flat - flat.mean(1, keepdims=True)
    fit = np.repeat(row_pattern[None, :, None], k.shape[2], 2).reshape(1, -1) - row_pattern.mean()
    out["mask_row_r2"] = float(1 - ((resid - fit) ** 2).sum() / max((resid**2).sum(), 1e-12))
    col_pattern = k.mean((0, 1))  # (w,)
    fit = np.repeat(col_pattern[None, None, :], k.shape[1], 1).reshape(1, -1) - col_pattern.mean()
    out["mask_col_r2"] = float(1 - ((resid - fit) ** 2).sum() / max((resid**2).sum(), 1e-12))
    # keep mass on the image's left half minus the right half (image left = ego left), per maneuver
    half = k.shape[2] // 2
    lr = k[:, :, :half].mean((1, 2)) - k[:, :, k.shape[2] - half :].mean((1, 2))
    for m in sorted({x["maneuver"] for x in records if "keep_cur" in x}):
        sel = np.array([x["maneuver"] == m for x in records if "keep_cur" in x])
        out[f"mask_lr_diff/{m}"] = float(lr[sel].mean())
    for suffix in ("", "_textdrop"):
        if f"attn_text_mass{suffix}" in records[0]:
            out[f"attn_text_mass{suffix}"] = float(np.mean([x[f"attn_text_mass{suffix}"] for x in records]))
        key = "attn_cur" if suffix == "" else "attn_textdrop_cur"
        if key in records[0]:
            a = np.stack([x[key] for x in records]).reshape(len(records), -1)
            out[f"rankcorr_mask_attn{suffix}"] = float(_rank_corr_rows(flat, a).mean())
    return out


def plot_bev(
    meta: dict,
    pred_xy: np.ndarray,
    image: Optional[Image.Image] = None,
    extra_preds: Optional[List[np.ndarray]] = None,
    title: str = "",
    keep_map: Optional[np.ndarray] = None,
    extra_maps: Optional[List] = None,
    reference: Optional[np.ndarray] = None,
    candidate_anchors: Optional[np.ndarray] = None,
):
    """Front image (left), optionally the visual gate's keep map over the frame (middle), and the
    bird's-eye view (right) with past / GT / predicted ego trajectories.

    BEV axes: up = ego forward (+x), right = ego right (-y)."""
    extra_maps = list(extra_maps or []) if image is not None else []
    panels = ([image] if image is not None else []) + ([keep_map] if image is not None and keep_map is not None else [])
    panels += [m for _, m in extra_maps]
    ratios = [1.6] * len(panels) + [1]
    panel_w = 8.3 if len(panels) <= 2 else 6.0
    fig, axes = plt.subplots(1, len(ratios), figsize=(5.2 + panel_w * len(panels), 5.2), gridspec_kw={"width_ratios": ratios})
    axes = np.atleast_1d(axes)
    if image is not None:
        axes[0].imshow(image)
        axes[0].set_axis_off()
        axes[0].set_title("front camera (t0)", fontsize=9)
    if image is not None and keep_map is not None:
        # keep values on an absolute 0..1 colour scale over a dimmed grey copy of the frame (readable at night too)
        grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        grey = np.clip(grey / max(float(np.percentile(grey, 99)), 1e-3), 0.0, 1.0)  # stretch dark frames
        axes[1].imshow(grey, cmap="gray", vmin=0.0, vmax=1.0)
        heat = axes[1].imshow(np.asarray(keep_map, dtype=np.float32), cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.6,
                              interpolation="nearest", extent=(-0.5, image.width - 0.5, image.height - 0.5, -0.5))
        fig.colorbar(heat, ax=axes[1], fraction=0.03, pad=0.01)
        axes[1].set_axis_off()
        axes[1].set_title(f"keep m on current frame, absolute scale (min {float(np.min(keep_map)):.2f} / mean "
                          f"{float(np.mean(keep_map)):.2f} / max {float(np.max(keep_map)):.2f})", fontsize=9)
    for k, (name, values) in enumerate(extra_maps):
        ax_m = axes[len(panels) - len(extra_maps) + k]
        values = np.asarray(values, dtype=np.float32)
        grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        grey = np.clip(grey / max(float(np.percentile(grey, 99)), 1e-3), 0.0, 1.0)
        ax_m.imshow(grey, cmap="gray", vmin=0.0, vmax=1.0)
        lo, hi = (float(values.min()), float(values.max())) if "score" in name else (0.0, float(values.max()))
        im = ax_m.imshow(values, cmap="magma" if "attention" in name else "coolwarm", vmin=lo, vmax=hi if hi > lo else lo + 1,
                         alpha=0.6, interpolation="nearest", extent=(-0.5, image.width - 0.5, image.height - 0.5, -0.5))
        fig.colorbar(im, ax=ax_m, fraction=0.03, pad=0.01)
        ax_m.set_axis_off()
        ax_m.set_title(f"{name} (sum on frame {float(values.sum()):.3f})" if "attention" in name else name, fontsize=9)
    ax = axes[-1]

    def _plot(xy, **kw):
        xy = np.asarray(xy)
        if len(xy):
            ax.plot(-xy[:, 1], xy[:, 0], **kw)

    _plot(meta["past_xy"], color="0.55", lw=1.5, label="past (2 s)")
    _plot(meta["future_xy"], color="tab:green", lw=1.0, ls="--", alpha=0.6, label="GT future (5 s)")
    for k, xy in enumerate(candidate_anchors if candidate_anchors is not None else []):
        _plot(xy, color="tab:purple", lw=0.8, alpha=0.3, label="top-N anchors" if k == 0 else None)
    if reference is not None:
        _plot(reference, color="tab:purple", lw=1.8, ls="--", label="anchor reference")
    for k, xy in enumerate(extra_preds or []):
        _plot(xy, color="tab:red", lw=0.8, alpha=0.25, label="pred samples" if k == 0 else None)
    _plot(meta["action_xy"], color="tab:green", lw=2.5, marker="o", ms=3, label="GT (horizon)")
    _plot(pred_xy, color="tab:red", lw=2.0, marker="o", ms=3, label="prediction")
    tgt = np.asarray(meta["target_xy"])
    ax.scatter([-tgt[1]], [tgt[0]], marker="*", s=120, color="tab:orange", zorder=5, label="target point")
    # approximate ego footprint (rig origin at the rear axle)
    ax.add_patch(plt.Rectangle((-0.95, -1.0), 1.9, 4.8, fill=False, color="k", lw=1.2))

    pts = np.concatenate([np.asarray(meta["past_xy"]).reshape(-1, 2), np.asarray(meta["future_xy"]).reshape(-1, 2),
                          np.asarray(pred_xy).reshape(-1, 2), tgt[None]]
                         + ([np.asarray(reference).reshape(-1, 2)] if reference is not None else []), axis=0)
    span = max(15.0, float(np.abs(pts).max()) * 1.1)
    xs, ys = -pts[:, 1], pts[:, 0]
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    half = max(span / 2, (xs.max() - xs.min()) / 2 + 3, (ys.max() - ys.min()) / 2 + 3)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("lateral [m] (right +)")
    ax.set_ylabel("forward [m]")
    ax.legend(fontsize=7, loc="best")
    err = np.linalg.norm(np.asarray(pred_xy) - np.asarray(meta["action_xy"]), axis=-1)
    ax.set_title(f"{title}\nADE {err.mean():.2f} m  FDE {err[-1]:.2f} m", fontsize=9)
    fig.tight_layout()
    return fig
