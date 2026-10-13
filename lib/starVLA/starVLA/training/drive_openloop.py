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


def plot_bev(
    meta: dict,
    pred_xy: np.ndarray,
    image: Optional[Image.Image] = None,
    extra_preds: Optional[List[np.ndarray]] = None,
    title: str = "",
    keep_map: Optional[np.ndarray] = None,
):
    """Front image (left), optionally the image weighted by the visual gate's keep map (middle), and the
    bird's-eye view (right) with past / GT / predicted ego trajectories.

    BEV axes: up = ego forward (+x), right = ego right (-y)."""
    panels = ([image] if image is not None else []) + ([keep_map] if image is not None and keep_map is not None else [])
    ratios = [1.6] * len(panels) + [1]
    fig, axes = plt.subplots(1, len(ratios), figsize=(5.2 + 8.3 * len(panels), 5.2), gridspec_kw={"width_ratios": ratios})
    axes = np.atleast_1d(axes)
    if image is not None:
        axes[0].imshow(image)
        axes[0].set_axis_off()
        axes[0].set_title("front camera (t0)", fontsize=9)
    if image is not None and keep_map is not None:
        # each merged token's patch scaled by its min-max normalised keep value (bright = attended most)
        lo, hi = float(np.min(keep_map)), float(np.max(keep_map))
        norm = (keep_map - lo) / (hi - lo) if hi > lo else np.ones_like(keep_map)
        mask = Image.fromarray((norm * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)
        gated = np.asarray(image, dtype=np.float32) * (np.asarray(mask, dtype=np.float32)[..., None] / 255.0)
        axes[1].imshow(gated.astype(np.uint8))
        axes[1].set_axis_off()
        axes[1].set_title(f"mask on current frame, min-max normalised (m: min {lo:.2f} / mean "
                          f"{float(np.mean(keep_map)):.2f} / max {hi:.2f})", fontsize=9)
    ax = axes[-1]

    def _plot(xy, **kw):
        xy = np.asarray(xy)
        if len(xy):
            ax.plot(-xy[:, 1], xy[:, 0], **kw)

    _plot(meta["past_xy"], color="0.55", lw=1.5, label="past (2 s)")
    _plot(meta["future_xy"], color="tab:green", lw=1.0, ls="--", alpha=0.6, label="GT future (5 s)")
    for k, xy in enumerate(extra_preds or []):
        _plot(xy, color="tab:red", lw=0.8, alpha=0.25, label="pred samples" if k == 0 else None)
    _plot(meta["action_xy"], color="tab:green", lw=2.5, marker="o", ms=3, label="GT (horizon)")
    _plot(pred_xy, color="tab:red", lw=2.0, marker="o", ms=3, label="prediction")
    tgt = np.asarray(meta["target_xy"])
    ax.scatter([-tgt[1]], [tgt[0]], marker="*", s=120, color="tab:orange", zorder=5, label="target point")
    # approximate ego footprint (rig origin at the rear axle)
    ax.add_patch(plt.Rectangle((-0.95, -1.0), 1.9, 4.8, fill=False, color="k", lw=1.2))

    pts = np.concatenate([np.asarray(meta["past_xy"]).reshape(-1, 2), np.asarray(meta["future_xy"]).reshape(-1, 2),
                          np.asarray(pred_xy).reshape(-1, 2), tgt[None]], axis=0)
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
