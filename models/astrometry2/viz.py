"""Visualization helpers for the standalone astrometry2 pipeline."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def make_tile_diagnostic_figure(
    item: Dict,
    tile_id: str,
    rubin_band: str,
    input_bands: list[str],
) -> plt.Figure:
    vis = np.asarray(item["vis_image"], dtype=np.float32)
    mesh = item["mesh"]
    vis_xy = np.asarray(item["vis_xy"], dtype=np.float32)
    raw_offsets = np.asarray(item["raw_offsets"], dtype=np.float32)
    pred_offsets = np.asarray(item["pred_offsets"], dtype=np.float32)
    sigma_mas = np.asarray(item["sigma_arcsec"], dtype=np.float32) * 1000.0

    dra = np.asarray(mesh["dra"], dtype=np.float32)
    ddec = np.asarray(mesh["ddec"], dtype=np.float32)
    x_mesh = np.asarray(mesh["x_mesh"], dtype=np.float32)
    y_mesh = np.asarray(mesh["y_mesh"], dtype=np.float32)

    resid = pred_offsets - raw_offsets
    resid_mag_mas = np.hypot(resid[:, 0], resid[:, 1]) * 1000.0
    raw_mag_mas = np.hypot(raw_offsets[:, 0], raw_offsets[:, 1]) * 1000.0
    pred_mag_mas = np.hypot(pred_offsets[:, 0], pred_offsets[:, 1]) * 1000.0
    mesh_mag_mas = np.hypot(dra, ddec) * 1000.0

    p1, p99 = np.percentile(vis, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p1 >= p99:
        p1 = float(np.min(vis))
        p99 = float(np.max(vis))
        if p1 >= p99:
            p1, p99 = 0.0, 1.0

    yy, xx = np.meshgrid(y_mesh, x_mesh, indexing="ij")
    step = max(1, min(dra.shape[0], dra.shape[1]) // 18)
    xx_s = xx[::step, ::step]
    yy_s = yy[::step, ::step]
    qu_u = dra[::step, ::step] / 0.1
    qu_v = ddec[::step, ::step] / 0.1
    qu_scale = max(1.0, float(np.percentile(np.hypot(qu_u, qu_v), 95)))
    qu_div = max(1.0, qu_scale / 12.0)

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    sc = ax.scatter(vis_xy[:, 0], vis_xy[:, 1], c=raw_mag_mas, s=12, cmap="magma", alpha=0.9)
    ax.set_title("VIS + matched anchors (raw |offset|)")
    plt.colorbar(sc, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    ax.quiver(
        xx_s,
        yy_s,
        qu_u / qu_div,
        qu_v / qu_div,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0024,
        alpha=0.85,
    )
    ax.set_title("Predicted mesh field (quiver on VIS)")

    ax = fig.add_subplot(2, 3, 3)
    im = ax.imshow(mesh_mag_mas, origin="lower", cmap="viridis")
    ax.quiver(
        xx_s,
        yy_s,
        qu_u / qu_div,
        qu_v / qu_div,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0022,
        alpha=0.7,
    )
    ax.set_title("Predicted mesh |offset|")
    plt.colorbar(im, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    n = vis_xy.shape[0]
    keep = np.arange(n)
    if n > 200:
        keep = np.random.choice(n, 200, replace=False)
    ax.quiver(
        vis_xy[keep, 0],
        vis_xy[keep, 1],
        resid[keep, 0] * 1000.0 / 12.0,
        resid[keep, 1] * 1000.0 / 12.0,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="crimson",
        width=0.0025,
        alpha=0.8,
    )
    ax.set_xlim(0, vis.shape[1])
    ax.set_ylim(0, vis.shape[0])
    ax.set_title("Anchor residuals (pred - raw)")

    ax = fig.add_subplot(2, 3, 5)
    m = min(2000, vis_xy.shape[0])
    keep2 = np.arange(vis_xy.shape[0])
    if keep2.size > m:
        keep2 = np.random.choice(keep2, m, replace=False)
    ax.scatter(raw_offsets[keep2, 0] * 1000.0, pred_offsets[keep2, 0] * 1000.0, s=5, alpha=0.35, label="DRA*")
    ax.scatter(raw_offsets[keep2, 1] * 1000.0, pred_offsets[keep2, 1] * 1000.0, s=5, alpha=0.35, label="DDec")
    lim = max(
        20.0,
        float(np.percentile(np.abs(np.concatenate([
            raw_offsets[keep2, 0] * 1000.0,
            raw_offsets[keep2, 1] * 1000.0,
            pred_offsets[keep2, 0] * 1000.0,
            pred_offsets[keep2, 1] * 1000.0,
        ])), 99)),
    )
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Raw match (mas)")
    ax.set_ylabel("Pred local (mas)")
    ax.set_title("Anchor offsets: raw vs pred")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    summary = (
        f"tile: {tile_id}\n"
        f"target: {rubin_band} -> euclid_VIS\n"
        f"input: {', '.join(input_bands)}\n"
        f"anchors: {vis_xy.shape[0]}\n"
        f"raw |offset| median: {float(np.median(raw_mag_mas)):.1f} mas\n"
        f"pred |offset| median: {float(np.median(pred_mag_mas)):.1f} mas\n"
        f"anchor residual median: {float(np.median(resid_mag_mas)):.1f} mas\n"
        f"anchor residual p68: {float(np.percentile(resid_mag_mas, 68)):.1f} mas\n"
        f"sigma median: {float(np.median(sigma_mas)):.1f} mas\n"
        f"mesh |offset| p95: {float(np.percentile(mesh_mag_mas, 95)):.1f} mas"
    )
    ax.text(
        0.05,
        0.95,
        summary,
        transform=ax.transAxes,
        va="top",
        fontfamily="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.95),
    )

    fig.suptitle(f"{tile_id} | standalone local matcher field diagnostic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def save_tile_diagnostic(
    item: Dict,
    tile_id: str,
    rubin_band: str,
    input_bands: list[str],
    path: str,
) -> str:
    fig = make_tile_diagnostic_figure(item, tile_id, rubin_band, input_bands)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
