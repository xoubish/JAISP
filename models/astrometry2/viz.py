"""Visualization helpers for the standalone astrometry2 pipeline."""

from pathlib import Path
from typing import Dict, Optional

import math

import matplotlib.pyplot as plt
import numpy as np


def _downsample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_n, dtype=np.int64)


def _sky_to_display_vectors(item: Dict, dra_arcsec: np.ndarray, ddec_arcsec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dra_arcsec = np.asarray(dra_arcsec, dtype=np.float32)
    ddec_arcsec = np.asarray(ddec_arcsec, dtype=np.float32)
    hdr = item.get("vis_wcs_header")
    if hdr is not None:
        try:
            from astropy.wcs import WCS

            vis = np.asarray(item["vis_image"], dtype=np.float32)
            h, w = int(vis.shape[0]), int(vis.shape[1])
            cx = 0.5 * max(0.0, float(w - 1))
            cy = 0.5 * max(0.0, float(h - 1))
            wcs = WCS(hdr)
            ra0, dec0 = wcs.wcs_pix2world([[cx, cy]], 0)[0]
            ra_x, dec_x = wcs.wcs_pix2world([[cx + 1.0, cy]], 0)[0]
            ra_y, dec_y = wcs.wcs_pix2world([[cx, cy + 1.0]], 0)[0]
            cos_dec = np.cos(np.deg2rad(dec0))
            pix_to_sky = np.array(
                [
                    [(ra_x - ra0) * cos_dec * 3600.0, (ra_y - ra0) * cos_dec * 3600.0],
                    [(dec_x - dec0) * 3600.0, (dec_y - dec0) * 3600.0],
                ],
                dtype=np.float32,
            )
            sky_to_pix = np.linalg.pinv(pix_to_sky)
            flat = np.stack([dra_arcsec.reshape(-1), ddec_arcsec.reshape(-1)], axis=0)
            disp = sky_to_pix @ flat
            return disp[0].reshape(dra_arcsec.shape), disp[1].reshape(dra_arcsec.shape)
        except Exception:
            pass

    mag = np.hypot(dra_arcsec, ddec_arcsec)
    scale = max(1e-4, float(np.percentile(mag, 95)))
    return (dra_arcsec / scale) * 8.0, (ddec_arcsec / scale) * 8.0


def _mesh_extent(vis_shape: tuple[int, int]) -> list[float]:
    h, w = int(vis_shape[0]), int(vis_shape[1])
    return [0.0, float(w), 0.0, float(h)]


def _outlined_quiver(ax, x, y, u, v, color: str = "black", halo: str = "white", width: float = 0.0024) -> None:
    ax.quiver(
        x,
        y,
        u,
        v,
        color=halo,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=width * 2.0,
        alpha=0.9,
    )
    ax.quiver(
        x,
        y,
        u,
        v,
        color=color,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=width,
        alpha=0.95,
    )


def make_tile_diagnostic_figure(
    item: Dict,
    tile_id: str,
    rubin_band: str,
    input_bands: list[str],
) -> plt.Figure:
    """
    3×3 tile-level diagnostic figure.

    Row 1 — The concordance product itself:
      (0,0) VIS image + NN-predicted offset magnitudes at source anchors
      (0,1) Solved ΔRA* field (mas, coolwarm)
      (0,2) Solved ΔDec field (mas, coolwarm)

    Row 2 — Quality checks:
      (1,0) Solved field magnitude + direction arrows overlaid on VIS
      (1,1) Sky-coord scatter of predicted offsets, colored by σ
            → shows if there is a systematic offset direction
      (1,2) Uncertainty calibration: |pred − smooth field| vs predicted σ
            → points near y=x mean σ correctly reflects actual scatter

    Row 3 — Model performance statistics:
      (2,0) Distribution of predicted ΔRA* and ΔDec in mas
      (2,1) Raw WCS measurement vs NN prediction (y≈x → NN confirms match)
      (2,2) Summary stats
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    vis = np.asarray(item["vis_image"], dtype=np.float32)
    mesh = item["mesh"]
    raw_anchor_xy = np.asarray(item.get("raw_anchor_xy", item["vis_xy"]), dtype=np.float32)
    vis_anchor_xy = np.asarray(item.get("vis_anchor_xy", item["vis_xy"]), dtype=np.float32)
    vis_xy = np.asarray(item["vis_xy"], dtype=np.float32)
    raw_offsets = np.asarray(item["raw_offsets"], dtype=np.float32)
    pred_offsets = np.asarray(item["pred_offsets"], dtype=np.float32)
    sigma_mas = np.asarray(item["sigma_arcsec"], dtype=np.float32) * 1000.0

    dra = np.asarray(mesh["dra"], dtype=np.float32)
    ddec = np.asarray(mesh["ddec"], dtype=np.float32)
    x_mesh = np.asarray(mesh["x_mesh"], dtype=np.float32)
    y_mesh = np.asarray(mesh["y_mesh"], dtype=np.float32)

    pred_mas = pred_offsets * 1000.0
    raw_mas = raw_offsets * 1000.0
    resid_mas = raw_mas - pred_mas
    mesh_mag_mas = np.hypot(dra, ddec) * 1000.0
    resid_mag_mas = np.hypot(resid_mas[:, 0], resid_mas[:, 1])

    # Evaluate the smooth field at each anchor — used for the summary stats only.
    resid_from_field_mas = np.zeros(len(vis_xy), dtype=np.float32)
    valid_calib = np.zeros(len(vis_xy), dtype=bool)
    if _has_scipy and len(x_mesh) > 1 and len(y_mesh) > 1:
        try:
            interp_dra = RegularGridInterpolator(
                (y_mesh.astype(float), x_mesh.astype(float)), dra.astype(float),
                method="linear", bounds_error=False, fill_value=np.nan)
            interp_ddec = RegularGridInterpolator(
                (y_mesh.astype(float), x_mesh.astype(float)), ddec.astype(float),
                method="linear", bounds_error=False, fill_value=np.nan)
            pts = np.stack([vis_xy[:, 1].astype(float), vis_xy[:, 0].astype(float)], axis=1)
            fd = interp_dra(pts).astype(np.float32)
            fdd = interp_ddec(pts).astype(np.float32)
            resid_from_field_mas = np.hypot(
                (pred_offsets[:, 0] - fd) * 1000.0,
                (pred_offsets[:, 1] - fdd) * 1000.0,
            )
            valid_calib = np.isfinite(resid_from_field_mas)
        except Exception as exc:
            print(f'[viz] Field interpolation failed: {exc}')

    # Display setup
    p1v, p99v = np.percentile(vis, [1, 99])
    if not np.isfinite(p1v) or p1v >= p99v:
        p1v, p99v = float(vis.min()), float(vis.max())
    vis_extent = _mesh_extent(vis.shape)
    comp_lim_mas = max(12.0, float(
        np.percentile(np.abs(np.concatenate([dra.ravel(), ddec.ravel()])), 95)) * 1000.0)
    dstep = int(round(float(x_mesh[1] - x_mesh[0]))) if len(x_mesh) > 1 else 1

    # Quiver arrows for field magnitude panel
    yy, xx = np.meshgrid(y_mesh, x_mesh, indexing="ij")
    qstep = max(1, min(dra.shape[0], dra.shape[1]) // 14)
    xx_q = xx[::qstep, ::qstep]
    yy_q = yy[::qstep, ::qstep]
    qu_dx, qu_dy = _sky_to_display_vectors(item, dra[::qstep, ::qstep], ddec[::qstep, ::qstep])
    qu_div = max(1.0, float(np.percentile(np.hypot(qu_dx, qu_dy), 95)) / 18.0)
    qu_u = qu_dx / qu_div
    qu_v = qu_dy / qu_div

    keep_s = _downsample_indices(len(vis_xy), 700)
    keep_l = _downsample_indices(len(vis_xy), 1600)

    title_fs = 10
    label_fs = 9
    tick_fs = 8

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), constrained_layout=True)

    # ── Row 1: Anchor coverage + field magnitude + calibration ─────────────

    # (0,0) VIS + NN predicted offset magnitudes at source positions
    ax = axes[0, 0]
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1v, vmax=p99v, aspect="auto")
    keep_raw = _downsample_indices(len(raw_anchor_xy), 1200)
    keep_vis = _downsample_indices(len(vis_anchor_xy), 1200)
    plot_mag = np.hypot(pred_mas[keep_s, 0], pred_mas[keep_s, 1])
    vmax_mag = max(1.0, float(np.percentile(plot_mag, 95))) if len(plot_mag) > 0 else 1.0
    if len(keep_raw) > 0:
        ax.scatter(
            raw_anchor_xy[keep_raw, 0], raw_anchor_xy[keep_raw, 1],
            s=18, facecolors="none", edgecolors="deepskyblue",
            linewidths=0.7, alpha=0.45, label="raw detector anchors",
        )
    if len(keep_vis) > 0:
        ax.scatter(
            vis_anchor_xy[keep_vis, 0], vis_anchor_xy[keep_vis, 1],
            s=10, facecolors="none", edgecolors="springgreen",
            linewidths=0.6, alpha=0.45, label="VIS-kept anchors",
        )
    sc = ax.scatter(
        vis_xy[keep_s, 0], vis_xy[keep_s, 1],
        c=plot_mag,
        s=14, cmap="plasma", alpha=0.95, vmin=0, vmax=vmax_mag,
        edgecolors="white", linewidths=0.25, label="target-band anchors",
    )
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="|pred| (mas)")
    counts_txt = (
        f"raw={len(raw_anchor_xy)}  "
        f"VIS-kept={len(vis_anchor_xy)}  "
        f"target-kept={len(vis_xy)}"
    )
    ax.text(
        0.02, 0.98, counts_txt,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
    )
    ax.set_xlim(0, vis.shape[1]); ax.set_ylim(0, vis.shape[0])
    ax.set_title("NN predicted corrections at source anchors", fontsize=title_fs)
    ax.set_xlabel("VIS x (px)", fontsize=label_fs); ax.set_ylabel("VIS y (px)", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.legend(fontsize=7, loc="lower right")

    # (0,1) Solved field magnitude + direction arrows (most useful summary)
    ax = axes[0, 1]
    ax.set_facecolor("white")
    im = ax.imshow(mesh_mag_mas, origin="lower", extent=vis_extent, cmap="magma",
                   vmin=0, vmax=float(np.percentile(mesh_mag_mas, 95)),
                   interpolation="bilinear", alpha=0.95, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="|field| (mas)")
    arrow_mag = np.hypot(qu_dx, qu_dy).ravel() / qu_div  # un-scaled magnitude
    vmax_arrow = float(np.percentile(np.hypot(dra, ddec) * 1000.0, 95))
    ax.quiver(
        xx_q.ravel(), yy_q.ravel(), qu_u.ravel(), qu_v.ravel(),
        arrow_mag,
        cmap="plasma", clim=(0, max(vmax_arrow, 1.0)),
        angles="xy", scale_units="xy", scale=1,
        width=0.0032, alpha=0.85,
    )
    ax.set_xlim(0, vis.shape[1]); ax.set_ylim(0, vis.shape[0])
    ax.set_title("Solved field magnitude + direction", fontsize=title_fs)
    ax.set_xlabel("VIS x (px)", fontsize=label_fs); ax.set_ylabel("VIS y (px)", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)

    # (0,2) Uncertainty calibration: does high σ correlate with the NN
    # disagreeing more with the raw WCS measurement?
    # x = predicted σ,  y = |raw_offset − pred_offset|
    # Points near y=x → σ correctly tracks how much the NN deviates from raw.
    # Points above y=x → model is overconfident.
    # Points below y=x → model is underconfident.
    ax = axes[0, 2]
    raw_pred_diff_mas = np.hypot(
        (raw_offsets[:, 0] - pred_offsets[:, 0]) * 1000.0,
        (raw_offsets[:, 1] - pred_offsets[:, 1]) * 1000.0,
    )
    keep_c = _downsample_indices(len(sigma_mas), 600)
    s_c = sigma_mas[keep_c]
    r_c = raw_pred_diff_mas[keep_c]
    ax.scatter(s_c, r_c, s=8, alpha=0.45, color="steelblue")
    smax = max(float(np.percentile(s_c, 97)), 1.0)
    rmax = max(float(np.percentile(r_c, 97)), 1.0)
    diag_max = max(smax, rmax)
    ax.plot([0, diag_max], [0, diag_max], "k--", lw=1.2, label="y = x  (perfect calibration)")
    ax.set_xlim(0, smax * 1.1); ax.set_ylim(0, rmax * 1.2)
    ax.legend(fontsize=8)
    ax.set_xlabel("Predicted σ (mas)", fontsize=label_fs)
    ax.set_ylabel("|raw WCS − NN pred| (mas)", fontsize=label_fs)
    ax.set_title("Uncertainty calibration\n(does σ track |raw − pred|?)", fontsize=title_fs)
    ax.tick_params(labelsize=tick_fs)

    # ── Row 2: Residual stats + summary ─────────────────────────────────────

    # (1,0) Distribution of residual offset components in mas.
    # Both components should be centered near zero if corrections are good.
    ax = axes[1, 0]
    all_vals = np.concatenate([resid_mas[:, 0], resid_mas[:, 1]])
    lim_h = max(20.0, float(np.percentile(np.abs(all_vals), 98)))
    bins = np.linspace(-lim_h, lim_h, 20)
    ax.hist(resid_mas[:, 0], bins=bins, alpha=0.7, label="ΔRA*", color="tab:blue", density=True)
    ax.hist(resid_mas[:, 1], bins=bins, alpha=0.7, label="ΔDec", color="tab:orange", density=True)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Residual offset component (mas)", fontsize=label_fs)
    ax.set_ylabel("Density", fontsize=label_fs)
    ax.set_title("Distribution of residual offset components", fontsize=title_fs)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=tick_fs)

    # (1,1) Raw WCS measurements vs NN corrections.
    # Points near y=x → the NN is reproducing the WCS-matched measurements.
    # Scatter around y=x → NN is smoothing noisy WCS measurements (expected).
    # Systematic offset from y=x → NN is correcting a bias in the raw WCS.
    ax = axes[1, 1]
    lim3 = max(20.0, float(np.percentile(np.abs(np.concatenate([
        raw_mas[keep_l, 0], raw_mas[keep_l, 1],
        pred_mas[keep_l, 0], pred_mas[keep_l, 1],
    ])), 99)))
    ax.scatter(raw_mas[keep_l, 0], pred_mas[keep_l, 0], s=5, alpha=0.3,
               color="tab:blue", label="ΔRA*")
    ax.scatter(raw_mas[keep_l, 1], pred_mas[keep_l, 1], s=5, alpha=0.3,
               color="tab:orange", label="ΔDec")
    ax.plot([-lim3, lim3], [-lim3, lim3], "k--", lw=1.0, label="y = x")
    ax.set_xlim(-lim3, lim3); ax.set_ylim(-lim3, lim3)
    ax.set_xlabel("Raw WCS measurement (mas)", fontsize=label_fs)
    ax.set_ylabel("NN correction (mas)", fontsize=label_fs)
    ax.set_title("Raw WCS vs NN corrections\n(y ≈ x → NN confirms match)", fontsize=title_fs)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=tick_fs)

    # (1,2) Summary stats text
    ax = axes[1, 2]
    ax.axis("off")
    n = len(vis_xy)
    raw_n = len(raw_anchor_xy)
    vis_n = len(vis_anchor_xy)
    raw_med = float(np.median(np.hypot(raw_mas[:, 0], raw_mas[:, 1])))
    pred_med = float(np.median(np.hypot(pred_mas[:, 0], pred_mas[:, 1])))
    resid_offset_med = float(np.median(resid_mag_mas))
    sigma_med = float(np.median(sigma_mas))
    field_p95 = float(np.percentile(mesh_mag_mas, 95))
    resid_field_med = (float(np.median(resid_from_field_mas[valid_calib]))
                       if valid_calib.sum() > 0 else float("nan"))
    txt = (
        f"Tile:   {tile_id}\n"
        f"Band:   {rubin_band} → euclid_VIS\n"
        f"Input:  {', '.join(input_bands)}\n\n"
        f"Raw anchors:          {raw_n}\n"
        f"VIS-kept anchors:     {vis_n}\n"
        f"Target-kept anchors:  {n}\n\n"
        f"Raw |offset| median:\n  {raw_med:.1f} mas\n\n"
        f"NN correction |offset| median:\n  {pred_med:.1f} mas\n\n"
        f"Residual |offset| median:\n  {resid_offset_med:.1f} mas\n\n"
        f"σ median:\n  {sigma_med:.1f} mas\n\n"
        f"|pred − field| median:\n  {resid_field_med:.1f} mas\n\n"
        f"Field |offset| p95:\n  {field_p95:.1f} mas\n\n"
        f"Mesh: {dra.shape[0]}×{dra.shape[1]} nodes\n"
        f"DSTEP: {dstep} VIS px ({dstep * 0.1:.1f}\")"
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, va="top",
            fontfamily="monospace", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", alpha=0.95))
    ax.set_title("Summary", fontsize=9)

    fig.suptitle(
        f"{tile_id}  |  {rubin_band} → euclid_VIS  |  astrometry preview",
        fontsize=11, y=0.995,
    )
    return fig


def _short_band_label(band: str) -> str:
    if band.startswith("rubin_"):
        return band.split("_", 1)[1]
    if band.startswith("nisp_"):
        return band.split("_", 1)[1]
    return band


def _prep_axes(n: int, ncols: int):
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.0), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    return fig, axes, nrows, ncols


def make_band_montage(
    items: Dict[str, Dict],
    plot_kind: str,
    ncols: int = 5,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Create a montage with one subplot per band for a given plot type."""
    bands = list(items.keys())
    if not bands:
        return None

    fig, axes, nrows, ncols = _prep_axes(len(bands), ncols)
    flat_axes = axes.ravel()

    # Shared ranges across bands.
    if plot_kind == "anchors":
        all_pred = []
        for item in items.values():
            pred_offsets = np.asarray(item["pred_offsets"], dtype=np.float32)
            all_pred.append(np.hypot(pred_offsets[:, 0], pred_offsets[:, 1]) * 1000.0)
        vmax = float(np.percentile(np.concatenate(all_pred), 95)) if all_pred else 1.0
        cmap = "plasma"
    elif plot_kind == "field":
        all_mag = []
        for item in items.values():
            mesh = item["mesh"]
            dra = np.asarray(mesh["dra"], dtype=np.float32)
            ddec = np.asarray(mesh["ddec"], dtype=np.float32)
            all_mag.append(np.hypot(dra, ddec).ravel() * 1000.0)
        vmax = float(np.percentile(np.concatenate(all_mag), 95)) if all_mag else 1.0
        cmap = "magma"
    elif plot_kind == "residual_hist":
        all_vals = []
        for item in items.values():
            raw = np.asarray(item["raw_offsets"], dtype=np.float32) * 1000.0
            pred = np.asarray(item["pred_offsets"], dtype=np.float32) * 1000.0
            resid = raw - pred
            all_vals.append(resid[:, 0]); all_vals.append(resid[:, 1])
        all_vals = np.concatenate(all_vals) if all_vals else np.zeros((1,), dtype=np.float32)
        lim = max(20.0, float(np.percentile(np.abs(all_vals), 98)))
        bins = np.linspace(-lim, lim, 18)
    elif plot_kind == "raw_vs_pred":
        all_vals = []
        for item in items.values():
            raw = np.asarray(item["raw_offsets"], dtype=np.float32) * 1000.0
            pred = np.asarray(item["pred_offsets"], dtype=np.float32) * 1000.0
            all_vals.append(raw[:, 0]); all_vals.append(raw[:, 1])
            all_vals.append(pred[:, 0]); all_vals.append(pred[:, 1])
        all_vals = np.concatenate(all_vals) if all_vals else np.zeros((1,), dtype=np.float32)
        lim = max(20.0, float(np.percentile(np.abs(all_vals), 99)))
    else:
        raise ValueError(f"Unknown plot_kind: {plot_kind}")

    # Render each band.
    mappable = None
    for i, band in enumerate(bands):
        ax = flat_axes[i]
        item = items[band]

        if plot_kind == "anchors":
            vis = np.asarray(item["vis_image"], dtype=np.float32)
            p1v, p99v = np.percentile(vis, [1, 99])
            if not np.isfinite(p1v) or p1v >= p99v:
                p1v, p99v = float(vis.min()), float(vis.max())
            ax.imshow(vis, origin="lower", cmap="gray", vmin=p1v, vmax=p99v, aspect="auto")
            vis_xy = np.asarray(item["vis_xy"], dtype=np.float32)
            raw_anchor_xy = np.asarray(item.get("raw_anchor_xy", vis_xy), dtype=np.float32)
            vis_anchor_xy = np.asarray(item.get("vis_anchor_xy", vis_xy), dtype=np.float32)
            pred_offsets = np.asarray(item["pred_offsets"], dtype=np.float32) * 1000.0
            plot_mag = np.hypot(pred_offsets[:, 0], pred_offsets[:, 1])
            keep_raw = _downsample_indices(len(raw_anchor_xy), 600)
            keep_vis = _downsample_indices(len(vis_anchor_xy), 600)
            keep = _downsample_indices(len(vis_xy), 700)
            if len(keep_raw) > 0:
                ax.scatter(raw_anchor_xy[keep_raw, 0], raw_anchor_xy[keep_raw, 1],
                           s=8, facecolors="none", edgecolors="deepskyblue",
                           linewidths=0.5, alpha=0.35)
            if len(keep_vis) > 0:
                ax.scatter(vis_anchor_xy[keep_vis, 0], vis_anchor_xy[keep_vis, 1],
                           s=6, facecolors="none", edgecolors="springgreen",
                           linewidths=0.4, alpha=0.35)
            sc = ax.scatter(vis_xy[keep, 0], vis_xy[keep, 1],
                            c=plot_mag[keep], s=8, cmap=cmap, vmin=0, vmax=vmax,
                            edgecolors="white", linewidths=0.2, alpha=0.9)
            mappable = sc
            ax.set_xlim(0, vis.shape[1]); ax.set_ylim(0, vis.shape[0])

        elif plot_kind == "field":
            mesh = item["mesh"]
            dra = np.asarray(mesh["dra"], dtype=np.float32)
            ddec = np.asarray(mesh["ddec"], dtype=np.float32)
            x_mesh = np.asarray(mesh["x_mesh"], dtype=np.float32)
            y_mesh = np.asarray(mesh["y_mesh"], dtype=np.float32)
            mesh_mag_mas = np.hypot(dra, ddec) * 1000.0
            vis_shape = item.get("vis_shape") or item["vis_image"].shape
            extent = _mesh_extent(vis_shape)
            im = ax.imshow(mesh_mag_mas, origin="lower", extent=extent, cmap=cmap,
                           vmin=0, vmax=vmax, interpolation="bilinear", alpha=0.95, aspect="auto")
            mappable = im
            yy, xx = np.meshgrid(y_mesh, x_mesh, indexing="ij")
            qstep = max(1, min(dra.shape[0], dra.shape[1]) // 10)
            xx_q = xx[::qstep, ::qstep]
            yy_q = yy[::qstep, ::qstep]
            qu_dx, qu_dy = _sky_to_display_vectors(item, dra[::qstep, ::qstep], ddec[::qstep, ::qstep])
            qu_div = max(1.0, float(np.percentile(np.hypot(qu_dx, qu_dy), 95)) / 18.0)
            qu_u = qu_dx / qu_div
            qu_v = qu_dy / qu_div
            ax.quiver(
                xx_q.ravel(), yy_q.ravel(), qu_u.ravel(), qu_v.ravel(),
                angles="xy", scale_units="xy", scale=1,
                width=0.0025, alpha=0.7, color="black",
            )
            ax.set_xlim(0, vis_shape[1]); ax.set_ylim(0, vis_shape[0])

        elif plot_kind == "residual_hist":
            raw = np.asarray(item["raw_offsets"], dtype=np.float32) * 1000.0
            pred = np.asarray(item["pred_offsets"], dtype=np.float32) * 1000.0
            resid = raw - pred
            ax.hist(resid[:, 0], bins=bins, alpha=0.7, color="tab:blue", density=True)
            ax.hist(resid[:, 1], bins=bins, alpha=0.7, color="tab:orange", density=True)
            ax.axvline(0, color="gray", lw=0.7)

        elif plot_kind == "raw_vs_pred":
            raw = np.asarray(item["raw_offsets"], dtype=np.float32) * 1000.0
            pred = np.asarray(item["pred_offsets"], dtype=np.float32) * 1000.0
            keep = _downsample_indices(len(raw), 700)
            ax.scatter(raw[keep, 0], pred[keep, 0], s=6, alpha=0.35, color="tab:blue")
            ax.scatter(raw[keep, 1], pred[keep, 1], s=6, alpha=0.35, color="tab:orange")
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

        # Title per band
        ax.set_title(_short_band_label(band), fontsize=9)
        ax.tick_params(labelsize=7)

        # Reduce clutter for montage
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Turn off unused axes
    for j in range(len(bands), len(flat_axes)):
        flat_axes[j].axis("off")

    # Add shared colorbar for map-like plots.
    if plot_kind in ("anchors", "field") and mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("|pred| (mas)" if plot_kind == "anchors" else "|field| (mas)")

    if title:
        fig.suptitle(title, fontsize=11)
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


# ──────────────────────────────────────────────────────────────────────────────
# Global concordance visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_global_concordance(
    result: dict,
    sources: dict,
    output_path: str,
    target_band: str = 'r',
) -> str:
    """
    3×3 diagnostic figure for the global sky-coordinate concordance field.

    Row 1 — The field itself:
      (0,0) Field magnitude (mas) + source positions scattered on top
      (0,1) ΔRA* component (mas, coolwarm)
      (0,2) ΔDec component (mas, coolwarm)

    Row 2 — Quality / coverage:
      (1,0) Quiver arrows on magnitude background (direction + strength)
      (1,1) Per-source offset scatter colored by predicted σ
      (1,2) Coverage map (arcsec to nearest source anchor)

    Row 3 — Statistics:
      (2,0) Histogram of predicted offset components
      (2,1) Raw WCS measurements vs NN predictions
      (2,2) Summary text

    Axes are in RA / Dec degrees with standard astronomical orientation
    (RA increases to the left, Dec increases upward).
    """
    mesh     = result['mesh']
    ra_grid  = np.asarray(result['ra_grid'],  dtype=np.float64)   # [W] RA  of mesh cols
    dec_grid = np.asarray(result['dec_grid'], dtype=np.float64)   # [H] Dec of mesh rows
    dra      = np.asarray(mesh['dra'],  dtype=np.float32) * 1000.0   # → mas
    ddec     = np.asarray(mesh['ddec'], dtype=np.float32) * 1000.0   # → mas
    cov      = np.asarray(mesh['coverage'], dtype=np.float32) if mesh.get('coverage') is not None else None

    ra_src   = np.asarray(sources['ra'],  dtype=np.float64)
    dec_src  = np.asarray(sources['dec'], dtype=np.float64)
    pred     = np.asarray(sources['pred_offsets'], dtype=np.float32) * 1000.0   # → mas
    raw      = np.asarray(sources['raw_offsets'],  dtype=np.float32) * 1000.0
    sigma    = np.asarray(sources['sigma'], dtype=np.float32) * 1000.0

    pred_mag = np.hypot(pred[:, 0], pred[:, 1])
    raw_mag  = np.hypot(raw[:, 0],  raw[:, 1])
    field_mag = np.hypot(dra, ddec)

    # imshow extent: [left, right, bottom, top]
    # RA increases left → use [ra_max, ra_min, dec_min, dec_max]
    ra_min, ra_max   = float(ra_grid.min()),  float(ra_grid.max())
    dec_min, dec_max = float(dec_grid.min()), float(dec_grid.max())
    extent_sky = [ra_max, ra_min, dec_min, dec_max]   # RA left-to-right reversed

    comp_lim = max(20.0, float(np.percentile(np.abs(np.concatenate([dra.ravel(), ddec.ravel()])), 97)))

    # Subsampled quiver grid (≤15×15 arrows)
    qstep_h = max(1, dra.shape[0] // 15)
    qstep_w = max(1, dra.shape[1] // 15)
    ra_q    = ra_grid[::qstep_w]
    dec_q   = dec_grid[::qstep_h]
    dra_q   = dra[::qstep_h, ::qstep_w]
    ddec_q  = ddec[::qstep_h, ::qstep_w]
    mag_q   = np.hypot(dra_q, ddec_q)
    rr_q, dd_q = np.meshgrid(ra_q, dec_q)

    # Scale quiver so median arrow spans ~3% of the field width in RA
    field_width_deg = ra_max - ra_min
    med_mag_deg = float(np.median(mag_q[mag_q > 0])) / 3600.0 if (mag_q > 0).any() else 1e-4
    arrow_scale = field_width_deg * 0.03 / max(med_mag_deg, 1e-9)
    # Convert arcsec offsets to degrees for display
    # Note: quiver u = ΔRA in deg (positive = East = left), so negate for display
    qu_u = -(dra_q  / 3600.0) * arrow_scale   # negate: RA increases left
    qu_v =  (ddec_q / 3600.0) * arrow_scale

    keep = _downsample_indices(len(ra_src), 1500)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.subplots_adjust(hspace=0.44, wspace=0.30)

    # ── Row 1: Field maps ───────────────────────────────────────────────────

    # (0,0) Field magnitude + source positions
    ax = axes[0, 0]
    im = ax.imshow(field_mag, origin='lower', extent=extent_sky, cmap='magma',
                   vmin=0, vmax=float(np.percentile(field_mag, 97)), aspect='auto',
                   interpolation='bilinear')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='|offset| (mas)')
    sc = ax.scatter(ra_src[keep], dec_src[keep], c=pred_mag[keep],
                    s=8, cmap='magma', vmin=0, vmax=float(np.percentile(field_mag, 97)),
                    edgecolors='white', linewidths=0.3, alpha=0.85)
    ax.set_xlim(ra_max, ra_min)   # RA increases left
    ax.set_ylim(dec_min, dec_max)
    ax.set_xlabel('RA (deg)', fontsize=8); ax.set_ylabel('Dec (deg)', fontsize=8)
    ax.set_title('Global field magnitude + source anchors', fontsize=9)

    # (0,1) ΔRA* component
    ax = axes[0, 1]
    im = ax.imshow(dra, origin='lower', extent=extent_sky, cmap='coolwarm',
                   vmin=-comp_lim, vmax=comp_lim, aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='mas')
    ax.set_xlim(ra_max, ra_min); ax.set_ylim(dec_min, dec_max)
    ax.set_xlabel('RA (deg)', fontsize=8); ax.set_ylabel('Dec (deg)', fontsize=8)
    ax.set_title('Solved ΔRA* field  (East-West correction)', fontsize=9)

    # (0,2) ΔDec component
    ax = axes[0, 2]
    im = ax.imshow(ddec, origin='lower', extent=extent_sky, cmap='coolwarm',
                   vmin=-comp_lim, vmax=comp_lim, aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='mas')
    ax.set_xlim(ra_max, ra_min); ax.set_ylim(dec_min, dec_max)
    ax.set_xlabel('RA (deg)', fontsize=8); ax.set_ylabel('Dec (deg)', fontsize=8)
    ax.set_title('Solved ΔDec field  (North-South correction)', fontsize=9)

    # ── Row 2: Quality / coverage ───────────────────────────────────────────

    # (1,0) Quiver arrows on magnitude background
    ax = axes[1, 0]
    ax.imshow(field_mag, origin='lower', extent=extent_sky, cmap='Greys',
              vmin=0, vmax=float(np.percentile(field_mag, 97)), aspect='auto',
              interpolation='bilinear', alpha=0.6)
    q = ax.quiver(
        rr_q.ravel(), dd_q.ravel(), qu_u.ravel(), qu_v.ravel(),
        mag_q.ravel(),
        cmap='plasma', clim=(0, comp_lim),
        angles='xy', scale_units='xy', scale=1,
        width=0.003, alpha=0.9,
    )
    plt.colorbar(q, ax=ax, fraction=0.03, pad=0.02, label='|offset| (mas)')
    ax.set_xlim(ra_max, ra_min); ax.set_ylim(dec_min, dec_max)
    ax.set_xlabel('RA (deg)', fontsize=8); ax.set_ylabel('Dec (deg)', fontsize=8)
    ax.set_title('Offset direction field\n(arrow color = magnitude, length display-scaled)', fontsize=9)

    # (1,1) Per-source scatter colored by σ
    ax = axes[1, 1]
    vmax_s = float(np.percentile(sigma, 95)) if len(sigma) > 0 else 100.0
    sc = ax.scatter(pred[keep, 0], pred[keep, 1],
                    c=sigma[keep], s=8, cmap='plasma', alpha=0.65,
                    vmin=0, vmax=vmax_s)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='σ (mas)')
    lim2 = max(30.0, float(np.percentile(np.abs(pred), 99)) * 1.2)
    ax.axhline(0, color='lightgray', lw=0.8); ax.axvline(0, color='lightgray', lw=0.8)
    ax.set_xlim(-lim2, lim2); ax.set_ylim(-lim2, lim2)
    ax.set_aspect('equal')
    ax.set_xlabel('ΔRA* (mas)  →  East', fontsize=9)
    ax.set_ylabel('ΔDec (mas)  →  North', fontsize=9)
    ax.set_title('Per-source offset scatter (color = σ)\ntight cluster = consistent field', fontsize=9)

    # (1,2) Coverage map
    ax = axes[1, 2]
    if cov is not None:
        im = ax.imshow(cov, origin='lower', extent=extent_sky, cmap='viridis_r',
                       vmin=0, aspect='auto', interpolation='bilinear')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='dist to nearest source (arcsec)')
        ax.set_title('Coverage map\n(high = extrapolating, threshold ~150")', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No coverage map', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.set_title('Coverage map', fontsize=9)
    ax.set_xlim(ra_max, ra_min); ax.set_ylim(dec_min, dec_max)
    ax.set_xlabel('RA (deg)', fontsize=8); ax.set_ylabel('Dec (deg)', fontsize=8)

    # ── Row 3: Statistics ───────────────────────────────────────────────────

    # (2,0) Histogram of offset components
    ax = axes[2, 0]
    lim_h = max(20.0, float(np.percentile(np.abs(pred), 98)))
    bins = np.linspace(-lim_h, lim_h, 30)
    ax.hist(pred[:, 0], bins=bins, alpha=0.7, label='ΔRA*', color='tab:blue',  density=True)
    ax.hist(pred[:, 1], bins=bins, alpha=0.7, label='ΔDec', color='tab:orange', density=True)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel('Predicted offset component (mas)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Distribution of predicted offset components\n(all sources, all tiles)', fontsize=9)
    ax.legend(fontsize=8)

    # (2,1) Raw WCS vs NN predictions
    ax = axes[2, 1]
    keep2 = _downsample_indices(len(ra_src), 2000)
    lim3 = max(20.0, float(np.percentile(np.abs(np.concatenate([
        raw[keep2, 0], raw[keep2, 1], pred[keep2, 0], pred[keep2, 1],
    ])), 99)))
    ax.scatter(raw[keep2, 0], pred[keep2, 0], s=5, alpha=0.3, color='tab:blue',   label='ΔRA*')
    ax.scatter(raw[keep2, 1], pred[keep2, 1], s=5, alpha=0.3, color='tab:orange', label='ΔDec')
    ax.plot([-lim3, lim3], [-lim3, lim3], 'k--', lw=1.0, label='y = x')
    ax.set_xlim(-lim3, lim3); ax.set_ylim(-lim3, lim3)
    ax.set_xlabel('Raw WCS measurement (mas)', fontsize=9)
    ax.set_ylabel('NN prediction (mas)', fontsize=9)
    ax.set_title('Raw WCS vs NN predictions\n(y ≈ x → NN confirms match; scatter → noise suppression)', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')

    # (2,2) Summary text
    ax = axes[2, 2]
    ax.axis('off')
    n_tiles = len(set(sources['tile_ids']))
    n_src   = len(ra_src)
    raw_med  = float(np.median(raw_mag))
    pred_med = float(np.median(pred_mag))
    sig_med  = float(np.median(sigma))
    field_p95 = float(np.percentile(field_mag, 95))
    txt = (
        f"Global concordance field\n"
        f"Band:   {target_band} → euclid_VIS\n"
        f"Solve:  single sky-coord field\n\n"
        f"Tiles processed:     {n_tiles}\n"
        f"Total sources:       {n_src}\n\n"
        f"Mesh shape: {dra.shape[0]}×{dra.shape[1]}\n"
        f"Mesh step:  {result['dstep_arcsec']:.1f}\"\n"
        f"Footprint:  {result['field_shape_arcsec'][1]:.0f}\" × {result['field_shape_arcsec'][0]:.0f}\"\n\n"
        f"Raw |offset| median:\n  {raw_med:.1f} mas\n\n"
        f"NN |offset| median:\n  {pred_med:.1f} mas\n\n"
        f"σ median:  {sig_med:.1f} mas\n\n"
        f"Field p95: {field_p95:.1f} mas\n"
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, va='top',
            fontfamily='monospace', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', alpha=0.95))
    ax.set_title('Summary', fontsize=9)

    band_label = target_band if target_band.startswith('rubin_') else f'rubin_{target_band}'
    fig.suptitle(
        f'Global sky-coord concordance  |  {band_label} → euclid_VIS  |  {n_src} sources, {n_tiles} tiles',
        fontsize=11, y=0.998,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved global concordance plot: {out_path}')
    return str(out_path)
