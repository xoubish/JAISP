"""Visualization helpers for the standalone astrometry2 pipeline."""

from pathlib import Path
from typing import Dict

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
    qu_dx, qu_dy = _sky_to_display_vectors(item, dra[::step, ::step], ddec[::step, ::step])
    qu_mag_px = np.hypot(qu_dx, qu_dy)
    qu_div = max(1.0, float(np.percentile(qu_mag_px, 95)) / 20.0)
    qu_u = qu_dx / qu_div
    qu_v = qu_dy / qu_div
    vis_extent = _mesh_extent(vis.shape)
    comp_lim_mas = max(20.0, float(np.percentile(np.abs(np.concatenate([dra.ravel(), ddec.ravel()])), 99)) * 1000.0)

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    sc = ax.scatter(vis_xy[:, 0], vis_xy[:, 1], c=raw_mag_mas, s=12, cmap="magma", alpha=0.9)
    ax.set_xlim(0, vis.shape[1])
    ax.set_ylim(0, vis.shape[0])
    ax.set_title("Raw matched-source offsets on VIS")
    plt.colorbar(sc, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    im = ax.imshow(
        mesh_mag_mas,
        origin="lower",
        extent=vis_extent,
        cmap="magma",
        alpha=0.55,
        interpolation="bilinear",
    )
    _outlined_quiver(ax, xx_s, yy_s, qu_u, qu_v, color="black", halo="white", width=0.0019)
    ax.set_xlim(0, vis.shape[1])
    ax.set_ylim(0, vis.shape[0])
    ax.set_title("Solved field |offset| with display arrows")
    plt.colorbar(im, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 3)
    im = ax.imshow(
        dra * 1000.0,
        origin="lower",
        extent=vis_extent,
        cmap="coolwarm",
        vmin=-comp_lim_mas,
        vmax=comp_lim_mas,
        interpolation="bilinear",
    )
    ax.set_title("Solved DRA* field")
    plt.colorbar(im, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(
        ddec * 1000.0,
        origin="lower",
        extent=vis_extent,
        cmap="coolwarm",
        vmin=-comp_lim_mas,
        vmax=comp_lim_mas,
        interpolation="bilinear",
    )
    ax.set_title("Solved DDec field")
    plt.colorbar(im, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    keep = _downsample_indices(int(vis_xy.shape[0]), 200)
    resid_dx, resid_dy = _sky_to_display_vectors(item, resid[keep, 0], resid[keep, 1])
    resid_mag = np.hypot(resid_dx, resid_dy)
    resid_div = max(1.0, float(np.percentile(resid_mag, 95)) / 20.0)
    _outlined_quiver(
        ax,
        vis_xy[keep, 0],
        vis_xy[keep, 1],
        resid_dx / resid_div,
        resid_dy / resid_div,
        color="crimson",
        halo="white",
        width=0.0019,
    )
    ax.set_xlim(0, vis.shape[1])
    ax.set_ylim(0, vis.shape[0])
    ax.set_title("Anchor residuals (network - raw)")

    ax = fig.add_subplot(2, 3, 6)
    keep2 = _downsample_indices(int(vis_xy.shape[0]), 2000)
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
    ax.set_xlabel("Raw matched-source offset (mas)")
    ax.set_ylabel("Network local prediction (mas)")
    ax.set_title("Anchor comparison")
    ax.legend(loc="upper left")
    summary = (
        f"tile: {tile_id}\n"
        f"target: {rubin_band} -> euclid_VIS\n"
        f"input: {', '.join(input_bands)}\n"
        f"anchors: {vis_xy.shape[0]}\n"
        f"raw = WCS-matched source measurement\n"
        f"pred = network output at same anchor\n"
        f"raw |offset| median: {float(np.median(raw_mag_mas)):.1f} mas\n"
        f"pred |offset| median: {float(np.median(pred_mag_mas)):.1f} mas\n"
        f"anchor residual median: {float(np.median(resid_mag_mas)):.1f} mas\n"
        f"anchor residual p68: {float(np.percentile(resid_mag_mas, 68)):.1f} mas\n"
        f"sigma median: {float(np.median(sigma_mas)):.1f} mas\n"
        f"field |offset| p95: {float(np.percentile(mesh_mag_mas, 95)):.1f} mas\n"
        f"arrows are display-scaled for visibility"
    )
    ax.text(
        0.03,
        0.97,
        summary,
        transform=ax.transAxes,
        va="top",
        fontfamily="monospace",
        fontsize=9,
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
