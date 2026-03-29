"""
Shared utilities for turning Rubin<->VIS source matches into smooth teacher fields.

These helpers are used by:
  - the explicit non-neural baseline
  - the multiband neural trainer, where the fitted field acts as a dense target
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import RBFInterpolator

from jaisp_dataset_v4 import RUBIN_BAND_ORDER, _to_float32
from source_matching import (
    build_detection_image,
    detect_sources,
    match_sources_wcs,
    refine_centroids_in_band,
)


def normalize_band_name(name: str) -> str:
    band = str(name).strip().lower()
    if band in RUBIN_BAND_ORDER:
        band = f"rubin_{band}"
    if not band.startswith("rubin_"):
        band = f"rubin_{band}"
    short = band.split("_", 1)[1]
    if short not in RUBIN_BAND_ORDER:
        raise ValueError(f"Invalid Rubin band: {name}")
    return band


def normalize_band_list(names: List[str]) -> List[str]:
    raw = [str(x).strip().lower() for x in names if str(x).strip()]
    if not raw:
        return []
    if any(x == "all" for x in raw):
        return [f"rubin_{b}" for b in RUBIN_BAND_ORDER]
    out = []
    seen = set()
    for name in raw:
        band = normalize_band_name(name)
        if band not in seen:
            seen.add(band)
            out.append(band)
    return out


def _fit_affine(coords_norm: np.ndarray, values: np.ndarray) -> np.ndarray:
    x = coords_norm[:, 0]
    y = coords_norm[:, 1]
    design = np.stack([np.ones_like(x), x, y], axis=1)
    coeffs = []
    for k in range(values.shape[1]):
        c, _, _, _ = np.linalg.lstsq(design, values[:, k], rcond=None)
        coeffs.append(c.astype(np.float64))
    return np.stack(coeffs, axis=0)


def _eval_affine(coords_norm: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    x = coords_norm[:, 0]
    y = coords_norm[:, 1]
    design = np.stack([np.ones_like(x), x, y], axis=1)
    out = []
    for k in range(coeffs.shape[0]):
        out.append(design @ coeffs[k])
    return np.stack(out, axis=1)


def fit_smooth_field(
    vis_xy: np.ndarray,
    offsets: np.ndarray,
    vis_shape: Tuple[int, int],
    dstep: int,
    smoothing: float,
    neighbors: int,
    kernel: str,
) -> Dict[str, np.ndarray]:
    """
    Fit affine + smooth RBF residual and evaluate on a VIS mesh.
    """
    height, width = int(vis_shape[0]), int(vis_shape[1])
    if vis_xy.shape[0] < 3:
        raise ValueError("Need at least 3 matches to fit a field.")

    xy = vis_xy.astype(np.float64, copy=False)
    denom_x = max(1.0, float(width - 1))
    denom_y = max(1.0, float(height - 1))
    coords_norm = np.empty_like(xy, dtype=np.float64)
    coords_norm[:, 0] = 2.0 * (xy[:, 0] / denom_x) - 1.0
    coords_norm[:, 1] = 2.0 * (xy[:, 1] / denom_y) - 1.0

    coeffs = _fit_affine(coords_norm, offsets.astype(np.float64, copy=False))
    affine_at_pts = _eval_affine(coords_norm, coeffs)
    residual = offsets.astype(np.float64, copy=False) - affine_at_pts

    nn = max(4, min(int(neighbors), int(vis_xy.shape[0])))
    rbf = RBFInterpolator(
        coords_norm,
        residual,
        kernel=kernel,
        smoothing=float(smoothing),
        neighbors=nn,
    )

    y_mesh = np.arange(0, height, int(max(1, dstep)), dtype=np.float64)
    x_mesh = np.arange(0, width, int(max(1, dstep)), dtype=np.float64)
    yy, xx = np.meshgrid(y_mesh, x_mesh, indexing="ij")
    mesh_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    mesh_norm = np.empty_like(mesh_xy, dtype=np.float64)
    mesh_norm[:, 0] = 2.0 * (mesh_xy[:, 0] / denom_x) - 1.0
    mesh_norm[:, 1] = 2.0 * (mesh_xy[:, 1] / denom_y) - 1.0

    affine_mesh = _eval_affine(mesh_norm, coeffs)
    residual_mesh = rbf(mesh_norm)
    pred = affine_mesh + residual_mesh

    mesh_h = y_mesh.size
    mesh_w = x_mesh.size
    dra = pred[:, 0].reshape(mesh_h, mesh_w).astype(np.float32)
    ddec = pred[:, 1].reshape(mesh_h, mesh_w).astype(np.float32)

    fit_at_pts = _eval_affine(coords_norm, coeffs) + rbf(coords_norm)
    point_resid = fit_at_pts - offsets.astype(np.float64, copy=False)
    point_resid_mas = np.hypot(point_resid[:, 0], point_resid[:, 1]) * 1000.0

    return {
        "dra": dra,
        "ddec": ddec,
        "x_mesh": x_mesh.astype(np.float32),
        "y_mesh": y_mesh.astype(np.float32),
        "affine_coeffs": coeffs.astype(np.float32),
        "fit_offsets": fit_at_pts.astype(np.float32),
        "point_resid_mas": point_resid_mas.astype(np.float32),
    }


def build_teacher_field(
    rubin_cube: np.ndarray,
    vis_img: np.ndarray,
    rubin_wcs: WCS,
    vis_wcs: WCS,
    rubin_band: str,
    detect_bands: List[str],
    *,
    dstep: int,
    min_matches: int,
    max_matches: int,
    max_sep_arcsec: float,
    clip_sigma: float,
    rubin_nsig: float,
    vis_nsig: float,
    rubin_smooth: float,
    vis_smooth: float,
    rubin_min_dist: int,
    vis_min_dist: int,
    max_sources_rubin: int,
    max_sources_vis: int,
    detect_clip_sigma: float,
    refine_band_centroids: bool,
    refine_radius: int,
    refine_flux_floor_sigma: float,
    rbf_smoothing: float,
    rbf_neighbors: int,
    rbf_kernel: str,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Build a smooth dense teacher field for one Rubin band against VIS.
    """
    rubin_band = normalize_band_name(rubin_band)
    idx = RUBIN_BAND_ORDER.index(rubin_band.split("_", 1)[1])
    if idx >= rubin_cube.shape[0]:
        return None

    rubin_img = np.nan_to_num(_to_float32(rubin_cube[idx]), nan=0.0)
    vis_img = np.nan_to_num(_to_float32(vis_img), nan=0.0)
    rubin_det = build_detection_image(rubin_cube, detect_bands, clip_sigma=detect_clip_sigma)

    rx, ry = detect_sources(
        rubin_det,
        nsig=rubin_nsig,
        smooth_sigma=rubin_smooth,
        min_dist=rubin_min_dist,
        max_sources=max_sources_rubin,
    )
    vx, vy = detect_sources(
        vis_img,
        nsig=vis_nsig,
        smooth_sigma=vis_smooth,
        min_dist=vis_min_dist,
        max_sources=max_sources_vis,
    )
    matched = match_sources_wcs(
        rx,
        ry,
        vx,
        vy,
        rubin_wcs,
        vis_wcs,
        max_sep_arcsec=max_sep_arcsec,
        clip_sigma=clip_sigma,
        max_matches=max_matches,
    )
    if matched["vis_xy"].shape[0] == 0:
        return None

    if refine_band_centroids:
        rubin_xy_band = refine_centroids_in_band(
            rubin_img,
            matched["rubin_xy"],
            radius=refine_radius,
            flux_floor_sigma=refine_flux_floor_sigma,
        )
        r_ra, r_dec = rubin_wcs.wcs_pix2world(rubin_xy_band[:, 0], rubin_xy_band[:, 1], 0)
        v_ra, v_dec = vis_wcs.wcs_pix2world(matched["vis_xy"][:, 0], matched["vis_xy"][:, 1], 0)
        dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
        ddec = (v_dec - r_dec) * 3600.0
        matched["rubin_xy"] = rubin_xy_band.astype(np.float32)
        matched["offsets"] = np.stack([dra, ddec], axis=1).astype(np.float32)

    nmatch = int(matched["vis_xy"].shape[0])
    if nmatch < int(min_matches):
        return None

    fit = fit_smooth_field(
        vis_xy=matched["vis_xy"],
        offsets=matched["offsets"],
        vis_shape=vis_img.shape,
        dstep=dstep,
        smoothing=rbf_smoothing,
        neighbors=rbf_neighbors,
        kernel=rbf_kernel,
    )
    return {
        "rubin_band": rubin_band,
        "matched": matched,
        "fit": fit,
        "vis_shape": tuple(int(x) for x in vis_img.shape),
    }
