"""
TileDetectionDataset: tile loader with pseudo-label source detection.

Each item returns:
    images    : {band_name: [1, H, W]}
    rms       : {band_name: [1, H, W]}
    centroids : [M, 2]  float32  (x, y) normalised to [0, 1]
    classes   : [M]     int64    0=star, 1=galaxy
    tile_id   : str
    tile_hw   : (H, W)

When a real source catalog becomes available, replace _pseudo_labels() with a
catalog lookup and drop the classical detection dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE   = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v10 import RUBIN_BANDS, EUCLID_BANDS, ALL_BANDS
from jaisp_dataset_v10 import JAISPDatasetV10
from astrometry2.source_matching import detect_sources, build_detection_image


PSEUDO_LABEL_CACHE_VERSION = 3


def _axis_angle_distance(a: float, b: float) -> float:
    """Smallest distance between two unoriented line angles in radians."""
    return abs(((a - b + np.pi / 2.0) % np.pi) - np.pi / 2.0)


def _smooth_circular_hist(hist: np.ndarray) -> np.ndarray:
    """Light circular smoothing for angular histograms."""
    if hist.size < 5:
        return hist
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    return np.convolve(padded, kernel / kernel.sum(), mode='valid')


def _pseudo_labels(
    rubin_img: np.ndarray,   # [6, H, W]
    nsig: float,
    max_sources: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Detect sources from Rubin g+r+i coadd.

    Returns (centroids_norm [M,2], classes [M], H, W).
    """
    H, W = rubin_img.shape[1], rubin_img.shape[2]
    det = build_detection_image(rubin_img, ['rubin_g', 'rubin_r', 'rubin_i'])
    xs, ys = detect_sources(det, nsig=nsig, max_sources=max_sources)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,),   dtype=np.int64),
            H, W,
        )

    classes = np.zeros(len(xs), dtype=np.int64)  # all class 0 = source

    centroids = np.stack([
        xs / max(W - 1, 1),
        ys / max(H - 1, 1),
    ], axis=1).astype(np.float32)

    return centroids, classes, H, W


def _vis_bright_core_and_spike_mask(
    vis_img: np.ndarray,
    spike_radius: int = 40,
    min_star_area: int = 20,
    max_spike_radius: int = 400,
    spike_width: float = 3.0,
    spike_width_slope: float = 0.004,
    n_angle_bins: int = 180,
    max_spike_angles: int = 6,
    include_core: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bright-source centroids and a thin mask on VIS spike ridges.

    Bright saturated cores are found from the upper VIS percentile. Around each
    core, this estimates dominant radial spike angles from high-flux pixels in
    an annulus, then masks only narrow line segments with observed spike
    evidence. That preserves real sources in the gaps between spikes while
    vetoing the repeated false peaks that sit directly on a diffraction ridge.

    ``spike_radius`` controls the base radial search length; it still scales
    with saturated-core area and is capped by ``max_spike_radius``.
    When ``include_core`` is true, the compact saturated core is included in the
    returned mask; this is useful during classical peak filtering because the
    bright-core centroid is returned separately. Teacher/export vetoes should
    set it false so bright objects are not suppressed as artifacts.
    """
    from scipy.ndimage import center_of_mass, distance_transform_edt, find_objects, label as ndlabel

    H, W = vis_img.shape
    bright_xs = np.empty(0, dtype=np.float64)
    bright_ys = np.empty(0, dtype=np.float64)
    spike_mask = np.zeros((H, W), dtype=bool)

    nonzero_vals = vis_img[vis_img > 0]
    if len(nonzero_vals) <= 100:
        return bright_xs, bright_ys, spike_mask

    finite_vals = vis_img[np.isfinite(vis_img)]
    if finite_vals.size <= 100:
        finite_vals = nonzero_vals

    sat_thresh = np.percentile(nonzero_vals, 99.5)
    halo_vals = finite_vals[finite_vals < sat_thresh]
    if halo_vals.size <= 100:
        halo_vals = finite_vals

    med = float(np.median(finite_vals))
    mad = float(np.median(np.abs(finite_vals - med)))
    sig = max(1.4826 * mad, 1e-6)
    angle_thr = max(float(np.percentile(halo_vals, 98.5)), med + 6.0 * sig)
    line_thr = max(float(np.percentile(halo_vals, 93.0)), med + 2.5 * sig)

    sat_mask = vis_img >= sat_thresh
    labeled, n_blobs = ndlabel(sat_mask)
    if n_blobs <= 0:
        return bright_xs, bright_ys, spike_mask

    areas = np.bincount(labeled.ravel())  # areas[0] = background
    core_labels = [lb for lb in range(1, n_blobs + 1) if areas[lb] >= min_star_area]
    if not core_labels:
        return bright_xs, bright_ys, spike_mask

    coms = center_of_mass(np.ones_like(vis_img), labeled, core_labels)
    if isinstance(coms, tuple):
        coms = [coms]  # single blob: center_of_mass returns a bare tuple
    bright_ys = np.array([c[0] for c in coms], dtype=np.float64)
    bright_xs = np.array([c[1] for c in coms], dtype=np.float64)

    blob_slices = find_objects(labeled)

    angle_sep = np.deg2rad(8.0)
    min_radial_start = 6.0

    # Work per bright core so large masks stay local even for saturated stars.
    for lb, cx, cy in zip(core_labels, bright_xs, bright_ys):
        core_area = max(float(areas[lb]), 1.0)
        core_r = max(2.0, np.sqrt(core_area / np.pi))
        r_search = int(spike_radius * np.sqrt(core_area / min_star_area))
        r_search = min(max(r_search, spike_radius), max_spike_radius)

        blob_slice = blob_slices[lb - 1]
        if blob_slice is None:
            continue

        y_slice, x_slice = blob_slice
        if r_search <= 0 or spike_width <= 0:
            if include_core:
                spike_mask[y_slice, x_slice] |= labeled[y_slice, x_slice] == lb
            continue

        pad = int(np.ceil(r_search + core_r + 4))
        y0 = max(0, y_slice.start - pad)
        y1 = min(H, y_slice.stop + pad)
        x0 = max(0, x_slice.start - pad)
        x1 = min(W, x_slice.stop + pad)

        local_img = vis_img[y0:y1, x0:x1]
        local_blob = labeled[y0:y1, x0:x1] == lb
        yy, xx = np.indices(local_img.shape, dtype=np.float32)
        dx = (xx + x0) - float(cx)
        dy = (yy + y0) - float(cy)
        rr = np.hypot(dx, dy)

        # During pseudo-label generation the saturated core is masked here
        # because its centroid is returned separately. For teacher/export vetoes
        # keep the core unmasked so very bright objects can remain detections.
        core_guard = distance_transform_edt(~local_blob) <= max(2.0, min(core_r, 10.0))
        local_mask = core_guard.copy() if include_core else np.zeros_like(core_guard)

        annulus = (
            np.isfinite(local_img)
            & (rr >= max(core_r * 1.6, min_radial_start))
            & (rr <= float(r_search))
        )
        candidates = annulus & (local_img >= angle_thr)
        if candidates.sum() < 8:
            spike_mask[y0:y1, x0:x1] |= local_mask
            continue

        theta = (np.arctan2(dy[candidates], dx[candidates]) + np.pi) % np.pi
        weights = (
            np.log1p(np.maximum(local_img[candidates] - angle_thr, 0.0))
            * np.sqrt(np.maximum(rr[candidates], 1.0))
        )
        hist, edges = np.histogram(
            theta, bins=n_angle_bins, range=(0.0, np.pi), weights=weights
        )
        hist = _smooth_circular_hist(hist.astype(np.float64))
        if hist.max() <= 0:
            spike_mask[y0:y1, x0:x1] |= local_mask
            continue

        hist_med = float(np.median(hist))
        hist_mad = float(np.median(np.abs(hist - hist_med)))
        peak_floor = max(hist_med + 5.0 * 1.4826 * hist_mad, float(hist.max()) * 0.18)
        peak_idx = [
            i for i in range(hist.size)
            if hist[i] >= peak_floor
            and hist[i] >= hist[(i - 1) % hist.size]
            and hist[i] >= hist[(i + 1) % hist.size]
        ]
        peak_idx.sort(key=lambda i: hist[i], reverse=True)

        angles = []
        for i in peak_idx:
            angle = 0.5 * (edges[i] + edges[i + 1])
            if all(_axis_angle_distance(angle, old) >= angle_sep for old in angles):
                angles.append(float(angle))
            if len(angles) >= max_spike_angles:
                break

        for angle in angles:
            ca = np.cos(angle)
            sa = np.sin(angle)
            along = dx * ca + dy * sa
            perp = np.abs(dx * sa - dy * ca)
            width = spike_width + spike_width_slope * rr
            line_axis = (
                annulus
                & (perp <= width)
                & (rr >= max(core_r, min_radial_start))
            )

            # Split the unoriented axis into two rays so a one-sided spike does
            # not veto the opposite side of the star.
            for sign in (-1.0, 1.0):
                ray = line_axis & (along * sign > 0)
                evidence = ray & (local_img >= line_thr)
                if evidence.sum() < 5:
                    continue
                radial = np.abs(along[evidence])
                r_min = max(core_r, float(np.percentile(radial, 2.0)) - 5.0)
                r_max = min(float(r_search), float(np.percentile(radial, 98.0)) + 25.0)
                if r_max <= r_min:
                    continue
                local_mask |= ray & (np.abs(along) >= r_min) & (np.abs(along) <= r_max)

        spike_mask[y0:y1, x0:x1] |= local_mask

    return bright_xs, bright_ys, spike_mask


def _point_scores(vis_img: np.ndarray, xs: np.ndarray, ys: np.ndarray, half: int = 2) -> np.ndarray:
    """Local robust brightness score for ordering/deduplicating classical labels."""
    H, W = vis_img.shape
    scores = np.zeros(len(xs), dtype=np.float32)
    clean = np.nan_to_num(np.asarray(vis_img, dtype=np.float32), nan=0.0)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xi = int(np.clip(round(float(x)), 0, W - 1))
        yi = int(np.clip(round(float(y)), 0, H - 1))
        x0, x1 = max(0, xi - half), min(W, xi + half + 1)
        y0, y1 = max(0, yi - half), min(H, yi + half + 1)
        patch = clean[y0:y1, x0:x1]
        scores[i] = float(np.nanmax(patch)) if patch.size else 0.0
    return scores


def _dedupe_points_by_score(
    xs: np.ndarray,
    ys: np.ndarray,
    scores: np.ndarray,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily keep the highest-scoring point in each local neighborhood."""
    if len(xs) <= 1 or radius_px <= 0:
        return xs.astype(np.float32), ys.astype(np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
    kept: list[int] = []
    r2 = float(radius_px) ** 2
    for idx in order:
        if not kept:
            kept.append(int(idx))
            continue
        old = pts[np.asarray(kept, dtype=np.int64)]
        d2 = ((old - pts[idx]) ** 2).sum(axis=1)
        if float(d2.min()) >= r2:
            kept.append(int(idx))
    kept = np.asarray(kept, dtype=np.int64)
    return xs[kept].astype(np.float32), ys[kept].astype(np.float32)


def _peak_center_in_component(
    smooth: np.ndarray,
    comp: np.ndarray,
    threshold: float,
    half_window: int = 3,
) -> tuple[float, float]:
    """Refine the smoothed peak of a resolved VIS component."""
    yy, xx = np.nonzero(comp)
    if yy.size == 0:
        return 0.0, 0.0
    k = int(np.argmax(smooth[yy, xx]))
    py = int(yy[k])
    px = int(xx[k])
    H, W = smooth.shape
    y0, y1 = max(0, py - half_window), min(H, py + half_window + 1)
    x0, x1 = max(0, px - half_window), min(W, px + half_window + 1)
    weights = np.clip(smooth[y0:y1, x0:x1] - float(threshold), 0.0, None)
    weights *= comp[y0:y1, x0:x1].astype(np.float32)
    wsum = float(weights.sum())
    if wsum <= 0:
        return float(px), float(py)
    yyw, xxw = np.indices(weights.shape, dtype=np.float32)
    cx = float(np.sum((xxw + x0) * weights) / wsum)
    cy = float(np.sum((yyw + y0) * weights) / wsum)
    return cx, cy


def _merge_resolved_vis_components(
    vis_img: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    spike_mask: Optional[np.ndarray] = None,
    thresh_nsig: float = 2.0,
    min_area: int = 45,
    min_radius: float = 5.0,
    min_peak_snr: float = 8.0,
    match_radius_scale: float = 1.5,
    match_radius_min: float = 8.0,
    match_radius_max: float = 35.0,
    max_area_frac: float = 0.05,
    max_components: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge/recenter classical labels on resolved VIS components.

    The local-max detector is good for compact sources, but broad galaxies and
    saturated objects can produce several nearby maxima or a centroid on a
    lobe. This object-level pass replaces labels near one resolved component
    with one smoothed-core center.
    """
    from scipy.ndimage import gaussian_filter, label as ndlabel

    img = np.asarray(vis_img, dtype=np.float32)
    finite = np.isfinite(img)
    if finite.sum() <= 100:
        return xs.astype(np.float32), ys.astype(np.float32)

    clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    vals = img[finite]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sig = max(1.4826 * mad, 1e-6)
    if not np.isfinite(sig) or sig <= 0:
        return xs.astype(np.float32), ys.astype(np.float32)

    H, W = clean.shape
    smooth = gaussian_filter(clean, sigma=1.2)
    threshold = med + float(thresh_nsig) * sig
    candidate_mask = finite & (smooth >= threshold)
    if spike_mask is not None and spike_mask.shape == candidate_mask.shape:
        candidate_mask &= ~spike_mask

    labeled, n_comp = ndlabel(candidate_mask)
    if n_comp <= 0:
        return xs.astype(np.float32), ys.astype(np.float32)

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    keep = np.ones(len(pts), dtype=bool)
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1) if len(pts) else np.zeros(0, dtype=int)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1) if len(pts) else np.zeros(0, dtype=int)
    max_area = max(float(min_area), float(H * W) * float(max_area_frac))

    components = []
    for lb in range(1, n_comp + 1):
        cyy, cxx = np.nonzero(labeled == lb)
        area = int(cxx.size)
        if area < int(min_area) or area > max_area:
            continue
        vals_comp = clean[cyy, cxx]
        peak_snr = float((np.max(vals_comp) - med) / sig)
        if peak_snr < float(min_peak_snr):
            continue
        weights = np.clip(smooth[cyy, cxx] - threshold, 0.0, None).astype(np.float64)
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights, dtype=np.float64)
        wsum = float(weights.sum())
        cx_mom = float(np.sum(cxx * weights) / wsum)
        cy_mom = float(np.sum(cyy * weights) / wsum)
        moment_r = float(np.sqrt(np.sum(((cxx - cx_mom) ** 2 + (cyy - cy_mom) ** 2) * weights) / wsum))
        footprint_r = float(np.sqrt(area / np.pi))
        radius = max(moment_r, footprint_r)
        if radius < float(min_radius):
            continue
        comp = labeled == lb
        cx, cy = _peak_center_in_component(smooth, comp, threshold)
        score = peak_snr * np.sqrt(area)
        components.append((score, lb, cx, cy, radius))

    components.sort(reverse=True, key=lambda row: row[0])
    if max_components > 0:
        components = components[:int(max_components)]

    additions: list[tuple[float, float]] = []
    for _, lb, cx, cy, radius in components:
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) if len(pts) else np.zeros(0, dtype=np.float32)
        match_radius = float(np.clip(
            match_radius_scale * radius,
            match_radius_min,
            match_radius_max,
        ))
        in_comp = (labeled[yi, xi] == lb) if len(pts) else np.zeros(0, dtype=bool)
        near_comp = d <= match_radius
        replace = keep & (in_comp | near_comp)
        if replace.any():
            keep[replace] = False
            additions.append((cx, cy))
        elif not np.any(keep & (d <= match_radius)):
            additions.append((cx, cy))

    if additions:
        add = np.asarray(additions, dtype=np.float32)
        pts = np.concatenate([pts[keep], add], axis=0)
    else:
        pts = pts[keep]
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    scores = _point_scores(clean, pts[:, 0], pts[:, 1])
    return _dedupe_points_by_score(pts[:, 0], pts[:, 1], scores, radius_px=5.0)


def _sep_vis_proposals(
    vis_img: np.ndarray,
    thresh: float,
    minarea: int = 4,
    deblend_nthresh: int = 32,
    deblend_cont: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Optional SEP proposal pass; final labels are still classical-canonicalized."""
    try:
        import sep
    except Exception:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    a = np.ascontiguousarray(np.nan_to_num(
        np.asarray(vis_img, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ))
    if a.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    try:
        bkg = sep.Background(a)
        det = a - bkg.back()
    except Exception:
        det = a - float(np.median(a[np.isfinite(a)]))

    sig = max(float(np.median(np.abs(det - np.median(det)))) * 1.4826, 1e-6)
    kernel = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0],
    ], dtype=np.float32) / 16.0

    try:
        cat = sep.extract(
            np.ascontiguousarray(det.astype(np.float32)),
            thresh=float(thresh),
            err=float(sig),
            minarea=int(minarea),
            deblend_nthresh=int(deblend_nthresh),
            deblend_cont=float(deblend_cont),
            filter_kernel=kernel,
            filter_type="matched",
            clean=True,
            clean_param=1.0,
        )
    except Exception:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    if len(cat) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return (
        np.asarray(cat["x"], dtype=np.float32),
        np.asarray(cat["y"], dtype=np.float32),
    )


def _disk_mask_px(shape: tuple[int, int], xs: np.ndarray, ys: np.ndarray, radius_px: float) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    r = int(np.ceil(max(0.0, float(radius_px))))
    if r <= 0 or len(xs) == 0:
        return mask
    r2 = float(radius_px) ** 2
    for x, y in zip(xs, ys):
        x0 = max(0, int(np.floor(float(x) - r)))
        x1 = min(W, int(np.ceil(float(x) + r)) + 1)
        y0 = max(0, int(np.floor(float(y) - r)))
        y1 = min(H, int(np.ceil(float(y) + r)) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= ((xx - float(x)) ** 2 + (yy - float(y)) ** 2) <= r2
    return mask


def _proposal_ignore_mask_vis(
    vis_img: np.ndarray,
    positives_norm: np.ndarray,
    nsig: float,
    spike_mask: Optional[np.ndarray] = None,
    proposal_nsig: float = 1.8,
    ignore_radius_px: float = 5.0,
    positive_match_px: float = 6.0,
    max_proposals: int = 2000,
) -> np.ndarray:
    """Mask uncertain low-threshold proposals so missed sources are not hard negatives."""
    H, W = vis_img.shape
    proposal_xs: list[np.ndarray] = []
    proposal_ys: list[np.ndarray] = []
    for nsig_i, smooth_sigma, min_dist in [
        (min(float(nsig), float(proposal_nsig)), 0.9, 5),
        (min(float(nsig), float(proposal_nsig)), 1.6, 9),
        (min(float(nsig), float(proposal_nsig)), 2.6, 13),
    ]:
        sx, sy = detect_sources(
            vis_img,
            nsig=nsig_i,
            smooth_sigma=smooth_sigma,
            min_dist=min_dist,
            max_sources=max_proposals,
        )
        if sx.size:
            proposal_xs.append(sx)
            proposal_ys.append(sy)

    sx, sy = _sep_vis_proposals(
        vis_img,
        thresh=max(1.3, float(proposal_nsig)),
        minarea=3,
        deblend_nthresh=32,
        deblend_cont=0.02,
    )
    if sx.size:
        proposal_xs.append(sx)
        proposal_ys.append(sy)

    if not proposal_xs:
        return np.zeros((H, W), dtype=bool)

    xs = np.concatenate(proposal_xs).astype(np.float32)
    ys = np.concatenate(proposal_ys).astype(np.float32)
    xi = np.clip(np.round(xs).astype(int), 0, W - 1)
    yi = np.clip(np.round(ys).astype(int), 0, H - 1)
    keep = np.ones(len(xs), dtype=bool)
    if spike_mask is not None and spike_mask.shape == (H, W):
        keep &= ~spike_mask[yi, xi]

    positives_px = np.zeros((0, 2), dtype=np.float32)
    if positives_norm is not None and len(positives_norm) > 0:
        p = np.asarray(positives_norm, dtype=np.float32).reshape(-1, 2)
        good = np.isfinite(p).all(axis=1)
        if good.any():
            p = p[good]
            positives_px = np.stack([
                p[:, 0] * max(W - 1, 1),
                p[:, 1] * max(H - 1, 1),
            ], axis=1).astype(np.float32)

    if len(positives_px) > 0 and len(xs) > 0:
        d2 = (
            (xs[:, None] - positives_px[None, :, 0]) ** 2
            + (ys[:, None] - positives_px[None, :, 1]) ** 2
        )
        keep &= d2.min(axis=1) > float(positive_match_px) ** 2

    xs = xs[keep]
    ys = ys[keep]
    if len(xs) == 0:
        return np.zeros((H, W), dtype=bool)

    scores = _point_scores(vis_img, xs, ys)
    xs, ys = _dedupe_points_by_score(xs, ys, scores, radius_px=max(2.0, positive_match_px))
    if len(xs) > max_proposals:
        order = np.argsort(_point_scores(vis_img, xs, ys))[::-1][:int(max_proposals)]
        xs, ys = xs[order], ys[order]

    return _disk_mask_px((H, W), xs, ys, radius_px=ignore_radius_px)


def _inject_synthetic_sources(
    images: dict,
    rms: dict,
    centroids_np: np.ndarray,
    source_weights_np: np.ndarray,
    n_sources: int,
    prob: float,
    min_snr: float,
    max_snr: float,
    min_sigma_px: float,
    max_sigma_px: float,
    source_weight: float,
    min_dist_norm: float = 0.02,
    border_norm: float = 0.04,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Inject synthetic Gaussian sources with perfect labels into loaded bands."""
    if n_sources <= 0 or prob <= 0 or np.random.random() > float(prob):
        return images, centroids_np, source_weights_np

    if not images:
        return images, centroids_np, source_weights_np

    out_images = {band: img.clone() for band, img in images.items()}
    centroids = np.asarray(centroids_np, dtype=np.float32).reshape(-1, 2)
    weights = np.asarray(source_weights_np, dtype=np.float32).reshape(-1)
    existing = centroids.copy()
    added: list[list[float]] = []

    ref = out_images.get("euclid_VIS", next(iter(out_images.values())))
    ref_h, ref_w = ref.shape[-2:]
    attempts = 0
    max_attempts = max(100, int(n_sources) * 100)
    while len(added) < int(n_sources) and attempts < max_attempts:
        attempts += 1
        x = float(np.random.uniform(border_norm, 1.0 - border_norm))
        y = float(np.random.uniform(border_norm, 1.0 - border_norm))
        pt = np.array([x, y], dtype=np.float32)
        if existing.size:
            d = np.hypot(existing[:, 0] - x, existing[:, 1] - y)
            if float(d.min()) < float(min_dist_norm):
                continue
        if added:
            a = np.asarray(added, dtype=np.float32)
            d = np.hypot(a[:, 0] - x, a[:, 1] - y)
            if float(d.min()) < float(min_dist_norm):
                continue
        added.append([x, y])
        existing = np.concatenate([existing, pt[None, :]], axis=0) if existing.size else pt[None, :]

    if not added:
        return images, centroids_np, source_weights_np

    for x_norm, y_norm in added:
        sigma_vis = float(np.random.uniform(min_sigma_px, max_sigma_px))
        base_snr = float(np.exp(np.random.uniform(np.log(min_snr), np.log(max_snr))))
        mode = str(np.random.choice(["all", "vis", "nir"], p=[0.55, 0.30, 0.15]))
        theta = float(np.random.uniform(0.0, np.pi))
        axis_ratio = float(np.random.uniform(0.7, 1.0))

        for band, img in out_images.items():
            if band not in rms:
                continue
            band_is_nir = band in {"euclid_Y", "euclid_J", "euclid_H"}
            band_is_vis = band == "euclid_VIS"
            if mode == "nir" and band_is_vis:
                amp_scale = 0.0
            elif mode == "vis" and band_is_nir:
                amp_scale = float(np.random.uniform(0.0, 0.25))
            else:
                amp_scale = float(np.random.uniform(0.6, 1.4))
            if amp_scale <= 0:
                continue

            H, W = img.shape[-2:]
            x = x_norm * max(W - 1, 1)
            y = y_norm * max(H - 1, 1)
            scale = max(W / max(ref_w, 1), H / max(ref_h, 1))
            sig_x = max(0.75, sigma_vis * scale)
            sig_y = max(0.75, sigma_vis * axis_ratio * scale)
            radius = int(max(3, np.ceil(4.0 * max(sig_x, sig_y))))
            x0, x1 = max(0, int(np.floor(x)) - radius), min(W, int(np.floor(x)) + radius + 1)
            y0, y1 = max(0, int(np.floor(y)) - radius), min(H, int(np.floor(y)) + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            rms_arr = rms[band][0].detach().cpu().numpy()
            xi = int(np.clip(round(x), 0, W - 1))
            yi = int(np.clip(round(y), 0, H - 1))
            rx0, rx1 = max(0, xi - 3), min(W, xi + 4)
            ry0, ry1 = max(0, yi - 3), min(H, yi + 4)
            local = rms_arr[ry0:ry1, rx0:rx1]
            good = np.isfinite(local) & (local > 0)
            local_rms = float(np.median(local[good])) if good.any() else 1.0

            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx = xx - x
            dy = yy - y
            ca, sa = np.cos(theta), np.sin(theta)
            xp = ca * dx + sa * dy
            yp = -sa * dx + ca * dy
            stamp = np.exp(-0.5 * ((xp / sig_x) ** 2 + (yp / sig_y) ** 2)).astype(np.float32)
            stamp *= float(base_snr * amp_scale * local_rms)
            stamp_t = torch.from_numpy(stamp).to(device=img.device, dtype=img.dtype)
            img[0, y0:y1, x0:x1] = img[0, y0:y1, x0:x1] + stamp_t

    added_np = np.asarray(added, dtype=np.float32)
    centroids = np.concatenate([centroids, added_np], axis=0) if len(centroids) else added_np
    add_w = np.full((len(added_np),), float(source_weight), dtype=np.float32)
    weights = np.concatenate([weights, add_w], axis=0) if len(weights) else add_w
    return out_images, centroids.astype(np.float32), weights.astype(np.float32)


def _pseudo_labels_vis(
    vis_img: np.ndarray,       # [H_vis, W_vis]
    nsig: float = 3.0,
    max_sources: int = 1000,
    spike_radius: int = 40,
    min_star_area: int = 20,
    spike_width: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Detect sources from Euclid VIS at native 0.1"/px resolution.

    Returns centroids normalized to VIS frame [0,1], preserving the full
    VIS spatial precision without projecting through a coarser grid.

    Saturated blobs with area >= min_star_area pixels are identified as bright
    cores. Their centroids are recorded as detections, then thin radial spike
    ridges are masked. Classical peak-finding runs on the full image but its
    results are filtered to exclude positions on those ridges. This means bright
    sources are always detected while fake spike sources are suppressed without
    masking the gaps between spikes.
    Small saturated blobs (hot pixels, CRs) are ignored.

    Returns (centroids_norm [M,2], classes [M], H_vis, W_vis).
    """
    H, W = vis_img.shape

    bright_xs, bright_ys, spike_mask = _vis_bright_core_and_spike_mask(
        vis_img,
        spike_radius=spike_radius,
        min_star_area=min_star_area,
        spike_width=spike_width,
    )

    # Classical peak-finding on the full image, then filter spike regions.
    # Use multiple smoothing scales: compact/faint peaks need a sharp filter,
    # while broader galaxies are much more stable at a larger scale.
    peak_xs: list[np.ndarray] = []
    peak_ys: list[np.ndarray] = []
    scale_specs = [
        (float(nsig), 0.9, 7),
        (float(nsig), 1.2, 9),
        (max(2.0, float(nsig) - 0.5), 2.0, 13),
        (max(2.0, float(nsig) - 0.8), 3.0, 17),
    ]
    for nsig_i, smooth_sigma, min_dist in scale_specs:
        sx, sy = detect_sources(
            vis_img,
            nsig=nsig_i,
            max_sources=max_sources,
            smooth_sigma=smooth_sigma,
            min_dist=min_dist,
        )
        if sx.size:
            peak_xs.append(sx)
            peak_ys.append(sy)

    sep_x, sep_y = _sep_vis_proposals(
        vis_img,
        thresh=max(1.8, float(nsig) - 1.0),
        minarea=4,
        deblend_nthresh=32,
        deblend_cont=0.01,
    )
    if sep_x.size:
        peak_xs.append(sep_x)
        peak_ys.append(sep_y)

    if peak_xs:
        xs = np.concatenate(peak_xs).astype(np.float32)
        ys = np.concatenate(peak_ys).astype(np.float32)
        xi = np.clip(xs.round().astype(int), 0, W - 1)
        yi = np.clip(ys.round().astype(int), 0, H - 1)
        keep = ~spike_mask[yi, xi]
        xs, ys = xs[keep], ys[keep]
        scores = _point_scores(vis_img, xs, ys)
        xs, ys = _dedupe_points_by_score(xs, ys, scores, radius_px=5.0)
    else:
        xs = np.zeros(0, dtype=np.float32)
        ys = np.zeros(0, dtype=np.float32)

    # Prepend bright-core centroids (always kept regardless of spike mask)
    xs = np.concatenate([bright_xs, xs])
    ys = np.concatenate([bright_ys, ys])

    if xs.size > 0:
        scores = _point_scores(vis_img, xs, ys)
        xs, ys = _dedupe_points_by_score(xs, ys, scores, radius_px=5.0)
    xs, ys = _merge_resolved_vis_components(
        vis_img,
        xs,
        ys,
        spike_mask=spike_mask,
    )
    if max_sources and xs.size > max_sources:
        scores = _point_scores(vis_img, xs, ys)
        keep = np.argsort(scores)[::-1][:int(max_sources)]
        xs, ys = xs[keep], ys[keep]

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            H, W,
        )

    classes = np.zeros(len(xs), dtype=np.int64)
    centroids = np.stack([
        xs / max(W - 1, 1),
        ys / max(H - 1, 1),
    ], axis=1).astype(np.float32)

    return centroids, classes, H, W


def _pseudo_labels_vis_sep(
    vis_img: np.ndarray,       # [H_vis, W_vis]
    nsig: float = 3.0,
    max_sources: int = 1000,
    spike_radius: int = 40,
    min_star_area: int = 20,
    spike_width: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """SEP-primary VIS pseudo-labels with the same cleaning wrapper as
    ``_pseudo_labels_vis``.

    The primary detector is SEP (Source Extractor) instead of the multi-scale
    local-peak finder. The cleaning steps kept are: bright-core centroids (one
    detection per saturated star, always retained), thin spike-ridge masking,
    and greedy score-based dedup. The ``_merge_resolved_vis_components`` step
    used by ``_pseudo_labels_vis`` is deliberately *omitted* here -- on SEP
    inputs it was found to re-introduce multiple detections near bright stars
    and to cost completeness against MER.

    Motivation/tuning (validated on 10 ECDFS tiles vs the clean Q1 MER
    catalogue, 0.5'' match): bare SEP scores highest vs MER (comp ~98%, pur
    ~57%) but only because it splits nearly every bright star into multiples
    (227/228 cores) and fires on spikes (~17% on-spike) -- useless as training
    labels. This SEP+cleaning(no-merge) recipe gives comp ~88% / pur ~53%
    while staying clean (1/228 multi-star cores, ~6% on-spike), a modest but
    real gain over the multi-scale peak finder (comp 86% / pur 47%). Used for
    the SEP-relabel detector retrain (see DOCUMENTATION, "Detection").

    Returns (centroids_norm [M,2], classes [M], H_vis, W_vis).
    """
    H, W = vis_img.shape

    bright_xs, bright_ys, spike_mask = _vis_bright_core_and_spike_mask(
        vis_img,
        spike_radius=spike_radius,
        min_star_area=min_star_area,
        spike_width=spike_width,
    )

    # SEP is the primary (and only) peak source here, run at the requested
    # significance. Matches the bare-SEP config that wins against MER
    # (thresh~nsig, minarea=5, fine deblending) -- the cleaning below removes
    # its spike/bright-star artifacts.
    sep_x, sep_y = _sep_vis_proposals(
        vis_img,
        thresh=max(1.5, float(nsig)),
        minarea=5,
        deblend_nthresh=32,
        deblend_cont=0.005,
    )

    if sep_x.size:
        xs = np.asarray(sep_x, dtype=np.float32)
        ys = np.asarray(sep_y, dtype=np.float32)
        xi = np.clip(xs.round().astype(int), 0, W - 1)
        yi = np.clip(ys.round().astype(int), 0, H - 1)
        keep = ~spike_mask[yi, xi]
        xs, ys = xs[keep], ys[keep]
        scores = _point_scores(vis_img, xs, ys)
        xs, ys = _dedupe_points_by_score(xs, ys, scores, radius_px=5.0)
    else:
        xs = np.zeros(0, dtype=np.float32)
        ys = np.zeros(0, dtype=np.float32)

    # Prepend bright-core centroids (always kept regardless of spike mask)
    xs = np.concatenate([bright_xs, xs])
    ys = np.concatenate([bright_ys, ys])

    if xs.size > 0:
        scores = _point_scores(vis_img, xs, ys)
        xs, ys = _dedupe_points_by_score(xs, ys, scores, radius_px=5.0)
    # NB: no _merge_resolved_vis_components here -- on SEP inputs it re-creates
    # multi-detections near bright stars and costs MER completeness.
    if max_sources and xs.size > max_sources:
        scores = _point_scores(vis_img, xs, ys)
        keep = np.argsort(scores)[::-1][:int(max_sources)]
        xs, ys = xs[keep], ys[keep]

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            H, W,
        )

    classes = np.zeros(len(xs), dtype=np.int64)
    centroids = np.stack([
        xs / max(W - 1, 1),
        ys / max(H - 1, 1),
    ], axis=1).astype(np.float32)

    return centroids, classes, H, W


def _vis_bright_extended_rescue_labels(
    vis_img: np.ndarray,
    existing_norm: Optional[np.ndarray] = None,
    thresh_nsig: float = 2.5,
    min_area: int = 45,
    min_radius: float = 5.0,
    min_peak_snr: float = 8.0,
    match_radius_scale: float = 1.5,
    match_radius_min: float = 8.0,
    match_radius_max: float = 35.0,
    spike_radius: int = 40,
    spike_width: float = 3.0,
    min_star_area: int = 20,
    max_area_frac: float = 0.05,
    max_rescue_per_tile: int = 32,
) -> Tuple[np.ndarray, list]:
    """Find missing bright/extended VIS objects as one-center rescue labels.

    This is intentionally different from the thin spike veto.  It looks for
    resolved high-S/N connected components, estimates one centroid per component,
    and only returns candidates that are not already covered by an existing
    detection.  Coverage uses an adaptive radius tied to the component footprint:
    a broad galaxy is allowed a larger match radius than a compact source.
    """
    from scipy.ndimage import gaussian_filter, label as ndlabel

    img = np.asarray(vis_img, dtype=np.float32)
    finite = np.isfinite(img)
    if finite.sum() <= 100:
        return np.zeros((0, 2), dtype=np.float32), []

    clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    vals = img[finite]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sig = max(1.4826 * mad, 1e-6)
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros((0, 2), dtype=np.float32), []

    H, W = clean.shape
    smooth = gaussian_filter(clean, sigma=1.0)
    threshold = med + float(thresh_nsig) * sig
    candidate_mask = finite & (smooth >= threshold)

    if spike_radius > 0 and spike_width > 0:
        try:
            _, _, spike_mask = _vis_bright_core_and_spike_mask(
                clean,
                spike_radius=spike_radius,
                min_star_area=min_star_area,
                spike_width=spike_width,
                include_core=False,
            )
            candidate_mask &= ~spike_mask
        except Exception:
            pass

    labeled, n_comp = ndlabel(candidate_mask)
    if n_comp <= 0:
        return np.zeros((0, 2), dtype=np.float32), []

    existing_px = np.zeros((0, 2), dtype=np.float32)
    if existing_norm is not None and len(existing_norm) > 0:
        ex = np.asarray(existing_norm, dtype=np.float32)
        good = np.isfinite(ex).all(axis=1)
        if good.any():
            ex = ex[good]
            existing_px = np.stack(
                [ex[:, 0] * max(W - 1, 1), ex[:, 1] * max(H - 1, 1)],
                axis=1,
            ).astype(np.float32)

    max_area = max(float(min_area), float(H * W) * float(max_area_frac))
    candidates = []
    for lb in range(1, n_comp + 1):
        yy, xx = np.nonzero(labeled == lb)
        area = int(len(xx))
        if area < int(min_area) or area > max_area:
            continue

        vals_comp = clean[yy, xx]
        peak_snr = float((np.max(vals_comp) - med) / sig)
        if peak_snr < float(min_peak_snr):
            continue

        weights = np.clip(smooth[yy, xx] - threshold, 0.0, None).astype(np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights, dtype=np.float64)
        wsum = float(weights.sum())
        cx = float(np.sum(xx * weights) / wsum)
        cy = float(np.sum(yy * weights) / wsum)

        moment_r = float(np.sqrt(np.sum(((xx - cx) ** 2 + (yy - cy) ** 2) * weights) / wsum))
        footprint_r = float(np.sqrt(area / np.pi))
        radius = max(moment_r, footprint_r)
        if radius < float(min_radius):
            continue

        match_radius = float(np.clip(
            match_radius_scale * radius,
            match_radius_min,
            match_radius_max,
        ))
        if existing_px.shape[0] > 0:
            d = np.hypot(existing_px[:, 0] - cx, existing_px[:, 1] - cy)
            if float(d.min()) <= match_radius:
                continue

        candidates.append({
            'xy': np.array([cx / max(W - 1, 1), cy / max(H - 1, 1)], dtype=np.float32),
            'score': peak_snr * np.sqrt(area),
            'area': area,
            'radius_px': radius,
            'match_radius_px': match_radius,
            'peak_snr': peak_snr,
        })

    if not candidates:
        return np.zeros((0, 2), dtype=np.float32), []

    candidates.sort(key=lambda c: c['score'], reverse=True)
    if max_rescue_per_tile > 0:
        candidates = candidates[:int(max_rescue_per_tile)]
    centroids = np.stack([c['xy'] for c in candidates], axis=0).astype(np.float32)
    return centroids, candidates


class TileDetectionDataset(Dataset):
    """
    Parameters
    ----------
    rubin_dir      : Rubin tile directory (tile_x*_y*.npz)
    euclid_dir     : Euclid tile directory (optional)
    nsig           : detection significance for pseudo-labels
    max_sources    : cap on pseudo-labels per tile
    use_all_bands  : if True include Euclid bands; else Rubin-only
    augment        : random 90° rotations + flips (matches MAE training)
    """

    def __init__(
        self,
        rubin_dir:        str,
        euclid_dir:       Optional[str] = None,
        nsig:             float = 5.0,
        max_sources:      int = 500,
        use_all_bands:    bool = False,
        augment:          bool = True,
        labels_mode:      str = "vis_peak",
        multiband_kwargs: Optional[dict] = None,
        uncertain_ignore: bool = False,
        uncertain_nsig:   float = 1.8,
        uncertain_radius_px: float = 5.0,
        synthetic_sources_per_tile: int = 0,
        synthetic_prob:   float = 0.0,
        synthetic_min_snr: float = 5.0,
        synthetic_max_snr: float = 20.0,
        synthetic_min_sigma_px: float = 1.1,
        synthetic_max_sigma_px: float = 3.5,
        synthetic_weight: float = 1.5,
    ):
        """labels_mode:
            "vis_peak"  - legacy single-band peak finder via _pseudo_labels_vis.
            "multiband" - sep-based multi-band labels via labels_multiband
                          (VIS + NIR chi-squared stack, with cross-band
                          deblending). Requires Euclid bands; falls back to
                          "vis_peak" on tiles where Euclid is unavailable.
        """
        self.nsig         = nsig
        self.max_sources  = max_sources
        self.use_all_bands = use_all_bands and (euclid_dir is not None)
        self.bands        = ALL_BANDS if self.use_all_bands else RUBIN_BANDS

        if labels_mode not in ("vis_peak", "multiband", "vis_sep"):
            raise ValueError(f"unknown labels_mode={labels_mode!r}")
        self.labels_mode = labels_mode
        # VIS label fn: SEP-primary for "vis_sep", else the multi-scale peak
        # finder. "multiband" uses this as its fallback when Euclid is missing.
        self._vis_label_fn = (
            _pseudo_labels_vis_sep if labels_mode == "vis_sep" else _pseudo_labels_vis
        )
        self.multiband_kwargs = dict(multiband_kwargs or {})
        self.uncertain_ignore = bool(uncertain_ignore)
        self.uncertain_nsig = float(uncertain_nsig)
        self.uncertain_radius_px = float(uncertain_radius_px)
        self.synthetic_sources_per_tile = int(synthetic_sources_per_tile)
        self.synthetic_prob = float(synthetic_prob)
        self.synthetic_min_snr = float(synthetic_min_snr)
        self.synthetic_max_snr = float(synthetic_max_snr)
        self.synthetic_min_sigma_px = float(synthetic_min_sigma_px)
        self.synthetic_max_sigma_px = float(synthetic_max_sigma_px)
        self.synthetic_weight = float(synthetic_weight)

        self._base = JAISPDatasetV10(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir or rubin_dir,
            augment=augment,
            load_euclid=self.use_all_bands,
        )

        if labels_mode == "multiband":
            from detection.labels_multiband import make_multiband_labels
            _multiband_fn = make_multiband_labels
        else:
            _multiband_fn = None

        self._label_cache = {}
        self._weight_cache = {}
        self._ignore_cache = {} if self.uncertain_ignore else None
        n_mb, n_vis, n_rubin = 0, 0, 0
        print(f'  Pre-computing pseudo-labels (mode={labels_mode}, nsig={nsig}) ...',
              end=' ', flush=True)
        for idx in range(len(self._base)):
            tile = self._base.tiles[idx]
            euclid_path = tile.get('euclid_path')

            centroids_np = None
            classes_np   = None
            H = W = None

            if labels_mode == "multiband" and euclid_path and Path(euclid_path).exists():
                try:
                    edata = np.load(euclid_path, allow_pickle=True, mmap_mode='r')
                    def _g(k):
                        return np.nan_to_num(
                            np.asarray(edata[k], dtype=np.float32), nan=0.0,
                        )
                    labels, _info = _multiband_fn(
                        _g('img_VIS'), np.asarray(edata['var_VIS'], dtype=np.float32),
                        _g('img_Y'),   np.asarray(edata['var_Y'],   dtype=np.float32),
                        _g('img_J'),   np.asarray(edata['var_J'],   dtype=np.float32),
                        _g('img_H'),   np.asarray(edata['var_H'],   dtype=np.float32),
                        **self.multiband_kwargs,
                    )
                    if labels.x_norm.size > 0:
                        centroids_np = labels.centroids()
                        classes_np   = np.zeros(len(centroids_np), dtype=np.int64)
                        src = np.asarray(labels.sources, dtype=object)
                        weights_np = np.ones(len(centroids_np), dtype=np.float32)
                        weights_np[src == "both"] = 1.0
                        weights_np[src == "vis"] = 0.9
                        weights_np[src == "nir"] = 0.7
                        weights_np[src == "nir_deblend"] = 0.8
                        H, W = labels.H, labels.W
                        if max_sources and len(centroids_np) > max_sources:
                            centroids_np = centroids_np[:max_sources]
                            classes_np   = classes_np[:max_sources]
                            weights_np   = weights_np[:max_sources]
                        n_mb += 1
                except Exception:
                    pass  # fall through to vis_peak / rubin path

            if centroids_np is None and euclid_path and Path(euclid_path).exists():
                try:
                    edata = np.load(euclid_path, allow_pickle=True, mmap_mode='r')
                    vis_img = np.nan_to_num(
                        np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0,
                    )
                    centroids_np, classes_np, H, W = self._vis_label_fn(
                        vis_img, self.nsig, self.max_sources,
                    )
                    weights_np = np.ones(len(centroids_np), dtype=np.float32)
                    n_vis += 1
                except Exception:
                    centroids_np = None

            if centroids_np is None:
                raw_path = tile['rubin_path']
                raw_data = np.load(raw_path, allow_pickle=True, mmap_mode='r')
                raw_img = np.nan_to_num(
                    np.asarray(raw_data['img'], dtype=np.float32), nan=0.0,
                )
                centroids_np, classes_np, H, W = _pseudo_labels(
                    raw_img, self.nsig, self.max_sources,
                )
                weights_np = np.ones(len(centroids_np), dtype=np.float32)
                n_rubin += 1

            self._label_cache[idx] = (centroids_np, classes_np, H, W)
            self._weight_cache[idx] = weights_np

            if self._ignore_cache is not None and euclid_path and Path(euclid_path).exists():
                try:
                    edata = np.load(euclid_path, allow_pickle=True, mmap_mode='r')
                    vis_img = np.nan_to_num(
                        np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0,
                    )
                    _, _, spike_mask = _vis_bright_core_and_spike_mask(
                        vis_img,
                        spike_radius=40,
                        spike_width=3.0,
                        include_core=False,
                    )
                    self._ignore_cache[idx] = _proposal_ignore_mask_vis(
                        vis_img,
                        positives_norm=centroids_np,
                        nsig=self.nsig,
                        spike_mask=spike_mask,
                        proposal_nsig=self.uncertain_nsig,
                        ignore_radius_px=self.uncertain_radius_px,
                    )
                except Exception:
                    pass
        print(f'done ({len(self._label_cache)} tiles: '
              f'{n_mb} multiband, {n_vis} VIS, {n_rubin} Rubin)')

    @staticmethod
    def _transform_centroids(
        centroids: np.ndarray, n_rot: int, flip_ud: bool, flip_lr: bool,
    ) -> np.ndarray:
        """Apply the same augmentation to normalized (x, y) centroids."""
        if centroids.shape[0] == 0:
            return centroids
        xy = centroids.copy()
        # rot90: each 90° rotation maps (x, y) -> (y, 1-x)
        for _ in range(n_rot % 4):
            xy = np.stack([xy[:, 1], 1.0 - xy[:, 0]], axis=1)
        if flip_ud:
            xy[:, 1] = 1.0 - xy[:, 1]
        if flip_lr:
            xy[:, 0] = 1.0 - xy[:, 0]
        return xy

    @staticmethod
    def _transform_mask(mask: np.ndarray, n_rot: int, flip_ud: bool, flip_lr: bool) -> np.ndarray:
        """Apply the same augmentation to a 2D mask."""
        m = np.asarray(mask, dtype=bool)
        if n_rot % 4:
            m = np.rot90(m, n_rot % 4).copy()
        if flip_ud:
            m = np.flipud(m).copy()
        if flip_lr:
            m = np.fliplr(m).copy()
        return m

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        item = self._base[idx]

        # Build per-band image/rms dicts
        images: dict = {}
        rms:    dict = {}

        for band in RUBIN_BANDS:
            if band in item['rubin']:
                images[band] = item['rubin'][band]['image']
                rms[band]    = item['rubin'][band]['rms']

        if self.use_all_bands:
            for band in EUCLID_BANDS:
                if band in item.get('euclid', {}):
                    images[band] = item['euclid'][band]['image']
                    rms[band]    = item['euclid'][band]['rms']

        # Retrieve cached pseudo-labels and transform to match augmentation
        centroids_np, classes_np, H, W = self._label_cache[idx]
        weights_np = self._weight_cache.get(
            idx,
            np.ones(len(centroids_np), dtype=np.float32),
        )
        aug_params = item.get('aug_params', (0, False, False))
        centroids_np = self._transform_centroids(centroids_np, *aug_params)

        if self.synthetic_sources_per_tile > 0 and self._base.augment:
            images, centroids_np, weights_np = _inject_synthetic_sources(
                images,
                rms,
                centroids_np,
                weights_np,
                n_sources=self.synthetic_sources_per_tile,
                prob=self.synthetic_prob,
                min_snr=self.synthetic_min_snr,
                max_snr=self.synthetic_max_snr,
                min_sigma_px=self.synthetic_min_sigma_px,
                max_sigma_px=self.synthetic_max_sigma_px,
                source_weight=self.synthetic_weight,
            )
            classes_np = np.zeros(len(centroids_np), dtype=np.int64)

        result = {
            'images':    images,                              # {band: [1,H,W]}
            'rms':       rms,                                 # {band: [1,H,W]}
            'centroids': torch.from_numpy(centroids_np),     # [M, 2]
            'classes':   torch.from_numpy(classes_np),       # [M]
            'source_weights': torch.from_numpy(weights_np.astype(np.float32)),
            'tile_id':   item['tile_id'],
            'tile_hw':   (H, W),
        }
        ignore_cache = getattr(self, '_ignore_cache', None)
        if ignore_cache is not None and idx in ignore_cache:
            ignore_np = self._transform_mask(ignore_cache[idx], *aug_params)
            result['ignore_mask'] = torch.from_numpy(ignore_np.astype(np.bool_))
        return result


def collate_fn(batch: List[dict]) -> dict:
    """
    Stack images/rms per band across the batch; keep centroids/classes as lists
    (variable number of sources per tile).
    """
    bands = list(batch[0]['images'].keys())
    images = {b: torch.stack([s['images'][b] for s in batch]) for b in bands}
    rms    = {b: torch.stack([s['rms'][b]    for s in batch]) for b in bands}
    result = {
        'images':    images,
        'rms':       rms,
        'centroids': [s['centroids'] for s in batch],
        'classes':   [s['classes']   for s in batch],
        'source_weights': [s['source_weights'] for s in batch],
        'tile_id':   [s['tile_id']   for s in batch],
        'tile_hw':   [s['tile_hw']   for s in batch],
    }
    if all('ignore_mask' in s for s in batch):
        result['ignore_mask'] = torch.stack([s['ignore_mask'] for s in batch])
    return result
