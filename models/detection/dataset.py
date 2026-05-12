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


PSEUDO_LABEL_CACHE_VERSION = 2


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

    # Classical peak-finding on the full image, then filter spike regions
    xs, ys = detect_sources(vis_img, nsig=nsig, max_sources=max_sources,
                            smooth_sigma=1.2, min_dist=9)
    if xs.size > 0:
        xi = np.clip(xs.round().astype(int), 0, W - 1)
        yi = np.clip(ys.round().astype(int), 0, H - 1)
        keep = ~spike_mask[yi, xi]
        xs, ys = xs[keep], ys[keep]

    # Prepend bright-core centroids (always kept regardless of spike mask)
    xs = np.concatenate([bright_xs, xs])
    ys = np.concatenate([bright_ys, ys])

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
        rubin_dir:     str,
        euclid_dir:    Optional[str] = None,
        nsig:          float = 5.0,
        max_sources:   int = 500,
        use_all_bands: bool = False,
        augment:       bool = True,
    ):
        self.nsig         = nsig
        self.max_sources  = max_sources
        self.use_all_bands = use_all_bands and (euclid_dir is not None)
        self.bands        = ALL_BANDS if self.use_all_bands else RUBIN_BANDS

        self._base = JAISPDatasetV10(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir or rubin_dir,
            augment=augment,
            load_euclid=self.use_all_bands,
        )

        # Cache pseudo-labels from raw (unaugmented) tiles — run detection once.
        # Use VIS when Euclid data is available (0.1"/px, sharper centroids);
        # fall back to Rubin g+r+i coadd when not.
        self._label_cache = {}
        n_vis, n_rubin = 0, 0
        print(f'  Pre-computing pseudo-labels (nsig={nsig}) ...', end=' ', flush=True)
        for idx in range(len(self._base)):
            tile = self._base.tiles[idx]

            # Try VIS first
            euclid_path = tile.get('euclid_path')
            used_vis = False
            if euclid_path and Path(euclid_path).exists():
                try:
                    edata = np.load(euclid_path, allow_pickle=True, mmap_mode='r')
                    vis_img = np.nan_to_num(
                        np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0
                    )
                    centroids_np, classes_np, H, W = _pseudo_labels_vis(
                        vis_img, self.nsig, self.max_sources
                    )
                    used_vis = True
                    n_vis += 1
                except Exception:
                    pass

            if not used_vis:
                raw_path = tile['rubin_path']
                raw_data = np.load(raw_path, allow_pickle=True, mmap_mode='r')
                raw_img = np.nan_to_num(
                    np.asarray(raw_data['img'], dtype=np.float32), nan=0.0
                )
                centroids_np, classes_np, H, W = _pseudo_labels(
                    raw_img, self.nsig, self.max_sources
                )
                n_rubin += 1

            self._label_cache[idx] = (centroids_np, classes_np, H, W)
        print(f'done ({len(self._label_cache)} tiles: {n_vis} VIS, {n_rubin} Rubin)')

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
        aug_params = item.get('aug_params', (0, False, False))
        centroids_np = self._transform_centroids(centroids_np, *aug_params)

        result = {
            'images':    images,                              # {band: [1,H,W]}
            'rms':       rms,                                 # {band: [1,H,W]}
            'centroids': torch.from_numpy(centroids_np),     # [M, 2]
            'classes':   torch.from_numpy(classes_np),       # [M]
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
        'tile_id':   [s['tile_id']   for s in batch],
        'tile_hw':   [s['tile_hw']   for s in batch],
    }
    if all('ignore_mask' in s for s in batch):
        result['ignore_mask'] = torch.stack([s['ignore_mask'] for s in batch])
    return result
