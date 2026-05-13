"""Multi-band detection-label generator for v10 detector head training.

Replaces the single-band ``_pseudo_labels_vis`` peak-finder with a Source
Extractor-based, multi-band pipeline. Goal: training labels include positives
the foundation should learn to detect by fusing multi-band evidence, not just
VIS peaks.

Categories emitted:
  - ``vis``         : detected in VIS only
  - ``both``        : VIS + NIR (matched within ``cross_match_px``)
  - ``nir``         : NIR-only (no VIS counterpart within ``cross_match_px``)
  - ``nir_deblend`` : extra NIR peaks inside a single VIS segment whose
                      distance from the VIS centroid exceeds
                      ``cross_deblend_min_px`` -- "cross-band deblending".

Conservative extended VIS rescues are emitted as ``vis`` labels and counted
separately in diagnostics as ``n_vis_extended_rescue``.

VIS, Y, J, H are all on the same 1084x1084 grid at 0.1"/px (MER mosaic),
so no reprojection is needed. Rubin is not yet incorporated; the v10
foundation passes Rubin through the frozen encoder, so the detector head
already sees Rubin features. Rubin-anchored positives can be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import sep


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _prepare(img: np.ndarray) -> np.ndarray:
    """sep requires contiguous, native-byte-order float32."""
    a = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.dtype.byteorder not in ("=", "|"):
        a = a.byteswap().newbyteorder()
    return a


def _safe_rms_from_var(var: np.ndarray, fallback: float = 1.0) -> np.ndarray:
    var = np.asarray(var, dtype=np.float32)
    good = np.isfinite(var) & (var > 0) & (var < 1e20)
    if not good.any():
        return np.full(var.shape, fallback, dtype=np.float32)
    fb = float(np.sqrt(np.median(var[good])))
    out = np.full(var.shape, fb, dtype=np.float32)
    out[good] = np.sqrt(var[good])
    return out


# ---------------------------------------------------------------------------
# per-band sep wrapper
# ---------------------------------------------------------------------------


@dataclass
class BandCatalog:
    x:    np.ndarray
    y:    np.ndarray
    flux: np.ndarray
    snr:  np.ndarray
    seg:  np.ndarray
    band: str

    def __len__(self) -> int:
        return int(self.x.size)


def _empty_cat(H: int, W: int, band: str) -> BandCatalog:
    return BandCatalog(
        x=np.zeros(0, dtype=np.float32),
        y=np.zeros(0, dtype=np.float32),
        flux=np.zeros(0, dtype=np.float32),
        snr=np.zeros(0, dtype=np.float32),
        seg=np.zeros((H, W), dtype=np.int32),
        band=band,
    )


_DEFAULT_FILTER_KERNEL = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 4.0, 2.0],
    [1.0, 2.0, 1.0],
], dtype=np.float32) / 16.0


def _sep_extract(
    img:             np.ndarray,
    rms:             np.ndarray,
    thresh:          float,
    minarea:         int,
    deblend_nthresh: int,
    deblend_cont:    float,
    band:            str,
    subtract_background: bool,
    filter_kernel:   "np.ndarray | None" = _DEFAULT_FILTER_KERNEL,
) -> BandCatalog:
    """Run sep.Background + sep.extract on a single image.

    ``thresh`` is in units of ``err``. ``filter_kernel`` is a small Gaussian
    matched filter applied before thresholding -- crucial for recovering
    compact faint sources that don't have many pixels above threshold without
    smoothing. Default is a 3x3 Gaussian (sigma ~1 px). Pass ``None`` to skip.
    """
    a = _prepare(img)
    H, W = a.shape

    if subtract_background:
        try:
            bkg = sep.Background(a)
            a = a - bkg.back()
        except Exception:
            pass

    err = _prepare(rms)
    if (err <= 0).all():
        return _empty_cat(H, W, band)

    try:
        cat, seg = sep.extract(
            a, thresh=float(thresh), err=err,
            minarea=int(minarea),
            deblend_nthresh=int(deblend_nthresh),
            deblend_cont=float(deblend_cont),
            filter_kernel=filter_kernel,
            filter_type="matched",
            clean=True, clean_param=1.0,
            segmentation_map=True,
        )
    except Exception:
        return _empty_cat(H, W, band)

    if len(cat) == 0:
        return BandCatalog(
            x=np.zeros(0, dtype=np.float32),
            y=np.zeros(0, dtype=np.float32),
            flux=np.zeros(0, dtype=np.float32),
            snr=np.zeros(0, dtype=np.float32),
            seg=np.asarray(seg, dtype=np.int32),
            band=band,
        )

    x = np.asarray(cat["x"], dtype=np.float32)
    y = np.asarray(cat["y"], dtype=np.float32)
    flux = np.asarray(cat["flux"], dtype=np.float32)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    peak = a[yi, xi]
    er = err[yi, xi]
    er = np.where(er > 0, er, np.median(err[err > 0]))
    snr = (peak / er).astype(np.float32)
    return BandCatalog(
        x=x, y=y, flux=flux, snr=snr,
        seg=np.asarray(seg, dtype=np.int32),
        band=band,
    )


# ---------------------------------------------------------------------------
# chi^2 stack
# ---------------------------------------------------------------------------


def chi2_significance(
    images: List[np.ndarray],
    vars_:  List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a Gaussian-equivalent significance image from N co-registered bands.

    For each pixel, ``S = sum_i (img_i^2 / var_i)`` is chi^2-distributed with N
    degrees of freedom under pure-sky noise. We return ``(S - N) / sqrt(2N)``,
    which has zero mean and unit variance under sky -- so it can be fed to
    sep.extract with ``err=1`` and ``thresh`` interpreted as sigma.
    """
    n = len(images)
    H, W = images[0].shape
    s = np.zeros((H, W), dtype=np.float32)
    for img, var in zip(images, vars_):
        a = np.asarray(img, dtype=np.float32)
        v = np.asarray(var, dtype=np.float32)
        good = np.isfinite(a) & np.isfinite(v) & (v > 0) & (v < 1e20)
        s += np.where(good, (a * a) / np.where(good, v, 1.0), 0.0)
    det = ((s - n) / np.sqrt(2.0 * n)).astype(np.float32)
    return det, np.ones_like(det, dtype=np.float32)


# ---------------------------------------------------------------------------
# top-level multi-band label builder
# ---------------------------------------------------------------------------


@dataclass
class MultibandLabels:
    x_norm:  np.ndarray   # [N] normalized to VIS frame [0,1]
    y_norm:  np.ndarray
    sources: np.ndarray   # [N] object array of {'vis','both','nir','nir_deblend'}
    H:       int          # VIS frame height (px)
    W:       int          # VIS frame width (px)

    def centroids(self) -> np.ndarray:
        return np.stack([self.x_norm, self.y_norm], axis=1).astype(np.float32)


def _filter_spike(cat: BandCatalog, spike_mask: np.ndarray) -> BandCatalog:
    if len(cat) == 0 or spike_mask is None:
        return cat
    H, W = spike_mask.shape
    xi = np.clip(np.round(cat.x).astype(int), 0, W - 1)
    yi = np.clip(np.round(cat.y).astype(int), 0, H - 1)
    keep = ~spike_mask[yi, xi]
    return BandCatalog(
        x=cat.x[keep], y=cat.y[keep],
        flux=cat.flux[keep], snr=cat.snr[keep],
        seg=cat.seg, band=cat.band,
    )


def _greedy_dedup_xy(xy: np.ndarray, radius_px: float) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.zeros(0, dtype=int)
    r2 = float(radius_px) ** 2
    kept = [0]
    kept_xy = xy[:1].copy()
    for i in range(1, xy.shape[0]):
        d2 = ((kept_xy - xy[i]) ** 2).sum(axis=1)
        if d2.min() >= r2:
            kept.append(i)
            kept_xy = np.vstack([kept_xy, xy[i]])
    return np.asarray(kept, dtype=int)


def make_multiband_labels(
    img_vis: np.ndarray, var_vis: np.ndarray,
    img_Y:   np.ndarray, var_Y:   np.ndarray,
    img_J:   np.ndarray, var_J:   np.ndarray,
    img_H:   np.ndarray, var_H:   np.ndarray,
    vis_thresh:          float = 1.2,
    nir_thresh:          float = 1.2,
    minarea:             int   = 3,
    deblend_nthresh:     int   = 32,
    deblend_cont:        float = 0.005,
    cross_match_px:      float = 3.0,
    cross_deblend_min_px: float = 4.0,
    dedup_px:            float = 2.0,
    apply_spike_veto:    bool  = True,
    spike_radius:        int   = 40,
    spike_width:         float = 3.0,
    extended_rescue:     bool  = True,
    extended_rescue_nsig: float = 2.0,
    extended_rescue_min_area: int = 35,
    extended_rescue_min_radius: float = 4.0,
    extended_rescue_min_peak_snr: float = 6.0,
    extended_rescue_max_per_tile: int = 32,
) -> Tuple[MultibandLabels, dict]:
    """Build multi-band labels for one tile.

    Returns the final label set (positions normalized to the VIS frame) and
    a small diagnostic dict with per-stage counts.
    """
    H, W = img_vis.shape
    info: dict = {}

    vis_cat = _sep_extract(
        img_vis, _safe_rms_from_var(var_vis),
        thresh=vis_thresh, minarea=minarea,
        deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
        band="VIS", subtract_background=True,
    )
    info["n_vis_raw"] = len(vis_cat)

    # Saturated/very-bright stars often defeat sep: a bright peak inflates the
    # local sep.Background estimate so the background-subtracted residual
    # falls below thresh*err, and the star is missed entirely. The old
    # _pseudo_labels_vis pipeline worked around this by separately detecting
    # bright saturated cores morphologically and unconditionally prepending
    # them to the catalog. Do the same here so saturated cores never go
    # unlabeled. Match radius is generous (8 px) because sep often finds a
    # spurious peak in the bright-star halo offset from the actual core.
    from detection.dataset import _vis_bright_core_and_spike_mask
    bright_xs, bright_ys, _ = _vis_bright_core_and_spike_mask(
        np.asarray(img_vis, dtype=np.float32),
        spike_radius=spike_radius,
        spike_width=spike_width,
        include_core=False,
    )
    n_bright_added = 0
    if bright_xs.size > 0:
        if len(vis_cat) > 0:
            dx = bright_xs[:, None] - vis_cat.x[None, :]
            dy = bright_ys[:, None] - vis_cat.y[None, :]
            bright_to_vis = np.sqrt(dx * dx + dy * dy).min(axis=1)
            new_mask = bright_to_vis > 8.0
        else:
            new_mask = np.ones(bright_xs.size, dtype=bool)
        if new_mask.any():
            add_x = bright_xs[new_mask].astype(np.float32)
            add_y = bright_ys[new_mask].astype(np.float32)
            zeros = np.zeros(add_x.size, dtype=np.float32)
            vis_cat = BandCatalog(
                x   =np.concatenate([vis_cat.x,    add_x]),
                y   =np.concatenate([vis_cat.y,    add_y]),
                flux=np.concatenate([vis_cat.flux, zeros]),
                snr =np.concatenate([vis_cat.snr,  zeros]),
                seg =vis_cat.seg,
                band=vis_cat.band,
            )
            n_bright_added = int(add_x.size)
    info["n_vis_bright_core_added"] = n_bright_added

    nir_det, nir_rms = chi2_significance(
        [img_Y, img_J, img_H], [var_Y, var_J, var_H],
    )
    nir_cat = _sep_extract(
        nir_det, nir_rms,
        thresh=nir_thresh, minarea=minarea,
        deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
        band="NIR", subtract_background=False,
    )
    info["n_nir_raw"] = len(nir_cat)

    if apply_spike_veto:
        # Reuse the existing helper so spike masking is consistent with the
        # rest of the pipeline.
        from detection.dataset import _vis_bright_core_and_spike_mask
        _, _, spike_mask = _vis_bright_core_and_spike_mask(
            np.asarray(img_vis, dtype=np.float32),
            spike_radius=spike_radius,
            spike_width=spike_width,
            include_core=False,
        )
        vis_cat = _filter_spike(vis_cat, spike_mask)
        nir_cat = _filter_spike(nir_cat, spike_mask)
        info["n_vis_post_spike"] = len(vis_cat)
        info["n_nir_post_spike"] = len(nir_cat)

    out_xy: List[Tuple[float, float]] = []
    out_src: List[str] = []

    # Match: each VIS to nearest NIR, each NIR to nearest VIS.
    if len(vis_cat) > 0 and len(nir_cat) > 0:
        dx = vis_cat.x[:, None] - nir_cat.x[None, :]
        dy = vis_cat.y[:, None] - nir_cat.y[None, :]
        vis_to_nir = np.sqrt(dx * dx + dy * dy)
        vis_min = vis_to_nir.min(axis=1)
        vis_has_nir = vis_min <= cross_match_px
        nir_min = vis_to_nir.min(axis=0)
        nir_nearest_vis = vis_to_nir.argmin(axis=0)
    else:
        vis_has_nir    = np.zeros(len(vis_cat), dtype=bool)
        nir_min        = np.full(len(nir_cat), np.inf, dtype=np.float32)
        nir_nearest_vis = np.zeros(len(nir_cat), dtype=int)

    # 1. VIS detections -> 'vis' (no NIR partner) or 'both' (NIR within cross_match_px).
    for i in range(len(vis_cat)):
        out_xy.append((float(vis_cat.x[i]), float(vis_cat.y[i])))
        out_src.append("both" if vis_has_nir[i] else "vis")

    # 2. NIR peaks: classify mutually exclusively.
    #    First check distance to nearest VIS peak (authoritative "same source"
    #    radius), THEN fall back to VIS-segment membership for cross-band
    #    deblending. The distance-first rule fixes a previous bug where an NIR
    #    peak 3-4 px from a compact VIS source -- but just outside the small
    #    VIS segment -- was tagged 'nir' on top of the matching 'both'.
    #
    #    - within cross_match_px of a VIS peak -> already 'both', skip.
    #    - else if segment=0 (sky)           -> 'nir' (no VIS counterpart).
    #    - else if segment=k, far AND seg has >=2 NIR peaks -> 'nir_deblend'.
    #    - else                              -> skip (single faint knot in halo).
    n_nir_only = 0
    n_deblend  = 0
    if len(nir_cat) > 0 and vis_cat.seg.shape == (H, W):
        nyi = np.clip(np.round(nir_cat.y).astype(int), 0, H - 1)
        nxi = np.clip(np.round(nir_cat.x).astype(int), 0, W - 1)
        nir_in_segid = vis_cat.seg[nyi, nxi]  # 0 = sky, k = VIS source k (1-indexed)
    else:
        nir_in_segid = np.zeros(len(nir_cat), dtype=np.int32)

    seg_counts: dict = {}
    for sid in nir_in_segid:
        sid = int(sid)
        if sid > 0:
            seg_counts[sid] = seg_counts.get(sid, 0) + 1

    for j in range(len(nir_cat)):
        # Distance-first guard: any VIS peak within cross_match_px means the
        # NIR peak is already represented as 'both'. Skip regardless of seg.
        if nir_min[j] <= cross_match_px:
            continue
        seg_id = int(nir_in_segid[j])
        if seg_id == 0:
            out_xy.append((float(nir_cat.x[j]), float(nir_cat.y[j])))
            out_src.append("nir")
            n_nir_only += 1
            continue
        if seg_counts.get(seg_id, 0) < 2:
            continue
        vis_i = seg_id - 1
        if not (0 <= vis_i < len(vis_cat)):
            continue
        d = float(np.hypot(
            nir_cat.x[j] - vis_cat.x[vis_i],
            nir_cat.y[j] - vis_cat.y[vis_i],
        ))
        if d >= cross_deblend_min_px:
            out_xy.append((float(nir_cat.x[j]), float(nir_cat.y[j])))
            out_src.append("nir_deblend")
            n_deblend += 1

    info["n_nir_only"]    = n_nir_only
    info["n_nir_deblend"] = n_deblend

    # SEP can miss obvious broad galaxies when local background subtraction or
    # deblending eats the extended footprint. Add one conservative VIS-center
    # rescue label for high-S/N resolved components not already covered by the
    # VIS/NIR label union. Keep the source tag as "vis" so existing notebook
    # plotting and training code include it without needing a new class.
    n_extended_rescue = 0
    if extended_rescue:
        try:
            from detection.dataset import _vis_bright_extended_rescue_labels

            existing_norm = np.zeros((0, 2), dtype=np.float32)
            if out_xy:
                out_arr = np.asarray(out_xy, dtype=np.float32)
                existing_norm = np.stack([
                    out_arr[:, 0] / max(W - 1, 1),
                    out_arr[:, 1] / max(H - 1, 1),
                ], axis=1).astype(np.float32)

            rescue_norm, _ = _vis_bright_extended_rescue_labels(
                np.asarray(img_vis, dtype=np.float32),
                existing_norm=existing_norm,
                thresh_nsig=extended_rescue_nsig,
                min_area=extended_rescue_min_area,
                min_radius=extended_rescue_min_radius,
                min_peak_snr=extended_rescue_min_peak_snr,
                spike_radius=spike_radius if apply_spike_veto else 0,
                spike_width=spike_width if apply_spike_veto else 0.0,
                max_rescue_per_tile=extended_rescue_max_per_tile,
            )
            for x_norm, y_norm in rescue_norm:
                out_xy.append((float(x_norm) * max(W - 1, 1), float(y_norm) * max(H - 1, 1)))
                out_src.append("vis")
            n_extended_rescue = int(len(rescue_norm))
        except Exception:
            n_extended_rescue = 0
    info["n_vis_extended_rescue"] = n_extended_rescue

    if not out_xy:
        info["n_total"] = 0
        return (
            MultibandLabels(
                x_norm=np.zeros(0, dtype=np.float32),
                y_norm=np.zeros(0, dtype=np.float32),
                sources=np.zeros(0, dtype=object),
                H=H, W=W,
            ),
            info,
        )

    xy = np.asarray(out_xy, dtype=np.float32)
    src = np.asarray(out_src, dtype=object)
    keep = _greedy_dedup_xy(xy, radius_px=float(dedup_px))
    xy = xy[keep]
    src = src[keep]

    info["n_total"]          = int(len(xy))
    info["n_both"]           = int((src == "both").sum())
    info["n_vis_only_final"] = int((src == "vis").sum())
    info["n_nir_only_final"] = int((src == "nir").sum())
    info["n_deblend_final"]  = int((src == "nir_deblend").sum())

    labels = MultibandLabels(
        x_norm=(xy[:, 0] / max(W - 1, 1)).astype(np.float32),
        y_norm=(xy[:, 1] / max(H - 1, 1)).astype(np.float32),
        sources=src,
        H=H, W=W,
    )
    return labels, info
