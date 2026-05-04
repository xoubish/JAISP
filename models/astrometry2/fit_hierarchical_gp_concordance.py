"""Global hierarchical GP-style concordance fit from saved anchor caches.

This script is meant for the post-latent-head concordance use case:

  * one global field over the whole mosaic, not one field per tile;
  * CenterNet anchors are supported directly via ``anchors_centernet.npz``;
  * ``*_head_resid`` is the default target, so per-source centering is already
    handled before the field model sees the data;
  * the multi-band structure is hierarchical rather than fully pooled or fully
    separate:

        field_band(x, y) = common(x, y)
                         + instrument_group(x, y)
                         + band_specific(x, y)

The implementation is a finite-rank, Bayesian linear approximation to a
multi-scale GP.  It uses individual anchors, not binned measurements.  The
finite basis is a computational scaffold: compact radial basis functions at
several length scales play the role of inducing features, while Gaussian priors
on the feature weights provide shrinkage and posterior field uncertainty.

This is intentionally less assumption-heavy than a fixed-resolution grid, but
it is still a model: the length-scale list, feature budget and component priors
define what structure the fit can express.  Spatial-block holdout and posterior
calibration are included so the uncertainty maps are not taken on faith.

Example
-------
    python models/astrometry2/fit_hierarchical_gp_concordance.py \\
        --anchors models/checkpoints/latent_position_v8_no_psf/anchors_centernet.npz \\
        --output  models/checkpoints/latent_position_v8_no_psf/concordance_hgp_head_resid.fits \\
        --offset-kind head_resid \\
        --pool all \\
        --length-scales 60,180,600 \\
        --dstep-arcsec 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import linalg
from scipy import sparse
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_MODELS, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.dataset import BAND_TO_IDX, NISP_BAND_ORDER
from astrometry2.source_matching import RUBIN_BAND_ORDER


RUBIN_BANDS = list(RUBIN_BAND_ORDER)
NISP_BANDS = [f"nisp_{b}" for b in NISP_BAND_ORDER]
ALL_BANDS = RUBIN_BANDS + NISP_BANDS
GROUP_NAMES = ["rubin", "nisp"]
GROUP_TO_IDX = {name: i for i, name in enumerate(GROUP_NAMES)}
DEFAULT_LENGTH_SCALES = (60.0, 180.0, 600.0)


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _split_csv_float(text: str) -> list[float]:
    vals = [float(x) for x in _split_csv(text)]
    if not vals:
        raise ValueError("expected at least one comma-separated float")
    return vals


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.nanmedian(x)
    return 1.4826 * float(np.nanmedian(np.abs(x - med)))


def radial_mas(offset_arcsec: np.ndarray) -> np.ndarray:
    off = np.asarray(offset_arcsec)
    return np.hypot(off[:, 0], off[:, 1]) * 1000.0


def median_radial_mas(offset_arcsec: np.ndarray) -> float:
    vals = radial_mas(offset_arcsec)
    return float(np.nanmedian(vals)) if vals.size else float("nan")


def madxy_mas(offset_arcsec: np.ndarray) -> float:
    off = np.asarray(offset_arcsec, dtype=np.float64) * 1000.0
    if off.size == 0:
        return float("nan")
    return 0.5 * (robust_sigma(off[:, 0]) + robust_sigma(off[:, 1]))


def band_group(band: str) -> str:
    return "nisp" if band.startswith("nisp_") else "rubin"


def band_to_int(band: str) -> int:
    if band.startswith("nisp_"):
        key = band
    else:
        key = f"rubin_{band}"
    return int(BAND_TO_IDX[key])


def band_fits_prefix(band: str) -> str:
    if band.startswith("nisp_"):
        return band.replace("nisp_", "NISP_")
    return band.upper()


def normalize_band_name(band: str) -> str:
    band = band.strip()
    if band.startswith("rubin_"):
        return band.split("_", 1)[1]
    if band in RUBIN_BANDS:
        return band
    if band.startswith("nisp_"):
        short = band.split("_", 1)[1]
        return f"nisp_{short}"
    if band in NISP_BAND_ORDER:
        return f"nisp_{band}"
    raise ValueError(f"unknown band name: {band}")


def sky_to_tangent_plane(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, float, float]:
    ra0 = float(np.nanmedian(ra))
    dec0 = float(np.nanmedian(dec))
    cosdec = np.cos(np.deg2rad(dec0))
    x = (np.asarray(ra) - ra0) * cosdec * 3600.0
    y = (np.asarray(dec) - dec0) * 3600.0
    return np.stack([x, y], axis=1).astype(np.float32), ra0, dec0


@dataclass
class AnchorTable:
    ra: np.ndarray
    dec: np.ndarray
    pos_shifted: np.ndarray
    offsets: np.ndarray
    snr: np.ndarray
    sigma_arcsec: np.ndarray
    weights: np.ndarray
    band_idx: np.ndarray
    band_name_idx: np.ndarray
    group_idx: np.ndarray
    band_names: list[str]
    n_per_band: dict[str, int]
    ra0: float
    dec0: float
    pos_min: np.ndarray
    pos_max: np.ndarray


@dataclass
class BasisSpec:
    centers: np.ndarray
    scales: np.ndarray
    scale_labels: np.ndarray
    metadata: list[dict]


@dataclass
class HierarchySpec:
    n_base: int
    band_names: list[str]
    use_common: bool
    use_group: bool
    use_band: bool
    common_slice: slice | None
    group_slices: dict[int, slice]
    band_slices: dict[int, slice]
    prior_precision: np.ndarray
    n_features: int


@dataclass
class HGPModel:
    basis: BasisSpec
    hierarchy: HierarchySpec
    coef: np.ndarray
    cho: tuple[np.ndarray, bool]
    posterior_scale: np.ndarray
    jitter: float
    train_summary: dict


def load_anchor_cache(
    cache_path: Path,
    bands: Sequence[str],
    offset_kind: str,
    pool: str,
    snr_min: float,
    snr_classical: float,
    clip_mas: float,
    clip_sigma: float,
    snr_weight_floor: float,
    snr_weight_cap: float,
    snr_weight_power: float,
    min_sigma_mas: float,
    max_sigma_mas: float,
    max_anchors: int | None,
    seed: int,
) -> AnchorTable:
    cache = np.load(cache_path, allow_pickle=True)
    rng = np.random.default_rng(seed)

    offset_suffix = "head_resid" if offset_kind == "head_resid" else "raw"
    ra_list: list[np.ndarray] = []
    dec_list: list[np.ndarray] = []
    off_list: list[np.ndarray] = []
    snr_list: list[np.ndarray] = []
    sigma_list: list[np.ndarray] = []
    weight_list: list[np.ndarray] = []
    band_idx_list: list[np.ndarray] = []
    band_name_idx_list: list[np.ndarray] = []
    group_idx_list: list[np.ndarray] = []
    n_per_band: dict[str, int] = {}

    used_bands: list[str] = []
    for band_name_i, band in enumerate(bands):
        required = [f"{band}_ra", f"{band}_dec", f"{band}_{offset_suffix}"]
        if not all(k in cache.files for k in required):
            print(f"  {band:7s}: missing required keys, skipping")
            continue

        ra = np.asarray(cache[f"{band}_ra"], dtype=np.float64)
        dec = np.asarray(cache[f"{band}_dec"], dtype=np.float64)
        offsets = np.asarray(cache[f"{band}_{offset_suffix}"], dtype=np.float32)
        if f"{band}_snr" in cache.files:
            snr = np.asarray(cache[f"{band}_snr"], dtype=np.float32)
        else:
            snr = np.full(len(offsets), np.nan, dtype=np.float32)

        good = (
            np.isfinite(ra)
            & np.isfinite(dec)
            & np.isfinite(offsets).all(axis=1)
            & np.isfinite(snr)
            & (snr > float(snr_min))
        )
        if pool == "classical":
            good &= snr >= float(snr_classical)
        elif pool == "nonclassical":
            good &= snr < float(snr_classical)
        elif pool != "all":
            raise ValueError(f"unknown pool: {pool}")

        if clip_mas > 0:
            good &= radial_mas(offsets) <= float(clip_mas)

        if good.sum() < 10:
            print(f"  {band:7s}: too few usable anchors after cuts ({good.sum()}), skipping")
            continue

        # Robust per-band outlier clip around the band median vector.
        if clip_sigma > 0:
            off_good = offsets[good]
            med_vec = np.nanmedian(off_good, axis=0)
            resid_mas = radial_mas(off_good - med_vec[None, :])
            sig = robust_sigma(resid_mas)
            if np.isfinite(sig) and sig > 0:
                keep_good = resid_mas <= (np.nanmedian(resid_mas) + clip_sigma * sig)
                good_idx = np.where(good)[0]
                good[good_idx[~keep_good]] = False

        if good.sum() < 10:
            print(f"  {band:7s}: too few usable anchors after robust clip ({good.sum()}), skipping")
            continue

        offsets = offsets[good]
        ra = ra[good]
        dec = dec[good]
        snr = snr[good]

        band_scatter_mas = madxy_mas(offsets)
        if not np.isfinite(band_scatter_mas) or band_scatter_mas <= 0:
            band_scatter_mas = 20.0

        snr_c = np.clip(snr, snr_weight_floor, snr_weight_cap)
        snr_ref = max(float(np.nanmedian(snr_c)), snr_weight_floor)
        rel_w = np.clip((snr_c / snr_ref) ** snr_weight_power, 0.05, 25.0)
        sigma_mas = np.clip(band_scatter_mas / np.sqrt(rel_w), min_sigma_mas, max_sigma_mas)
        sigma_arcsec = (sigma_mas / 1000.0).astype(np.float32)
        weights = (1.0 / np.maximum(sigma_arcsec, min_sigma_mas / 1000.0) ** 2).astype(np.float32)

        n_per_band[band] = int(len(offsets))
        used_bands.append(band)
        ra_list.append(ra)
        dec_list.append(dec)
        off_list.append(offsets)
        snr_list.append(snr.astype(np.float32))
        sigma_list.append(sigma_arcsec)
        weight_list.append(weights)
        band_idx_list.append(np.full(len(offsets), band_to_int(band), dtype=np.int16))
        band_name_idx_list.append(np.full(len(offsets), band_name_i, dtype=np.int16))
        group_idx_list.append(np.full(len(offsets), GROUP_TO_IDX[band_group(band)], dtype=np.int16))
        print(
            f"  {band:7s}: {len(offsets):8,d} anchors, "
            f"med |offset|={median_radial_mas(offsets):5.1f} mas, "
            f"MADxy={band_scatter_mas:5.1f} mas, SNR_ref={snr_ref:5.1f}"
        )

    if not ra_list:
        raise RuntimeError("No usable anchors found in cache")

    ra_all = np.concatenate(ra_list)
    dec_all = np.concatenate(dec_list)
    offsets_all = np.concatenate(off_list).astype(np.float32)
    snr_all = np.concatenate(snr_list).astype(np.float32)
    sigma_all = np.concatenate(sigma_list).astype(np.float32)
    weights_all = np.concatenate(weight_list).astype(np.float32)
    band_idx_all = np.concatenate(band_idx_list)
    band_name_idx_all = np.concatenate(band_name_idx_list)
    group_idx_all = np.concatenate(group_idx_list)

    if max_anchors is not None and max_anchors > 0 and len(ra_all) > max_anchors:
        keep = rng.choice(len(ra_all), size=int(max_anchors), replace=False)
        keep.sort()
        ra_all = ra_all[keep]
        dec_all = dec_all[keep]
        offsets_all = offsets_all[keep]
        snr_all = snr_all[keep]
        sigma_all = sigma_all[keep]
        weights_all = weights_all[keep]
        band_idx_all = band_idx_all[keep]
        band_name_idx_all = band_name_idx_all[keep]
        group_idx_all = group_idx_all[keep]
        n_per_band = {
            band: int((band_name_idx_all == i).sum())
            for i, band in enumerate(bands)
            if int((band_name_idx_all == i).sum()) > 0
        }
        print(f"  Subsampled to {len(ra_all):,} anchors for this run")

    pos, ra0, dec0 = sky_to_tangent_plane(ra_all, dec_all)
    pos_min = pos.min(axis=0).astype(np.float32)
    pos_max = pos.max(axis=0).astype(np.float32)
    pos_shifted = (pos - pos_min[None, :]).astype(np.float32)

    print(f"\nLoaded {len(ra_all):,} anchors from {cache_path}")
    print(f"  offset kind       : {offset_kind}")
    print(f"  pool              : {pool}")
    print(f"  median |offset|   : {median_radial_mas(offsets_all):.2f} mas")
    print(f"  field extent      : {np.ptp(pos[:, 0]):.0f} x {np.ptp(pos[:, 1]):.0f} arcsec")

    return AnchorTable(
        ra=ra_all,
        dec=dec_all,
        pos_shifted=pos_shifted,
        offsets=offsets_all,
        snr=snr_all,
        sigma_arcsec=sigma_all,
        weights=weights_all,
        band_idx=band_idx_all,
        band_name_idx=band_name_idx_all,
        group_idx=group_idx_all,
        band_names=list(bands),
        n_per_band=n_per_band,
        ra0=ra0,
        dec0=dec0,
        pos_min=pos_min,
        pos_max=pos_max,
    )


def make_basis(
    pos_shifted: np.ndarray,
    length_scales: Sequence[float],
    spacing_factor: float,
    max_centers_per_scale: int,
    support_factor: float,
) -> BasisSpec:
    extent = np.maximum(pos_shifted.max(axis=0), 1.0)
    area = max(float(extent[0] * extent[1]), 1.0)
    centers_all: list[np.ndarray] = []
    scales_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    metadata: list[dict] = []

    for scale_i, length_scale in enumerate(length_scales):
        length_scale = float(length_scale)
        if length_scale <= 0:
            raise ValueError("length scales must be positive")
        target_spacing = max(spacing_factor * length_scale, 1.0)
        budget_spacing = math.sqrt(area / max(1, int(max_centers_per_scale)))
        spacing = max(target_spacing, budget_spacing)

        pad = support_factor * length_scale
        xs = np.arange(-pad, extent[0] + pad + 0.5 * spacing, spacing, dtype=np.float32)
        ys = np.arange(-pad, extent[1] + pad + 0.5 * spacing, spacing, dtype=np.float32)
        if len(xs) * len(ys) > max_centers_per_scale:
            spacing *= math.sqrt((len(xs) * len(ys)) / max_centers_per_scale)
            xs = np.arange(-pad, extent[0] + pad + 0.5 * spacing, spacing, dtype=np.float32)
            ys = np.arange(-pad, extent[1] + pad + 0.5 * spacing, spacing, dtype=np.float32)

        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        centers = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        scales = np.full(len(centers), length_scale, dtype=np.float32)
        labels = np.full(len(centers), scale_i, dtype=np.int16)
        centers_all.append(centers)
        scales_all.append(scales)
        labels_all.append(labels)
        metadata.append(
            {
                "length_scale_arcsec": length_scale,
                "spacing_arcsec": float(spacing),
                "n_centers": int(len(centers)),
            }
        )
        print(
            f"  basis scale {length_scale:7.1f}\" : "
            f"{len(centers):5,d} centers, spacing={spacing:.1f}\""
        )

    return BasisSpec(
        centers=np.concatenate(centers_all, axis=0),
        scales=np.concatenate(scales_all),
        scale_labels=np.concatenate(labels_all),
        metadata=metadata,
    )


def make_hierarchy(
    n_base: int,
    band_names: Sequence[str],
    use_common: bool,
    use_group: bool,
    use_band: bool,
    prior_common_mas: float,
    prior_group_mas: float,
    prior_band_mas: float,
) -> HierarchySpec:
    start = 0
    common_slice = None
    group_slices: dict[int, slice] = {}
    band_slices: dict[int, slice] = {}
    prior_blocks: list[np.ndarray] = []

    def add_block(prior_std_mas: float) -> slice:
        nonlocal start
        sl = slice(start, start + n_base)
        start += n_base
        prior_std_arcsec = max(float(prior_std_mas), 1e-3) / 1000.0
        prior_blocks.append(np.full(n_base, 1.0 / prior_std_arcsec**2, dtype=np.float64))
        return sl

    if use_common:
        common_slice = add_block(prior_common_mas)
    if use_group:
        for group_i in range(len(GROUP_NAMES)):
            group_slices[group_i] = add_block(prior_group_mas)
    if use_band:
        for band_i, _ in enumerate(band_names):
            band_slices[band_i] = add_block(prior_band_mas)

    if not prior_blocks:
        raise ValueError("at least one hierarchy component must be enabled")

    prior_precision = np.concatenate(prior_blocks)
    return HierarchySpec(
        n_base=n_base,
        band_names=list(band_names),
        use_common=use_common,
        use_group=use_group,
        use_band=use_band,
        common_slice=common_slice,
        group_slices=group_slices,
        band_slices=band_slices,
        prior_precision=prior_precision,
        n_features=start,
    )


def base_features(pos: np.ndarray, basis: BasisSpec, support_factor: float) -> np.ndarray:
    pos64 = np.asarray(pos, dtype=np.float64)
    centers = basis.centers.astype(np.float64, copy=False)
    scales = basis.scales.astype(np.float64, copy=False)
    dx = pos64[:, None, 0] - centers[None, :, 0]
    dy = pos64[:, None, 1] - centers[None, :, 1]
    r2 = dx * dx + dy * dy
    scale2 = scales[None, :] ** 2
    phi = np.exp(-0.5 * r2 / scale2)
    support2 = (support_factor * scales[None, :]) ** 2
    phi[r2 > support2] = 0.0
    return phi.astype(np.float64, copy=False)


def design_matrix(
    pos: np.ndarray,
    group_idx: np.ndarray,
    band_name_idx: np.ndarray,
    basis: BasisSpec,
    hierarchy: HierarchySpec,
    support_factor: float,
    component: tuple[str, int] | None = None,
) -> sparse.csr_matrix:
    base_dense = base_features(pos, basis, support_factor)
    base = sparse.csr_matrix(base_dense)
    blocks: list[sparse.csr_matrix] = []

    def masked_base(mask: np.ndarray) -> sparse.csr_matrix:
        if mask.all():
            return base
        return base.multiply(mask.astype(np.float64)[:, None]).tocsr()

    if hierarchy.common_slice is not None:
        if component is None or component == ("common", -1):
            blocks.append(base)
        else:
            blocks.append(sparse.csr_matrix((len(pos), hierarchy.n_base), dtype=np.float64))

    if hierarchy.use_group:
        for group_i, sl in hierarchy.group_slices.items():
            del sl
            if component is None or component == ("group", group_i):
                blocks.append(masked_base(group_idx == group_i))
            else:
                blocks.append(sparse.csr_matrix((len(pos), hierarchy.n_base), dtype=np.float64))

    if hierarchy.use_band:
        for band_i, sl in hierarchy.band_slices.items():
            del sl
            if component is None or component == ("band", band_i):
                blocks.append(masked_base(band_name_idx == band_i))
            else:
                blocks.append(sparse.csr_matrix((len(pos), hierarchy.n_base), dtype=np.float64))

    if not blocks:
        return sparse.csr_matrix((len(pos), hierarchy.n_features), dtype=np.float64)
    return sparse.hstack(blocks, format="csr")


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def accumulate_precision(
    anchors: AnchorTable,
    basis: BasisSpec,
    hierarchy: HierarchySpec,
    support_factor: float,
    train_idx: np.ndarray,
    robust_weights: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = hierarchy.n_features
    precision = np.diag(hierarchy.prior_precision).astype(np.float64)
    rhs = np.zeros((n_features, 2), dtype=np.float64)
    y = anchors.offsets.astype(np.float64)

    for idx in iter_batches(train_idx, batch_size):
        x = design_matrix(
            anchors.pos_shifted[idx],
            anchors.group_idx[idx],
            anchors.band_name_idx[idx],
            basis,
            hierarchy,
            support_factor,
        )
        w = (anchors.weights[idx] * robust_weights[idx]).astype(np.float64)
        wx = x.multiply(np.sqrt(w)[:, None])
        precision += (wx.T @ wx).toarray()
        rhs += np.asarray(x.T @ (w[:, None] * y[idx]))

    return precision, rhs


def predict(
    model: HGPModel,
    pos: np.ndarray,
    group_idx: np.ndarray,
    band_name_idx: np.ndarray,
    support_factor: float,
    batch_size: int,
    return_std: bool = False,
    component: tuple[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    all_idx = np.arange(len(pos))
    for idx in iter_batches(all_idx, batch_size):
        x = design_matrix(
            pos[idx],
            group_idx[idx],
            band_name_idx[idx],
            model.basis,
            model.hierarchy,
            support_factor,
            component=component,
        )
        means.append(np.asarray(x @ model.coef).astype(np.float32))
        if return_std:
            xt = x.T.toarray()
            solved = linalg.cho_solve(model.cho, xt, check_finite=False)
            var = np.asarray(x.multiply(solved.T).sum(axis=1)).ravel()
            var = np.maximum(var, 0.0)
            std = np.stack(
                [
                    np.sqrt(var) * float(model.posterior_scale[0]),
                    np.sqrt(var) * float(model.posterior_scale[1]),
                ],
                axis=1,
            )
            stds.append(std.astype(np.float32))
    mean = np.concatenate(means, axis=0)
    std = np.concatenate(stds, axis=0) if return_std else None
    return mean, std


def fit_hgp(
    anchors: AnchorTable,
    basis: BasisSpec,
    hierarchy: HierarchySpec,
    support_factor: float,
    train_mask: np.ndarray,
    robust_iters: int,
    huber_k: float,
    batch_size: int,
    jitter: float,
) -> HGPModel:
    train_idx = np.where(train_mask)[0]
    if len(train_idx) < max(10, hierarchy.n_features // 3):
        print(
            f"Warning: only {len(train_idx):,} training anchors for "
            f"{hierarchy.n_features:,} features; expect strong prior shrinkage."
        )

    robust_weights = np.ones(len(anchors.offsets), dtype=np.float32)
    coef = np.zeros((hierarchy.n_features, 2), dtype=np.float64)
    cho = None
    precision = None
    t0 = time.time()
    n_iter = max(1, int(robust_iters))

    for it in range(n_iter):
        print(f"  HGP solve iteration {it + 1}/{n_iter}")
        precision, rhs = accumulate_precision(
            anchors,
            basis,
            hierarchy,
            support_factor,
            train_idx,
            robust_weights,
            batch_size,
        )
        if jitter > 0:
            precision.flat[:: precision.shape[0] + 1] += float(jitter)
        cho = linalg.cho_factor(precision, lower=True, check_finite=False)
        coef = linalg.cho_solve(cho, rhs, check_finite=False)

        if it == n_iter - 1:
            break

        pred, _ = predict(
            HGPModel(
                basis=basis,
                hierarchy=hierarchy,
                coef=coef,
                cho=cho,
                posterior_scale=np.ones(2, dtype=np.float64),
                jitter=jitter,
                train_summary={},
            ),
            anchors.pos_shifted[train_idx],
            anchors.group_idx[train_idx],
            anchors.band_name_idx[train_idx],
            support_factor,
            batch_size,
            return_std=False,
        )
        resid = anchors.offsets[train_idx].astype(np.float64) - pred.astype(np.float64)
        sigma = np.maximum(anchors.sigma_arcsec[train_idx].astype(np.float64), 1e-5)
        r_norm = np.hypot(resid[:, 0] / sigma, resid[:, 1] / sigma)
        new_w = np.ones_like(r_norm, dtype=np.float32)
        bad = r_norm > float(huber_k)
        new_w[bad] = (float(huber_k) / np.maximum(r_norm[bad], 1e-6)).astype(np.float32)
        robust_weights[train_idx] = new_w
        print(
            f"    robust downweight: {(new_w < 1).sum():,}/{len(new_w):,} "
            f"({100.0 * (new_w < 1).mean():.1f}%)"
        )

    assert cho is not None and precision is not None

    pred_train, _ = predict(
        HGPModel(
            basis=basis,
            hierarchy=hierarchy,
            coef=coef,
            cho=cho,
            posterior_scale=np.ones(2, dtype=np.float64),
            jitter=jitter,
            train_summary={},
        ),
        anchors.pos_shifted[train_idx],
        anchors.group_idx[train_idx],
        anchors.band_name_idx[train_idx],
        support_factor,
        batch_size,
        return_std=False,
    )
    resid = anchors.offsets[train_idx].astype(np.float64) - pred_train.astype(np.float64)
    w = (anchors.weights[train_idx] * robust_weights[train_idx]).astype(np.float64)
    dof = max(len(train_idx) - hierarchy.n_features, max(1, len(train_idx) // 2))
    chi2 = np.array(
        [
            float(np.sum(w * resid[:, 0] ** 2) / dof),
            float(np.sum(w * resid[:, 1] ** 2) / dof),
        ],
        dtype=np.float64,
    )
    posterior_scale = np.sqrt(np.maximum(chi2, 1e-6))
    train_summary = {
        "n_train": int(len(train_idx)),
        "n_features": int(hierarchy.n_features),
        "robust_downweighted": int((robust_weights[train_idx] < 1.0).sum()),
        "train_resid_med_mas": median_radial_mas(resid),
        "train_resid_madxy_mas": madxy_mas(resid),
        "posterior_scale_dra": float(posterior_scale[0]),
        "posterior_scale_ddec": float(posterior_scale[1]),
        "solve_seconds": float(time.time() - t0),
    }
    print(
        f"  train residual: med={train_summary['train_resid_med_mas']:.2f} mas, "
        f"MADxy={train_summary['train_resid_madxy_mas']:.2f} mas"
    )

    return HGPModel(
        basis=basis,
        hierarchy=hierarchy,
        coef=coef,
        cho=cho,
        posterior_scale=posterior_scale,
        jitter=jitter,
        train_summary=train_summary,
    )


def make_holdout_mask(
    pos_shifted: np.ndarray,
    frac: float,
    mode: str,
    block_arcsec: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(pos_shifted)
    if frac <= 0:
        return np.zeros(n, dtype=bool)
    frac = min(max(float(frac), 0.0), 0.8)

    if mode == "random":
        return rng.random(n) < frac
    if mode != "spatial":
        raise ValueError(f"unknown holdout mode: {mode}")

    block_arcsec = max(float(block_arcsec), 1.0)
    blocks_x = np.floor(pos_shifted[:, 0] / block_arcsec).astype(np.int64)
    blocks_y = np.floor(pos_shifted[:, 1] / block_arcsec).astype(np.int64)
    block_id = blocks_x * 1_000_003 + blocks_y
    unique = np.unique(block_id)
    rng.shuffle(unique)

    holdout = np.zeros(n, dtype=bool)
    target = int(round(frac * n))
    for bid in unique:
        holdout |= block_id == bid
        if holdout.sum() >= target:
            break
    return holdout


def evaluate_holdout(
    model: HGPModel,
    anchors: AnchorTable,
    holdout_mask: np.ndarray,
    support_factor: float,
    batch_size: int,
    max_eval: int,
    seed: int,
) -> tuple[dict, float]:
    idx = np.where(holdout_mask)[0]
    if len(idx) == 0:
        return {}, 1.0
    if max_eval > 0 and len(idx) > max_eval:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_eval, replace=False)
        idx.sort()

    mean, std = predict(
        model,
        anchors.pos_shifted[idx],
        anchors.group_idx[idx],
        anchors.band_name_idx[idx],
        support_factor,
        batch_size,
        return_std=True,
    )
    resid = anchors.offsets[idx].astype(np.float64) - mean.astype(np.float64)
    sigma = np.maximum(anchors.sigma_arcsec[idx].astype(np.float64), 1e-5)
    pred_sigma = np.sqrt(sigma[:, None] ** 2 + np.maximum(std.astype(np.float64), 0.0) ** 2)
    z = resid / pred_sigma
    z_std = np.nanstd(z, axis=0)
    z_std_scalar = float(np.nanmean(z_std))
    cal = max(1.0, z_std_scalar)
    summary = {
        "n_holdout_eval": int(len(idx)),
        "holdout_resid_med_mas": median_radial_mas(resid),
        "holdout_resid_madxy_mas": madxy_mas(resid),
        "z_std_dra": float(z_std[0]),
        "z_std_ddec": float(z_std[1]),
        "uncertainty_calibration_factor": float(cal),
    }
    print(
        f"  holdout residual: med={summary['holdout_resid_med_mas']:.2f} mas, "
        f"MADxy={summary['holdout_resid_madxy_mas']:.2f} mas, "
        f"std(z)=({summary['z_std_dra']:.2f}, {summary['z_std_ddec']:.2f})"
    )
    return summary, cal


def build_output_wcs(anchors: AnchorTable, dstep_arcsec: float) -> fits.Header:
    w = WCS(naxis=2)
    w.wcs.crpix = [
        1.0 - float(anchors.pos_min[0]) / float(dstep_arcsec),
        1.0 - float(anchors.pos_min[1]) / float(dstep_arcsec),
    ]
    w.wcs.crval = [float(anchors.ra0), float(anchors.dec0)]
    w.wcs.cdelt = [float(dstep_arcsec) / 3600.0, float(dstep_arcsec) / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    return w.to_header()


def mesh_grid(anchors: AnchorTable, dstep_arcsec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extent = anchors.pos_max - anchors.pos_min
    xs = np.arange(0.0, float(extent[0]) + dstep_arcsec, dstep_arcsec, dtype=np.float32)
    ys = np.arange(0.0, float(extent[1]) + dstep_arcsec, dstep_arcsec, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pos = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return xs, ys, pos


def output_hdu(
    data: np.ndarray,
    name: str,
    header: fits.Header,
    dstep_arcsec: float,
    unit: str,
    comment: str,
    target_band: str | None = None,
    n_sources: int | None = None,
    grid_shape: tuple[int, int] | None = None,
    method: str = "hier_basis_gp",
) -> fits.ImageHDU:
    hdu = fits.ImageHDU(data=data.astype(np.float32), name=name)
    hdu.header.update(header)
    hdu.header["DSTEP"] = (float(dstep_arcsec), "Mesh step size in arcsec")
    hdu.header["DUNIT"] = (unit, "Unit of image values")
    hdu.header["BUNIT"] = (unit, "Image value unit")
    hdu.header["INTERP"] = ("bilinear", "Recommended interpolation method")
    hdu.header["CONCRDNC"] = (True, "Global concordance field")
    hdu.header["REFFRAME"] = ("euclid_VIS", "Reference astrometric frame")
    hdu.header["SOLVETYP"] = ("global_hgp", "Global hierarchical GP-style solve")
    hdu.header["METHOD"] = (method, "Concordance method")
    if target_band is not None:
        hdu.header["TGTBAND"] = (target_band, "Target band")
    if n_sources is not None:
        hdu.header["NSRC"] = (int(n_sources), "Total anchors used in solve")
    if grid_shape is not None:
        hdu.header["GRIDH"] = (int(grid_shape[0]), "Output mesh height")
        hdu.header["GRIDW"] = (int(grid_shape[1]), "Output mesh width")
    hdu.header["COMMENT"] = comment
    return hdu


def write_fits(
    output: Path,
    model: HGPModel,
    anchors: AnchorTable,
    args: argparse.Namespace,
    holdout_summary: dict,
) -> dict:
    dstep = float(args.dstep_arcsec)
    xs, ys, mesh_pos = mesh_grid(anchors, dstep)
    mesh_shape = (len(ys), len(xs))
    header = build_output_wcs(anchors, dstep)

    hdus: list[fits.ImageHDU] = []
    summary: dict = {
        "output": str(output),
        "mesh_shape": [int(mesh_shape[0]), int(mesh_shape[1])],
        "bands": {},
        "components": {},
    }

    for band_i, band in enumerate(anchors.band_names):
        if anchors.n_per_band.get(band, 0) <= 0:
            continue
        group_i = GROUP_TO_IDX[band_group(band)]
        group_arr = np.full(len(mesh_pos), group_i, dtype=np.int16)
        band_arr = np.full(len(mesh_pos), band_i, dtype=np.int16)
        mean, std = predict(
            model,
            mesh_pos,
            group_arr,
            band_arr,
            float(args.support_factor),
            int(args.batch_size),
            return_std=True,
        )
        dra = mean[:, 0].reshape(mesh_shape)
        ddec = mean[:, 1].reshape(mesh_shape)
        sdra = std[:, 0].reshape(mesh_shape)
        sddec = std[:, 1].reshape(mesh_shape)
        prefix = band_fits_prefix(band)
        hdu_kwargs = {
            "target_band": band,
            "n_sources": len(anchors.ra),
            "grid_shape": mesh_shape,
        }
        hdus.append(output_hdu(dra, f"{prefix}.DRA", header, dstep, "arcsec", f"{band} dRA* mean", **hdu_kwargs))
        hdus.append(output_hdu(ddec, f"{prefix}.DDE", header, dstep, "arcsec", f"{band} dDec mean", **hdu_kwargs))
        hdus.append(output_hdu(sdra, f"{prefix}.DRA_STD", header, dstep, "arcsec", f"{band} dRA* posterior std", **hdu_kwargs))
        hdus.append(output_hdu(sddec, f"{prefix}.DDE_STD", header, dstep, "arcsec", f"{band} dDec posterior std", **hdu_kwargs))
        summary["bands"][band] = {
            "n_anchors": int(anchors.n_per_band.get(band, 0)),
            "field_rms_mas": float(np.sqrt(np.nanmean(dra**2 + ddec**2)) * 1000.0),
            "median_std_mas": float(np.nanmedian(np.hypot(sdra, sddec)) * 1000.0),
        }
        print(
            f"  {band:7s}: field RMS={summary['bands'][band]['field_rms_mas']:.2f} mas, "
            f"median posterior std={summary['bands'][band]['median_std_mas']:.2f} mas"
        )

    if args.save_components:
        component_specs: list[tuple[str, tuple[str, int]]] = []
        if model.hierarchy.common_slice is not None:
            component_specs.append(("COMMON", ("common", -1)))
        for group_i, group_name in enumerate(GROUP_NAMES):
            if group_i in model.hierarchy.group_slices:
                component_specs.append((f"GROUP_{group_name.upper()}", ("group", group_i)))
        for band_i, band in enumerate(anchors.band_names):
            if band_i in model.hierarchy.band_slices:
                component_specs.append((f"BAND_{band_fits_prefix(band)}", ("band", band_i)))

        for name, component in component_specs:
            if component[0] == "group":
                group_i = component[1]
                band_i = 0
            elif component[0] == "band":
                band_i = component[1]
                group_i = GROUP_TO_IDX[band_group(anchors.band_names[band_i])]
            else:
                group_i = 0
                band_i = 0
            group_arr = np.full(len(mesh_pos), group_i, dtype=np.int16)
            band_arr = np.full(len(mesh_pos), band_i, dtype=np.int16)
            mean, _ = predict(
                model,
                mesh_pos,
                group_arr,
                band_arr,
                float(args.support_factor),
                int(args.batch_size),
                return_std=False,
                component=component,
            )
            dra = mean[:, 0].reshape(mesh_shape)
            ddec = mean[:, 1].reshape(mesh_shape)
            hdus.append(output_hdu(
                dra, f"COMP_{name}_DRA", header, dstep, "arcsec", f"{name} dRA* component",
                target_band=name, n_sources=len(anchors.ra), grid_shape=mesh_shape,
            ))
            hdus.append(output_hdu(
                ddec, f"COMP_{name}_DDE", header, dstep, "arcsec", f"{name} dDec component",
                target_band=name, n_sources=len(anchors.ra), grid_shape=mesh_shape,
            ))
            summary["components"][name] = {
                "field_rms_mas": float(np.sqrt(np.nanmean(dra**2 + ddec**2)) * 1000.0),
            }

    if args.write_coverage:
        tree = cKDTree(anchors.pos_shifted.astype(np.float32))
        dist, _ = tree.query(mesh_pos, workers=-1)
        coverage = dist.reshape(mesh_shape).astype(np.float32)
        hdus.append(output_hdu(
            coverage, "COVERAGE", header, dstep, "arcsec", "Distance to nearest anchor",
            target_band="ALL", n_sources=len(anchors.ra), grid_shape=mesh_shape,
        ))

    primary = fits.PrimaryHDU()
    primary.header["CONCRDNC"] = (True, "JAISP hierarchical GP concordance product")
    primary.header["METHOD"] = ("hier_basis_gp", "Hierarchical finite-rank GP")
    primary.header["ANCHORS"] = (Path(args.anchors).name, "Anchor cache")
    primary.header["OFFKIND"] = (args.offset_kind, "Offset key used")
    primary.header["POOL"] = (args.pool, "SNR pool")
    primary.header["NBANDS"] = (len([b for b in anchors.band_names if anchors.n_per_band.get(b, 0) > 0]), "Bands fitted")
    primary.header["NANCHOR"] = (len(anchors.ra), "Total anchors")
    primary.header["NFEAT"] = (model.hierarchy.n_features, "Basis features")
    primary.header["DSTEP"] = (float(args.dstep_arcsec), "Mesh step arcsec")
    primary.header["RA0"] = (float(anchors.ra0), "Reference RA deg")
    primary.header["DEC0"] = (float(anchors.dec0), "Reference Dec deg")
    primary.header["CALFAC"] = (float(holdout_summary.get("uncertainty_calibration_factor", 1.0)), "Posterior std calibration")

    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary] + hdus).writeto(output, overwrite=True)
    print(f"\nWrote {output}")
    print(f"  extensions: {len(hdus)}")
    return summary


def save_summary(
    path: Path,
    args: argparse.Namespace,
    anchors: AnchorTable,
    basis: BasisSpec,
    hierarchy: HierarchySpec,
    model: HGPModel,
    holdout_summary: dict,
    output_summary: dict,
) -> None:
    summary = {
        "args": vars(args),
        "n_anchors": int(len(anchors.ra)),
        "n_per_band": {k: int(v) for k, v in anchors.n_per_band.items()},
        "ra0": float(anchors.ra0),
        "dec0": float(anchors.dec0),
        "pos_min_arcsec": anchors.pos_min.astype(float).tolist(),
        "pos_max_arcsec": anchors.pos_max.astype(float).tolist(),
        "basis": {
            "n_base": int(basis.centers.shape[0]),
            "scales": basis.metadata,
        },
        "hierarchy": {
            "n_features": int(hierarchy.n_features),
            "use_common": bool(hierarchy.use_common),
            "use_group": bool(hierarchy.use_group),
            "use_band": bool(hierarchy.use_band),
        },
        "train": model.train_summary,
        "holdout": holdout_summary,
        "output": output_summary,
    }
    out = path.with_suffix(path.suffix + ".json") if path.suffix else Path(str(path) + ".json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary {out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit a global hierarchical GP-style concordance field from saved anchors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--anchors", type=str, required=True, help="Anchor .npz from eval_latent_position.py --save-anchors.")
    p.add_argument("--output", type=str, required=True, help="Output FITS path.")
    p.add_argument("--offset-kind", choices=["head_resid", "raw"], default="head_resid")
    p.add_argument("--bands", type=str, default=",".join(ALL_BANDS), help="Comma-separated bands. Use u,g,...,nisp_Y.")
    p.add_argument("--pool", choices=["all", "classical", "nonclassical"], default="all")
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-classical", type=float, default=30.0)
    p.add_argument("--clip-mas", type=float, default=200.0, help="Hard radial offset clip. 0 disables.")
    p.add_argument("--clip-sigma", type=float, default=5.0, help="Per-band robust radial clip. 0 disables.")
    p.add_argument("--snr-weight-floor", type=float, default=3.0)
    p.add_argument("--snr-weight-cap", type=float, default=150.0)
    p.add_argument("--snr-weight-power", type=float, default=1.0, help="Relative SNR weight exponent; 2 is classical inverse variance.")
    p.add_argument("--min-sigma-mas", type=float, default=2.0)
    p.add_argument("--max-sigma-mas", type=float, default=80.0)
    p.add_argument("--length-scales", type=str, default=",".join(str(x) for x in DEFAULT_LENGTH_SCALES))
    p.add_argument("--spacing-factor", type=float, default=1.0)
    p.add_argument("--max-centers-per-scale", type=int, default=120)
    p.add_argument("--support-factor", type=float, default=2.5)
    p.add_argument("--no-common", action="store_true", help="Disable shared common field component.")
    p.add_argument("--no-group", action="store_true", help="Disable Rubin/NISP group components.")
    p.add_argument("--no-band", action="store_true", help="Disable band-specific components.")
    p.add_argument("--prior-common-mas", type=float, default=25.0)
    p.add_argument("--prior-group-mas", type=float, default=12.0)
    p.add_argument("--prior-band-mas", type=float, default=6.0)
    p.add_argument("--robust-iters", type=int, default=3)
    p.add_argument("--huber-k", type=float, default=3.0)
    p.add_argument("--jitter", type=float, default=1e-8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--holdout-mode", choices=["spatial", "random"], default="spatial")
    p.add_argument("--holdout-block-arcsec", type=float, default=300.0)
    p.add_argument("--max-holdout-eval", type=int, default=20000)
    p.add_argument("--dstep-arcsec", type=float, default=5.0)
    p.add_argument("--save-components", action="store_true", help="Write common/group/band component maps.")
    p.add_argument("--write-coverage", action="store_true", help="Write nearest-anchor distance map.")
    p.add_argument("--max-anchors", type=int, default=0, help="Random subsample for testing. 0 uses all.")
    p.add_argument("--dry-run", action="store_true", help="Load anchors and build basis, then exit before fitting.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    t0 = time.time()
    bands = [normalize_band_name(b) for b in _split_csv(args.bands)]
    max_anchors = int(args.max_anchors) if int(args.max_anchors) > 0 else None

    print("Loading anchors")
    anchors = load_anchor_cache(
        cache_path=Path(args.anchors),
        bands=bands,
        offset_kind=args.offset_kind,
        pool=args.pool,
        snr_min=float(args.snr_min),
        snr_classical=float(args.snr_classical),
        clip_mas=float(args.clip_mas),
        clip_sigma=float(args.clip_sigma),
        snr_weight_floor=float(args.snr_weight_floor),
        snr_weight_cap=float(args.snr_weight_cap),
        snr_weight_power=float(args.snr_weight_power),
        min_sigma_mas=float(args.min_sigma_mas),
        max_sigma_mas=float(args.max_sigma_mas),
        max_anchors=max_anchors,
        seed=int(args.seed),
    )

    print("\nBuilding multi-scale basis")
    basis = make_basis(
        anchors.pos_shifted,
        length_scales=_split_csv_float(args.length_scales),
        spacing_factor=float(args.spacing_factor),
        max_centers_per_scale=int(args.max_centers_per_scale),
        support_factor=float(args.support_factor),
    )
    hierarchy = make_hierarchy(
        n_base=int(basis.centers.shape[0]),
        band_names=anchors.band_names,
        use_common=not bool(args.no_common),
        use_group=not bool(args.no_group),
        use_band=not bool(args.no_band),
        prior_common_mas=float(args.prior_common_mas),
        prior_group_mas=float(args.prior_group_mas),
        prior_band_mas=float(args.prior_band_mas),
    )
    print(f"  total base functions: {basis.centers.shape[0]:,}")
    print(f"  hierarchy features  : {hierarchy.n_features:,}")

    if args.dry_run:
        print("Dry run requested; stopping before fit.")
        return 0

    holdout_mask = make_holdout_mask(
        anchors.pos_shifted,
        frac=float(args.holdout_frac),
        mode=args.holdout_mode,
        block_arcsec=float(args.holdout_block_arcsec),
        seed=int(args.seed),
    )
    print(f"\nHoldout anchors: {holdout_mask.sum():,}/{len(holdout_mask):,}")

    holdout_summary: dict = {}
    calibration = 1.0
    if holdout_mask.any():
        print("\nFitting holdout-training model")
        holdout_model = fit_hgp(
            anchors,
            basis,
            hierarchy,
            support_factor=float(args.support_factor),
            train_mask=~holdout_mask,
            robust_iters=int(args.robust_iters),
            huber_k=float(args.huber_k),
            batch_size=int(args.batch_size),
            jitter=float(args.jitter),
        )
        holdout_summary, calibration = evaluate_holdout(
            holdout_model,
            anchors,
            holdout_mask,
            support_factor=float(args.support_factor),
            batch_size=int(args.batch_size),
            max_eval=int(args.max_holdout_eval),
            seed=int(args.seed),
        )

    print("\nFitting final model on all anchors")
    final_model = fit_hgp(
        anchors,
        basis,
        hierarchy,
        support_factor=float(args.support_factor),
        train_mask=np.ones(len(anchors.ra), dtype=bool),
        robust_iters=int(args.robust_iters),
        huber_k=float(args.huber_k),
        batch_size=int(args.batch_size),
        jitter=float(args.jitter),
    )
    final_model.posterior_scale *= float(calibration)
    if calibration != 1.0:
        print(f"  Applied conservative posterior-std calibration factor: {calibration:.2f}")

    print("\nWriting FITS maps")
    output_summary = write_fits(Path(args.output), final_model, anchors, args, holdout_summary)
    save_summary(Path(args.output), args, anchors, basis, hierarchy, final_model, holdout_summary, output_summary)

    print(f"\nDone in {(time.time() - t0) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
