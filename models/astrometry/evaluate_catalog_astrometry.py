"""
Evaluate matched-source astrometric residuals before/after concordance correction.

This script is intended to mirror the robust workflow used in io/00_joint_visual_check:
  - one-to-one greedy nearest-neighbor matching within a max separation
  - robust component-wise MAD clipping
  - residual metrics in milliarcseconds (mas)

It accepts two source catalogs (reference and candidate) with RA/Dec columns.
If a concordance FITS file is provided, it samples DRA/DDE corrections and evaluates
"after correction" metrics as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter, zoom

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, search_around_sky
    from astropy.io import fits
    from astropy.table import Table
    from astropy.wcs import WCS
except ImportError as e:
    raise SystemExit(
        "This evaluator requires astropy. Install with `pip install astropy` "
        "in your runtime environment."
    ) from e


_RUBIN_BAND_ORDER = ["u", "g", "r", "i", "z", "y"]


def _extract_npz_scalar(obj):
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()
    return obj


def _decode_header_string(value) -> str:
    v = _extract_npz_scalar(value)
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def _parse_fits_header_string(header_text: str) -> fits.Header:
    # Euclid NPZ tiles usually store a 2880-byte FITS card block without newlines.
    # Some pipelines may store newline-separated card text instead.
    text = str(header_text).replace("\x00", "")
    if "\n" in text:
        try:
            return fits.Header.fromstring(text, sep="\n")
        except Exception:
            return fits.Header.fromstring(text, sep="")
    try:
        return fits.Header.fromstring(text, sep="")
    except Exception:
        return fits.Header.fromstring(text, sep="\n")


def _wcs_from_rubin_hdr_dict(hdr_dict: Dict[str, object]) -> WCS:
    # Rubin tiles store a FITS-style key/value dict.
    hdr = fits.Header()
    for k, v in hdr_dict.items():
        try:
            hdr[k] = v
        except Exception:
            hdr[str(k)] = str(v)
    w = WCS(hdr)
    if not w.has_celestial:
        raise ValueError("Rubin tile WCS header does not contain celestial axes.")
    return w


def _load_rubin_tile_for_autocat(path: str, band_key: str) -> Tuple[np.ndarray, WCS]:
    band = band_key.lower().replace("rubin_", "")
    if band not in _RUBIN_BAND_ORDER:
        raise ValueError(f"Unknown Rubin band '{band_key}'. Expected one of {_RUBIN_BAND_ORDER}.")
    idx = _RUBIN_BAND_ORDER.index(band)

    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to load Rubin tile with pickled WCS metadata. "
            "Use an environment with compatible numpy/pickle support."
        ) from e

    if "img" not in d:
        raise KeyError(f"Rubin tile missing 'img': {path}")
    img = np.asarray(d["img"][idx], dtype=float)

    if "wcs_hdr" not in d:
        raise KeyError(f"Rubin tile missing 'wcs_hdr': {path}")
    wcs_hdr_obj = _extract_npz_scalar(d["wcs_hdr"])
    if not isinstance(wcs_hdr_obj, dict):
        raise TypeError(f"Unexpected Rubin wcs_hdr type: {type(wcs_hdr_obj)}")

    wcs = _wcs_from_rubin_hdr_dict(wcs_hdr_obj)
    return img, wcs


def _load_euclid_tile_for_autocat(path: str, band_key: str = "VIS") -> Tuple[np.ndarray, WCS]:
    band = band_key.upper()
    d = np.load(path, allow_pickle=False)

    img_key = f"img_{band}"
    wcs_key = f"wcs_{band}"
    if img_key not in d.files:
        raise KeyError(f"Euclid tile missing '{img_key}': {path}")
    if wcs_key not in d.files:
        raise KeyError(f"Euclid tile missing '{wcs_key}': {path}")

    img = np.asarray(d[img_key], dtype=float)
    wcs_header_str = _decode_header_string(d[wcs_key])
    hdr = _parse_fits_header_string(wcs_header_str)
    wcs = WCS(hdr)
    if not wcs.has_celestial:
        raise ValueError("Euclid tile WCS header does not contain celestial axes.")
    return img, wcs


def _find_peaks(
    img: np.ndarray,
    nsigma: float = 10.0,
    border: int = 10,
    top: Optional[int] = 5000,
    local_box: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.nan_to_num(img, nan=-np.inf)
    ny, nx = data.shape
    b = max(4, int(local_box))

    py = (-ny) % b
    px = (-nx) % b
    data_p = np.pad(data, ((0, py), (0, px)), mode="edge") if (py or px) else data
    Ny, Nx = data_p.shape
    by, bx = Ny // b, Nx // b

    blocks = data_p.reshape(by, b, bx, b)
    m_blk = np.nanmedian(blocks, axis=(1, 3))
    mad_blk = np.nanmedian(np.abs(blocks - m_blk[:, None, :, None]), axis=(1, 3))
    s_blk = 1.4826 * mad_blk

    m = zoom(m_blk, (b, b), order=1)[:Ny, :Nx][:ny, :nx]
    s = zoom(s_blk, (b, b), order=1)[:Ny, :Nx][:ny, :nx]
    thr = m + float(nsigma) * s

    local_max = maximum_filter(data, size=3) == data
    mask = (data > thr) & local_max & np.isfinite(data)

    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    vals = data[ys, xs]
    order = np.argsort(vals)[::-1]
    if top is not None:
        order = order[: int(top)]
    return xs[order].astype(float), ys[order].astype(float)


def _parabolic_subpixel_1d(fm1: float, f0: float, fp1: float) -> float:
    denom = fm1 - 2.0 * f0 + fp1
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    dx = 0.5 * (fm1 - fp1) / denom
    if not np.isfinite(dx):
        return 0.0
    return float(np.clip(dx, -0.75, 0.75))


def _refine_centroids(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, r: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if len(xs) == 0:
        return xs, ys
    outx, outy = [], []
    ny, nx = img.shape
    data = np.nan_to_num(img, nan=-np.inf)

    for x0, y0 in zip(xs, ys):
        ix, iy = int(round(float(x0))), int(round(float(y0)))
        y1, y2 = max(0, iy - r), min(ny, iy + r + 1)
        x1, x2 = max(0, ix - r), min(nx, ix + r + 1)
        cut = data[y1:y2, x1:x2]
        if cut.size == 0 or not np.isfinite(cut).any():
            outx.append(float(x0))
            outy.append(float(y0))
            continue

        bkg = np.nanmedian(cut[np.isfinite(cut)])
        w = np.clip(cut - bkg, 0, None)
        if np.sum(w) <= 0:
            outx.append(float(x0))
            outy.append(float(y0))
            continue

        py, px = np.unravel_index(np.argmax(w), w.shape)
        if py <= 0 or py >= w.shape[0] - 1 or px <= 0 or px >= w.shape[1] - 1:
            outx.append(float(x1 + px))
            outy.append(float(y1 + py))
            continue

        f0 = float(w[py, px])
        dx = _parabolic_subpixel_1d(float(w[py, px - 1]), f0, float(w[py, px + 1]))
        dy = _parabolic_subpixel_1d(float(w[py - 1, px]), f0, float(w[py + 1, px]))
        outx.append(float(x1 + px + dx))
        outy.append(float(y1 + py + dy))

    return np.asarray(outx, dtype=float), np.asarray(outy, dtype=float)


def _detect_sources_to_sky(
    img: np.ndarray,
    wcs: WCS,
    nsigma: float,
    top: int,
    local_box: int,
    border: int,
    refine_r: int,
) -> Tuple[SkyCoord, np.ndarray, np.ndarray]:
    xs, ys = _find_peaks(img, nsigma=nsigma, border=border, top=top, local_box=local_box)
    if refine_r > 0:
        xs, ys = _refine_centroids(img, xs, ys, r=refine_r)
    if len(xs) == 0:
        empty = SkyCoord(ra=np.array([]) * u.deg, dec=np.array([]) * u.deg, frame="icrs")
        return empty, xs, ys
    sky = SkyCoord.from_pixel(xs, ys, wcs, origin=0)
    return sky, xs, ys


def _load_columnar_catalog(
    path: str,
    ra_col: str,
    dec_col: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    if p.suffix.lower() == ".npz":
        data = np.load(p, allow_pickle=False)
        keys = set(data.files)
        missing = [k for k in [ra_col, dec_col] if k not in keys]
        if missing:
            raise KeyError(f"Missing required columns in NPZ {path}: {missing}. Available: {sorted(keys)}")
        ra = np.asarray(data[ra_col], dtype=float).ravel()
        dec = np.asarray(data[dec_col], dtype=float).ravel()
        out = {"ra_deg": ra, "dec_deg": dec}
        if x_col and y_col:
            if x_col not in keys or y_col not in keys:
                raise KeyError(
                    f"Missing x/y columns in NPZ {path}: {(x_col, y_col)}. Available: {sorted(keys)}"
                )
            out["x"] = np.asarray(data[x_col], dtype=float).ravel()
            out["y"] = np.asarray(data[y_col], dtype=float).ravel()
        return _validate_catalog_arrays(out, path)

    tab = Table.read(path)
    cols = set(tab.colnames)
    missing = [k for k in [ra_col, dec_col] if k not in cols]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}. Available: {sorted(cols)}")

    out = {
        "ra_deg": np.asarray(tab[ra_col], dtype=float).ravel(),
        "dec_deg": np.asarray(tab[dec_col], dtype=float).ravel(),
    }
    if x_col and y_col:
        if x_col not in cols or y_col not in cols:
            raise KeyError(f"Missing x/y columns in {path}: {(x_col, y_col)}. Available: {sorted(cols)}")
        out["x"] = np.asarray(tab[x_col], dtype=float).ravel()
        out["y"] = np.asarray(tab[y_col], dtype=float).ravel()
    return _validate_catalog_arrays(out, path)


def _validate_catalog_arrays(cat: Dict[str, np.ndarray], path: str) -> Dict[str, np.ndarray]:
    n = len(cat["ra_deg"])
    if len(cat["dec_deg"]) != n:
        raise ValueError(f"RA/Dec length mismatch in {path}: {len(cat['ra_deg'])} vs {len(cat['dec_deg'])}")
    if "x" in cat and len(cat["x"]) != n:
        raise ValueError(f"RA/x length mismatch in {path}: {len(cat['ra_deg'])} vs {len(cat['x'])}")
    if "y" in cat and len(cat["y"]) != n:
        raise ValueError(f"RA/y length mismatch in {path}: {len(cat['ra_deg'])} vs {len(cat['y'])}")
    return cat


def _nn_greedy_unique_match(
    ref_sky: SkyCoord,
    cand_sky: SkyCoord,
    max_sep_arcsec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-to-one greedy matching within max_sep.
    Returns (ri, ci, sep_arcsec), where:
      ri indexes ref_sky
      ci indexes cand_sky
    """
    if len(ref_sky) == 0 or len(cand_sky) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    idx_ref, idx_cand, sep2d, _ = search_around_sky(ref_sky, cand_sky, max_sep_arcsec * u.arcsec)
    if len(idx_ref) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    sep = sep2d.to(u.arcsec).value
    order = np.argsort(sep)
    used_ref = np.zeros(len(ref_sky), dtype=bool)
    used_cand = np.zeros(len(cand_sky), dtype=bool)

    ri_out, ci_out, sep_out = [], [], []
    for k in order:
        r = int(idx_ref[k])
        c = int(idx_cand[k])
        if used_ref[r] or used_cand[c]:
            continue
        used_ref[r] = True
        used_cand[c] = True
        ri_out.append(r)
        ci_out.append(c)
        sep_out.append(float(sep[k]))

    return np.asarray(ri_out, dtype=int), np.asarray(ci_out, dtype=int), np.asarray(sep_out, dtype=float)


def _mad_sigma(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def _robust_keep_mask(dra_mas: np.ndarray, ddec_mas: np.ndarray, clip_sigma: float) -> np.ndarray:
    if len(dra_mas) == 0:
        return np.zeros(0, dtype=bool)
    med_ra = float(np.median(dra_mas))
    med_dec = float(np.median(ddec_mas))
    sig_ra = _mad_sigma(dra_mas)
    sig_dec = _mad_sigma(ddec_mas)
    if not np.isfinite(sig_ra) or sig_ra <= 0:
        sig_ra = float(np.std(dra_mas) + 1e-9)
    if not np.isfinite(sig_dec) or sig_dec <= 0:
        sig_dec = float(np.std(ddec_mas) + 1e-9)
    return (
        (np.abs(dra_mas - med_ra) < clip_sigma * sig_ra)
        & (np.abs(ddec_mas - med_dec) < clip_sigma * sig_dec)
    )


def _summarize_offsets(dra_mas: np.ndarray, ddec_mas: np.ndarray) -> Dict[str, float]:
    if len(dra_mas) == 0:
        return {
            "n": 0,
            "median_dra_mas": float("nan"),
            "median_ddec_mas": float("nan"),
            "median_offset_mas": float("nan"),
            "rms_offset_mas": float("nan"),
            "p68_offset_mas": float("nan"),
            "p95_offset_mas": float("nan"),
            "centered_rms_offset_mas": float("nan"),
            "centered_p68_offset_mas": float("nan"),
        }

    med_dra = float(np.median(dra_mas))
    med_ddec = float(np.median(ddec_mas))
    r = np.hypot(dra_mas, ddec_mas)
    rc = np.hypot(dra_mas - med_dra, ddec_mas - med_ddec)
    return {
        "n": int(len(dra_mas)),
        "median_dra_mas": med_dra,
        "median_ddec_mas": med_ddec,
        "median_offset_mas": float(np.hypot(med_dra, med_ddec)),
        "rms_offset_mas": float(np.sqrt(np.mean(r * r))),
        "p68_offset_mas": float(np.quantile(r, 0.68)),
        "p95_offset_mas": float(np.quantile(r, 0.95)),
        "centered_rms_offset_mas": float(np.sqrt(np.mean(rc * rc))),
        "centered_p68_offset_mas": float(np.quantile(rc, 0.68)),
    }


def _compute_offsets_mas(ref: SkyCoord, cand: SkyCoord) -> Tuple[np.ndarray, np.ndarray]:
    cosdec = np.cos(np.deg2rad(ref.dec.deg))
    dra_mas = (cand.ra.deg - ref.ra.deg) * 3600.0 * 1000.0 * cosdec
    ddec_mas = (cand.dec.deg - ref.dec.deg) * 3600.0 * 1000.0
    return np.asarray(dra_mas, dtype=float), np.asarray(ddec_mas, dtype=float)


def _evaluate_pair(
    ref_ra_deg: np.ndarray,
    ref_dec_deg: np.ndarray,
    cand_ra_deg: np.ndarray,
    cand_dec_deg: np.ndarray,
    max_sep_arcsec: float,
    clip_sigma: float,
) -> Dict[str, object]:
    ref_sky = SkyCoord(ra=ref_ra_deg * u.deg, dec=ref_dec_deg * u.deg, frame="icrs")
    cand_sky = SkyCoord(ra=cand_ra_deg * u.deg, dec=cand_dec_deg * u.deg, frame="icrs")
    ri, ci, sep_arcsec = _nn_greedy_unique_match(ref_sky, cand_sky, max_sep_arcsec=max_sep_arcsec)

    out: Dict[str, object] = {
        "n_ref": int(len(ref_sky)),
        "n_cand": int(len(cand_sky)),
        "n_matches": int(len(ri)),
        "max_sep_arcsec": float(max_sep_arcsec),
        "sep_median_arcsec": float(np.median(sep_arcsec)) if len(sep_arcsec) else float("nan"),
        "sep_p95_arcsec": float(np.quantile(sep_arcsec, 0.95)) if len(sep_arcsec) else float("nan"),
    }
    if len(ri) == 0:
        out["all_matches"] = _summarize_offsets(np.array([], dtype=float), np.array([], dtype=float))
        out["clipped_matches"] = _summarize_offsets(np.array([], dtype=float), np.array([], dtype=float))
        return out

    dra_mas, ddec_mas = _compute_offsets_mas(ref_sky[ri], cand_sky[ci])
    keep = _robust_keep_mask(dra_mas, ddec_mas, clip_sigma=clip_sigma)
    out["all_matches"] = _summarize_offsets(dra_mas, ddec_mas)
    out["clipped_matches"] = _summarize_offsets(dra_mas[keep], ddec_mas[keep])
    out["clip_sigma"] = float(clip_sigma)
    out["n_clipped"] = int(np.sum(keep))
    return out


def _evaluate_fixed_pairs(
    ref_ra_deg: np.ndarray,
    ref_dec_deg: np.ndarray,
    cand_before_ra_deg: np.ndarray,
    cand_before_dec_deg: np.ndarray,
    cand_after_ra_deg: np.ndarray,
    cand_after_dec_deg: np.ndarray,
    max_sep_arcsec: float,
    clip_sigma: float,
) -> Dict[str, object]:
    ref_sky = SkyCoord(ra=ref_ra_deg * u.deg, dec=ref_dec_deg * u.deg, frame="icrs")
    cand_before_sky = SkyCoord(ra=cand_before_ra_deg * u.deg, dec=cand_before_dec_deg * u.deg, frame="icrs")
    cand_after_sky = SkyCoord(ra=cand_after_ra_deg * u.deg, dec=cand_after_dec_deg * u.deg, frame="icrs")

    ri, ci, sep_arcsec = _nn_greedy_unique_match(ref_sky, cand_before_sky, max_sep_arcsec=max_sep_arcsec)
    out: Dict[str, object] = {
        "n_ref": int(len(ref_sky)),
        "n_cand": int(len(cand_before_sky)),
        "n_pairs": int(len(ri)),
        "max_sep_arcsec": float(max_sep_arcsec),
        "pair_sep_median_arcsec": float(np.median(sep_arcsec)) if len(sep_arcsec) else float("nan"),
        "pair_sep_p95_arcsec": float(np.quantile(sep_arcsec, 0.95)) if len(sep_arcsec) else float("nan"),
        "clip_sigma": float(clip_sigma),
    }

    if len(ri) == 0:
        empty = _summarize_offsets(np.array([], dtype=float), np.array([], dtype=float))
        out["before_all"] = empty
        out["before_clipped"] = empty
        out["after_all"] = empty
        out["after_clipped"] = empty
        out["n_clipped"] = 0
        out["delta_all"] = {
            "median_offset_mas": float("nan"),
            "centered_p68_offset_mas": float("nan"),
            "rms_offset_mas": float("nan"),
        }
        out["delta_clipped"] = {
            "median_offset_mas": float("nan"),
            "centered_p68_offset_mas": float("nan"),
            "rms_offset_mas": float("nan"),
        }
        return out

    dra_before_mas, ddec_before_mas = _compute_offsets_mas(ref_sky[ri], cand_before_sky[ci])
    dra_after_mas, ddec_after_mas = _compute_offsets_mas(ref_sky[ri], cand_after_sky[ci])
    keep = _robust_keep_mask(dra_before_mas, ddec_before_mas, clip_sigma=clip_sigma)

    before_all = _summarize_offsets(dra_before_mas, ddec_before_mas)
    before_clipped = _summarize_offsets(dra_before_mas[keep], ddec_before_mas[keep])
    after_all = _summarize_offsets(dra_after_mas, ddec_after_mas)
    after_clipped = _summarize_offsets(dra_after_mas[keep], ddec_after_mas[keep])

    out["before_all"] = before_all
    out["before_clipped"] = before_clipped
    out["after_all"] = after_all
    out["after_clipped"] = after_clipped
    out["n_clipped"] = int(np.sum(keep))
    out["delta_all"] = {
        "median_offset_mas": float(after_all["median_offset_mas"] - before_all["median_offset_mas"]),
        "centered_p68_offset_mas": float(after_all["centered_p68_offset_mas"] - before_all["centered_p68_offset_mas"]),
        "rms_offset_mas": float(after_all["rms_offset_mas"] - before_all["rms_offset_mas"]),
    }
    out["delta_clipped"] = {
        "median_offset_mas": float(after_clipped["median_offset_mas"] - before_clipped["median_offset_mas"]),
        "centered_p68_offset_mas": float(
            after_clipped["centered_p68_offset_mas"] - before_clipped["centered_p68_offset_mas"]
        ),
        "rms_offset_mas": float(after_clipped["rms_offset_mas"] - before_clipped["rms_offset_mas"]),
    }
    return out


def _resolve_extnames(
    hdul: fits.HDUList,
    dra_ext: Optional[str],
    dde_ext: Optional[str],
    tile_id: Optional[str],
    band_key: Optional[str],
) -> Tuple[str, str]:
    if dra_ext and dde_ext:
        return dra_ext, dde_ext
    if not tile_id or not band_key:
        raise ValueError("Provide either --dra-ext/--dde-ext or --tile-id/--band-key.")
    band = band_key.lower().replace("rubin_", "")
    return f"{tile_id}.{band}.DRA", f"{tile_id}.{band}.DDE"


def _get_hdu_data_by_name(hdul: fits.HDUList, extname: str) -> Tuple[np.ndarray, fits.Header]:
    wanted = extname.upper()
    for hdu in hdul:
        if hdu.name.upper() == wanted:
            if hdu.data is None:
                raise ValueError(f"HDU '{extname}' has no data.")
            return np.asarray(hdu.data, dtype=float), hdu.header
    names = [h.name for h in hdul]
    raise KeyError(f"Extension '{extname}' not found. Available: {names}")


def _bilinear_sample(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = data.shape
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= (w - 1)) & (y <= (h - 1))
    if not np.any(valid):
        return out

    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(int)
    y0 = np.floor(yv).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = xv - x0
    wy = yv - y0

    f00 = data[y0, x0]
    f01 = data[y0, x1]
    f10 = data[y1, x0]
    f11 = data[y1, x1]

    outv = (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f01
        + (1.0 - wx) * wy * f10
        + wx * wy * f11
    )
    out[valid] = outv
    return out


def _map_xy_from_vis_xy(
    x_vis: np.ndarray,
    y_vis: np.ndarray,
    map_w: int,
    map_h: int,
    full_w: int,
    full_h: int,
    xy_origin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x_vis, dtype=float) - float(xy_origin)
    y0 = np.asarray(y_vis, dtype=float) - float(xy_origin)
    # align_corners=False style mapping between pixel-center grids
    x_map = ((x0 + 0.5) / float(full_w)) * float(map_w) - 0.5
    y_map = ((y0 + 0.5) / float(full_h)) * float(map_h) - 0.5
    return x_map, y_map


def _apply_concordance_to_catalog(
    cand_ra_deg: np.ndarray,
    cand_dec_deg: np.ndarray,
    dra_arcsec: np.ndarray,
    ddec_arcsec: np.ndarray,
    apply_sign: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dec_rad = np.deg2rad(cand_dec_deg)
    cosdec = np.cos(dec_rad)
    eps = 1e-12
    safe = np.where(np.abs(cosdec) > eps, cosdec, np.sign(cosdec) * eps + (cosdec == 0) * eps)

    d_ra_deg = apply_sign * (dra_arcsec / (3600.0 * safe))
    d_dec_deg = apply_sign * (ddec_arcsec / 3600.0)
    return cand_ra_deg + d_ra_deg, cand_dec_deg + d_dec_deg


def _print_summary(tag: str, result: Dict[str, object]) -> None:
    allm = result["all_matches"]
    clipm = result["clipped_matches"]
    print(f"\n[{tag}]")
    print(
        f"  matches: {result['n_matches']} / ref={result['n_ref']} cand={result['n_cand']} "
        f"(sep_med={result['sep_median_arcsec']:.4f}\" sep_p95={result['sep_p95_arcsec']:.4f}\")"
    )
    print(
        "  all     : "
        f"median_offset={allm['median_offset_mas']:.2f} mas "
        f"rms={allm['rms_offset_mas']:.2f} mas "
        f"centered_p68={allm['centered_p68_offset_mas']:.2f} mas "
        f"(n={allm['n']})"
    )
    print(
        "  clipped : "
        f"median_offset={clipm['median_offset_mas']:.2f} mas "
        f"rms={clipm['rms_offset_mas']:.2f} mas "
        f"centered_p68={clipm['centered_p68_offset_mas']:.2f} mas "
        f"(n={clipm['n']})"
    )


def _print_fixed_pair_summary(tag: str, result: Dict[str, object]) -> None:
    b_all = result["before_all"]
    a_all = result["after_all"]
    b_clip = result["before_clipped"]
    a_clip = result["after_clipped"]
    d_all = result["delta_all"]
    d_clip = result["delta_clipped"]

    print(f"\n[{tag}]")
    print(
        f"  fixed pairs: {result['n_pairs']} / ref={result['n_ref']} cand={result['n_cand']} "
        f"(pair_sep_med={result['pair_sep_median_arcsec']:.4f}\" pair_sep_p95={result['pair_sep_p95_arcsec']:.4f}\")"
    )
    print(
        "  all     : "
        f"median_offset {b_all['median_offset_mas']:.2f} -> {a_all['median_offset_mas']:.2f} mas "
        f"(Δ={d_all['median_offset_mas']:+.2f}) | "
        f"centered_p68 {b_all['centered_p68_offset_mas']:.2f} -> {a_all['centered_p68_offset_mas']:.2f} mas "
        f"(Δ={d_all['centered_p68_offset_mas']:+.2f})"
    )
    print(
        "  clipped : "
        f"median_offset {b_clip['median_offset_mas']:.2f} -> {a_clip['median_offset_mas']:.2f} mas "
        f"(Δ={d_clip['median_offset_mas']:+.2f}) | "
        f"centered_p68 {b_clip['centered_p68_offset_mas']:.2f} -> {a_clip['centered_p68_offset_mas']:.2f} mas "
        f"(Δ={d_clip['centered_p68_offset_mas']:+.2f}) "
        f"(n={a_clip['n']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matched-source astrometry before/after concordance correction.")
    parser.add_argument("--ref-catalog", type=str, default="", help="Reference catalog path (e.g. Euclid VIS).")
    parser.add_argument("--cand-catalog", type=str, default="", help="Candidate catalog path (e.g. Rubin band).")
    parser.add_argument("--ref-ra-col", type=str, default="ra")
    parser.add_argument("--ref-dec-col", type=str, default="dec")
    parser.add_argument("--cand-ra-col", type=str, default="ra")
    parser.add_argument("--cand-dec-col", type=str, default="dec")
    parser.add_argument("--cand-x-col", type=str, default="", help="Optional candidate x-column for map sampling.")
    parser.add_argument("--cand-y-col", type=str, default="", help="Optional candidate y-column for map sampling.")
    parser.add_argument("--max-sep-arcsec", type=float, default=0.1, help="Matching radius in arcseconds.")
    parser.add_argument("--clip-sigma", type=float, default=3.5, help="MAD clipping sigma threshold.")

    parser.add_argument(
        "--auto-from-tiles",
        action="store_true",
        help="Build catalogs automatically from Rubin/Euclid tile images via peak detection.",
    )
    parser.add_argument("--rubin-tile", type=str, default="", help="Rubin tile NPZ path for auto mode.")
    parser.add_argument("--euclid-tile", type=str, default="", help="Euclid tile NPZ path for auto mode.")
    parser.add_argument("--rubin-band", type=str, default="r", help="Rubin band key for auto mode (u/g/r/i/z/y).")
    parser.add_argument("--euclid-band", type=str, default="VIS", help="Euclid band key for auto mode (VIS/Y/J/H).")
    parser.add_argument("--detect-nsigma", type=float, default=10.0)
    parser.add_argument("--detect-top", type=int, default=5000)
    parser.add_argument("--detect-local-box", type=int, default=128)
    parser.add_argument("--detect-border", type=int, default=10)
    parser.add_argument("--detect-refine-r", type=int, default=5)

    parser.add_argument("--concordance-fits", type=str, default="", help="Optional concordance FITS path.")
    parser.add_argument("--dra-ext", type=str, default="", help="DRA extension name.")
    parser.add_argument("--dde-ext", type=str, default="", help="DDE extension name.")
    parser.add_argument("--tile-id", type=str, default="", help="Tile ID to build ext names if dra/dde not given.")
    parser.add_argument("--band-key", type=str, default="", help="Band key (e.g. r, i) to build ext names.")
    parser.add_argument("--xy-space", type=str, default="vis", choices=["vis", "mesh"])
    parser.add_argument("--xy-origin", type=int, default=0, choices=[0, 1], help="Column origin convention for x/y.")
    parser.add_argument("--dstep", type=int, default=0, help="Override DSTEP if needed (0 => read from header or 8).")
    parser.add_argument("--vis-width", type=int, default=0, help="VIS image width in pixels (for xy-space=vis).")
    parser.add_argument("--vis-height", type=int, default=0, help="VIS image height in pixels (for xy-space=vis).")
    parser.add_argument(
        "--apply-sign",
        type=float,
        default=1.0,
        help="Sign for applying concordance: +1 means corrected = candidate + delta (default).",
    )

    parser.add_argument("--output-json", type=str, default="", help="Optional output JSON path.")
    args = parser.parse_args()

    ref: Dict[str, np.ndarray]
    cand: Dict[str, np.ndarray]

    if args.auto_from_tiles:
        if not args.rubin_tile or not args.euclid_tile:
            raise ValueError("Auto mode requires --rubin-tile and --euclid-tile.")
        rubin_img, rubin_wcs = _load_rubin_tile_for_autocat(args.rubin_tile, args.rubin_band)
        eu_img, eu_wcs = _load_euclid_tile_for_autocat(args.euclid_tile, args.euclid_band)

        ref_sky, _, _ = _detect_sources_to_sky(
            img=eu_img,
            wcs=eu_wcs,
            nsigma=args.detect_nsigma,
            top=args.detect_top,
            local_box=args.detect_local_box,
            border=args.detect_border,
            refine_r=args.detect_refine_r,
        )
        cand_sky, _, _ = _detect_sources_to_sky(
            img=rubin_img,
            wcs=rubin_wcs,
            nsigma=args.detect_nsigma,
            top=args.detect_top,
            local_box=args.detect_local_box,
            border=args.detect_border,
            refine_r=args.detect_refine_r,
        )
        cand_x_vis, cand_y_vis = eu_wcs.world_to_pixel(cand_sky)

        ref = {"ra_deg": np.asarray(ref_sky.ra.deg, dtype=float), "dec_deg": np.asarray(ref_sky.dec.deg, dtype=float)}
        cand = {
            "ra_deg": np.asarray(cand_sky.ra.deg, dtype=float),
            "dec_deg": np.asarray(cand_sky.dec.deg, dtype=float),
            "x": np.asarray(cand_x_vis, dtype=float),
            "y": np.asarray(cand_y_vis, dtype=float),
        }
    else:
        if not args.ref_catalog or not args.cand_catalog:
            raise ValueError("Provide --ref-catalog and --cand-catalog, or use --auto-from-tiles.")
        ref = _load_columnar_catalog(
            args.ref_catalog,
            ra_col=args.ref_ra_col,
            dec_col=args.ref_dec_col,
            x_col=None,
            y_col=None,
        )
        cand = _load_columnar_catalog(
            args.cand_catalog,
            ra_col=args.cand_ra_col,
            dec_col=args.cand_dec_col,
            x_col=args.cand_x_col or None,
            y_col=args.cand_y_col or None,
        )

    results: Dict[str, object] = {
        "config": {
            "ref_catalog": str(args.ref_catalog),
            "cand_catalog": str(args.cand_catalog),
            "max_sep_arcsec": float(args.max_sep_arcsec),
            "clip_sigma": float(args.clip_sigma),
            "auto_from_tiles": bool(args.auto_from_tiles),
        }
    }
    if args.auto_from_tiles:
        results["config"]["rubin_tile"] = str(args.rubin_tile)
        results["config"]["euclid_tile"] = str(args.euclid_tile)
        results["config"]["rubin_band"] = str(args.rubin_band)
        results["config"]["euclid_band"] = str(args.euclid_band)
        results["config"]["detect"] = {
            "nsigma": float(args.detect_nsigma),
            "top": int(args.detect_top),
            "local_box": int(args.detect_local_box),
            "border": int(args.detect_border),
            "refine_r": int(args.detect_refine_r),
        }
        print(
            f"\n[auto_from_tiles] ref_sources={len(ref['ra_deg'])} "
            f"cand_sources={len(cand['ra_deg'])}"
        )

    before_all = _evaluate_pair(
        ref_ra_deg=ref["ra_deg"],
        ref_dec_deg=ref["dec_deg"],
        cand_ra_deg=cand["ra_deg"],
        cand_dec_deg=cand["dec_deg"],
        max_sep_arcsec=args.max_sep_arcsec,
        clip_sigma=args.clip_sigma,
    )
    results["before_all_candidates"] = before_all
    _print_summary("before_all_candidates", before_all)

    if args.concordance_fits:
        with fits.open(args.concordance_fits) as hdul:
            tile_id_auto = ""
            band_key_auto = ""
            if args.auto_from_tiles:
                tile_id_auto = Path(args.rubin_tile).stem
                band_key_auto = args.rubin_band
            dra_name, dde_name = _resolve_extnames(
                hdul=hdul,
                dra_ext=args.dra_ext or None,
                dde_ext=args.dde_ext or None,
                tile_id=(args.tile_id or tile_id_auto) or None,
                band_key=(args.band_key or band_key_auto) or None,
            )
            dra_map, dra_hdr = _get_hdu_data_by_name(hdul, dra_name)
            dde_map, _ = _get_hdu_data_by_name(hdul, dde_name)

        map_h, map_w = dra_map.shape
        if dde_map.shape != dra_map.shape:
            raise ValueError(f"DRA/DDE shape mismatch: {dra_map.shape} vs {dde_map.shape}")

        if "x" in cand and "y" in cand:
            if args.xy_space == "vis":
                dstep_hdr = int(dra_hdr.get("DSTEP", 8))
                dstep = int(args.dstep) if args.dstep > 0 else dstep_hdr
                full_w = int(args.vis_width) if args.vis_width > 0 else int(map_w * dstep)
                full_h = int(args.vis_height) if args.vis_height > 0 else int(map_h * dstep)
                x_map, y_map = _map_xy_from_vis_xy(
                    x_vis=cand["x"],
                    y_vis=cand["y"],
                    map_w=map_w,
                    map_h=map_h,
                    full_w=full_w,
                    full_h=full_h,
                    xy_origin=args.xy_origin,
                )
            else:
                x_map = np.asarray(cand["x"], dtype=float) - float(args.xy_origin)
                y_map = np.asarray(cand["y"], dtype=float) - float(args.xy_origin)
        else:
            wcs = WCS(dra_hdr)
            if not wcs.has_celestial:
                raise ValueError(
                    "No candidate x/y columns provided and FITS extension lacks celestial WCS. "
                    "Provide --cand-x-col/--cand-y-col with --xy-space."
                )
            cand_sky = SkyCoord(ra=cand["ra_deg"] * u.deg, dec=cand["dec_deg"] * u.deg, frame="icrs")
            x_map, y_map = wcs.world_to_pixel(cand_sky)

        dra_arcsec = _bilinear_sample(dra_map, x_map, y_map)
        ddec_arcsec = _bilinear_sample(dde_map, x_map, y_map)
        valid_corr = np.isfinite(dra_arcsec) & np.isfinite(ddec_arcsec)
        corr_mag = np.hypot(dra_arcsec[valid_corr], ddec_arcsec[valid_corr])

        if corr_mag.size:
            corr_stats = {
                "median_arcsec": float(np.median(corr_mag)),
                "p68_arcsec": float(np.quantile(corr_mag, 0.68)),
                "p95_arcsec": float(np.quantile(corr_mag, 0.95)),
                "max_arcsec": float(np.max(corr_mag)),
            }
        else:
            corr_stats = {
                "median_arcsec": float("nan"),
                "p68_arcsec": float("nan"),
                "p95_arcsec": float("nan"),
                "max_arcsec": float("nan"),
            }

        print(
            "\n[correction_field] "
            f"|delta| median={corr_stats['median_arcsec']*1000:.1f} mas "
            f"p68={corr_stats['p68_arcsec']*1000:.1f} mas "
            f"p95={corr_stats['p95_arcsec']*1000:.1f} mas "
            f"max={corr_stats['max_arcsec']*1000:.1f} mas"
        )

        before_correctable = _evaluate_pair(
            ref_ra_deg=ref["ra_deg"],
            ref_dec_deg=ref["dec_deg"],
            cand_ra_deg=cand["ra_deg"][valid_corr],
            cand_dec_deg=cand["dec_deg"][valid_corr],
            max_sep_arcsec=args.max_sep_arcsec,
            clip_sigma=args.clip_sigma,
        )
        results["before_correctable_subset"] = before_correctable
        _print_summary("before_correctable_subset", before_correctable)

        ra_corr, dec_corr = _apply_concordance_to_catalog(
            cand_ra_deg=cand["ra_deg"][valid_corr],
            cand_dec_deg=cand["dec_deg"][valid_corr],
            dra_arcsec=dra_arcsec[valid_corr],
            ddec_arcsec=ddec_arcsec[valid_corr],
            apply_sign=float(args.apply_sign),
        )
        after_corrected = _evaluate_pair(
            ref_ra_deg=ref["ra_deg"],
            ref_dec_deg=ref["dec_deg"],
            cand_ra_deg=ra_corr,
            cand_dec_deg=dec_corr,
            max_sep_arcsec=args.max_sep_arcsec,
            clip_sigma=args.clip_sigma,
        )
        fixed_pairs = _evaluate_fixed_pairs(
            ref_ra_deg=ref["ra_deg"],
            ref_dec_deg=ref["dec_deg"],
            cand_before_ra_deg=cand["ra_deg"][valid_corr],
            cand_before_dec_deg=cand["dec_deg"][valid_corr],
            cand_after_ra_deg=ra_corr,
            cand_after_dec_deg=dec_corr,
            max_sep_arcsec=args.max_sep_arcsec,
            clip_sigma=args.clip_sigma,
        )
        results["after_correction"] = after_corrected
        results["fixed_pairs"] = fixed_pairs
        results["correction_sampling"] = {
            "concordance_fits": str(args.concordance_fits),
            "dra_ext": dra_name,
            "dde_ext": dde_name,
            "n_candidates_total": int(len(cand["ra_deg"])),
            "n_candidates_correctable": int(np.sum(valid_corr)),
            "sampled_correction_abs_arcsec": corr_stats,
            "apply_sign": float(args.apply_sign),
            "xy_space": str(args.xy_space),
            "xy_origin": int(args.xy_origin),
        }
        _print_summary("after_correction", after_corrected)
        _print_fixed_pair_summary("fixed_pairs_after_correction", fixed_pairs)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
