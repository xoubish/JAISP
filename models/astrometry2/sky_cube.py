"""Extract an aligned multi-band sky cube at a given RA/Dec.

All 10 bands (Rubin u/g/r/i/z/y + Euclid VIS/Y/J/H) are resampled onto
a common pixel grid at VIS resolution (0.1 arcsec/px), with the concordance
astrometric correction applied per Rubin band.

Output is a [10, H, W] float32 array where every slice is on the same sky
coordinate grid — ready for forced photometry, SED fitting, deblending, or
as input to the JAISP foundation model.

Usage
-----
    from sky_cube import SkyCubeExtractor

    extractor = SkyCubeExtractor(
        rubin_dir='data/rubin_tiles_ecdfs',
        euclid_dir='data/euclid_tiles_ecdfs',
        concordance_path='checkpoints/astrometry_v6_phaseB2/concordance_r.fits',
    )

    result = extractor.extract(ra=53.1234, dec=-27.8765, size_arcsec=10.0)

    cube      = result['cube']       # [10, 100, 100] float32
    rms_cube  = result['rms_cube']   # [10, 100, 100] float32 — per-pixel noise
    bands     = result['bands']      # ['rubin_u', ..., 'euclid_VIS', ...]
    coverage  = result['coverage']   # [100, 100] — concordance coverage (VIS px to nearest anchor)
    wcs       = result['patch_wcs']  # astropy WCS for the output patch
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

# Make sure sibling modules are importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR  = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _MODELS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apply_concordance import ConcordanceMap
from source_matching import safe_header_from_card_string

VIS_PIXEL_SCALE  = 0.1           # arcsec / VIS pixel
RUBIN_BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']
EUCLID_BANDS     = ['VIS', 'Y', 'J', 'H']
ALL_BANDS        = [f'rubin_{b}' for b in RUBIN_BAND_ORDER] + [f'euclid_{b}' for b in EUCLID_BANDS]


def _to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_rms(var: np.ndarray, img: np.ndarray) -> np.ndarray:
    """sqrt(variance), replacing non-positive/nan pixels with MAD estimate."""
    rms = np.zeros_like(var)
    good = np.isfinite(var) & (var > 0)
    rms[good] = np.sqrt(var[good])
    bad = ~good
    if bad.any():
        finite = img[np.isfinite(img)]
        mad = np.median(np.abs(finite - np.median(finite))) if len(finite) > 10 else 1.0
        rms[bad] = max(1.4826 * mad, 1e-10)
    return rms


def _sample_image(
    image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    out_shape: Tuple[int, int],
    order: int = 3,
) -> np.ndarray:
    """
    Resample image at fractional pixel coords (xs, ys), reshape to out_shape.

    order=1  bilinear  — fast, adequate for flux/alignment checks
    order=3  bicubic   — better preservation of PSF shape (default)

    Note: this is geometric resampling only — no new information is created.
    For scientifically correct Rubin-at-VIS-resolution the v6 foundation model
    super-resolution decoder should be used instead (run the decoder conditioned
    on euclid_VIS with Rubin context bands).
    """
    coords = np.stack([ys, xs], axis=0)   # map_coordinates expects [row, col]
    sampled = map_coordinates(
        image.astype(np.float64),
        coords,
        order=order,
        mode='constant',
        cval=np.nan,
    )
    return sampled.reshape(out_shape).astype(np.float32)


def _build_patch_wcs(
    ra_center: float,
    dec_center: float,
    size_px: int,
    pixel_scale_deg: float = VIS_PIXEL_SCALE / 3600.0,
) -> WCS:
    """Build a simple tangent-projection WCS for the output patch."""
    w = WCS(naxis=2)
    w.wcs.crpix = [(size_px + 1) / 2.0, (size_px + 1) / 2.0]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]   # RA increases right→left
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w


class SkyCubeExtractor:
    """
    Extracts aligned 10-band sky cubes at any RA/Dec covered by the survey.

    For each requested position:
      - Finds the tile covering that position.
      - Builds an output pixel grid at VIS resolution centred on the position.
      - Samples Euclid VIS/NISP directly using their WCS.
      - Samples each Rubin band using the inverse concordance correction so that
        each Rubin slice is aligned to the VIS reference frame.

    Parameters
    ----------
    rubin_dir         : directory containing tile_x*_y*.npz Rubin files
    euclid_dir        : directory containing tile_x*_y*_euclid.npz files
    concordance_path  : path to concordance FITS from infer_concordance.py
                        If None, Rubin bands are projected with WCS only (no correction).
    """

    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        concordance_path: Optional[str] = None,
    ):
        self.rubin_dir  = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)

        # Load concordance
        self.cmap: Optional[ConcordanceMap] = None
        if concordance_path:
            self.cmap = ConcordanceMap(concordance_path)

        # Index tiles: tile_id → {ra_center, dec_center, rubin_path, euclid_path}
        self._tile_index: List[dict] = []
        self._build_tile_index()

    # ------------------------------------------------------------------
    # Tile indexing
    # ------------------------------------------------------------------

    def _build_tile_index(self) -> None:
        rubin_files = sorted(self.rubin_dir.glob('tile_x*_y*.npz'))
        for rp in rubin_files:
            tile_id = rp.stem
            ep = self.euclid_dir / f'{tile_id}_euclid.npz'
            if not ep.exists():
                continue
            try:
                d = np.load(rp, allow_pickle=True, mmap_mode='r')
                self._tile_index.append({
                    'tile_id':    tile_id,
                    'rubin_path': rp,
                    'euclid_path': ep,
                    'ra_center':  float(d['ra_center']),
                    'dec_center': float(d['dec_center']),
                })
            except Exception:
                continue
        print(f'SkyCubeExtractor: indexed {len(self._tile_index)} tiles')

    def _find_tile(self, ra: float, dec: float) -> Optional[dict]:
        """Return the tile covering (ra, dec), choosing the one whose centre is closest."""
        if not self._tile_index:
            return None
        cos_dec = np.cos(np.deg2rad(dec))
        dists = [
            ((t['ra_center'] - ra) * cos_dec) ** 2 + (t['dec_center'] - dec) ** 2
            for t in self._tile_index
        ]
        # Sort by distance, return the first tile that actually contains the position
        for idx in np.argsort(dists):
            tile = self._tile_index[idx]
            try:
                rdata = np.load(tile['rubin_path'], allow_pickle=True, mmap_mode='r')
                rwcs  = WCS(rdata['wcs_hdr'].item())
                x, y  = rwcs.wcs_world2pix(ra, dec, 0)
                h, w  = int(rdata['img'].shape[1]), int(rdata['img'].shape[2])
                if 0 <= float(x) < w and 0 <= float(y) < h:
                    return tile
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        ra: float,
        dec: float,
        size_arcsec: float = 10.0,
        tile_id: Optional[str] = None,
    ) -> dict:
        """
        Extract an aligned [10, H, W] cube centred on (ra, dec).

        Parameters
        ----------
        ra, dec      : sky position in degrees (ICRS)
        size_arcsec  : side length of the output patch in arcsec
        tile_id      : override tile selection (useful if you already know the tile)

        Returns
        -------
        dict with keys:
          cube       [10, H, W] float32 — all bands, aligned to VIS grid
          rms_cube   [10, H, W] float32 — per-pixel noise estimate
          bands      list of 10 band names
          coverage   [H, W]    float32 — concordance coverage map (VIS px to nearest anchor)
                                         NaN where no concordance available
          patch_wcs  astropy WCS for the output patch (tangent projection at VIS scale)
          tile_id    str
          ra, dec    float (requested centre)
        """
        size_px = max(1, int(round(size_arcsec / VIS_PIXEL_SCALE)))

        # Find tile
        if tile_id is None:
            tile = self._find_tile(ra, dec)
        else:
            tile = next((t for t in self._tile_index if t['tile_id'] == tile_id), None)
        if tile is None:
            raise ValueError(f'No tile found covering RA={ra:.4f} Dec={dec:.4f}')

        tile_id = tile['tile_id']

        # Load raw data
        rdata = np.load(tile['rubin_path'],  allow_pickle=True)
        edata = np.load(tile['euclid_path'], allow_pickle=True)

        rwcs = WCS(rdata['wcs_hdr'].item())
        vwcs = WCS(safe_header_from_card_string(edata['wcs_VIS'].item()))

        # Build output patch WCS at VIS resolution, centred on (ra, dec)
        patch_wcs = _build_patch_wcs(ra, dec, size_px)

        # Pixel grid for the output patch
        px_idx = np.arange(size_px, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(px_idx, px_idx)   # [H, W]
        out_shape = (size_px, size_px)

        # Sky coords for each output pixel
        ra_grid, dec_grid = patch_wcs.wcs_pix2world(
            grid_x.ravel(), grid_y.ravel(), 0
        )

        cube     = np.full((10, size_px, size_px), np.nan, dtype=np.float32)
        rms_cube = np.full((10, size_px, size_px), np.nan, dtype=np.float32)
        coverage = np.full((size_px, size_px), np.nan, dtype=np.float32)

        # ---- Rubin bands (with concordance correction) -------------------
        rubin_img = _to_float32(rdata['img'])   # [6, H, W]
        rubin_var = _to_float32(rdata['var']) if 'var' in rdata else np.ones_like(rubin_img)

        # Pre-compute VIS pixel coords for the output grid (shared by all bands)
        vis_x_raw, vis_y_raw = vwcs.wcs_world2pix(ra_grid, dec_grid, 0)
        vis_xy = np.stack(
            [vis_x_raw.astype(np.float32), vis_y_raw.astype(np.float32)], axis=1
        )

        # Coverage (same spatial field for all Rubin bands — grab once from r-band)
        if self.cmap is not None:
            for ck in (list(self.cmap.tiles.get(tile_id.lower(), {}).keys()) or ['r']):
                cov_vals = self.cmap.coverage(tile_id, ck, vis_xy)
                if cov_vals is not None:
                    coverage = cov_vals.reshape(out_shape)
                    break

        for bi, band_key in enumerate(RUBIN_BAND_ORDER):
            # Uncorrected Rubin pixels for the output sky grid
            rx_raw, ry_raw = rwcs.wcs_world2pix(ra_grid, dec_grid, 0)

            if self.cmap is not None:
                # --- Inverse concordance correction ---
                # The concordance says: rubin at (rx,ry) maps to VIS at
                # (rx,ry) + (dra,ddec). To find the Rubin pixel that maps to
                # our target sky coord, subtract the correction (one iteration).
                try:
                    # Try per-band concordance, fall back to first available band
                    avail = list(self.cmap.tiles.get(tile_id.lower(), {}).keys())
                    fallback = avail[0] if avail else 'r'
                    try:
                        dra, ddec = self.cmap._correction_at_vis_xy(tile_id, band_key, vis_xy)
                    except KeyError:
                        dra, ddec = self.cmap._correction_at_vis_xy(tile_id, fallback, vis_xy)

                    # Apply inverse: sky - correction → Rubin pixel
                    cos_dec = np.cos(np.deg2rad(dec_grid))
                    ra_inv  = ra_grid  - (dra  / 3600.0) / cos_dec
                    dec_inv = dec_grid - (ddec / 3600.0)
                    rx, ry  = rwcs.wcs_world2pix(ra_inv, dec_inv, 0)
                except Exception:
                    rx, ry = rx_raw, ry_raw  # fall back to uncorrected
            else:
                rx, ry = rx_raw, ry_raw

            img_band = _to_float32(rubin_img[bi])
            var_band = _to_float32(rubin_var[bi])
            rms_band = _safe_rms(var_band, img_band)

            cube[bi]     = _sample_image(img_band, rx, ry, out_shape)
            rms_cube[bi] = _sample_image(rms_band, rx, ry, out_shape)

        # ---- Euclid VIS (reference — no correction) ----------------------
        vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
        vis_var = _to_float32(edata['var_VIS']) if 'var_VIS' in edata else np.ones_like(vis_img)
        vis_x, vis_y = vwcs.wcs_world2pix(ra_grid, dec_grid, 0)
        cube[6]     = _sample_image(vis_img, vis_x, vis_y, out_shape)
        rms_cube[6] = _sample_image(_safe_rms(vis_var, vis_img), vis_x, vis_y, out_shape)

        # ---- Euclid NISP bands (Y, J, H) — WCS reprojection to VIS grid --
        for ni, nb in enumerate(EUCLID_BANDS[1:]):   # Y, J, H
            band_idx = 7 + ni
            wcs_key  = f'wcs_{nb}'
            img_key  = f'img_{nb}'
            var_key  = f'var_{nb}'
            if img_key not in edata or wcs_key not in edata:
                continue
            try:
                nisp_wcs = WCS(safe_header_from_card_string(edata[wcs_key].item()))
                nisp_img = np.nan_to_num(_to_float32(edata[img_key]), nan=0.0)
                nisp_var = _to_float32(edata[var_key]) if var_key in edata else np.ones_like(nisp_img)
                nx, ny = nisp_wcs.wcs_world2pix(ra_grid, dec_grid, 0)
                cube[band_idx]     = _sample_image(nisp_img, nx, ny, out_shape)
                rms_cube[band_idx] = _sample_image(_safe_rms(nisp_var, nisp_img), nx, ny, out_shape)
            except Exception:
                continue

        return {
            'cube':      cube,
            'rms_cube':  rms_cube,
            'bands':     ALL_BANDS,
            'coverage':  coverage,
            'patch_wcs': patch_wcs,
            'tile_id':   tile_id,
            'ra':        ra,
            'dec':       dec,
            'size_px':   size_px,
        }
