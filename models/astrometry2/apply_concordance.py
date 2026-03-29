"""Apply the astrometric concordance correction to align Rubin → Euclid VIS.

The concordance FITS file produced by infer_concordance.py contains a smooth
offset field (DRA*, DDec) in arcsec sampled on a coarse mesh over each VIS tile.
This module bilinearly interpolates that field at any sky position and applies
the correction so that a Rubin coordinate maps correctly onto the VIS pixel grid.

Typical usage
-------------
    from apply_concordance import ConcordanceMap

    cmap = ConcordanceMap('concordance_r.fits')

    # Project a single Rubin source onto VIS
    vis_x, vis_y = cmap.rubin_to_vis(
        rubin_x=247.3, rubin_y=183.1,
        rubin_wcs=rwcs,
        vis_wcs=vwcs,
        tile_id='tile_x01024_y00000',
        band='r',
    )

    # Project a full Rubin image pixel grid onto VIS (for resampling)
    H, W = rubin_img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    vis_coords = cmap.rubin_grid_to_vis(
        rubin_xs=xx.ravel(), rubin_ys=yy.ravel(),
        rubin_wcs=rwcs, vis_wcs=vwcs,
        tile_id='tile_x01024_y00000', band='r',
    )
    # vis_coords shape: [N, 2] columns = (vis_x, vis_y)
    # Feed directly into scipy.ndimage.map_coordinates or cv2.remap.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from typing import Dict, Optional, Tuple


class ConcordanceMap:
    """
    Loads a concordance FITS file and provides methods to apply corrections.

    Parameters
    ----------
    fits_path : str
        Path to the concordance FITS file produced by infer_concordance.py.

    Attributes
    ----------
    tiles : dict
        Nested dict: tiles[tile_id][band] = {'dra': ndarray, 'dde': ndarray,
        'cov': ndarray|None, 'dstep': int, 'vis_wcs': WCS}
        dra/dde are the offset meshes in arcsec at DSTEP spacing.
        cov is min-distance-to-nearest-anchor in VIS pixels (None if absent).
    """

    def __init__(self, fits_path: str):
        self.fits_path = fits_path
        self.tiles: Dict[str, Dict[str, dict]] = {}
        self._load(fits_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, fits_path: str) -> None:
        with fits.open(fits_path) as hdul:
            for hdu in hdul[1:]:   # skip PrimaryHDU
                name = hdu.name    # e.g. 'TILE_X01024_Y00000.R.DRA'
                if not name:
                    continue
                parts = name.split('.')
                if len(parts) < 3:
                    continue
                tile_id  = parts[0].lower()   # tile_x01024_y00000
                band     = parts[1].lower()    # r
                field    = parts[2].upper()    # DRA / DDE / COV

                if tile_id not in self.tiles:
                    self.tiles[tile_id] = {}
                if band not in self.tiles[tile_id]:
                    self.tiles[tile_id][band] = {
                        'dra': None, 'dde': None, 'cov': None,
                        'dstep': int(hdu.header.get('DSTEP', 8)),
                        'vis_wcs': WCS(hdu.header, naxis=2) if 'CRVAL1' in hdu.header else None,
                    }

                data = hdu.data.astype(np.float32) if hdu.data is not None else None
                if field == 'DRA':
                    self.tiles[tile_id][band]['dra'] = data
                elif field == 'DDE':
                    self.tiles[tile_id][band]['dde'] = data
                elif field == 'COV':
                    self.tiles[tile_id][band]['cov'] = data

        n_tiles = len(self.tiles)
        n_bands = sum(len(v) for v in self.tiles.values())
        print(f'ConcordanceMap: loaded {n_tiles} tiles, {n_bands} tile-band pairs from {fits_path}')

    # ------------------------------------------------------------------
    # Core interpolation
    # ------------------------------------------------------------------

    def _interpolate_field(
        self,
        mesh: np.ndarray,
        vis_xy: np.ndarray,
        dstep: int,
    ) -> np.ndarray:
        """
        Bilinearly interpolate a concordance mesh at VIS pixel positions.

        Parameters
        ----------
        mesh    : [H_mesh, W_mesh] float32 — DRA or DDE mesh at DSTEP spacing
        vis_xy  : [N, 2] float32 — VIS pixel positions (x, y)
        dstep   : int — mesh pixel spacing (from FITS header DSTEP)

        Returns
        -------
        [N] float32 — interpolated values in arcsec
        """
        # Convert VIS pixel coords to mesh-grid fractional indices
        # mesh pixel (0,0) corresponds to VIS pixel (0,0)
        mesh_x = vis_xy[:, 0] / dstep
        mesh_y = vis_xy[:, 1] / dstep

        # map_coordinates: coords are [row, col] = [y, x]
        coords = np.stack([mesh_y, mesh_x], axis=0)
        return map_coordinates(
            mesh.astype(np.float64),
            coords,
            order=1,           # bilinear
            mode='nearest',    # extrapolate at edges by clamping
        ).astype(np.float32)

    def _correction_at_vis_xy(
        self,
        tile_id: str,
        band: str,
        vis_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (dra_arcsec, ddec_arcsec) at VIS pixel positions.

        Parameters
        ----------
        vis_xy : [N, 2] — VIS pixel (x, y)

        Returns
        -------
        dra  : [N] — ΔRA* in arcsec  (= Δα × cos δ, ready to add to RA)
        ddec : [N] — ΔDec in arcsec
        """
        entry = self._get_entry(tile_id, band)
        dstep = entry['dstep']
        dra   = self._interpolate_field(entry['dra'], vis_xy, dstep)
        ddec  = self._interpolate_field(entry['dde'], vis_xy, dstep)
        return dra, ddec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_entry(self, tile_id: str, band: str) -> dict:
        tile_id = tile_id.lower()
        band    = band.lower().replace('rubin_', '')
        if tile_id not in self.tiles:
            raise KeyError(f'Tile "{tile_id}" not in concordance. Available: {list(self.tiles)[:5]}...')
        if band not in self.tiles[tile_id]:
            raise KeyError(f'Band "{band}" not in tile "{tile_id}". Available: {list(self.tiles[tile_id])}')
        entry = self.tiles[tile_id][band]
        if entry['dra'] is None or entry['dde'] is None:
            raise ValueError(f'DRA/DDE mesh missing for {tile_id}/{band}')
        return entry

    def rubin_to_vis(
        self,
        rubin_x: float | np.ndarray,
        rubin_y: float | np.ndarray,
        rubin_wcs: WCS,
        vis_wcs: WCS,
        tile_id: str,
        band: str = 'r',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project Rubin pixel coordinate(s) onto the VIS pixel grid with
        concordance correction applied.

        Parameters
        ----------
        rubin_x, rubin_y : scalar or array — Rubin pixel coordinates
        rubin_wcs        : astropy WCS for the Rubin tile
        vis_wcs          : astropy WCS for the VIS tile
        tile_id          : e.g. 'tile_x01024_y00000'
        band             : Rubin band key, e.g. 'r' or 'rubin_r'

        Returns
        -------
        vis_x, vis_y : arrays of VIS pixel coordinates (same shape as input)
        """
        rubin_x = np.atleast_1d(np.asarray(rubin_x, dtype=np.float64))
        rubin_y = np.atleast_1d(np.asarray(rubin_y, dtype=np.float64))

        # 1. Rubin pixel → sky
        ra, dec = rubin_wcs.wcs_pix2world(rubin_x, rubin_y, 0)

        # 2. Uncorrected Rubin sky → VIS pixel (to look up concordance)
        vis_x_raw, vis_y_raw = vis_wcs.wcs_world2pix(ra, dec, 0)
        vis_xy = np.stack([vis_x_raw, vis_y_raw], axis=1).astype(np.float32)

        # 3. Interpolate concordance field at those VIS positions
        dra, ddec = self._correction_at_vis_xy(tile_id, band, vis_xy)

        # 4. Apply correction: dra is ΔRA* = Δα·cos(δ)
        cos_dec = np.cos(np.deg2rad(dec))
        ra_corr  = ra  + (dra  / 3600.0) / cos_dec
        dec_corr = dec + (ddec / 3600.0)

        # 5. Corrected sky → VIS pixel
        vis_x, vis_y = vis_wcs.wcs_world2pix(ra_corr, dec_corr, 0)
        return vis_x.squeeze(), vis_y.squeeze()

    def rubin_grid_to_vis(
        self,
        rubin_xs: np.ndarray,
        rubin_ys: np.ndarray,
        rubin_wcs: WCS,
        vis_wcs: WCS,
        tile_id: str,
        band: str = 'r',
    ) -> np.ndarray:
        """
        Project a set of Rubin pixel coordinates onto the VIS grid.

        Convenience wrapper around rubin_to_vis for bulk reprojection.
        Feed the result into scipy.ndimage.map_coordinates or cv2.remap.

        Returns
        -------
        vis_coords : [N, 2] float32 — columns are (vis_x, vis_y)
        """
        vis_x, vis_y = self.rubin_to_vis(
            rubin_xs, rubin_ys, rubin_wcs, vis_wcs, tile_id, band,
        )
        return np.stack([vis_x, vis_y], axis=1).astype(np.float32)

    def coverage(
        self,
        tile_id: str,
        band: str,
        vis_xy: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Return the coverage value (min distance in VIS px to nearest anchor)
        at the given VIS positions. High coverage = model is extrapolating.

        Returns None if the concordance file has no coverage HDU.
        """
        entry = self._get_entry(tile_id, band)
        if entry['cov'] is None:
            return None
        return self._interpolate_field(entry['cov'], vis_xy, entry['dstep'])

    def list_tiles(self) -> list:
        return sorted(self.tiles.keys())

    def list_bands(self, tile_id: str) -> list:
        return sorted(self.tiles[tile_id.lower()].keys())
