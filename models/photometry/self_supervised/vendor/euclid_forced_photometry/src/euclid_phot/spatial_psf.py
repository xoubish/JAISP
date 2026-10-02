"""Nearest measured PSF at each source position, in one Tractor scene.

The supplied stamps are samples of the instrumental PSF on the mosaic.
Sources keep their own profiles; this class selects the convolution kernel
at each rendered centroid. All sources still contribute to the joint fit.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib

import numpy as np
from scipy.spatial import cKDTree
from tractor.psf import PixelizedPSF
from tractor.utils import BaseParams


def _unit_vectors(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.stack((np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra), np.sin(dec)), axis=-1)


class SpatialPixelizedPSF(BaseParams):
    """A fixed PSF field sampled by CATALOG-PSF or GRID-PSF stamps.

    Nearest is measured by angular separation, including RA wrap. No
    interpolation or 8x8/3x3 grouping is applied. The PSF is constant over
    each source's profile and selected at its current centroid. Thus this
    is a piecewise-constant spatial approximation, not a continuous PSF
    model or a treatment of variation within a very extended galaxy.

    ``wcs`` is the Tractor WCS for the uncropped image. Cropped blob
    images must use ``getShifted`` so their pixels address the same sky.
    """

    is_spatial = True

    def __init__(self, psf_data, wcs):
        stamps = np.array(psf_data['stamps'], dtype=np.float32, copy=True)
        ra = np.asarray(psf_data['ra'], dtype=float)
        dec = np.asarray(psf_data['dec'], dtype=float)
        if (stamps.ndim != 3 or len(stamps) == 0 or
                ra.shape != (len(stamps),) or dec.shape != ra.shape):
            raise ValueError('PSF stamps and one-dimensional sky coordinates must align')
        if any(n % 2 != 1 for n in stamps.shape[1:]):
            raise ValueError('Pixelized PSF stamps must have odd dimensions')
        sums = stamps.sum(axis=(1, 2), dtype=np.float64)
        if (not np.isfinite(stamps).all() or not np.isfinite(ra).all()
                or not np.isfinite(dec).all() or np.any(np.abs(dec) > 90)
                or np.any(sums <= 0)):
            raise ValueError('PSF samples require finite coordinates and finite positive-sum stamps')
        stamps /= sums[:, None, None]
        stamps.flags.writeable = False
        self.stamps = stamps
        self._tree = cKDTree(_unit_vectors(ra, dec))
        self.wcs = wcs
        self.x0 = self.y0 = 0.0
        digest = hashlib.sha256(stamps.tobytes())
        digest.update(ra.tobytes()); digest.update(dec.tobytes())
        self._digest = digest.hexdigest()
        self._positions = OrderedDict()
        self._kernels = OrderedDict()

    @property
    def shape(self):
        return self.stamps.shape[1:]

    def getRadius(self):
        return float(np.hypot(self.shape[0] / 2., self.shape[1] / 2.))

    def hashkey(self):
        return ('SpatialPixelizedPSF', self._digest, self.wcs.hashkey(), self.x0, self.y0)

    def copy(self):
        other = copy.copy(self)
        other._positions = OrderedDict()
        other._kernels = OrderedDict()
        return other

    def __deepcopy__(self, memo):
        other = self.copy()
        memo[id(self)] = other
        return other

    def getShifted(self, x0, y0):
        other = self.copy()
        other.x0 += x0
        other.y0 += y0
        return other

    def index_at(self, px, py):
        key = (float(px + self.x0), float(py + self.y0))
        if key not in self._positions:
            pos = self.wcs.pixelToPosition(*key)
            vector = _unit_vectors(float(pos.ra), float(pos.dec))
            if not np.isfinite(vector).all():
                raise ValueError('Cannot select a PSF at a non-finite sky position')
            self._positions[key] = int(self._tree.query(vector)[1])
            if len(self._positions) > 4096:
                self._positions.popitem(last=False)
        self._positions.move_to_end(key)
        return self._positions[key]

    def constantPsfAt(self, px, py):
        idx = self.index_at(px, py)
        if idx not in self._kernels:
            self._kernels[idx] = PixelizedPSF(self.stamps[idx])
            if len(self._kernels) > 64:
                self._kernels.popitem(last=False)
        self._kernels.move_to_end(idx)
        return self._kernels[idx]

    def getImage(self, px, py):
        return self.stamps[self.index_at(px, py)]

    def getPointSourcePatch(self, px, py, **kwargs):
        return self.constantPsfAt(px, py).getPointSourcePatch(px, py, **kwargs)

    def getFourierTransform(self, px, py, radius):
        return self.constantPsfAt(px, py).getFourierTransform(px, py, radius)
