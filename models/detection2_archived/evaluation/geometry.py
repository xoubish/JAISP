"""Prediction-independent ownership of overlapping tile footprints."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from astropy.wcs import WCS
from scipy.spatial import cKDTree


@dataclass
class Tile:
    name: str
    wcs: WCS
    mask: np.ndarray

    def edge_distance(self, ra, dec):
        x, y = self.wcs.all_world2pix(ra, dec, 0)
        h, w = self.mask.shape
        valid = np.isfinite(x) & np.isfinite(y) & (x >= 4) & (x < w-4) & (y >= 4) & (y < h-4)
        idx = np.flatnonzero(valid)
        valid[idx] &= ~self.mask[y[idx].astype(int), x[idx].astype(int)]
        distance = np.minimum.reduce([x-4, w-4-x, y-4, h-4-y])
        return np.where(valid, distance, -np.inf)


def owners(tiles, ra, dec):
    """Use the eligible tile with the largest distance from its four-pixel edge.

    Ties use sorted tile order. No score, label type or recovery outcome enters
    this decision. The same partition applies to references and detections.
    """
    if [t.name for t in tiles] != sorted(t.name for t in tiles):
        raise ValueError('Tile order must be sorted')
    best = np.full(len(ra), -np.inf)
    owner = np.full(len(ra), -1, dtype=np.int32)
    for i, tile in enumerate(tiles):
        distance = tile.edge_distance(ra, dec)
        take = distance > best
        owner[take], best[take] = i, distance[take]
    return owner


def vectors(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def recovered(reference_ra, reference_dec, detection_ra, detection_dec, radius_arcsec=.5):
    if not len(detection_ra):
        return np.zeros(len(reference_ra), dtype=bool)
    distances = cKDTree(vectors(detection_ra, detection_dec)).query(vectors(reference_ra, reference_dec))[0]
    return distances < 2*np.sin(np.deg2rad(radius_arcsec/3600)/2)


def tangent(ra, dec, center):
    """Gnomonic projection in arcminutes, adequate across RA wrapping too."""
    ra, dec, ra0, dec0 = np.deg2rad(ra), np.deg2rad(dec), *np.deg2rad(center)
    dra = ra-ra0
    den = np.sin(dec0)*np.sin(dec)+np.cos(dec0)*np.cos(dec)*np.cos(dra)
    x = np.cos(dec)*np.sin(dra)/den
    y = (np.cos(dec0)*np.sin(dec)-np.sin(dec0)*np.cos(dec)*np.cos(dra))/den
    return np.column_stack([x, y])*180/np.pi*60


def regions(ra, dec, frame, side):
    xy = tangent(ra, dec, frame['center_deg'])
    low, high = np.asarray(frame['low_arcmin']), np.asarray(frame['high_arcmin'])
    ij = np.floor((xy-low)/(high-low)*side).astype(int)
    ij = np.clip(ij, 0, side-1)
    return ij[:, 0] + side*ij[:, 1]
