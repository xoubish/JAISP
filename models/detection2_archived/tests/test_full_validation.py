"""Scientific invariants: overlap ownership, spherical matching, paired resampling."""
import unittest
import numpy as np
from astropy.wcs import WCS

from models.detection2.evaluation.geometry import Tile, owners, recovered, regions
from models.detection2.evaluation.statistics import bootstrap


def tile(name, center_pixel, mask=None):
    w = WCS(naxis=2)
    w.wcs.crpix = center_pixel
    w.wcs.cdelt = np.array([-.1/3600, .1/3600])
    w.wcs.crval = [53., -28.]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return Tile(name, w, np.zeros((100, 100), bool) if mask is None else mask)


class SpatialProtocol(unittest.TestCase):
    def test_overlapping_copies_have_one_owner(self):
        tiles = [tile('a', [50, 50]), tile('b', [75, 50])]
        owner = owners(tiles, np.array([53., 53.]), np.array([-28., -28.]))
        np.testing.assert_array_equal(owner, [0, 0])
        # Two tile detections at the same sky position: exactly one survives.
        self.assertEqual(int((owner == np.array([0, 1])).sum()), 1)

    def test_masked_tile_cannot_own_source(self):
        tiles = [tile('a', [50, 50], np.ones((100, 100), bool)), tile('b', [75, 50])]
        np.testing.assert_array_equal(owners(tiles, [53.], [-28.]), [1])
        np.testing.assert_array_equal(owners(tiles, [54.], [-28.]), [-1])

    def test_spherical_matching_and_empty_catalogue(self):
        hit = recovered([0., 0.], [0., 1.], [.4999/3600, .5001/3600], [0., 1.])
        np.testing.assert_array_equal(hit, [True, False])
        np.testing.assert_array_equal(recovered([0.], [0.], [], []), [False])

    def test_bootstrap_keeps_models_paired(self):
        # Strongly different sky regions, identical models: paired difference
        # must have exactly zero uncertainty despite uncertain absolute rates.
        num = np.tile(np.array([1, 9, 2, 8]), (3, 1, 1, 1))
        den = np.full_like(num, 10)
        point, draws, weights = bootstrap(num, den, replicates=1000, seed=7)
        np.testing.assert_array_equal(draws[2]-draws[0], np.zeros_like(draws[0]))
        self.assertGreater(float(draws[0].std()), 5.)
        np.testing.assert_array_equal(weights.sum(1), np.full(1000, 4))
        self.assertTrue(np.all(point == 50))

    def test_region_assignment_same_for_references_and_predictions(self):
        frame = {'center_deg': [53., -28.], 'low_arcmin': [-1, -1], 'high_arcmin': [1, 1]}
        ra, dec = np.array([53.001, 52.999]), np.array([-28.001, -27.999])
        a = regions(ra, dec, frame, 4)
        np.testing.assert_array_equal(a, regions(ra[::-1], dec[::-1], frame, 4)[::-1])
        self.assertTrue(np.all((a >= 0) & (a < 16)))


if __name__ == '__main__':
    unittest.main()
