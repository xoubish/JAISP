"""Offline tests for the detection-catalog pipeline: scene geometry and MER matching."""
import unittest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from models.photometry.self_supervised.detcat.prepare import scene_for_source, _crop
from models.photometry.self_supervised.detcat.mer import match_sky
from models.photometry.self_supervised.core import BANDS


def tan_wcs(ra, dec, scale_arcsec, shape):
    w = WCS(naxis=2); w.wcs.ctype = ['RA---TAN', 'DEC--TAN']; w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2 + 1, shape[0] / 2 + 1]; w.wcs.cd = np.diag([-scale_arcsec, scale_arcsec]) / 3600
    return w


class SceneBuilderTest(unittest.TestCase):
    def inputs(self):
        ra, dec = 53.2, -28.1; rng = np.random.default_rng(1)
        offsets = np.array([[0, 0], [2, 0], [0, 5], [20, 0], [-40, -40], [52, 0]], float)  # arcsec east/north
        sky = np.column_stack((ra + offsets[:, 0] / 3600 / np.cos(np.deg2rad(dec)), dec + offsets[:, 1] / 3600))
        inputs = dict(sky=sky, source=np.arange(len(sky)), tile='synthetic')
        for short in ('VIS', 'Y', 'J', 'H'):
            shape = (1160, 1160); w = tan_wcs(ra, dec, .1, shape); b = 'euclid_' + short
            inputs.update({b + '__image': rng.normal(size=shape).astype('float32'), b + '__variance': np.ones(shape, 'float32'),
                           b + '__mask': np.ones(shape, bool), b + '__wcs': w.to_header().tostring(), b + '__magzero': 24.6,
                           b + '__psf_stamps': np.ones((2, 5, 5), 'float32') / 25, b + '__psf_sky': np.array([[ra, dec], [ra, dec + .01]]),
                           b + '__positions': np.column_stack(w.world_to_pixel_values(*sky.T)), b + '__kernel_index': np.zeros(len(sky), int)})
        shape = (512, 512); w = tan_wcs(ra, dec, .2, shape)
        inputs.update({'rubin__image': rng.normal(size=(6, *shape)).astype('float32'), 'rubin__variance': np.ones((6, *shape), 'float32'),
                       'rubin__mask': np.ones((6, *shape), bool), 'rubin__wcs': w.to_header().tostring(), 'rubin__scale_arcsec': .2,
                       'rubin__positions': np.column_stack(w.world_to_pixel_values(*sky.T))})
        return inputs

    def test_scene_geometry_and_membership(self):
        inputs = self.inputs(); sigmas = {b: 1. for b in BANDS}
        scene, info = scene_for_source(inputs, 0, sigmas)
        self.assertIsNotNone(scene); self.assertEqual(set(scene['bands']), set(BANDS))
        self.assertEqual(scene['source_indices'].tolist(), [0, 1, 2])  # 2" and 5" neighbours inside 14"; 20" neighbour excluded
        self.assertEqual(scene['central'], 0); self.assertEqual(info['n_sources'], 3)
        vis = scene['bands']['euclid_VIS']; self.assertEqual(tuple(vis['image'].shape), (121, 121))
        np.testing.assert_allclose(vis['positions'][0].numpy(), [60, 60], atol=.5)
        np.testing.assert_allclose(vis['positions'][1].numpy() - vis['positions'][0].numpy(), [-20, 0], atol=.05)  # 2" east = -20 px
        rubin = scene['bands']['rubin_r']; self.assertEqual(tuple(rubin['image'].shape), (61, 61))
        np.testing.assert_allclose(np.linalg.inv(vis['sky_to_pixel'].numpy()) @ [1, 0], [-.1, 0], atol=1e-3)
        self.assertEqual(len(vis['psf_kernels']), 3)

    def test_rubin_edge_is_padded_and_euclid_edge_rejected(self):
        inputs = self.inputs(); sigmas = {b: 1. for b in BANDS}
        scene, info = scene_for_source(inputs, 5, sigmas)  # 52" east: inside Euclid (58" half), outside Rubin (51.2" half)
        self.assertIsNotNone(scene)
        self.assertLess(info['valid_fraction']['rubin_r'], 1.); self.assertEqual(info['valid_fraction']['euclid_VIS'], 1.)
        rubin = scene['bands']['rubin_r']; self.assertFalse(rubin['mask'].numpy().all()); self.assertTrue(np.isnan(rubin['image'].numpy()[~rubin['mask'].numpy()]).all())
        inputs['euclid_VIS__positions'][5] = [1150, 580]
        scene, reason = scene_for_source(inputs, 5, sigmas); self.assertIsNone(scene); self.assertEqual(reason, 'edge_VIS')

    def test_masked_center_is_rejected(self):
        inputs = self.inputs(); inputs['euclid_VIS__mask'][:] = False
        scene, reason = scene_for_source(inputs, 0, {b: 1. for b in BANDS}); self.assertIsNone(scene); self.assertEqual(reason, 'masked_center')

    def test_crop_padding(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        self.assertIsNone(_crop(image, image, image > -1, np.array([1., 1.]), 3))
        im, var, mask, origin = _crop(image, image, image > -1, np.array([1., 1.]), 3, pad=True)
        self.assertEqual(im.shape, (7, 7)); self.assertEqual(origin.tolist(), [-2, -2]); self.assertEqual(mask.sum(), 25); self.assertEqual(im[2, 2], 0.)


class MatchTest(unittest.TestCase):
    def test_nearest_within_radius(self):
        center = (53., -28.); cosd = np.cos(np.deg2rad(-28))
        det = np.array([[53., -28.], [53. + 1 / 3600 / cosd, -28.], [53., -28. + 10 / 3600]])
        ref = np.array([[53. + .3 / 3600 / cosd, -28.], [53. + 1.2 / 3600 / cosd, -28. + .2 / 3600]])
        index, sep, second, within = match_sky(det, ref, center, radius=.5)
        self.assertEqual(index.tolist(), [0, 1, -1]); np.testing.assert_allclose(sep[:2], [.3, np.hypot(.2, .2)], atol=1e-3)
        self.assertEqual(within.tolist(), [1, 2, 0]); self.assertTrue(np.isinf(sep[2]) or sep[2] > 5)


if __name__ == '__main__': unittest.main()
