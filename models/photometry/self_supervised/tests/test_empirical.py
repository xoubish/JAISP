"""Tests of independent truth rendering and injection bookkeeping."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from models.photometry.self_supervised.empirical import (
    render, unit, reconstruct, gaussian_psf, oracle_fit, positive_amplitude, BACKGROUND_REGIONS, load_scene, sky_object_mask)
from models.photometry.self_supervised.core import BANDS


class EmpiricalTests(unittest.TestCase):
    def test_pixel_response_once_and_phase(self):
        latent = np.zeros((41, 41)); latent[20, 20] = 1
        psf = gaussian_psf(1.3)
        matrix = np.diag([-.1, .1])
        for phase in (0., .37):
            result = render(latent, matrix, matrix, [40+phase, 40], (81, 81), kernel=psf)
            self.assertAlmostEqual(result.sum(), 1., places=5)
            if phase == 0:
                expected = np.zeros((81, 81)); expected[22:59, 22:59] = psf
                # Fourier interpolation has a small approximation error; a
                # second pixel integration would add 1/12 pixel^2, much larger.
                np.testing.assert_allclose(result, expected, atol=2.5e-4)
                yy, xx = np.indices(result.shape)
                self.assertLess(abs((result*(xx-40)**2).sum()-(expected*(xx-40)**2).sum()), .01)
            yy, xx = np.indices(result.shape)
            self.assertAlmostEqual((result*xx).sum()/result.sum(), 40+phase, places=3)

    def test_color_centers_wcs_rotation_and_crop(self):
        stamp = np.zeros((41, 41)); stamp[17, 21] = .7; stamp[24, 15] = .3
        dm = np.diag([-.1, .1]); tm = np.diag([-.2, .2])
        center = [19.2, 20.1]
        a = render(stamp, dm, tm, [30., 30.], (61, 61), angle=.6, donor_center=center,
                   kernel=gaussian_psf(2.))
        self.assertAlmostEqual(a.sum(), 1., places=3)
        intrinsic_centroid = np.array([21*.7+15*.3, 17*.7+24*.3])
        rotation = np.array([[np.cos(.6), -np.sin(.6)], [np.sin(.6), np.cos(.6)]])
        expected = np.array([30., 30.])+np.linalg.solve(tm, rotation@dm@(intrinsic_centroid-center))
        yy, xx = np.indices(a.shape)
        np.testing.assert_allclose([(a*xx).sum()/a.sum(), (a*yy).sum()/a.sum()], expected, atol=.02)
        cropped = render(stamp, dm, tm, [-1, -1], (21, 21), donor_center=center)
        self.assertLess(cropped.sum(), .8)  # Never normalize missing wings.

    def test_nonparametric_clumps_and_mask(self):
        yy, xx = np.indices((61, 61))
        latent = unit(np.exp(-((xx-27)**2+(yy-29)**2)/8)+.5*np.exp(-((xx-35)**2+(yy-33)**2)/5))
        psf = gaussian_psf(1.2); image = 1000*fftconvolve(latent, psf, mode='same')
        valid = np.ones(image.shape, bool); valid[3:8, 4:12] = False
        noisy = image+np.random.default_rng(4).normal(0, .3, image.shape)
        noisy[~valid] = 1e9
        q, _ = reconstruct(noisy, np.full(image.shape, .09), valid, psf, np.hypot(xx-30, yy-30)<23)
        self.assertTrue((q >= 0).all())
        self.assertLess(abs(q.sum()/1000-1), .03)
        self.assertGreater(q[29, 27], q[29, 31]); self.assertGreater(q[33, 35], q[33, 39])

    def test_known_flux_blend_negative_amplitudes_and_background(self):
        stamp = np.zeros((41, 41)); stamp[20, 20] = 1
        matrix = np.diag([-.1, .1]); psf = gaussian_psf(2)
        profiles = [render(stamp, matrix, matrix, p, (61, 61), kernel=psf) for p in ([30, 30], [33, 30])]
        image = 8.3+12*profiles[0]-3*profiles[1]
        flux, errors = oracle_fit(image, np.ones(image.shape), np.ones(image.shape, bool), profiles)
        np.testing.assert_allclose(flux, [12, -3], atol=1e-10)
        self.assertTrue((errors > 0).all())
        self.assertGreater(positive_amplitude(-2., 1.), 0.)

    def test_truth_blind_input_adapter(self):
        z = dict(sky=np.array([[53., -28.]]))
        for b in BANDS:
            z.update({b+'__image':np.ones((5,5),np.float32), b+'__variance':np.ones((5,5),np.float32),
                      b+'__mask':np.ones((5,5),bool), b+'__positions':np.array([[2.,2.]]),
                      b+'__sky_to_pixel':np.eye(2), b+'__psf_sigma':1.,
                      b+'__psf_kernels':np.ones((1,3,3))/9, b+'__truth':np.array([123456.])})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scene.npz'; np.savez(path,**z); scene=load_scene(path)
        self.assertEqual(set(scene['bands']),set(BANDS))
        for band in scene['bands'].values():
            self.assertNotIn('truth',band)
            np.testing.assert_array_equal(band['image'].numpy(),np.ones((5,5)))

    def test_sky_guard_detects_non_VIS_source_and_keeps_noise(self):
        yy,xx=np.indices((256,256));valid=np.ones((256,256),bool)
        sky=np.random.default_rng(42).normal(0,1,valid.shape)
        vis_mask,vis_sources=sky_object_mask(sky,valid,.2)
        red=sky+30*np.exp(-((xx-120)**2+(yy-130)**2)/32)
        red_mask,red_sources=sky_object_mask(red,valid,.2)
        self.assertTrue(vis_mask[130,120]);self.assertFalse(red_mask[130,120])
        self.assertGreater(vis_mask.mean(),.99)
        self.assertTrue(np.min(np.linalg.norm(red_sources-[120,130],axis=1))<1)


if __name__ == '__main__': unittest.main()
