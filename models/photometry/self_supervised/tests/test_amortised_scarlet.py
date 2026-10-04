"""Renderer conventions for the amortised scarlet head: flux conservation, placement, PSF convolution."""
import unittest
import numpy as np
import torch
from scipy.signal import fftconvolve
from models.photometry.self_supervised.amortised_scarlet import (MORPH_SIZE, MORPH_SCALE, render_unconvolved, fft_convolve,
                                                                  render_templates, ScarletHead, band_morphologies, monotone_profile, knot_radii, N_KNOTS, R_MAX)
from models.photometry.self_supervised.core import BANDS


def gaussian_morph(sigma_px, centre_shift=(0., 0.)):
    g = torch.arange(MORPH_SIZE, dtype=torch.float32) - (MORPH_SIZE - 1) / 2
    vv, uu = torch.meshgrid(g, g, indexing='ij')
    m = torch.exp(-((uu - centre_shift[0]) ** 2 + (vv - centre_shift[1]) ** 2) / (2 * sigma_px ** 2))
    return (m / m.sum())[None, None]


class RendererTest(unittest.TestCase):
    def test_flux_conserved_on_rotated_and_coarser_grids(self):
        morph = gaussian_morph(4.)
        for scale, angle in ((.1, 0.), (.2, .3), (.1, -.7)):
            rot = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=torch.float32)
            sky_to_pixel = rot @ torch.diag(torch.tensor([-1 / scale, 1 / scale]))
            canvas = render_unconvolved(morph, torch.tensor([[30.2, 29.7]]), sky_to_pixel, (61, 61), pad=20, oversample=4)
            self.assertAlmostEqual(float(canvas.sum()), 1., places=2)

    def test_source_lands_at_requested_position(self):
        morph = gaussian_morph(2.5)
        pos = torch.tensor([[40.3, 20.6]]); sky_to_pixel = torch.diag(torch.tensor([-10., 10.]))
        canvas = render_unconvolved(morph, pos, sky_to_pixel, (61, 61), pad=0, oversample=2)[0, 0]
        yy, xx = torch.meshgrid(torch.arange(61.), torch.arange(61.), indexing='ij')
        cx = float((canvas * xx).sum() / canvas.sum()); cy = float((canvas * yy).sum() / canvas.sum())
        self.assertAlmostEqual(cx, 40.3, places=2); self.assertAlmostEqual(cy, 20.6, places=2)

    def test_east_is_negative_x(self):
        # Shift the morphology 5 grid pixels east (0.5 arcsec): with sky_to_pixel = diag(-10, 10) that is -5 image pixels.
        morph = gaussian_morph(2., centre_shift=(5., 0.))
        canvas = render_unconvolved(morph, torch.tensor([[30., 30.]]), torch.diag(torch.tensor([-10., 10.])), (61, 61), pad=0, oversample=2)[0, 0]
        xx = torch.arange(61.); self.assertAlmostEqual(float((canvas.sum(0) * xx).sum() / canvas.sum()), 25., places=1)

    def test_fft_convolution_matches_scipy_same(self):
        rng = np.random.default_rng(0); image = rng.normal(size=(1, 1, 40, 37)).astype('float32'); kernel = rng.random((1, 7, 7)).astype('float32')
        ours = fft_convolve(torch.tensor(image), torch.tensor(kernel))[0, 0].numpy()
        reference = fftconvolve(image[0, 0], kernel[0], mode='same')
        np.testing.assert_allclose(ours, reference, atol=1e-4)

    def test_templates_on_mixed_resolution_scene_have_unit_flux(self):
        scene = dict(sky=np.zeros((1, 2)), bands={})
        for band in BANDS:
            scale = .2 if band.startswith('rubin') else .1; n = 61 if scale == .2 else 121
            scene['bands'][band] = dict(image=torch.zeros(n, n), variance=torch.ones(n, n), mask=torch.ones(n, n, dtype=torch.bool),
                                        positions=torch.tensor([[(n - 1) / 2 + .3, (n - 1) / 2 - .2]]), sky_to_pixel=torch.diag(torch.tensor([-1 / scale, 1 / scale])), psf_sigma=1.5)
        scene['bands']['euclid_VIS']['psf_kernels'] = np.ones((1, 9, 9)) / 81
        head = ScarletHead(width=8, steps=0); params = dict(vector=head.init_vector.clone(), eps=torch.zeros(1, 1, MORPH_SIZE, MORPH_SIZE))
        templates = render_templates(scene, band_morphologies(head, params), torch.device('cpu'))
        for band, t in templates.items():
            self.assertEqual(tuple(t.shape), ((61 if band.startswith('rubin') else 121) ** 2, 1))
            self.assertAlmostEqual(float(t.sum()), 1., places=2, msg=band)



class MonotoneProfileTest(unittest.TestCase):
    def test_profile_is_decreasing_centred_and_finite(self):
        head = ScarletHead(width=8, steps=0)
        v = head.init_vector.clone(); v[0, 2 * N_KNOTS] = 2.; v[0, 2 * N_KNOTS + 1] = .5; v[0, 2 * N_KNOTS + 2] = .8   # elliptical, rotated
        bands, comps, mix = head.morphologies(dict(vector=v, eps=torch.randn(1, 1, MORPH_SIZE, MORPH_SIZE)))
        self.assertAlmostEqual(float(comps[0, 0].sum()), 1., places=5); self.assertAlmostEqual(float(comps[0, 1].sum()), 1., places=5)
        radius = torch.linspace(0, R_MAX * 1.2, 200)[None]
        profile = monotone_profile(v[:, :N_KNOTS], radius[..., None])[0, :, 0]
        self.assertTrue(bool((profile[1:] <= profile[:-1] + 1e-6).all())); self.assertLess(float(profile[-1]), 1e-5)
        c = (MORPH_SIZE - 1) // 2; m = comps[0, 0]
        yy, xx = torch.meshgrid(torch.arange(MORPH_SIZE), torch.arange(MORPH_SIZE), indexing='ij')
        self.assertAlmostEqual(float((m * xx).sum()), c, delta=.4); self.assertAlmostEqual(float((m * yy).sum()), c, delta=.4)
        self.assertEqual(tuple(bands['euclid_VIS'].shape), (1, 1, MORPH_SIZE, MORPH_SIZE))


if __name__ == '__main__': unittest.main()
