"""Independent checks of compression, pixel centering, units and cropped flux."""
import unittest
import numpy as np
import torch

from models.jaisp_foundation_v10 import compress_snr, decompress_snr
from models.photometry.self_supervised.audit_flux_path import gaussian_pixels
from models.photometry.self_supervised.core import templates, fit_flux
from models.photometry.self_supervised.pixel_psf import convolved_profile
from models.photometry.self_supervised.scene_features import windows
from models.photometry.self_supervised.mixture import dictionary, positive_profile_fit, signed_measurement, SCALES


class FluxPathAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_asinh_preserves_core_order_and_has_correct_inverse(self):
        snr = torch.tensor([-1e5, -100., 0., 1., 50., 100., 500., 5000., 1e5], dtype=torch.float64)
        encoded = compress_snr(snr, 'asinh50')
        self.assertTrue(bool((encoded[1:] > encoded[:-1]).all()))
        torch.testing.assert_close(decompress_snr(encoded, 'asinh50'), snr, rtol=1e-12, atol=1e-10)
        # Identity * RMS is not an inverse for a compressed decoder output.
        self.assertLess(float(encoded[-1]), float(snr[-1])/100)

    def test_exact_gaussian_stars_recover_flux_at_subpixel_phases(self):
        for phase in (0., .17, .37, .49):
            position = torch.tensor([[30.+phase, 30.-phase]], dtype=torch.float64)
            exact = gaussian_pixels((61, 61), position[0].numpy(), .932)
            model = templates(position, torch.zeros(1, 2, 2, dtype=torch.float64),
                              torch.eye(2, dtype=torch.float64)*10, .932, (61, 61))
            self.assertAlmostEqual(float(model.sum()), 1., places=6)
            fit = fit_flux(torch.tensor(12345.*exact-7), torch.ones(61, 61), model)
            self.assertLess(abs(float(fit['flux'][0])/12345.-1), 4e-6)

    def test_signed_blend_cropped_wings_masked_cores_and_units(self):
        shape = (41, 41)
        first = gaussian_pixels(shape, (.2, 20.37), 2.)
        second = gaussian_pixels(shape, (4.73, 21.11), 2.3)
        self.assertLess(first.sum(), .7)  # preserve missing wings, not unit crop sum
        bank = torch.tensor(np.column_stack((first.ravel(), second.ravel())))
        truth = torch.tensor([321., -29.], dtype=torch.float64)
        image = (bank @ truth).reshape(shape)-3.5
        variance = torch.linspace(.2, 3, np.prod(shape), dtype=torch.float64).reshape(shape)
        mask = torch.ones(shape, dtype=torch.bool)
        mask[18:24, :6] = False  # both source cores masked
        image[~mask] = float('nan')
        fit = fit_flux(image, variance, bank, mask)
        torch.testing.assert_close(fit['flux'], truth, atol=1e-8, rtol=1e-10)
        changed = fit_flux(image*7, variance*49, bank[:, [1, 0]], mask)
        torch.testing.assert_close(changed['flux'], truth[[1, 0]]*7, atol=1e-8, rtol=1e-10)
        torch.testing.assert_close(changed['error'], fit['error'][[1, 0]]*7, atol=1e-8, rtol=1e-10)

    def test_column_preconditioning_restores_template_amplitude(self):
        shape = (51, 51)
        bank = torch.tensor(np.column_stack([gaussian_pixels(shape, (x, 25.), 2.).ravel()
                                            for x in (23., 27.)]))
        true = torch.tensor([197., 311.], dtype=torch.float64)
        im = (bank @ true).reshape(shape)+2.
        scale = torch.tensor([1e-7, 1e5], dtype=torch.float64)
        a = fit_flux(im, torch.ones(shape), bank)
        b = fit_flux(im, torch.ones(shape), bank*scale)
        torch.testing.assert_close(b['flux']*scale, true, atol=1e-8, rtol=1e-10)
        torch.testing.assert_close(b['error']*scale, a['error'], atol=1e-8, rtol=1e-10)

    def test_asymmetric_psf_keeps_orientation_negative_lobes_and_crop(self):
        stamp = np.zeros((9, 9)); stamp[4, 4] = .75; stamp[3, 6] = .30; stamp[5, 3] = -.05
        result = convolved_profile((41, 41), (20, 20), np.zeros((2, 2)), stamp)
        np.testing.assert_allclose(result[16:25, 16:25], stamp, atol=1e-12)
        # Left side leaves the image. Neither the stamp lobes nor the cropped
        # footprint is clipped/renormalized by the renderer.
        edge = convolved_profile((41, 41), (-1, 20), np.zeros((2, 2)), stamp)
        self.assertAlmostEqual(edge.sum(), .30, places=10)

    def test_feature_windows_preserve_fractional_pixel_center(self):
        yy, xx = torch.meshgrid(torch.arange(23.), torch.arange(27.), indexing='ij')
        ramp = torch.stack((xx, yy))[None]
        positions = torch.tensor([[10.37, 12.19], [6.49, 9.17]])
        sampled = windows(ramp, positions, 3)
        torch.testing.assert_close(sampled[:, :, 1, 1], positions, atol=2e-6, rtol=0)
        torch.testing.assert_close(sampled[:, :, 0, 2], positions+torch.tensor([1., -1.]), atol=2e-6, rtol=0)

    def test_supplied_stellar_classification_removes_size_floor_in_mixed_blend(self):
        shape = (81, 81)
        positions = np.array([[37.37, 40.19], [43.17, 40.63]])
        star = gaussian_pixels(shape, positions[0], .932)
        galaxy = gaussian_pixels(shape, positions[1], np.hypot(.932, SCALES[3]/.1))
        truth = np.array([300., 190.])
        d = dict(image=torch.tensor(truth[0]*star+truth[1]*galaxy-3),
                 variance=torch.full(shape, .01), mask=torch.ones(shape, dtype=torch.bool),
                 positions=torch.tensor(positions, dtype=torch.float32),
                 sky_to_pixel=torch.eye(2)/.1, psf_sigma=.932)
        scene = dict(sky=np.zeros((2, 2)), bands={'euclid_VIS':d}, point_sources=np.array([True, False]))
        bank = dictionary(scene, 'euclid_VIS', torch.eye(2).repeat(2, 1, 1))
        np.testing.assert_allclose(bank[:, 0, 0], bank[:, 0, -1], atol=1e-12)
        self.assertGreater(np.abs(bank[:, 1, 0]-bank[:, 1, -1]).max(), .01)
        # The stellar prior strongly prefers an extended profile. Supplied
        # classification must override that preference without changing flux units.
        prior = np.zeros((2, len(SCALES))); prior[0, -1] = 1; prior[1, 3] = 1
        weights, _ = positive_profile_fit(d, bank, prior, strength=1e8, reference_flux=truth)
        flux = signed_measurement(d, bank, weights)['flux']
        np.testing.assert_allclose(flux, truth, rtol=5e-6, atol=.001)
        with self.assertRaisesRegex(ValueError, 'boolean mask'):
            dictionary(dict(scene, point_sources=np.array([1, 0])), 'euclid_VIS', torch.eye(2).repeat(2, 1, 1))


if __name__ == '__main__':
    unittest.main()
