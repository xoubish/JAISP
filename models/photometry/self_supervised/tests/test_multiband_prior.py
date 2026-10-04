"""Physics and isolation checks for the experimental band-specific prior."""
import unittest
import numpy as np
import torch

from models.photometry.self_supervised.core import BANDS
from models.photometry.self_supervised.mixture import SCALES, dictionary
from models.photometry.self_supervised.multiband_prior import (
    BandMorphologyPrior, extract_features, fit_band_priors, SpatialProjection,
)


class BandPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_independent_band_profiles_and_signed_flux(self):
        scene = dict(sky=np.zeros((2, 2)), bands={})
        priors, truth, banks = {}, {}, {}
        for j, band in enumerate(BANDS):
            d = dict(image=torch.zeros(61, 61), variance=torch.full((61, 61), .01),
                     mask=torch.ones(61, 61, dtype=torch.bool),
                     positions=torch.tensor([[28.2, 30.1], [33.4, 31.2]]),
                     sky_to_pixel=torch.eye(2)/.1, psf_sigma=1.5)
            scene['bands'][band] = d
            bank = dictionary(scene, band, torch.eye(2).repeat(2, 1, 1))
            weights = np.zeros((2, len(SCALES)))
            weights[:, [1, 3]] = [[.1+.08*j, .9-.08*j], [.8-.06*j, .2+.06*j]]
            flux = np.array([1200.+30*j, -12.-j])
            d['image'] = torch.tensor(np.einsum('pnk,nk,n->p', bank, weights, flux).reshape(61, 61)-2.)
            priors[band], truth[band], banks[band] = weights, flux, bank
        precisions = {band: np.eye(len(SCALES)) for band in BANDS}
        # Strong morphology prior still does not constrain the signed final flux.
        fits = fit_band_priors(scene, priors, precisions, strength=1e12, banks=banks)
        for band in BANDS:
            np.testing.assert_allclose(fits[band]['flux'], truth[band], atol=.002, rtol=1e-5)
            self.assertLess(fits[band]['flux'][1], 0)

    def test_features_mask_invalid_pixels_and_keep_coverage(self):
        bands = {}
        for band in BANDS:
            image, variance = torch.ones(31, 31), torch.ones(31, 31)
            image[15, 15] = float('nan')
            variance[15, 16] = float('inf')
            bands[band] = dict(image=image, variance=variance, mask=torch.ones(31, 31, dtype=torch.bool),
                               positions=torch.tensor([[15., 15.]]))
        features = extract_features(dict(sky=np.zeros((1, 2)), bands=bands), None)
        self.assertTrue(all(np.isfinite(x).all() for x in features.values()))
        coverage = features['vis_image'].reshape(1, 2, 17, 17)[0, 1]
        self.assertEqual(coverage[8, 8], 0)
        self.assertEqual(coverage[8, 9], 0)
        self.assertEqual(coverage[7, 7], 1)

    def test_projection_is_deterministic_and_prior_normalized(self):
        rng = np.random.default_rng(33)
        features = dict(vis_image=rng.normal(size=(60, 2, 9, 9)))
        targets, good = {}, {}
        for i, band in enumerate(BANDS):
            logits = rng.normal(size=(60, len(SCALES)))
            logits[:, i % len(SCALES)] += features['vis_image'][:, 0, 4, 4]
            weights = np.exp(logits-logits.max(1, keepdims=True))
            targets[band] = weights/weights.sum(1, keepdims=True)
            good[band] = np.ones(60, bool)
        split = lambda sl: dict(features={k:v[sl] for k,v in features.items()},
                                targets={k:v[sl] for k,v in targets.items()},
                                good={k:v[sl] for k,v in good.items()})
        train, validation = split(slice(0,40)), split(slice(40,None))
        a = SpatialProjection().fit(train['features']['vis_image'])
        b = SpatialProjection().fit(train['features']['vis_image'])
        np.testing.assert_allclose(a.projection, b.projection)
        head = BandMorphologyPrior().fit(train, validation, 'vis_image')
        prediction = head.predict(validation['features'])
        for band in BANDS:
            self.assertTrue((prediction[band] >= 0).all())
            np.testing.assert_allclose(prediction[band].sum(1), 1)

    def test_validation_groups_preserve_scenes_and_rare_band_coverage(self):
        from models.photometry.self_supervised.try_multiband_prior import validation_partitions
        items = [dict(identifier=i, good={b: np.array([i < 10 if b == 'rubin_u' else True])
                                        for b in BANDS}) for i in range(20)]
        tuning, calibration = validation_partitions(items, 42)
        ids = [{item['identifier'] for item in group} for group in (tuning, calibration)]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual(ids[0] | ids[1], set(range(20)))
        rare = [sum(int(item['good']['rubin_u'].sum()) for item in group)
                for group in (tuning, calibration)]
        self.assertEqual(rare, [5, 5])

    def test_raw_multiband_control_matches_capacity_without_validation_pca(self):
        rng = np.random.default_rng(92)
        features = {name: rng.normal(size=(100, width)) for name, width in
                    [('vis_image', 50), ('other_images', 80), ('vis_stem', 30), ('bottleneck', 40)]}
        targets = {band: rng.dirichlet(np.ones(len(SCALES)), size=100) for band in BANDS}
        data = lambda sl: dict(features={k:v[sl] for k,v in features.items()},
                               targets={k:v[sl] for k,v in targets.items()},
                               good={b:np.ones(len(targets[b][sl]), bool) for b in BANDS})
        train, validation = data(slice(0,70)), data(slice(70,None))
        raw = BandMorphologyPrior().fit(train, validation, 'all_images')
        foundation = BandMorphologyPrior().fit(train, validation, 'foundation')
        self.assertEqual(raw.design(validation['features']).shape[1], 73)
        self.assertEqual(foundation.design(validation['features']).shape[1], 73)
        for view, transform in foundation.transforms.items():
            np.testing.assert_allclose(transform.mean, train['features'][view].mean(0))


if __name__ == '__main__':
    unittest.main()
