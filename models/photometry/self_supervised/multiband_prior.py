"""Experimental band-specific morphology priors, with matched pixel controls.

Every method selects positive profiles, then remeasures signed fluxes using the
same native-pixel solver. Image-fitted teacher profiles never impose a flux label.
The existing VIS-first prior and its checkpoints remain supported separately.
"""
import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from .core import BANDS
from .mixture import (
    SCALES, amplitude_scale, dictionary, ellipse_from_pixels,
    positive_profile_fit, signed_measurement,
)
from .scene_features import windows

MODES = ('population', 'vis_image', 'all_images', 'foundation')
VIEWS = {
    'vis_image': ('vis_image',),
    'all_images': ('vis_image', 'other_images'),
    'foundation': ('vis_image', 'other_images', 'vis_stem', 'bottleneck'),
}


def extract_features(scene, encoder):
    """S/N pixels and coverage in all bands; frozen 9x9 stem/5x5 bottleneck."""
    raw = {}
    for band in BANDS:
        d = scene['bands'][band]
        valid = d['mask'] & torch.isfinite(d['image']) & torch.isfinite(d['variance']) & (d['variance'] > 0)
        if not valid.any():
            raise ValueError(f'{band}: no valid pixels for feature extraction')
        bg = d['image'][valid].median()
        safe_image = torch.where(valid, d['image'] - bg, 0)
        safe_variance = torch.where(valid, d['variance'], 1)
        snr = torch.asinh(safe_image / safe_variance.sqrt() / 5)
        channels = torch.stack((snr, valid.float()))[None]
        raw[band] = windows(channels, d['positions'], 17).numpy().reshape(len(scene['sky']), -1)
    result = dict(vis_image=raw['euclid_VIS'],
                  other_images=np.concatenate([raw[b] for b in BANDS if b != 'euclid_VIS'], axis=1))
    if encoder is not None:
        result.update(encoder.feature_views(scene, bottleneck_window=5, stem_window=9))
    return result


def profile_cdf():
    radii = np.geomspace(.03, 3, 30)
    return 1 - np.exp(-radii[None, :] ** 2 / (2 * (SCALES[:, None] ** 2 + .095 ** 2)))


def profile_error(prediction, target):
    return float(np.mean(((prediction - target) @ profile_cdf()) ** 2))


def precision_from_errors(prediction, target):
    """Independent calibration scenes; bias counts in the second moment."""
    if len(target) < 8:
        raise ValueError('At least eight independent calibration profiles per band required')
    cdf = profile_cdf()
    residual = (prediction - target) @ cdf
    second = residual.T @ residual / len(residual)
    covariance = .8 * second + .2 * np.diag(np.diag(second)) + .03 ** 2 * np.eye(cdf.shape[1])
    precision = cdf @ np.linalg.solve(covariance, cdf.T)
    return (precision + precision.T) / 2


class SpatialProjection:
    """Training-only, deterministic randomized PCA with whitened scores."""
    def fit(self, features, components=24):
        x = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
        if len(x) < 8 or not np.isfinite(x).all():
            raise ValueError('Insufficient or nonfinite training features')
        self.mean = x.mean(0)
        self.scale = x.std(0).clip(.01)
        x = (x - self.mean) / self.scale
        rank = min(components, len(x) - 2, x.shape[1])
        width = min(rank + 12, len(x), x.shape[1])
        rng = np.random.default_rng(1729)
        with threadpool_limits(limits=2):
            q, _ = np.linalg.qr(x @ rng.normal(size=(x.shape[1], width)), mode='reduced')
            for _ in range(2):
                z, _ = np.linalg.qr(x.T @ q, mode='reduced')
                q, _ = np.linalg.qr(x @ z, mode='reduced')
            _, singular, vectors = np.linalg.svd(q.T @ x, full_matrices=False)
        self.projection = vectors[:rank].T / np.maximum(singular[:rank] / np.sqrt(len(x)), .01)
        return self

    def __call__(self, features):
        x = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
        return ((x - self.mean) / self.scale) @ self.projection


class BandMorphologyPrior:
    """Shared spatial projections, independently selected band regressions."""
    def fit(self, train, validation, mode):
        if mode not in VIEWS:
            raise ValueError(mode)
        self.mode = mode
        dimensions = {view: (24 if 'image' in view else 12) for view in VIEWS[mode]}
        if mode == 'all_images':
            # Match the foundation head's 72 latent dimensions using pixels only.
            dimensions['other_images'] = 48
        self.transforms = {view: SpatialProjection().fit(train['features'][view], components=dimensions[view])
                           for view in VIEWS[mode]}
        z = self.design(train['features'])
        vz = self.design(validation['features'])
        self.heads = {}
        for band in BANDS:
            good, vgood = train['good'][band], validation['good'][band]
            target, vtarget = train['targets'][band][good], validation['targets'][band][vgood]
            if good.sum() < 12 or vgood.sum() < 8:
                raise ValueError(f'{band}: insufficient independent teacher coverage for {mode}')
            population = target.mean(0)
            population /= population.sum()
            baseline = np.tile(population, (len(vtarget), 1))
            best = dict(beta=None, fraction=0., error=profile_error(baseline, vtarget))
            log_target = np.log(np.maximum(target, .01))
            log_target -= log_target.mean(1, keepdims=True)
            a = z[good]
            trials = []
            for alpha in (1., 10., 100.):
                penalty = np.eye(a.shape[1]) * alpha
                penalty[0, 0] = 0
                beta = np.linalg.solve(a.T @ a + penalty, a.T @ log_target)
                prediction = softmax(np.clip(vz[vgood] @ beta, -8, 8), axis=1)
                for fraction in (.25, .5, 1.):
                    error = profile_error(fraction * prediction + (1 - fraction) * baseline, vtarget)
                    trials.append(dict(alpha=alpha, fraction=fraction, error=error))
                    if error < best['error']:
                        best = dict(beta=beta, fraction=fraction, error=error, alpha=alpha)
            self.heads[band] = dict(**best, population=population,
                population_error=profile_error(baseline, vtarget), trials=trials,
                training_examples=int(good.sum()), validation_examples=int(vgood.sum()))
        return self

    def design(self, features):
        latent = [self.transforms[view](features[view]) for view in VIEWS[self.mode]]
        return np.c_[np.ones(len(latent[0])), np.concatenate(latent, axis=1)]

    def predict(self, features):
        z = self.design(features)
        result = {}
        for band, head in self.heads.items():
            base = np.tile(head['population'], (len(z), 1))
            result[band] = base if head['beta'] is None else (
                head['fraction'] * softmax(np.clip(z @ head['beta'], -8, 8), axis=1)
                + (1 - head['fraction']) * base)
        return result


def fit_band_priors(scene, priors, precisions, strength=1., banks=None):
    """Band-dependent profile priors, identical solver and background for controls."""
    if set(priors) != set(BANDS) or set(precisions) != set(BANDS):
        raise ValueError('All ten band priors and precision matrices are required')
    ellipse = ellipse_from_pixels(scene)
    banks = {} if banks is None else banks
    result = {}
    for band in BANDS:
        d = scene['bands'][band]
        if band not in banks:
            banks[band] = dictionary(scene, band, ellipse)
        prior = priors[band]
        if prior.shape != (len(scene['sky']), len(SCALES)):
            raise ValueError(f'{band}: inconsistent prior dimensions')
        if not np.isfinite(prior).all() or (prior < 0).any() or not np.allclose(prior.sum(1), 1):
            raise ValueError(f'{band}: prior must be positive and unit sum')
        reference = amplitude_scale(d, banks[band], prior)
        weights, _ = positive_profile_fit(d, banks[band], prior, strength,
                                           reference, precision=precisions[band])
        result[band] = signed_measurement(d, banks[band], weights)
        result[band]['weights'] = weights
    return result
