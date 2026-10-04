"""CPU checks for photometric behavior, masks and training/checkpoint integration."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from simple_regressor.config import Config
from simple_regressor.data import StampDataset, balanced_weights
from simple_regressor.model import build_model
from simple_regressor.losses import LogScaler, TargetScaler
from simple_regressor.baseline import fit_baseline
from simple_regressor.train import train
from simple_regressor.evaluate import evaluate, _load
from simple_regressor.prepare import prepare, _spatial_split
from simple_regressor.geometry import pixel_scale_arcsec


def sample_cache(n=48, size=64):
    rng = np.random.default_rng(12)
    yy, xx = np.mgrid[:size, :size]
    r2 = (xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2
    amps = np.geomspace(1, 100, n)
    rng.shuffle(amps)
    widths = rng.uniform(1, 4, n)
    profiles = np.exp(-r2[None] / (2 * widths[:, None, None] ** 2))
    profiles /= profiles.sum((1, 2))[:, None, None]
    image = amps[:, None, None] * profiles + 0.2
    rms = np.full_like(image, 0.01)
    # An exactly measurable target: largest native aperture, calibrated by 2.7.
    flux = 2.7 * ((image - 0.2) * (r2 <= 15 ** 2)).sum((1, 2))
    return dict(stamps=np.stack([image, rms], 1).astype('f4'), flux=flux.astype('f4'),
                fluxerr=(flux * .02).astype('f4'), mag=(23.9 - 2.5 * np.log10(flux)).astype('f4'),
                object_id=np.arange(n), split=np.array(['train'] * (n-16) + ['val'] * 8 + ['test'] * 8))


class RegressorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masks_and_background(self):
        cache = sample_cache()
        cache['stamps'][0, 0, 5, 5] = np.nan
        cache['stamps'][0, 1, 6, 6] = 0
        cache['stamps'][0, 0, 6, 6] = 1e8
        cache['stamps'][0, 1, 7, 7] = np.inf
        ds = StampDataset(cache, np.arange(1), 1., bin_factor=1)
        feat, linear, _, _ = ds[0]
        self.assertTrue(torch.isfinite(feat).all())
        self.assertTrue(torch.isfinite(linear).all())
        for i in (5, 6, 7):
            self.assertEqual(linear[:, i, i].tolist(), [0., 0.])
            self.assertEqual(feat[1, i, i].item(), 0.)
        self.assertLess(abs(linear[0, 0, 0].item()), 1e-6)

    def test_native_flux_survives_binning(self):
        c = sample_cache()
        a = StampDataset(c, np.arange(1), 1., bin_factor=1)[0]
        b = StampDataset(c, np.arange(1), 1., bin_factor=2)[0]
        torch.testing.assert_close(a[1], b[1])
        torch.testing.assert_close(torch.sinh(a[0][0]).sum(), torch.sinh(b[0][0]).sum())

    def test_zero_variance_rejected(self):
        c = sample_cache(); c['stamps'][:, 1] = 0
        with self.assertRaisesRegex(ValueError, 'variance'):
            StampDataset(c, np.arange(1), 1.)[0]

    def test_calibration_and_amplitude(self):
        c = sample_cache()
        cfg = Config(stamp=64, bin_factor=1, dropout=0)
        model = build_model(cfg).eval()
        ds = StampDataset(c, c['split'] == 'train', 1., bin_factor=1)
        scaler = LogScaler.fit(c['flux'][c['split'] == 'train'])
        fit_baseline(model, DataLoader(ds, batch_size=16), scaler, 'cpu')
        feat, linear, truth, _ = next(iter(DataLoader(StampDataset(c, c['split'] == 'test', 1., bin_factor=1), batch_size=8)))
        with torch.no_grad():
            pred = scaler.to_flux(model(feat, linear))
            torch.testing.assert_close(pred, truth, rtol=2e-4, atol=2e-5)
            doubled = linear.clone(); doubled[:, 0] *= 2
            # Amplitude bypass remains functional even with identical context features.
            torch.testing.assert_close(scaler.to_flux(model(feat, doubled)), 2 * pred, rtol=2e-5, atol=2e-5)

    def test_flux_loss_transform(self):
        cfg = Config(stamp=64, loss='flux', dropout=0)
        c = sample_cache(); model = build_model(cfg).eval()
        ds = StampDataset(c, np.ones(48, bool), 1.)
        scaler = TargetScaler.fit(c['flux'])
        loader = DataLoader(ds, batch_size=48)
        fit_baseline(model, loader, scaler, 'cpu')
        feat, linear, truth, _ = next(iter(loader))
        with torch.no_grad():
            torch.testing.assert_close(scaler.to_flux(model(feat, linear)), truth, rtol=2e-4, atol=2e-5)

    def test_training_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = sample_cache()
            cfg = Config(output_dir=tmp, stamp=64, epochs=2, batch_size=16, dropout=0.,
                         augment=False, balanced_sampling=False, device='cpu', torch_threads=2)
            np.savez(Path(tmp) / 'stamps_cache.npz', **c)
            (Path(tmp) / 'metadata.json').write_text(json.dumps(dict(cache_version=2, config=cfg.to_json())))
            train(cfg)
            result = evaluate(tmp, save_plot=False)
            self.assertLess(result['overall']['nmad_fractional_flux_error'], 1e-3)
            self.assertLess(result['aperture_baseline']['nmad_fractional_flux_error'], 1e-3)
            self.assertEqual(result['overall']['n'], 8)
            model, _, _ = _load(tmp, 'last.pt')
            self.assertGreater(float(model.head[-1].weight.abs().sum()), 0)
            # Identical seed produces identical fitted weights and ordering.
            before = {k: v.clone() for k,v in model.state_dict().items()}
            train(cfg)
            again, _, _ = _load(tmp, 'last.pt')
            for k, v in again.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_learns_correction_beyond_linear_apertures(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = sample_cache(n=96)
            c['flux'] *= (1 + .15 * np.log(c['flux']))
            c['fluxerr'] = .02 * c['flux']
            c['mag'] = 23.9 - 2.5 * np.log10(c['flux'])
            cfg = Config(output_dir=tmp, stamp=64, epochs=30, patience=30, batch_size=32,
                         dropout=0., augment=False, balanced_sampling=False, device='cpu',
                         torch_threads=2, lr=1e-3)
            np.savez(Path(tmp) / 'stamps_cache.npz', **c)
            (Path(tmp) / 'metadata.json').write_text(json.dumps(dict(cache_version=2, config=cfg.to_json())))
            train(cfg)
            history = json.loads((Path(tmp) / 'history.json').read_text())
            self.assertLess(history['best_val_loss'], .5 * history['initial_val_loss'])
            result = evaluate(tmp, save_plot=False)
            self.assertLess(result['overall']['nmad_fractional_flux_error'],
                            result['aperture_baseline']['nmad_fractional_flux_error'])

    def test_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(model_version=1)
            model = build_model(cfg)
            saved = cfg.to_json(); saved.pop('model_version')
            torch.save(dict(model=model.state_dict(), config=saved, input_scale=1.,
                            target_scaler=LogScaler().state()), Path(tmp) / 'best.pt')
            loaded, config, _ = _load(tmp, 'best.pt')
            self.assertEqual(config.model_version, 1)
            self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))

    def test_split_and_constant_magnitudes(self):
        np.testing.assert_array_equal(balanced_weights(np.ones(5)), np.ones(5))
        cfg = Config(val_patches=['b'], test_patches=['b'])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            _spatial_split(np.arange(3), np.arange(3), np.array(['a','b','c']), cfg)

    def test_prepare_masks_and_rotated_wcs(self):
        from astropy.wcs import WCS
        from astropy.table import Table
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = WCS(naxis=2); w.wcs.crpix = [65,65]; w.wcs.crval = [53., -28.]
            w.wcs.ctype = ['RA---TAN','DEC--TAN']
            angle = .7
            w.wcs.cd = .1/3600 * np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
            self.assertAlmostEqual(pixel_scale_arcsec(w), .1)
            image = np.ones((128,128), 'f4'); var = np.ones_like(image)
            image[60,60] = np.inf; var[62,62] = 0
            patch = root / 'tiles' / 'patch_a'; patch.mkdir(parents=True)
            np.savez(patch / 'tile_euclid.npz', img_VIS=image, var_VIS=var, wcs_VIS=w.to_header().tostring())
            ra, dec = w.pixel_to_world_values(64.,64.)
            Table(dict(object_id=[1], ra=[ra], dec=[dec], flux_detection_total=[10.],
                       fluxerr_detection_total=[1.], mag_detection_total=[21.4])).write(root/'cat.fits')
            cfg = Config(tiles_root=str(root/'tiles'), catalog=str(root/'cat.fits'), output_dir=str(root/'out'), stamp=64)
            path = prepare(cfg)
            with np.load(path) as z:
                self.assertEqual(z['stamps'].dtype, np.float32)
                np.testing.assert_array_equal(z['stamps'][0,:,28,28], [0,0])
                np.testing.assert_array_equal(z['stamps'][0,:,30,30], [0,0])


if __name__ == '__main__':
    unittest.main()
