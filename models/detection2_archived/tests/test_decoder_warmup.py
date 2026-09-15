"""Scientific invariants of the fixed-catalogue renderer warm-up."""
import unittest

import numpy as np
import torch

from models.detection2.decoder.model import ObjectRenderer, objective
from models.detection2.decoder.prepare import cutout, sample_object_features


class RendererWarmup(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        self.cfg = dict(appearance_dim=4, hidden_dim=8, flux_scale=10.,
                        huber_delta=3., centering_weight=.01)
        self.model = ObjectRenderer(12, self.cfg, [(20, 20), (40, 40)], [5, 9], [.2, .1])
        self.features = torch.randn(2, 3, 12)
        self.present = torch.tensor([[True, True, False], [True, False, False]])
        self.positions = torch.tensor([[[[6.2, 6.5], [12.4, 13.]],
                                         [[12.1, 11.3], [24.2, 22.6]],
                                         [[9., 9.], [18., 18.]]]]).expand(2, -1, -1, -1).clone()

    def test_padding_and_catalogue_order_cannot_change_images(self):
        expected = self.model(self.features, self.present, self.positions)
        altered = self.features.clone()
        altered[~self.present] = torch.randn_like(altered[~self.present])*20
        order = torch.tensor([2, 0, 1])
        actual = self.model(altered[:, order], self.present[:, order], self.positions[:, order])
        for a, b in zip(expected['images'], actual['images']):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(expected['centering_arcsec2'], actual['centering_arcsec2'])

    def test_object_flux_conserved_and_no_dense_background_bypass(self):
        output = self.model(self.features, self.present, self.positions)
        for band, im in enumerate(output['images']):
            flux = (im-output['background'][:, band, None, None]).sum((-2, -1))
            torch.testing.assert_close(flux, (output['flux'][:, :, band]*self.present).sum(1))
        # A nonzero background is still spatially constant when all objects are absent.
        with torch.no_grad(): self.model.background_net[-1].bias.fill_(2.)
        empty = self.model(self.features, self.present & False, self.positions)
        for im in empty['images']: torch.testing.assert_close(im, torch.full_like(im, 2.))

    def test_reconstruction_trains_appearance_templates_and_flux(self):
        targets = [torch.zeros(2, h, h) for h in (20, 40)]
        targets[0][:, 6, 6] = 12.; targets[1][:, 13, 12] = 12.
        batch = {'targets': targets, 'valid': [torch.ones_like(t, dtype=torch.bool) for t in targets]}
        output = self.model(self.features, self.present, self.positions)
        loss, _ = objective(output, batch, self.cfg)
        loss.backward()
        for module in (self.model.projector, self.model.shape_net, self.model.flux_net):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_invalid_pixels_do_not_affect_loss_or_gradients(self):
        prediction = torch.randn(1, 4, 4, requires_grad=True)
        valid = torch.ones_like(prediction, dtype=torch.bool); valid[:, :2] = False
        output = {'images': [prediction], 'centering_arcsec2': prediction.new_tensor(0.)}
        target = torch.zeros_like(prediction)
        baseline, _ = objective(output, {'targets': [target], 'valid': [valid]}, self.cfg)
        target[:, :2] = 10000.
        loss, _ = objective(output, {'targets': [target], 'valid': [valid]}, self.cfg)
        torch.testing.assert_close(loss, baseline)
        loss.backward()
        self.assertEqual(float(prediction.grad[~valid].abs().sum()), 0.)

    def test_crop_padding_and_feature_registration(self):
        image = np.arange(16).reshape(4, 4)
        expected = np.array([[-1, 4, 5], [-1, 8, 9], [-1, 12, 13]])
        np.testing.assert_array_equal(cutout(image, (-1, 1), (3, 3), fill=-1), expected)
        yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing='ij')
        features = torch.stack([xx, yy])[None].float()
        # Pixel (4,4) in a 9x9 image is feature (2,2), with one-cell ROI offsets.
        sampled = sample_object_features(features, np.array([[4., 4.]]), (9, 9), 3).reshape(9, 2)
        np.testing.assert_array_equal(sampled, np.array([[x, y] for y in (1, 2, 3) for x in (1, 2, 3)]))


if __name__ == '__main__':
    unittest.main()
