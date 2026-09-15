"""Loss semantics, augmentation geometry, and the learned object bottleneck."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from models.detection2.common import read_config
from models.detection2.data import disk_mask, novel_candidates
from models.detection2.object_decoder import LearnedObjectDecoder, splat
from detection.centernet_loss import focal_loss
from detection.dataset import TileDetectionDataset


class LossAndGeometry(unittest.TestCase):
    def test_unknown_removes_only_negative_gradient(self):
        pred = torch.tensor([[[[0.8, 0.7, 0.6]]]], requires_grad=True)
        target = torch.tensor([[[[1., 0., 0.]]]])
        focal_loss(pred, target, ignore_mask=torch.tensor([[[[True, True, False]]]])).backward()
        self.assertLess(float(pred.grad[0, 0, 0, 0]), 0)
        self.assertEqual(float(pred.grad[0, 0, 0, 1]), 0)
        self.assertGreater(float(pred.grad[0, 0, 0, 2]), 0)

    def test_all_rotation_flip_combinations(self):
        points = np.array([[0.21, 0.63], [0.77, 0.16]], dtype=np.float64)
        original = disk_mask(points, (101, 101), 3.2)
        for rotation in range(4):
            for ud in (False, True):
                for lr in (False, True):
                    transformed = TileDetectionDataset._transform_centroids(points, rotation, ud, lr)
                    expected = TileDetectionDataset._transform_mask(original, rotation, ud, lr)
                    np.testing.assert_array_equal(disk_mask(transformed, (101, 101), 3.2), expected)

    def test_matching_boundary(self):
        np.testing.assert_array_equal(novel_candidates(np.array([[3., 4.], [3., 4.01]]),
                                                       np.array([[0., 0.]]), 5.), [False, True])


class LearnedBottleneck(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.decoder = LearnedObjectDecoder(appearance_dim=4, n_bands=2, stamp_size=5)
        self.z = torch.randn(1, 3, 4)
        self.gates = torch.tensor([[1., .8, .6]])
        self.positions = torch.tensor([[[[8.2, 7.3], [8.2, 7.3]],
                                        [[12.1, 14.4], [12.1, 14.4]],
                                        [[15.2, 9.7], [15.2, 9.7]]]])
        self.shapes = [(24, 24), (24, 24)]

    def test_no_objects_means_no_source_reconstruction(self):
        images = self.decoder(self.z, self.gates*0, self.positions, self.shapes,
                              background=torch.tensor([[2., 3.]]))
        for image, bg in zip(images, (2., 3.)):
            torch.testing.assert_close(image, torch.full_like(image, bg))
        empty = self.decoder(self.z[:, :0], self.gates[:, :0], self.positions[:, :0], self.shapes)
        self.assertTrue(all((im == 0).all() for im in empty))

    def test_catalogue_order_does_not_change_reconstruction(self):
        expected = self.decoder(self.z, self.gates, self.positions, self.shapes)
        order = torch.tensor([2, 0, 1])
        actual = self.decoder(self.z[:, order], self.gates[:, order], self.positions[:, order], self.shapes)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b)

    def test_position_and_appearance_receive_gradients(self):
        z = self.z.clone().requires_grad_()
        pos = self.positions.clone().requires_grad_()
        images = self.decoder(z, self.gates, pos, self.shapes)
        ramp = torch.arange(24).float()[None, None, :]
        sum((im*ramp).sum() for im in images).backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)
        self.assertGreater(float(pos.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(z.grad).all() and torch.isfinite(pos.grad).all())

    def test_subpixel_placement_conserves_flux_and_centroid(self):
        stamp = torch.zeros(1, 1, 3, 3)
        stamp[0, 0, 1, 1] = 7.
        image = splat(stamp, torch.tensor([[[5.2, 6.7]]]), (12, 12))[0]
        y, x = torch.meshgrid(torch.arange(12), torch.arange(12), indexing='ij')
        self.assertAlmostEqual(float(image.sum()), 7., places=5)
        self.assertAlmostEqual(float((image*x).sum()/7), 5.2, places=5)
        self.assertAlmostEqual(float((image*y).sum()/7), 6.7, places=5)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
