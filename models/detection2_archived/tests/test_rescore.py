"""Checks for additive object removal and equal-agreement evaluation."""
import unittest
import numpy as np
import torch

from models.detection2.rescore.score import image_evidence, combine_scores
from models.detection2.rescore.metrics import ranked, operating_point, resampled_completeness


class FrozenRescoring(unittest.TestCase):
    def test_real_signal_positive_empty_signal_negative(self):
        source = torch.zeros(1, 7, 7); source[:, 3, 3] = 2.
        background = torch.full_like(source, 3.)
        valid = torch.ones_like(source, dtype=torch.bool)
        positive = image_evidence([background+source], [source], [background+source], [valid])
        negative = image_evidence([background+source], [source], [background], [valid])
        self.assertGreater(float(positive), 0)
        self.assertLess(float(negative), 0)
        torch.testing.assert_close(positive, -negative)

    def test_invalid_pixels_and_absent_object_have_no_evidence(self):
        source = torch.ones(1, 3, 3)
        for mask, component in [(torch.zeros_like(source, dtype=torch.bool), source),
                                 (torch.ones_like(source, dtype=torch.bool), source*0)]:
            score = image_evidence([source], [component], [source*0], [mask])
            torch.testing.assert_close(score, torch.zeros_like(score))
        p = np.array([.2, .4, .8])
        np.testing.assert_allclose(combine_scores(p, np.zeros(3)), np.log(p/(1-p)))

    def test_unique_reference_recovery_and_score_ties(self):
        neighbours = [np.array([0, 1]), np.array([2]), np.array([], int), np.array([3])]
        groups = np.ones((4, 4), bool)
        curve = ranked(np.array([.9, .9, .5, .1]), np.ones(4, bool), neighbours, groups)
        np.testing.assert_array_equal(curve['ends'], [2, 3, 4])
        np.testing.assert_array_equal(curve['completeness'][:, 0], [25, 50, 75])

    def test_equal_agreement_and_paired_identity(self):
        n = 200
        scores = np.arange(n, 0, -1).astype(float)
        matched = np.arange(n) < 150
        neighbours = [np.array([i]) for i in range(n)]
        groups = np.ones((n, 4), bool)
        curve = ranked(scores, matched, neighbours, groups)
        point = operating_point(curve, 90., matched, groups)
        self.assertEqual(point['detections'], 166)
        self.assertGreaterEqual(point['agreement_percent'], 90.)
        regions = np.arange(n) % 4
        weights = np.random.default_rng(12).multinomial(4, np.full(4, .25), size=50)
        a = resampled_completeness(curve, matched, groups, regions, regions, weights, 90.)
        b = resampled_completeness(curve, matched, groups, regions, regions, weights, 90.)
        np.testing.assert_array_equal(a-b, np.zeros_like(a))


if __name__ == '__main__':
    unittest.main()
