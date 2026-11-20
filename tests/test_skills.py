import unittest

import torch

from ascis.skills import METRICS, WIDTH, SkillMixture, compare


class SkillTests(unittest.TestCase):
    def test_hybrid_learns_with_both_branches_and_produces_bounded_probabilities(self):
        model = SkillMixture(WIDTH, "residual_cross")
        features = torch.linspace(-1, 1, WIDTH).repeat(4, 1)
        reference = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
        probabilities = model(features, reference)
        self.assertEqual(probabilities.shape, (4, 2))
        self.assertTrue(((probabilities > 0) & (probabilities < 1)).all())
        probabilities.sum().backward()
        for skill in model.skills:
            self.assertGreater(skill.output.weight.grad.abs().sum().item(), 0)

    def test_released_outputs_are_compared_to_the_existing_model(self):
        def report(normal, stress):
            return {
                cohort: {name: {"learned_mse": mse, "baseline_mse": 0.20} for name in METRICS}
                for cohort, mse in (("normal", normal), ("stress", stress))
            }

        candidate = report(0.12, 0.15)
        incumbent = report(0.10, 0.16)
        results = compare(candidate, incumbent, (True, False, False, False))
        self.assertLess(results["sell_through"], 0)
        self.assertGreater(results["full_sellout"], 0)
