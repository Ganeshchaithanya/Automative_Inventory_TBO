from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ascis.intelligence import CompanyPolicy, Position, evaluate
from ascis.market import episodes, fit_calibration, fit_demand, forecast_benchmark
from ascis.policy import ActorCritic, load, save


class ForecastCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, _ = fit_demand()
        cls.training = episodes(27103, 4000) + episodes(27104, 2000, shock_mode=True)
        cls.calibration = episodes(27105, 3000) + episodes(27106, 1500, shock_mode=True)
        cls.model = fit_calibration(cls.baseline, cls.training, cls.calibration)

    def test_independent_cohorts_improve_early_coverage_and_volatile_error(self):
        holdout = episodes(27109, 1000)
        original = forecast_benchmark(self.baseline, holdout)
        calibrated = forecast_benchmark(self.model, holdout)
        self.assertLess(abs(calibrated["opening"]["coverage_pct"] - 80),
                        abs(original["opening"]["coverage_pct"] - 80))
        self.assertGreaterEqual(calibrated["opening"]["coverage_pct"], 75)
        self.assertLess(calibrated["deadline_volatile"]["mae"], original["deadline_volatile"]["mae"])

    def test_checkpoint_preserves_calibrated_demand_paths(self):
        example = self.training[0].opening
        company = CompanyPolicy()
        policy = ActorCritic(len(evaluate(example, Position(), self.model, company).features))
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "policy.pt"
            save(checkpoint, policy, self.model, company, {})
            _, loaded, _, _ = load(checkpoint)
        expected = self.model.scenarios(example, count=256)
        np.testing.assert_array_equal(loaded.scenarios(example, count=256), expected)
        self.assertTrue(np.all(np.diff(expected) >= 0))
        self.assertTrue(np.all(expected >= 0))

    def test_training_and_calibration_cannot_share_events(self):
        with self.assertRaisesRegex(ValueError, "separate events"):
            fit_calibration(self.baseline, self.training, self.training)
