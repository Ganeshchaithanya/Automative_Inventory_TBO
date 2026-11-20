import sqlite3
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from ascis.execution import MockExecutor, proposal
from ascis.intelligence import Action, CompanyPolicy, Position, evaluate, settlement, supplier_quotes
from ascis.market import bengaluru, fit_demand
from ascis.policy import ActorCritic


class ProcurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demand, cls.fit = fit_demand()
        cls.company = CompanyPolicy()
        cls.event = bengaluru(volatile=False)
        cls.verified = replace(cls.event.opening, quote_verified=True)

    def test_censored_fit_corrects_stockout_mean(self):
        self.assertGreater(self.fit["stockout_events"], 0)
        self.assertGreater(self.fit["fitted_demand_mean"], self.fit["observed_booking_mean"])

    def test_shared_pool_and_unverified_quote(self):
        report = evaluate(self.event.opening, Position(), self.demand, self.company)
        contracts = report.report["domain_skills"][0]
        self.assertEqual(contracts["pool_capacity"]["firm-allocation"], 500)
        self.assertFalse(report.mask[Action.HOLD_300_PLUS_300])
        self.assertTrue(report.mask[Action.ABSTAIN])
        model = ActorCritic(len(report.features))
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.zero_()
            model.actor.bias[Action.HOLD_300_PLUS_300] = 10
        self.assertEqual(model.choose(report), Action.ABSTAIN)

    def test_mock_approval_is_exact_and_commit_is_idempotent(self):
        report = evaluate(self.verified, Position(), self.demand, self.company)
        requested = proposal(report, Action.HOLD_300_PLUS_300, self.company)
        args = (self.verified, self.demand, self.company)
        with sqlite3.connect(":memory:") as db:
            executor = MockExecutor(db)
            with self.assertRaisesRegex(ValueError, "exact proposal"):
                executor.commit(requested, "other", *args)
            with self.assertRaisesRegex(ValueError, "outside its decision window"):
                executor.commit(requested, requested["id"], self.event.deadline, self.demand, self.company)
            receipt = executor.commit(requested, requested["id"], *args)
            self.assertEqual(receipt, executor.commit(requested, requested["id"], *args))
            with self.assertRaisesRegex(ValueError, "contents changed"):
                executor.commit({**requested, "firm_units": 500}, requested["id"], *args)
            self.assertEqual(db.execute("SELECT count(*) FROM execution").fetchone()[0], 1)
        position = Position(**receipt["position"])
        self.assertEqual(position.committed_cash, 1_080_000)
        self.assertEqual(settlement(position, 800)["realized_contribution"], -280_000)

    def test_conflicting_pool_capacity_requires_review_even_with_verified_quotes(self):
        direct, reseller, flex = supplier_quotes(-60)
        conflicting = direct, replace(reseller, capacity=600), flex
        with patch("ascis.intelligence.supplier_quotes", return_value=conflicting):
            report = evaluate(self.verified, Position(), self.demand, self.company)
        self.assertEqual(report.report["domain_skills"][0]["pool_capacity"], {flex.pool: 600})
        for action in (Action.BUY_300, Action.BUY_500):
            self.assertFalse(report.mask[action])
            self.assertIn("conflicting supplier capacity", report.report["counterfactuals"][action]["excluded_by"])

    def test_release_option_only_at_deadline(self):
        held = Position(300, 1_080_000, True)
        opening = evaluate(self.verified, held, self.demand, self.company)
        self.assertFalse(opening.mask[Action.EXERCISE_300])
        volatile = bengaluru(volatile=True).deadline
        deadline = evaluate(volatile, held, self.demand, self.company)
        self.assertTrue(deadline.mask[Action.ABSTAIN])
        self.assertFalse(deadline.mask[Action.EXERCISE_300])


if __name__ == "__main__":
    unittest.main()
