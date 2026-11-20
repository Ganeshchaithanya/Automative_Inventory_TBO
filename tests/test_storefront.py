"""Keep unapproved and optional allocations out of the published inventory."""

from pathlib import Path
from datetime import date
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from ascis import web
from ascis.catalog import promotion_seconds
from ascis.intelligence import Action, CompanyPolicy
from ascis.market import fit_demand


class OfferPolicy:
    def choose(self, analysis):
        if analysis.report["day"] == -10:
            return Action.EXERCISE_300 if analysis.mask[Action.EXERCISE_300] else Action.ABSTAIN
        return Action.HOLD_300_PLUS_300 if analysis.mask[Action.HOLD_300_PLUS_300] else Action.ABSTAIN


class StorefrontTests(unittest.TestCase):
    def test_promotion_time_rewards_demand_and_remaining_cost(self):
        self.assertEqual(promotion_seconds(0, 0, 0), 5)
        self.assertGreater(promotion_seconds(1.15, 0, 0), promotion_seconds(0.3, 0, 0))
        self.assertEqual(promotion_seconds(1.15, 1_080_000, 0), 10)
        self.assertEqual(promotion_seconds(1.15, 1_080_000, 640_000), 9)
        self.assertEqual(promotion_seconds(1.15, 1_080_000, 1_200_000), 8)

    def test_scenario_lab_gates_commitments_without_writing_to_ledger(self):
        demand, _ = fit_demand()
        checkpoint = (OfferPolicy(), demand, CompanyPolicy(), {})
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(web, "LEDGER", Path(temporary) / "lab.sqlite"),
            patch.object(web.app.state, "checkpoint", checkpoint, create=True),
        ):
            inputs = web.Scenario(
                event_id="icc", phase="opening", signal=0.85, volatile=False,
                quote_verified=True, budget=3_000_000, downside_limit=600_000,
            )
            stable = web.run_scenario(inputs)
            self.assertEqual(stable["selected_action"], "HOLD_300_PLUS_300")
            self.assertTrue(stable["actor_ran"])
            self.assertNotIn("transfer inventory", stable["workflow"]["activated"])
            package = web.run_scenario(inputs.model_copy(update={"offer": "package"}))
            self.assertIn("transfer inventory", package["workflow"]["activated"])
            self.assertEqual(package["selected_action"], stable["selected_action"])
            tight = web.run_scenario(inputs.model_copy(update={"budget": 700_000}))
            offer = next(row for row in tight["candidates"] if row["action"] == "HOLD_300_PLUS_300")
            self.assertIn("cash limit", offer["excluded_by"])
            self.assertEqual(tight["selected_action"], "ABSTAIN")
            quiet = web.run_scenario(inputs.model_copy(update={"signal": 0.05}))
            self.assertFalse(quiet["actor_ran"])
            self.assertFalse(web.LEDGER.exists())

    def test_only_approved_firm_stock_reaches_catalog(self):
        demand, _ = fit_demand()
        checkpoint = (OfferPolicy(), demand, CompanyPolicy(), {})
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(web, "LEDGER", Path(temporary) / "storefront.sqlite"),
            patch.object(web, "ACTIVE", {}),
            patch.object(web.app.state, "checkpoint", checkpoint, create=True),
        ):
            web.prepare_ledger()
            web.prepare_agent_requests()

            def read_stock():
                return {hotel["name"]: hotel["stock"] for hotel in web.catalog()}

            self.assertNotIn("/approvals", (web.ASSETS / "index.html").read_text(encoding="utf-8"))
            before = read_stock()
            approved, declined, volatile = web.agent_requests()
            proposal_id = approved["proposal"]["id"]
            pdf = web.review_pdf(approved["id"], proposal_id)
            self.assertTrue(pdf.body.startswith(b"%PDF"))
            with self.assertRaises(HTTPException) as error:
                web.approve(approved["id"], web.Approval(proposal_id=proposal_id, acknowledge_simulation=False))
            self.assertEqual(error.exception.status_code, 422)
            self.assertEqual(read_stock(), before)
            accepted = web.Approval(proposal_id=proposal_id, acknowledge_simulation=True)
            self.assertEqual(web.approve(approved["id"], accepted), web.approve(approved["id"], accepted))
            self.assertEqual(web.decline_agent_request(declined["id"], web.Approval(
                proposal_id=declined["proposal"]["id"], acknowledge_simulation=False,
            )), {"status": "declined"})
            after = read_stock()
            self.assertEqual(after[approved["title"]], before[approved["title"]] + approved["proposal"]["firm_units"])
            self.assertEqual(after[declined["title"]], before[declined["title"]])
            declined_market = next(hotel for hotel in web.catalog() if hotel["name"] == declined["title"])
            self.assertFalse(declined_market["event_active"])
            self.assertIsNone(declined_market["promotion_seconds"])
            self.assertNotIn("fifa", [event["id"] for event in web.scenario_inputs()["events"]])
            with self.assertRaises(HTTPException) as error:
                web.run_scenario(web.Scenario(
                    event_id="fifa", phase="opening", signal=0.88, volatile=False, quote_verified=True,
                    budget=3_000_000, downside_limit=600_000,
                ))
            self.assertEqual(error.exception.status_code, 409)
            web.approve(volatile["id"], web.Approval(
                proposal_id=volatile["proposal"]["id"], acknowledge_simulation=True,
            ))
            web.review_due_options(now=0)
            self.assertEqual(web.agent_requests(), [])
            web.ACTIVE.clear()
            web.prepare_agent_requests()
            web.review_due_options(now=float("inf"))
            follow_up, = web.agent_requests()
            self.assertEqual(follow_up["stage"], "Option deadline")
            self.assertEqual(follow_up["id"], approved["id"])
            self.assertIn("Option lapsed", next(
                item["status"] for item in web.agent_activity() if item["event"] == volatile["event_title"]
            ))
            before_exercise = read_stock()
            web.approve(follow_up["id"], web.Approval(
                proposal_id=follow_up["proposal"]["id"], acknowledge_simulation=True,
            ))
            self.assertEqual(read_stock()[approved["title"]], before_exercise[approved["title"]] + 300)
            self.assertEqual(read_stock()[volatile["title"]], before[volatile["title"]] + 300)
            self.assertEqual(web.agent_requests(), [])
            with self.assertRaises(HTTPException):
                web.review_pdf(approved["id"], proposal_id)
            property = next(hotel for hotel in web.catalog() if hotel["name"] == approved["title"])
            check_in = date.fromisoformat(property["available_from"])
            check_out = date.fromisoformat(sorted(property["rooms_by_date"])[2])
            request = web.Booking(
                request_id=uuid4(), hotel_id=property["id"], offer="package", check_in=check_in,
                check_out=check_out, rooms=2,
            )
            package = web.book(request)
            self.assertEqual(package["subtotal_inr"], 18_400)
            self.assertEqual(package, web.book(request))
            remaining = next(hotel for hotel in web.catalog() if hotel["id"] == property["id"])
            self.assertEqual(remaining["stock"], property["stock"] - 4)
            self.assertEqual(remaining["featured_stock"], property["featured_stock"] - 4)
            self.assertEqual(remaining["transfers_by_date"][str(check_in)], property["transfers_by_date"][str(check_in)] - 2)
            hotel_booking = web.book(request.model_copy(update={"request_id": uuid4(), "offer": "hotel"}))
            self.assertEqual(hotel_booking["subtotal_inr"], 16_000)
            impact = next(row for row in web.impact()["properties"] if row["id"] == property["id"])
            self.assertEqual(impact["featured_sold"], 8)
            self.assertEqual(impact["revenue"], 34_400)

            reset = web.reset_demo(web.ResetDemo(confirm="reset-demo", revision=0))
            self.assertEqual(reset, {"revision": 1, "pending_proposals": 3})
            self.assertEqual(read_stock(), before)
            self.assertIn("fifa", [event["id"] for event in web.scenario_inputs()["events"]])
            self.assertEqual(web.agent_activity(), [])
            self.assertEqual(web.impact()["bookings"], [])
            self.assertEqual(web.demo_status()["revision"], 1)
            fresh = web.agent_requests()
            self.assertTrue(all(item["id"].endswith("-r1") for item in fresh))
            with self.assertRaises(HTTPException):
                web.approve(approved["id"], accepted)
            with self.assertRaises(HTTPException):
                web.book(request)
            with self.assertRaises(HTTPException):
                web.reset_demo(web.ResetDemo(confirm="reset-demo", revision=0))

            fresh_opening = next(item for item in fresh if item["title"] == approved["title"])
            web.approve(fresh_opening["id"], web.Approval(
                proposal_id=fresh_opening["proposal"]["id"], acknowledge_simulation=True,
            ))
            web.review_due_options(now=float("inf"))
            self.assertTrue(any(
                item["stage"] == "Option deadline" and item["id"] == fresh_opening["id"]
                for item in web.agent_requests()
            ))
