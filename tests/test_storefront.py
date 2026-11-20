"""Keep unapproved and optional allocations out of the published inventory."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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
    def test_seat_and_room_capacity_survive_racing_requests_and_retries(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(web, "LEDGER", Path(directory) / "bookings.sqlite"):
            web.prepare_ledger()
            hotel = next(item for item in web.catalog() if item["id"] == "icc")
            first, second = map(date.fromisoformat, sorted(hotel["rooms_by_date"])[:2])
            booking = web.Booking(request_id=uuid4(), hotel_id="icc", offer="hotel", check_in=first,
                                  check_out=second, rooms=12)
            rival = booking.model_copy(update={"request_id": uuid4(), "offer": "package"})

            def submit(request):
                try:
                    return web.book(request)
                except HTTPException as error:
                    return error.status_code

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(submit, (booking, rival)))
            self.assertIn(409, results)
            winner = next(index for index, result in enumerate(results) if isinstance(result, dict))
            receipt = results[winner]
            self.assertEqual(submit((booking, rival)[winner]), receipt)
            live = next(item for item in web.catalog() if item["id"] == "icc")
            self.assertEqual(live["rooms_by_date"][first.isoformat()], hotel["rooms_by_date"][first.isoformat()] - 12)
            consumed_transfers = 12 if winner == 1 else 0
            self.assertEqual(live["transfers_by_date"][first.isoformat()], 24 - consumed_transfers)

    def test_transfer_sellout_still_allows_hotel_only_checkout(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(web, "LEDGER", Path(directory) / "transfers.sqlite"),
            patch.object(web, "approved_allocations", return_value={"India vs Pakistan · Cricket World Cup": (300, 1_080_000)}),
        ):
            web.prepare_ledger()
            hotel = next(item for item in web.catalog() if item["id"] == "icc")
            first, second = map(date.fromisoformat, sorted(hotel["rooms_by_date"])[:2])
            package = web.Booking(request_id=uuid4(), hotel_id="icc", offer="package", check_in=first,
                                  check_out=second, rooms=12)
            web.book(package)
            web.book(package.model_copy(update={"request_id": uuid4()}))
            with self.assertRaises(HTTPException) as rejected:
                web.book(package.model_copy(update={"request_id": uuid4(), "rooms": 1}))
            self.assertIn("transfers", rejected.exception.detail)
            room = web.book(package.model_copy(update={"request_id": uuid4(), "offer": "hotel", "rooms": 1}))
            self.assertEqual(room["subtotal_inr"], 4_000)

    def test_bengaluru_selection_survives_restart_and_only_publishes_after_approval(self):
        demand, _ = fit_demand()
        checkpoint = (OfferPolicy(), demand, CompanyPolicy(), {})
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(web, "LEDGER", Path(temporary) / "scenario.sqlite"),
            patch.object(web, "ACTIVE", {}),
            patch.object(web.app.state, "checkpoint", checkpoint, create=True),
        ):
            web.prepare_ledger()
            inputs = web.Scenario(
                event_id="blr", phase="opening", signal=0.78, volatile=False, quote_verified=True,
                budget=3_000_000, downside_limit=600_000, queue_for_review=True,
            )
            self.assertIsNone(web.run_scenario(inputs.model_copy(update={"signal": 0.05}))["review_status"])
            self.assertEqual(web.run_scenario(inputs)["review_status"], "queued")
            self.assertEqual(web.run_scenario(inputs)["review_status"], "existing request")
            initial, = web.agent_requests()
            self.assertNotIn("blr", {hotel["id"] for hotel in web.catalog()})
            web.ACTIVE.clear()
            web.prepare_agent_requests()
            restored = next(request for request in web.agent_requests() if request["id"] == initial["id"])
            self.assertEqual(restored["proposal"]["id"], initial["proposal"]["id"])
            self.assertTrue(web.review_pdf(restored["id"], restored["proposal"]["id"]).body.startswith(b"%PDF"))
            web.approve(restored["id"], web.Approval(
                proposal_id=restored["proposal"]["id"], acknowledge_simulation=True,
            ))
            hotel = next(hotel for hotel in web.catalog() if hotel["id"] == "blr")
            self.assertEqual(hotel["featured_stock"], 300)
            self.assertEqual(web.run_scenario(inputs)["review_status"], "already approved")

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
