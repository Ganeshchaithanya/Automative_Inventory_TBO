import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ascis import evidence_workflow, web
from ascis.evidence import local_extract, source_fact, supplier_evidence
from ascis.intelligence import Action, CompanyPolicy, Position
from ascis.market import bengaluru, fit_demand
from ascis.research import EVENTS


DOCUMENT = {
    "id": "known", "url": "https://example.com/event", "title": "Announcement",
    "text": "The date is 2027-06-24. The date 2027-02-30 is invalid.",
}


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_grounding_requires_exact_span_and_valid_date(self):
        facts = local_extract(DOCUMENT)
        self.assertEqual([fact["value"] for fact in facts], ["2027-06-24"])
        self.assertIsNone(source_fact(DOCUMENT, "date mention", "2027-06-25", 12, 22, "test"))
        self.assertEqual(facts[0]["document_id"], DOCUMENT["id"])

    async def test_mcp_failure_preserves_curated_evidence_without_claiming_connectivity(self):
        with patch.object(evidence_workflow, "request_sources", AsyncMock(side_effect=RuntimeError())):
            response = await evidence_workflow.research_event(EVENTS[0])
        self.assertEqual(response["evidence"]["discovery_state"], "unavailable")
        self.assertTrue(response["evidence"]["facts"])
        self.assertEqual(response["evidence"]["facts"][0]["method"], "curated event calendar")

    async def test_mcp_excerpt_is_extracted_locally_with_source_provenance(self):
        packet = {
            "checked_at": "2026-11-20T18:00:00+00:00",
            "sources": [{"url": DOCUMENT["url"], "title": DOCUMENT["title"], "excerpt": DOCUMENT["text"]}],
        }
        with patch.object(evidence_workflow, "request_sources", AsyncMock(return_value=packet)):
            response = await evidence_workflow.research_event(EVENTS[0])
        self.assertEqual(response["evidence"]["discovery_state"], "connected")
        fact = response["evidence"]["facts"][-1]
        self.assertEqual(fact["value"], "2027-06-24")
        self.assertEqual(fact["method"], "local text parser")
        self.assertEqual(fact["source_url"], DOCUMENT["url"])

    def test_missing_stale_and_conflicting_evidence_block_inventory_changes(self):
        demand, _ = fit_demand()
        company = CompanyPolicy()
        options = bengaluru(volatile=False)
        for state in ("missing", "stale", "conflicting"):
            for observation, position, expected in (
                (options.opening, Position(), Action.REQUEST_TERMS),
                (options.deadline, Position(300, 1_080_000, True), Action.ABSTAIN),
            ):
                evidence = supplier_evidence(observation.day, True, state)
                analysis, action, actor_ran, _ = web.assess_decision(
                    None, demand, company, observation, position, evidence,
                )
                self.assertEqual(action, expected)
                self.assertFalse(actor_ran)
                self.assertFalse(any(analysis.mask[candidate.action] for candidate in analysis.candidates if candidate.units))
                self.assertFalse(analysis.mask[Action.RELEASE_OPTION])

    def test_secondary_mask_rejects_a_disallowed_model_action(self):
        demand, _ = fit_demand()
        observation = bengaluru(volatile=False).deadline
        model = SimpleNamespace(choose=lambda analysis: Action.HOLD_300_PLUS_300)
        evidence = supplier_evidence(observation.day, True)
        _, action, actor_ran, _ = web.assess_decision(model, demand, CompanyPolicy(), observation, Position(), evidence)
        self.assertTrue(actor_ran)
        self.assertEqual(action, Action.ABSTAIN)
