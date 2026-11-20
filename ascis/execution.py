"""Local mock approval, execution and inventory ledger."""

from dataclasses import asdict
from hashlib import sha256
import json
import sqlite3

from ascis.intelligence import Action, Analysis, CompanyPolicy, Position, apply_position, evaluate
from ascis.market import DemandModel, Observation


def proposal(analysis: Analysis, action: Action, company: CompanyPolicy) -> dict:
    candidate = analysis.candidates[action]
    payload = {
        "event": analysis.report["event"],
        "day": analysis.report["day"],
        "report_version": analysis.report["version"],
        "company_policy": company.version,
        "action": action.name,
        "pool": candidate.pool,
        "firm_units": candidate.units,
        "option_units": candidate.optional_units,
        "cost_inr": candidate.cost,
        "maximum_cash_inr": candidate.peak_cash,
        "decision_expires_on_day": analysis.report["day"],
    }
    payload["id"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


class MockExecutor:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS execution (
                id TEXT PRIMARY KEY,
                event TEXT NOT NULL,
                day INTEGER NOT NULL,
                receipt TEXT NOT NULL,
                UNIQUE(event, day)
            )
        """)
        self.connection.commit()

    def commit(
        self, requested: dict, approval_id: str, live: Observation,
        demand: DemandModel, company: CompanyPolicy,
    ) -> dict:
        if approval_id != requested["id"]:
            raise ValueError("Approval must match the exact proposal.")
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        with connection:
            existing = connection.execute(
                "SELECT receipt FROM execution WHERE id = ?", (approval_id,),
            ).fetchone()
            if existing:
                receipt = json.loads(existing[0])
                if receipt["proposal"] != requested:
                    raise ValueError("Approved proposal contents changed.")
                return receipt
            if live.day != requested["decision_expires_on_day"]:
                raise ValueError("The approved quote is outside its decision window.")
            if live.event != requested["event"]:
                raise ValueError("Approval is for another opportunity.")
            prior = connection.execute(
                "SELECT receipt FROM execution WHERE event = ? ORDER BY day DESC LIMIT 1", (live.event,),
            ).fetchone()
            position = Position()
            if prior:
                prior_receipt = json.loads(prior[0])
                if prior_receipt["proposal"]["day"] >= live.day:
                    raise ValueError("Decision window already processed.")
                position = Position(**prior_receipt["position"])
            action = Action[requested["action"]]
            analysis = evaluate(live, position, demand, company, report_version=requested["report_version"])
            if not analysis.mask[action] or proposal(analysis, action, company) != requested:
                raise ValueError("Live terms or eligibility changed. Reassessment required.")
            candidate = analysis.candidates[action]
            if not candidate.units and action != Action.RELEASE_OPTION:
                raise ValueError("Only inventory actions go to the executor.")
            position = apply_position(position, candidate)
            receipt = {
                "type": "mock_inventory_update", "proposal": requested, "position": asdict(position),
            }
            connection.execute(
                "INSERT INTO execution VALUES (?, ?, ?, ?)",
                (approval_id, live.event, live.day, json.dumps(receipt)),
            )
            return receipt
