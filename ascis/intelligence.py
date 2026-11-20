"""Supplier terms, versioned skill outputs, counterfactuals and company rules."""

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Literal

import numpy as np

from ascis.market import BASE_STOCK, SELL_PRICE, UNIT_COST, DemandModel, Observation


class Action(IntEnum):
    ABSTAIN = 0
    WAIT = 1
    REJECT = 2
    REQUEST_TERMS = 3
    BUY_300 = 4
    BUY_500 = 5
    HOLD_300_PLUS_300 = 6
    EXERCISE_300 = 7
    RELEASE_OPTION = 8


@dataclass(frozen=True)
class CompanyPolicy:
    version: str = "synthetic-company-v1"
    budget: int = 3_000_000
    downside_limit: int = 600_000
    availability_value: int = 280
    capital_weight: float = 0.025
    downside_penalty: float = 0.45
    unsold_inventory_penalty: int = 0


@dataclass(frozen=True)
class Position:
    units: int = 0
    committed_cash: int = 0
    option_open: bool = False


@dataclass(frozen=True)
class Quote:
    supplier: str
    pool: str
    capacity: int
    unit_cost: int
    terms: str


@dataclass(frozen=True)
class Candidate:
    action: Action
    units: int = 0
    cost: int = 0
    optional_units: int = 0
    pool: str = ""

    @property
    def peak_cash(self) -> int:
        return self.cost + self.optional_units * UNIT_COST


@dataclass(frozen=True)
class Analysis:
    candidates: tuple[Candidate, ...]
    mask: tuple[bool, ...]
    features: tuple[float, ...]
    report: dict


def supplier_quotes(day: int) -> tuple[Quote, ...]:
    firm_price = UNIT_COST if day < -10 else 3_800
    return (
        Quote("Bengaluru Direct", "firm-allocation", 500, firm_price, "room-only, firm"),
        Quote("Regional Wholesale", "firm-allocation", 500, firm_price + 50, "breakfast, firm"),
        Quote("Direct Flex", "flex-allocation", 600, UNIT_COST, "300 firm, 300 optional at T−10"),
    )


def contract_comparison(quotes: tuple[Quote, ...], verified: bool) -> dict:
    sizes: dict[str, set[int]] = {}
    for quote in quotes:
        if quote.pool.strip():
            sizes.setdefault(quote.pool.strip(), set()).add(quote.capacity)
    conflicts = sorted(pool for pool, capacities in sizes.items() if len(capacities) > 1 or min(capacities) <= 0)
    pool_caps = {pool: min(capacities) for pool, capacities in sizes.items() if pool not in conflicts}
    return {
        "skill": "contract_comparison", "version": 2, "quotes": [asdict(q) for q in quotes],
        "pool_capacity": pool_caps, "verified": verified, "conflicting_pools": conflicts,
        "objection": (
            "Resolve conflicting supplier capacity before committing the affected pool." if conflicts
            else None if verified else "Verify optional-allotment expiry and allocation before buying."
        ),
    }


def route_workflow(offer: Literal["hotel", "package"]) -> dict:
    required = ["hotel inventory", "supplier terms"]
    optional = ["transfer inventory", "package eligibility"] if offer == "package" else []
    return {"method": "explicit product selection", "offer": offer, "activated": required + optional}


def candidate_actions(day: int) -> tuple[Candidate, ...]:
    firm = supplier_quotes(day)[0]
    return (
        Candidate(Action.ABSTAIN),
        Candidate(Action.WAIT),
        Candidate(Action.REJECT),
        Candidate(Action.REQUEST_TERMS),
        Candidate(Action.BUY_300, 300, firm.unit_cost * 300, pool=firm.pool),
        Candidate(Action.BUY_500, 500, firm.unit_cost * 500, pool=firm.pool),
        Candidate(Action.HOLD_300_PLUS_300, 300, UNIT_COST * 300, 300, "flex-allocation"),
        Candidate(Action.EXERCISE_300, 300, UNIT_COST * 300, pool="flex-allocation"),
        Candidate(Action.RELEASE_OPTION),
    )


def metrics(draws: np.ndarray, candidate: Candidate, position: Position) -> dict:
    remaining = np.maximum(draws - BASE_STOCK - position.units, 0)
    sold = np.minimum(remaining, candidate.units)
    profit = sold * SELL_PRICE - candidate.cost
    losses = np.maximum(-profit, 0)
    tail_count = max(1, (len(draws) + 9) // 10)
    worst_losses = np.partition(losses, len(losses) - tail_count)[-tail_count:]
    return {
        "expected_profit": float(profit.mean()),
        "downside_cvar90": float(worst_losses.mean()),
        "expected_fulfilled": float(sold.mean()),
        "expected_sell_through_pct": float(100 * sold.mean() / candidate.units) if candidate.units else None,
        "peak_cash": candidate.peak_cash,
        "loss_probability": float(np.mean(profit < 0)),
        "full_sellout_probability": float(np.mean(sold == candidate.units)) if candidate.units else None,
    }


def evaluate(
    observation: Observation,
    position: Position,
    demand: DemandModel,
    company: CompanyPolicy,
    *,
    scenarios: int = 256,
    report_version: int = 1,
    offer: Literal["hotel", "package"] = "hotel",
    evidence_issue: str | None = None,
) -> Analysis:
    draws = demand.scenarios(observation, count=scenarios)
    p10, median, p90 = np.quantile(draws, [0.10, 0.50, 0.90])
    candidates = candidate_actions(observation.day)
    contracts = contract_comparison(supplier_quotes(observation.day), observation.quote_verified)
    assessments = [metrics(draws, candidate, position) for candidate in candidates]
    headroom = company.budget - position.committed_cash
    mask = []
    evaluations = []
    opening = observation.day < -10
    for candidate, assessment in zip(candidates, assessments):
        action = candidate.action
        reasons = []
        if action in (Action.WAIT, Action.REQUEST_TERMS, Action.HOLD_300_PLUS_300) and not opening:
            reasons.append("decision window closed")
        if action in (Action.EXERCISE_300, Action.RELEASE_OPTION) and (opening or not position.option_open):
            reasons.append("no exercisable option")
        if action in (Action.BUY_300, Action.BUY_500) and position.option_open:
            reasons.append("reassess existing option before a separate purchase")
        if (candidate.units or action == Action.RELEASE_OPTION) and not observation.quote_verified:
            reasons.append(evidence_issue or "supplier terms are unverified")
        if candidate.units:
            if candidate.peak_cash > headroom:
                reasons.append("cash limit")
            if assessment["downside_cvar90"] > company.downside_limit:
                reasons.append("downside limit")
            if candidate.pool in contracts["conflicting_pools"]:
                reasons.append("conflicting supplier capacity")
            elif candidate.units + candidate.optional_units > contracts["pool_capacity"].get(candidate.pool, 0):
                reasons.append("allocation capacity")
        mask.append(not reasons)
        evaluations.append({"action": action.name, **assessment, "excluded_by": reasons})

    forecast_skill = {
        "skill": "forecast_uncertainty", "version": 1,
        "range_width": round(float(p90 - p10), 1),
        "calibrated": demand.calibration is not None,
        "stress_regime": observation.volatile,
        "objection": "Demand is sensitive to the cancellation stress branch." if observation.volatile else None,
    }
    capital_skill = {
        "skill": "capital_preservation", "version": 1, "policy_version": company.version,
        "headroom": headroom,
        "eligible_commitments": [
            c.action.name for c, allowed in zip(candidates, mask) if allowed and c.units
        ],
    }
    growth_skill = {
        "skill": "availability_alignment", "version": 1,
        "scenario_service_value": {
            item["action"]: round(item["expected_fulfilled"] * company.availability_value)
            for item in evaluations
        },
    }
    features = (
        observation.signal, float(observation.volatile), float(observation.quote_verified),
        float(opening), float(position.option_open), position.units / 600,
        headroom / 3_000_000, float(median) / 1_500, float(p90 - p10) / 1_000,
        company.downside_limit / 1_000_000, company.availability_value / 1_000,
        company.capital_weight * 10,
        *(x["expected_profit"] / 1_000_000 for x in evaluations),
        *(x["downside_cvar90"] / 1_000_000 for x in evaluations),
    )
    return Analysis(candidates, tuple(mask), features, {
        "event": observation.event, "day": observation.day, "version": report_version,
        "workflow_router": route_workflow(offer),
        "existing_room_nights": BASE_STOCK + position.units,
        "forecast_room_nights": {"p10": round(float(p10)), "p50": round(float(median)), "p90": round(float(p90))},
        "domain_skills": [contracts, forecast_skill],
        "company_skills": [capital_skill, growth_skill],
        "counterfactuals": evaluations,
    })


def apply_position(position: Position, candidate: Candidate) -> Position:
    action = candidate.action
    option_open = position.option_open
    if action == Action.HOLD_300_PLUS_300:
        option_open = True
    elif action in (Action.EXERCISE_300, Action.RELEASE_OPTION):
        option_open = False
    return Position(position.units + candidate.units, position.committed_cash + candidate.cost, option_open)


def settlement(position: Position, demand: int) -> dict:
    fulfilled = min(position.units, max(0, demand - BASE_STOCK))
    return {
        "realized_demand": demand,
        "incremental_sale_nights": fulfilled,
        "procured_room_nights": position.units,
        "unsold_procured_nights": position.units - fulfilled,
        "unmet_room_nights": max(0, demand - BASE_STOCK - position.units),
        "realized_contribution": fulfilled * SELL_PRICE - position.committed_cash,
    }
