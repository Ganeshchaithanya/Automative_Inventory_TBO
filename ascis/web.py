"""Separate local storefront and operations sites sharing a simulated ledger."""

import asyncio
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, replace
from datetime import date
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock
from time import time
from typing import Literal
from uuid import UUID, uuid4

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ascis.catalog import HOTELS, TRANSFER_PRICE, consume_booking, demo_quote, listings, promotion_seconds
from ascis.event_evidence import HISTORY, PROPERTIES, attendance_adjustment, hotel_search
from ascis.evidence import EvidenceState, curated_context, supplier_evidence
from ascis.evidence_workflow import research_event
from ascis.execution import MockExecutor, proposal
from ascis.intelligence import Action, Analysis, Position, apply_position, candidate_actions, evaluate
from ascis.live_hotels import available_hotels
from ascis.market import SELL_PRICE, UNIT_COST, Observation, bengaluru, trigger
from ascis.policy import load
from ascis.research import EVENTS
from ascis.skills import load_skills


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "website"
LEDGER = Path(os.getenv("ASCIS_LEDGER", str(ROOT / "runs" / "web-ledger.sqlite")))
OPTION_REVIEW_SECONDS = 20
VOLATILE_MARKETS = {"expo"}


class Approval(BaseModel):
    proposal_id: str
    acknowledge_simulation: bool


class BookingItem(BaseModel):
    hotel_id: str
    offer: Literal["hotel", "package"]
    check_in: date
    check_out: date
    rooms: int = Field(ge=1)
    origin: str = Field(default="", max_length=100)


class Booking(BookingItem):
    request_id: UUID
    demo_revision: int = Field(default=0, ge=0)


class BookingCart(BaseModel):
    request_id: UUID
    demo_revision: int = Field(default=0, ge=0)
    items: list[BookingItem] = Field(min_length=1)


class ResetDemo(BaseModel):
    confirm: Literal["reset-demo"]
    revision: int = Field(ge=0)


class Scenario(BaseModel):
    event_id: str
    phase: Literal["opening", "deadline"]
    offer: Literal["hotel", "package"] = "hotel"
    signal: float = Field(ge=-0.2, le=1.15, allow_inf_nan=False)
    volatile: bool
    quote_verified: bool
    evidence_state: EvidenceState = "current"
    budget: int = Field(ge=0, le=3_000_000)
    downside_limit: int = Field(ge=0, le=600_000)
    queue_for_review: bool = False


@dataclass
class DecisionRun:
    id: str
    event: dict
    observation: Observation
    position: Position
    analysis: Analysis
    action: Action
    proposal: dict | None
    status: str
    trace: list[dict]


def prepare_ledger():
    LEDGER.parent.mkdir(exist_ok=True, parents=True)
    with closing(sqlite3.connect(LEDGER)) as connection:
        MockExecutor(connection)
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS demo_session (id INTEGER PRIMARY KEY CHECK (id = 1), "
                "revision INTEGER NOT NULL)"
            )
            connection.execute("INSERT OR IGNORE INTO demo_session VALUES (1, 0)")
            connection.execute("CREATE TABLE IF NOT EXISTS demo_event (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS agent_decline (id TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE IF NOT EXISTS scenario_review (id TEXT PRIMARY KEY, inputs TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS demo_booking "
                "(id TEXT PRIMARY KEY, request TEXT NOT NULL, receipt TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS demo_booking_cart "
                "(id TEXT PRIMARY KEY, request TEXT NOT NULL, receipt TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS option_review "
                "(event TEXT PRIMARY KEY, due_at REAL NOT NULL, outcome TEXT)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO option_review (event, due_at) "
                "SELECT event, ? FROM execution WHERE day = -60 AND event LIKE 'agent-opening-%' "
                "AND json_extract(receipt, '$.position.option_open') = 1",
                (time() + OPTION_REVIEW_SECONDS,),
            )
            connection.execute(
                "UPDATE option_review SET outcome = ? WHERE outcome IS NULL "
                "AND event IN (SELECT event FROM execution WHERE day = -10)",
                ("Option decision executed. The catalog reflects the approved allocation.",),
            )


@asynccontextmanager
async def start_server(app: FastAPI):
    checkpoint = Path(os.getenv("ASCIS_CHECKPOINT", str(ROOT / "runs" / "policy.pt")))
    if not checkpoint.is_file():
        raise RuntimeError("Run python -m ascis train before starting the website.")
    app.state.checkpoint = load(checkpoint)
    skills_path = Path(os.getenv("ASCIS_SKILLS_CHECKPOINT", str(ROOT / "runs" / "dual-skills.pt")))
    app.state.skills = load_skills(skills_path, checkpoint)
    prepare_ledger()
    prepare_agent_requests()
    app.state.market_evidence = {}
    app.state.live_hotels = {}
    async with anyio.create_task_group() as jobs:
        jobs.start_soon(collect_market_evidence, app)
        jobs.start_soon(watch_deadlines)
        yield
        jobs.cancel_scope.cancel()


app = FastAPI(title="ASCIS Demo", lifespan=start_server, docs_url=None, redoc_url=None, openapi_url=None)
storefront = FastAPI(title="Avoflare Stays", docs_url=None, redoc_url=None, openapi_url=None)
allowed_hosts = ["127.0.0.1", "localhost", "testserver"]
if hosted_name := os.getenv("RENDER_EXTERNAL_HOSTNAME"):
    allowed_hosts.append(hosted_name)
for site in (app, storefront):
    site.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    site.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
ACTIVE: dict[str, DecisionRun] = {}
LOCK = Lock()


@storefront.get("/")
def index():
    return FileResponse(ASSETS / "index.html")


@app.get("/approvals")
def approvals_page():
    return FileResponse(ASSETS / "approvals.html")


@app.get("/impact")
@app.get("/")
def impact_page():
    return FileResponse(ASSETS / "impact.html")


@app.get("/scenarios")
def scenarios_page():
    return FileResponse(ASSETS / "scenarios.html")


@app.get("/api/scenarios")
def scenario_inputs():
    _, _, company, _ = app.state.checkpoint
    declined = read_declined_markets()
    return {
        "events": [{
            "id": event["id"], "title": event["title"], "destination": event["destination"],
            "signal": round(event["signal"] + attendance_adjustment(event["id"]), 2),
        } for event in EVENTS if event["id"] not in declined],
        "budget": company.budget, "downside_limit": company.downside_limit,
    }


@app.post("/api/scenarios")
def run_scenario(body: Scenario):
    event = next((event for event in EVENTS if event["id"] == body.event_id), None)
    if not event:
        raise HTTPException(404, "Unknown event market.")
    if body.event_id in read_declined_markets():
        raise HTTPException(409, "This event allocation was declined and is no longer an active scenario.")
    model, demand, configured, training = app.state.checkpoint
    company = scenario_policy(configured, body)
    at_deadline = body.phase == "deadline"
    day = -10 if at_deadline else -60
    position = Position()
    if at_deadline:
        position = apply_position(position, candidate_actions(-60)[Action.HOLD_300_PLUS_300])
    observation = scenario_observation(body, event["title"], day)
    evidence = supplier_evidence(day, observation.quote_verified, body.evidence_state)
    analysis, action, actor_ran, market_trigger = assess_decision(
        model, demand, company, observation, position, evidence, offer=body.offer,
    )
    assessment = analysis.report
    skills = getattr(app.state, "skills", None)
    validation_group = body.phase + ("_volatile" if body.volatile else "_stable")
    validation = {
        name: result["calibrated"][validation_group]
        for name, result in training.get("forecast_evaluation", {}).items()
    }
    review_status = None
    if body.queue_for_review and not at_deadline and analysis.candidates[action].units:
        review_status = queue_scenario(body, event, observation, analysis, action, company)
    action_weights = model.action_weights(analysis) if actor_ran and hasattr(model, "action_weights") else {}
    return {
        "event_id": event["id"], "event": event["title"], "market": event["destination"],
        "phase": body.phase, "trigger": market_trigger,
        "signal": body.signal, "volatile": body.volatile, "quote_verified": evidence["eligible"],
        "supplier_evidence": evidence,
        "existing_room_nights": assessment["existing_room_nights"],
        "option_open": position.option_open, "cash_headroom": company.budget - position.committed_cash,
        "budget": company.budget, "downside_limit": company.downside_limit,
        "action_weights": action_weights,
        "forecast": assessment["forecast_room_nights"], "forecast_skill": assessment["domain_skills"][1],
        "selected_action": action.name, "actor_ran": actor_ran,
        "candidates": assessment["counterfactuals"], "fit": training.get("fit", {}),
        "forecast_validation": validation,
        "workflow": assessment["workflow_router"],
        "skill_reviews": skills.assess(observation, analysis) if skills else [],
        "review_status": review_status,
    }


@app.get("/storefront")
def storefront_link():
    if storefront_url := os.getenv("ASCIS_STOREFRONT_URL"):
        return RedirectResponse(storefront_url)
    if os.getenv("RENDER_EXTERNAL_HOSTNAME") == "automative-inventory-tbo.onrender.com":
        return RedirectResponse("https://af-stays.vercel.app/")
    public_port = int(os.getenv("ASCIS_PORT", "8080"))
    return RedirectResponse(f"http://127.0.0.1:{public_port}/")


async def collect_market_evidence(app: FastAPI):
    """Refresh real event context internally. Procurement forecasts use simulation inputs."""
    async def collect(event: dict):
        app.state.market_evidence[event["id"]] = await research_event(event)
        hotel = next(item for item in listings({}) if item["id"] == event["id"])
        app.state.live_hotels[event["id"]] = await anyio.to_thread.run_sync(
            available_hotels, event["id"], hotel["available_from"], hotel["available_until"],
        )

    async with anyio.create_task_group() as jobs:
        for event in EVENTS:
            if event["id"] in HOTELS and event["url"]:
                jobs.start_soon(collect, event)


async def watch_deadlines():
    while True:
        await anyio.to_thread.run_sync(review_due_options)
        await anyio.sleep(2)


def assess_decision(
    model, demand, company, observation, position, evidence, *, report_version=1, offer="hotel",
):
    effective = replace(observation, quote_verified=evidence["eligible"])
    market_trigger = trigger(effective)
    analysis = evaluate(effective, position, demand, company, report_version=report_version,
                        offer=offer, evidence_issue=evidence["reason"])
    triggered = market_trigger["active"] or effective.day == -10
    actor_ran = triggered and evidence["eligible"]
    action = Action.ABSTAIN
    if actor_ran:
        proposed = model.choose(analysis)
        if analysis.mask[proposed]:
            action = proposed
    elif triggered and effective.day < -10 and not evidence["eligible"]:
        action = Action.REQUEST_TERMS
    analysis.report["supplier_evidence"] = evidence
    return analysis, action, actor_ran, market_trigger


def decision(
    run_id: str, event: dict, observation: Observation, position: Position, version: int, trace: list[dict],
    *, scenario: Scenario | None = None,
):
    model, demand, company, _ = app.state.checkpoint
    if scenario:
        company = scenario_policy(company, scenario)
    evidence = supplier_evidence(observation.day, observation.quote_verified, scenario.evidence_state if scenario else "current")
    analysis, action, _, _ = assess_decision(
        model, demand, company, observation, position, evidence, report_version=version,
        offer=scenario.offer if scenario else "hotel",
    )
    skills = getattr(app.state, "skills", None)
    if skills:
        analysis.report["skill_reviews"] = skills.assess(observation, analysis)
    candidate = analysis.candidates[action]
    requested = proposal(analysis, action, company) if candidate.units or action == Action.RELEASE_OPTION else None
    status = "awaiting approval" if requested else "no commitment"
    trace = [*trace, {"day": observation.day, "action": action.name, "status": status}]
    return DecisionRun(run_id, event, observation, position, analysis, action, requested, status, trace)


def active(run_id: str) -> DecisionRun:
    try:
        return ACTIVE[run_id]
    except KeyError:
        raise HTTPException(404, "This decision session is no longer active. Run a new scenario.") from None


def demo_revision(connection: sqlite3.Connection) -> int:
    return connection.execute("SELECT revision FROM demo_session WHERE id = 1").fetchone()[0]


def opening_id(event_id: str, revision: int) -> str:
    base = "agent-opening-" + event_id
    return f"{base}-r{revision}" if revision else base


def declined_markets(connection: sqlite3.Connection) -> set[str]:
    revision = demo_revision(connection)
    declines = {row[0] for row in connection.execute("SELECT id FROM agent_decline")}
    return {event_id for event_id in HOTELS if opening_id(event_id, revision) in declines}


def read_declined_markets() -> set[str]:
    if not LEDGER.is_file():
        return set()
    with closing(sqlite3.connect(LEDGER)) as connection:
        connection.execute("BEGIN")
        return declined_markets(connection)


def scenario_policy(company, body: Scenario):
    return replace(company, budget=min(company.budget, body.budget),
                   downside_limit=min(company.downside_limit, body.downside_limit))


def scenario_observation(body: Scenario, identifier: str, day: int) -> Observation:
    return replace(
        bengaluru(volatile=body.volatile).opening, event=identifier, day=day, signal=body.signal,
        searches=int(250 + 280 * max(0, body.signal)), volatile=body.volatile, quote_verified=body.quote_verified,
    )


def queue_scenario(body: Scenario, event: dict, observation: Observation, analysis: Analysis, action: Action, company):
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        identifier = opening_id(event["id"], demo_revision(connection))
        if connection.execute("SELECT 1 FROM agent_decline WHERE id = ?", (identifier,)).fetchone():
            return "declined"
        if connection.execute("SELECT 1 FROM execution WHERE event = ? AND day = -60", (identifier,)).fetchone():
            return "already approved"
        if identifier in ACTIVE:
            return "existing request"
        observation = replace(observation, event=identifier)
        report = {**analysis.report, "event": identifier}
        analysis = replace(analysis, report=report)
        requested = proposal(analysis, action, company)
        connection.execute("INSERT OR IGNORE INTO demo_event VALUES (?, ?)", (identifier, event["title"]))
        connection.execute("INSERT INTO scenario_review VALUES (?, ?)", (identifier, body.model_dump_json()))
        run = DecisionRun(identifier, event, observation, Position(), analysis, action, requested, "awaiting approval", [
            {"day": observation.day, "action": action.name, "status": "awaiting approval"},
        ])
        connection.commit()
        ACTIVE[identifier] = run
        return "queued"


def opening_runs(connection: sqlite3.Connection, skip: set[str]) -> dict[str, DecisionRun]:
    revision = demo_revision(connection)
    runs = {}
    saved = dict(connection.execute("SELECT id, inputs FROM scenario_review"))
    for event in EVENTS:
        if event["id"] not in HOTELS:
            continue
        identifier = opening_id(event["id"], revision)
        if identifier in skip:
            continue
        if identifier in saved:
            body = Scenario.model_validate_json(saved[identifier])
            observation = scenario_observation(body, identifier, -60)
            run = decision(identifier, event, observation, Position(), 1, [], scenario=body)
        elif event["url"]:
            signal = event["signal"] + attendance_adjustment(event["id"])
            observation = replace(
                bengaluru(volatile=False).opening, event=identifier, signal=signal,
                searches=int(250 + 280 * signal), quote_verified=True,
            )
            run = decision(identifier, event, observation, Position(), 1, [])
        else:
            continue
        if run.proposal:
            connection.execute("INSERT OR IGNORE INTO demo_event VALUES (?, ?)", (identifier, event["title"]))
            runs[identifier] = run
    return runs


def prepare_agent_requests():
    """Run the internal policy and queue only proposals that need a human."""
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection:
        with connection:
            resolved = {
                row[0] for row in connection.execute(
                    "SELECT id FROM agent_decline UNION SELECT event FROM execution WHERE day = -60"
                )
            }
            runs = opening_runs(connection, resolved | set(ACTIVE))
        ACTIVE.update(runs)


@app.get("/api/demo/status")
def demo_status():
    with closing(sqlite3.connect(LEDGER)) as connection:
        revision = demo_revision(connection)
    results = getattr(app.state, "market_evidence", {})
    markets = [
        {
            "destination": event["destination"],
            "state": "connected" if result.get("checked_at") else ("unavailable" if result else "checking"),
            "providers": result.get("providers", {}),
            "sources": len(result.get("sources", [])),
        }
        for event in EVENTS if event["id"] in HOTELS and event["url"]
        for result in [results.get(event["id"], {})]
    ]
    return {
        "revision": revision, "transport": "in-process", "tool": "search_public_events",
        "markets": markets,
        "live_hotel_provider_configured": bool(os.getenv("BOOKING_DEMAND_TOKEN") and os.getenv("BOOKING_AFFILIATE_ID")),
    }


@app.post("/api/demo/reset")
def reset_demo(body: ResetDemo):
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if body.revision != demo_revision(connection):
                raise HTTPException(409, "Another tab already reset the demo.")
            connection.execute("DELETE FROM demo_booking")
            connection.execute("DELETE FROM demo_booking_cart")
            connection.execute("DELETE FROM option_review")
            connection.execute("DELETE FROM agent_decline")
            connection.execute("DELETE FROM scenario_review")
            connection.execute("DELETE FROM execution")
            connection.execute("DELETE FROM demo_event")
            connection.execute("UPDATE demo_session SET revision = revision + 1 WHERE id = 1")
            new_revision = demo_revision(connection)
            fresh_runs = opening_runs(connection, set())
        ACTIVE.clear()
        ACTIVE.update(fresh_runs)
    return {"revision": new_revision, "pending_proposals": len(fresh_runs)}


def review_due_options(now: float | None = None):
    """Replay the option deadline using the recorded opening position."""
    cutoff = time() if now is None else now
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection, connection:
        markets = {opening_id(event["id"], demo_revision(connection)): event for event in EVENTS}
        due = connection.execute(
            "SELECT option_review.event, execution.receipt, scenario_review.inputs FROM option_review "
            "JOIN execution ON execution.event = option_review.event AND execution.day = -60 "
            "LEFT JOIN scenario_review ON scenario_review.id = option_review.event "
            "WHERE option_review.outcome IS NULL AND due_at <= ?", (cutoff,),
        ).fetchall()
        for identifier, receipt, inputs in due:
            if identifier in ACTIVE and ACTIVE[identifier].observation.day == -10:
                continue
            event = markets.get(identifier)
            if event is None:
                continue
            position = Position(**json.loads(receipt)["position"])
            if not position.option_open:
                continue
            scenario = Scenario.model_validate_json(inputs) if inputs else None
            volatile = scenario.volatile if scenario else event["id"] in VOLATILE_MARKETS
            opening_signal = scenario.signal if scenario else event["signal"]
            signal = min(0.35, opening_signal) if volatile else min(1.15, opening_signal + 0.12)
            observation = replace(
                bengaluru(volatile=volatile).deadline, event=identifier, signal=signal,
                searches=int(250 + 280 * signal), quote_verified=True,
            )
            trace = ACTIVE[identifier].trace if identifier in ACTIVE else []
            run = decision(identifier, event, observation, position, 2, trace, scenario=scenario)
            if not run.proposal:
                run.position = replace(position, option_open=False)
                connection.execute(
                    "UPDATE option_review SET outcome = ? WHERE event = ?",
                    ("Option lapsed. The agent selected no new commitment.", identifier),
                )
            ACTIVE[identifier] = run


@app.get("/api/agent/requests")
def agent_requests():
    with LOCK:
        return [
            {
                "id": run.id, "title": HOTELS[run.event["id"]]["name"], "destination": run.event["destination"],
                "event_title": run.event["title"], "proposal": run.proposal,
                "stage": "Option deadline" if run.observation.day == -10 else "Opening allocation",
                "reason": "The policy selected this allocation after reviewing simulated demand, supplier terms and downside exposure.",
                "supplier_evidence": run.analysis.report["supplier_evidence"],
            }
            for run in ACTIVE.values() if run.id.startswith("agent-opening-") and run.status == "awaiting approval"
        ]


@app.post("/api/agent/requests/{run_id}/decline")
def decline_agent_request(run_id: str, approval: Approval):
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection, connection:
        run = active(run_id)
        if not run_id.startswith("agent-opening-") or run.status != "awaiting approval":
            raise HTTPException(409, "This request has already been resolved.")
        if not run.proposal or run.proposal["id"] != approval.proposal_id:
            raise HTTPException(409, "This request has changed.")
        if run.observation.day == -10:
            connection.execute(
                "UPDATE option_review SET outcome = ? WHERE event = ?",
                ("Optional allocation declined. The firm allocation remains in the catalog.", run_id),
            )
        else:
            connection.execute("INSERT INTO agent_decline VALUES (?)", (run_id,))
        run.status = "declined"
    return {"status": "declined"}


@app.post("/api/agent/requests/{run_id}/approve")
def approve(run_id: str, approval: Approval):
    if not approval.acknowledge_simulation:
        raise HTTPException(422, "Acknowledge the simulated inventory update to continue.")
    with LOCK:
        run = active(run_id)
        if run.status == "declined":
            raise HTTPException(409, "This request was declined.")
        if not run.proposal or approval.proposal_id != run.proposal["id"]:
            raise HTTPException(409, "The selected proposal changed. Review the current proposal.")
        _, demand, company, _ = app.state.checkpoint
        with closing(sqlite3.connect(LEDGER)) as connection:
            try:
                receipt = MockExecutor(connection).commit(
                    run.proposal, approval.proposal_id, run.observation, demand, company,
                )
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
            with connection:
                if run.observation.day == -10:
                    outcome = (
                        "Option released. Firm room-nights remain in the catalog."
                        if run.action == Action.RELEASE_OPTION
                        else "Option exercised. Additional room-nights were added to the catalog."
                    )
                    connection.execute(
                        "UPDATE option_review SET outcome = ? WHERE event = ?",
                        (outcome, run_id),
                    )
                elif receipt["position"]["option_open"]:
                    connection.execute(
                        "INSERT OR IGNORE INTO option_review (event, due_at) VALUES (?, ?)",
                        (run_id, time() + OPTION_REVIEW_SECONDS),
                    )
        run.position = Position(**receipt["position"])
        run.status = "added to demo inventory" if run.proposal["firm_units"] else "demo option released"
        run.trace[-1]["status"] = run.status
        return {"status": run.status, "firm_added": run.proposal["firm_units"]}


@app.get("/api/agent/activity")
def agent_activity():
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection:
        rows = connection.execute(
            "SELECT demo_event.title, option_review.event, outcome FROM option_review "
            "JOIN demo_event ON demo_event.id = option_review.event ORDER BY due_at DESC"
        ).fetchall()
        declines = connection.execute(
            "SELECT demo_event.title FROM agent_decline JOIN demo_event ON demo_event.id = agent_decline.id "
            "ORDER BY agent_decline.rowid DESC"
        ).fetchall()
        return [
            {"event": title, "status": "Opening allocation declined. Event promotion withdrawn."}
            for (title,) in declines
        ] + [
            {"event": title, "status": outcome or (
                "Option decision awaits approval."
                if identifier in ACTIVE and ACTIVE[identifier].observation.day == -10
                else "Firm stock published. Agent is monitoring the optional allocation."
            )}
            for title, identifier, outcome in rows
        ]


@app.get("/api/agent/requests/{run_id}/review.pdf")
def review_pdf(run_id: str, proposal_id: str):
    from ascis.review_pdf import procurement_review

    with LOCK:
        run = active(run_id)
        if not run.proposal or run.proposal["id"] != proposal_id:
            raise HTTPException(409, "This proposal changed. Download the current review.")
        assessment = next(
            row for row in run.analysis.report["counterfactuals"] if row["action"] == run.action.name
        )
        content = procurement_review(
            event=run.event, hotel=HOTELS[run.event["id"]]["name"], proposal=run.proposal,
            forecast=run.analysis.report["forecast_room_nights"], assessment=assessment, status=run.status,
        )
    return Response(
        content, media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="ASCIS-Review-{proposal_id[:12]}.pdf"',
            "Cache-Control": "no-store",
        },
    )


def approved_allocations(connection: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    rows = connection.execute(
        "SELECT demo_event.title, execution.receipt FROM execution "
        "JOIN demo_event ON execution.event = demo_event.id ORDER BY execution.rowid DESC"
    ).fetchall()
    approved: dict[str, tuple[int, int]] = {}
    for title, receipt in rows:
        proposal = json.loads(receipt)["proposal"]
        units, cost = approved.get(title, (0, 0))
        approved[title] = units + proposal["firm_units"], cost + proposal["cost_inr"]
    return approved


def approved_stock(connection: sqlite3.Connection) -> dict[str, int]:
    return {title: units for title, (units, _) in approved_allocations(connection).items()}


def booking_receipts(connection: sqlite3.Connection) -> list[dict]:
    return [json.loads(row[0]) for row in connection.execute("SELECT receipt FROM demo_booking ORDER BY rowid DESC")]


@storefront.get("/api/catalog")
def catalog(response: Response = None):
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection:
        connection.execute("BEGIN")
        revision = demo_revision(connection)
        if response is not None:
            response.headers["X-ASCIS-Demo-Revision"] = str(revision)
        declined = declined_markets(connection)
        allocations = approved_allocations(connection)
        bookings = booking_receipts(connection)
        recovered: dict[str, int] = {}
        for booking in bookings:
            acquired_nights = sum(booking.get("acquired_nights", {}).values())
            recovered[booking["hotel_id"]] = recovered.get(booking["hotel_id"], 0) + acquired_nights * SELL_PRICE
        properties = listings({title: units for title, (units, _) in allocations.items()}, bookings)
        for hotel in properties:
            market = hotel["id"]
            event = next(item for item in EVENTS if item["id"] == market)
            run = ACTIVE.get(opening_id(market, revision))
            signal = run.observation.signal if run else event["signal"] + attendance_adjustment(market)
            _, cost = allocations.get(hotel["event"], (0, 0))
            hotel["event_active"] = market not in declined
            hotel["promotion_seconds"] = (
                promotion_seconds(signal, cost, recovered.get(market, 0))
                if hotel["event_active"] and hotel["stock"] else None
            )
        return properties


@storefront.post("/api/bookings")
def book(body: Booking):
    request_id = str(body.request_id)
    request = body.model_dump_json()
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        if body.demo_revision != demo_revision(connection):
            raise HTTPException(409, "The demo was reset. Reopen the stay before booking.")
        existing = connection.execute(
            "SELECT request, receipt FROM demo_booking WHERE id = ?", (request_id,),
        ).fetchone()
        if existing:
            if Booking.model_validate_json(existing[0]) != body:
                raise HTTPException(409, "This booking request was already used.")
            receipt = json.loads(existing[1])
            receipt.pop("cost_inr")
            return receipt
        hotel = next((item for item in listings(approved_stock(connection), booking_receipts(connection))
                      if item["id"] == body.hotel_id), None)
        if hotel is None:
            raise HTTPException(404, "Hotel not found.")
        receipt = booking_details(hotel, body)
        connection.execute(
            "INSERT INTO demo_booking VALUES (?, ?, ?)", (request_id, request, json.dumps(receipt)),
        )
        receipt.pop("cost_inr")
        return receipt


def booking_details(hotel: dict, item: BookingItem) -> dict:
    try:
        details = demo_quote(hotel, item.offer, item.check_in, item.check_out, item.rooms)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {**details, "reference": "DEMO-" + uuid4().hex[:8].upper(), "origin": item.origin.strip()}


@storefront.post("/api/booking-carts")
def book_cart(body: BookingCart):
    request_id = str(body.request_id)
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        if body.demo_revision != demo_revision(connection):
            raise HTTPException(409, "The demo was reset. Rebuild your cart with the current inventory.")
        previous = connection.execute(
            "SELECT request, receipt FROM demo_booking_cart WHERE id = ?", (request_id,),
        ).fetchone()
        if previous:
            if BookingCart.model_validate_json(previous[0]) != body:
                raise HTTPException(409, "This cart request was already used.")
            return json.loads(previous[1])
        hotels = {hotel["id"]: hotel for hotel in listings(approved_stock(connection), booking_receipts(connection))}
        bookings = []
        for index, item in enumerate(body.items, start=1):
            hotel = hotels.get(item.hotel_id)
            if not hotel:
                raise HTTPException(409, f"Cart item {index}: this property is no longer available.")
            try:
                receipt = booking_details(hotel, item)
            except HTTPException as error:
                raise HTTPException(409, f"Cart item {index} ({hotel['name']}): {error.detail}") from error
            consume_booking(hotel["rooms_by_date"], hotel["acquired_by_date"], hotel["transfers_by_date"], receipt)
            line = Booking(**item.model_dump(), request_id=uuid4(), demo_revision=body.demo_revision)
            connection.execute(
                "INSERT INTO demo_booking VALUES (?, ?, ?)",
                (str(line.request_id), line.model_dump_json(), json.dumps(receipt)),
            )
            receipt.pop("cost_inr")
            bookings.append(receipt)
        confirmation = {
            "reference": "CART-" + body.request_id.hex[:12].upper(),
            "bookings": bookings,
            "subtotal_inr": sum(booking["subtotal_inr"] for booking in bookings),
        }
        connection.execute(
            "INSERT INTO demo_booking_cart VALUES (?, ?, ?)",
            (request_id, body.model_dump_json(), json.dumps(confirmation)),
        )
        return confirmation


@app.get("/api/impact")
def impact():
    _, _, company, _ = app.state.checkpoint
    with LOCK, closing(sqlite3.connect(LEDGER)) as connection:
        connection.execute("BEGIN")
        revision = demo_revision(connection)
        allocations, bookings = approved_allocations(connection), booking_receipts(connection)
        approved = {title: units for title, (units, _) in allocations.items()}
        declined = declined_markets(connection)
        rows = []
        for hotel in listings(approved, bookings):
            sales = [booking for booking in bookings if booking["hotel_id"] == hotel["id"]]
            event = next(item for item in EVENTS if item["id"] == hotel["id"])
            added = approved.get(hotel["event"], 0)
            sold = sum(booking["room_nights"] for booking in sales)
            revenue = sum(booking["subtotal_inr"] for booking in sales)
            realized_cost = sum(booking["cost_inr"] for booking in sales)
            event_active = hotel["id"] not in declined
            active = ACTIVE.get(opening_id(hotel["id"], revision)) if event_active else None
            evidence = getattr(app.state, "market_evidence", {}).get(hotel["id"], {})
            rows.append({
                "id": hotel["id"], "hotel": hotel["name"], "destination": hotel["destination"], "event": hotel["event"],
                "event_active": event_active,
                "available": hotel["stock"], "added": added, "sold": sold,
                "revenue": revenue, "gross_margin_on_sold": revenue - realized_cost,
                "committed_cost": allocations.get(hotel["event"], (0, 0))[1],
                "unit_cost": UNIT_COST, "night_rate": hotel["price"],
                "transfer_rate": TRANSFER_PRICE,
                "policy_action": active.action.name if active else None,
                "policy_stage": active.observation.day if active else None,
                "volatile": active.observation.volatile if active else False,
                "budget": company.budget, "downside_limit": company.downside_limit,
                "forecast": active.analysis.report["forecast_room_nights"] if active else None,
                "forecast_calibrated": bool(active and active.analysis.report["domain_skills"][1]["calibrated"]),
                "evidence_count": len(evidence.get("sources", [])),
                "evidence": evidence.get("evidence") or {
                    "discovery_state": "checking", "as_of": None, "facts": curated_context(event),
                },
                "event_date": event["date"], "event_source": event["url"],
                "history": HISTORY.get(hotel["id"], ()), "real_hotels": PROPERTIES.get(hotel["id"], ()),
                "live_hotels": getattr(app.state, "live_hotels", {}).get(hotel["id"], {
                    "status": "Checking event sources and live provider configuration.", "offers": [],
                }),
                "hotel_search": hotel_search(event, hotel["available_from"], hotel["available_until"]),
                "check_in": hotel["available_from"], "check_out": hotel["available_until"],
                "signal_prior": attendance_adjustment(hotel["id"]),
                "featured_stock": hotel["featured_stock"],
                "featured_sold": sum(sum(booking.get("acquired_nights", {}).values()) for booking in sales),
                "candidate_assessment": next(
                    (row for row in active.analysis.report["counterfactuals"] if row["action"] == active.action.name),
                    None,
                ) if active else None,
            })
        return {"properties": rows, "bookings": bookings[:10]}


async def serve():
    import uvicorn

    prepare_ledger()
    servers = [
        uvicorn.Server(uvicorn.Config(site, host="127.0.0.1", port=port))
        for site, port in ((storefront, int(os.getenv("ASCIS_PORT", "8080"))),
                           (app, int(os.getenv("ASCIS_OPS_PORT", "8083"))))
    ]
    tasks = [asyncio.create_task(server.serve()) for server in servers]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for server in servers:
        server.should_exit = True
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(serve())
