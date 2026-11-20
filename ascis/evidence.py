"""Provenance and eligibility for public context and demo supplier terms."""

from datetime import datetime, timezone
from hashlib import sha256
import re
from typing import Literal

from ascis.event_evidence import HISTORY


EvidenceState = Literal["current", "missing", "stale", "conflicting"]
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_MENTION = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}} {MONTH} \d{{4}}|{MONTH} \d{{1,2}}, \d{{4}})\b", re.I,
)
ATTENDANCE_MENTION = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{1,8})\s+(?:fans|spectators|attendees|visitors)\b", re.I)


def supplier_evidence(day: int, verified: bool, state: EvidenceState = "current") -> dict:
    if not verified and state == "current":
        state = "missing"
    reasons = {
        "current": None,
        "missing": "Verified supplier terms are missing.",
        "stale": "Supplier terms expired before this decision window.",
        "conflicting": "Supplier terms disagree on the applicable allocation.",
    }
    return {
        "state": state, "eligible": state == "current", "reason": reasons[state],
        "origin": "simulated supplier terms", "decision_day": day,
        "source_day": None if state == "missing" else day - 1 if state == "stale" else day,
    }


def curated_context(event: dict) -> list[dict]:
    facts = []
    if event["url"] and event["date"]:
        facts.append({
            "field": "event date", "value": event["date"], "source_url": event["url"],
            "source_title": event["publisher"], "method": "curated event calendar",
            "use": "event context", "excerpt": None,
        })
    for row in HISTORY.get(event["id"], ()):
        facts.append({
            "field": row["metric"], "value": f"{row['count']:,}", "source_url": row["url"],
            "source_title": row["edition"], "method": "curated published statistic",
            "use": "historical context", "excerpt": None,
        })
    return facts


def evidence_documents(sources: list[dict]) -> list[dict]:
    return [
        {
            "id": sha256(source["url"].encode()).hexdigest()[:16],
            "url": source["url"], "title": source["title"], "text": source["excerpt"][:2400],
        }
        for source in sources if source.get("excerpt")
    ]


def source_fact(document: dict, field: str, value: str, start: int, end: int, method: str) -> dict | None:
    if not 0 <= start < end <= len(document["text"]) or document["text"][start:end] != value:
        return None
    if field == "date mention":
        normalized = normalize_date(value)
        if normalized is None:
            return None
    else:
        normalized = value.replace(",", "")
    return {
        "field": field, "value": value, "source_url": document["url"], "source_title": document["title"],
        "method": method, "use": "unverified source mention",
        "excerpt": document["text"][max(0, start - 60):end + 60],
        "document_id": document["id"], "start": start, "end": end,
        "normalized_value": normalized,
    }


def normalize_date(value: str) -> str | None:
    for pattern in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def local_extract(document: dict) -> list[dict]:
    facts = []
    for field, pattern, group in (
        ("date mention", DATE_MENTION, 0), ("attendance mention", ATTENDANCE_MENTION, 1),
    ):
        for match in pattern.finditer(document["text"]):
            if fact := source_fact(document, field, match[group], *match.span(group), "local text parser"):
                facts.append(fact)
    return sorted(facts, key=lambda fact: fact["start"])


def evidence_summary(event: dict, discovery: dict, facts: list[dict]) -> dict:
    mentions = {
        (fact["document_id"], fact["field"], fact["start"], fact["end"]): fact for fact in facts
    }
    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "discovery_state": "connected" if discovery.get("checked_at") else "unavailable",
        "facts": curated_context(event) + list(mentions.values())[:12],
    }
