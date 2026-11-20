"""Read-only event evidence provider. Commercial inputs stay in simulation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
import os
import re
from urllib.parse import quote, quote_plus, urlsplit

import httpx


EVENTS = (
    {"id": "icc", "title": "India vs Pakistan · Cricket World Cup", "destination": "Johannesburg, South Africa",
     "date": "2027-10-10", "category": "Cricket", "signal": 0.83, "query": "ICC Cricket World Cup 2027 India Pakistan Johannesburg",
     "publisher": "International Cricket Council",
     "url": "https://www.icc-cricket.com/news/icc-men-s-cricket-world-cup-2027-schedule-revealed-and-ballot-system-opens"},
    {"id": "fifa", "title": "Women’s World Cup · Opening match", "destination": "Rio de Janeiro, Brazil",
     "date": "2027-06-24", "category": "Football", "signal": 0.88, "query": "FIFA Women's World Cup 2027 Rio de Janeiro opening match",
     "publisher": "FIFA",
     "url": "https://inside.fifa.com/organisation/president/news/womens-world-cup-brazil-2027-gianni-infantino-vai-ser-epico"},
    {"id": "expo", "title": "Expo 2027 · Opening week", "destination": "Belgrade, Serbia",
     "date": "2027-05-15", "category": "Exhibition", "signal": 0.67, "query": "Expo 2027 Belgrade 15 May",
     "publisher": "Expo 2027 Belgrade",
     "url": "https://expobelgrade2027.org/en/experience/programme"},
    {"id": "blr", "title": "Bengaluru cricket final", "destination": "Bengaluru, India",
     "date": None, "category": "Synthetic replay", "signal": 0.78, "query": "Bengaluru cricket travel",
     "publisher": None, "url": None},
)

HEADERS = {"User-Agent": "ASCIS-VoyageHack-Demo/1.0 (public event research)"}


def source(title: str, url: str, provider: str, excerpt: str = "") -> dict | None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    return {
        "title": title[:240], "url": url, "provider": provider, "publisher": parsed.hostname,
        "excerpt": " ".join(unescape(re.sub(r"<[^>]+>", "", excerpt[:4000])).split())[:2400],
    }


def _firecrawl(query: str, key: str) -> list[dict]:
    response = httpx.post(
        "https://api.firecrawl.dev/v2/search",
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query, "limit": 4, "sources": [{"type": "web"}, {"type": "news"}]},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise ValueError("Search provider declined the request.")
    data = payload.get("data", {})
    return [
        item for group in ("web", "news") for row in data.get(group, [])
        if (item := source(row.get("title") or "Source", row.get("url", ""), f"Firecrawl {group}",
                           row.get("description") or row.get("snippet") or ""))
    ]


def _wikipedia(query: str) -> list[dict]:
    response = httpx.get(
        "https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=6,
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": 4, "format": "json"},
    )
    response.raise_for_status()
    return [
        source(row["title"], "https://en.wikipedia.org/wiki/" + quote(row["title"].replace(" ", "_")),
               "Wikipedia", row.get("snippet", ""))
        for row in response.json()["query"]["search"]
    ]


def _gdelt(query: str) -> list[dict]:
    response = httpx.get(
        "https://api.gdeltproject.org/api/v2/doc/doc", headers=HEADERS, timeout=6,
        params={"query": query, "mode": "ArtList", "format": "json", "maxrecords": 5, "timespan": "3months"},
    )
    response.raise_for_status()
    return [
        item for row in response.json().get("articles", [])
        if (item := source(row["title"], row["url"], "GDELT news"))
    ]


def discover(query: str) -> dict:
    query = " ".join(query.split())[:160]
    if len(query) < 3:
        raise ValueError("Enter at least three characters.")
    key = os.environ.get("FIRECRAWL_API_KEY")
    providers = {"Firecrawl": lambda: _firecrawl(query, key)} if key else {
        "Wikipedia": lambda: _wikipedia(query), "GDELT news": lambda: _gdelt(query),
    }
    statuses, sources = {}, {}
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    for event in EVENTS:
        event_terms = set(re.findall(r"[a-z0-9]+", event["query"].lower()))
        shared = query_terms & event_terms
        if event["url"] and len(shared) >= 2 and shared - {"2027", "world", "cup", "match"}:
            sources[event["url"]] = {
                "title": event["title"], "url": event["url"], "publisher": event["publisher"],
                "provider": "Pinned organizer reference",
            }
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {label: pool.submit(search) for label, search in providers.items()}
        for label, future in futures.items():
            try:
                rows = future.result()
                statuses[label] = "available" if rows else "no results"
                for row in rows:
                    sources.setdefault(row["url"], row)
            except (httpx.HTTPError, ValueError, KeyError):
                statuses[label] = "unavailable"
    return {
        "query": query, "sources": list(sources.values())[:8], "providers": statuses,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "google_url": "https://www.google.com/search?q=" + quote_plus(query),
        "google_news_url": "https://news.google.com/search?q=" + quote_plus(query),
    }
