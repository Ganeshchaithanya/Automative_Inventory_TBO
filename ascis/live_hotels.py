"""Optional, read-only partner lookup for event-date hotel availability."""

from datetime import datetime, timezone
import os
from urllib.parse import urlsplit

import httpx


AIRPORTS = {"icc": "JNB", "fifa": "RIO", "expo": "BEG"}
API = "https://demandapi.booking.com/3.3/accommodations/"


def available_hotels(event_id: str, check_in: str, check_out: str) -> dict:
    token, affiliate = os.getenv("BOOKING_DEMAND_TOKEN"), os.getenv("BOOKING_AFFILIATE_ID")
    if not token or not affiliate:
        return {"status": "Live availability is unverified. Use the property links or connect an authorized hotel provider.", "offers": []}
    params = {
        "airport": AIRPORTS[event_id], "checkin": check_in, "checkout": check_out, "rows": 10,
        "booker": {"country": "in", "platform": "desktop"},
        "guests": {"number_of_adults": 2, "number_of_rooms": 1},
        "currency": "INR",
    }
    try:
        with httpx.Client(headers={"Authorization": f"Bearer {token}", "X-Affiliate-Id": affiliate}, timeout=8) as client:
            search = client.post(API + "search", json=params)
            search.raise_for_status()
            matches = search.json()["data"][:5]
            names = {}
            if matches:
                details = client.post(API + "details", json={
                    "accommodations": [row["id"] for row in matches], "languages": ["en-gb"],
                })
                details.raise_for_status()
                for row in details.json()["data"]:
                    name = row.get("name")
                    names[row["id"]] = name.get("en-gb") if isinstance(name, dict) else name
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {"status": "The live provider could not verify availability for these dates.", "offers": []}

    offers = []
    for item in matches:
        url = item.get("url")
        if not isinstance(url, str) or urlsplit(url).scheme != "https" or (
            urlsplit(url).hostname not in {"www.booking.com", "booking.com", "secure.booking.com"}
        ):
            continue
        price = item.get("price", {}).get("total")
        currency = item.get("currency")
        offers.append({
            "name": names.get(item["id"]) or f"Booking.com property {item['id']}", "url": url,
            "total": price if isinstance(price, (int, float)) else None,
            "currency": currency if isinstance(currency, str) and len(currency) == 3 and currency.isalpha() else None,
        })
    return {
        "status": "Live date search completed." if offers else "No displayable live offers were returned.",
        "offers": offers, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
