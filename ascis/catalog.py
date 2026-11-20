"""Fictional saleable hotel inventory for the storefront demo."""

from collections.abc import Sequence
from datetime import date, timedelta

from ascis.market import SELL_PRICE, UNIT_COST
from ascis.research import EVENTS


HOTELS = {
    "icc": {"name": "Jacaranda House", "area": "Rosebank", "style": "Boutique hotel",
            "amenities": ["Breakfast available", "Airport transfers"], "stock": 120,
            "description": "A relaxed, design-led city stay with a garden terrace and easy access to central Johannesburg."},
    "fifa": {"name": "Avenida Mar Hotel", "area": "Botafogo", "style": "Coastal city stay",
             "amenities": ["Rooftop pool", "Breakfast available"], "stock": 85,
             "description": "An airy coastal stay with a rooftop pool and views toward Rio’s bay."},
    "expo": {"name": "Danube Court", "area": "New Belgrade", "style": "Contemporary hotel",
             "amenities": ["Breakfast available", "Group-friendly"], "stock": 60,
             "description": "A contemporary address with generous social spaces for city breaks, groups and event travel."},
    "blr": {"name": "Garden City Hotel", "area": "Central Bengaluru", "style": "City stay",
            "amenities": ["Airport transfers", "Group-friendly"], "stock": 0,
            "description": "A fictional Bengaluru stay for the simulated cricket final."},
}
TRANSFER_PRICE = 1_200
TRANSFER_COST = 850
TRANSFERS_PER_DAY = 24


def dates_between(start: date, end: date) -> list[str]:
    return [(start + timedelta(days=index)).isoformat() for index in range((end - start).days)]


def promotion_seconds(signal: float, committed: int, recovered: int) -> int:
    demand = min(1.0, max(0.0, signal / 1.15))
    unrecovered = max(0, committed - recovered) / committed if committed else 0
    return 5 + round(3 * demand + 2 * unrecovered)


def consume_booking(rooms: dict, acquired: dict, transfers: dict, booking: dict) -> None:
    for day in booking["nights"]:
        rooms[day] -= booking["rooms"]
        acquired[day] -= booking.get("acquired_nights", {}).get(day, 0)
    if booking["offer"] == "package":
        transfers[booking["nights"][0]] -= booking["rooms"]


def listings(approved: dict[str, int], bookings: Sequence[dict] = ()) -> list[dict]:
    records = []
    for event in EVENTS:
        key = event["id"]
        if key not in HOTELS:
            continue
        hotel = HOTELS[key]
        if not (hotel["stock"] or approved.get(event["title"])):
            continue
        event_date = date.fromisoformat(event["date"])
        start, end = event_date - timedelta(days=3), event_date + timedelta(days=4)
        dates = dates_between(start, end)
        each, remainder = divmod(hotel["stock"], len(dates))
        acquired_each, acquired_remainder = divmod(approved.get(event["title"], 0), len(dates))
        acquired = {day: acquired_each + (index < acquired_remainder) for index, day in enumerate(dates)}
        rooms = {day: each + (index < remainder) + acquired[day] for index, day in enumerate(dates)}
        transfers = dict.fromkeys(dates, TRANSFERS_PER_DAY)
        for booking in bookings:
            if booking["hotel_id"] != key:
                continue
            consume_booking(rooms, acquired, transfers, booking)
        records.append({
            **hotel, "id": key, "destination": event["destination"], "price": SELL_PRICE,
            "stock": sum(rooms.values()), "featured_stock": sum(acquired.values()), "event": event["title"],
            "available_from": start.isoformat(), "available_until": end.isoformat(),
            "rooms_by_date": rooms, "acquired_by_date": acquired, "transfers_by_date": transfers,
            "transfer_price": TRANSFER_PRICE,
            "hotel_image": f"/assets/photo-{key if key != 'blr' else 'expo'}.jpg",
            "city_image": f"/assets/{'city-' + key if key != 'blr' else 'photo-expo'}.jpg",
        })
    return sorted(records, key=lambda hotel: hotel["featured_stock"], reverse=True)


def demo_quote(hotel: dict, offer: str, check_in: date, check_out: date, rooms: int) -> dict:
    start, end = check_in.isoformat(), check_out.isoformat()
    if not hotel["available_from"] <= start < end <= hotel["available_until"]:
        raise ValueError("Choose dates within the property's demo stay window.")
    nights = dates_between(check_in, check_out)
    if any(hotel["rooms_by_date"][day] < rooms for day in nights):
        available = min(hotel["rooms_by_date"][day] for day in nights)
        raise ValueError(f"Only {available} rooms are available for the full stay.")
    package = offer == "package"
    if package and hotel["transfers_by_date"][start] < rooms:
        raise ValueError(f"Only {hotel['transfers_by_date'][start]} airport transfers are available on arrival.")
    room_nights = rooms * len(nights)
    return {
        "hotel_id": hotel["id"], "hotel": hotel["name"], "offer": offer, "rooms": rooms,
        "nights": nights, "room_nights": room_nights, "check_out": end,
        "acquired_nights": {day: min(hotel["acquired_by_date"][day], rooms) for day in nights},
        "subtotal_inr": room_nights * SELL_PRICE + (rooms * TRANSFER_PRICE if package else 0),
        "cost_inr": room_nights * UNIT_COST + (rooms * TRANSFER_COST if package else 0),
    }
