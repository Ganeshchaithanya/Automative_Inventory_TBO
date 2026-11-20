"""Organizer-reported demand proxies and verified public hotel candidates."""

from math import log
from urllib.parse import urlencode


ICC_REPORT = "https://www.icc-cricket.com/news/more-history-made-at-world-cup-as-new-record-for-spectators-broken"
FIFA_REPORT = "https://inside.fifa.com/official-documents/annual-report/2023/competitions/fifa-womens-world-cup-australia-and-new-zealand-2023/marketing-commercial-broadcast-communications"
BIE_REPORT = "https://social.bie-paris.org/site/en/2017-astana"

HISTORY = {
    "icc": (
        {"edition": "2015 · Australia & New Zealand", "count": 1_016_420, "metric": "Tournament admissions", "url": ICC_REPORT},
        {"edition": "2019 · England & Wales", "count": 752_000, "metric": "Tournament admissions", "url": ICC_REPORT},
        {"edition": "2023 · India", "count": 1_250_307, "metric": "Tournament admissions", "url": ICC_REPORT},
    ),
    "fifa": (
        {"edition": "2023 · Australia & New Zealand", "count": 1_978_274, "metric": "Tournament admissions", "url": FIFA_REPORT},
        {"edition": "2023 · Sydney", "count": 610_822, "metric": "Sydney match admissions", "url": "https://files.parliament.nsw.gov.au/fileapi/ParlFiles/GetArtifact/S2425-028%20DNSW%20Annual%20Report%2023-24_Entire_30-10-24.pdf?serverRelativeUrl=%2Ftp%2Ffiles%2F189724%2FS2425-028+DNSW+Annual+Report+23-24_Entire_30-10-24.pdf"},
    ),
    "expo": (
        {"edition": "2017 · Astana Specialized Expo", "count": 3_977_545, "metric": "Expo visits across the full event", "url": BIE_REPORT},
    ),
}

PROPERTIES = {
    "icc": (
        {"name": "Hyatt House Johannesburg Rosebank", "url": "https://www.hyatt.com/en-US/hotel/south-africa/hyatt-house-johannesburg-rosebank/johxr"},
    ),
    "fifa": (
        {"name": "Hilton Rio de Janeiro Copacabana", "url": "https://www.hilton.com/pt/hotels/riocchh-hilton-rio-de-janeiro-copacabana/hotel-info/"},
    ),
    "expo": (
        {"name": "Courtyard Belgrade City Center", "url": "https://www.marriott.com/en-us/hotels/begcy-courtyard-belgrade-city-center/overview/"},
    ),
}


def attendance_adjustment(event_id: str) -> float:
    """Adjust the synthetic signal only when comparable editions share a metric."""
    series = HISTORY.get(event_id, ())
    comparable = [row["count"] for row in series if row["metric"] == "Tournament admissions"]
    return round(0.08 * max(-0.5, min(0.5, log(comparable[-1] / comparable[0]))), 3) if len(comparable) > 1 else 0.0


def hotel_search(event: dict, start: str, end: str) -> str:
    query = f"hotels in {event['destination']} check-in {start} check-out {end}"
    return "https://www.google.com/travel/search?" + urlencode({"q": query, "hl": "en"})
