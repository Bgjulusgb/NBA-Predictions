"""TheScore.com NBA scores (best-effort).

TheScore exposes a public JSON API used by their mobile app. We try three
endpoints in priority order until one returns usable data. Response shape
varies slightly across endpoints; we normalise to the same SourceResult the
pipeline expects.
"""
from __future__ import annotations

import datetime as dt

from .. import config
from ..http_util import fetch_json
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

THESCORE_HEADERS = {
    "User-Agent": "TheScore/24.0.0 (iOS 17.0; iPhone)",
    "Accept": "application/json",
    "x-api-version": "6",
}

# Endpoints tried in order; first one that returns data wins.
def _endpoints(date: str) -> list[str]:
    return [
        f"https://mobile-statsv2.thescore.com/nba/events?date={date}&league_id=15",
        "https://mobile-statsv2.thescore.com/nba/events?status=in_progress",
        f"https://api.thescore.com/nba/events?date={date}",
        f"https://api.thescore.com/nba/events/daily_events?date={date}",
    ]


def _matches(event: dict, home: dict, away: dict) -> bool:
    h = ((event.get("home_team") or {}).get("full_name") or "").lower()
    a = ((event.get("away_team") or {}).get("full_name") or "").lower()
    # Also check abbreviated names / short names.
    h_short = ((event.get("home_team") or {}).get("short_name") or "").lower()
    a_short = ((event.get("away_team") or {}).get("short_name") or "").lower()
    home_ok = (any(al in h for al in home["aliases"])
               or any(al in h_short for al in home["aliases"]))
    away_ok = (any(al in a for al in away["aliases"])
               or any(al in a_short for al in away["aliases"]))
    return home_ok and away_ok


def _extract_events(data) -> list[dict]:
    if isinstance(data, list):
        return data
    for key in ("data", "events", "games"):
        val = data.get(key)
        if isinstance(val, list):
            return val
    return []


def fetch_game(date: str | None = None,
               home: dict | None = None,
               away: dict | None = None) -> SourceResult:
    home = home or config.GAME["home"]
    away = away or config.GAME["away"]
    if not date:
        date = config.GAME["date_et"]

    last_error = "all endpoints failed"
    for url in _endpoints(date):
        data, res = fetch_json(url, headers=THESCORE_HEADERS)
        if not res.ok or data is None:
            last_error = res.error or f"HTTP error on {url}"
            continue
        events = _extract_events(data)
        for ev in events:
            if _matches(ev, home, away):
                return _normalise(ev)
        # Endpoint worked but our game wasn't there; record for diagnostics.
        last_error = (
            f"target game not in feed ({len(events)} events on {date}) "
            f"via {url.split('?')[0]}"
        )

    return SourceResult("thescore", STATUS_PARTIAL, error=last_error)


def _normalise(ev: dict) -> SourceResult:
    status = (ev.get("status") or "").lower()
    state_map = {"pre_game": "pre", "in_progress": "in", "final": "post"}
    state = next((v for k, v in state_map.items() if k in status), "pre")
    box = ev.get("box_score") or {}
    score = box.get("score") or {}
    game = {
        "id": ev.get("id"),
        "name": (ev.get("description") or "").strip(),
        "state": state,
        "status_detail": ev.get("status_string") or ev.get("status"),
        "period": (ev.get("progress") or {}).get("segment"),
        "clock": (ev.get("progress") or {}).get("clock"),
        "home": {
            "name": (ev.get("home_team") or {}).get("full_name"),
            "abbr": (ev.get("home_team") or {}).get("abbreviation"),
            "score": _to_int((score.get("home") or {}).get("score")),
        },
        "away": {
            "name": (ev.get("away_team") or {}).get("full_name"),
            "abbr": (ev.get("away_team") or {}).get("abbreviation"),
            "score": _to_int((score.get("away") or {}).get("score")),
        },
    }
    return SourceResult("thescore", STATUS_OK, records=[], meta={"game": game})


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
