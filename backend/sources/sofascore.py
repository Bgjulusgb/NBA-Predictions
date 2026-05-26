"""Sofascore scraper — the most data-rich free NBA source we use.

Sofascore's public web app talks to an open JSON API at api.sofascore.com.
No key required, but it inspects headers, so we send browser-ish defaults.
This module exposes everything the pipeline + dashboard care about:

  * fetch_game()              - scheduled / in-play game (score, status, period)
  * fetch_statistics()        - team box stats (FG, 3P, FT, REB, AST, TOV, ...)
  * fetch_incidents()         - play-by-play (scoring + non-scoring events)
  * fetch_lineups()           - starting + bench (with positions / minutes)
  * fetch_odds()              - moneyline + spread + total per provider
  * fetch_form()              - last-5 form for both sides
  * fetch_h2h()               - head-to-head summary
  * fetch_graph()             - per-minute scoring graph (line chart data)
  * fetch_featured_players()  - sofascore's "key players" picks
  * fetch_all()               - one-shot bundle, parallelised

All requests degrade gracefully: missing fields just disappear from the result,
network failures return a STATUS_ERROR SourceResult, and the pipeline keeps
going. Output records always carry a `source` tag so downstream code can tell
which scraper a metric came from.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from .. import config
from ..http_util import fetch_json, run_parallel
from .base import (SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL)

# --- API base + headers ----------------------------------------------------
SOFASCORE_API = "https://api.sofascore.com/api/v1"
SOFASCORE_WEB = "https://www.sofascore.com"

# Sofascore checks for typical browser headers; without them you get 403.
SOFASCORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Site": "same-origin",
    "Cache-Control": "no-cache",
}

# Sofascore's internal NBA tournament id is 132; we keep it configurable
# because pre-season / Summer League / play-in have different ids.
NBA_TOURNAMENT_ID = 132
# Sport key in the URL path.
NBA_SPORT_KEY = "basketball"


def _date_key(d: str | None = None) -> str:
    """Sofascore expects YYYY-MM-DD; default to today UTC."""
    if d:
        return d
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Game discovery
# ---------------------------------------------------------------------------
def _matches_target(event: dict, home: dict, away: dict) -> bool:
    """True if this event matches our configured target game.

    Sofascore returns names like "New York Knicks" + short "NY Knicks", we
    match on lowercased alias overlap so casing & punctuation don't bite.
    """
    h = (event.get("homeTeam") or {}).get("name", "").lower()
    a = (event.get("awayTeam") or {}).get("name", "").lower()
    h_alias = any(al in h for al in home["aliases"])
    a_alias = any(al in a for al in away["aliases"])
    return h_alias and a_alias


def discover_event_id(date: str | None = None, home: dict | None = None,
                      away: dict | None = None) -> tuple[str | None, SourceResult]:
    """Find Sofascore's numeric event id for our target game.

    Returns (event_id, SourceResult). The SourceResult is informational so the
    pipeline can show "we found id 12345" or "we couldn't find your matchup".
    """
    home = home or config.GAME["home"]
    away = away or config.GAME["away"]
    date = _date_key(date)
    url = f"{SOFASCORE_API}/sport/{NBA_SPORT_KEY}/scheduled-events/{date}"

    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or data is None:
        return None, SourceResult("sofascore_discover", STATUS_ERROR,
                                   error=res.error)
    events = data.get("events", []) or []
    # Narrow to NBA first to skip Euroleague / NCAA noise.
    nba_events = [e for e in events
                  if (e.get("tournament", {}).get("uniqueTournament", {}) or {})
                  .get("id") == NBA_TOURNAMENT_ID
                  or "nba" in (e.get("tournament", {}).get("name") or "").lower()]
    for ev in nba_events or events:
        if _matches_target(ev, home, away):
            return (str(ev.get("id")),
                    SourceResult("sofascore_discover", STATUS_OK,
                                  meta={"event_id": ev.get("id"),
                                         "names": (
                                             ev["homeTeam"]["name"],
                                             ev["awayTeam"]["name"])}))
    return None, SourceResult(
        "sofascore_discover", STATUS_PARTIAL,
        error=f"no matching event on {date}",
        meta={"sampled": [(e.get("homeTeam", {}).get("name"),
                            e.get("awayTeam", {}).get("name"))
                          for e in events[:8]]})


# ---------------------------------------------------------------------------
# Per-resource fetchers — each returns a SourceResult.
# ---------------------------------------------------------------------------
def fetch_game(event_id: str | int) -> SourceResult:
    """Live game state: score, status, period, clock."""
    url = f"{SOFASCORE_API}/event/{event_id}"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_game", STATUS_ERROR, error=res.error)
    ev = data.get("event") or {}
    home_team = ev.get("homeTeam") or {}
    away_team = ev.get("awayTeam") or {}
    home_score = ev.get("homeScore") or {}
    away_score = ev.get("awayScore") or {}
    status = ev.get("status") or {}
    # Sofascore status.type: notstarted | inprogress | finished
    state_map = {"notstarted": "pre", "inprogress": "in", "finished": "post"}
    period = (ev.get("time") or {}).get("currentPeriodStartTimestamp")
    game = {
        "id": ev.get("id"),
        "name": f"{away_team.get('name', '')} @ {home_team.get('name', '')}",
        "state": state_map.get((status.get("type") or "").lower(), "pre"),
        "status_detail": status.get("description"),
        "period": ev.get("currentPeriod") or _period_from_status(status),
        "clock": _clock_from_status(status),
        "home": {
            "name": home_team.get("name"),
            "abbr": home_team.get("nameCode"),
            "score": home_score.get("current"),
            "period_scores": _period_scores(home_score),
        },
        "away": {
            "name": away_team.get("name"),
            "abbr": away_team.get("nameCode"),
            "score": away_score.get("current"),
            "period_scores": _period_scores(away_score),
        },
        "attendance": ev.get("attendance"),
        "venue": ((ev.get("venue") or {}).get("stadium") or {}).get("name"),
    }
    status_flag = STATUS_OK if game["state"] in ("in", "post") else STATUS_PARTIAL
    return SourceResult("sofascore_game", status_flag, records=[],
                         meta={"game": game, "raw_status": status})


def _period_from_status(status: dict) -> int | None:
    desc = (status.get("description") or "").lower()
    if "1st" in desc or "q1" in desc: return 1
    if "2nd" in desc or "q2" in desc: return 2
    if "3rd" in desc or "q3" in desc: return 3
    if "4th" in desc or "q4" in desc: return 4
    if "ot" in desc: return 5
    return None


def _clock_from_status(status: dict) -> str | None:
    # Sofascore exposes time differently per sport; basketball uses initial
    # "12:00" countdown via status.description like "Q3 7:24".
    desc = status.get("description") or ""
    parts = desc.split()
    for p in parts:
        if ":" in p:
            return p
    return None


def _period_scores(score: dict) -> list[int]:
    """Return the per-period scoring list when present (period1..period4 [+OT])."""
    out = []
    for i in range(1, 6):
        v = score.get(f"period{i}")
        if v is None:
            break
        out.append(v)
    return out


def fetch_statistics(event_id: str | int) -> SourceResult:
    """Box-score-style team statistics."""
    url = f"{SOFASCORE_API}/event/{event_id}/statistics"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_stats", STATUS_PARTIAL, error=res.error)
    stats = {"all": {}, "by_period": {}}
    for block in (data.get("statistics") or []):
        period = block.get("period", "ALL")
        bucket = stats["by_period"].setdefault(period, {})
        for group in block.get("groups", []) or []:
            for item in group.get("statisticsItems", []) or []:
                name = item.get("name") or item.get("key")
                if not name:
                    continue
                bucket[name] = {
                    "home": item.get("home"),
                    "away": item.get("away"),
                    "home_total": item.get("homeTotal"),
                    "away_total": item.get("awayTotal"),
                }
        if period == "ALL":
            stats["all"] = bucket
    return SourceResult("sofascore_stats",
                         STATUS_OK if stats["all"] else STATUS_PARTIAL,
                         records=[], meta={"stats": stats})


def fetch_incidents(event_id: str | int) -> SourceResult:
    """Play-by-play stream: scoring events, fouls, subs, period boundaries."""
    url = f"{SOFASCORE_API}/event/{event_id}/incidents"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_incidents", STATUS_PARTIAL,
                             error=res.error)
    incidents = data.get("incidents") or []
    records: list[dict] = []
    for inc in incidents:
        text = inc.get("text") or inc.get("incidentType") or ""
        side = inc.get("isHome")
        team_hint = "home" if side is True else "away" if side is False else None
        points = inc.get("points") or _infer_points(inc)
        # Sofascore reverses the incident list (newest first); we keep that
        # order so downstream "recent_events" tails stay tight.
        records.append({
            "id": f"sofascore_inc:{inc.get('id') or inc.get('sequence')}",
            "source": "sofascore_incidents",
            "kind": "live",
            "title": text[:160],
            "text": text,
            "url": "",
            "author": None,
            "published": _iso_from_minute(inc),
            "engagement": 0,
            "team": team_hint,
            "points": points,
            "period": inc.get("period") or inc.get("currentPeriod"),
            "clock": inc.get("time") or inc.get("addedTime") or "",
            "incident_type": inc.get("incidentType"),
            "incident_class": inc.get("incidentClass"),
            "score_home": (inc.get("homeScore") if isinstance(inc.get("homeScore"), int)
                            else None),
            "score_away": (inc.get("awayScore") if isinstance(inc.get("awayScore"), int)
                            else None),
        })
    return SourceResult("sofascore_incidents",
                         STATUS_OK if records else STATUS_PARTIAL,
                         records=records)


def _infer_points(inc: dict) -> int:
    """Sofascore doesn't always set incident.points; infer from text / type."""
    typ = (inc.get("incidentType") or "").lower()
    cls = (inc.get("incidentClass") or "").lower()
    if "three" in typ or cls == "threepointer":
        return 3
    if "two" in typ or cls == "twopointer":
        return 2
    if "free" in typ or cls == "freethrow":
        return 1
    text = (inc.get("text") or "").lower()
    if "3pt" in text or "3-pt" in text or "three pointer" in text:
        return 3
    if "free throw" in text or "ft" in text and "made" in text:
        return 1
    if "made" in text and ("layup" in text or "jumper" in text or "dunk" in text):
        return 2
    return 0


def _iso_from_minute(inc: dict) -> str | None:
    """Sofascore incidents use a 'time' field (minute mark, not wall clock).

    For our purposes we attach the snapshot time so timeline charts can order
    them; the precise minute lives in the 'clock'/'period' fields anyway.
    """
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fetch_lineups(event_id: str | int) -> SourceResult:
    """Starting lineup + bench, with positions and minute-by-minute when live."""
    url = f"{SOFASCORE_API}/event/{event_id}/lineups"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_lineups", STATUS_PARTIAL, error=res.error)
    out = {"home": _side_lineup(data.get("home") or {}),
            "away": _side_lineup(data.get("away") or {}),
            "confirmed": data.get("confirmed", False)}
    return SourceResult("sofascore_lineups",
                         STATUS_OK if (out["home"] or out["away"]) else STATUS_PARTIAL,
                         records=[], meta={"lineups": out})


def _side_lineup(side: dict) -> dict:
    starters = []
    bench = []
    for entry in side.get("players", []) or []:
        player = entry.get("player") or {}
        stats = entry.get("statistics") or {}
        slot = {
            "name": player.get("name"),
            "short_name": player.get("shortName"),
            "id": player.get("id"),
            "position": entry.get("position"),
            "jersey": entry.get("shirtNumber") or entry.get("jerseyNumber"),
            "minutes_played": stats.get("minutesPlayed"),
            "pts": stats.get("points"),
            "ast": stats.get("assists"),
            "reb": stats.get("rebounds"),
            "stl": stats.get("steals"),
            "blk": stats.get("blocks"),
            "fgm": stats.get("fieldGoalsMade"),
            "fga": stats.get("fieldGoalAttempts"),
            "fg3m": stats.get("threePointsMade"),
            "fg3a": stats.get("threePointAttempts"),
            "ftm": stats.get("freeThrowsMade"),
            "fta": stats.get("freeThrowAttempts"),
            "tov": stats.get("turnovers"),
            "plus_minus": stats.get("plusMinus"),
        }
        if entry.get("substitute"):
            bench.append(slot)
        else:
            starters.append(slot)
    return {
        "starters": starters,
        "bench": bench,
        "formation": side.get("formation"),
        "missing": [_missing(m) for m in side.get("missingPlayers", []) or []],
    }


def _missing(entry: dict) -> dict:
    p = entry.get("player") or {}
    return {
        "name": p.get("name"),
        "id": p.get("id"),
        "type": entry.get("type"),       # "missing" | "doubtful" | ...
        "reason": entry.get("reason"),   # textual reason (injury / suspension)
    }


def fetch_odds(event_id: str | int) -> SourceResult:
    """Sofascore aggregates odds from multiple bookmakers — perfect for compare."""
    url = f"{SOFASCORE_API}/event/{event_id}/odds/1/all"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_odds", STATUS_PARTIAL, error=res.error)

    markets: dict[str, list[dict]] = {"moneyline": [], "spread": [], "total": []}
    for block in (data.get("markets") or []):
        name = (block.get("marketName") or "").lower()
        choices = block.get("choices") or []
        provider = (block.get("provider") or {}).get("name") or block.get("providerId")
        if "home/away" in name or "moneyline" in name or block.get("marketName") == "Match winner":
            entry = {"provider": str(provider),
                      "home_decimal": _decimal_for(choices, ("1", "home")),
                      "away_decimal": _decimal_for(choices, ("2", "away"))}
            markets["moneyline"].append(entry)
        elif "spread" in name or "handicap" in name:
            markets["spread"].append({"provider": str(provider),
                                       "choices": [(c.get("name"), c.get("fractionalValue"))
                                                    for c in choices]})
        elif "total" in name or "over/under" in name:
            markets["total"].append({"provider": str(provider),
                                       "choices": [(c.get("name"), c.get("fractionalValue"))
                                                    for c in choices]})

    has_data = any(markets.values())
    return SourceResult("sofascore_odds",
                         STATUS_OK if has_data else STATUS_PARTIAL,
                         records=[], meta={"odds": markets})


def _decimal_for(choices: list[dict], keys: tuple[str, ...]) -> float | None:
    """Pull a decimal price for a moneyline choice (home / away)."""
    for c in choices:
        nm = (c.get("name") or "").lower()
        if any(k == nm or k in nm for k in keys):
            frac = c.get("fractionalValue") or c.get("decimalValue")
            return _to_decimal(frac)
    return None


def _to_decimal(frac: Any) -> float | None:
    """Sofascore returns fractional ('11/10') or decimal ('2.1') strings."""
    if frac is None:
        return None
    s = str(frac)
    if "/" in s:
        try:
            n, d = s.split("/")
            return round(1 + float(n) / float(d), 4)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def fetch_form(event_id: str | int) -> SourceResult:
    """Recent-form summary (W/L/streak) for both sides."""
    url = f"{SOFASCORE_API}/event/{event_id}/form"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_form", STATUS_PARTIAL, error=res.error)
    return SourceResult("sofascore_form", STATUS_OK, records=[],
                         meta={"form": {
                             "home": _form_block(data.get("homeTeam") or {}),
                             "away": _form_block(data.get("awayTeam") or {}),
                         }})


def _form_block(side: dict) -> dict:
    pos = side.get("position")
    form_str = side.get("form") or ""
    return {
        "position": pos,
        "form": form_str,
        "value": side.get("value"),
        "wins": form_str.count("W"),
        "losses": form_str.count("L"),
    }


def fetch_h2h(event_id: str | int) -> SourceResult:
    """Head-to-head: wins/losses in past meetings."""
    url = f"{SOFASCORE_API}/event/{event_id}/h2h"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_h2h", STATUS_PARTIAL, error=res.error)
    tt = data.get("teamDuel") or {}
    return SourceResult("sofascore_h2h", STATUS_OK, records=[],
                         meta={"h2h": {
                             "home_wins": tt.get("homeWins"),
                             "away_wins": tt.get("awayWins"),
                             "draws": tt.get("draws"),
                         }})


def fetch_graph(event_id: str | int) -> SourceResult:
    """Minute-by-minute scoring (great for a live line chart)."""
    url = f"{SOFASCORE_API}/event/{event_id}/graph"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_graph", STATUS_PARTIAL, error=res.error)
    points = data.get("graphPoints") or []
    out = [{"minute": p.get("minute"), "value": p.get("value")}
           for p in points if p.get("minute") is not None]
    return SourceResult("sofascore_graph",
                         STATUS_OK if out else STATUS_PARTIAL,
                         records=[], meta={"points": out})


def fetch_featured_players(event_id: str | int) -> SourceResult:
    """Sofascore's algorithmic 'key players' picks."""
    url = f"{SOFASCORE_API}/event/{event_id}/featured-players"
    data, res = fetch_json(url, headers=SOFASCORE_HEADERS)
    if not res.ok or not data:
        return SourceResult("sofascore_featured", STATUS_PARTIAL,
                             error=res.error)
    feat = []
    for key in ("homeTopRatingPlayer", "awayTopRatingPlayer"):
        block = data.get(key) or {}
        player = block.get("player") or {}
        if not player:
            continue
        feat.append({
            "side": "home" if "home" in key else "away",
            "name": player.get("name"),
            "id": player.get("id"),
            "rating": (block.get("statistics") or {}).get("rating"),
            "stats": block.get("statistics"),
        })
    return SourceResult("sofascore_featured",
                         STATUS_OK if feat else STATUS_PARTIAL,
                         records=[], meta={"featured": feat})


# ---------------------------------------------------------------------------
# One-shot: discover the event id, then pull every panel in parallel.
# ---------------------------------------------------------------------------
def fetch_all(date: str | None = None) -> dict:
    """Pull every panel for the target game. Returns a dict of SourceResult.

    Anything that fails individually returns a partial SourceResult — the
    overall dict still contains a key for it so the pipeline can render
    a per-panel status badge.
    """
    event_id, discover = discover_event_id(date=date)
    bundle: dict[str, SourceResult] = {"discover": discover}
    if not event_id:
        return bundle
    tasks = {
        "game": lambda: fetch_game(event_id),
        "stats": lambda: fetch_statistics(event_id),
        "incidents": lambda: fetch_incidents(event_id),
        "lineups": lambda: fetch_lineups(event_id),
        "odds": lambda: fetch_odds(event_id),
        "form": lambda: fetch_form(event_id),
        "h2h": lambda: fetch_h2h(event_id),
        "graph": lambda: fetch_graph(event_id),
        "featured": lambda: fetch_featured_players(event_id),
    }
    parallel = run_parallel(tasks, max_workers=6)
    for k, v in parallel.items():
        if isinstance(v, Exception):
            bundle[k] = SourceResult(f"sofascore_{k}", STATUS_ERROR, error=str(v))
        else:
            bundle[k] = v
    return bundle
