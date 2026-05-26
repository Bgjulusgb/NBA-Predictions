"""Action Network public-betting + sharp-money data (best-effort).

Tries two approaches in order:
  1. The public widget JSON API (no auth, used by their embeds).
  2. The web page's __NEXT_DATA__ blob (HTML scrape fallback).

Returns: home/away public_bet_pct, public_money_pct, and moneylines when
available so the model can blend sharp-money signals.
"""
from __future__ import annotations

import datetime as dt
import json
import re

from .. import config
from ..http_util import fetch, fetch_json
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

AN_API = "https://api.actionnetwork.com/web/v1/games?sport=nba&date={date}"
AN_API_ALT = "https://www.actionnetwork.com/api/v1/matchups/nba"
AN_WEB = "https://www.actionnetwork.com/nba/public-betting"

HEADERS_API = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.actionnetwork.com/",
    "Origin": "https://www.actionnetwork.com",
}
HEADERS_WEB = {
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', re.DOTALL)


def _matches(game: dict, home: dict, away: dict) -> bool:
    teams = game.get("teams") or []
    names = " ".join((t.get("full_name") or t.get("name") or "").lower()
                     for t in teams)
    home_ok = any(al in names for al in home["aliases"])
    away_ok = any(al in names for al in away["aliases"])
    return home_ok and away_ok


def _extract_pcts(game: dict) -> dict:
    """Pull public bet % and money % from an Action Network game object."""
    out: dict = {}
    # Different response shapes depending on endpoint.
    for block in (game.get("odds") or []):
        if not isinstance(block, dict):
            continue
        for side_key, label in (("home_team_id", "home"), ("away_team_id", "away")):
            bets_pct = block.get(f"{label}_betting_percentage")
            money_pct = block.get(f"{label}_money_percentage")
            if bets_pct is not None:
                out[f"{label}_public_bet_pct"] = float(bets_pct)
            if money_pct is not None:
                out[f"{label}_public_money_pct"] = float(money_pct)
        ml_home = block.get("ml_home")
        ml_away = block.get("ml_away")
        if ml_home is not None:
            out["home_ml"] = int(ml_home)
        if ml_away is not None:
            out["away_ml"] = int(ml_away)
    # Flatten public betting directly on game object (alternate shape).
    if "public_bets_home_pct" in game:
        out["home_public_bet_pct"] = float(game["public_bets_home_pct"])
    if "public_bets_away_pct" in game:
        out["away_public_bet_pct"] = float(game["public_bets_away_pct"])
    return out


def _try_api(home, away, date: str) -> tuple[dict | None, str | None]:
    """Try the public JSON API endpoints."""
    urls = [AN_API.format(date=date), AN_API_ALT]
    for url in urls:
        data, res = fetch_json(url, headers=HEADERS_API)
        if not res.ok or not data:
            continue
        games = (data.get("games") or data.get("matchups")
                 or (data if isinstance(data, list) else []))
        for g in games:
            if _matches(g, home, away):
                pcts = _extract_pcts(g)
                pcts["source"] = "action_network"
                return pcts, None
        return None, f"target not in API response ({len(games)} games)"
    return None, "all API endpoints failed"


def _try_html(home, away) -> tuple[dict | None, str | None]:
    """Fallback: scrape the __NEXT_DATA__ blob from the web page."""
    res = fetch(AN_WEB, headers=HEADERS_WEB)
    if not res.ok or not res.body:
        return None, res.error
    m = _NEXT_DATA.search(res.body)
    if not m:
        return None, "no __NEXT_DATA__ blob on page"
    try:
        blob = json.loads(m.group(1))
    except ValueError as e:
        return None, f"JSON parse failed: {e}"
    games = _find_list(blob)
    if not games:
        return None, "game list not located in __NEXT_DATA__"
    for g in games:
        if _matches(g, home, away):
            pcts = _extract_pcts(g)
            pcts["source"] = "action_network"
            return pcts, None
    return None, f"target not in HTML data ({len(games)} games)"


def _find_list(node, depth: int = 0):
    if depth > 7:
        return None
    if isinstance(node, dict):
        for key in ("publicBetting", "public_betting", "games", "matchups"):
            v = node.get(key)
            if isinstance(v, list):
                return v
        if {"teams", "odds"}.issubset(node.keys()):
            return [node]
        for v in node.values():
            r = _find_list(v, depth + 1)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_list(v, depth + 1)
            if r:
                return r
    return None


def fetch_public_betting() -> SourceResult:
    home = config.GAME["home"]
    away = config.GAME["away"]
    date = config.GAME["date_et"]

    result, err = _try_api(home, away, date)
    if result:
        return SourceResult("action_network", STATUS_OK, records=[],
                             meta={"betting": result})

    api_err = err
    result, err = _try_html(home, away)
    if result:
        return SourceResult("action_network", STATUS_OK, records=[],
                             meta={"betting": result})

    return SourceResult("action_network", STATUS_PARTIAL,
                         error=f"api: {api_err}; html: {err}")
