"""Flashscore live scores scraper (best-effort).

Tries two approaches in order:
  1. Flashscore's d.flashscore.com feed (pipe/section-delimited custom string).
  2. HTML fallback on the NBA league page — parses team names + scores via regex.

Flashscore is geo/IP-sensitive and rotates feed URLs; if both approaches fail
we report PARTIAL and continue, exactly like every other source here.
"""
from __future__ import annotations

import re

from ..http_util import fetch
from .. import config
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

FLASHSCORE_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.flashscore.com/",
    "Origin": "https://www.flashscore.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "x-fsign": "SW9D1eZo",
}

# Primary: the custom feed endpoint; NBA sport/tournament codes.
FLASHSCORE_FEED = "https://d.flashscore.com/x/feed/f_2_0_3_en_1"
# Secondary: HTML league page.
FLASHSCORE_NBA = "https://www.flashscore.com/basketball/usa/nba/"

TAGS = {
    "AA": "id",
    "AC": "stage",
    "AD": "start_unix",
    "AE": "home_team",
    "AF": "away_team",
    "AG": "home_score",
    "AH": "away_score",
    "AS": "status_id",
    "AT": "status_text",
}

# Patterns for HTML fallback: match team names and nearby score digits.
_HTML_SCORE = re.compile(
    r'class="[^"]*event__score[^"]*"[^>]*>\s*(\d+)\s*</div>'
    r'(?:.*?class="[^"]*event__score[^"]*"[^>]*>\s*(\d+)\s*</div>)?',
    re.IGNORECASE | re.DOTALL,
)
_HTML_TEAM = re.compile(
    r'class="[^"]*event__participant[^"]*"[^>]*>\s*([^<]+)\s*</div>',
    re.IGNORECASE,
)


def _split_blocks(payload: str) -> list[dict]:
    rows = payload.split("¬~AA÷")
    out: list[dict] = []
    for raw in rows[1:]:
        block: dict = {}
        chunks = raw.split("¬")
        chunks[0] = "AA÷" + chunks[0]
        for c in chunks:
            if "÷" not in c:
                continue
            key, _, val = c.partition("÷")
            if key in TAGS:
                block[TAGS[key]] = val
        if block:
            out.append(block)
    return out


def _match_filter(blocks: list[dict], home: dict, away: dict) -> dict | None:
    for b in blocks:
        h = (b.get("home_team") or "").lower()
        a = (b.get("away_team") or "").lower()
        if (any(al in h for al in home["aliases"])
                and any(al in a for al in away["aliases"])):
            return b
    return None


def _try_feed(home, away) -> tuple[dict | None, str | None]:
    """Try the d.flashscore.com pipe-delimited feed. Returns (game_dict, error)."""
    res = fetch(FLASHSCORE_FEED, headers=FLASHSCORE_HEADERS)
    if not res.ok or not res.body:
        return None, res.error
    blocks = _split_blocks(res.body)
    if not blocks:
        return None, "no blocks parsed from feed"
    target = _match_filter(blocks, home, away)
    if not target:
        return None, f"target not in feed ({len(blocks)} blocks)"
    state_map = {"1": "pre", "2": "in", "3": "post"}
    return {
        "id": target.get("id"),
        "name": f"{target.get('away_team')} @ {target.get('home_team')}",
        "state": state_map.get(target.get("status_id"), "pre"),
        "status_detail": target.get("status_text"),
        "home": {"name": target.get("home_team"),
                  "score": _to_int(target.get("home_score"))},
        "away": {"name": target.get("away_team"),
                  "score": _to_int(target.get("away_score"))},
    }, None


def _try_html(home, away) -> tuple[dict | None, str | None]:
    """HTML fallback: scrape the NBA league page for matching team + scores."""
    res = fetch(FLASHSCORE_NBA, headers=FLASHSCORE_HEADERS)
    if not res.ok or not res.body:
        return None, res.error
    body = res.body
    teams_found = _HTML_TEAM.findall(body)
    for i, t in enumerate(teams_found):
        t_low = t.strip().lower()
        if any(al in t_low for al in home["aliases"]):
            # Found home team; check next entry for away match.
            if i + 1 < len(teams_found):
                a_low = teams_found[i + 1].strip().lower()
                if any(al in a_low for al in away["aliases"]):
                    scores = _HTML_SCORE.findall(body)
                    idx = i // 2
                    h_score = a_score = None
                    if idx < len(scores):
                        h_score = _to_int(scores[idx][0])
                        a_score = _to_int(scores[idx][1]) if len(scores[idx]) > 1 else None
                    return {
                        "id": None,
                        "name": f"{t.strip()} vs {teams_found[i+1].strip()}",
                        "state": "pre",
                        "status_detail": "HTML scrape",
                        "home": {"name": t.strip(), "score": h_score},
                        "away": {"name": teams_found[i + 1].strip(), "score": a_score},
                    }, None
    return None, "target game not found in HTML page"


def fetch_game(home: dict | None = None, away: dict | None = None) -> SourceResult:
    home = home or config.GAME["home"]
    away = away or config.GAME["away"]

    game, err = _try_feed(home, away)
    if game:
        return SourceResult("flashscore", STATUS_OK, records=[],
                             meta={"game": game, "method": "feed"})

    feed_err = err
    game, err = _try_html(home, away)
    if game:
        return SourceResult("flashscore", STATUS_OK, records=[],
                             meta={"game": game, "method": "html"})

    return SourceResult("flashscore", STATUS_PARTIAL,
                         error=f"feed: {feed_err}; html: {err}")


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
