"""Rotowire NBA lineups + injuries (HTML scrape).

Rotowire publishes daily lineups + injury statuses on public pages. We
scrape both pages with regex patterns targeting the current CSS class names.
Patterns are written defensively — a parse failure returns [] rather than
crashing the pipeline.

  fetch_lineups() → kind='lineup' records
  fetch_injuries() → kind='injury' records
"""
from __future__ import annotations

import html
import re

from .. import config
from ..http_util import fetch
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

ROTOWIRE_LINEUPS = "https://www.rotowire.com/basketball/nba-lineups.php"
ROTOWIRE_INJURIES = "https://www.rotowire.com/basketball/injury-report.php"

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.rotowire.com/",
}

# --- Lineup selectors -------------------------------------------------------
# Match each <ul class="lineup__list"> block (visitor + home per game).
_LINEUP_LIST = re.compile(
    r'<ul[^>]*class="[^"]*lineup__list[^"]*"[^>]*>(.*?)</ul>',
    re.IGNORECASE | re.DOTALL,
)
# Player anchor inside a lineup__player list item.
_LINEUP_PLAYER = re.compile(
    r'<li[^>]*class="[^"]*lineup__player[^"]*"[^>]*>.*?<a[^>]*>([^<]+)</a>',
    re.IGNORECASE | re.DOTALL,
)
# Team name block that precedes each lineup list.
_LINEUP_TEAM = re.compile(
    r'<div[^>]*class="[^"]*lineup__team-name[^"]*"[^>]*>\s*([^<]+)',
    re.IGNORECASE,
)
# Fallback: any player link to basketball/player.php (broad catch).
_PLAYER_LINK = re.compile(
    r'<a[^>]*href="[^"]*basketball/player\.php[^"]*"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)

# --- Injury selectors -------------------------------------------------------
_INJURY_TABLE = re.compile(
    r'<table[^>]*class="[^"]*injury-report[^"]*"[^>]*>(.*?)</table>',
    re.IGNORECASE | re.DOTALL,
)
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", _TAG.sub("", s))).strip()


def fetch_lineups() -> SourceResult:
    """Best-effort scrape of today's projected lineups."""
    try:
        res = fetch(ROTOWIRE_LINEUPS, headers=HEADERS)
        if not res.ok or not res.body:
            return SourceResult("rotowire_lineups", STATUS_ERROR, error=res.error)
        body = res.body
        records: list[dict] = []

        # Primary: structured lineup__list approach.
        lineup_lists = _LINEUP_LIST.findall(body)
        team_names = _LINEUP_TEAM.findall(body)

        for idx, ul_html in enumerate(lineup_lists):
            players = _LINEUP_PLAYER.findall(ul_html)
            team = _strip(team_names[idx]) if idx < len(team_names) else ""
            for name in players:
                clean = _strip(name)
                if not clean:
                    continue
                records.append({
                    "id": f"rotowire_lineup:{clean}",
                    "source": "rotowire_lineups",
                    "kind": "lineup",
                    "title": clean,
                    "text": f"{team} {clean}",
                    "url": ROTOWIRE_LINEUPS,
                    "published": None,
                    "engagement": 0,
                    "team_hint": None,
                    "player": clean,
                    "team_name": team,
                })

        # Fallback: broad player-link sweep if primary found nothing.
        if not records:
            for name in _PLAYER_LINK.findall(body):
                clean = _strip(name)
                if not clean:
                    continue
                records.append({
                    "id": f"rotowire_lineup:{clean}",
                    "source": "rotowire_lineups",
                    "kind": "lineup",
                    "title": clean,
                    "text": clean,
                    "url": ROTOWIRE_LINEUPS,
                    "published": None,
                    "engagement": 0,
                    "team_hint": None,
                    "player": clean,
                })

        status_flag = STATUS_OK if records else STATUS_PARTIAL
        return SourceResult("rotowire_lineups", status_flag, records=records,
                             meta={"teams_seen": [_strip(t) for t in team_names[:20]]})
    except Exception as exc:
        return SourceResult("rotowire_lineups", STATUS_ERROR, error=str(exc))


def fetch_injuries() -> SourceResult:
    """Scrape Rotowire's injury report table."""
    try:
        res = fetch(ROTOWIRE_INJURIES, headers=HEADERS)
        if not res.ok or not res.body:
            return SourceResult("rotowire_injuries", STATUS_ERROR, error=res.error)
        body = res.body

        home_aliases = config.GAME["home"]["aliases"]
        away_aliases = config.GAME["away"]["aliases"]
        home_abbr = config.GAME["home"]["abbr"].lower()
        away_abbr = config.GAME["away"]["abbr"].lower()

        records: list[dict] = []

        # Try injury-report table first.
        tables = _INJURY_TABLE.findall(body)
        source_html = tables[0] if tables else body

        for row_html in _TR.findall(source_html):
            cells = [_strip(c) for c in _TD.findall(row_html)]
            if len(cells) < 4:
                continue
            player = cells[0]
            team_cell = cells[1] if len(cells) > 1 else ""
            position = cells[2] if len(cells) > 2 else ""
            status = cells[3] if len(cells) > 3 else ""
            injury = cells[4] if len(cells) > 4 else ""
            est_return = cells[5] if len(cells) > 5 else ""

            if not player or player.lower() in ("player", "name"):
                continue

            # Filter to our two teams by abbreviation or alias.
            team_low = team_cell.lower()
            is_target = (
                team_low == home_abbr
                or team_low == away_abbr
                or any(al in team_low for al in home_aliases)
                or any(al in team_low for al in away_aliases)
            )
            if not is_target:
                continue

            records.append({
                "id": f"rotowire_injury:{player}",
                "source": "rotowire_injuries",
                "kind": "injury",
                "title": f"{player}: {status}",
                "text": (
                    f"{player} ({position}) [{team_cell}] "
                    f"status: {status} — {injury} — return: {est_return}"
                ),
                "url": ROTOWIRE_INJURIES,
                "published": None,
                "engagement": 0,
                "team_hint": None,
                "player": player,
                "team": team_cell,
                "position": position,
                "status_text": status,
                "injury_desc": injury,
                "expected_return": est_return,
            })

        # If table filtering found nothing, fall back to scraping all rows
        # (non-target injuries still feed the corpus for sentiment).
        if not records and tables:
            for row_html in _TR.findall(tables[0]):
                cells = [_strip(c) for c in _TD.findall(row_html)]
                if len(cells) < 4:
                    continue
                player = cells[0]
                if not player or player.lower() in ("player", "name"):
                    continue
                records.append({
                    "id": f"rotowire_injury:{player}",
                    "source": "rotowire_injuries",
                    "kind": "injury",
                    "title": f"{player}: {cells[3] if len(cells) > 3 else ''}",
                    "text": " ".join(cells),
                    "url": ROTOWIRE_INJURIES,
                    "published": None,
                    "engagement": 0,
                    "team_hint": None,
                    "player": player,
                })

        return SourceResult("rotowire_injuries",
                             STATUS_OK if records else STATUS_PARTIAL,
                             records=records)
    except Exception as exc:
        return SourceResult("rotowire_injuries", STATUS_ERROR, error=str(exc))
