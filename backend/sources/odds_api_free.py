"""Multi-book odds scraper (free, no API key).

Tries two public sources in order:
  1. VegasInsider NBA moneylines (HTML scrape).
  2. OddsPortal NBA odds (HTML scrape, fallback).

Returns a list of {book, home_ml, away_ml} dicts so the dashboard can show
a multi-book odds table.
"""
from __future__ import annotations

import html
import re

from .. import config
from ..http_util import fetch
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

VI_URL = "https://www.vegasinsider.com/nba/odds/las-vegas/"
OP_URL = "https://www.oddsportal.com/basketball/usa/nba/"

VI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": "https://www.vegasinsider.com/",
}
OP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html",
}

_ML_CELL = re.compile(r"([+\-]\d{3,4})")
_TEAM_CELL = re.compile(
    r'class="[^"]*team[^"]*"[^>]*>\s*<[^>]*>([^<]{3,40})<',
    re.IGNORECASE,
)
# OddsPortal decimal prices.
_DECIMAL = re.compile(r">\s*(\d+\.\d{2})\s*<")


def _decimal_to_american(dec: float) -> int:
    """Convert decimal odds to American moneyline."""
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    if dec > 1.0:
        return int(round(-100 / (dec - 1)))
    return 0


def _scrape_vi(home, away) -> list[dict]:
    res = fetch(VI_URL, headers=VI_HEADERS)
    if not res.ok or not res.body:
        return []
    body = res.body
    books: list[dict] = []

    # VegasInsider embeds moneylines in a structured table.
    # We look for blocks of two consecutive team-matching rows near ML cells.
    team_hits: list[tuple[int, str]] = []
    for m in re.finditer(
        r'(?:cavaliers|cavs|knicks|new\s*york|cleveland)',
        body, re.IGNORECASE
    ):
        team_hits.append((m.start(), m.group(0).lower()))

    # Pull all moneylines from page and pair them up as (away_ml, home_ml).
    all_mls = [int(x) for x in _ML_CELL.findall(body)]
    if len(all_mls) >= 2:
        # Group into pairs; first is usually away, second is home.
        for i in range(0, len(all_mls) - 1, 2):
            away_ml, home_ml = all_mls[i], all_mls[i + 1]
            if abs(away_ml) > 1000 or abs(home_ml) > 1000:
                continue
            books.append({
                "book": "vegasinsider",
                "home_ml": home_ml,
                "away_ml": away_ml,
                "source": "odds_scraper",
            })
            break  # take first matching pair

    return books


def _scrape_op(home, away) -> list[dict]:
    res = fetch(OP_URL, headers=OP_HEADERS)
    if not res.ok or not res.body:
        return []
    body = res.body
    books: list[dict] = []
    decimals = _DECIMAL.findall(body)
    if len(decimals) >= 2:
        try:
            home_dec = float(decimals[0])
            away_dec = float(decimals[1])
            books.append({
                "book": "oddsportal",
                "home_ml": _decimal_to_american(home_dec),
                "away_ml": _decimal_to_american(away_dec),
                "home_decimal": home_dec,
                "away_decimal": away_dec,
                "source": "odds_scraper",
            })
        except (ValueError, IndexError):
            pass
    return books


def fetch_odds() -> SourceResult:
    home = config.GAME["home"]
    away = config.GAME["away"]
    books: list[dict] = []
    errors: list[str] = []

    try:
        vi_books = _scrape_vi(home, away)
        books.extend(vi_books)
    except Exception as exc:
        errors.append(f"vegasinsider: {exc}")

    if not books:
        try:
            op_books = _scrape_op(home, away)
            books.extend(op_books)
        except Exception as exc:
            errors.append(f"oddsportal: {exc}")

    if not books:
        return SourceResult("odds_scraper", STATUS_PARTIAL,
                             error="; ".join(errors) or "no odds found")
    return SourceResult("odds_scraper", STATUS_OK, records=[],
                         meta={"books": books},
                         error="; ".join(errors) or None)
