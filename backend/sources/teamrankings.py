"""TeamRankings NBA power-ranking & ATS-trend scraper (HTML).

The site publishes simple HTML tables — ATS records, over/under trends, power
ratings — that are stable to scrape. Feeds the prediction model with an extra
independent power-rating signal next to Elo and Sofascore's form.
"""

from __future__ import annotations

import html
import re

from ..http_util import fetch
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

TR_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.teamrankings.com/",
}

POWER_URL = "https://www.teamrankings.com/nba/ranking/predictive-by-other"
ATS_URL = "https://www.teamrankings.com/nba/trends/ats_trends/"
OVER_UNDER_URL = "https://www.teamrankings.com/nba/trends/ou_trends/"

# A generic <tr>...<td>...</td>... extractor. Cheap, robust to most edits.
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _cells(row: str) -> list[str]:
    return [html.unescape(_TAG.sub("", c)).strip() for c in _TD.findall(row)]


def _table_rows(body: str, min_cells: int = 3) -> list[list[str]]:
    out = []
    for row in _TR.findall(body):
        c = _cells(row)
        if len(c) >= min_cells:
            out.append(c)
    return out


def fetch_power_ratings() -> SourceResult:
    """Pull TeamRankings power ratings (rank, team, rating)."""
    res = fetch(POWER_URL, headers=TR_HEADERS)
    if not res.ok or not res.body:
        return SourceResult("teamrankings_power", STATUS_ERROR, error=res.error)
    rows = _table_rows(res.body, min_cells=4)
    rankings = []
    for row in rows:
        # Skip header rows (where first cell isn't a rank number).
        try:
            rank = int(row[0])
        except (ValueError, IndexError):
            continue
        rankings.append({
            "rank": rank,
            "team": row[1] if len(row) > 1 else None,
            "rating": _safe_float(row[2] if len(row) > 2 else None),
            "last_3": _safe_float(row[3] if len(row) > 3 else None),
        })
    return SourceResult("teamrankings_power",
                         STATUS_OK if rankings else STATUS_PARTIAL,
                         records=[], meta={"rankings": rankings})


def fetch_ats_trends() -> SourceResult:
    """Pull ATS (against-the-spread) trend rows."""
    res = fetch(ATS_URL, headers=TR_HEADERS)
    if not res.ok or not res.body:
        return SourceResult("teamrankings_ats", STATUS_ERROR, error=res.error)
    rows = _table_rows(res.body, min_cells=5)
    trends = [{"trend": row[0], "wins": row[1], "losses": row[2],
                "win_pct": row[3], "roi": row[4]}
              for row in rows[:50]]
    return SourceResult("teamrankings_ats",
                         STATUS_OK if trends else STATUS_PARTIAL,
                         records=[], meta={"trends": trends})


def fetch_ou_trends() -> SourceResult:
    """Pull over/under trend rows."""
    res = fetch(OVER_UNDER_URL, headers=TR_HEADERS)
    if not res.ok or not res.body:
        return SourceResult("teamrankings_ou", STATUS_ERROR, error=res.error)
    rows = _table_rows(res.body, min_cells=5)
    trends = [{"trend": row[0], "overs": row[1], "unders": row[2],
                "over_pct": row[3], "roi": row[4]}
              for row in rows[:50]]
    return SourceResult("teamrankings_ou",
                         STATUS_OK if trends else STATUS_PARTIAL,
                         records=[], meta={"trends": trends})


def _safe_float(s):
    if s is None:
        return None
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)))
    except (ValueError, TypeError):
        return None
