"""OddsPortal NBA odds movement (HTML scrape, best-effort).

OddsPortal's listing page renders moneyline / spread / total per book on the
match detail page. We're frugal: just pull the headline list of upcoming /
in-play games and their best-price moneyline. For deeper book-by-book history
the user should consider the Action Network endpoint (see action_network.py).

If the page renders odds via XHR (which they have been moving toward), the
HTML scrape will return a partial result — exactly the contract every other
source uses.
"""

from __future__ import annotations

import html
import re

from ..http_util import fetch
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

OP_HEADERS = {
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

OP_NBA = "https://www.oddsportal.com/basketball/usa/nba/"

# Per-match anchor + the two adjacent decimal cells.
_MATCH = re.compile(
    r'<a[^>]*href="(?P<href>/basketball/usa/nba/[^"]+)"[^>]*>'
    r'(?P<teams>[^<]+)</a>',
    re.IGNORECASE)
_DECIMAL = re.compile(r'>\s*(\d+\.\d{2})\s*<')


def fetch_listing() -> SourceResult:
    """Top NBA games shown on the OP NBA page with best-headline odds."""
    res = fetch(OP_NBA, headers=OP_HEADERS)
    if not res.ok or not res.body:
        return SourceResult("oddsportal", STATUS_ERROR, error=res.error)

    body = res.body
    matches = _MATCH.findall(body)
    decimals = _DECIMAL.findall(body)

    # We pair matches with decimals positionally — fragile but it's the only
    # thing that survives OddsPortal's frequent markup updates.
    rows: list[dict] = []
    dec_iter = iter(decimals)
    for href, teams in matches:
        try:
            home_dec = float(next(dec_iter))
            away_dec = float(next(dec_iter))
        except (StopIteration, ValueError):
            home_dec = away_dec = None
        rows.append({
            "match_url": "https://www.oddsportal.com" + html.unescape(href),
            "teams": html.unescape(teams).strip(),
            "home_decimal": home_dec,
            "away_decimal": away_dec,
        })
    return SourceResult("oddsportal",
                         STATUS_OK if rows else STATUS_PARTIAL,
                         records=[], meta={"matches": rows[:30]})
