"""Multi-book odds comparison: best price, arbitrage, no-vig consensus, CLV.

ESPN's scoreboard typically reports one set of odds, but the codebase already
strips them into a structured dict, so any caller that supplies a list of
{provider, home_moneyline, away_moneyline} can plug into these helpers.

Functions:

  * best_price            - cheapest book for each side
  * no_vig_consensus      - de-vig per book, then average the implied probs
  * arbitrage             - is the best-bid pair an arb? returns stake split
  * sharp_movement        - detect sharp money via line drift vs the open
  * closing_line_value    - your prob vs the closing implied prob (post-game)
"""

from __future__ import annotations

import math
from typing import Sequence

from . import model


def best_price(books: Sequence[dict]) -> dict:
    """Pick the best American moneyline for each side across a list of books.

    Each book: {'provider', 'home_moneyline', 'away_moneyline'}.
    The "best" home price is the one with the highest implied payout (= highest
    decimal odds), and vice-versa.
    """
    if not books:
        return {"home": None, "away": None}
    home_best = away_best = None
    home_dec = away_dec = 0.0
    for b in books:
        dh = model.american_to_decimal(b.get("home_moneyline"))
        da = model.american_to_decimal(b.get("away_moneyline"))
        if dh is not None and dh > home_dec:
            home_dec = dh
            home_best = {"provider": b.get("provider"),
                         "moneyline": b.get("home_moneyline"),
                         "decimal": round(dh, 4)}
        if da is not None and da > away_dec:
            away_dec = da
            away_best = {"provider": b.get("provider"),
                         "moneyline": b.get("away_moneyline"),
                         "decimal": round(da, 4)}
    return {"home": home_best, "away": away_best}


def no_vig_consensus(books: Sequence[dict]) -> dict:
    """De-vig every book then mean the resulting probabilities."""
    rows = []
    for b in books:
        m = model.devig_two_way(b.get("home_moneyline"), b.get("away_moneyline"))
        if m:
            rows.append(m)
    if not rows:
        return {"home": None, "away": None, "n_books": 0}
    home = sum(r["home"] for r in rows) / len(rows)
    away = sum(r["away"] for r in rows) / len(rows)
    avg_vig = sum(r["margin_pct"] for r in rows) / len(rows)
    return {
        "home": round(home, 4),
        "away": round(away, 4),
        "n_books": len(rows),
        "avg_vig_pct": round(avg_vig, 2),
    }


def arbitrage(best: dict, stake: float = 100.0) -> dict | None:
    """Detect an arbitrage from the output of best_price().

    Arb exists when 1/dec_home + 1/dec_away < 1. Returns the stake split,
    locked profit, and ROI; None when no arb.
    """
    h = (best or {}).get("home")
    a = (best or {}).get("away")
    if not h or not a:
        return None
    dh, da = h["decimal"], a["decimal"]
    if dh <= 1 or da <= 1:
        return None
    inv_sum = 1.0 / dh + 1.0 / da
    if inv_sum >= 1.0:
        return None
    stake_home = stake * (1.0 / dh) / inv_sum
    stake_away = stake * (1.0 / da) / inv_sum
    payout = stake_home * dh                  # equal payout both sides
    profit = payout - stake
    return {
        "stake_home": round(stake_home, 2),
        "stake_away": round(stake_away, 2),
        "guaranteed_payout": round(payout, 2),
        "profit": round(profit, 2),
        "roi_pct": round(100.0 * profit / stake, 3),
    }


def sharp_movement(open_book: dict, current_book: dict) -> dict:
    """How far has the implied probability moved since the open?

    Positive home_drift_pct means the home implied prob went UP (line moved
    against home backers, i.e. home favourite price shortened).
    """
    o = model.devig_two_way(open_book.get("home_moneyline"),
                            open_book.get("away_moneyline"))
    c = model.devig_two_way(current_book.get("home_moneyline"),
                            current_book.get("away_moneyline"))
    if not o or not c:
        return {"home_drift_pct": 0.0, "away_drift_pct": 0.0}
    return {
        "home_drift_pct": round(100.0 * (c["home"] - o["home"]), 2),
        "away_drift_pct": round(100.0 * (c["away"] - o["away"]), 2),
        "open_home": round(o["home"], 4),
        "close_home": round(c["home"], 4),
    }


def closing_line_value(your_prob: float, closing_market_prob: float) -> dict:
    """CLV = the edge you locked in relative to the closing line.

    Positive CLV is the long-run profitable signal in sports betting.
    """
    if your_prob is None or closing_market_prob is None:
        return {"clv_pct": 0.0}
    return {
        "your_prob": round(your_prob, 4),
        "closing_prob": round(closing_market_prob, 4),
        "clv_pct": round(100.0 * (your_prob - closing_market_prob), 2),
    }


def hold_pct(books: Sequence[dict]) -> float:
    """Average book hold (vig %) across the supplied books."""
    holds = []
    for b in books:
        m = model.devig_two_way(b.get("home_moneyline"),
                                b.get("away_moneyline"))
        if m:
            holds.append(m["margin_pct"])
    return round(sum(holds) / len(holds), 2) if holds else 0.0
