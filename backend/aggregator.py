"""Cross-source aggregator: merge data from multiple scrapers into one view.

When ESPN, Sofascore, Flashscore and TheScore all report a score, we keep all
of them, agree on a consensus value and surface disagreement (a 2-point
difference between ESPN and Sofascore = something is mid-update). Same idea
for odds (one row per book), play-by-play (deduped), lineups (most recent
confirmed wins) and form (median of W counts).

The pipeline calls these to populate snapshot['sources_unified'].
"""

from __future__ import annotations

import statistics
from typing import Iterable

# ---------------------------------------------------------------------------
# Score consensus across sources
# ---------------------------------------------------------------------------
def merge_scores(per_source: dict[str, dict]) -> dict:
    """Build a consensus score view from {source_name: game_dict}.

    Returns: {
      home_scores: [(source, score), ...],
      away_scores: [(source, score), ...],
      consensus_home, consensus_away,
      max_disagreement: largest point-difference between any two sources.
    }
    """
    home_pairs, away_pairs = [], []
    for name, game in per_source.items():
        if not game:
            continue
        h = (game.get("home") or {}).get("score")
        a = (game.get("away") or {}).get("score")
        if isinstance(h, int):
            home_pairs.append((name, h))
        if isinstance(a, int):
            away_pairs.append((name, a))

    def _consensus(pairs):
        if not pairs:
            return None
        vals = [v for _, v in pairs]
        # Median is robust to one stale source.
        return int(statistics.median(vals))

    def _disagreement(pairs):
        vals = [v for _, v in pairs]
        return max(vals) - min(vals) if len(vals) >= 2 else 0

    return {
        "home_scores": home_pairs,
        "away_scores": away_pairs,
        "consensus_home": _consensus(home_pairs),
        "consensus_away": _consensus(away_pairs),
        "max_disagreement": max(_disagreement(home_pairs), _disagreement(away_pairs)),
        "n_sources": len({n for n, _ in home_pairs + away_pairs}),
    }


# ---------------------------------------------------------------------------
# Multi-book odds aggregation
# ---------------------------------------------------------------------------
def merge_odds_books(*books_blocks: dict) -> dict:
    """Combine ESPN-style odds + Sofascore-style moneyline blocks.

    Each input can be either a single book dict {home_moneyline, away_moneyline}
    or a structured Sofascore moneyline block {moneyline: [{provider, ...}]}.
    Output: list of normalised books + best-price summary.
    """
    rows: list[dict] = []
    for block in books_blocks:
        if not block:
            continue
        # ESPN-style flat book.
        if "home_moneyline" in block:
            rows.append({
                "provider": block.get("provider") or "ESPN",
                "home_moneyline": block.get("home_moneyline"),
                "away_moneyline": block.get("away_moneyline"),
                "spread": block.get("spread"),
                "over_under": block.get("over_under"),
            })
            continue
        # Sofascore-style structured.
        for ml in block.get("moneyline", []) or []:
            rows.append({
                "provider": ml.get("provider"),
                "home_decimal": ml.get("home_decimal"),
                "away_decimal": ml.get("away_decimal"),
            })
    return {"books": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# Play-by-play merge with dedupe
# ---------------------------------------------------------------------------
def merge_pbp(*streams: Iterable[dict]) -> list[dict]:
    """Combine PBP streams from multiple sources. De-dupes by (period, clock, points, team).

    Sources sometimes lag — by ordering by (period DESC, clock DESC) we keep
    the freshest description for a duplicate event.
    """
    seen: dict[tuple, dict] = {}
    for stream in streams:
        for ev in (stream or []):
            key = (
                ev.get("period"),
                _clock_to_secs(ev.get("clock")),
                ev.get("points", 0),
                (ev.get("team") or "").upper(),
            )
            if key not in seen or len(ev.get("desc", "") or "") > len(
                    seen[key].get("desc", "") or ""):
                seen[key] = ev
    out = list(seen.values())
    # Sort: latest period first, lowest clock first within a period.
    out.sort(key=lambda e: (e.get("period") or 0,
                              -(_clock_to_secs(e.get("clock")) or 0)))
    return out


def _clock_to_secs(clock):
    if not clock:
        return None
    s = str(clock)
    if ":" in s:
        try:
            m, sec = s.split(":")
            return int(m) * 60 + int(float(sec))
        except (ValueError, TypeError):
            return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Lineup merge
# ---------------------------------------------------------------------------
def pick_lineup(*candidates: dict | None) -> dict | None:
    """Pick the most-complete lineup payload from multiple sources.

    Preference: confirmed > non-confirmed > most starters listed.
    """
    best = None
    best_score = -1
    for cand in candidates:
        if not cand:
            continue
        home = cand.get("home") or {}
        away = cand.get("away") or {}
        score = ((cand.get("confirmed") and 100 or 0)
                 + len(home.get("starters") or [])
                 + len(away.get("starters") or []))
        if score > best_score:
            best, best_score = cand, score
    return best


# ---------------------------------------------------------------------------
# Source health summary across the whole pipeline
# ---------------------------------------------------------------------------
def source_health(sources: list) -> dict:
    """Quick by-status counts the dashboard renders as a green/yellow/red dot grid."""
    counts = {"ok": 0, "partial": 0, "error": 0}
    by_name = []
    for s in sources:
        status = s.get("status") if isinstance(s, dict) else getattr(s, "status", "error")
        counts[status] = counts.get(status, 0) + 1
        name = s.get("name") if isinstance(s, dict) else getattr(s, "name", "?")
        by_name.append({"name": name, "status": status})
    return {
        "counts": counts,
        "uptime_pct": round(
            100.0 * counts.get("ok", 0) / max(1, sum(counts.values())), 1),
        "by_source": by_name,
    }
