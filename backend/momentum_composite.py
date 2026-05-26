"""Composite momentum index blending several signals.

Each signal already lives in the codebase; this module just combines them
into one number in [-1, +1] (positive = home momentum, negative = away).

Signals & default weights:

    scoring_momentum   0.40  (model.momentum: exp-weighted scoring diff)
    run_momentum       0.25  (current scoring run, normalised)
    sentiment_momentum 0.15  (rolling live sentiment z-score)
    lineup_momentum    0.10  (live plus/minus from starters on the floor)
    pace_momentum      0.10  (recent possessions ratio vs full-game baseline)

Weights re-normalise across whichever signals are actually present, so it
degrades gracefully when only a subset is available.
"""

from __future__ import annotations

import math
from typing import Iterable

WEIGHTS = {
    "scoring": 0.40,
    "run": 0.25,
    "sentiment": 0.15,
    "lineup": 0.10,
    "pace": 0.10,
}


def _norm(x: float, scale: float = 1.0) -> float:
    """Map any real number to [-1, 1] via tanh."""
    return math.tanh(x / max(scale, 1e-6))


def composite(*, scoring_momentum: float | None = None,
              current_run: dict | None = None,
              home_abbr: str | None = None,
              sentiment_zscore: float | None = None,
              starting_plus_minus_diff: float | None = None,
              pace_ratio: float | None = None) -> dict:
    """Return {value, weighted, components}.

    `value` is the bounded composite in [-1, +1]; `components` shows the raw
    + normalised contribution of each signal for the dashboard tooltip.
    """
    components: dict[str, dict] = {}
    weight_total = 0.0
    weighted = 0.0

    def _add(name: str, raw, normalised: float):
        nonlocal weight_total, weighted
        if raw is None:
            return
        w = WEIGHTS[name]
        weight_total += w
        weighted += w * normalised
        components[name] = {"raw": raw, "norm": round(normalised, 4),
                             "weight": w}

    # 1. Scoring momentum is already in [-1, 1].
    _add("scoring", scoring_momentum,
         max(-1, min(1, scoring_momentum or 0)) if scoring_momentum is not None else None)

    # 2. Current run normalised by 12-point scale, signed by team.
    run_val = None
    if current_run and home_abbr:
        team = current_run.get("team")
        pts = current_run.get("points") or 0
        signed = pts if team == home_abbr else -pts
        run_val = signed
        _add("run", signed, _norm(signed, scale=12.0))

    # 3. Sentiment z-score normalised by 2 sigma.
    if sentiment_zscore is not None:
        _add("sentiment", sentiment_zscore, _norm(sentiment_zscore, scale=2.0))

    # 4. Starter +/- diff (home - away), normalised by 25 (full game blow-out).
    if starting_plus_minus_diff is not None:
        _add("lineup", starting_plus_minus_diff,
             _norm(starting_plus_minus_diff, scale=25.0))

    # 5. Pace ratio: > 1 means home is generating possessions faster than its
    # season baseline (often a momentum tell).
    if pace_ratio is not None:
        _add("pace", pace_ratio, _norm(pace_ratio - 1.0, scale=0.20))

    value = weighted / weight_total if weight_total else 0.0
    return {
        "value": round(value, 4),
        "weight_total": round(weight_total, 4),
        "components": components,
    }
