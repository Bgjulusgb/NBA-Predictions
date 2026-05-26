"""Backtesting / calibration metrics for the prediction models.

Lets you score predicted win probabilities against actual outcomes so the
ensemble weights and sentiment cap can be tuned empirically rather than guessed.

Outcome convention: 1 = home team won, 0 = home team lost.
Run:  python3 -m backend.run evaluate --results data/results.json
where results.json is a list of {"prob_home": 0.62, "home_won": 1, ...}.
"""

import math


def _clamp(p, eps=1e-9):
    return max(eps, min(1 - eps, p))


def brier_score(prob_home, home_won):
    """Squared error of the probability (lower is better, 0..1)."""
    return (prob_home - home_won) ** 2


def log_loss(prob_home, home_won):
    """Negative log-likelihood of the outcome (lower is better)."""
    p = _clamp(prob_home)
    return -(home_won * math.log(p) + (1 - home_won) * math.log(1 - p))


def evaluate(predictions):
    """Score a list of {'prob_home', 'home_won'} predictions.

    Returns aggregate Brier, log-loss, accuracy and a baseline comparison
    against always predicting 0.5 (skill = how much better than a coin flip).
    """
    rows = [r for r in predictions
            if r.get("prob_home") is not None and r.get("home_won") in (0, 1)]
    n = len(rows)
    if n == 0:
        return {"n": 0}

    brier = sum(brier_score(r["prob_home"], r["home_won"]) for r in rows) / n
    ll = sum(log_loss(r["prob_home"], r["home_won"]) for r in rows) / n
    correct = sum(1 for r in rows
                  if (r["prob_home"] >= 0.5) == bool(r["home_won"]))
    baseline_brier = sum(brier_score(0.5, r["home_won"]) for r in rows) / n

    return {
        "n": n,
        "brier": round(brier, 4),
        "log_loss": round(ll, 4),
        "accuracy": round(correct / n, 4),
        "baseline_brier": round(baseline_brier, 4),
        "skill_score": round(1 - brier / baseline_brier, 4) if baseline_brier else 0.0,
    }
