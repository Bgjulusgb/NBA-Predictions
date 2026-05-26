"""Monte Carlo simulators for the NBA prediction pipeline.

Three layers, all reading existing model probabilities so a single config can
drive game-, series- and possession-level views:

  1. simulate_game        - margin distribution via a normal model
  2. simulate_series      - best-of-7 from any current state (uses real per-game probs)
  3. simulate_possession  - possession-by-possession Poisson + 3pt mixture model
  4. simulate_alt_lines   - over/under and alternate-spread cover probabilities
  5. simulate_team_score  - per-team total score distribution
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Sequence

from . import advanced_math, model

_RNG = random.Random()


def seed(s: int | None) -> None:
    _RNG.seed(s)
    advanced_math.seed(s)


# ---------------------------------------------------------------------------
def simulate_game(home_win_prob: float, sigma: float = 12.0,
                  trials: int = 10000) -> dict:
    """Sample margin outcomes consistent with a given home-win probability.

    Computes the expected margin from the win probability via the inverse
    normal CDF then simulates around it. Returns aggregate stats + histogram.
    """
    if not 0 < home_win_prob < 1:
        home_win_prob = max(0.01, min(0.99, home_win_prob))
    mean_margin = sigma * model._inv_norm_cdf(home_win_prob)
    margins = [_RNG.gauss(mean_margin, sigma) for _ in range(trials)]
    home_wins = sum(1 for m in margins if m > 0)
    # 10-point-wide histogram from -40..+40, plus tails.
    bins = list(range(-40, 41, 5))
    hist = []
    for low, high in zip(bins[:-1], bins[1:]):
        hist.append({
            "low": low, "high": high,
            "count": sum(1 for m in margins if low <= m < high),
        })
    return {
        "trials": trials,
        "implied_mean_margin": round(mean_margin, 2),
        "home_win_prob_sim": round(home_wins / trials, 4),
        "median_margin": round(advanced_math.percentile(margins, 50), 2),
        "p10_margin": round(advanced_math.percentile(margins, 10), 2),
        "p90_margin": round(advanced_math.percentile(margins, 90), 2),
        "stdev_margin": round(statistics.pstdev(margins), 2),
        "cover_minus_3": round(sum(1 for m in margins if m > 3) / trials, 4),
        "cover_plus_3": round(sum(1 for m in margins if m > -3) / trials, 4),
        "histogram": hist,
    }


# ---------------------------------------------------------------------------
def simulate_series(per_game_p: Sequence[float], leader_wins: int,
                     trailer_wins: int, trials: int = 20000) -> dict:
    """Simulate a best-of-7 remainder using per-game leader-win probabilities."""
    return advanced_math.monte_carlo_series(per_game_p, leader_wins,
                                            trailer_wins, trials)


# ---------------------------------------------------------------------------
def simulate_team_score(team_pace: float, team_eff: float,
                         opp_eff: float, league_eff: float = 113.0,
                         trials: int = 5000) -> dict:
    """Sample team final scores using a Poisson points model.

    team_pace: estimated possessions for the team. team_eff/opp_eff/league_eff
    are offensive ratings (points per 100 possessions). Lambda = pace * (own +
    opp) / 2 / league_eff * league_eff / 100 = pace * (own + opp) / 200.
    """
    expected = team_pace * (team_eff + (200.0 - opp_eff)) / 200.0
    # Add a small over-dispersion: scale lambda by per-trial gamma noise.
    scores = []
    for _ in range(trials):
        noise = _RNG.gauss(1.0, 0.05)
        lam = max(40.0, expected * noise)
        # Approximate Poisson via normal for large lambda; faster than summing
        # exponentials and accurate when lambda is in NBA territory (80-130).
        s = _RNG.gauss(lam, math.sqrt(lam))
        scores.append(round(s))
    return {
        "trials": trials,
        "mean_score": round(statistics.fmean(scores), 2),
        "p10": advanced_math.percentile(scores, 10),
        "p50": advanced_math.percentile(scores, 50),
        "p90": advanced_math.percentile(scores, 90),
    }


# ---------------------------------------------------------------------------
def simulate_alt_lines(home_win_prob: float, total: float = 220.0,
                       sigma_total: float = 18.0, sigma_margin: float = 12.0,
                       trials: int = 10000) -> dict:
    """Probabilities for a grid of alternate spread + total lines."""
    mean_margin = sigma_margin * model._inv_norm_cdf(max(0.01, min(0.99, home_win_prob)))
    margins = [_RNG.gauss(mean_margin, sigma_margin) for _ in range(trials)]
    totals = [_RNG.gauss(total, sigma_total) for _ in range(trials)]
    spreads = [-10, -6.5, -3.5, -1.5, 1.5, 3.5, 6.5, 10]
    spread_grid = {
        s: round(sum(1 for m in margins if m > s) / trials, 4) for s in spreads
    }
    totals_grid = {
        f"over_{int(total - 10)}": round(sum(1 for t in totals if t > total - 10) / trials, 4),
        f"over_{int(total - 5)}": round(sum(1 for t in totals if t > total - 5) / trials, 4),
        f"over_{int(total)}": round(sum(1 for t in totals if t > total) / trials, 4),
        f"over_{int(total + 5)}": round(sum(1 for t in totals if t > total + 5) / trials, 4),
        f"over_{int(total + 10)}": round(sum(1 for t in totals if t > total + 10) / trials, 4),
    }
    return {
        "trials": trials,
        "spread_cover_prob": spread_grid,
        "total_over_prob": totals_grid,
        "implied_margin": round(mean_margin, 2),
    }


# ---------------------------------------------------------------------------
# 3-point shooting + free-throw mixture: each possession ends with the
# offence taking a 2pt, 3pt or FT. Used for a more textured live model than
# the normal margin one — exposes things like "how does a hot 3pt night
# change the win prob".
# ---------------------------------------------------------------------------
def simulate_possession(possessions: int = 100,
                         pace: float = 100.0,
                         home: dict | None = None,
                         away: dict | None = None,
                         trials: int = 1500) -> dict:
    """Possession-level mixture sim for one game.

    Each side's dict carries: efg (eff. FG%), three_rate, ft_rate, tov_rate,
    plus 2pt/3pt/FT scoring rates. Reasonable NBA defaults are filled in.
    """
    home = {**_DEFAULT_OFFENCE, **(home or {})}
    away = {**_DEFAULT_OFFENCE, **(away or {})}
    h_margins = []
    for _ in range(trials):
        hp, ap = 0, 0
        for _ in range(possessions):
            hp += _one_possession(home)
            ap += _one_possession(away)
        h_margins.append(hp - ap)
    home_wins = sum(1 for m in h_margins if m > 0)
    return {
        "trials": trials,
        "possessions_per_team": possessions,
        "pace": pace,
        "home_win_prob": round(home_wins / trials, 4),
        "mean_margin": round(statistics.fmean(h_margins), 2),
        "stdev_margin": round(statistics.pstdev(h_margins), 2),
        "p10_margin": round(advanced_math.percentile(h_margins, 10), 2),
        "p90_margin": round(advanced_math.percentile(h_margins, 90), 2),
    }


_DEFAULT_OFFENCE = {
    "three_rate": 0.40,        # share of FGA from beyond the arc
    "two_pct": 0.54,           # 2P%
    "three_pct": 0.37,         # 3P%
    "ft_rate": 0.22,           # FT attempts per FGA (proxied as poss-end share)
    "ft_pct": 0.78,
    "tov_rate": 0.135,
}


def _one_possession(side: dict) -> int:
    r = _RNG.random()
    if r < side["tov_rate"]:                        # turnover -> 0 points
        return 0
    r2 = _RNG.random()
    if r2 < side["ft_rate"]:                        # foul -> two FTs
        made = (1 if _RNG.random() < side["ft_pct"] else 0) + \
               (1 if _RNG.random() < side["ft_pct"] else 0)
        return made
    if _RNG.random() < side["three_rate"]:          # 3PA
        return 3 if _RNG.random() < side["three_pct"] else 0
    return 2 if _RNG.random() < side["two_pct"] else 0


# ---------------------------------------------------------------------------
def what_if_player_out(home_win_prob: float, lost_share: float = 0.18,
                       trials: int = 5000) -> dict:
    """Sensitivity: how does the win prob shift if a key player misses the game?

    Crude rule of thumb: removing a star (~18% of team value) shifts margin
    by ~3-4 points; that's modelled as a leftward shift of the margin dist.
    """
    sigma = 12.0
    base_margin = sigma * model._inv_norm_cdf(max(0.01, min(0.99, home_win_prob)))
    shift = -lost_share * 22.0   # 0.18 * 22 ≈ -4 points
    sims = [_RNG.gauss(base_margin + shift, sigma) for _ in range(trials)]
    new_wp = sum(1 for m in sims if m > 0) / trials
    return {
        "lost_share": lost_share,
        "estimated_margin_shift": round(shift, 2),
        "new_home_win_prob": round(new_wp, 4),
        "delta_win_prob": round(new_wp - home_win_prob, 4),
    }
