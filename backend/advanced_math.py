"""Advanced mathematics & statistics, pure Python stdlib.

This module deliberately re-implements everything from scratch so the project
stays keyless / dependency-free. It is the math heart of the prediction stack:

  * Probability:        Beta-Binomial Bayes updating, Poisson/Skellam, Brier
  * Regression:         OLS linear, logistic (gradient descent), ridge
  * Time series:        EWMA, SMA, WMA, autocorrelation, exponential smoothing
  * Statistics:         percentile, IQR, skewness, kurtosis, Pearson, Spearman
  * Ratings:            Glicko-2 incremental update
  * Sims:               Monte Carlo NBA game, series, possession Poisson model
  * Bets:               Kelly criterion (full + fractional), Sharpe / Sortino
  * Basketball:         Pythagorean expectation, Four Factors, pace adjustment
  * Resampling:         Bootstrap mean + percentile CI
  * Information:        Shannon entropy, KL divergence, mutual information

Everything is unit-tested in backend.tests.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence

# Module-local PRNG so callers can seed reproducibly without touching globals.
_RNG = random.Random()


def seed(s: int | None) -> None:
    """Seed the module-local PRNG (None -> non-deterministic)."""
    _RNG.seed(s)


# ===========================================================================
# 1. Bayesian Beta-Binomial updating
# ===========================================================================
def beta_update(alpha: float, beta: float, wins: int, losses: int):
    """Conjugate update for a Bernoulli rate under Beta(alpha, beta) prior.

    Returns (alpha', beta'). Mean of the posterior is alpha'/(alpha'+beta'),
    variance is alpha' * beta' / ((alpha'+beta')^2 * (alpha'+beta'+1)).
    """
    return (alpha + wins, beta + losses)


def beta_mean_var(alpha: float, beta: float):
    """Mean and variance of Beta(alpha, beta)."""
    s = alpha + beta
    mean = alpha / s if s else 0.5
    var = (alpha * beta) / (s * s * (s + 1)) if s and (s + 1) else 0.0
    return mean, var


def beta_credible_interval(alpha: float, beta: float, level: float = 0.95,
                           n: int = 4000):
    """Approximate central credible interval for Beta(alpha, beta).

    Uses the inverse-CDF via repeated bisection on the regularised incomplete
    beta isn't in stdlib; instead we Monte-Carlo sample with the PRNG and
    quantile, which is plenty accurate for n>=2000 and avoids extra deps.
    """
    if alpha <= 0 or beta <= 0:
        return (0.0, 1.0)
    samples = sorted(_RNG.betavariate(alpha, beta) for _ in range(n))
    lo = (1 - level) / 2
    hi = 1 - lo
    return (samples[int(lo * n)], samples[min(int(hi * n), n - 1)])


# ===========================================================================
# 2. Poisson / Skellam (scoring distribution: home_pts - away_pts)
# ===========================================================================
def poisson_pmf(k: int, lam: float) -> float:
    if k < 0 or lam < 0:
        return 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def skellam_pmf(k: int, lam1: float, lam2: float, terms: int = 80) -> float:
    """P(X1 - X2 = k) for independent X_i ~ Poisson(lam_i).

    Computed as e^(-(l1+l2)) * (l1/l2)^(k/2) * I_{|k|}(2*sqrt(l1 l2)).
    We approximate the modified Bessel function via its series:
        I_n(z) = sum_{m=0..inf} (z/2)^(2m+n) / (m! * (m+n)!)
    `terms` controls truncation; 80 is well above convergence for NBA scales.
    """
    if lam1 < 0 or lam2 < 0:
        return 0.0
    if lam1 == 0 and lam2 == 0:
        return 1.0 if k == 0 else 0.0
    n = abs(k)
    z = 2 * math.sqrt(lam1 * lam2)
    half = z / 2.0
    bessel = 0.0
    for m in range(terms):
        try:
            term = (half ** (2 * m + n)) / (math.factorial(m) * math.factorial(m + n))
        except OverflowError:
            break
        bessel += term
        if term < 1e-18 and m > n:
            break
    ratio = (lam1 / lam2) ** (k / 2.0) if lam2 > 0 else float("inf")
    return math.exp(-(lam1 + lam2)) * ratio * bessel


def skellam_win_prob(lam_home: float, lam_away: float, terms: int = 80) -> float:
    """P(home wins) = sum_{k>=1} P(X_h - X_a = k). Half ties go to home."""
    # NBA can't tie, but be safe: split a numerical tie mass.
    win = sum(skellam_pmf(k, lam_home, lam_away, terms) for k in range(1, 60))
    tie = skellam_pmf(0, lam_home, lam_away, terms)
    return win + 0.5 * tie


# ===========================================================================
# 3. Regression: OLS linear, logistic (gradient descent), ridge
# ===========================================================================
def linear_regression(xs: Sequence[float], ys: Sequence[float]):
    """Simple OLS y = a + b x. Returns dict(slope, intercept, r2)."""
    n = len(xs)
    if n < 2:
        return {"slope": 0.0, "intercept": ys[0] if ys else 0.0, "r2": 0.0}
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return {"slope": 0.0, "intercept": my, "r2": 0.0}
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def ridge_regression(xs: Sequence[float], ys: Sequence[float], lam: float = 1.0):
    """Univariate ridge with bias term. Returns dict(slope, intercept)."""
    if not xs:
        return {"slope": 0.0, "intercept": 0.0}
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) + lam
    slope = num / den if den else 0.0
    return {"slope": slope, "intercept": my - slope * mx}


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logistic_regression(
    rows: Sequence[Sequence[float]],
    ys: Sequence[int],
    lr: float = 0.05,
    epochs: int = 300,
    l2: float = 0.0,
):
    """Plain-Python logistic regression (gradient descent, mean cross-entropy).

    rows: list of feature vectors. ys: list of 0/1. A bias term is added
    automatically. Returns dict(weights, bias, log_loss).
    """
    if not rows:
        return {"weights": [], "bias": 0.0, "log_loss": 0.0}
    d = len(rows[0])
    w = [0.0] * d
    b = 0.0
    n = len(rows)
    last_loss = 0.0
    for _ in range(epochs):
        dw = [0.0] * d
        db = 0.0
        loss = 0.0
        for x, y in zip(rows, ys):
            z = b + sum(w[j] * x[j] for j in range(d))
            p = sigmoid(z)
            err = p - y
            for j in range(d):
                dw[j] += err * x[j]
            db += err
            # Clip for numerical safety in log:
            p_c = min(1 - 1e-12, max(1e-12, p))
            loss += -(y * math.log(p_c) + (1 - y) * math.log(1 - p_c))
        for j in range(d):
            w[j] -= lr * (dw[j] / n + l2 * w[j])
        b -= lr * (db / n)
        last_loss = loss / n
    return {"weights": w, "bias": b, "log_loss": last_loss}


def logistic_predict(weights: Sequence[float], bias: float,
                     x: Sequence[float]) -> float:
    return sigmoid(bias + sum(w * v for w, v in zip(weights, x)))


# ===========================================================================
# 4. Time series helpers
# ===========================================================================
def sma(values: Sequence[float], window: int) -> list[float]:
    """Simple moving average. Result aligned to the right (length=len-window+1)."""
    if window <= 0 or len(values) < window:
        return []
    out = []
    s = sum(values[:window])
    out.append(s / window)
    for i in range(window, len(values)):
        s += values[i] - values[i - window]
        out.append(s / window)
    return out


def ewma(values: Sequence[float], alpha: float) -> list[float]:
    """Exponentially-weighted moving average with smoothing factor alpha in (0,1]."""
    if not values:
        return []
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def wma(values: Sequence[float], window: int) -> list[float]:
    """Weighted moving average: weights 1..window assigned to the window."""
    if window <= 0 or len(values) < window:
        return []
    weights = list(range(1, window + 1))
    denom = sum(weights)
    out = []
    for i in range(window - 1, len(values)):
        w_sum = sum(values[i - window + 1 + k] * weights[k] for k in range(window))
        out.append(w_sum / denom)
    return out


def double_exponential_smoothing(values: Sequence[float], alpha: float = 0.4,
                                  beta: float = 0.2) -> list[float]:
    """Holt's linear method. Returns smoothed levels (no forecast horizon)."""
    if not values:
        return []
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0.0
    out = [level]
    for v in values[1:]:
        new_level = alpha * v + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
        out.append(level)
    return out


def autocorrelation(values: Sequence[float], lag: int = 1) -> float:
    """Pearson autocorrelation at the given lag (0 if too short)."""
    n = len(values)
    if n <= lag or lag < 1:
        return 0.0
    mu = statistics.fmean(values)
    num = sum((values[i] - mu) * (values[i - lag] - mu) for i in range(lag, n))
    den = sum((v - mu) ** 2 for v in values)
    if den == 0:
        return 0.0
    return num / den


# ===========================================================================
# 5. Distribution moments + percentiles
# ===========================================================================
def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 100])."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (k - f) * (s[c] - s[f])


def iqr(values: Sequence[float]) -> float:
    """Interquartile range Q3 - Q1."""
    return percentile(values, 75) - percentile(values, 25)


def skewness(values: Sequence[float]) -> float:
    """Sample skewness (Fisher-Pearson, biased)."""
    n = len(values)
    if n < 3:
        return 0.0
    mu = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd == 0:
        return 0.0
    return sum(((v - mu) / sd) ** 3 for v in values) / n


def kurtosis(values: Sequence[float], excess: bool = True) -> float:
    """Sample kurtosis. If `excess`, subtracts 3 (normal-distribution baseline)."""
    n = len(values)
    if n < 4:
        return 0.0
    mu = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd == 0:
        return 0.0
    k = sum(((v - mu) / sd) ** 4 for v in values) / n
    return k - 3.0 if excess else k


def z_score(value: float, history: Sequence[float]) -> float:
    if not history:
        return 0.0
    mu = statistics.fmean(history)
    sd = statistics.pstdev(history)
    if sd == 0:
        return 0.0
    return (value - mu) / sd


# ===========================================================================
# 6. Correlation (Pearson + Spearman)
# ===========================================================================
def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _rank(values: Sequence[float]) -> list[float]:
    """Average rank (1-based) for tied values."""
    indexed = sorted(((v, i) for i, v in enumerate(values)), key=lambda t: t[0])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson_correlation(_rank(xs), _rank(ys))


# ===========================================================================
# 7. Glicko-2 (incremental player/team strength + uncertainty)
# ===========================================================================
GLICKO_TAU = 0.5
GLICKO_SCALE = 173.7178            # convert rating <-> internal mu
GLICKO_BASE = 1500.0


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def glicko2_update(rating: float, rd: float, vol: float,
                   opponents: Sequence[tuple[float, float, float]],
                   tau: float = GLICKO_TAU):
    """Glicko-2 rating update.

    opponents: list of (opp_rating, opp_rd, score) with score in {0, 0.5, 1}.
    Returns (new_rating, new_rd, new_vol).
    """
    if not opponents:
        # Inactivity step: RD increases by sqrt(rd^2 + vol^2).
        phi = rd / GLICKO_SCALE
        new_phi = math.sqrt(phi * phi + vol * vol)
        return (rating, new_phi * GLICKO_SCALE, vol)

    mu = (rating - GLICKO_BASE) / GLICKO_SCALE
    phi = rd / GLICKO_SCALE

    v_inv = 0.0
    delta_acc = 0.0
    for opp_r, opp_rd, s in opponents:
        mu_j = (opp_r - GLICKO_BASE) / GLICKO_SCALE
        phi_j = opp_rd / GLICKO_SCALE
        g_j = _g(phi_j)
        E_j = _E(mu, mu_j, phi_j)
        v_inv += g_j * g_j * E_j * (1 - E_j)
        delta_acc += g_j * (s - E_j)
    v = 1.0 / v_inv if v_inv > 0 else 1e9
    delta = v * delta_acc

    # Volatility update (Mark-Glickman, Illinois method).
    a = math.log(vol * vol)

    def f(x: float) -> float:
        e = math.exp(x)
        return (e * (delta * delta - phi * phi - v - e) /
                (2 * (phi * phi + v + e) ** 2) - (x - a) / (tau * tau))

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau
    fa, fb = f(A), f(B)
    for _ in range(50):
        if abs(B - A) < 1e-6:
            break
        C = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if fc * fb <= 0:
            A, fa = B, fb
        else:
            fa /= 2
        B, fb = C, fc
    new_vol = math.exp(A / 2)

    phi_star = math.sqrt(phi * phi + new_vol * new_vol)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = mu + new_phi * new_phi * delta_acc

    return (new_mu * GLICKO_SCALE + GLICKO_BASE,
            new_phi * GLICKO_SCALE, new_vol)


def glicko2_win_prob(home_rating: float, home_rd: float,
                     away_rating: float, away_rd: float,
                     home_court: float = 50.0) -> float:
    """Predicted P(home wins) under Glicko-2, with optional rating-point HCA."""
    mu_h = (home_rating + home_court - GLICKO_BASE) / GLICKO_SCALE
    mu_a = (away_rating - GLICKO_BASE) / GLICKO_SCALE
    phi_a = away_rd / GLICKO_SCALE
    return _E(mu_h, mu_a, phi_a)


# ===========================================================================
# 8. Monte Carlo: game margin + series simulator
# ===========================================================================
def monte_carlo_game(mean_margin_home: float, sigma: float = 12.0,
                     trials: int = 10000) -> dict:
    """Sample game outcomes from a normal margin model.

    Returns aggregate stats: home_win_prob, mean/std/percentiles of margin.
    """
    wins = 0
    margins = []
    for _ in range(trials):
        m = _RNG.gauss(mean_margin_home, sigma)
        margins.append(m)
        if m > 0:
            wins += 1
    return {
        "trials": trials,
        "home_win_prob": wins / trials,
        "mean_margin": statistics.fmean(margins),
        "std_margin": statistics.pstdev(margins),
        "p5_margin": percentile(margins, 5),
        "p50_margin": percentile(margins, 50),
        "p95_margin": percentile(margins, 95),
    }


def monte_carlo_series(per_game_p: Sequence[float], leader_wins: int,
                        trailer_wins: int, trials: int = 10000) -> dict:
    """Simulate the remainder of a best-of-7 from current series state.

    per_game_p: P(leader wins game k) for each remaining game in order.
    Returns dict with leader_clinch_prob, expected games, distribution.
    """
    target = 4
    leader_total = 0
    game_count = {4: 0, 5: 0, 6: 0, 7: 0}
    for _ in range(trials):
        l, t = leader_wins, trailer_wins
        for p in per_game_p:
            if l >= target or t >= target:
                break
            if _RNG.random() < p:
                l += 1
            else:
                t += 1
        if l >= target:
            leader_total += 1
        games_played = l + t
        if games_played in game_count:
            game_count[games_played] += 1
    expected_games = sum(g * c for g, c in game_count.items()) / trials \
        if any(game_count.values()) else 0.0
    return {
        "trials": trials,
        "leader_clinch_prob": leader_total / trials,
        "expected_games": round(expected_games, 3),
        "ends_in": {g: round(c / trials, 4) for g, c in game_count.items()},
    }


# ===========================================================================
# 9. Betting math: Kelly, fractional Kelly, Sharpe / Sortino
# ===========================================================================
def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Optimal Kelly stake fraction. Negative means no bet.

    f* = (b p - q) / b where b = decimal_odds - 1, q = 1 - p.
    """
    if decimal_odds is None or decimal_odds <= 1 or prob is None:
        return 0.0
    b = decimal_odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    return max(0.0, f)


def fractional_kelly(prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """A safer Kelly variant (typically 1/4 or 1/2 Kelly)."""
    return fraction * kelly_fraction(prob, decimal_odds)


def expected_value(prob: float, decimal_odds: float) -> float:
    """EV per unit stake: p*(odds-1) - (1-p)."""
    if decimal_odds is None or prob is None:
        return 0.0
    return prob * (decimal_odds - 1) - (1 - prob)


def sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free for r in returns]
    sd = statistics.pstdev(excess)
    if sd == 0:
        return 0.0
    return statistics.fmean(excess) / sd


def sortino_ratio(returns: Sequence[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free for r in returns]
    downside = [min(0.0, r) for r in excess]
    sd = math.sqrt(sum(d * d for d in downside) / len(downside))
    if sd == 0:
        return 0.0
    return statistics.fmean(excess) / sd


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (0..1)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = max(dd, (peak - v) / peak)
    return dd


# ===========================================================================
# 10. Basketball-specific math
# ===========================================================================
def pythagorean_expectation(points_for: float, points_against: float,
                             exponent: float = 13.91) -> float:
    """Pythagorean win expectation (Daryl Morey's basketball exponent ~13.91)."""
    if points_for <= 0 and points_against <= 0:
        return 0.5
    pf = points_for ** exponent
    pa = points_against ** exponent
    return pf / (pf + pa)


def four_factors(box: dict) -> dict:
    """Dean Oliver's Four Factors from a box-score dict.

    Required keys: fgm, fga, fg3m, fta, tov, orb, opp_drb, ft.
    Returns: efg, tov_rate, orb_rate, ft_rate.
    """
    fgm = box.get("fgm", 0)
    fga = max(1, box.get("fga", 0))
    fg3m = box.get("fg3m", 0)
    tov = box.get("tov", 0)
    fta = box.get("fta", 0)
    orb = box.get("orb", 0)
    opp_drb = box.get("opp_drb", 0)
    ft = box.get("ft", 0)
    efg = (fgm + 0.5 * fg3m) / fga
    possessions = fga + 0.44 * fta + tov
    tov_rate = tov / max(1, possessions)
    orb_rate = orb / max(1, orb + opp_drb)
    ft_rate = ft / fga
    return {
        "efg": round(efg, 4),
        "tov_rate": round(tov_rate, 4),
        "orb_rate": round(orb_rate, 4),
        "ft_rate": round(ft_rate, 4),
    }


def estimated_possessions(fga: float, fta: float, orb: float, tov: float) -> float:
    """Dean Oliver's possession estimate."""
    return fga - orb + tov + 0.44 * fta


def offensive_rating(points: float, possessions: float) -> float:
    if possessions <= 0:
        return 0.0
    return 100.0 * points / possessions


def pace(possessions: float, minutes: float = 48.0) -> float:
    if minutes <= 0:
        return 0.0
    return 48.0 * possessions / minutes


# ===========================================================================
# 11. Resampling: bootstrap mean + CI
# ===========================================================================
def bootstrap_mean_ci(values: Sequence[float], trials: int = 2000,
                      level: float = 0.95) -> dict:
    """Percentile bootstrap CI for the sample mean."""
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    n = len(values)
    boot_means = []
    for _ in range(trials):
        sample = [values[_RNG.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.fmean(sample))
    lo = (1 - level) / 2 * 100
    hi = (1 + level) / 2 * 100
    return {
        "mean": statistics.fmean(values),
        "low": percentile(boot_means, lo),
        "high": percentile(boot_means, hi),
    }


# ===========================================================================
# 12. Information theory
# ===========================================================================
def shannon_entropy(probs: Sequence[float]) -> float:
    """H(p) = -sum p_i log2 p_i (zeros skipped)."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q). Both must be valid probability vectors of equal length."""
    if len(p) != len(q):
        raise ValueError("p, q must have the same length")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        if qi <= 0:
            return float("inf")
        total += pi * math.log2(pi / qi)
    return total


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Symmetric, bounded distance based on KL: sqrt of JS divergence."""
    if len(p) != len(q):
        raise ValueError("p, q must have the same length")
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ===========================================================================
# 13. Markov chain (momentum states: home_hot / neutral / away_hot)
# ===========================================================================
def stationary_distribution(transition: Sequence[Sequence[float]],
                            iterations: int = 200) -> list[float]:
    """Power-iteration stationary distribution of a row-stochastic matrix."""
    n = len(transition)
    if n == 0:
        return []
    pi = [1.0 / n] * n
    for _ in range(iterations):
        new = [0.0] * n
        for i in range(n):
            for j in range(n):
                new[j] += pi[i] * transition[i][j]
        s = sum(new)
        if s > 0:
            new = [v / s for v in new]
        if max(abs(a - b) for a, b in zip(pi, new)) < 1e-9:
            return new
        pi = new
    return pi


def fit_momentum_chain(states: Sequence[int], n_states: int = 3) -> dict:
    """Estimate row-stochastic transition matrix from an integer state sequence."""
    if not states:
        return {"transition": [[1.0 / n_states] * n_states for _ in range(n_states)],
                "counts": [[0] * n_states for _ in range(n_states)]}
    counts = [[0] * n_states for _ in range(n_states)]
    for a, b in zip(states[:-1], states[1:]):
        if 0 <= a < n_states and 0 <= b < n_states:
            counts[a][b] += 1
    trans = []
    for row in counts:
        s = sum(row)
        trans.append([c / s for c in row] if s else [1.0 / n_states] * n_states)
    return {"transition": trans, "counts": counts}
