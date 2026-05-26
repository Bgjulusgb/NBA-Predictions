"""Prediction mathematics.

Several independent formulas, then an ensemble:

  1. Odds  -> implied probability + DE-VIG (remove bookmaker margin)
  2. Elo / log5 win probability (with home-court advantage)
  3. Sentiment-adjusted probability (bounded, never dominates the market)
  4. Heat / Hype / Toxicity meters (0..100)
  5. Live momentum + scoring-run detection (exponential time decay)
  6. Sentiment-spike detection (rolling z-score)
  7. Ensemble win probability + cross-source confidence
  8. Series-clinch probability given the 3-0 lead

Everything is pure Python (math/statistics only).

Heavier-weight math (Monte Carlo, Glicko-2, regression, Kelly, four factors,
bootstrap, KL/JS, Markov chains) lives in `advanced_math.py` and is composed
on top of these primitives by `pipeline.py`.
"""

import datetime as _dt
import math
import statistics

from . import config


def recency_weight(published, half_life_days=2.0, now=None):
    """Exponential time-decay weight in (0, 1] for a record's timestamp.

    Recent items count more. Missing/unparseable timestamps get a neutral 0.5.
    Accepts ISO8601 timestamps, plain date strings ("2026-05-25") and None.
    """
    if not published:
        return 0.5
    s = str(published).strip().replace("Z", "+00:00")
    t = None
    for parser in (_dt.datetime.fromisoformat,
                   lambda x: _dt.datetime.combine(_dt.date.fromisoformat(x),
                                                  _dt.time(0, 0))):
        try:
            t = parser(s)
            break
        except (ValueError, TypeError):
            continue
    if t is None:
        return 0.5
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    now = now or _dt.datetime.now(_dt.timezone.utc)
    age_days = max(0.0, (now - t).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)

# --- Elo seeds (refined by Basketball Reference SRS when available) --------
ELO_SEED = {"home": 1600.0, "away": 1612.0}   # CLE / NYK baseline
HOME_COURT_ELO = 100.0                         # ~ +100 Elo for the home side
ELO_PER_SRS = 28.0                             # Elo points per SRS point

# Sentiment is a bounded add-on to the ensemble (see MODEL_WEIGHTS below).
SENT_MAX_DELTA = 0.06                          # max +-6 pts from sentiment
SENT_K = 1.5                                   # sentiment differential gain


# ===========================================================================
# 1. Odds -> implied probability + de-vig
# ===========================================================================
def american_to_decimal(american):
    if american is None:
        return None
    a = float(american)
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def decimal_to_implied(decimal):
    if not decimal or decimal <= 1.0:
        return None
    return 1.0 / decimal


# NBA single-game margin standard deviation (points). Used to turn a point
# spread into a win probability via the normal CDF.
MARGIN_SIGMA = 12.0


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_from_spread(spread, sigma=MARGIN_SIGMA):
    """Favorite win probability implied by a point spread magnitude."""
    if spread is None:
        return None
    return _norm_cdf(abs(spread) / sigma)


def devig_two_way(home_ml, away_ml):
    """Remove the bookmaker margin from a two-way moneyline.

    Returns {'home', 'away', 'overround'} where probs sum to 1. Accepts
    American odds (the format ESPN returns).
    """
    dh = american_to_decimal(home_ml)
    da = american_to_decimal(away_ml)
    ph = decimal_to_implied(dh)
    pa = decimal_to_implied(da)
    if ph is None or pa is None:
        return None
    overround = ph + pa                  # > 1 because of the vig
    return {
        "home": ph / overround,
        "away": pa / overround,
        "overround": overround,
        "margin_pct": round((overround - 1.0) * 100, 2),
    }


# ===========================================================================
# 2. Elo / log5
# ===========================================================================
def elo_expected(rating_home, rating_away, home_court=HOME_COURT_ELO):
    """Expected score (= win probability) for the home team."""
    diff = (rating_home + home_court) - rating_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def log5(p_a, p_b):
    """log5: probability A beats B given each team's base win rate."""
    denom = p_a * (1 - p_b) + (1 - p_a) * p_b
    if denom == 0:
        return 0.5
    return (p_a * (1 - p_b)) / denom


ELO_PER_FORM_WIN = 16.0                        # Elo nudge per net win in last 5


def elo_from_history(bref_meta):
    """Refine Elo seeds using Basketball Reference SRS if present."""
    ratings = {"home": ELO_SEED["home"], "away": ELO_SEED["away"]}
    teams = (bref_meta or {}).get("teams", {})
    for side in ("home", "away"):
        srs = (teams.get(side) or {}).get("srs")
        if srs is not None:
            ratings[side] = 1500.0 + srs * ELO_PER_SRS
    return ratings


def elo_adjust_for_form(ratings, form):
    """Nudge Elo ratings by recent form (net wins in the last five games)."""
    if not form:
        return ratings
    out = dict(ratings)
    for side in ("home", "away"):
        f = form.get(side)
        if not f:
            continue
        net = f.get("wins", 0) - f.get("losses", 0)
        out[side] = out[side] + net * ELO_PER_FORM_WIN
    return out


# ===========================================================================
# 3. Sentiment-adjusted probability (bounded)
# ===========================================================================
def sentiment_delta(team_sentiment):
    """Bounded probability nudge for the home team from sentiment differential.

    team_sentiment: {'home': mean_compound, 'away': mean_compound} in [-1,1].
    Returns a delta in [-SENT_MAX_DELTA, +SENT_MAX_DELTA].
    """
    diff = team_sentiment.get("home", 0.0) - team_sentiment.get("away", 0.0)
    return SENT_MAX_DELTA * math.tanh(SENT_K * diff)


# ===========================================================================
# 4. Heat / Hype / Toxicity meters (0..100)
# ===========================================================================
def _saturate(x, scale):
    """Map [0, inf) -> [0, 100) with a saturating curve."""
    return 100.0 * (1.0 - math.exp(-x / scale))


def mood_meters(records):
    """Compute Heat Index, Hype Meter, Toxicity Meter from social/article recs."""
    if not records:
        return {"heat": 0.0, "hype": 0.0, "toxicity": 0.0, "volume": 0,
                "mean_sentiment": 0.0}

    comps = [r["sentiment"]["compound"] for r in records if "sentiment" in r]
    toxes = [r["sentiment"]["toxicity"] for r in records if "sentiment" in r]
    engagement = sum(max(0, r.get("engagement", 0)) for r in records)
    volume = len(records)

    mean_comp = statistics.fmean(comps) if comps else 0.0
    var_comp = statistics.pvariance(comps) if len(comps) > 1 else 0.0
    mean_tox = statistics.fmean(toxes) if toxes else 0.0

    # Heat = how much is being said + emotional spread (engagement + variance).
    # Scale chosen so a heavily-covered marquee game sits in the 70-90 band
    # rather than pinning at 100, keeping the meter discriminative.
    heat = _saturate(volume + engagement / 200.0 + var_comp * 60.0, scale=260.0)

    # Hype = mean positive emotional energy per item, with a bounded volume
    # bonus, so it reflects intensity rather than raw article count.
    n = len(records)
    pos_energy = sum(max(0.0, r["sentiment"]["compound"]) *
                     (1 + max(0, r.get("engagement", 0)) / 100.0)
                     for r in records if "sentiment" in r)
    mean_pos = pos_energy / n if n else 0.0
    hype = min(100.0, mean_pos * 130.0 + _saturate(volume, scale=400.0) * 0.35)

    # Toxicity = mean toxicity boosted by negative-sentiment density.
    neg_density = sum(1 for c in comps if c <= -0.35) / max(1, len(comps))
    toxicity = min(100.0, (mean_tox * 70.0) + neg_density * 60.0)

    return {
        "heat": round(heat, 1),
        "hype": round(hype, 1),
        "toxicity": round(toxicity, 1),
        "volume": volume,
        "engagement": engagement,
        "mean_sentiment": round(mean_comp, 4),
        "sentiment_variance": round(var_comp, 4),
    }


def team_sentiment(records):
    """Mean compound sentiment per team ('home'/'away'), engagement-weighted."""
    buckets = {"home": [], "away": []}
    weights = {"home": [], "away": []}
    for r in records:
        team = r.get("team")
        if team not in ("home", "away"):
            continue
        comp = r.get("sentiment", {}).get("compound", 0.0)
        # Weight by engagement AND recency so game-day chatter dominates.
        w = (1.0 + max(0, r.get("engagement", 0)) / 100.0) * \
            recency_weight(r.get("published"))
        buckets[team].append(comp * w)
        weights[team].append(w)

    out = {}
    for side in ("home", "away"):
        if buckets[side]:
            out[side] = round(sum(buckets[side]) / sum(weights[side]), 4)
        else:
            out[side] = 0.0
    out["count_home"] = len(buckets["home"])
    out["count_away"] = len(buckets["away"])
    return out


# ===========================================================================
# 5. Live momentum + run detection
# ===========================================================================
def detect_current_run(scoring_events):
    """Find the active scoring run from ordered scoring events.

    scoring_events: list of {'team': tricode, 'points': int} in time order.
    Returns {'team', 'points'} for the most recent uninterrupted run.
    """
    run_team = None
    run_points = 0
    for ev in reversed(scoring_events):
        if ev.get("points", 0) <= 0:
            continue
        team = ev.get("team")
        if run_team is None:
            run_team, run_points = team, ev["points"]
        elif team == run_team:
            run_points += ev["points"]
        else:
            break
    return {"team": run_team, "points": run_points}


def momentum(scoring_events, half_life=6):
    """Exponentially-weighted recent scoring differential (home minus away).

    Positive => home momentum. `half_life` is measured in scoring events.
    """
    decay = math.log(2) / half_life
    home_tri = config.GAME["home"]["abbr"]
    score = 0.0
    weight_sum = 0.0
    n = len(scoring_events)
    for i, ev in enumerate(scoring_events):
        pts = ev.get("points", 0)
        if pts <= 0:
            continue
        age = n - 1 - i
        w = math.exp(-decay * age)
        signed = pts if ev.get("team") == home_tri else -pts
        score += signed * w
        weight_sum += w * pts
    if weight_sum == 0:
        return 0.0
    return round(score / weight_sum, 4)        # in [-1, 1]


REGULATION_SECONDS = 48 * 60        # NBA regulation length
PERIOD_SECONDS = 12 * 60


def seconds_remaining(period, clock):
    """Approximate seconds left in regulation from period + 'M:SS' clock."""
    if not period:
        return REGULATION_SECONDS
    try:
        if ":" in str(clock):
            m, s = str(clock).split(":")
            left_in_period = int(m) * 60 + float(s)
        else:
            left_in_period = float(clock or 0)
    except (TypeError, ValueError):
        left_in_period = 0.0
    full_periods_left = max(0, 4 - period)        # periods after the current
    return max(0.0, full_periods_left * PERIOD_SECONDS + left_in_period)


def live_win_probability(margin_home, period, clock, pregame_home=0.5):
    """In-game home win probability from score margin + time remaining.

    Uses a normal model on the projected final margin: early in the game the
    estimate is anchored to the pre-game prior, late it is dominated by the
    current margin. Returns home win probability in (0, 1).
    """
    secs = seconds_remaining(period, clock)
    if secs <= 0:
        return 1.0 if margin_home > 0 else 0.0 if margin_home < 0 else pregame_home
    frac_left = secs / REGULATION_SECONDS
    # Pre-game prior expressed as an expected margin (inverse of the spread map).
    prior_margin = MARGIN_SIGMA * _inv_norm_cdf(min(0.999, max(0.001, pregame_home)))
    # Blend current margin with the prior, weighting the prior by time left.
    expected_final = margin_home + prior_margin * frac_left
    sigma = MARGIN_SIGMA * math.sqrt(max(frac_left, 1e-4))
    return round(_norm_cdf(expected_final / sigma), 4)


def _inv_norm_cdf(p):
    """Rational approximation of the inverse normal CDF (Acklam)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ===========================================================================
# 6. Sentiment-spike detection (rolling z-score over time buckets)
# ===========================================================================
def sentiment_spike(bucket_values):
    """z-score of the latest bucket vs the rolling history.

    bucket_values: chronological list of per-bucket metric (e.g. mean
    sentiment or volume). Returns z-score of the last bucket (0 if too few).
    """
    if len(bucket_values) < 3:
        return 0.0
    history = bucket_values[:-1]
    mu = statistics.fmean(history)
    sigma = statistics.pstdev(history)
    if sigma == 0:
        return 0.0
    return round((bucket_values[-1] - mu) / sigma, 3)


# ===========================================================================
# 7. Ensemble + confidence
# ===========================================================================
# Relative weight of each model. Only the models actually present are used,
# and their weights are renormalised so they always sum to 1.
# `power` is the TeamRankings power-rating signal (added when available).
MODEL_WEIGHTS = {"market": 0.40, "espn": 0.25, "elo": 0.20, "power": 0.15}


def ensemble(model_probs, sent_delta):
    """Weighted blend of any available models, then a bounded sentiment nudge.

    model_probs: {name: home_win_prob}. Recognised names are weighted by
    MODEL_WEIGHTS (unknown names get the smallest known weight). All
    probabilities are for the HOME team. Returns home/away + components.
    """
    components = {}
    weighted = 0.0
    weight_total = 0.0
    fallback_w = min(MODEL_WEIGHTS.values())
    for name, p in model_probs.items():
        if p is None:
            continue
        w = MODEL_WEIGHTS.get(name, fallback_w)
        weighted += w * p
        weight_total += w
        components[name] = round(p, 4)

    base = (weighted / weight_total) if weight_total else 0.5
    home = base + (sent_delta or 0.0)
    home = max(0.01, min(0.99, home))
    components["sentiment_delta"] = round(sent_delta or 0.0, 4)
    return {"home": round(home, 4), "away": round(1 - home, 4),
            "components": components}


def confidence(model_probs, n_sentiment):
    """0..100 confidence from model agreement + sentiment sample size.

    model_probs: {name: home_win_prob}. Agreement falls as the spread between
    the models widens.
    """
    preds = [p for p in model_probs.values() if p is not None]
    if len(preds) >= 2:
        spread = max(preds) - min(preds)
        agreement = max(0.0, 1.0 - spread / 0.25)     # 0 spread -> 1
    else:
        agreement = 0.5
    data_factor = min(1.0, n_sentiment / 40.0)
    return round(100.0 * (0.7 * agreement + 0.3 * data_factor), 1)


# ===========================================================================
# 8. Series-clinch probability (best-of-7, leader at 3-0)
# ===========================================================================
def elo_from_net_rating(ratings, pbp_team_stats, home_abbr, away_abbr):
    """Nudge Elo ratings by NetRating delta from pbpstats data.

    ~0.4 Elo points per NetRating point is calibrated so that a typical
    10-point NetRating advantage translates to ~4 Elo points, a modest but
    non-trivial signal relative to home-court (100 Elo).
    """
    home_nr = away_nr = None
    for row in (pbp_team_stats or []):
        abbr = (row.get("team") or "").upper()
        nr = row.get("net_rtg")
        if abbr == home_abbr.upper() and nr is not None:
            home_nr = float(nr)
        elif abbr == away_abbr.upper() and nr is not None:
            away_nr = float(nr)
    if home_nr is None or away_nr is None:
        return ratings
    net_delta = home_nr - away_nr
    nudge = net_delta * 0.4
    out = dict(ratings)
    out["home"] = out["home"] + nudge
    out["away"] = out["away"] - nudge
    return out


def apply_public_bet_nudge(conf, an_betting):
    """Blend Action Network public-money % into the confidence score.

    A heavy one-sided public-money tilt (sharp vs square divergence) slightly
    shifts confidence toward or away from the crowd's favourite.
    Bounded to ±3 confidence points.
    """
    if not an_betting:
        return conf
    away_money_pct = an_betting.get("away_public_money_pct")
    if away_money_pct is None:
        return conf
    nudge = 0.03 * (float(away_money_pct) - 50) / 100 * 100  # ±1.5 pts max
    nudge = max(-3.0, min(3.0, nudge))
    return round(conf + nudge, 1)


def series_clinch(leader_game_probs):
    """Probability the 3-0 leader wins the series.

    The leader needs ONE more win, so the trailer must win EVERY remaining
    game. leader_game_probs: list of P(leader wins) for each remaining game
    (already venue-adjusted). Series-win = 1 - P(trailer sweeps the rest).
    """
    p_trailer_wins_all = 1.0
    for p in leader_game_probs:
        p_trailer_wins_all *= (1 - p)
    return round(1 - p_trailer_wins_all, 4)
