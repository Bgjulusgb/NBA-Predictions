"""Advanced player- and team-level statistics.

Implements standard basketball efficiency metrics from box-score-style inputs:

  * effective_fg_pct    - (FGM + 0.5*3PM)/FGA
  * true_shooting_pct   - PTS / (2*(FGA + 0.44*FTA))
  * usage_rate          - share of possessions a player ends
  * game_score          - Hollinger's quick game-impact score
  * player_efficiency   - simplified PER (no league averages required)
  * box_plus_minus_lite - approximate BPM from raw box numbers
  * versus_average      - z-score vs a reference distribution

All inputs use dict-of-numbers; missing keys default to zero so callers don't
need every stat to use the helpers they care about.
"""

from __future__ import annotations

from typing import Iterable

from . import advanced_math


def _g(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    return float(v) if v is not None else default


def effective_fg_pct(box: dict) -> float:
    fga = _g(box, "fga")
    if fga <= 0:
        return 0.0
    return (_g(box, "fgm") + 0.5 * _g(box, "fg3m")) / fga


def true_shooting_pct(box: dict) -> float:
    denom = 2.0 * (_g(box, "fga") + 0.44 * _g(box, "fta"))
    if denom <= 0:
        return 0.0
    return _g(box, "pts") / denom


def usage_rate(player: dict, team: dict) -> float:
    """Approximate USG%: how often the player ends a team possession when on floor.

    Standard formula: 100 * ((FGA + 0.44*FTA + TOV) * (Team_MP/5)) /
                     (Min * (Team_FGA + 0.44*Team_FTA + Team_TOV)).
    """
    p_acts = _g(player, "fga") + 0.44 * _g(player, "fta") + _g(player, "tov")
    t_acts = _g(team, "fga") + 0.44 * _g(team, "fta") + _g(team, "tov")
    t_mp = _g(team, "mp", 240.0)
    p_mp = _g(player, "mp")
    if t_acts <= 0 or p_mp <= 0:
        return 0.0
    return 100.0 * (p_acts * (t_mp / 5.0)) / (p_mp * t_acts)


def assist_rate(player: dict, team: dict) -> float:
    """Assist rate: % of teammate FGM the player assisted while on floor."""
    t_fgm = _g(team, "fgm")
    p_fgm = _g(player, "fgm")
    p_mp = _g(player, "mp")
    t_mp = _g(team, "mp", 240.0)
    if t_mp <= 0 or p_mp <= 0:
        return 0.0
    floor_share = (p_mp / (t_mp / 5.0))
    teammates_fgm = (t_fgm * floor_share) - p_fgm
    if teammates_fgm <= 0:
        return 0.0
    return 100.0 * _g(player, "ast") / teammates_fgm


def turnover_rate(player: dict) -> float:
    """TOV per 100 plays the player was involved in."""
    plays = _g(player, "fga") + 0.44 * _g(player, "fta") + _g(player, "tov")
    if plays <= 0:
        return 0.0
    return 100.0 * _g(player, "tov") / plays


def game_score(player: dict) -> float:
    """Hollinger's Game Score: a quick player-rating heuristic.

    Coefficients are the published Hollinger formula.
    """
    return (
        _g(player, "pts")
        + 0.4 * _g(player, "fgm")
        - 0.7 * _g(player, "fga")
        - 0.4 * (_g(player, "fta") - _g(player, "ftm"))
        + 0.7 * _g(player, "orb")
        + 0.3 * _g(player, "drb")
        + _g(player, "stl")
        + 0.7 * _g(player, "ast")
        + 0.7 * _g(player, "blk")
        - 0.4 * _g(player, "pf")
        - _g(player, "tov")
    )


def player_efficiency(player: dict) -> float:
    """Simplified PER (no league pace normalisation, no opp scaling).

    Approximation of Hollinger's PER per-minute weights:
      uPER = 1/MP * (3P + (2/3)*AST + (2 - factor*team_AST/team_FG)*FG
                    + FT*0.5*(1 + (1-team_AST/team_FG) + (2/3)*team_AST/team_FG)
                    - VOP*TO - VOP*DRBP*(FGA - FG) - VOP*0.44*(0.44 + 0.56*DRBP)*(FTA - FT)
                    + VOP*(1 - DRBP)*(TRB - ORB) + VOP*DRBP*ORB
                    + VOP*STL + VOP*DRBP*BLK - PF*(LgFT/LgPF - 0.44*LgFTA/LgPF*VOP))
    For zero-dependency context we use a constant-VOP approximation: weights
    tuned to land in 0-35 range for typical NBA box scores.
    """
    mp = _g(player, "mp")
    if mp <= 0:
        return 0.0
    raw = (
        _g(player, "fg3m") * 1.0
        + 0.67 * _g(player, "ast")
        + 1.0 * _g(player, "fgm")
        + 0.5 * _g(player, "ftm")
        + 0.7 * _g(player, "orb")
        + 0.3 * _g(player, "drb")
        + 1.0 * _g(player, "stl")
        + 0.9 * _g(player, "blk")
        - 1.0 * _g(player, "tov")
        - 0.4 * _g(player, "pf")
        - 0.7 * (_g(player, "fga") - _g(player, "fgm"))
        - 0.4 * (_g(player, "fta") - _g(player, "ftm"))
    )
    return round(15.0 * raw / mp, 2)


def box_plus_minus_lite(player: dict, team: dict) -> float:
    """Very-rough BPM proxy in [-15, +15] from box-only inputs.

    Real BPM regresses against on/off plus-minus which we don't have here.
    This proxy uses TS%, USG and turnover rate to keep it directional.
    """
    ts = true_shooting_pct(player)
    usg = usage_rate(player, team)
    tov_r = turnover_rate(player)
    # League-ish baselines: TS 0.56, USG 20.
    val = (
        100.0 * (ts - 0.56) * 0.20
        + (usg - 20.0) * 0.10
        + (_g(player, "ast") - 4.0) * 0.10
        + (_g(player, "stl") + _g(player, "blk") - 1.5) * 0.50
        - (tov_r - 12.0) * 0.05
    )
    return round(max(-15.0, min(15.0, val)), 2)


def versus_average(value: float, reference: Iterable[float]) -> dict:
    """Compare a single metric vs a reference distribution: z, percentile."""
    ref = list(reference)
    if not ref:
        return {"z": 0.0, "percentile": 50.0}
    z = advanced_math.z_score(value, ref)
    rank = sum(1 for v in ref if v <= value)
    return {"z": round(z, 3),
            "percentile": round(100.0 * rank / len(ref), 1)}


def project_player_line(history_lines: list[dict],
                         minute_pace: float | None = None) -> dict:
    """Project a player's stat line using recent-game EWMA.

    history_lines: list of {pts, reb, ast, stl, blk, mp, ...} in chrono order.
    Returns projection dict for the next game. EWMA so recent games matter more.
    """
    if not history_lines:
        return {}
    keys = ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga",
            "fg3m", "ftm", "fta", "mp"]
    out = {}
    for k in keys:
        series = [_g(line, k) for line in history_lines]
        smoothed = advanced_math.ewma(series, alpha=0.35)
        proj = smoothed[-1] if smoothed else 0.0
        # Optional minutes adjustment.
        if minute_pace and k != "mp" and _g(history_lines[-1], "mp") > 0:
            proj *= minute_pace / _g(history_lines[-1], "mp")
        out[k] = round(proj, 2)
    return out
