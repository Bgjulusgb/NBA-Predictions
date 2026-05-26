"""Streak detection across scores, sentiment, runs and lead changes.

Pulls together a small zoo of pattern-finders the dashboard surfaces:

  * win_loss_streak       - current and longest W/L streaks from a result list
  * scoring_streak        - longest consecutive scoring stretch by one team
  * lead_change_count     - how many times the lead flipped in a score timeline
  * largest_run           - biggest single uninterrupted run + when it happened
  * sentiment_streak      - longest stretch of positive/negative sentiment
  * shooting_streak       - longest stretch of made FGs / missed FGs
  * comeback_distance     - biggest deficit overcome in a single game
"""

from __future__ import annotations

from typing import Sequence


def win_loss_streak(results: Sequence[str]) -> dict:
    """Analyse a list of single-char W/L results (most-recent first OR last).

    Treats input as chronological. Returns current_streak (signed: +n for wins,
    -n for losses), longest_win, longest_loss.
    """
    if not results:
        return {"current_streak": 0, "longest_win": 0, "longest_loss": 0}
    longest_w = longest_l = 0
    run_type = None
    run_len = 0
    for r in results:
        if r == "W":
            if run_type == "W":
                run_len += 1
            else:
                run_type = "W"
                run_len = 1
            longest_w = max(longest_w, run_len)
        elif r == "L":
            if run_type == "L":
                run_len += 1
            else:
                run_type = "L"
                run_len = 1
            longest_l = max(longest_l, run_len)
    sign = 1 if run_type == "W" else -1 if run_type == "L" else 0
    return {
        "current_streak": sign * run_len,
        "longest_win": longest_w,
        "longest_loss": longest_l,
    }


def scoring_streak(events: Sequence[dict]) -> dict:
    """Find the longest consecutive scoring stretch (one team only).

    Each event needs keys 'team' and 'points'. Returns the biggest such run
    found anywhere in the sequence, with its points total.
    """
    best = {"team": None, "points": 0}
    cur_team, cur_pts = None, 0
    for ev in events:
        pts = ev.get("points", 0)
        if pts <= 0:
            continue
        team = ev.get("team")
        if team == cur_team:
            cur_pts += pts
        else:
            cur_team, cur_pts = team, pts
        if cur_pts > best["points"]:
            best = {"team": cur_team, "points": cur_pts}
    return best


def largest_run(events: Sequence[dict]) -> dict:
    """Same as scoring_streak but also returns the run index range."""
    best = {"team": None, "points": 0, "start": None, "end": None}
    cur_team = None
    cur_pts = 0
    cur_start = 0
    for i, ev in enumerate(events):
        pts = ev.get("points", 0)
        if pts <= 0:
            continue
        team = ev.get("team")
        if team == cur_team:
            cur_pts += pts
        else:
            cur_team, cur_pts, cur_start = team, pts, i
        if cur_pts > best["points"]:
            best = {"team": cur_team, "points": cur_pts,
                    "start": cur_start, "end": i}
    return best


def lead_change_count(timeline: Sequence[dict]) -> dict:
    """Count lead changes and how often the game was tied.

    timeline: list of {home, away} score dicts in chronological order.
    """
    changes = ties = 0
    prev = 0
    for row in timeline:
        h = row.get("home", 0) or 0
        a = row.get("away", 0) or 0
        diff = h - a
        sign = 0 if diff == 0 else (1 if diff > 0 else -1)
        if diff == 0:
            ties += 1
        if prev != 0 and sign != 0 and sign != prev:
            changes += 1
        if sign != 0:
            prev = sign
    return {"lead_changes": changes, "tied_moments": ties}


def sentiment_streak(values: Sequence[float], threshold: float = 0.2) -> dict:
    """Longest positive / negative sentiment streak (above/below threshold)."""
    longest_pos = longest_neg = 0
    cur_pos = cur_neg = 0
    for v in values:
        if v >= threshold:
            cur_pos += 1
            cur_neg = 0
            longest_pos = max(longest_pos, cur_pos)
        elif v <= -threshold:
            cur_neg += 1
            cur_pos = 0
            longest_neg = max(longest_neg, cur_neg)
        else:
            cur_pos = cur_neg = 0
    return {"longest_positive": longest_pos, "longest_negative": longest_neg}


def shooting_streak(makes: Sequence[bool]) -> dict:
    """Longest hit / miss streak from a sequence of booleans."""
    hot = cold = 0
    cur_hot = cur_cold = 0
    for m in makes:
        if m:
            cur_hot += 1
            cur_cold = 0
            hot = max(hot, cur_hot)
        else:
            cur_cold += 1
            cur_hot = 0
            cold = max(cold, cur_cold)
    return {"longest_made": hot, "longest_missed": cold}


def comeback_distance(timeline: Sequence[dict]) -> dict:
    """Biggest deficit overcome (home and away separately)."""
    if not timeline:
        return {"home_max_deficit": 0, "away_max_deficit": 0,
                "home_came_back": False, "away_came_back": False}
    final_h = timeline[-1].get("home", 0) or 0
    final_a = timeline[-1].get("away", 0) or 0
    home_max_def = away_max_def = 0
    for row in timeline:
        h = row.get("home", 0) or 0
        a = row.get("away", 0) or 0
        if a - h > home_max_def:
            home_max_def = a - h
        if h - a > away_max_def:
            away_max_def = h - a
    return {
        "home_max_deficit": home_max_def,
        "away_max_deficit": away_max_def,
        "home_came_back": home_max_def > 0 and final_h > final_a,
        "away_came_back": away_max_def > 0 and final_a > final_h,
    }


def biggest_lead(timeline: Sequence[dict]) -> dict:
    """Largest lead either team held during the game."""
    home_max = away_max = 0
    for row in timeline:
        h = row.get("home", 0) or 0
        a = row.get("away", 0) or 0
        home_max = max(home_max, h - a)
        away_max = max(away_max, a - h)
    return {"home_biggest_lead": home_max, "away_biggest_lead": away_max}
