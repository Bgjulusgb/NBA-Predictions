"""ANSI-coloured CLI dashboard.

Pure stdlib. Reads data/snapshot.json (and optionally data/live.json) and
renders an information-dense terminal layout: Mood Mirror, Press Review,
Prediction, Series, Players, Narratives, Live (when present), and the new
math additions (Glicko-2, Monte Carlo, Kelly, four-factors, simulation).

Run:
    python3 -m backend.run dashboard
    python3 -m backend.run dashboard --watch     # auto-refresh
    python3 -m backend.run dashboard --tab prediction
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Iterable

from . import config


# --- ANSI helpers ---------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDER = "\033[4m"
INV = "\033[7m"


def color(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}{RESET}"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def hex_color(rgb: tuple[int, int, int], text: str) -> str:
    if not _supports_color():
        return text
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}{RESET}"


def red(t): return color("31", t)
def green(t): return color("32", t)
def yellow(t): return color("33", t)
def blue(t): return color("34", t)
def magenta(t): return color("35", t)
def cyan(t): return color("36", t)
def white(t): return color("37", t)
def bold(t): return f"{BOLD}{t}{RESET}" if _supports_color() else t
def dim(t): return f"{DIM}{t}{RESET}" if _supports_color() else t
def under(t): return f"{UNDER}{t}{RESET}" if _supports_color() else t


def term_width() -> int:
    try:
        return max(60, shutil.get_terminal_size().columns)
    except Exception:
        return 80


# --- Rendering primitives -------------------------------------------------
def hline(char: str = "-", width: int | None = None) -> str:
    return (char * (width or term_width()))


def title_bar(text: str) -> str:
    w = term_width()
    pad = (w - len(text) - 4) // 2
    if pad < 0:
        return bold(text)
    return f"{cyan('=' * pad)} {bold(text)} {cyan('=' * pad)}"


def section(title: str) -> str:
    return bold(cyan(f"\n>> {title}"))


def kv(label: str, value, label_w: int = 22) -> str:
    return f"  {label.ljust(label_w)} {bold(str(value))}"


def progress_bar(value: float, scale: float = 100.0, width: int = 28,
                 colorfn=cyan) -> str:
    pct = max(0.0, min(1.0, value / scale))
    filled = int(round(pct * width))
    return "[" + colorfn("#" * filled) + dim("-" * (width - filled)) + f"] {value:>5.1f}"


def colored_pct(value: float) -> str:
    """Colour a probability/percentage by direction (negative / neutral / positive)."""
    if value >= 0.55:
        return green(f"{value*100:5.1f}%")
    if value <= 0.45:
        return red(f"{value*100:5.1f}%")
    return yellow(f"{value*100:5.1f}%")


def sentiment_arrow(value: float) -> str:
    if value > 0.1:
        return green("▲")
    if value < -0.1:
        return red("▼")
    return yellow("●")


# --- Tabs ------------------------------------------------------------------
def render_header(snap: dict) -> str:
    game = snap.get("game") or {}
    teams = snap.get("teams") or {}
    home = teams.get("home", {}).get("name", "Home")
    away = teams.get("away", {}).get("name", "Away")
    label = snap.get("label", "Game")
    mode = snap.get("mode", "?")
    when = snap.get("generated_at", "")[:19].replace("T", " ")
    venue = snap.get("venue", "")
    state = game.get("state", "?")

    line1 = bold(f"{away}  @  {home}")
    line2 = dim(f"{label}  ·  {venue}  ·  state={state}  ·  mode={mode}  ·  generated {when}")
    return f"\n{title_bar('NBA MOOD MIRROR — PREDICTION DASHBOARD')}\n{line1}\n{line2}"


def render_mood(snap: dict) -> str:
    mood = snap.get("mood", {})
    overall = mood.get("overall", {})
    ts = mood.get("team_sentiment", {})
    emos = mood.get("emotions", {})
    out = [section("Mood Mirror")]
    out.append(kv("Heat Index", progress_bar(overall.get("heat", 0))))
    out.append(kv("Hype Meter", progress_bar(overall.get("hype", 0), colorfn=green)))
    out.append(kv("Toxicity", progress_bar(overall.get("toxicity", 0), colorfn=red)))
    out.append(kv("Volume", overall.get("volume", 0)))
    out.append(kv("Mean sentiment", f"{overall.get('mean_sentiment', 0):+.4f}"))
    out.append("")
    h_arrow = sentiment_arrow(ts.get("home", 0))
    a_arrow = sentiment_arrow(ts.get("away", 0))
    home_name = snap["teams"]["home"]["name"]
    away_name = snap["teams"]["away"]["name"]
    out.append(kv(f"{home_name}", f"{h_arrow} {ts.get('home', 0):+.4f}  ({ts.get('count_home',0)} items)"))
    out.append(kv(f"{away_name}", f"{a_arrow} {ts.get('away', 0):+.4f}  ({ts.get('count_away',0)} items)"))
    if emos:
        out.append("")
        out.append(kv("Emotion mix", " · ".join(f"{k}={v:.2f}" for k, v in emos.items())))
    return "\n".join(out)


def render_prediction(snap: dict) -> str:
    pred = snap.get("prediction", {}) or {}
    out = [section("Prediction Ensemble")]
    ens = pred.get("ensemble", {})
    if ens:
        out.append(kv("Home win",
                      colored_pct(ens.get("home", 0)) + dim(f"   away={ens.get('away',0):.4f}")))
    market = pred.get("market") or {}
    if market:
        method = market.get("method", "")
        out.append(kv("Market (devig)",
                      f"{colored_pct(market.get('home', 0))}  "
                      f"{dim('overround=' + str(market.get('overround') or '?'))}"
                      f"{dim('  method=' + method)}"))
    elo = pred.get("elo") or {}
    if elo:
        out.append(kv("Elo / log5", colored_pct(elo.get("home", 0))))
        ratings = elo.get("ratings", {})
        if ratings:
            out.append(kv("Elo ratings",
                          f"home={ratings.get('home')}  away={ratings.get('away')}"))
    espn_p = pred.get("espn_predictor")
    if espn_p:
        out.append(kv("ESPN predictor", colored_pct(espn_p.get("home", 0))))
    out.append(kv("Sentiment delta", f"{pred.get('sentiment_delta', 0):+.4f}"))
    out.append(kv("Confidence", progress_bar(pred.get("confidence", 0))))

    vb = pred.get("value_bet")
    if vb:
        side = vb.get("side")
        if vb.get("has_value"):
            label = green(f"VALUE on {side.upper()}  edge={vb.get('edge_pct')}%")
        else:
            label = dim(f"no value (best {side}, edge={vb.get('edge_pct')}%)")
        out.append(kv("Value bet", f"{label}  ml={vb.get('moneyline')}  EV={vb.get('expected_value')}"))

    series = pred.get("series") or {}
    if series:
        out.append("")
        out.append(kv("Series leader", series.get("leader")))
        out.append(kv("Series lead", series.get("lead")))
        out.append(kv("Clinch probability",
                      progress_bar(100 * (series.get("leader_clinch_probability") or 0),
                                    colorfn=green)))
        per_game = series.get("per_game") or []
        if per_game:
            joined = "  ".join(
                f"G{g['game']}@{g['venue']}={g['leader_win']:.2f}" for g in per_game)
            out.append(kv("Per-game leader %", joined))
    # New advanced math block (added by pipeline as 'advanced').
    adv = snap.get("advanced") or {}
    if adv:
        out.append("")
        out.append(under(dim("advanced math (Monte Carlo / Kelly / Glicko)")))
        mc = adv.get("monte_carlo_game") or {}
        if mc:
            out.append(kv("MC home win", f"{mc.get('home_win_prob_sim'):.4f} "
                                          f"({mc.get('trials')} trials)"))
            out.append(kv("MC margin p10/50/90",
                          f"{mc.get('p10_margin'):+.1f} / {mc.get('median_margin'):+.1f} / {mc.get('p90_margin'):+.1f}"))
        mcs = adv.get("monte_carlo_series") or {}
        if mcs:
            ei = mcs.get("ends_in", {})
            out.append(kv("MC series clinch", f"{mcs.get('leader_clinch_prob'):.4f}"))
            out.append(kv("MC expected games", mcs.get("expected_games")))
            if ei:
                out.append(kv("MC ends-in", " · ".join(f"G{k}={v:.3f}" for k, v in ei.items())))
        kel = adv.get("kelly") or {}
        if kel:
            out.append(kv("Kelly fraction (full)", f"{kel.get('full', 0)*100:.2f}%"))
            out.append(kv("Kelly fraction (1/4)", f"{kel.get('quarter', 0)*100:.2f}%"))
        glk = adv.get("glicko") or {}
        if glk:
            out.append(kv("Glicko-2 home WP",
                          colored_pct(glk.get("home_win_prob", 0.5))))
    return "\n".join(out)


def render_players(snap: dict, top: int = 8) -> str:
    out = [section("Players (Buzz · Sentiment)")]
    players = snap.get("players") or []
    if not players:
        out.append(dim("  (no player chatter)"))
        return "\n".join(out)
    home_name = snap["teams"]["home"]["name"]
    away_name = snap["teams"]["away"]["name"]
    out.append(f"  {bold('PLAYER'.ljust(28))}{bold('TEAM'.ljust(20))}"
               f"{bold('MENT.'.rjust(6))}  {bold('SENT.'.rjust(8))}  {bold('BUZZ'.rjust(8))}")
    out.append("  " + dim("-" * 72))
    for p in players[:top]:
        team = home_name if p["team"] == "home" else away_name
        s = p["mean_sentiment"]
        sentiment_str = green(f"{s:+.4f}") if s > 0 else red(f"{s:+.4f}") if s < 0 else yellow(f"{s:+.4f}")
        out.append(f"  {p['name'][:27].ljust(28)}{team[:19].ljust(20)}"
                   f"{str(p['mentions']).rjust(6)}  {sentiment_str.rjust(8)}  "
                   f"{p['buzz']:>8.2f}")
    return "\n".join(out)


def render_narratives(snap: dict, top: int = 10) -> str:
    out = [section("Trending Narratives")]
    nar = snap.get("narratives") or []
    meta = snap.get("narrative_meta") or {}
    if meta:
        pol = meta.get("polarity") or {}
        out.append(kv("Concentration (HHI)", f"{meta.get('concentration', 0):.4f}  "
                                              f"(0=fragmented, 1=single story)"))
        out.append(kv("Polarity",
                      f"pos={pol.get('positive', 0):.2%}  "
                      f"neu={pol.get('neutral', 0):.2%}  "
                      f"neg={pol.get('negative', 0):.2%}"))
    if not nar:
        out.append(dim("  (no narratives yet)"))
        return "\n".join(out)
    out.append("")
    out.append(f"  {bold('TERM'.ljust(18))}{bold('COUNT'.rjust(8))}  {bold('MEAN SENT.'.rjust(12))}")
    out.append("  " + dim("-" * 46))
    for n in nar[:top]:
        s = n["mean_sentiment"]
        sent_str = green(f"{s:+.4f}") if s > 0 else red(f"{s:+.4f}") if s < 0 else yellow(f"{s:+.4f}")
        out.append(f"  {n['term'][:17].ljust(18)}{str(n['count']).rjust(8)}  {sent_str.rjust(12)}")
    return "\n".join(out)


def render_press(snap: dict, top: int = 10) -> str:
    out = [section("Press Review (top headlines)")]
    items = snap.get("press_review") or []
    if not items:
        out.append(dim("  (no articles)"))
        return "\n".join(out)
    for it in items[:top]:
        when = (it.get("published") or "")[:16]
        comp = it.get("sentiment", {}).get("compound", 0)
        sent_str = green(f"{comp:+.2f}") if comp > 0.05 else red(f"{comp:+.2f}") if comp < -0.05 else yellow(f"{comp:+.2f}")
        outlet = it.get("source", "?")[:14]
        title = (it.get("title") or "").strip()[:max(40, term_width() - 50)]
        out.append(f"  {dim(when):17}  {sent_str}  {dim(outlet):14}  {title}")
    return "\n".join(out)


def render_sources(snap: dict) -> str:
    out = [section("Source status")]
    for s in snap.get("sources", []):
        status = s.get("status")
        if status == "ok":
            badge = green("OK")
        elif status == "partial":
            badge = yellow("PART")
        else:
            badge = red("ERR")
        out.append(f"  {badge:8} {s.get('name'):20} count={s.get('count')}  "
                   f"{dim(s.get('error') or '')}")
    return "\n".join(out)


def render_live(live: dict, snap: dict | None = None) -> str:
    out = [section("Live (in-game)")]
    if not live or live.get("mode") != "live":
        out.append(dim("  (no live update yet)"))
        return "\n".join(out)
    game = live.get("game") or {}
    L = live.get("live") or {}
    home_name = snap["teams"]["home"]["name"] if snap else "Home"
    away_name = snap["teams"]["away"]["name"] if snap else "Away"
    hs = (game.get("home") or {}).get("score")
    as_ = (game.get("away") or {}).get("score")
    out.append(kv("Score", f"{away_name}  {as_}  @  {home_name}  {hs}"))
    out.append(kv("Period / clock", f"Q{game.get('period')}  {game.get('clock')}"))
    wp = L.get("win_probability") or {}
    if wp:
        out.append(kv("Live home WP", colored_pct(wp.get("home", 0.5))))
        out.append(kv("Pregame home WP", colored_pct(wp.get("pregame_home", 0.5))))
    run = L.get("current_run") or {}
    if run.get("team") and run.get("points", 0) > 0:
        out.append(kv("Current run", f"{run['team']} on {run['points']}-0"))
    out.append(kv("Momentum (home+)",
                  f"{L.get('momentum', 0):+.3f}"))
    out.append(kv("Sentiment spike (z)", f"{L.get('sentiment_spike', 0):+.2f}"))
    alerts = live.get("alerts") or []
    if alerts:
        out.append("")
        out.append(under(bold("Alerts")))
        for a in alerts:
            sev = a.get("severity", "low")
            text = a.get("text", "")
            line = bold(text) if sev == "high" else text
            badge = red("HIGH") if sev == "high" else yellow("MED") if sev == "medium" else dim("LOW")
            out.append(f"  {badge:>10}  {line}")
    return "\n".join(out)


def render_history(snap: dict, last_n: int = 12) -> str:
    rows = snap.get("history") or []
    if not rows:
        return ""
    out = [section("Recent history (movement)")]
    out.append(f"  {bold('TIME'.ljust(20))}{bold('MODE'.ljust(8))}"
               f"{bold('ENS').rjust(8)}  {bold('MKT').rjust(8)}  "
               f"{bold('ELO').rjust(8)}  {bold('HEAT').rjust(6)}  "
               f"{bold('TOX').rjust(6)}  {bold('CLINCH').rjust(8)}")
    for r in rows[-last_n:]:
        out.append(
            f"  {(r.get('ts') or '')[:19].ljust(20)}"
            f"{(r.get('mode') or '?').ljust(8)}"
            f"{(r.get('ens_home') or 0):>8.3f}  {(r.get('market_home') or 0):>8.3f}  "
            f"{(r.get('elo_home') or 0):>8.3f}  "
            f"{(r.get('heat') or 0):>6.1f}  {(r.get('toxicity') or 0):>6.1f}  "
            f"{(r.get('clinch') or 0):>8.3f}"
        )
    return "\n".join(out)


TABS = {
    "all": "complete dashboard",
    "mood": "mood meters + team sentiment",
    "prediction": "prediction + Monte Carlo + Kelly + Glicko",
    "press": "press review",
    "players": "player buzz",
    "narratives": "trending terms",
    "live": "live in-game view",
    "history": "history table",
    "sources": "source health",
    "categories": "topic category breakdown",
    "lineups": "lineups + missing players",
    "odds": "multi-book odds",
    "power": "power & advanced stats",
}


def render_categories(snap):
    out = [section("Topic Categories")]
    cats = (snap.get("categories") or {}).get("categories") or []
    if not cats:
        out.append(dim("  (no records to categorise yet)"))
        return "\n".join(out)
    out.append(kv("Total records", (snap.get("categories") or {}).get("total")))
    out.append("")
    out.append(f"  {bold('CATEGORY'.ljust(18))}{bold('COUNT'.rjust(7))}  {bold('MEAN SENT.'.rjust(12))}")
    out.append("  " + dim("-" * 42))
    for c in cats:
        s = c["mean_sentiment"]
        col = green(f"{s:+.4f}") if s > 0 else red(f"{s:+.4f}") if s < 0 else yellow(f"{s:+.4f}")
        out.append(f"  {c['category'].ljust(18)}{str(c['count']).rjust(7)}  {col.rjust(12)}")
    return "\n".join(out)


def render_lineups(snap):
    out = [section("Lineups")]
    L = snap.get("lineups_unified")
    meta = snap.get("lineup_meta") or {}
    if not L:
        out.append(dim("  (no lineups available; Sofascore usually publishes ~1h before tip)"))
        return "\n".join(out)
    adv = meta.get("starting_advantage") or {}
    if adv:
        out.append(kv("Starting value home", adv.get("home_starting_value")))
        out.append(kv("Starting value away", adv.get("away_starting_value")))
        out.append(kv("Net (home)", adv.get("net_advantage_home")))
    for side_key, label in (("home", snap["teams"]["home"]["name"]),
                              ("away", snap["teams"]["away"]["name"])):
        side = L.get(side_key)
        if not side:
            continue
        out.append("")
        out.append(bold(label))
        for p in (side.get("starters") or []):
            out.append(f"  {dim(p.get('position') or '?')}  {p.get('name') or '?'}")
        out.append(dim("  bench:"))
        for p in (side.get("bench") or [])[:5]:
            out.append(f"    {dim(p.get('position') or '?')}  {p.get('name') or '?'}")
        miss = (meta.get(f"{side_key}_missing") or {}).get("players") or []
        if miss:
            out.append(red("  missing:"))
            for m in miss:
                out.append(f"    {m.get('name')} — {m.get('type')} ({m.get('reason')})")
    return "\n".join(out)


def render_odds(snap):
    out = [section("Multi-book odds")]
    odds = snap.get("odds_unified") or {}
    books = odds.get("books") or []
    if not books:
        out.append(dim("  (no odds; Sofascore odds endpoint failed or no provider published)"))
        return "\n".join(out)
    out.append(kv("Books reporting", odds.get("count")))
    out.append("")
    out.append(f"  {bold('PROVIDER'.ljust(20))}{bold('HOME ML'.rjust(9))}  "
               f"{bold('AWAY ML'.rjust(9))}  {bold('H DEC'.rjust(8))}  "
               f"{bold('A DEC'.rjust(8))}  {bold('SPREAD'.rjust(8))}")
    for b in books:
        out.append(f"  {(b.get('provider') or '?')[:19].ljust(20)}"
                   f"{str(b.get('home_moneyline') or '—').rjust(9)}  "
                   f"{str(b.get('away_moneyline') or '—').rjust(9)}  "
                   f"{str(b.get('home_decimal') or '—').rjust(8)}  "
                   f"{str(b.get('away_decimal') or '—').rjust(8)}  "
                   f"{str(b.get('spread') or '—').rjust(8)}")
    return "\n".join(out)


def render_power_stats(snap: dict) -> str:
    """Power & Advanced Stats: standings, PBP ratings, public money, odds."""
    out = [section("Power & Advanced Stats")]
    teams = snap.get("teams") or {}
    home_name = teams.get("home", {}).get("name", "Home")
    away_name = teams.get("away", {}).get("name", "Away")
    home_abbr = teams.get("home", {}).get("abbr", "HME")
    away_abbr = teams.get("away", {}).get("abbr", "AWY")

    # --- NBA Standings (playoffs) -------------------------------------------
    v2 = snap.get("standings_v2") or []
    reg = snap.get("standings") or {}
    home_stand = next((r for r in v2 if r.get("abbr") == home_abbr), None)
    away_stand = next((r for r in v2 if r.get("abbr") == away_abbr), None)
    # Fall back to regular-season standings.
    if not home_stand:
        home_stand = reg.get("home")
    if not away_stand:
        away_stand = reg.get("away")

    if home_stand or away_stand:
        out.append("")
        out.append(under(bold("NBA Standings")))
        out.append(f"  {'TEAM'.ljust(26)}{'W-L'.rjust(6)}  "
                   f"{'L10'.rjust(6)}  {'STREAK'.rjust(8)}  "
                   f"{'PTS/G'.rjust(7)}  {'OPP/G'.rjust(7)}")
        out.append("  " + dim("-" * 68))
        for label, row in ((home_name, home_stand), (away_name, away_stand)):
            if not row:
                continue
            w = row.get("wins", "?")
            lo = row.get("losses", "?")
            l10 = row.get("l10") or row.get("last_10") or "—"
            streak = row.get("streak") or "—"
            pts = row.get("pts_pg") or row.get("points_for_pg") or "—"
            opp = row.get("opp_pts_pg") or row.get("points_against_pg") or "—"
            out.append(
                f"  {label[:25].ljust(26)}{str(w)+'-'+str(lo):>6}  "
                f"{str(l10):>6}  {str(streak):>8}  "
                f"{str(pts):>7}  {str(opp):>7}"
            )

    # --- PBP Stats (advanced ratings) ---------------------------------------
    pbp = snap.get("pbpstats") or []
    if pbp:
        out.append("")
        out.append(under(bold("Advanced Ratings (pbpstats)")))
        out.append(f"  {'TEAM'.ljust(26)}{'OFFRTG':>8}  {'DEFRTG':>8}  "
                   f"{'NETRTG':>8}  {'PACE':>8}")
        out.append("  " + dim("-" * 60))
        abbr_to_name = {home_abbr: home_name, away_abbr: away_name}
        for row in pbp:
            abbr = row.get("team", "")
            label = abbr_to_name.get(abbr, abbr)
            off = row.get("off_rtg")
            defe = row.get("def_rtg")
            net = row.get("net_rtg")
            pace = row.get("pace")
            net_str = f"{net:+.1f}" if net is not None else "—"
            net_col = (green(net_str) if net is not None and net > 0
                       else red(net_str) if net is not None and net < 0
                       else dim(net_str))
            out.append(
                f"  {label[:25].ljust(26)}"
                f"{(f'{off:.1f}' if off is not None else '—'):>8}  "
                f"{(f'{defe:.1f}' if defe is not None else '—'):>8}  "
                f"{net_col:>8}  "
                f"{(f'{pace:.1f}' if pace is not None else '—'):>8}"
            )

    # --- Public Bet % (Action Network) -------------------------------------
    an = snap.get("action_network") or {}
    if an:
        out.append("")
        out.append(under(bold("Public Betting (Action Network)")))
        hb = an.get("home_public_bet_pct")
        ab = an.get("away_public_bet_pct")
        hm = an.get("home_public_money_pct")
        am = an.get("away_public_money_pct")
        hml = an.get("home_ml")
        aml = an.get("away_ml")
        if hb is not None or ab is not None:
            out.append(kv("Bet tickets (home/away)",
                          f"{hb or '?':.1f}% / {ab or '?':.1f}%"
                          if (hb is not None and ab is not None)
                          else f"home={hb}  away={ab}"))
        if hm is not None or am is not None:
            out.append(kv("Money % (home/away)",
                          f"{hm or '?':.1f}% / {am or '?':.1f}%"
                          if (hm is not None and am is not None)
                          else f"home={hm}  away={am}"))
        if hml is not None or aml is not None:
            out.append(kv("Moneyline (home/away)",
                          f"{hml:+d} / {aml:+d}"
                          if (hml is not None and aml is not None)
                          else f"home={hml}  away={aml}"))

    # --- Multi-book odds (odds_scraper) ------------------------------------
    books = snap.get("odds_scraper") or []
    if not books:
        books = (snap.get("odds_unified") or {}).get("books") or []
    if books:
        out.append("")
        out.append(under(bold("Multi-book Odds")))
        out.append(f"  {'BOOK'.ljust(18)}{'HOME ML':>9}  {'AWAY ML':>9}")
        out.append("  " + dim("-" * 42))
        for b in books[:8]:
            bname = (b.get("book") or b.get("provider") or "?")[:17]
            hml = b.get("home_ml") or b.get("home_moneyline")
            aml = b.get("away_ml") or b.get("away_moneyline")
            out.append(
                f"  {bname.ljust(18)}"
                f"{(f'{hml:+d}' if hml else '—'):>9}  "
                f"{(f'{aml:+d}' if aml else '—'):>9}"
            )

    if len(out) == 1:
        out.append(dim("  (no advanced stats yet — sources still fetching)"))
    return "\n".join(out)


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def render(snap: dict, live: dict | None, tab: str) -> str:
    parts = [render_header(snap)]
    if tab in ("all", "mood"):
        parts.append(render_mood(snap))
    if tab in ("all", "prediction"):
        parts.append(render_prediction(snap))
    if tab in ("all", "players"):
        parts.append(render_players(snap))
    if tab in ("all", "narratives"):
        parts.append(render_narratives(snap))
    if tab in ("all", "press"):
        parts.append(render_press(snap))
    if tab in ("all", "live") and live:
        parts.append(render_live(live, snap))
    if tab in ("all", "history"):
        h = render_history(snap)
        if h:
            parts.append(h)
    if tab in ("all", "categories"):
        parts.append(render_categories(snap))
    if tab in ("all", "lineups"):
        parts.append(render_lineups(snap))
    if tab in ("all", "odds"):
        parts.append(render_odds(snap))
    if tab in ("all", "power"):
        parts.append(render_power_stats(snap))
    if tab in ("all", "sources"):
        parts.append(render_sources(snap))
    parts.append("")
    return "\n".join(parts)


def show(tab: str = "all", watch: bool = False, interval: int = 5) -> int:
    while True:
        snap = _load(config.SNAPSHOT_PATH)
        if not snap:
            print(red(f"No snapshot at {config.SNAPSHOT_PATH}. "
                       f"Run: python3 -m backend.run snapshot"))
            return 1
        live = _load(config.LIVE_PATH)
        if watch:
            sys.stdout.write("\033[2J\033[H")    # clear + home
        print(render(snap, live, tab))
        if not watch:
            return 0
        try:
            time.sleep(max(1, interval))
        except KeyboardInterrupt:
            print(dim("\nbye."))
            return 0
