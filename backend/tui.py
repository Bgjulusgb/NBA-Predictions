"""Interactive curses TUI for the NBA Mood Mirror.

Stdlib only (`curses` ships with Python). Provides a multi-tab, keyboard-driven
dashboard. The user requested "build a UI" — this is the Python-native one,
complementing the existing single-file HTML dashboard.

Keys (case-insensitive):
    1  Mood Mirror      4  Live
    2  Prediction       5  History
    3  Players          6  Math (Monte Carlo + Glicko + Kelly)
    7  Press                R  Refresh now
    arrows / pgup pgdn  Scroll       q / esc  Quit
    s                   Toggle sort (players)
    /                   Filter (press)

Run: python3 -m backend.run tui
"""

from __future__ import annotations

import curses
import json
import os
import time
from typing import Any

from . import config


def _load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --- Curses helpers --------------------------------------------------------
COLOR_HEADER = 1
COLOR_GOOD = 2
COLOR_BAD = 3
COLOR_NEUTRAL = 4
COLOR_DIM = 5
COLOR_HOT = 6
COLOR_TAB = 7
COLOR_ALERT = 8


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(COLOR_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_GOOD, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_BAD, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_NEUTRAL, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_HOT, curses.COLOR_MAGENTA, -1)
    curses.init_pair(COLOR_TAB, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(COLOR_ALERT, curses.COLOR_WHITE, curses.COLOR_RED)


def _pair(code):
    try:
        return curses.color_pair(code)
    except curses.error:
        return 0


def _safe_addstr(win, y, x, text, attr=0):
    """Add a string without crashing on the bottom-right cell or width overflows."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    text = str(text)[: max(0, w - x - 1)]
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _bar(width: int, value: float, scale: float = 100.0) -> str:
    pct = max(0.0, min(1.0, value / scale))
    filled = int(round(pct * width))
    return "#" * filled + "-" * (width - filled)


def _prob_color(p):
    if p is None:
        return _pair(COLOR_DIM)
    if p >= 0.55:
        return _pair(COLOR_GOOD)
    if p <= 0.45:
        return _pair(COLOR_BAD)
    return _pair(COLOR_NEUTRAL)


# --- Tab renderers ---------------------------------------------------------
TABS = [
    ("1", "Mood"),
    ("2", "Prediction"),
    ("3", "Players"),
    ("4", "Live"),
    ("5", "History"),
    ("6", "Math"),
    ("7", "Press"),
]


def _render_tab_bar(win, active_idx: int):
    h, w = win.getmaxyx()
    x = 1
    for i, (key, label) in enumerate(TABS):
        text = f" [{key}] {label} "
        attr = _pair(COLOR_TAB) | curses.A_BOLD if i == active_idx else _pair(COLOR_DIM)
        _safe_addstr(win, 1, x, text, attr)
        x += len(text) + 1


def _header(win, snap):
    h, w = win.getmaxyx()
    teams = snap.get("teams") or {}
    away = teams.get("away", {}).get("name", "?")
    home = teams.get("home", {}).get("name", "?")
    label = snap.get("label", "")
    venue = snap.get("venue", "")
    mode = snap.get("mode", "")
    when = (snap.get("generated_at") or "")[:19].replace("T", " ")
    _safe_addstr(win, 0, 1,
                  f" NBA MOOD MIRROR · {away} @ {home} · {label} · {venue} · mode={mode} · {when}",
                  _pair(COLOR_HEADER) | curses.A_BOLD)


def _render_mood(win, top: int, snap):
    mood = snap.get("mood", {})
    overall = mood.get("overall", {})
    ts = mood.get("team_sentiment", {})
    emos = mood.get("emotions", {})
    home_name = snap["teams"]["home"]["name"]
    away_name = snap["teams"]["away"]["name"]
    y = top
    _safe_addstr(win, y, 2, "MOOD MIRROR", _pair(COLOR_HEADER) | curses.A_BOLD)
    y += 2
    _safe_addstr(win, y, 4,
                  f"Heat      [{_bar(28, overall.get('heat', 0))}] {overall.get('heat', 0):5.1f}",
                  _pair(COLOR_HOT))
    y += 1
    _safe_addstr(win, y, 4,
                  f"Hype      [{_bar(28, overall.get('hype', 0))}] {overall.get('hype', 0):5.1f}",
                  _pair(COLOR_GOOD))
    y += 1
    _safe_addstr(win, y, 4,
                  f"Toxicity  [{_bar(28, overall.get('toxicity', 0))}] {overall.get('toxicity', 0):5.1f}",
                  _pair(COLOR_BAD))
    y += 1
    _safe_addstr(win, y, 4,
                  f"Volume    {overall.get('volume', 0):5}    mean_sent={overall.get('mean_sentiment', 0):+.4f}",
                  _pair(COLOR_DIM))
    y += 2
    h_s = ts.get("home", 0)
    a_s = ts.get("away", 0)
    _safe_addstr(win, y, 4, f"{home_name:35} {h_s:+.4f}  ({ts.get('count_home', 0)} items)",
                  _pair(COLOR_GOOD) if h_s > 0 else _pair(COLOR_BAD) if h_s < 0 else _pair(COLOR_NEUTRAL))
    y += 1
    _safe_addstr(win, y, 4, f"{away_name:35} {a_s:+.4f}  ({ts.get('count_away', 0)} items)",
                  _pair(COLOR_GOOD) if a_s > 0 else _pair(COLOR_BAD) if a_s < 0 else _pair(COLOR_NEUTRAL))
    y += 2
    _safe_addstr(win, y, 4, "Emotion mix:", curses.A_BOLD)
    y += 1
    for k, v in (emos or {}).items():
        _safe_addstr(win, y, 6, f"{k:14} {_bar(20, v, scale=1.0)} {v:.3f}", _pair(COLOR_HEADER))
        y += 1
    return y


def _render_prediction(win, top, snap):
    pred = snap.get("prediction", {}) or {}
    y = top
    _safe_addstr(win, y, 2, "PREDICTION", _pair(COLOR_HEADER) | curses.A_BOLD)
    y += 2
    ens = pred.get("ensemble") or {}
    _safe_addstr(win, y, 4, f"Ensemble HOME  {ens.get('home', 0)*100:5.2f}%   AWAY {ens.get('away', 0)*100:5.2f}%",
                  _prob_color(ens.get("home")))
    y += 1
    market = pred.get("market") or {}
    if market:
        _safe_addstr(win, y, 4, f"Market   HOME  {market.get('home', 0)*100:5.2f}%   "
                                  f"(method={market.get('method')} overround={market.get('overround')})",
                      _prob_color(market.get("home")))
        y += 1
    elo = pred.get("elo") or {}
    if elo:
        _safe_addstr(win, y, 4, f"Elo      HOME  {elo.get('home', 0)*100:5.2f}%   "
                                  f"ratings home={(elo.get('ratings') or {}).get('home')} "
                                  f"away={(elo.get('ratings') or {}).get('away')}",
                      _prob_color(elo.get("home")))
        y += 1
    espn_p = pred.get("espn_predictor")
    if espn_p:
        _safe_addstr(win, y, 4, f"ESPN     HOME  {espn_p.get('home', 0)*100:5.2f}%",
                      _prob_color(espn_p.get("home")))
        y += 1
    y += 1
    _safe_addstr(win, y, 4, f"Sentiment delta {pred.get('sentiment_delta', 0):+.4f}", _pair(COLOR_DIM))
    y += 1
    _safe_addstr(win, y, 4, f"Confidence      {pred.get('confidence', 0):5.1f}", _pair(COLOR_HEADER))
    y += 1

    vb = pred.get("value_bet")
    if vb:
        attr = (_pair(COLOR_GOOD) | curses.A_BOLD) if vb.get("has_value") else _pair(COLOR_DIM)
        _safe_addstr(win, y, 4, f"Value bet:  side={vb.get('side')}  edge={vb.get('edge_pct')}%  "
                                  f"ml={vb.get('moneyline')}  EV={vb.get('expected_value')}", attr)
        y += 1

    series = pred.get("series", {})
    if series:
        y += 1
        _safe_addstr(win, y, 4, f"Series  {series.get('leader')} leads {series.get('lead')}",
                      curses.A_BOLD)
        y += 1
        clinch = series.get("leader_clinch_probability") or 0
        _safe_addstr(win, y, 4, f"Clinch  [{_bar(28, clinch*100)}] {clinch*100:5.2f}%",
                      _pair(COLOR_GOOD))
        y += 1
        for g in series.get("per_game", []) or []:
            _safe_addstr(win, y, 6, f"G{g['game']} @ {g['venue']:3}  "
                                      f"leader win {g['leader_win']*100:5.2f}%",
                          _pair(COLOR_DIM))
            y += 1
    return y


def _render_players(win, top, snap, sort_by="buzz"):
    y = top
    _safe_addstr(win, y, 2, f"PLAYERS  (sort=s to toggle, current={sort_by})",
                  _pair(COLOR_HEADER) | curses.A_BOLD)
    y += 2
    players = list(snap.get("players") or [])
    players.sort(key=lambda p: p.get(sort_by, 0), reverse=True)
    _safe_addstr(win, y, 4, f"{'PLAYER':28} {'TEAM':8} {'MENT':>5} {'SENT':>10} {'BUZZ':>8}",
                  curses.A_BOLD)
    y += 1
    for p in players[:20]:
        s = p["mean_sentiment"]
        attr = _pair(COLOR_GOOD) if s > 0 else _pair(COLOR_BAD) if s < 0 else _pair(COLOR_NEUTRAL)
        team = "HOME" if p["team"] == "home" else "AWAY"
        _safe_addstr(win, y, 4,
                      f"{p['name'][:27]:28} {team:8} {p['mentions']:5} {s:+10.4f} {p['buzz']:8.2f}",
                      attr)
        y += 1
    return y


def _render_live(win, top, live, snap):
    y = top
    _safe_addstr(win, y, 2, "LIVE", _pair(COLOR_HEADER) | curses.A_BOLD)
    y += 2
    if not live or live.get("mode") != "live":
        _safe_addstr(win, y, 4, "(no live snapshot — game not in progress)", _pair(COLOR_DIM))
        return y
    game = live.get("game") or {}
    L = live.get("live") or {}
    hs = (game.get("home") or {}).get("score")
    as_ = (game.get("away") or {}).get("score")
    home = snap["teams"]["home"]["name"]
    away = snap["teams"]["away"]["name"]
    _safe_addstr(win, y, 4, f"{away} {as_}  @  {home} {hs}    Q{game.get('period')}  {game.get('clock')}",
                  curses.A_BOLD)
    y += 1
    wp = L.get("win_probability") or {}
    if wp:
        _safe_addstr(win, y, 4, f"Live WP   HOME {wp.get('home', 0)*100:5.2f}%   "
                                  f"pregame HOME {wp.get('pregame_home', 0)*100:5.2f}%",
                      _prob_color(wp.get("home")))
        y += 1
    run = L.get("current_run") or {}
    if run.get("team") and run.get("points", 0) > 0:
        _safe_addstr(win, y, 4, f"Current run: {run['team']} on {run['points']}-0",
                      _pair(COLOR_HOT) | curses.A_BOLD)
        y += 1
    _safe_addstr(win, y, 4, f"Momentum (home+) {L.get('momentum', 0):+.3f}", _pair(COLOR_DIM))
    y += 1
    _safe_addstr(win, y, 4, f"Sentiment spike z={L.get('sentiment_spike', 0):+.2f}", _pair(COLOR_DIM))
    y += 2
    alerts = live.get("alerts", [])
    if alerts:
        _safe_addstr(win, y, 4, "Alerts:", curses.A_BOLD)
        y += 1
        for a in alerts:
            sev_attr = (_pair(COLOR_ALERT) if a.get("severity") == "high"
                        else _pair(COLOR_NEUTRAL))
            _safe_addstr(win, y, 6, f"[{a.get('severity','low').upper()}]  {a.get('text', '')}",
                          sev_attr | curses.A_BOLD)
            y += 1
    return y


def _render_history(win, top, snap):
    y = top
    _safe_addstr(win, y, 2, "HISTORY (rolling)", _pair(COLOR_HEADER) | curses.A_BOLD)
    y += 2
    rows = snap.get("history") or []
    _safe_addstr(win, y, 4,
                  f"{'TIME':19}  {'MODE':6}  {'ENS':>6}  {'MKT':>6}  {'ELO':>6}  "
                  f"{'HEAT':>6}  {'TOX':>6}  {'CLINCH':>7}", curses.A_BOLD)
    y += 1
    for r in rows[-18:]:
        _safe_addstr(win, y, 4,
                      f"{(r.get('ts') or '')[:19]:19}  {(r.get('mode') or '?'):6}  "
                      f"{(r.get('ens_home') or 0):6.3f}  {(r.get('market_home') or 0):6.3f}  "
                      f"{(r.get('elo_home') or 0):6.3f}  "
                      f"{(r.get('heat') or 0):6.1f}  {(r.get('toxicity') or 0):6.1f}  "
                      f"{(r.get('clinch') or 0):7.3f}",
                      _pair(COLOR_DIM))
        y += 1
    return y


def _render_math(win, top, snap):
    y = top
    _safe_addstr(win, y, 2, "ADVANCED MATH", _pair(COLOR_HEADER) | curses.A_BOLD)
    y += 2
    adv = snap.get("advanced") or {}
    if not adv:
        _safe_addstr(win, y, 4, "(rerun snapshot to compute the advanced metrics)",
                      _pair(COLOR_DIM))
        return y

    mc = adv.get("monte_carlo_game") or {}
    if mc:
        _safe_addstr(win, y, 4, "Monte Carlo (game)", curses.A_BOLD)
        y += 1
        _safe_addstr(win, y, 6,
                      f"trials={mc.get('trials')}  home WP={mc.get('home_win_prob_sim'):.4f}",
                      _pair(COLOR_DIM))
        y += 1
        _safe_addstr(win, y, 6,
                      f"margin p10={mc.get('p10_margin'):+5.1f}   "
                      f"p50={mc.get('median_margin'):+5.1f}   "
                      f"p90={mc.get('p90_margin'):+5.1f}",
                      _pair(COLOR_DIM))
        y += 1
        _safe_addstr(win, y, 6,
                      f"std margin {mc.get('stdev_margin'):.2f}", _pair(COLOR_DIM))
        y += 2

    mcs = adv.get("monte_carlo_series") or {}
    if mcs:
        _safe_addstr(win, y, 4, "Monte Carlo (series)", curses.A_BOLD)
        y += 1
        _safe_addstr(win, y, 6,
                      f"clinch={mcs.get('leader_clinch_prob'):.4f}   "
                      f"expected games={mcs.get('expected_games')}",
                      _pair(COLOR_DIM))
        y += 1
        ei = mcs.get("ends_in") or {}
        if ei:
            for g, p in ei.items():
                _safe_addstr(win, y, 8,
                              f"ends in G{g} : [{_bar(20, p*100)}] {p:.4f}",
                              _pair(COLOR_DIM))
                y += 1
        y += 1

    glk = adv.get("glicko") or {}
    if glk:
        _safe_addstr(win, y, 4, "Glicko-2", curses.A_BOLD)
        y += 1
        _safe_addstr(win, y, 6,
                      f"home rating={glk.get('home_rating')}  RD={glk.get('home_rd')}",
                      _pair(COLOR_DIM))
        y += 1
        _safe_addstr(win, y, 6,
                      f"away rating={glk.get('away_rating')}  RD={glk.get('away_rd')}",
                      _pair(COLOR_DIM))
        y += 1
        _safe_addstr(win, y, 6,
                      f"P(home wins) = {glk.get('home_win_prob'):.4f}",
                      _prob_color(glk.get("home_win_prob")) | curses.A_BOLD)
        y += 2

    kel = adv.get("kelly") or {}
    if kel:
        _safe_addstr(win, y, 4, "Kelly criterion", curses.A_BOLD)
        y += 1
        _safe_addstr(win, y, 6,
                      f"full Kelly = {kel.get('full', 0)*100:.2f}%   "
                      f"quarter Kelly = {kel.get('quarter', 0)*100:.2f}%   "
                      f"EV = {kel.get('ev', 0)*100:.2f}%",
                      _pair(COLOR_HEADER))
        y += 2

    inj = adv.get("injuries") or {}
    if inj:
        _safe_addstr(win, y, 4, "Injury impact", curses.A_BOLD)
        y += 1
        _safe_addstr(win, y, 6,
                      f"home share lost {inj.get('home', 0)*100:.1f}%  "
                      f"away share lost {inj.get('away', 0)*100:.1f}%",
                      _pair(COLOR_BAD))
        y += 1
        for p in inj.get("players_flagged", []):
            _safe_addstr(win, y, 8,
                          f"FLAG {p['name']:20} ({p['team']}) share={p['share']:.2f} "
                          f"sig={p['net_signal']}",
                          _pair(COLOR_NEUTRAL))
            y += 1
    return y


def _render_press(win, top, snap, filter_text=""):
    y = top
    _safe_addstr(win, y, 2, f"PRESS REVIEW  (/ to filter, current='{filter_text}')",
                  _pair(COLOR_HEADER) | curses.A_BOLD)
    y += 2
    items = snap.get("press_review") or []
    filt = filter_text.lower().strip()
    if filt:
        items = [i for i in items if filt in (i.get("title", "").lower())]
    _safe_addstr(win, y, 4,
                  f"{'WHEN':17}  {'SENT':>6}  {'OUTLET':15}  TITLE", curses.A_BOLD)
    y += 1
    h, w = win.getmaxyx()
    for it in items[: max(0, h - y - 3)]:
        when = (it.get("published") or "")[:16]
        comp = it.get("sentiment", {}).get("compound", 0)
        outlet = (it.get("source") or "?")[:15]
        title = (it.get("title") or "").strip()
        attr = (_pair(COLOR_GOOD) if comp > 0.05
                else _pair(COLOR_BAD) if comp < -0.05
                else _pair(COLOR_NEUTRAL))
        _safe_addstr(win, y, 4, f"{when:17}  {comp:+6.2f}  {outlet:15}  {title}", attr)
        y += 1
    return y


# --- Main loop -------------------------------------------------------------
def _refresh_data():
    snap = _load(config.SNAPSHOT_PATH) or {}
    live = _load(config.LIVE_PATH)
    return snap, live


def _run(stdscr):
    curses.curs_set(0)
    _init_colors()
    stdscr.nodelay(True)
    stdscr.timeout(800)
    tab_idx = 0
    sort_by = "buzz"
    filter_text = ""
    snap, live = _refresh_data()
    last_load = time.time()

    while True:
        if time.time() - last_load > 8:
            snap, live = _refresh_data()
            last_load = time.time()
        stdscr.erase()
        if not snap:
            _safe_addstr(stdscr, 0, 0, "snapshot.json not found. Run: python3 -m backend.run snapshot",
                         _pair(COLOR_BAD))
            stdscr.refresh()
        else:
            _header(stdscr, snap)
            _render_tab_bar(stdscr, tab_idx)
            top = 3
            try:
                if tab_idx == 0:
                    _render_mood(stdscr, top, snap)
                elif tab_idx == 1:
                    _render_prediction(stdscr, top, snap)
                elif tab_idx == 2:
                    _render_players(stdscr, top, snap, sort_by=sort_by)
                elif tab_idx == 3:
                    _render_live(stdscr, top, live, snap)
                elif tab_idx == 4:
                    _render_history(stdscr, top, snap)
                elif tab_idx == 5:
                    _render_math(stdscr, top, snap)
                elif tab_idx == 6:
                    _render_press(stdscr, top, snap, filter_text=filter_text)
            except Exception as e:                       # never crash the UI
                _safe_addstr(stdscr, top, 2, f"render error: {e}", _pair(COLOR_BAD))
            h, w = stdscr.getmaxyx()
            _safe_addstr(stdscr, h - 1, 1,
                          "1-7 tabs · r refresh · s sort · / filter · q quit",
                          _pair(COLOR_DIM))
            stdscr.refresh()

        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            break
        if ch == -1:
            continue
        if ch in (ord("q"), 27):
            break
        if ord("1") <= ch <= ord("7"):
            tab_idx = ch - ord("1")
        elif ch in (ord("r"), ord("R")):
            snap, live = _refresh_data()
            last_load = time.time()
        elif ch in (ord("s"), ord("S")):
            sort_by = "mentions" if sort_by == "buzz" else "buzz"
        elif ch == ord("/"):
            filter_text = _prompt(stdscr, "filter> ")


def _prompt(stdscr, prompt_text: str) -> str:
    """Synchronous single-line input at the bottom row."""
    h, w = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    stdscr.nodelay(False)
    _safe_addstr(stdscr, h - 2, 1, " " * (w - 2))
    _safe_addstr(stdscr, h - 2, 1, prompt_text)
    try:
        s = stdscr.getstr(h - 2, 1 + len(prompt_text), 60).decode("utf-8", "replace")
    except Exception:
        s = ""
    curses.noecho()
    curses.curs_set(0)
    stdscr.nodelay(True)
    return s.strip()


def main():
    if not os.environ.get("TERM"):
        print("TUI requires a TTY (TERM env var unset).")
        return 1
    try:
        curses.wrapper(_run)
    except curses.error as e:
        print(f"curses error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
