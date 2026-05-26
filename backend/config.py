"""Central configuration for the NBA Mood Mirror backend.

Everything here is free / keyless. No API keys, no paid services.
All identifiers and user-facing strings are English.
"""

import os

# --- Target game -----------------------------------------------------------
# New York Knicks @ Cleveland Cavaliers, Eastern Conference Finals Game 4.
# Mon May 25, 2026, 8:00 PM ET. Rocket Arena, Cleveland. Knicks lead 3-0.
GAME = {
    "label": "Eastern Conference Finals - Game 4",
    "date_et": "2026-05-25",          # used to query ESPN (?dates=YYYYMMDD)
    "tipoff_et": "2026-05-25T20:00:00-04:00",
    "venue": "Rocket Arena, Cleveland",
    "home": {
        "name": "Cleveland Cavaliers",
        "abbr": "CLE",
        "espn_abbr": "CLE",
        "subreddit": "clevelandcavs",
        "aliases": ["cavaliers", "cavs", "cleveland", "cle"],
    },
    "away": {
        "name": "New York Knicks",
        "abbr": "NYK",
        "espn_abbr": "NY",
        "subreddit": "NYKnicks",
        "aliases": ["knicks", "new york", "nyk", "ny knicks"],
    },
    "series": {"leader": "New York Knicks", "lead": "3-0", "home_games_won": 0,
               "away_games_won": 3},
}

# --- Source endpoints ------------------------------------------------------
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
ESPN_NEWS_RSS = "https://www.espn.com/espn/rss/nba/news"

# Google News RSS search queries (English, US edition).
GOOGLE_NEWS_QUERIES = [
    "Knicks Cavaliers",
    "New York Knicks Eastern Conference Finals",
    "Cleveland Cavaliers Eastern Conference Finals",
    "Knicks Cavaliers Game 4",
    "Knicks Cavaliers injury report",
    "Jalen Brunson Knicks",
    "Donovan Mitchell Cavaliers",
]
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

# Additional free, keyless press RSS feeds (general NBA, used for the press
# review + narrative tracking). Tagged "general" unless a team is mentioned.
EXTRA_PRESS_FEEDS = [
    ("https://www.espn.com/espn/rss/nba/news", "espn_news"),
    ("https://sports.yahoo.com/nba/rss.xml", "yahoo_news"),
    ("https://www.cbssports.com/rss/headlines/nba/", "cbs_news"),
]

# --- Team rosters (for player-level sentiment) -----------------------------
# name -> match aliases (lower-cased, matched as whole-ish tokens downstream).
ROSTERS = {
    "home": {  # Cleveland Cavaliers
        "Donovan Mitchell": ["donovan mitchell", "mitchell", "spida"],
        "Darius Garland": ["darius garland", "garland"],
        "Evan Mobley": ["evan mobley", "mobley"],
        "Jarrett Allen": ["jarrett allen", "allen"],
        "Max Strus": ["max strus", "strus"],
        "De'Andre Hunter": ["de'andre hunter", "deandre hunter", "hunter"],
        "Ty Jerome": ["ty jerome", "jerome"],
    },
    "away": {  # New York Knicks
        "Jalen Brunson": ["jalen brunson", "brunson"],
        "Karl-Anthony Towns": ["karl-anthony towns", "karl anthony towns",
                               "towns", "kat"],
        "OG Anunoby": ["og anunoby", "anunoby"],
        "Josh Hart": ["josh hart", "hart"],
        "Mikal Bridges": ["mikal bridges", "bridges"],
        "Mitchell Robinson": ["mitchell robinson", "robinson"],
        "Miles McBride": ["miles mcbride", "mcbride", "deuce"],
    },
}

# Reddit (best-effort, keyless .json). Requires a descriptive User-Agent.
REDDIT_SUBREDDITS = ["nba", "NYKnicks", "clevelandcavs", "sportsbook"]
REDDIT_LISTING = "https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
REDDIT_LIMIT = 25

# NBA.com live data CDN (flaky / bot-protected; optional enrichment).
NBA_CDN_SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_CDN_PLAYBYPLAY = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

# Basketball Reference (historical / Elo seed, best-effort HTML scrape).
BREF_TEAM_PAGE = "https://www.basketball-reference.com/teams/{abbr}/2026.html"

# --- HTTP behaviour --------------------------------------------------------
USER_AGENT = (
    "MoodMirror/1.0 (NBA sentiment dashboard; educational; "
    "contact: local-user) Python-urllib"
)
HTTP_TIMEOUT = 15          # seconds per request
HTTP_RETRIES = 3           # attempts before giving up on a source
HTTP_BACKOFF = 2.0         # exponential backoff base (2s, 4s, 8s)

# --- Output paths ----------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
SNAPSHOT_PATH = os.path.join(DATA_DIR, "snapshot.json")
LIVE_PATH = os.path.join(DATA_DIR, "live.json")

# --- Live mode -------------------------------------------------------------
LIVE_POLL_SECONDS = 25     # how often live mode refreshes during the game
