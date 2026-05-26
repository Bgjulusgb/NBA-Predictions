"""Lexicon-based sentiment engine (VADER-style), pure Python, no downloads.

Implements: valence lexicon (general + NBA slang), negation handling,
booster/intensifier words, ALL-CAPS emphasis, exclamation emphasis, and a
compound score normalised to [-1, 1]. Also exposes a toxicity score in [0, 1].

This is intentionally self-contained so it runs with zero dependencies and
zero cost. The NBA lexicon captures fan vocabulary the user called out:
"clutch", "washed", "rigged", "choke", "MVP", "bum", "robbed", etc.
"""

import math
import re

# --- Tuning constants (mirrors VADER) --------------------------------------
B_INCR = 0.293          # booster boost amount
B_DECR = -0.293
C_INCR = 0.733          # ALL-CAPS emphasis
N_SCALAR = -0.74        # negation dampening factor
ALPHA = 15.0            # compound normalisation constant
EXCL_INCR = 0.292       # per "!" (max 4)
QUES_INCR = 0.18        # per "?" group

# --- General valence lexicon (curated subset, scores roughly -3.5..3.5) ----
GENERAL_LEXICON = {
    "good": 1.9, "great": 3.1, "amazing": 3.2, "awesome": 3.1, "excellent": 3.2,
    "fantastic": 3.3, "incredible": 3.0, "love": 3.2, "loved": 2.9, "best": 3.2,
    "perfect": 3.1, "win": 2.6, "wins": 2.4, "winning": 2.5, "won": 2.3,
    "victory": 2.8, "dominant": 2.6, "dominate": 2.5, "dominated": 2.5,
    "strong": 2.0, "solid": 1.8, "impressive": 2.6, "beautiful": 2.9,
    "happy": 2.7, "hope": 1.9, "confident": 2.0, "easy": 1.4, "smooth": 1.3,
    "brilliant": 2.9, "elite": 2.4, "unstoppable": 2.7, "fire": 2.0,
    "hot": 1.4, "lit": 1.8, "smart": 1.7, "deserved": 1.5, "proud": 2.2,
    "bad": -2.5, "terrible": -3.1, "awful": -3.0, "horrible": -3.1,
    "worst": -3.2, "lose": -2.3, "loses": -2.2, "losing": -2.4, "lost": -2.0,
    "loss": -2.0, "defeat": -2.2, "weak": -2.1, "trash": -2.6, "garbage": -2.7,
    "embarrassing": -2.8, "embarrassed": -2.6, "disaster": -3.0, "fail": -2.4,
    "failure": -2.6, "disappointing": -2.3, "disappointed": -2.3, "sad": -2.1,
    "angry": -2.2, "mad": -1.9, "hate": -3.1, "hated": -2.9, "ugly": -2.3,
    "pathetic": -2.8, "useless": -2.6, "scared": -1.6, "worried": -1.6,
    "nervous": -1.3, "panic": -2.2, "broken": -1.9, "hurt": -1.7, "pain": -1.6,
    "boring": -1.5, "overrated": -1.8, "doubt": -1.2, "concern": -1.0,
    "concerned": -1.2, "struggle": -1.6, "struggling": -1.7, "cold": -1.0,
    "wow": 2.0, "yes": 1.3, "no": -1.0, "not": 0.0, "lol": 0.7, "lmao": 0.9,
}

# --- NBA fan / basketball-specific lexicon ---------------------------------
NBA_LEXICON = {
    "clutch": 2.8, "mvp": 2.9, "goat": 3.0, "splash": 2.0, "buckets": 1.8,
    "cooking": 2.0, "cooked": 1.6, "locked": 1.6, "lockdown": 2.2,
    "poster": 1.9, "dunk": 1.6, "swish": 1.7, "sweep": 1.5, "closeout": 1.0,
    "hustle": 1.4, "killer": 1.9, "cold-blooded": 2.4, "dagger": 2.3,
    "heater": 1.8, "unguardable": 2.5, "carry": 1.6, "carried": 1.2,
    "buzzer": 1.5, "and-one": 1.6, "clutchness": 2.6, "sniper": 2.0,
    "scorching": 2.2, "leveled": 1.4, "ballin": 1.8, "balling": 1.8,
    "money": 1.7, "automatic": 2.0, "flame": 1.6, "ascendant": 2.0,
    "washed": -2.6, "rigged": -2.9, "choke": -2.8, "choked": -2.8,
    "choking": -2.7, "bum": -2.4, "bums": -2.4, "robbed": -2.6, "rob": -2.0,
    "scam": -2.5, "fraud": -2.6, "fraudulent": -2.6, "bust": -2.3,
    "bricks": -1.6, "brick": -1.5, "airball": -1.8, "turnover": -1.4,
    "turnovers": -1.5, "blown": -2.0, "collapse": -2.4, "collapsed": -2.4,
    "injury": -1.8, "injured": -1.9, "hurt": -1.7, "out": -0.6, "ejected": -1.4,
    "flopping": -1.6, "flop": -1.5, "blowout": -0.8, "benched": -1.3,
    "softer": -1.5, "soft": -1.4, "exposed": -1.8, "overrated": -1.8,
    "tanking": -1.6, "blew": -2.1, "blowing": -1.8, "refball": -2.2,
    "yikes": -1.4, "atrocious": -2.8, "abysmal": -2.7, "imploded": -2.3,
    "imploding": -2.3, "embarrassed": -2.4, "frustrated": -1.4,
    "frustration": -1.3, "stinker": -2.0,
}

LEXICON = {**GENERAL_LEXICON, **NBA_LEXICON}

# Multi-word phrases scored as a unit (matched before single words).
PHRASE_LEXICON = {
    "buzzer beater": 2.6, "game winner": 2.6, "game-winner": 2.6,
    "on fire": 2.2, "ice cold": -1.8, "triple double": 1.9,
    "double double": 1.2, "career high": 2.1, "dagger three": 2.5,
    "blown lead": -2.4, "blew the lead": -2.5, "must win": -0.4,
    "must-win": -0.4, "no answer": -1.3, "foul trouble": -1.4,
    "season ending": -2.6, "bad call": -1.9, "missed call": -1.7,
    "bounce back": 1.0, "load management": -0.8, "garbage time": -0.7,
    "play of the year": 2.6, "rising star": 2.1, "stomped on": -1.8,
    "in their bag": 2.0, "going off": 2.0, "running away": 1.6,
    "made history": 2.4, "writing himself": 2.0, "shutting down": 1.6,
    "no chance": -1.6, "wave the white flag": -2.2, "no contest": 1.0,
    "season high": 2.1, "season low": -1.6, "career low": -2.0,
}

# A few emoji with sports-fan valence.
EMOJI_LEXICON = {
    "🔥": 1.6, "🐐": 2.5, "💯": 1.6, "🙌": 1.6, "❤": 1.5, "👏": 1.2,
    "😤": 0.5, "💀": -0.4, "🤡": -2.2, "😭": -1.2, "🚮": -2.0, "👎": -1.5,
    "😡": -2.0, "🤬": -2.4,
}

CONTRAST_WORDS = {"but", "however", "although", "though", "yet"}

_PHRASE_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(PHRASE_LEXICON, key=len, reverse=True)),
    re.IGNORECASE,
)

# Referee/conspiracy/insult terms that drive a "toxicity" reading.
TOXICITY_TERMS = {
    "rigged", "refball", "robbed", "scam", "fraud", "trash", "garbage",
    "bum", "bums", "washed", "pathetic", "useless", "hate", "ugly",
    "embarrassing", "choke", "choked", "fix", "fixed", "corrupt", "clown",
    "clowns", "idiot", "idiots", "stupid", "disgrace",
}

BOOSTERS = {
    "absolutely": B_INCR, "completely": B_INCR, "extremely": B_INCR,
    "incredibly": B_INCR, "really": B_INCR, "so": B_INCR, "totally": B_INCR,
    "very": B_INCR, "super": B_INCR, "insanely": B_INCR, "literally": B_INCR,
    "fucking": B_INCR, "freaking": B_INCR, "hella": B_INCR, "mad": B_INCR,
    "barely": B_DECR, "hardly": B_DECR, "slightly": B_DECR, "kinda": B_DECR,
    "somewhat": B_DECR, "marginally": B_DECR, "almost": B_DECR,
}

NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "cannot", "cant", "can't", "won't", "wont", "wouldn't", "wouldnt",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "isn't", "isnt",
    "aint", "ain't", "without", "lack", "lacks",
}

_WORD_RE = re.compile(r"[A-Za-z'\-]+|[!?]+")


def _is_allcaps(token):
    return token.isupper() and len(token) > 1


def _tokenize(text):
    return _WORD_RE.findall(text)


def score_text(text):
    """Return sentiment for a piece of text.

    Output dict: compound (-1..1), pos, neg, neu (proportions, sum~=1),
    toxicity (0..1).
    """
    if not text or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0,
                "toxicity": 0.0}

    # Multi-word phrases are scored as a unit, then removed so their component
    # words are not double-counted by the single-word pass below.
    phrase_valences = []
    def _take_phrase(m):
        phrase_valences.append(PHRASE_LEXICON[m.group(0).lower()])
        return " "
    text = _PHRASE_RE.sub(_take_phrase, text)

    # Emoji valences (searched on the raw text).
    emoji_valences = [v * text.count(e) for e, v in EMOJI_LEXICON.items()
                      if e in text]
    emoji_valences = [v for v in emoji_valences if v]

    tokens = _tokenize(text)
    words_lower = [t.lower() for t in tokens]
    exclamations = text.count("!")
    questions = text.count("?")
    # Caps emphasis only matters when the whole text isn't shouting.
    mostly_caps = sum(1 for t in tokens if _is_allcaps(t)) > len(tokens) * 0.6
    # Position of the first contrastive word, for "but"-style re-weighting.
    contrast_at = next((i for i, w in enumerate(words_lower)
                        if w in CONTRAST_WORDS), None)

    valences = []          # (valence, token_index)
    tox_hits = 0
    content_words = 0

    for i, word in enumerate(words_lower):
        if word in ("!", "?") or not word.isalpha() and "'" not in word and "-" not in word:
            # punctuation tokens handled globally below
            if word.strip("!?") == "":
                continue
        if word in BOOSTERS or word in NEGATIONS:
            continue
        content_words += 1
        if word in TOXICITY_TERMS:
            tox_hits += 1

        valence = LEXICON.get(word)
        if valence is None:
            continue

        # ALL-CAPS emphasis on the original token.
        if not mostly_caps and i < len(tokens) and _is_allcaps(tokens[i]):
            valence += C_INCR if valence > 0 else -C_INCR

        # Booster words in the 3 preceding tokens (with step damping).
        for step in range(1, 4):
            j = i - step
            if j < 0:
                break
            prev = words_lower[j]
            if prev in BOOSTERS:
                b = BOOSTERS[prev]
                if valence < 0:
                    b = -b
                valence += b * (1.0 - 0.05 * (step - 1))
            if not mostly_caps and tokens[j].isupper() and tokens[j].lower() in BOOSTERS:
                valence += (C_INCR if valence > 0 else -C_INCR) * 0.5

        # Negation in the 3 preceding tokens flips & dampens.
        negated = any(words_lower[i - k] in NEGATIONS
                      for k in range(1, 4) if i - k >= 0)
        if negated:
            valence *= N_SCALAR

        valences.append((valence, i))

    # Apply "but"-style contrast: de-emphasise clauses before the contrast
    # word, emphasise those after it (mirrors VADER).
    weighted = []
    for v, idx in valences:
        if contrast_at is not None:
            v *= 0.5 if idx < contrast_at else 1.5
        weighted.append(v)
    # Phrase and emoji valences carry full weight.
    weighted.extend(phrase_valences)
    weighted.extend(emoji_valences)

    if not weighted:
        compound = 0.0
    else:
        total = sum(weighted)
        # Punctuation emphasis pushes magnitude away from zero.
        punct_emph = _punct_emphasis(exclamations, questions)
        if total > 0:
            total += punct_emph
        elif total < 0:
            total -= punct_emph
        compound = total / math.sqrt(total * total + ALPHA)
        compound = max(-1.0, min(1.0, compound))

    pos_sum = sum(v for v in weighted if v > 0)
    neg_sum = sum(-v for v in weighted if v < 0)
    neu_count = max(0, content_words - len(valences))
    norm = pos_sum + neg_sum + neu_count
    if norm == 0:
        pos = neg = 0.0
        neu = 1.0
    else:
        pos = pos_sum / norm
        neg = neg_sum / norm
        neu = neu_count / norm

    toxicity = min(1.0, tox_hits / max(1, content_words) * 4.0)

    return {"compound": round(compound, 4), "pos": round(pos, 4),
            "neg": round(neg, 4), "neu": round(neu, 4),
            "toxicity": round(toxicity, 4)}


def _punct_emphasis(exclamations, questions):
    emph = 0.0
    emph += min(exclamations, 4) * EXCL_INCR
    if questions > 1:
        emph += min(questions, 3) * QUES_INCR
    return emph


def label(compound):
    """Map a compound score to a coarse English label."""
    if compound >= 0.35:
        return "positive"
    if compound <= -0.35:
        return "negative"
    return "neutral"


# --- Emotion classification (lightweight, lexicon-based) -------------------
EMOTION_LEXICON = {
    "joy": {"win", "won", "clutch", "amazing", "love", "loved", "happy",
            "hype", "beautiful", "dominant", "mvp", "celebrate", "dub", "lit",
            "ecstatic", "thrilled", "goat", "splash", "dagger", "incredible",
            "unstoppable", "elite", "cooking", "buckets", "proud"},
    "anger": {"rigged", "refball", "robbed", "scam", "fraud", "hate", "hated",
              "trash", "garbage", "mad", "furious", "angry", "disgrace",
              "clown", "clowns", "corrupt", "fix", "fixed", "soft", "flop",
              "flopping", "bum", "bums", "embarrassing"},
    "fear": {"nervous", "worried", "scared", "panic", "choke", "choked",
             "choking", "collapse", "collapsed", "blow", "blown", "anxiety",
             "doubt", "concern", "concerned", "shaky", "scary"},
    "sadness": {"lost", "loss", "sad", "heartbroken", "disappointing",
                "disappointed", "eliminated", "devastated", "hurt", "injury",
                "injured", "depressing", "done", "over"},
    "anticipation": {"tonight", "gameday", "tipoff", "preview", "prediction",
                     "matchup", "series", "clinch", "decider", "showdown",
                     "must", "closeout", "swing", "pivotal", "huge"},
}


def emotions(text):
    """Return a normalised emotion distribution for a piece of text.

    Keys: joy, anger, fear, sadness, anticipation. Values sum to 1.0 when any
    emotion word is present, otherwise all zero.
    """
    base = {k: 0 for k in EMOTION_LEXICON}
    if not text:
        return {k: 0.0 for k in base}
    words = {w.lower() for w in _tokenize(text)}
    for emo, lex in EMOTION_LEXICON.items():
        base[emo] = len(words & lex)
    total = sum(base.values())
    if total == 0:
        return {k: 0.0 for k in base}
    return {k: round(v / total, 4) for k, v in base.items()}

