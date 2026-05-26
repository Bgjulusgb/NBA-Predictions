"""Shared contract for every source module.

A source returns a SourceResult: a status plus a list of normalised records.
The pipeline treats all sources identically, so adding a new website only
means writing a module that returns this shape.

Record shape (dict), used by enrichment:
    {
        "id":        unique string,
        "source":    e.g. "google_news" / "reddit:nba" / "espn",
        "kind":      "article" | "social" | "live" | "stat" | "odds",
        "title":     str,
        "text":      str (body / selftext / comment text; may be ""),
        "url":       str,
        "author":    str | None,
        "published": ISO8601 str | None,
        "engagement": int (upvotes / comments / 0),
        "team_hint": str | None,   # set by per-source modules when obvious
        "raw":       dict (optional source-specific extras),
    }
"""

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_ERROR = "error"


class SourceResult:
    def __init__(self, name, status, records=None, error=None, meta=None):
        self.name = name
        self.status = status              # ok | partial | error
        self.records = records or []
        self.error = error
        self.meta = meta or {}            # source-specific structured payload

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "count": len(self.records),
            "error": self.error,
            "meta": self.meta,
        }

    def __repr__(self):
        return (f"<SourceResult {self.name} status={self.status} "
                f"records={len(self.records)} error={self.error}>")
