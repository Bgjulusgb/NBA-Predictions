"""Tiny stdlib HTTP helper: per-request User-Agent, timeout, retry/backoff.

Used by every source module so a single flaky endpoint never crashes the run.
No third-party dependencies (urllib only).
"""

import gzip
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import config


class FetchResult:
    """Outcome of a fetch attempt: ok flag + payload or error string."""

    def __init__(self, ok, body=None, status=None, error=None, url=None):
        self.ok = ok
        self.body = body          # str (decoded text)
        self.status = status      # HTTP status code if known
        self.error = error        # human-readable error
        self.url = url

    def __repr__(self):
        return f"<FetchResult ok={self.ok} status={self.status} url={self.url}>"


def _header(headers, name):
    """Case-insensitive header lookup (proxies may lower-case header names)."""
    name = name.lower()
    for k, v in headers.items():
        if k.lower() == name:
            return v
    return ""


def _decode(raw, headers):
    # Detect gzip by magic bytes too, since header casing/presence varies.
    enc = _header(headers, "Content-Encoding").lower()
    if enc == "gzip" or raw[:2] == b"\x1f\x8b":
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    charset = "utf-8"
    ctype = _header(headers, "Content-Type")
    if "charset=" in ctype:
        charset = ctype.split("charset=")[-1].split(";")[0].strip()
    return raw.decode(charset, errors="replace")


def fetch(url, headers=None, timeout=None, retries=None):
    """GET a URL with retry/backoff. Returns a FetchResult (never raises)."""
    timeout = timeout or config.HTTP_TIMEOUT
    retries = retries if retries is not None else config.HTTP_RETRIES
    req_headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip",
    }
    if headers:
        req_headers.update(headers)

    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = _decode(resp.read(), dict(resp.headers))
                return FetchResult(True, body=body, status=resp.status, url=url)
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}"
            # 4xx (except 429) won't fix themselves; stop retrying.
            if 400 <= e.code < 500 and e.code != 429:
                return FetchResult(False, status=e.code, error=last_error, url=url)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 - defensive: a source must never crash the run
            last_error = f"{type(e).__name__}: {e}"

        if attempt < retries - 1:
            time.sleep(config.HTTP_BACKOFF * (2 ** attempt))

    return FetchResult(False, error=last_error or "unknown error", url=url)


def fetch_json(url, headers=None, timeout=None, retries=None):
    """Fetch and parse JSON. Returns (data_or_None, FetchResult)."""
    res = fetch(url, headers=headers, timeout=timeout, retries=retries)
    if not res.ok:
        return None, res
    try:
        return json.loads(res.body), res
    except (ValueError, TypeError) as e:
        res.ok = False
        res.error = f"JSON parse error: {e}"
        return None, res


def run_parallel(tasks, max_workers=6):
    """Run {name: callable} concurrently. Returns {name: result_or_Exception}.

    Each callable is run in its own thread so independent sources/feeds are
    fetched at the same time. Exceptions are captured, never raised, so one
    failing task cannot abort the others.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in futures:
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:  # noqa: BLE001 - isolate per-task failures
                results[name] = e
    return results
