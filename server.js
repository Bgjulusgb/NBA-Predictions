// Zero-dependency static + API server (Node built-in http only).
// Mirrors the Python api_server.py: serves index.html / data files AND a
// JSON REST API at /api/*. Alternative to `python3 -m backend.run api`.
//
//   node server.js                  # serves on http://localhost:8000
//   PORT=3000 node server.js
//   FORWARD_PY=http://127.0.0.1:8001 node server.js   # proxy /api/* to Python
//
// Without FORWARD_PY we serve the API directly by reading data/snapshot.json
// and data/live.json. With FORWARD_PY we transparently proxy /api/* to a
// running Python api_server (useful when Python owns the snapshot rebuilds).

const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const PORT = process.env.PORT || 8000;
const FORWARD_PY = process.env.FORWARD_PY || "";  // e.g. http://127.0.0.1:8001
const SNAPSHOT_PATH = path.join(ROOT, "data", "snapshot.json");
const LIVE_PATH = path.join(ROOT, "data", "live.json");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".js":   "text/javascript; charset=utf-8",
  ".css":  "text/css; charset=utf-8",
  ".svg":  "image/svg+xml",
};

// In-memory cache so /api/snapshot doesn't re-read the JSON each request.
const cache = { snap: null, snapMtime: 0, live: null, liveMtime: 0 };

function loadJson(filePath, key) {
  return new Promise(resolve => {
    fs.stat(filePath, (err, stat) => {
      if (err) return resolve(null);
      if (cache[key + "Mtime"] === stat.mtimeMs && cache[key]) return resolve(cache[key]);
      fs.readFile(filePath, "utf-8", (err2, body) => {
        if (err2) return resolve(null);
        try {
          const data = JSON.parse(body);
          cache[key] = data;
          cache[key + "Mtime"] = stat.mtimeMs;
          resolve(data);
        } catch { resolve(null); }
      });
    });
  });
}

function jsonReply(res, payload, status = 200) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

async function handleApi(subpath, query, res) {
  const snap = await loadJson(SNAPSHOT_PATH, "snap");
  const live = await loadJson(LIVE_PATH, "live");

  if (subpath === "health") {
    const counts = (snap?.source_health || {}).counts || { ok: 0, partial: 0, error: 0 };
    return jsonReply(res, {
      ok: true, version: "1.0",
      snapshotLoaded: Boolean(snap),
      liveLoaded: Boolean(live),
      counts,
    });
  }
  if (subpath === "snapshot")       return jsonReply(res, snap || { error: "no snapshot" });
  if (subpath.startsWith("snapshot/")) {
    const sec = subpath.split("/", 2)[1];
    if (!snap) return jsonReply(res, { error: "no snapshot" }, 404);
    if (!(sec in snap)) return jsonReply(res, { error: `unknown section ${sec}` }, 404);
    return jsonReply(res, snap[sec]);
  }
  if (subpath === "live")           return jsonReply(res, live || { mode: "unavailable" });
  if (subpath === "categories")     return jsonReply(res, snap?.categories || {});
  if (subpath === "players") {
    let items = snap?.players || [];
    if (query.team === "home" || query.team === "away")
      items = items.filter(p => p.team === query.team);
    const sort = query.sort || "buzz";
    items = items.slice().sort((a, b) => (b[sort] || 0) - (a[sort] || 0));
    return jsonReply(res, { count: items.length, items });
  }
  if (subpath === "odds")           return jsonReply(res, snap?.odds_unified || {});
  if (subpath === "lineups")        return jsonReply(res, snap?.lineups_unified || {});
  if (subpath === "sources")        return jsonReply(res, snap?.source_health || {});
  if (subpath === "advanced")       return jsonReply(res, snap?.advanced || {});
  if (subpath === "narratives")     return jsonReply(res, {
    list: snap?.narratives || [], meta: snap?.narrative_meta || {} });
  if (subpath === "sofascore")      return jsonReply(res, snap?.sofascore || {});
  if (subpath === "teamrankings")   return jsonReply(res, snap?.teamrankings || {});
  if (subpath === "standings")      return jsonReply(res, snap?.standings || {});

  return jsonReply(res, { error: `unknown api path: ${subpath}` }, 404);
}

function proxyPython(req, res) {
  // FORWARD_PY mode: forward request to the Python api_server.
  const target = new URL(FORWARD_PY + req.url);
  const opts = {
    hostname: target.hostname,
    port: target.port || 80,
    path: target.pathname + target.search,
    method: req.method,
    headers: req.headers,
  };
  const px = http.request(opts, ur => {
    res.writeHead(ur.statusCode || 502, ur.headers);
    ur.pipe(res);
  });
  px.on("error", e => jsonReply(res, { error: "proxy failed: " + e.message }, 502));
  req.pipe(px);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const pathname = decodeURIComponent(url.pathname);

  // API.
  if (pathname.startsWith("/api/")) {
    if (FORWARD_PY) return proxyPython(req, res);
    const query = Object.fromEntries(url.searchParams.entries());
    return handleApi(pathname.slice(5), query, res);
  }

  // Static files.
  let urlPath = pathname === "/" ? "/index.html" : pathname;
  const filePath = path.normalize(path.join(ROOT, urlPath));
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403).end("Forbidden");
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404, { "Content-Type": "text/plain" }).end("Not found"); return; }
    const type = TYPES[path.extname(filePath)] || "application/octet-stream";
    res.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store",
                          "Access-Control-Allow-Origin": "*" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`NBA Mood Mirror server: http://localhost:${PORT}/`);
  console.log(`  REST API:       http://localhost:${PORT}/api/snapshot`);
  console.log(`  Categories:     http://localhost:${PORT}/api/categories`);
  console.log(`  Source health:  http://localhost:${PORT}/api/sources`);
  if (FORWARD_PY) console.log(`  /api/* → proxied to ${FORWARD_PY}`);
});
