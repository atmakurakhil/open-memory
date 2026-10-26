"""A local viewer for your memory, in the spirit of Obsidian: a list, a reading pane, a graph.

  open-memory ui            # http://127.0.0.1:7878, bound to this machine only

Read-mostly: facts can be edited or forgotten from the page, and every change goes through the
engine, so it lands in the audit log like any other write. Nothing is fetched from the internet.
"""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .engine import Memory

HTML = os.path.join(os.path.dirname(__file__), "ui.html")


def _title(src, first_chunk: str, limit: int = 70) -> str:
    if src["title"]:
        return src["title"]
    line = next((ln.strip() for ln in first_chunk.split("\n") if ln.strip()), "untitled")
    for p in ("[user]", "[assistant]", "user:", "assistant:"):
        if line.lower().startswith(p):
            line = line[len(p):].strip()
    return line[:limit] + ("…" if len(line) > limit else "")


def graph(m: Memory, space: str, links: int = 2, min_sim: float = 0.55) -> dict:
    db = m.db
    srcs = db.execute("SELECT * FROM sources WHERE space=? AND deleted=0 ORDER BY occurred_at, recorded_at",
                      (space,)).fetchall()
    first = {r["source_id"]: r["text"] for r in db.execute(
        "SELECT source_id, text FROM chunks WHERE space=? AND deleted=0 AND seq=0", (space,))}
    nodes, edges = [], []
    for s in srcs:
        n = db.execute("SELECT COUNT(*) FROM chunks WHERE source_id=? AND deleted=0", (s["id"],)).fetchone()[0]
        nodes.append({"id": f"source:{s['id']}", "type": s["kind"], "label": _title(s, first.get(s["id"], "")),
                      "date": s["occurred_at"], "size": n})
    facts = db.execute("SELECT f.*, c.source_id FROM facts f LEFT JOIN chunks c ON c.id=f.chunk_id "
                       "WHERE f.space=? AND f.deleted=0", (space,)).fetchall()
    for f in facts:
        nodes.append({"id": f"fact:{f['id']}", "type": "fact", "label": f["text"][:70], "key": f["key"],
                      "date": f["valid_from"], "current": f["valid_to"] is None})
        if f["source_id"]:
            edges.append({"a": f"fact:{f['id']}", "b": f"source:{f['source_id']}", "kind": "from"})
        if f["superseded_by"]:
            edges.append({"a": f"fact:{f['id']}", "b": f"fact:{f['superseded_by']}", "kind": "replaced"})

    # semantic links: each source joins its nearest neighbours, by the mean of its chunk vectors
    mat, owners = m._dense(space)
    if len(owners) and len(srcs) > 1:
        cid_src = {r[0]: r[1] for r in db.execute("SELECT id, source_id FROM chunks WHERE space=? AND deleted=0",
                                                   (space,))}
        groups: dict[str, list[int]] = {}
        for i, (kind, oid) in enumerate(owners):
            if kind == "chunk" and oid in cid_src:
                groups.setdefault(cid_src[oid], []).append(i)
        ids = [s["id"] for s in srcs if s["id"] in groups]
        if len(ids) > 1:
            cent = np.vstack([mat[groups[i]].mean(0) for i in ids])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
            sims = cent @ cent.T
            np.fill_diagonal(sims, -1)
            seen = set()
            for i, sid in enumerate(ids):
                for j in np.argsort(-sims[i])[:links]:
                    pair = tuple(sorted((sid, ids[j])))
                    if sims[i, j] >= min_sim and pair not in seen:
                        seen.add(pair)
                        edges.append({"a": f"source:{pair[0]}", "b": f"source:{pair[1]}", "kind": "related",
                                      "w": round(float(sims[i, j]), 3)})
    return {"nodes": nodes, "edges": edges, "stats": m.stats(space)}


def note(m: Memory, space: str, ref: str) -> dict:
    kind, _, oid = ref.partition(":")
    db = m.db
    if kind == "source":
        s = db.execute("SELECT * FROM sources WHERE id=? AND space=?", (oid, space)).fetchone()
        if not s:
            return {"error": "not found"}
        chunks = [dict(r) for r in db.execute(
            "SELECT id, seq, role, text, occurred_at FROM chunks WHERE source_id=? AND deleted=0 ORDER BY seq", (oid,))]
        facts = [dict(r) for r in db.execute(
            "SELECT f.id, f.key, f.text, f.valid_from, f.valid_to FROM facts f JOIN chunks c ON c.id=f.chunk_id "
            "WHERE c.source_id=? AND f.deleted=0", (oid,))]
        return {"ref": ref, "kind": s["kind"], "title": _title(s, chunks[0]["text"] if chunks else "", 240),
                "date": s["occurred_at"], "recorded": s["recorded_at"], "chunks": chunks, "facts": facts}
    if kind == "fact":
        f = db.execute("SELECT f.*, c.source_id FROM facts f LEFT JOIN chunks c ON c.id=f.chunk_id "
                       "WHERE f.id=? AND f.space=?", (int(oid), space)).fetchone()
        if not f:
            return {"error": "not found"}
        hist = m.history(f["key"], space) if f["key"] else [dict(f)]
        return {"ref": ref, "kind": "fact", "fact": dict(f), "history": hist}
    return {"error": "bad ref"}


def serve(m: Memory, space: str = "default", host: str = "127.0.0.1", port: int = 7878, open_browser=True):
    page = open(HTML, encoding="utf-8").read()
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                with lock:
                    if u.path == "/":
                        return self._send(200, page, "text/html")
                    if u.path == "/api/graph":
                        return self._send(200, graph(m, space))
                    if u.path == "/api/note":
                        return self._send(200, note(m, space, q.get("ref", "")))
                    if u.path == "/api/search":
                        return self._send(200, m.search(q.get("q", ""), space=space, k=int(q.get("k", 12))))
                return self._send(404, {"error": "not found"})
            except Exception as e:  # shown on the page, never swallowed
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.headers.get("Origin") not in (None, f"http://{host}:{port}", f"http://localhost:{port}"):
                return self._send(403, {"error": "cross-origin writes refused"})
            try:
                with lock:
                    if u.path == "/api/edit":
                        return self._send(200, {"fact": m.edit(int(body["id"]), body["text"])})
                    if u.path == "/api/forget":
                        return self._send(200, m.forget(body["ref"], space=space, reason="forgotten from the viewer"))
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    srv = ThreadingHTTPServer((host, port), H)
    url = f"http://{host}:{port}"
    print(f"open-memory viewer on {url}  (space: {space}, ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
