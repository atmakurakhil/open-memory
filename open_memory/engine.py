"""The memory engine. One SQLite file, nothing hidden.

Three things live here:
  sources  what came in (a conversation, a document), with when it happened
  chunks   the verbatim text, split into rounds or paragraphs. Never rewritten.
  facts    short statements derived from chunks, each with a validity window
           (valid_from / valid_to) and a pointer back to the chunk it came from.

Retrieval is hybrid: dense vectors over chunks and their keys, BM25 over the
words, trigram matching for scripts without spaces and for typos, fused with
weighted reciprocal rank, then boosted by the time window the query names.
Every write is synchronous: when add_* returns, search can see it.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime

import numpy as np

from . import chunk as chunking
from .embed import Embedder, from_name
from .timeparse import parse_date, window

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, space TEXT NOT NULL, kind TEXT NOT NULL, title TEXT,
  uri TEXT, occurred_at TEXT, recorded_at TEXT NOT NULL, meta TEXT, deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, space TEXT NOT NULL, seq INTEGER NOT NULL,
  role TEXT, text TEXT NOT NULL, keytext TEXT, occurred_at TEXT, deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS chunks_src ON chunks(source_id, seq);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY, space TEXT NOT NULL, key TEXT, text TEXT NOT NULL,
  valid_from TEXT, valid_to TEXT, recorded_at TEXT NOT NULL, superseded_by INTEGER,
  chunk_id INTEGER, confidence REAL DEFAULT 1.0, origin TEXT DEFAULT 'user', deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS facts_key ON facts(space, key);
CREATE TABLE IF NOT EXISTS vecs (
  owner TEXT NOT NULL, owner_id INTEGER NOT NULL, space TEXT NOT NULL, v BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS vecs_space ON vecs(space);
CREATE TABLE IF NOT EXISTS events (
  at TEXT NOT NULL, op TEXT NOT NULL, target TEXT, detail TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(text, keytext, tokenize='porter unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_tri USING fts5(text, tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(text, tokenize='porter unicode61 remove_diacritics 2');
"""

STOP = set("""a an the and or but if of to in on at for with about from by as is are was were be been
being do does did have has had i me my mine we our you your he she it its they them their this that
these those what which who whom whose when where why how can could should would will shall may might
must not no yes so than then there here just also very too any some all each every more most other
such only own same few both into over under again further once up down out off am s t don't i'm i've
did tell remind know remember recall ever""".split())

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _iso(d) -> str | None:
    p = parse_date(d) if d is not None else None
    return p.isoformat(timespec="seconds") if p else None


def focus(text: str, query: str, limit: int) -> str:
    """Clip a chunk to `limit` characters keeping the lines that share the most words with the
    query, in their original order. Head-and-tail clipping drops the one line that mattered."""
    if len(text) <= limit:
        return text
    terms = {t for t in re.findall(r"\w+", query.lower()) if t not in STOP and len(t) > 2}
    lines = [ln for ln in text.split("\n") if ln.strip()]
    def score(ln):
        words = set(re.findall(r"\w+", ln.lower()))
        return len(terms & words) + 0.3 * bool(re.search(r"\d", ln))
    keep, used = set(), 0
    first = lines[0][:300]
    used += len(first)
    for i in sorted(range(1, len(lines)), key=lambda i: (-score(lines[i]), i)):
        if used + len(lines[i]) > limit:
            continue
        keep.add(i)
        used += len(lines[i]) + 1
    out, prev = [first], 0
    for i in sorted(keep):
        if i > prev + 1:
            out.append("…")
        out.append(lines[i])
        prev = i
    return "\n".join(out)


class MemoryError_(RuntimeError):
    """Raised loudly. This engine never fails silently."""


class Memory:
    def __init__(self, path: str = ":memory:", embedder: str | Embedder = "ollama:nomic-embed-text",
                 cache_path: str | None = None, llm=None, reranker: str | None = None,
                 split_assistant: int | None = None):
        self.path = path
        self.split_assistant = split_assistant
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.emb = embedder if isinstance(embedder, Embedder) else from_name(embedder, cache_path)
        self.llm = llm
        self._rr_name, self._rr = reranker, None
        self._lock = threading.RLock()
        self._index: dict[str, tuple[np.ndarray, list[tuple[str, int]]]] = {}

    # ------------------------------------------------------------------ writes
    def _log(self, op: str, target: str, detail=None):
        self.db.execute("INSERT INTO events VALUES (?,?,?,?)",
                        (_now(), op, target, json.dumps(detail) if detail is not None else None))

    def _add_source(self, kind, pieces, space, occurred_at, source_id, title, uri, meta):
        sid = source_id or uuid.uuid4().hex[:12]
        occ = _iso(occurred_at)
        with self._lock:
            if self.db.execute("SELECT 1 FROM sources WHERE id=?", (sid,)).fetchone():
                raise MemoryError_(f"source {sid} already exists; forget it first or use a new id")
            self.db.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,0)",
                            (sid, space, kind, title, uri, occ, _now(), json.dumps(meta or {})))
            texts, owners = [], []
            for seq, p in enumerate(pieces):
                cur = self.db.execute(
                    "INSERT INTO chunks (source_id, space, seq, role, text, keytext, occurred_at) VALUES (?,?,?,?,?,?,?)",
                    (sid, space, seq, p["role"], p["text"], p.get("key") or "", occ))
                cid = cur.lastrowid
                self.db.execute("INSERT INTO chunk_fts (rowid, text, keytext) VALUES (?,?,?)",
                                (cid, p["text"], p.get("key") or ""))
                self.db.execute("INSERT INTO chunk_tri (rowid, text) VALUES (?,?)", (cid, p["text"]))
                texts.append(p["text"])
                owners.append(cid)
                if p.get("key") and p["key"] != p["text"]:
                    texts.append(p["key"])
                    owners.append(cid)
            vecs = self.emb.embed(texts)
            self.db.executemany("INSERT INTO vecs VALUES ('chunk',?,?,?)",
                                [(o, space, v.tobytes()) for o, v in zip(owners, vecs)])
            self._log("add", f"source:{sid}", {"kind": kind, "chunks": len(pieces)})
            self.db.commit()
            self._index.pop(space, None)
        return sid

    def add_conversation(self, turns: list[dict], space: str = "default", occurred_at=None,
                         source_id: str | None = None, title: str | None = None, meta: dict | None = None,
                         extract: bool = False) -> str:
        """turns: [{"role": "user"|"assistant", "content": str}, ...]"""
        sid = self._add_source("conversation", chunking.rounds(turns, self.split_assistant), space, occurred_at,
                               source_id, title, None, meta)
        if extract:
            self.extract_facts(sid)
        return sid

    def add_document(self, text: str, space: str = "default", occurred_at=None, source_id=None,
                     title=None, uri=None, meta=None, extract: bool = False) -> str:
        sid = self._add_source("document", chunking.paragraphs(text), space, occurred_at,
                               source_id, title, uri, meta)
        if extract:
            self.extract_facts(sid)
        return sid

    def remember(self, text: str, space: str = "default", key: str | None = None, valid_from=None,
                 chunk_id: int | None = None, confidence: float = 1.0, origin: str = "user") -> int:
        """Store a fact. A fact with the same key supersedes by *event* time, not insertion order:
        learning an older fact late never overwrites the newer truth."""
        vf = _iso(valid_from) or _now()
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO facts (space, key, text, valid_from, recorded_at, chunk_id, confidence, origin) "
                "VALUES (?,?,?,?,?,?,?,?)", (space, key, text, vf, _now(), chunk_id, confidence, origin))
            fid = cur.lastrowid
            if key:
                rows = self.db.execute(
                    "SELECT id, valid_from FROM facts WHERE space=? AND key=? AND id!=? AND deleted=0 "
                    "AND valid_to IS NULL", (space, key, fid)).fetchall()
                for r in rows:
                    if (r["valid_from"] or "") <= vf:
                        self.db.execute("UPDATE facts SET valid_to=?, superseded_by=? WHERE id=?", (vf, fid, r["id"]))
                    else:
                        self.db.execute("UPDATE facts SET valid_to=?, superseded_by=? WHERE id=?",
                                        (r["valid_from"], r["id"], fid))
            self.db.execute("INSERT INTO fact_fts (rowid, text) VALUES (?,?)", (fid, text))
            v = self.emb.embed([text])[0]
            self.db.execute("INSERT INTO vecs VALUES ('fact',?,?,?)", (fid, space, v.tobytes()))
            if chunk_id:  # key merging: the fact also becomes a way to find its chunk
                self.db.execute("INSERT INTO vecs VALUES ('chunk',?,?,?)", (chunk_id, space, v.tobytes()))
                self.db.execute("UPDATE chunk_fts SET keytext = keytext || ' ' || ? WHERE rowid=?", (text, chunk_id))
            self._log("remember", f"fact:{fid}", {"key": key, "valid_from": vf})
            self.db.commit()
            self._index.pop(space, None)
        return fid

    def edit(self, fact_id: int, text: str) -> int:
        """Correct a fact. The old version is kept in history; the new one is current."""
        old = self.db.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        if not old:
            raise MemoryError_(f"no fact {fact_id}")
        key = old["key"] or f"fact:{fact_id}"
        if not old["key"]:
            self.db.execute("UPDATE facts SET key=? WHERE id=?", (key, fact_id))
        return self.remember(text, old["space"], key=key, valid_from=_now(), chunk_id=old["chunk_id"],
                             origin="edit")

    def forget(self, target: str, space: str = "default", reason: str | None = None) -> dict:
        """target: 'fact:<id>', 'source:<id>', 'chunk:<id>', or a query string (forgets the
        best-matching facts and returns them, so the caller can see what went)."""
        with self._lock:
            kind, _, ident = target.partition(":")
            gone = []
            if kind == "fact" and ident.isdigit():
                self.db.execute("UPDATE facts SET deleted=1 WHERE id=?", (int(ident),))
                gone.append(target)
            elif kind == "source":
                self.db.execute("UPDATE sources SET deleted=1 WHERE id=?", (ident,))
                ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE source_id=?", (ident,))]
                self.db.execute("UPDATE chunks SET deleted=1 WHERE source_id=?", (ident,))
                if ids:
                    self.db.execute(f"UPDATE facts SET deleted=1 WHERE chunk_id IN ({','.join('?' * len(ids))})", ids)
                gone.append(target)
            elif kind == "chunk" and ident.isdigit():
                self.db.execute("UPDATE chunks SET deleted=1 WHERE id=?", (int(ident),))
                self.db.execute("UPDATE facts SET deleted=1 WHERE chunk_id=?", (int(ident),))
                gone.append(target)
            else:
                for f in self.search(target, space=space, k=0, facts_k=3)["facts"]:
                    if f["score"] >= 0.5:
                        self.db.execute("UPDATE facts SET deleted=1 WHERE id=?", (f["id"],))
                        gone.append(f"fact:{f['id']} {f['text']}")
            self._log("forget", target, {"reason": reason, "gone": gone})
            self.db.commit()
            self._index.pop(space, None)
        return {"forgotten": gone}

    def extract_facts(self, source_id: str) -> list[int]:
        """LLM pass: pull durable facts out of a source, each keyed so later facts supersede.
        Needs an llm (see open_memory.llm). Facts always point back at the chunk they came from."""
        if self.llm is None:
            raise MemoryError_("extract_facts needs an llm; pass Memory(llm=LLM(...))")
        src = self.db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        rows = self.db.execute("SELECT id, text FROM chunks WHERE source_id=? AND deleted=0 ORDER BY seq",
                               (source_id,)).fetchall()
        existing = self.db.execute(
            "SELECT DISTINCT key FROM facts WHERE space=? AND key IS NOT NULL AND deleted=0 AND valid_to IS NULL",
            (src["space"],)).fetchall()
        keys = ", ".join(r[0] for r in existing[:200]) or "(none yet)"
        batches, cur = [], []
        for r in rows:  # batch so each call fits small context windows and per-minute token caps
            cur.append(f"<chunk id={r['id']}>\n{r['text'][:2500]}\n</chunk>")
            if sum(map(len, cur)) > 9000:
                batches.append(cur)
                cur = []
        if cur:
            batches.append(cur)
        found = []
        for b in batches:
            prompt = EXTRACT_PROMPT.format(date=src["occurred_at"] or "unknown", keys=keys, body="\n\n".join(b))
            out = self.llm.json(prompt)
            found += out.get("facts", []) if isinstance(out, dict) else []
        ids = []
        valid_chunks = {r["id"] for r in rows}
        for f in found:
            if not f.get("text"):
                continue
            cid = f.get("chunk") if f.get("chunk") in valid_chunks else None
            ids.append(self.remember(f["text"], src["space"], key=f.get("key"),
                                     valid_from=f.get("valid_from") or src["occurred_at"],
                                     chunk_id=cid, confidence=float(f.get("confidence", 0.8)), origin="extract"))
        return ids

    # ------------------------------------------------------------------- reads
    def _dense(self, space: str):
        with self._lock:
            if space not in self._index:
                rows = self.db.execute(
                    "SELECT v.owner, v.owner_id, v.v FROM vecs v "
                    "LEFT JOIN chunks c ON v.owner='chunk' AND c.id=v.owner_id "
                    "LEFT JOIN facts f ON v.owner='fact' AND f.id=v.owner_id "
                    "WHERE v.space=? AND COALESCE(c.deleted, f.deleted, 0)=0", (space,)).fetchall()
                if rows:
                    mat = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                else:
                    mat = np.zeros((0, 1), dtype=np.float32)
                self._index[space] = (mat, [(r[0], r[1]) for r in rows])
            return self._index[space]

    @staticmethod
    def _match(query: str, trigram: bool = False) -> str | None:
        if trigram:
            toks = [t for t in re.findall(r"\w{3,}", query.lower()) if t not in STOP]
            if _CJK.search(query):
                toks += [query[i:i + 3] for i in range(len(query) - 2) if _CJK.search(query[i:i + 3])]
        else:
            toks = [t for t in re.findall(r"\w+", query.lower()) if t not in STOP and len(t) > 1]
        toks = list(dict.fromkeys(toks))[:32]
        return " OR ".join('"' + t.replace('"', "") + '"' for t in toks) or None

    def _fts(self, table: str, space_table: str, query: str, space: str, n: int, trigram=False) -> list[int]:
        m = self._match(query, trigram)
        if not m:
            return []
        sql = (f"SELECT x.rowid FROM {table} x JOIN {space_table} c ON c.id = x.rowid "
               f"WHERE {table} MATCH ? AND c.space=? AND c.deleted=0 ORDER BY bm25({table}) LIMIT ?")
        try:
            return [r[0] for r in self.db.execute(sql, (m, space, n))]
        except sqlite3.OperationalError:
            return []

    def _reranker(self):
        if self._rr is None and self._rr_name:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._rr = TextCrossEncoder(self._rr_name)
        return self._rr

    def search(self, query: str, space: str = "default", k: int = 10, now=None, facts_k: int = 10,
               include_history: bool = False, rerank: bool | None = None, expand: int = 0,
               weights=(0.7, 0.3, 0.15), pool: int = 100) -> dict:
        """Returns {"chunks": [...], "facts": [...], "window": [start, end] | None}.
        Each chunk carries its source, date and score; each fact its validity window and origin chunk."""
        now_dt = parse_date(now) if now else datetime.now()
        win = window(query, now_dt)
        qv = self.emb.embed([query], query=True)[0]
        mat, owners = self._dense(space)
        dense_chunks, dense_facts = [], []
        if len(owners):
            sims = mat @ qv
            order = np.argsort(-sims)
            seen = set()
            for i in order[: pool * 4]:
                o = owners[i]
                if o in seen:
                    continue
                seen.add(o)
                (dense_chunks if o[0] == "chunk" else dense_facts).append((o[1], float(sims[i])))
                if len(dense_chunks) >= pool and len(dense_facts) >= pool:
                    break
        out = {"window": [win[0].isoformat(), win[1].isoformat()] if win else None, "chunks": [], "facts": []}

        if k:
            wd, wb, wt = weights
            fused: dict[int, float] = {}
            lists = [([c for c, _ in dense_chunks[:pool]], wd),
                     (self._fts("chunk_fts", "chunks", query, space, pool), wb)]
            if _CJK.search(query) or len(lists[1][0]) < 5:
                lists.append((self._fts("chunk_tri", "chunks", query, space, pool, trigram=True), wt))
            for ids, w in lists:
                for rank, cid in enumerate(ids):
                    fused[cid] = fused.get(cid, 0.0) + w / (60 + rank)
            if not fused:
                pass
            else:
                ids = list(fused)
                rows = {r["id"]: r for r in self.db.execute(
                    f"SELECT c.*, s.title, s.kind FROM chunks c JOIN sources s ON s.id=c.source_id "
                    f"WHERE c.id IN ({','.join('?' * len(ids))})", ids)}
                if win:
                    for cid, r in rows.items():
                        d = parse_date(r["occurred_at"])
                        if d and win[0] <= d < win[1]:
                            fused[cid] *= 1.6
                ranked = sorted(fused, key=lambda c: -fused[c])
                use_rr = rerank if rerank is not None else bool(self._rr_name)
                if use_rr and self._reranker():
                    top = ranked[:50]
                    scores = list(self._reranker().rerank(query, [rows[c]["text"][:2000] for c in top]))
                    rr_rank = {c: i for i, c in enumerate(sorted(top, key=lambda c: -scores[top.index(c)]))}
                    for c in top:
                        fused[c] += 0.5 / (60 + rr_rank[c])
                    ranked = sorted(fused, key=lambda c: -fused[c])
                picked = ranked[:k]
                if expand:
                    extra = []
                    for cid in picked:
                        r = rows[cid]
                        for nb in self.db.execute(
                                "SELECT id FROM chunks WHERE source_id=? AND deleted=0 AND seq BETWEEN ? AND ? AND id!=?",
                                (r["source_id"], r["seq"] - expand, r["seq"] + expand, cid)):
                            if nb[0] not in picked and nb[0] not in extra:
                                extra.append(nb[0])
                    if extra:
                        for r in self.db.execute(
                                f"SELECT c.*, s.title, s.kind FROM chunks c JOIN sources s ON s.id=c.source_id "
                                f"WHERE c.id IN ({','.join('?' * len(extra))})", extra):
                            rows[r["id"]] = r
                            fused[r["id"]] = 0.0
                    picked += extra
                for cid in picked:
                    r = rows[cid]
                    out["chunks"].append({"id": cid, "source": r["source_id"], "seq": r["seq"],
                                          "title": r["title"], "date": r["occurred_at"], "text": r["text"],
                                          "score": round(fused[cid] * 60, 4), "neighbor": fused[cid] == 0.0})

        if facts_k:
            fused_f: dict[int, float] = {}
            for rank, (fid, _) in enumerate(dense_facts[:pool]):
                fused_f[fid] = fused_f.get(fid, 0) + 0.7 / (60 + rank)
            for rank, fid in enumerate(self._fts("fact_fts", "facts", query, space, pool)):
                fused_f[fid] = fused_f.get(fid, 0) + 0.3 / (60 + rank)
            sim = dict(dense_facts)
            if fused_f:
                ids = list(fused_f)
                rows = self.db.execute(f"SELECT * FROM facts WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
                for r in sorted(rows, key=lambda r: -fused_f[r["id"]]):
                    if r["valid_to"] and not include_history:
                        continue
                    out["facts"].append({"id": r["id"], "key": r["key"], "text": r["text"],
                                         "valid_from": r["valid_from"], "valid_to": r["valid_to"],
                                         "chunk": r["chunk_id"], "origin": r["origin"],
                                         "score": round(sim.get(r["id"], 0.0), 4)})
                    if len(out["facts"]) >= facts_k:
                        break
        return out

    def context(self, query: str, space: str = "default", k: int = 10, now=None, budget: int = 16000,
                **kw) -> str:
        """A reader-ready block: current facts, then verbatim excerpts in date order, every item dated."""
        res = self.search(query, space=space, k=k, now=now, **kw)
        lines = []
        if res["facts"]:
            lines.append("## Extracted memories (current)")
            for f in res["facts"]:
                lines.append(f"- [{(f['valid_from'] or '?')[:10]}] {f['text']}")
            lines.append("")
        if res["chunks"]:
            lines.append("## Conversation excerpts (oldest first)")
            by_src: dict[str, list] = {}
            for c in res["chunks"]:
                by_src.setdefault(c["source"], []).append(c)
            for src, cs in sorted(by_src.items(), key=lambda kv: kv[1][0]["date"] or ""):
                date = (cs[0]["date"] or "unknown date").replace("T", " ")[:16]
                lines.append(f"### Session {src} — {date}")
                per = max(500, budget // max(1, len(res["chunks"])))
                for c in sorted(cs, key=lambda c: c["seq"]):
                    lines.append(focus(c["text"], query, per))
                lines.append("")
        return "\n".join(lines).strip()

    def ask(self, question: str, space: str = "default", now=None, k: int = 12) -> dict:
        """Answer from memory with the configured LLM. Returns the answer plus the exact
        context it was given, so every answer can be checked against its sources."""
        if self.llm is None:
            raise MemoryError_("ask needs an llm; set OPEN_MEMORY_LLM=provider:model (your own key)")
        now_s = (parse_date(now) or datetime.now()).strftime("%Y-%m-%d %H:%M")
        ctx = self.context(question, space=space, k=k, now=now)
        raw = self.llm.chat(ASK_PROMPT.format(now=now_s, context=ctx or "(memory is empty)", question=question),
                            max_tokens=1500)
        answer = raw.split("Answer:")[-1].strip() if "Answer:" in raw else raw
        return {"answer": answer, "reasoning": raw, "context": ctx}

    def profile(self, space: str = "default") -> list[dict]:
        rows = self.db.execute("SELECT * FROM facts WHERE space=? AND deleted=0 AND valid_to IS NULL "
                               "ORDER BY key, valid_from", (space,)).fetchall()
        return [dict(r) for r in rows]

    def history(self, key: str, space: str = "default") -> list[dict]:
        rows = self.db.execute("SELECT * FROM facts WHERE space=? AND key=? ORDER BY valid_from",
                               (space, key)).fetchall()
        return [dict(r) for r in rows]

    def export(self, space: str, out_dir: str) -> list[str]:
        """Everything as plain markdown you can read, grep, diff and take with you."""
        os.makedirs(out_dir, exist_ok=True)
        written = []
        prof = os.path.join(out_dir, "facts.md")
        with open(prof, "w") as fh:
            fh.write(f"# Facts — {space}\n\n")
            for f in self.db.execute("SELECT * FROM facts WHERE space=? AND deleted=0 ORDER BY key, valid_from",
                                     (space,)):
                state = "current" if not f["valid_to"] else f"until {f['valid_to'][:10]}"
                fh.write(f"- **{f['key'] or '-'}** · {f['text']} · from {(f['valid_from'] or '?')[:10]} · {state}"
                         f" · fact:{f['id']}" + (f" · chunk:{f['chunk_id']}" if f["chunk_id"] else "") + "\n")
        written.append(prof)
        for s in self.db.execute("SELECT * FROM sources WHERE space=? AND deleted=0", (space,)).fetchall():
            p = os.path.join(out_dir, "sources", f"{s['id']}.md")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(f"---\nid: {s['id']}\nkind: {s['kind']}\ntitle: {s['title'] or ''}\n"
                         f"occurred_at: {s['occurred_at'] or ''}\nuri: {s['uri'] or ''}\n---\n\n")
                for c in self.db.execute("SELECT * FROM chunks WHERE source_id=? AND deleted=0 ORDER BY seq",
                                         (s["id"],)):
                    fh.write(f"<!-- chunk:{c['id']} -->\n{c['text']}\n\n")
            written.append(p)
        return written

    def stats(self, space: str | None = None) -> dict:
        w, a = ("WHERE space=? AND deleted=0", (space,)) if space else ("WHERE deleted=0", ())
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t} {w}", a).fetchone()[0]
                for t in ("sources", "chunks", "facts")}

    def close(self):
        self.db.close()


ASK_PROMPT = """You are answering a question from your long-term memory of the user.
Today is {now}.

{context}

Question: {question}

First note which memories are relevant and their dates. When memories disagree, the most recent
is the current truth. Compute time questions from the dates shown. For "how many" questions,
list every distinct item before counting. If memory does not contain the answer, say so plainly.
End with a line starting "Answer:"."""


EXTRACT_PROMPT = """You maintain a long-term memory about a user. Read the chunks below (dated {date})
and return the durable facts about the user worth remembering in three months: identity, relationships,
possessions, preferences, plans, events with dates, changes of state. Skip small talk and anything the
assistant said unless the user adopted it.

Rules:
- One fact per item, written as a standalone sentence with names and dates in full. Keep numbers,
  times and amounts exactly as written (25:50, $1,200, 3 cats), never as words.
- "key": a short slot like "user.job" or "user.pet.dog.name". Reuse an existing key when the fact
  replaces an older value of that slot, so the old value can be retired. Existing keys: {keys}
- "valid_from": ISO date the fact became true if the text says so, else omit.
- "chunk": the id of the chunk the fact came from.
Return JSON only: {{"facts": [{{"key": "...", "text": "...", "valid_from": "...", "chunk": 0, "confidence": 0.9}}]}}

{body}
"""
