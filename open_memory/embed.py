"""Embedding providers with a content-addressed cache.

Every vector is keyed by sha256(model + text), so re-ingesting the same text
never pays twice, and two spaces that share text share vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request

import numpy as np

MAX_CHARS = 2000


def _h(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class Embedder:
    """Base class. Subclasses implement _embed(list[str]) -> list[list[float]]."""

    model = "base"
    doc_prefix = ""
    query_prefix = ""

    def __init__(self, cache_path: str | None = None):
        self._lock = threading.Lock()
        self._cache = None
        if cache_path:
            self._cache = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache.execute("PRAGMA journal_mode=WAL")
            self._cache.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: list[str], query: bool = False, batch: int = 64) -> np.ndarray:
        prefix = self.query_prefix if query else self.doc_prefix
        texts = [prefix + (t or " ")[:MAX_CHARS] for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        keys = [_h(self.model, t) for t in texts]
        missing = list(range(len(texts)))
        if self._cache is not None and texts:
            with self._lock:
                found = {}
                for i in range(0, len(keys), 900):
                    part = keys[i:i + 900]
                    q = f"SELECT h, v FROM emb WHERE h IN ({','.join('?' * len(part))})"
                    found.update(dict(self._cache.execute(q, part).fetchall()))
            missing = []
            for i, k in enumerate(keys):
                if k in found:
                    out[i] = np.frombuffer(found[k], dtype=np.float32)
                else:
                    missing.append(i)
        for s in range(0, len(missing), batch):
            idx = missing[s:s + batch]
            vecs = self._embed([texts[i] for i in idx])
            rows = []
            for i, v in zip(idx, vecs):
                a = np.asarray(v, dtype=np.float32)
                a /= (np.linalg.norm(a) or 1.0)
                out[i] = a
                rows.append((keys[i], a.tobytes()))
            if self._cache is not None:
                with self._lock:
                    self._cache.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
                    self._cache.commit()
        if not texts:
            return np.zeros((0, self.dim()), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    def dim(self) -> int:
        return len(self.embed(["dim probe"])[0])


class OllamaEmbedder(Embedder):
    """Local embeddings through Ollama (Metal on Apple silicon). Default."""

    def __init__(self, model: str = "nomic-embed-text", host: str | None = None, cache_path: str | None = None):
        super().__init__(cache_path)
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if "nomic" in model:
            self.doc_prefix, self.query_prefix = "search_document: ", "search_query: "

    def _embed(self, texts, tries: int = 12):
        body = json.dumps({"model": self.model, "input": texts, "truncate": True}).encode()
        for attempt in range(tries):
            req = urllib.request.Request(f"{self.host}/api/embed", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.load(r)["embeddings"]
            except urllib.error.HTTPError as e:
                if e.code == 400 and len(texts) > 1:  # one bad input: split so the rest still embed
                    mid = len(texts) // 2
                    return self._embed(texts[:mid]) + self._embed(texts[mid:])
                if e.code == 400:  # a single input the model rejects: embed a stripped copy
                    clean = "".join(ch for ch in texts[0] if ch.isprintable())[:1000] or "empty"
                    if clean != texts[0]:
                        return self._embed([clean])
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == tries - 1:  # server down: wait it out (an app restart takes seconds)
                    raise
                time.sleep(min(30, 2 ** attempt))


class FastEmbedder(Embedder):
    """ONNX embeddings through fastembed; no server needed."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_path: str | None = None):
        super().__init__(cache_path)
        from fastembed import TextEmbedding
        self.model = model
        self._m = TextEmbedding(model)

    def _embed(self, texts):
        return [v.tolist() for v in self._m.embed(texts)]


class HashEmbedder(Embedder):
    """Deterministic bag-of-hashed-tokens vectors. For tests and offline use only."""

    model = "hash-256"

    def _embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for tok in t.lower().split():
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
            out.append(v.tolist())
        return out


def from_name(name: str, cache_path: str | None = None) -> Embedder:
    kind, _, model = name.partition(":")
    if kind == "ollama":
        return OllamaEmbedder(model or "nomic-embed-text", cache_path=cache_path)
    if kind == "fastembed":
        return FastEmbedder(model or "BAAI/bge-small-en-v1.5", cache_path=cache_path)
    if kind == "hash":
        return HashEmbedder(cache_path)
    raise ValueError(f"unknown embedder {name!r}; use ollama:<model>, fastembed:<model> or hash")
