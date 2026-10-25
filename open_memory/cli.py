"""open-memory command line.

  open-memory add "note text" [--title T]       open-memory search "query"
  open-memory remember "fact" --key user.city   open-memory forget fact:12
  open-memory profile                           open-memory history user.city
  open-memory export ./out                      open-memory stats
  open-memory --llm openai:gpt-5-mini ask "what's my 5k best?"   (your own key, from the environment)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import Memory
from .llm import from_spec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="open-memory")
    ap.add_argument("--db", default=os.path.expanduser(os.environ.get("OPEN_MEMORY_DB", "~/.open-memory/memory.db")))
    ap.add_argument("--space", default=os.environ.get("OPEN_MEMORY_SPACE", "default"))
    ap.add_argument("--embedder", default=os.environ.get("OPEN_MEMORY_EMBEDDER", "ollama:nomic-embed-text"))
    ap.add_argument("--llm", default=os.environ.get("OPEN_MEMORY_LLM", ""),
                    help="provider:model for ask/extract, using your own key (e.g. openai:gpt-5-mini, ollama:llama3.2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add"); p.add_argument("text"); p.add_argument("--title"); p.add_argument("--date")
    p.add_argument("--extract", action="store_true", help="also pull durable facts out with --llm")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("-k", type=int, default=12)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=8)
    p.add_argument("--history", action="store_true")
    p = sub.add_parser("context"); p.add_argument("query"); p.add_argument("-k", type=int, default=8)
    p = sub.add_parser("remember"); p.add_argument("fact"); p.add_argument("--key"); p.add_argument("--since")
    p = sub.add_parser("forget"); p.add_argument("target"); p.add_argument("--reason")
    sub.add_parser("profile")
    p = sub.add_parser("history"); p.add_argument("key")
    p = sub.add_parser("export"); p.add_argument("dir")
    sub.add_parser("stats")
    a = ap.parse_args(argv)
    if a.cmd == "add" and a.text == "-":
        a.text = sys.stdin.read()
    m = Memory(a.db, embedder=a.embedder, cache_path=os.path.join(os.path.dirname(a.db), "emb-cache.db"),
               llm=from_spec(a.llm))
    out = {
        "add": lambda: {"source": m.add_document(a.text, space=a.space, title=a.title, occurred_at=a.date,
                                                 extract=a.extract)},
        "ask": lambda: m.ask(a.question, space=a.space, k=a.k)["answer"],
        "search": lambda: m.search(a.query, space=a.space, k=a.k, include_history=a.history),
        "context": lambda: m.context(a.query, space=a.space, k=a.k),
        "remember": lambda: {"fact": m.remember(a.fact, space=a.space, key=a.key, valid_from=a.since)},
        "forget": lambda: m.forget(a.target, space=a.space, reason=a.reason),
        "profile": lambda: m.profile(a.space),
        "history": lambda: m.history(a.key, a.space),
        "export": lambda: {"written": m.export(a.space, a.dir)},
        "stats": lambda: m.stats(a.space),
    }[a.cmd]()
    print(out if isinstance(out, str) else json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
