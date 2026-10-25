"""MCP server: plug the memory into Claude, Cursor, or any MCP client.

  claude mcp add hal-memory -- /path/to/.venv/bin/python -m hal_memory.mcp_server

Env: HAL_MEMORY_DB (default ~/.hal-memory/memory.db), HAL_MEMORY_EMBEDDER
(default ollama:nomic-embed-text), HAL_MEMORY_SPACE (default "default"),
HAL_MEMORY_LLM (optional provider:model for fact extraction, with your own key).
"""
from __future__ import annotations

import json
import os

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from .engine import Memory
from .llm import from_spec

DB = os.path.expanduser(os.environ.get("HAL_MEMORY_DB", "~/.hal-memory/memory.db"))
SPACE = os.environ.get("HAL_MEMORY_SPACE", "default")
mem = Memory(DB, embedder=os.environ.get("HAL_MEMORY_EMBEDDER", "ollama:nomic-embed-text"),
             cache_path=os.path.join(os.path.dirname(DB), "emb-cache.db"),
             llm=from_spec(os.environ.get("HAL_MEMORY_LLM", "")))
app = FastMCP("hal-memory")


@app.tool()
def memory_search(query: str, k: int = 8, include_history: bool = False) -> str:
    """Search long-term memory. Returns current facts and verbatim excerpts, each dated and with its source id."""
    return json.dumps(mem.search(query, space=SPACE, k=k, include_history=include_history), default=str)


@app.tool()
def memory_context(query: str, k: int = 8) -> str:
    """A ready-to-read memory block for answering a question: facts first, then dated excerpts, oldest first."""
    return mem.context(query, space=SPACE, k=k)


@app.tool()
def memory_add(text: str, title: str = "", occurred_at: str = "") -> str:
    """Save a note, document or pasted conversation verbatim. Searchable the moment this returns."""
    sid = mem.add_document(text, space=SPACE, title=title or None, occurred_at=occurred_at or None)
    return json.dumps({"source": sid, **mem.stats(SPACE)})


@app.tool()
def memory_add_conversation(turns_json: str, occurred_at: str = "", extract: bool = False) -> str:
    """Save a conversation: JSON list of {"role": "user"|"assistant", "content": "..."}. With extract=true and
    HAL_MEMORY_LLM set, durable facts are pulled out and keyed so later facts retire earlier ones."""
    sid = mem.add_conversation(json.loads(turns_json), space=SPACE, occurred_at=occurred_at or None,
                               extract=extract and mem.llm is not None)
    return json.dumps({"source": sid, **mem.stats(SPACE)})


@app.tool()
def memory_remember(fact: str, key: str = "", valid_from: str = "") -> str:
    """Save one fact. Give a key (e.g. 'user.city') so a later fact with the same key retires this one."""
    fid = mem.remember(fact, space=SPACE, key=key or None, valid_from=valid_from or None)
    return json.dumps({"fact": fid})


@app.tool()
def memory_forget(target: str, reason: str = "") -> str:
    """Forget 'fact:<id>', 'source:<id>', 'chunk:<id>', or the facts best matching a description. Returns what went."""
    return json.dumps(mem.forget(target, space=SPACE, reason=reason or None))


@app.tool()
def memory_profile() -> str:
    """Everything currently believed true about the user, one fact per line, with keys and dates."""
    return "\n".join(f"{f['key'] or '-'}: {f['text']} (since {(f['valid_from'] or '?')[:10]}, fact:{f['id']})"
                     for f in mem.profile(SPACE)) or "(no facts yet)"


@app.tool()
def memory_history(key: str) -> str:
    """How a fact changed over time: every value a key has held, with its validity window."""
    return json.dumps(mem.history(key, SPACE), default=str)


def main():
    app.run()


if __name__ == "__main__":
    main()
