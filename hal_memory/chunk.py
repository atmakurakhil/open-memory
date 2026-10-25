"""Chunking. Conversations split into rounds; documents into paragraphs.

A round is one user turn plus the assistant reply that follows it. The user
turn alone is kept as an extra retrieval key, because most facts about a
person are stated in their own words.
"""
from __future__ import annotations

import re


def _split(text: str, limit: int) -> list[str]:
    parts, buf = [], ""
    for para in re.split(r"\n(?=\s*\n)|\n(?=\s*(?:[-*•]|\d+[.)])\s)", text):
        if buf and len(buf) + len(para) > limit:
            parts.append(buf.strip())
            buf = ""
        buf += para + "\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts or [text]


def rounds(turns: list[dict], split_assistant: int | None = None) -> list[dict]:
    """split_assistant: when an assistant reply is longer than this many characters, cut it at
    paragraph or list boundaries into extra chunks. A detail buried in the ninth paragraph of a
    long answer then gets its own vector instead of being averaged away."""
    out, cur_u, cur_a = [], [], []

    def flush():
        if cur_u or cur_a:
            u, a = "\n".join(cur_u).strip(), "\n".join(cur_a).strip()
            text = (f"[user] {u}\n" if u else "") + (f"[assistant] {a}" if a else "")
            if not split_assistant or len(a) <= split_assistant:
                out.append({"text": text.strip(), "key": u, "role": "user" if u else "assistant"})
                return
            head = u if len(u) <= 400 else u[:400] + " …"
            for i, part in enumerate(_split(a, split_assistant)):
                body = (f"[user] {u}\n" if i == 0 else f"[user, continued] {head}\n") + f"[assistant] {part}"
                out.append({"text": body.strip(), "key": u if i == 0 else "", "role": "user" if i == 0 else "assistant"})

    for t in turns:
        role, content = t.get("role", "user"), t.get("content") or ""
        if role == "user":
            if cur_a:
                flush()
                cur_u.clear()
                cur_a.clear()
            cur_u.append(content)
        else:
            cur_a.append(content)
    flush()
    return out


def paragraphs(text: str, target: int = 1200) -> list[dict]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > target:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return [{"text": t, "key": "", "role": "doc"} for t in out]
