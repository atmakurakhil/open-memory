# hal-memory

Long-term memory for AI agents that you can read, edit, and take with you.
One SQLite file on your machine. Every fact is dated and points to the words it came from.

This is the memory engine behind Hal, my personal AI, opened up. The engine is MIT.
My own memories are not included.

## Why another memory layer

I took apart the most popular hosted memory API before writing a line of this. Its users' top complaints
(GitHub issues, Product Hunt) were the same five things, so those are the design spec:

| What breaks elsewhere | What this does |
|---|---|
| "Open source", but the engine ships as a closed binary | All of it is here: about 1,000 lines of Python |
| A write succeeds, the search finds nothing for minutes | Writes are synchronous. When `add` returns, `search` sees it |
| Chinese and Japanese text is never indexed | Trigram index next to the word index; CJK works by default |
| A corrected fact keeps coming back; memories can't be edited or exported | Facts have validity windows. A new value retires the old one by when it became true, not by insertion order. `edit`, `forget`, `history`, and `export` to plain markdown |
| Benchmarks nobody can reproduce | `eval/` runs LongMemEval-S end to end with the official judge prompts. Every number below comes from that file |

## How it works

```
add_conversation / add_document          remember(fact, key, valid_from)
        │                                         │
   chunk into rounds (user turn + reply)     fact row: valid_from, valid_to,
   keep the text verbatim, forever           superseded_by, chunk_id (provenance)
        │                                         │
   index three ways ───────────────┬──────────────┘
   dense vectors (round + user turn as a separate key + every fact derived from it)
   BM25 over words, trigram for CJK and typos
        │
search(query, now=…)
   weighted reciprocal-rank fusion (dense 0.7, BM25 0.3, trigram 0.15)
   + the time window the query names ("two months ago", "last Saturday", "in March") as a boost
   + optional cross-encoder rerank, optional neighbour expansion
        │
context(query) → current facts, then excerpts grouped by session, oldest first, every line dated
```

Facts are optional. With an LLM attached, `extract=True` pulls durable facts out of each source,
keyed (`user.running.pb_5k`) so a later value supersedes an earlier one, each pointing at the chunk it
came from. The verbatim text is never replaced by the summary: published ablations show storing only
extracted facts costs about 20 points on LongMemEval.

## Results

LongMemEval-S (500 questions, ~115k tokens of chat history each), scored with the benchmark's own judge prompts.
Everything here was run on a free tier, so read it as a first measurement, not a leaderboard entry.

**End-to-end QA, stratified sample (seed 7), reader `gpt-oss-120b`, judge `gpt-oss-20b`, v0.1:**

| Question type | Correct |
|---|---|
| knowledge-update | 10 / 10 |
| single-session-assistant | 9 / 10 |
| multi-session | 6 / 10 |
| single-session-preference | 4 / 6 |
| **total** | **29 / 36 (80.6%)** |

Against naive RAG (dense top-k, no dates, no ordering) on the 13 questions both arms reached, with the same reader and judge: 12 vs 11.
The run stopped on the provider's daily token cap. Published numbers (supermemory 81.6, Mastra 84.2, Redis 86.5) use gpt-4o
as reader, so these are not like for like. Run `eval/` with your own key to get the comparable number.

The misses were counting across many sessions and one detail buried in a long assistant reply. v0.2 adds query-focused
clipping and splits long replies into their own chunks (both covered by tests); it has not been re-scored yet.

**Retrieval, all 470 answerable questions, no LLM, runs free on a laptop** (`nomic-embed-text` via Ollama).
Session-level recall: *any@5* = at least one evidence session in the top 5; *all@5* = every evidence session in the top 5.

| Mode | any@1 | any@5 | all@5 | all@10 | NDCG@10 |
|---|---|---|---|---|---|
| dense only | 90.6 | 98.1 | 91.1 | 95.5 | 93.7 |
| BM25 only | 88.7 | 97.4 | 86.2 | 92.8 | 91.5 |
| **hybrid (default)** | **92.3** | **99.4** | **91.7** | **96.4** | **95.0** |
| hybrid + time window | 93.0 | 99.6 | 91.1 | 95.7 | 94.9 |

Hybrid wins on every column but one. The time-window boost helps the top hit and costs 0.6 on *all@5*, so it stays a soft boost.
The weakest type is temporal reasoning (82.7 all@5); multi-session is 87.6. Raw output: `results/retrieval.json`.

## Use it

```bash
pip install -e ".[mcp]"
ollama pull nomic-embed-text            # local embeddings; or use fastembed:<model>

hal-memory remember "I live in Hyderabad" --key user.city --since 2020-01-01
hal-memory remember "I moved to Bangalore" --key user.city --since 2026-09-01
hal-memory profile                      # only Bangalore is current
hal-memory history user.city            # both, with their windows
hal-memory add "meeting notes ..." --date 2026-10-20
hal-memory search "where do I live"
hal-memory export ./my-memory           # plain markdown, yours
```

As an MCP server for Claude Code, Claude Desktop, Cursor:

```bash
claude mcp add hal-memory -- /path/to/.venv/bin/python -m hal_memory.mcp_server
```

Tools: `memory_search`, `memory_context`, `memory_add`, `memory_remember`, `memory_forget`,
`memory_profile`, `memory_history`.

From Python:

```python
from hal_memory.engine import Memory
from hal_memory.llm import LLM

m = Memory("me.db", llm=LLM("openai/gpt-oss-120b", provider="groq"))
m.add_conversation(turns, occurred_at="2026-10-25", extract=True)
print(m.context("what's my 5k personal best?", now="2026-10-25"))
```

## Reproduce the numbers

```bash
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python eval/warm_cache.py longmemeval_s_cleaned.json .cache/emb.db          # embeds once, cached by content hash
python eval/longmemeval.py retrieval longmemeval_s_cleaned.json --out results/retrieval.json
GROQ_API_KEY=... python eval/longmemeval.py qa longmemeval_s_cleaned.json --arm engine --out results/qa-engine.jsonl
GROQ_API_KEY=... python eval/longmemeval.py qa longmemeval_s_cleaned.json --arm naive  --out results/qa-naive.jsonl
```

Judge prompts are copied verbatim from the LongMemEval repo (MIT, Di Wu 2024).

## Tests

`pytest tests/`. Each test pins a failure users reported in another memory product: write lag, insertion-order
supersession, CJK blindness, forgetting that doesn't take, memories you can't export.

## License

MIT.
