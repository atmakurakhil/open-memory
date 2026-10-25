"""Pre-embed every round in LongMemEval-S so eval runs never wait on the GPU.

Usage: python eval/warm_cache.py <longmemeval_s_cleaned.json> <cache.db>
"""
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("/eval/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from open_memory.chunk import rounds  # noqa: E402
from open_memory.embed import OllamaEmbedder  # noqa: E402

data = json.load(open(sys.argv[1]))
if "--qa-first" in sys.argv:  # embed the QA sample's haystacks before the rest
    from longmemeval import sample  # noqa: E402
    picked = sample(data, 10, 7)
    ids = {q["question_id"] for q in picked}
    data = picked + [q for q in data if q["question_id"] not in ids]
emb = OllamaEmbedder(cache_path=sys.argv[2])
seen, texts = set(), []
for q in data:
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        if sid in seen:
            continue
        seen.add(sid)
        for r in rounds(sess):
            texts.append(r["text"])
            if r["key"]:
                texts.append(r["key"])
print(f"{len(seen)} sessions, {len(texts)} texts", flush=True)
t0 = time.time()
step = 2048
for i in range(0, len(texts), step):
    emb.embed(texts[i:i + step], batch=128)
    done = min(i + step, len(texts))
    rate = done / (time.time() - t0)
    print(f"{done}/{len(texts)}  {rate:.0f}/s  eta {(len(texts) - done) / rate / 60:.1f} min", flush=True)
print("done", flush=True)
