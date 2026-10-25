"""LongMemEval-S harness. Two modes, both reproducible from this file alone.

retrieval  Session-level recall@k over every non-abstention question (470). No LLM, no cost.
           Arms: dense, bm25, hybrid, hybrid+time, hybrid+time+rerank.
qa         End-to-end accuracy on a stratified sample: reader answers from the memory's
           context, the official LongMemEval judge prompt grades it (vendored from
           github.com/xiaowu0162/LongMemEval, MIT). Arms: engine, naive (dense top-k, no dates).

  python eval/longmemeval.py retrieval DATA.json --cache .cache/emb.db --out results/retrieval.json
  python eval/longmemeval.py qa DATA.json --cache .cache/emb.db --per-type 10 --arm engine \
        --reader openai/gpt-oss-120b --judge openai/gpt-oss-20b --out results/qa-engine.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hal_memory.embed import OllamaEmbedder  # noqa: E402
from hal_memory.engine import Memory, focus  # noqa: E402
from hal_memory.llm import LLM  # noqa: E402

ARMS = {
    "dense": dict(weights=(1.0, 0.0, 0.0), time=False, rerank=False),
    "bm25": dict(weights=(0.0, 1.0, 0.0), time=False, rerank=False),
    "hybrid": dict(weights=(0.7, 0.3, 0.15), time=False, rerank=False),
    "hybrid+time": dict(weights=(0.7, 0.3, 0.15), time=True, rerank=False),
    "hybrid+time+rerank": dict(weights=(0.7, 0.3, 0.15), time=True, rerank=True),
}


def build(q, emb, reranker=None, split=None):
    m = Memory(":memory:", embedder=emb, reranker=reranker, split_assistant=split)
    seen = set()  # a few haystacks repeat a session id; index each session once
    for sid, date, sess in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        if sid not in seen:
            seen.add(sid)
            m.add_conversation(sess, occurred_at=date, source_id=sid)
    return m


def session_rank(m, q, arm):
    cfg = ARMS[arm]
    res = m.search(q["question"], k=60, facts_k=0, now=q["question_date"] if cfg["time"] else "1900-01-01",
                   weights=cfg["weights"], rerank=cfg["rerank"])
    seen = []
    for c in res["chunks"]:
        if c["source"] not in seen:
            seen.append(c["source"])
    return seen


def ndcg(ranked, gold, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, s in enumerate(ranked[:k]) if s in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def run_retrieval(data, args):
    emb = OllamaEmbedder(cache_path=args.cache)
    arms = args.arms.split(",")
    rr = "Xenova/ms-marco-MiniLM-L-6-v2" if any(ARMS[a]["rerank"] for a in arms) else None
    qs = [q for q in data if not q["question_id"].endswith("_abs")]
    if args.limit:
        qs = qs[: args.limit]
    ks = (1, 3, 5, 10)
    agg = {a: defaultdict(list) for a in arms}
    per_type = {a: defaultdict(lambda: defaultdict(list)) for a in arms}
    t0 = time.time()
    for n, q in enumerate(qs, 1):
        m = build(q, emb, rr)
        gold = set(q["answer_session_ids"])
        for a in arms:
            ranked = session_rank(m, q, a)
            for k in ks:
                hit_any = float(bool(gold & set(ranked[:k])))
                hit_all = float(gold <= set(ranked[:k]))
                agg[a][f"any@{k}"].append(hit_any)
                agg[a][f"all@{k}"].append(hit_all)
                per_type[a][q["question_type"]][f"all@{k}"].append(hit_all)
            agg[a]["ndcg@10"].append(ndcg(ranked, gold))
        m.close()
        if n % 25 == 0 or n == len(qs):
            line = " | ".join(f"{a} all@5={100 * sum(agg[a]['all@5']) / n:.1f}" for a in arms)
            print(f"[{n}/{len(qs)}] {time.time() - t0:.0f}s  {line}", flush=True)
    out = {"n": len(qs), "embedder": emb.model, "arms": {}}
    for a in arms:
        out["arms"][a] = {k: round(100 * sum(v) / len(v), 2) for k, v in agg[a].items()}
        out["arms"][a]["by_type_all@5"] = {t: round(100 * sum(v["all@5"]) / len(v["all@5"]), 1)
                                           for t, v in per_type[a].items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- QA

def judge_prompt(task, question, answer, response, abstention=False):
    # Verbatim from LongMemEval src/evaluation/evaluate_qa.py (MIT, Di Wu 2024).
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            t = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "temporal-reasoning":
            t = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "knowledge-update":
            t = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "single-session-preference":
            t = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError(task)
    else:
        t = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return t.format(question, answer, response)


READER = """You are answering a question using your long-term memory of past conversations with the user.
Today's date is {now}.

{context}

Question: {question}

Instructions:
1. First write brief notes: which excerpts are relevant, with their dates. If several excerpts disagree,
   the most recent one reflects the current state.
2. For questions about time, compute from the dates shown (today is {now}).
3. For counting or "how many" questions, list every distinct item you find across all sessions before counting.
4. If the memory does not contain the answer, say you don't have that information.
5. End with a line starting "Answer:" giving the final answer concisely."""

NAIVE_READER = """Use the retrieved memory to answer the question.

{context}

Question: {question}
Answer concisely."""


def clip(text, n):
    return text if len(text) <= n else text[: n - 40] + " …[truncated]… " + text[-30:]


COUNTING = re.compile(r"\b(how many|how much|total|in total|altogether|number of|count)\b", re.I)


def engine_context(m, q, k, budget, v2=False):
    if v2 and COUNTING.search(q["question"]):
        k = max(k, 20)  # aggregation questions need breadth; focused clipping pays for it
    res = m.search(q["question"], k=k, facts_k=0, now=q["question_date"], expand=0)
    by_src = defaultdict(list)
    for c in res["chunks"]:
        by_src[c["source"]].append(c)
    per = max(600, budget // max(1, len(res["chunks"])))
    blocks = []
    for src, cs in sorted(by_src.items(), key=lambda kv: kv[1][0]["date"] or ""):
        blocks.append(f"### Session on {(cs[0]['date'] or '?').replace('T', ' ')[:16]}")
        for c in sorted(cs, key=lambda c: c["seq"]):
            blocks.append(focus(c["text"], q["question"], per) if v2 else clip(c["text"], per))
    return "\n".join(blocks)[:budget]


def naive_context(m, q, k, budget):
    res = m.search(q["question"], k=k, facts_k=0, weights=(1.0, 0.0, 0.0), now="1900-01-01")
    per = max(600, budget // max(1, len(res["chunks"])))
    return "\n---\n".join(clip(c["text"], per) for c in res["chunks"])[:budget]


def sample(data, per_type, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for q in data:
        by[q["question_type"]].append(q)
    out = []
    for t in sorted(by):
        out += rng.sample(by[t], min(per_type, len(by[t])))
    return out


def run_qa(data, args):
    emb = OllamaEmbedder(cache_path=args.cache)
    reader = LLM(args.reader, provider=args.provider)
    judge = LLM(args.judge, provider=args.provider)
    qs = sample(data, args.per_type, args.seed)
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done[r["question_id"]] = r
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fh = open(args.out, "a")
    for n, q in enumerate(qs, 1):
        if q["question_id"] in done:
            continue
        m = build(q, emb, split=1500 if args.v2 else None)
        if args.arm == "engine":
            ctx = engine_context(m, q, args.k, args.budget, v2=args.v2)
            prompt = READER.format(now=q["question_date"], context=ctx, question=q["question"])
        else:
            ctx = naive_context(m, q, args.k, args.budget)
            prompt = NAIVE_READER.format(context=ctx, question=q["question"])
        m.close()
        hyp = reader.chat(prompt, max_tokens=1500, reasoning_effort="medium")
        final = hyp.split("Answer:")[-1].strip() if "Answer:" in hyp else hyp
        verdict = judge.chat(judge_prompt(q["question_type"], q["question"], q["answer"], final,
                                          "_abs" in q["question_id"]), max_tokens=400,
                             reasoning_effort="low")
        ok = "yes" in verdict.lower()
        rec = {"question_id": q["question_id"], "type": q["question_type"], "abs": "_abs" in q["question_id"],
               "question": q["question"], "answer": str(q["answer"]), "hypothesis": final,
               "raw": hyp, "verdict": verdict, "correct": ok, "ctx_chars": len(ctx)}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        done[q["question_id"]] = rec
        acc = 100 * sum(r["correct"] for r in done.values()) / len(done)
        print(f"[{len(done)}/{len(qs)}] {q['question_type']:<26} {'OK ' if ok else 'X  '} running={acc:.1f}%",
              flush=True)
    summary = defaultdict(list)
    for r in done.values():
        summary[r["type"]].append(r["correct"])
    rep = {t: round(100 * sum(v) / len(v), 1) for t, v in sorted(summary.items())}
    rep["overall"] = round(100 * sum(r["correct"] for r in done.values()) / len(done), 1)
    rep["task_avg"] = round(sum(v for k, v in rep.items() if k != "overall") / len(summary), 1)
    rep["n"] = len(done)
    rep["reader_usage"] = reader.usage
    print(json.dumps(rep, indent=2))
    json.dump(rep, open(args.out.replace(".jsonl", ".summary.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrieval", "qa"])
    ap.add_argument("data")
    ap.add_argument("--cache", default=os.path.join(ROOT, ".cache/emb.db"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="dense,bm25,hybrid,hybrid+time")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arm", default="engine", choices=["engine", "naive"])
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--reader", default="openai/gpt-oss-120b")
    ap.add_argument("--judge", default="openai/gpt-oss-20b")
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--v2", action="store_true", help="split long assistant replies, focused clipping, wider k for counting")
    a = ap.parse_args()
    data = json.load(open(a.data))
    (run_retrieval if a.mode == "retrieval" else run_qa)(data, a)
