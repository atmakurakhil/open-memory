"""Black-box head-to-head: Supermemory Local vs open-memory, same inputs, same queries.

  supermemory-server                      # their free local server, with any LLM provider
  SM_KEY=sm_... python eval/head_to_head.py

Uses only Supermemory's public HTTP API. Nothing from its binary is read."""
import json, os, sys, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_memory.engine import Memory

SM = "http://127.0.0.1:6767"
KEY = os.environ["SM_KEY"]
RUN = uuid.uuid4().hex[:6]


def sm(method, path, body=None):
    req = urllib.request.Request(SM + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            t = r.read().decode()
            return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:400]}


def sm_add(tag, text, date=None):
    body = {"content": text, "containerTag": tag}
    if date:
        body["metadata"] = {"date": date}
    return sm("POST", "/v3/documents", body)


def sm_wait(ids, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        st = [sm("GET", f"/v3/documents/{i}").get("status") for i in ids]
        if all(s in ("done", "failed") for s in st):
            return st, time.time() - t0
        time.sleep(2)
    return st, time.time() - t0


def sm_search(tag, q, k=5):
    r = sm("POST", "/v4/search", {"q": q, "containerTag": tag, "limit": k, "searchMode": "hybrid", "threshold": 0})
    out = []
    for x in r.get("results", []):
        out.append(" ".join(str(x.get(f) or "") for f in ("memory", "chunk", "content")) +
                   " " + " ".join(c.get("content", "") for c in (x.get("chunks") or [])))
    if "_error" in r:
        out = [f"ERROR {r}"]
    return out


om = Memory(":memory:", embedder="ollama:nomic-embed-text")


def om_search(space, q, k=5, now=None):
    r = om.search(q, space=space, k=k, now=now)
    return [f"[{c['date'] or ''}] " + c["text"] for c in r["chunks"]]


CASES = [
    dict(name="write-then-read (searchable the moment add returns)", docs=[("My locker code at the gym is 4417.", None)],
         q="what is my gym locker code", want="4417", instant=True),
    dict(name="newer fact by event date wins (learned out of order)",
         docs=[("Update: I moved to Bangalore in September 2026, I live there now.", "2026-09-05"),
               ("I live in Hyderabad, have since 2020.", "2020-01-01")],
         q="which city do I live in now", want="Bangalore", top1=True),
    dict(name="Chinese text", docs=[("我最喜欢的食物是寿司，每周五都去吃。", None)], q="寿司", want="寿司"),
    dict(name="Japanese text", docs=[("私の猫の名前はモチです。", None)], q="猫の名前", want="モチ"),
    dict(name="typo in query", docs=[("My sister's name is Lahari and she studies medicine.", None)],
         q="what does lahri study", want="medicine"),
    dict(name="exact code lookup", docs=[("Flight to Delhi booked, PNR XK72QZ, seat 14C.", None),
                                         ("Train to Chennai, PNR AB19CD.", None)], q="XK72QZ", want="Delhi", top1=True),
    dict(name="time window (last Saturday)", now="2026-10-25",
         docs=[("Went bowling with Ravi.", "2026-10-18"), ("Went swimming with Ravi.", "2026-10-04")],
         q="what did I do with Ravi last Saturday", want="bowling", top1=True),
    dict(name="counting across sessions (all 4 in top 5)",
         docs=[("Bought a Logitech mouse today.", "2026-10-01"), ("Picked up new AirPods.", "2026-10-05"),
               ("Got a mechanical keyboard on sale.", "2026-10-09"), ("Ordered a USB-C hub for the laptop.", "2026-10-12"),
               ("Had biryani for lunch.", "2026-10-13"), ("Called mom in the evening.", "2026-10-14")],
         q="how many tech accessories did I buy this month", want=["mouse", "AirPods", "keyboard", "hub"]),
    dict(name="detail buried in a long reply",
         docs=[("user: give me a full plan for my Goa trip\nassistant: " + " ".join(
             ["Day %d: beaches, markets and local food stalls, keep it relaxed." % i for i in range(1, 15)]) +
             " Important: your hotel check-in code is HX-5521. " + " ".join(
             ["Evening %d: sunset point, dinner by the water." % i for i in range(1, 15)]), None)],
         q="what was the hotel check-in code you gave me", want="HX-5521"),
    dict(name="forget really forgets", docs=[("My bank PIN hint is the year Dad was born.", None)],
         q="bank PIN hint", want=None, forget=True),
]


def hit(results, want, top1=False):
    pool = results[:1] if top1 else results
    blob = " ".join(pool).lower()
    if isinstance(want, list):
        return sum(w.lower() in blob for w in want), len(want)
    return int(want.lower() in blob), 1


report = []
for i, c in enumerate(CASES):
    tag = f"p{RUN}{i}"
    # supermemory
    ids, t_add = [], time.time()
    for text, date in c["docs"]:
        r = sm_add(tag, (f"[{date}] " if date else "") + text, date)
        ids.append(r.get("id"))
        if c.get("instant"):
            instant_sm = sm_search(tag, c["q"])
    st, waited = sm_wait([x for x in ids if x])
    if c.get("forget"):
        sm("DELETE", "/v3/documents/bulk", {"ids": ids})
        time.sleep(3)
    smr = sm_search(tag, c["q"])
    # open-memory
    oids = []
    for text, date in c["docs"]:
        oids.append(om.add_document(text, space=tag, occurred_at=date))
        if c.get("instant"):
            instant_om = om_search(tag, c["q"])
    if c.get("forget"):
        for s in oids:
            om.forget(f"source:{s}", space=tag)
    omr = om_search(tag, c["q"], now=c.get("now"))
    row = dict(case=c["name"], sm_status=st, sm_ingest_s=round(waited, 1))
    if c.get("forget"):
        word = "PIN"
        row["sm"] = (int(not any(word.lower() in x.lower() for x in smr)), 1)
        row["om"] = (int(not any(word.lower() in x.lower() for x in omr)), 1)
    else:
        row["sm"] = hit(smr, c["want"], c.get("top1"))
        row["om"] = hit(omr, c["want"], c.get("top1"))
    if c.get("instant"):
        row["sm_instant"] = hit(instant_sm, c["want"])
        row["om_instant"] = hit(instant_om, c["want"])
    row["sm_top"] = [x[:140] for x in smr[:3]]
    row["om_top"] = [x[:140] for x in omr[:3]]
    report.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

json.dump(report, open(os.environ.get("OUT", "head_to_head.json"), "w"), ensure_ascii=False, indent=1)
