"""Each test pins a failure users reported in other memory products."""
import os
import tempfile

from open_memory.embed import HashEmbedder
from open_memory.engine import Memory
from open_memory.timeparse import window, parse_date


def mem():
    return Memory(":memory:", embedder=HashEmbedder())


def test_write_is_visible_immediately():
    m = mem()
    m.add_conversation([{"role": "user", "content": "My dog is called Biscuit."},
                        {"role": "assistant", "content": "Cute name!"}], occurred_at="2026-01-02")
    hits = m.search("what is my dog called", k=3)["chunks"]
    assert hits and "Biscuit" in hits[0]["text"]


def test_supersede_by_event_time_not_insertion_order():
    m = mem()
    new = m.remember("User lives in Austin.", key="user.city", valid_from="2026-03-01")
    old = m.remember("User lives in Denver.", key="user.city", valid_from="2025-06-01")  # learned late
    current = [f["id"] for f in m.profile()]
    assert new in current and old not in current
    hist = m.history("user.city")
    assert [h["text"] for h in hist] == ["User lives in Denver.", "User lives in Austin."]
    assert hist[0]["valid_to"] == hist[1]["valid_from"]


def test_correction_wins_and_history_kept():
    m = mem()
    fid = m.remember("User's favourite colour is blue.", key="user.colour")
    m.edit(fid, "User's favourite colour is green.")
    res = m.search("favourite colour", k=0)["facts"]
    assert res[0]["text"].endswith("green.")
    assert len(m.history("user.colour")) == 2
    assert any("blue" in f["text"] for f in m.search("favourite colour", k=0, include_history=True)["facts"])


def test_forget_source_removes_chunks_and_derived_facts():
    m = mem()
    sid = m.add_document("The launch code is 4471.\n\nUnrelated paragraph.", source_id="doc1")
    cid = m.search("launch code", k=1)["chunks"][0]["id"]
    m.remember("Launch code is 4471.", chunk_id=cid)
    m.forget(f"source:{sid}")
    res = m.search("launch code 4471", k=5)
    assert res["chunks"] == [] and res["facts"] == []


def test_cjk_text_is_findable():
    m = mem()
    m.add_document("私の猫の名前はミケです。", source_id="jp")
    m.add_document("Something else entirely.", source_id="en")
    hits = m.search("猫の名前", k=1)["chunks"]
    assert hits and hits[0]["source"] == "jp"


def test_time_window_boost():
    m = mem()
    m.add_conversation([{"role": "user", "content": "I went hiking at the lake."}], occurred_at="2026-01-05")
    m.add_conversation([{"role": "user", "content": "I went hiking in the hills."}], occurred_at="2026-03-02")
    hits = m.search("where did I go hiking two months ago", k=2, now="2026-03-05")["chunks"]
    assert "lake" in hits[0]["text"]


def test_present_tense_query_prefers_latest_event():
    # learned out of order: the newer move arrives first, the old address second
    m = mem()
    m.add_document("I live in Bangalore now, moved in September.", occurred_at="2026-09-05")
    m.add_document("I live in Hyderabad.", occurred_at="2020-01-01")
    top = m.search("which city do I live in now", now="2026-10-25", k=2)["chunks"][0]
    assert "Bangalore" in top["text"]


def test_window_parser():
    now = parse_date("2023/05/30 (Tue) 23:40")
    s, e = window("what did I buy last week", now)
    assert s < now and e > s
    s, e = window("what happened in March", now)
    assert s.month == 3 and e.month == 4
    assert window("what is my name", now) is None


def test_export_is_plain_markdown_with_provenance():
    m = mem()
    sid = m.add_document("Akhil prefers dark mode.", source_id="prefs")
    cid = m.search("dark mode", k=1)["chunks"][0]["id"]
    m.remember("Prefers dark mode.", key="user.theme", chunk_id=cid)
    with tempfile.TemporaryDirectory() as d:
        files = m.export("default", d)
        facts = open(os.path.join(d, "facts.md")).read()
        assert f"chunk:{cid}" in facts and "user.theme" in facts
        assert any(f.endswith(f"{sid}.md") for f in files)


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.db")
        m = Memory(p, embedder=HashEmbedder())
        m.remember("User is 15.", key="user.age")
        m.close()
        m2 = Memory(p, embedder=HashEmbedder())
        assert m2.search("how old is the user", k=0)["facts"][0]["text"] == "User is 15."


def test_spaces_are_isolated():
    m = mem()
    m.remember("Alice likes tea.", space="alice")
    m.remember("Bob likes coffee.", space="bob")
    assert all("Bob" not in f["text"] for f in m.search("likes", space="alice", k=0)["facts"])


def test_detail_deep_in_long_assistant_reply_is_its_own_chunk():
    from open_memory.chunk import rounds
    long = "\n\n".join(f"Section {i}: generic advice about campaigns." * 3 for i in range(30))
    long += "\n\nInfluencer marketing budget: $2,000."
    rs = rounds([{"role": "user", "content": "plan my campaign"}, {"role": "assistant", "content": long}],
                split_assistant=1500)
    assert len(rs) > 1 and "$2,000" in rs[-1]["text"] and rs[-1]["text"].startswith("[user, continued]")
    assert rs[0]["key"] == "plan my campaign" and rs[-1]["key"] == ""


def test_focus_keeps_the_matching_line():
    from open_memory.engine import focus
    text = "[user] q\n" + "filler words here\n" * 200 + "Influencer marketing: $2,000\n" + "more filler\n" * 200
    out = focus(text, "how much for influencer marketing", 400)
    assert "$2,000" in out and len(out) <= 450


def test_ask_without_llm_says_how_to_configure():
    import pytest
    from open_memory.engine import MemoryError_
    with pytest.raises(MemoryError_, match="OPEN_MEMORY_LLM"):
        mem().ask("anything")


def test_ask_passes_dated_context_to_your_llm():
    class Fake:
        def chat(self, prompt, **kw):
            self.prompt = prompt
            return "notes...\nAnswer: Biscuit"
    m = Memory(":memory:", embedder=HashEmbedder(), llm=Fake())
    m.add_conversation([{"role": "user", "content": "My dog is called Biscuit."}], occurred_at="2026-01-02")
    out = m.ask("what is my dog called", now="2026-02-01")
    assert out["answer"] == "Biscuit" and "2026-01-02" in m.llm.prompt and "Biscuit" in out["context"]


def test_provider_spec_requires_your_own_key(monkeypatch):
    import pytest
    from open_memory.llm import from_spec
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        from_spec("openai:gpt-5-mini")
    assert from_spec("ollama:llama3.2").base.startswith("http://localhost")
    assert from_spec("") is None


def test_viewer_graph_links_facts_to_sources_and_versions():
    from open_memory.ui import graph, note
    m = mem()
    sid = m.add_document("I live in Pune.", occurred_at="2026-03-01")
    cid = m.db.execute("SELECT id FROM chunks WHERE source_id=?", (sid,)).fetchone()[0]
    old = m.remember("Lives in Pune", key="user.city", valid_from="2026-03-01", chunk_id=cid)
    new = m.remember("Lives in Bangalore", key="user.city", valid_from="2026-07-28")
    g = graph(m, "default")
    kinds = {(e["a"], e["b"], e["kind"]) for e in g["edges"]}
    assert (f"fact:{old}", f"source:{sid}", "from") in kinds
    assert (f"fact:{old}", f"fact:{new}", "replaced") in kinds
    assert [h["text"] for h in note(m, "default", f"fact:{old}")["history"]] == ["Lives in Pune", "Lives in Bangalore"]
