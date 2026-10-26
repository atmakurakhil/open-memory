"""Fill a fresh database with a made-up person's memories, so the viewer has something to show.

  python examples/demo_memory.py demo.db
  open-memory --db demo.db ui
"""
import sys

from open_memory.engine import Memory

NOTES = [
    ("2026-03-02", "conversation", [("user", "I just moved to Pune for the new job at Lumen Labs. Still unpacking boxes."),
                                    ("assistant", "Congrats on the move. Want help planning the first week?")],
     [("user.city", "Lives in Pune"), ("user.job", "Works at Lumen Labs as a frontend engineer")]),
    ("2026-03-09", "document", "Started running again. 5k in 31:40 along the river. Knees fine.",
     [("user.running.pb_5k", "5k personal best is 31:40")]),
    ("2026-03-20", "conversation", [("user", "My sister Anika's birthday is on April 14. Remind me to get her that pottery class."),
                                    ("assistant", "Noted: April 14, pottery class for Anika.")],
     [("people.anika.birthday", "Sister Anika's birthday is April 14")]),
    ("2026-04-02", "document", "Trying vegetarian for a month. Day 1: dal, rice, too much chai.",
     [("user.diet", "Trying a vegetarian diet")]),
    ("2026-04-14", "document", "Anika loved the pottery class. She made a lopsided bowl and refuses to fix it.", []),
    ("2026-04-21", "document", "5k down to 28:55. The interval sessions are working.",
     [("user.running.pb_5k", "5k personal best is 28:55")]),
    ("2026-05-05", "conversation", [("user", "Vegetarian month went well, I'm keeping it. Eggs are still allowed though."),
                                    ("assistant", "So: vegetarian plus eggs, going forward.")],
     [("user.diet", "Vegetarian, eats eggs")]),
    ("2026-05-30", "document", "Lumen Labs shipped the dashboard redesign. Led the charts work. Proud of this one.", []),
    ("2026-06-18", "conversation", [("user", "I accepted an offer from Northwind. Senior frontend role, starting August."),
                                    ("assistant", "Big step. Want a checklist for leaving Lumen well?")],
     [("user.job", "Senior frontend engineer at Northwind, from August 2026")]),
    ("2026-07-10", "document", "Booked the flat in Bangalore. Moving on July 28, Northwind's office is 20 minutes away.", []),
    ("2026-07-28", "document", "Moved to Bangalore today. The flat smells like fresh paint.",
     [("user.city", "Lives in Bangalore")]),
    ("2026-08-16", "document", "First Bangalore run: 10k in 61:02. Hills everywhere.",
     [("user.running.pb_10k", "10k personal best is 61:02")]),
    ("2026-09-03", "document", "我在班加罗尔学了一点中文：我最喜欢的菜是饺子。", []),
    ("2026-09-22", "document", "5k personal best: 27:41 at the Cubbon Park parkrun.",
     [("user.running.pb_5k", "5k personal best is 27:41")]),
    ("2026-10-12", "conversation", [("user", "Planning Anika's visit in November. She wants to see the pottery studios here."),
                                    ("assistant", "I'll pull a list of studios near Indiranagar.")], []),
]


def main(path):
    m = Memory(path)
    for date, kind, body, facts in NOTES:
        if kind == "conversation":
            sid = m.add_conversation([{"role": r, "content": t} for r, t in body], occurred_at=date)
        else:
            sid = m.add_document(body, occurred_at=date)
        cid = m.db.execute("SELECT id FROM chunks WHERE source_id=? ORDER BY seq LIMIT 1", (sid,)).fetchone()[0]
        for key, text in facts:
            m.remember(text, key=key, valid_from=date, chunk_id=cid, origin="demo")
    print(m.stats("default"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo.db")
