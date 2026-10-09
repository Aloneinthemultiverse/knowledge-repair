"""Before/after: an assistant answering from the corrupted vs the repaired knowledge base.

Questions target facts the corruptor damaged. Both sides use the same retriever
(truthguard ChunkStore: turbovec + BM25), and the answer is read from the top
retrieved person record, so the score isolates the knowledge base itself.

    python -m kb.ask_demo [--seed 7]
"""
import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict

from truthguard.chunk_store import ChunkStore

from .corrupt import corrupt
from .load import load
from .repair import KBRepairer


def _y(d):
    return str(d)[:4] if d and str(d)[:4].isdigit() else ""


def person_docs(kb, canon_place=None):
    P, L, E, R = (kb[t] for t in ("people", "places", "events", "relationships"))
    place = {r.place_id: (r.name or "", r.country or "") for r in L.itertuples()}
    ev = {r.event_id: (r.prize, r.year) for r in E.itertuples()}
    awards, family = defaultdict(dict), defaultdict(list)
    names = dict(zip(P.person_id, P.name))
    for r in R.itertuples():
        if r.rel == "received" and r.dst in ev:
            awards[r.src][ev[r.dst][0]] = str(r.year or ev[r.dst][1])
        elif r.rel != "received" and r.dst in names:
            family[r.src].append((r.rel, names[r.dst]))
    docs = []
    for r in P.itertuples():
        bp = place.get(r.birth_place_id, ("", ""))
        f = {"name": r.name or "", "birth_year": _y(r.birth_date), "death_year": _y(r.death_date),
             "birth_place": bp[0], "birth_country": bp[1], "awards": awards[r.person_id]}
        text = (f"{f['name']}. Born {f['birth_year']} in {bp[0]}, {bp[1]}. "
                + (f"Died {f['death_year']}. " if f["death_year"] else "")
                + " ".join(f"Received the {p} in {y}." for p, y in f["awards"].items())
                + " " + " ".join(f"{rel}: {n}." for rel, n in family[r.person_id]))
        docs.append({"id": r.person_id, "text": text, "fields": f, "source_file": "kb", "page": 0,
                     "extraction": "structured", "content_type": "prose"})
    return docs


def store(docs, d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    json.dump([{k: v for k, v in x.items() if k != "fields"} for x in docs], open(f"{d}/chunks.json", "w"))
    st = ChunkStore(d)
    st.build()
    return st


def ask(st, q, k=5):
    score = defaultdict(float)
    for hits in (st.vector_search(q, 25), st.keyword_search(q, 25)):
        for rank, (cid, _) in enumerate(hits):
            score[cid] += 1 / (60 + rank)
    return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]


def questions(clean, dirty, truth, log, rng):
    C = {t: clean[t].set_index(i) for t, i in (("people", "person_id"), ("places", "place_id"), ("events", "event_id"))}
    gold_docs = {d["id"]: d["fields"] for d in person_docs(clean)}
    qs, seen = [], set()
    targets = []
    for _, e in log.iterrows():
        if e.table == "XP" and e.col in ("birth_date", "birth_place_id"):
            targets.append((truth["people"][e.id], "birth_year" if e.col == "birth_date" else "birth_place", e.kind))
        elif e.table == "XP" and e.kind == "DUPLICATE":
            targets.append((truth["people"][e.id], rng.choice(["birth_year", "birth_place"]), "DUPLICATE"))
        elif e.table == "XL" and e.col == "country":
            born_there = [p for p, b in C["people"].birth_place_id.items() if b == truth["places"][e.id]]
            if born_there:
                targets.append((born_there[0], "birth_country", e.kind))
        elif e.table == "XE" and e.col == "year":
            ev = truth["events"][e.id]
            winners = clean["relationships"][(clean["relationships"].rel == "received") & (clean["relationships"].dst == ev)]
            if len(winners):
                targets.append((winners.src.iloc[0], "award:" + C["events"].at[ev, "prize"], "CONTRADICTION"))
        elif e.table == "XR" and e.kind == "CONTRADICTION":
            pid = truth["people"].get(e.id, e.id)
            if pid in gold_docs and gold_docs[pid]["awards"]:
                targets.append((pid, "award:" + next(iter(gold_docs[pid]["awards"])), "CONTRADICTION"))
    for pid, field, kind in targets:
        if (pid, field) in seen or pid not in gold_docs:
            continue
        seen.add((pid, field))
        g = gold_docs[pid]
        name = g["name"]
        if field == "birth_year":
            q, gold = f"When was {name} born?", g["birth_year"]
        elif field == "birth_place":
            q, gold = f"Where was {name} born?", g["birth_place"]
        elif field == "birth_country":
            q, gold = f"In which country was {name} born?", g["birth_country"]
        else:
            prize = field.split(":", 1)[1]
            q, gold = f"In which year did {name} receive the {prize}?", g["awards"].get(prize, "")
        if gold:
            qs.append({"q": q, "field": field, "gold": gold, "kind": kind, "person": pid})
    return qs


def answer(docs_by_id, top, field):
    if not top:
        return ""
    f = docs_by_id[top[0]]["fields"]
    if field.startswith("award:"):
        return f["awards"].get(field.split(":", 1)[1], "")
    return f.get(field, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    clean = load()
    dirty, truth, log = corrupt(clean, seed=a.seed)
    rep = KBRepairer(dirty).run()
    qs = questions(clean, dirty, truth, log, rng)
    sides = {"before (corrupted KB)": person_docs(dirty), "after (repaired KB)": person_docs(rep.output())}
    res, examples = {"questions": len(qs)}, []
    for label, docs in sides.items():
        st = store(docs, f"storage/kb_ask_{label.split()[0]}")
        by = {d["id"]: d for d in docs}
        per = defaultdict(lambda: [0, 0])
        lat = []
        for q in qs:
            t = time.perf_counter()
            top = ask(st, q["q"])
            lat.append(time.perf_counter() - t)
            ans = answer(by, top, q["field"])
            ok = ans.strip().lower() == q["gold"].strip().lower()
            per[q["kind"]][0] += ok
            per[q["kind"]][1] += 1
            q.setdefault("answers", {})[label] = ans
        tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
        res[label] = {k: f"{c}/{n} = {c / n:.0%}" for k, (c, n) in sorted(per.items())}
        res[label]["ALL"] = f"{tot[0]}/{tot[1]} = {tot[0] / tot[1]:.1%}"
        res[label]["query_ms_p50"] = round(1000 * sorted(lat)[len(lat) // 2], 1)
    for q in qs:
        b, af = q["answers"].values()
        if b.lower() != q["gold"].lower() and af.lower() == q["gold"].lower():
            examples.append({"question": q["q"], "before": b, "after": af, "truth": q["gold"]})
    res["examples_fixed"] = examples[:8]
    os.makedirs("out_kb_bench", exist_ok=True)
    json.dump(res, open("out_kb_bench/ask_demo.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
