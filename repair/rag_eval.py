"""Compare: (A) dirty -> turbovec RAG, (B) repaired -> turbovec RAG, (C) repaired -> graph.
Uses truthguard.chunk_store.ChunkStore unchanged. No LLM: answers are read from the
top retrieved chunk's structured fields, so the score isolates data + retrieval quality.

    python -m repair.rag_eval [--n 300] [--q 150]
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict

import networkx as nx
import pandas as pd
from rapidfuzz import fuzz, process

from truthguard.chunk_store import ChunkStore

from .engine import Repairer
from .synth import corrupt, make_clean

FIELDS = ["name", "birth_year", "city", "country", "email"]


def _s(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    v = str(v).strip()
    return v[:-2] if v.endswith(".0") else v


def _chunk(cid, row, children, extra=""):
    text = (f"Person: {_s(row['name'])}. Born {_s(row['birth_year'])}. Lives in {_s(row['city'])}, "
            f"{_s(row['country'])}. Email {_s(row['email'])}. "
            f"Children: {', '.join(children) or 'none'}. {extra}")
    return {"id": cid, "text": text, "source_file": "people", "page": 0, "extraction": "structured",
            "content_type": "prose", "fields": {**{f: _s(row[f]) for f in FIELDS}, "children": children}}


def build_store(chunks, d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(d, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    st = ChunkStore(d)
    engine = st.build()
    return st, engine


def retrieve(st, q, k=5):
    """RRF over turbovec dense + BM25 (same fusion idea as truthguard.retrieve)."""
    score = defaultdict(float)
    for hits in (st.vector_search(q, 25), st.keyword_search(q, 25)):
        for rank, (cid, _) in enumerate(hits):
            score[cid] += 1 / (60 + rank)
    return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]


def make_questions(clean, rels, nq, seed=3):
    rng = random.Random(seed)
    name = clean.set_index("entity_id").name
    kids = defaultdict(list)
    for _, r in rels[rels.rel == "parent_of"].iterrows():
        kids[r.src].append(name[r.dst])
    qs = []
    for _, p in clean.sample(nq, random_state=seed).iterrows():
        f = rng.choice(["birth_year", "city", "email", "children"])
        if f == "children" and not kids[p.entity_id]:
            f = "birth_year"
        # identify ONE person: add a qualifier that is not the asked field and is unique
        same = clean[clean.name == p["name"]]
        who = p["name"]
        if len(same) > 1:
            for qcol in ["city", "birth_year"]:
                if qcol != f and (same[qcol] == p[qcol]).sum() == 1:
                    who = f"{p['name']} ({'from ' + p['city'] if qcol == 'city' else 'born ' + str(p['birth_year'])})"
                    break
            else:
                continue
        text = {"birth_year": f"When was {who} born?", "city": f"Which city does {who} live in?",
                "email": f"What is the email address of {who}?",
                "children": f"Who are the children of {who}?"}[f]
        gold = sorted(kids[p.entity_id]) if f == "children" else _s(p[f])
        qs.append({"q": text, "field": f, "gold": gold, "entity": p.entity_id, "name": who})
    return qs


def _correct(ans, gold):
    if isinstance(gold, list):
        return sorted(a.lower() for a in ans) == sorted(g.lower() for g in gold)
    return _s(ans).lower() == gold.lower()


def evaluate(qs, answer_fn, relevant_fn):
    hit1 = rec5 = prec5 = acc = 0
    for q in qs:
        top, ans = answer_fn(q)
        rel = relevant_fn(q)
        got = [c for c in top if c in rel]
        hit1 += bool(top) and top[0] in rel
        rec5 += len(set(got)) / len(rel) if rel else 0
        prec5 += len(got) / len(top) if top else 0
        acc += _correct(ans, q["gold"])
    n = len(qs)
    return {"hit@1": round(hit1 / n, 3), "recall@5": round(rec5 / n, 3),
            "precision@5": round(prec5 / n, 3), "answer_accuracy": round(acc / n, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--q", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    clean, rels = make_clean(a.n, seed=1)
    dirty, drels, truth, _ = corrupt(clean, rels, seed=a.seed)
    rep = Repairer(dirty, drels).run()
    qs = make_questions(clean, rels, a.q)
    res = {}

    # ---- A: dirty records straight into turbovec ----
    dname = dirty.set_index("record_id").name
    dkids = defaultdict(list)
    for _, r in drels[drels.rel == "parent_of"].iterrows():
        if r.dst in dname.index:
            dkids[r.src].append(dname[r.dst])
    chA = [_chunk(r.record_id, r, dkids[r.record_id]) for _, r in dirty.iterrows()]
    stA, engA = build_store(chA, "storage/repair_eval_dirty")
    byA = {c["id"]: c for c in chA}
    recs_of = defaultdict(set)
    for rid, eid in truth.items():
        recs_of[eid].add(rid)

    def ansA(q):
        top = retrieve(stA, q["q"])
        f = byA[top[0]]["fields"] if top else {}
        return top, f.get(q["field"], [] if q["field"] == "children" else "")
    res["A_dirty_turbovec"] = {**evaluate(qs, ansA, lambda q: recs_of[q["entity"]]), "engine": engA,
                               "chunks": len(chA)}

    # ---- B: repaired entities into turbovec ----
    ent = rep.entities.set_index("entity_id")
    ename = ent.name
    ekids = defaultdict(list)
    for _, r in rep.entity_rels[rep.entity_rels.rel == "parent_of"].iterrows():
        ekids[r.src].append(_s(ename[r.dst]))
    review = defaultdict(list)
    for x in rep.actions:
        if x["status"] != "applied" and x["col"]:
            for r in x["records"].split(","):
                review[rep.record_to_entity.get(r, r)].append(x["col"])
    chB = [_chunk(eid, r, ekids[eid],
                  f"Source records: {r['source_records']}." +
                  (f" Unverified: {', '.join(sorted(set(review[eid])))}." if review[eid] else ""))
           for eid, r in ent.iterrows()]
    stB, engB = build_store(chB, "storage/repair_eval_repaired")
    byB = {c["id"]: c for c in chB}
    ent_of_true = defaultdict(set)
    for rid, eid in truth.items():
        ent_of_true[eid].add(rep.record_to_entity[rid])

    def ansB(q):
        top = retrieve(stB, q["q"])
        f = byB[top[0]]["fields"] if top else {}
        return top, f.get(q["field"], [] if q["field"] == "children" else "")
    res["B_repaired_turbovec"] = {**evaluate(qs, ansB, lambda q: ent_of_true[q["entity"]]), "engine": engB,
                                  "chunks": len(chB)}

    # ---- C: repaired entities as a graph; name lookup + edge traversal ----
    G = nx.DiGraph()
    for eid, r in ent.iterrows():
        G.add_node(eid, **{f: _s(r[f]) for f in FIELDS})
    for _, r in rep.entity_rels.iterrows():
        G.add_edge(r.src, r.dst, rel=r.rel)
    names = {eid: f"{d['name']} from {d['city']} born {d['birth_year']}" for eid, d in G.nodes(data=True)}

    def ansC(q):
        hits = process.extract(q["name"].replace("(", "").replace(")", ""), names, scorer=fuzz.token_set_ratio, limit=5)
        top = [h[2] for h in hits]
        if not top:
            return [], ""
        n = top[0]
        if q["field"] == "children":
            return top, [G.nodes[d]["name"] for d in G.successors(n) if G.edges[n, d]["rel"] == "parent_of"]
        return top, G.nodes[n][q["field"]]
    res["C_repaired_graph"] = {**evaluate(qs, ansC, lambda q: ent_of_true[q["entity"]]),
                               "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    res["questions"] = len(qs)
    os.makedirs("out_repair", exist_ok=True)
    with open("out_repair/rag_compare.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
