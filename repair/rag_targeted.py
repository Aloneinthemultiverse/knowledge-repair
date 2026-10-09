"""Before/after RAG on the facts that were actually corrupted.

Every question targets a cell the corruptor damaged (typo, missing, contradiction,
FD violation) or an entity that was duplicated. Same turbovec retriever both
sides; only the knowledge base differs (dirty records vs repaired entities).

    python -m repair.rag_targeted
"""
import json
import os
import random
import time
from collections import defaultdict

from .engine import Repairer
from .rag_eval import _chunk, _s, build_store, retrieve
from .synth import corrupt, make_clean

ASK = {"birth_year": "When was {w} born?", "city": "Which city does {w} live in?",
       "country": "Which country does {w} live in?", "email": "What is the email address of {w}?"}


def who(clean, p, f):
    same = clean[clean.name == p["name"]]
    if len(same) == 1:
        return p["name"]
    for q in ("city", "birth_year", "email"):
        if q != f and (same[q] == p[q]).sum() == 1:
            return f"{p['name']} ({'from ' + p['city'] if q == 'city' else 'born ' + str(p['birth_year']) if q == 'birth_year' else 'email ' + p['email']})"
    return None


def main(n=300, cseed=1, dseed=7):
    clean, rels = make_clean(n, cseed)
    dirty, drels, truth, log = corrupt(clean, rels, dseed)
    rep = Repairer(dirty, drels).run()
    C = clean.set_index("entity_id")

    # ---- questions on corrupted facts ----
    qs, seen = [], set()
    for _, e in log.iterrows():
        if e.kind in ("TYPO", "MISSING", "CONTRADICTION", "FD_VIOLATION") and e.col in ASK:
            eid = truth[e.record_id]
            if (eid, e.col) in seen:
                continue
            w = who(clean, C.loc[eid].to_dict() | {"entity_id": eid}, e.col)
            if w:
                seen.add((eid, e.col))
                qs.append({"kind": e.kind, "field": e.col, "entity": eid,
                           "q": ASK[e.col].format(w=w), "gold": _s(C.at[eid, e.col])})
    dup_ents = [eid for eid in set(truth.values()) if list(truth.values()).count(eid) > 1]
    rng = random.Random(5)
    for eid in dup_ents:                      # duplicates: does the KB give ONE consistent answer?
        f = rng.choice(["birth_year", "city", "email"])
        w = who(clean, C.loc[eid].to_dict(), f)
        if w and (eid, f) not in seen:
            qs.append({"kind": "DUPLICATE", "field": f, "entity": eid,
                       "q": ASK[f].format(w=w), "gold": _s(C.at[eid, f])})

    # ---- knowledge bases ----
    chA = [_chunk(r.record_id, r, []) for _, r in dirty.iterrows()]
    ent = rep.entities.set_index("entity_id")
    review = defaultdict(set)
    for x in rep.actions:
        if x["status"] != "applied" and x["col"]:
            for r in x["records"].split(","):
                review[rep.record_to_entity.get(r, r)].add(x["col"])
    chB = [_chunk(eid, r, [], f"Source records: {r['source_records']}.") for eid, r in ent.iterrows()]
    t0 = time.perf_counter(); stA, _ = build_store(chA, "storage/rt_dirty"); tA = time.perf_counter() - t0
    t0 = time.perf_counter(); stB, _ = build_store(chB, "storage/rt_repaired"); tB = time.perf_counter() - t0
    byA, byB = {c["id"]: c for c in chA}, {c["id"]: c for c in chB}

    def run(st, by, conflict_check):
        res = defaultdict(lambda: [0, 0, 0])      # correct, conflicting-evidence, total
        lat = []
        for q in qs:
            t = time.perf_counter(); top = retrieve(st, q["q"], k=5); lat.append(time.perf_counter() - t)
            ans = by[top[0]]["fields"].get(q["field"], "") if top else ""
            res[q["kind"]][0] += ans.lower() == q["gold"].lower()
            res[q["kind"]][2] += 1
            if conflict_check:   # do other retrieved chunks for the same person disagree?
                same = [by[c]["fields"] for c in top[1:] if by[c]["fields"]["name"].lower() == by[top[0]]["fields"]["name"].lower()]
                res[q["kind"]][1] += any(f.get(q["field"]) and f.get(q["field"]) != ans for f in same)
        tot = [sum(v[i] for v in res.values()) for i in range(3)]
        out = {k: f"{v[0]}/{v[2]} = {v[0] / v[2]:.0%}" for k, v in res.items()}
        out["ALL"] = f"{tot[0]}/{tot[2]} = {tot[0] / tot[2]:.1%}"
        if conflict_check:
            out["answers_with_contradicting_evidence"] = f"{tot[1]}/{tot[2]}"
        lat.sort()
        out["query_ms_p50"] = round(1000 * lat[len(lat) // 2], 1)
        return out

    res = {"questions": len(qs),
           "BEFORE_dirty_kb": run(stA, byA, True) | {"chunks": len(chA), "build_s": round(tA, 1)},
           "AFTER_repaired_kb": run(stB, byB, True) | {"chunks": len(chB), "build_s": round(tB, 1)}}
    os.makedirs("out_repair", exist_ok=True)
    json.dump(res, open("out_repair/rag_targeted.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
