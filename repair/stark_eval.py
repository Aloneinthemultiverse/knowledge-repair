"""STaRK-Prime retrieval benchmark on our turbovec stack (truthguard ChunkStore, unchanged).

Metrics follow the STaRK leaderboard: Hit@1, Hit@5, Recall@20, MRR on the test split.

    python -m repair.stark_eval build      # docs + turbovec index (slow, once)
    python -m repair.stark_eval eval [--limit N]
"""
import ast
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = "bench/stark_prime"
STORE = "storage/stark_prime"


def node_doc(n, rel):
    d = n.get("details", "")
    try:
        d = ast.literal_eval(d) if isinstance(d, str) and d.startswith("{") else d
    except Exception:
        pass
    parts = [f"{n['type']}: {n['name']}."]
    if isinstance(d, dict):
        for k in ("alias", "summary", "description", "mayo_symptoms", "mayo_causes", "indication",
                  "mechanism_of_action", "pharmacodynamics"):
            if d.get(k):
                parts.append(f"{k}: {str(d[k])[:300]}")
    elif d:
        parts.append(str(d)[:600])
    for et, names in rel.items():
        parts.append(f"{et}: {', '.join(names[:6])}")
    return " ".join(parts)[:2000]


def build():
    import torch
    P = f"{ROOT}/skb/prime/processed"
    ni = pickle.load(open(f"{P}/node_info.pkl", "rb"))
    et = pickle.load(open(f"{P}/edge_type_dict.pkl", "rb"))
    ei = torch.load(f"{P}/edge_index.pt").numpy()
    ety = torch.load(f"{P}/edge_types.pt").numpy()
    rel = defaultdict(lambda: defaultdict(list))
    for s, d, t in zip(ei[0], ei[1], ety):
        r = rel[int(s)][et[int(t)]]
        if len(r) < 6:
            r.append(ni[int(d)]["name"])
    chunks = [{"id": str(i), "text": node_doc(ni[i], rel[i]), "source_file": "stark-prime", "page": 0,
               "extraction": "structured", "content_type": "prose"} for i in sorted(ni)]
    os.makedirs(STORE, exist_ok=True)
    json.dump(chunks, open(f"{STORE}/chunks.json", "w", encoding="utf-8"))
    from truthguard.chunk_store import ChunkStore
    t = time.perf_counter()
    st = ChunkStore(STORE)
    st.embedder.max_seq_length = 256
    eng = st.build()
    print(json.dumps({"nodes": len(chunks), "engine": eng, "build_s": round(time.perf_counter() - t, 1)}))


def evaluate(limit=None, k=100):
    from truthguard.chunk_store import ChunkStore
    st = ChunkStore(STORE)
    qa = pd.read_csv(f"{ROOT}/qa/prime/stark_qa/stark_qa.csv").set_index("id")
    test = [int(x) for x in open(f"{ROOT}/qa/prime/split/test.index").read().split()]
    if limit:
        test = test[:limit]
    m = defaultdict(list)
    lat = []
    for qid in test:
        q = qa.loc[qid]
        gold = set(str(x) for x in ast.literal_eval(q.answer_ids))
        t = time.perf_counter()
        score = defaultdict(float)
        for hits in (st.vector_search(q.query, k), st.keyword_search(q.query, k)):
            for r, (cid, _) in enumerate(hits):
                score[cid] += 1 / (60 + r)
        top = [c for c, _ in sorted(score.items(), key=lambda x: -x[1])][:k]
        lat.append(time.perf_counter() - t)
        m["hit@1"].append(float(top[:1] and top[0] in gold))
        m["hit@5"].append(float(any(c in gold for c in top[:5])))
        m["recall@20"].append(len(gold & set(top[:20])) / len(gold))
        rr = next((1 / (i + 1) for i, c in enumerate(top) if c in gold), 0.0)
        m["mrr"].append(rr)
    out = {k_: round(100 * float(np.mean(v)), 2) for k_, v in m.items()}
    lat.sort()
    out.update({"questions": len(test), "query_ms_p50": round(1000 * lat[len(lat) // 2], 1),
                "query_ms_p95": round(1000 * lat[int(.95 * len(lat)) - 1], 1)})
    sizes = {f: round(os.path.getsize(f"{STORE}/{f}") / 1e6, 1) for f in os.listdir(STORE)}
    out["storage_MB"] = sizes
    print(json.dumps(out, indent=1))
    json.dump(out, open("out_repair/stark_prime.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        evaluate(lim)
