"""FEBRL person-record deduplication benchmark (Christen, ANU).

Ours: multi-key blocking + field-weighted fuzzy agreement (rapidfuzz), unsupervised,
one fixed threshold for every dataset. Baseline: recordlinkage's ECM classifier
(unsupervised Fellegi-Sunter) on the same candidate pairs.

    python -m repair.febrl_bench
"""
import itertools
import json
import time
from collections import defaultdict

import jellyfish
import pandas as pd
import recordlinkage as rl
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA
from recordlinkage.datasets import load_febrl1, load_febrl2, load_febrl3

FIELDS = {  # weight, similarity kind
    "given_name": (1.0, "name"), "surname": (1.5, "name"), "street_number": (0.5, "exact"),
    "address_1": (1.0, "text"), "address_2": (0.5, "text"), "suburb": (1.0, "text"),
    "postcode": (1.0, "code"), "state": (0.3, "exact"), "date_of_birth": (1.5, "code"),
    "soc_sec_id": (2.0, "code"),
}


def sim(kind, a, b):
    if not a or not b:
        return None
    if kind == "exact":
        return float(a == b)
    if kind == "code":        # ids/dates/postcodes: allow one swap or one wrong digit
        return 1.0 if a == b else 0.7 if OSA.distance(a, b) <= 1 else 0.0
    if kind == "name":
        s = max(OSA.normalized_similarity(a, b), fuzz.ratio(a, b) / 100)
        return s if s >= 0.75 else (0.6 if jellyfish.soundex(a) == jellyfish.soundex(b) else 0.0)
    return fuzz.token_set_ratio(a, b) / 100


def candidates(df):
    keys = [lambda r: "S" + jellyfish.soundex(r.surname) if r.surname else None,
            lambda r: "G" + jellyfish.soundex(r.given_name) if r.given_name else None,
            lambda r: "D" + r.date_of_birth if r.date_of_birth else None,
            lambda r: "I" + r.soc_sec_id if r.soc_sec_id else None,
            lambda r: "P" + r.postcode + (r.street_number or "") if r.postcode else None]
    blocks = defaultdict(list)
    for r in df.itertuples():
        for k in keys:
            kv = k(r)
            if kv:
                blocks[kv].append(r.Index)
    pairs = set()
    for ids in blocks.values():
        if len(ids) <= 60:
            pairs.update(itertools.combinations(sorted(ids), 2))
    return pairs


def score(df, pairs):
    rows = df.to_dict("index")
    out = {}
    for a, b in pairs:
        ra, rb = rows[a], rows[b]
        num = den = 0.0
        for f, (w, kind) in FIELDS.items():
            s = sim(kind, ra[f], rb[f])
            if s is not None:
                num += w * s
                den += w
        out[(a, b)] = num / den if den else 0.0
    return out


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0
    r = tp / len(gold) if gold else 0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0


def run(loader, threshold=0.50):   # calibrated on FEBRL1 only
    df, links = loader(return_links=True)
    df = df.fillna("").astype(str).apply(lambda c: c.str.strip())
    gold = {tuple(sorted(p)) for p in links}
    t = time.perf_counter()
    cand = candidates(df)
    sc = score(df, cand)
    pred = {p for p, s in sc.items() if s >= threshold}
    t_ours = time.perf_counter() - t
    blocking_recall = len(gold & cand) / len(gold)
    # baseline: recordlinkage ECM on the same candidate pairs
    t = time.perf_counter()
    idx = pd.MultiIndex.from_tuples(sorted(cand))
    cmp = rl.Compare()
    cmp.string("given_name", "given_name", method="jarowinkler", threshold=0.85, label="g")
    cmp.string("surname", "surname", method="jarowinkler", threshold=0.85, label="s")
    cmp.exact("date_of_birth", "date_of_birth", label="d")
    cmp.exact("suburb", "suburb", label="sb")
    cmp.exact("state", "state", label="st")
    cmp.string("address_1", "address_1", threshold=0.85, label="a")
    cmp.exact("postcode", "postcode", label="p")
    cmp.exact("soc_sec_id", "soc_sec_id", label="id")
    feats = cmp.compute(idx, df)
    ecm = rl.ECMClassifier(binarize=0.5)
    ecm_pred = {tuple(sorted(p)) for p in ecm.fit_predict(feats)}
    t_ecm = time.perf_counter() - t
    return {"records": len(df), "true_pairs": len(gold), "candidate_pairs": len(cand),
            "blocking_recall": round(blocking_recall, 4),
            "ours_P_R_F1": prf(pred, gold), "ours_s": round(t_ours, 2),
            "ecm_P_R_F1": prf(ecm_pred, gold), "ecm_s": round(t_ecm, 2)}


if __name__ == "__main__":
    res = {f.__name__.replace("load_", ""): run(f) for f in (load_febrl1, load_febrl2, load_febrl3)}
    json.dump(res, open("out_repair/febrl.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
