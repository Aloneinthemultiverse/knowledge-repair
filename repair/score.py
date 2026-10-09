"""Quality measurement: (A) against clean ground truth, (B) truth-free DQ score."""
import itertools
from collections import defaultdict

import pandas as pd

from .synth import PEOPLE_COLS


def _eq(a, b):
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return False
    def n(x):
        x = str(x).strip()
        return x[:-2] if x.endswith(".0") else x
    return n(a) == n(b)


def against_truth(rep, clean, truth):
    clean = clean.set_index("entity_id")
    ent = rep.entities.set_index("entity_id")
    dirty = rep.orig.set_index("record_id")
    before = after = harm = fixed = total = 0
    for rid, eid in truth.items():
        for col in PEOPLE_COLS:
            gold = clean.at[eid, col]
            b = _eq(dirty.at[rid, col], gold)
            a = _eq(ent.at[rep.record_to_entity[rid], col], gold)
            total += 1; before += b; after += a
            harm += b and not a
            fixed += a and not b
    # duplicate detection: pairwise precision/recall over records
    def pairs(groups):
        return {p for g in groups.values() for p in itertools.combinations(sorted(g), 2)}
    tg, pg = defaultdict(list), defaultdict(list)
    for r, e in truth.items():
        tg[e].append(r)
    for r, e in rep.record_to_entity.items():
        pg[e].append(r)
    tp, pp = pairs(tg), pairs(pg)
    p = len(tp & pp) / len(pp) if pp else 1.0
    r = len(tp & pp) / len(tp) if tp else 1.0
    return {
        "cell_accuracy_before": round(before / total, 4),
        "cell_accuracy_after": round(after / total, 4),
        "cells_fixed": fixed, "cells_broken (harm)": harm, "cells_total": total,
        "harm_rate": round(harm / total, 4),
        "dup_precision": round(p, 4), "dup_recall": round(r, 4),
        "dup_f1": round(2 * p * r / (p + r), 4) if p + r else 0,
        "true_entities": len(tg), "entities_after": len(pg), "records_before": len(truth),
    }


def dq_score(people, id_col, rels=None, valid_ids=None):
    """Truth-free score = mean(completeness, validity, consistency, ref_integrity)."""
    df = people.copy()
    def empty(v):
        return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip().lower() in {"", "n/a", "unknown", "none"}
    completeness = 1 - df[PEOPLE_COLS].map(empty).values.mean()
    cc = df[~df.city.map(empty)].groupby("city").country.agg(lambda s: s.value_counts().iloc[0] / len(s))
    consistency = cc.mean() if len(cc) else 1.0
    uniq = df.email.dropna()
    uniq = uniq[~uniq.map(empty)]
    uniqueness = uniq.nunique() / len(uniq) if len(uniq) else 1.0
    ref = 1.0
    if rels is not None and len(rels):
        ok = rels.src.isin(valid_ids) & rels.dst.isin(valid_ids)
        ref = ok.mean()
    parts = {"completeness": completeness, "consistency(city->country)": consistency,
             "uniqueness(email)": uniqueness, "ref_integrity": ref}
    parts = {k: round(float(v), 4) for k, v in parts.items()}
    parts["DQ"] = round(sum(parts.values()) / 4, 4)
    return parts
