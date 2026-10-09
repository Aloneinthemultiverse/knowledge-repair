"""Score a KB repair against the clean Wikidata KB.

  * cell accuracy per table, before (corrupted record) vs after (its canonical entity)
  * harm: cells correct before and wrong after
  * per error kind: share of injected errors whose cell is correct after repair
  * duplicate detection: pairwise precision / recall / F1 per entity table
  * relationships: precision / recall of the (src, rel, dst, year) set, mapped to true ids
"""
import itertools
from collections import Counter, defaultdict

COLS = {"people": ["name", "birth_date", "death_date", "gender", "birth_place_id", "death_place_id"],
        "places": ["name", "country"], "events": ["name", "year", "place_id"]}
IDC = {"people": "person_id", "places": "place_id", "events": "event_id"}
REF = {("people", "birth_place_id"): "places", ("people", "death_place_id"): "places", ("events", "place_id"): "places"}


def _n(v):
    return None if v is None or (isinstance(v, float) and v != v) or str(v).strip() == "" else str(v).strip()


def score(rep, clean, truth, log):
    res, harm_total, cells_total = {}, 0, 0
    # canonical id -> true id (majority of its source records)
    can2true = {}
    for t in COLS:
        members = defaultdict(list)
        for rid, cid in rep.canon[t].items():
            members[cid].append(truth[t].get(rid, rid))
        can2true[t] = {c: Counter(m).most_common(1)[0][0] for c, m in members.items()}

    def as_true(t, col, v):
        if (t, col) in REF and v is not None:
            tt = REF[(t, col)]
            return can2true[tt].get(rep.canon[tt].get(v, v), truth[tt].get(v, v))
        return v

    out_tables = {"people": rep.P, "places": rep.L, "events": rep.E}
    fixed_by_kind = defaultdict(lambda: [0, 0])
    for t, cols in COLS.items():
        gold = clean[t].set_index(IDC[t])
        orig = rep.orig[t].set_index(IDC[t])
        rep_t = out_tables[t]
        before = after = harm = 0
        for rid in orig.index:
            tid = truth[t].get(rid)
            if tid not in gold.index:
                continue
            cid = rep.canon[t][rid]
            for c in cols:
                g = _n(gold.at[tid, c])
                b = _n(as_true(t, c, _n(orig.at[rid, c])) if (t, c) in REF else orig.at[rid, c])
                a = _n(as_true(t, c, _n(rep_t.at[cid, c])) if (t, c) in REF else rep_t.at[cid, c])
                before += b == g
                after += a == g
                harm += (b == g) and (a != g)
                cells_total += 1
        n = len(orig.index) * len(cols)
        harm_total += harm
        res[f"{t}_cell_acc"] = f"{before / n:.3f} -> {after / n:.3f}"
        res[f"{t}_harm"] = harm
        # duplicates
        tg, pg = defaultdict(list), defaultdict(list)
        for rid in orig.index:
            tg[truth[t][rid]].append(rid)
            pg[rep.canon[t][rid]].append(rid)
        pairs = lambda gr: {p for g in gr.values() for p in itertools.combinations(sorted(g), 2)}
        tp_, pp = pairs(tg), pairs(pg)
        p = len(tp_ & pp) / len(pp) if pp else 1.0
        r = len(tp_ & pp) / len(tp_) if tp_ else 1.0
        res[f"{t}_dup_P/R/F1"] = f"{p:.3f}/{r:.3f}/{(2 * p * r / (p + r) if p + r else 0):.3f}"
        # error-kind repair rate for cell errors
        tab = {"people": "XP", "places": "XL", "events": "XE"}[t]
        for _, e in log[(log.table == tab) & log.col.notna()].iterrows():
            if e.col not in cols or e.id not in rep.canon[t]:
                continue
            tid = truth[t][e.id]
            g = _n(gold.at[tid, e.col])
            cid = rep.canon[t][e.id]
            a = _n(as_true(t, e.col, _n(rep_t.at[cid, e.col])) if (t, e.col) in REF else rep_t.at[cid, e.col])
            k = f"{t}:{e.kind}"
            fixed_by_kind[k][0] += a == g
            fixed_by_kind[k][1] += 1
    # relationships
    def key(t, v):
        return truth[t].get(v, v)
    gold_rel = {(r.src, r.rel, r.dst, _n(r.year)) for r in clean["relationships"].itertuples()}
    before_rel = {(key("people", r.src), r.rel, key("events" if r.rel == "received" else "people", r.dst), _n(r.year))
                  for r in rep.orig["relationships"].itertuples()}
    after_rel = {(can2true["people"].get(r.src, r.src), r.rel,
                  can2true["events" if r.rel == "received" else "people"].get(r.dst, r.dst), _n(r.year))
                 for r in rep.Rout.itertuples()}
    for name, s in (("before", before_rel), ("after", after_rel)):
        p = len(s & gold_rel) / len(s) if s else 0
        r = len(s & gold_rel) / len(gold_rel)
        res[f"relationships_{name}_P/R/F1"] = f"{p:.3f}/{r:.3f}/{2 * p * r / (p + r):.3f}"
    orig_R = rep.orig["relationships"]
    flagged_rows = {a["records"] for a in rep.actions if a["table"] == "relationships" and a["kind"] == "FLAG"}
    for _, e in log[log.table == "XR"].iterrows():
        k = f"relationships:{e.kind}"
        if e.kind == "MISSING":                       # dropped family link: is it back?
            ok = any(gs == key("people", e.id) and gr == e.col and gd == key("people", e.old) for gs, gr, gd, _ in after_rel)
        elif e.kind == "CONTRADICTION":               # wrong award year: is the gold triple present?
            row = orig_R[(orig_R.year == e.new)]
            ok = any((key("people", r.src), "received", key("events", r.dst)) in {(a, b, c) for a, b, c, _ in after_rel & gold_rel}
                     for r in row.itertuples() if key("people", r.src) == key("people", e.id))
        else:                                         # broken ref: unrecoverable, success = flagged
            ok = any(str(e.new) in a["before"] if isinstance(a["before"], str) else False
                     for a in rep.actions if a["table"] == "relationships" and a["kind"] == "FLAG")
            k += " (flagged)"
        fixed_by_kind[k][0] += bool(ok)
        fixed_by_kind[k][1] += 1
    # detection: was every injected cell error surfaced as an issue on that record?
    touched = defaultdict(set)
    for i in rep.issues:
        for rr in i["records"].split(","):
            touched[rr].add(i["col"])
    det = defaultdict(lambda: [0, 0])
    G = {t: clean[t].set_index(IDC[t]) for t in COLS}
    for _, e in log[log.table.isin(["XP", "XL", "XE"]) & log.col.notna()].iterrows():
        k = f"{ {'XP': 'people', 'XL': 'places', 'XE': 'events'}[e.table]}:{e.kind}"
        tb = {'XP': 'people', 'XL': 'places', 'XE': 'events'}[e.table]
        cols_hit = touched.get(e.id, set()) | touched.get(rep.canon[tb].get(e.id), set())
        g = _n(G[tb].at[truth[tb][e.id], e.col])
        cid = rep.canon[tb][e.id]
        v = out_tables[tb].at[cid, e.col]
        fixed = _n(as_true(tb, e.col, _n(v)) if (tb, e.col) in REF else v) == g
        det[k][0] += any(e.col in (c or "") for c in cols_hit) or fixed
        det[k][1] += 1
    res["detected_by_error"] = {k: f"{a}/{b} = {a / b:.0%}" for k, (a, b) in sorted(det.items())}
    res["fix_rate_by_error"] = {k: f"{a}/{b} = {a / b:.0%}" for k, (a, b) in sorted(fixed_by_kind.items()) if b and a <= b}
    res["harm_total"] = f"{harm_total}/{cells_total}"
    res["actions"] = Counter(f"{a['kind']}/{a['status']}" for a in rep.actions)
    return res
