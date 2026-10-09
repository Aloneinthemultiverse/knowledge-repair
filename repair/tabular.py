"""Generic table repair (any schema) + benchmark scoring in the standard
Raha/Baran/HoloClean protocol: repair precision / recall / F1 over cells.

Method: discover approximate functional dependencies A -> B from the dirty data,
then inside each A-group repair B cells that disagree with a clear majority.
Iterated, because fixing a LHS cell unlocks more groups. Every change is logged
with its FD, group size and vote share (traceability).

    python -m repair.tabular bench/hospital
"""
import sys
import time
from collections import Counter

import pandas as pd

from .formats import repair_formats
from .truth import detect_key_source, resolve


def discover_fds(df, min_conf=0.90, min_group_rows=0.3, skip=("index", "tuple_id")):
    """A->B holds if, in groups of A with >=2 rows, the majority B covers >= min_conf of rows."""
    cols = [c for c in df.columns if c not in skip]
    fds = []
    for a in cols:
        ga = df.groupby(a, sort=False)
        sizes = ga.size()
        multi = sizes[sizes >= 2]
        if multi.sum() < min_group_rows * len(df) or df[a].nunique() < 2:
            continue                                   # A is (near) unique or constant: useless
        for b in cols:
            if a == b:
                continue
            maj = df[df[a].isin(multi.index)].groupby(a, sort=False)[b].agg(lambda s: s.value_counts().iloc[0]).sum()
            conf = maj / multi.sum()
            if conf >= min_conf:
                fds.append((a, b, round(conf, 4)))
    return fds


def repair(df, fds, min_share=0.5, min_votes=2, rounds=3, max_minority=2, min_major=0.6):
    df = df.copy()
    log = []
    for rnd in range(rounds):
        changed = 0
        for a, b, conf in fds:
            for key, idx in df.groupby(a, sort=False).groups.items():
                if len(idx) < 3:
                    continue
                vals = Counter(df.loc[idx, b])
                (best, n), *rest = vals.most_common()
                if not rest or n < min_votes or n / len(idx) <= min_share:
                    continue
                if rest[0][1] == n:
                    continue                           # tie: don't guess
                for i in idx:
                    v = df.at[i, b]
                    # errors are rare one-offs: a value repeated 3+ times in its group is a
                    # legitimate variant (accidental FD like city->owner), never "fix" it
                    if v != best and vals[v] <= max_minority and n / len(idx) >= min_major:
                        log.append({"row": i, "col": b, "before": v, "after": best,
                                    "rule": f"{a}->{b}", "group": key, "votes": f"{n}/{len(idx)}",
                                    "fd_conf": conf, "round": rnd})
                        df.at[i, b] = best
                        changed += 1
        if not changed:
            break
    return df, pd.DataFrame(log)


def score(dirty, repaired, clean):
    cols = [c for c in clean.columns if c != "index"]
    err = (dirty[cols] != clean[cols])
    chg = (dirty[cols] != repaired[cols])
    ok = chg & (repaired[cols] == clean[cols])
    n_err, n_chg, n_ok = int(err.values.sum()), int(chg.values.sum()), int(ok.values.sum())
    p = n_ok / n_chg if n_chg else 0.0
    r = n_ok / n_err if n_err else 0.0
    return {"errors": n_err, "repairs": n_chg, "correct_repairs": n_ok,
            "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def main(path):
    dirty = pd.read_csv(f"{path}/dirty.csv", dtype=str, keep_default_na=False)
    clean = pd.read_csv(f"{path}/clean.csv", dtype=str, keep_default_na=False)
    clean.columns = dirty.columns          # same schema, different header spelling
    t0 = time.perf_counter()
    flog = []
    stage = repair_formats(dirty, flog)                       # 1. formats
    key, src = detect_key_source(stage)                       # 2. multi-source conflicts
    if key:
        stage, weights = resolve(stage, key, src, flog, skip=("index", "tuple_id"))
    fds = discover_fds(stage)                                 # 3. dependency repair
    t1 = time.perf_counter()
    rep, log = repair(stage, fds)
    log = pd.concat([pd.DataFrame(flog), log], ignore_index=True)
    t2 = time.perf_counter()
    s = score(dirty, rep, clean)
    s.update({"key/source": f"{key}/{src}", "fds": len(fds), "seconds": round(t2 - t0, 2),
              "rows": len(dirty)})
    print(s)
    log.to_csv(f"{path}/repair_log.csv", index=False)
    return s


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/hospital")
