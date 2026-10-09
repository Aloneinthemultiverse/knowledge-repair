"""Multi-source conflict resolution (truth discovery, TruthFinder-style).

When several sources describe the same entity (rows sharing a key, one row per
source), vote per attribute with source weights, then re-estimate each source's
weight as its agreement with the consensus. Iterate. Missing values are filled
from the consensus. Only runs when a key + source column pair is detected.
"""
from collections import Counter, defaultdict


def detect_key_source(df, max_sources=100):
    """source: low-cardinality column whose values rarely repeat inside a key group.
    key: column with many groups of size >= 2 that the source column is spread across."""
    best = None
    named = [c for c in df.columns if c.lower() in ("src", "source", "source_name", "provenance")]
    for s in (named or df.columns):
        ns = df[s].nunique()
        if not 2 <= ns <= max_sources:
            continue
        if not named:   # a real source never determines the attributes it reports
            if any(df.groupby(s)[c].agg(lambda x: x.value_counts().iloc[0]).sum() >= 0.9 * len(df)
                   for c in df.columns if c != s and df[c].nunique() > 1):
                continue
        for k in df.columns:
            if k == s:
                continue
            g = df.groupby(k)[s]
            sizes = g.size()
            multi = sizes[sizes >= 3]
            if len(multi) < 0.05 * df[k].nunique() or multi.sum() < 0.5 * len(df):
                continue
            distinct = (g.nunique()[multi.index] / multi).mean()   # 1.0 = every row a different source
            if distinct < 0.95:
                continue
            # sources describing the SAME entity must mostly agree; a cross-product
            # (hospital x measure) has attributes that differ on almost every row
            sub = df[df[k].isin(multi.index)]
            others = [c for c in df.columns if c not in (k, s)]
            agree = sum(sub.groupby(k)[c].agg(lambda x: x.value_counts().iloc[0] / len(x)).mean()
                        for c in others) / len(others)
            rank = (round(distinct, 2), int(multi.sum()))     # one row per source, covering most rows
            if agree >= (0.4 if named else 0.6) and (best is None or rank > best[2]):
                best = (k, s, rank)
    return (best[0], best[1]) if best else (None, None)


def resolve(df, key, src, log, iters=5, skip=()):
    df = df.copy()
    attrs = [c for c in df.columns if c not in (key, src, *skip)]
    w = defaultdict(lambda: 0.8)
    groups = df.groupby(key).groups
    for _ in range(iters):
        truth = {}
        for kv, idx in groups.items():
            for a in attrs:
                score = Counter()
                for i in idx:
                    v = df.at[i, a]
                    if v:
                        score[v] += w[df.at[i, src]]
                if score:
                    truth[(kv, a)] = score.most_common(1)[0][0]
        agree, total = Counter(), Counter()
        for kv, idx in groups.items():
            for a in attrs:
                t = truth.get((kv, a))
                for i in idx:
                    v = df.at[i, a]
                    if t and v:
                        total[df.at[i, src]] += 1
                        agree[df.at[i, src]] += v == t
        w = defaultdict(lambda: 0.8, {s: (agree[s] + 1) / (total[s] + 2) for s in total})
    for kv, idx in groups.items():
        if len(idx) < 2:
            continue
        for a in attrs:
            t = truth.get((kv, a))
            if not t:
                continue
            for i in idx:
                if df.at[i, a] != t:
                    log.append({"row": i, "col": a, "before": df.at[i, a], "after": t,
                                "rule": f"truth discovery {key}/{src}",
                                "src_weight": round(w[df.at[i, src]], 3)})
                    df.at[i, a] = t
    return df, dict(w)
