"""Data-quality score that needs no clean copy.

The same checks run on the corrupted and on the repaired knowledge base, so
before/after are measured identically. Each dimension is in [0, 1]; DQ is their mean.

  completeness   share of required cells that hold a value (death_date may legitimately be empty)
  validity       share of filled cells that are well-formed (dates, placeholders)
  uniqueness     1 - share of records that look like copies (same normalised key)
  consistency    1 - share of entities breaking a cross-record rule
  integrity      share of references that point to an existing record
"""
import re
from collections import Counter

PH = {"", "n/a", "na", "unknown", "?", "none", "null", "-"}
REQUIRED = {"people": ["name", "birth_date", "gender", "birth_place_id"],
            "places": ["name", "country"], "events": ["name", "prize", "year", "place_id"]}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _v(x):
    return None if x is None or (isinstance(x, float) and x != x) or str(x).strip().lower() in PH else str(x).strip()


def _yr(x):
    x = _v(x)
    return int(x[:4]) if x and x[:4].isdigit() else None


def _person_key(n):
    n = (_v(n) or "").lower()
    if "," in n:
        last, first = [p.strip() for p in n.split(",", 1)]
        n = f"{first} {last}"
    parts = n.split()
    return f"{parts[0][0]}|{parts[-1]}" if len(parts) > 1 else n


def dq(kb):
    P, L, E, R = kb["people"], kb["places"], kb["events"], kb["relationships"]
    out, viol = {}, Counter()
    # completeness / validity
    filled = total = valid = nvalid = 0
    for t, cols in REQUIRED.items():
        for c in cols:
            for v in kb[t][c]:
                total += 1
                vv = _v(v)
                filled += vv is not None
                if vv is not None:
                    nvalid += 1
                    valid += bool(DATE.match(vv)) if c == "birth_date" else (vv.isdigit() and len(vv) == 4) if c == "year" else True
    out["completeness"] = filled / total
    out["validity"] = valid / nvalid if nvalid else 1.0
    # uniqueness: records sharing a normalised identity key
    keys = Counter((_person_key(r.name), (_v(r.birth_date) or "")[5:]) for r in P.itertuples())
    pk = Counter((re.sub(r",.*$", "", (_v(r.name) or "")).lower().strip()) for r in L.itertuples())
    ek = Counter((r.prize, re.search(r"(\d{4})\s*$", _v(r.name) or "") and re.search(r"(\d{4})\s*$", r.name).group(1)) for r in E.itertuples())
    dup = sum(n - 1 for k in (keys, pk, ek) for n in k.values() if n > 1)
    out["uniqueness"] = 1 - dup / (len(P) + len(L) + len(E))
    # consistency: cross-record rules
    ids = {"people": set(P.person_id), "places": set(L.place_id), "events": set(E.event_id)}
    ey = {r.event_id: _yr(r.year) for r in E.itertuples()}
    born = {r.person_id: _yr(r.birth_date) for r in P.itertuples()}
    died = {r.person_id: _yr(r.death_date) for r in P.itertuples()}
    for r in E.itertuples():
        m = re.search(r"(\d{4})\s*$", _v(r.name) or "")
        if m and _v(r.year) and m.group(1) != _v(r.year):
            viol["event year != label"] += 1
    for pid in ids["people"]:
        b, d = born.get(pid), died.get(pid)
        if b and d and not 15 <= d - b <= 110:
            viol["impossible lifespan"] += 1
    rel_set = set(zip(R.src, R.rel, R.dst))
    for r in R.itertuples():
        if r.rel == "received" and r.dst in ey:
            if _v(r.year) and ey[r.dst] and int(_v(r.year)) != ey[r.dst]:
                viol["award year != event year"] += 1
            b = born.get(r.src)
            if b and ey[r.dst] and not 17 <= ey[r.dst] - b <= 100:
                viol["impossible age at award"] += 1
        if r.rel in ("father", "mother") and (r.dst, "child", r.src) not in rel_set:
            viol["parent link without child link"] += 1
        if r.rel == "spouse" and (r.dst, "spouse", r.src) not in rel_set:
            viol["one-sided spouse link"] += 1
    n_ent = len(P) + len(E) + len(R)
    out["consistency"] = 1 - sum(viol.values()) / n_ent
    # referential integrity
    refs = ok = 0
    for c in ("birth_place_id", "death_place_id"):
        for v in P[c]:
            if _v(v):
                refs += 1
                ok += v in ids["places"]
    for v in E.place_id:
        if _v(v):
            refs += 1
            ok += v in ids["places"]
    for r in R.itertuples():
        refs += 2
        ok += (r.src in ids["people"]) + (r.dst in (ids["events"] if r.rel == "received" else ids["people"]))
    out["integrity"] = ok / refs if refs else 1.0
    out = {k: round(v, 4) for k, v in out.items()}
    out["DQ"] = round(sum(out.values()) / len(out), 4)
    out["violations"] = dict(viol)
    return out
