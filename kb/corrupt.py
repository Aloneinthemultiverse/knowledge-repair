"""Seeded corruption of the clean KB (BART/REIN-style error injection) with a full error log.

Every injected error is one row of `log` (table, kind, id, col, old, new). Duplicate
records get fresh ids; `truth[table][id] -> clean id` is the ground-truth identity map.
"""
import random

import pandas as pd

NICK = {"William": "Bill", "Robert": "Bob", "Richard": "Dick", "James": "Jim", "John": "Jack",
        "Charles": "Charlie", "Edward": "Ed", "Michael": "Mike", "Thomas": "Tom", "Joseph": "Joe"}
PLACEHOLDERS = ["", "N/A", "unknown", "?"]


def typo(s, rng):
    if not s or len(s) < 4:
        return s
    i = rng.randint(1, len(s) - 3)
    op = rng.choice("sdi")
    if op == "s":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "d":
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def person_variant(name, rng):
    parts = name.split()
    if len(parts) < 2:
        return typo(name, rng)
    f, l = parts[0], parts[-1]
    opts = [f"{l}, {' '.join(parts[:-1])}", f"{f[0]}. {' '.join(parts[1:])}", name.upper(), typo(name, rng)]
    if f in NICK:
        opts.append(" ".join([NICK[f]] + parts[1:]))
    return rng.choice(opts)


def place_variant(name, country, rng):
    opts = [name.upper(), typo(name, rng), f"{name}, {country}" if country else name + " City"]
    return rng.choice(opts)


def shift_year(d, k):
    return f"{int(d[:4]) + k:04d}{d[4:]}" if d else d


def corrupt(kb, seed=7, dup=0.12, typo_r=0.05, miss=0.05, contra=0.05):
    rng = random.Random(seed)
    log, truth = [], {}
    P, L, E, R = (kb[t].copy() for t in ("people", "places", "events", "relationships"))

    def add_dups(df, idc, prefix, vary):
        rows, tr = [], {i: i for i in df[idc]}
        for _, r in df.iterrows():
            if rng.random() < dup:
                d = dict(r)
                d[idc] = f"{prefix}{rng.randint(10**6, 10**7 - 1)}"
                vary(d)
                rows.append(d)
                tr[d[idc]] = r[idc]
                log.append({"table": prefix, "kind": "DUPLICATE", "id": d[idc], "col": None,
                            "old": r[idc], "new": d[idc]})
        return pd.concat([df, pd.DataFrame(rows)], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True), tr

    # ---- duplicates in every entity table ----
    def vp(d):
        d["name"] = person_variant(d["name"], rng)
        if rng.random() < 0.4:
            d["birth_place_id"] = None
    P, truth["people"] = add_dups(P, "person_id", "XP", vp)
    L, truth["places"] = add_dups(L, "place_id", "XL", lambda d: d.update(name=place_variant(d["name"], d["country"], rng)))
    E, truth["events"] = add_dups(E, "event_id", "XE", lambda d: d.update(name=typo(d["name"], rng)))

    def cell(df, table, idc, i, col, new, kind):
        log.append({"table": table, "kind": kind, "id": df.at[i, idc], "col": col, "old": df.at[i, col], "new": new})
        df.at[i, col] = new

    # ---- people: typos, missing, contradictions ----
    for i in P.index:
        if rng.random() < typo_r:
            cell(P, "XP", "person_id", i, "name", typo(P.at[i, "name"], rng), "TYPO")
        if rng.random() < miss:
            col = rng.choice(["birth_date", "birth_place_id", "gender"])
            if P.at[i, col]:
                cell(P, "XP", "person_id", i, col, rng.choice(PLACEHOLDERS), "MISSING")
        if rng.random() < contra and str(P.at[i, "birth_date"])[:4].isdigit():
            k = rng.choice([-40, -30, 30, 40, 60])          # pushes award age out of a human range
            cell(P, "XP", "person_id", i, "birth_date", shift_year(P.at[i, "birth_date"], k), "CONTRADICTION")
        elif rng.random() < contra / 2 and str(P.at[i, "death_date"])[:4].isdigit() and str(P.at[i, "birth_date"])[:4].isdigit():
            b, d = P.at[i, "birth_date"], P.at[i, "death_date"]       # swapped dates: died before born
            cell(P, "XP", "person_id", i, "birth_date", d, "CONTRADICTION")
            cell(P, "XP", "person_id", i, "death_date", b, "CONTRADICTION")
    # ---- places: typos, missing / wrong country ----
    countries = sorted(set(c for c in L.country if c))
    for i in L.index:
        if rng.random() < typo_r:
            cell(L, "XL", "place_id", i, "name", typo(L.at[i, "name"], rng), "TYPO")
        if rng.random() < miss and L.at[i, "country"]:
            cell(L, "XL", "place_id", i, "country", rng.choice(PLACEHOLDERS), "MISSING")
        elif rng.random() < typo_r and L.at[i, "country"]:
            cell(L, "XL", "place_id", i, "country", typo(L.at[i, "country"], rng), "TYPO")
        elif rng.random() < contra / 2 and L.at[i, "country"]:
            cell(L, "XL", "place_id", i, "country", rng.choice(countries), "CONTRADICTION")
    # ---- events: wrong year / wrong ceremony place ----
    for i in E.index:
        if rng.random() < contra:
            cell(E, "XE", "event_id", i, "year", str(int(E.at[i, "year"]) + rng.choice([-2, -1, 1, 3])), "CONTRADICTION")
        if rng.random() < contra:
            cell(E, "XE", "event_id", i, "place_id", "Q585" if E.at[i, "place_id"] != "Q585" else "Q1754", "CONTRADICTION")
    # ---- relationships: point to duplicate copies, wrong years, missing inverses, broken refs ----
    copies = {t: {} for t in truth}
    for t, tr in truth.items():
        for new, old in tr.items():
            if new != old:
                copies[t].setdefault(old, []).append(new)
    for i in R.index:
        for side, t in (("src", "people"), ("dst", "events" if R.at[i, "rel"] == "received" else "people")):
            if R.at[i, side] in copies[t] and rng.random() < 0.5:
                R.at[i, side] = rng.choice(copies[t][R.at[i, side]])   # same entity, other copy (not an error)
        if R.at[i, "rel"] == "received" and rng.random() < contra:
            cell(R, "XR", "src", i, "year", str(int(R.at[i, "year"]) + rng.choice([-1, 1, 2])), "CONTRADICTION")
    drop = [i for i in R.index if R.at[i, "rel"] in ("child", "spouse") and rng.random() < 0.3]
    for i in drop:
        log.append({"table": "XR", "kind": "MISSING", "id": R.at[i, "src"], "col": R.at[i, "rel"],
                    "old": R.at[i, "dst"], "new": None})
    R = R.drop(index=drop)
    for i in R.sample(frac=0.02, random_state=seed).index:
        cell(R, "XR", "src", i, "dst", f"Q9{rng.randint(10**6, 10**7)}", "BROKEN_REF")
    for (t, df, idc) in (("XP", P, "person_id"), ("XL", L, "place_id")):
        pass
    for i in P.sample(frac=0.02, random_state=seed + 1).index:
        cell(P, "XP", "person_id", i, "birth_place_id", f"Q9{rng.randint(10**6, 10**7)}", "BROKEN_REF")
    return {"people": P, "places": L, "events": E, "relationships": R.reset_index(drop=True)}, truth, pd.DataFrame(log)
