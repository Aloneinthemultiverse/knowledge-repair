"""Synthetic clean dataset + seeded corruptor with a full error log (ground truth)."""
import random
import pandas as pd

FIRST = ["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda",
         "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
         "Charles", "Karen", "Daniel", "Nancy", "Matthew", "Lisa", "Anthony", "Priya", "Arjun"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Kumar"]
CITIES = {"Paris": "France", "Lyon": "France", "Berlin": "Germany", "Munich": "Germany",
          "Chennai": "India", "Mumbai": "India", "Boston": "USA", "Chicago": "USA",
          "Toronto": "Canada", "Madrid": "Spain"}
NICK = {"William": "Bill", "Robert": "Bob", "Michael": "Mike", "Elizabeth": "Liz",
        "Richard": "Rick", "Joseph": "Joe", "Thomas": "Tom", "Daniel": "Dan",
        "Matthew": "Matt", "Anthony": "Tony", "Patricia": "Pat", "Jennifer": "Jen"}
PEOPLE_COLS = ["name", "birth_year", "city", "country", "email"]


def make_clean(n_people=300, seed=1):
    rng = random.Random(seed)
    people, used = [], set()
    while len(people) < n_people:
        f, l = rng.choice(FIRST), rng.choice(LAST)
        y = rng.randint(1940, 2005)
        if (f, l, y) in used:
            continue
        used.add((f, l, y))
        c = rng.choice(list(CITIES))
        pid = f"P{len(people):04d}"
        people.append({"entity_id": pid, "name": f"{f} {l}", "birth_year": y, "city": c,
                       "country": CITIES[c], "email": f"{f}.{l}{y % 100}@mail.com".lower()})
    people = pd.DataFrame(people)
    rels, ids = [], list(people.entity_id)
    by_year = people.set_index("entity_id").birth_year
    for _ in range(n_people // 2):
        a, b = rng.sample(ids, 2)
        if by_year[a] + 18 <= by_year[b]:
            rels.append({"src": a, "rel": "parent_of", "dst": b})
        else:
            rels.append({"src": a, "rel": "friend_of", "dst": b})
    return people, pd.DataFrame(rels).drop_duplicates()


def _typo(s, rng):
    if len(s) < 4:
        return s + "x"
    i = rng.randint(1, len(s) - 2)
    op = rng.choice(["swap", "drop", "dup", "sub"])
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "drop":
        return s[:i] + s[i + 1:]
    if op == "dup":
        return s[:i] + s[i] + s[i:]
    return s[:i] + rng.choice("aeiou") + s[i + 1:]


def _name_variant(name, rng):
    f, l = name.split(" ", 1)
    opts = [f"{l}, {f}", f"{f[0]}. {l}", name.upper(), f"{_typo(f, rng)} {l}"]
    if f in NICK:
        opts.append(f"{NICK[f]} {l}")
    return rng.choice(opts)


def corrupt(people, rels, seed=7, dup_rate=0.15, typo_rate=0.05, miss_rate=0.06,
            conflict_rate=0.04, fd_rate=0.03, broken_ref_rate=0.03):
    """Return (dirty_people with record_id, dirty_rels, truth{record->entity}, error_log)."""
    rng = random.Random(seed)
    log = []
    recs = []
    for _, p in people.iterrows():
        recs.append(dict(p))
    # duplicates
    for p in list(recs):
        if rng.random() < dup_rate:
            d = dict(p)
            d["name"] = _name_variant(p["name"], rng)
            if rng.random() < 0.5:
                d["email"] = None
            recs.append(d)
            log.append({"kind": "DUPLICATE", "entity_id": p["entity_id"], "detail": d["name"]})
    rng.shuffle(recs)
    for i, r in enumerate(recs):
        r["record_id"] = f"R{i:05d}"
    df = pd.DataFrame(recs)
    truth = dict(zip(df.record_id, df.entity_id))
    df = df.drop(columns=["entity_id"])

    def cell(kind, i, col, new):
        log.append({"kind": kind, "record_id": df.at[i, "record_id"], "col": col,
                    "old": df.at[i, col], "new": new})
        df.at[i, col] = new

    df["birth_year"] = df["birth_year"].astype(object)
    for i in df.index:
        if rng.random() < typo_rate:
            col = rng.choice(["city", "name"])
            if pd.notna(df.at[i, col]):
                cell("TYPO", i, col, _typo(str(df.at[i, col]), rng))
        if rng.random() < miss_rate:
            col = rng.choice(["city", "country", "birth_year", "email"])
            if pd.notna(df.at[i, col]):
                cell("MISSING", i, col, rng.choice([None, "", "N/A", "unknown"]))
        if rng.random() < conflict_rate and str(df.at[i, "birth_year"]).isdigit():
            cell("CONTRADICTION", i, "birth_year", int(df.at[i, "birth_year"]) + rng.choice([-3, -1, 1, 2, 5]))
        if rng.random() < fd_rate and pd.notna(df.at[i, "country"]):
            wrong = rng.choice([c for c in set(CITIES.values()) if c != df.at[i, "country"]])
            cell("FD_VIOLATION", i, "country", wrong)
    # relationships point at records (first record of each entity)
    first_rec = {}
    for rid, eid in truth.items():
        first_rec.setdefault(eid, rid)
    r = rels.copy()
    r["src"] = r["src"].map(first_rec)
    r["dst"] = r["dst"].map(first_rec)
    for i in r.index:
        if rng.random() < broken_ref_rate:
            log.append({"kind": "BROKEN_REF", "rel_index": int(i), "old": r.at[i, "dst"]})
            r.at[i, "dst"] = f"R9{rng.randint(1000, 9999)}"
    return df[["record_id"] + PEOPLE_COLS], r.reset_index(drop=True), truth, pd.DataFrame(log)
