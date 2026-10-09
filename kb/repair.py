"""Knowledge-base repair across linked tables (people, places, events, relationships).

Order matters: places -> events -> people -> relationships, because each step uses
the repaired, deduplicated tables before it as evidence. Originals are never mutated;
every change is an Action (issue, records, column, before, after, confidence, rule, why).
Confidence bands: >=0.90 applied, 0.60-0.90 applied+needs_review, <0.60 flag only.
"""
import itertools
import re
from collections import Counter, defaultdict

import jellyfish
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA

from .corrupt import NICK

PH = {"", "n/a", "na", "unknown", "?", "none", "null", "-"}
NICK_REV = {v.lower(): k.lower() for k, v in NICK.items()}
AUTO, REVIEW = 0.90, 0.60


def year(d):
    return int(str(d)[:4]) if d and str(d)[:4].isdigit() else None


class KBRepairer:
    def __init__(self, kb):
        self.orig = {k: v.copy() for k, v in kb.items()}
        self.P = kb["people"].copy().set_index("person_id", drop=False)
        self.L = kb["places"].copy().set_index("place_id", drop=False)
        self.E = kb["events"].copy().set_index("event_id", drop=False)
        self.R = kb["relationships"].copy()
        self.issues, self.actions = [], []
        self.canon = {"people": {}, "places": {}, "events": {}}   # record id -> canonical id
        self.lineage = []                                          # one row per repaired value

    # ------------------------------------------------------------ bookkeeping
    def issue(self, table, kind, records, detail, col=None):
        iid = f"I{len(self.issues):05d}"
        self.issues.append({"issue_id": iid, "table": table, "kind": kind, "col": col,
                            "records": ",".join(map(str, records)), "detail": detail})
        return iid

    def act(self, iid, kind, table, records, col, before, after, conf, rule, why):
        st = "applied" if conf >= AUTO else "needs_review" if conf >= REVIEW else "flagged_only"
        self.actions.append({"action_id": f"A{len(self.actions):05d}", "issue_id": iid, "table": table,
                             "kind": kind, "records": ",".join(map(str, records)), "col": col,
                             "before": before, "after": after, "confidence": round(conf, 3),
                             "status": st, "rule": rule, "explanation": why})
        return st != "flagged_only"

    def setv(self, df, table, rid, col, new, kind, conf, rule, why, issue_kind):
        old = df.at[rid, col]
        iid = self.issue(table, issue_kind, [rid], why, col)
        if self.act(iid, kind, table, [rid], col, old, new, conf, rule, why):
            df.at[rid, col] = new

    # ------------------------------------------------------------ 1. missing / placeholders
    def placeholders(self):
        for name, df in (("people", self.P), ("places", self.L), ("events", self.E)):
            for col in df.columns:
                for rid, v in df[col].items():
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        df.at[rid, col] = None
                    elif str(v).strip().lower() in PH:
                        iid = self.issue(name, "MISSING", [rid], f"{col}={v!r} is a placeholder", col)
                        self.act(iid, "NORMALIZE", name, [rid], col, v, None, 1.0, "placeholder",
                                 f"{v!r} carries no value; stored as null")
                        df.at[rid, col] = None

    # ------------------------------------------------------------ 2. places
    def vocab_fix(self, df, table, col, min_common=3, min_sim=80):
        """Noisy-channel correction: a rare value close to a frequent one is a typo of it."""
        cnt = Counter(v for v in df[col] if v)
        common = [v for v, c in cnt.items() if c >= min_common]
        for v, c in list(cnt.items()):
            if c >= min_common:
                continue
            best = max(common, key=lambda x: OSA.normalized_similarity(v.lower(), x.lower()), default=None)
            if not best:
                continue
            sim = OSA.normalized_similarity(v.lower(), best.lower()) * 100
            if sim < min_sim:
                continue
            conf = sim / 100 * cnt[best] / (cnt[best] + c)
            for rid in df.index[df[col] == v]:
                self.setv(df, table, rid, col, best, "CORRECT", conf, "noisy-channel vocabulary",
                          f"{v!r} ({c}x) is {sim:.0f}% similar to {best!r} ({cnt[best]}x)", "TYPO")

    def place_key(self, name, country):
        """Strip a trailing ', <country>' even when that country is misspelled."""
        n = (name or "").strip()
        if "," in n:
            head, tail = n.rsplit(",", 1)
            tail = re.sub(r"\s+", " ", tail).strip().lower()
            if any(OSA.normalized_similarity(tail, c.lower()) >= 0.8 for c in self.countries):
                n = head
        return re.sub(r"\s+", " ", n).lower()

    def dedupe(self, df, table, idc, key_fn, sim_fn, threshold, guard=None):
        blocks = defaultdict(set)
        for rid, r in df.iterrows():
            for k in key_fn(r):
                if k:
                    blocks[k].add(rid)
        parent = {r: r for r in df.index}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        evidence = {}
        pairs = sorted({p for ids in blocks.values() if len(ids) <= 40 for p in itertools.combinations(sorted(ids), 2)})
        for a, b in pairs:
            s, ev = sim_fn(df.loc[a], df.loc[b])
            if s >= threshold and find(a) != find(b) and not (guard and guard(a, b, find)):
                parent[find(a)] = find(b)
                evidence[(a, b)] = (s, ev)
        groups = defaultdict(list)
        for r in df.index:
            groups[find(r)].append(r)
        return [sorted(g) for g in groups.values()], evidence

    def merge(self, df, table, idc, clusters, evidence, cols, choose=None):
        rows = []
        orig = self.orig[table].set_index(idc)
        for g in clusters:
            cid = g[0]                                     # canonical = one of its own source ids
            for r in g:
                self.canon[table][r] = cid
            row = {idc: cid, "source_records": ",".join(g)}
            if len(g) > 1:
                ev = [f"{a}~{b} {s:.2f} ({'; '.join(e)})" for (a, b), (s, e) in evidence.items() if a in g and b in g]
                iid = self.issue(table, "DUPLICATE", g, f"{len(g)} records describe one {table[:-1]}")
                self.act(iid, "MERGE", table, g, None, None, cid, 0.95, "blocking + similarity",
                         " | ".join(ev[:2]))
            for c in cols:
                vals = [(r, df.at[r, c]) for r in g if df.at[r, c] is not None]
                if choose and c in choose:
                    row[c] = choose[c](vals, g)
                elif not vals:
                    row[c] = None
                else:
                    cnt = Counter(str(v) for _, v in vals)
                    (best, n), *rest = cnt.most_common()
                    row[c] = next(v for _, v in vals if str(v) == best)
                    if rest:
                        self._resolve_log(table, g, c, cnt, row[c], n, rest, len(vals))
                src = [r for r, v in vals if str(v) == str(row[c])]
                changed = [r for r in src if r in orig.index and str(orig.at[r, c]) != str(row[c])]
                self.lineage.append({"table": table, "entity_id": cid, "column": c, "value": row[c],
                                     "from_records": ",".join(src),
                                     "status": "derived" if not src and row[c] is not None else
                                               "repaired" if changed or len(set(map(str, (v for _, v in vals)))) > 1 else
                                               "original"})
            rows.append(row)
        return pd.DataFrame(rows).set_index(idc, drop=False)

    def _resolve_log(self, table, g, c, cnt, chosen, n, rest, nvals):
        if True:
            tie = rest[0][1] == n
            iid = self.issue(table, "CONTRADICTION", g, f"{c}: {dict(cnt)}", c)
            self.act(iid, "RESOLVE", table, g, c, dict(cnt), None if tie else chosen,
                     0.5 if tie else n / nvals, "majority over duplicates",
                     "duplicates tie; kept first, needs a human" if tie else f"{n}/{nvals} copies agree")

    def repair_places(self):
        L = self.L
        self.vocab_fix(L, "places", "country")

        self.countries = sorted({c for c in L["country"] if c})

        def keys(r):
            k = self.place_key(r["name"], r["country"])
            return [k, "S" + jellyfish.soundex(k)]          # no country in the key: it may be missing

        def sim(a, b):
            ka, kb_ = self.place_key(a["name"], a["country"]), self.place_key(b["name"], b["country"])
            if a["country"] and b["country"] and a["country"] != b["country"]:
                return 0.0, ["different country"]            # namesakes: Cambridge UK vs US
            if re.findall(r"\d+", ka) != re.findall(r"\d+", kb_):
                return 0.0, ["different number"]             # 13th vs 14th arrondissement
            ta, tb_ = ka.split(), kb_.split()
            if len(ta) > 1 and len(tb_) > 1 and any(max(OSA.normalized_similarity(x, y) for y in tb_) < 0.6 for x in ta):
                return 0.0, ["a word differs entirely"]      # 7th arrondissement of Paris vs Lyon
            strip = lambda x: re.sub(r"\s+c+ity$", "", x)
            s = max(OSA.normalized_similarity(ka, kb_), OSA.normalized_similarity(strip(ka), strip(kb_)))
            return s, [f"name {s:.2f}"]
        def guard(a, b, find):                               # never chain two countries together
            ca = {L.at[x, "country"] for x in L.index if find(x) == find(a)} - {None}
            cb = {L.at[x, "country"] for x in L.index if find(x) == find(b)} - {None}
            return bool(ca and cb and not (ca & cb))
        cl, ev = self.dedupe(L, "places", "place_id", keys, sim, 0.80, guard)

        def best_name(vals, g):
            names = [v for _, v in vals]
            clean = [n for n in names if n != n.upper() and "," not in n and not n.lower().endswith(" city")] or names
            return Counter(clean).most_common(1)[0][0] if clean else None
        self.L = self.merge(L, "places", "place_id", cl, ev, ["name", "country"], {"name": best_name})

    # ------------------------------------------------------------ 3. events
    def repair_events(self):
        E = self.E
        name_year = lambda r: (re.search(r"(\d{4})\s*$", r["name"] or "") or [None, None])[1]
        fd = defaultdict(Counter)                         # prize -> ceremony place
        for _, r in E.iterrows():
            fd[r["prize"]][r["place_id"]] += 1
        for rid, r in E.iterrows():
            best, n = fd[r["prize"]].most_common(1)[0]
            share = n / sum(fd[r["prize"]].values())
            if r["place_id"] != best and share >= 0.7:
                self.setv(E, "events", rid, "place_id", best, "CORRECT", share, "dependency prize->place",
                          f"{share:.0%} of '{r['prize']}' events are held at {best}", "CONTRADICTION")
        # identity = prize + year written in the label (the year column is the corruptible field)
        cl, ev = self.dedupe(E, "events", "event_id",
                             lambda r: [f"{r['prize']}|{name_year(r) or r['year']}"],
                             lambda a, b: (1.0, ["same prize, same year in label"]), 0.99)
        links = defaultdict(Counter)
        for _, r in self.R[self.R.rel == "received"].iterrows():
            if r.year:
                links[r.dst][r.year] += 1

        def vote_year(vals, g):
            votes = Counter()
            for x in g:
                if name_year(E.loc[x]):
                    votes[name_year(E.loc[x])] += 1
                if E.at[x, "year"]:
                    votes[E.at[x, "year"]] += 1
                votes.update(links[x])
            best, n = votes.most_common(1)[0]
            wrong = [x for x in g if E.at[x, "year"] and E.at[x, "year"] != best]
            for x in wrong:
                iid = self.issue("events", "CONTRADICTION", [x], f"year {E.at[x, 'year']} vs evidence {dict(votes)}", "year")
                self.act(iid, "CORRECT", "events", [x], "year", E.at[x, "year"], best,
                         min(0.99, n / sum(votes.values()) + 0.3), "vote: label + copies + award links",
                         f"label, copies and {sum(sum(links[y].values()) for y in g)} award link(s) give {best}")
            return best
        canon_name = lambda vals, g: f"{E.at[g[0], 'prize']} {vote_year(None, g)}"
        self.E = self.merge(E, "events", "event_id", cl, ev, ["name", "prize", "year", "place_id"],
                            {"name": canon_name, "year": lambda v, g: vote_year(v, g)})
        # year column gets logged twice (name + year) - drop the duplicate log entries
        seen, keep = set(), []
        for a in self.actions:
            k = (a["kind"], a["records"], a["col"], str(a["before"]), str(a["after"]))
            if k not in seen:
                seen.add(k)
                keep.append(a)
        self.actions = keep

    # ------------------------------------------------------------ 4. people
    @staticmethod
    def norm_name(n):
        n = (n or "").strip()
        if "," in n:
            last, first = [x.strip() for x in n.split(",", 1)]
            n = f"{first} {last}"
        parts = n.lower().split()
        if parts:
            parts[0] = NICK_REV.get(parts[0], parts[0])
        return " ".join(parts)

    def award_years(self):
        ey = self.E["year"].to_dict()
        out = defaultdict(list)
        for _, r in self.R[self.R.rel == "received"].iterrows():
            e = self.canon["events"].get(r.dst)
            if e and ey.get(e):
                out[r.src].append(int(ey[e]))
        return out

    def repair_people(self):
        P = self.P
        for rid in P.index:                                   # remap place refs to canonical places
            for c in ("birth_place_id", "death_place_id"):
                v = P.at[rid, c]
                if v and v not in self.canon["places"]:
                    self.setv(P, "people", rid, c, None, "FLAG", 0.0, "referential integrity",
                              f"{c}={v} points to no known place", "BROKEN_REF")
                elif v:
                    P.at[rid, c] = self.canon["places"][v]
        aw = self.award_years()
        for rid, r in P.iterrows():                           # died before born -> swapped dates
            b, d = year(r["birth_date"]), year(r["death_date"])
            if b and d and d < b:
                a = aw.get(rid, [])
                ok = all(d + 15 <= x <= b + 1 for x in a) if a else True
                conf = 0.95 if ok and a else 0.7
                iid = self.issue("people", "CONTRADICTION", [rid], f"death {d} before birth {b}", "birth_date")
                if self.act(iid, "CORRECT", "people", [rid], "birth_date/death_date", f"{r['birth_date']}/{r['death_date']}",
                            f"{r['death_date']}/{r['birth_date']}", conf, "temporal order + award dates",
                            "swapping restores birth < award(s) < death" if a else "swapping restores birth < death"):
                    P.at[rid, "birth_date"], P.at[rid, "death_date"] = r["death_date"], r["birth_date"]

        # name tokens: noisy channel per token (first names repeat across people)
        tok = Counter(t for n in P["name"] if n for t in self.norm_name(n).split() if len(t) > 2)
        common = [t for t, c in tok.items() if c >= 3]
        for rid, n in P["name"].items():
            if not n:
                continue
            new = n
            for t in self.norm_name(n).split():
                if len(t) <= 2 or tok[t] > 1 or t in NICK_REV:
                    continue
                # only typing errors that real name variants never are: adjacent swap, doubled letter
                def typing_error(c):
                    swap = len(c) == len(t) and sorted(c) == sorted(t) and OSA.distance(t, c) == 1
                    dbl = any(t[:i] + t[i + 1:] == c for i in range(1, len(t)) if t[i] == t[i - 1])
                    return swap or dbl
                best = next((c for c in sorted(common, key=lambda c: -tok[c]) if typing_error(c)), None)
                if best:
                    new = re.sub(re.escape(t), best.upper() if t.upper() in n else best.title(), new, flags=re.I)
            if new != n:
                self.setv(P, "people", rid, "name", new, "CORRECT", 0.85, "noisy-channel name token",
                          f"rare token in {n!r} is one edit from a common name", "TYPO")
        # gender implied by family links
        for _, r in self.R[self.R.rel.isin(["father", "mother"])].iterrows():
            g = {"father": "male", "mother": "female"}[r.rel]
            if r.dst in P.index and P.at[r.dst, "gender"] is None:
                self.setv(P, "people", r.dst, "gender", g, "FILL", 0.95, "implied by family link",
                          f"{r.src} lists {r.dst} as {r.rel}", "MISSING")
            elif r.dst in P.index and P.at[r.dst, "gender"] not in (None, g):
                iid = self.issue("people", "CONTRADICTION", [r.dst], f"gender {P.at[r.dst, 'gender']} but listed as {r.rel}", "gender")
                self.act(iid, "FLAG", "people", [r.dst], "gender", P.at[r.dst, "gender"], None, 0.0,
                         "family link vs gender", f"listed as {r.rel} of {r.src}")

        won = defaultdict(set)
        for _, r in self.R[self.R.rel == "received"].iterrows():
            if r.dst in self.canon["events"]:
                won[r.src].add(self.canon["events"][r.dst])

        def keys(r):
            nn = self.norm_name(r["name"]).split()
            return ["L" + jellyfish.soundex(nn[-1]) if nn else None,
                    "B" + str(r["birth_date"]) if r["birth_date"] else None,
                    "M" + str(r["birth_date"])[4:] + (nn[-1][:3] if nn else "") if r["birth_date"] else None]

        won = defaultdict(set)                               # person record -> canonical award events
        for _, r in self.R[self.R.rel == "received"].iterrows():
            if r.dst in self.canon["events"]:
                won[r.src].add(self.canon["events"][r.dst])

        def sim(a, b):
            na, nb = self.norm_name(a["name"]), self.norm_name(b["name"])
            shared = won[a.name] & won[b.name]
            s = fuzz.token_sort_ratio(na, nb) / 100
            fa, fb = na.split(), nb.split()
            if fa and fb and fa[-1] == fb[-1] and fa[0][0] == fb[0][0] and (len(fa[0].rstrip(".")) == 1 or len(fb[0].rstrip(".")) == 1):
                s = max(s, 0.9)
            ev = [f"name {s:.2f}"]
            score = 0.6 * s
            if a["birth_date"] and b["birth_date"]:
                same = a["birth_date"] == b["birth_date"]
                md = a["birth_date"][4:] == b["birth_date"][4:]   # same day+month, year differs: 1-in-365 coincidence
                score += 0.3 if same else 0.15 if md else -0.3
                ev.append("same birth date" if same else "same day/month, year conflicts" if md else "different birth date")
            if a["gender"] and b["gender"] and a["gender"] != b["gender"]:
                score -= 0.5
            if a["birth_place_id"] and b["birth_place_id"]:
                score += 0.1 if a["birth_place_id"] == b["birth_place_id"] else -0.1
            return score, ev
        ages = sorted(y - year(P.at[r, "birth_date"]) for r in P.index if P.at[r, "birth_date"]
                      for y in aw.get(r, []) if 17 <= y - year(P.at[r, "birth_date"]) <= 100)
        self.typical_age = ages[len(ages) // 2] if ages else 55
        cl, ev = self.dedupe(P, "people", "person_id", keys, sim, 0.62)

        def pick_birth(vals, g):                              # duplicates disagree: award ages decide
            if not vals:
                return None
            cnt = Counter(v for _, v in vals)
            if len(cnt) == 1:
                return vals[0][1]
            awards = [y for r in g for y in aw.get(r, [])]
            deaths = [year(P.at[r, "death_date"]) for r in g if P.at[r, "death_date"]]
            plaus = lambda d: (all(17 <= y - year(d) <= 100 for y in awards) if awards else True) and                               all(15 <= x - year(d) <= 110 for x in deaths)
            good = [v for v in cnt if plaus(v)]
            # tie-break: the age at award closest to the typical laureate age (learned from the data)
            dev = lambda v: min((abs(y - year(v) - self.typical_age) for y in awards), default=0)
            choice = max(good or cnt, key=lambda v: (cnt[v], -dev(v)))
            iid = self.issue("people", "CONTRADICTION", g, f"birth_date: {dict(cnt)}", "birth_date")
            self.act(iid, "RESOLVE", "people", g, "birth_date", dict(cnt), choice,
                     0.95 if len(good) == 1 else 0.6, "duplicates + age at award",
                     f"only {choice} gives a plausible age at award {awards}" if len(good) == 1 else "copies disagree")
            return choice

        def pick_name(vals, g):
            names = [v for _, v in vals]
            good = [n for n in names if n != n.upper() and "," not in n and not re.match(r"^\w\. ", n)
                    and n.split()[0].lower() not in NICK_REV] or names
            return Counter(good).most_common(1)[0][0] if good else None
        self.P = self.merge(P, "people", "person_id", cl, ev,
                            ["name", "birth_date", "death_date", "gender", "birth_place_id", "death_place_id"],
                            {"birth_date": pick_birth, "name": pick_name})
        aw = self.award_years()                               # single-record age check (no evidence to fix)
        for pid, r in self.P.iterrows():
            b, dd = year(r["birth_date"]), year(r["death_date"])
            if b and dd and not 15 <= dd - b <= 110:
                iid = self.issue("people", "CONTRADICTION", [pid], f"lifespan {dd - b} years", "birth_date")
                self.act(iid, "FLAG", "people", [pid], "birth_date", r["birth_date"], None, 0.0,
                         "human lifespan", f"born {b}, died {dd}: a {dd - b}-year life is impossible")
                continue
            for y in {y for rr in r["source_records"].split(",") for y in aw.get(rr, [])}:
                if b and not 17 <= y - b <= 100:
                    iid = self.issue("people", "CONTRADICTION", [pid], f"born {b} but award in {y}", "birth_date")
                    self.act(iid, "FLAG", "people", [pid], "birth_date", r["birth_date"], None, 0.0,
                             "age at award", f"age {y - b} at award is impossible; true birth year unknown")
                    break

    # ------------------------------------------------------------ 5. relationships
    def repair_relationships(self):
        out, ey = [], self.E["year"].to_dict()
        gender = self.P["gender"].to_dict()
        for i, r in self.R.iterrows():
            s = self.canon["people"].get(r.src)
            dt = "events" if r.rel == "received" else "people"
            d = self.canon[dt].get(r.dst)
            if not s or not d:
                iid = self.issue("relationships", "BROKEN_REF", [r.src, r.dst], f"{r.rel} -> unknown id")
                self.act(iid, "FLAG", "relationships", [f"row{i}"], "dst", r.dst, None, 0.0, "referential integrity",
                         "target does not exist in the knowledge base; cannot infer it")
                continue
            yv = r.year
            if r.rel == "received" and ey.get(d) and yv != ey[d]:
                iid = self.issue("relationships", "CONTRADICTION", [f"row{i}"], f"award year {yv} vs event {ey[d]}", "year")
                self.act(iid, "CORRECT", "relationships", [f"row{i}"], "year", yv, ey[d], 0.95,
                         "event is authoritative", f"the award event {d} took place in {ey[d]}")
                yv = ey[d]
            out.append({"src": s, "rel": r.rel, "dst": d, "year": yv, "from_row": i})
        R = pd.DataFrame(out).drop_duplicates(["src", "rel", "dst"])
        # inverse completeness, learned from the data itself
        have = set(zip(R.src, R.rel, R.dst))
        inv = lambda rel, s: {"spouse": "spouse", "sibling": "sibling",
                              "father": "child", "mother": "child"}.get(rel) or \
            ({"male": "father", "female": "mother"}.get(gender.get(s)) if rel == "child" else None)
        fam = [(s, rel, d) for s, rel, d in have if rel != "received"]
        if fam:
            closed = sum((d, inv(rel, d), s) in have for s, rel, d in fam if inv(rel, d)) / len(fam)
            if closed >= 0.6:
                add = []
                for s, rel, d in fam:
                    ir = inv(rel, d)
                    if ir and (d, ir, s) not in have:
                        iid = self.issue("relationships", "MISSING", [s, d], f"{s} {rel} {d} has no inverse")
                        self.act(iid, "FILL", "relationships", [s, d], ir, None, f"{d} {ir} {s}", closed,
                                 "learned inverse completeness", f"{closed:.0%} of family links have their inverse")
                        add.append({"src": d, "rel": ir, "dst": s, "year": None, "from_row": None})
                        have.add((d, ir, s))
                R = pd.concat([R, pd.DataFrame(add)], ignore_index=True)
        births = {k: year(v) for k, v in self.P["birth_date"].items()}
        for _, r in R[R.rel.isin(["father", "mother"])].iterrows():   # parent must be older
            bc, bp = births.get(r.src), births.get(r.dst)
            if bc and bp and bp > bc - 12:
                iid = self.issue("relationships", "CONTRADICTION", [r.src, r.dst], f"parent born {bp}, child {bc}")
                self.act(iid, "FLAG", "relationships", [r.src, r.dst], None, None, None, 0.0, "parent age",
                         "parent is not at least 12 years older than the child")
        self.Rout = R.reset_index(drop=True)

    def output(self):
        """The repaired knowledge base as 4 tables (each entity lists its source records)."""
        return {"people": self.P.reset_index(drop=True), "places": self.L.reset_index(drop=True),
                "events": self.E.reset_index(drop=True), "relationships": self.Rout.reset_index(drop=True)}

    def run(self):
        self.placeholders()
        self.repair_places()
        self.repair_events()
        self.repair_people()
        self.repair_relationships()
        return self
