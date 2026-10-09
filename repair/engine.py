"""Detect + repair with full lineage. Original records are never mutated:
every change is an Action pointing at the record_ids/cells it came from."""
import itertools
import re
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA

from .synth import NICK, PEOPLE_COLS

PLACEHOLDERS = {"", "n/a", "na", "unknown", "none", "null", "-", "?"}
NICK_REV = {v.lower(): k.lower() for k, v in NICK.items()}
AUTO, REVIEW = 0.90, 0.60          # confidence bands (see DESIGN.md §3)


class Repairer:
    def __init__(self, people: pd.DataFrame, rels: pd.DataFrame):
        self.orig = people.copy()
        self.df = people.copy().set_index("record_id", drop=False)
        self.rels = rels.copy()
        self.issues, self.actions = [], []
        self.derived = set()          # (record_id, col) filled by inference, not observed

    # ---------- bookkeeping ----------
    def _issue(self, kind, records, detail, col=None):
        iid = f"I{len(self.issues):05d}"
        self.issues.append({"issue_id": iid, "kind": kind, "col": col,
                            "records": ",".join(records), "detail": detail})
        return iid

    def _act(self, iid, kind, records, col, before, after, conf, method, why):
        status = "applied" if conf >= AUTO else "needs_review" if conf >= REVIEW else "flagged_only"
        self.actions.append({"action_id": f"A{len(self.actions):05d}", "issue_id": iid,
                             "kind": kind, "records": ",".join(records), "col": col,
                             "before": before, "after": after, "confidence": round(conf, 3),
                             "status": status, "method": method, "explanation": why})
        return status != "flagged_only"

    def _set(self, rid, col, new, iid, kind, conf, method, why):
        old = self.df.at[rid, col]
        if self._act(iid, kind, [rid], col, old, new, conf, method, why):
            self.df.at[rid, col] = new

    # ---------- D2: missing / placeholders ----------
    def detect_missing(self):
        for rid, row in self.df.iterrows():
            for col in PEOPLE_COLS:
                v = row[col]
                if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip().lower() in PLACEHOLDERS:
                    iid = self._issue("MISSING", [rid], f"{col} empty/placeholder ({v!r})", col)
                    self._act(iid, "NORMALIZE", [rid], col, v, None, 1.0, "placeholder->null",
                              f"{v!r} is a placeholder, stored as null")
                    self.df.at[rid, col] = None

    # ---------- D3: typos in categorical columns ----------
    def fix_typos(self, col="city", min_ratio=85):
        counts = Counter(v for v in self.df[col] if v)
        frequent = [v for v, c in counts.items() if c >= 5]
        for v, c in counts.items():
            if c >= 5:
                continue
            best = max(frequent, key=lambda f: fuzz.ratio(v.lower(), f.lower()), default=None)
            if not best:
                continue
            sim = fuzz.ratio(v.lower(), best.lower())
            if sim < min_ratio:
                continue
            conf = (sim / 100) * (counts[best] / (counts[best] + c))
            for rid in self.df.index[self.df[col] == v]:
                iid = self._issue("TYPO", [rid], f"{col}={v!r} ~ {best!r}", col)
                self._set(rid, col, best, iid, "CORRECT", conf, "fuzzy-vocab",
                          f"{v!r} seen {c}x, {sim:.0f}% similar to {best!r} seen {counts[best]}x")

    # ---------- D4a: functional dependency city -> country ----------
    def fix_fd(self, lhs="city", rhs="country"):
        groups = self.df[self.df[lhs].notna()].groupby(lhs)
        for city, g in groups:
            vals = Counter(v for v in g[rhs] if v)
            if not vals:
                continue
            best, n = vals.most_common(1)[0]
            share = n / sum(vals.values())
            for rid, v in g[rhs].items():
                if v is None:
                    iid = self._issue("MISSING", [rid], f"{rhs} null, derivable from {lhs}={city}", rhs)
                    self._set(rid, rhs, best, iid, "FILL", share, f"FD {lhs}->{rhs}",
                              f"{share:.0%} of {lhs}={city} rows have {rhs}={best}")
                elif v != best:
                    iid = self._issue("CONTRADICTION", [rid], f"{lhs}={city} but {rhs}={v} (majority {best})", rhs)
                    self._set(rid, rhs, best, iid, "CORRECT", share, f"FD {lhs}->{rhs}",
                              f"{share:.0%} of {lhs}={city} rows say {best}; this row says {v}")

    # ---------- D1: duplicates ----------
    @staticmethod
    def norm_name(n):
        if not n:
            return ""
        n = n.strip()
        if "," in n:
            last, first = [p.strip() for p in n.split(",", 1)]
            n = f"{first} {last}"
        parts = n.lower().split()
        if parts:
            parts[0] = NICK_REV.get(parts[0], parts[0])
        return " ".join(parts)

    def _obs_email(self, rid):
        return None if (rid, "email") in self.derived else self.df.at[rid, "email"]

    def _pair_score(self, a, b):
        na, nb = self.norm_name(a["name"]), self.norm_name(b["name"])
        name = fuzz.token_sort_ratio(na, nb) / 100
        fa, fb = na.split(), nb.split()
        if fa and fb and (len(fa[0].rstrip(".")) == 1 or len(fb[0].rstrip(".")) == 1):
            if fa[-1] == fb[-1] and fa[0][0] == fb[0][0]:
                name = max(name, 0.9)            # "J. Smith" vs "John Smith"
        a, b = a.copy(), b.copy()
        a["email"], b["email"] = self._obs_email(a.name), self._obs_email(b.name)
        same_email = bool(a["email"]) and a["email"] == b["email"]
        if fa and fb and fuzz.ratio(fa[-1], fb[-1]) < 80 and not same_email:
            return -1.0, ["different last name"]      # hard gate: no surname match, no merge
        ev, s = [f"name {name:.2f}"], 0.5 * name
        if a["email"] and b["email"]:
            if a["email"] == b["email"]:
                s += 0.3; ev.append("same email")
            else:
                s -= 0.3; ev.append("different email")
        if a["birth_year"] and b["birth_year"]:
            d = abs(int(a["birth_year"]) - int(b["birth_year"]))
            s += 0.2 if d == 0 else 0.05 if d <= 5 else -0.4
            ev.append(f"year diff {d}")
        if a["city"] and b["city"]:
            s += 0.1 if a["city"] == b["city"] else -0.1
            ev.append("same city" if a["city"] == b["city"] else "different city")
        return s, ev

    def _sibling(self, ra, rb, find, fields=("birth_year", "city")):
        """Collective check (from team NextGen's ER pipeline): a value that >= 2 records of
        one cluster agree on is a consensus; the other cluster deviating from it on an
        identity field means a sibling entity, not another copy."""
        ca = [x for x in self.df.index if find(x) == ra]
        cb = [x for x in self.df.index if find(x) == rb]
        for f in fields:
            va = Counter(str(self.df.at[x, f]) for x in ca if self.df.at[x, f] is not None)
            vb = Counter(str(self.df.at[x, f]) for x in cb if self.df.at[x, f] is not None)
            if not va or not vb:
                continue
            (ta, na), (tb, nb) = va.most_common(1)[0], vb.most_common(1)[0]
            if ta != tb and max(na, nb) >= 2 and not (set(va) & set(vb)):
                self.sibling_vetoes.append((ca, cb, f, ta, tb))
                return True
        return False

    def find_duplicates(self, threshold=0.62):
        self.sibling_vetoes = []
        blocks = defaultdict(set)
        for rid, r in self.df.iterrows():
            nn = self.norm_name(r["name"]).split()
            if nn:
                blocks["L:" + nn[-1]].add(rid)
            if r["email"]:
                blocks["E:" + r["email"]].add(rid)
            if r["birth_year"] and r["city"]:
                blocks[f"YC:{r['birth_year']}:{r['city']}"].add(rid)
        pairs = set()
        for ids in blocks.values():
            if len(ids) <= 50:
                pairs.update(itertools.combinations(sorted(ids), 2))
        parent = {r: r for r in self.df.index}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        self.match_evidence = {}
        for a, b in sorted(pairs):                     # deterministic merge order
            s, ev = self._pair_score(self.df.loc[a], self.df.loc[b])
            if s >= threshold:
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                ea = {self._obs_email(x) for x in self.df.index if find(x) == ra} - {None}
                eb = {self._obs_email(x) for x in self.df.index if find(x) == rb} - {None}
                if ea and eb and not (ea & eb):
                    continue                             # clusters carry different emails: keep apart
                if self._sibling(ra, rb, find):
                    continue
                parent[ra] = rb
                self.match_evidence[(a, b)] = (s, ev)
        clusters = defaultdict(list)
        for r in self.df.index:
            clusters[find(r)].append(r)
        return list(clusters.values())

    # ---------- merge + D4b contradictions inside a cluster ----------
    def merge(self, clusters):
        out, self.record_to_entity, lineage = [], {}, []
        self.first_freq = Counter(self.norm_name(n).split()[0] for n in self.df["name"] if n)
        self.rec_issues = Counter()
        for i in self.issues:
            for r in i["records"].split(","):
                self.rec_issues[r] += 1
        for r, n in self.df["name"].items():   # name variants count as a defect of that record
            if n and (n != n.title() or "," in n or len(n.split()[0].rstrip(".")) == 1
                      or n.split()[0].lower() in NICK_REV):
                self.rec_issues[r] += 1
        for k, recs in enumerate(sorted(clusters, key=lambda c: min(c))):
            eid = f"E{k:05d}"
            for r in recs:
                self.record_to_entity[r] = eid
            row = {"entity_id": eid, "source_records": ",".join(recs)}
            if len(recs) > 1:
                ev = [f"{a}~{b}: {s:.2f} ({'; '.join(e)})" for (a, b), (s, e) in self.match_evidence.items()
                      if a in recs and b in recs]
                iid = self._issue("DUPLICATE", recs, f"{len(recs)} records are one entity")
                self._act(iid, "MERGE", recs, None, None, eid, 0.95, "blocking+fuzzy score",
                          " | ".join(ev[:3]))
            for col in PEOPLE_COLS:
                vals = [(r, self.df.at[r, col]) for r in recs if self.df.at[r, col] is not None]
                if col == "name":
                    vals = [(r, v) for r, v in vals]
                    # prefer the most "complete" clean form: 'First Last', title case, no initials
                    # canonical form: expand nicknames/"Last, First", then pick the spelling
                    # whose first name is most common across the whole dataset (typos are rare)
                    cands = [self.norm_name(v).title() for _, v in vals]
                    cands = [c for c in cands if len(c.split()[0].rstrip(".")) > 1] or cands
                    chosen = max(cands, key=lambda v: (self.first_freq[v.split()[0].lower()],
                                                       cands.count(v)), default=None)
                    row[col] = chosen
                elif not vals:
                    row[col] = None
                else:
                    c = Counter(str(v) for _, v in vals)
                    (best, n), *rest = c.most_common()
                    if rest and rest[0][1] == n:   # tie: trust the record that needed fewer repairs
                        tied = {k for k, m in c.items() if m == n}
                        best = str(min(((r, v) for r, v in vals if str(v) in tied),
                                       key=lambda rv: self.rec_issues[rv[0]])[1])
                    row[col] = next(v for _, v in vals if str(v) == best)
                    if rest:
                        conf = n / len(vals)
                        tie = rest[0][1] == n
                        iid = self._issue("CONTRADICTION", recs, f"{col}: {dict(c)}", col)
                        self._act(iid, "RESOLVE", recs, col, dict(c), None if tie else row[col],
                                  0.5 if tie else conf, "majority vote",
                                  "tie between duplicates: kept first value, needs human" if tie
                                  else f"{n}/{len(vals)} duplicate records agree")
                lineage.append({"entity_id": eid, "col": col, "value": row[col],
                                "from_records": ",".join(r for r, v in vals if str(v) == str(row[col]))})
            out.append(row)
        self.entities = pd.DataFrame(out)
        self.lineage = pd.DataFrame(lineage)

    # ---------- relationships ----------
    def fix_relationships(self):
        r = self.rels.copy()
        rows = []
        years = self.entities.set_index("entity_id").birth_year
        for i, x in r.iterrows():
            s, d = self.record_to_entity.get(x.src), self.record_to_entity.get(x.dst)
            if s is None or d is None:
                iid = self._issue("BROKEN_REF", [str(x.src), str(x.dst)], f"{x.rel} points to unknown record")
                self._act(iid, "FLAG", [str(x.src), str(x.dst)], None, x.dst, None, 0.0, "ref check",
                          "target record does not exist; cannot infer who was meant")
                continue
            ys, yd = years.get(s), years.get(d)
            if x.rel == "parent_of" and pd.notna(ys) and pd.notna(yd) and ys is not None and yd is not None and int(years[s]) >= int(years[d]) - 12:
                iid = self._issue("CONTRADICTION", [s, d], f"parent born {years[s]}, child {years[d]}")
                self._act(iid, "FLAG", [s, d], "birth_year", None, None, 0.0, "age constraint",
                          "parent is not at least 12 years older than child")
            rows.append({"src": s, "rel": x.rel, "dst": d, "from_rel_row": i})
        self.entity_rels = pd.DataFrame(rows).drop_duplicates(["src", "rel", "dst"])

    # ---------- D3b: name typos (noisy channel: frequency prior x edit similarity) ----------
    def fix_name_typos(self, min_target=5, min_ratio=80):
        def toks(n):
            return [t for t in re.split(r"[ ,]+", self.norm_name(n)) if len(t.rstrip(".")) > 1]
        freq = Counter(t for n in self.df["name"] if n for t in toks(n))
        common = [t for t, c in freq.items() if c >= min_target]
        for rid, n in self.df["name"].items():
            if not n:
                continue
            new = n
            for t in set(toks(n)):
                if freq[t] > 2 or t in NICK_REV:
                    continue
                sim = lambda c: OSA.normalized_similarity(t, c) * 100   # swaps count as 1 edit
                best = max(common, key=sim, default=None)
                if not best or sim(best) < min_ratio:
                    continue
                prior = freq[best] / (freq[best] + freq[t])
                conf = sim(best) / 100 * prior
                repl = best.upper() if t.upper() in n else best.title()
                new = re.sub(re.escape(t), repl, new, flags=re.I)
            if new != n:
                iid = self._issue("TYPO", [rid], f"name {n!r}", "name")
                self._set(rid, "name", new, iid, "CORRECT", conf, "noisy-channel name",
                          f"rare spelling in {n!r} replaced by frequent {best!r} ({freq[best]}x)")

    # ---------- D4c: learned cross-field dependencies (email <-> name, year) ----------
    def fix_cross_field(self, min_agree=0.8):
        def parts(rid):
            r = self.df.loc[rid]
            nm = self.norm_name(r["name"]).split()
            if len(nm) < 2 or len(nm[0].rstrip(".")) < 2:
                return None
            return nm[0], nm[-1], r["birth_year"], r["email"]
        rows = {rid: parts(rid) for rid in self.df.index}
        full = [(rid, p) for rid, p in rows.items() if p and p[2] is not None and p[3]]
        doms = Counter(p[3].split("@")[-1] for _, p in full)
        if not full:
            return
        dom = doms.most_common(1)[0][0]

        def mk(pad):
            return lambda f, l, y: f"{f}.{l}{int(float(y)) % 100:0{2 if pad else 1}d}@{dom}"
        template = max((mk(True), mk(False)), key=lambda T: sum(T(*p[:3]) == p[3] for _, p in full))
        agree = sum(template(*p[:3]) == p[3] for _, p in full) / len(full)
        if agree < min_agree:
            return                                    # no stable email template in this data
        years = [int(float(y)) for y in self.df["birth_year"] if y is not None and str(y).replace(".0", "").isdigit()]
        lo, hi = min(years), max(years)
        for rid, p in rows.items():
            if not p:
                continue
            f, l, y, e = p
            m = re.match(rf"{re.escape(f)}\.{re.escape(l)}(\d{{1,2}})@", e or "")
            if e and m:                               # email matches the person -> trust its year digits
                yy = int(m.group(1))
                cands = [c for c in (1900 + yy, 2000 + yy) if lo <= c <= hi]
                if len(cands) != 1:
                    continue
                if y is None:
                    iid = self._issue("MISSING", [rid], "birth_year null, derivable from email", "birth_year")
                    self._set(rid, "birth_year", cands[0], iid, "FILL", agree, "learned email<->year",
                              f"{agree:.0%} of complete rows have email digits = birth year; email {e}")
                elif int(float(y)) % 100 != yy:
                    iid = self._issue("CONTRADICTION", [rid], f"birth_year {y} vs email {e}", "birth_year")
                    self._set(rid, "birth_year", cands[0], iid, "CORRECT", agree * 0.95, "learned email<->year",
                              f"email {e} encodes {cands[0]}; {agree:.0%} of rows follow this pattern")
            elif not e and y is not None:
                iid = self._issue("MISSING", [rid], "email null, derivable from name+year template", "email")
                self.derived.add((rid, "email"))
                self._set(rid, "email", template(f, l, y), iid, "FILL", agree * 0.95, "learned email template",
                          f"{agree:.0%} of complete rows have email = first.lastYY@{dom}")

    def run(self):
        self.detect_missing()
        self.fix_name_typos()
        self.fix_typos("city")
        self.fix_fd("city", "country")
        self.fix_cross_field()
        self.merge(self.find_duplicates())
        self.fix_relationships()
        return self
