"""JSON API for the Knowledge Repair frontend (kb-ui/).

    python -m kb.api            # http://127.0.0.1:8766  (serves kb-ui/dist when built)

Runs are kept in memory: the repairer, the input KB and, lazily, two turbovec
stores (corrupted vs repaired) used by /ask. Answers are read from the retrieved
record's fields, so nothing is generated.
"""
import io
import json
import time
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .__main__ import TABLES
from .ask_demo import ask as retrieve, person_docs
from .dq import dq
from .llm import available as llm_available, grounded_sentence
from .repair import KBRepairer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "sample_input")
REQUIRED = {"people": ["person_id", "name", "birth_date", "death_date", "gender", "birth_place_id", "death_place_id"],
            "places": ["place_id", "name", "country"], "events": ["event_id", "name", "prize", "year", "place_id"],
            "relationships": ["src", "rel", "dst", "year"]}
IDC = {"people": "person_id", "places": "place_id", "events": "event_id"}
RUNS: dict = {}
LOCK = threading.Lock()

app = FastAPI(title="Knowledge Repair API")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5179", "http://localhost:5179"],
                   allow_methods=["*"], allow_headers=["*"])


def _clean(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v


def _rows(df):
    return [{k: _clean(v) for k, v in r.items()} for r in df.astype(object).to_dict("records")]


def _frame(raw):
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    return df.where(df != "", None)


def _start(kb, label):
    rep = KBRepairer(kb).run()
    rid = uuid.uuid4().hex[:8]
    RUNS[rid] = {"kb": kb, "rep": rep, "fixed": rep.output(), "label": label, "stores": {},
                 "orig": {t: df.copy() for t, df in kb.items()}, "sample": label.startswith("Sample")}
    threading.Thread(target=_stores, args=(RUNS[rid],), daemon=True).start()   # warm /ask in the background
    return rid


def _run(rid):
    if rid not in RUNS:
        raise HTTPException(404, "This repair run no longer exists. Start a new repair.")
    return RUNS[rid]


@app.post("/api/runs/sample")
def run_sample():
    kb = {t: _frame(open(os.path.join(SAMPLE, f"{t}.csv"), "rb").read()) for t in TABLES}
    return summary(_start(kb, "Sample: corrupted Nobel laureate KB (Wikidata)"))


@app.post("/api/runs")
async def run_upload(people: UploadFile = File(...), places: UploadFile = File(...),
                     events: UploadFile = File(...), relationships: UploadFile = File(...)):
    kb = {t: _frame(await f.read()) for t, f in zip(TABLES, (people, places, events, relationships))}
    missing = {t: sorted(set(c) - set(kb[t].columns)) for t, c in REQUIRED.items()}
    missing = {t: m for t, m in missing.items() if m}
    if missing:
        raise HTTPException(400, "; ".join(f"{t}.csv is missing {', '.join(m)}" for t, m in missing.items()))
    return summary(_start(kb, "Uploaded knowledge base"))


@app.get("/api/runs/{rid}")
def summary(rid: str):
    r = _run(rid)
    rep, kb, fixed = r["rep"], r["kb"], r["fixed"]
    if "dq" not in r:
        r["dq"] = {"before": dq(kb), "after": dq(fixed)}
    issues = Counter((i["table"], i["kind"]) for i in rep.issues)
    status = Counter(a["status"] for a in rep.actions)
    merged = [e for e in _rows(fixed["people"]) if "," in (e.get("source_records") or "")]
    return {"id": rid, "label": r["label"], "quality": r["dq"],
            "records": {t: [len(kb[t]), len(fixed[t])] for t in TABLES},
            "issues": [{"table": t, "kind": k, "count": n} for (t, k), n in sorted(issues.items())],
            "actions": {"applied": status["applied"], "needs_review": status["needs_review"],
                        "flagged_only": status["flagged_only"], "total": len(rep.actions)},
            "examples": [e["person_id"] for e in merged[:6]]}


@app.get("/api/runs/{rid}/ask/ready")
def ask_ready(rid: str):
    return {"ready": len(_run(rid)["stores"]) == 2}


@app.get("/api/runs/{rid}/actions")
def actions(rid: str, table: str = "", kind: str = "", status: str = "", q: str = "", limit: int = 200, offset: int = 0):
    acts = _run(rid)["rep"].actions
    ql = q.lower()
    rows = [a for a in acts if (not table or a["table"] == table) and (not kind or a["kind"] == kind)
            and (not status or a["status"] == status)
            and (not ql or ql in str(a).lower())]
    out = []
    for a in rows[offset:offset + limit]:
        b = a["before"]
        out.append({**a, "before": b if not isinstance(b, dict) else ", ".join(f"{k} ×{v}" for k, v in b.items())})
    return {"total": len(rows), "kinds": sorted({a["kind"] for a in acts}), "rows": out}


@app.get("/api/runs/{rid}/trace/{eid}")
def trace(rid: str, eid: str):
    r = _run(rid)
    rep, kb, fixed = r["rep"], r["kb"], r["fixed"]
    hit = next(((t, rep.canon[t][eid]) for t in IDC if eid in rep.canon[t]), None)
    if not hit:
        raise HTTPException(404, f"No person, place or event with ID {eid}.")
    t, cid = hit
    ent = fixed[t].set_index(IDC[t]).loc[cid]
    srcs = (ent.get("source_records") or cid).split(",")
    orig = kb[t].set_index(IDC[t])
    cols = REQUIRED[t][1:]
    names = {**dict(zip(fixed["people"].person_id, fixed["people"].name)),
             **dict(zip(fixed["places"].place_id, fixed["places"].name)),
             **dict(zip(fixed["events"].event_id, fixed["events"].name))}
    rels = fixed["relationships"]
    rels = rels[(rels.src == cid) | (rels.dst == cid)]
    return {
        "table": t, "entity_id": cid, "columns": cols,
        "repaired": {c: _clean(ent.get(c)) for c in cols},
        "originals": [{"record_id": s, **{c: _clean(orig.at[s, c]) if s in orig.index else None for c in cols}} for s in srcs],
        "lineage": [l for l in rep.lineage if l["table"] == t and l["entity_id"] == cid],
        "relationships": [{"rel": x.rel, "src": x.src, "src_name": names.get(x.src, x.src), "dst": x.dst,
                           "dst_name": names.get(x.dst, x.dst), "year": _clean(x.year)} for x in rels.itertuples()],
        "actions": [{**a, "before": str(a["before"]) if isinstance(a["before"], dict) else a["before"]}
                    for a in rep.actions if any(x in srcs or x == cid for x in a["records"].split(","))],
        "place_names": {v: names.get(v, v) for row in [{c: ent.get(c) for c in cols}] +
                        [{c: (orig.at[x, c] if x in orig.index else None) for c in cols} for x in srcs]
                        for c, v in row.items() if c.endswith("place_id") and v},
    }


# ----------------------------------------------------------------- ask
class Ask(BaseModel):
    question: str


PRIZES = {"physics": "Nobel Prize in Physics", "chemistry": "Nobel Prize in Chemistry",
          "medicine": "Nobel Prize in Physiology or Medicine", "physiology": "Nobel Prize in Physiology or Medicine",
          "literature": "Nobel Prize in Literature", "peace": "Nobel Peace Prize",
          "economic": "Nobel Memorial Prize in Economic Sciences", "economics": "Nobel Memorial Prize in Economic Sciences"}


def intent(q):
    ql = q.lower()
    if "country" in ql:
        return "birth_country", "country of birth"
    if re.search(r"\bwhere\b", ql) and "born" in ql:
        return "birth_place", "birthplace"
    if "die" in ql or "death" in ql:
        return "death_year", "year of death"
    if "born" in ql or "birth" in ql:
        return "birth_year", "year of birth"
    if any(w in ql for w in ("award", "prize", "nobel", "win", "won", "receive")):
        prize = next((v for k, v in PRIZES.items() if k in ql), None)
        return "award:" + (prize or "*"), "award year"
    return "summary", "record"


def _build_stores(kb, fixed):
    tmp = tempfile.mkdtemp(prefix="kbask_")
    from .ask_demo import store
    out = {}
    for side, data in (("before", kb), ("after", fixed)):
        docs = person_docs(data)
        d = os.path.join(tmp, side)
        st = store(docs, d)
        size = lambda f: os.path.getsize(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else 0
        meta = json.load(open(os.path.join(d, "index_meta.json")))
        out[side] = (st, {x["id"]: x for x in docs},
                     {"engine": meta["engine"], "vectors": meta["count"],
                      "index_kb": round(size("turbovec.idx") / 1024, 1),
                      "float32_kb": round(size("vectors.npy") / 1024, 1)})
    return out


def _stores(r):
    with LOCK:
        if not r["stores"]:
            r["stores"].update(_build_stores(r["kb"], r["fixed"]))
    return r["stores"]


def _refresh_after_edit(r):
    """Edits show up at once: each indexed record's fields are swapped for the edited ones.
    The vectors are rebuilt in the background (new or renamed records become searchable)."""
    if r["stores"]:
        for side, data in (("before", r["kb"]), ("after", r["fixed"])):
            st, by, info = r["stores"][side]
            by.clear()
            by.update({x["id"]: x for x in person_docs(data)})
    r["index_version"] = r.get("index_version", 0) + 1
    version = r["index_version"]

    def rebuild():
        fresh = _build_stores(r["kb"], r["fixed"])
        if r.get("index_version") == version:          # a newer edit supersedes this rebuild
            r["stores"].update(fresh)
            r["index_stale"] = False
    r["index_stale"] = True
    threading.Thread(target=rebuild, daemon=True).start()


@app.post("/api/runs/{rid}/ask")
def ask(rid: str, body: Ask):
    r = _run(rid)
    q = body.question.strip()
    if len(q) < 4:
        raise HTTPException(400, "Type a question, for example: When was Alan J. Heeger born?")
    field, label = intent(q)
    out = {"question": q, "asked_for": label}
    for side, (st, by, info) in _stores(r).items():
        t0 = time.perf_counter()
        cand = [c for c in retrieve(st, q, k=25) if c in by]
        ms = round((time.perf_counter() - t0) * 1000, 1)
        vec = dict(st.vector_search(q, 25))
        qt = set(re.findall(r"[\w'-]+", q.lower()))

        def named(c):                                  # share of the record's name words found in the question
            words = re.findall(r"[\w'-]+", by[c]["fields"]["name"].lower().replace(",", " "))
            words = [w for w in words if len(w.rstrip(".")) > 1]
            hit = sum(any(fuzz.ratio(w, t) >= 85 for t in qt) for w in words)
            return hit / len(words) if words else 0
        top = sorted(cand, key=lambda c: -named(c))[:3]   # stable: ties keep retrieval order
        retrieval = {**info, "query_ms": ms,
                     "hits": [{"id": c, "name": by[c]["fields"]["name"], "rank": cand.index(c) + 1,
                               "vector_score": round(vec[c], 3) if c in vec else None} for c in cand[:5]]}
        if not top:
            out[side] = {"answer": None, "record": None, "alternatives": []}
            continue
        f = by[top[0]]["fields"]
        if named(top[0]) < 0.5:                        # nobody with that name: say so, don't answer for someone else
            out[side] = {"answer": None, "record_id": None, "record": None, "name": None, "conflicting": [],
                         "not_found": True, "retrieval": retrieval,
                         "alternatives": [{"id": c, "name": by[c]["fields"]["name"]} for c in top]}
            continue
        if field.startswith("award:"):
            p = field.split(":", 1)[1]
            ans = "; ".join(f"{k}: {v}" for k, v in f["awards"].items()) if p == "*" else f["awards"].get(p)
        elif field == "summary":
            ans = by[top[0]]["text"]
        else:
            ans = f.get(field) or None
        # same-name records the retriever also surfaced: conflicting answers a user would see
        conflict = sorted({by[c]["fields"].get(field) for c in top[1:]
                           if field in by[c]["fields"] and by[c]["fields"]["name"].lower().replace(",", "")[:6]
                           == f["name"].lower().replace(",", "")[:6] and by[c]["fields"].get(field)} - {ans})
        out[side] = {"answer": ans, "record_id": top[0], "record": by[top[0]]["text"],
                     "name": f["name"], "conflicting": conflict, "not_found": False, "retrieval": retrieval,
                     "alternatives": [{"id": c, "name": by[c]["fields"]["name"]} for c in top[1:]]}
    r.setdefault("asked", {})[q] = out
    return out


class Explain(BaseModel):
    question: str
    side: str


@app.post("/api/runs/{rid}/explain")
def explain(rid: str, body: Explain):
    """One grounded sentence for one side of the last /ask, from an OpenRouter free model."""
    r = _run(rid)
    res = r.get("asked", {}).get(body.question)
    if not res or body.side not in ("before", "after"):
        raise HTTPException(400, "Ask the question first.")
    side = res[body.side]
    if side.get("not_found"):
        return {"sentence": None, "model": None, "status": "rejected", "note": "No matching person to describe."}
    return grounded_sentence(body.question, side.get("record"), side.get("answer"))


@app.get("/api/llm")
def llm_status():
    return {"available": llm_available()}


# ----------------------------------------------------------------- live editing
class Edit(BaseModel):
    changes: dict


def _rerun(r):
    r["rep"] = KBRepairer(r["kb"]).run()
    r["fixed"] = r["rep"].output()
    r.pop("dq", None)
    r.pop("asked", None)
    _refresh_after_edit(r)


def _focus(rid, r, table, record_id, prev_actions):
    """What the repair now says about the edited record."""
    rep = r["rep"]
    cid = rep.canon.get(table, {}).get(record_id, record_id)
    srcs = set((r["fixed"][table].set_index(IDC[table]).loc[cid].get("source_records") or cid).split(","))
    mine = [a for a in rep.actions if srcs & set(a["records"].split(",")) or cid in a["records"].split(",")]
    key = lambda a: (a["kind"], a["records"], str(a["col"]), str(a["before"]), str(a["after"]))
    old = {key(a) for a in prev_actions}
    return {"summary": summary(rid), "trace": trace(rid, record_id),
            "actions": [{**a, "before": str(a["before"]) if isinstance(a["before"], dict) else a["before"],
                         "new": key(a) not in old} for a in mine]}


@app.get("/api/runs/{rid}/records")
def records(rid: str, q: str = "", limit: int = 30):
    r = _run(rid)
    P = r["kb"]["people"]
    ql = q.lower().strip()
    rows = P[P.name.fillna("").str.lower().str.contains(re.escape(ql))] if ql else P
    places = dict(zip(r["kb"]["places"].place_id, r["kb"]["places"].name))
    return {"total": len(rows),
            "places": [{"id": k, "name": v} for k, v in sorted(places.items(), key=lambda x: str(x[1]))],
            "rows": _rows(rows.head(limit))}


@app.patch("/api/runs/{rid}/records/{table}/{record_id}")
def edit(rid: str, table: str, record_id: str, body: Edit):
    from .live import edit_record
    r = _run(rid)
    prev = list(r["rep"].actions)
    try:
        before = edit_record(r["kb"], table, record_id, body.changes)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e).strip("'"))
    t0 = time.perf_counter()
    _rerun(r)
    out = _focus(rid, r, table, record_id, prev)
    out.update({"edited": {"table": table, "record_id": record_id, "before": before, "after": body.changes},
                "repair_ms": round((time.perf_counter() - t0) * 1000)})
    return out


@app.post("/api/runs/{rid}/records/people/{record_id}/duplicate")
def duplicate(rid: str, record_id: str):
    from .live import add_duplicate
    r = _run(rid)
    prev = list(r["rep"].actions)
    try:
        new_id, name = add_duplicate(r["kb"], record_id, seed=len(r["kb"]["people"]))
    except KeyError as e:
        raise HTTPException(400, str(e).strip("'"))
    t0 = time.perf_counter()
    _rerun(r)
    out = _focus(rid, r, "people", new_id, prev)
    out.update({"edited": {"table": "people", "record_id": new_id, "before": {}, "after": {"name": name},
                           "duplicate_of": record_id}, "repair_ms": round((time.perf_counter() - t0) * 1000)})
    return out


@app.post("/api/runs/{rid}/reset")
def reset(rid: str):
    r = _run(rid)
    r["kb"] = {t: df.copy() for t, df in r["orig"].items()}
    _rerun(r)
    return summary(rid)


@app.get("/api/runs/{rid}/index")
def index_status(rid: str):
    r = _run(rid)
    return {"ready": len(r["stores"]) == 2, "stale": bool(r.get("index_stale"))}


# ----------------------------------------------------------------- playback
@app.get("/api/runs/{rid}/timeline")
def timeline(rid: str):
    from .live import stage_timeline
    r = _run(rid)
    if "timeline" not in r or r.get("timeline_version") != r.get("index_version"):
        r["timeline"] = stage_timeline(r["kb"])
        r["timeline_version"] = r.get("index_version")
    corruption = None
    log_path = os.path.join(SAMPLE, "injected_errors.csv")
    if r["sample"] and os.path.exists(log_path):
        from .load import load
        log = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        names = dict(zip(r["orig"]["people"].person_id, r["orig"]["people"].name))
        names.update(zip(r["orig"]["places"].place_id, r["orig"]["places"].name))
        names.update(zip(r["orig"]["events"].event_id, r["orig"]["events"].name))
        table = {"XP": "people", "XL": "places", "XE": "events", "XR": "relationships"}
        if "clean_quality" not in RUNS:
            RUNS["clean_quality"] = dq(load())["DQ"]
        corruption = {"clean_quality": RUNS["clean_quality"],
                      "errors": [{"table": table.get(e["table"], e["table"]), "kind": e["kind"], "id": e["id"],
                                  "name": names.get(e["id"], e["id"]), "col": e["col"] or None,
                                  "old": e["old"] or None, "new": e["new"] or None} for e in log.to_dict("records")]}
    return {"corruption": corruption, "repair": r["timeline"]}


@app.get("/api/runs/{rid}/download/{name}")
def download(rid: str, name: str):
    r = _run(rid)
    files = {"repaired_people.csv": r["fixed"]["people"], "repaired_places.csv": r["fixed"]["places"],
             "repaired_events.csv": r["fixed"]["events"], "repaired_relationships.csv": r["fixed"]["relationships"],
             "lineage.csv": pd.DataFrame(r["rep"].lineage), "actions.csv": pd.DataFrame(r["rep"].actions),
             "issues.csv": pd.DataFrame(r["rep"].issues)}
    if name not in files:
        raise HTTPException(404, "Unknown file")
    path = os.path.join(tempfile.gettempdir(), f"kb_{rid}_{name}")
    files[name].to_csv(path, index=False)
    return FileResponse(path, filename=name, media_type="text/csv")


DIST = os.path.join(ROOT, "kb-ui", "dist")
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8766)
