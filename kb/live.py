"""Live editing and repair playback for the web app.

edit_record / add_duplicate change the working (corrupted) knowledge base; the API
then re-runs the full repair. stage_timeline replays the repair stage by stage and
measures real quality after each stage.
"""
import random

import pandas as pd

from .dq import dq
from .repair import KBRepairer

EDITABLE = {"people": ["name", "birth_date", "death_date", "gender", "birth_place_id"],
            "places": ["name", "country"], "events": ["name", "year", "place_id"]}
IDC = {"people": "person_id", "places": "place_id", "events": "event_id"}


def edit_record(kb, table, rid, changes):
    if table not in EDITABLE:
        raise ValueError(f"{table} cannot be edited here")
    df = kb[table]
    hit = df.index[df[IDC[table]] == rid]
    if not len(hit):
        raise KeyError(f"No {table[:-1]} with ID {rid}")
    before = {}
    for col, val in changes.items():
        if col not in EDITABLE[table]:
            raise ValueError(f"{col} is not an editable field")
        before[col] = df.at[hit[0], col]
        df.at[hit[0], col] = (val if val not in ("", None) else None)
    return before


def add_duplicate(kb, rid, seed=None):
    """Copy a person under a name variant, as a messy second source would."""
    P = kb["people"]
    row = P[P.person_id == rid]
    if not len(row):
        raise KeyError(f"No person with ID {rid}")
    d = row.iloc[0].to_dict()
    parts = (d["name"] or "").split()
    rng = random.Random(seed)
    if len(parts) > 1:
        d["name"] = rng.choice([f"{parts[0][0]}. {' '.join(parts[1:])}", f"{parts[-1]}, {' '.join(parts[:-1])}", d["name"].upper()])
    d["person_id"] = f"LIVE{rng.randint(10**5, 10**6 - 1)}"
    kb["people"] = pd.concat([P, pd.DataFrame([d])], ignore_index=True)
    # the copy also carries the award link, as the original source did
    R = kb["relationships"]
    links = R[(R.src == rid) & (R.rel == "received")].copy()
    if len(links):
        links["src"] = d["person_id"]
        kb["relationships"] = pd.concat([R, links], ignore_index=True)
    return d["person_id"], d["name"]


STAGES = [("placeholders", "Missing values", "Placeholders such as N/A become explicit gaps"),
          ("repair_places", "Places", "Country typos fixed, duplicate places merged"),
          ("repair_events", "Events", "Award years settled by vote, duplicate events merged"),
          ("repair_people", "People", "Duplicate people merged, birth dates checked against awards"),
          ("repair_relationships", "Relationships", "Links re-pointed, award years corrected, missing links restored")]


def _snapshot(rep, kb, done):
    """The knowledge base as it stands after the stages in `done`."""
    out = {t: df.copy() for t, df in kb.items()}
    if "repair_places" in done:
        out["places"] = rep.L.reset_index(drop=True)
        for c in ("birth_place_id", "death_place_id"):
            out["people"][c] = out["people"][c].map(lambda v: rep.canon["places"].get(v, v) if v else v)
        out["events"]["place_id"] = out["events"]["place_id"].map(lambda v: rep.canon["places"].get(v, v) if v else v)
    if "repair_events" in done:
        out["events"] = rep.E.reset_index(drop=True)
    if "repair_people" in done:
        out["people"] = rep.P.reset_index(drop=True)
    if "repair_relationships" in done:
        out["relationships"] = rep.Rout.reset_index(drop=True)
    return out


def stage_timeline(kb):
    rep = KBRepairer(kb)
    steps, done, seen = [{"stage": "start", "label": "Corrupted input", "quality": dq(kb)["DQ"], "actions": []}], [], 0
    for fn, label, what in STAGES:
        getattr(rep, fn)()
        done.append(fn)
        if fn == "repair_relationships":
            snap = rep.output()
        else:
            snap = _snapshot(rep, kb, done)
        new = rep.actions[seen:]
        seen = len(rep.actions)
        steps.append({"stage": fn, "label": label, "what": what, "quality": dq(snap)["DQ"],
                      "actions": [{k: (str(v) if isinstance(v, dict) else v) for k, v in a.items()} for a in new]})
    return steps
