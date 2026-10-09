"""Knowledge Repair Tool.

    python -m kb run <in_dir> [--out out_kb]     repair people/places/events/relationships CSVs
    python -m kb bench [--seeds 7,11,23,42,99] [--out out_kb_bench]
                                                  real Wikidata KB -> inject errors -> repair -> score

`run` writes: repaired_<table>.csv, lineage.csv, issues.csv, actions.csv, quality.json, report.html
"""
import argparse
import json
import os
import time

import pandas as pd

from .dq import dq
from .repair import KBRepairer
from .report import build as build_report

TABLES = ("people", "places", "events", "relationships")


def read_kb(path):
    kb = {}
    for t in TABLES:
        df = pd.read_csv(os.path.join(path, f"{t}.csv"), dtype=str, keep_default_na=False)
        kb[t] = df.where(df != "", None)
    return kb


def repair_and_export(kb, out, truth_metrics=None):
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    rep = KBRepairer(kb).run()
    secs = time.perf_counter() - t0
    fixed = rep.output()
    for t, df in fixed.items():
        df.to_csv(os.path.join(out, f"repaired_{t}.csv"), index=False)
    pd.DataFrame(rep.lineage).to_csv(os.path.join(out, "lineage.csv"), index=False)
    pd.DataFrame(rep.issues).to_csv(os.path.join(out, "issues.csv"), index=False)
    pd.DataFrame(rep.actions).to_csv(os.path.join(out, "actions.csv"), index=False)
    quality = {"before": dq(kb), "after": dq(fixed), "seconds": round(secs, 2),
               "records_before": {t: len(kb[t]) for t in TABLES},
               "records_after": {t: len(fixed[t]) for t in TABLES}}
    if truth_metrics:
        quality["against_ground_truth"] = truth_metrics
    with open(os.path.join(out, "quality.json"), "w") as f:
        json.dump(quality, f, indent=1, default=str)
    build_report(os.path.join(out, "report.html"), kb, rep, quality)
    return rep, quality


def main():
    ap = argparse.ArgumentParser(prog="python -m kb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("in_dir")
    r.add_argument("--out", default="out_kb")
    b = sub.add_parser("bench")
    b.add_argument("--seeds", default="7,11,23,42,99")
    b.add_argument("--out", default="out_kb_bench")
    a = ap.parse_args()

    if a.cmd == "run":
        rep, q = repair_and_export(read_kb(a.in_dir), a.out)
        print(json.dumps({"DQ": f"{q['before']['DQ']} -> {q['after']['DQ']}", "issues": len(rep.issues),
                          "actions": len(rep.actions), "seconds": q["seconds"], "out": a.out}, indent=1))
        return

    from .corrupt import corrupt
    from .load import load
    from .score import score
    clean = load()
    results = {}
    for i, sd in enumerate(int(x) for x in a.seeds.split(",")):
        dirty, truth, log = corrupt(clean, seed=sd)
        rep = KBRepairer(dirty).run()
        m = score(rep, clean, truth, log)
        m["actions"] = dict(m["actions"])
        out = os.path.join(a.out, f"seed{sd}")
        os.makedirs(out, exist_ok=True)
        for t, df in dirty.items():
            df.to_csv(os.path.join(out, f"dirty_{t}.csv"), index=False)
        log.to_csv(os.path.join(out, "injected_errors.csv"), index=False)
        _, q = repair_and_export(dirty, out, truth_metrics=m)
        results[sd] = {"truth": m, "DQ": f"{q['before']['DQ']} -> {q['after']['DQ']}", "seconds": q["seconds"]}
        print(sd, results[sd]["DQ"], m["relationships_after_P/R/F1"], m["harm_total"], f"{q['seconds']}s")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()
