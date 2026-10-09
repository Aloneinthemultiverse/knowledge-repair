"""python -m repair bench [--n 300] [--seed 7] [--out out_repair]"""
import argparse
import json
import os

import pandas as pd

from .engine import Repairer
from .report import build as build_report
from .score import against_truth, dq_score
from .synth import corrupt, make_clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="out_repair")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    clean, rels = make_clean(a.n, seed=1)
    dirty, drels, truth, log = corrupt(clean, rels, seed=a.seed)
    rep = Repairer(dirty, drels).run()

    for name, df in {"clean": clean, "dirty_people": dirty, "dirty_rels": drels, "injected_errors": log,
                     "issues": pd.DataFrame(rep.issues), "actions": pd.DataFrame(rep.actions),
                     "repaired_entities": rep.entities, "repaired_rels": rep.entity_rels,
                     "lineage": rep.lineage}.items():
        df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)

    issues = pd.DataFrame(rep.issues)
    actions = pd.DataFrame(rep.actions)
    res = {
        "injected": log.kind.value_counts().to_dict(),
        "detected": issues.kind.value_counts().to_dict(),
        "actions": actions.groupby(["kind", "status"]).size().rename("n").reset_index()
                          .apply(lambda r: f"{r.kind}/{r.status}", axis=1).to_list(),
        "truth_metrics": against_truth(rep, clean, truth),
        "dq_before": dq_score(dirty, "record_id", drels, set(dirty.record_id)),
        "dq_after": dq_score(rep.entities, "entity_id", rep.entity_rels, set(rep.entities.entity_id)),
    }
    res["action_counts"] = actions.groupby(["kind", "status"]).size().to_dict()
    res["action_counts"] = {f"{k[0]}/{k[1]}": int(v) for k, v in res["action_counts"].items()}
    del res["actions"]
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    build_report(os.path.join(a.out, "report.html"), res, rep, dirty)
    print(json.dumps(res["truth_metrics"], indent=2))
    print("report:", os.path.join(a.out, "report.html"))


if __name__ == "__main__":
    main()
