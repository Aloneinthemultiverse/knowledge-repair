"""Phase A: turn adjudications into training rows.

A resolved conflict is the only supervised signal this system produces — a human
looked at two claims with their provenance and said which one holds. Stored as a
graph node that is unusable for learning; this extracts it as a row.

One row per SIDE, not per conflict, with `won` as the target. Per-side rows let a
model score a single claim in isolation, which is what the review queue actually
needs: rank fifteen conflicts by which side is likely right, rather than only
compare pairs. The `conflict_id` column keeps the pairing recoverable for anyone
who wants pairwise training instead.

Every feature is already computed elsewhere — trust scores, sensitivity joins,
derivation depth. Nothing here is invented; it is extraction.

Two things deliberately NOT included:

  the claim's value       a model that learns "250 rps beats 100 rps" has learned
                          this corpus, not adjudication. Only provenance features.
  rule-generated labels   marked separately via `resolved_by`, because a label
                          produced by "2024 supersedes 2023" teaches the model
                          that rule rather than the human's judgement, and mixing
                          them silently would make a good score meaningless.

    python -m truthguard.labels_export --namespace fleet --out labels.jsonl
"""
import json
import os
import time


def _depth(cg, node: str, _seen=None, _d: int = 0) -> int:
    """How many premises deep this claim sits. A conclusion built on a long chain
    is more fragile than a direct reading, and that is exactly the kind of thing
    a human weighs without articulating."""
    if _seen is None:
        _seen = set()
    if node in _seen or _d > 6 or not cg.g.has_node(node):
        return _d
    _seen.add(node)
    best = _d
    for src in cg.g.successors(node):
        if cg.g.edges[node, src].get("relation") in ("derived_from", "informed_by"):
            best = max(best, _depth(cg, src, _seen, _d + 1))
    return best


def _corroboration(cg, node: str) -> int:
    """How many other claims independently confirm this one."""
    if not cg.g.has_node(node):
        return 0
    return sum(1 for _, dst, e in cg.g.out_edges(node, data=True)
               if e.get("relation") == "confirms") + \
           sum(1 for src, _, e in cg.g.in_edges(node, data=True)
               if e.get("relation") == "confirms")


def features(cg, node: str, namespace: str, now: float = None) -> dict:
    """Provenance features for one claim. No value, no subject text."""
    from . import trust, labels as lab
    now = now or time.time()
    d = cg.g.nodes.get(node) or {}
    claimant = str(d.get("claimant") or d.get("agent_id") or "")
    agent = str(d.get("agent_id") or "")
    # A document asserting a fact and an agent concluding one are different kinds
    # of evidence, and this is the single most likely feature to matter.
    is_doc = bool(claimant) and claimant != agent
    try:
        rel = trust.score(cg, agent, namespace).get("score", 500)
    except Exception:
        rel = 500
    return {
        "claim_id": node,
        "claimant_is_document": int(is_doc),
        "confidence": float(d.get("confidence") or 0.0),
        "age_hours": round(max(0.0, now - float(d.get("asserted_at") or now)) / 3600, 3),
        "n_sources": len(d.get("sources") or []),
        "derivation_depth": _depth(cg, node),
        "corroborations": _corroboration(cg, node),
        "agent_reliability": rel,
        "identity_verified": int(bool(d.get("identity_verified"))),
        "sensitivity_rank": lab.rank(lab.effective_sensitivity(cg, node)),
        "severity_rank": {"low": 0, "normal": 1, "high": 2, "critical": 3}.get(
            str(d.get("severity", "normal")).lower(), 1),
        "reasserted": int(d.get("reasserted") or 0),
        "was_quarantined": int(d.get("write_verdict") == "QUARANTINED"),
    }


def export(cg, namespace: str = None, include_rule_labels: bool = True) -> list:
    """All adjudicated conflicts as training rows."""
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    now = time.time()
    rows = []
    for cid, d in cg.g.nodes(data=True):
        if d.get("plane") != "conflict" or d.get("status") != "RESOLVED":
            continue
        if d.get("namespace") != namespace:
            continue
        winner = d.get("winner")
        if not winner:
            continue                      # resolved without a ruling; nothing to learn
        by = str(d.get("resolved_by") or "")
        # A rule-generated label teaches the model the rule. Kept, but marked, so
        # a model trained on both can be evaluated on human labels alone.
        rule_made = by.lower() in ("rule", "auto", "script", "")
        if rule_made and not include_rule_labels:
            continue
        for side in (d.get("claims") or []):
            n = side.get("node")
            if not n:
                continue
            rows.append({
                **features(cg, n, namespace, now),
                "conflict_id": cid,
                "subject": d.get("subject"),
                "relation": d.get("relation"),
                "won": int(n == winner),
                "resolved_by": by,
                "label_source": "rule" if rule_made else "human",
                "resolved_at": d.get("resolved_at"),
            })
    return rows


def summary(rows: list) -> dict:
    """What the dataset can and cannot support, stated before anyone trains on it."""
    import collections
    human = [r for r in rows if r["label_source"] == "human"]
    conflicts = {r["conflict_id"] for r in rows}
    feats = [k for k in (rows[0] if rows else {})
             if k not in ("claim_id", "conflict_id", "subject", "relation", "won",
                          "resolved_by", "label_source", "resolved_at")]
    # Constant features carry no signal and will confuse feature-importance
    # readings; better to name them now than to discover it after training.
    constant = [f for f in feats if len({r[f] for r in rows}) <= 1] if rows else []
    return {
        "rows": len(rows), "conflicts": len(conflicts),
        "human_labelled_rows": len(human),
        "rule_labelled_rows": len(rows) - len(human),
        "won_balance": dict(collections.Counter(r["won"] for r in rows)),
        "features": len(feats),
        "constant_features": constant,
        "ready_for": ("nothing yet — collect more" if len(human) < 40 else
                      "baseline trees" if len(human) < 300 else "graph model"),
        "note": "train and report on human rows only; rule rows teach the rule.",
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default=os.getenv("TG_NAMESPACE", "default"))
    ap.add_argument("--out", default="labels.jsonl")
    ap.add_argument("--human-only", action="store_true")
    a = ap.parse_args()
    from .context_graph import ContextGraph
    cg = ContextGraph()
    rows = export(cg, a.namespace, include_rule_labels=not a.human_only)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(json.dumps(summary(rows), indent=1))
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
