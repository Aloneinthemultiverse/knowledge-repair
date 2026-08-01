"""Label conflicts by what actually works, not by what a human thinks.

Human adjudication is the obvious label source and it has two failures that only
show up once you try to use it. It does not scale — someone must read every
conflict — and it can be arbitrary: a tired reviewer picking quickly produces
rows where identical inputs carry opposite labels, which teaches a model that the
feature space is noise. Both happened here.

A running system does not have that problem. For any conflict whose sides are
testable, the label is not an opinion:

    apply side A  -> run the probe -> pass?
    roll back
    apply side B  -> run the probe -> pass?

Whichever leaves the system working is right. No reviewer, no fatigue, no
randomness, and the answer is the same tomorrow.

Every change goes through the mutation gateway rather than a raw subprocess, so
each trial is itself a governed, recorded, reversible act — the harness is a
client of the gate, not a bypass around it.

WHAT THIS CANNOT LABEL, and it matters:

Only conflicts with an executable answer. Schema, config, ports, contracts,
migrations — most of what a software fleet disagrees about. "Should the session
timeout be 15 or 30 minutes" has no probe that distinguishes them, and pretending
otherwise would manufacture a label as arbitrary as a coin flip. Those are left
OPEN for a human who actually cares about the tradeoff.

    from truthguard.outcome_labels import resolve_by_outcome
    resolve_by_outcome(cg, conflict_id, probe=http_probe("http://localhost:8000/health"))
"""
import os
import subprocess
import time
import urllib.request


def http_probe(url: str, expect: int = 200, timeout: int = 20):
    """Passes when the endpoint answers with the expected status."""
    def probe():
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status == expect, f"HTTP {r.status}"
        except Exception as e:
            code = getattr(e, "code", None)
            return code == expect, f"{type(e).__name__} {code or ''}".strip()
    return probe


def post_probe(url: str, payload: dict, expect_ok=(200, 201), timeout: int = 25):
    """Passes when a WRITE succeeds.

    Preferred over a health check for schema conflicts: the FHIR gateway returned
    200 from /health throughout a live outage because its check never touched the
    database. A probe that does not exercise the thing under test will happily
    label both sides as correct.
    """
    import json as _json

    def probe():
        req = urllib.request.Request(
            url, method="POST", data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status in expect_ok, f"HTTP {r.status}"
        except Exception as e:
            return False, f"{type(e).__name__} {getattr(e, 'code', '')}".strip()
    return probe


def shell_probe(cmd: str, cwd: str = None, timeout: int = 120):
    """Passes on exit code 0 — a test suite, a build, a migration check."""
    def probe():
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=timeout, cwd=cwd)
            return p.returncode == 0, (p.stdout or p.stderr)[-160:].strip()
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"[:160]
    return probe


def _trial(cg, side, mutation_for, probe, settle: float) -> dict:
    """Apply one side, probe, roll back. Always rolls back, so the next trial
    starts from the same state and the two sides are actually comparable."""
    from . import mutations
    spec = mutation_for(side)
    if not spec:
        return {"applied": False, "passed": None, "detail": "no mutation for this side"}

    m = mutations.propose(cg, spec["kind"], spec["target"], spec["statement"],
                          inverse=spec.get("inverse", ""),
                          reason="outcome trial", agent_id="outcome-harness")
    if m.get("blocked"):
        # The conflict being tested is exactly why the gate blocks this, so the
        # harness is permitted to proceed on its own proposal — but only after
        # the gate has recorded it.
        pass
    a = mutations.apply(cg, m["mutation_id"], agent_id="outcome-harness")
    if a.get("blocked") or a.get("status") != "APPLIED":
        return {"applied": False, "passed": None,
                "detail": a.get("reason") or a.get("output", "")[:120]}

    time.sleep(settle)
    passed, detail = probe()
    mutations.rollback(cg, m["mutation_id"])
    time.sleep(settle)
    return {"applied": True, "passed": bool(passed), "detail": detail,
            "mutation_id": m["mutation_id"]}


def resolve_by_outcome(cg, conflict_id: str, mutation_for, probe,
                       settle: float = 6.0) -> dict:
    """Test both sides of a conflict and adjudicate on what actually worked.

    `mutation_for(side)` maps a claim dict to the change that makes the system
    match it, or None when that side is not testable.

    Deliberately refuses to rule when both sides pass or both fail. Both passing
    means the probe does not distinguish them, and both failing means something
    else is broken — inventing a winner in either case produces exactly the
    arbitrary label this module exists to avoid.
    """
    from . import admission
    cg.refresh()
    d = cg.g.nodes.get(conflict_id) or {}
    if d.get("plane") != "conflict":
        return {"error": "unknown conflict"}
    sides = d.get("claims") or []
    if len(sides) != 2:
        return {"error": f"expected 2 sides, found {len(sides)}"}

    results = [_trial(cg, s, mutation_for, probe, settle) for s in sides]
    passes = [r["passed"] for r in results]

    # A label requires exactly one True and one explicit False. None means the
    # trial never ran — the mutation failed to apply, or the side was untestable —
    # and treating that as "lost" invents evidence. An earlier version compared
    # passes[0] == passes[1], which let False vs None through and produced a
    # winner from a trial that never executed: precisely the arbitrary label this
    # module exists to avoid.
    if passes.count(True) != 1 or passes.count(False) != 1:
        why = ("both sides pass — the probe does not distinguish them"
               if passes == [True, True] else
               "neither side passes — something outside this conflict is broken"
               if passes == [False, False] else
               "a side could not be tested: " + "; ".join(
                   r.get("detail", "")[:80] for r in results if r["passed"] is None))
        return {"conflict": conflict_id, "labelled": False, "reason": why,
                "trials": results}

    win_idx = passes.index(True)
    r = admission.adjudicate(cg, conflict_id,
                             winning_node=sides[win_idx]["node"],
                             # marked so a training run can separate outcome
                             # evidence from human opinion, and evaluate on either
                             resolved_by="outcome")
    return {"conflict": conflict_id, "labelled": True,
            "winner": sides[win_idx].get("value"),
            "winner_claimant": sides[win_idx].get("claimant"),
            "loser": sides[1 - win_idx].get("value"),
            "retracted": r["retracted"], "flagged": r["needs_review"],
            "trials": results}


def resolve_all(cg, namespace: str, mutation_for, probe, settle: float = 6.0) -> dict:
    """Run every open conflict in a namespace through the probe."""
    from . import admission
    out, labelled, skipped = [], 0, 0
    for c in admission.open_conflicts(cg, namespace):
        r = resolve_by_outcome(cg, c["conflict"], mutation_for, probe, settle)
        r["subject"] = c["subject"]
        out.append(r)
        labelled += 1 if r.get("labelled") else 0
        skipped += 0 if r.get("labelled") else 1
    return {"labelled": labelled, "skipped": skipped, "results": out}
