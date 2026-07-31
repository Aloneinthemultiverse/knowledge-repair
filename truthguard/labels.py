"""Sensitivity labels on claims, joined automatically across derivations.

Adapted from the information-flow-control model in Microsoft's Agent Governance
Toolkit (MIT licensed): a sensitivity lattice, dominance, no-write-down at the
sink, and "the label of a derived result is the join of its sources".

Where this deliberately differs, and why it matters:

Their core is STATELESS by design — it "does not store labels, propagate taint"
and "the host owns provenance tracking". The host must instrument every path,
attach labels to every hop, and pass them back in on the next call. Their own
docs name the consequence: "uninstrumented paths, host side label loss, and sinks
that do not call ACS are outside the guarantee". Correctness rests on nobody ever
forgetting to thread a label through.

TruthGuard already stores the provenance. `derived_from` edges record which claims
a conclusion was reasoned from, so the join can be COMPUTED from the graph rather
than supplied by a caller. A claim cannot lose its label by being passed through
code that forgot to carry it, because the label is never carried — it is derived
on demand from what the claim is built on.

That turns their instrumentation requirement into a graph walk, which is the same
walk trust propagation already does when a premise is retracted.

    effective_sensitivity(cg, claim_id)  -> highest label among self + ancestors
    may_read(agent_clearance, label)     -> no-write-down check
"""
import os

# Total order, matching the reference lattice so labels are interchangeable with
# theirs. `_RANK` is the whole definition of dominance for an ordered lattice.
DEFAULT_LATTICE = ("public", "internal", "confidential", "secret")
_RANK = {name: i for i, name in enumerate(DEFAULT_LATTICE)}

MAX_DEPTH = 6          # same bound as trust propagation; chains deeper are absent


def rank(label: str) -> int:
    """Position in the lattice. Unknown labels fail CLOSED — an unrecognised
    label is treated as maximally sensitive rather than ignored, so a typo
    restricts access instead of silently publishing."""
    if not label:
        return _RANK["public"]
    return _RANK.get(str(label).strip().lower(), len(DEFAULT_LATTICE) - 1)


def join(labels) -> str:
    """Maximum sensitivity of a set — the label a result derived from all of them
    must carry."""
    best = "public"
    for l in labels or ():
        if rank(l) > rank(best):
            best = str(l).strip().lower() if str(l).strip().lower() in _RANK else DEFAULT_LATTICE[-1]
    return best


def dominates(clearance: str, label: str) -> bool:
    """True when `clearance` is at least as high as `label`."""
    return rank(clearance) >= rank(label)


def may_read(clearance: str, label: str) -> bool:
    """No read up: a sink may receive data only when its clearance dominates."""
    return dominates(clearance, label)


def effective_sensitivity(cg, node: str, _depth: int = 0, _seen=None) -> str:
    """The label a claim actually carries, including everything it was built on.

    A conclusion is only as public as its most sensitive premise. Declaring a
    derived claim `internal` while it rests on a `secret` one is precisely the
    laundering an information-flow model exists to stop — and because the
    derivation is recorded as an edge, nobody has to remember to declare it.
    """
    if _seen is None:
        _seen = set()
    if node in _seen or _depth > MAX_DEPTH or not cg.g.has_node(node):
        return "public"
    _seen.add(node)
    d = cg.g.nodes[node]
    own = d.get("sensitivity") or "public"
    labels = [own]
    for src in cg.g.successors(node):
        rel = cg.g.edges[node, src].get("relation")
        if rel in ("derived_from", "grounds"):
            labels.append(effective_sensitivity(cg, src, _depth + 1, _seen))
    return join(labels)


def agent_clearance(cg, agent_id: str, namespace: str = None) -> str:
    """An agent's clearance, from its registration. Absent means `public`, so an
    unregistered agent sees only unclassified claims rather than everything."""
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    rec = cg.g.nodes.get(f"agent:{agent_id}") or {}
    return rec.get("clearance") or "public"


def set_clearance(cg, agent_id: str, clearance: str, save: bool = True) -> dict:
    """Operator action: grant an agent a clearance level."""
    nid = f"agent:{agent_id}"
    if not cg.g.has_node(nid):
        return {"error": f"unknown agent '{agent_id}' — register it first"}
    if str(clearance).strip().lower() not in _RANK:
        return {"error": f"unknown label '{clearance}'; expected one of {DEFAULT_LATTICE}"}
    cg.g.nodes[nid]["clearance"] = str(clearance).strip().lower()
    if save:
        cg.save()
    return {"agent_id": agent_id, "clearance": cg.g.nodes[nid]["clearance"]}


def visible_to(cg, agent_id: str, claim_ids, namespace: str = None) -> list:
    """Filter claims to those this agent is cleared to read.

    Applied at the READ path rather than at the write, because a claim's
    sensitivity can rise after it is written — the moment something confidential
    is derived from it, or a source is reclassified. Checking at read time means
    that change takes effect immediately for everyone, with nothing to re-stamp.
    """
    clearance = agent_clearance(cg, agent_id, namespace)
    out = []
    for cid in claim_ids:
        if may_read(clearance, effective_sensitivity(cg, cid)):
            out.append(cid)
    return out
