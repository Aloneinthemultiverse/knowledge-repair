"""The write admission gate: what may enter shared memory.

The assessment gate protects one direction — it decides whether a question may
be *answered*. In a fleet the same machinery has to protect the other: whether a
conclusion may be *remembered*. Without it the last writer wins, so one agent's
wrong conclusion silently becomes another agent's premise and hallucination
compounds as the fleet gets busier.

Three verdicts:

  ACCEPTED     clean; visible to every agent
  QUARANTINED  below the namespace confidence floor; stored, not served
  CONFLICTED   contradicts a stored claim over an overlapping validity window;
               BOTH persist, neither wins, a conflict node is materialised

The rule is deliberately the same one assess.py already applies within a single
query — same subject and relation, overlapping validity, different canonical
value — pointed at stored claims across agents instead of chunks within a query.
It is an interval check: no model call, no ambiguity.

Nothing is ever silently overwritten. That is the whole point.
"""
import os
import time

from . import config
from .assess import _canon_value, _temporal_overlap

# Below this, a claim is stored but not served. Overridable per namespace so a
# clinical tenant can demand more confidence than a documentation one.
DEFAULT_FLOOR = float(os.getenv("TG_ADMIT_FLOOR", "0.35"))


def _floor(cg, namespace: str) -> float:
    """Per-namespace confidence floor, falling back to the global default."""
    n = cg.g.nodes.get(f"ns:{namespace}", {})
    return float(n.get("admit_floor", DEFAULT_FLOOR))


def _stored_claims(cg, namespace: str) -> list:
    """Every claim currently asserted as fact in this namespace."""
    out = []
    for n, d in cg.g.nodes(data=True):
        if d.get("plane") != "claim" or d.get("namespace") != namespace:
            continue
        if d.get("write_verdict") == "QUARANTINED" or d.get("retracted"):
            continue
        out.append((n, d))
    return out


def admit(cg, claim: dict, agent_id: str = None, namespace: str = None,
          episode_id: str = None, save: bool = True) -> dict:
    """Submit one claim to shared memory.

    claim = {subject, relation, object, confidence,
             valid_from?, valid_until?, sources?, severity?}

    Returns the verdict plus the node id, and — when CONFLICTED — the id of the
    conflict node holding both sides.
    """
    agent_id = agent_id or os.getenv("TG_AGENT_ID", "default")
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")

    # The conflict check below is read-then-write. Two agents submitting
    # contradicting claims at the same moment could otherwise both read "no
    # conflict" and both write, and the contradiction would never be detected.
    # Holding the write lock for the whole decision makes the second agent block,
    # re-read, and correctly see the first agent's claim.
    if save:
        from . import graph_store
        with graph_store.writer(cg.storage_dir) as conn:
            cg.refresh()                       # see anything committed meanwhile
            r = _admit_locked(cg, claim, agent_id, namespace, episode_id)
            cg.save(conn=conn)
            return r
    return _admit_locked(cg, claim, agent_id, namespace, episode_id)


# ── policy: what each agent is allowed to assert ─────────────────────────────
# Attribution alone is not governance. Without this, any agent may assert
# anything about anything — a symptom checker could assert a dose, and the only
# evidence would be an agent_id on a claim nobody read. Capability is declared
# per namespace and enforced at the gate.

def set_policy(cg, agent_id: str, relations: list, namespace: str = None,
               quota_per_hour: int = 0, save: bool = True) -> dict:
    """Declare what an agent may assert. relations=['*'] permits everything."""
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    nid = f"policy:{namespace}:{agent_id}"
    cg.g.add_node(nid, plane="policy", namespace=namespace, agent_id=agent_id,
                  relations=[r.strip().lower() for r in relations],
                  quota_per_hour=int(quota_per_hour), updated_at=time.time())
    if save:
        cg.save()
    return {"agent_id": agent_id, "namespace": namespace,
            "relations": relations, "quota_per_hour": quota_per_hour}


def get_policy(cg, agent_id: str, namespace: str) -> dict:
    """No policy means unrestricted, so existing single-agent setups are
    unaffected until a policy is declared."""
    return cg.g.nodes.get(f"policy:{namespace}:{agent_id}", {})


def _policy_allows(cg, agent_id: str, namespace: str, relation: str) -> bool:
    p = get_policy(cg, agent_id, namespace)
    rels = p.get("relations")
    if not rels:
        return True
    return "*" in rels or relation.lower() in rels


def _quota_exceeded(cg, agent_id: str, namespace: str) -> tuple:
    """Cap claims per agent per hour so one looping agent cannot drown the graph.

    Counted over claims actually written, not tool calls, because the cost being
    limited is pollution of shared memory rather than compute.
    """
    p = get_policy(cg, agent_id, namespace)
    limit = int(p.get("quota_per_hour") or 0)
    if limit <= 0:
        return False, 0, 0
    cutoff = time.time() - 3600
    n = sum(1 for _, d in cg.g.nodes(data=True)
            if d.get("plane") == "claim" and d.get("agent_id") == agent_id
            and d.get("namespace") == namespace
            and float(d.get("asserted_at") or 0) >= cutoff)
    return n >= limit, n, limit


def _admit_locked(cg, claim: dict, agent_id: str, namespace: str,
                  episode_id: str) -> dict:
    """The gate decision itself. Caller owns the transaction and the save."""
    subj = str(claim.get("subject", "")).strip().lower()
    rel = str(claim.get("relation", "")).strip().lower()
    val = _canon_value(str(claim.get("object", "")))
    conf = float(claim.get("confidence") or 0.0)
    if not subj or not rel:
        return {"verdict": "REJECTED", "reason": "claim needs subject and relation"}

    # policy and quota are checked before anything is written: a claim an agent
    # was never permitted to make should not occupy shared memory at all.
    if not _policy_allows(cg, agent_id, namespace, rel):
        return {"verdict": "DENIED", "reason":
                f"agent '{agent_id}' is not permitted to assert '{rel}' in '{namespace}'",
                "allowed": get_policy(cg, agent_id, namespace).get("relations")}
    over, used, limit = _quota_exceeded(cg, agent_id, namespace)
    if over:
        return {"verdict": "THROTTLED", "reason":
                f"agent '{agent_id}' has written {used}/{limit} claims this hour"}

    nid = f"claim:{abs(hash((subj, rel, val, agent_id, time.time())))%10**12}"
    node = {"plane": "claim", "namespace": namespace, "agent_id": agent_id,
            "subject": subj, "relation": rel, "object": claim.get("object"),
            "canonical": val, "confidence": conf,
            "valid_from": claim.get("valid_from"),
            "valid_until": claim.get("valid_until"),
            "severity": claim.get("severity", "normal"),
            "sources": (claim.get("sources") or [])[:8],
            "asserted_at": time.time()}

    # 1) confidence floor — stored, but not visible to other agents
    if conf < _floor(cg, namespace):
        node["write_verdict"] = "QUARANTINED"
        cg.g.add_node(nid, **node)
        _link(cg, nid, node, episode_id)
        return {"verdict": "QUARANTINED", "node": nid,
                "reason": f"confidence {conf:.2f} below floor {_floor(cg, namespace):.2f}"}

    # 2) contradiction against what is already asserted
    for other_id, other in _stored_claims(cg, namespace):
        if other.get("subject") != subj or other.get("relation") != rel:
            continue
        if other.get("canonical") == val:
            # independent agreement from a different agent is evidence, not noise
            if other.get("agent_id") != agent_id:
                cg.g.add_edge(nid, other_id, relation="confirms")
            continue
        if not _temporal_overlap(node, other):
            # separate windows: a timeline, not a conflict
            cg.g.add_edge(nid, other_id, relation="supersedes")
            continue
        # same subject+relation, different value, overlapping validity -> conflict
        node["write_verdict"] = "CONFLICTED"
        cg.g.add_node(nid, **node)
        _link(cg, nid, node, episode_id)
        conflict = _materialise_conflict(cg, nid, node, other_id, other, namespace)
        return {"verdict": "CONFLICTED", "node": nid, "conflict": conflict,
                "conflicts_with": other_id,
                "reason": f"'{other.get('object')}' vs '{claim.get('object')}' "
                          f"over overlapping validity"}

    node["write_verdict"] = "ACCEPTED"
    cg.g.add_node(nid, **node)
    _link(cg, nid, node, episode_id)
    return {"verdict": "ACCEPTED", "node": nid}


def _link(cg, nid: str, node: dict, episode_id: str) -> None:
    """Wire the claim to the run that produced it and the sources it used."""
    if episode_id and cg.g.has_node(episode_id):
        cg.g.add_edge(episode_id, nid, relation="produced")
    for s in node.get("sources") or []:
        if cg.g.has_node(s):
            cg.g.add_edge(nid, s, relation="grounds")


def _materialise_conflict(cg, a_id, a, b_id, b, namespace) -> str:
    """Both claims persist; neither wins. The disagreement becomes a first-class
    node so an agent reading this fact receives both sides plus the flag."""
    cid = f"conflict:{abs(hash((a['subject'], a['relation'], time.time())))%10**12}"
    cg.g.add_node(cid, plane="conflict", namespace=namespace,
                  subject=a["subject"], relation=a["relation"],
                  status="OPEN",
                  severity=max(a.get("severity", "normal"),
                               b.get("severity", "normal"), key=_sev_rank),
                  claims=[{"node": a_id, "agent_id": a.get("agent_id"),
                           "value": a.get("object"), "confidence": a.get("confidence"),
                           "sources": a.get("sources")},
                          {"node": b_id, "agent_id": b.get("agent_id"),
                           "value": b.get("object"), "confidence": b.get("confidence"),
                           "sources": b.get("sources")}],
                  opened_at=time.time(), resolved_by=None)
    cg.g.add_edge(a_id, b_id, relation="contradicts")
    cg.g.add_edge(cid, a_id, relation="holds")
    cg.g.add_edge(cid, b_id, relation="holds")
    return cid


_SEV = {"low": 0, "normal": 1, "high": 2, "critical": 3}


def _sev_rank(s):
    return _SEV.get(str(s).lower(), 1)


def open_conflicts(cg, namespace: str = None, min_severity: str = "low") -> list:
    """The clinician queue: unresolved conflicts, most severe first.

    Ranked and filterable because escalating everything equally is how clinical
    decision support earns alert fatigue and gets switched off.
    """
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    floor = _sev_rank(min_severity)
    out = []
    for n, d in cg.g.nodes(data=True):
        if d.get("plane") != "conflict" or d.get("status") != "OPEN":
            continue
        if d.get("namespace") != namespace or _sev_rank(d.get("severity")) < floor:
            continue
        out.append({"conflict": n, "subject": d["subject"], "relation": d["relation"],
                    "severity": d.get("severity"), "claims": d.get("claims"),
                    "opened_at": d.get("opened_at")})
    out.sort(key=lambda c: (-_sev_rank(c["severity"]), c["opened_at"] or 0))
    return out


def adjudicate(cg, conflict_id: str, winning_node: str = None,
               resolved_by: str = "clinician", save: bool = True) -> dict:
    """Resolve a conflict. Suppressed from the queue afterwards, so the same
    disagreement is not re-raised once a human has ruled on it.

    The losing claim is retracted rather than deleted — the record of what was
    believed, by whom, and when it was overturned is the audit trail.
    """
    d = cg.g.nodes.get(conflict_id)
    if not d or d.get("plane") != "conflict":
        return {"error": "unknown conflict"}
    d["status"] = "RESOLVED"
    d["resolved_by"] = resolved_by
    d["resolved_at"] = time.time()
    d["winner"] = winning_node
    retracted = []
    for c in d.get("claims") or []:
        if winning_node and c["node"] != winning_node and cg.g.has_node(c["node"]):
            cg.g.nodes[c["node"]]["retracted"] = True
            cg.g.nodes[c["node"]]["retracted_at"] = time.time()
            retracted.append(c["node"])
    flagged = []
    for r in retracted:
        flagged += propagate_retraction(cg, r, save=False)
    if save:
        cg.save()
    return {"conflict": conflict_id, "status": "RESOLVED",
            "winner": winning_node, "retracted": retracted,
            "needs_review": flagged}


# ── trust propagation ────────────────────────────────────────────────────────
# A conclusion built on a claim that later proves wrong is itself suspect. Without
# this, one refuted fact quietly poisons everything downstream of it — the exact
# failure mode a fleet amplifies, because agents build on each other's output.
#
# derived_from edges are what make the blast radius walkable. They are written
# whenever a conclusion grounds on another *conclusion* rather than on a source
# chunk, so the chain records reasoning, not just evidence.

MAX_DEPTH = 6          # a cycle guard and a sanity bound; chains this deep are rare


def link_derivation(cg, conclusion: str, built_on: list, save: bool = True) -> int:
    """Record that `conclusion` was reasoned from other stored conclusions."""
    n = 0
    for src in built_on or []:
        if cg.g.has_node(src) and not cg.g.has_edge(conclusion, src):
            cg.g.add_edge(conclusion, src, relation="derived_from")
            n += 1
    if save and n:
        cg.save()
    return n


def propagate_retraction(cg, node: str, save: bool = True) -> list:
    """Walk derived_from backwards from a retracted claim and flag dependents.

    Flagging, not deleting: a downstream conclusion may still hold for other
    reasons, so the system marks it for review rather than deciding on the
    clinician's behalf — the same principle as materialising conflicts instead
    of resolving them.
    """
    seen, queue, flagged = {node}, [(node, 0)], []
    while queue:
        cur, depth = queue.pop(0)
        if depth >= MAX_DEPTH:
            continue
        # who was built ON cur? those are the dependents
        for dep in cg.g.predecessors(cur):
            if dep in seen:
                continue
            if cg.g.edges[dep, cur].get("relation") != "derived_from":
                continue
            seen.add(dep)
            d = cg.g.nodes[dep]
            d["needs_review"] = True
            d["review_reason"] = f"derived from retracted {cur}"
            d["flagged_at"] = time.time()
            flagged.append(dep)
            queue.append((dep, depth + 1))
    if save and flagged:
        cg.save()
    return flagged


def retract(cg, node: str, reason: str = "", save: bool = True) -> dict:
    """Retract a claim directly and propagate. Used when a source is found wrong
    outside of an adjudicated conflict."""
    if not cg.g.has_node(node):
        return {"error": "unknown node"}
    cg.g.nodes[node]["retracted"] = True
    cg.g.nodes[node]["retracted_at"] = time.time()
    cg.g.nodes[node]["retraction_reason"] = reason
    flagged = propagate_retraction(cg, node, save=False)
    if save:
        cg.save()
    return {"retracted": node, "needs_review": flagged}


def review_queue(cg, namespace: str = None) -> list:
    """Conclusions flagged by propagation and not yet re-verified."""
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    return [{"node": n, "subject": d.get("subject"), "object": d.get("object"),
             "agent_id": d.get("agent_id"), "reason": d.get("review_reason")}
            for n, d in cg.g.nodes(data=True)
            if d.get("needs_review") and not d.get("retracted")
            and d.get("namespace") == namespace]
