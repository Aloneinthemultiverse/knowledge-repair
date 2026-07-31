"""Agent reliability, scored from what the gate already recorded.

Shape borrowed from agent-mesh's trust store (MIT): a 0-1000 score, updated on
interaction outcomes, rate-limited so an agent cannot inflate itself by spamming
successes. That rate limit is the non-obvious part and worth keeping — without
it, trust is trivially farmable by any agent that can call a cheap tool in a loop.

Where this differs, and it is the reason it can exist at all:

Their store needs a HOST to call `record_interaction(agent, success=True)`, so
"success" means whatever the host decided it meant, and an agent whose caller
never reports failures looks perfect forever. TruthGuard already records the
outcome the gate reached — ACCEPTED, CONFLICTED, retracted, adjudicated against —
so the score is derived from evidence the agent did not supply about itself.

That distinction matters most in the case worth catching: an agent whose claims
keep LOSING adjudications is unreliable even though every one of its calls
"succeeded". Nothing a host reports would show that.

Signals, weighted by how much they actually say:

  adjudicated against   strongest — a human looked at both sides and ruled
  retracted             strong    — the claim turned out to be wrong
  conflicted            weak      — being contradicted is not being wrong;
                                    the other agent may be the mistaken one
  accepted              baseline  — went in cleanly

Deliberately NOT counted: how many claims an agent makes. Volume is not
reliability, and rewarding it recreates the farming problem the rate limit exists
to prevent.
"""
import os
import time

BASE_SCORE = 500          # neutral start, matching agent-mesh's default
MAX_SCORE = 1000
MIN_SCORE = 0
MAX_UPDATES_PER_MINUTE = 10       # anti-inflation, borrowed directly

_WEIGHTS = {
    "accepted": +2,
    "confirmed": +6,       # another agent's independent claim agreed
    "conflicted": -3,
    "retracted": -25,
    "lost_adjudication": -60,
    "won_adjudication": +30,
}

_update_times = {}


def _rate_limited(agent_id: str) -> bool:
    """True when this agent has already been updated too often this minute."""
    now = time.time()
    times = [t for t in _update_times.setdefault(agent_id, []) if t > now - 60]
    _update_times[agent_id] = times
    if len(times) >= MAX_UPDATES_PER_MINUTE:
        return True
    times.append(now)
    return False


def score(cg, agent_id: str, namespace: str = None) -> dict:
    """Reliability for one agent, recomputed from the graph.

    Recomputed rather than incrementally stored: an adjudication that happens
    weeks later changes the meaning of a claim written long ago, and a running
    counter would have to be walked backwards to reflect it. Deriving from
    current state means a verdict reached today is priced in immediately.
    """
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    tally = {k: 0 for k in _WEIGHTS}
    claims = [(n, d) for n, d in cg.g.nodes(data=True)
              if d.get("plane") == "claim" and d.get("namespace") == namespace
              and d.get("agent_id") == agent_id]

    for nid, d in claims:
        if d.get("retracted"):
            tally["retracted"] += 1
        elif d.get("write_verdict") == "CONFLICTED":
            tally["conflicted"] += 1
        elif d.get("write_verdict") == "ACCEPTED":
            tally["accepted"] += 1
        for _, dst, e in cg.g.out_edges(nid, data=True):
            if e.get("relation") == "confirms":
                tally["confirmed"] += 1

    # Adjudications: the strongest signal, because a human compared both sides.
    mine = {n for n, _ in claims}
    for _, d in cg.g.nodes(data=True):
        if d.get("plane") != "conflict" or d.get("status") != "RESOLVED":
            continue
        if d.get("namespace") != namespace:
            continue
        winner = d.get("winner")
        for c in d.get("claims") or []:
            if c.get("node") not in mine:
                continue
            if winner and c["node"] == winner:
                tally["won_adjudication"] += 1
            elif winner:
                tally["lost_adjudication"] += 1

    raw = BASE_SCORE + sum(_WEIGHTS[k] * v for k, v in tally.items())
    return {"agent_id": agent_id, "namespace": namespace,
            "score": max(MIN_SCORE, min(MAX_SCORE, raw)),
            "claims": len(claims), "signals": {k: v for k, v in tally.items() if v},
            "band": band(max(MIN_SCORE, min(MAX_SCORE, raw)))}


def band(s: int) -> str:
    """Coarse label, because a raw number invites false precision. 500 is neutral
    and means "no evidence yet", not "average"."""
    if s >= 700:
        return "reliable"
    if s >= 450:
        return "unproven"
    if s >= 250:
        return "questionable"
    return "unreliable"


def fleet_scores(cg, namespace: str = None) -> list:
    """Every agent that has written here, worst first — the review order."""
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    agents = {d.get("agent_id") for _, d in cg.g.nodes(data=True)
              if d.get("plane") == "claim" and d.get("namespace") == namespace
              and d.get("agent_id")}
    out = [score(cg, a, namespace) for a in sorted(agents)]
    out.sort(key=lambda r: r["score"])
    return out


def record_interaction(cg, agent_id: str, success: bool, namespace: str = None,
                       save: bool = True) -> dict:
    """Host-reported outcome, for work the gate never saw.

    Kept for compatibility with agent-mesh's interface and for actions outside
    shared memory — a deploy, an API call. Stored as an adjustment rather than
    folded into the derived score, so evidence the agent supplied about itself
    stays distinguishable from evidence the gate observed.
    """
    if _rate_limited(agent_id):
        return {"agent_id": agent_id, "ignored": "rate limited",
                "limit_per_minute": MAX_UPDATES_PER_MINUTE}
    nid = f"agent:{agent_id}"
    if not cg.g.has_node(nid):
        return {"error": f"unknown agent '{agent_id}'"}
    d = cg.g.nodes[nid]
    d["host_reported_ok"] = int(d.get("host_reported_ok") or 0) + (1 if success else 0)
    d["host_reported_fail"] = int(d.get("host_reported_fail") or 0) + (0 if success else 1)
    if save:
        cg.save()
    return {"agent_id": agent_id, "success": success,
            "host_reported_ok": d["host_reported_ok"],
            "host_reported_fail": d["host_reported_fail"]}
