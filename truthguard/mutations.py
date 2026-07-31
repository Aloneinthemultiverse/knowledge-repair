"""Mutations: make the gate enforce rather than advise.

Until now the gate governed what agents SAID. Tonight it proved it does not
govern what they DO — two agents proposed opposite repairs to a live database,
the gate correctly raised CONFLICTED, and one of them renamed the column anyway
using its own shell. TruthGuard's record showed known_facts and submit_claim.
The production change that actually happened is absent from it.

That is the whole gap. "If CONFLICTED, stop" is a request, and a request is not
governance.

Two phases, because checking and acting must be separable:

  propose_mutation   records the intent and runs it through the same admission
                     gate as any claim. Returns a verdict. Changes nothing.
  apply_mutation     executes ONLY if that verdict still holds. A conflict raised
                     between propose and apply blocks it, because the world may
                     have changed while the agent was thinking.

Enforcement does not come from the tool. An agent that still holds the database
password can bypass any tool offered to it. It comes from the agent not having
the credential: the executor here reads secrets from ITS OWN environment, and
the agent's environment has none. The gateway is then not the polite path, it is
the only path.

Every mutation carries its `inverse`. A change that cannot be undone should not
be easy to make, and requiring the caller to state the undo is the cheapest way
to force that thought before the fact rather than during an incident.
"""
import json
import os
import subprocess
import time
import uuid

from . import admission


# Executors are named, not passed as shell strings. An agent that could supply
# the command would simply supply `psql` and be back where it started; naming a
# whitelisted executor means the set of possible actions is fixed by the operator.
def _exec_psql(target: str, statement: str, timeout: int) -> dict:
    """Run SQL against a database. Credentials come from THIS process's
    environment — never from the agent, never from the arguments."""
    db = target.split(".")[0]
    cmd = os.getenv("TG_PSQL_CMD", "docker compose exec -T postgres psql -U postgres -d {db} -c").format(db=db)
    p = subprocess.run(cmd.split() + [statement], capture_output=True, text=True,
                       timeout=timeout, cwd=os.getenv("TG_MUTATION_CWD") or None)
    return {"exit_code": p.returncode,
            "stdout": (p.stdout or "")[-800:], "stderr": (p.stderr or "")[-800:]}


def _exec_shell(target: str, statement: str, timeout: int) -> dict:
    """Operator-enabled escape hatch, off unless TG_ALLOW_SHELL=1. Present so a
    deployment step can be governed, not so arbitrary commands can be."""
    if os.getenv("TG_ALLOW_SHELL") != "1":
        return {"exit_code": 126, "stdout": "",
                "stderr": "shell executor disabled; set TG_ALLOW_SHELL=1 to enable"}
    p = subprocess.run(statement, shell=True, capture_output=True, text=True,
                       timeout=timeout, cwd=os.getenv("TG_MUTATION_CWD") or None)
    return {"exit_code": p.returncode,
            "stdout": (p.stdout or "")[-800:], "stderr": (p.stderr or "")[-800:]}


EXECUTORS = {"schema": _exec_psql, "sql": _exec_psql, "deploy": _exec_shell}


def propose(cg, kind: str, target: str, statement: str, inverse: str = "",
            reason: str = "", agent_id: str = None, namespace: str = None,
            token: str = None, episode_id: str = None) -> dict:
    """Declare an intended change. Executes nothing.

    The proposal goes through admission.admit as a claim about the target's
    intended state, so two agents proposing different changes to the same target
    collide exactly as two agents asserting different facts do — the contradiction
    machinery is reused rather than reimplemented, and the conflict lands in the
    same review queue a human already watches.
    """
    if kind not in EXECUTORS:
        return {"error": f"unknown kind '{kind}'; expected {sorted(EXECUTORS)}"}
    if not statement.strip():
        return {"error": "statement is required"}

    agent_id = agent_id or os.getenv("TG_AGENT_ID", "default")
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    mid = "mut:" + uuid.uuid4().hex[:10]

    v = admission.admit(cg, {
        "subject": f"{target}", "relation": f"{kind} change",
        "object": statement.strip()[:400],
        "confidence": 0.9, "claimant": agent_id,
        "severity": "high" if kind in ("schema", "deploy") else "normal",
    }, agent_id=agent_id, namespace=namespace, token=token,
        episode_id=episode_id, save=False)

    cg.g.add_node(mid, plane="mutation", namespace=namespace, agent_id=agent_id,
                  kind=kind, target=target, statement=statement.strip(),
                  inverse=inverse.strip(), reason=reason[:300],
                  verdict=v.get("verdict"), claim=v.get("node"),
                  status="PROPOSED", proposed_at=time.time())
    if v.get("node"):
        cg.g.add_edge(mid, v["node"], relation="proposes")
    cg.save()

    blocked = v.get("verdict") in ("CONFLICTED", "DENIED", "THROTTLED")
    return {"mutation_id": mid, "verdict": v.get("verdict"),
            "blocked": blocked,
            "reason": v.get("reason"),
            "conflict": v.get("conflict"),
            "next": ("adjudicate the conflict, then apply_mutation"
                     if blocked else f"apply_mutation('{mid}')")}


def apply(cg, mutation_id: str, agent_id: str = None, timeout: int = 60) -> dict:
    """Execute a proposal, but only if its verdict still holds.

    Re-checked rather than trusted: a conflict may have been raised between the
    proposal and this call, and applying a change whose justification has since
    been contradicted is precisely the race the gate exists to prevent.
    """
    cg.refresh()
    m = cg.g.nodes.get(mutation_id)
    if not m or m.get("plane") != "mutation":
        return {"error": "unknown mutation"}
    if m.get("status") != "PROPOSED":
        return {"error": f"mutation is {m.get('status')}, not PROPOSED"}

    claim = m.get("claim")
    if claim and cg.g.has_node(claim):
        c = cg.g.nodes[claim]
        if c.get("retracted"):
            return {"blocked": True, "reason": "the claim behind this was retracted"}
        if c.get("write_verdict") == "CONFLICTED":
            return {"blocked": True, "reason":
                    "still CONFLICTED — a human must adjudicate before this applies"}
    if m.get("verdict") in ("DENIED", "THROTTLED"):
        return {"blocked": True, "reason": f"verdict was {m['verdict']}"}

    fn = EXECUTORS[m["kind"]]
    t0 = time.time()
    try:
        res = fn(m["target"], m["statement"], timeout)
    except Exception as e:
        res = {"exit_code": -1, "stdout": "", "stderr": f"{type(e).__name__}: {e}"[:400]}

    ok = res["exit_code"] == 0
    # What was ACTUALLY run, and what came back. This is the record that was
    # missing when an agent renamed a column outside the gate's view.
    m.update({"status": "APPLIED" if ok else "FAILED",
              "applied_at": time.time(),
              "applied_by": agent_id or os.getenv("TG_AGENT_ID", "default"),
              "exit_code": res["exit_code"],
              "stdout": res["stdout"], "stderr": res["stderr"],
              "duration_s": round(time.time() - t0, 2)})
    cg.save()
    return {"mutation_id": mutation_id, "status": m["status"],
            "exit_code": res["exit_code"],
            "output": (res["stdout"] or res["stderr"])[:300],
            "rollback": (f"rollback_mutation('{mutation_id}')"
                         if ok and m.get("inverse") else None)}


def rollback(cg, mutation_id: str, timeout: int = 60) -> dict:
    """Undo by running the inverse the proposer had to state up front."""
    cg.refresh()
    m = cg.g.nodes.get(mutation_id)
    if not m or m.get("plane") != "mutation":
        return {"error": "unknown mutation"}
    if m.get("status") != "APPLIED":
        return {"error": f"cannot roll back a {m.get('status')} mutation"}
    if not m.get("inverse"):
        return {"error": "no inverse was recorded; this change is not reversible"}

    res = EXECUTORS[m["kind"]](m["target"], m["inverse"], timeout)
    ok = res["exit_code"] == 0
    m.update({"status": "ROLLED_BACK" if ok else "ROLLBACK_FAILED",
              "rolled_back_at": time.time(),
              "rollback_exit": res["exit_code"],
              "rollback_output": (res["stdout"] or res["stderr"])[:400]})
    # the claim that justified it no longer holds, so anything built on it is
    # flagged by the same propagation a retraction uses
    if ok and m.get("claim") and cg.g.has_node(m["claim"]):
        admission.retract(cg, m["claim"], reason=f"mutation {mutation_id} rolled back",
                          save=False)
    cg.save()
    return {"mutation_id": mutation_id, "status": m["status"],
            "exit_code": res["exit_code"],
            "output": (res["stdout"] or res["stderr"])[:300]}


def log(cg, namespace: str = None, limit: int = 20) -> list:
    """Every mutation and what it actually did — the z-plane for actions."""
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    out = [{"mutation_id": n, "kind": d.get("kind"), "target": d.get("target"),
            "statement": str(d.get("statement"))[:120],
            "agent": d.get("agent_id"), "verdict": d.get("verdict"),
            "status": d.get("status"), "exit_code": d.get("exit_code"),
            "at": d.get("applied_at") or d.get("proposed_at")}
           for n, d in cg.g.nodes(data=True)
           if d.get("plane") == "mutation" and d.get("namespace") == namespace]
    out.sort(key=lambda x: -(x["at"] or 0))
    return out[:limit]
