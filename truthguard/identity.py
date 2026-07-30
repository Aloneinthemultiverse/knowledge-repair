"""Verified agent identity.

Until now an agent's identity was `TG_AGENT_ID` — an environment variable it set
about itself. Every governance control downstream of it inherited that weakness:
an agent whose capability policy forbade a relation could simply declare a
different agent_id, and the quota, the policy and the audit trail would all
follow the lie. Attribution that the attributed party controls is not
attribution.

So identity becomes a registration: the fleet operator registers each agent once
and receives a secret. The agent presents it (`TG_AGENT_TOKEN`) and the gate
verifies it before honouring the claimed id. Only the hash is stored, so the
graph is not a credential store.

Enforcement is per namespace and opt-in via `require_auth`, for one reason: a
single-agent install has no identity problem to solve, and turning verification
on unconditionally would break every existing deployment at import time. A
namespace that has registered agents and set require_auth rejects unverified
claims outright.

    secret = register_agent(cg, "drug-safety")     # operator, once
    TG_AGENT_ID=drug-safety TG_AGENT_TOKEN=<secret> ...   # agent, thereafter
"""
import hashlib
import hmac
import os
import secrets
import time


def _hash(agent_id: str, token: str) -> str:
    """Salted with the agent id so the same token under two ids differs, and
    slow-ish by design — this is a credential, not a cache key."""
    return hashlib.pbkdf2_hmac(
        "sha256", token.encode(), f"tg:{agent_id}".encode(), 50_000).hex()


def register_agent(cg, agent_id: str, namespace: str = None,
                   save: bool = True) -> dict:
    """Register an agent and return its secret ONCE. Only the hash is kept.

    Re-registering rotates the secret, which is also how a leaked token is
    revoked: the old one stops verifying immediately.
    """
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    token = secrets.token_urlsafe(24)
    cg.g.add_node(f"agent:{agent_id}", plane="agent", agent_id=agent_id,
                  namespace=namespace, token_hash=_hash(agent_id, token),
                  registered_at=time.time())
    if save:
        cg.save()
    return {"agent_id": agent_id, "namespace": namespace, "token": token,
            "note": "store this now; it is not recoverable from the graph"}


def revoke_agent(cg, agent_id: str, save: bool = True) -> dict:
    """Drop the hash. The id can still appear in history — past claims keep their
    attribution — but it can no longer authenticate."""
    nid = f"agent:{agent_id}"
    if not cg.g.has_node(nid):
        return {"error": "unknown agent"}
    cg.g.nodes[nid]["token_hash"] = None
    cg.g.nodes[nid]["revoked_at"] = time.time()
    if save:
        cg.save()
    return {"revoked": agent_id}


def is_registered(cg, agent_id: str) -> bool:
    return bool((cg.g.nodes.get(f"agent:{agent_id}") or {}).get("token_hash"))


def any_registered(cg) -> bool:
    """Whether this graph uses verified identity at all. Lets unregistered
    single-agent installs keep working untouched."""
    return any(d.get("plane") == "agent" and d.get("token_hash")
               for _, d in cg.g.nodes(data=True))


def requires_auth(cg, namespace: str) -> bool:
    """Namespaces opt in. Default follows TG_REQUIRE_AUTH so an operator can turn
    it on fleet-wide with one variable."""
    n = cg.g.nodes.get(f"ns:{namespace}") or {}
    if "require_auth" in n:
        return bool(n["require_auth"])
    return os.getenv("TG_REQUIRE_AUTH", "0") == "1"


def verify(cg, agent_id: str, token: str = None) -> dict:
    """Decide whether this process may act as `agent_id`.

    Returns {"ok": bool, "reason": str}. `ok` is True in open mode so that
    verification can be introduced to a running fleet without a flag day: agents
    are registered first, then the namespace is switched to require_auth.
    """
    token = token if token is not None else os.getenv("TG_AGENT_TOKEN", "")
    rec = cg.g.nodes.get(f"agent:{agent_id}") or {}
    expected = rec.get("token_hash")

    if not expected:
        if is_registered(cg, agent_id) is False and rec.get("revoked_at"):
            return {"ok": False, "reason": f"identity '{agent_id}' is revoked"}
        return {"ok": True, "reason": "no registration on file (open mode)",
                "verified": False}
    if not token:
        return {"ok": False, "reason": f"'{agent_id}' is registered but no "
                                       f"TG_AGENT_TOKEN was presented"}
    # constant-time so a wrong token cannot be narrowed down by timing
    if not hmac.compare_digest(_hash(agent_id, token), expected):
        return {"ok": False, "reason": f"token does not match registered "
                                       f"identity '{agent_id}'"}
    return {"ok": True, "reason": "verified", "verified": True}


def check(cg, agent_id: str, namespace: str, token: str = None) -> dict:
    """The gate's entry point: verify, then apply the namespace's policy on
    unverified callers."""
    v = verify(cg, agent_id, token)
    if not v["ok"]:
        return v
    if v.get("verified"):
        return v
    if requires_auth(cg, namespace):
        return {"ok": False, "reason":
                f"namespace '{namespace}' requires verified identity and "
                f"'{agent_id}' is not registered"}
    return v


def roster(cg) -> list:
    """Who may act in this fleet — never includes secrets."""
    return sorted(
        ({"agent_id": d.get("agent_id"), "namespace": d.get("namespace"),
          "active": bool(d.get("token_hash")),
          "registered_at": d.get("registered_at")}
         for _, d in cg.g.nodes(data=True) if d.get("plane") == "agent"),
        key=lambda r: r["agent_id"] or "")
